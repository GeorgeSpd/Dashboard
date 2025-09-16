# pylint: disable=C0114
# pylint: disable=C0103
# pylint: disable=C0116
# pylint: disable=C0301
# pylint: disable=I1101
# pylint: disable=W0401
# pylint: disable=W0614
# pylint: disable=C0303
# pylint: disable=W0105
# pylint: disable=W0621
# pylint: disable=W0718
# pylint: disable=C0302

"""
Rendering logic for the **single-scenario view** in the dashboard.

Features
--------
- Displays KPIs per house and for all houses
- Breaks down costs, emissions, comfort, and efficiency
- Generates interactive Plotly charts for:
  - Temperature profiles
  - Solar irradiance
  - Heat flows
  - Efficiency (COP/EER)
  - Monthly summaries (electricity, heating, cooling, CO₂)
- Provides CSV export for full simulation results
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config.colors import COLORS


# ------------------------------- Helpers -------------------------------
def _dt_hours(index: pd.DatetimeIndex) -> pd.Series:
    """Step duration (hours) for possibly irregular time series."""
    if len(index) == 0:
        return pd.Series(dtype=float, index=index)
    t = pd.to_datetime(index)
    if len(t) == 1:
        return pd.Series([1.0], index=index)
    dt = (t[1:] - t[:-1]).total_seconds() / 3600.0
    dt = np.r_[dt[0], dt]
    return pd.Series(dt, index=index, dtype=float)

def _seasonal_efficiencies(sim: pd.DataFrame, cfg: dict) -> dict:
    """
    Seasonal efficiencies over the *selected window*:
      - SCOP_HP   : seasonal COP for electric heating techs
      - ETA_GAS   : seasonal thermal efficiency for gas heating techs
      - SEER_ELEC : seasonal EER for electric cooling techs
    Uses per-tech series and includes service+distribution energy where available.
    """
    out = {"SCOP_HP": float("nan"), "ETA_GAS": float("nan"), "SEER_ELEC": float("nan")}
    if sim is None or sim.empty:
        return out

    def _tech_names(chain_key: str, fuel: str) -> set[str]:
        is_central = cfg.get("arch") == "Centralized"
        chain = (cfg.get("arch_cfg") or {}).get(chain_key, []) if is_central else (cfg.get(chain_key) or [])
        return {
            str(t.get("name", "")).strip()
            for t in chain
            if str(t.get("fuel", "elec")) == fuel and t.get("name")
        }

    hp_heat_names   = _tech_names("heating_chain", "elec")
    gas_heat_names  = _tech_names("heating_chain", "gas")
    elec_cool_names = _tech_names("cooling_chain",  "elec")

    def _sum_cols(prefixes: list[str], techs: set[str], suffix="_kWh") -> pd.Series:
        if not techs:
            return pd.Series(0.0, index=sim.index)
        parts = []
        for tech in techs:
            for p in prefixes:
                col = f"{p}{tech}{suffix}"
                if col in sim.columns:
                    parts.append(sim[col])
        return sum(parts) if parts else pd.Series(0.0, index=sim.index)

    def _sum_Q(prefix: str, techs: set[str]) -> pd.Series:
        """Sum thermal outputs [W] per tech family and convert to kWh."""
        if not techs:
            return pd.Series(0.0, index=sim.index)
        cols = [f"{prefix}{t}_W" for t in techs if f"{prefix}{t}_W" in sim.columns]
        if not cols:
            return pd.Series(0.0, index=sim.index)
        Q_W  = sim[cols].sum(axis=1)
        dt_h = sim.index.to_series().diff().dt.total_seconds().div(3600.0)
        if len(dt_h) > 1 and (pd.isna(dt_h.iloc[0]) or dt_h.iloc[0] <= 0):
            dt_h.iloc[0] = dt_h.iloc[1]
        dt_h = dt_h.fillna(1.0).clip(lower=1e-9)
        return (Q_W * dt_h / 1000.0).reindex(sim.index, fill_value=0.0)

    # Heating families
    Q_heat_HP_kWh  = _sum_Q("Q_heat_", hp_heat_names)
    Q_heat_GAS_kWh = _sum_Q("Q_heat_", gas_heat_names)
    E_elec_HP_kWh  = _sum_cols(["E_elec_heat_", "E_elec_heat_dist_"], hp_heat_names)
    E_gas_kWh      = _sum_cols(["E_gas_heat_",  "E_gas_heat_dist_"], gas_heat_names)

    # Cooling (electric)
    Q_cool_E_kWh     = _sum_Q("Q_cool_", elec_cool_names)
    E_elec_cool_kWh  = _sum_cols(["E_elec_cool_", "E_elec_cool_dist_"], elec_cool_names)

    # Aggregates
    hp_elec = float(E_elec_HP_kWh.sum())
    gas_in  = float(E_gas_kWh.sum())
    cool_el = float(E_elec_cool_kWh.sum())

    out["SCOP_HP"]   = float(Q_heat_HP_kWh.sum()  / hp_elec) if hp_elec > 0 else float("nan")
    out["ETA_GAS"]   = float(Q_heat_GAS_kWh.sum() / gas_in)  if gas_in  > 0 else float("nan")
    out["SEER_ELEC"] = float(Q_cool_E_kWh.sum()   / cool_el) if cool_el > 0 else float("nan")
    return out

def _per_tech_elec_total(sim: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Returns per-tech electricity (kWh) for 'heating' or 'cooling',
    combining service + distribution. Uses explicit per-tech electricity
    if present; otherwise apportions totals using per-tech thermal shares.
    Columns of the returned frame are technology keys (already stripped).
    """
    assert mode in ("heating", "cooling")
    # prefixes & total column names
    pref_e     = "E_elec_heat_" if mode == "heating" else "E_elec_cool_"
    total_svc  = sim.get(f"E_elec_kWh_{'heat' if mode=='heating' else 'cool'}_service")
    total_dist = sim.get(f"E_elec_kWh_dist_{'heat' if mode=='heating' else 'cool'}")

    # 1) If explicit per-tech electricity exists, use it (+ distribute the dist total proportionally)
    per_cols = [c for c in sim.columns if c.startswith(pref_e) and c.endswith("_kWh") and "_dist_" not in c]
    if per_cols and sum(float(sim[c].sum()) for c in per_cols) > 0:
        per = sim[per_cols].copy()
        per.columns = [c.removeprefix(pref_e).removesuffix("_kWh") for c in per.columns]
        if total_dist is not None and float(total_dist.sum()) > 0:
            svc_sum = per.sum(axis=1).replace(0, np.nan)
            w = per.div(svc_sum, axis=0).fillna(0.0)
            per = per + w.mul(total_dist, axis=0)
        return per

    # 2) Fallback: apportion from thermal split shares
    q_tot_col = "Q_heat_W" if mode == "heating" else "Q_cool_W"
    q_total = sim.get(q_tot_col)
    if q_total is None or (q_total == 0).all():
        return pd.DataFrame(index=sim.index)

    q_cols = [c for c in sim.columns if c.startswith(f"Q_{'heat' if mode=='heating' else 'cool'}_") and c.endswith("_W")]
    q_cols = [c for c in q_cols if c not in (q_tot_col, f"Q_unmet_{'heat' if mode=='heating' else 'cool'}_W")]
    if not q_cols:
        return pd.DataFrame(index=sim.index)

    q_per = sim[q_cols].copy()
    # from "Q_heat_<TECH>_W" → "<TECH>"
    q_per.columns = [c.split("_", 2)[2][:-2] for c in q_cols]

    denom = q_total.replace(0, np.nan)
    w = q_per.div(denom, axis=0).fillna(0.0)

    # service part
    if total_svc is None or float(total_svc.sum()) == 0.0:
        return pd.DataFrame(index=sim.index, columns=q_per.columns).fillna(0.0)
    per = w.mul(total_svc, axis=0)

    # add distribution part proportionally if present
    if total_dist is not None and float(total_dist.sum()) > 0:
        per = per + w.mul(total_dist, axis=0)

    return per

def _stacked_bar(df_long: pd.DataFrame, title: str, ylab: str, color_col: str | None = None):
    """Small wrapper to standardize stacked bar formatting."""
    if color_col is None:
        if "Category" in df_long.columns:
            color_col = "Category"
        elif "Technology" in df_long.columns:
            color_col = "Technology"
        elif "Source" in df_long.columns:
            color_col = "Source"
    fig = px.bar(
        df_long, x="time", y="Value", color=color_col, barmode="stack",
        labels={"time": "Month", "Value": ylab}
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(title=title)
    return fig

# ----------------------------- Main renderer -----------------------------
def render_single(label: str, cfg: dict, sim: pd.DataFrame, view: pd.DataFrame):
    price_elec = float(cfg.get("price_elec", cfg.get("price", 0.30)))
    price_gas  = float(cfg.get("price_gas", 0.10))
    co2_elec   = float(cfg.get("co2_per_kwh_elec", cfg.get("co2_per_kwh", 0.40)))
    co2_gas    = float(cfg.get("co2_per_kwh_gas", 0.20))

    # --- KPIs (robust integration for loads) ---
    dt_h = _dt_hours(view.index)
    heat_kwh = float((view.get("Q_heat_W", pd.Series(0, index=view.index)) * dt_h).sum() / 1000.0)
    cool_kwh = float((view.get("Q_cool_W", pd.Series(0, index=view.index)) * dt_h).sum() / 1000.0)
    elec_kwh = float(view.get("E_elec_kWh", pd.Series(0, index=view.index)).sum())
    gas_kwh  = float(view.get("E_gas_kWh",  pd.Series(0, index=view.index)).sum())

    effs = _seasonal_efficiencies(view, cfg) or {}
    SCOP_HP   = effs.get("SCOP_HP",   float("nan"))
    ETA_GAS   = effs.get("ETA_GAS",   float("nan"))
    SEER_ELEC = effs.get("SEER_ELEC", float("nan"))

    op_co2_house_window = float(view.get("CO2_kg", pd.Series(0, index=view.index)).sum())
    embodied_all = float(cfg.get("co2_embodied_kg", 0.0)) * cfg["nr_houses"]
    op_co2_all   = op_co2_house_window * cfg["nr_houses"]
    total_co2_all = embodied_all + op_co2_all

    capex_house     = float(cfg.get("capex_house", 0.0))
    op_cost_house   = elec_kwh * price_elec + gas_kwh * price_gas
    total_cost_house = op_cost_house + capex_house

    # -------------------- Overview KPIs --------------------
    st.subheader("Results")
    with st.expander("Result Overview", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown("### Per house")
            st.metric("Heating load [kWh]", f"{heat_kwh:.0f}")
            st.metric("Cooling load [kWh]", f"{cool_kwh:.0f}")
            st.metric("Electricity [kWh]", f"{elec_kwh:.0f}")
            st.metric("Operational costs [€]", f"{op_cost_house:.0f}")
            st.metric("Capital costs [€]", f"{capex_house:.0f}")
            st.metric("Total costs [€]", f"{total_cost_house:.0f}")

        with col2:
            st.markdown("### All houses")
            n = int(cfg["nr_houses"])
            st.metric("Heating load [kWh]", f"{heat_kwh * n:.0f}")
            st.metric("Cooling load [kWh]", f"{cool_kwh * n:.0f}")
            st.metric("Electricity [kWh]", f"{elec_kwh * n:.0f}")
            st.metric("Operational costs [€]", f"{op_cost_house * n:.0f}")
            st.metric("Capital costs [€]", f"{float(capex_house * n):.0f}")
            st.metric("Total costs [€]", f"{float(total_cost_house * n):.0f}")

        with col3:
            st.markdown("### Total Emissions")
            st.metric("Capital CO₂ [kg]", f"{embodied_all:.0f}")
            st.metric("Operational CO₂ [kg]", f"{op_co2_all:.0f}")
            st.metric("Total CO₂ [kg]", f"{total_co2_all:.0f}")

        with col4:
            st.markdown("### Comfort")
            st.metric("Avg indoor [°C]", f"{view['T_in_pred'].mean():.1f}")
            st.metric("Min/Max [°C]", f"{view['T_in_pred'].min():.1f} / {view['T_in_pred'].max():.1f}")

        with col5:
            st.markdown("### Efficiency")
            st.metric("SCOP (heat pumps)",  "—" if np.isnan(SCOP_HP)   else f"{SCOP_HP:.2f}")
            st.metric("SEER (electric)",    "—" if np.isnan(SEER_ELEC) else f"{SEER_ELEC:.2f}")
            st.metric("Boiler η (seasonal)","—" if np.isnan(ETA_GAS)   else f"{ETA_GAS:.2f}")

    # -------------------- Plots --------------------
    with st.expander("Plots", expanded=False):
        tab_temp, tab_solar, tab_flows, tab_eff, tab_summary = st.tabs(
            ["🌡️ Temperature", "☀️ Solar", "🔁 Energy Flows", "⚙️ Efficiency", "📊 Summary"]
        )

        # Temperature
        with tab_temp:
            st.caption("Temperature profiles")
            fig_t = px.line(
                view.reset_index(), x="time", y=["T_out", "T_in_pred", "T_wall_pred"],
                labels={"value": "Temperature [°C]", "time": "Time", "variable": "Series"},
                color_discrete_map=COLORS,
            )
            fig_t.update_layout(hovermode="x unified")
            st.plotly_chart(fig_t, use_container_width=True)

        # Solar
        with tab_solar:
            st.caption("Solar irradiance (global horizontal)")
            if "G_solar" in view.columns:
                fig_g = px.line(
                    view.reset_index(), x="time", y="G_solar",
                    labels={"G_solar": "Solar irradiance [W/m²]", "time": "Time"},
                )
                fig_g.update_traces(line_color=COLORS.get("G_solar", None))
                fig_g.update_layout(hovermode="x unified")
                st.plotly_chart(fig_g, use_container_width=True)
            else:
                st.info("No solar data in view.")

        # Energy flows + stacks
        with tab_flows:
            st.caption("Heat flows & technology dispatch")

            # Envelope/solar context + totals
            y_ctx = [c for c in ["Q_solar_W", "Q_cond_W", "Q_inf_W", "Q_net_W"] if c in view.columns]
            if y_ctx:
                fig_ctx = px.line(
                    view.reset_index(), x="time", y=y_ctx,
                    labels={"value": "Power [W]", "variable": "Flow", "time": "Time"},
                    color_discrete_map=COLORS,
                )
                if "Q_heat_W" in view.columns:
                    fig_ctx.add_scatter(x=view.index, y=view["Q_heat_W"], mode="lines",
                                        name="Q_heat_W", line=dict(color=COLORS.get("Q_heat_W"), width=2))
                if "Q_cool_W" in view.columns:
                    fig_ctx.add_scatter(x=view.index, y=view["Q_cool_W"], mode="lines",
                                        name="Q_cool_W", line=dict(color=COLORS.get("Q_cool_W"), width=2))
                fig_ctx.update_layout(title="Envelope & solar flows (+ totals)", hovermode="x unified")
                st.plotly_chart(fig_ctx, use_container_width=True)

            # Per-tech heating (stacked area)
            heat_cols = [c for c in view.columns if c.startswith("Q_heat_") and c.endswith("_W")]
            heat_cols = [c for c in heat_cols if c not in ("Q_heat_W", "Q_unmet_heat_W")]
            if heat_cols:
                v_heat = view[heat_cols].rename(columns=lambda c: c.removeprefix("Q_heat_").removesuffix("_W"))
                fig_hstack = px.area(
                    v_heat.reset_index().melt(id_vars="time", var_name="Technology", value_name="W"),
                    x="time", y="W", color="Technology",
                    labels={"W": "Heating power [W]", "time": "Time"},
                )
                fig_hstack.update_traces(line=dict(width=0))
                if "Q_unmet_heat_W" in view.columns and view["Q_unmet_heat_W"].abs().sum() > 0:
                    fig_hstack.add_scatter(
                        x=view.index, y=view["Q_unmet_heat_W"], mode="lines",
                        name="Unmet heat [W]", line=dict(width=2, dash="dash"),
                    )
                fig_hstack.update_layout(title="Heating by technology (stacked)", hovermode="x unified")
                st.plotly_chart(fig_hstack, use_container_width=True)

            # Per-tech cooling (stacked area)
            cool_cols = [c for c in view.columns if c.startswith("Q_cool_") and c.endswith("_W")]
            cool_cols = [c for c in cool_cols if c not in ("Q_cool_W", "Q_unmet_cool_W")]
            if cool_cols:
                v_cool = view[cool_cols].rename(columns=lambda c: c.removeprefix("Q_cool_").removesuffix("_W"))
                fig_cstack = px.area(
                    v_cool.reset_index().melt(id_vars="time", var_name="Technology", value_name="W"),
                    x="time", y="W", color="Technology",
                    labels={"W": "Cooling power [W]", "time": "Time"},
                )
                fig_cstack.update_traces(line=dict(width=0))
                if "Q_unmet_cool_W" in view.columns and view["Q_unmet_cool_W"].abs().sum() > 0:
                    fig_cstack.add_scatter(
                        x=view.index, y=view["Q_unmet_cool_W"], mode="lines",
                        name="Unmet cool [W]", line=dict(width=2, dash="dash"),
                    )
                fig_cstack.update_layout(title="Cooling by technology (stacked)", hovermode="x unified")
                st.plotly_chart(fig_cstack, use_container_width=True)

        # Efficiency (effective + scatter)
        with tab_eff:
            st.caption("Efficiency")

            # Bars over time
            y_eff = []
            if "COP_heating_eff" in view.columns:
                y_eff.append("COP_heating_eff")
            if "EER_cooling_eff" in view.columns:
                y_eff.append("EER_cooling_eff")

            if y_eff:
                v_eff_long = view.reset_index().melt(
                    id_vars=["time"], value_vars=y_eff, var_name="Metric", value_name="Value"
                )
                v_eff_long = v_eff_long.replace([np.inf, -np.inf], np.nan).dropna(subset=["Value"])
                v_eff_long = v_eff_long[v_eff_long["Value"] > 0]
                fig_eff_bar = px.bar(
                    v_eff_long, x="time", y="Value", color="Metric", barmode="group",
                    labels={"Value": "Efficiency", "time": "Time"},
                )
                fig_eff_bar.update_traces(marker_line_width=0)
                fig_eff_bar.update_layout(hovermode="x unified", title="Effective COP/EER over time")
                st.plotly_chart(fig_eff_bar, use_container_width=True)

                # Scatter vs outdoor temp
                vf_long = view.reset_index().melt(
                    id_vars=["time", "T_out"], value_vars=y_eff, var_name="Metric", value_name="Efficiency"
                )
                vf_long = vf_long.replace([np.inf, -np.inf], np.nan).dropna(subset=["Efficiency"])
                vf_long = vf_long[vf_long["Efficiency"] > 0]
                fig_sc = px.scatter(vf_long, x="T_out", y="Efficiency", color="Metric",
                                    opacity=0.6, labels={"T_out": "Outdoor temperature [°C]"})
                fig_sc.update_layout(title="Efficiency vs Outdoor Temperature")
                st.plotly_chart(fig_sc, use_container_width=True)
            else:
                st.info("No effective COP/EER series found.")

        # Summary (monthly stacks)
        with tab_summary:
            st.subheader("Summary per house")

            def _monthly_sum(sim: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
                present = [c for c in cols if c in sim.columns]
                if not present:
                    if len(sim.index) == 0:
                        return pd.DataFrame()
                return sim[present].resample("M").sum() if present else pd.DataFrame(index=sim.resample("M").sum().index)

            # needs_tab, elec_tab, gas_tab, co2_tab, cost_tab = st.tabs(
            #     ["🔥 Thermal needs", "⚡ Electricity", "🛢️ Gas", "🌍 CO₂", "💶 Costs"]
            # )

            view_key = st.radio(
                "View", ["🔥 Thermal needs", "⚡ Electricity", "🛢️ Gas", "🌍 CO₂", "💶 Costs"],
                horizontal=True, key=f"summary_view_{label}"
            )

            # Thermal needs
            # with needs_tab:
            if view_key == "🔥 Thermal needs":
                if "E_hvac_kWh" in sim.columns:
                    monthly_heat = sim["E_hvac_kWh"].clip(lower=0).resample("M").sum().rename("Heating (kWh)")
                    monthly_cool = (-sim["E_hvac_kWh"].clip(upper=0)).resample("M").sum().rename("Cooling (kWh)")
                    hc_long = pd.concat([monthly_heat, monthly_cool], axis=1).reset_index().melt(
                        id_vars="time", var_name="Type", value_name="Energy [kWh]"
                    )
                    fig_hc = px.bar(hc_long, x="time", y="Energy [kWh]", color="Type", barmode="group")
                    fig_hc.update_layout(title="Monthly thermal needs")
                    st.plotly_chart(fig_hc, use_container_width=True)
                else:
                    st.info("No thermal needs available.")

            # Electricity
            elif view_key == "⚡ Electricity":
                st.markdown("#### Electricity by source")
                elec_src_cols = {
                    "Cooling (service)": "E_elec_kWh_cool_service",
                    "Cooling losses":    "E_elec_kWh_dist_cool",
                    "Heating (service)": "E_elec_kWh_heat_service",
                    "Heating losses":    "E_elec_kWh_dist_heat",
                    "Auxiliaries":       "E_elec_kWh_aux",
                }
                m_elec_src = _monthly_sum(sim, list(elec_src_cols.values()))
                if not m_elec_src.empty:
                    m_elec_src = m_elec_src.rename(columns={v: k for k, v in elec_src_cols.items() if v in m_elec_src.columns})
                    elec_src_long = m_elec_src.reset_index().melt(id_vars="time", var_name="Category", value_name="Value")
                    st.plotly_chart(_stacked_bar(elec_src_long, "Monthly electricity by source", "kWh"),
                                    use_container_width=True)
                else:
                    st.info("No electricity source components found.")

                st.divider()
                st.markdown("#### Electricity by technology (service + distribution)")
                elec_c1, elec_c2 = st.columns(2)

                # Heating tech electricity (service + dist), robust for decentralized
                m_heat_per = _per_tech_elec_total(sim, "heating").resample("M").sum()
                if not m_heat_per.empty:
                    ht_long = m_heat_per.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                    with elec_c1:
                        st.plotly_chart(_stacked_bar(ht_long, "Monthly electricity by technology (heating)", "kWh", color_col="Technology"),
                                        use_container_width=True)
                else:
                    with elec_c1:
                        st.info("No heating electricity by technology found.")

                # Cooling tech electricity (service + dist), robust for decentralized
                m_cool_per = _per_tech_elec_total(sim, "cooling").resample("M").sum()
                if not m_cool_per.empty:
                    ct_long = m_cool_per.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                    with elec_c2:
                        st.plotly_chart(_stacked_bar(ct_long, "Monthly electricity by technology (cooling)", "kWh", color_col="Technology"),
                                        use_container_width=True)
                else:
                    with elec_c2:
                        st.info("No cooling electricity by technology found.")

            # Gas
            elif view_key == "🛢️ Gas":
                st.markdown("#### Gas by source")
                gas_src_cols = {
                    "Heating (service)": "E_gas_kWh_heat_service",
                    "Heating losses":    "E_gas_kWh_dist_heat",
                }
                m_gas_src = _monthly_sum(sim, list(gas_src_cols.values()))
                if not m_gas_src.empty:
                    m_gas_src = m_gas_src.rename(columns={v: k for k, v in gas_src_cols.items() if v in m_gas_src.columns})
                    gas_src_long = m_gas_src.reset_index().melt(id_vars="time", var_name="Category", value_name="Value")
                    st.plotly_chart(_stacked_bar(gas_src_long, "Monthly gas by source", "kWh"), use_container_width=True)
                else:
                    st.info("No gas source components found.")

            # CO₂
            elif view_key == "🌍 CO₂":
                st.markdown("#### CO₂ by source")

                # keep service and losses as separate bars
                elec_src_cols = {
                    "Elec Cooling (service)": "E_elec_kWh_cool_service",
                    "Elec Cooling losses":    "E_elec_kWh_dist_cool",
                    "Elec Heating (service)": "E_elec_kWh_heat_service",
                    "Elec Heating losses":    "E_elec_kWh_dist_heat",
                }
                gas_src_cols = {
                    "Gas Heating (service)":  "E_gas_kWh_heat_service",
                    "Gas Heating losses":     "E_gas_kWh_dist_heat",
                    "Auxiliaries":             "E_elec_kWh_aux",
                }

                # monthly energy
                m_elec_src_E = _monthly_sum(sim, list(elec_src_cols.values()))
                m_gas_src_E  = _monthly_sum(sim, list(gas_src_cols.values()))

                frames = []
                if not m_elec_src_E.empty:
                    m_elec_src_E = m_elec_src_E.rename(columns={v: k for k, v in elec_src_cols.items() if v in m_elec_src_E.columns})
                    m_elec_src_CO2 = m_elec_src_E * co2_elec
                    frames.append(m_elec_src_CO2)
                if not m_gas_src_E.empty:
                    m_gas_src_E = m_gas_src_E.rename(columns={v: k for k, v in gas_src_cols.items() if v in m_gas_src_E.columns})
                    m_gas_src_CO2 = m_gas_src_E * co2_gas
                    frames.append(m_gas_src_CO2)

                if frames:
                    msrc_co2 = pd.concat(frames, axis=1).fillna(0.0)
                    co2_long = msrc_co2.reset_index().melt(id_vars="time", var_name="Category", value_name="Value")
                    st.plotly_chart(_stacked_bar(co2_long, "Monthly CO₂ by source", "CO₂ [kg]"), use_container_width=True)
                else:
                    st.info("No CO₂ source components found.")

                st.divider()
                st.markdown("#### CO₂ by technology (service + distribution)")
                co2_c1, co2_c2 = st.columns(2)

                # factors
                co2_elec = float(cfg.get("co2_per_kwh_elec", cfg.get("co2_per_kwh", 0.40)))
                co2_gas  = float(cfg.get("co2_per_kwh_gas", 0.20))

                # ===== HEATING =====
                # collect per-tech service/dist (elec + gas), then combine to a single per-tech total
                heat_elec_service = [c for c in sim.columns if c.startswith("E_elec_heat_")      and c.endswith("_kWh") and "_dist_" not in c]
                heat_elec_dist    = [c for c in sim.columns if c.startswith("E_elec_heat_dist_") and c.endswith("_kWh")]
                heat_gas_service  = [c for c in sim.columns if c.startswith("E_gas_heat_")       and c.endswith("_kWh") and "_dist_" not in c]
                heat_gas_dist     = [c for c in sim.columns if c.startswith("E_gas_heat_dist_")  and c.endswith("_kWh")]

                map_e_serv = {c: c.removeprefix("E_elec_heat_").removesuffix("_kWh")      for c in heat_elec_service}
                map_e_dist = {c: c.removeprefix("E_elec_heat_dist_").removesuffix("_kWh") for c in heat_elec_dist}
                map_g_serv = {c: c.removeprefix("E_gas_heat_").removesuffix("_kWh")       for c in heat_gas_service}
                map_g_dist = {c: c.removeprefix("E_gas_heat_dist_").removesuffix("_kWh")  for c in heat_gas_dist}

                m_e_serv = sim[heat_elec_service].rename(columns=map_e_serv).resample("M").sum() if heat_elec_service else pd.DataFrame()
                m_e_dist = sim[heat_elec_dist].rename(columns=map_e_dist).resample("M").sum()     if heat_elec_dist     else pd.DataFrame()
                m_g_serv = sim[heat_gas_service].rename(columns=map_g_serv).resample("M").sum()  if heat_gas_service  else pd.DataFrame()
                m_g_dist = sim[heat_gas_dist].rename(columns=map_g_dist).resample("M").sum()      if heat_gas_dist      else pd.DataFrame()

                techs = sorted(set(m_e_serv.columns) | set(m_e_dist.columns) | set(m_g_serv.columns) | set(m_g_dist.columns))
                if techs:
                    idx = sim.resample("M").sum().index
                    m_e_total = m_e_serv.reindex(index=idx, columns=techs, fill_value=0.0) + m_e_dist.reindex(index=idx, columns=techs, fill_value=0.0)
                    m_g_total = m_g_serv.reindex(index=idx, columns=techs, fill_value=0.0) + m_g_dist.reindex(index=idx, columns=techs, fill_value=0.0)
                    m_heat_co2 = m_e_total * co2_elec + m_g_total * co2_gas

                    h_long = m_heat_co2.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                    with co2_c1:
                        st.plotly_chart(_stacked_bar(h_long, "Monthly CO₂ by technology (heating)", "CO₂ [kg]", color_col="Technology"),
                                        use_container_width=True)
                else:
                    with co2_c1:
                        st.info("No heating CO₂ by technology found.")

                # ===== COOLING =====
                cool_elec_service = [c for c in sim.columns if c.startswith("E_elec_cool_")      and c.endswith("_kWh") and "_dist_" not in c]
                cool_elec_dist    = [c for c in sim.columns if c.startswith("E_elec_cool_dist_") and c.endswith("_kWh")]
                cool_gas_service  = [c for c in sim.columns if c.startswith("E_gas_cool_")       and c.endswith("_kWh") and "_dist_" not in c]
                cool_gas_dist     = [c for c in sim.columns if c.startswith("E_gas_cool_dist_")  and c.endswith("_kWh")]

                map_ec_serv = {c: c.removeprefix("E_elec_cool_").removesuffix("_kWh")      for c in cool_elec_service}
                map_ec_dist = {c: c.removeprefix("E_elec_cool_dist_").removesuffix("_kWh") for c in cool_elec_dist}
                map_gc_serv = {c: c.removeprefix("E_gas_cool_").removesuffix("_kWh")       for c in cool_gas_service}
                map_gc_dist = {c: c.removeprefix("E_gas_cool_dist_").removesuffix("_kWh")  for c in cool_gas_dist}

                m_ec_serv = sim[cool_elec_service].rename(columns=map_ec_serv).resample("M").sum() if cool_elec_service else pd.DataFrame()
                m_ec_dist = sim[cool_elec_dist].rename(columns=map_ec_dist).resample("M").sum()     if cool_elec_dist     else pd.DataFrame()
                m_gc_serv = sim[cool_gas_service].rename(columns=map_gc_serv).resample("M").sum()  if cool_gas_service  else pd.DataFrame()
                m_gc_dist = sim[cool_gas_dist].rename(columns=map_gc_dist).resample("M").sum()      if cool_gas_dist      else pd.DataFrame()

                techs_c = sorted(set(m_ec_serv.columns) | set(m_ec_dist.columns) | set(m_gc_serv.columns) | set(m_gc_dist.columns))
                if techs_c:
                    idx = sim.resample("M").sum().index
                    m_e_total = m_ec_serv.reindex(index=idx, columns=techs_c, fill_value=0.0) + m_ec_dist.reindex(index=idx, columns=techs_c, fill_value=0.0)
                    m_g_total = m_gc_serv.reindex(index=idx, columns=techs_c, fill_value=0.0) + m_gc_dist.reindex(index=idx, columns=techs_c, fill_value=0.0)
                    m_cool_co2 = m_e_total * co2_elec + m_g_total * co2_gas

                    c_long = m_cool_co2.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                    with co2_c2:
                        st.plotly_chart(_stacked_bar(c_long, "Monthly CO₂ by technology (cooling)", "CO₂ [kg]", color_col="Technology"),
                                        use_container_width=True)
                else:
                    with co2_c2:
                        st.info("No cooling CO₂ by technology found.")

            # Costs
            elif view_key == "💶 Costs":
                st.markdown("#### Cost by source")

                # keep service and losses as separate bars
                elec_src_cols = {
                    "Elec Cooling (service)": "E_elec_kWh_cool_service",
                    "Elec Cooling losses":    "E_elec_kWh_dist_cool",
                    "Elec Heating (service)": "E_elec_kWh_heat_service",
                    "Elec Heating losses":    "E_elec_kWh_dist_heat",
                }
                gas_src_cols = {
                    "Gas Heating (service)":  "E_gas_kWh_heat_service",
                    "Gas Heating losses":     "E_gas_kWh_dist_heat",
                    "Auxiliaries":             "E_elec_kWh_aux",
                }

                # monthly energy → monthly cost
                m_elec_src_E = _monthly_sum(sim, list(elec_src_cols.values()))
                m_gas_src_E  = _monthly_sum(sim, list(gas_src_cols.values()))

                frames = []
                if not m_elec_src_E.empty:
                    m_elec_src_E = m_elec_src_E.rename(columns={v: k for k, v in elec_src_cols.items() if v in m_elec_src_E.columns})
                    m_elec_src_cost = m_elec_src_E * price_elec
                    frames.append(m_elec_src_cost)
                if not m_gas_src_E.empty:
                    m_gas_src_E = m_gas_src_E.rename(columns={v: k for k, v in gas_src_cols.items() if v in m_gas_src_E.columns})
                    m_gas_src_cost = m_gas_src_E * price_gas
                    frames.append(m_gas_src_cost)

                if frames:
                    msrc_cost = pd.concat(frames, axis=1).fillna(0.0)
                    cost_long = msrc_cost.reset_index().melt(id_vars="time", var_name="Category", value_name="Value")
                    st.plotly_chart(_stacked_bar(cost_long, "Monthly cost by source", "Costs [€]"), use_container_width=True)
                else:
                    st.info("No cost source components found.")

                st.divider()
                st.markdown("#### Cost by technology (service + distribution)")
                cost_c1, cost_c2 = st.columns(2)
                price_elec = float(cfg.get("price_elec", cfg.get("price", 0.30)))
                price_gas  = float(cfg.get("price_gas", 0.10))

                # ===== HEATING =====
                heat_elec_service = [c for c in sim.columns if c.startswith("E_elec_heat_")      and c.endswith("_kWh") and "_dist_" not in c]
                heat_elec_dist    = [c for c in sim.columns if c.startswith("E_elec_heat_dist_") and c.endswith("_kWh")]
                heat_gas_service  = [c for c in sim.columns if c.startswith("E_gas_heat_")       and c.endswith("_kWh") and "_dist_" not in c]
                heat_gas_dist     = [c for c in sim.columns if c.startswith("E_gas_heat_dist_")  and c.endswith("_kWh")]

                map_e_serv = {c: c.removeprefix("E_elec_heat_").removesuffix("_kWh")      for c in heat_elec_service}
                map_e_dist = {c: c.removeprefix("E_elec_heat_dist_").removesuffix("_kWh") for c in heat_elec_dist}
                map_g_serv = {c: c.removeprefix("E_gas_heat_").removesuffix("_kWh")       for c in heat_gas_service}
                map_g_dist = {c: c.removeprefix("E_gas_heat_dist_").removesuffix("_kWh")  for c in heat_gas_dist}

                m_e_serv = sim[heat_elec_service].rename(columns=map_e_serv).resample("M").sum() if heat_elec_service else pd.DataFrame()
                m_e_dist = sim[heat_elec_dist].rename(columns=map_e_dist).resample("M").sum()     if heat_elec_dist     else pd.DataFrame()
                m_g_serv = sim[heat_gas_service].rename(columns=map_g_serv).resample("M").sum()  if heat_gas_service  else pd.DataFrame()
                m_g_dist = sim[heat_gas_dist].rename(columns=map_g_dist).resample("M").sum()      if heat_gas_dist      else pd.DataFrame()

                techs = sorted(set(m_e_serv.columns) | set(m_e_dist.columns) | set(m_g_serv.columns) | set(m_g_dist.columns))
                if techs:
                    idx = sim.resample("M").sum().index
                    m_e_total = m_e_serv.reindex(index=idx, columns=techs, fill_value=0.0) + m_e_dist.reindex(index=idx, columns=techs, fill_value=0.0)
                    m_g_total = m_g_serv.reindex(index=idx, columns=techs, fill_value=0.0) + m_g_dist.reindex(index=idx, columns=techs, fill_value=0.0)
                    m_heat_cost = m_e_total * price_elec + m_g_total * price_gas  # single bar per tech (includes losses)

                    h_cost_long = m_heat_cost.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                    with cost_c1:
                        st.plotly_chart(_stacked_bar(h_cost_long, "Monthly cost by technology (heating)", "Costs [€]", color_col="Technology"),
                                        use_container_width=True)
                else:
                    with cost_c1:
                        st.info("No heating tech cost data.")

                # ===== COOLING =====
                cool_elec_service = [c for c in sim.columns if c.startswith("E_elec_cool_")      and c.endswith("_kWh") and "_dist_" not in c]
                cool_elec_dist    = [c for c in sim.columns if c.startswith("E_elec_cool_dist_") and c.endswith("_kWh")]
                cool_gas_service  = [c for c in sim.columns if c.startswith("E_gas_cool_")       and c.endswith("_kWh") and "_dist_" not in c]
                cool_gas_dist     = [c for c in sim.columns if c.startswith("E_gas_cool_dist_")  and c.endswith("_kWh")]

                map_ec_serv = {c: c.removeprefix("E_elec_cool_").removesuffix("_kWh")      for c in cool_elec_service}
                map_ec_dist = {c: c.removeprefix("E_elec_cool_dist_").removesuffix("_kWh") for c in cool_elec_dist}
                map_gc_serv = {c: c.removeprefix("E_gas_cool_").removesuffix("_kWh")       for c in cool_gas_service}
                map_gc_dist = {c: c.removeprefix("E_gas_cool_dist_").removesuffix("_kWh")  for c in cool_gas_dist}

                m_ec_serv = sim[cool_elec_service].rename(columns=map_ec_serv).resample("M").sum() if cool_elec_service else pd.DataFrame()
                m_ec_dist = sim[cool_elec_dist].rename(columns=map_ec_dist).resample("M").sum()     if cool_elec_dist     else pd.DataFrame()
                m_gc_serv = sim[cool_gas_service].rename(columns=map_gc_serv).resample("M").sum()  if cool_gas_service  else pd.DataFrame()
                m_gc_dist = sim[cool_gas_dist].rename(columns=map_gc_dist).resample("M").sum()      if cool_gas_dist      else pd.DataFrame()

                techs_c = sorted(set(m_ec_serv.columns) | set(m_ec_dist.columns) | set(m_gc_serv.columns) | set(m_gc_dist.columns))
                if techs_c:
                    idx = sim.resample("M").sum().index
                    m_e_total = m_ec_serv.reindex(index=idx, columns=techs_c, fill_value=0.0) + m_ec_dist.reindex(index=idx, columns=techs_c, fill_value=0.0)
                    m_g_total = m_gc_serv.reindex(index=idx, columns=techs_c, fill_value=0.0) + m_gc_dist.reindex(index=idx, columns=techs_c, fill_value=0.0)
                    m_cool_cost = m_e_total * price_elec + m_g_total * price_gas

                    c_cost_long = m_cool_cost.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                    with cost_c2:
                        st.plotly_chart(_stacked_bar(c_cost_long, "Monthly cost by technology (cooling)", "Costs [€]", color_col="Technology"),
                                        use_container_width=True)
                else:
                    with cost_c2:
                        st.info("No cooling tech cost data.")

    # Export
    st.subheader("Export")
    csv = sim.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(f"Download Scenario {label} CSV", csv, f"scenario_{label}.csv", "text/csv")
