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
def _dt_hours(idx):
    if len(idx) <= 1:
        return pd.Series([1.0], index=idx)
    dt = (idx.to_series().diff().dt.total_seconds().fillna(method="bfill")) / 3600.0
    return dt.reindex(idx)

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

def _tech_display_map(cfg: dict, chain_key: str) -> dict[str, str]:
    """
    Map internal tech 'name' -> display label. Prefer: curve_key > label > name.
    Supports: 'heating_chain', 'cooling_chain', 'dhw'/'dhw_chain'.
    """
    is_central = cfg.get("arch") == "Centralized"

    if chain_key in ("heating_chain", "cooling_chain"):
        chain = (cfg.get("arch_cfg") or {}).get(chain_key, []) if is_central else (cfg.get(chain_key) or [])
    elif chain_key in ("dhw", "dhw_chain"):
        chain = ((cfg.get("dhw") or {}).get("chain") or [])
    else:
        chain = (cfg.get(chain_key) or [])

    disp = {}
    for t in chain:
        internal = str(t.get("name", "")).strip()
        if not internal:
            continue
        pretty = str(t.get("curve_key") or t.get("label") or internal).strip()
        disp[internal] = pretty
    return disp

def _monthly_pertech_total(
    sim: pd.DataFrame,
    prefix_service: str,
    prefix_dist: str,
    cfg: dict,
    chain_key: str,
    include_dist: bool = False,   # default: exclude distribution
) -> pd.DataFrame:
    """
    Monthly totals per technology.

    Returns service-only by default. Set include_dist=True to add distribution.
    """
    # service columns: exact tech series, explicitly exclude *_dist_*
    serv_cols = [
        c for c in sim.columns
        if c.startswith(prefix_service)
        and c.endswith("_kWh")
        and "_dist_" not in c          # ← critical line
    ]

    # dist columns only if requested
    dist_cols = []
    if include_dist and prefix_dist:
        dist_cols = [
            c for c in sim.columns
            if c.startswith(prefix_dist) and c.endswith("_kWh")
        ]

    # monthly frames
    m_serv = sim[serv_cols].rename(
        columns=lambda c: c.removeprefix(prefix_service).removesuffix("_kWh")
    ).resample("M").sum() if serv_cols else pd.DataFrame()

    m_dist = sim[dist_cols].rename(
        columns=lambda c: c.removeprefix(prefix_dist).removesuffix("_kWh")
    ).resample("M").sum() if dist_cols else pd.DataFrame()

    techs = sorted(set(m_serv.columns) | set(m_dist.columns))
    if not techs:
        idx = sim.resample("M").sum().index if len(sim.index) else pd.DatetimeIndex([])
        return pd.DataFrame(index=idx)

    idx = sim.resample("M").sum().index
    m_total = (
        m_serv.reindex(index=idx, columns=techs, fill_value=0.0) +
        m_dist.reindex(index=idx, columns=techs, fill_value=0.0)
    )

    # pretty display names
    disp_map = _tech_display_map(cfg, chain_key)
    if disp_map:
        m_total = m_total.rename(columns=lambda k: disp_map.get(k, k))

    return m_total

# ----------------------------- Main renderer -----------------------------
def render_single(label: str, cfg: dict, sim: pd.DataFrame, view: pd.DataFrame):
    price_elec = float(cfg.get("price_elec", cfg.get("price", 0.30)))
    price_gas  = float(cfg.get("price_gas", 0.10))
    co2_elec   = float(cfg.get("co2_per_kwh_elec", cfg.get("co2_per_kwh", 0.40)))
    co2_gas    = float(cfg.get("co2_per_kwh_gas", 0.20))

    # --- KPIs (robust integration for loads) ---
    dt_h = _dt_hours(view.index)
    
    # Overall efficiencies (service-only, window-wide)
    th_heat_kWh = (view.get("Q_heat_W", pd.Series(0, index=view.index)) * dt_h) / 1000.0
    th_cool_kWh = (view.get("Q_cool_W", pd.Series(0, index=view.index)) * dt_h) / 1000.0
    th_dhw_kWh  = (view.get("Q_dhw_W",  pd.Series(0, index=view.index)) * dt_h) / 1000.0

    Ein_heat = (view.get("E_elec_kWh_heat_service", pd.Series(0, index=view.index)) +
                view.get("E_gas_kWh_heat_service",  pd.Series(0, index=view.index)))
    Ein_cool = (view.get("E_elec_kWh_cool_service", pd.Series(0, index=view.index)) +
                view.get("E_gas_kWh_cool_service",  pd.Series(0, index=view.index)))
    Ein_dhw  = (view.get("E_elec_kWh_dhw_service",  pd.Series(0, index=view.index)) +
                view.get("E_gas_kWh_dhw_service",   pd.Series(0, index=view.index)))

    eta_heat = float(th_heat_kWh.sum()) / float(Ein_heat.sum()) if float(Ein_heat.sum()) > 0 else float("nan")
    eta_cool = float(th_cool_kWh.sum()) / float(Ein_cool.sum()) if float(Ein_cool.sum()) > 0 else float("nan")
    eta_dhw  = float(th_dhw_kWh.sum())  / float(Ein_dhw.sum())  if float(Ein_dhw.sum())  > 0 else float("nan")
    
    heat_kwh = float((view.get("Q_heat_W", pd.Series(0, index=view.index)) * dt_h).sum() / 1000.0)
    dhw_kwh = float((view.get("Q_dhw_W", pd.Series(0, index=view.index)) * dt_h).sum() / 1000.0)
    cool_kwh = float((view.get("Q_cool_W", pd.Series(0, index=view.index)) * dt_h).sum() / 1000.0)
    elec_kwh = float(view.get("E_elec_kWh", pd.Series(0, index=view.index)).sum())
    gas_kwh  = float(view.get("E_gas_kWh",  pd.Series(0, index=view.index)).sum())

    op_co2_house_window = float(view.get("CO2_kg", pd.Series(0, index=view.index)).sum())
    op_cost_house   = elec_kwh * price_elec + gas_kwh * price_gas

    # -------------------- Overview KPIs --------------------
    st.subheader("Results")
    with st.expander("Result Overview", expanded=False):
        n = int(cfg["nr_houses"])

        # Toggle only when we have multiple houses
        show_totals = st.checkbox(
            "All houses", value=False, key=f"all_houses_toggle_{label}"
        ) if n > 1 else False

        mult = n if show_totals else 1
        scope_label = "All houses" if show_totals else ("Per house" if n > 1 else "House")

        # Scale-out values for view mode
        heat_disp = heat_kwh * mult
        cool_disp = cool_kwh * mult
        dhw_disp  = dhw_kwh * mult
        elec_disp = elec_kwh * mult
        gas_disp  = gas_kwh * mult
        op_cost_disp = op_cost_house * mult
        capex_disp = float(cfg.get("capex_house", 0.0)) * mult
        total_cost_disp = (op_cost_house + float(cfg.get("capex_house", 0.0))) * mult

        embodied_disp = float(cfg.get("co2_embodied_kg", 0.0)) * mult
        op_co2_disp = op_co2_house_window * mult
        total_co2_disp = embodied_disp + op_co2_disp

        gas_kwh_per_m3 = float(cfg.get("gas_kwh_per_m3", 9.77))
        gas_m3_disp = (gas_kwh * mult) / gas_kwh_per_m3

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.markdown(f"### {scope_label}")
            st.metric("Heating load [kWh]", f"{heat_disp:.0f}")
            st.metric("DHW load [kWh]", f"{(dhw_disp):.0f}")
            st.metric("Cooling load [kWh]", f"{cool_disp:.0f}")

        with col2:
            st.markdown("### Efficiency")
            def fmt(x):
                return "—" if np.isnan(x) else f"{x:.2f}"
            st.metric("Heating", fmt(eta_heat))
            st.metric("DHW",     fmt(eta_dhw))
            st.metric("Cooling", fmt(eta_cool))
            
        with col3:
            st.markdown("### Energy")
            st.metric("Electricity [kWh]", f"{elec_disp:.0f}")
            st.metric("Gas [kWh]", f"{gas_disp:.0f}", delta=f"{gas_m3_disp:.1f} m³", delta_color="off")

        with col4:
            st.markdown("### Comfort")
            def _num_or_none(x):
                try:
                    x = float(x)
                    return x if np.isfinite(x) else None
                except (TypeError, ValueError):
                    return None

            sp_h = _num_or_none(cfg.get("set_heat", cfg.get("setpoint_heat", 20.0)))
            sp_c = _num_or_none(cfg.get("set_cool", cfg.get("setpoint_cool", 26.0)))
            Tin = view["T_in_pred"]
            tol = float(cfg.get("comfort_tolerance_deg", 0.1))
            minmax_text = f"{Tin.min():.1f} / {Tin.max():.1f}"

            if sp_h is None and sp_c is None:
                st.metric("Min/Max [°C]", minmax_text, delta="No comfort band", delta_color="off")
            else:
                if sp_h is None:  # cooling only
                    in_band = Tin.max() <= sp_c + tol
                    mask_out = Tin > sp_c + tol
                elif sp_c is None:  # heating only
                    in_band = Tin.min() >= sp_h - tol
                    mask_out = Tin < sp_h - tol
                else:  # both
                    in_band = (Tin.min() >= sp_h - tol) and (Tin.max() <= sp_c + tol)
                    mask_out = (Tin < sp_h - tol) | (Tin > sp_c + tol)

                if in_band:
                    st.metric("Min/Max [°C]", minmax_text, delta="Comfort reached")
                else:
                    dt_h = _dt_hours(view.index)
                    hrs_out = float(dt_h[mask_out].sum())
                    hrs_tot = float(dt_h.sum()) if len(dt_h) else 0.0
                    pct_in  = 100.0 * (1.0 - (hrs_out / hrs_tot)) if hrs_tot > 0 else 0.0
                    st.metric("Min/Max [°C]", minmax_text, delta=f"{pct_in:.1f}% met", delta_color="inverse")

            # --- DHW comfort (percent met as value, hours unmet as delta) ---
            dhw_enabled = bool((cfg.get("dhw") or {}).get("enabled", True))
            if dhw_enabled:
                # Demand = delivered + unmet
                q_deliv  = view.get("Q_dhw_W", pd.Series(0.0, index=view.index)).clip(lower=0)
                q_unmet  = view.get("Q_unmet_dhw_W", pd.Series(0.0, index=view.index)).clip(lower=0)
                q_demand = q_deliv + q_unmet

                # Integrate to kWh over the window (you already have dt_h earlier)
                E_deliv  = (q_deliv  * dt_h) / 1000.0
                E_demand = (q_demand * dt_h) / 1000.0

                if float(E_demand.sum()) <= 1e-9:
                    st.metric("Hot water [% met]", "—", delta="No DHW in window", delta_color="off")
                else:
                    pct_met  = 100.0 * float(E_deliv.sum()) / float(E_demand.sum())
                    # only count hours unmet when there was demand
                    hrs_unmet = float(dt_h[(q_unmet > 0) & (q_demand > 0)].sum())

                    if float(q_unmet.sum()) <= 1e-9:
                        st.metric("Hot water [% met]", f"{pct_met:.1f}%", delta="Comfort reached")
                    else:
                        st.metric("Hot water [% met]", f"{pct_met:.2f}%", delta=f"{hrs_unmet:.0f} instances unmet", delta_color="inverse")

        with col5:
            st.markdown("### Emissions")
            st.metric("Capital CO₂ [kg]", f"{embodied_disp:.0f}")
            st.metric("Operational CO₂ [kg]", f"{op_co2_disp:.0f}")
            st.metric("Total CO₂ [kg]", f"{total_co2_disp:.0f}")
                    
        with col6:
            st.markdown("### Costs")
            st.metric("Capital costs [€]", f"{capex_disp:.0f}")
            st.metric("Operational costs [€]", f"{op_cost_disp:.0f}")
            st.metric("Total costs [€]", f"{total_cost_disp:.0f}")

    # -------------------- Plots --------------------
    with st.expander("Plots", expanded=False):
        tab_balance, tab_mix, tab_energy, tab_eff, tab_impact, tab_env = st.tabs(
            ["⚖️ Suppy vs Demand", "🧩 Technology mix", "⚡ Energy use", "📈 Efficiency", "🌍 Impact", "🌤️ Environment"]
        )

        # =============== ⚖️ Balance ===============
        with tab_balance:
            st.caption("Delivered vs unmet demand by service")
            if "E_hvac_kWh" not in sim.columns and "Q_dhw_W" not in sim.columns:
                st.info("No thermal needs available.")
            else:
                # Heating & cooling (existing logic)
                m_heat_deliv = sim.get("E_hvac_kWh", pd.Series(0.0, index=sim.index)).clip(lower=0).resample("M").sum()
                m_cool_deliv = (-sim.get("E_hvac_kWh", pd.Series(0.0, index=sim.index))).clip(lower=0).resample("M").sum()

                if {"Q_unmet_heat_W_rc", "Q_unmet_cool_W_rc"}.issubset(sim.columns):
                    dt_h = _dt_hours(sim.index)
                    m_heat_unmet = (sim["Q_unmet_heat_W_rc"] * dt_h / 1000.0).resample("M").sum()
                    m_cool_unmet = (sim["Q_unmet_cool_W_rc"] * dt_h / 1000.0).resample("M").sum()
                else:
                    m_heat_unmet = m_heat_deliv * 0.0
                    m_cool_unmet = m_cool_deliv * 0.0

                heat_long = pd.DataFrame({
                    "time": m_heat_deliv.index,
                    "Delivered": m_heat_deliv.values,
                    "Unmet": m_heat_unmet.reindex(m_heat_deliv.index, fill_value=0).values,
                }).melt(id_vars="time", var_name="Component", value_name="kWh")

                cool_long = pd.DataFrame({
                    "time": m_cool_deliv.index,
                    "Delivered": m_cool_deliv.values,
                    "Unmet": m_cool_unmet.reindex(m_cool_deliv.index, fill_value=0).values,
                }).melt(id_vars="time", var_name="Component", value_name="kWh")

                # --- NEW: DHW delivered vs unmet ---
                dt_h = _dt_hours(sim.index)
                Q_dhw_W = sim.get("Q_dhw_W", pd.Series(0.0, index=sim.index)).fillna(0.0)
                m_dhw_deliv = (Q_dhw_W * dt_h / 1000.0).resample("M").sum()
                if "Q_unmet_dhw_W" in sim.columns:
                    m_dhw_unmet = (sim["Q_unmet_dhw_W"] * dt_h / 1000.0).resample("M").sum()
                else:
                    m_dhw_unmet = m_dhw_deliv * 0.0

                dhw_long = pd.DataFrame({
                    "time": m_dhw_deliv.index,
                    "Delivered": m_dhw_deliv.values,
                    "Unmet": m_dhw_unmet.reindex(m_dhw_deliv.index, fill_value=0).values,
                }).melt(id_vars="time", var_name="Component", value_name="kWh")

                # Layout: two charts top (heat/cool), one full-width bottom (DHW)
                col_h, col_c = st.columns(2)
                with col_h:
                    fig_h = px.bar(heat_long, x="time", y="kWh", color="Component", barmode="stack",
                                labels={"time": "Month", "kWh": "Energy [kWh]"})
                    fig_h.update_layout(title="Heating — Delivered vs Unmet", height=420, hovermode="x unified")
                    st.plotly_chart(fig_h, use_container_width=True)

                with col_c:
                    fig_c = px.bar(cool_long, x="time", y="kWh", color="Component", barmode="stack",
                                labels={"time": "Month", "kWh": "Energy [kWh]"})
                    fig_c.update_layout(title="Cooling — Delivered vs Unmet", height=420, hovermode="x unified")
                    st.plotly_chart(fig_c, use_container_width=True)

                fig_d = px.bar(dhw_long, x="time", y="kWh", color="Component", barmode="stack",
                            labels={"time": "Month", "kWh": "Energy [kWh]"})
                fig_d.update_layout(title="DHW — Delivered vs Unmet", height=420, hovermode="x unified")
                st.plotly_chart(fig_d, use_container_width=True)

        # =============== 🧩 Technology mix ===============
        with tab_mix:
            st.caption("Per-technology thermal dispatch (stacked)")
            # --- Per-tech heating (stacked) ---
            heat_cols = [c for c in view.columns if c.startswith("Q_heat_") and c.endswith("_W")]
            heat_cols = [c for c in heat_cols if c not in ("Q_heat_W", "Q_unmet_heat_W")]
            if heat_cols:
                h_map = {c: c.removeprefix("Q_heat_").removesuffix("_W") for c in heat_cols}
                v_heat = view[heat_cols].rename(columns=h_map)

                # NEW: apply pretty names
                heat_disp = _tech_display_map(cfg, "heating_chain")
                if heat_disp:
                    v_heat = v_heat.rename(columns=lambda k: heat_disp.get(k, k))

                if v_heat.columns.duplicated().any():
                    st.error("⚠️ Duplicate heating technologies detected. Please choose a different secondary heating technology.")
                else:
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

            # --- Per-tech DHW (stacked) ---
            dhw_cols = [c for c in view.columns if c.startswith("Q_dhw_") and c.endswith("_W")]
            dhw_cols = [c for c in dhw_cols if c not in ("Q_dhw_W", "Q_unmet_dhw_W")]
            
            
            # --- DHW (daily vs yearly) ---
            dhw_enabled = bool((cfg.get("dhw") or {}).get("enabled", True))
            if dhw_enabled and "Q_dhw_W" in view.columns and view["Q_dhw_W"].abs().sum() > 0:
                # integration step (h)
                dt_h = _dt_hours(view.index)

                # ===== Left: single day profile (W vs time) =====
                tz = view.index.tz
                year0 = int(view.index[0].year)
                first_day = pd.Timestamp(year0, 1, 1, tz=tz).normalize()

                idx_norm = view.index.normalize()
                if (idx_norm == first_day).any():
                    day = first_day
                else:
                    mask_year = view.index.year == year0
                    day = view.index[mask_year][0].normalize() if mask_year.any() else idx_norm[0]

                day_mask = idx_norm == day
                s_day = view.loc[day_mask, "Q_dhw_W"]

                # ===== Right: yearly profile as daily energy (kWh/day) =====
                c_left, c_right = st.columns(2)

                with c_left:
                    if not s_day.empty:
                        df_day = s_day.rename("DHW total [W]").rename_axis("time").reset_index()
                        fig_dhw_total = px.line(
                            df_day, x="time", y="DHW total [W]",
                            labels={"DHW total [W]": "Power [W]", "time": "Time"},
                            color_discrete_map={"DHW total [W]": COLORS.get("Q_dhw_W", "#ff7f0e")},
                        )
                        fig_dhw_total.update_traces(line=dict(width=2))
                        fig_dhw_total.update_layout(
                            title=f"DHW daily profile — {day.strftime('%Y-%m-%d')}",
                            hovermode="x unified",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_dhw_total, use_container_width=True)
                    else:
                        st.info("No DHW data for the selected day.")

                with c_right:
                    if dhw_enabled and dhw_cols:
                        d_map = {c: c.removeprefix("Q_dhw_").removesuffix("_W") for c in dhw_cols}
                        v_dhw = view[dhw_cols].rename(columns=d_map)

                        dhw_disp = _tech_display_map(cfg, "dhw")
                        if dhw_disp:
                            v_dhw = v_dhw.rename(columns=lambda k: dhw_disp.get(k, k))

                        if v_dhw.columns.duplicated().any():
                            st.error("⚠️ Duplicate DHW technologies detected. Please rename or change selection.")
                        else:
                            fig_dstack = px.area(
                                v_dhw.reset_index().melt(id_vars="time", var_name="Technology", value_name="W"),
                                x="time", y="W", color="Technology",
                                labels={"W": "Thermal power [W]", "time": "Time"},
                            )
                            # remove outlines like heating/cooling
                            fig_dstack.update_traces(line=dict(width=0))
                            fig_dstack.update_layout(title="DHW per technology (stacked)", hovermode="x unified")
                            st.plotly_chart(fig_dstack, use_container_width=True)

            # --- Per-tech cooling (stacked) ---
            cool_cols = [c for c in view.columns if c.startswith("Q_cool_") and c.endswith("_W")]
            cool_cols = [c for c in cool_cols if c not in ("Q_cool_W", "Q_unmet_cool_W")]
            if cool_cols:
                c_map = {c: c.removeprefix("Q_cool_").removesuffix("_W") for c in cool_cols}
                v_cool = view[cool_cols].rename(columns=c_map)

                # NEW: apply pretty names
                cool_disp = _tech_display_map(cfg, "cooling_chain")
                if cool_disp:
                    v_cool = v_cool.rename(columns=lambda k: cool_disp.get(k, k))

                if v_cool.columns.duplicated().any():
                    st.error("⚠️ Duplicate cooling technologies detected. Please choose a different secondary cooling technology.")
                else:
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

        # =============== ⚡ Energy use ===============
        with tab_energy:
            view_energy = st.radio(
                "View",
                ["⚡ Electricity", "🛢️ Gas"],
                horizontal=True,
                key=f"energy_view_{label}"
            )

            def _monthly_sum(sim: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
                present = [c for c in cols if c in sim.columns]
                if not present:
                    return pd.DataFrame(index=sim.resample("M").sum().index if len(sim.index) else [])
                return sim[present].resample("M").sum()

            st.caption("Monthly energy by source and by technology")

            if view_energy == "⚡ Electricity":
                # ---- by source
                elec_src_cols = {
                    "Cooling (service)": "E_elec_kWh_cool_service",
                    "Cooling losses":    "E_elec_kWh_dist_cool",
                    "Heating (service)": "E_elec_kWh_heat_service",
                    "Heating losses":    "E_elec_kWh_dist_heat",
                    "DHW (service)":     "E_elec_kWh_dhw_service",
                    "DHW losses":        "E_elec_kWh_dist_dhw",
                    "Auxiliaries":       "E_elec_kWh_aux",
                }
                m_elec_src = _monthly_sum(sim, list(elec_src_cols.values()))
                if not m_elec_src.empty:
                    m_elec_src = m_elec_src.rename(columns={v:k for k,v in elec_src_cols.items() if v in m_elec_src.columns})
                    elec_src_long = m_elec_src.reset_index().melt(id_vars="time", var_name="Category", value_name="Value")
                    st.plotly_chart(_stacked_bar(elec_src_long, "Monthly electricity by source", "Energy [kWh]"),
                                    use_container_width=True)
                else:
                    st.info("No electricity source components found.")

                st.divider()
                st.markdown("#### Electricity by technology (service + distribution)")
                elec_c1, elec_c2 = st.columns(2)

                # heating tech
                m_total_heat = _monthly_pertech_total(sim, "E_elec_heat_", "E_elec_heat_dist_", cfg, "heating_chain", include_dist=True)
                with elec_c1:
                    if not m_total_heat.empty:
                        ht_long = m_total_heat.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                        st.plotly_chart(_stacked_bar(ht_long, "Heating — electricity by technology", "Energy [kWh]", "Technology"),
                                        use_container_width=True)
                    else:
                        st.info("No heating electricity by technology found.")

                # cooling tech
                m_total_cool = _monthly_pertech_total(sim, "E_elec_cool_", "E_elec_cool_dist_", cfg, "cooling_chain", include_dist=True)
                with elec_c2:
                    if not m_total_cool.empty:
                        ct_long = m_total_cool.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                        st.plotly_chart(_stacked_bar(ct_long, "Cooling — electricity by technology", "Energy [kWh]", "Technology"),
                                        use_container_width=True)
                    else:
                        st.info("No cooling electricity by technology found.")

                # DHW tech
                m_elec_dhw = _monthly_pertech_total(sim, "E_elec_dhw_", "E_elec_dhw_dist_", cfg, "dhw", include_dist=True)
                if not m_elec_dhw.empty:
                    d_elec_long = m_elec_dhw.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                    st.plotly_chart(_stacked_bar(d_elec_long, "DHW — electricity by technology", "Energy [kWh]", "Technology"),
                                    use_container_width=True)
                else:
                    st.info("No DHW electricity by technology found.")

            else:  # 🛢️ Gas
                gas_src_cols = {
                    "Heating (service)": "E_gas_kWh_heat_service",
                    "Heating losses":    "E_gas_kWh_dist_heat",
                    "DHW (service)":     "E_gas_kWh_dhw_service",
                    "DHW losses":        "E_gas_kWh_dist_dhw",
                }
                m_gas_src = _monthly_sum(sim, list(gas_src_cols.values()))
                if not m_gas_src.empty:
                    m_gas_src = m_gas_src.rename(columns={v:k for k,v in gas_src_cols.items() if v in m_gas_src.columns})
                    gas_src_long = m_gas_src.reset_index().melt(id_vars="time", var_name="Category", value_name="Value")
                    st.plotly_chart(_stacked_bar(gas_src_long, "Monthly gas by source", "Energy [kWh]"),
                                    use_container_width=True)
                else:
                    st.info("No gas source components found.")

                st.divider()
                st.markdown("#### Gas by technology (service + distribution)")
                # heating tech
                m_gas_heat = _monthly_pertech_total(sim, "E_gas_heat_", "E_gas_heat_dist_", cfg, "heating_chain", include_dist=True)
                if not m_gas_heat.empty:
                    h_long = m_gas_heat.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                    st.plotly_chart(_stacked_bar(h_long, "Heating — gas by technology", "Energy [kWh]", "Technology"),
                                    use_container_width=True)
                else:
                    st.info("No heating gas by technology found.")

                # DHW tech
                m_gas_dhw = _monthly_pertech_total(sim, "E_gas_dhw_", "E_gas_dhw_dist_", cfg, "dhw", include_dist=True)
                if not m_gas_dhw.empty:
                    d_long = m_gas_dhw.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                    st.plotly_chart(_stacked_bar(d_long, "DHW — gas by technology", "Energy [kWh]", "Technology"),
                                    use_container_width=True)
                else:
                    st.info("No DHW gas by technology found.")

        # =============== 📈 Efficiency ===============
        with tab_eff:
            st.caption("Overall and per-component efficiencies")
            eff_overall_tab, eff_comp_tab = st.tabs(["Overall", "Components"])

            # Config & helpers
            dhw_enabled = bool((cfg.get("dhw") or {}).get("enabled", True))
            # eps_W = float(cfg.get("eff_plot_threshold_W", 50.0))  # small noise threshold
            dt_h = _dt_hours(view.index)
            th_heat_kWh = (view.get("Q_heat_W", pd.Series(0.0, index=view.index)) * dt_h) / 1000.0
            th_cool_kWh = (view.get("Q_cool_W", pd.Series(0.0, index=view.index)) * dt_h) / 1000.0
            th_dhw_kWh  = (view.get("Q_dhw_W",  pd.Series(0.0, index=view.index)) * dt_h) / 1000.0

            def _monthly_efficiency(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
                """Energy-weighted monthly efficiency = sum(out) / sum(in) per month."""
                df = pd.DataFrame({"num": numerator.fillna(0.0), "den": denominator.fillna(0.0)})
                m = df.resample("M").sum(min_count=1)
                return (m["num"] / m["den"]).replace([np.inf, -np.inf], np.nan)

            # Build (tech_name -> {fuel, pretty}) from cfg
            def _tech_meta(cfg: dict, chain_key: str) -> dict[str, dict]:
                is_central = cfg.get("arch") == "Centralized"
                chain = (cfg.get("arch_cfg") or {}).get(chain_key, []) if is_central else (cfg.get(chain_key) or [])
                meta = {}
                for t in chain:
                    name = str(t.get("name", "")).strip()
                    if not name:
                        continue
                    pretty = str(t.get("curve_key") or t.get("label") or name).strip()
                    fuel = (t.get("fuel") or "elec").lower().strip()
                    meta[name] = {"fuel": fuel, "pretty": pretty}
                return meta

            with eff_overall_tab:
                st.caption("Per-technology thermal dispatch (stacked)")
                # Inputs (service-only, consistent with your HEAT_eff_overall / COOL_eff_overall)
                Ein_heat = (view.get("E_elec_kWh_heat_service", pd.Series(0.0, index=view.index)) +
                            view.get("E_gas_kWh_heat_service",  pd.Series(0.0, index=view.index)))
                Ein_cool = (view.get("E_elec_kWh_cool_service", pd.Series(0.0, index=view.index)) +
                            view.get("E_gas_kWh_cool_service",  pd.Series(0.0, index=view.index)))
                Ein_dhw  = (view.get("E_elec_kWh_dhw_service",  pd.Series(0.0, index=view.index)) +
                            view.get("E_gas_kWh_dhw_service",   pd.Series(0.0, index=view.index)))

                m_eff = {}
                # Heating overall (can exceed 1 if HPs dominate)
                m_eff["η (heating overall)"] = _monthly_efficiency(th_heat_kWh, Ein_heat)
                # Cooling overall (usually == electric EER with your current tech set)
                m_eff["η (cooling overall)"] = _monthly_efficiency(th_cool_kWh, Ein_cool)
                # DHW overall if DHW enabled
                if bool((cfg.get("dhw") or {}).get("enabled", True)):
                    m_eff["η (DHW overall)"] = _monthly_efficiency(th_dhw_kWh, Ein_dhw)

                dfm = pd.DataFrame(m_eff).dropna(how="all")
                if not dfm.empty:
                    v = dfm.reset_index(names="time").melt(id_vars="time", var_name="Metric", value_name="Value").dropna(subset=["Value"])
                    fig = px.bar(v, x="time", y="Value", color="Metric", barmode="group",
                                labels={"Value": "Overall efficiency (kWh_th per kWh_in)", "time": "Month"})
                    fig.update_traces(marker_line_width=0)
                    fig.update_layout(title="Overall system efficiencies — monthly", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No overall monthly efficiencies available.")

            with eff_comp_tab:
                st.caption("Component efficiencies per technology (monthly)")

                dt_h = _dt_hours(view.index)

                def _pertech_monthly_eff(
                    sim: pd.DataFrame,
                    q_prefix: str,                       # e.g. "Q_heat_", "Q_cool_", "Q_dhw_"
                    elec_prefix: str,                    # e.g. "E_elec_heat_", "E_elec_cool_", "E_elec_dhw_"
                    gas_prefix: str,                     # e.g. "E_gas_heat_",  "E_gas_cool_",  "E_gas_dhw_"
                    chain_key: str,                      # "heating_chain" | "cooling_chain" | "dhw"
                    cfg_for_names: dict | None = None,   # optional when using DHW's custom map
                ) -> pd.DataFrame:
                    """
                    Return wide DataFrame: index = month end, columns = pretty tech names,
                    values = monthly efficiency (sum Out kWh / sum In kWh).
                    """
                    # 1) Collect per-tech Q columns present
                    q_cols = [c for c in sim.columns if c.startswith(q_prefix) and c.endswith("_W")]
                    # exclude totals if present
                    q_cols = [c for c in q_cols if c not in (f"{q_prefix[:-1]}_W", f"{q_prefix}W")]

                    if not q_cols:
                        return pd.DataFrame()

                    # Map tech internal -> pretty display
                    disp_map = _tech_display_map(cfg if cfg_for_names is None else cfg_for_names, chain_key)

                    # 2) Build per-tech thermal OUT (kWh) monthly
                    #    Take each tech's Q_W series, integrate to kWh, then resample to months
                    m_out = {}
                    for qc in q_cols:
                        tech = qc.removeprefix(q_prefix).removesuffix("_W")
                        qW = sim.get(qc)
                        if qW is None:
                            continue
                        e_kWh = (qW.fillna(0.0) * dt_h.fillna(0.0)) / 1000.0
                        m_out[tech] = e_kWh.resample("M").sum(min_count=1)

                    if not m_out:
                        return pd.DataFrame()

                    m_out = pd.DataFrame(m_out)

                    # 3) Per-tech electric & gas input columns (service only)
                    def _find_inputs(prefix: str) -> pd.DataFrame:
                        cols = [c for c in sim.columns if c.startswith(prefix) and c.endswith("_kWh")]
                        # strip prefix/suffix to tech name
                        data = {}
                        for c in cols:
                            tech = c.removeprefix(prefix).removesuffix("_kWh")
                            data[tech] = sim[c]
                        return pd.DataFrame(data) if data else pd.DataFrame(index=sim.index)

                    Ein_e = _find_inputs(elec_prefix)
                    Ein_g = _find_inputs(gas_prefix)
                    Ein   = (Ein_e.reindex(index=sim.index, columns=m_out.columns, fill_value=0.0) +
                            Ein_g.reindex(index=sim.index, columns=m_out.columns, fill_value=0.0))

                    m_in = Ein.resample("M").sum(min_count=1).reindex(index=m_out.resample("M").sum(min_count=1).index)

                    # 4) Efficiency = monthly_out / monthly_in
                    eff = (m_out / m_in).replace([np.inf, -np.inf], np.nan)

                    # 5) Pretty names
                    if disp_map:
                        eff = eff.rename(columns=lambda k: disp_map.get(k, k))

                    # Drop all-NaN columns
                    eff = eff.dropna(axis=1, how="all")
                    return eff

                tab_h, tab_c, tab_d = st.tabs(["Heating components", "Cooling components", "DHW components"])

                # ----- HEATING -----
                with tab_h:
                    eff_h = _pertech_monthly_eff(
                        sim=view,
                        q_prefix="Q_heat_",
                        elec_prefix="E_elec_heat_",
                        gas_prefix="E_gas_heat_",
                        chain_key="heating_chain",
                    )
                    if not eff_h.empty:
                        v = eff_h.reset_index(names="time").melt(id_vars="time", var_name="Component", value_name="Value").dropna(subset=["Value"])
                        fig = px.bar(
                            v, x="time", y="Value", color="Component", barmode="group",
                            labels={"Value": "Monthly COP / η", "time": "Month"}
                        )
                        fig.update_traces(marker_line_width=0)
                        fig.update_layout(title="Heating — per-component efficiency (monthly)", hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No per-component heating efficiencies available.")

                # ----- COOLING -----
                with tab_c:
                    eff_c = _pertech_monthly_eff(
                        sim=view,
                        q_prefix="Q_cool_",
                        elec_prefix="E_elec_cool_",
                        gas_prefix="E_gas_cool_",   # kept for completeness if you add gas cooling later
                        chain_key="cooling_chain",
                    )
                    if not eff_c.empty:
                        v = eff_c.reset_index(names="time").melt(id_vars="time", var_name="Component", value_name="Value").dropna(subset=["Value"])
                        fig = px.bar(
                            v, x="time", y="Value", color="Component", barmode="group",
                            labels={"Value": "Monthly EER", "time": "Month"}
                        )
                        fig.update_traces(marker_line_width=0)
                        fig.update_layout(title="Cooling — per-component efficiency (monthly)", hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No per-component cooling efficiencies available.")

                # ----- DHW -----
                with tab_d:
                    # DHW tech names come from cfg["dhw"]["chain"]
                    eff_d = _pertech_monthly_eff(
                        sim=view,
                        q_prefix="Q_dhw_",
                        elec_prefix="E_elec_dhw_",
                        gas_prefix="E_gas_dhw_",
                        chain_key="dhw",
                    )
                    if not eff_d.empty:
                        v = eff_d.reset_index(names="time").melt(id_vars="time", var_name="Component", value_name="Value").dropna(subset=["Value"])
                        fig = px.bar(
                            v, x="time", y="Value", color="Component", barmode="group",
                            labels={"Value": "Monthly COP / η", "time": "Month"}
                        )
                        fig.update_traces(marker_line_width=0)
                        fig.update_layout(title="DHW — per-component efficiency (monthly)", hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No per-component DHW efficiencies available.")

        # =============== 🌍 Impact ===============
        with tab_impact:
            mode = st.radio("View", ["CO₂", "Costs"], horizontal=True, key=f"impact_mode_{label}")
            if mode == "CO₂":
                st.markdown("#### CO₂ by source")

                elec_src_cols = {
                    "Elec Cooling (service)": "E_elec_kWh_cool_service",
                    "Elec Cooling losses":    "E_elec_kWh_dist_cool",
                    "Elec Heating (service)": "E_elec_kWh_heat_service",
                    "Elec Heating losses":    "E_elec_kWh_dist_heat",
                    "Elec DHW (service)":     "E_elec_kWh_dhw_service",
                    "Elec DHW losses":        "E_elec_kWh_dist_dhw",
                }
                gas_src_cols = {
                    "Gas Heating (service)":  "E_gas_kWh_heat_service",
                    "Gas Heating losses":     "E_gas_kWh_dist_heat",
                    "Gas DHW (service)":      "E_gas_kWh_dhw_service",
                    "Gas DHW losses":         "E_gas_kWh_dist_dhw",
                    "Auxiliaries":            "E_elec_kWh_aux",
                }

                m_elec_src_E = _monthly_sum(sim, list(elec_src_cols.values()))
                m_gas_src_E  = _monthly_sum(sim, list(gas_src_cols.values()))

                frames = []
                co2_elec = float(cfg.get("co2_per_kwh_elec", cfg.get("co2_per_kwh", 0.40)))
                co2_gas  = float(cfg.get("co2_per_kwh_gas", 0.20))

                if not m_elec_src_E.empty:
                    m_elec_src_E = m_elec_src_E.rename(columns={v: k for k, v in elec_src_cols.items() if v in m_elec_src_E.columns})
                    frames.append(m_elec_src_E * co2_elec)
                if not m_gas_src_E.empty:
                    m_gas_src_E = m_gas_src_E.rename(columns={v: k for k, v in gas_src_cols.items() if v in m_gas_src_E.columns})
                    frames.append(m_gas_src_E * co2_gas)

                if frames:
                    msrc_co2 = pd.concat(frames, axis=1).fillna(0.0)
                    co2_long = msrc_co2.reset_index().melt(id_vars="time", var_name="Category", value_name="Value")
                    st.plotly_chart(_stacked_bar(co2_long, "Monthly CO₂ by source", "CO₂ [kg]"), use_container_width=True)
                else:
                    st.info("No CO₂ source components found.")

                st.divider()
                st.markdown("#### CO₂ by technology (service + distribution)")
                # Two plots side-by-side (heating/cooling) + DHW below
                top_left, top_right = st.columns(2)

                # ===== HEATING =====
                m_elec_heat = _monthly_pertech_total(sim, "E_elec_heat_", "E_elec_heat_dist_", cfg, "heating_chain", include_dist=True)
                m_gas_heat  = _monthly_pertech_total(sim, "E_gas_heat_",  "E_gas_heat_dist_",  cfg, "heating_chain", include_dist=True)
                if not (m_elec_heat.empty and m_gas_heat.empty):
                    idx   = (m_elec_heat.index if not m_elec_heat.empty else m_gas_heat.index)
                    techs = sorted(set(m_elec_heat.columns) | set(m_gas_heat.columns))
                    m_heat_co2 = m_elec_heat.reindex(index=idx, columns=techs, fill_value=0.0) * co2_elec \
                            + m_gas_heat.reindex(index=idx,  columns=techs, fill_value=0.0) * co2_gas
                    h_long = m_heat_co2.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                    with top_left:
                        st.plotly_chart(
                            _stacked_bar(h_long, "Monthly CO₂ by technology (heating)", "CO₂ [kg]", color_col="Technology"),
                            use_container_width=True,
                        )
                else:
                    with top_left:
                        st.info("No heating CO₂ by technology found.")

                # ===== COOLING =====
                m_elec_cool = _monthly_pertech_total(sim, "E_elec_cool_", "E_elec_cool_dist_", cfg, "cooling_chain", include_dist=True)
                m_gas_cool  = _monthly_pertech_total(sim, "E_gas_cool_",  "E_gas_cool_dist_",  cfg, "cooling_chain", include_dist=True)
                if not (m_elec_cool.empty and m_gas_cool.empty):
                    idx   = (m_elec_cool.index if not m_elec_cool.empty else m_gas_cool.index)
                    techs = sorted(set(m_elec_cool.columns) | set(m_gas_cool.columns))
                    m_cool_co2 = m_elec_cool.reindex(index=idx, columns=techs, fill_value=0.0) * co2_elec \
                            + m_gas_cool.reindex(index=idx,  columns=techs, fill_value=0.0) * co2_gas
                    c_long = m_cool_co2.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                    with top_right:
                        st.plotly_chart(
                            _stacked_bar(c_long, "Monthly CO₂ by technology (cooling)", "CO₂ [kg]", color_col="Technology"),
                            use_container_width=True,
                        )
                else:
                    with top_right:
                        st.info("No cooling CO₂ by technology found.")

                # ===== DHW (full width below) =====
                m_elec_dhw = _monthly_pertech_total(sim, "E_elec_dhw_", "E_elec_dhw_dist_", cfg, "dhw", include_dist=True)
                m_gas_dhw  = _monthly_pertech_total(sim, "E_gas_dhw_",  "E_gas_dhw_dist_",  cfg, "dhw", include_dist=True)
                if not (m_elec_dhw.empty and m_gas_dhw.empty):
                    idx   = (m_elec_dhw.index if not m_elec_dhw.empty else m_gas_dhw.index)
                    techs = sorted(set(m_elec_dhw.columns) | set(m_gas_dhw.columns))
                    m_dhw_co2 = m_elec_dhw.reindex(index=idx, columns=techs, fill_value=0.0) * co2_elec \
                            + m_gas_dhw.reindex(index=idx,  columns=techs, fill_value=0.0) * co2_gas
                    d_long = m_dhw_co2.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                    st.plotly_chart(
                        _stacked_bar(d_long, "Monthly CO₂ by technology (DHW)", "CO₂ [kg]", color_col="Technology"),
                        use_container_width=True,
                    )
                else:
                    st.info("No DHW CO₂ by technology found.")

            else:
                st.markdown("#### Costs by source")
                # Prices
                price_elec = float(cfg.get("price_elec", cfg.get("price", 0.30)))
                price_gas  = float(cfg.get("price_gas", 0.10))

                # Energy-by-source (monthly)
                elec_src_cols = {
                    "Elec Cooling (service)": "E_elec_kWh_cool_service",
                    "Elec Cooling losses":    "E_elec_kWh_dist_cool",
                    "Elec Heating (service)": "E_elec_kWh_heat_service",
                    "Elec Heating losses":    "E_elec_kWh_dist_heat",
                    "Elec DHW (service)":     "E_elec_kWh_dhw_service",
                    "Elec DHW losses":        "E_elec_kWh_dist_dhw",
                }
                gas_src_cols = {
                    "Gas Heating (service)":  "E_gas_kWh_heat_service",
                    "Gas Heating losses":     "E_gas_kWh_dist_heat",
                    "Gas DHW (service)":      "E_gas_kWh_dhw_service",
                    "Gas DHW losses":         "E_gas_kWh_dist_dhw",
                    "Auxiliaries":            "E_elec_kWh_aux",
                }

                m_elec_src_E = _monthly_sum(sim, list(elec_src_cols.values()))
                m_gas_src_E  = _monthly_sum(sim, list(gas_src_cols.values()))

                frames_cost = []
                if not m_elec_src_E.empty:
                    m_elec_src_E = m_elec_src_E.rename(columns={v: k for k, v in elec_src_cols.items() if v in m_elec_src_E.columns})
                    frames_cost.append(m_elec_src_E * price_elec)
                if not m_gas_src_E.empty:
                    m_gas_src_E = m_gas_src_E.rename(columns={v: k for k, v in gas_src_cols.items() if v in m_gas_src_E.columns})
                    frames_cost.append(m_gas_src_E * price_gas)

                if frames_cost:
                    msrc_cost = pd.concat(frames_cost, axis=1).fillna(0.0)
                    cost_long = msrc_cost.reset_index().melt(id_vars="time", var_name="Category", value_name="Value")
                    st.plotly_chart(_stacked_bar(cost_long, "Monthly cost by source", "Costs [€]"), use_container_width=True)
                else:
                    st.info("No cost source components found.")
                
                st.divider()
                st.markdown("#### Costs by technology (service + distribution)")
                cost_c1, cost_c2 = st.columns(2)

                # ===== HEATING =====
                m_elec_heat = _monthly_pertech_total(sim, "E_elec_heat_", "E_elec_heat_dist_", cfg, "heating_chain", include_dist=True)
                m_gas_heat  = _monthly_pertech_total(sim, "E_gas_heat_",  "E_gas_heat_dist_",  cfg, "heating_chain", include_dist=True)

                if m_elec_heat.columns.duplicated().any() or m_gas_heat.columns.duplicated().any():
                    st.error("⚠️ Duplicate heating technologies detected. Please choose a different secondary heating technology.")
                else:
                    if not (m_elec_heat.empty and m_gas_heat.empty):
                        idx   = (m_elec_heat.index if not m_elec_heat.empty else m_gas_heat.index)
                        techs = sorted(set(m_elec_heat.columns) | set(m_gas_heat.columns))
                        m_heat_cost = m_elec_heat.reindex(index=idx, columns=techs, fill_value=0.0) * price_elec \
                                    + m_gas_heat.reindex(index=idx,  columns=techs, fill_value=0.0) * price_gas

                        h_cost_long = m_heat_cost.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                        with cost_c1:
                            st.plotly_chart(
                                _stacked_bar(h_cost_long, "Monthly cost by technology (heating)", "Costs [€]", color_col="Technology"),
                                use_container_width=True,
                            )
                    else:
                        with cost_c1:
                            st.info("No heating tech cost data.")

                # ===== COOLING =====
                m_elec_cool = _monthly_pertech_total(sim, "E_elec_cool_", "E_elec_cool_dist_", cfg, "cooling_chain", include_dist=True)
                m_gas_cool  = _monthly_pertech_total(sim, "E_gas_cool_",  "E_gas_cool_dist_",  cfg, "cooling_chain", include_dist=True)

                if m_elec_cool.columns.duplicated().any() or m_gas_cool.columns.duplicated().any():
                    st.error("⚠️ Duplicate cooling technologies detected. Please choose a different secondary cooling technology.")
                else:
                    if not (m_elec_cool.empty and m_gas_cool.empty):
                        idx   = (m_elec_cool.index if not m_elec_cool.empty else m_gas_cool.index)
                        techs = sorted(set(m_elec_cool.columns) | set(m_gas_cool.columns))
                        m_cool_cost = m_elec_cool.reindex(index=idx, columns=techs, fill_value=0.0) * price_elec \
                                    + m_gas_cool.reindex(index=idx,  columns=techs, fill_value=0.0) * price_gas

                        c_cost_long = m_cool_cost.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                        with cost_c2:
                            st.plotly_chart(
                                _stacked_bar(c_cost_long, "Monthly cost by technology (cooling)", "Costs [€]", color_col="Technology"),
                                use_container_width=True,
                            )
                    else:
                        with cost_c2:
                            st.info("No cooling tech cost data.")

                st.divider()
                st.markdown("#### Cost by technology (DHW)")
                # Include distribution if you emit per-tech dist; helper will add it when columns exist
                m_elec_dhw = _monthly_pertech_total(sim, "E_elec_dhw_", "E_elec_dhw_dist_", cfg, "dhw", include_dist=True)
                m_gas_dhw  = _monthly_pertech_total(sim, "E_gas_dhw_",  "E_gas_dhw_dist_",  cfg, "dhw", include_dist=True)

                if not (m_elec_dhw.empty and m_gas_dhw.empty):
                    idx   = (m_elec_dhw.index if not m_elec_dhw.empty else m_gas_dhw.index)
                    techs = sorted(set(m_elec_dhw.columns) | set(m_gas_dhw.columns))

                    m_dhw_cost = m_elec_dhw.reindex(index=idx, columns=techs, fill_value=0.0) * price_elec \
                            + m_gas_dhw.reindex(index=idx,  columns=techs, fill_value=0.0) * price_gas

                    d_cost_long = m_dhw_cost.reset_index().melt(id_vars="time", var_name="Technology", value_name="Value")
                    st.plotly_chart(
                        _stacked_bar(d_cost_long, "Monthly cost by technology (DHW)", "Costs [€]", color_col="Technology"),
                        use_container_width=True,
                    )
                else:
                    st.info("No DHW tech cost data.")

        # =============== 🌤️ Environment ===============
        with tab_env:

            st.caption("Temperature profiles")
            fig_t = px.line(
                view.reset_index(), x="time", y=["T_out", "T_in_pred", "T_wall_pred"],
                labels={"value": "Temperature [°C]", "time": "Time", "variable": "Series"},
                color_discrete_map=COLORS,
            )
            fig_t.update_layout(hovermode="x unified")
            st.plotly_chart(fig_t, use_container_width=True)

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
                fig_ctx.update_layout(title="Energy flows", hovermode="x unified")
                st.plotly_chart(fig_ctx, use_container_width=True)
    
    # Export
    st.subheader("Export")
    csv = sim.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(f"Download Scenario {label} CSV", csv, f"scenario_{label}.csv", "text/csv")
