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
Rendering logic for the **multi-scenario comparison** view.

Features
--------
- Select comparison horizon: Selected window / Full-year / Lifecycle
- Computes per-scenario KPIs (costs, CO₂, SCOP/SEER)
- Builds cumulative cost & CO₂ curves (time- or year-indexed)
- Pareto plot (Total € vs Total CO₂) and cost breakdown
- CSV export of cumulative totals
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from services.metrics import scop_seer

def _sum_or_zero(df: pd.DataFrame, col: str) -> float:
    return float(df[col].sum()) if col in df.columns else 0.0

def _get_prices_co2(cfg: dict):
    price_elec = float(cfg.get("price_elec", cfg.get("price", 0.30)))
    price_gas  = float(cfg.get("price_gas", 0.10))
    co2_elec   = float(cfg.get("co2_per_kwh_elec", cfg.get("co2_per_kwh", 0.40)))
    co2_gas    = float(cfg.get("co2_per_kwh_gas", 0.20))
    return price_elec, price_gas, co2_elec, co2_gas

def _component_kwh(df: pd.DataFrame):
    """
    Return kWh components tolerant to missing cols.
    For decentralized, 'base' falls back to service terms.
    """
    # electricity
    e_base = _sum_or_zero(df, "E_elec_kWh_base")
    if e_base == 0.0:
        e_base = (
            _sum_or_zero(df, "E_elec_kWh_heat_service") +
            _sum_or_zero(df, "E_elec_kWh_cool_service")
        )
    e_dloss = (
        _sum_or_zero(df, "E_elec_kWh_dist_heat") +
        _sum_or_zero(df, "E_elec_kWh_dist_cool")
    )
    e_aux = _sum_or_zero(df, "E_elec_kWh_aux")

    # gas
    g_base  = _sum_or_zero(df, "E_gas_kWh_heat_service")
    g_dloss = _sum_or_zero(df, "E_gas_kWh_dist_heat")

    return {"e_base": e_base, "e_dloss": e_dloss, "e_aux": e_aux,
            "g_base": g_base,  "g_dloss": g_dloss}

def _dt_hours(index: pd.DatetimeIndex) -> pd.Series:
    if len(index) == 0:
        return pd.Series(dtype=float, index=index)
    t = pd.to_datetime(index)
    if len(t) == 1:
        return pd.Series([1.0], index=index)
    dt = (t[1:] - t[:-1]).total_seconds() / 3600.0
    dt = np.r_[dt[0], dt]
    return pd.Series(dt, index=index, dtype=float)

def _comfort_hour_counters(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Returns hours of unmet demand and hours outside setpoints for the given window df.
    Uses:
      - Q_unmet_heat_W / Q_unmet_cool_W (if present) for 'unmet' hours
      - T_in_pred vs setpoint_heat / setpoint_cool (defaults 20/26 °C) for 'too cold' / 'too hot' hours
    """
    out = dict(
        hours_unmet_heat=0.0,
        hours_unmet_cool=0.0,
        hours_too_cold=0.0,
        hours_too_hot=0.0,
    )
    if df is None or df.empty:
        return out

    dt_h = _dt_hours(df.index)

    # --- unmet demand hours (any positive unmet power during a step counts the whole step) ---
    if "Q_unmet_heat_W" in df.columns:
        unmet_h = (df["Q_unmet_heat_W"] > 0).astype(float)
        out["hours_unmet_heat"] = float((unmet_h * dt_h).sum())
    if "Q_unmet_cool_W" in df.columns:
        unmet_c = (df["Q_unmet_cool_W"] > 0).astype(float)
        out["hours_unmet_cool"] = float((unmet_c * dt_h).sum())

    # --- hours outside setpoints (too cold / too hot) ---
    if "T_in_pred" in df.columns:
        sp_h = float(cfg.get("setpoint_heat", 20.0))
        sp_c = float(cfg.get("setpoint_cool", 26.0))
        too_cold = (df["T_in_pred"] < sp_h - 0.1).astype(float)
        too_hot  = (df["T_in_pred"] > sp_c + 0.1).astype(float)
        out["hours_too_cold"] = float((too_cold * dt_h).sum())
        out["hours_too_hot"]  = float((too_hot  * dt_h).sum())

    return out

def render_multi(scenarios: list[dict], sims: dict[str, pd.DataFrame], views: dict[str, pd.DataFrame]):
    # ----- Sidebar: comparison horizon -----
    with st.sidebar.expander("Comparison Horizon", expanded=True):
        horizon_mode = st.radio("Compare over:", ["Selected window", "Lifecycle"], index=0, key="cmp_mode")

        lifetime_years = 15
        disc_rate      = 0.04
        price_escal    = 0.00
        co2_change     = -0.03

        if horizon_mode == "Lifecycle":
            lifetime_years = st.number_input("Lifetime [years]", 1, 50, 15, 1, key="life_years")
            disc_rate   = st.number_input("Discount rate [%/yr]", 0.0, 50.0, 4.0, 0.1, key="disc_rate") / 100.0
            price_escal = st.number_input("Price escalation [%/yr]", -50.0, 50.0, 0.0, 0.1, key="price_escal") / 100.0
            co2_change  = st.number_input("Grid CO₂ change [%/yr]", -50.0, 50.0, 0.0, 0.1, key="co2_change") / 100.0

    st.subheader("Comparison Results")
    
    cum_cost: dict[str, pd.Series] = {}
    cum_co2:  dict[str, pd.Series] = {}
    kpis = []
    houses_by_name = {}
    comfort_rows = []
    
    for cfg in scenarios:
        label = cfg["label"]
        name  = cfg.get("disp_name", cfg.get("name", label))
        n_h   = int(cfg.get("nr_houses", 1))
        houses_by_name[name] = n_h

        v_window = views[label].copy().sort_index()
        v_full   = sims[label].copy().sort_index()

        price_elec, price_gas, co2_elec, co2_gas = _get_prices_co2(cfg)
        capex     = float(cfg.get("capex_house", 0.0))
        embodied  = float(cfg.get("co2_embodied_kg", 0.0))

        # Selected window vs Lifecycle cumulative curves (per house)
        if horizon_mode == "Selected window":
            parts = _component_kwh(v_window)
            v = v_window.copy()
            v["cost_step"] = (
                (v.get("E_elec_kWh", pd.Series(0.0, index=v.index))) * price_elec +
                (v.get("E_gas_kWh",  pd.Series(0.0, index=v.index))) * price_gas
            )
            v["co2_step_kg"] = (
                (v.get("E_elec_kWh", pd.Series(0.0, index=v.index))) * co2_elec +
                (v.get("E_gas_kWh",  pd.Series(0.0, index=v.index))) * co2_gas
            )
            v["cost_cum"]   = v["cost_step"].cumsum() + float(cfg.get("capex_house", 0.0))
            v["co2_cum_kg"] = v["co2_step_kg"].cumsum() + float(cfg.get("co2_embodied_kg", 0.0))
            cum_cost[name] = v["cost_cum"]
            cum_co2[name]  = v["co2_cum_kg"]

        else:
            # Lifecycle: need a representative full year; otherwise fall back to window logic
            src = v_full if len(v_full) >= 8000 else v_window

            E_elec_y = _sum_or_zero(src, "E_elec_kWh")
            E_gas_y  = _sum_or_zero(src, "E_gas_kWh") or _sum_or_zero(src, "E_gas_KWh")

            op_eur_y0 = E_elec_y * price_elec + E_gas_y * price_gas
            op_co2_y0 = E_elec_y * co2_elec   + E_gas_y * co2_gas

            Y = int(lifetime_years)
            years = np.arange(1, Y + 1)

            # price escalation / grid CO2 change / discount
            price_stream = (1.0 + price_escal) ** (years - 1)
            co2_stream   = (1.0 + co2_change) ** (years - 1)
            disc         = (1.0 + disc_rate) ** years

            cost_cum_house = capex + np.cumsum(op_eur_y0 * price_stream / disc)
            co2_cum_house  = embodied + np.cumsum(op_co2_y0 * co2_stream)

            idx = pd.Index(years, name="Year")
            cum_cost[name] = pd.Series(cost_cum_house, index=idx, name=name)
            cum_co2[name]  = pd.Series(co2_cum_house,  index=idx, name=name)

        # KPIs over the selected window (per house)
        SCOP, SEER = scop_seer(v_window)
        E_elec_win = _sum_or_zero(v_window, "E_elec_kWh")
        E_gas_win  = _sum_or_zero(v_window, "E_gas_kWh") or _sum_or_zero(v_window, "E_gas_KWh")

        op_eur_house    = float(
            (v_window.get("E_elec_kWh", pd.Series(0.0, index=v_window.index)) * price_elec).sum() +
            (v_window.get("E_gas_kWh",  pd.Series(0.0, index=v_window.index)) * price_gas).sum()
        )
        total_eur_house = op_eur_house + float(cfg.get("capex_house", 0.0))
        co2_house_total = float(
            (v_window.get("E_elec_kWh", pd.Series(0.0, index=v_window.index)) * co2_elec).sum() +
            (v_window.get("E_gas_kWh",  pd.Series(0.0, index=v_window.index)) * co2_gas).sum() +
            float(cfg.get("co2_embodied_kg", 0.0))
)
        
        # Thermal loads (window) — optional, uses signed E_hvac if present
        heat_kwh = float(v_window.get("E_hvac_kWh", pd.Series(0.0, index=v_window.index)).clip(lower=0).sum())
        cool_kwh = float((-v_window.get("E_hvac_kWh", pd.Series(0.0, index=v_window.index)).clip(upper=0)).sum())

        # Comfort calculation
        comf = _comfort_hour_counters(v_window, cfg)
        # append a row for the comfort plots
        comfort_rows.append({
            "Scenario": name,
            "Unmet heat [h]": comf["hours_unmet_heat"],
            "Unmet cool [h]": comf["hours_unmet_cool"],
            "Too cold [h]":   comf["hours_too_cold"],
            "Too hot [h]":    comf["hours_too_hot"],
        })

        kpis.append({
            "Scenario": name,
            "Elec kWh (house)": E_elec_win,
            "Gas kWh (house)":  E_gas_win,
            "Heating load [kWh]": heat_kwh,
            "Cooling load [kWh]": cool_kwh,
            "OpEx per house [€]": op_eur_house,
            "CapEx per house [€]": capex,
            "Total € (house) incl CapEx": total_eur_house,
            "Total € (all houses) incl CapEx": total_eur_house * n_h,
            "CO₂ (house) [kg] incl embodied": co2_house_total,
            "CO₂ (all houses) [kg] incl embodied": co2_house_total * n_h,
            "SCOP": SCOP,
            "SEER": SEER,
        })

        # ----- Comparison tables & cumulative curves -----
    with st.expander("Result Overview", expanded=False):
        kpi_df = pd.DataFrame(kpis)
        kpi_df["Houses"] = kpi_df["Scenario"].map({cfg.get("disp_name", cfg["label"]): int(cfg.get("nr_houses", 1)) for cfg in scenarios})

        kpi_df["Total Cost [€]"] = kpi_df["Total € (house) incl CapEx"] * kpi_df["Houses"]
        kpi_df["Total CO₂ [kg]"] = kpi_df["CO₂ (house) [kg] incl embodied"] * kpi_df["Houses"]

        st.dataframe(
            kpi_df[[
                "Scenario",
                "Heating load [kWh]", "Cooling load [kWh]",
                "Elec kWh (house)", "Gas kWh (house)",
                "Total € (house) incl CapEx", "CO₂ (house) [kg] incl embodied",
            ]],
            use_container_width=True,
        )

    with st.expander("Plots", expanded=False):
        # Cumulative cost (total, scaled by houses)
        df_cost_total = pd.concat({n: s * houses_by_name[n] for n, s in cum_cost.items()}, axis=1)
        if horizon_mode == "Lifecycle":
            df_cost_total.index.name = "Year"
            x_axis = "Year"
            title_cost = "Cumulative Expenditure (Total, lifecycle)"
        else:
            df_cost_total.index.name = "time"
            x_axis = "time"
            title_cost = "Cumulative Expenditure (Total)"

        fig_cost = px.line(df_cost_total.reset_index(), x=x_axis, y=df_cost_total.columns,
                        labels={"value": "Expenditure [€]", x_axis: x_axis, "variable": "Scenario"},
                        title=title_cost)
        fig_cost.update_layout(hovermode="x unified")
        st.plotly_chart(fig_cost, use_container_width=True)

        # Cumulative CO2 (total, scaled by houses)
        df_co2_total = pd.concat({n: s * houses_by_name[n] for n, s in cum_co2.items()}, axis=1)
        if horizon_mode == "Lifecycle":
            df_co2_total.index.name = "Year"
            x_axis = "Year"
            title_co2 = "Cumulative CO₂ (Total, lifecycle)"
        else:
            df_co2_total.index.name = "time"
            x_axis = "time"
            title_co2 = "Cumulative CO₂ (Total)"

        fig_co2cum = px.line(df_co2_total.reset_index(), x=x_axis, y=df_co2_total.columns,
                            labels={"value": "CO₂ [kg]", x_axis: x_axis, "variable": "Scenario"},
                            title=title_co2)
        fig_co2cum.update_layout(hovermode="x unified")
        st.plotly_chart(fig_co2cum, use_container_width=True)

        # ----- Breakdown & Pareto -----
        tab_breakdown_cost_co2, tab_pareto, tab_comfort = st.tabs(["Cost & CO₂ breakdown", "Cost vs CO₂", "Comfort Comparison"])

        with tab_breakdown_cost_co2:
            rows_cost = []
            rows_co2  = []

            for cfg in scenarios:
                name = cfg.get("disp_name", cfg["label"])
                n    = int(cfg.get("nr_houses", 1))
                price_elec, price_gas, co2_elec, co2_gas = _get_prices_co2(cfg)

                vwin  = views[cfg["label"]]
                vfull = sims[cfg["label"]]

                # CapEx & embodied (per house)
                cap_user  = float(cfg.get("capex_house_user", cfg.get("capex_house", 0.0)))
                cap_plant = float(cfg.get("capex_alloc_plant", 0.0))
                cap_dist  = float(cfg.get("capex_alloc_dist", 0.0))
                emb_user  = float(cfg.get("embodied_user_kg", 0.0))
                emb_plant = float(cfg.get("embodied_plant_kg", 0.0))
                emb_dist  = float(cfg.get("embodied_dist_kg", 0.0))

                if horizon_mode == "Selected window":
                    parts = _component_kwh(vwin)

                    opex_base  = price_elec * parts["e_base"]  + price_gas * parts["g_base"]
                    opex_dloss = price_elec * parts["e_dloss"] + price_gas * parts["g_dloss"]
                    opex_aux   = price_elec * parts["e_aux"]

                    co2_base   = co2_elec * parts["e_base"]  + co2_gas * parts["g_base"]
                    co2_dloss  = co2_elec * parts["e_dloss"] + co2_gas * parts["g_dloss"]
                    co2_aux    = co2_elec * parts["e_aux"]
                else:
                    # Lifecycle: use representative year (full if available, else window)
                    src   = vfull if len(vfull) >= 8000 else vwin
                    parts = _component_kwh(src)

                    Y = int(lifetime_years)
                    years = np.arange(1, Y + 1)
                    price_stream = (1.0 + price_escal) ** (years - 1)
                    co2_stream   = (1.0 + co2_change) ** (years - 1)
                    disc         = (1.0 + disc_rate) ** years

                    # annualized base/dloss/aux -> escalate/discount streams
                    e_base_y,  g_base_y  = parts["e_base"],  parts["g_base"]
                    e_dloss_y, g_dloss_y = parts["e_dloss"], parts["g_dloss"]
                    e_aux_y              = parts["e_aux"]

                    opex_base  = ((price_elec * e_base_y  + price_gas * g_base_y)  * price_stream / disc).sum()
                    opex_dloss = ((price_elec * e_dloss_y + price_gas * g_dloss_y) * price_stream / disc).sum()
                    opex_aux   = ((price_elec * e_aux_y)                          * price_stream / disc).sum()

                    co2_base   = ((co2_elec * e_base_y  + co2_gas * g_base_y)  * co2_stream).sum()
                    co2_dloss  = ((co2_elec * e_dloss_y + co2_gas * g_dloss_y) * co2_stream).sum()
                    co2_aux    = ((co2_elec * e_aux_y)                        * co2_stream).sum()

                # Scale to all houses
                rows_cost.append({
                    "Scenario": name,
                    "OpEx: HVAC base": opex_base * n,
                    "OpEx: Dist losses": opex_dloss * n,
                    "OpEx: Auxiliaries": opex_aux * n,
                    "CapEx: Per-house equip": cap_user * n,
                    "CapEx: Plant": cap_plant * n,
                    "CapEx: Distribution": cap_dist * n,
                })

                rows_co2.append({
                    "Scenario": name,
                    "Operational: HVAC base": co2_base * n,
                    "Operational: Dist losses": co2_dloss * n,
                    "Operational: Auxiliaries": co2_aux * n,
                    "Embodied: Per-house equip": emb_user * n,
                    "Embodied: Plant": emb_plant * n,
                    "Embodied: Distribution": emb_dist * n,
                })

            # Cost breakdown
            df_break = pd.DataFrame(rows_cost)
            opex_cols  = ["OpEx: HVAC base", "OpEx: Dist losses", "OpEx: Auxiliaries"]
            capex_cols = ["CapEx: Per-house equip", "CapEx: Plant", "CapEx: Distribution"]
            df_opex_long  = df_break.melt(id_vars="Scenario", value_vars=opex_cols,  var_name="Type", value_name="€")
            df_capex_long = df_break.melt(id_vars="Scenario", value_vars=capex_cols, var_name="Type", value_name="€")

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(df_opex_long, x="Scenario", y="€", color="Type", barmode="stack", title="OpEx breakdown")
                fig.update_layout(hovermode="x unified")
                fig.update_yaxes(title_text="Cost [€]")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(df_capex_long, x="Scenario", y="€", color="Type", barmode="stack", title="CapEx breakdown")
                fig.update_layout(hovermode="x unified")
                fig.update_yaxes(title_text="Cost [€]")
                st.plotly_chart(fig, use_container_width=True)

            # CO₂ breakdown
            df_co2 = pd.DataFrame(rows_co2)
            op_cols  = ["Operational: HVAC base", "Operational: Dist losses", "Operational: Auxiliaries"]
            emb_cols = ["Embodied: Per-house equip", "Embodied: Plant", "Embodied: Distribution"]
            df_co2_op  = df_co2.melt(id_vars="Scenario", value_vars=op_cols,  var_name="Type", value_name="kg CO₂")
            df_co2_emb = df_co2.melt(id_vars="Scenario", value_vars=emb_cols, var_name="Type", value_name="kg CO₂")

            c3, c4 = st.columns(2)
            with c3:
                fig = px.bar(df_co2_op, x="Scenario", y="kg CO₂", color="Type", barmode="stack", title="Operational CO₂ breakdown")
                fig.update_layout(hovermode="x unified")
                fig.update_yaxes(title_text="CO₂ [kg]")
                st.plotly_chart(fig, use_container_width=True)
            with c4:
                fig = px.bar(df_co2_emb, x="Scenario", y="kg CO₂", color="Type", barmode="stack", title="Embodied CO₂ breakdown")
                fig.update_layout(hovermode="x unified")
                fig.update_yaxes(title_text="CO₂ [kg]")
                st.plotly_chart(fig, use_container_width=True)

        with tab_pareto:
            fig = px.scatter(
                kpi_df, x="Total Cost [€]", y="Total CO₂ [kg]",
                color="Scenario", size="Houses",
                hover_data=["Elec kWh (house)", "Gas kWh (house)", "SCOP", "SEER"],
                title="Cost vs CO₂",
            )
            fig.update_traces(marker=dict(opacity=0.85))
            fig.update_layout(hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)

        # ---- Comfort comparison ----
        with tab_comfort:
            if comfort_rows:
                df_comf = pd.DataFrame(comfort_rows)
                
                df_out = df_comf.melt(id_vars="Scenario",
                                    value_vars=["Too cold [h]", "Too hot [h]"],
                                    var_name="Type", value_name="Hours")
                fig_out = px.bar(df_out, x="Scenario", y="Hours", color="Type",
                                barmode="group", title="Hours outside setpoints")
                fig_out.update_layout(hovermode="x unified")
                st.plotly_chart(fig_out, use_container_width=True)
            else:
                st.info("No comfort data available for the selected window.")

    # Export cumulative costs (total)
    st.subheader("Export (comparison)")
    out_tot = pd.concat({n: s * houses_by_name[n] for n, s in cum_cost.items()}, axis=1)
    csv = out_tot.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download cumulative cost (totals, all scenarios)", csv, "cumulative_costs_total.csv", "text/csv")
