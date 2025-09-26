# # pylint: disable=C0114
# # pylint: disable=C0103
# # pylint: disable=C0116
# # pylint: disable=C0301
# # pylint: disable=I1101
# # pylint: disable=W0401
# # pylint: disable=W0614
# # pylint: disable=C0303
# # pylint: disable=W0105
# # pylint: disable=W0621
# # pylint: disable=W0718
# # pylint: disable=W0640

import numpy as np
import pandas as pd

def simulate_RC(
    weather: pd.DataFrame,
    params: dict,
    T0: float,
    dt_hours: float = 1.0,
    setpoint_heat: float = 20.0,   # lower comfort edge [°C]
    setpoint_cool: float = 26.0,   # upper comfort edge [°C]
    P_max_heat: float = 5000.0,    # W (thermal)
    P_max_cool: float = 5000.0,    # W (thermal)
) -> pd.DataFrame:
    """
    Two-node RC (Zone + Wall) with a backward-Euler step.

    Control policy (thermal only):
      1) Compute end-of-step free drift (Q_hvac=0).
      2) If free drift ends inside [setpoint_heat, setpoint_cool] → do nothing.
      3) If below band → heat to setpoint_heat (limited by P_max_heat).
      4) If above band → cool to setpoint_cool (limited by P_max_cool).
         When acting, solve for the HVAC power that lands exactly on the edge
         this step via bisection; clamp to capacity if needed.

    Weather columns:
      - T_out [°C]
      - (optional) G_solar [W/m²] (defaults to 0)

    Returns
    -------
    DataFrame with (at least):
      - T_in_pred, T_wall_pred
      - Q_hvac_W   (thermal power, + = heating, − = cooling)
      - Q_cond_W, Q_inf_W, Q_solar_W, Q_net_W
      - E_cond_kWh, E_inf_kWh, E_solar_kWh, E_hvac_kWh, E_net_kWh (thermal integrals)
    """
    # Inputs
    df = weather.copy()
    dt_s = float(dt_hours) * 3600.0

    # Air props
    AIR_RHO = 1.2      # [kg/m³]
    AIR_CP  = 1005.0   # [J/kgK]

    # Archetypes
    S2V = {
        "Detached": 0.75,
        "Semi-detached": 0.60,
        "Terraced": 0.45,
        "Apartment (mid-floor)": 0.30,
        "Apartment (corner/top)": 0.50,
    }
    def U_by_year(y: int) -> float:
        if y <= 1975:
            return 1.5
        if y <= 1991:
            return 1.0
        if y <= 2005:
            return 0.70
        if y <= 2012:
            return 0.50
        if y <= 2020:
            return 0.35
        return 0.25

    # Unpack params
    btype = params.get("building_type")
    if btype not in S2V:
        raise ValueError(f"Unknown building_type: {btype}")
    year_built      = int(params.get("year_built"))
    V_m3            = float(params.get("V", 250.0))
    ACH             = float(params.get("ACH", 0.5))
    A_solar_m2      = float(params.get("A_solar", 10.0))
    g_solar         = float(params.get("g_solar", 0.5))
    solar_zone_frac = float(params.get("solar_zone_fraction", 0.55))
    hvac_zone_frac  = float(params.get("hvac_zone_fraction", 0.85))

    # Derived physics
    A_env = S2V[btype] * V_m3
    A_floor = max(V_m3 / 2.5, 1.0)

    U_env = U_by_year(year_built)                   # [W/m²K]
    UA    = U_env * A_env                           # [W/K]

    # Split the envelope into zone-wall and wall-outdoor resistances in series
    G_zw = 2.0 * UA                                 # [W/K]
    G_wo = 2.0 * UA                                 # [W/K]
    G_inf = ACH * V_m3 / 3600.0 * AIR_RHO * AIR_CP  # [W/K] (infiltration)

    # Capacitances (J/K)
    CAP_ZONE_AREAL = 10.0 * 1e3                     # 10  [kJ/m²K]
    CAP_WALL_AREAL = 120.0 * 1e3                    # 120 [kJ/m²K]
    C_air  = AIR_RHO * AIR_CP * V_m3
    C_zone = CAP_ZONE_AREAL * A_floor + C_air
    C_wall = CAP_WALL_AREAL * A_env

    # Weather arrays
    T_out = df["T_out"].to_numpy(float)
    G_sol = df.get("G_solar", pd.Series(0.0, index=df.index)).to_numpy(float)
    n = len(df)

    # Initial state
    T_zone = float(T0)
    T_wall = float((T0 + T_out[0]) / 2.0)

    T0_need = float(np.clip(T0, setpoint_heat, setpoint_cool if np.isfinite(setpoint_cool) else 1e9))
    T_zone_need = T0_need
    T_wall_need = float((T0_need + T_out[0]) / 2.0)

    # Implicit step
    def backward_euler_iter(
        Tz0: float, Tw0: float, To: float, Gs: float, Q_hvac_W: float,
        tol: float = 1e-5, max_iter: int = 20
    ):
        # Constant sources during the hour
        QsZ = g_solar * A_solar_m2 * Gs * solar_zone_frac
        QsW = g_solar * A_solar_m2 * Gs * (1.0 - solar_zone_frac)
        QhZ = hvac_zone_frac * Q_hvac_W
        QhW = (1.0 - hvac_zone_frac) * Q_hvac_W

        denom_z = (C_zone / dt_s) + G_inf + G_zw
        denom_w = (C_wall / dt_s) + G_zw + G_wo

        Tz, Tw = float(Tz0), float(Tw0)
        for _ in range(max_iter):
            rhs_z = (C_zone/dt_s)*Tz0 + G_inf*To + G_zw*Tw + QsZ + QhZ
            Tz_new = rhs_z / denom_z
            rhs_w = (C_wall/dt_s)*Tw0 + G_wo*To + G_zw*Tz_new + QsW + QhW
            Tw_new = rhs_w / denom_w
            if abs(Tz_new - Tz) < tol and abs(Tw_new - Tw) < tol:
                Tz, Tw = Tz_new, Tw_new
                break
            Tz, Tw = Tz_new, Tw_new

        Q_zw_end  = G_zw * (Tw - Tz)   # into zone
        Q_inf_end = G_inf * (To - Tz)  # into zone
        Q_wo_end  = G_wo * (To - Tw)   # into wall
        avg_Q_zone_sources_W = QsZ + QhZ
        avg_Q_total_solar_W  = QsZ + QsW
        return (Tz, Tw, avg_Q_zone_sources_W, Q_zw_end, Q_inf_end, Q_wo_end, avg_Q_total_solar_W)

    # Outputs
    out_T_zone = np.empty(n)
    out_T_wall = np.empty(n)
    out_Q_hvac = np.zeros(n)
    out_Q_inf  = np.zeros(n)
    out_Q_cond = np.zeros(n)
    out_Q_zw   = np.zeros(n)
    out_Q_sol  = np.zeros(n)
    out_Q_net  = np.zeros(n)
    out_Q_need      = np.zeros(n)
    out_Q_need_h    = np.zeros(n)
    out_Q_need_c    = np.zeros(n)
    out_Q_unmet_h   = np.zeros(n)
    out_Q_unmet_c   = np.zeros(n)
    
    # Main loop
    for i in range(n):
        To, Gs = T_out[i], G_sol[i]

        (free_Tz1, free_Tw1,
         free_Q_zone_src, free_Q_zw, free_Q_inf, free_Q_wo, free_Q_solar) = \
            backward_euler_iter(T_zone, T_wall, To, Gs, 0.0)

        if setpoint_heat <= free_Tz1 <= setpoint_cool:
            Q_need = 0.0
            Q_hvac_W = 0.0
            zone_T1, wall_T1 = free_Tz1, free_Tw1
            avg_Q_zone_src, avg_Q_zw, avg_Q_inf, avg_Q_wo, avg_Q_solar = \
                free_Q_zone_src, free_Q_zw, free_Q_inf, free_Q_wo, free_Q_solar
        else:
            target_T = setpoint_heat if free_Tz1 < setpoint_heat else setpoint_cool
            Q_ref = 1.0
            Tz_ref, *_ = backward_euler_iter(T_zone, T_wall, To, Gs, Q_ref)
            slope = (Tz_ref - free_Tz1) / Q_ref
            Q_need = 0.0 if abs(slope) < 1e-12 else (target_T - free_Tz1) / slope
            if target_T == setpoint_heat:
                Q_hvac_W = min(max(Q_need, 0.0), P_max_heat)
            else:
                Q_hvac_W = max(min(Q_need, 0.0), -P_max_cool)

            (zone_T1, wall_T1,
             avg_Q_zone_src, avg_Q_zw, avg_Q_inf, avg_Q_wo, avg_Q_solar) = \
                backward_euler_iter(T_zone, T_wall, To, Gs, Q_hvac_W)

        out_T_zone[i] = zone_T1
        out_T_wall[i] = wall_T1
        out_Q_hvac[i] = Q_hvac_W
        out_Q_zw[i]   = avg_Q_zw
        out_Q_inf[i]  = avg_Q_inf
        out_Q_cond[i] = avg_Q_wo
        out_Q_sol[i]  = avg_Q_solar
        out_Q_net[i]  = avg_Q_zw + avg_Q_inf + avg_Q_zone_src
        out_Q_need[i]    = Q_need
        out_Q_need_h[i]  = max(Q_need, 0.0)
        out_Q_need_c[i]  = max(-Q_need, 0.0)
        out_Q_unmet_h[i] = max(out_Q_need_h[i] - max(out_Q_hvac[i], 0.0), 0.0)
        out_Q_unmet_c[i] = max(out_Q_need_c[i] - max(-out_Q_hvac[i], 0.0), 0.0)

        T_zone, T_wall = zone_T1, wall_T1

        (free_Tz1_n, *_ignore) = backward_euler_iter(T_zone_need, T_wall_need, To, Gs, 0.0)

        Q_need = 0.0
        # only act if drifting outside band (otherwise zero need this step)
        if free_Tz1_n < setpoint_heat:
            target_n = setpoint_heat
            Q_ref = 1.0
            Tz_ref_n, *_ = backward_euler_iter(T_zone_need, T_wall_need, To, Gs, Q_ref)
            slope_n = (Tz_ref_n - free_Tz1_n) / Q_ref
            Q_need = 0.0 if abs(slope_n) < 1e-12 else (target_n - free_Tz1_n) / slope_n
        elif np.isfinite(setpoint_cool) and (free_Tz1_n > setpoint_cool):
            target_n = setpoint_cool
            Q_ref = 1.0
            Tz_ref_n, *_ = backward_euler_iter(T_zone_need, T_wall_need, To, Gs, Q_ref)
            slope_n = (Tz_ref_n - free_Tz1_n) / Q_ref
            Q_need = 0.0 if abs(slope_n) < 1e-12 else (target_n - free_Tz1_n) / slope_n
        # else: inside band → 0

        # advance needs state with Q_need (unlimited capacity)
        TzN1, TwN1, *_ = backward_euler_iter(T_zone_need, T_wall_need, To, Gs, Q_need)
        T_zone_need, T_wall_need = TzN1, TwN1

        # store needs (split by sign)
        out_Q_need_h[i] = max(Q_need, 0.0)
        out_Q_need_c[i] = max(-Q_need, 0.0)

        # unmet due to capacity this step (delivered vs need)
        out_Q_unmet_h[i] = max(out_Q_need_h[i] - max(out_Q_hvac[i], 0.0), 0.0)
        out_Q_unmet_c[i] = max(out_Q_need_c[i] - max(-out_Q_hvac[i], 0.0), 0.0)

    # Output dataframe
    out = df.copy()
    out["T_in_pred"]   = out_T_zone
    out["T_wall_pred"] = out_T_wall
    out["Q_hvac_W"]    = out_Q_hvac
    out["Q_cond_W"]    = out_Q_cond
    out["Q_inf_W"]     = out_Q_inf
    out["Q_solar_W"]   = out_Q_sol
    out["Q_net_W"]     = out_Q_net
    out["Q_need_heat_W"] = out_Q_need_h
    out["Q_need_cool_W"] = out_Q_need_c
    out["Q_unmet_heat_W_rc"] = out_Q_unmet_h
    out["Q_unmet_cool_W_rc"] = out_Q_unmet_c

    # Thermal energy integrals (kWh) — optional but handy for QA; remove if you prefer
    out["E_cond_kWh"]  = out["Q_cond_W"]  * dt_s / 3_600_000.0
    out["E_inf_kWh"]   = out["Q_inf_W"]   * dt_s / 3_600_000.0
    out["E_solar_kWh"] = out["Q_solar_W"] * dt_s / 3_600_000.0
    out["E_hvac_kWh"]  = out["Q_hvac_W"]  * dt_s / 3_600_000.0
    out["E_net_kWh"]   = out["Q_net_W"]   * dt_s / 3_600_000.0
    out["E_need_heat_kWh"] = out["Q_need_heat_W"] * dt_s / 3_600_000.0
    out["E_need_cool_kWh"] = out["Q_need_cool_W"] * dt_s / 3_600_000.0
    return out
