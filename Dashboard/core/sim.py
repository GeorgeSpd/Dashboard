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
    COP_heat=3.0,                  # float or callable(T_out)->float
    EER_cool=3.0,                  # float or callable(T_out)->float
) -> pd.DataFrame:
    """
    Two-node RC (Zone + Wall) with a *backward-Euler* step

    Control policy:
      1) Compute end-of-step *free drift* (Q_hvac=0).
      2) If free drift ends INSIDE [setpoint_heat, setpoint_cool] → do NOTHING.
      3) If below the band → heat to setpoint_heat (if capacity allows).
      4) If above the band → cool to setpoint_cool (if capacity allows).
         When acting, find the HVAC power that lands exactly on the edge this
         step via *bisection*; clamp to capacity if needed.

    Weather columns:
      - T_out [°C]
      - (optional) G_solar [W/m²] (defaults to 0)
    """

    # Helper
    def _safe_eff(eff_or_fn, t_out: float) -> float:
        try:
            v = float(eff_or_fn(float(t_out))) if callable(eff_or_fn) else float(eff_or_fn)
        except Exception:
            v = 0.1
        return max(v, 0.1)

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
    CAP_ZONE_AREAL = 20.0 * 1e3                     # 100 [kJ/m²K]
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

    # Implicit solver
    def backward_euler_iter(
        Tz0: float, Tw0: float, To: float, Gs: float, Q_hvac_W: float,
        tol: float = 1e-5, max_iter: int = 20
    ):
        """
        One backward-Euler step with fixed-point iteration.
        Returns:
          (Tz1, Tw1,
           avg_Q_zone_sources_W,   # zone share: solar+hvac (constant this hour)
           avg_Q_zone_from_wall_W,
           avg_Q_zone_from_outdoor_W,
           avg_Q_wall_from_outdoor_W,
           avg_Q_total_solar_W)    # total solar (zone+wall)
        """
        # Constant sources during the hour
        QsZ = g_solar * A_solar_m2 * Gs * solar_zone_frac
        QsW = g_solar * A_solar_m2 * Gs * (1.0 - solar_zone_frac)
        QhZ = hvac_zone_frac * Q_hvac_W
        QhW = (1.0 - hvac_zone_frac) * Q_hvac_W

        # Denominators for the two implicit updates
        denom_z = (C_zone / dt_s) + G_inf + G_zw
        denom_w = (C_wall / dt_s) + G_zw + G_wo

        # Start from previous state
        Tz = float(Tz0)
        Tw = float(Tw0)

        for _ in range(max_iter):
            # Update zone using *current Tw*
            rhs_z = (C_zone/dt_s)*Tz0 + G_inf*To + G_zw*Tw + QsZ + QhZ
            Tz_new = rhs_z / denom_z

            # Update wall using *new Tz*
            rhs_w = (C_wall/dt_s)*Tw0 + G_wo*To + G_zw*Tz_new + QsW + QhW
            Tw_new = rhs_w / denom_w

            if abs(Tz_new - Tz) < tol and abs(Tw_new - Tw) < tol:
                Tz, Tw = Tz_new, Tw_new
                break

            Tz, Tw = Tz_new, Tw_new

        # End-of-step flows
        Q_zw_end  = G_zw * (Tw - Tz)   # into zone
        Q_inf_end = G_inf * (To - Tz)  # into zone
        Q_wo_end  = G_wo * (To - Tw)   # into wall

        avg_Q_zone_sources_W = QsZ + QhZ
        avg_Q_total_solar_W  = QsZ + QsW

        return (Tz, Tw,
                avg_Q_zone_sources_W,
                Q_zw_end,
                Q_inf_end,
                Q_wo_end,
                avg_Q_total_solar_W)

    # Outputs
    out_T_zone = np.empty(n)
    out_T_wall = np.empty(n)
    out_Q_hvac = np.zeros(n)
    out_P_elec = np.zeros(n)
    out_COP    = np.full(n, np.nan)
    out_EER    = np.full(n, np.nan)
    out_Q_inf  = np.zeros(n)
    out_Q_cond = np.zeros(n)
    out_Q_zw   = np.zeros(n)
    out_Q_sol  = np.zeros(n)
    out_Q_net  = np.zeros(n)

    # Main loop
    for i in range(n):
        To, Gs = T_out[i], G_sol[i]

        # Initial check
        (free_Tz1, free_Tw1,
         free_Q_zone_src, free_Q_zw, free_Q_inf, free_Q_wo, free_Q_solar) = \
            backward_euler_iter(T_zone, T_wall, To, Gs, 0.0)

        # If temp is between setpoints
        if setpoint_heat <= free_Tz1 <= setpoint_cool:
            # Do nothing
            Q_hvac_W = 0.0
            zone_T1, wall_T1 = free_Tz1, free_Tw1
            avg_Q_zone_src, avg_Q_zw, avg_Q_inf, avg_Q_wo, avg_Q_solar = \
                free_Q_zone_src, free_Q_zw, free_Q_inf, free_Q_wo, free_Q_solar
        else:
            target_T = setpoint_heat if free_Tz1 < setpoint_heat else setpoint_cool

            # Compute slope dTz/dQ using a single reference power
            Q_ref = 1.0
            Tz_ref, *_ = backward_euler_iter(T_zone, T_wall, To, Gs, Q_ref)
            slope = (Tz_ref - free_Tz1) / Q_ref

            # Avoid divide-by-zero
            if abs(slope) < 1e-12:
                Q_star = 0.0
            else:
                Q_star = (target_T - free_Tz1) / slope

            # Respect capacity
            if target_T == setpoint_heat:
                Q_hvac_W = min(max(Q_star, 0.0), P_max_heat)
            else:
                Q_hvac_W = max(min(Q_star, 0.0), -P_max_cool)

            # Final pass to get end temps and flows with correct Q_hvac
            (zone_T1, wall_T1,
            avg_Q_zone_src, avg_Q_zw, avg_Q_inf, avg_Q_wo, avg_Q_solar) = \
                backward_euler_iter(T_zone, T_wall, To, Gs, Q_hvac_W)

        # Electrical power
        if Q_hvac_W > 0:
            cop = _safe_eff(COP_heat, To)
            Pel = Q_hvac_W / cop
            out_COP[i] = cop
        elif Q_hvac_W < 0:
            eer = _safe_eff(EER_cool, To)
            Pel = abs(Q_hvac_W) / eer
            out_EER[i] = eer
        else:
            Pel = 0.0

        # Store
        out_T_zone[i] = zone_T1
        out_T_wall[i] = wall_T1
        out_Q_hvac[i] = Q_hvac_W
        out_P_elec[i] = Pel
        out_Q_zw[i]   = avg_Q_zw
        out_Q_inf[i]  = avg_Q_inf
        out_Q_cond[i] = avg_Q_wo
        out_Q_sol[i]  = avg_Q_solar
        out_Q_net[i]  = avg_Q_zw + avg_Q_inf + avg_Q_zone_src

        # Advance state
        T_zone, T_wall = zone_T1, wall_T1

    # Output dataframe
    out = df.copy()
    out["T_in_pred"]   = out_T_zone
    out["T_wall_pred"] = out_T_wall
    out["Q_hvac_W"] = out_Q_hvac
    out["P_elec_W"] = out_P_elec
    out["COP_used"] = out_COP
    out["EER_used"] = out_EER
    out["Q_cond_W"]  = out_Q_cond
    out["Q_inf_W"]   = out_Q_inf
    out["Q_solar_W"] = out_Q_sol
    out["Q_net_W"]   = out_Q_net

    # Energies per step
    out["E_elec_kWh"]  = out["P_elec_W"]  * dt_s / 3_600_000.0
    out["E_cond_kWh"]  = out["Q_cond_W"]  * dt_s / 3_600_000.0
    out["E_inf_kWh"]   = out["Q_inf_W"]   * dt_s / 3_600_000.0
    out["E_solar_kWh"] = out["Q_solar_W"] * dt_s / 3_600_000.0
    out["E_hvac_kWh"]  = out["Q_hvac_W"]  * dt_s / 3_600_000.0
    out["E_net_kWh"]   = out["Q_net_W"]   * dt_s / 3_600_000.0
    return out
