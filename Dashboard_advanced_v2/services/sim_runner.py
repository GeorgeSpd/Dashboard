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
Thermal-service dispatch over RC-simulated loads.

Pipeline
--------
1) Run `simulate_RC` with per-house capacity caps so temperatures/comfort and
   delivered thermal respond to equipment limits.
2) Dispatch the delivered heating/cooling/DHW load to prioritized technology
   chains (merit order) with temperature-dependent efficiency curves.
3) If centralized, include distribution losses in the *same* dispatch to
   respect plant capacity, then partition results into service vs. losses.
4) Compute energy, costs, CO₂, and expose per-tech splits and effective COP/EER
   time-series for plotting/analysis.

Inputs
------
- Weather/time series `wx` (DataFrame with at least outdoor temperature).
- Scenario config `cfg` returned by the UI layer (see `scenario_controls`).

Outputs
-------
A time-indexed DataFrame aligned to the simulation grid with:
- Delivered thermal by sign: `Q_heat_W`, `Q_cool_W` (per house).
- Merit-order splits per tech (thermal W and energy kWh).
- Effective efficiency series: `COP_heating_eff`, `EER_cooling_eff`,
  plus per-tech efficiency series masked to NaN when the tech is off.
- Final energy totals (`E_elec_kWh`, `E_gas_kWh`), cost (`Cost_eur`),
  operational CO₂ (`CO2_kg`), cumulative CO₂ including embodied
  (`CO2_cum_house_kg`), and unmet loads.

Notes
-----
- Room/house capacities are in **W**; plant capacities (central) provided in
  **kW** are scaled to per-house **W** using diversity and house count.
- `simulate_RC` electric consumption is neutralized (huge COP/EER); this module
  recomputes energy based on the dispatched chains.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from core.sim import simulate_RC
from core.efficiency import HEATING_COP, COOLING_EER, BOILER_EFF

# Helpers
def _dt_hours(index: pd.DatetimeIndex) -> pd.Series:
    """
    Compute step durations in hours for a (possibly irregular) time index.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Simulation time grid.

    Returns
    -------
    pd.Series
        Series aligned with `index` containing step lengths in hours.
        The first step is duplicated from the second to keep length consistent.
    """
    if len(index) <= 1:
        return pd.Series([1.0], index=index)
    t = pd.to_datetime(index)
    dt = (t[1:] - t[:-1]).total_seconds() / 3600.0
    dt = np.r_[dt[0], dt]
    return pd.Series(dt, index=index, dtype=float)

def _eval_curve(curve_dict: dict, key: str, Tout: np.ndarray) -> np.ndarray:
    """
    Evaluate a performance curve (COP/EER/η) for an outdoor temperature vector.

    Parameters
    ----------
    curve_dict : dict
        Mapping {curve_key: callable_or_constant}. Callables must accept a
        NumPy array of outdoor temperatures and return a vector.
    key : str
        Curve identifier to look up.
    Tout : np.ndarray
        Outdoor temperature sequence.

    Returns
    -------
    np.ndarray
        Vector of efficiencies, same shape as `Tout`. If the key is missing,
        returns zeros; if the item is a scalar, returns a constant vector.
    """
    item = curve_dict.get(key, 0.0)
    if callable(item):
        return np.asarray(item(Tout), dtype=float)
    return np.full_like(Tout, float(item), dtype=float)

def _get_chains_in_W(cfg: dict, is_central: bool, cooling_enabled: bool) -> tuple[list[dict], list[dict]]:
    """
    Normalize heating/cooling technology chains to **per-house Watts**.

    Decentralized:
        - Tech `Pmax` is already per-house W → passed through.

    Centralized:
        - Plant `Pmax` provided in kW → converted to W and then scaled to
          per-house available capacity using:  P_house_W = kW * 1000 * diversity / nr_houses

    Parameters
    ----------
    cfg : dict
        Scenario configuration (from UI).
    is_central : bool
        Whether the architecture is centralized.
    cooling_enabled : bool
        Whether cooling is enabled (empty chain if False).

    Returns
    -------
    (heating_chain_W, cooling_chain_W) : tuple[list[dict], list[dict]]
        Each list contains dicts with the original tech fields and a per-house
        `Pmax` in Watts, suitable for dispatch.
    """
    nr = max(int(cfg.get("nr_houses", 1)), 1)
    if not is_central:
        heat_chain = [{**d, "Pmax": float(d.get("Pmax", 0.0))} for d in (cfg.get("heating_chain", []))]
        cool_chain = [{**d, "Pmax": float(d.get("Pmax", 0.0))} for d in (cfg.get("cooling_chain", []) if cooling_enabled else [])]
        return heat_chain, cool_chain

    arch = cfg.get("arch_cfg") or {}
    diversity = float(arch.get("diversity", 1.0))
    scale = (1000.0 * diversity) / nr

    heat_src = arch.get("heating_chain") or arch.get("heat_sys_base") or []
    cool_src = (arch.get("cooling_chain") or arch.get("cool_sys_base") or []) if cooling_enabled else []

    def _scale(src):
        out = []
        for d in src:
            out.append({**d, "Pmax": float(d.get("Pmax", 0.0)) * scale})
        return out

    return _scale(heat_src), _scale(cool_src)

def _dispatch_merit_order(load_W: pd.Series, Tout: pd.Series, chain: list[dict], mode: str):
    """
    Dispatch a thermal load in priority order across a technology chain.

    For each step:
      - Serve up to each tech's `Pmax` in order of `priority` (1 = first).
      - Evaluate efficiency (COP/EER for electric; η for gas boilers) from
        the selected curve dictionary and `Tout`.
      - Convert delivered thermal to input energy by fuel type.

    Parameters
    ----------
    load_W : pd.Series
        Positive thermal power demand to be served [W], time-indexed.
    Tout : pd.Series
        Outdoor temperature [°C], aligned with `load_W`.
    chain : list[dict]
        Priority-ordered tech descriptors with keys:
          - "name": str
          - "curve_key": str
          - "Pmax": float (per-house W)
          - "priority": int (1..N)
          - "fuel": "elec" | "gas"
    mode : str
        "heating" or "cooling" (controls which curve dict is used).

    Returns
    -------
    dict
        {
          "served_W_total": pd.Series,             # total thermal served [W]
          "unmet_W": pd.Series,                    # residual [W]
          "E_elec_kWh_total": pd.Series,           # electric input energy
          "E_gas_kWh_total":  pd.Series,           # gas input energy
          "E_th_elec_kWh": pd.Series,              # thermal served by electric techs
          "E_th_gas_kWh":  pd.Series,              # thermal served by gas techs
          "per_tech_W": dict[str, pd.Series],      # thermal per tech [W]
          "per_tech_elec_kWh": dict[str, pd.Series],
          "per_tech_gas_kWh":  dict[str, pd.Series],
          "per_tech_eff": dict[str, pd.Series],    # COP/EER/η masked to NaN when off
        }
    """
    idx = load_W.index
    dt_h = _dt_hours(idx)
    Tout_arr = Tout.values.astype(float)

    chain_sorted = sorted(chain or [], key=lambda d: int(d.get("priority", 1)))

    residual = load_W.astype(float).clip(lower=0.0).copy()
    served_sum = pd.Series(0.0, index=idx)

    # Totals
    E_elec_kWh_total = pd.Series(0.0, index=idx)
    E_gas_kWh_total  = pd.Series(0.0, index=idx)

    # Thermal delivered by elec/gas techs (for effective efficiency)
    E_th_elec_kWh = pd.Series(0.0, index=idx)
    E_th_gas_kWh  = pd.Series(0.0, index=idx)

    # Per-tech outputs
    per_tech_W = {}
    per_tech_elec = {}
    per_tech_gas  = {}
    per_tech_eff  = {}  # COP or η time-series used (masked when off)

    for i, tech in enumerate(chain_sorted):
        name = str(tech.get("name", f"Tech #{i+1}"))
        Pmax_W = float(tech.get("Pmax", 0.0))
        fuel   = (tech.get("fuel") or "elec").lower().strip()
        curve_key = str(tech.get("curve_key", "")).strip()

        # Pick the right curve dictionary
        if mode == "heating" and fuel == "gas":
            curve_dict = BOILER_EFF
        elif mode == "heating" and fuel == "elec":
            curve_dict = HEATING_COP
        else:  # cooling → electric
            curve_dict = COOLING_EER

        eff_vec = np.maximum(_eval_curve(curve_dict, curve_key, Tout_arr), 0.01)

        # Allocate thermal
        q_vals = np.minimum(residual.values, Pmax_W)
        q_W = pd.Series(q_vals, index=idx)
        residual -= q_W
        served_sum += q_W

        # Energy inputs
        e_th_kWh = (q_vals * dt_h.values) / 1000.0
        if fuel == "elec":
            e_elec = (q_vals / eff_vec) * dt_h.values / 1000.0
            e_gas  = np.zeros_like(e_elec)
            E_th_elec_kWh += pd.Series(e_th_kWh, index=idx)
        else:
            e_gas  = (q_vals / eff_vec) * dt_h.values / 1000.0
            e_elec = np.zeros_like(e_gas)
            E_th_gas_kWh  += pd.Series(e_th_kWh, index=idx)

        E_elec_kWh_total += pd.Series(e_elec, index=idx)
        E_gas_kWh_total  += pd.Series(e_gas,  index=idx)

        # Store splits
        per_tech_W[name]     = q_W
        per_tech_elec[name]  = pd.Series(e_elec, index=idx)
        per_tech_gas[name]   = pd.Series(e_gas,  index=idx)

        # Efficiency shown only when tech is contributing
        eff_series = pd.Series(eff_vec, index=idx)
        per_tech_eff[name] = eff_series.where(q_W > 0, other=np.nan)

    unmet = residual.clip(lower=0.0)

    return {
        "served_W_total": served_sum,
        "unmet_W": unmet,
        "E_elec_kWh_total": E_elec_kWh_total,
        "E_gas_kWh_total":  E_gas_kWh_total,
        "E_th_elec_kWh": E_th_elec_kWh,
        "E_th_gas_kWh":  E_th_gas_kWh,
        "per_tech_W": per_tech_W,
        "per_tech_elec_kWh": per_tech_elec,
        "per_tech_gas_kWh":  per_tech_gas,
        "per_tech_eff": per_tech_eff,
    }

def _dhw_weights24(preset: str, peakiness: float, main_hour: int) -> np.ndarray:
    """Return 24 hourly weights that sum to 1.0."""
    pnorm = (preset or "").strip().lower().replace(" ", "")
    sigma = np.clip(3.0 * np.exp(-0.35 * float(peakiness)), 0.20, 3.0)

    w = np.ones(24, dtype=float)
    if pnorm in ("flat",):
        w[:] = 1.0
    else:
        def bump(center, sig):
            h = np.arange(24)
            d = np.minimum(np.abs(h - center), 24 - np.abs(h - center))  # circular distance
            return np.exp(-(d**2) / (2 * sig**2))

        morning = bump(int(main_hour) % 24, sigma)
        evening = bump((int(main_hour) + 12) % 24, sigma)

        if pnorm in ("morning+evening", "morningevening"):
            w = morning + 0.9 * evening
        elif pnorm in ("evening-heavy", "eveningheavy"):
            w = 0.6 * morning + 1.4 * evening
        else:
            w = morning + 0.9 * evening

    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 0 else np.ones(24) / 24

def _monthly_factor_12(amp: float, peak_month: int) -> np.ndarray:
    m = np.arange(1, 13)
    phase = (m - peak_month) / 12.0
    f = 1.0 + amp * np.cos(2 * np.pi * phase)
    return f

def _expand_24h_weights_to_index(weights24: np.ndarray, index: pd.DatetimeIndex) -> pd.Series:
    df = pd.DataFrame(index=index)
    df["hour"] = df.index.hour
    hourly = pd.Series({h: float(weights24[h]) for h in range(24)}, name="w")
    df = df.join(hourly, on="hour")
    day_sum = df["w"].groupby(df.index.date).transform("sum").replace(0, 1.0)
    return df["w"] / day_sum

def _get_dhw_chain_in_W(cfg: dict, is_central: bool) -> list[dict]:
    """Return DHW chain with Pmax in per-house Watts."""
    chain = (cfg.get("dhw") or {}).get("chain") or []
    if not is_central:
        # return [{**d, "Pmax": float(d.get("Pmax", 0.0))} for d in chain]
        # Scale DHW tech capacities to 1/4 (per-house W already)
        return [{**d, "Pmax": float(d.get("Pmax", 0.0)) * 0.25} for d in chain]
    nr = max(int(cfg.get("nr_houses", 1)), 1)
    arch = cfg.get("arch_cfg") or {}
    diversity = float(arch.get("diversity", 1.0))
    scale = (1000.0 * diversity) / nr
    # return [{**d, "Pmax": float(d.get("Pmax", 0.0)) * scale} for d in chain]
    # Central plant capacities are given in kW → per-house W via `scale`.
    # Then quarter them for DHW only.
    return [{**d, "Pmax": float(d.get("Pmax", 0.0)) * scale * 0.25} for d in chain]

def _dispatch_with_losses(load_W: pd.Series, loss_frac: float, Tout: pd.Series,
                          chain: list[dict], mode: str):
    """
    Dispatch service load including network losses in a single pass:
      required_W = load_W / (1 - loss_frac)

    Returns (res_total, ratio_service) where:
      - res_total: _dispatch_merit_order(...) result for required_W
      - ratio_service: Series in [0,1] to split total into service vs losses
    """
    loss_frac = float(loss_frac)
    if loss_frac <= 0:
        return _dispatch_merit_order(load_W, Tout, chain, mode), None

    required_W = load_W / (1.0 - loss_frac)
    res_total = _dispatch_merit_order(required_W, Tout, chain, mode)

    ratio_service = (load_W / required_W).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(0.0, 1.0)
    return res_total, ratio_service

# Main function
def run_scenario(wx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Run a full per-house scenario: RC simulation → merit-order dispatch (incl. losses) →
    energy/cost/CO₂ KPIs.

    Parameters
    ----------
    wx : pd.DataFrame
        Weather/time-step inputs required by `simulate_RC`. Must include
        outdoor temperature column (commonly "T_out") and a monotonic
        DatetimeIndex or a "time" column (will be set as index).
    cfg : dict
        Scenario configuration from the UI layer. Expected keys include:
          - Building & comfort: "building_type", "year_built", "V", "ACH",
            "A_solar", "g_solar", "T0", "set_heat", "set_cool",
            "cooling_enabled".
          - Architecture: "is_central" or "arch" == "Centralized",
            and tech chains:
              * Decentralized: "heating_chain", "cooling_chain" (Pmax in W)
              * Centralized: "arch_cfg" with "heating_chain"/"cooling_chain"
                (Pmax in kW), "diversity", "dist_loss_*", "aux_kw_total"
                (plant-level; must be converted to per-house upstream if used
                as `aux_kw_house` here).
          - Prices & CO₂: "price_elec", "price_gas",
            "co2_per_kwh_elec", "co2_per_kwh_gas",
            optional legacy "price", "co2_per_kwh".
          - Embodied CO₂: "co2_embodied_kg" (decentralized) or central plant /
            distribution embodied values (handled upstream; only per-house
            embodied is accumulated here via "co2_embodied_kg").
          - Auxiliaries (per house): "aux_kw_house" [kW] (electric).

    Returns
    -------
    pd.DataFrame
        Time-indexed per-house results with at least:
          Thermal:
            - Q_hvac_W (from RC), Q_heat_W, Q_cool_W, Q_unmet_heat_W, Q_unmet_cool_W
          Dispatch (per tech):
            - Q_heat_{name}_W, Q_cool_{name}_W
            - E_elec_heat_{name}_kWh, E_gas_heat_{name}_kWh
            - E_elec_cool_{name}_kWh
            - COP_heat_{name}, EER_cool_{name}  # masked when off
            - If centralized losses are present:
              E_elec_heat_dist_{name}_kWh, E_gas_heat_dist_{name}_kWh,
              E_elec_cool_dist_{name}_kWh, E_gas_cool_dist_{name}_kWh
          Totals (service + distribution + auxiliaries):
            - E_elec_kWh_heat_service, E_gas_kWh_heat_service
            - E_elec_kWh_cool_service, E_gas_kWh_cool_service
            - E_elec_kWh_dist_heat, E_gas_kWh_dist_heat
            - E_elec_kWh_dist_cool, E_gas_kWh_dist_cool
            - E_elec_kWh_aux
            - E_elec_kWh, E_gas_kWh
          Efficiencies:
            - COP_heating_eff (elec-only), HEAT_eff_overall (dual-fuel),
              EER_cooling_eff
          Economics & CO₂:
            - Cost_eur
            - CO2_kg
            - CO2_cum_house_kg (cumulative operational CO₂ + embodied offset)
    """
    params = {
        "building_type": cfg["building_type"],
        "year_built":    cfg["year_built"],
        "V": cfg["V"], "ACH": cfg["ACH"],
        "A_solar": cfg["A_solar"], "g_solar": cfg["g_solar"],
        "min_on_minutes": 10.0, "min_off_minutes": 10.0,
    }
    is_central = bool(cfg.get("is_central", cfg.get("arch") == "Centralized"))
    cooling_enabled = bool(cfg.get("cooling_enabled", True))

    # Capacity caps (per-house W)
    heat_chain, cool_chain = _get_chains_in_W(cfg, is_central, cooling_enabled)
    cap_heat_W = sum(t["Pmax"] for t in heat_chain)
    cap_cool_W = sum(t["Pmax"] for t in cool_chain)

    # Simulate temperatures & delivered thermal with those caps.
    sim = simulate_RC(
        wx, params,
        T0=cfg["T0"],
        setpoint_heat=cfg["set_heat"],
        setpoint_cool=(cfg["set_cool"] if cooling_enabled and cfg.get("set_cool") is not None else float("inf")),
        P_max_heat=cap_heat_W,
        P_max_cool=cap_cool_W,
    )
    if sim.index.name != "time":
        sim = sim.set_index("time")

    # Common time-step durations
    dt_h = _dt_hours(sim.index)

    # Split delivered thermal by sign
    sim["Q_heat_W"] = sim["Q_hvac_W"].clip(lower=0.0)
    sim["Q_cool_W"] = (-sim["Q_hvac_W"]).clip(lower=0.0)
    Tout = sim["T_out"]

    # =================== HEATING (single dispatch incl. losses) ==================
    Lh = 0.0
    if is_central:
        arch = cfg.get("arch_cfg") or {}
        Lh = float(arch.get("dist_loss_heat_pct", 0.0)) / 100.0

    heat_all, heat_ratio = _dispatch_with_losses(sim["Q_heat_W"], Lh, Tout, heat_chain, mode="heating")
    ratio_h = 1.0 if heat_ratio is None else heat_ratio

    # Service portions
    sim["E_elec_kWh_heat_service"] = heat_all["E_elec_kWh_total"] * ratio_h
    sim["E_gas_kWh_heat_service"]  = heat_all["E_gas_kWh_total"]  * ratio_h

    sim["E_th_elec_kWh_heat"] = heat_all["E_th_elec_kWh"] * ratio_h
    sim["E_th_gas_kWh_heat"]  = heat_all["E_th_gas_kWh"]  * ratio_h

    sim["Q_unmet_heat_W"] = heat_all["unmet_W"] * (0.0 if heat_ratio is None else ratio_h)

    # Distribution portions (totals)
    sim["E_elec_kWh_dist_heat"] = 0.0 if heat_ratio is None else heat_all["E_elec_kWh_total"] * (1.0 - ratio_h)
    sim["E_gas_kWh_dist_heat"]  = 0.0 if heat_ratio is None else heat_all["E_gas_kWh_total"]  * (1.0 - ratio_h)

    # Per-tech columns
    for tech_name, sW in heat_all["per_tech_W"].items():
        sim[f"Q_heat_{tech_name}_W"] = sW * ratio_h
    for tech_name, sElec in heat_all["per_tech_elec_kWh"].items():
        if heat_ratio is None:
            sim[f"E_elec_heat_{tech_name}_kWh"] = sElec
        else:
            sim[f"E_elec_heat_{tech_name}_kWh"]      = sElec * ratio_h
            sim[f"E_elec_heat_dist_{tech_name}_kWh"] = sElec * (1.0 - ratio_h)
    for tech_name, sGas in heat_all["per_tech_gas_kWh"].items():
        if heat_ratio is None:
            sim[f"E_gas_heat_{tech_name}_kWh"] = sGas
        else:
            sim[f"E_gas_heat_{tech_name}_kWh"]      = sGas * ratio_h
            sim[f"E_gas_heat_dist_{tech_name}_kWh"] = sGas * (1.0 - ratio_h)
    for tech_name, s_eff in heat_all["per_tech_eff"].items():
        sim[f"COP_heat_{tech_name}"] = s_eff

    # =================== COOLING (single dispatch incl. losses) ==================
    Lc = 0.0
    if is_central:
        Lc = float((cfg.get("arch_cfg") or {}).get("dist_loss_cool_pct", 0.0)) / 100.0

    cool_chain_eff = cool_chain if cooling_enabled else []
    cool_all, cool_ratio = _dispatch_with_losses(sim["Q_cool_W"], Lc, Tout, cool_chain_eff, mode="cooling")
    ratio_c = 1.0 if cool_ratio is None else cool_ratio

    sim["E_elec_kWh_cool_service"] = cool_all["E_elec_kWh_total"] * ratio_c
    sim["E_gas_kWh_cool_service"]  = cool_all["E_gas_kWh_total"]  * ratio_c
    sim["Q_unmet_cool_W"]          = cool_all["unmet_W"]          * (0.0 if cool_ratio is None else ratio_c)

    sim["E_elec_kWh_dist_cool"] = 0.0 if cool_ratio is None else cool_all["E_elec_kWh_total"] * (1.0 - ratio_c)
    sim["E_gas_kWh_dist_cool"]  = 0.0 if cool_ratio is None else cool_all["E_gas_kWh_total"]  * (1.0 - ratio_c)

    for tech_name, sW in cool_all["per_tech_W"].items():
        sim[f"Q_cool_{tech_name}_W"] = sW * ratio_c
    for tech_name, sElec in cool_all["per_tech_elec_kWh"].items():
        if cool_ratio is None:
            sim[f"E_elec_cool_{tech_name}_kWh"] = sElec
        else:
            sim[f"E_elec_cool_{tech_name}_kWh"]      = sElec * ratio_c
            sim[f"E_elec_cool_dist_{tech_name}_kWh"] = sElec * (1.0 - ratio_c)
    for tech_name, sGas in cool_all["per_tech_gas_kWh"].items():
        if cool_ratio is None:
            sim[f"E_gas_cool_{tech_name}_kWh"] = sGas
        else:
            sim[f"E_gas_cool_{tech_name}_kWh"]      = sGas * ratio_c
            sim[f"E_gas_cool_dist_{tech_name}_kWh"] = sGas * (1.0 - ratio_c)
    for tech_name, s_eff in cool_all["per_tech_eff"].items():
        sim[f"EER_cool_{tech_name}"] = s_eff

    # =================== DHW (single dispatch incl. losses) ======================
    dhw_chain_W = []
    dhw_cfg = cfg.get("dhw") or {}
    if bool(dhw_cfg.get("enabled", True)):
        prof = dhw_cfg.get("profile") or {}
        season = dhw_cfg.get("season") or {}

        w24 = _dhw_weights24(
            preset=prof.get("preset", "Morning+Evening"),
            peakiness=float(prof.get("peakiness", 1.5)),
            main_hour=int(prof.get("main_peak_hour", 7)),
        )
        day_w = _expand_24h_weights_to_index(w24, sim.index)
        m12 = _monthly_factor_12(
            amp=float(season.get("amp", 0.15)),
            peak_month=int(season.get("peak_month", 1)),
        )
        month_factor = pd.Series(m12[(sim.index.month - 1).values], index=sim.index)

        occ   = float(dhw_cfg.get("occupancy", 3.0))
        lppd  = float(dhw_cfg.get("lppd", 50.0))
        tap   = float(dhw_cfg.get("tap_C", 55.0))
        cold  = float(dhw_cfg.get("cold_C", 10.0))
        deltaT = max(0.0, tap - cold)
        daily_kWh = (occ * lppd * 4.186 * deltaT) / 3600.0  # kWh/day

        step_kWh = daily_kWh * day_w * month_factor
        Q_dhw_W = (step_kWh * 1000.0) / dt_h.replace(0, np.nan)
        Q_dhw_W = Q_dhw_W.fillna(0.0).clip(lower=0.0)
        sim["Q_dhw_need_W"] = Q_dhw_W  # preserve the original need

        dhw_chain_W = _get_dhw_chain_in_W(cfg, is_central)

        Ldhw = 0.0
        if is_central:
            Ldhw = float((cfg.get("arch_cfg") or {}).get("dist_loss_dhw_pct", 0.0)) / 100.0

        dhw_all, dhw_ratio = _dispatch_with_losses(Q_dhw_W, Ldhw, Tout, dhw_chain_W, mode="heating")
        ratio_d = 1.0 if dhw_ratio is None else dhw_ratio

        # Served at service-end (exclude network losses via ratio_d)
        sim["Q_dhw_served_W"] = dhw_all["served_W_total"] * ratio_d
        # Backward-compat: many plots expect Q_dhw_W to be *delivered*
        sim["Q_dhw_W"]        = sim["Q_dhw_served_W"]
        # Define unmet against the original *need*, not dispatcher internals
        sim["Q_unmet_dhw_W"]  = (sim["Q_dhw_need_W"] - sim["Q_dhw_served_W"]).clip(lower=0.0)

        sim["E_elec_kWh_dhw_service"] = dhw_all["E_elec_kWh_total"] * ratio_d
        sim["E_gas_kWh_dhw_service"]  = dhw_all["E_gas_kWh_total"]  * ratio_d

        for tech_name, s in dhw_all["per_tech_W"].items():
            sim[f"Q_dhw_{tech_name}_W"] = s * ratio_d
        for tech_name, s in dhw_all["per_tech_elec_kWh"].items():
            if dhw_ratio is None:
                sim[f"E_elec_dhw_{tech_name}_kWh"] = s
            else:
                sim[f"E_elec_dhw_{tech_name}_kWh"]      = s * ratio_d
                sim[f"E_elec_dhw_dist_{tech_name}_kWh"] = s * (1.0 - ratio_d)
        for tech_name, s in dhw_all["per_tech_gas_kWh"].items():
            if dhw_ratio is None:
                sim[f"E_gas_dhw_{tech_name}_kWh"] = s
            else:
                sim[f"E_gas_dhw_{tech_name}_kWh"]      = s * ratio_d
                sim[f"E_gas_dhw_dist_{tech_name}_kWh"] = s * (1.0 - ratio_d)
        for tech_name, s_eff in dhw_all["per_tech_eff"].items():
            sim[f"COP_dhw_{tech_name}"] = s_eff

        # Thermal splits for DHW (kWh_th, service-end)
        sim["E_th_elec_kWh_dhw"] = dhw_all["E_th_elec_kWh"] * ratio_d
        sim["E_th_gas_kWh_dhw"]  = dhw_all["E_th_gas_kWh"]  * ratio_d

        # Dist totals
        sim["E_elec_kWh_dist_dhw"] = 0.0 if dhw_ratio is None else dhw_all["E_elec_kWh_total"] * (1.0 - ratio_d)
        sim["E_gas_kWh_dist_dhw"]  = 0.0 if dhw_ratio is None else dhw_all["E_gas_kWh_total"]  * (1.0 - ratio_d)

    else:
        sim["Q_dhw_W"] = 0.0
        sim["Q_unmet_dhw_W"] = 0.0
        sim["E_elec_kWh_dhw_service"] = 0.0
        sim["E_gas_kWh_dhw_service"]  = 0.0
        sim["E_th_elec_kWh_dhw"] = 0.0
        sim["E_th_gas_kWh_dhw"]  = 0.0
        sim["E_elec_kWh_dist_dhw"] = 0.0
        sim["E_gas_kWh_dist_dhw"]  = 0.0

    # =================== Effective efficiencies (time-series) ====================
    # Heating
    E_heat_elec_th = sim["E_th_elec_kWh_heat"]
    E_heat_gas_th  = sim["E_th_gas_kWh_heat"]
    E_heat_total_th = E_heat_elec_th + E_heat_gas_th
    E_heat_elec_in  = sim["E_elec_kWh_heat_service"]
    E_heat_gas_in   = sim["E_gas_kWh_heat_service"]
    E_heat_in_total = E_heat_elec_in + E_heat_gas_in

    sim["COP_heating_eff"]  = np.where(E_heat_elec_in > 0, E_heat_elec_th / E_heat_elec_in, np.nan)
    sim["HEAT_eff_overall"] = np.where(E_heat_in_total > 0, E_heat_total_th / E_heat_in_total, np.nan)

    # Cooling
    E_cool_elec_th = cool_all["E_th_elec_kWh"] * ratio_c
    E_cool_gas_th  = cool_all["E_th_gas_kWh"]  * ratio_c
    E_cool_total_th  = E_cool_elec_th + E_cool_gas_th
    E_cool_elec_in = sim["E_elec_kWh_cool_service"]
    E_cool_gas_in  = sim["E_gas_kWh_cool_service"]
    E_cool_total_in  = E_cool_elec_in + E_cool_gas_in

    sim["EER_cooling_eff"] = np.where(E_cool_elec_in > 0, E_cool_elec_th / E_cool_elec_in, np.nan)
    sim["COOL_eff_overall"] = np.where(E_cool_total_in > 0, E_cool_total_th / E_cool_total_in, np.nan)

    # DHW
    E_elec_dhw_in = sim["E_elec_kWh_dhw_service"]
    E_gas_dhw_in  = sim["E_gas_kWh_dhw_service"]
    E_th_elec_dhw = sim["E_th_elec_kWh_dhw"]
    E_th_gas_dhw  = sim["E_th_gas_kWh_dhw"]
    E_th_tot_dhw  = E_th_elec_dhw + E_th_gas_dhw

    sim["COP_dhw_eff"]     = np.where(E_elec_dhw_in > 0, E_th_elec_dhw / E_elec_dhw_in, np.nan)
    sim["DHW_eff_overall"] = np.where((E_elec_dhw_in + E_gas_dhw_in) > 0,
                                    E_th_tot_dhw / (E_elec_dhw_in + E_gas_dhw_in),
                                    np.nan)

    # Combined Heating + DHW (electric-only COP)
    E_th_elec_heat = sim["E_th_elec_kWh_heat"]
    E_elec_heat_in = sim["E_elec_kWh_heat_service"]
    E_th_elec_sum  = E_th_elec_heat + E_th_elec_dhw
    E_elec_in_sum  = E_elec_heat_in + E_elec_dhw_in
    sim["COP_heat_dhw_eff"] = np.where(E_elec_in_sum > 0, E_th_elec_sum / E_elec_in_sum, np.nan)

    # Auxiliaries (per-house) — assumed electric
    aux_kw_house = float(cfg.get("aux_kw_house", 0.0))
    sim["E_elec_kWh_aux"] = aux_kw_house * dt_h if aux_kw_house > 0 else 0.0

    # Final energy totals
    sim["E_elec_kWh"] = (
        sim["E_elec_kWh_heat_service"] + sim["E_elec_kWh_cool_service"] + sim["E_elec_kWh_dhw_service"]
        + sim["E_elec_kWh_dist_heat"]    + sim["E_elec_kWh_dist_cool"]    + sim["E_elec_kWh_dist_dhw"]
        + sim["E_elec_kWh_aux"]
    )

    sim["E_gas_kWh"] = (
        sim["E_gas_kWh_heat_service"] + sim["E_gas_kWh_cool_service"] + sim["E_gas_kWh_dhw_service"]
        + sim["E_gas_kWh_dist_heat"]    + sim["E_gas_kWh_dist_cool"]    + sim["E_gas_kWh_dist_dhw"]
    )

    # CO₂ and Cost (dual-fuel)
    price_elec = float(cfg.get("price_elec", cfg.get("price", 0.30)))
    price_gas  = float(cfg.get("price_gas", 0.10))
    co2_elec   = float(cfg.get("co2_per_kwh_elec", cfg.get("co2_per_kwh", 0.40)))
    co2_gas    = float(cfg.get("co2_per_kwh_gas", 0.20))

    sim["Cost_eur"] = sim["E_elec_kWh"] * price_elec + sim["E_gas_kWh"] * price_gas
    sim["CO2_kg"]   = sim["E_elec_kWh"] * co2_elec  + sim["E_gas_kWh"] * co2_gas

    embodied = float(cfg.get("co2_embodied_kg", 0.0))
    sim["CO2_cum_house_kg"] = sim["CO2_kg"].cumsum() + embodied

    return sim
