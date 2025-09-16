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

from __future__ import annotations
import numpy as np
import pandas as pd

from core.sim import simulate_RC
from core.efficiency import HEATING_COP, COOLING_EER, BOILER_EFF


# --------------------------- helpers ---------------------------

def _dt_hours(index: pd.DatetimeIndex) -> pd.Series:
    """Vector of step durations (hours) for possibly irregular time series."""
    if len(index) <= 1:
        return pd.Series([1.0], index=index)
    t = pd.to_datetime(index)
    dt = (t[1:] - t[:-1]).total_seconds() / 3600.0
    dt = np.r_[dt[0], dt]
    return pd.Series(dt, index=index, dtype=float)


def _eval_curve(curve_dict: dict, key: str, Tout: np.ndarray) -> np.ndarray:
    """Return a vector of COP/EER given a curve key (callable or constant)."""
    item = curve_dict.get(key, 0.0)
    if callable(item):
        return np.asarray(item(Tout), dtype=float)
    return np.full_like(Tout, float(item), dtype=float)


def _get_chains_in_W(cfg: dict, is_central: bool, cooling_enabled: bool) -> tuple[list[dict], list[dict]]:
    """
    Return (heating_chain_W, cooling_chain_W) where each entry has:
      - same keys as input tech dicts (name, curve_key, priority, ...)
      - Pmax scaled to **per-house Watts**
        * decentralized: already in W, pass through
        * centralized: kW -> W, then *diversity / nr_houses*
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
    Allocate 'load_W' across 'chain' in priority order.
    Returns electricity & gas consumptions, per-tech splits, and effective efficiency series.
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
        e_th_kWh = (q_vals * dt_h.values) / 1000.0  # thermal kWh delivered by this tech
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

    # Effective efficiencies (electric-only + overall)
    # (Caller can compute ratios they need; we return components)
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
        "per_tech_eff": per_tech_eff,  # dict of series (COP/η or EER)
    }





# --------------------------- main entry ---------------------------

def run_scenario(wx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    1) Run simulate_RC with capacity limits so physics + comfort respond.
    2) Split delivered loads by merit order chain(s) to compute electricity per tech.
    3) Add network losses (central) and auxiliaries; compute CO2.
    4) Provide effective and per-tech COP/EER series for plotting.
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
    # Use huge COP/EER so simulate_RC's *own* electricity is ~0 (we'll recompute with dispatch).
    sim = simulate_RC(
        wx, params,
        T0=cfg["T0"],
        setpoint_heat=cfg["set_heat"],
        setpoint_cool=(cfg["set_cool"] if cooling_enabled and cfg.get("set_cool") is not None else float("inf")),
        P_max_heat=cap_heat_W,
        P_max_cool=cap_cool_W,
        COP_heat=lambda T: 1e6,  # neutralize internal electric calc
        EER_cool=lambda T: 1e6,
    )
    if sim.index.name != "time":
        sim = sim.set_index("time")

    # Split delivered thermal by sign
    sim["Q_heat_W"] = sim["Q_hvac_W"].clip(lower=0.0)
    sim["Q_cool_W"] = (-sim["Q_hvac_W"]).clip(lower=0.0)
    Tout = sim["T_out"]

    # Merit-order dispatch with fuel
    heat_res = _dispatch_merit_order(sim["Q_heat_W"], Tout, heat_chain, mode="heating")
    cool_res = _dispatch_merit_order(sim["Q_cool_W"], Tout, cool_chain if cooling_enabled else [], mode="cooling")

    # Base energy (service, before network losses/aux)
    sim["E_elec_kWh_heat_service"] = heat_res["E_elec_kWh_total"]
    sim["E_gas_kWh_heat_service"]  = heat_res["E_gas_kWh_total"]
    sim["E_elec_kWh_cool_service"] = cool_res["E_elec_kWh_total"]
    sim["E_gas_kWh_cool_service"]  = cool_res["E_gas_kWh_total"]  # likely 0 for our tech set

    # Effective efficiencies (time-series)
    dt_h = _dt_hours(sim.index)
    E_heat_elec_th = heat_res["E_th_elec_kWh"]
    E_heat_gas_th  = heat_res["E_th_gas_kWh"]
    E_heat_total_th = E_heat_elec_th + E_heat_gas_th
    E_heat_elec_in  = heat_res["E_elec_kWh_total"]
    E_heat_gas_in   = heat_res["E_gas_kWh_total"]
    E_heat_in_total = E_heat_elec_in + E_heat_gas_in

    sim["COP_heating_eff"]  = np.where(E_heat_elec_in > 0, E_heat_elec_th / E_heat_elec_in, np.nan)
    sim["HEAT_eff_overall"] = np.where(E_heat_in_total > 0, E_heat_total_th / E_heat_in_total, np.nan)

    # (cooling is electric in your tech set; if you add gas-driven chillers later this still works)
    E_cool_elec_th = cool_res["E_th_elec_kWh"]
    E_cool_elec_in = cool_res["E_elec_kWh_total"]

    sim["EER_cooling_eff"] = np.where(E_cool_elec_in > 0, E_cool_elec_th / E_cool_elec_in, np.nan)

    # Per-tech series for plots
    for tech_name, s in heat_res["per_tech_W"].items():
        sim[f"Q_heat_{tech_name}_W"] = s
    for tech_name, s in cool_res["per_tech_W"].items():
        sim[f"Q_cool_{tech_name}_W"] = s
    for tech_name, s in heat_res["per_tech_elec_kWh"].items():
        sim[f"E_elec_heat_{tech_name}_kWh"] = s
    for tech_name, s in heat_res["per_tech_gas_kWh"].items():
        sim[f"E_gas_heat_{tech_name}_kWh"] = s
    for tech_name, s in cool_res["per_tech_elec_kWh"].items():
        sim[f"E_elec_cool_{tech_name}_kWh"] = s

    # Efficiency per tech (COP or η for heating; EER for cooling). Hidden when off (NaN)
    for tech_name, s_eff in heat_res["per_tech_eff"].items():
        sim[f"COP_heat_{tech_name}"] = s_eff
    for tech_name, s_eff in cool_res["per_tech_eff"].items():
        sim[f"EER_cool_{tech_name}"] = s_eff

    # --- Centralized network losses: dispatch extra with same chains (fuel-aware) ---
    sim["E_elec_kWh_dist_heat"] = 0.0
    sim["E_gas_kWh_dist_heat"]  = 0.0
    sim["E_elec_kWh_dist_cool"] = 0.0
    sim["E_gas_kWh_dist_cool"]  = 0.0

    if is_central:
        arch = cfg.get("arch_cfg") or {}
        Lh = float(arch.get("dist_loss_heat_pct", 0.0)) / 100.0
        Lc = float(arch.get("dist_loss_cool_pct", 0.0)) / 100.0

        if Lh > 0:
            mult_h = (1.0 / (1.0 - Lh)) - 1.0
            extra_Q_W = sim["Q_heat_W"] * mult_h
            extra_heat = _dispatch_merit_order(extra_Q_W, Tout, heat_chain, mode="heating")

            sim["E_elec_kWh_dist_heat"] = extra_heat["E_elec_kWh_total"]
            sim["E_gas_kWh_dist_heat"]  = extra_heat["E_gas_kWh_total"]

            # NEW: per-tech dist series (elec & gas)
            for tech_name, s in extra_heat["per_tech_elec_kWh"].items():
                sim[f"E_elec_heat_dist_{tech_name}_kWh"] = s
            for tech_name, s in extra_heat["per_tech_gas_kWh"].items():
                sim[f"E_gas_heat_dist_{tech_name}_kWh"] = s

        if Lc > 0 and cooling_enabled:
            mult_c = (1.0 / (1.0 - Lc)) - 1.0
            extra_Qc_W = sim["Q_cool_W"] * mult_c
            extra_cool = _dispatch_merit_order(extra_Qc_W, Tout, cool_chain, mode="cooling")

            sim["E_elec_kWh_dist_cool"] = extra_cool["E_elec_kWh_total"]
            sim["E_gas_kWh_dist_cool"]  = extra_cool["E_gas_kWh_total"]  # likely 0 with current tech set

            # NEW: per-tech dist series (mostly electric)
            for tech_name, s in extra_cool["per_tech_elec_kWh"].items():
                sim[f"E_elec_cool_dist_{tech_name}_kWh"] = s
            for tech_name, s in extra_cool["per_tech_gas_kWh"].items():
                sim[f"E_gas_cool_dist_{tech_name}_kWh"] = s

    # Auxiliaries (per-house) — assumed electric
    aux_kw_house = float(cfg.get("aux_kw_house", 0.0))
    sim["E_elec_kWh_aux"] = aux_kw_house * dt_h if aux_kw_house > 0 else 0.0

    # Final energy totals
    sim["E_elec_kWh"] = (
        sim["E_elec_kWh_heat_service"] + sim["E_elec_kWh_cool_service"]
        + sim["E_elec_kWh_dist_heat"] + sim["E_elec_kWh_dist_cool"] + sim["E_elec_kWh_aux"]
    )
    sim["E_gas_kWh"] = (
        sim["E_gas_kWh_heat_service"] + sim["E_gas_kWh_cool_service"]
        + sim["E_gas_kWh_dist_heat"] + sim["E_gas_kWh_dist_cool"]
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

    # Expose unmet loads (if chain empty/small)
    sim["Q_unmet_heat_W"] = heat_res["unmet_W"]
    sim["Q_unmet_cool_W"] = cool_res["unmet_W"]

    return sim
