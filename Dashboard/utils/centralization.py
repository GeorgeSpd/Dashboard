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
Utility functions for translating centralized plant inputs
into effective per-house parameters.

- Adjusts plant capacity with diversity factor
- Converts total plant capacity into per-house Pmax values
- Maps performance baseline curves to heating/cooling systems
- Allocates plant CAPEX and embodied CO₂ per house
- Divides auxiliaries among houses

Used to normalize scenario configs so that centralized
and decentralized architectures are treated consistently
downstream in the simulation.
"""

from __future__ import annotations
from core.efficiency import HEATING_COP, COOLING_EER
from .common import _f


def adapt_to_per_house(cfg: dict) -> dict:
    """
    Translate centralized plant inputs into effective per-house parameters.

    Parameters
    ----------
    cfg : dict
        Scenario configuration dictionary, as returned from `scenario_controls`.
        Must contain at least:
        - "arch" : str
        - "nr_houses" : int
        - "arch_cfg" : dict with centralized parameters if applicable.

    Returns
    -------
    dict
        Updated configuration dictionary with effective per-house values:
        - "Pmax_heat" : float, per-house max heating power [W]
        - "Pmax_cool" : float | None, per-house max cooling power [W]
        - "heat_sys"  : str, selected heating system key
        - "cool_sys"  : str | None, selected cooling system key
        - "capex_house" : float, allocated per-house CAPEX [€]
        - "co2_embodied_kg" : float, allocated per-house embodied CO₂ [kg]
        - "aux_kw_house" : float, allocated per-house auxiliaries [kW]

    Notes
    -----
    - For decentralized configs, the input dictionary is returned unchanged.
    - Ensures that downstream simulation always has per-house values,
      regardless of centralization.
    """
    if cfg.get("arch") != "Centralized":
        out = dict(cfg)
        # Ensure attribution keys exist for plotting uniformity
        out.setdefault("capex_house_user", float(out.get("capex_house", 0.0)))
        out.setdefault("capex_alloc_plant", 0.0)
        out.setdefault("capex_alloc_dist",  0.0)
        out.setdefault("embodied_user_kg",  float(out.get("co2_embodied_kg", 0.0)))
        out.setdefault("embodied_plant_kg", 0.0)
        out.setdefault("embodied_dist_kg",  0.0)
        out.setdefault("aux_kw_house", 0.0)
        return out

    arch = cfg.get("arch_cfg") or {}
    n = max(int(_f(cfg.get("nr_houses"), 1)), 1)

    # Capacity/diversity → per house
    div = _f(arch.get("diversity"), 0.7)
    plant_kw = _f(arch.get("plant_kw"), 0.0)
    per_house_heat_kw = plant_kw * div / n

    plant_cool_kw = _f(arch.get("plant_cool_kw"), 0.0) if cfg.get("cooling_enabled", True) else 0.0
    per_house_cool_kw = plant_cool_kw * div / n

    Pmax_heat = max(_f(cfg.get("Pmax_heat"), 0.0), per_house_heat_kw * 1000.0)
    Pmax_cool = None
    if cfg.get("cooling_enabled", True) and plant_cool_kw > 0.0:
        Pmax_cool = max(_f(cfg.get("Pmax_cool"), 0.0), per_house_cool_kw * 1000.0)

    # Perf baselines
    heat_sys_key = arch.get("heat_sys_base") or cfg.get("heat_sys") or next(iter(HEATING_COP.keys()))
    cool_sys_key = None
    if cfg.get("cooling_enabled", True) and (plant_cool_kw > 0.0):
        cool_sys_key = arch.get("cool_sys_base") or cfg.get("cool_sys") or next(iter(COOLING_EER.keys()))

    # CAPEX/embodied allocation
    plant_capex = _f(arch.get("plant_capex_eur"), 0.0)
    dist_capex_ph = _f(arch.get("dist_capex_per_house_eur"), 0.0)

    user_capex = cfg.get("capex_house")
    if isinstance(user_capex, (int, float)) and user_capex > 0:
        capex_house_eff = float(user_capex)
    else:
        capex_house_eff = plant_capex / n + dist_capex_ph

    embodied_house_eff = _f(cfg.get("co2_embodied_kg"), 0.0) \
                       + _f(arch.get("plant_embodied_kg"), 0.0) / n \
                       + _f(arch.get("dist_embodied_per_house_kg"), 0.0)

    # Auxiliaries share
    aux_kw_house = _f(arch.get("aux_kw_total"), 0.0) / n

    out = dict(cfg)
    # Per-house breakdowns
    capex_house_user = float(cfg.get("capex_house") or 0.0)
    capex_alloc_plant = _f(arch.get("plant_capex_eur"), 0.0) / n
    capex_alloc_dist  = _f(arch.get("dist_capex_per_house_eur"), 0.0)

    emb_user  = _f(cfg.get("co2_embodied_kg"), 0.0)
    emb_plant = _f(arch.get("plant_embodied_kg"), 0.0) / n
    emb_dist  = _f(arch.get("dist_embodied_per_house_kg"), 0.0)

    out.update({
        "Pmax_heat": Pmax_heat,
        "Pmax_cool": Pmax_cool,
        "heat_sys":  heat_sys_key,
        "cool_sys":  cool_sys_key,

        # Totals (per house)
        "capex_house": capex_house_eff,
        "co2_embodied_kg": embodied_house_eff,

        # Attribution (per house)
        "capex_house_user": capex_house_user,
        "capex_alloc_plant": capex_alloc_plant,
        "capex_alloc_dist":  capex_alloc_dist,

        "embodied_user_kg":  emb_user,
        "embodied_plant_kg": emb_plant,
        "embodied_dist_kg":  emb_dist,

        # Aux shares (per house)
        "aux_kw_house": aux_kw_house,
    })
    return out
