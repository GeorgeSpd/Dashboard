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
# pylint: disable=W0613

"""
Streamlit sidebar UI for configuring a single simulation scenario.

Overview
--------
This UI renders one scenario as an expandable section in the sidebar with
structured inputs for:

- Building characteristics
- Comfort settings
- Equipment / system architecture (decentralized or centralized)
- Economics (CAPEX and energy prices)
- CO₂ accounting (embodied and operational)

The function `scenario_controls()` collects all inputs for one scenario and
returns a single configuration dictionary. That dictionary is later adapted and
passed to the simulation layer.

Key Concepts
------------
- *Decentralized*: All equipment is per-house. Capacities are input in **W**.
- *Centralized*: Shared plant with distribution to houses. Plant capacities are
  input in **kW**. Additional distribution losses, auxiliaries, and diversity
  are included.

Session State & Keys
--------------------
All widgets are keyed with the provided scenario label (e.g., "A", "B") to avoid
collisions when multiple scenarios are on screen simultaneously.

Return Schema (summary)
-----------------------
`scenario_controls(label, defaults)` returns a `dict` with:

Identification:
    - label, name

Building:
    - nr_houses, building_type, year_built, V, ACH, A_solar, g_solar

Comfort:
    - T0, set_heat, set_cool (None if cooling disabled)
    - cooling_enabled

Energy Prices:
    - price_elec, price_gas

Economics:
    - capex_house (decentralized), or plant_capex_eur and
      dist_capex_per_house_eur (centralized)

CO₂:
    - co2_embodied_kg (decentralized) or plant_embodied_kg and
      dist_embodied_per_house_kg (centralized)
    - co2_per_kwh_elec, co2_per_kwh_gas

Architecture:
    - arch, is_central
    - heating_chain, cooling_chain (priority-ordered tech lists)
    - arch_cfg (centralized-only extras; empty for decentralized)

Dependencies
------------
Relies on performance curve registries imported from `core.efficiency`:
`HEATING_ALL`, `COOLING_EER`, and `FUEL_BY_HEAT_CURVE`.
"""

from __future__ import annotations
import streamlit as st
from core.efficiency import HEATING_ALL, DHW_ALL, COOLING_EER, FUEL_BY_HEAT_CURVE

# Helper
def multi_tech_selector(
    section_title: str,
    label_name: str,
    curve_dict: dict,
    unit_label: str = "W",
    base_key: str = "heat",
    defaults: dict | None = None,
    fuel_by_key: dict | None = None,
):
    """
    Render a compact selector for 1-4 prioritized technologies of the same type
    (e.g., heating or cooling) and return their configuration.

    The UI includes:
      - A counter for the number of technologies (1-4)
      - For each technology (priority = row index):
          * Performance curve dropdown (keys from `curve_dict`)
          * Maximum capacity input (units controlled by `unit_label`)
          * Read-only priority badge (1 = highest priority / dispatched first)

    Parameters
    ----------
    section_title : str
        Human-friendly name shown above the technology list (informative only).
    label_name : str
        Scenario label used to namespace Streamlit widget keys (e.g., "A").
    curve_dict : dict
        Mapping of curve keys to curve objects/metadata. Only keys are used for
        the dropdown (e.g., `HEATING_ALL` or `COOLING_EER`).
    unit_label : str, optional
        Capacity unit shown in the UI. Use "W" for room systems and "kW" for
        plant-level equipment. Default is "W".
    base_key : str, optional
        Base key prefix to build widget keys and to read defaults, e.g.:
        "heat", "cool", "plant_heat", "plant_cool". Default is "heat".
    defaults : dict | None, optional
        Optional initial values. Recognized keys (0-based index i):
          - f"{base_key}_n": int (1..4), number of technologies
          - f"{base_key}_curve_{i}": str, preselected curve key
          - f"{base_key}_Pmax_{i}": float, capacity
    fuel_by_key : dict | None, optional
        Optional mapping `curve_key -> fuel_string` to annotate each technology
        with a fuel type (e.g., "elec", "gas"). If not provided, defaults to
        "elec" for all rows.

    Returns
    -------
    list[dict]
        Priority-ordered list of technology dicts:
        [
          {
            "name": "Tech #<priority>",
            "curve_key": "<key from curve_dict>",
            "Pmax": <float>,                # capacity in units of `unit_label`
            "priority": <int>,              # 1..N (1 = highest)
            "fuel": "<fuel string>",        # from fuel_by_key or "elec"
          },
          ...
        ]
    """
    defaults = defaults or {}
    curve_choices = list(curve_dict.keys())

    n_key = f"{base_key}_n_{label_name}"
    n_default = min(max(int(defaults.get(f"{base_key}_n", 1)), 1), 4)

    # number of technologies with +/- like scenarios
    n = st.number_input(
        "Number of technologies",
        min_value=1,
        max_value=4,
        value=n_default,
        step=1,
        key=n_key,
    )

    techs = []
    for i in range(n):
        idx = i + 1
        with st.container(border=True):
            cols = st.columns([2, 2, 1])

            # Default curve for this row
            row_curve_default = defaults.get(
                f"{base_key}_curve_{i}",
                (curve_choices[0] if curve_choices else "")
            )
            try:
                curve_index = curve_choices.index(row_curve_default)
            except ValueError:
                curve_index = 0

            curve = cols[0].selectbox(
                f"Tech #{idx} performance curve",
                curve_choices,
                index=curve_index,
                key=f"{base_key}_curve_{i}_{label_name}",
            )

            # Default capacity
            cap_default = float(defaults.get(
                f"{base_key}_Pmax_{i}",
                6000.0 if unit_label == "W" else 10.0
            ))
            step = 500.0 if unit_label == "W" else 1.0

            cap = cols[1].number_input(
                f"Tech #{idx} max capacity [{unit_label}]",
                value=cap_default,
                step=step,
                key=f"{base_key}_Pmax_{i}_{label_name}",
            )

            cols[2].markdown(f"**Priority:** {idx}")

            # Fuel from mapping (fallback "elec")
            fuel = (fuel_by_key or {}).get(curve, "elec")

            techs.append({
                "name": f"Tech #{idx}",
                "curve_key": curve,
                "Pmax": float(cap),
                "priority": idx,
                "fuel": fuel,
            })

    return techs

def render_dhw_section(
    label_name: str,
    is_central: bool,
    dhw_defaults: dict,
    dist_loss_default: float = 0.05,
) -> tuple[dict, list[dict], float]:
    """
    Render a DHW block consistent with Heating/Cooling sections.

    Returns:
      dhw_inputs: dict with occupancy/Lppd/temps + profile + season knobs
      dhw_chain:  prioritized tech list from multi_tech_selector
    """
    base_key   = "plant_dhw" if is_central else "dhw"
    title      = "DHW Plant" if is_central else "Domestic Hot Water"
    unit_label = "kW" if is_central else "W"

    # Extract starting values (fallbacks are safe)
    _prof = (dhw_defaults.get("profile") or {})
    _seas = (dhw_defaults.get("season")  or {})

    occ   = float(dhw_defaults.get("occupancy", 3.0))
    lppd  = float(dhw_defaults.get("lppd", 50.0))
    tap   = float(dhw_defaults.get("tap_C", 60.0))
    cold  = float(dhw_defaults.get("cold_C", 10.0))

    preset     = _prof.get("preset", "Evening-heavy")
    peakiness  = float(_prof.get("peakiness", 10))
    main_hour  = int(_prof.get("main_peak_hour", 7))

    amp        = float(_seas.get("amp", 0.40))
    peak_month = int(_seas.get("peak_month", 1))

    with st.expander(title, expanded=False):
        # Inputs (same layout style as your Heating/Cooling)
        c1, c2 = st.columns(2)
        with c1:
            occ  = st.number_input("Occupancy [persons]", min_value=0.0, value=occ,  key=f"{base_key}_occ_{label_name}")
        with c2:
            lppd = st.number_input("Use [L/person/day]",  min_value=0.0, value=lppd, key=f"{base_key}_lppd_{label_name}")
            
        c1, c2 = st.columns(2)
        with c1:
            tap  = st.number_input("Charge setpoint [°C]",   min_value=30.0, max_value=70.0, value=tap,  key=f"{base_key}_tap_{label_name}")
        with c2:
            cold = st.number_input("Cold inlet [°C]",     min_value=0.0,  max_value=30.0, value=cold, key=f"{base_key}_cold_{label_name}")

        p1, p2, p3 = st.columns([2,1,1])
        with p1:
            preset = st.selectbox(
                "Preset", ["Flat","Morning + Evening","Evening-heavy"],
                index={"Flat":0, "Morning + Evening":1, "Evening-heavy":2}[preset if preset in ["Flat","Morning + Evening","Evening-heavy"] else "Morning + Evening"],
                key=f"{base_key}_prof_preset_{label_name}"
            )
        with p2:
            peakiness = st.slider("Peakiness", 0.0, 10.0, peakiness, 0.1, key=f"{base_key}_peakiness_{label_name}")
        with p3:
            main_hour = st.slider("Main peak hour", 0, 23, main_hour, 1, key=f"{base_key}_mainhour_{label_name}")

        s1, s2 = st.columns(2)
        with s1:
            amp  = st.slider("Season amplitude", 0.0, 0.5, amp, 0.01, key=f"{base_key}_season_amp_{label_name}")
        with s2:
            peak_month = st.slider("Peak month (1–12)", 1, 12, peak_month, 1, key=f"{base_key}_season_peak_{label_name}")

        # Build UI defaults for multi_tech_selector from dhw_defaults["chain"] if present
        ui_defaults = {}
        chain_init = (dhw_defaults.get("chain") or []) if isinstance(dhw_defaults, dict) else []
        if chain_init:
            ui_defaults[f"{base_key}_n"] = min(max(len(chain_init), 1), 4)
            for i, t in enumerate(chain_init[:4]):
                ui_defaults[f"{base_key}_curve_{i}"] = t.get("curve_key", "")
                ui_defaults[f"{base_key}_Pmax_{i}"]  = float(t.get("Pmax", 6000.0 if unit_label == "W" else 10.0))
        else:
            # fallback: allow a simple top-level override key like defaults["dhw_default_curve"]
            maybe_curve = (dhw_defaults.get("default_curve") if isinstance(dhw_defaults, dict) else None) or "HP_low_temp"
            ui_defaults[f"{base_key}_n"] = 1
            ui_defaults[f"{base_key}_curve_0"] = maybe_curve
            ui_defaults[f"{base_key}_Pmax_0"]  = 6000.0 if unit_label == "W" else 10.0

        # Technologies (exactly like Heating/Cooling)
        dhw_chain = multi_tech_selector(
            section_title=title,
            label_name=label_name,
            curve_dict=DHW_ALL,
            unit_label=unit_label,
            base_key=base_key,
            defaults=ui_defaults,
            fuel_by_key=FUEL_BY_HEAT_CURVE,
        )

        if is_central:
            dist_loss_dhw_pct = st.number_input(
                "Distribution loss (DHW) [%]",
                min_value=0.0, max_value=50.0,
                value=5.0,
                step=1.0,
                key=f"dhl_dhw_{label_name}",
            )
        else:
            dist_loss_dhw_pct = 0.0

    dhw_inputs = {
        "occupancy": float(occ),
        "lppd": float(lppd),
        "tap_C": float(tap),
        "cold_C": float(cold),
        "profile": {
            "preset": preset,
            "peakiness": float(peakiness),
            "main_peak_hour": int(main_hour),
        },
        "season": {
            "amp": float(amp),
            "peak_month": int(peak_month),
        },
    }
    return dhw_inputs, dhw_chain, dist_loss_dhw_pct

# Main function
def scenario_controls(label_name: str, defaults: dict | None = None) -> dict:
    """
    Render all sidebar controls for a single scenario and return a normalized
    configuration dictionary.

    The UI is grouped into:
      1) Building Specs
      2) Comfort
      3) Equipment (decentralized vs centralized)
      4) Economics (CAPEX, energy prices)
      5) CO₂ (embodied, operational)

    Architecture Modes
    ------------------
    - Decentralized:
        * Per-house equipment
        * Capacities in **W**
        * `heating_chain` / `cooling_chain` are room/system-level
        * `capex_house` and `co2_embodied_kg` are per house
    - Centralized:
        * Shared plant + distribution
        * Plant capacities in **kW**
        * Extra fields for distribution losses, auxiliaries, diversity
        * Adds plant/distribution CAPEX and embodied CO₂
        * `arch_cfg` includes central-only details

    Parameters
    ----------
    label_name : str
        Scenario identifier used to namespace Streamlit widget keys ("A", "B", ...).
    defaults : dict | None, optional
        Prefill values for any of the fields below. Recognized keys include:

        Identification:
            - "name": str

        Building:
            - "nr_houses": int
            - "building_type": str
            - "year_built": int
            - "V": float                     # m³
            - "ACH": float                   # 1/h
            - "A_solar": float               # m²
            - "g_solar": float               # 0..1

        Comfort:
            - "cooling_enabled": bool
            - "T0": float                    # °C
            - "set_heat": float              # °C
            - "set_cool": float              # °C

        Equipment chains (see `multi_tech_selector` for per-row keys):
            - For decentralized: "heat_*" / "cool_*"
            - For centralized:   "plant_heat_*" / "plant_cool_*"

        Economics:
            - "capex_house": float
            - "plant_capex_eur": float
            - "dist_capex_per_house_eur": float
            - "price_elec" (or legacy "price"): float  # €/kWh
            - "price_gas": float                       # €/kWh

        CO₂:
            - "co2_embodied_kg": float                 # kg per house (decentralized)
            - "plant_embodied_kg": float               # kg total
            - "dist_embodied_per_house_kg": float      # kg per house
            - "co2_per_kwh_elec" (or legacy "co2_per_kwh"): float  # kg/kWh
            - "co2_per_kwh_gas": float                 # kg/kWh

    Returns
    -------
    dict
        A normalized scenario configuration:

        Identification:
            - "label": str
            - "name": str

        Building:
            - "nr_houses": int
            - "building_type": str
            - "year_built": int
            - "V": float
            - "ACH": float
            - "A_solar": float
            - "g_solar": float

        Comfort:
            - "T0": float
            - "set_heat": float
            - "set_cool": float | None
            - "cooling_enabled": bool

        Prices & Economics:
            - "price_elec": float
            - "price_gas": float
            - "capex_house": float                     # 0.0 for centralized
            - "plant_capex_eur": float | None          # None for decentralized
            - "dist_capex_per_house_eur": float | None

        CO₂:
            - "co2_embodied_kg": float                 # 0.0 for centralized
            - "co2_per_kwh_elec": float
            - "co2_per_kwh_gas": float
            - "plant_embodied_kg": float | None
            - "dist_embodied_per_house_kg": float | None

        Architecture:
            - "arch": Literal["Decentralized","Centralized"]
            - "is_central": bool
            - "heating_chain": list[dict]              # from multi_tech_selector
            - "cooling_chain": list[dict]
            - "arch_cfg": dict                         # centralized-only, else {}

              For centralized mode, `arch_cfg` includes:
                {
                  "diversity": float,
                  "dist_loss_heat_pct": float,
                  "dist_loss_cool_pct": float,
                  "heating_chain": list[dict],   # plant (kW)
                  "cooling_chain": list[dict],   # plant (kW)
                  "aux_kw_total": float,
                  "plant_capex_eur": float | None,
                  "dist_capex_per_house_eur": float | None,
                  "plant_embodied_kg": float | None,
                  "dist_embodied_per_house_kg": float | None,
                }
    """
    if defaults is None:
        defaults = {}

    # ---- DHW defaults & safe pre-inits ----
    _dhw_defaults = defaults.get("dhw", {})
    _dhw_enabled_default = bool(defaults.get("dhw_enabled", _dhw_defaults.get("enabled", True)))

    # basic inputs
    dhw_enabled   = _dhw_enabled_default
    dhw_occ       = float(_dhw_defaults.get("occupancy", 3.0))
    dhw_lppd      = float(_dhw_defaults.get("lppd", 50.0))
    dhw_tap       = float(_dhw_defaults.get("tap_C", 40.0))
    dhw_cold      = float(_dhw_defaults.get("cold_C", 10.0))

    # profile knobs
    _prof = _dhw_defaults.get("profile", {})
    dhw_profile_preset = _prof.get("preset", "Morning + Evening")
    dhw_peakiness      = float(_prof.get("peakiness", 1.5))
    dhw_main_hour      = int(_prof.get("main_peak_hour", 7))

    # season knobs
    _seas = _dhw_defaults.get("season", {})
    dhw_season_amp     = float(_seas.get("amp", 0.15))
    dhw_season_peak    = int(_seas.get("peak_month", 1))

    # tech chain pre-init
    dhw_state = {
        "occupancy": dhw_occ,
        "lppd": dhw_lppd,
        "tap_C": dhw_tap,
        "cold_C": dhw_cold,
        "profile": {
            "preset": dhw_profile_preset,
            "peakiness": dhw_peakiness,
            "main_peak_hour": dhw_main_hour,
        },
        "season": {
            "amp": dhw_season_amp,
            "peak_month": dhw_season_peak,
        },
    }
    dhw_chain = []

    # Unique name management
    name_key = f"name_{label_name}"
    default_name = defaults.get("name", f"Scenario {label_name}")
    current_name = st.session_state.get(name_key, default_name)
    expander_title = (current_name or "").strip() or default_name

    with st.sidebar.expander(expander_title, expanded=False):
        name = st.text_input("Scenario name", value=current_name, key=name_key)

        arch = st.selectbox("System architecture", ["Decentralized", "Centralized"], index=0, key=f"arch_{label_name}")
        is_central = arch == "Centralized"

        st.caption("Building")

        # Building & Model
        with st.expander("Building Specs", expanded=False):
            nr_houses = st.number_input("Number of houses", 1, 100, defaults.get("nr_houses", 1), 1, key=f"nr_{label_name}")
            building_type = st.selectbox(
                "Building type",
                ["Detached", "Semi-detached", "Terraced", "Apartment (mid-floor)", "Apartment (corner/top)"],
                index=0, key=f"btype_{label_name}",
            )
            year_built = st.number_input("Year built", 1900, 2100, defaults.get("year_built", 2005), 1, key=f"year_{label_name}")
            V = st.number_input("Indoor volume V [m³]", value=defaults.get("V", 250.0), key=f"V_{label_name}")
            ACH = st.number_input("Infiltration ACH [1/h]", value=defaults.get("ACH", 0.5), step=0.1, key=f"ACH_{label_name}")
            A_solar = st.number_input("Effective solar aperture A_solar [m²]", value=defaults.get("A_solar", 10.0), key=f"A_{label_name}")
            g_solar = st.slider("Solar gain factor g_solar", 0.0, 1.0, defaults.get("g_solar", 0.5), 0.05, key=f"g_{label_name}")

        # Comfort
        with st.expander("Comfort", expanded=False):
            st.caption("Mode")
            dhw_enabled = st.checkbox("DHW enabled", value=_dhw_enabled_default, key=f"dhw_enabled_{label_name}")
            cooling_enabled = st.checkbox("Cooling enabled", value=defaults.get("cooling_enabled", True), key=f"cooling_enabled_{label_name}")
            set_heat = st.number_input("Heating setpoint [°C]", value=defaults.get("set_heat", 20.0), format="%.1f", step=0.5, key=f"seth_{label_name}")
            set_cool = st.number_input("Cooling setpoint [°C]", value=defaults.get("set_cool", 26.0), format="%.1f", step=0.5, key=f"setc_{label_name}") if cooling_enabled else None
            T0 = st.number_input("Initial indoor temperature [°C]", value=defaults.get("T0", set_heat), format="%.1f", step=0.5, key=f"T0_{label_name}")

        # Equipment
        st.caption("Equipment")
        heating_chain = []
        cooling_chain = []
        plant_heating_chain = []
        plant_cooling_chain = []

        dist_loss_heat_pct = 0.0
        dist_loss_cool_pct = 0.0
        dist_loss_dhw_pct = 0.0
        aux_kw_total = 0.0
        diversity = 1.0
        plant_capex_eur = None
        dist_capex_per_house = None
        plant_embodied_kg = None
        dist_embodied_per_house_kg = None
        arch_cfg = {}

        if not is_central:
            with st.expander("Heating", expanded=False):
                heating_chain = multi_tech_selector(
                    section_title="Heating",
                    label_name=label_name,
                    curve_dict=HEATING_ALL,
                    unit_label="W",
                    base_key="heat",
                    defaults=defaults,
                    fuel_by_key=FUEL_BY_HEAT_CURVE,
                )

            if dhw_enabled:
                dhw_state, dhw_chain, _ = render_dhw_section(
                    label_name=label_name,
                    is_central=False,
                    dhw_defaults=(defaults.get("dhw") or _dhw_defaults),
                )

            if cooling_enabled:
                with st.expander("Cooling", expanded=False):
                    cooling_chain = multi_tech_selector(
                        section_title="Cooling",
                        label_name=label_name,
                        curve_dict=COOLING_EER,
                        unit_label="W",
                        base_key="cool",
                        defaults=defaults,
                    )
            else:
                cooling_chain = []
        else:
            with st.expander("Heating Plant", expanded=False):
                plant_heating_chain = multi_tech_selector(
                    section_title="Heating Plant",
                    label_name=label_name,
                    curve_dict=HEATING_ALL,
                    unit_label="kW",
                    base_key="plant_heat",
                    defaults=defaults,
                    fuel_by_key=FUEL_BY_HEAT_CURVE,
                )
                dist_loss_heat_pct = st.number_input("Distribution loss (heating) [%]", 0.0, 50.0, 5.0, 1.0, key=f"dhl_{label_name}")

            if dhw_enabled:
                dhw_state, dhw_chain, dist_loss_dhw_pct = render_dhw_section(
                    label_name=label_name,
                    is_central=True,
                    dhw_defaults=(defaults.get("dhw") or _dhw_defaults),
                    dist_loss_default=dist_loss_dhw_pct,
                )

            if cooling_enabled:
                with st.expander("Cooling Plant", expanded=False):
                    plant_cooling_chain = multi_tech_selector(
                        section_title="Cooling Plant",
                        label_name=label_name,
                        curve_dict=COOLING_EER,
                        unit_label="kW",
                        base_key="plant_cool",
                        defaults=defaults,
                    )
                    dist_loss_cool_pct = st.number_input("Distribution loss (cooling) [%]", 0.0, 50.0, 5.0, 1.0, key=f"dcl_{label_name}")
            else:
                dist_loss_cool_pct = 0.0
                plant_cooling_chain = []

            aux_kw_total = st.number_input("Plant auxiliaries (pumps/controls) [kW]", 0.0, 100.0, 0.5, 0.1, key=f"aux_{label_name}")
            diversity  = st.slider("Diversity / simultaneity factor", 0.3, 1.0, 1.0, 0.05, key=f"div_{label_name}")

            arch_cfg = {
                "diversity": diversity,
                "dist_loss_heat_pct": dist_loss_heat_pct,
                "dist_loss_dhw_pct":  dist_loss_dhw_pct,
                "dist_loss_cool_pct": dist_loss_cool_pct,
                "heating_chain": plant_heating_chain,
                "cooling_chain": plant_cooling_chain,
                "aux_kw_total": aux_kw_total,
            }

        # Economics
        st.caption("Economics")
        if not is_central:
            with st.expander("Capital Costs", expanded=False):
                capex_house = st.number_input("CAPEX per house [€]", value=defaults.get("capex_house", 0.0), step=500.0, key=f"capex_{label_name}")
        else:
            capex_house = 0.0
            with st.expander("Capital Costs", expanded=False):
                plant_capex_eur = st.number_input("Plant CAPEX [€]", 0.0, 1e9, 60000.0, 1000.0, key=f"capexp_{label_name}")
                dist_capex_per_house = st.number_input("Distribution CAPEX per house [€]", 0.0, 1e7, 1500.0, 100.0, key=f"capexd_{label_name}")
        # Operational Costs
        with st.expander("Operational Costs", expanded=False):
            price_elec = st.number_input("Electricity price [€/kWh]", value=defaults.get("price_elec", defaults.get("price", 0.30)), step=0.01, key=f"price_elec_{label_name}")
            price_gas  = st.number_input("Gas price [€/kWh]",        value=defaults.get("price_gas", 0.10), step=0.01, key=f"price_gas_{label_name}")

        # CO₂
        st.caption("CO₂")
        if not is_central:
            with st.expander("Embodied CO₂", expanded=False):
                co2_embodied_kg = st.number_input("Capital CO₂ per house [kg CO₂]", value=defaults.get("co2_embodied_kg", 0.0), step=50.0, key=f"co2_embodied_{label_name}")
        else:
            co2_embodied_kg = 0.0
            with st.expander("Embodied CO₂", expanded=False):
                plant_embodied_kg = st.number_input("Plant Embodied CO₂ [kg]", 0.0, 1e9, 0.0, 100.0, key=f"co2p_{label_name}")
                dist_embodied_per_house_kg = st.number_input("Distribution Embodied CO₂ per house [kg]", 0.0, 1e6, 0.0, 10.0, key=f"co2d_{label_name}")
        # Operational CO₂
        with st.expander("Operational CO₂", expanded=False):
            co2_elec = st.number_input("Electricity CO₂ [kg/kWh]", value=defaults.get("co2_per_kwh_elec", defaults.get("co2_per_kwh", 0.40)), step=0.05, key=f"co2_elec_{label_name}")
            co2_gas  = st.number_input("Gas CO₂ [kg/kWh]",         value=defaults.get("co2_per_kwh_gas", 0.20), step=0.05, key=f"co2_gas_{label_name}")
            
            arch_cfg.update({
                "plant_capex_eur": plant_capex_eur,
                "dist_capex_per_house_eur": dist_capex_per_house,
                "plant_embodied_kg": plant_embodied_kg,
                "dist_embodied_per_house_kg": dist_embodied_per_house_kg,
            })

            dhw_config = {
                "enabled": bool(dhw_enabled),
                **dhw_state,
                "chain": dhw_chain if dhw_enabled else [],
            }

    return {
        "label": label_name,
        "name": name,
        "nr_houses": nr_houses,
        "building_type": building_type,
        "year_built": year_built,
        "V": V, "ACH": ACH, "A_solar": A_solar, "g_solar": g_solar,
        "T0": T0, "set_heat": set_heat, "set_cool": set_cool,

        "cooling_enabled": cooling_enabled,
        "dhw": dhw_config,

        # Price & CO2:
        "price_elec": price_elec,
        "price_gas":  price_gas,
        "capex_house": capex_house,
        "co2_embodied_kg": co2_embodied_kg,
        "co2_per_kwh_elec": co2_elec,
        "co2_per_kwh_gas":  co2_gas,

        # Architecture:
        "arch": arch,
        "is_central": is_central,

        # Priority chains (always present, structure depends on arch):
        "heating_chain": plant_heating_chain if is_central else heating_chain,
        "cooling_chain": plant_cooling_chain if is_central else cooling_chain,

        # Central-only extras (None otherwise):
        "arch_cfg": arch_cfg if is_central else {},
        "plant_capex_eur": plant_capex_eur,
        "dist_capex_per_house_eur": dist_capex_per_house,
        "plant_embodied_kg": plant_embodied_kg,
        "dist_embodied_per_house_kg": dist_embodied_per_house_kg,
    }
