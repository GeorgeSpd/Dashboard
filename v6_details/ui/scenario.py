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
Streamlit UI for defining a single simulation scenario.

Each scenario is shown in the sidebar as an expandable section
with fields for:
- Building characteristics
- Comfort settings
- Equipment / system architecture
- Economics (CAPEX, prices)
- CO₂ (embodied + operational)

The returned configuration dict is later adapted and passed to
simulation functions.
"""

from __future__ import annotations
import streamlit as st
from core.efficiency import HEATING_ALL, COOLING_EER, FUEL_BY_HEAT_CURVE


def multi_tech_selector(
    section_title: str,
    label_name: str,
    curve_dict: dict,              # e.g., HEATING_ALL or COOLING_EER
    unit_label: str = "W",         # "W" for room systems, "kW" for plants
    base_key: str = "heat",        # "heat", "cool", "plant_heat", "plant_cool"
    defaults: dict | None = None,  # scenario defaults
    fuel_by_key: dict | None = None,   # <--- NEW: map curve_key -> "elec" | "gas"
):
    """
    Render 1-4 technologies with priorities.
    Returns: list of dicts like:
      [{"name": "Tech #1", "curve_key": "...", "Pmax": <float>, "priority": 1, "fuel": "elec"|"gas"}, ...]
    """
    defaults = defaults or {}
    curve_choices = list(curve_dict.keys())

    n_key = f"{base_key}_n_{label_name}"
    n_default = min(max(int(defaults.get(f"{base_key}_n", 1)), 1), 4)

    st.write("")  # small spacer
    n = st.slider("Number of technologies", 1, 4, n_default, key=n_key,
                  help="1 = only main tech; order = priority")

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

            # Default capacity for this row
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
                "curve_key": curve,         # key into HEATING/COOLING/BOILER dicts
                "Pmax": float(cap),         # W or kW depending on section
                "priority": idx,            # order = priority
                "fuel": fuel,
            })

    return techs

def scenario_controls(label_name: str, defaults: dict | None = None) -> dict:
    """
    Render sidebar controls for defining one scenario.

    Parameters
    ----------
    label_name : str
        Scenario identifier (e.g. "A", "B"). Used for widget keys
        to avoid collisions.
    defaults : dict | None, optional
        Prefill values for the fields. Keys can include:
        - "name", "nr_houses", "building_type", "year_built",
          "V", "ACH", "A_solar", "g_solar"
        - "T0", "set_heat", "set_cool"
        - "Pmax_heat", "Pmax_cool"
        - "heat_sys", "cool_sys"
        - "price", "capex_house"
        - "cooling_enabled"
        - "co2_embodied_kg", "co2_per_kwh"
        - For centralized configs: plant/distribution CAPEX & CO₂.

    Returns
    -------
    dict
        Scenario configuration dictionary with keys:
        - Identification: "label", "name"
        - Building: "nr_houses", "building_type", "year_built", "V", "ACH",
          "A_solar", "g_solar"
        - Comfort: "T0", "set_heat", "set_cool", "cooling_enabled"
        - Equipment: "Pmax_heat", "Pmax_cool", "heat_sys", "cool_sys",
          "arch", "arch_cfg"
        - Economics: "capex_house", "plant_capex_eur",
          "dist_capex_per_house_eur", "price"
        - CO₂: "co2_embodied_kg", "co2_per_kwh", "plant_embodied_kg",
          "dist_embodied_per_house_kg"
    """
    
    if defaults is None:
        defaults = {}

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
            cooling_enabled = st.checkbox("Cooling enabled", value=defaults.get("cooling_enabled", True), key=f"cooling_enabled_{label_name}")
            T0 = st.number_input("Initial indoor temperature [°C]", value=defaults.get("T0", 20.0), format="%.1f", step=0.5, key=f"T0_{label_name}")
            set_heat = st.number_input("Heating setpoint [°C]", value=defaults.get("set_heat", 20.0), format="%.1f", step=0.5, key=f"seth_{label_name}")
            set_cool = st.number_input("Cooling setpoint [°C]", value=defaults.get("set_cool", 26.0), format="%.1f", step=0.5, key=f"setc_{label_name}") if cooling_enabled else None

        # Equipment
        st.caption("Equipment")
        heating_chain = []
        cooling_chain = []
        plant_heating_chain = []
        plant_cooling_chain = []

        dist_loss_heat_pct = 0.0
        dist_loss_cool_pct = 0.0
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
                "dist_loss_cool_pct": dist_loss_cool_pct,
                "heating_chain": plant_heating_chain,   # capacities in kW
                "cooling_chain": plant_cooling_chain,   # capacities in kW
                "aux_kw_total": aux_kw_total,           # plant-level aux; adapt_to_per_house should divide per house
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

    return {
        "label": label_name,
        "name": name,
        "nr_houses": nr_houses,
        "building_type": building_type,
        "year_built": year_built,
        "V": V, "ACH": ACH, "A_solar": A_solar, "g_solar": g_solar,
        "T0": T0, "set_heat": set_heat, "set_cool": set_cool,

        # Price & CO2:
        "price_elec": price_elec,
        "price_gas":  price_gas,
        "capex_house": capex_house,
        "cooling_enabled": cooling_enabled,
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
