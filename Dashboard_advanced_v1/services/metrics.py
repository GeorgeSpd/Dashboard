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
Utility functions for computing performance indicators
from building energy simulations.

Functions
---------
- annual_totals_from_sim : Aggregate annual electricity, costs, and CO₂.
- scop_seer              : Compute seasonal performance factors (SCOP/SEER)
                           from a simulation view.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def annual_totals_from_sim(sim: pd.DataFrame, price_per_kwh: float, co2_per_kwh: float) -> tuple[float, float, float]:
    """
    Compute annual totals from a full-year or TMY simulation.

    Parameters
    ----------
    sim : pandas.DataFrame
        Simulation results with at least the column:
        - "E_elec_kWh" : electricity consumption per timestep [kWh].
    price_per_kwh : float
        Electricity price [€/kWh].
    co2_per_kwh : float
        CO₂ intensity [kg/kWh].

    Returns
    -------
    tuple of (float, float, float)
        - Total electricity use [kWh]
        - Annual operational cost [€]
        - Annual operational CO₂ emissions [kg]

    Notes
    -----
    Works both for 8760-hour years and representative TMY datasets,
    regardless of leap year adjustments.
    """
    elec_kwh = float(sim["E_elec_kWh"].sum())
    op_eur   = elec_kwh * float(price_per_kwh)
    op_co2   = elec_kwh * float(co2_per_kwh)
    return elec_kwh, op_eur, op_co2


def scop_seer(view: pd.DataFrame) -> tuple[float, float]:
    """
    Compute SCOP (seasonal COP for heating) and SEER (seasonal EER for cooling).

    Parameters
    ----------
    view : pandas.DataFrame
        Subset of simulation results with at least:
        - "Q_hvac_W"    : net HVAC power [W] (+heating, -cooling).
        - "E_hvac_kWh"  : thermal energy exchanged per timestep [kWh].
        - "E_elec_kWh"  : electricity consumption per timestep [kWh].

    Returns
    -------
    tuple of (float, float)
        - SCOP : Seasonal Coefficient of Performance for heating
        - SEER : Seasonal Energy Efficiency Ratio for cooling

    Notes
    -----
    - If no heating or cooling is active, the respective metric is NaN.
    - Masks are used to separate heating (Q_hvac_W > 0) and cooling (Q_hvac_W < 0).
    """
    heat_mask = view["Q_hvac_W"] > 0
    cool_mask = view["Q_hvac_W"] < 0
    Q_heat_kWh = view.loc[heat_mask, "E_hvac_kWh"].sum()
    E_heat_el_kWh = view.loc[heat_mask, "E_elec_kWh"].sum()
    SCOP = (Q_heat_kWh / E_heat_el_kWh) if E_heat_el_kWh > 0 else np.nan
    Q_cool_kWh = view.loc[cool_mask, "E_hvac_kWh"].clip(upper=0).abs().sum()
    E_cool_el_kWh = view.loc[cool_mask, "E_elec_kWh"].sum()
    SEER = (Q_cool_kWh / E_cool_el_kWh) if E_cool_el_kWh > 0 else np.nan
    return SCOP, SEER
