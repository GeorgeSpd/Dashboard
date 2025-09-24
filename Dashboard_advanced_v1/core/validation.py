"""
core/validation.py

Validation and testing utilities for the energy simulation project.

- Validates that weather dataframes follow the expected schema.
- Provides synthetic weather data for testing without relying on an API.
- Runs a series of self-tests on the building simulation to check invariants
  like energy positivity, timeline sorting, and determinism.
"""

import pandas as pd
import numpy as np
from core.sim import simulate_RC

REQUIRED_WEATHER_COLS = {"T_out", "RH_out", "wind", "G_solar"}


def validate_weather_schema(df: pd.DataFrame) -> None:
    """
    Validate that a weather dataframe matches the expected schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate. Must be indexed by a DatetimeIndex and contain
        the columns {"T_out", "RH_out", "wind", "G_solar"}.

    Raises
    ------
    AssertionError
        If the index type, required columns, or chronological order are not correct.
    """
    assert isinstance(df.index, pd.DatetimeIndex), "weather index must be DatetimeIndex"
    missing = REQUIRED_WEATHER_COLS.difference(df.columns)
    assert not missing, f"weather df missing columns: {missing}"
    assert df.index.is_monotonic_increasing, "weather times must be sorted"


def synthetic_weather(n: int = 24, start=None, tz: str = "Europe/Amsterdam") -> pd.DataFrame:
    """
    Generate synthetic hourly weather data for testing.

    Parameters
    ----------
    n : int, optional
        Number of hours to generate (default: 24).
    start : datetime-like, optional
        Start time. If None, uses the current time in the given timezone.
    tz : str, optional
        IANA timezone string (default: "Europe/Amsterdam").

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by time with outdoor temperature, humidity, wind,
        and solar radiation profiles.
    """
    start = pd.Timestamp.now(tz) if start is None else pd.Timestamp(start).tz_convert(tz)
    times = pd.date_range(start=start.floor("H"), periods=n, freq="H")
    tbase = 5 + 7 * np.sin(np.linspace(0, 2 * np.pi, n))
    g = np.clip(600 * np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, n)), 0, None)
    df = pd.DataFrame({
        "time": times,
        "T_out": tbase,
        "RH_out": np.clip(70 - 10 * np.sin(np.linspace(0, 2 * np.pi, n)), 20, 100),
        "wind": np.clip(3 + np.random.randn(n), 0, None),
        "G_solar": g,
    }).set_index("time")
    return df


def run_self_tests():
    """
    Run built-in self-tests for the weather validation and simulation logic.

    Returns
    -------
    list of tuple
        Each element is (test_name, ok, message) where:
        - test_name : str, name of the test
        - ok : bool, True if the test passed
        - message : str, details or error description
    """
    results = []
    try:
        wx = synthetic_weather(48)
        validate_weather_schema(wx)
        results.append(("weather_schema", True, "weather df schema OK"))
    except AssertionError as e:
        results.append(("weather_schema", False, str(e)))
        return results

    try:
        # dict-based params (no BuildingParams)
        params = {
            "C": 30e6,
            "R": 0.005,
            "A_solar": 10.0,
            "g_solar": 0.5,
            "ACH": 0.5,
            "V": 250,
        }
        sim = simulate_RC(wx, params, T0=18, setpoint_heat=20, setpoint_cool=26)
        assert (sim["E_elec_kWh"] >= 0).all(), "electric energy must be non-negative"
        assert sim.index.is_monotonic_increasing, "sim timeline must be sorted"
        assert sim["Q_hvac_W"].gt(0).any(), "should heat at least once when T0 < setpoint"
        results.append(("simulate_basics", True, "simulation invariants OK"))
    except AssertionError as e:
        results.append(("simulate_basics", False, str(e)))

    try:
        params = {
            "C": 30e6,
            "R": 0.005,
            "A_solar": 10.0,
            "g_solar": 0.5,
            "ACH": 0.5,
            "V": 250,
        }
        sim1 = simulate_RC(wx, params, T0=21)
        sim2 = simulate_RC(wx, params, T0=21)
        pd.testing.assert_frame_equal(sim1, sim2, check_dtype=False)
        results.append(("determinism", True, "deterministic outputs for same inputs"))
    except AssertionError as e:
        results.append(("determinism", False, str(e)))

    return results
