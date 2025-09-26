"""
Utility helpers shared across the dashboard modules.

Functions
---------
- _f : safe float conversion with fallback default.
- make_unique : ensures unique names in a list by appending suffixes.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def _f(x, default=0.0) -> float:
    """
    Safe float conversion.

    Parameters
    ----------
    x : any
        Input value to convert.
    default : float, optional
        Value to return if `x` is None, NaN, or cannot be cast to float.
        Defaults to 0.0.

    Returns
    -------
    float
        Converted float value, or the fallback default.

    Examples
    --------
    >>> _f("3.14")
    3.14
    >>> _f(None, default=1.0)
    1.0
    >>> _f("bad", default=-1.0)
    -1.0
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    if v != v:  # NaN check
        return float(default)
    return v

def make_unique(names: list[str]) -> list[str]:
    """
    Ensure unique names by appending suffixes if duplicates are found.

    Parameters
    ----------
    names : list of str
        Input names, possibly with duplicates.

    Returns
    -------
    list of str
        List with guaranteed unique names. If duplicates are found,
        they are suffixed with " (2)", " (3)", etc.

    Examples
    --------
    >>> make_unique(["Scenario", "Scenario", "Alt"])
    ['Scenario', 'Scenario (2)', 'Alt']
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        base = (n or "Scenario").strip() or "Scenario"
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base} ({seen[base]})")
    return out

def normalize_weights_24(weights: list[float]) -> np.ndarray:
    """Clamp negatives to 0, handle None/NaN, and normalize to sum=1 over 24 slots."""
    w = np.array([(0.0 if (x is None or np.isnan(x) or x < 0) else float(x)) for x in weights], dtype=float)
    s = w.sum()
    if s <= 0:
        # default to uniform profile if all-zero
        w[:] = 1.0 / 24.0
    else:
        w /= s
    return w

def expand_24h_weights_to_index(weights24: np.ndarray, index: pd.DatetimeIndex) -> pd.Series:
    """
    Expand 24-hour weights to a full-resolution day-vector aligned to `index`.
    Works for sub-hourly timesteps. Returns a series of non-negative weights whose
    daily integral (sum across a calendar day) equals 1.0 for that day.
    """
    # Build an hourly template aligned per day, then forward-fill into sub-hourly bins.
    df = pd.DataFrame(index=index)
    df['hour'] = df.index.hour
    hourly_weights = pd.Series({h: weights24[h] for h in range(24)}, name='w')
    df = df.join(hourly_weights, on='hour')
    # For sub-hourly steps, w is constant within the hour; we’ll renormalize PER DAY:
    day_sum = df['w'].groupby(df.index.date).transform('sum')
    day_sum = day_sum.replace(0, 1.0)
    df['w_norm'] = df['w'] / day_sum
    return df['w_norm']

def monthly_factor_series(index: pd.DatetimeIndex, monthly_factors: list[float] | None) -> pd.Series:
    """Return multiplicative seasonal factor (length 12) aligned to index month, defaults to 1.0."""
    if not monthly_factors or len(monthly_factors) != 12:
        return pd.Series(1.0, index=index)
    m = np.array([1.0 if (x is None or np.isnan(x)) else float(x) for x in monthly_factors], dtype=float)
    m[m <= 0] = 1.0
    month_idx = pd.Index(index.month - 1)  # 0..11
    return pd.Series(m[month_idx.values], index=index)

def dhw_daily_energy_kwh(occupants: float, lppd: float, tap_C: float, cold_C: float) -> float:
    """
    Convert liters/person/day + occupants + deltaT into *thermal* energy (kWh/day).
    Q = m * cp * dT. 1 liter ≈ 1 kg. cp ≈ 4.186 kJ/kgK.
    """
    deltaT = max(0.0, tap_C - cold_C)
    liters_day = max(0.0, occupants) * max(0.0, lppd)
    kJ = liters_day * 4.186 * deltaT
    kWh = kJ / 3600.0
    return kWh