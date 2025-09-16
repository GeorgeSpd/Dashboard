"""
Utility helpers shared across the dashboard modules.

Functions
---------
- _f : safe float conversion with fallback default.
- make_unique : ensures unique names in a list by appending suffixes.
"""

from __future__ import annotations


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
