"""
Service layer for fetching and validating weather data for the
Streamlit dashboard. Provides a single entry point (`load_weather`)
that abstracts away forecast, historical, and TMY sources.

- Forecast (short-term) → Open-Meteo via `fetch_weather`
- Historical (full year) → archive via `fetch_weather_archive`
- TMY (typical year) → PVGIS dataset via `fetch_tmy_pvgis`

All returned datasets are validated against the expected schema
using `validate_weather_schema`.
"""

from __future__ import annotations
import streamlit as st
from data.weather import fetch_weather, fetch_weather_archive, fetch_tmy_pvgis
from core.validation import validate_weather_schema


def load_weather(
    mode: str,
    lat: float,
    lon: float,
    tz: str,
    hours: int | None = None,
    year: int | None = None,
):
    """
    Fetch and validate weather data for the dashboard, based on the chosen mode.

    Parameters
    ----------
    mode : str
        One of:
        - "Forecast (next days)" → fetch a short-term forecast (uses `hours` horizon).
        - "Historical (full year)" → fetch archived weather for a full calendar year (uses `year`).
        - "TMY (full year)" → fetch typical meteorological year (TMY) dataset.
    lat : float
        Latitude of the location in decimal degrees.
    lon : float
        Longitude of the location in decimal degrees.
    tz : str
        IANA timezone string (e.g., "Europe/Amsterdam").
    hours : int | None, optional
        Forecast horizon in hours (only used if `mode="Forecast (next days)"`).
        Defaults to 48 if not provided.
    year : int | None, optional
        Year to fetch historical data for (only used if `mode="Historical (full year)"`).
        Defaults to 2024 if not provided.

    Returns
    -------
    pandas.DataFrame
        Weather data with a DatetimeIndex and at least the required columns
        for simulation (e.g. "T_out", "G_solar").
        Raises an exception if validation fails.

    Notes
    -----
    - Uses `validate_weather_schema` to ensure the returned data has the expected structure.
    - Displays a Streamlit success message with the number of hours loaded.
    """
    if mode == "Forecast (next days)":
        wx = fetch_weather(lat, lon, tz, int(hours or 48))
    elif mode == "Historical (full year)":
        y = int(year or 2024)
        start_date = f"{y:04d}-01-01"
        end_date = f"{y:04d}-12-31"
        wx = fetch_weather_archive(lat, lon, tz, start_date, end_date)
    else:
        wx = fetch_tmy_pvgis(lat, lon, tz)

    validate_weather_schema(wx)
    st.success(f"Weather data loaded: {len(wx)} hours")
    return wx
