"""
data/weather.py

Fetch weather forecast data from the Open-Meteo API and return it as a Pandas DataFrame.

This module provides a simple wrapper for pulling hourly data (temperature, humidity, wind,
and solar radiation) for a given latitude/longitude and timezone. The result is ready to
use in building/energy simulation models.
"""

import pandas as pd
import requests
from pvlib.iotools import get_pvgis_tmy

def fetch_weather(lat: float, lon: float, tz: str, hours: int = 48) -> pd.DataFrame:
    """
    Fetch hourly forecast weather data from Open-Meteo API.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees.
    lon : float
        Longitude in decimal degrees.
    tz : str
        IANA timezone string, e.g. "Europe/Amsterdam".
    hours : int, optional
        Number of forecast hours to retrieve (default: 48, max ~72).

    Returns
    -------
    pd.DataFrame
        Weather dataframe indexed by time with columns:
        - T_out [°C]
        - RH_out [%]
        - wind [m/s]
        - G_solar [W/m²]
    """
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation"
        f"&forecast_days=3&timezone={tz}"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    hourly = data["hourly"]

    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "T_out": hourly["temperature_2m"],
        "RH_out": hourly["relative_humidity_2m"],
        "wind": hourly["wind_speed_10m"],
        "G_solar": hourly["shortwave_radiation"],
    })
    df = df.set_index("time").sort_index().iloc[:hours]
    return df

def fetch_weather_archive(
    lat: float,
    lon: float,
    tz: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch hourly historical weather data from Open-Meteo Archive API.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees.
    lon : float
        Longitude in decimal degrees.
    tz : str
        IANA timezone string, e.g. "Europe/Amsterdam".
    start_date : str
        Start date, format "YYYY-MM-DD".
    end_date : str
        End date, format "YYYY-MM-DD".

    Returns
    -------
    pd.DataFrame
        Weather dataframe with columns:
        - T_out [°C]
        - RH_out [%]
        - wind [m/s]
        - G_solar [W/m²]
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation"
        f"&start_date={start_date}&end_date={end_date}"
        f"&timezone={tz}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    hourly = r.json()["hourly"]

    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "T_out": hourly["temperature_2m"],
        "RH_out": hourly["relative_humidity_2m"],
        "wind": hourly["wind_speed_10m"],
        "G_solar": hourly["shortwave_radiation"],
    }).set_index("time").sort_index()
    return df

def fetch_tmy_pvgis(lat: float, lon: float, tz: str) -> pd.DataFrame:
    """
    Fetch a Typical Meteorological Year (TMY) from PVGIS and map to our schema.

    Returns a DataFrame indexed by local time with:
      - T_out [°C]
      - RH_out [%]
      - wind [m/s]
      - G_solar [W/m²]  (global horizontal irradiance)
    """
    # PVGIS returns a typical year time series; pvlib maps variables to standard names
    df, meta = get_pvgis_tmy(lat, lon, map_variables=True, outputformat="json")

    # Ensure timezone-aware index in the user's tz
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    df = df.tz_convert(tz)

    # Map pvlib variable names to your project schema
    out = pd.DataFrame(index=df.index)
    # pvlib columns are typically: 'temp_air', 'relative_humidity', 'wind_speed', 'ghi'
    out["T_out"] = df.get("temp_air")
    out["RH_out"] = df.get("relative_humidity")
    out["wind"] = df.get("wind_speed")
    out["G_solar"] = df.get("ghi")
    out.index.name = "time"
    # Basic sanity
    out = out.sort_index()
    return out
