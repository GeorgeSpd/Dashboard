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
Streamlit dashboard for simulating a simple RC building thermal model
with scenario comparison.

- Set Time & Location across scenarios (shared)
- Each scenario has its own expander with Building/HVAC/Efficiency/Price/CO₂ settings
- If only 1 scenario: show detailed plots (temp/solar/flows/efficiency/summary)
- If 2+ scenarios: hide detailed plots and show cumulative € comparison + KPIs
"""

from datetime import timedelta
import pandas as pd
import streamlit as st

from ui.scenario import scenario_controls
from utils.centralization import adapt_to_per_house
from utils.common import make_unique
from services.weather_service import load_weather
from services.sim_runner import run_scenario
from views.single import render_single
from views.multi import render_multi

st.set_page_config(page_title="Energy Simulation Dashboard", page_icon="🔥", layout="wide")

st.sidebar.title("⚙️ Scenario Manager")

# Sidebar: Time & Location
st.sidebar.header("Time & Location")
with st.sidebar.expander("Time", expanded=True):
    mode = st.radio("Data of Interest", ["None", "Historical (full year)", "TMY (full year)", "Forecast (next days)"])

    tz = st.text_input("Timezone (IANA)", value="Europe/Amsterdam")

    if mode == "Forecast (next days)":
        hours = st.slider("Forecast horizon (hours)", 12, 72, 48, step=6)
        year = None
    elif mode == "Historical (full year)":
        hours = None
        year = st.number_input("Year", min_value=2000, max_value=2100, value=2024, step=1)
    elif mode == "TMY (full year)":
        hours = None
        year = None
    else:
        hours = None
        year = None

if mode == "None":
    st.warning("Select a data source (Historical, TMY, or Forecast) to start the simulation.")
    st.stop()

with st.sidebar.expander("Location", expanded=True):
    lat = st.number_input("Latitude", value=52.0, format="%.1f", step=0.1)
    lon = st.number_input("Longitude", value=4.9, format="%.1f", step=0.1)

# Title + badge
st.title("🔥 Energy Simulation Dashboard")
badge_color = {"Forecast (next days)": "#3778FF", "Historical (full year)": "#8A2BE2", "TMY (full year)": "#10B981",}[mode]
st.markdown(f"""
<div style=\"display:flex;gap:10px;align-items:center;\">
  <span style=\"background:{badge_color};color:white;padding:4px 10px;border-radius:999px;font-weight:600;\">{mode}</span>
  <span style=\"opacity:0.8\">lat: {lat:.2f} lon: {lon:.2f} timezone: {tz}</span>
</div>
""", unsafe_allow_html=True)

# Weather fetch
st.subheader("Weather data")
try:
    wx = load_weather(mode, lat, lon, tz, hours=hours, year=year)
except Exception as e:
    st.error(f"Weather fetch/validation failed: {e}")
    st.stop()

# Sidebar: Scenario Manager
st.sidebar.header("Scenario Definition")
num_scen = st.sidebar.number_input("Number of scenarios", min_value=1, max_value=10, value=1, step=1)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:num_scen]

# Create scenario configs
raw_scenarios = [scenario_controls(lbl) for lbl in labels]
scenarios = [adapt_to_per_house(s) for s in raw_scenarios]
disp_names = make_unique([cfg["name"] for cfg in scenarios])
for cfg, nm in zip(scenarios, disp_names):
    cfg["disp_name"] = nm

# Shared zoom window
st.sidebar.header("Focus Window")
t_min = pd.to_datetime(wx.index.min()).to_pydatetime()
t_max = pd.to_datetime(wx.index.max()).to_pydatetime()
win_start, win_end = st.sidebar.slider("Select window of interest", min_value=t_min, max_value=t_max, value=(t_min, t_max), step=timedelta(hours=1))

# Run all scenarios
sims = {cfg["label"]: run_scenario(wx, cfg) for cfg in scenarios}
views = {lbl: df.loc[win_start:win_end] for lbl, df in sims.items()}

# Single vs Multi
if num_scen == 1:
    label = labels[0]
    render_single(label, scenarios[0], sims[label], views[label])
else:
    render_multi(scenarios, sims, views)
