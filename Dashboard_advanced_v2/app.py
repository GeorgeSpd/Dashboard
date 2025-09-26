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
import pytz

from ui.scenario import scenario_controls
from utils.centralization import adapt_to_per_house
from utils.common import make_unique
from services.weather_service import load_weather
from services.sim_runner import run_scenario
from views.single import render_single
from views.multi import render_multi

# Page title
st.set_page_config(page_title="Energy Simulation Dashboard", page_icon="🔥", layout="wide")

# Screen title
st.title("🔥 Energy Simulation Dashboard")

# Info text
st.subheader("General Info")
st.info(
    """
    **What this tool does:**  
    - 📅 Select time period & location  
    - 🏠 Configure one or more scenarios  
    - 🏗 Define building setup, demand profiles, and energy solutions  
    - 📊 View detailed plots for a single scenario or comparisons across many  
    
    Use it to quickly explore how design choices and external conditions impact building performance.
    """
)

# Sidebar title
st.sidebar.markdown(
    "<h1 style='font-size:40px; font-weight:800;'>⚙️ Scenario Manager</h1>",
    unsafe_allow_html=True
)

# Sidebar: Time
st.sidebar.header("Time & Location")
with st.sidebar.expander("Time", expanded=True):
    mode = st.radio("Data of Interest", ["None", "Historical (full year)", "TMY (full year)", "Forecast (next days)"],
                    help="Choose the type of weather dataset to use:\n- Historical: Chosen past year\n- TMY: Typical meteorological year\n- Forecast: Upcoming days",
                    )

    timezones = pytz.all_timezones
    tz = st.selectbox("Timezone (IANA)", options=timezones, index=timezones.index("Europe/Amsterdam"))

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

# Sidebar: Location
with st.sidebar.expander("Location", expanded=True):
    lat = st.number_input("Latitude", value=52.0, format="%.1f", step=0.1)
    lon = st.number_input("Longitude", value=4.9, format="%.1f", step=0.1)

# if no mode is chosen ask for it and stop
if mode == "None":
    st.warning("TO DO: Select data of interest (Time Period & Location) to import weather data.")
    st.stop()

# Weather fetch
st.subheader("Weather data")
try:
    wx = load_weather(mode, lat, lon, tz, hours=hours, year=year)
except Exception as e:
    st.error(f"Weather fetch/validation failed: {e}")
    st.stop()

# Sidebar: Scenario Manager
st.sidebar.header("Scenario Definition")
num_scen = st.sidebar.number_input("Number of scenarios", min_value=0, max_value=10, value=0, step=1,
                                   help="Define number of scenarios:\n- (1): Scenario-specific results\n- (≥2): Scenario comparison")
if num_scen == 0:
    st.warning("TO DO: Create and define a scenario.")
    st.stop()

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:num_scen]

# Create scenario configs
# collect inputs from user interface
raw_scenarios = [scenario_controls(lbl) for lbl in labels]
# standardize inputs for centralized and decentralized scenarios
scenarios = [adapt_to_per_house(s) for s in raw_scenarios]
# make scenario names unique
disp_names = make_unique([cfg["name"] for cfg in scenarios])
# match names to scenarios
for cfg, nm in zip(scenarios, disp_names):
    cfg["disp_name"] = nm

# Sidebar: Shared zoom window
st.sidebar.header("Focus Window")
t_min = pd.to_datetime(wx.index.min()).to_pydatetime()
t_max = pd.to_datetime(wx.index.max()).to_pydatetime()
win_start, win_end = st.sidebar.slider("Select window of interest", min_value=t_min, max_value=t_max, value=(t_min, t_max), step=timedelta(hours=1))

# Run all scenarios
# run whole scenario on background
sims = {cfg["label"]: run_scenario(wx, cfg) for cfg in scenarios}
# view only what is selected by focus window
views = {lbl: df.loc[win_start:win_end] for lbl, df in sims.items()}

# Single vs Multi
if num_scen == 1:
    label = labels[0]
    # if only one scenario is selected show scenario specific results
    render_single(label, scenarios[0], sims[label], views[label])
else:
    # if more than one scenarios are selected show comparison results
    render_multi(scenarios, sims, views)
