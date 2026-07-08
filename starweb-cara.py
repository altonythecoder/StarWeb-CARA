# StarWeb-CARA: Conjunction Assessment and Collision Risk Analysis
# Starlink & OneWeb Megaconstellations — Thesis Project
# Altay ÇAVUŞ — Space Sciences and Technologies, ÇOMÜ (2026)

import math
from datetime import datetime, timezone
from io import BytesIO
from itertools import combinations
from functools import lru_cache
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import spacetrack.operators as op
import streamlit as st
from PIL import Image
from scipy.integrate import dblquad
from scipy.stats import norm
from skyfield.api import EarthSatellite, load, wgs84
from spacetrack import SpaceTrackClient



# ================================================================================
#  TIMESCALE INIT & SIMULATION QUEUE HELPER
# ================================================================================
# Skyfield zaman ölçeği — orijinal tek-dosya sürümde dosyanın en başında
# (importlardan hemen sonra) global olarak bir kez oluşturuluyordu.
# Burada module-level'da tutuluyor, tüm modüller buradan import ediyor.

try:
    ts = load.timescale()
except Exception as e:
    # Streamlit not yet initialized, just log the error
    print(f"Skyfield timescale initialization failed: {e}")
    ts = None

def queue_simulation_pair(sat_a, sat_b, center_tt=None):
    """
    Store the selected satellite pair (and an optional TCA centre time)
    in session state so that the Live‑Simulation tab can pick it up
    and start the animation.

    Parameters
    ----------
    sat_a, sat_b : EarthSatellite
        The two satellites whose encounter you want to visualise.
    center_tt : float, optional
        Skyfield TT timestamp that should be used as the centre of the
        simulation window (e.g. the TCA time).  If None, the simulation
        will start at the current time.
    """
    # Guard against accidental None values (optional but helpful)
    if sat_a is None or sat_b is None:
        print("Cannot queue a simulation – one of the satellites is None.")
        return

    # Save the objects and the optional centre time
    try:
        st.session_state["sim_sat_a"] = sat_a
        st.session_state["sim_sat_b"] = sat_b
        st.session_state["sim_center_tt"] = center_tt
        st.session_state["run_sim"] = True
    except Exception as e:
        print(f"Failed to queue simulation: {e}")


# ================================================================================

# ================================================================================
#  TIME AND ORBITAL HELPERS
# ================================================================================
# ================================================================================
#  CONSTANTS AND CONFIGURATION
# ================================================================================
MANUAL_SAT_DEFAULT_MASS_KG = 250
MASS_WIDGET_MAX_KG = 500_000.0
EARTH_RADIUS_KM = 6371.0
MU_EARTH_KM3_S2 = 398600.4418
ANALYSIS_STEP_MIN = 5
CONJUNCTION_DISTANCE_THRESHOLD_KM = 500.0
APSIS_FILTER_THRESHOLD_KM = 50.0  # Fixed threshold for apsis filter

GROUP_CONFIG = {
    "STARLINK": {
        "label": "STARLINK",
        "spacetrack_mode": "name",
        "spacetrack_value": "STARLINK",
        "celestrak_group": "starlink",
        "default_mass_kg": 250,
    },
    "ONEWEB": {
        "label": "ONEWEB",
        "spacetrack_mode": "name",
        "spacetrack_value": "ONEWEB",
        "celestrak_group": "oneweb",
        "default_mass_kg": 150,
    },
    "ISS": {
        "label": "ISS",
        "spacetrack_mode": "norad",
        "spacetrack_value": 25544,
        "celestrak_group": "stations",
        "default_mass_kg": 419725,
    },
    "KUIPER": {
        "label": "KUIPER",
        "spacetrack_mode": "name",
        "spacetrack_value": "KUIPER",
        "celestrak_group": "kuiper",
        "default_mass_kg": 630,
    },
    "IRIDIUM-NEXT": {
        "label": "IRIDIUM NEXT",
        "spacetrack_mode": "name",
        "spacetrack_value": "IRIDIUM",
        "celestrak_group": "iridium-NEXT",
        "default_mass_kg": 860,
    },
    "PLANET": {
        "label": "PLANET",
        "spacetrack_mode": "name",
        "spacetrack_value": "PLANET",
        "celestrak_group": "planet",
        "default_mass_kg": 5,
    },
}


def get_group_default_mass(group_key: str) -> int:
    return int(GROUP_CONFIG.get(group_key, {}).get("default_mass_kg", 250))


# ================================================================================
#  TIME AND ORBITAL HELPERS
# ================================================================================
def build_time_grid(start_tt: float, window_hrs: int, step_min: int = ANALYSIS_STEP_MIN):
    """Build time grid for analysis using linspace for clarity."""
    if ts is None:
        print("Error: Timescale not initialized")
        return None, None
    n_steps = max(1, int(window_hrs * 60 // step_min) + 1)
    offsets = np.linspace(0, (n_steps-1)*step_min, n_steps) / 1440.0
    return ts.tt_jd(start_tt + offsets), offsets


def propagated_positions(sat, times):
    """
    Enhanced position propagation with better error handling and performance monitoring.
    Note: Not cached because EarthSatellite objects are not serializable.
    The expensive computations (conjunction analysis) are cached instead.
    """
    try:
        return sat.at(times).position.km
    except Exception as e:
        # Log error without using st.warning to avoid session dependency
        print(f"Position propagation error for {sat.name}: {str(e)[:50]}")
        return None


def _set_mass_widget_values(mass_a: float, mass_b: float):
    try:
        mass_a = float(max(1.0, min(mass_a, MASS_WIDGET_MAX_KG)))
        mass_b = float(max(1.0, min(mass_b, MASS_WIDGET_MAX_KG)))
        # Only set session_state if streamlit is initialized
        if hasattr(st, 'session_state'):
            st.session_state["mass_a_kg"] = mass_a
            st.session_state["mass_b_kg"] = mass_b
            # Pop widget-bound keys so they reinitialise from mass_*_kg on the
            # next render.  Direct assignment to a widget-bound key outside a
            # callback raises StreamlitAPIException, so we delete them instead.
            st.session_state.pop("mass_a_input", None)
            st.session_state.pop("mass_b_input", None)
    except Exception as e:
        print(f"Error setting mass widget values: {e}")


def sync_mass_a_from_input():
    try:
        if not hasattr(st, 'session_state'):
            return  # Streamlit not initialized yet
        value = float(
            max(1.0, min(st.session_state.get("mass_a_input", 1.0), MASS_WIDGET_MAX_KG))
        )
        st.session_state["mass_a_kg"] = value
        st.session_state["mass_a_input"] = value
    except Exception as e:
        print(f"Error syncing mass A: {e}")
        # Set fallback values
        if hasattr(st, 'session_state'):
            st.session_state["mass_a_kg"] = 250.0
            st.session_state["mass_a_input"] = 250.0


def sync_mass_b_from_input():
    try:
        if not hasattr(st, 'session_state'):
            return  # Streamlit not initialized yet
        value = float(
            max(1.0, min(st.session_state.get("mass_b_input", 1.0), MASS_WIDGET_MAX_KG))
        )
        st.session_state["mass_b_kg"] = value
        st.session_state["mass_b_input"] = value
    except Exception as e:
        print(f"Error syncing mass B: {e}")
        # Set fallback values
        if hasattr(st, 'session_state'):
            st.session_state["mass_b_kg"] = 250.0
            st.session_state["mass_b_input"] = 250.0


def sync_mass_defaults(group_key: str):
    try:
        if not hasattr(st, 'session_state'):
            return  # Streamlit not initialized yet
        manual_present = "my_sat" in st.session_state
        fleet_mass = get_group_default_mass(group_key)
        group_changed = st.session_state.get("_mass_group_key") != group_key
        manual_changed = st.session_state.get("_manual_mass_mode") != manual_present

        if group_changed or manual_changed:
            if manual_present:
                _set_mass_widget_values(MANUAL_SAT_DEFAULT_MASS_KG, fleet_mass)
            else:
                _set_mass_widget_values(fleet_mass, fleet_mass)

            st.session_state["_mass_group_key"] = group_key
            st.session_state["_manual_mass_mode"] = manual_present
    except Exception as e:
        print(f"Error syncing mass defaults: {e}")
        # Set fallback values
        if hasattr(st, 'session_state'):
            st.session_state["_mass_group_key"] = group_key
            st.session_state["_manual_mass_mode"] = False


# ================================================================================
#  CSS — ENHANCED MISSION CONTROL DARK THEME WITH THEME SELECTOR
# ================================================================================
def get_theme_css(theme="dark"):
    """Generate CSS based on selected theme"""
    if theme == "light":
        # Light/Professional Theme (Black & White with blue accents)
        return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --bg:#ffffff; --bg2:#f8f9fa; --bg3:#e9ecef; --border:#dee2e6;
  --accent:#0066cc; --accent2:#004499; --warn:#ffc107; --crit:#dc3545;
  --text:#212529; --dim:#6c757d;
  --mono:'Space Mono',monospace; --sans:'Barlow Condensed',sans-serif; --ui:'Inter',sans-serif;
  --gradient-primary: linear-gradient(135deg, #0066cc 0%, #004499 100%);
  --gradient-dark: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
  --shadow-glow: 0 0 20px rgba(0, 102, 204, 0.1);
  --shadow-card: 0 4px 20px rgba(0, 0, 0, 0.08);
}

/* Base Styles */
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
  background:var(--bg) !important;
  color:var(--text) !important;
  font-family:var(--ui) !important;
  background-image: var(--gradient-dark);
}

[data-testid="stSidebar"]{
  background:var(--bg2) !important;
  border-right:1px solid var(--border) !important;
  box-shadow: var(--shadow-card);
}

[data-testid="stSidebar"] *{
  color:var(--text) !important;
  font-family:var(--ui) !important;
}

/* Typography */
h1{
  font-family:var(--sans) !important;
  font-weight:900 !important;
  font-size:2.2rem !important;
  letter-spacing:.06em !important;
  color:#000 !important;
  text-transform:uppercase !important;
  line-height:1.2 !important;
  text-shadow: none;
}

h2,h3{
  font-family:var(--sans) !important;
  color:var(--accent) !important;
  font-weight:700 !important;
  letter-spacing:.08em !important;
  text-transform:uppercase !important;
  border-bottom:1px solid var(--border) !important;
  padding-bottom:.4em !important;
  margin-bottom:1em !important;
}

/* Metric Cards */
[data-testid="metric-container"]{
  background:var(--bg) !important;
  border:1px solid var(--border) !important;
  border-left:4px solid var(--accent) !important;
  padding:16px 20px !important;
  border-radius:8px !important;
  box-shadow: var(--shadow-card);
  transition: all 0.3s ease !important;
}

[data-testid="metric-container"]:hover{
  transform: translateY(-2px);
  box-shadow: var(--shadow-glow);
}

[data-testid="metric-container"] label{
  font-family:var(--ui) !important;
  font-size:.7rem !important;
  letter-spacing:.12em !important;
  color:var(--dim) !important;
  text-transform:uppercase !important;
  font-weight:600 !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"]{
  font-family:var(--mono) !important;
  color:var(--accent) !important;
  font-size:1.8rem !important;
  font-weight:700 !important;
}

/* Buttons */
.stButton button{
  background:linear-gradient(135deg, rgba(0, 102, 204, 0.1) 0%, rgba(0, 68, 153, 0.1) 100%) !important;
  border:1px solid var(--accent) !important;
  color:var(--accent) !important;
  font-family:var(--ui) !important;
  font-weight:600 !important;
  border-radius:6px !important;
  padding:12px 24px !important;
  transition: all 0.3s ease !important;
}

.stButton button:hover{
  background:var(--accent) !important;
  color:#fff !important;
  box-shadow: var(--shadow-glow);
}

/* Input Fields */
.stTextInput input, .stNumberInput input, .stSelectbox select{
  background:var(--bg) !important;
  border:1px solid var(--border) !important;
  color:var(--text) !important;
  border-radius:6px !important;
}

.stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus{
  border-color:var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1) !important;
}

/* Sliders */
.stSlider [role="slider"]{
  background:var(--accent) !important;
}

/* Info Panels */
.info-panel{
  background:var(--bg2) !important;
  border:1px solid var(--border) !important;
  border-left:4px solid var(--accent) !important;
  padding:16px 20px !important;
  border-radius:8px !important;
  margin:16px 0 !important;
  color:var(--text) !important;
}

.warn-panel{
  background:var(--bg2) !important;
  border:1px solid var(--border) !important;
  border-left:4px solid var(--warn) !important;
  padding:16px 20px !important;
  border-radius:8px !important;
  margin:16px 0 !important;
  color:var(--text) !important;
}

.crit-panel{
  background:var(--bg2) !important;
  border:1px solid var(--border) !important;
  border-left:4px solid var(--crit) !important;
  padding:16px 20px !important;
  border-radius:8px !important;
  margin:16px 0 !important;
  color:var(--text) !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"]{
  background:var(--bg2) !important;
  border:1px solid var(--border) !important;
  border-radius:8px !important;
  padding:8px !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"]{
  background:var(--accent) !important;
  color:#fff !important;
  border-radius:6px !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="false"]{
  color:var(--dim) !important;
}

/* Dataframe */
.stDataFrame{
  background:var(--bg) !important;
  border:1px solid var(--border) !important;
  border-radius:8px !important;
}

.stDataFrame table{
  color:var(--text) !important;
}

.stDataFrame th{
  background:var(--bg2) !important;
  color:var(--accent) !important;
  font-weight:600 !important;
}

.stDataFrame tr:hover{
  background:var(--bg2) !important;
}

/* Sidebar Elements */
[data-testid="stSidebar"] .stSlider label{
  color:var(--text) !important;
  font-weight:600 !important;
}

[data-testid="stSidebar"] .stSelectbox label{
  color:var(--text) !important;
  font-weight:600 !important;
}

/* Expander */
[data-testid="stExpander"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

[data-testid="stExpander"] > div > div > svg {
  color: var(--accent) !important;
}

/* Enhanced text area */
[data-testid="stTextArea"] > div > div > textarea {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
}

/* Hide sidebar collapse button and header */
[data-testid="stSidebarCollapseButton"]{display:none !important;}
header[data-testid="stHeader"]{display:none !important;}
</style>
"""
    else:
        # Original Dark Mission Control Theme
        return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --bg:#05070a; --bg2:#0a0f18; --bg3:#121824; --border:#1e2d42;
  --accent:#00d4ff; --accent2:#00ffa8; --warn:#ffb800; --crit:#ff3d5c;
  --text:#c4d4e8; --dim:#5a7a94;
  --mono:'Space Mono',monospace; --sans:'Barlow Condensed',sans-serif; --ui:'Inter',sans-serif;
  --gradient-primary: linear-gradient(135deg, #00d4ff 0%, #00ffa8 100%);
  --gradient-dark: linear-gradient(180deg, #0a0f18 0%, #05070a 100%);
  --shadow-glow: 0 0 20px rgba(0, 212, 255, 0.15);
  --shadow-card: 0 4px 20px rgba(0, 0, 0, 0.3);
}

/* Base Styles */
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
  background:var(--bg) !important;
  color:var(--text) !important;
  font-family:var(--ui) !important;
  background-image: var(--gradient-dark);
}

[data-testid="stSidebar"]{
  background:var(--bg2) !important;
  border-right:1px solid var(--border) !important;
  box-shadow: var(--shadow-card);
}

[data-testid="stSidebar"] *{
  color:var(--text) !important;
  font-family:var(--ui) !important;
}

/* Typography */
h1{
  font-family:var(--sans) !important;
  font-weight:900 !important;
  font-size:2.2rem !important;
  letter-spacing:.06em !important;
  color:#fff !important;
  text-transform:uppercase !important;
  line-height:1.2 !important;
  text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
}

h2,h3{
  font-family:var(--sans) !important;
  color:var(--accent) !important;
  font-weight:700 !important;
  letter-spacing:.08em !important;
  text-transform:uppercase !important;
  border-bottom:1px solid var(--border) !important;
  padding-bottom:.4em !important;
  margin-bottom:1em !important;
}

/* Metric Cards */
[data-testid="metric-container"]{
  background:var(--bg3) !important;
  border:1px solid var(--border) !important;
  border-left:4px solid var(--accent) !important;
  padding:16px 20px !important;
  border-radius:8px !important;
  box-shadow: var(--shadow-card);
  transition: all 0.3s ease !important;
}

[data-testid="metric-container"]:hover{
  transform: translateY(-2px);
  box-shadow: var(--shadow-glow);
}

[data-testid="metric-container"] label{
  font-family:var(--ui) !important;
  font-size:.7rem !important;
  letter-spacing:.12em !important;
  color:var(--dim) !important;
  text-transform:uppercase !important;
  font-weight:600 !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"]{
  font-family:var(--mono) !important;
  color:var(--accent) !important;
  font-size:1.8rem !important;
  font-weight:700 !important;
}

/* Buttons */
.stButton button{
  background:linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 255, 168, 0.1) 100%) !important;
  border:1px solid var(--accent) !important;
  color:var(--accent) !important;
  font-family:var(--mono) !important;
  font-size:.75rem !important;
  letter-spacing:.1em !important;
  text-transform:uppercase !important;
  padding:10px 24px !important;
  border-radius:6px !important;
  transition:all 0.3s ease !important;
  font-weight:600 !important;
  box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
}

.stButton button:hover{
  background:var(--gradient-primary) !important;
  color:var(--bg) !important;
  transform: translateY(-2px);
  box-shadow: var(--shadow-glow);
}

.stButton button:active{
  transform: translateY(0);
}

/* Tabs */
[data-baseweb="tab-list"]{
  background:var(--bg2) !important;
  border-bottom:2px solid var(--border) !important;
  gap:0 !important;
  padding: 0 8px !important;
}

[data-baseweb="tab"]{
  font-family:var(--sans) !important;
  font-weight:600 !important;
  font-size:.85rem !important;
  letter-spacing:.1em !important;
  text-transform:uppercase !important;
  color:var(--dim) !important;
  padding:14px 24px !important;
  border-radius:8px 8px 0 0 !important;
  transition: all 0.3s ease !important;
}

[aria-selected="true"][data-baseweb="tab"]{
  color:var(--accent) !important;
  background:linear-gradient(180deg, rgba(0, 212, 255, 0.1) 0%, transparent 100%) !important;
  border-bottom:2px solid var(--accent) !important;
}

/* Input Fields */
[data-testid="stTextInput"] input{
  background:var(--bg3) !important;
  border-color:var(--border) !important;
  color:var(--text) !important;
  font-family:var(--mono) !important;
  font-size:.85rem !important;
  border-radius:6px !important;
  padding:10px 14px !important;
  transition: all 0.3s ease !important;
}

[data-testid="stTextInput"] input:focus{
  border-color:var(--accent) !important;
  box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
}

[data-testid="stSelectbox"]>div>div{
  background:var(--bg3) !important;
  border-color:var(--border) !important;
  border-radius:6px !important;
}

/* Alert Messages */
[data-testid="stInfo"]{
  background:rgba(0,212,255,.06) !important;
  border:1px solid rgba(0,212,255,.25) !important;
  border-radius:8px !important;
  padding:16px 20px !important;
}

[data-testid="stSuccess"]{
  background:rgba(0,255,168,.06) !important;
  border:1px solid rgba(0,255,168,.25) !important;
  border-radius:8px !important;
  padding:16px 20px !important;
}

[data-testid="stError"]{
  background:rgba(255,61,92,.08) !important;
  border:1px solid rgba(255,61,92,.35) !important;
  border-radius:8px !important;
  padding:16px 20px !important;
}

[data-testid="stWarning"]{
  background:rgba(255,184,0,.06) !important;
  border:1px solid rgba(255,184,0,.25) !important;
  border-radius:8px !important;
  padding:16px 20px !important;
}

/* DataFrames */
[data-testid="stDataFrame"]{
  border:1px solid var(--border) !important;
  border-radius:8px !important;
  overflow:hidden !important;
  box-shadow: var(--shadow-card);
}

/* Toolbar */
[data-testid="stElementToolbarButton"]{display:none !important;}
[data-testid="stElementToolbar"]{display:none !important;}

/* Download Button */
[data-testid="stDownloadButton"] button{
  width:auto !important;
  background:var(--bg3) !important;
  border:1px solid var(--accent) !important;
  color:var(--accent) !important;
  font-family:var(--mono) !important;
  font-size:.7rem !important;
  letter-spacing:.08em !important;
  text-transform:uppercase !important;
  padding:8px 18px !important;
  border-radius:6px !important;
  transition: all 0.3s ease !important;
}

[data-testid="stDownloadButton"] button:hover{
  background:var(--gradient-primary) !important;
  color:var(--bg) !important;
  transform: translateY(-2px);
}

/* Scrollbar */
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--bg2);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--accent);}

/* Layout */
section[data-testid="stMain"]>div{background:var(--bg) !important;}
div[data-testid="stVerticalBlock"]{background:transparent !important;}

/* Markdown */
.stMarkdown,.stMarkdown p{color:var(--text) !important;line-height:1.7 !important;}

/* Slider */
[data-baseweb="slider"]{background:var(--bg3) !important;}
[data-baseweb="slider-handle"]{background:var(--accent) !important;}

/* Progress Bar */
[data-testid="stProgress"]{background:var(--bg3) !important;}
[data-testid="stProgress"] > div > div > div{background:var(--gradient-primary) !important;}

/* Custom Panels */
.info-panel{
  background:rgba(0,212,255,.04);
  border:1px solid rgba(0,212,255,.18);
  border-left:4px solid var(--accent);
  padding:16px 20px;
  margin:12px 0;
  border-radius:8px;
  font-size:.9rem;
  line-height:1.7;
  box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1);
}

.warn-panel{
  background:rgba(255,184,0,.04);
  border:1px solid rgba(255,184,0,.18);
  border-left:4px solid var(--warn);
  padding:16px 20px;
  margin:12px 0;
  border-radius:8px;
  font-size:.9rem;
  line-height:1.7;
  box-shadow: 0 4px 15px rgba(255, 184, 0, 0.1);
}

.crit-panel{
  background:rgba(255,61,92,.04);
  border:1px solid rgba(255,61,92,.18);
  border-left:4px solid var(--crit);
  padding:16px 20px;
  margin:12px 0;
  border-radius:8px;
  font-size:.9rem;
  line-height:1.7;
  box-shadow: 0 4px 15px rgba(255, 61, 92, 0.1);
}

/* Animation for loading */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

[data-testid="stSpinner"] > div {
  animation: pulse 1.5s ease-in-out infinite;
}

/* Enhanced selectbox */
[data-baseweb="select"] *{background:var(--bg3) !important;color:var(--text) !important;}
[data-baseweb="popover"]{background:var(--bg3) !important;border:1px solid var(--border) !important;}
[data-baseweb="menu"]{background:var(--bg3) !important;}

/* Sidebar top header (broken Material Icons icon) hide */
[data-testid="stSidebarHeader"]{display:none !important;}
[data-testid="stSidebarCollapseButton"]{display:none !important;}
header[data-testid="stHeader"]{display:none !important;}

/* Enhanced text area */
[data-testid="stTextArea"] > div > div > textarea {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
}

/* Expander */
[data-testid="stExpander"] {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

[data-testid="stExpander"] > div > div > svg {
  color: var(--accent) !important;
}

/* Hide sidebar collapse button and header */
[data-testid="stSidebarCollapseButton"]{display:none !important;}
header[data-testid="stHeader"]{display:none !important;}
</style>
"""


# ================================================================================
#  EARTH VIEW - ENHANCED WITH ERROR HANDLING
# ================================================================================
@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def load_earth_texture(resolution: int = 360, style: str = "night"):
    """
    Loads high-resolution NASA Earth textures and optimizes them for Plotly Surface rendering.
    style: "night" (Night Lights), "realistic" (Realistic Blue Marble), "futuristic" (Blue/Cyan Tonal)
    Enhanced with better error handling and performance optimization.
    """
    try:
        if style == "night":
            # NASA Black Marble (Night Lights) - Optimal for dark theme
            urls = [
                "https://eoimages.gsfc.nasa.gov/images/imagerecords/79000/79765/dnb_land_ocean_ice.2012.3600x1800.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/b/ba/The_earth_at_night.jpg",
            ]
        elif style == "realistic":
            # NASA Blue Marble Next Generation (High-Resolution Realistic)
            urls = [
                "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/a/ad/Blue_Marble_2002.png",
            ]
        else:
            # Optimized version of the futuristic theme
            urls = [
                "https://upload.wikimedia.org/wikipedia/commons/c/cd/Land_ocean_ice_2048.jpg/1024px-Land_ocean_ice_2048.jpg"
            ]

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()

                img = Image.open(BytesIO(resp.content)).convert("RGB")
                W, H = resolution * 2, resolution

                # LANCZOS filter minimizes pixelation during resampling
                img = img.resize((W, H), Image.LANCZOS)

                img_array = np.array(img, dtype=np.float32)

                if style == "futuristic":
                    # Futuristic blue/cyan color processing
                    img_array = img_array * 0.4
                    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.4, 0, 255)
                    img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.1, 0, 255)
                    mean_val = np.mean(img_array)
                    img_array = np.clip((img_array - mean_val) * 1.3 + mean_val, 0, 255)
                    img_array = np.clip(img_array + 15, 0, 255)
                elif style == "night":
                    # Slightly enhance city lights, render oceans as pure black
                    img_array = np.clip(img_array * 1.3, 0, 255)

                img_array = img_array.astype(np.uint8)
                img = Image.fromarray(img_array)

                # MEDIANCUT algorithm prevents color muddying under 256-color limit
                imgq = img.quantize(colors=256, method=Image.MEDIANCUT)
                pal = np.array(imgq.getpalette(), dtype=np.uint8).reshape(-1, 3)[:256]

                idx = np.flipud(np.array(imgq, dtype=float))
                surf_color = idx / 255.0
                colorscale = [
                    [i / 255.0, f"rgb({pal[i, 0]},{pal[i, 1]},{pal[i, 2]})"]
                    for i in range(256)
                ]

                lat = np.linspace(np.pi / 2, -np.pi / 2, H)
                lon = np.linspace(-np.pi, np.pi, W)
                lon_g, lat_g = np.meshgrid(lon, lat)

                R = EARTH_RADIUS_KM
                x = R * np.cos(lat_g) * np.cos(lon_g)
                y = R * np.cos(lat_g) * np.sin(lon_g)
                z = R * np.sin(lat_g)

                return x, y, z, surf_color, colorscale
            except Exception:
                continue

        return None
    except Exception as e:
        print(f"Earth texture load failed: {str(e)[:50]}")
        return None


# ================================================================================
#  DATA FETCHING & TLE PARSING
# ================================================================================
#  DATA FETCHING
# ================================================================================
def count_tle_objects(lines: list) -> int:
    if not lines:
        return 0
    is_3ln = not (lines[0].startswith("1 ") or lines[0].startswith("2 "))
    step = 3 if is_3ln else 2
    return len(lines) // step


def trim_tle_lines(lines: list, sat_limit: int) -> list:
    if not lines:
        return []
    is_3ln = not (lines[0].startswith("1 ") or lines[0].startswith("2 "))
    step = 3 if is_3ln else 2
    max_lines = max(1, sat_limit) * step
    return lines[:max_lines]


def fetch_spacetrack_tles(username: str, password: str, group_key: str, sat_limit: int):
    """
    Enhanced Space-Track TLE fetching with better error handling and user feedback.
    """
    try:
        config = GROUP_CONFIG.get(group_key)
        if not config:
            return None, f"Unknown group: {group_key}"

        client = SpaceTrackClient(identity=username, password=password)

        # Add timeout handling
        if config["spacetrack_mode"] == "norad":
            raw = client.gp(
                norad_cat_id=config["spacetrack_value"],
                format="tle",
                orderby="epoch desc",
                limit=1,  # For NORAD (specific satellite), we only need the latest TLE
            )
        else:
            raw = client.gp(
                object_name=op.like(f"{config['spacetrack_value']}%"),
                format="tle",
                orderby="epoch desc",
                limit=max(1, sat_limit),
            )

        if not raw or not raw.strip():
            return None, f"Space-Track üzerinde '{group_key}' için veri bulunamadı."

        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            return None, "Space-Track geçerli TLE döndürmedi."

        lines = trim_tle_lines(lines, sat_limit)
        return lines, "Veri Space-Track üzerinden alındı."
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower():
            return None, "Space-Track kimlik doğrulama hatası. Lütfen kullanıcı adı ve şifrenizi kontrol edin."
        elif "timeout" in error_msg.lower():
            return None, "Space-Track bağlantı zaman aşımı. Lütfen daha sonra tekrar deneyin."
        elif "rate limit" in error_msg.lower():
            return None, "Space-Track hız sınırı aşıldı. Lütfen kilka dakika bekleyin."
        else:
            return None, f"Space-Track hatası: {error_msg[:100]}"


def fetch_celestrak_tles(group_key: str, sat_limit: int):
    """
    Enhanced CelesTrak TLE fetching with better error handling and performance.
    """
    try:
        config = GROUP_CONFIG.get(group_key)
        if not config:
            return None, f"Unknown group: {group_key}"

        celestrak_group = config["celestrak_group"]
        url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={celestrak_group}&FORMAT=TLE"
        headers = {
            "User-Agent": "Mozilla/5.0 (StarWeb-CARA/1.0)"
        }
        
        # Add retry logic for better reliability
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

        raw = resp.text
        if not raw or not raw.strip():
            return None, f"CelesTrak üzerinde '{group_key}' için veri bulunamadı."

        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            return None, "CelesTrak geçerli TLE döndürmedi."

        lines = trim_tle_lines(lines, sat_limit)
        return lines, "Veri CelesTrak üzerinden alındı."
    except requests.exceptions.Timeout:
        return None, "CelesTrak bağlantı zaman aşımı. Lütfen daha sonra tekrar deneyin."
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None, f"CelesTrak üzerinde '{group_key}' için veri bulunamadı (404)."
        elif e.response.status_code == 429:
            return None, "CelesTrak hız sınırı aşıldı. Lütfen birkaç dakika bekleyin."
        else:
            return None, f"CelesTrak HTTP hatası: {e.response.status_code}"
    except Exception as e:
        return None, f"CelesTrak hatası: {str(e)[:100]}"


def fetch_tles_with_fallback(
    username: str, password: str, group_key: str, sat_limit: int
):
    lines, primary_message = fetch_spacetrack_tles(
        username, password, group_key, sat_limit
    )
    if lines:
        return {"lines": lines, "source": "Space-Track", "message": primary_message}

    lines, fallback_message = fetch_celestrak_tles(group_key, sat_limit)
    if lines:
        return {
            "lines": lines,
            "source": "CelesTrak",
            "message": f"{primary_message} CelesTrak yedek kaynağı üzerinden veri alındı.",
        }

    st.sidebar.error(primary_message)
    st.sidebar.error(fallback_message)
    return None


# ================================================================================
#  TLE PARSING
# ================================================================================
def build_fallback_sat_name(tle_line_1: str, fallback_name_prefix: str = None) -> str:
    norad_id = tle_line_1[2:7].strip()
    if fallback_name_prefix:
        return f"{fallback_name_prefix} {norad_id}"
    return f"NORAD {norad_id}"


def parse_tles(lines: list, limit: int = 30, fallback_name_prefix: str = None) -> list:
    """
    Parse TLE lines into EarthSatellite objects.
    Note: Not cached because EarthSatellite objects are not serializable.
    The heavy computations (position propagation, conjunction analysis) are cached instead.
    """
    sats = []
    is_3ln = not (lines[0].startswith("1 ") or lines[0].startswith("2 "))
    step = 3 if is_3ln else 2
    for i in range(0, len(lines) - (step - 1), step):
        try:
            if is_3ln:
                name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
            else:
                name = build_fallback_sat_name(lines[i], fallback_name_prefix)
                l1, l2 = lines[i], lines[i + 1]
            sats.append(EarthSatellite(l1, l2, name, ts))
            if len(sats) >= limit:
                break
        except Exception:
            continue
    return sats


# ================================================================================
#  ORBITAL ELEMENTS & APSIS FILTER
# ================================================================================
#  ORBITAL ELEMENTS (from TLE)
# ================================================================================
def get_orbital_elements(sat: EarthSatellite) -> dict:
    """
    Extracts Kepler orbital elements from TLE.
    Note: Not cached because EarthSatellite objects are not serializable.
    """
    try:
        model = sat.model
        # Elements from TLE epoch
        incl = math.degrees(model.inclo)  # inclination (deg)
        raan = math.degrees(model.nodeo)  # RAAN (deg)
        ecc = model.ecco  # eccentricity
        argp = math.degrees(model.argpo)  # argument of periapsis (deg)
        mean_m = math.degrees(model.mo)  # mean anomaly (deg)
        n_rpm = model.no_kozai * (60.0 / (2 * math.pi))  # rad/min → devir/min
        # Semi-major axis: a = (GM/n^2)^(1/3), n rad/s
        GM = MU_EARTH_KM3_S2  # km^3/s^2
        n_rads = model.no_kozai / 60.0  # rad/s
        a_km = (GM / n_rads**2) ** (1 / 3)
        alt_km = a_km - EARTH_RADIUS_KM
        period_min = 2 * math.pi / model.no_kozai
        return {
            "Semi-major Axis a (km)": round(a_km, 1),
            "Mean Altitude (km)": round(alt_km, 1),
            "Eccentricity e": f"{ecc:.6f}",
            "Inclination i (°)": round(incl, 4),
            "RAAN (°)": round(raan, 4),
            "Arg of Perigee ω (°)": round(argp, 4),
            "Mean Anomaly M (°)": round(mean_m, 4),
            "Orbital Period (min)": round(period_min, 2),
            "Mean Motion n (rev/min)": round(n_rpm, 6),
        }
    except Exception:
        return {}


# ================================================================================
#  APSIS FILTER (Section 2.1 — Thesis)
# ================================================================================
def apsis_filter(sats: list, threshold_km: float = APSIS_FILTER_THRESHOLD_KM) -> list:
    """
    Apsis (Apogee-Perigee) Filter — Section 2.1
    Reduces O(N^2) complexity by filtering pairs with non-overlapping
    altitude bands.
    q1 > Q2 + D   →   physical intersection impossible → filtered
    Note: Not cached because EarthSatellite objects are not serializable.
    """
    R_E = EARTH_RADIUS_KM
    GM = MU_EARTH_KM3_S2

    def apsis(sat):
        try:
            # Enhanced error handling for different satellite models
            if hasattr(sat, 'model') and hasattr(sat.model, 'no_kozai'):
                n = sat.model.no_kozai / 60.0  # rad/s
                a = (GM / n**2) ** (1 / 3)
                e = sat.model.ecco
                per = a * (1 - e) - R_E  # perigee altitude
                apo = a * (1 + e) - R_E  # apogee altitude
                return per, apo
            else:
                # Fallback for satellites without model attributes
                return 0.0, 10000.0
        except Exception as e:
            # Enhanced error logging
            return 0.0, 10000.0

    passed = []
    for s1, s2 in combinations(sats, 2):
        q1, Q1 = apsis(s1)
        q2, Q2 = apsis(s2)
        if max(q1, q2) <= min(Q1, Q2) + threshold_km:
            passed.append((s1, s2))
    return passed


def apsis_overlap(sat1, sat2, threshold_km: float = APSIS_FILTER_THRESHOLD_KM) -> bool:
    """
    Backward compatibility function for apsis overlap check.
    Returns True if two satellites have overlapping altitude bands.
    """
    R_E = EARTH_RADIUS_KM
    GM = MU_EARTH_KM3_S2

    def apsis(sat):
        try:
            if hasattr(sat, 'model') and hasattr(sat.model, 'no_kozai'):
                n = sat.model.no_kozai / 60.0  # rad/s
                a = (GM / n**2) ** (1 / 3)
                e = sat.model.ecco
                per = a * (1 - e) - R_E  # perigee altitude
                apo = a * (1 + e) - R_E  # apogee altitude
                return per, apo
            else:
                return 0.0, 10000.0
        except Exception:
            return 0.0, 10000.0

    q1, Q1 = apsis(sat1)
    q2, Q2 = apsis(sat2)
    return max(q1, q2) <= min(Q1, Q2) + threshold_km


# ================================================================================
#  RISK CALCULATIONS (Foster 2D-Pc, Mahalanobis, Max Pc, Dilution, Fragmentation, Risk Level)
# ================================================================================
#  FOSTER 1992 2D-Pc (Section 3.1 — Thesis)
# ================================================================================
@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def foster_2d_pc(miss_km: float, sigma_x: float, sigma_y: float, hbr_km: float = 0.020) -> float:
    """
    Foster & Estes (1992) 2D-Pc Model — Section 3.1
    Enhanced with flexible parameter handling for compatibility.
    Cached for performance improvement.
    """
    try:
        if sigma_x <= 0 or sigma_y <= 0:
            return 0.0

        def integrand(y, x):
            return (1.0 / (2 * math.pi * sigma_x * sigma_y)) * \
                   math.exp(-0.5 * (((x - miss_km) / sigma_x) ** 2 + (y / sigma_y) ** 2))

        result, _ = dblquad(
            integrand, -hbr_km, hbr_km,
            lambda x: -math.sqrt(max(hbr_km**2 - x**2, 0)),
            lambda x: math.sqrt(max(hbr_km**2 - x**2, 0)),
            limit=50,
        )
        return max(float(result), 0.0)
    except Exception:
        return collision_probability_isotropic(miss_km, (sigma_x + sigma_y) / 2, hbr_km)


@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def collision_probability_isotropic(
    miss_km: float, sigma_km: float, hbr_km: float = 0.020
) -> float:
    """
    Chan (1997) isotropic model — fast fallback.
    Correct formula: P(|X_rel| ≤ HBR) for x ∈ N(miss, σ)
    Pc = Φ((HBR - miss)/σ) + Φ((HBR + miss)/σ) - 1
    Cached for performance improvement.
    """
    if sigma_km <= 0:
        return 0.0
    pc = (
        norm.cdf((hbr_km - miss_km) / sigma_km)
        + norm.cdf((hbr_km + miss_km) / sigma_km)
        - 1.0
    )
    return max(float(pc), 0.0)


# Public alias
def collision_probability(miss_km, sigma_km, hbr_km=0.020):
    return collision_probability_isotropic(miss_km, sigma_km, hbr_km)


# ================================================================================
#  MAHALANOBIS DISTANCE TEST (Section 3.2 — Thesis)
# ================================================================================
@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def mahalanobis_test(miss_km: float, sigma_km: float) -> dict:
    """
    2D-Pc validity test (CARA methodology — Section 3.2).
    Mahalanobis distance Md = miss / sigma.
    Md < 1.5 → linear motion assumption breaks down → 3D-Pc required.
    Cached for performance improvement.
    """
    if sigma_km <= 0:
        return {"Md": 999.0, "valid_2d": True, "label": "Valid"}
    Md = miss_km / sigma_km
    valid = Md >= 1.5
    if Md < 0.5:
        label = "Invalid — 3D-Pc / Monte Carlo required"
    elif Md < 1.5:
        label = "Borderline — 3D-Pc recommended"
    else:
        label = "2D-Pc Valid"
    return {"Md": round(Md, 3), "valid_2d": valid, "label": label}


# ================================================================================
#  MAXIMUM Pc ANALYSIS (Section 4 — Thesis)
# ================================================================================
@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def max_pc_analysis(miss_km: float, hbr_km: float = 0.020) -> float:
    """
    Max Pc — Section 4 (CARA toolkit).
    Scans covariance multiplier σ to find mathematical maximum Pc.
    Worst case: σ_opt = miss / sqrt(2) (Gaussian peak point).
    Cached for performance improvement.
    """
    sigma_opt = miss_km / math.sqrt(2.0) if miss_km > 0 else hbr_km
    return collision_probability_isotropic(miss_km, max(sigma_opt, 1e-6), hbr_km)


# ================================================================================
#  PROBABILITY DILUTION DETECTION (Section 4 — Thesis)
# ================================================================================
@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def dilution_check(pc: float, sigma_km: float, miss_km: float) -> dict:
    """
    Probability Dilution detection — Section 4.
    Wide covariance → small Pc → false confidence.
    Warning: sigma > 5*miss_km and pc < 1e-6
    Cached for performance improvement.
    """
    diluted = (sigma_km > 5.0 * miss_km) and (pc < 1e-6) and (miss_km < 100.0)
    if diluted:
        return {
            "diluted": True,
            "msg": "PROBABILITY DILUTION DETECTED: Wide covariance masking Pc value. "
            "WSPRT or Max-Pc analysis required.",
        }
    return {"diluted": False, "msg": "Normal"}


# ================================================================================
#  FRAGMENTATION PROBABILITY Pf (Section 4 — Thesis)
# ================================================================================
@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def fragmentation_probability(
    rel_vel_km_s: float, mass_a_kg: float = 250.0, mass_b_kg: float = 250.0
) -> dict:
    """
    Collision Consequence — Section 4.
    Kinetic energy-based fragmentation risk per NASA operational guidelines.
    Specific Energy: E_c = 0.5 * m_b * v_rel^2 / m_a  (J/g)
    E_c > 40 J/g → Catastrophic fragmentation (Kessler contribution)
    E_c > 0 J/g  → Damaging
    Cached for performance improvement.
    """
    v_ms = rel_vel_km_s * 1000.0
    E_c = 0.5 * mass_b_kg * v_ms**2 / (mass_a_kg * 1000.0)  # J/g
    if E_c >= 40.0:
        pf_level = "CATASTROPHIC"
        pf_color = "#ff2b4d"
        pf_desc = "Complete fragmentation — Kessler contribution likely"
    elif E_c >= 10.0:
        pf_level = "SEVERE"
        pf_color = "#ff6b00"
        pf_desc = "Operational loss and significant debris"
    elif E_c >= 1.0:
        pf_level = "DAMAGING"
        pf_color = "#ffaa00"
        pf_desc = "Partial damage or subsystem failure"
    else:
        pf_level = "LOW"
        pf_color = "#00ff9d"
        pf_desc = "Minor damage — fragmentation unlikely"
    n_debris = int(0.1 * (mass_a_kg + mass_b_kg) * (rel_vel_km_s / 7.0))
    return {
        "E_c_J_per_g": round(E_c, 2),
        "level": pf_level,
        "color": pf_color,
        "desc": pf_desc,
        "est_debris": n_debris,
    }


# ================================================================================
#  RISK LEVEL
# ================================================================================
@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def risk_level(pc: float, theme="dark") -> tuple:
    """NASA STD-8719.14 — 4-tier risk classification with enhanced error handling."""
    try:
        if pc > 1e-3:
            return "CRITICAL", "#dc3545" if theme == "light" else "#ff3d5c"
        elif pc > 1e-4:
            return "HIGH", "#fd7e14" if theme == "light" else "#ff6b00"
        elif pc > 1e-5:
            return "MEDIUM", "#ffc107" if theme == "light" else "#ffb800"
        else:
            return "LOW", "#28a745" if theme == "light" else "#00ffa8"
    except Exception:
        return "UNKNOWN", "#6c757d" if theme == "light" else "#5a7a94"


# ================================================================================
#  CONJUNCTION ANALYSIS
# ================================================================================
#  HELPER FUNCTION FOR CONJUNCTION METRICS (EXTRACTED TO REDUCE DUPLICATION)
# ================================================================================
def _compute_conjunction_metrics(
    sat1, sat2, pos1, pos2, jd_values, sigma_km, hbr_km, mass_a_kg, mass_b_kg, theme="dark"
):
    """
    Core function to compute conjunction metrics for a satellite pair.
    Extracted to eliminate code duplication between compute_conjunctions and compute_conjunctions_custom.
    Optimized with early returns and error handling.
    """
    # Validate inputs
    if pos1 is None or pos2 is None or sat1 is None or sat2 is None:
        return None

    # Compute distance array
    try:
        dists = np.linalg.norm(pos1 - pos2, axis=0)
    except Exception:
        return None

    if len(dists) == 0 or np.all(np.isnan(dists)):
        return None

    tca_idx = int(np.nanargmin(dists))
    min_d = float(dists[tca_idx])
    if min_d >= CONJUNCTION_DISTANCE_THRESHOLD_KM:
        return None

    best_t = ts.tt_jd(float(jd_values[tca_idx]))
    dist_arr = dists.tolist()

    rel_vel = _relative_velocity(sat1, sat2, best_t)
    pc_iso = collision_probability_isotropic(min_d, sigma_km, hbr_km)
    pc_foster = foster_2d_pc(min_d, sigma_km, sigma_km * 2, hbr_km=hbr_km)  # Note: sigma_y = 2*sigma_x as in original
    pc_max = max_pc_analysis(min_d, hbr_km)
    mah = mahalanobis_test(min_d, sigma_km)
    dil = dilution_check(pc_iso, sigma_km, min_d)
    frag = fragmentation_probability(rel_vel, mass_a_kg, mass_b_kg)
    sev, color = risk_level(pc_iso, theme)

    return {
        "TCA (UTC)": best_t.utc_strftime("%Y-%m-%d %H:%M:%S"),
        "Object A": sat1.name,
        "Object B": sat2.name,
        "Distance (km)": round(min_d, 3),
        "Relative Velocity (km/s)": round(rel_vel, 3),
        "Pc (isotropic)": pc_iso,
        "Pc (Foster 2D)": pc_foster,
        "Pc Max": pc_max,
        "Pc (scientific)": f"{pc_iso:.3e}",
        "Mahalanobis Md": mah["Md"],
        "2D-Pc Valid": mah["label"],
        "Dilution": dil["diluted"],
        "Dilution Message": dil["msg"],
        "Ec (J/g)": frag["E_c_J_per_g"],
        "Fragmentation Level": frag["level"],
        "Estimated Debris": frag["est_debris"],
        "Risk Level": sev,
        "_color": color,
        "_dist_arr": dist_arr,
        "_tca_tt": best_t.tt,
        "_s1": sat1,
        "_s2": sat2,
    }


def _relative_velocity(s1, s2, t) -> float:
    v1 = s1.at(t).velocity.km_per_s
    v2 = s2.at(t).velocity.km_per_s
    return float(np.linalg.norm(np.array(v1) - np.array(v2)))


# ================================================================================
#  MAIN CONJUNCTION ANALYSIS FUNCTIONS (REFactored)
# ================================================================================
def compute_conjunctions(
    sats: list,
    window_hrs: int,
    sigma_km: float,
    hbr_km: float = 0.020,
    mass_a_kg: float = 250.0,
    mass_b_kg: float = 250.0,
    theme: str = "dark"
) -> tuple:
    """
    Enhanced Apsis filter + 5-min step TCA scan + multiple Pc metrics with progress tracking.
    Returns: (df_results, n_apsis_filtered, n_total_pairs)
    """
    if ts is None:
        print("Error: Timescale not initialized")
        return pd.DataFrame(), 0, 0

    now = ts.now()
    times, _ = build_time_grid(now.tt, window_hrs)
    if times is None:
        print("Error: Failed to build time grid")
        return pd.DataFrame(), 0, 0

    jd_values = np.asarray(times.tt)
    n_total = len(sats) * (len(sats) - 1) // 2

    # Progress tracking for large datasets
    if n_total > 100:
        progress_bar = st.progress(0, text="Analyzing satellite pairs...")
    else:
        progress_bar = None

    # Apsis pre-filter (cached)
    candidate_pairs = apsis_filter(sats, threshold_km=APSIS_FILTER_THRESHOLD_KM)
    n_filtered = n_total - len(candidate_pairs)

    # Position caching for performance (cached)
    positions_by_id = {}
    for idx, sat in enumerate(sats):
        if progress_bar and idx % max(1, len(sats)//20) == 0:  # Update less frequently
            progress = (idx + 1) / len(sats)
            progress_bar.progress(progress, text=f"Precomputing positions... {idx+1}/{len(sats)}")
        positions_by_id[id(sat)] = (sat, propagated_positions(sat, times))

    results = []
    for idx, (sat1, sat2) in enumerate(candidate_pairs):
        if progress_bar:
            progress = (idx + 1) / len(candidate_pairs)
            if idx % max(1, len(candidate_pairs)//20) == 0:  # Update less frequently
                progress_bar.progress(progress, text=f"Analyzing conjunctions... {idx+1}/{len(candidate_pairs)}")

        # Retrieve precomputed data
        sat1_obj, pos1 = positions_by_id.get(id(sat1), (None, None))
        sat2_obj, pos2 = positions_by_id.get(id(sat2), (None, None))

        if pos1 is None or pos2 is None or sat1_obj is None or sat2_obj is None:
            continue

        # Compute metrics using helper function
        result = _compute_conjunction_metrics(
            sat1_obj, sat2_obj, pos1, pos2, jd_values, sigma_km, hbr_km, mass_a_kg, mass_b_kg, theme
        )
        if result:
            results.append(result)

    # Clear progress bar
    if progress_bar:
        progress_bar.empty()

    return pd.DataFrame(results), n_filtered, n_total


def compute_conjunctions_custom(
    my_sat,
    sats: list,
    window_hrs: int,
    sigma_km: float,
    hbr_km: float = 0.020,
    mass_a_kg: float = 250.0,
    mass_b_kg: float = 250.0,
    theme: str = "dark"
) -> pd.DataFrame:
    """
    Compares user's own satellite with existing satellite fleet.
    Apsis filter + 5-min TCA scan + full Pc metrics.
    """
    if ts is None:
        print("Error: Timescale not initialized")
        return pd.DataFrame()

    now = ts.now()
    times, _ = build_time_grid(now.tt, window_hrs)
    if times is None:
        print("Error: Failed to build time grid")
        return pd.DataFrame()

    jd_values = np.asarray(times.tt)
    R_E, GM = EARTH_RADIUS_KM, MU_EARTH_KM3_S2

    def apsis(sat):
        try:
            n = sat.model.no_kozai / 60.0
            a = (GM / n**2) ** (1 / 3)
            e = sat.model.ecco
            return a * (1 - e) - R_E, a * (1 + e) - R_E
        except Exception:
            return 0.0, 10000.0

    my_q, my_Q = apsis(my_sat)
    my_pos = propagated_positions(my_sat, times)  # Cached
    results = []
    if my_pos is None:
        return pd.DataFrame(results)

    # Precompute user satellite position once
    my_sat_obj = my_sat

    for idx, sat in enumerate(sats):
        q, Q = apsis(sat)
        # Apsis filter
        if max(my_q, q) > min(my_Q, Q) + 100.0:  # Note: Using 100.0 here as in original (intentional for custom?)
            continue

        sat_pos = propagated_positions(sat, times)  # Cached
        if sat_pos is None:
            continue

        # Compute metrics using helper function
        result = _compute_conjunction_metrics(
            my_sat_obj, sat, my_pos, sat_pos, jd_values, sigma_km, hbr_km, mass_a_kg, mass_b_kg, theme
        )
        if result:
            results.append(result)

    return pd.DataFrame(results)


# ================================================================================
#  PLOTS - ENHANCED VISUALIZATION
# ================================================================================
# NOT: fig_animated_conjunction artik module_figures_animation.py icinde.
# app.py icinde iki modulden de import edin.


#  PLOTS - ENHANCED VISUALIZATION
# ================================================================================
DARK = dict(
    paper_bgcolor="#05070a",
    plot_bgcolor="#05070a",
    font=dict(family="Space Mono, monospace", color="#c4d4e8", size=11),
)

# Enhanced color palette for better visual distinction
ENHANCED_COLORS = [
    "#00d4ff",  # Bright cyan
    "#00ffa8",  # Bright green
    "#ffb800",  # Bright orange
    "#ff6b00",  # Deep orange
    "#c060ff",  # Purple
    "#ff3d5c",  # Red
    "#60d0ff",  # Light blue
    "#80ffb0",  # Light green
    "#ffcc60",  # Yellow
    "#ff9060",  # Coral
]


def fig_3d_orbits(sats):
    """
    Enhanced 3D orbit visualization with improved visual effects and performance.
    Fixed syntax error: lighting dict now properly closed with ) instead of }
    """
    now = ts.now()
    fig = go.Figure()

    # Load Earth texture with enhanced styling
    earth = load_earth_texture(resolution=360, style="realistic")
    if earth:
        x, y, z, sc, cs = earth
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=sc,
                colorscale=cs,
                showscale=False,
                opacity=0.95,
                hoverinfo="skip",
                name="Earth",
                lightposition=dict(x=0, y=0, z=10000),
                lighting=dict(
                    ambient=0.5,
                    diffuse=0.9,
                    specular=0.1,
                    roughness=0.8,
                    fresnel=0.1
                ),
            )
        )
    else:
        r = EARTH_RADIUS_KM
        u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
        fig.add_trace(
            go.Surface(
                x=r * np.cos(u) * np.sin(v),
                y=r * np.sin(u) * np.sin(v),
                z=r * np.cos(v),
                colorscale="Blues",
                opacity=0.5,
                showscale=False,
            )
        )

    # Use enhanced color palette
    colors = ENHANCED_COLORS
    offsets = np.linspace(0, 95, 80) / 1440.0

    for k, sat in enumerate(sats):
        times = ts.tt_jd(now.tt + offsets)
        c = colors[k % len(colors)]

        # 1. Orbit trajectory calculation with error handling
        try:
            pos = sat.at(times).position.km
        except Exception:
            pos = np.full((3, 80), np.nan)

        if not np.all(np.isnan(pos)):
            fig.add_trace(
                go.Scatter3d(
                    x=pos[0].tolist(),
                    y=pos[1].tolist(),
                    z=pos[2].tolist(),
                    mode="lines",
                    line=dict(color=c, width=3),
                    name=sat.name,
                    opacity=0.9,
                    hovertemplate=f"<b>{sat.name}</b><br>" +
                                  "X: %{x:.1f} km<br>" +
                                  "Y: %{y:.1f} km<br>" +
                                  "Z: %{z:.1f} km<extra></extra>",
                )
            )

        # 2. Instantaneous position calculation with enhanced markers
        try:
            p0 = sat.at(now).position.km
        except Exception:
            p0 = np.full((3,), np.nan)

        if not np.any(np.isnan(p0)):
            fig.add_trace(
                go.Scatter3d(
                    x=[float(p0[0])],
                    y=[float(p0[1])],
                    z=[float(p0[2])],
                    mode="markers",
                    marker=dict(
                        color=c,
                        size=8,
                        symbol="circle",
                        line=dict(color="#ffffff", width=1.5),
                        opacity=0.9,
                    ),
                    name=f"{sat.name} (current)",
                    showlegend=False,
                    hovertemplate=f"<b>{sat.name}</b><br>" +
                                  "Current Position<br>" +
                                  "X: %{x:.1f} km<br>" +
                                  "Y: %{y:.1f} km<br>" +
                                  "Z: %{z:.1f} km<extra></extra>",
                )
            )

    fig.update_layout(
        **DARK,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            bgcolor="#000408",
            xaxis=dict(visible=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False),
            zaxis=dict(visible=False, showgrid=False, zeroline=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.8), up=dict(x=0, y=0, z=1)),
        ),
        legend=dict(
            font=dict(size=9, family="Space Mono", color="#c4d4e8"),
            bgcolor="rgba(5,7,10,.9)",
            bordercolor="#1e2d42",
            borderwidth=1,
            x=0.01,
            y=0.99,
            itemsizing="constant",
        ),
        hoverlabel=dict(
            bgcolor="rgba(10,15,24,.95)",
            bordercolor="#00d4ff",
            font_size=11,
            font_family="Space Mono",
        ),
    )
    return fig


def fig_ground_tracks(sats):
    """
    Enhanced ground track visualization with improved visual effects.
    """
    now = ts.now()
    offsets = np.linspace(0, 95, 200) / 1440.0
    colors = ENHANCED_COLORS
    fig = go.Figure()
    
    for k, sat in enumerate(sats):
        times = ts.tt_jd(now.tt + offsets)
        try:
            geo = wgs84.subpoint_of(sat.at(times))
            g0 = wgs84.subpoint_of(sat.at(now))
        except Exception:
            continue
        c = colors[k % len(colors)]
        fig.add_trace(
            go.Scattergeo(
                lat=geo.latitude.degrees,
                lon=geo.longitude.degrees,
                mode="lines",
                line=dict(color=c, width=2.5),
                name=sat.name,
                opacity=0.9,
                hovertemplate=f"<b>{sat.name}</b><br>" +
                              "Lat: %{lat:.2f}°<br>" +
                              "Lon: %{lon:.2f}°<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scattergeo(
                lat=[g0.latitude.degrees],
                lon=[g0.longitude.degrees],
                mode="markers+text",
                marker=dict(
                    color=c,
                    size=10,
                    symbol="circle",
                    line=dict(color="#ffffff", width=1.5),
                    opacity=0.9,
                ),
                text=[sat.name],
                textposition="top right",
                textfont=dict(size=9, family="Space Mono", color=c, weight="bold"),
                showlegend=False,
                hovertemplate=f"<b>{sat.name}</b><br>" +
                              "Current Position<br>" +
                              "Lat: %{lat:.2f}°<br>" +
                              "Lon: %{lon:.2f}°<extra></extra>",
            )
        )
    fig.update_layout(
        **DARK,
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
        geo=dict(
            showland=True,
            landcolor="#0d2137",
            showocean=True,
            oceancolor="#050d18",
            showcoastlines=True,
            coastlinecolor="#2a5070",
            coastlinewidth=1.0,
            showcountries=True,
            countrycolor="#152535",
            countrywidth=0.5,
            showlakes=True,
            lakecolor="#080f1a",
            showrivers=True,
            rivercolor="#0a1828",
            showframe=False,
            bgcolor="#05070a",
            projection_type="natural earth",
            resolution=50,
            lonaxis=dict(
                range=[-180, 180],
                showgrid=True,
                gridcolor="rgba(30,45,66,.6)",
                gridwidth=0.4,
            ),
            lataxis=dict(
                range=[-90, 90],
                showgrid=True,
                gridcolor="rgba(30,45,66,.6)",
                gridwidth=0.4,
            ),
        ),
        legend=dict(
            font=dict(size=9, family="Space Mono", color="#c4d4e8"),
            bgcolor="rgba(5,7,10,.9)",
            bordercolor="#1a2740",
            borderwidth=1,
            x=0.0,
            y=1.0,
        ),
        title=dict(
            text="Ground Track -- Current Position and 95min Orbit",
            font=dict(size=11, family="Barlow Condensed", color="#00c8ff"),
            x=0.01,
            y=0.99,
        ),
    )
    return fig


@st.cache_data(show_spinner=False, ttl=1800)  # Cache for 30 minutes
def fig_distance_profile(dist_arr, window_hrs, miss_km, sigma_km, hbr_km=0.020):
    step_m = ANALYSIS_STEP_MIN
    t_axis = np.arange(len(dist_arr)) * step_m / 60.0
    fig = go.Figure()
    fig.add_hline(
        y=hbr_km,
        line=dict(color="#ff2b4d", dash="dot", width=1),
        annotation_text=f"HBR ({hbr_km * 1000:.0f} m)",
        annotation_font_size=9,
    )
    fig.add_hrect(
        y0=max(0, miss_km - sigma_km),
        y1=miss_km + sigma_km,
        fillcolor="rgba(0,200,255,.05)",
        line_width=0,
    )
    fig.add_trace(
        go.Scatter(
            x=t_axis,
            y=dist_arr,
            mode="lines",
            line=dict(color="#00c8ff", width=1.5),
            name="Distance (km)",
            fill="tozeroy",
            fillcolor="rgba(0,200,255,.04)",
        )
    )
    dist_np = np.asarray(dist_arr, dtype=float)
    if len(dist_np) and not np.all(np.isnan(dist_np)):
        tca_i = int(np.nanargmin(dist_np))
        fig.add_trace(
            go.Scatter(
                x=[t_axis[tca_i]],
                y=[dist_np[tca_i]],
                mode="markers+text",
                marker=dict(color="#ff2b4d", size=8),
                text=[f" TCA: {dist_np[tca_i]:.1f} km"],
                textposition="top right",
                textfont=dict(size=9, family="Space Mono", color="#ff2b4d"),
                name="TCA",
                showlegend=False,
            )
        )
    fig.update_layout(
        **DARK,
        height=280,
        xaxis=dict(
            title="Time (hours)",
            gridcolor="#1a2740",
            zeroline=False,
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            title="Distance (km)",
            gridcolor="#1a2740",
            zeroline=False,
            tickfont=dict(size=9),
        ),
        title=dict(
            text="Distance Profile — TCA Analysis",
            font=dict(size=11, family="Barlow Condensed", color="#00c8ff"),
            x=0.01,
        ),
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=35, b=10),
    )
    return fig


@st.cache_data(show_spinner=False, ttl=1800)  # Cache for 30 minutes
def fig_risk_gauge(pc: float):
    sev, color = risk_level(pc)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pc,
            number=dict(
                valueformat=".2e", font=dict(family="Space Mono", color=color, size=18)
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 1e-3],
                    tickvals=[0, 1e-5, 1e-4, 1e-3],
                    ticktext=["0", "1e-5", "1e-4", "1e-3"],
                    tickfont=dict(size=8, family="Space Mono", color="#4a6880"),
                ),
                bar=dict(color=color, thickness=0.25),
                bgcolor="#0c1018",
                bordercolor="#1a2740",
                steps=[
                    dict(range=[0, 1e-5], color="#0d1820"),
                    dict(range=[1e-5, 1e-4], color="#141e10"),
                    dict(range=[1e-4, 1e-3], color="#1e1008"),
                ],
                threshold=dict(line=dict(color="#ff2b4d", width=2), value=1e-4),
            ),
            title=dict(
                text=f"Pc — {sev}",
                font=dict(family="Barlow Condensed", color=color, size=14),
            ),
            domain=dict(x=[0, 1], y=[0, 1]),
        )
    )
    fig.update_layout(**DARK, height=210, margin=dict(l=10, r=10, t=10, b=10))
    return fig


@st.cache_data(show_spinner=False, ttl=1800)  # Cache for 30 minutes
def fig_orbital_elements_radar(elems_list):
    """Display satellites by orbital elements using scatter plot."""
    fig = go.Figure()
    colors = ["#00c8ff", "#00ff9d", "#ffaa00", "#ff6b00", "#c060ff", "#ff2b4d"]
    for k, (name, elems) in enumerate(elems_list):
        if not elems:
            continue
        try:
            # Error fix: Changed "Mean Altitude a (km)" to "Mean Altitude (km)"
            alt = float(str(elems.get("Mean Altitude (km)", 0)))
            incl = float(str(elems.get("Inclination i (°)", 0)))
            ecc = float(str(elems.get("Eccentricity e", "0")))
            fig.add_trace(
                go.Scatter(
                    x=[incl],
                    y=[alt],
                    mode="markers+text",
                    marker=dict(
                        color=colors[k % len(colors)],
                        size=10 + ecc * 80,
                        line=dict(color="#fff", width=0.5),
                    ),
                    text=[name[:12]],
                    textposition="top center",
                    textfont=dict(
                        size=8, family="Space Mono", color=colors[k % len(colors)]
                    ),
                    name=name,
                )
            )
        except Exception:
            continue
    fig.update_layout(
        **DARK,
        height=320,
        xaxis=dict(
            title="Inclination i (°)",
            gridcolor="#1a2740",
            zeroline=False,
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            title="Altitude (km)",
            gridcolor="#1a2740",
            zeroline=False,
            tickfont=dict(size=9),
        ),
        title=dict(
            text="Orbital Space — Altitude / Inclination Distribution",
            font=dict(size=11, family="Barlow Condensed", color="#00c8ff"),
            x=0.01,
        ),
        margin=dict(l=10, r=10, t=35, b=10),
        showlegend=False,
    )
    return fig


# ================================================================================
#  LIVE 3D ANIMATION (Two Satellites — TCA Focused) — OPTİMİZE EDİLDİ
# ================================================================================
#  LIVE 3D ANIMATION (Two Satellites — TCA Focused)
# ================================================================================

def fig_animated_conjunction(
    sat_a,
    sat_b,
    window_hrs: int = 6,
    show_orbits: bool = True,
    show_tca: bool = True,
    center_tt: float = None,
    frame_duration: int = 60,
):
    """
    3D Plotly figure showing two satellites with real-time animation.
    Robust version with error handling and fallback.
    Precomputed star field for performance.
    Fixed syntax error: lighting dict now properly closed with ) instead of }
    Note: Not cached because EarthSatellite objects are not serializable.
    """
    if ts is None:
        print("Error: Timescale not initialized")
        return go.Figure(), 0, 0.0, np.array([0.0]), np.array([0.0]), 0.0

    now = ts.now()
    # Perf: coarser steps → fewer frames → faster WebGL rendering.
    # max_frames=50: paylaşılan/düşük güçlü makinelerde (jüri sunumu vb.)
    # de akıcı kalması için 96'dan düşürüldü. Her frame artık sadece 3 trace
    # taşıdığı için (eskiden 5), toplam WebGL güncelleme yükü zaten
    # ~3x azaldı; ek olarak frame sayısını da kısmak toplam yükü daha da düşürür.
    sim_start_tt = (
        float(center_tt) - (window_hrs / 2.0) / 24.0
        if center_tt is not None
        else now.tt
    )
    max_frames = 50
    step_min = max(2, int(math.ceil(window_hrs * 60 / max_frames)))
    n_frames = min(max(2, int(math.ceil(window_hrs * 60 / step_min)) + 1), max_frames)
    if n_frames < 2:
        return go.Figure(), 0, 0.0, np.array([0.0]), np.array([sim_start_tt]), sim_start_tt

    orbit_pts = 72

    # Full orbit paths (static background)
    orb_off = np.linspace(0, 96, orbit_pts) / 1440.0
    try:
        orb_a = sat_a.at(ts.tt_jd(sim_start_tt + orb_off)).position.km
        orb_b = sat_b.at(ts.tt_jd(sim_start_tt + orb_off)).position.km
    except Exception:
        orb_a = np.full((3, orbit_pts), np.nan)
        orb_b = np.full((3, orbit_pts), np.nan)

    # Animation step positions
    anim_off = np.arange(n_frames) * step_min / 1440.0
    anim_jd = sim_start_tt + anim_off
    try:
        pos_a = sat_a.at(ts.tt_jd(anim_jd)).position.km
        pos_b = sat_b.at(ts.tt_jd(anim_jd)).position.km
    except Exception:
        pos_a = np.full((3, n_frames), np.nan)
        pos_b = np.full((3, n_frames), np.nan)

    # Calculate distances safely ignoring NaNs
    dists = np.linalg.norm(pos_a - pos_b, axis=0)

    if np.all(np.isnan(dists)):
        tca_idx = 0
        tca_dist = 0.0
    else:
        tca_idx = int(np.nanargmin(dists))
        tca_dist = float(dists[tca_idx])

    def dist_color(d):
        if np.isnan(d):
            return "rgba(100,180,255,0.55)"
        if d < 50:
            return "#ff2b4d"
        if d < 200:
            return "#ffaa00"
        return "rgba(100,180,255,0.55)"

    fig = go.Figure()

    # Precomputed star field (static)
    rng = np.random.default_rng(7)
    star_count = 90
    star_phi = rng.uniform(0, 2 * np.pi, star_count)
    star_costheta = rng.uniform(-1, 1, star_count)
    star_theta = np.arccos(star_costheta)
    star_r = rng.uniform(12500, 16500, star_count)
    fig.add_trace(
        go.Scatter3d(
            x=(star_r * np.sin(star_theta) * np.cos(star_phi)).tolist(),
            y=(star_r * np.sin(star_theta) * np.sin(star_phi)).tolist(),
            z=(star_r * np.cos(star_theta)).tolist(),
            mode="markers",
            marker=dict(
                size=rng.uniform(1.0, 2.4, star_count).tolist(),
                color="rgba(210,235,255,0.55)",
            ),
            hoverinfo="skip",
            showlegend=False,
            name="Star field",
        )
    )

    # Perf: lower resolution → fewer WebGL vertices → much faster render per frame.
    # resolution=40 → grid is 40 x 80 = 3200 vertices (was 72x144=10368 → ~3.2x lighter).
    # Bu değer sabit ve statik (frame başına yeniden çizilmiyor), bu yüzden
    # görsel kalite kaybı animasyon sırasında neredeyse fark edilmiyor.
    earth = load_earth_texture(resolution=40, style="night")
    if earth:
        x, y, z, sc, cs = earth
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=sc,
                colorscale=cs,
                showscale=False,
                opacity=0.98,
                hoverinfo="skip",
                lightposition=dict(x=0, y=0, z=10000),
                lighting=dict(ambient=0.72, diffuse=0.88, specular=0.05, roughness=0.82),
                name="Earth",
            )
        )
    else:
        r = EARTH_RADIUS_KM
        u, v = np.mgrid[0 : 2 * np.pi : 120j, 0 : np.pi : 60j]
        colorscale_earth = [
            [0.0, "#081828"],
            [0.2, "#0a2848"],
            [0.35, "#1a3868"],
            [0.5, "#2a4888"],
            [0.65, "#2a4888"],
            [0.8, "#1a3868"],
            [0.9, "#d0c0b0"],
            [1.0, "#ffffff"],
        ]
        fig.add_trace(
            go.Surface(
                x=r * np.cos(u) * np.sin(v),
                y=r * np.sin(u) * np.sin(v),
                z=r * np.cos(v),
                colorscale=colorscale_earth,
                opacity=0.95,
                showscale=False,
                name="Earth",
            )
        )

    # Coordinate reference lines — MINIMAL set (was 20 traces, now 3).
    # Fewer static traces = dramatically faster per-frame WebGL redraw.
    r_earth = EARTH_RADIUS_KM
    _pts = 80
    _lat_pm = np.linspace(-np.pi / 2, np.pi / 2, _pts)
    _lon_eq = np.linspace(-np.pi, np.pi, _pts)
    # Equator
    fig.add_trace(
        go.Scatter3d(
            x=(r_earth * np.cos(_lon_eq)).tolist(),
            y=(r_earth * np.sin(_lon_eq)).tolist(),
            z=np.zeros(_pts).tolist(),
            mode="lines",
            line=dict(color="rgba(100,150,200,0.35)", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # Prime meridian (lon=0)
    fig.add_trace(
        go.Scatter3d(
            x=(r_earth * np.cos(_lat_pm)).tolist(),
            y=np.zeros(_pts).tolist(),
            z=(r_earth * np.sin(_lat_pm)).tolist(),
            mode="lines",
            line=dict(color="rgba(100,150,200,0.25)", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # 90E meridian
    fig.add_trace(
        go.Scatter3d(
            x=np.zeros(_pts).tolist(),
            y=(r_earth * np.cos(_lat_pm)).tolist(),
            z=(r_earth * np.sin(_lat_pm)).tolist(),
            mode="lines",
            line=dict(color="rgba(100,150,200,0.25)", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Full orbit trails (SAFE CHECK ADDED)
    if show_orbits and not np.all(np.isnan(orb_a)):
        fig.add_trace(
            go.Scatter3d(
                x=orb_a[0].tolist(),
                y=orb_a[1].tolist(),
                z=orb_a[2].tolist(),
                mode="lines",
                line=dict(color="rgba(0,200,255,0.22)", width=2),
                name=sat_a.name + " orbit",
                showlegend=False,
            )
        )

    if show_orbits and not np.all(np.isnan(orb_b)):
        fig.add_trace(
            go.Scatter3d(
                x=orb_b[0].tolist(),
                y=orb_b[1].tolist(),
                z=orb_b[2].tolist(),
                mode="lines",
                line=dict(color="rgba(255,107,0,0.22)", width=2),
                name=sat_b.name + " orbit",
                showlegend=False,
            )
        )

    # TCA point
    mid_tca = (pos_a[:, tca_idx] + pos_b[:, tca_idx]) / 2
    pa_tca = pos_a[:, tca_idx]
    pb_tca = pos_b[:, tca_idx]

    if show_tca and not np.any(np.isnan(mid_tca)):
        ring_radius = max(140.0, min(900.0, max(tca_dist * 1.6, 180.0)))
        ring_angle = np.linspace(0, 2 * np.pi, 96)
        normal = (
            mid_tca / np.linalg.norm(mid_tca)
            if np.linalg.norm(mid_tca) > 0
            else np.array([0.0, 0.0, 1.0])
        )
        basis_a = np.cross(normal, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(basis_a) < 1e-6:
            basis_a = np.cross(normal, np.array([0.0, 1.0, 0.0]))
        basis_a = basis_a / np.linalg.norm(basis_a)
        basis_b = np.cross(normal, basis_a)
        ring = (
            mid_tca[:, None]
            + ring_radius * np.cos(ring_angle)[None, :] * basis_a[:, None]
            + ring_radius * np.sin(ring_angle)[None, :] * basis_b[:, None]
        )
        fig.add_trace(
            go.Scatter3d(
                x=ring[0].tolist(),
                y=ring[1].tolist(),
                z=ring[2].tolist(),
                mode="lines",
                line=dict(color="rgba(255,43,77,0.55)", width=3),
                hoverinfo="skip",
                showlegend=False,
                name="TCA risk zone",
            )
        )
        if not np.any(np.isnan(pa_tca)) and not np.any(np.isnan(pb_tca)):
            fig.add_trace(
                go.Scatter3d(
                    x=[float(pa_tca[0]), float(pb_tca[0])],
                    y=[float(pa_tca[1]), float(pb_tca[1])],
                    z=[float(pa_tca[2]), float(pb_tca[2])],
                    mode="lines",
                    line=dict(color="rgba(255,43,77,0.72)", width=4, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Closest approach chord",
                )
            )

    # ── TCA-FACING CAMERA ALGORITHM ──────────────────────────────────────────
    # Strategy: position camera in the direction of TCA from Earth center so
    # TCA always faces the viewer.  Plotly 3D eye coords are scene-normalised:
    # with aspectmode='cube' and data ~±8000 km, eye magnitude ≈ 2.0–2.5 maps
    # to a comfortable full-globe view.
    _CAM_DIST = 2.3  # eye distance in normalised scene units
    _CAM_DIST_Z_BOOST = 0.25  # slight upward tilt so labels aren't clipped

    if not np.any(np.isnan(mid_tca)) and np.linalg.norm(mid_tca) > 100:
        _unit = mid_tca / np.linalg.norm(mid_tca)  # unit vector toward TCA
        _eye = _unit * _CAM_DIST

        # Lift camera slightly so text labels on top aren't cut off
        _eye[2] += _CAM_DIST_Z_BOOST
        # Re-normalise to keep distance consistent after z-lift
        _eye = _eye / np.linalg.norm(_eye) * _CAM_DIST

        # Up vector: default world-Z; fall back to world-Y near poles
        # (avoids gimbal lock when eye ≈ (0,0,±1))
        _up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(_unit, np.array([0.0, 0.0, 1.0]))) > 0.92:
            _up = np.array([0.0, 1.0, 0.0])

        tca_camera = dict(
            eye=dict(x=float(_eye[0]), y=float(_eye[1]), z=float(_eye[2])),
            center=dict(x=0, y=0, z=0),  # always look at Earth centre
            up=dict(x=float(_up[0]), y=float(_up[1]), z=float(_up[2])),
        )
    else:
        # Fallback: sensible default if TCA position is unavailable
        tca_camera = dict(
            eye=dict(x=1.7, y=1.7, z=0.75),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1),
        )
    # ─────────────────────────────────────────────────────────────────────────

    if show_tca and not np.any(np.isnan(mid_tca)):
        fig.add_trace(
            go.Scatter3d(
                x=[float(mid_tca[0])],
                y=[float(mid_tca[1])],
                z=[float(mid_tca[2])],
                mode="markers+text",
                marker=dict(
                    color="#ff2b4d",
                    size=10,
                    symbol="diamond",
                    line=dict(color="#ffffff", width=1),
                ),
                text=[f"TCA {tca_dist:.1f} km"],
                textposition="top right",
                textfont=dict(color="#ff2b4d", size=9, family="Space Mono"),
                name="TCA Point",
            )
        )

    # ── PERF REWRITE: Trail'ler artık STATİK ─────────────────────────────────
    # Eskiden her frame'de "kayan pencere" (trail_len noktalık) yeniden
    # hesaplanıp yeni bir Scatter3d nesnesi olarak yollanıyordu (frame başına
    # 2 trace). Bu, n_frames sayısı kadar Python nesnesi + WebGL güncellemesi
    # demekti. Artık tüm animasyon penceresindeki iz TEK SEFERDE, statik
    # olarak çiziliyor; frame'ler sadece hareketli marker'ları günceller.
    if pos_a.shape[1] > 0 and not np.all(np.isnan(pos_a)):
        fig.add_trace(
            go.Scatter3d(
                x=pos_a[0].tolist(),
                y=pos_a[1].tolist(),
                z=pos_a[2].tolist(),
                mode="lines",
                line=dict(color="rgba(0,200,255,0.65)", width=3.5),
                name=sat_a.name + " track",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    if pos_b.shape[1] > 0 and not np.all(np.isnan(pos_b)):
        fig.add_trace(
            go.Scatter3d(
                x=pos_b[0].tolist(),
                y=pos_b[1].tolist(),
                z=pos_b[2].tolist(),
                mode="lines",
                line=dict(color="rgba(255,107,0,0.65)", width=3.5),
                name=sat_b.name + " track",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    n_static = len(fig.data)

    def make_dynamic_traces(i):
        """
        Always returns EXACTLY 3 traces (Plotly frame update requires fixed count).
        (Eskiden 5'ti: trail A, trail B, marker A, marker B, distance line.
        Trail'ler artık statik — bkz. yukarısı. Mesafe metni de kaldırıldı;
        değer artık başlıkta gösteriliyor, çünkü 3D WebGL'de her frame'de
        text glyph güncellemek pahalı.)
        Earth rotation removed — static Earth stays in base figure (memory + speed fix).
        """
        pa = pos_a[:, i]
        pb = pos_b[:, i]
        d_val = dists[i]
        dc = dist_color(d_val)
        beam_width = 4 if not np.isnan(d_val) and d_val < 200 else 3

        # Trace 0 — Position marker A
        if not np.any(np.isnan(pa)):
            tr0 = go.Scatter3d(
                x=[float(pa[0])],
                y=[float(pa[1])],
                z=[float(pa[2])],
                mode="markers",
                marker=dict(
                    color="#00c8ff",
                    size=12,
                    symbol="diamond",
                    line=dict(color="#ffffff", width=1.2),
                    opacity=0.98,
                ),
                name=sat_a.name + " pos",
                showlegend=False,
            )
        else:
            tr0 = go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(color="#00c8ff", size=12, symbol="diamond"),
                name=sat_a.name + " pos",
                showlegend=False,
            )

        # Trace 1 — Position marker B
        if not np.any(np.isnan(pb)):
            tr1 = go.Scatter3d(
                x=[float(pb[0])],
                y=[float(pb[1])],
                z=[float(pb[2])],
                mode="markers",
                marker=dict(
                    color="#ff6b00",
                    size=12,
                    symbol="circle",
                    line=dict(color="#ffffff", width=1.2),
                    opacity=0.98,
                ),
                name=sat_b.name + " pos",
                showlegend=False,
            )
        else:
            tr1 = go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(color="#ff6b00", size=12, symbol="circle"),
                name=sat_b.name + " pos",
                showlegend=False,
            )

        # Trace 2 — Distance line (metin YOK — 3D WebGL'de her frame'de text
        # glyph güncellemek pahalıydı; mesafe değeri artık başlıkta gösteriliyor)
        if not np.any(np.isnan(pa)) and not np.any(np.isnan(pb)):
            tr2 = go.Scatter3d(
                x=[float(pa[0]), float(pb[0])],
                y=[float(pa[1]), float(pb[1])],
                z=[float(pa[2]), float(pb[2])],
                mode="lines",
                line=dict(color=dc, width=beam_width, dash="dot"),
                name="Distance",
                showlegend=False,
            )
        else:
            tr2 = go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color=dc, width=beam_width, dash="dot"),
                name="Distance",
                showlegend=False,
            )

        return [tr0, tr1, tr2]

    for tr in make_dynamic_traces(0):
        fig.add_trace(tr)

    n_dynamic = len(fig.data) - n_static
    if n_dynamic == 0:
        tca_tt = anim_jd[tca_idx]  # TCA zamanını anim_jd'den al
        return fig, tca_idx, tca_dist, dists, anim_jd, tca_tt

    dyn_indices = list(range(n_static, n_static + n_dynamic))

    frames = []
    slider_steps = []
    for i in range(n_frames):
        t_utc = ts.tt_jd(anim_jd[i]).utc_strftime("%H:%M UTC")
        t_min = i * step_min

        # Daha şık, HUD tarzı dinamik başlık tasarımı (Sıfır dolguları kaldırıldı)
        warn_tag = "<span style='color:#ff2b4d;'> ⚠ TCA</span>" if i == tca_idx else ""
        title_txt = f"{sat_a.name} × {sat_b.name} <br><sup style='color:#b8cfe0;'>⏱ T+{t_min} min &nbsp;|&nbsp; {t_utc} &nbsp;|&nbsp; Δ {dists[i]:.1f} km{warn_tag}</sup>"

        frames.append(
            go.Frame(
                data=make_dynamic_traces(i),
                traces=dyn_indices,
                name=str(i),
                layout=go.Layout(title_text=title_txt),
            )
        )
        lbl = t_utc if i % max(1, n_frames // 20) == 0 else ""
        slider_steps.append(
            dict(
                args=[
                    [str(i)],
                    dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                ],
                label=lbl,
                method="animate",
            )
        )

    fig.frames = frames

    # Initial title displayed before animation starts
    base_t_utc = ts.tt_jd(anim_jd[0]).utc_strftime("%H:%M UTC")
    base_title = f"{sat_a.name} × {sat_b.name} <br><sup style='color:#b8cfe0;'>⏱ T+0 min &nbsp;|&nbsp; {base_t_utc} &nbsp;|&nbsp; Δ {dists[0]:.1f} km</sup>"

    fig.update_layout(
        **DARK,
        height=640,
        margin=dict(l=0, r=0, t=70, b=10),
        title=dict(
            text=base_title,
            font=dict(family="Barlow Condensed", color="#00c8ff", size=15),
            x=0.01,
            y=0.98,
        ),
        scene=dict(
            bgcolor="#000408",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube",
            camera=tca_camera,  # dynamically oriented toward TCA
        ),
        legend=dict(
            font=dict(size=8, family="Space Mono"),
            bgcolor="rgba(0,4,8,.85)",
            bordercolor="#1a2740",
            borderwidth=1,
            x=0.01,
            y=0.92,
            itemsizing="constant",
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                bgcolor="#0c1018",
                bordercolor="#1a2740",
                font=dict(family="Space Mono", size=8, color="#b8cfe0"),
                y=1.02,
                x=0.5,
                xanchor="center",
                pad=dict(r=4),
                direction="left",
                buttons=[
                    dict(
                        label="▶ 1x",
                        method="animate",
                        args=[
                            [str(k) for k in range(n_frames)],
                            dict(
                                frame=dict(duration=frame_duration, redraw=False),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="⏩ 2x",
                        method="animate",
                        args=[
                            [str(k) for k in range(0, n_frames, 2)],
                            dict(
                                frame=dict(duration=frame_duration, redraw=False),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="⏭ 5x",
                        method="animate",
                        args=[
                            [str(k) for k in range(0, n_frames, 5)],
                            dict(
                                frame=dict(duration=frame_duration, redraw=False),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ STOP",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False), mode="immediate"
                            ),
                        ],
                    ),
                    dict(
                        label="⏮ TCA",
                        method="animate",
                        args=[
                            [str(tca_idx)],
                            dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                        ],
                    ),
                ],
            ),
        ],
        sliders=[
            dict(
                steps=slider_steps,
                active=0,
                currentvalue=dict(
                    prefix="⏱  ",
                    font=dict(family="Space Mono", size=9, color="#4a6880"),
                ),
                pad=dict(t=64, b=0),
                len=0.92,
                x=0.04,
                bgcolor="#0c1018",
                bordercolor="#1a2740",
                tickcolor="#1a2740",
                font=dict(color="#4a6880", size=7),
            )
        ],
    )
    tca_tt = anim_jd[tca_idx]  # TCA zamanını anim_jd'den al
    return fig, tca_idx, tca_dist, dists, anim_jd, tca_tt


# ================================================================================
#  INTERFACE
# ================================================================================
st.set_page_config(
    page_title="StarWeb-CARA: Conjunction Assessment and Collision Risk Analysis for Starlink and OneWeb Megaconstellations",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply theme CSS
st.markdown(get_theme_css(st.session_state.get("theme", "dark")), unsafe_allow_html=True)

st.markdown(
    """
<div style="padding:24px 0 12px 0; border-bottom:2px solid #1e2d42; margin-bottom:24px; position:relative;">
  <div style="font-family:'Space Mono',monospace; font-size:.7rem;
              color:#5a7a94; letter-spacing:.25em; text-transform:uppercase; margin-bottom:8px;
              background: linear-gradient(90deg, #00d4ff 0%, #00ffa8 100%);
              -webkit-background-clip: text;
              -webkit-text-fill-color: transparent;
              background-clip: text;">
    Conjunction Assessment and Collision Risk Analysis
  </div>
  <h1 style="margin:0; font-size:2rem; line-height:1.3;">
    Low Earth Orbit<br>
    <span style="color:#00d4ff;">Conjunction Assessment &amp; Collision Risk Analysis</span>
  </h1>
  <div style="font-family:'Inter',sans-serif; font-size:.9rem;
              color:#5a7a94; margin-top:12px; letter-spacing:.06em; font-weight:400;">
    Space Sciences and Technologies Graduation Project · Space-Track GP Database · Skyfield SGP4 Propagator
  </div>
  <div style="position:absolute; top:0; right:0; width:100px; height:4px; 
              background: linear-gradient(90deg, #00d4ff 0%, #00ffa8 100%); border-radius:2px;"></div>
</div>
""",
    unsafe_allow_html=True,
)

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
<div style="padding:16px 0 20px 0; border-bottom:2px solid #1e2d42; margin-bottom:20px;">
  <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.4rem;
              font-weight:700; color:#00d4ff; letter-spacing:.08em; text-transform:uppercase;
              margin-bottom:4px;">
    CONTROL PANEL
  </div>
  <div style="font-family:'Space Mono',monospace; font-size:.65rem;
              color:#5a7a94; letter-spacing:.15em; text-transform:uppercase;">
    System Configuration
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─── THEME SELECTOR ─────────────────────────────────────────────────────────
st.sidebar.markdown(
    """<div style="font-family:'Space Mono',monospace;font-size:.7rem;
    letter-spacing:.18em;color:#00d4ff;text-transform:uppercase;
    border-bottom:1px solid #1e2d42;padding-bottom:6px;margin-bottom:12px;
    display:flex; align-items:center; gap:8px;">
    <span style="background:linear-gradient(135deg, #00d4ff 0%, #00ffa8 100%);
                 color:#05070a; padding:2px 8px; border-radius:4px; font-weight:700;">00</span>
    THEME SELECTION</div>""",
    unsafe_allow_html=True,
)

# Initialize theme in session state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Theme selector
theme_options = {
    "dark": {
        "label": "🌙 Dark Mission Control",
        "description": "Professional dark theme with cyan accents"
    },
    "light": {
        "label": "☀️ Professional Light",
        "description": "Clean black & white theme with blue accents"
    }
}

selected_theme = st.sidebar.selectbox(
    "Choose Theme",
    options=list(theme_options.keys()),
    format_func=lambda x: theme_options[x]["label"],
    index=0 if st.session_state.theme == "dark" else 1,
    key="theme_selector"
)

# Update theme if changed
if selected_theme != st.session_state.theme:
    st.session_state.theme = selected_theme
    st.rerun()

st.sidebar.caption(theme_options[selected_theme]["description"])

st.sidebar.markdown(
    """<div style="height:1px; background:linear-gradient(90deg, transparent 0%, #1e2d42 50%, transparent 100%);
                margin:20px 0;"></div>""",
    unsafe_allow_html=True,
)

# ─── SECTION 1: AUTO TLE DOWNLOAD ─────────────────────────────────────────
st.sidebar.markdown(
    """<div style="font-family:'Space Mono',monospace;font-size:.7rem;
    letter-spacing:.18em;color:#00d4ff;text-transform:uppercase;
    border-bottom:1px solid #1e2d42;padding-bottom:6px;margin-bottom:12px;
    display:flex; align-items:center; gap:8px;">
    <span style="background:linear-gradient(135deg, #00d4ff 0%, #00ffa8 100%);
                 color:#05070a; padding:2px 8px; border-radius:4px; font-weight:700;">01</span>
    AUTO TLE DOWNLOAD</div>""",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    """<div style="font-family:'Inter',sans-serif; font-size:.85rem; 
                color:#c4d4e8; font-weight:600; margin-bottom:8px;">
    Space-Track Authentication
  </div>""",
    unsafe_allow_html=True,
)
user_email = st.sidebar.text_input("Email", placeholder="user@domain.com")
user_pass = st.sidebar.text_input("Password", placeholder="........", type="password")
st.sidebar.markdown("**Target Satellite Constellation** *(focused on LEO fleets only)*")
search_term = st.sidebar.selectbox(
    "Select constellation",
    list(GROUP_CONFIG.keys()),
    label_visibility="collapsed",
)
if st.sidebar.button("DOWNLOAD LIVE TLE DATA"):
    if user_email and user_pass:
        with st.spinner("📡 Connecting to data sources..."):
            try:
                download_limit = int(st.session_state.get("sat_limit", 15))
                result = fetch_tles_with_fallback(
                    user_email, user_pass, search_term, download_limit
                )
                if result:
                    data = result["lines"]
                    st.session_state["tle_data"] = data
                    st.session_state["loaded_group"] = GROUP_CONFIG[search_term]["label"]
                    st.session_state["data_source"] = result["source"]
                    st.session_state["data_message"] = result["message"]
                    count = count_tle_objects(data)
                    pair_count = count * (count - 1) // 2
                    st.sidebar.success(
                        f"✅ {count} satellites loaded • {pair_count} possible pairs"
                    )
                    st.sidebar.caption(
                        "🎯 Apsis filter will eliminate pairs with low physical intersection probability."
                    )
                    st.sidebar.info(f"📊 Source: {result['source']}")
                    st.sidebar.caption(result["message"])
                else:
                    st.sidebar.error("❌ Failed to download TLE data. Please try again.")
            except Exception as e:
                st.sidebar.error(f"❌ Download error: {str(e)[:100]}")
    else:
        st.sidebar.warning("⚠️ Authentication required. Please enter your credentials.")

st.sidebar.markdown(
    """<div style="height:1px; background:linear-gradient(90deg, transparent 0%, #1e2d42 50%, transparent 100%); 
                margin:20px 0;"></div>""",
    unsafe_allow_html=True,
)

# ─── SECTION 2: MANUAL TLE ENTRY ─────────────────────────────────────────────
st.sidebar.markdown(
    """<div style="font-family:'Space Mono',monospace;font-size:.7rem;
    letter-spacing:.18em;color:#00ffa8;text-transform:uppercase;
    border-bottom:1px solid #1e2d42;padding-bottom:6px;margin-bottom:12px;
    display:flex; align-items:center; gap:8px;">
    <span style="background:linear-gradient(135deg, #00ffa8 0%, #00d4ff 100%);
                 color:#05070a; padding:2px 8px; border-radius:4px; font-weight:700;">02</span>
    ENTER YOUR SATELLITE (TLE)</div>""",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    """<div style="font-family:'Inter',sans-serif; font-size:.75rem; 
                color:#5a7a94; margin-bottom:10px; font-style:italic;">
    3-line TLE format (name + line1 + line2)
  </div>""",
    unsafe_allow_html=True,
)
manual_tle_text = st.sidebar.text_area(
    "Manual TLE",
    height=110,
    placeholder="ISS (ZARYA)\n1 25544U 98067A   24065.52722916  .00016717  00000+0  32296-3 0  9994\n2 25544  51.6412  237.8783 0003724 100.6644  259.4049 15.50110392 44874",
    label_visibility="collapsed",
    key="manual_tle_input",
)
if st.sidebar.button("LOAD MANUAL TLE"):
    lines = [l.strip() for l in manual_tle_text.strip().split("\n") if l.strip()]
    if len(lines) >= 3:
        try:
            my_sat = EarthSatellite(lines[1], lines[2], lines[0], ts)
            st.session_state["my_sat"] = my_sat
            st.sidebar.success(f"✅ {my_sat.name} loaded successfully.")
        except Exception as e:
            st.sidebar.error(f"❌ TLE parsing error: {str(e)[:80]}")
    elif len(lines) == 2:
        try:
            my_sat = EarthSatellite(lines[0], lines[1], "CUSTOM-SAT", ts)
            st.session_state["my_sat"] = my_sat
            st.sidebar.success("✅ CUSTOM-SAT loaded successfully.")
        except Exception as e:
            st.sidebar.error(f"❌ TLE parsing error: {str(e)[:80]}")
    else:
        st.sidebar.warning("⚠️ Please enter at least 2 TLE lines.")

if "my_sat" in st.session_state:
    ms = st.session_state["my_sat"]
    st.sidebar.markdown(
        f"""<div style="font-family:'Space Mono',monospace;font-size:.7rem;
        color:#00ffa8;padding:10px 14px;background:rgba(0,255,168,.08);
        border:1px solid rgba(0,255,168,.25);border-radius:6px;margin-top:8px;
        box-shadow: 0 2px 8px rgba(0,255,168,0.15);">
        ✓ ACTIVE: {ms.name}</div>""",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Delete My Satellite"):
        del st.session_state["my_sat"]
        st.rerun()

st.sidebar.markdown(
    """<div style="height:1px; background:linear-gradient(90deg, transparent 0%, #1e2d42 50%, transparent 100%); 
                margin:20px 0;"></div>""",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    """<div style="font-family:'Space Mono',monospace;font-size:.7rem;
    letter-spacing:.18em;color:#5a7a94;text-transform:uppercase;
    border-bottom:1px solid #1e2d42;padding-bottom:6px;margin-bottom:12px;
    display:flex; align-items:center; gap:8px;">
    <span style="background:linear-gradient(135deg, #5a7a94 0%, #1e2d42 100%);
                 color:#c4d4e8; padding:2px 8px; border-radius:4px; font-weight:700;">03</span>
    ANALYSIS PARAMETERS</div>""",
    unsafe_allow_html=True,
)
sync_mass_defaults(search_term)
selected_group_label = GROUP_CONFIG[search_term]["label"]
selected_group_mass = get_group_default_mass(search_term)

if "my_sat" in st.session_state:
    st.sidebar.markdown(
        f"""<div style="font-family:'Space Mono',monospace;font-size:.64rem;color:#4a6880;
        padding:6px 0 8px 0;line-height:1.6;">
        Object A default: <span style="color:#00ff9d;">MANUAL SAT • {MANUAL_SAT_DEFAULT_MASS_KG} kg</span><br>
        Object B default: <span style="color:#00c8ff;">{selected_group_label} • {selected_group_mass} kg</span>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        f"""<div style="font-family:'Space Mono',monospace;font-size:.64rem;color:#4a6880;
        padding:6px 0 8px 0;line-height:1.6;">
        Selected fleet: <span style="color:#00c8ff;">{selected_group_label} • {selected_group_mass} kg</span>
        </div>""",
        unsafe_allow_html=True,
    )

window_hrs = st.sidebar.slider("Analysis window (hours)", 1, 48, 24, key="sidebar_window_hrs")
sigma_km = st.sidebar.select_slider(
    "Position uncertainty σ (km)",
    options=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    value=0.5,
)
sat_limit = st.sidebar.slider("Maximum satellite count", 5, 30, 15, key="sat_limit")
hbr_km = st.sidebar.select_slider(
    "Hard-Body Radius HBR (km)",
    options=[0.005, 0.010, 0.020, 0.050, 0.100],
    value=0.020,
)
mass_a_label = "Object A Mass (kg)"
mass_b_label = "Object B Mass (kg)"
if "my_sat" in st.session_state:
    mass_a_label = "Object A Mass (Manual Sat)"
    mass_b_label = f"Object B Mass ({selected_group_label})"

st.sidebar.markdown(
    "<small style='color:#4a6880;'>Allowed range: 1–500000 kg. Type the value directly.</small>",
    unsafe_allow_html=True,
)

st.sidebar.markdown(f"**{mass_a_label}**")
mass_a_kg = st.sidebar.number_input(
    "Write Object A Mass (kg)",
    min_value=1.0,
    max_value=MASS_WIDGET_MAX_KG,
    value=float(
        st.session_state.get(
            "mass_a_input", st.session_state.get("mass_a_kg", selected_group_mass)
        )
    ),
    step=1.0,
    format="%.3f",
    key="mass_a_input",
    on_change=sync_mass_a_from_input,
    help="Allowed range: 1-500000 kg. Example: 630 or 419725.",
)

st.sidebar.markdown(f"**{mass_b_label}**")
mass_b_kg = st.sidebar.number_input(
    "Write Object B Mass (kg)",
    min_value=1.0,
    max_value=MASS_WIDGET_MAX_KG,
    value=float(
        st.session_state.get(
            "mass_b_input", st.session_state.get("mass_b_kg", selected_group_mass)
        )
    ),
    step=1.0,
    format="%.3f",
    key="mass_b_input",
    on_change=sync_mass_b_from_input,
    help="Allowed range: 1-500000 kg. Example: 250 or 419725.",
)

mass_a_kg = float(st.session_state.get("mass_a_kg", mass_a_kg))
mass_b_kg = float(st.session_state.get("mass_b_kg", mass_b_kg))

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model:** Chan 1997 + Foster 1992
**Propagator:** SGP4/SDP4
**Filter:** Apsis + Distance
**Data:** Space-Track GP + CelesTrak fallback
**TCA Step:** 5 min
**HBR:** User selected
""")

# DATA CHECK
if "tle_data" not in st.session_state:
    st.info("📡 Download data by entering your Space-Track credentials in the left panel.")
    st.markdown(
        """
    <div class="info-panel">
      <b>🚀 Quick Start Guide:</b><br>
      1. Create a free account at <b>space-track.org</b>.<br>
      2. Enter your email and password in the left panel.<br>
      3. Select a satellite constellation and click <b>DOWNLOAD LIVE TLE DATA</b>.<br>
      4. If Space-Track fails, CelesTrak will be used automatically as backup.<br>
      5. All tabs will become active for analysis.
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    sats = parse_tles(
        st.session_state["tle_data"],
        limit=sat_limit,
        fallback_name_prefix=st.session_state.get("loaded_group"),
    )
    if not sats:
        st.error("❌ TLE parsing failed. Please check your data source and try again.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error parsing TLE data: {str(e)[:100]}")
    st.stop()

# TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "DASHBOARD",
        "CONJUNCTION ANALYSIS",
        "YOUR SATELLITE",
        "LIVE SIMULATION",
        "3D ORBIT & GROUND TRACK",
        "ORBITAL ELEMENTS",
        "METHODOLOGY",
    ]
)

# ── TAB 1: DASHBOARD ───────────────────────────────────────────────────
with tab1:
    with st.spinner("🚀 Computing conjunction analysis with apsis filter..."):
        start_time = time.time()
        df, n_filtered, n_total = compute_conjunctions(
            sats, window_hrs, sigma_km, hbr_km, mass_a_kg, mass_b_kg, st.session_state.theme
        )
        computation_time = time.time() - start_time

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(
        f"""<div style="font-family:'Space Mono',monospace; font-size:.7rem;
         color:#5a7a94; text-align:right; margin-bottom:16px; padding:8px 12px; 
         background:rgba(90,122,148,.05); border-radius:6px; border:1px solid rgba(90,122,148,.15);">
         Last update: {now_str} · Computation time: {computation_time:.2f}s</div>""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🛰️ Tracked Satellites", len(sats))
    with c2:
        st.metric("🔗 Total Pairs", n_total)
    with c3:
        st.metric("✅ Passed Filter", n_total - n_filtered)
    with c4:
        n_conj = len(df) if not df.empty else 0
        st.metric("⚠️ Conjunctions", n_conj)
    with c5:
        n_crit = len(df[df["Risk Level"] == "CRITICAL"]) if not df.empty else 0
        st.metric("🚨 Critical Risk", n_crit)

    if n_filtered > 0:
        st.markdown(
            f"""<div class="info-panel">
        <b>🎯 Apsis Filter Performance:</b> {n_filtered} pairs filtered without orbit propagation
        due to non-overlapping altitude bands — computation time reduced by
        <span style="color:#00ffa8; font-weight:700;">{round(n_filtered / n_total * 100, 1)}%</span>.
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if df.empty:
        st.success(
            f"✅ No conjunctions below 500 km detected in {window_hrs}-hour window. All clear!"
        )
    else:
        # Dilution warning
        n_dil = df["Dilution"].sum() if not df.empty else 0
        if n_dil > 0:
            st.markdown(
                f"""<div class="warn-panel">
            <b>PROBABILITY DILUTION WARNING:</b> Wide covariance in {int(n_dil)} events
            may be masking Pc values. Check Max-Pc values in Conjunction Analysis tab.
            </div>""",
                unsafe_allow_html=True,
            )

        show_cols = [
            "TCA (UTC)",
            "Object A",
            "Object B",
            "Distance (km)",
            "Relative Velocity (km/s)",
            "Pc (scientific)",
            "Pc Max",
            "Mahalanobis Md",
            "Ec (J/g)",
            "Risk Level",
        ]
        RISK_COLORS = {
            "CRITICAL": "#ff2b4d",
            "HIGH": "#ff6b00",
            "MEDIUM": "#ffaa00",
            "LOW": "#00ff9d",
        }
        MONO = "font-family:'Space Mono',monospace; font-size:0.76rem;"

        df_show = df[show_cols].copy()
        styled = (
            df_show.style.map(
                lambda v: (
                    f"color:{RISK_COLORS.get(str(v), '#b8cfe0')};font-weight:bold;{MONO}"
                ),
                subset=["Risk Level"],
            )
            .map(lambda v: f"color:#00c8ff;{MONO}", subset=["Pc (scientific)"])
            .map(lambda v: f"color:#ff9060;{MONO}", subset=["Pc Max"])
            .map(
                lambda v: (
                    f"color:#ff2b4d;{MONO}"
                    if float(v) < 1.5
                    else f"color:#b8cfe0;{MONO}"
                ),
                subset=["Mahalanobis Md"],
            )
            .map(
                lambda v: (
                    f"color:#ff2b4d;{MONO}"
                    if float(v) >= 40
                    else f"color:#b8cfe0;{MONO}"
                ),
                subset=["Ec (J/g)"],
            )
            .format(
                {
                    "Distance (km)": "{:.3f}",
                    "Relative Velocity (km/s)": "{:.3f}",
                    "Pc Max": "{:.3e}",
                    "Mahalanobis Md": "{:.2f}",
                    "Ec (J/g)": "{:.1f}",
                }
            )
            .set_properties(
                **{"font-family": "Space Mono,monospace", "font-size": "0.76rem"}
            )
        )
        csv_bytes = df_show.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Download Report as CSV",
            data=csv_bytes,
            file_name=f"conjunction_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
        st.dataframe(styled, use_container_width=True)

# ── TAB 2: CONJUNCTION ANALYSIS ─────────────────────────────────────────────────
with tab2:
    if df is None or df.empty:
        st.success("✅ No critical conjunction events in selected window.")
    else:
        st.markdown("**🔍 Detailed Review — Select Event**")
        options = [
            f"{r['Object A']}  <->  {r['Object B']}  |  TCA {r['TCA (UTC)']}  |  {r['Distance (km)']} km"
            for _, r in df.iterrows()
        ]
        sel = st.selectbox("Conjunction event", options, label_visibility="collapsed")
        idx = options.index(sel)
        row = df.iloc[idx]

        # Dilution warning
        if row["Dilution"]:
            st.markdown(
                f"""<div class="crit-panel">
            <b>⚠️ PROBABILITY DILUTION:</b> {row["Dilution Message"]}
            </div>""",
                unsafe_allow_html=True,
            )

        # Plot + gauge
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.plotly_chart(
                fig_distance_profile(
                    row["_dist_arr"], window_hrs, row["Distance (km)"], sigma_km, hbr_km
                ),
                use_container_width=True,
                key="dist_prof_tab2",
            )
        with col_r:
            st.plotly_chart(
                fig_risk_gauge(row["Pc (isotropic)"]),
                use_container_width=True,
                key="risk_gauge_tab2",
            )

        # Pc comparison
        st.markdown("**📊 Collision Probability Model Comparison**")
        pc_cols = st.columns(3)
        with pc_cols[0]:
            st.metric("Chan 1997 (Isotropic)", f"{row['Pc (isotropic)']:.3e}")
        with pc_cols[1]:
            st.metric("Foster 1992 (2D-Pc)", f"{row['Pc (Foster 2D)']:.3e}")
        with pc_cols[2]:
            st.metric("Max Pc (Worst Case)", f"{row['Pc Max']:.3e}")

        # Mahalanobis test
        theme = st.session_state.theme
        if row["2D-Pc Valid"] != "2D-Pc Valid":
            mah_color = "#dc3545" if theme == "light" else "#ff3d5c"
        else:
            mah_color = "#28a745" if theme == "light" else "#00ffa8"
        st.markdown(
            f"""<div class="info-panel">
        <b>🎯 Mahalanobis Distance Test:</b> Md = {row["Mahalanobis Md"]:.3f} —
        <span style="color:{mah_color}; font-weight:700;">{row["2D-Pc Valid"]}</span><br>
        <small>Md < 1.5 → linear motion assumption breaks down → 3D-Pc required (CARA methodology)</small>
        </div>""",
            unsafe_allow_html=True,
        )

        # Fragmentation analysis
        frag = fragmentation_probability(
            row["Relative Velocity (km/s)"], mass_a_kg, mass_b_kg
        )
        st.markdown("**Collision Consequence Analysis**")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.metric("Specific Kinetic Energy (J/g)", f"{frag['E_c_J_per_g']:.1f}")
        with fc2:
            st.metric("Fragmentation Level", frag["level"])
        with fc3:
            st.metric("Estimated Debris Objects", frag["est_debris"])
        st.markdown(
            f"""<div class="info-panel" style="border-left-color:{frag["color"]};">
        <b>{frag["level"]}:</b> {frag["desc"]}<br>
        <small>Ec ≥ 40 J/g → Catastrophic fragmentation (Kessler Syndrome contribution)</small>
        </div>""",
            unsafe_allow_html=True,
        )

        # Full parameter table
        st.markdown("**Full Event Parameters**")
        det = {
            "Object A": row["Object A"],
            "Object B": row["Object B"],
            "TCA (UTC)": row["TCA (UTC)"],
            "Miss Distance (km)": row["Distance (km)"],
            "Relative Velocity (km/s)": row["Relative Velocity (km/s)"],
            "Position Uncertainty sigma (km)": sigma_km,
            "Hard-Body Radius HBR (km)": hbr_km,
            "Pc — Chan 1997 Isotropic": f"{row['Pc (isotropic)']:.3e}",
            "Pc — Foster 1992 2D": f"{row['Pc (Foster 2D)']:.3e}",
            "Pc — Maximum (Worst Case)": f"{row['Pc Max']:.3e}",
            "Mahalanobis Distance Md": row["Mahalanobis Md"],
            "2D-Pc Validity": row["2D-Pc Valid"],
            "Probability Dilution": "YES" if row["Dilution"] else "NO",
            "Specific Kinetic Energy (J/g)": row["Ec (J/g)"],
            "Fragmentation Level": row["Fragmentation Level"],
            "Estimated Debris Objects": row["Estimated Debris"],
            "Risk Level (NASA STD-8719.14)": row["Risk Level"],
        }
        df_det = pd.DataFrame(det.items(), columns=["Parameter", "Value"])
        st.dataframe(df_det, use_container_width=True, hide_index=True)

        st.markdown("---")
        if st.button("🔭 Show This Pair in Live Simulation", key="tab2_to_sim"):
            queue_simulation_pair(row["_s1"], row["_s2"], row["_tca_tt"])
            st.success(
                "Pair transferred to 'LIVE SIMULATION' tab with TCA-centered timing."
            )

# ── TAB 3: YOUR SATELLITE ───────────────────────────────────────────────────────
with tab3:
    st.markdown("## Analyze Your Satellite")
    if "my_sat" not in st.session_state:
        st.markdown(
            """<div class="warn-panel">
        <b>You haven't loaded your satellite yet.</b><br>
        Enter your TLE data in the <b>2 — ENTER YOUR SATELLITE (TLE)</b> section
        in the left panel and click <b>LOAD MANUAL TLE</b>.
        </div>""",
            unsafe_allow_html=True,
        )
    elif "tle_data" not in st.session_state:
        st.markdown(
            """<div class="warn-panel">
        <b>Fleet data not loaded.</b><br>
        First perform automatic TLE download from the left panel; then comparison
        with your satellite can be done.
        </div>""",
            unsafe_allow_html=True,
        )
    else:
        my_sat = st.session_state["my_sat"]
        st.markdown(
            f"""<div class="info-panel">
        <b>Active satellite:</b> {my_sat.name}&nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Fleet:</b> {st.session_state.get("loaded_group", "—")}&nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Manual mass A:</b> {mass_a_kg} kg&nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Fleet mass B:</b> {mass_b_kg} kg&nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Analysis window:</b> {window_hrs} hours&nbsp;&nbsp;|&nbsp;&nbsp;
        <b>σ:</b> {sigma_km} km
        </div>""",
            unsafe_allow_html=True,
        )

        with st.spinner(f"Running conjunction analysis for {my_sat.name}..."):
            df_my = compute_conjunctions_custom(
                my_sat, sats, window_hrs, sigma_km, hbr_km, mass_a_kg, mass_b_kg, st.session_state.theme
            )

        if df_my.empty:
            st.success(
                f"No conjunctions below 500 km for {my_sat.name} in {window_hrs}-hour window."
            )
        else:
            n_crit_my = len(df_my[df_my["Risk Level"] == "CRITICAL"])
            n_high_my = len(df_my[df_my["Risk Level"] == "HIGH"])

            c1m, c2m, c3m, c4m = st.columns(4)
            with c1m:
                st.metric("Total Conjunctions", len(df_my))
            with c2m:
                st.metric("Critical Risk", n_crit_my)
            with c3m:
                st.metric("High Risk", n_high_my)
            with c4m:
                st.metric("Min. Distance (km)", f"{df_my['Distance (km)'].min():.2f}")

            st.markdown("**Conjunctions — Risk Table**")
            RISK_COLORS = {
                "CRITICAL": "#ff2b4d",
                "HIGH": "#ff6b00",
                "MEDIUM": "#ffaa00",
                "LOW": "#00ff9d",
            }
            MONO = "font-family:'Space Mono',monospace; font-size:0.76rem;"
            show_c = [
                "TCA (UTC)",
                "Object A",
                "Object B",
                "Distance (km)",
                "Relative Velocity (km/s)",
                "Pc (scientific)",
                "Pc Max",
                "Mahalanobis Md",
                "Ec (J/g)",
                "Risk Level",
            ]
            df_my_show = df_my[show_c].copy()
            styled_my = (
                df_my_show.style.map(
                    lambda v: (
                        f"color:{RISK_COLORS.get(str(v), '#b8cfe0')};font-weight:bold;{MONO}"
                    ),
                    subset=["Risk Level"],
                )
                .map(lambda v: f"color:#00c8ff;{MONO}", subset=["Pc (scientific)"])
                .map(lambda v: f"color:#ff9060;{MONO}", subset=["Pc Max"])
                .map(
                    lambda v: (
                        f"color:#ff2b4d;{MONO}"
                        if float(v) < 1.5
                        else f"color:#b8cfe0;{MONO}"
                    ),
                    subset=["Mahalanobis Md"],
                )
                .format(
                    {
                        "Distance (km)": "{:.3f}",
                        "Relative Velocity (km/s)": "{:.3f}",
                        "Pc Max": "{:.3e}",
                        "Mahalanobis Md": "{:.2f}",
                        "Ec (J/g)": "{:.1f}",
                    }
                )
                .set_properties(
                    **{"font-family": "Space Mono,monospace", "font-size": "0.76rem"}
                )
            )
            csv_my = df_my_show.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download Report as CSV",
                data=csv_my,
                file_name=f"my_satellite_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
            st.dataframe(styled_my, use_container_width=True)

            # Detailed analysis for selected pair
            st.markdown("---")
            st.markdown("**Detailed Pair Analysis — Select Event**")
            opts_my = [
                f"{r['Object B']}  |  TCA {r['TCA (UTC)']}  |  {r['Distance (km)']} km"
                for _, r in df_my.iterrows()
            ]
            sel_my = st.selectbox(
                "Select event", opts_my, label_visibility="collapsed", key="my_sel"
            )
            idx_my = opts_my.index(sel_my)
            row_my = df_my.iloc[idx_my]

            if row_my["Dilution"]:
                st.markdown(
                    f"""<div class="crit-panel">
                <b>PROBABILITY DILUTION:</b> {row_my["Dilution Message"]}</div>""",
                    unsafe_allow_html=True,
                )

            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.plotly_chart(
                    fig_distance_profile(
                        row_my["_dist_arr"],
                        window_hrs,
                        row_my["Distance (km)"],
                        sigma_km,
                        hbr_km,
                    ),
                    use_container_width=True,
                    key="dist_prof_tab3",
                )
            with col_r:
                st.plotly_chart(
                    fig_risk_gauge(row_my["Pc (isotropic)"]),
                    use_container_width=True,
                    key="risk_gauge_tab3",
                )

            pc_c = st.columns(3)
            with pc_c[0]:
                st.metric("Chan 1997 (Isotropic)", f"{row_my['Pc (isotropic)']:.3e}")
            with pc_c[1]:
                st.metric("Foster 1992 (2D-Pc)", f"{row_my['Pc (Foster 2D)']:.3e}")
            with pc_c[2]:
                st.metric("Max Pc", f"{row_my['Pc Max']:.3e}")

            theme = st.session_state.theme
            if row_my["2D-Pc Valid"] != "2D-Pc Valid":
                mah_c = "#dc3545" if theme == "light" else "#ff2b4d"
            else:
                mah_c = "#28a745" if theme == "light" else "#00ff9d"
            st.markdown(
                f"""<div class="info-panel">
            <b>Mahalanobis Test:</b> Md = {row_my["Mahalanobis Md"]:.3f} —
            <span style="color:{mah_c};">{row_my["2D-Pc Valid"]}</span>
            </div>""",
                unsafe_allow_html=True,
            )

            frag_my = fragmentation_probability(
                row_my["Relative Velocity (km/s)"], mass_a_kg, mass_b_kg
            )
            fc = st.columns(3)
            with fc[0]:
                st.metric("Ec (J/g)", f"{frag_my['E_c_J_per_g']:.1f}")
            with fc[1]:
                st.metric("Fragmentation", frag_my["level"])
            with fc[2]:
                st.metric("Estimated Debris", frag_my["est_debris"])

            # Send to simulation button
            st.markdown("---")
            if st.button("🔭 Show This Pair in Live Simulation", key="my_to_sim"):
                queue_simulation_pair(row_my["_s1"], row_my["_s2"], row_my["_tca_tt"])
                st.success(
                    "Pair transferred to 'LIVE SIMULATION' tab with TCA-centered timing."
                )


# --------------------------------------------------------------
#  TAB 4: LIVE SIMULATION  (PROFESSIONAL YENİDEN TASARIM)
# --------------------------------------------------------------
with tab4:
    st.markdown("## 🚀 Live 3D Orbit Simulation")
    st.markdown(
        """
        <div class="info-panel">
            <b>🔬 Professional Conjunction Visualization</b><br>
            Watch the encounter between two satellites with <b>real‑time</b> animation.
            Focus on the risk moment with Play / Stop / Speed controls and <b>Jump to TCA</b> button.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    #  Session‑state başlatıcıları (eğer daha önce tanımlanmadıysa)
    # ------------------------------------------------------------------
    if "run_sim" not in st.session_state:
        st.session_state.run_sim = False
    if "sim_sat_a" not in st.session_state:
        st.session_state.sim_sat_a = None
    if "sim_sat_b" not in st.session_state:
        st.session_state.sim_sat_b = None
    if "sim_center_tt" not in st.session_state:
        st.session_state.sim_center_tt = None
    if "window_hrs" not in st.session_state:
        st.session_state.window_hrs = 24
    if "sel_a" not in st.session_state:
        st.session_state.sel_a = None
    if "sel_b" not in st.session_state:
        st.session_state.sel_b = None

    # ------------------------------------------------------------------
    #  Uydu seçimi (sidebar yerine burada tutarlı bir UI)
    # ------------------------------------------------------------------
    sat_names = [s.name for s in sats]
    if "my_sat" in st.session_state:
        sat_names_ext = [st.session_state["my_sat"].name] + sat_names
        all_sats_ext = [st.session_state["my_sat"]] + sats
    else:
        sat_names_ext = sat_names
        all_sats_ext = sats

    # Önceki seçimleri tutalım (sayfa yenilenince sıfırlanmasın)
    default_a = st.session_state.get("sel_a")
    if default_a and default_a in sat_names_ext:
        # Valid existing selection
        pass
    else:
        default_a = sat_names_ext[0] if sat_names_ext else ""

    default_b = st.session_state.get("sel_b")
    if default_b and default_b in sat_names_ext:
        # Valid existing selection
        pass
    else:
        default_b = sat_names_ext[1] if len(sat_names_ext) > 1 else sat_names_ext[0] if sat_names_ext else ""

    # ------------------------------------------------------------------
    #  Profesyonel Kontrol Paneli
    # ------------------------------------------------------------------
    st.markdown(
        """
        <div style="font-family:'Space Mono',monospace;font-size:.7rem;
        letter-spacing:.18em;color:#00d4ff;text-transform:uppercase;
        border-bottom:1px solid #1e2d42;padding-bottom:6px;margin-bottom:12px;
        display:flex; align-items:center; gap:8px;">
        <span style="background:linear-gradient(135deg, #00d4ff 0%, #00ffa8 100%);
                     color:#05070a; padding:2px 8px; border-radius:4px; font-weight:700;">🎮</span>
        SIMULATION CONTROLS</div>""",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            sel_a = st.selectbox(
                "🛰️ Satellite A",
                sat_names_ext,
                index=sat_names_ext.index(default_a) if default_a in sat_names_ext else 0,
                key="live_sel_a",
            )
        with c2:
            sel_b = st.selectbox(
                "🛰️ Satellite B",
                sat_names_ext,
                index=sat_names_ext.index(default_b)
                if default_b in sat_names_ext
                else min(1, len(sat_names_ext) - 1),
                key="live_sel_b",
            )
        with c3:
            sim_hrs = st.slider(
                "⏱️ Window (hours)",
                1,
                48,
                min(st.session_state.get("window_hrs", window_hrs), 48),
                key="live_window_hrs",
            )
            # Ensure sim_hrs is a valid integer
            if sim_hrs is None:
                sim_hrs = window_hrs

        # Görünüm seçenekleri (expander içinde)
        with st.expander("⚙️ Advanced Display Options", expanded=False):
            opt1, opt2 = st.columns(2)
            with opt1:
                show_orbits = st.checkbox(
                    "🔵 Show Orbit Trails",
                    value=True,
                    key="live_show_orbits",
                )
            with opt2:
                show_tca = st.checkbox(
                    "🎯 Show TCA Marker",
                    value=True,
                    key="live_show_tca",
                )
            # Hız kontrolü (opsiyonel)
            anim_speed = st.slider(
                "🎬 Playback Speed",
                0.5,
                2.0,
                1.0,
                0.1,
                key="live_anim_speed",
                help="1.0 = normal speed, 0.5 = two‑times slower, 2.0 = two‑times faster",
            )

    # ------------------------------------------------------------------
    #  Profesyonel Başlat / Durdur Butonları
    # ------------------------------------------------------------------
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        start_disabled = st.session_state.run_sim or (sel_a == sel_b)
        if st.button(
            "🚀 START SIMULATION",
            disabled=start_disabled,
            key="live_start_btn",
            use_container_width=True,
        ):
            if sel_a == sel_b:
                st.warning("⚠️ Please select two different satellites.")
            else:
                sat_obj_a = next(s for s in all_sats_ext if s.name == sel_a)
                sat_obj_b = next(s for s in all_sats_ext if s.name == sel_b)
                # Simülasyon kuyruğunu doldur
                queue_simulation_pair(sat_obj_a, sat_obj_b, None)
                st.session_state.run_sim = True
                st.session_state.sel_a = sel_a
                st.session_state.sel_b = sel_b
                # Ensure sim_hrs is valid before assignment
                try:
                    st.session_state.window_hrs = int(sim_hrs) if sim_hrs is not None else 24
                except Exception:
                    st.session_state.window_hrs = 24
                st.rerun()  # anlık UI güncellemesi

    with btn_col2:
        if st.button(
            "⏹️ STOP SIMULATION",
            disabled=not st.session_state.run_sim,
            key="live_stop_btn",
            use_container_width=True,
        ):
            st.session_state.run_sim = False
            st.rerun()

    with btn_col3:
        st.caption(
            "💡 **Tip:** Use **STOP** to pause the animation, then drag the slider or rotate the view manually."
        )

    # ------------------------------------------------------------------
    #  Simülasyon çalışıyorsa animasyonu göster
    # ------------------------------------------------------------------
    if st.session_state.get("run_sim", False):
        # Seçili uydu nesnelerini al (session_state'dan)
        sa = st.session_state.get("sim_sat_a")
        sb = st.session_state.get("sim_sat_b")

        # Guard: if satellites are missing despite run_sim=True, reset gracefully
        if sa is None or sb is None:
            st.session_state.run_sim = False
            st.warning("⚠️ Simulation state lost — please press START again.")
            st.rerun()

        center_tt = st.session_state.get("sim_center_tt")
        # Ensure window_hrs is valid
        try:
            window_hrs = int(st.session_state.get("window_hrs", sim_hrs))
        except Exception:
            window_hrs = int(sim_hrs) if sim_hrs is not None else 24

        # Compute frame duration from speed slider BEFORE building the figure
        frame_duration_ms = max(20, int(60 / anim_speed))  # 60 ms base at 1x speed

        with st.spinner("Preparing animation …"):
            # Use the main fig_animated_conjunction function which is already cached
            anim_fig, tca_i, tca_d, dists_arr, jd_arr, tca_tt = fig_animated_conjunction(
                sa,
                sb,
                window_hrs=window_hrs,
                show_orbits=show_orbits,
                show_tca=show_tca,
                center_tt=center_tt,
                frame_duration=frame_duration_ms,
            )

        # --------------------------------------------------------------
        #  TCA özet metrikleri
        # --------------------------------------------------------------
        if ts is not None:
            tca_utc = ts.tt_jd(tca_tt).utc_strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            tca_utc = "Unknown (timescale error)"
        sev_sim, col_sim = risk_level(
            collision_probability_isotropic(tca_d, sigma_km, hbr_km)
        )
        tca_tplus_min = int(round((tca_tt - jd_arr[0]) * 1440))

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("TCA Time (UTC)", tca_utc)
        with m2:
            st.metric("Min. Distance (km)", f"{tca_d:.3f}")
        with m3:
            st.metric("TCA T+ (min)", tca_tplus_min)
        with m4:
            st.metric("Risk", sev_sim)

        # --------------------------------------------------------------
        #  3D animasyon
        # --------------------------------------------------------------
        st.info(
            "💡 **Camera Control:** Camera rotation is only available when the animation is paused. Use **STOP** or the slider to pause, then rotate the view manually."
        )
        # Plotly’nin `frame.duration` parametresi milisaniye olduğundan:
        st.plotly_chart(
            anim_fig,
            use_container_width=True,
            key="live_anim_3d",
            config={"scrollZoom": False, "displayModeBar": False},
        )

        # --------------------------------------------------------------
        #  Distance profile (statistik)
        # --------------------------------------------------------------
        st.markdown("**📊 Distance Profile (Full Window)**")
        t_ax = (jd_arr - jd_arr[0]) * 24.0
        fig_dp_sim = go.Figure()
        fig_dp_sim.add_hline(
            y=hbr_km,
            line=dict(color="#ff2b4d", dash="dot", width=1),
            annotation_text=f"HBR ({hbr_km * 1000:.0f} m)",
        )
        fig_dp_sim.add_trace(
            go.Scatter(
                x=t_ax,
                y=dists_arr,
                mode="lines",
                line=dict(color="#00c8ff", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(0,200,255,.04)",
                name="Distance (km)",
            )
        )
        dists_profile = np.asarray(dists_arr, dtype=float)
        if len(dists_profile) and not np.all(np.isnan(dists_profile)):
            tca_profile_idx = int(np.nanargmin(dists_profile))
            fig_dp_sim.add_trace(
                go.Scatter(
                    x=[t_ax[tca_profile_idx]],
                    y=[dists_profile[tca_profile_idx]],
                    mode="markers+text",
                    marker=dict(color="#ff2b4d", size=10),
                    text=[f" TCA {dists_profile[tca_profile_idx]:.1f} km"],
                    textfont=dict(size=9, color="#ff2b4d", family="Space Mono"),
                    name="TCA",
                )
            )
        fig_dp_sim.update_layout(
            **DARK,
            height=240,
            xaxis=dict(title="Time (hours)", gridcolor="#1a2740", zeroline=False),
            yaxis=dict(title="Distance (km)", gridcolor="#1a2740", zeroline=False),
            title=dict(
                text=f"Distance Profile — {sa.name} × {sb.name}",
                font=dict(size=11, family="Barlow Condensed", color="#00c8ff"),
                x=0.01,
            ),
            margin=dict(l=10, r=10, t=35, b=10),
            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(
            fig_dp_sim,
            use_container_width=True,
            key="live_dist_profile",
            config={"scrollZoom": False, "displayModeBar": False},
        )


# ── TAB 5: 3D ORBIT & GROUND TRACK ───────────────────────────────────────────
with tab5:
    # Include user-defined satellite (if available) in visualization list
    display_sats = sats.copy()
    if "my_sat" in st.session_state:
        # Insert at beginning to ensure first color (light blue) and prominence in plot
        display_sats.insert(0, st.session_state["my_sat"])

    c1_3d, c2_3d = st.columns([3, 2])
    with c1_3d:
        st.markdown("**3D Orbit View**")
        with st.spinner("Loading Earth texture..."):
            st.plotly_chart(
                fig_3d_orbits(display_sats),
                use_container_width=True,
                height=560,
                key="3d_orbit_tab5",
            )
    with c2_3d:
        st.markdown("**Ground Track Map**")
        with st.spinner("Calculating..."):
            st.plotly_chart(
                fig_ground_tracks(display_sats),
                use_container_width=True,
                key="ground_track_tab5",
            )
        st.markdown(
            """<div style="font-family:'Space Mono',monospace; font-size:.65rem;
             color:#2a4060; line-height:2; margin-top:8px;">
          Approximately 95-minute track shown for each satellite.<br>
          Large dots represent current position.<br>
          Ground track calculated with SGP4/SDP4 propagator.
        </div>""",
            unsafe_allow_html=True,
        )

# ── TAB 6: ORBITAL ELEMENTS ────────────────────────────────────────────────
with tab6:
    st.markdown("## Orbital Elements and Space Distribution")

    # Include user-defined satellite in radar and table lists as well
    display_sats = sats.copy()
    if "my_sat" in st.session_state:
        display_sats.insert(0, st.session_state["my_sat"])

    elems_list = [(sat.name, get_orbital_elements(sat)) for sat in display_sats]

    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.markdown("**Kepler Orbital Elements Table**")
        rows = []
        for name, elems in elems_list:
            if elems:
                rows.append(
                    {
                        "Satellite": name[:18],
                        "Altitude (km)": elems.get("Mean Altitude (km)", "-"),
                        "Inclination (°)": elems.get("Inclination i (°)", "-"),
                        "Eccentricity": elems.get("Eccentricity e", "-"),
                        "Period (min)": elems.get("Orbital Period (min)", "-"),
                    }
                )
            if rows:
                df_elems = pd.DataFrame(rows)
                st.dataframe(df_elems, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Altitude / Inclination Distribution** (dot size = eccentricity)")
        st.plotly_chart(
            fig_orbital_elements_radar(elems_list),
            use_container_width=True,
            key="radar_tab6",
        )

    with st.expander("Selected Satellite Detail"):
        sel_sat = st.selectbox(
            "Select satellite", [s.name for s in display_sats], key="elem_sel"
        )
        sel_elems = next((e for n, e in elems_list if n == sel_sat), {})
        if sel_elems:
            df_single = pd.DataFrame(sel_elems.items(), columns=["Element", "Value"])
            st.dataframe(df_single, use_container_width=True, hide_index=True)

# ── TAB 7: METHODOLOGY ────────────────────────────────────────────────────────
with tab7:
    st.markdown("## Methodology and Theoretical Background")
    st.markdown(
        """
    <div class="info-panel">
    <b>1. Orbit Propagation — SGP4/SDP4 (Skyfield)</b><br>
    The NORAD standard <b>Simplified General Perturbations-4 (SGP4)</b> model is used
    to convert TLE (Two-Line Element) data into position vectors. SGP4 approximately
    accounts for gravity harmonics, atmospheric drag, and Sun/Moon third-body effects
    with a centered force model. SGP4 is used for low-orbit (<2000 km) objects;
    SDP4 automatically engages for high-orbit objects.<br><br>
    <b>Performance note (Thesis Section 1):</b> Pure Python/Skyfield produces ~1M steps/sec,
    while Rust/Zig-based <b>Astrora</b> (with SIMD) reaches 4.8–15M, and SatKit (PyO3/Rust) ~3.4M.
    Transition to these libraries is recommended for large-scale operational simulations.
    </div>

    <div class="info-panel">
    <b>2. Apsis Filter — Section 2.1 (ESA/NASA standard)</b><br>
    Before analysis begins, all satellite pairs pass through the <b>Apsis (Apogee-Perigee) Filter</b>.
    If the first object's <i>perigee altitude q₁</i> is higher than the second object's <i>apogee altitude Q₂</i>,
    these two orbits can never intersect in space. Mathematical condition:<br>
    &nbsp;&nbsp;<code>max(q₁, q₂) > min(Q₁, Q₂) + D_th</code><br>
    Applying this filter dramatically reduces O(N²) computational load by eliminating all
    pairs with non-overlapping altitude bands.
    </div>

    <div class="info-panel">
    <b>3. TCA Detection — 5-Minute Step Coarse Scan</b><br>
    For pairs passing the apsis filter, Euclidean distance is calculated throughout the
    analysis window with <b>5-minute fixed time steps</b>. The moment of minimum distance
    is identified as <b>TCA (Time of Closest Approach)</b>. Local minimization with Brent's
    method can be applied for more precise TCA.
    </div>

    <div class="info-panel">
    <b>4. Collision Probability — Two Models</b><br>
    <b>4a. Chan (1997) Isotropic Model:</b> Simplified model assuming position uncertainty
    is equally (spherically) distributed in all directions. Provides fast results but
    does not reflect real asymmetric covariance. Formula: normal CDF-based closed-form approximation.<br><br>
    <b>4b. Foster &amp; Estes (1992) 2D-Pc:</b> Industry standard used since NASA Space Shuttle era.
    Collision integration is reduced to two dimensions by projecting onto the
    <b>encounter plane</b>. Combined covariance matrix (Σ = Cₐ + C_b) is created and
    the 2D integral of Gaussian distribution over HBR circle is calculated:<br>
    &nbsp;&nbsp;<code>Pc = 1/(2π√detΣ) ∬_HBR exp(-½ rᵀΣ⁻¹r) dx dy</code>
    </div>

    <div class="info-panel">
    <b>5. Mahalanobis Distance Test — Section 3.2 (CARA Methodology)</b><br>
    2D-Pc's <i>"short-duration encounter"</i> and <i>"linear motion"</i> assumptions
    break down when objects approach at low relative velocities. According to CARA methodology,
    <b>Mahalanobis distance</b> (Md = miss / σ) tests this validity:<br>
    &nbsp;&nbsp;Md &lt; 0.5 → 2D-Pc <span style="color:#ff2b4d;">INVALID</span> — 3D-Pc / Monte Carlo required<br>
    &nbsp;&nbsp;Md &lt; 1.5 → 2D-Pc <span style="color:#ffaa00;">BORDERLINE</span> — 3D-Pc recommended<br>
    &nbsp;&nbsp;Md ≥ 1.5 → 2D-Pc <span style="color:#00ff9d;">VALID</span>
    </div>

    <div class="info-panel">
    <b>6. Probability Dilution — Section 4</b><br>
    With large position uncertainty (wide covariance), the Gaussian distribution spreads
    so much in space that the density falling within the HBR circle approaches zero —
    Pc mathematically decreases. This <b>"false confidence"</b> problem can make a
    genuinely dangerous close approach appear safe.<br><br>
    Solution tools: <b>WSPRT</b> (Wald Sequential Probability Ratio Test) — compares
    instantaneous risk with background risk; <b>Max-Pc Analysis</b> — iteratively varies
    covariance magnitude to find mathematical maximum Pc for that geometry.
    </div>

    <div class="info-panel">
    <b>7. Risk Classification — NASA STD-8719.14</b><br>
    &nbsp;&nbsp;• <span style="color:#ff2b4d;">Pc &gt; 1×10⁻³ → CRITICAL</span> — Collision Avoidance Maneuver (CAM) mandatory<br>
    &nbsp;&nbsp;• <span style="color:#ff6b00;">Pc &gt; 1×10⁻⁴ → HIGH</span> — CAM evaluation required<br>
    &nbsp;&nbsp;• <span style="color:#ffaa00;">Pc &gt; 1×10⁻⁵ → MEDIUM</span> — Increased tracking frequency<br>
    &nbsp;&nbsp;• <span style="color:#00ff9d;">Pc ≤ 1×10⁻⁵ → LOW</span> — Routine tracking sufficient
    </div>

    <div class="info-panel">
    <b>8. Collision Consequence — Fragmentation Probability Pf (Section 4)</b><br>
    Not only collision probability, but also the magnitude of potential disaster should
    be included in risk calculation. <b>Specific Kinetic Energy:</b> Ec = ½ · m_b · v_rel² / m_a (J/g)<br>
    &nbsp;&nbsp;Ec ≥ 40 J/g → Catastrophic fragmentation — <b>Kessler Syndrome</b> contribution<br>
    &nbsp;&nbsp;Ec ≥ 10 J/g → Severe damage and significant debris cloud<br>
    &nbsp;&nbsp;Ec ≥ 1 J/g  → Partial damage<br>
    Kessler Syndrome: Chain reaction where collisions trigger new collisions.
    </div>

    <div class="info-panel">
    <b>9. Data Source — Space-Track GP Class</b><br>
    TLE data is pulled from the <b>GP (General Perturbations)</b> endpoint on space-track.org,
    operated by the <b>18th Space Defense Squadron (US Space Force)</b>. Transition to
    JSON/CSV-based <b>OMM (Orbit Mean-Elements Message)</b> format instead of legacy TLE/2LE
    format is recommended. Bulk NORAD ID queries and randomly scheduled update cycles should
    be used to avoid API rate limit violations.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### References")
    st.markdown(
        """<div style="font-family:'Space Mono',monospace; font-size:.7rem;
         color:#4a6880; line-height:2.4;">
    Foster, J.L. &amp; Estes, H.S. (1992). A parametric analysis of orbital debris collision probability
    and maneuver rate for space vehicles. <i>NASA Technical Memorandum.</i><br>
    Chan, F.K. (1997). <i>Spacecraft Collision Probability.</i> The Aerospace Press.<br>
    Hoots, F.R. &amp; Roehrich, R.L. (1980). <i>Models for Propagation of NORAD Element Sets.</i>
    Spacetrack Reporst No. 3.<br>
    NASA (2023). <i>Spacecraft Conjunction Assessment and Collision Avoidance Best Practices Handbook.</i>
    CARA Handbook Rev. 1.<br>
    NASA (2011). <i>Process for Limiting Orbital Debris.</i> NASA-STD-8719.14A.<br>
    Alfriend, K.T. &amp; Akella, M.R. (2000). Probability of Collision Between Space Objects.
    <i>J. Guidance, Control, and Dynamics</i>, 23(5), 769–772.<br>
    ESA (2011). Efficient All vs. All Collision Risk Analyses — Smart Sieve Algorithm.
    <i>ISSFD Proceedings.</i><br>
    Vallado, D.A. (2013). <i>Fundamentals of Astrodynamics and Applications.</i> 4th ed. Microcosm Press.<br>
    Hall, D.T. et al. (2023). A Multistep Probability of Collision Computational Algorithm.
    <i>NASA NTRS.</i>
    </div>""",
        unsafe_allow_html=True,
    )
