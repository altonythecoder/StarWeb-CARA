# StarWeb-CARA: Conjunction Assessment and Collision Risk Analysis
# Altay ÇAVUŞ — Space Sciences and Technologies, 2026

import math
import time
from datetime import datetime, timezone
from io import BytesIO
from itertools import combinations

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
try:
    ts = load.timescale()
except Exception as e:
    print(f"Skyfield timescale initialization failed: {e}")
    ts = None

def queue_simulation_pair(sat_a, sat_b, center_tt=None):
    if sat_a is None or sat_b is None:
        print("Cannot queue a simulation – one of the satellites is None.")
        return
    try:
        st.session_state["sim_sat_a"] = sat_a
        st.session_state["sim_sat_b"] = sat_b
        st.session_state["sim_center_tt"] = center_tt
        st.session_state["run_sim"] = True
    except Exception as e:
        print(f"Failed to queue simulation: {e}")

# ================================================================================
#  CONSTANTS AND CONFIGURATION
# ================================================================================
MANUAL_SAT_DEFAULT_MASS_KG = 250
MASS_WIDGET_MAX_KG = 500_000.0
EARTH_RADIUS_KM = 6371.0
MU_EARTH_KM3_S2 = 398600.4418
ANALYSIS_STEP_MIN = 5
CONJUNCTION_DISTANCE_THRESHOLD_KM = 500.0
APSIS_FILTER_THRESHOLD_KM = 50.0

GROUP_CONFIG = {
    "STARLINK": {"label": "STARLINK", "spacetrack_mode": "name", "spacetrack_value": "STARLINK", "celestrak_group": "starlink", "default_mass_kg": 250},
    "ONEWEB": {"label": "ONEWEB", "spacetrack_mode": "name", "spacetrack_value": "ONEWEB", "celestrak_group": "oneweb", "default_mass_kg": 150},
    "ISS": {"label": "ISS", "spacetrack_mode": "norad", "spacetrack_value": 25544, "celestrak_group": "stations", "default_mass_kg": 419725},
    "KUIPER": {"label": "KUIPER", "spacetrack_mode": "name", "spacetrack_value": "KUIPER", "celestrak_group": "kuiper", "default_mass_kg": 630},
    "IRIDIUM-NEXT": {"label": "IRIDIUM NEXT", "spacetrack_mode": "name", "spacetrack_value": "IRIDIUM", "celestrak_group": "iridium-NEXT", "default_mass_kg": 860},
    "PLANET": {"label": "PLANET", "spacetrack_mode": "name", "spacetrack_value": "PLANET", "celestrak_group": "planet", "default_mass_kg": 5},
}

def get_group_default_mass(group_key: str) -> int:
    return int(GROUP_CONFIG.get(group_key, {}).get("default_mass_kg", 250))

# ================================================================================
#  TIME AND ORBITAL HELPERS
# ================================================================================
def build_time_grid(start_tt: float, window_hrs: int, step_min: int = ANALYSIS_STEP_MIN):
    if ts is None:
        return None, None
    n_steps = max(1, int(window_hrs * 60 // step_min) + 1)
    offsets = np.linspace(0, (n_steps - 1) * step_min, n_steps) / 1440.0
    return ts.tt_jd(start_tt + offsets), offsets

def propagated_positions(sat, times):
    try:
        return sat.at(times).position.km
    except Exception as e:
        print(f"Position propagation error for {sat.name}: {str(e)[:50]}")
        return None

def _set_mass_widget_values(mass_a: float, mass_b: float):
    try:
        mass_a = float(max(1.0, min(mass_a, MASS_WIDGET_MAX_KG)))
        mass_b = float(max(1.0, min(mass_b, MASS_WIDGET_MAX_KG)))
        if hasattr(st, "session_state"):
            st.session_state["mass_a_kg"] = mass_a
            st.session_state["mass_b_kg"] = mass_b
            st.session_state.pop("mass_a_input", None)
            st.session_state.pop("mass_b_input", None)
    except Exception as e:
        print(f"Error setting mass widget values: {e}")

def sync_mass_a_from_input():
    try:
        if not hasattr(st, "session_state"): return
        value = float(max(1.0, min(st.session_state.get("mass_a_input", 1.0), MASS_WIDGET_MAX_KG)))
        st.session_state["mass_a_kg"] = value
        st.session_state["mass_a_input"] = value
    except Exception:
        pass

def sync_mass_b_from_input():
    try:
        if not hasattr(st, "session_state"): return
        value = float(max(1.0, min(st.session_state.get("mass_b_input", 1.0), MASS_WIDGET_MAX_KG)))
        st.session_state["mass_b_kg"] = value
        st.session_state["mass_b_input"] = value
    except Exception:
        pass

def sync_mass_defaults(group_key: str):
    try:
        if not hasattr(st, "session_state"): return
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
    except Exception:
        pass

# ================================================================================
#  CSS — ENHANCED MISSION CONTROL THEME
# ================================================================================
def get_theme_css(theme="dark"):
    password_fix_css = """
    button[aria-label*="visibility"] span, button[aria-label*="show"] span, button[aria-label*="hide"] span { display: none !important; font-size: 0 !important; }
    button[aria-label*="visibility"]::before, button[aria-label*="show"]::before, button[aria-label*="hide"]::before { content: "👁"; font-size: 14px; }
    """
    if theme == "light":
        return f"""<style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        :root {{ --bg:#ffffff; --bg2:#f8f9fa; --bg3:#e9ecef; --border:#dee2e6; --accent:#0066cc; --warn:#ffc107; --crit:#dc3545; --text:#212529; --dim:#6c757d; --mono:'Space Mono',monospace; --sans:'Barlow Condensed',sans-serif; --ui:'Inter',sans-serif; --gradient-dark: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%); --shadow-card: 0 4px 20px rgba(0, 0, 0, 0.08); }}
        html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{{ background:var(--bg) !important; color:var(--text) !important; font-family:var(--ui) !important; background-image: var(--gradient-dark); }}
        [data-testid="stSidebar"]{{ background:var(--bg2) !important; border-right:1px solid var(--border) !important; box-shadow: var(--shadow-card); }}
        [data-testid="stSidebar"] *{{ color:var(--text) !important; font-family:var(--ui) !important; }}
        h1{{ font-family:var(--sans) !important; font-weight:900 !important; font-size:2.2rem !important; letter-spacing:.06em !important; color:#000 !important; text-transform:uppercase !important; }}
        h2,h3{{ font-family:var(--sans) !important; color:var(--accent) !important; font-weight:700 !important; letter-spacing:.08em !important; text-transform:uppercase !important; border-bottom:1px solid var(--border) !important; padding-bottom:.4em !important; margin-bottom:1em !important; }}
        [data-testid="metric-container"]{{ background:var(--bg) !important; border:1px solid var(--border) !important; border-left:4px solid var(--accent) !important; padding:16px 20px !important; border-radius:8px !important; box-shadow: var(--shadow-card); }}
        [data-testid="metric-container"] label{{ font-family:var(--ui) !important; font-size:.7rem !important; letter-spacing:.12em !important; color:var(--dim) !important; text-transform:uppercase !important; font-weight:600 !important; }}
        [data-testid="metric-container"] [data-testid="stMetricValue"]{{ font-family:var(--mono) !important; color:var(--accent) !important; font-size:1.8rem !important; font-weight:700 !important; }}
        .stButton button{{ background:linear-gradient(135deg, rgba(0, 102, 204, 0.1) 0%, rgba(0, 68, 153, 0.1) 100%) !important; border:1px solid var(--accent) !important; color:var(--accent) !important; font-family:var(--ui) !important; font-weight:600 !important; border-radius:6px !important; }}
        .stTextInput input, .stNumberInput input, .stSelectbox select{{ background:var(--bg) !important; border:1px solid var(--border) !important; color:var(--text) !important; border-radius:6px !important; }}
        .info-panel{{ background:var(--bg2) !important; border:1px solid var(--border) !important; border-left:4px solid var(--accent) !important; padding:16px 20px !important; border-radius:8px !important; margin:16px 0 !important; color:var(--text) !important; }}
        .warn-panel{{ background:var(--bg2) !important; border:1px solid var(--border) !important; border-left:4px solid var(--warn) !important; padding:16px 20px !important; border-radius:8px !important; margin:16px 0 !important; color:var(--text) !important; }}
        .crit-panel{{ background:var(--bg2) !important; border:1px solid var(--border) !important; border-left:4px solid var(--crit) !important; padding:16px 20px !important; border-radius:8px !important; margin:16px 0 !important; color:var(--text) !important; }}
        [data-testid="stSidebarCollapseButton"], header[data-testid="stHeader"]{{display:none !important;}}
        {password_fix_css}
        </style>"""
    else:
        return f"""<style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        :root {{ --bg:#05070a; --bg2:#0a0f18; --bg3:#121824; --border:#1e2d42; --accent:#00d4ff; --warn:#ffb800; --crit:#ff3d5c; --text:#c4d4e8; --dim:#5a7a94; --mono:'Space Mono',monospace; --sans:'Barlow Condensed',sans-serif; --ui:'Inter',sans-serif; --gradient-dark: linear-gradient(180deg, #0a0f18 0%, #05070a 100%); --shadow-card: 0 4px 20px rgba(0, 0, 0, 0.3); }}
        html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{{ background:var(--bg) !important; color:var(--text) !important; font-family:var(--ui) !important; background-image: var(--gradient-dark); }}
        [data-testid="stSidebar"]{{ background:var(--bg2) !important; border-right:1px solid var(--border) !important; box-shadow: var(--shadow-card); }}
        [data-testid="stSidebar"] *{{ color:var(--text) !important; font-family:var(--ui) !important; }}
        h1{{ font-family:var(--sans) !important; font-weight:900 !important; font-size:2.2rem !important; letter-spacing:.06em !important; color:#fff !important; text-transform:uppercase !important; text-shadow: 0 0 30px rgba(0, 212, 255, 0.3); }}
        h2,h3{{ font-family:var(--sans) !important; color:var(--accent) !important; font-weight:700 !important; letter-spacing:.08em !important; text-transform:uppercase !important; border-bottom:1px solid var(--border) !important; padding-bottom:.4em !important; margin-bottom:1em !important; }}
        [data-testid="metric-container"]{{ background:var(--bg3) !important; border:1px solid var(--border) !important; border-left:4px solid var(--accent) !important; padding:16px 20px !important; border-radius:8px !important; box-shadow: var(--shadow-card); transition: all 0.3s ease !important; }}
        [data-testid="metric-container"]:hover{{ transform: translateY(-2px); box-shadow: 0 0 20px rgba(0, 212, 255, 0.15); }}
        [data-testid="metric-container"] label{{ font-family:var(--ui) !important; font-size:.7rem !important; letter-spacing:.12em !important; color:var(--dim) !important; text-transform:uppercase !important; font-weight:600 !important; }}
        [data-testid="metric-container"] [data-testid="stMetricValue"]{{ font-family:var(--mono) !important; color:var(--accent) !important; font-size:1.8rem !important; font-weight:700 !important; }}
        .stButton button{{ background:linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 255, 168, 0.1) 100%) !important; border:1px solid var(--accent) !important; color:var(--accent) !important; font-family:var(--mono) !important; font-size:.75rem !important; letter-spacing:.1em !important; text-transform:uppercase !important; padding:10px 24px !important; border-radius:6px !important; font-weight:600 !important; }}
        .stButton button:hover{{ background:linear-gradient(135deg, #00d4ff 0%, #00ffa8 100%) !important; color:var(--bg) !important; transform: translateY(-2px); }}
        [data-baseweb="tab-list"]{{ background:var(--bg2) !important; border-bottom:2px solid var(--border) !important; gap:0 !important; padding: 0 8px !important; }}
        [data-baseweb="tab"]{{ font-family:var(--sans) !important; font-weight:600 !important; font-size:.85rem !important; letter-spacing:.1em !important; text-transform:uppercase !important; color:var(--dim) !important; padding:14px 24px !important; border-radius:8px 8px 0 0 !important; transition: all 0.3s ease !important; }}
        [aria-selected="true"][data-baseweb="tab"]{{ color:var(--accent) !important; background:linear-gradient(180deg, rgba(0, 212, 255, 0.1) 0%, transparent 100%) !important; border-bottom:2px solid var(--accent) !important; }}
        [data-testid="stTextInput"] input{{ background:var(--bg3) !important; border-color:var(--border) !important; color:var(--text) !important; font-family:var(--mono) !important; font-size:.85rem !important; border-radius:6px !important; padding:10px 14px !important; }}
        [data-testid="stSelectbox"]>div>div{{ background:var(--bg3) !important; border-color:var(--border) !important; border-radius:6px !important; }}
        .info-panel{{ background:rgba(0,212,255,.04); border:1px solid rgba(0,212,255,.18); border-left:4px solid var(--accent); padding:16px 20px; margin:12px 0; border-radius:8px; font-size:.9rem; line-height:1.7; }}
        .warn-panel{{ background:rgba(255,184,0,.04); border:1px solid rgba(255,184,0,.18); border-left:4px solid var(--warn); padding:16px 20px; margin:12px 0; border-radius:8px; font-size:.9rem; line-height:1.7; }}
        .crit-panel{{ background:rgba(255,61,92,.04); border:1px solid rgba(255,61,92,.18); border-left:4px solid var(--crit); padding:16px 20px; margin:12px 0; border-radius:8px; font-size:.9rem; line-height:1.7; }}
        [data-testid="stDataFrame"]{{ border:1px solid var(--border) !important; border-radius:8px !important; overflow:hidden !important; box-shadow: var(--shadow-card); }}
        [data-testid="stSidebarCollapseButton"], header[data-testid="stHeader"]{{display:none !important;}}
        {password_fix_css}
        </style>"""

# ================================================================================
#  EARTH VIEW TEXTURE LOADER
# ================================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def load_earth_texture(resolution: int = 80, style: str = "futuristic"):
    try:
        if style == "night":
            urls = ["https://eoimages.gsfc.nasa.gov/images/imagerecords/79000/79765/dnb_land_ocean_ice.2012.3600x1800.jpg", "https://upload.wikimedia.org/wikipedia/commons/b/ba/The_earth_at_night.jpg"]
        elif style == "realistic" or style == "futuristic":
            urls = ["https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg", "https://upload.wikimedia.org/wikipedia/commons/a/ad/Blue_Marble_2002.png"]
        else:
            urls = ["https://upload.wikimedia.org/wikipedia/commons/c/cd/Land_ocean_ice_2048.jpg/1024px-Land_ocean_ice_2048.jpg"]

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                W, H = resolution * 2, resolution
                img = img.resize((W, H), Image.LANCZOS)
                img_array = np.array(img, dtype=np.float32)

                if style == "futuristic":
                    img_array = img_array * 0.35
                    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.5, 0, 255)
                    img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.2, 0, 255)
                    mean_val = np.mean(img_array)
                    img_array = np.clip((img_array - mean_val) * 1.4 + mean_val, 0, 255)
                    img_array = np.clip(img_array + 10, 0, 255)
                elif style == "night":
                    img_array = np.clip(img_array * 1.3, 0, 255)

                img_array = img_array.astype(np.uint8)
                img = Image.fromarray(img_array)
                imgq = img.quantize(colors=256, method=Image.MEDIANCUT)
                pal = np.array(imgq.getpalette(), dtype=np.uint8).reshape(-1, 3)[:256]
                idx = np.flipud(np.array(imgq, dtype=float))
                surf_color = idx / 255.0
                colorscale = [[i / 255.0, f"rgb({pal[i, 0]},{pal[i, 1]},{pal[i, 2]})"] for i in range(256)]
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

        # Fallback Wireframe
        lat = np.linspace(np.pi / 2, -np.pi / 2, 50)
        lon = np.linspace(-np.pi, np.pi, 100)
        lon_g, lat_g = np.meshgrid(lon, lat)
        R = EARTH_RADIUS_KM
        x = R * np.cos(lat_g) * np.cos(lon_g)
        y = R * np.cos(lat_g) * np.sin(lon_g)
        z = R * np.sin(lat_g)
        surf_color = np.ones_like(lat_g) * 0.5
        colorscale = [[0, "rgb(10,20,40)"], [1, "rgb(30,60,120)"]]
        return x, y, z, surf_color, colorscale
    except Exception as e:
        print(f"Earth texture load failed: {str(e)[:50]}")
        return None

# ================================================================================
#  DATA FETCHING & TLE PARSING
# ================================================================================
def count_tle_objects(lines: list) -> int:
    if not lines: return 0
    is_3ln = not (lines[0].startswith("1 ") or lines[0].startswith("2 "))
    step = 3 if is_3ln else 2
    return len(lines) // step

def trim_tle_lines(lines: list, sat_limit: int) -> list:
    if not lines: return []
    is_3ln = not (lines[0].startswith("1 ") or lines[0].startswith("2 "))
    step = 3 if is_3ln else 2
    max_lines = max(1, sat_limit) * step
    return lines[:max_lines]

def fetch_spacetrack_tles(username: str, password: str, group_key: str, sat_limit: int):
    try:
        config = GROUP_CONFIG.get(group_key)
        if not config: return None, f"Unknown group: {group_key}"
        client = SpaceTrackClient(identity=username, password=password)
        if config["spacetrack_mode"] == "norad":
            raw = client.gp(norad_cat_id=config["spacetrack_value"], format="tle", orderby="epoch desc", limit=1)
        else:
            raw = client.gp(object_name=op.like(f"{config['spacetrack_value']}%"), format="tle", orderby="epoch desc", limit=max(1, sat_limit))
        if not raw or not raw.strip(): return None, f"No data found for '{group_key}' on Space-Track."
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        if len(lines) < 2: return None, "Invalid TLE returned."
        lines = trim_tle_lines(lines, sat_limit)
        return lines, "Data fetched from Space-Track."
    except Exception as e:
        err = str(e).lower()
        if "authentication" in err: return None, "Space-Track Auth Error. Check credentials."
        elif "timeout" in err: return None, "Space-Track Timeout. Try again."
        elif "rate limit" in err: return None, "Space-Track Rate Limit Exceeded. Wait a few minutes."
        return None, f"Space-Track error: {err[:100]}"

def fetch_celestrak_tles(group_key: str, sat_limit: int):
    try:
        config = GROUP_CONFIG.get(group_key)
        if not config: return None, f"Unknown group: {group_key}"
        url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={config['celestrak_group']}&FORMAT=TLE"
        headers = {"User-Agent": "Mozilla/5.0 (StarWeb-CARA/1.0)"}
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException:
                if attempt == 2: raise
                time.sleep(2**attempt)
        raw = resp.text
        if not raw or not raw.strip(): return None, "No data on CelesTrak."
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        if len(lines) < 2: return None, "Invalid TLE returned."
        lines = trim_tle_lines(lines, sat_limit)
        return lines, "Data fetched from CelesTrak fallback."
    except Exception as e:
        return None, f"CelesTrak error: {str(e)[:100]}"

def fetch_tles_with_fallback(username: str, password: str, group_key: str, sat_limit: int):
    lines, msg1 = fetch_spacetrack_tles(username, password, group_key, sat_limit)
    if lines: return {"lines": lines, "source": "Space-Track", "message": msg1}
    lines, msg2 = fetch_celestrak_tles(group_key, sat_limit)
    if lines: return {"lines": lines, "source": "CelesTrak", "message": f"{msg1} Fallback used."}
    st.sidebar.error(msg1)
    st.sidebar.error(msg2)
    return None

def build_fallback_sat_name(tle_line_1: str, fallback_name_prefix: str = None) -> str:
    norad_id = tle_line_1[2:7].strip()
    if fallback_name_prefix: return f"{fallback_name_prefix} {norad_id}"
    return f"NORAD {norad_id}"

def parse_tles(lines: list, limit: int = 30, fallback_name_prefix: str = None) -> list:
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
            if len(sats) >= limit: break
        except Exception:
            continue
    return sats

# ================================================================================
#  ORBITAL ELEMENTS & APSIS FILTER
# ================================================================================
def get_orbital_elements(sat: EarthSatellite) -> dict:
    try:
        model = sat.model
        incl = math.degrees(model.inclo)
        raan = math.degrees(model.nodeo)
        ecc = model.ecco
        argp = math.degrees(model.argpo)
        mean_m = math.degrees(model.mo)
        n_rpm = model.no_kozai * (60.0 / (2 * math.pi))
        n_rads = model.no_kozai / 60.0
        a_km = (MU_EARTH_KM3_S2 / n_rads**2) ** (1 / 3)
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

def apsis_filter(sats: list, threshold_km: float = APSIS_FILTER_THRESHOLD_KM) -> list:
    def apsis(sat):
        try:
            n = sat.model.no_kozai / 60.0
            a = (MU_EARTH_KM3_S2 / n**2) ** (1 / 3)
            e = sat.model.ecco
            return a * (1 - e) - EARTH_RADIUS_KM, a * (1 + e) - EARTH_RADIUS_KM
        except Exception:
            return 0.0, 10000.0

    passed = []
    for s1, s2 in combinations(sats, 2):
        q1, Q1 = apsis(s1)
        q2, Q2 = apsis(s2)
        if max(q1, q2) <= min(Q1, Q2) + threshold_km:
            passed.append((s1, s2))
    return passed

# ================================================================================
#  RISK CALCULATIONS
# ================================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def foster_2d_pc(miss_km: float, sigma_x: float, sigma_y: float, hbr_km: float = 0.020) -> float:
    try:
        if sigma_x <= 0 or sigma_y <= 0: return 0.0
        def integrand(y, x):
            return (1.0 / (2 * math.pi * sigma_x * sigma_y)) * math.exp(-0.5 * (((x - miss_km) / sigma_x) ** 2 + (y / sigma_y) ** 2))
        res, _ = dblquad(integrand, -hbr_km, hbr_km, lambda x: -math.sqrt(max(hbr_km**2 - x**2, 0)), lambda x: math.sqrt(max(hbr_km**2 - x**2, 0)), limit=50)
        return max(float(res), 0.0)
    except Exception:
        return collision_probability_isotropic(miss_km, (sigma_x + sigma_y) / 2, hbr_km)

@st.cache_data(show_spinner=False, ttl=3600)
def collision_probability_isotropic(miss_km: float, sigma_km: float, hbr_km: float = 0.020) -> float:
    if sigma_km <= 0: return 0.0
    pc = norm.cdf((hbr_km - miss_km) / sigma_km) + norm.cdf((hbr_km + miss_km) / sigma_km) - 1.0
    return max(float(pc), 0.0)

@st.cache_data(show_spinner=False, ttl=3600)
def mahalanobis_test(miss_km: float, sigma_km: float) -> dict:
    if sigma_km <= 0: return {"Md": 999.0, "valid_2d": True, "label": "Valid"}
    Md = miss_km / sigma_km
    valid = Md >= 1.5
    if Md < 0.5: label = "Invalid — 3D-Pc / Monte Carlo required"
    elif Md < 1.5: label = "Borderline — 3D-Pc recommended"
    else: label = "2D-Pc Valid"
    return {"Md": round(Md, 3), "valid_2d": valid, "label": label}

@st.cache_data(show_spinner=False, ttl=3600)
def max_pc_analysis(miss_km: float, hbr_km: float = 0.020) -> float:
    sigma_opt = miss_km / math.sqrt(2.0) if miss_km > 0 else hbr_km
    return collision_probability_isotropic(miss_km, max(sigma_opt, 1e-6), hbr_km)

@st.cache_data(show_spinner=False, ttl=3600)
def dilution_check(pc: float, sigma_km: float, miss_km: float) -> dict:
    diluted = (sigma_km > 5.0 * miss_km) and (pc < 1e-6) and (miss_km < 100.0)
    if diluted: return {"diluted": True, "msg": "PROBABILITY DILUTION DETECTED. WSPRT or Max-Pc analysis required."}
    return {"diluted": False, "msg": "Normal"}

@st.cache_data(show_spinner=False, ttl=3600)
def fragmentation_probability(rel_vel_km_s: float, mass_a_kg: float = 250.0, mass_b_kg: float = 250.0) -> dict:
    v_ms = rel_vel_km_s * 1000.0
    E_c = 0.5 * mass_b_kg * v_ms**2 / (mass_a_kg * 1000.0)
    if E_c >= 40.0: pf_level, pf_color, pf_desc = "CATASTROPHIC", "#ff2b4d", "Complete fragmentation — Kessler contribution likely"
    elif E_c >= 10.0: pf_level, pf_color, pf_desc = "SEVERE", "#ff6b00", "Operational loss and significant debris"
    elif E_c >= 1.0: pf_level, pf_color, pf_desc = "DAMAGING", "#ffaa00", "Partial damage or subsystem failure"
    else: pf_level, pf_color, pf_desc = "LOW", "#00ff9d", "Minor damage — fragmentation unlikely"
    n_debris = int(0.1 * (mass_a_kg + mass_b_kg) * (rel_vel_km_s / 7.0))
    return {"E_c_J_per_g": round(E_c, 2), "level": pf_level, "color": pf_color, "desc": pf_desc, "est_debris": n_debris}

@st.cache_data(show_spinner=False, ttl=3600)
def risk_level(pc: float, theme="dark") -> tuple:
    try:
        if pc > 1e-3: return "CRITICAL", "#dc3545" if theme == "light" else "#ff3d5c"
        elif pc > 1e-4: return "HIGH", "#fd7e14" if theme == "light" else "#ff6b00"
        elif pc > 1e-5: return "MEDIUM", "#ffc107" if theme == "light" else "#ffb800"
        else: return "LOW", "#28a745" if theme == "light" else "#00ffa8"
    except Exception: return "UNKNOWN", "#6c757d" if theme == "light" else "#5a7a94"

def _relative_velocity(s1, s2, t) -> float:
    v1 = s1.at(t).velocity.km_per_s
    v2 = s2.at(t).velocity.km_per_s
    return float(np.linalg.norm(np.array(v1) - np.array(v2)))

def _compute_conjunction_metrics(sat1, sat2, pos1, pos2, jd_values, sigma_km, hbr_km, mass_a_kg, mass_b_kg, theme="dark"):
    if pos1 is None or pos2 is None or sat1 is None or sat2 is None: return None
    try: dists = np.linalg.norm(pos1 - pos2, axis=0)
    except Exception: return None
    if len(dists) == 0 or np.all(np.isnan(dists)): return None
    tca_idx = int(np.nanargmin(dists))
    min_d = float(dists[tca_idx])
    if min_d >= CONJUNCTION_DISTANCE_THRESHOLD_KM: return None
    best_t = ts.tt_jd(float(jd_values[tca_idx]))
    dist_arr = dists.tolist()
    rel_vel = _relative_velocity(sat1, sat2, best_t)
    pc_iso = collision_probability_isotropic(min_d, sigma_km, hbr_km)
    pc_foster = foster_2d_pc(min_d, sigma_km, sigma_km * 2, hbr_km=hbr_km)
    pc_max = max_pc_analysis(min_d, hbr_km)
    mah = mahalanobis_test(min_d, sigma_km)
    dil = dilution_check(pc_iso, sigma_km, min_d)
    frag = fragmentation_probability(rel_vel, mass_a_kg, mass_b_kg)
    sev, color = risk_level(pc_iso, theme)
    return {
        "TCA (UTC)": best_t.utc_strftime("%Y-%m-%d %H:%M:%S"),
        "Object A": sat1.name, "Object B": sat2.name,
        "Distance (km)": round(min_d, 3), "Relative Velocity (km/s)": round(rel_vel, 3),
        "Pc (isotropic)": pc_iso, "Pc (Foster 2D)": pc_foster, "Pc Max": pc_max,
        "Pc (scientific)": f"{pc_iso:.3e}", "Mahalanobis Md": mah["Md"], "2D-Pc Valid": mah["label"],
        "Dilution": dil["diluted"], "Dilution Message": dil["msg"],
        "Ec (J/g)": frag["E_c_J_per_g"], "Fragmentation Level": frag["level"], "Estimated Debris": frag["est_debris"],
        "Risk Level": sev, "_color": color, "_dist_arr": dist_arr, "_tca_tt": best_t.tt, "_s1": sat1, "_s2": sat2,
    }

def compute_conjunctions(sats: list, window_hrs: int, sigma_km: float, hbr_km: float = 0.020, mass_a_kg: float = 250.0, mass_b_kg: float = 250.0, theme: str = "dark") -> tuple:
    if ts is None: return pd.DataFrame(), 0, 0
    times, _ = build_time_grid(ts.now().tt, window_hrs)
    if times is None: return pd.DataFrame(), 0, 0
    jd_values = np.asarray(times.tt)
    n_total = len(sats) * (len(sats) - 1) // 2
    progress_bar = st.progress(0, text="Analyzing satellite pairs...") if n_total > 100 else None
    candidate_pairs = apsis_filter(sats, threshold_km=APSIS_FILTER_THRESHOLD_KM)
    n_filtered = n_total - len(candidate_pairs)
    positions_by_id = {}
    for idx, sat in enumerate(sats):
        if progress_bar and idx % max(1, len(sats) // 20) == 0: progress_bar.progress((idx + 1) / len(sats), text=f"Precomputing positions... {idx + 1}/{len(sats)}")
        positions_by_id[id(sat)] = (sat, propagated_positions(sat, times))
    results = []
    for idx, (sat1, sat2) in enumerate(candidate_pairs):
        if progress_bar and idx % max(1, len(candidate_pairs) // 20) == 0: progress_bar.progress((idx + 1) / len(candidate_pairs), text=f"Analyzing conjunctions... {idx + 1}/{len(candidate_pairs)}")
        sat1_obj, pos1 = positions_by_id.get(id(sat1), (None, None))
        sat2_obj, pos2 = positions_by_id.get(id(sat2), (None, None))
        if pos1 is None or pos2 is None: continue
        result = _compute_conjunction_metrics(sat1_obj, sat2_obj, pos1, pos2, jd_values, sigma_km, hbr_km, mass_a_kg, mass_b_kg, theme)
        if result: results.append(result)
    if progress_bar: progress_bar.empty()
    return pd.DataFrame(results), n_filtered, n_total

def compute_conjunctions_custom(my_sat, sats: list, window_hrs: int, sigma_km: float, hbr_km: float = 0.020, mass_a_kg: float = 250.0, mass_b_kg: float = 250.0, theme: str = "dark") -> pd.DataFrame:
    if ts is None: return pd.DataFrame()
    times, _ = build_time_grid(ts.now().tt, window_hrs)
    if times is None: return pd.DataFrame()
    jd_values = np.asarray(times.tt)
    R_E, GM = EARTH_RADIUS_KM, MU_EARTH_KM3_S2
    def apsis(sat):
        try:
            n = sat.model.no_kozai / 60.0
            a = (GM / n**2) ** (1 / 3)
            e = sat.model.ecco
            return a * (1 - e) - R_E, a * (1 + e) - R_E
        except Exception: return 0.0, 10000.0
    my_q, my_Q = apsis(my_sat)
    my_pos = propagated_positions(my_sat, times)
    results = []
    if my_pos is None: return pd.DataFrame(results)
    for idx, sat in enumerate(sats):
        q, Q = apsis(sat)
        if max(my_q, q) > min(my_Q, Q) + 100.0: continue
        sat_pos = propagated_positions(sat, times)
        if sat_pos is None: continue
        result = _compute_conjunction_metrics(my_sat, sat, my_pos, sat_pos, jd_values, sigma_km, hbr_km, mass_a_kg, mass_b_kg, theme)
        if result: results.append(result)
    return pd.DataFrame(results)

# ================================================================================
#  PLOTS - ENHANCED VISUALIZATION
# ================================================================================
DARK = dict(paper_bgcolor="#05070a", plot_bgcolor="#05070a", font=dict(family="Space Mono, monospace", color="#c4d4e8", size=11))
ENHANCED_COLORS = ["#00d4ff", "#00ffa8", "#ffb800", "#ff6b00", "#c060ff", "#ff3d5c", "#60d0ff", "#80ffb0", "#ffcc60", "#ff9060"]

def fig_3d_orbits(sats):
    now = ts.now()
    fig = go.Figure()

    r_atm = EARTH_RADIUS_KM * 1.035
    u, v = np.mgrid[0 : 2 * np.pi : 60j, 0 : np.pi : 40j]
    x_atm = r_atm * np.cos(u) * np.sin(v)
    y_atm = r_atm * np.sin(u) * np.sin(v)
    z_atm = r_atm * np.cos(v)
    fig.add_trace(go.Surface(x=x_atm, y=y_atm, z=z_atm, surfacecolor=z_atm, colorscale=[[0, "rgba(0,212,255,0.0)"], [0.5, "rgba(0,212,255,0.06)"], [1, "rgba(0,212,255,0.18)"]], showscale=False, hoverinfo="skip", opacity=0.6, name="Atmosphere"))

    earth = load_earth_texture(resolution=100, style="futuristic")
    if earth:
        x, y, z, sc, cs = earth
        fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=sc, colorscale=cs, showscale=False, opacity=1.0, hoverinfo="skip", lightposition=dict(x=15000, y=0, z=15000), lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3, roughness=0.7), name="Earth"))
    else:
        r_e = EARTH_RADIUS_KM
        fig.add_trace(go.Surface(x=r_e * np.cos(u) * np.sin(v), y=r_e * np.sin(u) * np.sin(v), z=r_e * np.cos(v), colorscale=[[0, "#030a14"], [0.5, "#081b33"], [1, "#0a2240"]], showscale=False, opacity=1.0, hoverinfo="skip", name="Earth"))

    colors = ENHANCED_COLORS
    offsets = np.linspace(0, 95, 120) / 1440.0

    for k, sat in enumerate(sats):
        times = ts.tt_jd(now.tt + offsets)
        c = colors[k % len(colors)]
        try: pos = sat.at(times).position.km
        except Exception: pos = np.full((3, 120), np.nan)
        if not np.all(np.isnan(pos)):
            fig.add_trace(go.Scatter3d(x=pos[0].tolist(), y=pos[1].tolist(), z=pos[2].tolist(), mode="lines+markers", line=dict(color=c, width=1.0), marker=dict(size=2.5, color="#ffffff", line=dict(color=c, width=0.5), opacity=0.7), name=sat.name, opacity=0.6, hovertemplate=f"<b>{sat.name}</b><br>Orbit Path<extra></extra>"))
        try: p0 = sat.at(now).position.km
        except Exception: p0 = np.full((3,), np.nan)
        if not np.any(np.isnan(p0)):
            fig.add_trace(go.Scatter3d(x=[float(p0[0])], y=[float(p0[1])], z=[float(p0[2])], mode="markers", marker=dict(color="#ffffff", size=8, symbol="circle", line=dict(color=c, width=3.5), opacity=1.0), name=f"{sat.name} (current)", showlegend=False, hovertemplate=f"<b>{sat.name}</b><br>Current Position<br>X: %{{x:.1f}} km<br>Y: %{{y:.1f}} km<br>Z: %{{z:.1f}} km<extra></extra>"))

    fig.update_layout(**DARK, margin=dict(l=0, r=0, t=0, b=0), scene=dict(bgcolor="#000408", xaxis=dict(visible=False, showgrid=False, zeroline=False), yaxis=dict(visible=False, showgrid=False, zeroline=False), zaxis=dict(visible=False, showgrid=False, zeroline=False), aspectmode="cube", camera=dict(eye=dict(x=2.1, y=2.1, z=0.9), up=dict(x=0, y=0, z=1))), showlegend=False, hoverlabel=dict(bgcolor="rgba(10,15,24,.95)", bordercolor="#00d4ff", font_size=11, font_family="Space Mono"))
    return fig

def fig_ground_tracks(sats):
    now = ts.now()
    offsets = np.linspace(0, 95, 200) / 1440.0
    colors = ENHANCED_COLORS
    fig = go.Figure()
    for k, sat in enumerate(sats):
        times = ts.tt_jd(now.tt + offsets)
        try:
            geo = wgs84.subpoint_of(sat.at(times))
            g0 = wgs84.subpoint_of(sat.at(now))
        except Exception: continue
        c = colors[k % len(colors)]
        fig.add_trace(go.Scattergeo(lat=geo.latitude.degrees, lon=geo.longitude.degrees, mode="lines", line=dict(color=c, width=2.5), name=sat.name, opacity=0.9, hovertemplate=f"<b>{sat.name}</b><br>Lat: %{{lat:.2f}}°<br>Lon: %{{lon:.2f}}°<extra></extra>"))
        fig.add_trace(go.Scattergeo(lat=[g0.latitude.degrees], lon=[g0.longitude.degrees], mode="markers+text", marker=dict(color=c, size=10, symbol="circle", line=dict(color="#ffffff", width=1.5), opacity=0.9), text=[sat.name], textposition="top right", textfont=dict(size=9, family="Space Mono", color=c, weight="bold"), showlegend=False, hovertemplate=f"<b>{sat.name}</b><br>Current Position<br>Lat: %{{lat:.2f}}°<br>Lon: %{{lon:.2f}}°<extra></extra>"))
    fig.update_layout(**DARK, height=450, margin=dict(l=0, r=0, t=30, b=0), geo=dict(showland=True, landcolor="#0d2137", showocean=True, oceancolor="#050d18", showcoastlines=True, coastlinecolor="#2a5070", coastlinewidth=1.0, showcountries=True, countrycolor="#152535", countrywidth=0.5, showlakes=True, lakecolor="#080f1a", showrivers=True, rivercolor="#0a1828", showframe=False, bgcolor="#05070a", projection_type="natural earth", resolution=50, lonaxis=dict(range=[-180, 180], showgrid=True, gridcolor="rgba(30,45,66,.6)", gridwidth=0.4), lataxis=dict(range=[-90, 90], showgrid=True, gridcolor="rgba(30,45,66,.6)", gridwidth=0.4)), legend=dict(font=dict(size=9, family="Space Mono", color="#c4d4e8"), bgcolor="rgba(5,7,10,.9)", bordercolor="#1a2740", borderwidth=1, x=0.0, y=1.0), title=dict(text="Ground Track -- Current Position and 95min Orbit", font=dict(size=11, family="Barlow Condensed", color="#00c8ff"), x=0.01, y=0.99))
    return fig

@st.cache_data(show_spinner=False, ttl=1800)
def fig_distance_profile(dist_arr, window_hrs, miss_km, sigma_km, hbr_km=0.020):
    t_axis = np.arange(len(dist_arr)) * ANALYSIS_STEP_MIN / 60.0
    fig = go.Figure()
    fig.add_hline(y=hbr_km, line=dict(color="#ff2b4d", dash="dot", width=1), annotation_text=f"HBR ({hbr_km * 1000:.0f} m)", annotation_font_size=9)
    fig.add_hrect(y0=max(0, miss_km - sigma_km), y1=miss_km + sigma_km, fillcolor="rgba(0,200,255,.05)", line_width=0)
    fig.add_trace(go.Scatter(x=t_axis, y=dist_arr, mode="lines", line=dict(color="#00c8ff", width=1.5), name="Distance (km)", fill="tozeroy", fillcolor="rgba(0,200,255,.04)"))
    dist_np = np.asarray(dist_arr, dtype=float)
    if len(dist_np) and not np.all(np.isnan(dist_np)):
        tca_i = int(np.nanargmin(dist_np))
        fig.add_trace(go.Scatter(x=[t_axis[tca_i]], y=[dist_np[tca_i]], mode="markers+text", marker=dict(color="#ff2b4d", size=8), text=[f" TCA: {dist_np[tca_i]:.1f} km"], textposition="top right", textfont=dict(size=9, family="Space Mono", color="#ff2b4d"), name="TCA", showlegend=False))
    fig.update_layout(**DARK, height=280, xaxis=dict(title="Time (hours)", gridcolor="#1a2740", zeroline=False, tickfont=dict(size=9)), yaxis=dict(title="Distance (km)", gridcolor="#1a2740", zeroline=False, tickfont=dict(size=9)), title=dict(text="Distance Profile — TCA Analysis", font=dict(size=11, family="Barlow Condensed", color="#00c8ff"), x=0.01), legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"), margin=dict(l=10, r=10, t=35, b=10))
    return fig

@st.cache_data(show_spinner=False, ttl=1800)
def fig_risk_gauge(pc: float):
    sev, color = risk_level(pc)
    fig = go.Figure(go.Indicator(mode="gauge+number", value=pc, number=dict(valueformat=".2e", font=dict(family="Space Mono", color=color, size=18)), gauge=dict(axis=dict(range=[0, 1e-3], tickvals=[0, 1e-5, 1e-4, 1e-3], ticktext=["0", "1e-5", "1e-4", "1e-3"], tickfont=dict(size=8, family="Space Mono", color="#4a6880")), bar=dict(color=color, thickness=0.25), bgcolor="#0c1018", bordercolor="#1a2740", steps=[dict(range=[0, 1e-5], color="#0d1820"), dict(range=[1e-5, 1e-4], color="#141e10"), dict(range=[1e-4, 1e-3], color="#1e1008")], threshold=dict(line=dict(color="#ff2b4d", width=2), value=1e-4)), title=dict(text=f"Pc — {sev}", font=dict(family="Barlow Condensed", color=color, size=14)), domain=dict(x=[0, 1], y=[0, 1])))
    fig.update_layout(**DARK, height=210, margin=dict(l=10, r=10, t=10, b=10))
    return fig

@st.cache_data(show_spinner=False, ttl=1800)
def fig_orbital_elements_radar(elems_list):
    fig = go.Figure()
    colors = ["#00c8ff", "#00ff9d", "#ffaa00", "#ff6b00", "#c060ff", "#ff2b4d"]
    for k, (name, elems) in enumerate(elems_list):
        if not elems: continue
        try:
            alt = float(str(elems.get("Mean Altitude (km)", 0)))
            incl = float(str(elems.get("Inclination i (°)", 0)))
            ecc = float(str(elems.get("Eccentricity e", "0")))
            fig.add_trace(go.Scatter(x=[incl], y=[alt], mode="markers+text", marker=dict(color=colors[k % len(colors)], size=10 + ecc * 80, line=dict(color="#fff", width=0.5)), text=[name[:12]], textposition="top center", textfont=dict(size=8, family="Space Mono", color=colors[k % len(colors)]), name=name))
        except Exception: continue
    fig.update_layout(**DARK, height=320, xaxis=dict(title="Inclination i (°)", gridcolor="#1a2740", zeroline=False, tickfont=dict(size=9)), yaxis=dict(title="Altitude (km)", gridcolor="#1a2740", zeroline=False, tickfont=dict(size=9)), title=dict(text="Orbital Space — Altitude / Inclination Distribution", font=dict(size=11, family="Barlow Condensed", color="#00c8ff"), x=0.01), margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
    return fig

def fig_animated_conjunction(sat_a, sat_b, window_hrs=6, show_orbits=True, show_tca=True, center_tt=None, frame_duration=60):
    if ts is None: return go.Figure(), 0, 0.0, np.array([0.0]), np.array([0.0]), 0.0
    now = ts.now()

    sim_start_tt = float(center_tt) - (window_hrs / 2.0) / 24.0 if center_tt is not None else now.tt
    step_min = 5
    n_frames = int(math.ceil(window_hrs * 60 / step_min)) + 1

    if n_frames < 2: return go.Figure(), 0, 0.0, np.array([0.0]), np.array([sim_start_tt]), sim_start_tt

    orbit_pts = 120
    orb_off = np.linspace(0, 96, orbit_pts) / 1440.0
    try:
        orb_a = sat_a.at(ts.tt_jd(sim_start_tt + orb_off)).position.km
        orb_b = sat_b.at(ts.tt_jd(sim_start_tt + orb_off)).position.km
    except Exception:
        orb_a = orb_b = np.full((3, orbit_pts), np.nan)

    anim_off = np.arange(n_frames) * step_min / 1440.0
    anim_jd = sim_start_tt + anim_off
    try:
        pos_a = sat_a.at(ts.tt_jd(anim_jd)).position.km
        pos_b = sat_b.at(ts.tt_jd(anim_jd)).position.km
    except Exception:
        pos_a = pos_b = np.full((3, n_frames), np.nan)

    dists = np.linalg.norm(pos_a - pos_b, axis=0)
    if np.all(np.isnan(dists)):
        tca_idx, tca_dist = 0, 0.0
    else:
        tca_idx = int(np.nanargmin(dists))
        tca_dist = float(dists[tca_idx])

    def dist_color(d):
        if np.isnan(d): return "rgba(100,180,255,0.55)"
        if d < 50: return "#ff2b4d"
        if d < 200: return "#ffaa00"
        return "rgba(100,180,255,0.55)"

    fig = go.Figure()

    r_atm = EARTH_RADIUS_KM * 1.035
    u, v = np.mgrid[0 : 2 * np.pi : 60j, 0 : np.pi : 40j]
    x_atm = r_atm * np.cos(u) * np.sin(v)
    y_atm = r_atm * np.sin(u) * np.sin(v)
    z_atm = r_atm * np.cos(v)
    fig.add_trace(go.Surface(x=x_atm, y=y_atm, z=z_atm, surfacecolor=z_atm, colorscale=[[0, "rgba(0,212,255,0.0)"], [0.5, "rgba(0,212,255,0.06)"], [1, "rgba(0,212,255,0.18)"]], showscale=False, hoverinfo="skip", opacity=0.6, name="Atmosphere"))

    earth = load_earth_texture(resolution=80, style="futuristic")
    if earth:
        x, y, z, sc, cs = earth
        fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=sc, colorscale=cs, showscale=False, opacity=1.0, hoverinfo="skip", lightposition=dict(x=15000, y=0, z=15000), lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3, roughness=0.7), name="Earth"))
    else:
        r_e = EARTH_RADIUS_KM
        fig.add_trace(go.Surface(x=r_e * np.cos(u) * np.sin(v), y=r_e * np.sin(u) * np.sin(v), z=r_e * np.cos(v), colorscale=[[0, "#030a14"], [0.5, "#081b33"], [1, "#0a2240"]], showscale=False, opacity=1.0, hoverinfo="skip", name="Earth"))

    r_earth = EARTH_RADIUS_KM
    _pts = 80
    _lat_pm = np.linspace(-np.pi / 2, np.pi / 2, _pts)
    _lon_eq = np.linspace(-np.pi, np.pi, _pts)
    fig.add_trace(go.Scatter3d(x=(r_earth * np.cos(_lon_eq)).tolist(), y=(r_earth * np.sin(_lon_eq)).tolist(), z=np.zeros(_pts).tolist(), mode="lines", line=dict(color="rgba(100,150,200,0.35)", width=1), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=(r_earth * np.cos(_lat_pm)).tolist(), y=np.zeros(_pts).tolist(), z=(r_earth * np.sin(_lat_pm)).tolist(), mode="lines", line=dict(color="rgba(100,150,200,0.25)", width=1), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=np.zeros(_pts).tolist(), y=(r_earth * np.cos(_lat_pm)).tolist(), z=(r_earth * np.sin(_lat_pm)).tolist(), mode="lines", line=dict(color="rgba(100,150,200,0.25)", width=1), showlegend=False, hoverinfo="skip"))

    if show_orbits and not np.all(np.isnan(orb_a)):
        fig.add_trace(go.Scatter3d(x=orb_a[0].tolist(), y=orb_a[1].tolist(), z=orb_a[2].tolist(), mode="lines+markers", line=dict(color="rgba(0,200,255,0.65)", width=1.0), marker=dict(size=2.5, color="#ffffff", opacity=0.7), name=sat_a.name + " track", showlegend=False, hoverinfo="skip"))
    if show_orbits and not np.all(np.isnan(orb_b)):
        fig.add_trace(go.Scatter3d(x=orb_b[0].tolist(), y=orb_b[1].tolist(), z=orb_b[2].tolist(), mode="lines+markers", line=dict(color="rgba(255,107,0,0.65)", width=1.0), marker=dict(size=2.5, color="#ffffff", opacity=0.7), name=sat_b.name + " track", showlegend=False, hoverinfo="skip"))

    mid_tca = (pos_a[:, tca_idx] + pos_b[:, tca_idx]) / 2
    tca_tt_val = anim_jd[tca_idx]

    if show_tca and not np.any(np.isnan(mid_tca)):
        ring_radius = max(140.0, min(900.0, max(tca_dist * 1.6, 180.0)))
        ring_angle = np.linspace(0, 2 * np.pi, 96)
        normal = mid_tca / np.linalg.norm(mid_tca) if np.linalg.norm(mid_tca) > 0 else np.array([0.0, 0.0, 1.0])
        basis_a = np.cross(normal, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(basis_a) < 1e-6: basis_a = np.cross(normal, np.array([0.0, 1.0, 0.0]))
        basis_a = basis_a / np.linalg.norm(basis_a)
        basis_b = np.cross(normal, basis_a)
        ring = mid_tca[:, None] + ring_radius * np.cos(ring_angle)[None, :] * basis_a[:, None] + ring_radius * np.sin(ring_angle)[None, :] * basis_b[:, None]
        fig.add_trace(go.Scatter3d(x=ring[0].tolist(), y=ring[1].tolist(), z=ring[2].tolist(), mode="lines", line=dict(color="rgba(255,43,77,0.55)", width=3), hoverinfo="skip", showlegend=False, name="TCA risk zone"))

    _CAM_DIST = 2.4
    _CAM_DIST_Z_BOOST = 0.25
    if not np.any(np.isnan(mid_tca)) and np.linalg.norm(mid_tca) > 100:
        _unit = mid_tca / np.linalg.norm(mid_tca)
        _eye = _unit * _CAM_DIST
        _eye[2] += _CAM_DIST_Z_BOOST
        _eye = _eye / np.linalg.norm(_eye) * _CAM_DIST
        _up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(_unit, np.array([0.0, 0.0, 1.0]))) > 0.92: _up = np.array([0.0, 1.0, 0.0])
        tca_camera = dict(eye=dict(x=float(_eye[0]), y=float(_eye[1]), z=float(_eye[2])), center=dict(x=0, y=0, z=0), up=dict(x=float(_up[0]), y=float(_up[1]), z=float(_up[2])))
    else:
        tca_camera = dict(eye=dict(x=1.7, y=1.7, z=0.75), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1))

    if show_tca and not np.any(np.isnan(mid_tca)):
        tca_time_obj = ts.tt_jd(tca_tt_val)
        geo_a = wgs84.subpoint_of(sat_a.at(tca_time_obj))
        geo_b = wgs84.subpoint_of(sat_b.at(tca_time_obj))
        vel_a = np.linalg.norm(sat_a.at(tca_time_obj).velocity.km_per_s)
        vel_b = np.linalg.norm(sat_b.at(tca_time_obj).velocity.km_per_s)
        tca_hover_html = (
            f"<b><span style='color:#ff2b4d'>⚡ TCA POINT: {tca_dist:.2f} km</span></b><br>"
            f"Time: {tca_time_obj.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}<br><br>"
            f"<b>{sat_a.name}</b><br>Alt: {geo_a.elevation.km:.1f} km | Vel: {vel_a:.2f} km/s<br>Lat: {geo_a.latitude.degrees:.3f}° | Lon: {geo_a.longitude.degrees:.3f}°<br><br>"
            f"<b>{sat_b.name}</b><br>Alt: {geo_b.elevation.km:.1f} km | Vel: {vel_b:.2f} km/s<br>Lat: {geo_b.latitude.degrees:.3f}° | Lon: {geo_b.longitude.degrees:.3f}°<extra></extra>"
        )
        fig.add_trace(go.Scatter3d(x=[float(mid_tca[0])], y=[float(mid_tca[1])], z=[float(mid_tca[2])], mode="markers", marker=dict(color="#ff2b4d", size=14, symbol="diamond", line=dict(color="#ffffff", width=2)), hovertemplate=tca_hover_html, name="TCA Point"))

    n_static = len(fig.data)

    def make_dynamic_traces(i):
        pa, pb, d_val = pos_a[:, i], pos_b[:, i], dists[i]
        dc = dist_color(d_val)
        beam_width = 4 if not np.isnan(d_val) and d_val < 200 else 3
        tr0 = go.Scatter3d(x=[float(pa[0])] if not np.isnan(pa[0]) else [None], y=[float(pa[1])] if not np.isnan(pa[1]) else [None], z=[float(pa[2])] if not np.isnan(pa[2]) else [None], mode="markers", marker=dict(color="#ffffff", size=10, symbol="diamond", line=dict(color="#00c8ff", width=3), opacity=1.0), name=sat_a.name, showlegend=False, hoverinfo="skip")
        tr1 = go.Scatter3d(x=[float(pb[0])] if not np.isnan(pb[0]) else [None], y=[float(pb[1])] if not np.isnan(pb[1]) else [None], z=[float(pb[2])] if not np.isnan(pb[2]) else [None], mode="markers", marker=dict(color="#ffffff", size=10, symbol="circle", line=dict(color="#ff6b00", width=3), opacity=1.0), name=sat_b.name, showlegend=False, hoverinfo="skip")
        tr2 = go.Scatter3d(x=[float(pa[0]), float(pb[0])] if not np.isnan(pa[0]) and not np.isnan(pb[0]) else [None, None], y=[float(pa[1]), float(pb[1])] if not np.isnan(pa[1]) and not np.isnan(pb[1]) else [None, None], z=[float(pa[2]), float(pb[2])] if not np.isnan(pa[2]) and not np.isnan(pb[2]) else [None, None], mode="lines", line=dict(color=dc, width=beam_width, dash="solid"), name="Distance", showlegend=False, hoverinfo="skip")
        return [tr0, tr1, tr2]

    for tr in make_dynamic_traces(0): fig.add_trace(tr)

    n_dynamic = len(fig.data) - n_static
    if n_dynamic == 0: return fig, tca_idx, tca_dist, dists, anim_jd, tca_tt_val

    dyn_indices = list(range(n_static, n_static + n_dynamic))
    frames = []
    slider_steps = []

    for i in range(n_frames):
        t_utc = ts.tt_jd(anim_jd[i]).utc_strftime("%H:%M UTC")
        t_min = (i - tca_idx) * step_min
        time_lbl = f"T+{t_min}" if t_min >= 0 else f"T{t_min}"
        warn_tag = "<span style='color:#ff2b4d;'> ⚠ TCA ANIDIR</span>" if i == tca_idx else ""
        title_txt = f"<b>{sat_a.name} × {sat_b.name}</b> <br><sup style='color:#b8cfe0;'>⏱ {time_lbl} min &nbsp;|&nbsp; {t_utc} &nbsp;|&nbsp; Mesafe: {dists[i]:.1f} km{warn_tag}</sup>"
        frames.append(go.Frame(data=make_dynamic_traces(i), traces=dyn_indices, name=str(i), layout=go.Layout(title_text=title_txt)))
        lbl = t_utc if i % max(1, n_frames // 20) == 0 else ""
        # 0 Transition optimizasyonu: WebGL kasmasını engeller!
        slider_steps.append(dict(args=[[str(i)], dict(frame=dict(duration=0, redraw=True), transition=dict(duration=0), mode="immediate")], label=lbl, method="animate"))

    fig.frames = frames
    base_t_utc = ts.tt_jd(anim_jd[0]).utc_strftime("%H:%M UTC")
    base_t_min = (0 - tca_idx) * step_min
    base_title = f"<b>{sat_a.name} × {sat_b.name}</b> <br><sup style='color:#b8cfe0;'>⏱ {'T+' if base_t_min >= 0 else 'T'}{base_t_min} min &nbsp;|&nbsp; {base_t_utc} &nbsp;|&nbsp; Mesafe: {dists[0]:.1f} km</sup>"

    fig.update_layout(
        **DARK, height=680, margin=dict(l=0, r=0, t=80, b=10),
        title=dict(text=base_title, font=dict(family="Barlow Condensed", color="#00c8ff", size=16), x=0.01, y=0.97),
        scene=dict(bgcolor="#000408", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="cube", camera=tca_camera),
        legend=dict(font=dict(size=9, family="Space Mono"), bgcolor="rgba(0,4,8,.85)", bordercolor="#1a2740", borderwidth=1, x=0.01, y=0.90, itemsizing="constant"),
        updatemenus=[dict(type="buttons", showactive=True, bgcolor="#0c1018", bordercolor="#1a2740", font=dict(family="Space Mono", size=9, color="#b8cfe0"), y=1.02, x=0.5, xanchor="center", pad=dict(r=4), direction="left",
                buttons=[
                    dict(label="▶ 1x", method="animate", args=[[str(k) for k in range(n_frames)], dict(frame=dict(duration=frame_duration, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                    dict(label="⏩ 2x", method="animate", args=[[str(k) for k in range(0, n_frames, 2)], dict(frame=dict(duration=frame_duration//2, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                    dict(label="⏸ DUR", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=True), transition=dict(duration=0), mode="immediate")]),
                    dict(label="🎯 TCA'YA GİT", method="animate", args=[[str(tca_idx)], dict(frame=dict(duration=0, redraw=True), transition=dict(duration=0), mode="immediate")])])],
        sliders=[dict(steps=slider_steps, active=0, currentvalue=dict(prefix="⏱  ", font=dict(family="Space Mono", size=10, color="#4a6880")), pad=dict(t=64, b=0), len=0.92, x=0.04, bgcolor="#0c1018", bordercolor="#1a2740", tickcolor="#1a2740", font=dict(color="#4a6880", size=8))]
    )
    return fig, tca_idx, tca_dist, dists, anim_jd, tca_tt_val

# ================================================================================
#  INTERFACE & TABS
# ================================================================================
st.set_page_config(page_title="StarWeb-CARA: Conjunction Assessment", page_icon="S", layout="wide", initial_sidebar_state="expanded")
st.markdown(get_theme_css(st.session_state.get("theme", "dark")), unsafe_allow_html=True)
st.markdown("""<div style="padding:24px 0 12px 0; border-bottom:2px solid #1e2d42; margin-bottom:24px; position:relative;"><div style="font-family:'Space Mono',monospace; font-size:.7rem; color:#5a7a94; letter-spacing:.25em; text-transform:uppercase; margin-bottom:8px; background: linear-gradient(90deg, #00d4ff 0%, #00ffa8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Conjunction Assessment and Collision Risk Analysis</div><h1 style="margin:0; font-size:2rem; line-height:1.3;">Low Earth Orbit<br><span style="color:#00d4ff;">Conjunction Assessment &amp; Collision Risk Analysis</span></h1><div style="font-family:'Inter',sans-serif; font-size:.9rem; color:#5a7a94; margin-top:12px; letter-spacing:.06em; font-weight:400;">Space Sciences and Technologies Graduation Project · Space-Track GP Database · Skyfield SGP4 Propagator</div><div style="position:absolute; top:0; right:0; width:100px; height:4px; background: linear-gradient(90deg, #00d4ff 0%, #00ffa8 100%); border-radius:2px;"></div></div>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="padding:16px 0 20px 0; border-bottom:2px solid #1e2d42; margin-bottom:20px;"><div style="font-family:'Barlow Condensed',sans-serif; font-size:1.4rem; font-weight:700; color:#00d4ff; letter-spacing:.08em; text-transform:uppercase; margin-bottom:4px;">CONTROL PANEL</div><div style="font-family:'Space Mono',monospace; font-size:.65rem; color:#5a7a94; letter-spacing:.15em; text-transform:uppercase;">System Configuration</div></div>""", unsafe_allow_html=True)
st.sidebar.markdown("""<div style="font-family:'Space Mono',monospace;font-size:.7rem; letter-spacing:.18em;color:#00d4ff;text-transform:uppercase; border-bottom:1px solid #1e2d42;padding-bottom:6px;margin-bottom:12px;">THEME SELECTION</div>""", unsafe_allow_html=True)
if "theme" not in st.session_state: st.session_state.theme = "dark"
theme_options = {"dark": {"label": "🌙 Dark Mission Control", "desc": "Professional dark theme with cyan accents"}, "light": {"label": "☀️ Professional Light", "desc": "Clean black & white theme with blue accents"}}
selected_theme = st.sidebar.selectbox("Choose Theme", options=list(theme_options.keys()), format_func=lambda x: theme_options[x]["label"], index=0 if st.session_state.theme == "dark" else 1, key="theme_selector")
if selected_theme != st.session_state.theme:
    st.session_state.theme = selected_theme
    st.rerun()
st.sidebar.caption(theme_options[selected_theme]["desc"])
st.sidebar.markdown("""<div style="height:1px; background:linear-gradient(90deg, transparent 0%, #1e2d42 50%, transparent 100%); margin:20px 0;"></div>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="font-family:'Space Mono',monospace;font-size:.7rem; letter-spacing:.18em;color:#00d4ff;text-transform:uppercase; border-bottom:1px solid #1e2d42;padding-bottom:6px;margin-bottom:12px;">AUTO TLE DOWNLOAD</div>""", unsafe_allow_html=True)
st.sidebar.markdown("""<div style="font-family:'Inter',sans-serif; font-size:.85rem; color:#c4d4e8; font-weight:600; margin-bottom:8px;">Space-Track Authentication</div>""", unsafe_allow_html=True)
user_email = st.sidebar.text_input("Email", placeholder="user@domain.com")
user_pass = st.sidebar.text_input("Password", placeholder="........", type="password")
st.sidebar.markdown("**Target Satellite Constellation** *(focused on LEO fleets only)*")
search_term = st.sidebar.selectbox("Select constellation", list(GROUP_CONFIG.keys()), label_visibility="collapsed")
if st.sidebar.button("DOWNLOAD LIVE TLE DATA"):
    if user_email and user_pass:
        with st.spinner("📡 Connecting to data sources..."):
            try:
                download_limit = int(st.session_state.get("sat_limit", 15))
                result = fetch_tles_with_fallback(user_email, user_pass, search_term, download_limit)
                if result:
                    data = result["lines"]
                    st.session_state["tle_data"] = data
                    st.session_state["loaded_group"] = GROUP_CONFIG[search_term]["label"]
                    st.session_state["data_source"] = result["source"]
                    st.session_state["data_message"] = result["message"]
                    count = count_tle_objects(data)
                    pair_count = count * (count - 1) // 2
                    st.sidebar.success(f"✅ {count} satellites loaded • {pair_count} possible pairs")
                    st.sidebar.caption("🎯 Apsis filter will eliminate pairs with low physical intersection probability.")
                    st.sidebar.info(f"📊 Source: {result['source']}")
                    st.sidebar.caption(result["message"])
                else:
                    st.sidebar.error("❌ Failed to download TLE data. Please try again.")
            except Exception as e:
                st.sidebar.error(f"❌ Download error: {str(e)[:100]}")
    else:
        st.sidebar.warning("⚠️ Authentication required. Please enter your credentials.")
st.sidebar.markdown("""<div style="height:1px; background:linear-gradient(90deg, transparent 0%, #1e2d42 50%, transparent 100%); margin:20px 0;"></div>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="font-family:'Space Mono',monospace;font-size:.7rem; letter-spacing:.18em;color:#00ffa8;text-transform:uppercase; border-bottom:1px solid #1e2d42;padding-bottom:6px;margin-bottom:12px;">ENTER YOUR SATELLITE (TLE)</div>""", unsafe_allow_html=True)
st.sidebar.markdown("""<div style="font-family:'Inter',sans-serif; font-size:.75rem; color:#5a7a94; margin-bottom:10px; font-style:italic;">3-line TLE format (name + line1 + line2)</div>""", unsafe_allow_html=True)
manual_tle_text = st.sidebar.text_area("Manual TLE", height=110, placeholder="ISS (ZARYA)\n1 25544U 98067A   24065.52722916  .00016717  00000+0  32296-3 0  9994\n2 25544  51.6412  237.8783 0003724 100.6644  259.4049 15.50110392 44874", label_visibility="collapsed", key="manual_tle_input")
if st.sidebar.button("LOAD MANUAL TLE"):
    lines = [l.strip() for l in manual_tle_text.strip().split("\n") if l.strip()]
    if len(lines) >= 3:
        try:
            my_sat = EarthSatellite(lines[1], lines[2], lines[0], ts)
            st.session_state["my_sat"] = my_sat
            st.sidebar.success(f"✅ {my_sat.name} loaded successfully.")
        except Exception as e: st.sidebar.error(f"❌ TLE parsing error: {str(e)[:80]}")
    elif len(lines) == 2:
        try:
            my_sat = EarthSatellite(lines[0], lines[1], "CUSTOM-SAT", ts)
            st.session_state["my_sat"] = my_sat
            st.sidebar.success("✅ CUSTOM-SAT loaded successfully.")
        except Exception as e: st.sidebar.error(f"❌ TLE parsing error: {str(e)[:80]}")
    else: st.sidebar.warning("⚠️ Please enter at least 2 TLE lines.")

if "my_sat" in st.session_state:
    ms = st.session_state["my_sat"]
    st.sidebar.markdown(f"""<div style="font-family:'Space Mono',monospace;font-size:.7rem; color:#00ffa8;padding:10px 14px;background:rgba(0,255,168,.08); border:1px solid rgba(0,255,168,.25);border-radius:6px;margin-top:8px; box-shadow: 0 2px 8px rgba(0,255,168,0.15);"> ✓ ACTIVE: {ms.name}</div>""", unsafe_allow_html=True)
    if st.sidebar.button("Delete My Satellite"):
        del st.session_state["my_sat"]
        st.rerun()

st.sidebar.markdown("""<div style="height:1px; background:linear-gradient(90deg, transparent 0%, #1e2d42 50%, transparent 100%); margin:20px 0;"></div>""", unsafe_allow_html=True)
st.sidebar.markdown("""<div style="font-family:'Space Mono',monospace;font-size:.7rem; letter-spacing:.18em;color:#5a7a94;text-transform:uppercase; border-bottom:1px solid #1e2d42;padding-bottom:6px;margin-bottom:12px;">ANALYSIS PARAMETERS</div>""", unsafe_allow_html=True)

sync_mass_defaults(search_term)
selected_group_label = GROUP_CONFIG[search_term]["label"]
selected_group_mass = get_group_default_mass(search_term)

if "my_sat" in st.session_state: st.sidebar.markdown(f"""<div style="font-family:'Space Mono',monospace;font-size:.64rem;color:#4a6880; padding:6px 0 8px 0;line-height:1.6;">Object A default: <span style="color:#00ff9d;">MANUAL SAT • {MANUAL_SAT_DEFAULT_MASS_KG} kg</span><br>Object B default: <span style="color:#00c8ff;">{selected_group_label} • {selected_group_mass} kg</span></div>""", unsafe_allow_html=True)
else: st.sidebar.markdown(f"""<div style="font-family:'Space Mono',monospace;font-size:.64rem;color:#4a6880; padding:6px 0 8px 0;line-height:1.6;">Selected fleet: <span style="color:#00c8ff;">{selected_group_label} • {selected_group_mass} kg</span></div>""", unsafe_allow_html=True)

window_hrs = st.sidebar.slider("Analysis window (hours)", 1, 48, 24, key="sidebar_window_hrs")
sigma_km = st.sidebar.select_slider("Position uncertainty σ (km)", options=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0], value=0.5)
sat_limit = st.sidebar.slider("Maximum satellite count", 5, 30, 15, key="sat_limit")
hbr_km = st.sidebar.select_slider("Hard-Body Radius HBR (km)", options=[0.005, 0.010, 0.020, 0.050, 0.100], value=0.020)

mass_a_label = "Object A Mass (Manual Sat)" if "my_sat" in st.session_state else "Object A Mass (kg)"
mass_b_label = f"Object B Mass ({selected_group_label})" if "my_sat" in st.session_state else "Object B Mass (kg)"
st.sidebar.markdown("<small style='color:#4a6880;'>Allowed range: 1–500000 kg.</small>", unsafe_allow_html=True)

st.sidebar.markdown(f"**{mass_a_label}**")
mass_a_kg = st.sidebar.number_input("Write Object A Mass (kg)", min_value=1.0, max_value=MASS_WIDGET_MAX_KG, value=float(st.session_state.get("mass_a_input", st.session_state.get("mass_a_kg", selected_group_mass))), step=1.0, format="%.3f", key="mass_a_input", on_change=sync_mass_a_from_input)
st.sidebar.markdown(f"**{mass_b_label}**")
mass_b_kg = st.sidebar.number_input("Write Object B Mass (kg)", min_value=1.0, max_value=MASS_WIDGET_MAX_KG, value=float(st.session_state.get("mass_b_input", st.session_state.get("mass_b_kg", selected_group_mass))), step=1.0, format="%.3f", key="mass_b_input", on_change=sync_mass_b_from_input)

mass_a_kg = float(st.session_state.get("mass_a_kg", mass_a_kg))
mass_b_kg = float(st.session_state.get("mass_b_kg", mass_b_kg))

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Chan 1997 + Foster 1992\n**Propagator:** SGP4/SDP4\n**Filter:** Apsis + Distance\n**Data:** Space-Track GP + CelesTrak fallback\n**TCA Step:** 5 min\n**HBR:** User selected")

if "tle_data" not in st.session_state:
    st.info("📡 Download data by entering your Space-Track credentials in the left panel.")
    st.markdown("""<div class="info-panel"><b>🚀 Quick Start Guide:</b><br>1. Create a free account at <b>space-track.org</b>.<br>2. Enter your email and password in the left panel.<br>3. Select a satellite constellation and click <b>DOWNLOAD LIVE TLE DATA</b>.<br>4. If Space-Track fails, CelesTrak will be used automatically as backup.<br>5. All tabs will become active for analysis.</div>""", unsafe_allow_html=True)
    st.stop()

try:
    sats = parse_tles(st.session_state["tle_data"], limit=sat_limit, fallback_name_prefix=st.session_state.get("loaded_group"))
    if not sats:
        st.error("❌ TLE parsing failed. Please check your data source and try again.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error parsing TLE data: {str(e)[:100]}")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["DASHBOARD", "CONJUNCTION ANALYSIS", "YOUR SATELLITE", "LIVE SIMULATION", "3D ORBIT & GROUND TRACK", "ORBITAL ELEMENTS", "METHODOLOGY"])

with tab1:
    current_params = (window_hrs, sigma_km, hbr_km, mass_a_kg, mass_b_kg, st.session_state.theme)
    if "conj_df" not in st.session_state or st.session_state.get("conj_params") != current_params:
        with st.spinner("🚀 Computing conjunction analysis with apsis filter..."):
            start_time = time.time()
            df, n_filtered, n_total = compute_conjunctions(sats, window_hrs, sigma_km, hbr_km, mass_a_kg, mass_b_kg, st.session_state.theme)
            computation_time = time.time() - start_time
            st.session_state.conj_df = df
            st.session_state.conj_n_filtered = n_filtered
            st.session_state.conj_n_total = n_total
            st.session_state.conj_computation_time = computation_time
            st.session_state.conj_params = current_params
    else:
        df = st.session_state.conj_df
        n_filtered = st.session_state.conj_n_filtered
        n_total = st.session_state.conj_n_total
        computation_time = st.session_state.conj_computation_time

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(f"""<div style="font-family:'Space Mono',monospace; font-size:.7rem; color:#5a7a94; text-align:right; margin-bottom:16px; padding:8px 12px; background:rgba(90,122,148,.05); border-radius:6px; border:1px solid rgba(90,122,148,.15);">Last update: {now_str} · Computation time: {computation_time:.2f}s</div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🛰️ Tracked Satellites", len(sats))
    c2.metric("🔗 Total Pairs", n_total)
    c3.metric("✅ Passed Filter", n_total - n_filtered)
    c4.metric("⚠️ Conjunctions", len(df) if not df.empty else 0)
    c5.metric("🚨 Critical Risk", len(df[df["Risk Level"] == "CRITICAL"]) if not df.empty else 0)

    if n_filtered > 0:
        st.markdown(f"""<div class="info-panel"><b>🎯 Apsis Filter Performance:</b> {n_filtered} pairs filtered without orbit propagation due to non-overlapping altitude bands — computation time reduced by <span style="color:#00ffa8; font-weight:700;">{round(n_filtered / n_total * 100, 1)}%</span>.</div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if df.empty:
        st.success(f"✅ No conjunctions below 500 km detected in {window_hrs}-hour window. All clear!")
    else:
        n_dil = df["Dilution"].sum() if not df.empty else 0
        if n_dil > 0:
            st.markdown(f"""<div class="warn-panel"><b>PROBABILITY DILUTION WARNING:</b> Wide covariance in {int(n_dil)} events may be masking Pc values. Check Max-Pc values in Conjunction Analysis tab.</div>""", unsafe_allow_html=True)

        show_cols = ["TCA (UTC)", "Object A", "Object B", "Distance (km)", "Relative Velocity (km/s)", "Pc (scientific)", "Pc Max", "Mahalanobis Md", "Ec (J/g)", "Risk Level"]
        RISK_COLORS = {"CRITICAL": "#ff2b4d", "HIGH": "#ff6b00", "MEDIUM": "#ffaa00", "LOW": "#00ff9d"}
        MONO = "font-family:'Space Mono',monospace; font-size:0.76rem;"
        df_show = df[show_cols].copy()
        styled = (df_show.style.map(lambda v: f"color:{RISK_COLORS.get(str(v), '#b8cfe0')};font-weight:bold;{MONO}", subset=["Risk Level"])
            .map(lambda v: f"color:#00c8ff;{MONO}", subset=["Pc (scientific)"])
            .map(lambda v: f"color:#ff9060;{MONO}", subset=["Pc Max"])
            .map(lambda v: f"color:#ff2b4d;{MONO}" if float(v) < 1.5 else f"color:#b8cfe0;{MONO}", subset=["Mahalanobis Md"])
            .map(lambda v: f"color:#ff2b4d;{MONO}" if float(v) >= 40 else f"color:#b8cfe0;{MONO}", subset=["Ec (J/g)"])
            .format({"Distance (km)": "{:.3f}", "Relative Velocity (km/s)": "{:.3f}", "Pc Max": "{:.3e}", "Mahalanobis Md": "{:.2f}", "Ec (J/g)": "{:.1f}"})
            .set_properties(**{"font-family": "Space Mono,monospace", "font-size": "0.76rem"}))
        st.download_button("📥 Download Report as CSV", data=df_show.to_csv(index=False).encode("utf-8-sig"), file_name=f"conjunction_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
        st.dataframe(styled, use_container_width=True)

with tab2:
    if df is None or df.empty:
        st.success("✅ No critical conjunction events in selected window.")
    else:
        st.markdown("**🔍 Detailed Review — Select Event**")
        options = [f"{r['Object A']}  <->  {r['Object B']}  |  TCA {r['TCA (UTC)']}  |  {r['Distance (km)']} km" for _, r in df.iterrows()]
        sel = st.selectbox("Conjunction event", options, label_visibility="collapsed")
        idx = options.index(sel)
        row = df.iloc[idx]

        if row["Dilution"]: st.markdown(f"""<div class="crit-panel"><b>⚠️ PROBABILITY DILUTION:</b> {row["Dilution Message"]}</div>""", unsafe_allow_html=True)

        col_l, col_r = st.columns([2, 1])
        with col_l: st.plotly_chart(fig_distance_profile(row["_dist_arr"], window_hrs, row["Distance (km)"], sigma_km, hbr_km), use_container_width=True, key="dist_prof_tab2")
        with col_r: st.plotly_chart(fig_risk_gauge(row["Pc (isotropic)"]), use_container_width=True, key="risk_gauge_tab2")

        st.markdown("**📊 Collision Probability Model Comparison**")
        pc_cols = st.columns(3)
        pc_cols[0].metric("Chan 1997 (Isotropic)", f"{row['Pc (isotropic)']:.3e}")
        pc_cols[1].metric("Foster 1992 (2D-Pc)", f"{row['Pc (Foster 2D)']:.3e}")
        pc_cols[2].metric("Max Pc (Worst Case)", f"{row['Pc Max']:.3e}")

        mah_color = "#28a745" if row["2D-Pc Valid"] == "2D-Pc Valid" else "#dc3545"
        st.markdown(f"""<div class="info-panel"><b>🎯 Mahalanobis Distance Test:</b> Md = {row["Mahalanobis Md"]:.3f} — <span style="color:{mah_color}; font-weight:700;">{row["2D-Pc Valid"]}</span><br><small>Md < 1.5 → linear motion assumption breaks down → 3D-Pc required</small></div>""", unsafe_allow_html=True)

        frag = fragmentation_probability(row["Relative Velocity (km/s)"], mass_a_kg, mass_b_kg)
        st.markdown("**Collision Consequence Analysis**")
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Specific Kinetic Energy (J/g)", f"{frag['E_c_J_per_g']:.1f}")
        fc2.metric("Fragmentation Level", frag["level"])
        fc3.metric("Estimated Debris Objects", frag["est_debris"])
        st.markdown(f"""<div class="info-panel" style="border-left-color:{frag["color"]};"><b>{frag["level"]}:</b> {frag["desc"]}<br><small>Ec ≥ 40 J/g → Catastrophic fragmentation (Kessler Syndrome contribution)</small></div>""", unsafe_allow_html=True)

        st.markdown("**Full Event Parameters**")
        det = { "Object A": row["Object A"], "Object B": row["Object B"], "TCA (UTC)": row["TCA (UTC)"], "Miss Distance (km)": row["Distance (km)"], "Relative Velocity (km/s)": row["Relative Velocity (km/s)"], "Position Uncertainty sigma (km)": sigma_km, "Hard-Body Radius HBR (km)": hbr_km, "Pc — Chan 1997 Isotropic": f"{row['Pc (isotropic)']:.3e}", "Pc — Foster 1992 2D": f"{row['Pc (Foster 2D)']:.3e}", "Pc — Maximum (Worst Case)": f"{row['Pc Max']:.3e}", "Mahalanobis Distance Md": row["Mahalanobis Md"], "2D-Pc Validity": row["2D-Pc Valid"], "Probability Dilution": "YES" if row["Dilution"] else "NO", "Specific Kinetic Energy (J/g)": row["Ec (J/g)"], "Fragmentation Level": row["Fragmentation Level"], "Estimated Debris Objects": row["Estimated Debris"], "Risk Level": row["Risk Level"] }
        st.dataframe(pd.DataFrame(det.items(), columns=["Parameter", "Value"]), use_container_width=True, hide_index=True)
        st.markdown("---")
        if st.button("🔭 Show This Pair in Live Simulation", key="tab2_to_sim"):
            queue_simulation_pair(row["_s1"], row["_s2"], row["_tca_tt"])
            st.success("Pair transferred to 'LIVE SIMULATION' tab with TCA-centered timing.")

with tab3:
    st.markdown("## Analyze Your Satellite")
    if "my_sat" not in st.session_state:
        st.markdown("""<div class="warn-panel"><b>You haven't loaded your satellite yet.</b><br>Enter your TLE data in the <b>2 — ENTER YOUR SATELLITE (TLE)</b> section in the left panel and click <b>LOAD MANUAL TLE</b>.</div>""", unsafe_allow_html=True)
    elif "tle_data" not in st.session_state:
        st.markdown("""<div class="warn-panel"><b>Fleet data not loaded.</b><br>First perform automatic TLE download from the left panel; then comparison with your satellite can be done.</div>""", unsafe_allow_html=True)
    else:
        my_sat = st.session_state["my_sat"]
        st.markdown(f"""<div class="info-panel"><b>Active satellite:</b> {my_sat.name}&nbsp;&nbsp;|&nbsp;&nbsp;<b>Fleet:</b> {st.session_state.get("loaded_group", "—")}&nbsp;&nbsp;|&nbsp;&nbsp;<b>Manual mass A:</b> {mass_a_kg} kg&nbsp;&nbsp;|&nbsp;&nbsp;<b>Fleet mass B:</b> {mass_b_kg} kg&nbsp;&nbsp;|&nbsp;&nbsp;<b>Analysis window:</b> {window_hrs} hours&nbsp;&nbsp;|&nbsp;&nbsp;<b>σ:</b> {sigma_km} km</div>""", unsafe_allow_html=True)
        with st.spinner(f"Running conjunction analysis for {my_sat.name}..."):
            df_my = compute_conjunctions_custom(my_sat, sats, window_hrs, sigma_km, hbr_km, mass_a_kg, mass_b_kg, st.session_state.theme)
        if df_my.empty:
            st.success(f"No conjunctions below 500 km for {my_sat.name} in {window_hrs}-hour window.")
        else:
            c1m, c2m, c3m, c4m = st.columns(4)
            c1m.metric("Total Conjunctions", len(df_my))
            c2m.metric("Critical Risk", len(df_my[df_my["Risk Level"] == "CRITICAL"]))
            c3m.metric("High Risk", len(df_my[df_my["Risk Level"] == "HIGH"]))
            c4m.metric("Min. Distance (km)", f"{df_my['Distance (km)'].min():.2f}")
            st.markdown("**Conjunctions — Risk Table**")
            show_c = ["TCA (UTC)", "Object A", "Object B", "Distance (km)", "Relative Velocity (km/s)", "Pc (scientific)", "Pc Max", "Mahalanobis Md", "Ec (J/g)", "Risk Level"]
            df_my_show = df_my[show_c].copy()
            styled_my = (df_my_show.style.map(lambda v: f"color:{RISK_COLORS.get(str(v), '#b8cfe0')};font-weight:bold;{MONO}", subset=["Risk Level"])
                .map(lambda v: f"color:#00c8ff;{MONO}", subset=["Pc (scientific)"])
                .map(lambda v: f"color:#ff9060;{MONO}", subset=["Pc Max"])
                .map(lambda v: f"color:#ff2b4d;{MONO}" if float(v) < 1.5 else f"color:#b8cfe0;{MONO}", subset=["Mahalanobis Md"])
                .format({"Distance (km)": "{:.3f}", "Relative Velocity (km/s)": "{:.3f}", "Pc Max": "{:.3e}", "Mahalanobis Md": "{:.2f}", "Ec (J/g)": "{:.1f}"})
                .set_properties(**{"font-family": "Space Mono,monospace", "font-size": "0.76rem"}))
            st.download_button("Download Report as CSV", data=df_my_show.to_csv(index=False).encode("utf-8-sig"), file_name=f"my_satellite_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
            st.dataframe(styled_my, use_container_width=True)

            st.markdown("---")
            st.markdown("**Detailed Pair Analysis — Select Event**")
            opts_my = [f"{r['Object B']}  |  TCA {r['TCA (UTC)']}  |  {r['Distance (km)']} km" for _, r in df_my.iterrows()]
            sel_my = st.selectbox("Select event", opts_my, label_visibility="collapsed", key="my_sel")
            idx_my = opts_my.index(sel_my)
            row_my = df_my.iloc[idx_my]

            if row_my["Dilution"]: st.markdown(f"""<div class="crit-panel"><b>PROBABILITY DILUTION:</b> {row_my["Dilution Message"]}</div>""", unsafe_allow_html=True)
            col_l, col_r = st.columns([2, 1])
            with col_l: st.plotly_chart(fig_distance_profile(row_my["_dist_arr"], window_hrs, row_my["Distance (km)"], sigma_km, hbr_km), use_container_width=True, key="dist_prof_tab3")
            with col_r: st.plotly_chart(fig_risk_gauge(row_my["Pc (isotropic)"]), use_container_width=True, key="risk_gauge_tab3")

            pc_c = st.columns(3)
            pc_c[0].metric("Chan 1997 (Isotropic)", f"{row_my['Pc (isotropic)']:.3e}")
            pc_c[1].metric("Foster 1992 (2D-Pc)", f"{row_my['Pc (Foster 2D)']:.3e}")
            pc_c[2].metric("Max Pc", f"{row_my['Pc Max']:.3e}")
            mah_c = "#28a745" if row_my["2D-Pc Valid"] == "2D-Pc Valid" else "#dc3545"
            st.markdown(f"""<div class="info-panel"><b>Mahalanobis Test:</b> Md = {row_my["Mahalanobis Md"]:.3f} — <span style="color:{mah_c};">{row_my["2D-Pc Valid"]}</span></div>""", unsafe_allow_html=True)

            frag_my = fragmentation_probability(row_my["Relative Velocity (km/s)"], mass_a_kg, mass_b_kg)
            fc = st.columns(3)
            fc[0].metric("Ec (J/g)", f"{frag_my['E_c_J_per_g']:.1f}")
            fc[1].metric("Fragmentation", frag_my["level"])
            fc[2].metric("Estimated Debris", frag_my["est_debris"])
            st.markdown("---")
            if st.button("🔭 Show This Pair in Live Simulation", key="my_to_sim"):
                queue_simulation_pair(row_my["_s1"], row_my["_s2"], row_my["_tca_tt"])
                st.success("Pair transferred! Switch to the 'LIVE SIMULATION' tab to view.")

with tab4:
    st.markdown("## 🚀 Conjunction Encounter Simulator")
    st.markdown("""<div class="info-panel"><b>🔬 Real‑time Encounter Animation</b><br>Animation steps are strictly set to <b>5 minutes</b>. The window is automatically centered around the Time of Closest Approach (TCA).</div>""", unsafe_allow_html=True)
    sat_names_ext = [st.session_state["my_sat"].name] + [s.name for s in sats] if "my_sat" in st.session_state else [s.name for s in sats]
    all_sats_ext = [st.session_state["my_sat"]] + sats if "my_sat" in st.session_state else sats

    def_idx_a, def_idx_b = 0, min(1, len(sat_names_ext) - 1) if len(sat_names_ext) > 1 else 0
    queued_sat_a, queued_sat_b, center_tt = st.session_state.get("sim_sat_a"), st.session_state.get("sim_sat_b"), st.session_state.get("sim_center_tt")

    if queued_sat_a and queued_sat_a.name in sat_names_ext: def_idx_a = sat_names_ext.index(queued_sat_a.name)
    elif "live_sel_a" in st.session_state and st.session_state.live_sel_a in sat_names_ext: def_idx_a = sat_names_ext.index(st.session_state.live_sel_a)
    if queued_sat_b and queued_sat_b.name in sat_names_ext: def_idx_b = sat_names_ext.index(queued_sat_b.name)
    elif "live_sel_b" in st.session_state and st.session_state.live_sel_b in sat_names_ext: def_idx_b = sat_names_ext.index(st.session_state.live_sel_b)

    c1, c2, c3 = st.columns([2, 2, 1])
    sat_a_name = c1.selectbox("🛰️ Satellite A", sat_names_ext, index=def_idx_a, key="live_sel_a")
    sat_b_name = c2.selectbox("🛰️ Satellite B", sat_names_ext, index=def_idx_b, key="live_sel_b")

    # SAAT PROBLEMI ÇÖZÜLDÜ: Window hrs varsayılan olarak yan menüden geliyor.
    sim_hours = c3.slider("⏱️ Window (hrs)", 1, 48, window_hrs, key="live_window_hrs")

    with st.expander("⚙️ Display Options"):
        show_orbits = st.checkbox("Orbit Trails", value=True)
        show_tca = st.checkbox("TCA Risk Zone", value=True)
        anim_speed = st.slider("Playback Speed", 0.5, 3.0, 1.0, 0.1)

    auto_run = st.session_state.pop("run_sim", False)

    if st.button("🚀 GENERATE SIMULATION", use_container_width=True, disabled=(sat_a_name == sat_b_name)) or auto_run:
        if sat_a_name == sat_b_name:
            st.warning("Please select two different satellites.")
        else:
            sat_a = next(s for s in all_sats_ext if s.name == sat_a_name)
            sat_b = next(s for s in all_sats_ext if s.name == sat_b_name)
            if queued_sat_a and queued_sat_b and (sat_a.name != queued_sat_a.name or sat_b.name != queued_sat_b.name): center_tt = None

            with st.spinner("🧮 Computing 3D encounter geometry..."):
                frame_dur = int(60 / anim_speed)
                fig, tca_idx, tca_dist, dists, jd_arr, tca_tt_val = fig_animated_conjunction(sat_a, sat_b, sim_hours, show_orbits, show_tca, center_tt, frame_dur)

            tca_utc = ts.tt_jd(tca_tt_val).utc_strftime("%Y-%m-%d %H:%M:%S UTC")
            sev, col = risk_level(collision_probability_isotropic(tca_dist, sigma_km, hbr_km))
            tca_tplus = int(round((tca_tt_val - jd_arr[0]) * 1440))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("TCA Time", tca_utc)
            m2.metric("Min. Distance", f"{tca_dist:.3f} km")
            m3.metric("TCA @ T+", f"{tca_tplus} min")
            m4.metric("Risk Level", sev)
            st.info("💡 Use the **[🎯 TCA'YA GİT]** button to jump directly to the closest approach! Hover over the red diamond at TCA for exact metrics.")
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})

            st.markdown("**📊 Distance Profile**")
            t_ax = (jd_arr - jd_arr[0]) * 24.0
            fig_dp = go.Figure()
            fig_dp.add_hline(y=hbr_km, line=dict(color="#ff2b4d", dash="dot"), annotation_text=f"HBR ({hbr_km * 1000:.0f} m)")
            fig_dp.add_trace(go.Scatter(x=t_ax, y=dists, mode="lines", line=dict(color="#00c8ff", width=1.5), fill="tozeroy", fillcolor="rgba(0,200,255,.04)"))
            fig_dp.add_trace(go.Scatter(x=[t_ax[tca_idx]], y=[tca_dist], mode="markers+text", marker=dict(color="#ff2b4d", size=10), text=[f"TCA {tca_dist:.1f} km"], textfont=dict(size=10, color="#ff2b4d"), textposition="top right", showlegend=False))
            fig_dp.update_layout(**DARK, height=280, xaxis=dict(title="Time (hours)", gridcolor="#1a2740"), yaxis=dict(title="Distance (km)", gridcolor="#1a2740"), margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_dp, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("👆 Configure and press GENERATE SIMULATION to start the encounter animation.")

with tab5:
    display_sats = sats.copy()
    if "my_sat" in st.session_state: display_sats.insert(0, st.session_state["my_sat"])
    col_3d, col_gnd = st.columns([3, 2], gap="medium")
    with col_3d:
        st.markdown("**🛰️ 3D Orbit View**")
        with st.spinner("🌀 Loading Earth texture and propagating orbits…"):
            st.plotly_chart(fig_3d_orbits(display_sats), use_container_width=True, config={"scrollZoom": True, "displayModeBar": False}, key="3d_orbit_tab5")
    with col_gnd:
        st.markdown("**📍 Ground Track Map**")
        with st.spinner("🌐 Computing ground tracks…"):
            st.plotly_chart(fig_ground_tracks(display_sats), use_container_width=True, config={"scrollZoom": True, "displayModeBar": False}, key="ground_track_tab5")
        st.markdown(f"""<div style="font-family:'Space Mono',monospace; font-size:.65rem; color:{"#4a6880" if st.session_state.theme == "dark" else "#6c757d"}; line-height:2; margin-top:12px; padding:12px; background:{"rgba(10,15,24,.6)" if st.session_state.theme == "dark" else "#f8f9fa"}; border:1px solid {"#1e2d42" if st.session_state.theme == "dark" else "#dee2e6"}; border-radius:6px;"><span style="color:#00c8ff;">●</span> 95‑minute orbital paths shown.<br><span style="color:#00ffa8;">●</span> Large markers indicate current positions.<br><span style="color:#ffaa00;">●</span> Propagated with SGP4/SDP4.</div>""", unsafe_allow_html=True)

with tab6:
    st.markdown("## 📊 Orbital Elements & Space Distribution")
    st.caption("Keplerian elements and visual distribution of the fleet.")
    display_sats = sats.copy()
    if "my_sat" in st.session_state: display_sats.insert(0, st.session_state["my_sat"])
    elems_list = [(sat.name, get_orbital_elements(sat)) for sat in display_sats]

    left_col, right_col = st.columns([1, 2], gap="medium")
    with left_col:
        st.markdown("**🪐 Altitude vs Inclination**")
        st.caption("Bubble size ∝ eccentricity")
        st.plotly_chart(fig_orbital_elements_radar(elems_list), use_container_width=True, config={"displayModeBar": False}, key="radar_tab6")
    with right_col:
        st.markdown("**🛸 Fleet Orbital Table**")
        rows = []
        for name, elems in elems_list:
            if elems: rows.append({"Satellite": name[:20], "Alt (km)": elems.get("Mean Altitude (km)", "—"), "Inc (°)": elems.get("Inclination i (°)", "—"), "e": elems.get("Eccentricity e", "—"), "Period (min)": elems.get("Orbital Period (min)", "—")})
        if rows:
            df_elems = pd.DataFrame(rows)
            styled_df = (df_elems.style.set_properties(subset=["Satellite"], **{"font-family": "Space Mono, monospace", "color": "#00c8ff" if st.session_state.theme == "dark" else "#0066cc", "font-weight": "600"})
                .set_properties(subset=["Alt (km)", "Inc (°)", "e", "Period (min)"], **{"font-family": "Space Mono, monospace", "color": "#b8cfe0" if st.session_state.theme == "dark" else "#212529"})
                .set_table_styles([{"selector": "th", "props": [("font-family", "Barlow Condensed, sans-serif"), ("font-weight", "700"), ("color", "#00d4ff" if st.session_state.theme == "dark" else "#0066cc"), ("text-transform", "uppercase"), ("letter-spacing", "0.08em")]}, {"selector": "td", "props": [("padding", "8px 12px")]}]))
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("No orbital elements could be extracted.")
    st.markdown("---")
    with st.expander("🔬 Detailed View & Export Selected Satellite", expanded=False):
        sel_sat_name = st.selectbox("Select satellite to inspect and export", [name for name, _ in elems_list], key="elem_detail")
        if sel_sat_name:
            sel_elems = next((e for n, e in elems_list if n == sel_sat_name), {})
            if sel_elems:
                st.markdown(f"### 🛰️ Orbital Profile: `{sel_sat_name}`")
                c1, c2, c3 = st.columns(3)
                c1.metric("Semi-major Axis (a)", f"{sel_elems.get('Semi-major Axis a (km)', '—')} km")
                c2.metric("Mean Altitude", f"{sel_elems.get('Mean Altitude (km)', '—')} km")
                c3.metric("Orbital Period", f"{sel_elems.get('Orbital Period (min)', '—')} min")
                c4, c5, c6 = st.columns(3)
                c4.metric("Inclination (i)", f"{sel_elems.get('Inclination i (°)', '—')}°")
                c5.metric("Eccentricity (e)", f"{sel_elems.get('Eccentricity e', '—')}")
                c6.metric("Mean Motion (n)", f"{sel_elems.get('Mean Motion n (rev/min)', '—')} rev/m")
                c7, c8, c9 = st.columns(3)
                c7.metric("RAAN (Ω)", f"{sel_elems.get('RAAN (°)', '—')}°")
                c8.metric("Arg. of Perigee (ω)", f"{sel_elems.get('Arg of Perigee ω (°)', '—')}°")
                c9.metric("Mean Anomaly (M)", f"{sel_elems.get('Mean Anomaly M (°)', '—')}°")
                st.markdown("<br>", unsafe_allow_html=True)
                export_dict = {"Satellite Name": sel_sat_name}
                export_dict.update(sel_elems)
                st.download_button(label="📥 Download Elements to Excel (CSV)", data=pd.DataFrame([export_dict]).to_csv(index=False).encode("utf-8-sig"), file_name=f"orbital_elements_{sel_sat_name.replace(' ', '_')}.csv", mime="text/csv", use_container_width=True)
            else: st.warning("Orbital elements not available for this satellite.")
    st.markdown(f"""<div style="font-family:'Space Mono',monospace;font-size:.65rem; color:{"#4a6880" if st.session_state.theme == "dark" else "#6c757d"}; text-align:right; margin-top:24px;">Elements extracted from TLE at epoch · SGP4/SDP4 propagator</div>""", unsafe_allow_html=True)

with tab7:
    st.markdown("## 📚 Methodology and Theoretical Background")
    st.markdown(
        """
        <div class="info-panel">
        <b>🛰️ 1. Orbit Propagation — SGP4/SDP4 (Skyfield)</b><br>
        The NORAD standard <b>Simplified General Perturbations-4 (SGP4)</b> model is used to convert TLE data into Cartesian position vectors. SGP4 approximately accounts for gravity harmonics, atmospheric drag, and Sun/Moon third-body effects. SDP4 automatically engages for deep-space objects.<br>
        <i style="color:#b8cfe0;">Note: While pure Python/Skyfield operates at ~1M steps/sec, large-scale operational tools benefit from SIMD/Rust implementations (e.g., Astrora) reaching 15M+ steps/sec.</i>
        </div>

        <div class="info-panel">
        <b>🎯 2. Apsis Filter — Section 2.1 (ESA/NASA Standard)</b><br>
        A pre-filter that reduces O(N²) computational complexity. If the first object's perigee altitude (q₁) is higher than the second object's apogee altitude (Q₂), physical intersection is impossible:<br><br>
        <code style="background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px; color: #00d4ff;">max(q₁, q₂) > min(Q₁, Q₂) + D_threshold</code><br><br>
        This eliminates pairs with non-overlapping altitude bands before costly propagation.
        </div>

        <div class="info-panel">
        <b>⏱️ 3. Time of Closest Approach (TCA) Detection</b><br>
        Euclidean distance is calculated throughout the window using <b>fixed 5-minute discrete steps</b>. The global minimum in this array determines the TCA. The Live Simulator isolates a specific time window centered precisely on this TCA for high-resolution visual analysis.
        </div>

        <div class="info-panel">
        <b>📊 4. Collision Probability (Pc) Models</b><br>
        <b>• Chan (1997) Isotropic:</b> A fast closed-form approximation assuming spherical position uncertainty. Serves as a robust fallback.<br>
        <b>• Foster & Estes (1992) 2D-Pc:</b> The industry standard. Collision integration is reduced to 2D by projecting uncertainties onto the <b>encounter plane</b>. Evaluates the integral of the combined Gaussian probability density over the Hard-Body Radius (HBR) area.
        </div>

        <div class="info-panel">
        <b>📐 5. Mahalanobis Distance Test (CARA Methodology)</b><br>
        Evaluates the validity of the 2D-Pc assumptions (linear relative motion and short encounter duration).<br>
        Mahalanobis Distance: <code style="color: #00ffa8; background: transparent;">Md = miss_distance / σ</code><br>
        • Md &lt; 0.5 → <span style="color:#ff2b4d; font-weight:bold;">INVALID</span> (3D-Pc / Monte Carlo required)<br>
        • Md &lt; 1.5 → <span style="color:#ffaa00; font-weight:bold;">BORDERLINE</span><br>
        • Md ≥ 1.5 → <span style="color:#00ff9d; font-weight:bold;">VALID</span>
        </div>

        <div class="info-panel">
        <b>⚠️ 6. Probability Dilution</b><br>
        A mathematical anomaly where extreme position uncertainty (wide covariance) spreads the probability density so thinly that the Pc drops to "safe" levels, creating <b>false confidence</b>. Detected when <code style="color: #ffb800; background: transparent;">σ > 5 × miss_distance</code> and <code style="color: #ffb800; background: transparent;">Pc < 1e-6</code>. <b>Max-Pc</b> (worst-case scenario covariance) is calculated alongside standard Pc to flag dilution.
        </div>

        <div class="info-panel">
        <b>🚨 7. Risk Classification (NASA STD-8719.14)</b><br>
        • <span style="color:#ff2b4d; font-weight:bold;">Pc &gt; 1×10⁻³ → CRITICAL:</span> Collision Avoidance Maneuver (CAM) mandatory.<br>
        • <span style="color:#ff6b00; font-weight:bold;">Pc &gt; 1×10⁻⁴ → HIGH:</span> CAM evaluation required.<br>
        • <span style="color:#ffaa00; font-weight:bold;">Pc &gt; 1×10⁻⁵ → MEDIUM:</span> Increased tracking frequency.<br>
        • <span style="color:#00ff9d; font-weight:bold;">Pc ≤ 1×10⁻⁵ → LOW:</span> Routine tracking sufficient.
        </div>

        <div class="info-panel">
        <b>💥 8. Fragmentation Consequence (Pf)</b><br>
        Assesses the severity of a potential collision using Specific Kinetic Energy (Ec).<br>
        Formula: <code style="color: #00d4ff; background: transparent;">Ec = ½ · m_b · v_rel² / m_a (J/g)</code><br>
        • <b>Ec ≥ 40 J/g:</b> Catastrophic fragmentation (triggers Kessler Syndrome dynamics).<br>
        • <b>Ec ≥ 10 J/g:</b> Severe damage and significant debris cloud.<br>
        • <b>Ec ≥ 1 J/g:</b> Partial payload/subsystem damage.
        </div>

        <div class="info-panel">
        <b>📡 9. Data Source & Operational Standards</b><br>
        TLE data is fetched from the <b>Space-Track GP</b> endpoint (18th Space Defense Squadron). CelesTrak acts as a reliable fallback.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📖 References")
    st.markdown(
        """
        <div style="font-family:'Space Mono',monospace; font-size:.7rem; color:#5a7a94; line-height:2.0; padding: 16px; background: rgba(10,15,24,0.4); border-radius: 8px; border: 1px solid #1e2d42;">
        <b style="color:#c4d4e8;">Foster, J.L. & Estes, H.S. (1992).</b> A parametric analysis of orbital debris collision probability and maneuver rate for space vehicles. <i>NASA Technical Memorandum.</i><br>
        <b style="color:#c4d4e8;">Chan, F.K. (1997).</b> <i>Spacecraft Collision Probability.</i> The Aerospace Press.<br>
        <b style="color:#c4d4e8;">Hoots, F.R. & Roehrich, R.L. (1980).</b> <i>Models for Propagation of NORAD Element Sets.</i> Spacetrack Report No. 3.<br>
        <b style="color:#c4d4e8;">NASA (2023).</b> <i>Spacecraft Conjunction Assessment and Collision Avoidance Best Practices Handbook.</i> CARA Handbook Rev. 1.<br>
        <b style="color:#c4d4e8;">NASA (2011).</b> <i>Process for Limiting Orbital Debris.</i> NASA-STD-8719.14A.<br>
        <b style="color:#c4d4e8;">Alfriend, K.T. & Akella, M.R. (2000).</b> Probability of Collision Between Space Objects. <i>J. Guidance, Control, and Dynamics</i>, 23(5), 769–772.<br>
        <b style="color:#c4d4e8;">ESA (2011).</b> Efficient All vs. All Collision Risk Analyses — Smart Sieve Algorithm. <i>ISSFD Proceedings.</i><br>
        <b style="color:#c4d4e8;">Vallado, D.A. (2013).</b> <i>Fundamentals of Astrodynamics and Applications.</i> 4th ed. Microcosm Press.<br>
        <b style="color:#c4d4e8;">Hall, D.T. et al. (2023).</b> A Multistep Probability of Collision Computational Algorithm. <i>NASA NTRS.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )
