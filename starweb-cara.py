import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from skyfield.api import load, EarthSatellite, wgs84
from itertools import combinations
from spacetrack import SpaceTrackClient
import spacetrack.operators as op
from scipy.stats import norm, chi2
from scipy.integrate import dblquad
from datetime import datetime, timezone
import requests
from PIL import Image
from io import BytesIO
import math

ts = load.timescale()

# ================================================================================
#  CSS — MISSION CONTROL DARK THEME
# ================================================================================
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap');
:root {
  --bg:#07090f; --bg2:#0c1018; --bg3:#131b28; --border:#1a2740;
  --accent:#00c8ff; --accent2:#00ff9d; --warn:#ffaa00; --crit:#ff2b4d;
  --text:#b8cfe0; --dim:#4a6880;
  --mono:'Space Mono',monospace; --sans:'Barlow Condensed',sans-serif;
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#07090f !important;color:var(--text) !important;font-family:var(--sans) !important;}
[data-testid="stSidebar"]{background:var(--bg2) !important;border-right:1px solid var(--border) !important;}
[data-testid="stSidebar"] *{color:var(--text) !important;font-family:var(--sans) !important;}
h1{font-family:var(--sans) !important;font-weight:900 !important;font-size:2rem !important;letter-spacing:.08em !important;color:#fff !important;text-transform:uppercase !important;line-height:1.1 !important;}
h2,h3{font-family:var(--sans) !important;color:var(--accent) !important;font-weight:700 !important;letter-spacing:.1em !important;text-transform:uppercase !important;border-bottom:1px solid var(--border) !important;padding-bottom:.3em !important;}
[data-testid="metric-container"]{background:var(--bg3) !important;border:1px solid var(--border) !important;border-left:3px solid var(--accent) !important;padding:12px 16px !important;border-radius:2px !important;}
[data-testid="metric-container"] label{font-family:var(--sans) !important;font-size:.72rem !important;letter-spacing:.15em !important;color:var(--dim) !important;text-transform:uppercase !important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:var(--mono) !important;color:var(--accent) !important;font-size:1.7rem !important;}
.stButton button{background:transparent !important;border:1px solid var(--accent) !important;color:var(--accent) !important;font-family:var(--mono) !important;font-size:.75rem !important;letter-spacing:.12em !important;text-transform:uppercase !important;padding:8px 20px !important;border-radius:2px !important;width:100% !important;transition:all .2s !important;}
.stButton button:hover{background:var(--accent) !important;color:var(--bg) !important;}
[data-baseweb="tab-list"]{background:var(--bg2) !important;border-bottom:1px solid var(--border) !important;gap:0 !important;}
[data-baseweb="tab"]{font-family:var(--sans) !important;font-weight:600 !important;font-size:.82rem !important;letter-spacing:.12em !important;text-transform:uppercase !important;color:var(--dim) !important;padding:12px 22px !important;}
[aria-selected="true"][data-baseweb="tab"]{color:var(--accent) !important;background:transparent !important;}
[data-testid="stTextInput"] input{background:var(--bg3) !important;border-color:var(--border) !important;color:var(--text) !important;font-family:var(--mono) !important;font-size:.82rem !important;}
[data-testid="stSelectbox"]>div>div{background:var(--bg3) !important;border-color:var(--border) !important;}
[data-testid="stInfo"]{background:rgba(0,200,255,.04) !important;border:1px solid rgba(0,200,255,.25) !important;}
[data-testid="stSuccess"]{background:rgba(0,255,157,.04) !important;border:1px solid rgba(0,255,157,.25) !important;}
[data-testid="stError"]{background:rgba(255,43,77,.06) !important;border:1px solid rgba(255,43,77,.30) !important;}
[data-testid="stWarning"]{background:rgba(255,170,0,.04) !important;border:1px solid rgba(255,170,0,.25) !important;}
[data-testid="stDataFrame"]{border:1px solid var(--border) !important;}
[data-testid="stElementToolbarButton"]{display:none !important;}
[data-testid="stElementToolbar"]{display:none !important;}
[data-testid="stDownloadButton"] button{width:auto !important;background:var(--bg3) !important;border:1px solid var(--accent) !important;color:var(--accent) !important;font-family:var(--mono) !important;font-size:.72rem !important;letter-spacing:.1em !important;text-transform:uppercase !important;padding:6px 16px !important;border-radius:2px !important;}
[data-testid="stDownloadButton"] button:hover{background:var(--accent) !important;color:var(--bg) !important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg2);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
section[data-testid="stMain"]>div{background:var(--bg) !important;}
div[data-testid="stVerticalBlock"]{background:transparent !important;}
.stMarkdown,.stMarkdown p{color:var(--text) !important;}
[data-baseweb="select"] *{background:var(--bg3) !important;color:var(--text) !important;}
[data-baseweb="popover"]{background:var(--bg3) !important;border:1px solid var(--border) !important;}
[data-baseweb="menu"]{background:var(--bg3) !important;}
/* Sidebar top header (broken Material Icons icon) hide */
[data-testid="stSidebarHeader"]{display:none !important;}
[data-testid="stSidebarCollapseButton"]{display:none !important;}
header[data-testid="stHeader"]{display:none !important;}
.info-panel{background:var(--bg3);border:1px solid var(--border);border-left:3px solid var(--dim);padding:14px 18px;font-family:var(--sans);font-size:.88rem;line-height:1.7;margin:8px 0;}
.info-panel b{color:var(--accent);}
.warn-panel{background:rgba(255,170,0,.05);border:1px solid rgba(255,170,0,.3);border-left:3px solid #ffaa00;padding:12px 16px;font-family:var(--sans);font-size:.85rem;line-height:1.6;margin:8px 0;color:#b8cfe0;}
.crit-panel{background:rgba(255,43,77,.06);border:1px solid rgba(255,43,77,.35);border-left:3px solid #ff2b4d;padding:12px 16px;font-family:var(--sans);font-size:.85rem;line-height:1.6;margin:8px 0;color:#b8cfe0;}
</style>
"""


# ================================================================================
#  EARTH VIEW
# ================================================================================
@st.cache_data(show_spinner=False)
def load_earth_texture(resolution: int = 360, style: str = "night"):
    """
    Yüksek kaliteli NASA Dünya dokularını yükler ve Plotly Surface için optimize eder.
    style: "night" (Gece Işıkları), "realistic" (Gerçekçi Blue Marble), "futuristic" (Mavi/Siyan Tonlu)
    """
    try:
        if style == "night":
            # NASA Black Marble (Gece Işıkları) - Karanlık tema için EN İYİSİ
            urls = [
                "https://eoimages.gsfc.nasa.gov/images/imagerecords/79000/79765/dnb_land_ocean_ice.2012.3600x1800.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/b/ba/The_earth_at_night.jpg"
            ]
        elif style == "realistic":
            # NASA Blue Marble Next Generation (Yüksek Çözünürlüklü Gerçekçi)
            urls = [
                "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/a/ad/Blue_Marble_2002.png"
            ]
        else:
            # Senin eski fütüristik temanın optimize edilmiş hali
            urls = [
                "https://upload.wikimedia.org/wikipedia/commons/c/cd/Land_ocean_ice_2048.jpg/1024px-Land_ocean_ice_2048.jpg"
            ]
        
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        
        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                
                img  = Image.open(BytesIO(resp.content)).convert("RGB")
                W, H = resolution * 2, resolution
                
                # LANCZOS filtresi, yeniden boyutlandırmada piksellenmeyi en aza indirir
                img  = img.resize((W, H), Image.LANCZOS)
                
                img_array = np.array(img, dtype=np.float32)
                
                if style == "futuristic":
                    # Fütüristik mavi/siyan renk işlemleri
                    img_array = img_array * 0.4
                    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.4, 0, 255)
                    img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.1, 0, 255)
                    mean_val = np.mean(img_array)
                    img_array = np.clip((img_array - mean_val) * 1.3 + mean_val, 0, 255)
                    img_array = np.clip(img_array + 15, 0, 255)
                elif style == "night":
                    # Şehir ışıklarını hafifçe parlat, okyanusları tam siyah yap
                    img_array = np.clip(img_array * 1.3, 0, 255)
                
                img_array = img_array.astype(np.uint8)
                img = Image.fromarray(img_array)
                
                # MEDIANCUT algoritması, 256 renk limitinde oluşan çamurlaşmayı engeller
                imgq = img.quantize(colors=256, method=Image.MEDIANCUT)
                pal  = np.array(imgq.getpalette(), dtype=np.uint8).reshape(-1, 3)[:256]
                
                idx  = np.flipud(np.array(imgq, dtype=float))
                surf_color = idx / 255.0
                colorscale = [[i / 255.0, f"rgb({pal[i,0]},{pal[i,1]},{pal[i,2]})"] for i in range(256)]
                
                lat = np.linspace(np.pi / 2, -np.pi / 2, H)
                lon = np.linspace(-np.pi, np.pi, W)
                lon_g, lat_g = np.meshgrid(lon, lat)
                
                R = 6371.0
                x = R * np.cos(lat_g) * np.cos(lon_g)
                y = R * np.cos(lat_g) * np.sin(lon_g)
                z = R * np.sin(lat_g)
                
                return x, y, z, surf_color, colorscale
            except Exception:
                continue
                
        return None
    except Exception as e:
        st.sidebar.warning(f"Earth texture load failed: {str(e)[:50]}")
        return None

# ================================================================================
#  DATA FETCHING
# ================================================================================
def fetch_live_tles(username: str, password: str, search_query: str = "STARLINK"):
    try:
        client = SpaceTrackClient(identity=username, password=password)
        if search_query == "ISS":
            raw = client.gp(norad_cat_id=25544, format="tle", orderby="epoch desc", limit=1)
        else:
            raw = client.gp(object_name=op.like(f"{search_query}%"), format="tle",
                            orderby="epoch desc", limit=90)
        if not raw or not raw.strip():
            st.sidebar.error(f"No results found for '{search_query}'.")
            return None
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        return lines if len(lines) >= 2 else None
    except Exception as e:
        st.sidebar.error(f"API Error: {e}")
        return None

# ================================================================================
#  TLE PARSING
# ================================================================================
def parse_tles(lines: list, limit: int = 30) -> list:
    sats = []
    is_3ln = not (lines[0].startswith("1 ") or lines[0].startswith("2 "))
    step   = 3 if is_3ln else 2
    for i in range(0, len(lines) - (step - 1), step):
        try:
            if is_3ln:
                name, l1, l2 = lines[i], lines[i+1], lines[i+2]
            else:
                name = f"OBJ-{lines[i][2:7].strip()}"
                l1, l2 = lines[i], lines[i+1]
            sats.append(EarthSatellite(l1, l2, name, ts))
            if len(sats) >= limit:
                break
        except Exception:
            continue
    return sats

# ================================================================================
#  ORBITAL ELEMENTS (from TLE)
# ================================================================================
def get_orbital_elements(sat: EarthSatellite) -> dict:
    """Extracts Kepler orbital elements from TLE."""
    try:
        model = sat.model
        # Elements from TLE epoch
        incl   = math.degrees(model.inclo)          # inclination (deg)
        raan   = math.degrees(model.nodeo)           # RAAN (deg)
        ecc    = model.ecco                          # eccentricity
        argp   = math.degrees(model.argpo)           # argument of periapsis (deg)
        mean_m = math.degrees(model.mo)              # mean anomaly (deg)
        n_rpm  = model.no_kozai * (60.0 / (2*math.pi))  # rad/min → devir/min
        # Semi-major axis: a = (GM/n^2)^(1/3), n rad/s
        GM     = 398600.4418  # km^3/s^2
        n_rads = model.no_kozai / 60.0  # rad/s
        a_km   = (GM / n_rads**2) ** (1/3)
        alt_km = a_km - 6371.0
        period_min = 2 * math.pi / model.no_kozai
        return {
            "Semi-major Axis a (km)":   round(a_km, 1),
            "Mean Altitude (km)":       round(alt_km, 1),
            "Eccentricity e":             f"{ecc:.6f}",
            "Inclination i (°)":                round(incl, 4),
            "RAAN (°)": round(raan, 4),
            "Arg of Perigee ω (°)":      round(argp, 4),
            "Mean Anomaly M (°)":     round(mean_m, 4),
            "Orbital Period (min)":      round(period_min, 2),
            "Mean Motion n (rev/min)": round(n_rpm, 6),
        }
    except Exception:
        return {}

# ================================================================================
#  APSIS FILTER (Section 2.1 — Thesis)
# ================================================================================
def apsis_filter(sats: list, threshold_km: float = 50.0) -> list:
    """
    Apsis (Apogee-Perigee) Filter — Section 2.1
    Reduces O(N^2) complexity by filtering pairs with non-overlapping
    altitude bands.
    q1 > Q2 + D   →   physical intersection impossible → filtered
    """
    R_E = 6371.0
    GM  = 398600.4418

    def apsis(sat):
        try:
            n   = sat.model.no_kozai / 60.0  # rad/s
            a   = (GM / n**2) ** (1/3)
            e   = sat.model.ecco
            per = a * (1 - e) - R_E          # perigee altitude
            apo = a * (1 + e) - R_E          # apogee altitude
            return per, apo
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
#  FOSTER 1992 2D-Pc (Section 3.1 — Thesis)
# ================================================================================
def foster_2d_pc(miss_km: float, sigma_r: float, sigma_t: float,
                  sigma_n: float, hbr_km: float = 0.020) -> float:
    """
    Foster & Estes (1992) 2D-Pc Model — Section 3.1
    Gaussian integral projected onto encounter plane.
    sigma_r: radial (km), sigma_t: in-track (km), sigma_n: cross-track (km)
    Combined covariance: two components in encounter plane.
    """
    try:
        # Encounter plane components (radial + normal)
        sig_x = math.sqrt(sigma_r**2 + sigma_r**2)   # combined radial
        sig_y = math.sqrt(sigma_n**2 + sigma_n**2)   # combined normal
        if sig_x <= 0 or sig_y <= 0:
            return 0.0
        # Circle integral over 2D Gaussian (numerical)
        # Gaussian distribution centered at TCA mean miss point (miss_km, 0);
        # integration region is HBR-radius circle around origin (actual collision sphere).
        def integrand(y, x):
            return (1.0 / (2*math.pi*sig_x*sig_y) *
                    math.exp(-0.5 * (((x - miss_km)/sig_x)**2 + (y/sig_y)**2)))
        result, _ = dblquad(
            integrand,
            -hbr_km, hbr_km,
            lambda x: -math.sqrt(max(hbr_km**2 - x**2, 0)),
            lambda x:  math.sqrt(max(hbr_km**2 - x**2, 0)),
            limit=50,
        )
        return max(float(result), 0.0)
    except Exception:
        return collision_probability_isotropic(miss_km, (sigma_r+sigma_n)/2, hbr_km)

def collision_probability_isotropic(miss_km: float, sigma_km: float,
                                     hbr_km: float = 0.020) -> float:
    """
    Chan (1997) isotropic model — fast fallback.
    Correct formula: P(|X_rel| ≤ HBR) for x ∈ N(miss, σ)
    Pc = Φ((HBR - miss)/σ) + Φ((HBR + miss)/σ) - 1
    """
    if sigma_km <= 0:
        return 0.0
    pc = norm.cdf((hbr_km - miss_km) / sigma_km) + norm.cdf((hbr_km + miss_km) / sigma_km) - 1.0
    return max(float(pc), 0.0)

# Public alias
def collision_probability(miss_km, sigma_km, hbr_km=0.020):
    return collision_probability_isotropic(miss_km, sigma_km, hbr_km)

# ================================================================================
#  MAHALANOBIS DISTANCE TEST (Section 3.2 — Thesis)
# ================================================================================
def mahalanobis_test(miss_km: float, sigma_km: float) -> dict:
    """
    2D-Pc validity test (CARA methodology — Section 3.2).
    Mahalanobis distance Md = miss / sigma.
    Md < 1.5 → linear motion assumption breaks down → 3D-Pc required.
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
def max_pc_analysis(miss_km: float, hbr_km: float = 0.020) -> float:
    """
    Max Pc — Section 4 (CARA toolkit).
    Scans covariance multiplier σ to find mathematical maximum Pc.
    Worst case: σ_opt = miss / sqrt(2) (Gaussian peak point).
    """
    sigma_opt = miss_km / math.sqrt(2.0) if miss_km > 0 else hbr_km
    return collision_probability_isotropic(miss_km, max(sigma_opt, 1e-6), hbr_km)

# ================================================================================
#  PROBABILITY DILUTION DETECTION (Section 4 — Thesis)
# ================================================================================
def dilution_check(pc: float, sigma_km: float, miss_km: float) -> dict:
    """
    Probability Dilution detection — Section 4.
    Wide covariance → small Pc → false confidence.
    Warning: sigma > 5*miss_km and pc < 1e-6
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
def fragmentation_probability(rel_vel_km_s: float,
                               mass_a_kg: float = 250.0,
                               mass_b_kg: float = 250.0) -> dict:
    """
    Collision Consequence — Section 4.
    Kinetic energy-based fragmentation risk per NASA operational guidelines.
    Specific Energy: E_c = 0.5 * m_b * v_rel^2 / m_a  (J/g)
    E_c > 40 J/g → Catastrophic fragmentation (Kessler contribution)
    E_c > 0 J/g  → Damaging
    """
    v_ms = rel_vel_km_s * 1000.0
    E_c  = 0.5 * mass_b_kg * v_ms**2 / (mass_a_kg * 1000.0)  # J/g
    if E_c >= 40.0:
        pf_level = "CATASTROPHIC"
        pf_color = "#ff2b4d"
        pf_desc  = "Complete fragmentation — Kessler contribution likely"
    elif E_c >= 10.0:
        pf_level = "SEVERE"
        pf_color = "#ff6b00"
        pf_desc  = "Operational loss and significant debris"
    elif E_c >= 1.0:
        pf_level = "DAMAGING"
        pf_color = "#ffaa00"
        pf_desc  = "Partial damage or subsystem failure"
    else:
        pf_level = "LOW"
        pf_color = "#00ff9d"
        pf_desc  = "Minor damage — fragmentation unlikely"
    n_debris = int(0.1 * (mass_a_kg + mass_b_kg) * (rel_vel_km_s / 7.0))
    return {
        "E_c_J_per_g": round(E_c, 2),
        "level":       pf_level,
        "color":       pf_color,
        "desc":        pf_desc,
        "est_debris":  n_debris,
    }

# ================================================================================
#  RISK LEVEL
# ================================================================================
def risk_level(pc: float) -> tuple:
    """NASA STD-8719.14 — 4-tier risk classification."""
    if   pc > 1e-3: return "CRITICAL", "#ff2b4d"
    elif pc > 1e-4: return "HIGH", "#ff6b00"
    elif pc > 1e-5: return "MEDIUM",   "#ffaa00"
    else:           return "LOW",  "#00ff9d"

# ================================================================================
#  MAIN CONJUNCTION ANALYSIS (APSIS FILTERED)
# ================================================================================
def compute_conjunctions(sats: list, window_hrs: int, sigma_km: float, mass_a_kg: float = 250.0, mass_b_kg: float = 250.0) -> tuple:
    """
    Apsis filter + 5-min step TCA scan + multiple Pc metrics.
    Returns: (df_results, n_apsis_filtered, n_total_pairs)
    """
    now         = ts.now()
    step_m      = 5
    n_steps     = window_hrs * 60 // step_m
    n_total     = len(list(combinations(sats, 2)))

    # Apsis pre-filter
    candidate_pairs = apsis_filter(sats, threshold_km=100.0)
    n_filtered      = n_total - len(candidate_pairs)

    results = []
    for s1, s2 in candidate_pairs:
        min_d    = np.inf
        best_t   = None
        dist_arr = []

        for i in range(n_steps):
            t  = ts.tt_jd(now.tt + i * step_m / 1440.0)
            p1 = s1.at(t).position.km
            p2 = s2.at(t).position.km
            d  = float(np.linalg.norm(p1 - p2))
            dist_arr.append(d)
            if d < min_d:
                min_d, best_t = d, t

        if min_d >= 500:
            continue

        rel_vel   = _relative_velocity(s1, s2, best_t)
        pc_iso    = collision_probability_isotropic(min_d, sigma_km)
        pc_foster = foster_2d_pc(min_d, sigma_km, sigma_km*2, sigma_km)
        pc_max    = max_pc_analysis(min_d)
        mah       = mahalanobis_test(min_d, sigma_km)
        dil       = dilution_check(pc_iso, sigma_km, min_d)
        frag      = fragmentation_probability(rel_vel, mass_a_kg, mass_b_kg)
        sev, color = risk_level(pc_iso)

        results.append({
            "TCA (UTC)":           best_t.utc_strftime("%Y-%m-%d %H:%M:%S"),
            "Object A":              s1.name,
            "Object B":              s2.name,
            "Distance (km)":         round(min_d, 3),
            "Relative Velocity (km/s)":  round(rel_vel, 3),
            "Pc (isotropic)":      pc_iso,
            "Pc (Foster 2D)":      pc_foster,
            "Pc Max":              pc_max,
            "Pc (scientific)":       f"{pc_iso:.3e}",
            "Mahalanobis Md":      mah["Md"],
            "2D-Pc Valid":       mah["label"],
            "Dilution":            dil["diluted"],
            "Dilution Message":     dil["msg"],
            "Ec (J/g)":            frag["E_c_J_per_g"],
            "Fragmentation Level": frag["level"],
            "Estimated Debris":       frag["est_debris"],
            "Risk Level":       sev,
            "_color":              color,
            "_dist_arr":           dist_arr,
            "_s1":                 s1,
            "_s2":                 s2,
        })

    return pd.DataFrame(results), n_filtered, n_total

def _relative_velocity(s1, s2, t) -> float:
    v1 = s1.at(t).velocity.km_per_s
    v2 = s2.at(t).velocity.km_per_s
    return float(np.linalg.norm(np.array(v1) - np.array(v2)))

# ================================================================================
#  PLOTS
# ================================================================================
DARK = dict(
    paper_bgcolor="#07090f",
    plot_bgcolor="#07090f",
    font=dict(family="Space Mono, monospace", color="#b8cfe0", size=11),
)

def fig_3d_orbits(sats):
    now    = ts.now()
    fig    = go.Figure()
    
    # Dünya kaplamasını yükle
    earth = load_earth_texture(resolution=360, style="realistic")
    if earth:
        x, y, z, sc, cs = earth
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=sc, colorscale=cs,
            showscale=False, opacity=1.0, hoverinfo="skip", name="Dunya",
            lightposition=dict(x=0, y=0, z=10000),
            lighting=dict(ambient=0.6, diffuse=0.92, specular=0.04, roughness=0.85, fresnel=0.05),
        ))
    else:
        r = 6371
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        fig.add_trace(go.Surface(
            x=r*np.cos(u)*np.sin(v), y=r*np.sin(u)*np.sin(v), z=r*np.cos(v),
            colorscale="Blues", opacity=.4, showscale=False))
            
    colors  = ["#00c8ff","#00ff9d","#ffaa00","#ff6b00","#c060ff","#ff2b4d",
               "#60d0ff","#80ffb0","#ffcc60","#ff9060"]
    offsets = np.linspace(0, 95, 80) / 1440.0
    
    for k, sat in enumerate(sats):
        times = ts.tt_jd(now.tt + offsets)
        c     = colors[k % len(colors)]
        
        # 1. Yörünge çizgisi hesaplaması ve güvenlik kontrolü
        try:
            pos = sat.at(times).position.km
        except Exception:
            pos = np.full((3, 80), np.nan)
            
        if not np.all(np.isnan(pos)):
            fig.add_trace(go.Scatter3d(
                x=pos[0].tolist(), y=pos[1].tolist(), z=pos[2].tolist(), 
                mode="lines", line=dict(color=c, width=2.5), 
                name=sat.name, opacity=0.95
            ))
            
        # 2. Anlık pozisyon hesaplaması ve güvenlik kontrolü
        try:
            p0 = sat.at(now).position.km
        except Exception:
            p0 = np.full((3,), np.nan)
            
        if not np.any(np.isnan(p0)):
            fig.add_trace(go.Scatter3d(
                x=[float(p0[0])], y=[float(p0[1])], z=[float(p0[2])], 
                mode="markers",
                marker=dict(color=c, size=6, symbol="circle", line=dict(color="#ffffff", width=1)),
                name=f"{sat.name} (now)", showlegend=False
            ))
            
    fig.update_layout(
        **DARK, margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(bgcolor="#000408",
            xaxis=dict(visible=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False),
            zaxis=dict(visible=False, showgrid=False, zeroline=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.7), up=dict(x=0, y=0, z=1)),
        ),
        legend=dict(font=dict(size=8, family="Space Mono"),
            bgcolor="rgba(0,4,8,.85)", bordercolor="#1a2740", borderwidth=1,
            x=0.01, y=0.99, itemsizing="constant"),
    )
    return fig

def fig_ground_tracks(sats):
    now     = ts.now()
    offsets = np.linspace(0, 95, 200) / 1440.0
    colors  = ["#00c8ff","#00ff9d","#ffaa00","#ff6b00","#c060ff","#ff2b4d",
               "#60d0ff","#80ffb0","#ffcc60","#ff9060"]
    fig = go.Figure()
    for k, sat in enumerate(sats):
        times = ts.tt_jd(now.tt + offsets)
        geo   = wgs84.subpoint_of(sat.at(times))
        c     = colors[k % len(colors)]
        fig.add_trace(go.Scattergeo(lat=geo.latitude.degrees, lon=geo.longitude.degrees,
            mode="lines", line=dict(color=c, width=1.8), name=sat.name, opacity=.85))
        g0 = wgs84.subpoint_of(sat.at(ts.now()))
        fig.add_trace(go.Scattergeo(
            lat=[g0.latitude.degrees], lon=[g0.longitude.degrees],
            mode="markers+text",
            marker=dict(color=c, size=9, symbol="circle", line=dict(color="#ffffff", width=1)),
            text=[sat.name], textposition="top right",
            textfont=dict(size=8, family="Space Mono", color=c), showlegend=False))
    fig.update_layout(
        **DARK, height=420, margin=dict(l=0, r=0, t=30, b=0),
        geo=dict(
            showland=True,       landcolor="#0d2137",
            showocean=True,      oceancolor="#050d18",
            showcoastlines=True, coastlinecolor="#2a5070", coastlinewidth=0.8,
            showcountries=True,  countrycolor="#152535", countrywidth=0.4,
            showlakes=True,      lakecolor="#080f1a",
            showrivers=True,     rivercolor="#0a1828",
            showframe=False,     bgcolor="#07090f",
            projection_type="natural earth", resolution=50,
            lonaxis=dict(range=[-180,180], showgrid=True,
                         gridcolor="rgba(26,39,64,.5)", gridwidth=0.3),
            lataxis=dict(range=[-90,90], showgrid=True,
                         gridcolor="rgba(26,39,64,.5)", gridwidth=0.3),
        ),
        legend=dict(font=dict(size=8, family="Space Mono"),
            bgcolor="rgba(7,9,15,.85)", bordercolor="#1a2740", borderwidth=1, x=0.0, y=1.0),
        title=dict(text="Ground Track -- Current Position and 95min Orbit",
            font=dict(size=11, family="Barlow Condensed", color="#00c8ff"), x=0.01, y=0.99),
    )
    return fig

def fig_distance_profile(dist_arr, window_hrs, miss_km, sigma_km):
    step_m  = 5
    t_axis  = np.arange(len(dist_arr)) * step_m / 60.0
    fig = go.Figure()
    fig.add_hline(y=0.02, line=dict(color="#ff2b4d", dash="dot", width=1),
                  annotation_text="HBR (20 m)", annotation_font_size=9)
    fig.add_hrect(y0=max(0, miss_km-sigma_km), y1=miss_km+sigma_km,
                  fillcolor="rgba(0,200,255,.05)", line_width=0)
    fig.add_trace(go.Scatter(x=t_axis, y=dist_arr, mode="lines",
        line=dict(color="#00c8ff", width=1.5), name="Distance (km)",
        fill="tozeroy", fillcolor="rgba(0,200,255,.04)"))
    tca_i = int(np.argmin(dist_arr))
    fig.add_trace(go.Scatter(x=[t_axis[tca_i]], y=[dist_arr[tca_i]],
        mode="markers+text", marker=dict(color="#ff2b4d", size=8),
        text=[f" TCA: {dist_arr[tca_i]:.1f} km"], textposition="top right",
        textfont=dict(size=9, family="Space Mono", color="#ff2b4d"),
        name="TCA", showlegend=False))
    fig.update_layout(**DARK, height=280,
        xaxis=dict(title="Time (hours)", gridcolor="#1a2740", zeroline=False, tickfont=dict(size=9)),
        yaxis=dict(title="Distance (km)", gridcolor="#1a2740", zeroline=False, tickfont=dict(size=9)),
        title=dict(text="Distance Profile — TCA Analysis",
            font=dict(size=11, family="Barlow Condensed", color="#00c8ff"), x=0.01),
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=35, b=10),
    )
    return fig

def fig_risk_gauge(pc: float):
    sev, color = risk_level(pc)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pc,
        number=dict(valueformat=".2e", font=dict(family="Space Mono", color=color, size=18)),
        gauge=dict(
            axis=dict(range=[0, 1e-3],
                tickvals=[0, 1e-5, 1e-4, 1e-3],
                ticktext=["0","1e-5","1e-4","1e-3"],
                tickfont=dict(size=8, family="Space Mono", color="#4a6880")),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#0c1018", bordercolor="#1a2740",
            steps=[
                dict(range=[0,     1e-5], color="#0d1820"),
                dict(range=[1e-5,  1e-4], color="#141e10"),
                dict(range=[1e-4,  1e-3], color="#1e1008"),
            ],
            threshold=dict(line=dict(color="#ff2b4d", width=2), value=1e-4),
        ),
        title=dict(text=f"Pc — {sev}", font=dict(family="Barlow Condensed", color=color, size=14)),
        domain=dict(x=[0,1], y=[0,1]),
    ))
    fig.update_layout(**DARK, height=210, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def fig_orbital_elements_radar(elems_list):
    """Display satellites by orbital elements using scatter plot."""
    fig = go.Figure()
    colors = ["#00c8ff","#00ff9d","#ffaa00","#ff6b00","#c060ff","#ff2b4d"]
    for k, (name, elems) in enumerate(elems_list):
        if not elems:
            continue
        try:
            alt  = float(str(elems.get("Mean Altitude a (km)", 0)))
            incl = float(str(elems.get("Inclination i (°)", 0)))
            ecc  = float(str(elems.get("Eccentricity e", "0")))
            fig.add_trace(go.Scatter(
                x=[incl], y=[alt],
                mode="markers+text",
                marker=dict(color=colors[k % len(colors)], size=10+ecc*80,
                            line=dict(color="#fff", width=0.5)),
                text=[name[:12]], textposition="top center",
                textfont=dict(size=8, family="Space Mono", color=colors[k % len(colors)]),
                name=name,
            ))
        except Exception:
            continue
    fig.update_layout(
        **DARK, height=320,
        xaxis=dict(title="Inclination i (°)", gridcolor="#1a2740", zeroline=False, tickfont=dict(size=9)),
        yaxis=dict(title="Altitude (km)", gridcolor="#1a2740", zeroline=False, tickfont=dict(size=9)),
        title=dict(text="Orbital Space — Altitude / Inclination Distribution",
            font=dict(size=11, family="Barlow Condensed", color="#00c8ff"), x=0.01),
        margin=dict(l=10, r=10, t=35, b=10),
        showlegend=False,
    )
    return fig

# ================================================================================
#  CONJUNCTION ANALYSIS FOR OWN SATELLITE
# ================================================================================
def compute_conjunctions_custom(my_sat, sats: list, window_hrs: int, sigma_km: float, mass_a_kg: float = 250.0, mass_b_kg: float = 250.0) -> pd.DataFrame:
    """
    Compares user's own satellite with existing satellite fleet.
    Apsis filter + 5-min TCA scan + full Pc metrics.
    """
    now    = ts.now()
    step_m = 5
    n_steps = window_hrs * 60 // step_m
    R_E, GM = 6371.0, 398600.4418

    def apsis(sat):
        try:
            n = sat.model.no_kozai / 60.0
            a = (GM / n**2) ** (1/3)
            e = sat.model.ecco
            return a*(1-e)-R_E, a*(1+e)-R_E
        except Exception:
            return 0.0, 10000.0

    my_q, my_Q = apsis(my_sat)
    results = []

    for sat in sats:
        q, Q = apsis(sat)
        # Apsis filter
        if max(my_q, q) > min(my_Q, Q) + 100.0:
            continue

        min_d  = np.inf
        best_t = None
        dist_arr = []

        for i in range(n_steps):
            t  = ts.tt_jd(now.tt + i * step_m / 1440.0)
            p1 = my_sat.at(t).position.km
            p2 = sat.at(t).position.km
            d  = float(np.linalg.norm(p1 - p2))
            dist_arr.append(d)
            if d < min_d:
                min_d, best_t = d, t

        if min_d >= 500:
            continue

        rel_vel   = _relative_velocity(my_sat, sat, best_t)
        pc_iso    = collision_probability_isotropic(min_d, sigma_km)
        pc_foster = foster_2d_pc(min_d, sigma_km, sigma_km*2, sigma_km)
        pc_max    = max_pc_analysis(min_d)
        mah       = mahalanobis_test(min_d, sigma_km)
        dil       = dilution_check(pc_iso, sigma_km, min_d)
        frag      = fragmentation_probability(rel_vel, mass_a_kg, mass_b_kg)
        sev, color = risk_level(pc_iso)

        results.append({
            "TCA (UTC)":           best_t.utc_strftime("%Y-%m-%d %H:%M:%S"),
            "Object A":              my_sat.name,
            "Object B":              sat.name,
            "Distance (km)":         round(min_d, 3),
            "Relative Velocity (km/s)":  round(rel_vel, 3),
            "Pc (isotropic)":      pc_iso,
            "Pc (Foster 2D)":      pc_foster,
            "Pc Max":              pc_max,
            "Pc (scientific)":       f"{pc_iso:.3e}",
            "Mahalanobis Md":      mah["Md"],
            "2D-Pc Valid":       mah["label"],
            "Dilution":            dil["diluted"],
            "Dilution Message":     dil["msg"],
            "Ec (J/g)":            frag["E_c_J_per_g"],
            "Fragmentation Level": frag["level"],
            "Estimated Debris":       frag["est_debris"],
            "Risk Level":       sev,
            "_color":              color,
            "_dist_arr":           dist_arr,
            "_s1":                 my_sat,
            "_s2":                 sat,
        })

    return pd.DataFrame(results)


# ================================================================================
#  LIVE 3D ANIMATION (Two Satellites — TCA Focused)
# ================================================================================
def fig_animated_conjunction(sat_a, sat_b, window_hrs: int = 6, show_orbits: bool = True, show_tca: bool = True):
    """
    3D Plotly figure showing two satellites with real-time animation.
    Robust version with error handling and fallback.
    """
    now = ts.now()
    step_min = max(2, window_hrs * 60 // 360)
    n_frames = min(window_hrs * 60 // step_min, 360)
    if n_frames < 2:
        return go.Figure(), 0, 0.0, np.array([0.0]), np.array([now.tt])

    trail_len = 25
    orbit_pts = 100

    # Full orbit paths (static background)
    orb_off = np.linspace(0, 96, orbit_pts) / 1440.0
    try:
        orb_a = sat_a.at(ts.tt_jd(now.tt + orb_off)).position.km
        orb_b = sat_b.at(ts.tt_jd(now.tt + orb_off)).position.km
    except Exception:
        orb_a = np.full((3, orbit_pts), np.nan)
        orb_b = np.full((3, orbit_pts), np.nan)

    # Animation step positions
    anim_off = np.arange(n_frames) * step_min / 1440.0
    anim_jd = now.tt + anim_off
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
        if np.isnan(d): return "rgba(100,180,255,0.55)"
        if d < 50:   return "#ff2b4d"
        if d < 200:  return "#ffaa00"
        return "rgba(100,180,255,0.55)"

    fig = go.Figure()

    # Earth texture (with fallback)
    earth = load_earth_texture(resolution=240, style="night")
    earth_data = None
    if earth:
        x, y, z, sc, cs = earth
        earth_data = (x, y, z, sc, cs)
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=sc, colorscale=cs,
            showscale=False, opacity=1.0, hoverinfo="skip",
            lightposition=dict(x=0, y=0, z=10000),
            lighting=dict(ambient=0.6, diffuse=0.9, specular=0.03, roughness=0.85),
            name="Earth",
        ))
    else:
        r = 6371.0
        u, v = np.mgrid[0:2*np.pi:120j, 0:np.pi:60j]
        colorscale_earth = [
            [0.0, "#081828"], [0.2, "#0a2848"], [0.35, "#1a3868"],
            [0.5, "#2a4888"], [0.65, "#2a4888"], [0.8, "#1a3868"],
            [0.9, "#d0c0b0"], [1.0, "#ffffff"]
        ]
        fig.add_trace(go.Surface(
            x=r*np.cos(u)*np.sin(v), y=r*np.sin(u)*np.sin(v), z=r*np.cos(v),
            colorscale=colorscale_earth, opacity=0.95, showscale=False,
            name="Earth"))

    # Coordinate grid lines
    r_earth = 6371.0
    for lon_deg in range(-180, 181, 30):
        lon_rad = np.radians(lon_deg)
        lat_rad = np.linspace(-np.pi/2, np.pi/2, 50)
        xg = r_earth * np.cos(lat_rad) * np.cos(lon_rad)
        yg = r_earth * np.cos(lat_rad) * np.sin(lon_rad)
        zg = r_earth * np.sin(lat_rad)
        fig.add_trace(go.Scatter3d(
            x=xg.tolist(), y=yg.tolist(), z=zg.tolist(),
            mode="lines", line=dict(color="rgba(100,150,200,0.3)", width=1),
            showlegend=False, hoverinfo="skip"))
            
    for lat_deg in range(-90, 91, 30):
        lat_rad = np.radians(lat_deg)
        lon_rad = np.linspace(-np.pi, np.pi, 50)
        xg = r_earth * np.cos(lat_rad) * np.cos(lon_rad)
        yg = r_earth * np.cos(lat_rad) * np.sin(lon_rad)
        zg = r_earth * np.sin(lat_rad)
        fig.add_trace(go.Scatter3d(
            x=xg.tolist(), y=yg.tolist(), z=zg.tolist(),
            mode="lines", line=dict(color="rgba(100,150,200,0.3)", width=1),
            showlegend=False, hoverinfo="skip"))

    # Full orbit trails (SAFE CHECK ADDED)
    if show_orbits and not np.all(np.isnan(orb_a)):
        fig.add_trace(go.Scatter3d(
            x=orb_a[0].tolist(), y=orb_a[1].tolist(), z=orb_a[2].tolist(),
            mode="lines", line=dict(color="rgba(0,200,255,0.12)", width=1.5),
            name=sat_a.name+" orbit", showlegend=False))
            
    if show_orbits and not np.all(np.isnan(orb_b)):
        fig.add_trace(go.Scatter3d(
            x=orb_b[0].tolist(), y=orb_b[1].tolist(), z=orb_b[2].tolist(),
            mode="lines", line=dict(color="rgba(255,107,0,0.12)", width=1.5),
            name=sat_b.name+" orbit", showlegend=False))

    # TCA point (SAFE CHECK ADDED)
    mid_tca = (pos_a[:, tca_idx] + pos_b[:, tca_idx]) / 2
    if show_tca and not np.any(np.isnan(mid_tca)):
        fig.add_trace(go.Scatter3d(
            x=[float(mid_tca[0])], y=[float(mid_tca[1])], z=[float(mid_tca[2])],
            mode="markers+text",
            marker=dict(color="#ff2b4d", size=10, symbol="diamond",
                        line=dict(color="#ffffff", width=1)),
            text=[f"TCA {tca_dist:.1f} km"], textposition="top right",
            textfont=dict(color="#ff2b4d", size=9, family="Space Mono"),
            name="TCA Point",
        ))

    n_static = len(fig.data)

    def rotate_earth(x, y, z, angle_rad):
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        return x * cos_a - y * sin_a, x * sin_a + y * cos_a, z

    def make_dynamic_traces(i):
        t0 = max(0, i - trail_len)
        ta = pos_a[:, t0:i+1]
        tb = pos_b[:, t0:i+1]
        dc = dist_color(dists[i])
        traces = []

        # Safe Trails
        if ta.shape[1] > 0 and not np.all(np.isnan(ta)):
            traces.append(go.Scatter3d(
                x=ta[0].tolist(), y=ta[1].tolist(), z=ta[2].tolist(),
                mode="lines", line=dict(color="#00c8ff", width=2.5),
                name=sat_a.name, showlegend=False))
                
        if tb.shape[1] > 0 and not np.all(np.isnan(tb)):
            traces.append(go.Scatter3d(
                x=tb[0].tolist(), y=tb[1].tolist(), z=tb[2].tolist(),
                mode="lines", line=dict(color="#ff6b00", width=2.5),
                name=sat_b.name, showlegend=False))

        # Safe Current positions
        pa = pos_a[:, i]
        pb = pos_b[:, i]
        
        if not np.any(np.isnan(pa)):
            traces.append(go.Scatter3d(
                x=[float(pa[0])], y=[float(pa[1])], z=[float(pa[2])],
                mode="markers",
                marker=dict(color="#00c8ff", size=9, line=dict(color="#fff", width=1)),
                name=sat_a.name+" pos", showlegend=False))
                
        if not np.any(np.isnan(pb)):
            traces.append(go.Scatter3d(
                x=[float(pb[0])], y=[float(pb[1])], z=[float(pb[2])],
                mode="markers",
                marker=dict(color="#ff6b00", size=9, line=dict(color="#fff", width=1)),
                name=sat_b.name+" pos", showlegend=False))

        # Safe Distance line
        if not np.any(np.isnan(pa)) and not np.any(np.isnan(pb)):
            traces.append(go.Scatter3d(
                x=[float(pa[0]), float(pb[0])],
                y=[float(pa[1]), float(pb[1])],
                z=[float(pa[2]), float(pb[2])],
                mode="lines+text",
                line=dict(color=dc, width=2, dash="dot"),
                text=["", f"  Δ {dists[i]:.1f} km"],
                textfont=dict(color=dc, size=9, family="Space Mono"),
                name="Distance", showlegend=False))

        if earth_data is not None:
            x, y, z, sc, cs = earth_data
            time_hours = i * step_min / 60.0
            earth_angle = np.radians(time_hours * 15.0)
            ex_rot, ey_rot, ez_rot = rotate_earth(x, y, z, earth_angle)
            traces.insert(0, go.Surface(
                x=ex_rot, y=ey_rot, z=ez_rot,
                surfacecolor=sc, colorscale=cs,
                showscale=False, opacity=1.0, hoverinfo="skip",
                lightposition=dict(x=0, y=0, z=10000),
                lighting=dict(ambient=0.6, diffuse=0.9, specular=0.03, roughness=0.85),
                name="Earth",
            ))

        return traces

    for tr in make_dynamic_traces(0):
        fig.add_trace(tr)

    n_dynamic = len(fig.data) - n_static
    if n_dynamic == 0:
        return fig, tca_idx, tca_dist, dists, anim_jd

    dyn_indices = list(range(n_static, n_static + n_dynamic))

    frames = []
    slider_steps = []
    for i in range(n_frames):
        t_utc = ts.tt_jd(anim_jd[i]).utc_strftime("%H:%M UTC")
        t_min = i * step_min
        title_txt = (f"T+{t_min:04d} min  |  {t_utc}  |  "
                     f"Δ {dists[i]:.1f} km"
                     + ("  ⚠ TCA" if i == tca_idx else ""))
        frames.append(go.Frame(
            data=make_dynamic_traces(i),
            traces=dyn_indices,
            name=str(i),
            layout=go.Layout(title_text=title_txt),
        ))
        lbl = t_utc if i % max(1, n_frames//20) == 0 else ""
        slider_steps.append(dict(
            args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
            label=lbl, method="animate",
        ))

    fig.frames = frames

    fig.update_layout(
        **DARK,
        height=640,
        margin=dict(l=0, r=0, t=70, b=10),
        title=dict(
            text=f"{sat_a.name} × {sat_b.name} — TCA: {tca_dist:.2f} km (T+{tca_idx*step_min} min)",
            font=dict(family="Barlow Condensed", color="#00c8ff", size=13), x=0.01, y=0.98,
        ),
        scene=dict(
            bgcolor="#000408",
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.7, y=1.7, z=0.75), up=dict(x=0,y=0,z=1)),
        ),
        legend=dict(font=dict(size=8, family="Space Mono"),
                    bgcolor="rgba(0,4,8,.85)", bordercolor="#1a2740", borderwidth=1,
                    x=0.01, y=0.92, itemsizing="constant"),
        updatemenus=[
            dict(
                type="buttons", showactive=False,
                bgcolor="#0c1018", bordercolor="#1a2740",
                font=dict(family="Space Mono", size=8, color="#b8cfe0"),
                y=1.02, x=0.5, xanchor="center", pad=dict(r=4),
                direction="left",
                buttons=[
                    dict(label="▶ PLAY", method="animate",
                         args=[None, dict(frame=dict(duration=80, redraw=True),
                                         fromcurrent=True, mode="immediate")]),
                    dict(label="⏸ STOP", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                    dict(label="⏮ JUMP TO TCA", method="animate",
                         args=[[str(tca_idx)],
                               dict(frame=dict(duration=0, redraw=True), mode="immediate")]),
                ],
            ),
        ],
        sliders=[dict(
            steps=slider_steps, active=0,
            currentvalue=dict(prefix="⏱  ", font=dict(family="Space Mono",size=9,color="#4a6880")),
            pad=dict(t=64, b=0), len=0.92, x=0.04,
            bgcolor="#0c1018", bordercolor="#1a2740", tickcolor="#1a2740",
            font=dict(color="#4a6880", size=7),
        )],
    )
    return fig, tca_idx, tca_dist, dists, anim_jd


# ================================================================================
#  INTERFACE
# ================================================================================
st.set_page_config(page_title="StarWeb-CARA: Conjunction Assessment and Collision Risk Analysis for Starlink and OneWeb Megaconstellations", page_icon="S", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(STYLE, unsafe_allow_html=True)

st.markdown("""
<div style="padding:20px 0 8px 0; border-bottom:1px solid #1a2740; margin-bottom:20px;">
  <div style="font-family:'Space Mono',monospace; font-size:.68rem;
              color:#4a6880; letter-spacing:.2em; text-transform:uppercase; margin-bottom:4px;">
    Conjunction Assessment and Collision Risk Analysis (starlink/iss/oneweb)
  </div>
  <h1 style="margin:0; padding:0; font-size:1.7rem;">
    Low Earth Orbit<br>
    <span style="color:#00c8ff;">Conjunction Assessment &amp; Collision Risk Analysis </span>
  </h1>
  <div style="font-family:'Barlow Condensed',sans-serif; font-size:.95rem;
              color:#4a6880; margin-top:6px; letter-spacing:.05em;">
    Space Sciences and Technologies Graduation Project · Space-Track GP Database · Skyfield SGP4 Propagator
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("### CONTROL PANEL")

# ─── SECTION 1: AUTO TLE DOWNLOAD ─────────────────────────────────────────
st.sidebar.markdown("""<div style="font-family:'Space Mono',monospace;font-size:.65rem;
    letter-spacing:.15em;color:#00c8ff;text-transform:uppercase;
    border-bottom:1px solid #1a2740;padding-bottom:4px;margin-bottom:8px;">
    1 — AUTO TLE DOWNLOAD</div>""", unsafe_allow_html=True)
st.sidebar.markdown("**Space-Track Authentication**")
user_email  = st.sidebar.text_input("Email", placeholder="user@domain.com")
user_pass   = st.sidebar.text_input("Password", placeholder="........", type="password")
st.sidebar.markdown("**Target Satellite Constellation** *(focused on LEO fleets only)*")
search_term = st.sidebar.selectbox("Select constellation",
    ["STARLINK", "ISS", "ONEWEB"], label_visibility="collapsed")
if st.sidebar.button("DOWNLOAD LIVE TLE DATA"):
    if user_email and user_pass:
        with st.spinner("Connecting to Space-Track database..."):
            key = "ISS" if search_term == "ISS" else search_term
            data = fetch_live_tles(user_email, user_pass, key)
            if data:
                st.session_state["tle_data"] = data
                st.session_state["loaded_group"] = search_term
                is3 = not (data[0].startswith("1 ") or data[0].startswith("2 "))
                count = len(data) // 3 if is3 else len(data) // 2
                st.sidebar.success(f"{count} satellites loaded.")
    else:
        st.sidebar.warning("Authentication required.")

st.sidebar.markdown("---")

# ─── SECTION 2: MANUAL TLE ENTRY ─────────────────────────────────────────────
st.sidebar.markdown("""<div style="font-family:'Space Mono',monospace;font-size:.65rem;
    letter-spacing:.15em;color:#00ff9d;text-transform:uppercase;
    border-bottom:1px solid #1a2740;padding-bottom:4px;margin-bottom:8px;">
    2 — ENTER YOUR SATELLITE (TLE)</div>""", unsafe_allow_html=True)
st.sidebar.markdown("<small style='color:#4a6880;'>3-line TLE (name + line1 + line2)</small>",
                    unsafe_allow_html=True)
manual_tle_text = st.sidebar.text_area(
    "Manual TLE",
    height=110,
    placeholder="MY-SAT\n1 99999U ...\n2 99999  ...",
    label_visibility="collapsed",
    key="manual_tle_input",
)
if st.sidebar.button("LOAD MANUAL TLE"):
    lines = [l.strip() for l in manual_tle_text.strip().split("\n") if l.strip()]
    if len(lines) >= 3:
        try:
            my_sat = EarthSatellite(lines[1], lines[2], lines[0], ts)
            st.session_state["my_sat"] = my_sat
            st.sidebar.success(f"✓ {my_sat.name} loaded.")
        except Exception as e:
            st.sidebar.error(f"TLE error: {e}")
    elif len(lines) == 2:
        try:
            my_sat = EarthSatellite(lines[0], lines[1], "CUSTOM-SAT", ts)
            st.session_state["my_sat"] = my_sat
            st.sidebar.success("✓ CUSTOM-SAT loaded.")
        except Exception as e:
            st.sidebar.error(f"TLE error: {e}")
    else:
        st.sidebar.warning("Enter at least 2 TLE lines.")

if "my_sat" in st.session_state:
    ms = st.session_state["my_sat"]
    st.sidebar.markdown(f"""<div style="font-family:'Space Mono',monospace;font-size:.65rem;
        color:#00ff9d;padding:6px 10px;background:rgba(0,255,157,.05);
        border:1px solid rgba(0,255,157,.2);border-radius:2px;margin-top:4px;">
        ✓ ACTIVE: {ms.name}</div>""", unsafe_allow_html=True)
    if st.sidebar.button("Delete My Satellite"):
        del st.session_state["my_sat"]
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""<div style="font-family:'Space Mono',monospace;font-size:.65rem;
    letter-spacing:.15em;color:#4a6880;text-transform:uppercase;
    border-bottom:1px solid #1a2740;padding-bottom:4px;margin-bottom:8px;">
    3 — ANALYSIS PARAMETERS</div>""", unsafe_allow_html=True)
window_hrs  = st.sidebar.slider("Analysis window (hours)", 1, 48, 24)
sigma_km    = st.sidebar.select_slider("Position uncertainty σ (km)",
    options=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0], value=0.5)
sat_limit   = st.sidebar.slider("Maximum satellite count", 5, 30, 15)
hbr_km      = st.sidebar.select_slider("Hard-Body Radius HBR (km)",
    options=[0.005, 0.010, 0.020, 0.050, 0.100], value=0.020)
mass_a_kg   = st.sidebar.slider("Object A Mass (kg)", 10, 5000, 250)
mass_b_kg   = st.sidebar.slider("Object B Mass (kg)", 10, 5000, 250)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model:** Chan 1997 + Foster 1992  
**Propagator:** SGP4/SDP4  
**Filter:** Apsis + Distance  
**Data:** Space-Track GP  
**TCA Step:** 5 min  
**HBR:** User selected
""")

# DATA CHECK
if "tle_data" not in st.session_state:
    st.info("Download data by entering your Space-Track credentials in the left panel.")
    st.markdown("""
    <div class="info-panel">
      <b>How to use?</b><br>
      1. Create a free account at <b>space-track.org</b>.<br>
      2. Enter your email and password in the left panel.<br>
      3. Select a satellite constellation and click <b>DOWNLOAD LIVE TLE DATA</b>.<br>
      4. All tabs will become active.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

sats = parse_tles(st.session_state["tle_data"], limit=sat_limit)
if not sats:
    st.error("TLE parsing failed.")
    st.stop()

# TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "DASHBOARD",
    "CONJUNCTION ANALYSIS",
    "YOUR SATELLITE",
    "LIVE SIMULATION",
    "3D ORBIT & GROUND TRACK",
    "ORBITAL ELEMENTS",
    "METHODOLOGY",
])

# ── TAB 1: DASHBOARD ───────────────────────────────────────────────────
with tab1:
    with st.spinner("Apsis filter + conjunction analysis..."):
        df, n_filtered, n_total = compute_conjunctions(sats, window_hrs, sigma_km, mass_a_kg, mass_b_kg)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(f"""<div style="font-family:'Space Mono',monospace; font-size:.65rem;
         color:#4a6880; text-align:right; margin-bottom:14px;">Last update: {now_str}</div>""",
        unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Tracked Satellites", len(sats))
    with c2:
        st.metric("Total Pairs", n_total)
    with c3:
        st.metric("Passed Apsis Filter", n_total - n_filtered)
    with c4:
        n_conj = len(df) if not df.empty else 0
        st.metric("Conjunction Events (<500km)", n_conj)
    with c5:
        n_crit = len(df[df["Risk Level"] == "CRITICAL"]) if not df.empty else 0
        st.metric("Critical Risk", n_crit)

    if n_filtered > 0:
        st.markdown(f"""<div class="info-panel">
        <b>Apsis Filter:</b> {n_filtered} pairs filtered without orbit propagation
        due to non-overlapping altitude bands — computation time reduced by
        %{round(n_filtered/n_total*100,1)}.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if df.empty:
        st.success(f"No conjunctions below 500 km detected in {window_hrs}-hour window.")
    else:
        # Dilution warning
        n_dil = df["Dilution"].sum() if not df.empty else 0
        if n_dil > 0:
            st.markdown(f"""<div class="warn-panel">
            <b>PROBABILITY DILUTION WARNING:</b> Wide covariance in {int(n_dil)} events
            may be masking Pc values. Check Max-Pc values in Conjunction Analysis tab.
            </div>""", unsafe_allow_html=True)

        show_cols = ["TCA (UTC)", "Object A", "Object B", "Distance (km)",
                     "Relative Velocity (km/s)", "Pc (scientific)", "Pc Max",
                     "Mahalanobis Md", "Ec (J/g)", "Risk Level"]
        RISK_COLORS = {"CRITICAL":"#ff2b4d","HIGH":"#ff6b00","MEDIUM":"#ffaa00","LOW":"#00ff9d"}
        MONO = "font-family:'Space Mono',monospace; font-size:0.76rem;"

        df_show = df[show_cols].copy()
        styled = (
            df_show.style
            .map(lambda v: f"color:{RISK_COLORS.get(str(v),'#b8cfe0')};font-weight:bold;{MONO}",
                 subset=["Risk Level"])
            .map(lambda v: f"color:#00c8ff;{MONO}", subset=["Pc (scientific)"])
            .map(lambda v: f"color:#ff9060;{MONO}", subset=["Pc Max"])
            .map(lambda v: (f"color:#ff2b4d;{MONO}" if float(v) < 1.5 else f"color:#b8cfe0;{MONO}"),
                 subset=["Mahalanobis Md"])
            .map(lambda v: (f"color:#ff2b4d;{MONO}" if float(v) >= 40 else f"color:#b8cfe0;{MONO}"),
                 subset=["Ec (J/g)"])
            .format({"Distance (km)":"{:.3f}", "Relative Velocity (km/s)":"{:.3f}",
                     "Pc Max":"{:.3e}", "Mahalanobis Md":"{:.2f}", "Ec (J/g)":"{:.1f}"})
            .set_properties(**{"font-family":"Space Mono,monospace","font-size":"0.76rem"})
        )
        csv_bytes = df_show.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download Report as CSV", data=csv_bytes,
            file_name=f"conjunction_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
        st.dataframe(styled, use_container_width=True)

# ── TAB 2: CONJUNCTION ANALYSIS ─────────────────────────────────────────────────
with tab2:
    if df is None or df.empty:
        st.success("No critical conjunction events in selected window.")
    else:
        st.markdown("**Detailed Review — Select Event**")
        options = [f"{r['Object A']}  <->  {r['Object B']}  |  TCA {r['TCA (UTC)']}  |  {r['Distance (km)']} km"
                   for _, r in df.iterrows()]
        sel = st.selectbox("Conjunction event", options, label_visibility="collapsed")
        idx = options.index(sel)
        row = df.iloc[idx]

        # Dilution warning
        if row["Dilution"]:
            st.markdown(f"""<div class="crit-panel">
            <b>PROBABILITY DILUTION:</b> {row["Dilution Message"]}
            </div>""", unsafe_allow_html=True)

        # Plot + gauge
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.plotly_chart(fig_distance_profile(
                row["_dist_arr"], window_hrs, row["Distance (km)"], sigma_km),
                use_container_width=True)
        with col_r:
            st.plotly_chart(fig_risk_gauge(row["Pc (isotropic)"]), use_container_width=True)

        # Pc comparison
        st.markdown("**Collision Probability Model Comparison**")
        pc_cols = st.columns(3)
        with pc_cols[0]:
            st.metric("Chan 1997 (Isotropic)", f"{row['Pc (isotropic)']:.3e}")
        with pc_cols[1]:
            st.metric("Foster 1992 (2D-Pc)", f"{row['Pc (Foster 2D)']:.3e}")
        with pc_cols[2]:
            st.metric("Max Pc (Worst Case)", f"{row['Pc Max']:.3e}")

        # Mahalanobis test
        mah_color = "#ff2b4d" if row["2D-Pc Valid"] != "2D-Pc Valid" else "#00ff9d"
        st.markdown(f"""<div class="info-panel">
        <b>Mahalanobis Distance Test:</b> Md = {row['Mahalanobis Md']:.3f} —
        <span style="color:{mah_color};">{row['2D-Pc Valid']}</span><br>
        <small>Md < 1.5 → linear motion assumption breaks down → 3D-Pc required (CARA methodology)</small>
        </div>""", unsafe_allow_html=True)

        # Fragmentation analysis
        frag = fragmentation_probability(row["Relative Velocity (km/s)"])
        st.markdown("**Collision Consequence Analysis**")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.metric("Specific Kinetic Energy (J/g)", f"{frag['E_c_J_per_g']:.1f}")
        with fc2:
            st.metric("Fragmentation Level", frag["level"])
        with fc3:
            st.metric("Estimated Debris Objects", frag["est_debris"])
        st.markdown(f"""<div class="info-panel" style="border-left-color:{frag['color']};">
        <b>{frag['level']}:</b> {frag['desc']}<br>
        <small>Ec ≥ 40 J/g → Catastrophic fragmentation (Kessler Syndrome contribution)</small>
        </div>""", unsafe_allow_html=True)

        # Full parameter table
        st.markdown("**Full Event Parameters**")
        det = {
            "Object A": row["Object A"], "Object B": row["Object B"],
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

# ── TAB 3: YOUR SATELLITE ───────────────────────────────────────────────────────
with tab3:
    st.markdown("## Analyze Your Satellite")
    if "my_sat" not in st.session_state:
        st.markdown("""<div class="warn-panel">
        <b>You haven't loaded your satellite yet.</b><br>
        Enter your TLE data in the <b>2 — ENTER YOUR SATELLITE (TLE)</b> section
        in the left panel and click <b>LOAD MANUAL TLE</b>.
        </div>""", unsafe_allow_html=True)
    elif "tle_data" not in st.session_state:
        st.markdown("""<div class="warn-panel">
        <b>Fleet data not loaded.</b><br>
        First perform automatic TLE download from the left panel; then comparison
        with your satellite can be done.
        </div>""", unsafe_allow_html=True)
    else:
        my_sat = st.session_state["my_sat"]
        st.markdown(f"""<div class="info-panel">
        <b>Active satellite:</b> {my_sat.name}&nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Fleet:</b> {st.session_state.get('loaded_group','—')}&nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Analysis window:</b> {window_hrs} hours&nbsp;&nbsp;|&nbsp;&nbsp;
        <b>σ:</b> {sigma_km} km
        </div>""", unsafe_allow_html=True)

        with st.spinner(f"Running conjunction analysis for {my_sat.name}..."):
            df_my = compute_conjunctions_custom(my_sat, sats, window_hrs, sigma_km, mass_a_kg, mass_b_kg)

        if df_my.empty:
            st.success(f"No conjunctions below 500 km for {my_sat.name} in {window_hrs}-hour window.")
        else:
            n_crit_my = len(df_my[df_my["Risk Level"] == "CRITICAL"])
            n_high_my = len(df_my[df_my["Risk Level"] == "HIGH"])

            c1m, c2m, c3m, c4m = st.columns(4)
            with c1m: st.metric("Total Conjunctions", len(df_my))
            with c2m: st.metric("Critical Risk", n_crit_my)
            with c3m: st.metric("High Risk", n_high_my)
            with c4m: st.metric("Min. Distance (km)", f"{df_my['Distance (km)'].min():.2f}")

            st.markdown("**Conjunctions — Risk Table**")
            RISK_COLORS = {"CRITICAL":"#ff2b4d","HIGH":"#ff6b00","MEDIUM":"#ffaa00","LOW":"#00ff9d"}
            MONO = "font-family:'Space Mono',monospace; font-size:0.76rem;"
            show_c = ["TCA (UTC)","Object A","Object B","Distance (km)","Relative Velocity (km/s)",
                      "Pc (scientific)","Pc Max","Mahalanobis Md","Ec (J/g)","Risk Level"]
            df_my_show = df_my[show_c].copy()
            styled_my = (
                df_my_show.style
                .map(lambda v: f"color:{RISK_COLORS.get(str(v),'#b8cfe0')};font-weight:bold;{MONO}",
                     subset=["Risk Level"])
                .map(lambda v: f"color:#00c8ff;{MONO}", subset=["Pc (scientific)"])
                .map(lambda v: f"color:#ff9060;{MONO}", subset=["Pc Max"])
                .map(lambda v: (f"color:#ff2b4d;{MONO}" if float(v) < 1.5 else f"color:#b8cfe0;{MONO}"),
                     subset=["Mahalanobis Md"])
                .format({"Distance (km)":"{:.3f}","Relative Velocity (km/s)":"{:.3f}",
                         "Pc Max":"{:.3e}","Mahalanobis Md":"{:.2f}","Ec (J/g)":"{:.1f}"})
                .set_properties(**{"font-family":"Space Mono,monospace","font-size":"0.76rem"})
            )
            csv_my = df_my_show.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Download Report as CSV", data=csv_my,
                file_name=f"my_satellite_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")
            st.dataframe(styled_my, use_container_width=True)

            # Detailed analysis for selected pair
            st.markdown("---")
            st.markdown("**Detailed Pair Analysis — Select Event**")
            opts_my = [f"{r['Object B']}  |  TCA {r['TCA (UTC)']}  |  {r['Distance (km)']} km"
                       for _, r in df_my.iterrows()]
            sel_my  = st.selectbox("Select event", opts_my, label_visibility="collapsed", key="my_sel")
            idx_my  = opts_my.index(sel_my)
            row_my  = df_my.iloc[idx_my]

            if row_my["Dilution"]:
                st.markdown(f"""<div class="crit-panel">
                <b>PROBABILITY DILUTION:</b> {row_my["Dilution Message"]}</div>""",
                unsafe_allow_html=True)

            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.plotly_chart(
                    fig_distance_profile(row_my["_dist_arr"], window_hrs, row_my["Distance (km)"], sigma_km),
                    use_container_width=True)
            with col_r:
                st.plotly_chart(fig_risk_gauge(row_my["Pc (isotropic)"]), use_container_width=True)

            pc_c = st.columns(3)
            with pc_c[0]: st.metric("Chan 1997 (Isotropic)", f"{row_my['Pc (isotropic)']:.3e}")
            with pc_c[1]: st.metric("Foster 1992 (2D-Pc)",   f"{row_my['Pc (Foster 2D)']:.3e}")
            with pc_c[2]: st.metric("Max Pc",                 f"{row_my['Pc Max']:.3e}")

            mah_c = "#ff2b4d" if row_my["2D-Pc Valid"] != "2D-Pc Valid" else "#00ff9d"
            st.markdown(f"""<div class="info-panel">
            <b>Mahalanobis Test:</b> Md = {row_my['Mahalanobis Md']:.3f} —
            <span style="color:{mah_c};">{row_my['2D-Pc Valid']}</span>
            </div>""", unsafe_allow_html=True)

            frag_my = fragmentation_probability(row_my["Relative Velocity (km/s)"])
            fc = st.columns(3)
            with fc[0]: st.metric("Ec (J/g)", f"{frag_my['E_c_J_per_g']:.1f}")
            with fc[1]: st.metric("Fragmentation", frag_my["level"])
            with fc[2]: st.metric("Estimated Debris", frag_my["est_debris"])

            # Send to simulation button
            st.markdown("---")
            if st.button("🔭 Show This Pair in Live Simulation", key="my_to_sim"):
                st.session_state["sim_sat_a"] = row_my["_s1"]
                st.session_state["sim_sat_b"] = row_my["_s2"]
                st.success("Pair transferred to 'LIVE SIMULATION' tab.")


# ── TAB 4: LIVE SIMULATION ──────────────────────────────────────────────────
with tab4:
    st.markdown("## Live 3D Orbit Simulation")
    st.markdown("""<div class="info-panel">
    Watch the encounter between two satellites with <b>real-time</b> animation.
    Focus on the risk moment with Play / Stop / Speed controls and <b>Jump to TCA</b> button.
    </div>""", unsafe_allow_html=True)

    # Satellite selection source
    if "sim_sat_a" in st.session_state and "sim_sat_b" in st.session_state:
        default_a = st.session_state["sim_sat_a"].name
        default_b = st.session_state["sim_sat_b"].name
        st.markdown(f"""<div class="info-panel">
        <b>Selected pair:</b> {default_a} × {default_b}<br>
        <small>Use dropdown menus below to change.</small>
        </div>""", unsafe_allow_html=True)
    else:
        default_a = sats[0].name if sats else ""
        default_b = sats[1].name if len(sats) > 1 else ""

    sat_names  = [s.name for s in sats]
    if "my_sat" in st.session_state:
        sat_names_ext = [st.session_state["my_sat"].name] + sat_names
        all_sats_ext  = [st.session_state["my_sat"]] + sats
    else:
        sat_names_ext = sat_names
        all_sats_ext  = sats

    sc1, sc2, sc3 = st.columns([2, 2, 1])
    with sc1:
        sel_a = st.selectbox("Satellite A", sat_names_ext,
                             index=sat_names_ext.index(default_a) if default_a in sat_names_ext else 0,
                             key="sim_a")
    with sc2:
        sel_b = st.selectbox("Satellite B", sat_names_ext,
                             index=sat_names_ext.index(default_b) if default_b in sat_names_ext else min(1,len(sat_names_ext)-1),
                             key="sim_b")
    with sc3:
        sim_hrs = st.slider("Window (hours)", 1, 12, 6, key="sim_hrs")

    # Display options
    st.markdown("**Display Options**")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        show_orbits = st.checkbox("Show Orbit Trails", value=True, key="show_orbits")
    with col_opt2:
        show_tca = st.checkbox("Show TCA Marker", value=True, key="show_tca")

    if sel_a == sel_b:
        st.warning("Select two different satellites.")
    else:
        sat_obj_a = next(s for s in all_sats_ext if s.name == sel_a)
        sat_obj_b = next(s for s in all_sats_ext if s.name == sel_b)

        if st.button("▶ START SIMULATION", key="start_sim"):
            if sim_hrs < 0.5:
                st.warning("Simulation window must be at least 30 minutes.")
            else:
                st.session_state["sim_sat_a"] = sat_obj_a
                st.session_state["sim_sat_b"] = sat_obj_b
                st.session_state["run_sim"]   = True

        if st.session_state.get("run_sim") and \
           "sim_sat_a" in st.session_state and "sim_sat_b" in st.session_state:
            sa = st.session_state["sim_sat_a"]
            sb = st.session_state["sim_sat_b"]
            with st.spinner("Calculating orbits and creating animation..."):
                anim_fig, tca_i, tca_d, dists_arr, jd_arr = \
                    fig_animated_conjunction(sa, sb, sim_hrs, show_orbits, show_tca)

            # TCA info
            tca_utc = ts.tt_jd(jd_arr[tca_i]).utc_strftime("%Y-%m-%d %H:%M:%S UTC")
            sev_sim, col_sim = risk_level(
                collision_probability_isotropic(tca_d, sigma_km))
            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1: st.metric("TCA Time (UTC)", tca_utc)
            with tc2: st.metric("Min. Distance (km)", f"{tca_d:.3f}")
            with tc3: st.metric("TCA T+ (min)", tca_i * 2)
            with tc4: st.metric("Risk", sev_sim)

            # 3D animation
            st.info("ℹ️ Note: Camera rotation is only available when animation is paused. Use STOP or the slider to pause, then rotate the view.")
            st.plotly_chart(anim_fig, use_container_width=True)

            # Distance profile (static)
            st.markdown("**Distance Profile (Full Window)**")
            step_m_sim = 2
            t_ax = np.arange(len(dists_arr)) * step_m_sim / 60.0
            fig_dp_sim = go.Figure()
            fig_dp_sim.add_hline(y=0.02, line=dict(color="#ff2b4d", dash="dot", width=1),
                                 annotation_text="HBR (20 m)")
            fig_dp_sim.add_trace(go.Scatter(x=t_ax, y=dists_arr, mode="lines",
                line=dict(color="#00c8ff", width=1.5), fill="tozeroy",
                fillcolor="rgba(0,200,255,.04)", name="Distance (km)"))
            fig_dp_sim.add_trace(go.Scatter(
                x=[t_ax[tca_i]], y=[dists_arr[tca_i]],
                mode="markers+text",
                marker=dict(color="#ff2b4d", size=10),
                text=[f" TCA {dists_arr[tca_i]:.1f} km"],
                textfont=dict(size=9, color="#ff2b4d", family="Space Mono"),
                name="TCA"))
            fig_dp_sim.update_layout(
                **DARK, height=240,
                xaxis=dict(title="Time (hours)", gridcolor="#1a2740", zeroline=False),
                yaxis=dict(title="Distance (km)", gridcolor="#1a2740", zeroline=False),
                title=dict(text=f"Distance Profile — {sa.name} × {sb.name}",
                    font=dict(size=11, family="Barlow Condensed", color="#00c8ff"), x=0.01),
                margin=dict(l=10, r=10, t=35, b=10),
                legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_dp_sim, use_container_width=True)


# ── TAB 5: 3D ORBIT & GROUND TRACK ───────────────────────────────────────────
with tab5:
    c1_3d, c2_3d = st.columns([3, 2])
    with c1_3d:
        st.markdown("**3D Orbit View**")
        with st.spinner("Loading Earth texture..."):
            st.plotly_chart(fig_3d_orbits(sats), use_container_width=True, height=560)
    with c2_3d:
        st.markdown("**Ground Track Map**")
        with st.spinner("Calculating..."):
            st.plotly_chart(fig_ground_tracks(sats), use_container_width=True)
        st.markdown("""<div style="font-family:'Space Mono',monospace; font-size:.65rem;
             color:#2a4060; line-height:2; margin-top:8px;">
          Approximately 95-minute track shown for each satellite.<br>
          Large dots represent current position.<br>
          Ground track calculated with SGP4/SDP4 propagator.
        </div>""", unsafe_allow_html=True)

# ── TAB 6: ORBITAL ELEMENTS ────────────────────────────────────────────────
with tab6:
    st.markdown("## Orbital Elements and Space Distribution")
    elems_list = [(sat.name, get_orbital_elements(sat)) for sat in sats]

    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.markdown("**Kepler Orbital Elements Table**")
        rows = []
        for name, elems in elems_list:
            if elems:
                rows.append({
                    "Satellite": name[:18],
                    "Altitude (km)": elems.get("Mean Altitude (km)", "-"),
                    "Inclination (°)": elems.get("Inclination i (°)", "-"),
                    "Eccentricity": elems.get("Eccentricity e", "-"),
                    "Period (min)": elems.get("Orbital Period (min)", "-"),
                })
        if rows:
            df_elems = pd.DataFrame(rows)
            st.dataframe(df_elems, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Altitude / Inclination Distribution** (dot size = eccentricity)")
        st.plotly_chart(fig_orbital_elements_radar(elems_list), use_container_width=True)

    with st.expander("Selected Satellite Detail"):
        sel_sat = st.selectbox("Select satellite", [s.name for s in sats], key="elem_sel")
        sel_elems = next((e for n, e in elems_list if n == sel_sat), {})
        if sel_elems:
            df_single = pd.DataFrame(sel_elems.items(), columns=["Element", "Value"])
            st.dataframe(df_single, use_container_width=True, hide_index=True)

# ── TAB 7: METHODOLOGY ────────────────────────────────────────────────────────
with tab7:
    st.markdown("## Methodology and Theoretical Background")
    st.markdown("""
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
    """, unsafe_allow_html=True)

    st.markdown("### References")
    st.markdown("""<div style="font-family:'Space Mono',monospace; font-size:.7rem;
         color:#4a6880; line-height:2.4;">
    Foster, J.L. &amp; Estes, H.S. (1992). A parametric analysis of orbital debris collision probability
    and maneuver rate for space vehicles. <i>NASA Technical Memorandum.</i><br>
    Chan, F.K. (1997). <i>Spacecraft Collision Probability.</i> The Aerospace Press.<br>
    Hoots, F.R. &amp; Roehrich, R.L. (1980). <i>Models for Propagation of NORAD Element Sets.</i>
    Spacetrack Report No. 3.<br>
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
    </div>""", unsafe_allow_html=True)
