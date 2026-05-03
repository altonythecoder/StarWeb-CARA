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
#  CSS — GÖREV KONTROL KARANLIK TEMASI
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
/* Sidebar üst başlık (kırık Material Icons ikonu) gizle */
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
#  DÜNYA GÖRÜNÜMÜ
# ================================================================================
@st.cache_data(show_spinner=False)
def load_earth_texture(resolution: int = 270):
    try:
        url = ("https://upload.wikimedia.org/wikipedia/commons/thumb/"
               "c/cd/Land_ocean_ice_2048.jpg/1024px-Land_ocean_ice_2048.jpg")
        resp = requests.get(url, timeout=20)
        img  = Image.open(BytesIO(resp.content)).convert("RGB")
        W, H = resolution * 2, resolution
        img  = img.resize((W, H), Image.LANCZOS)
        imgq = img.quantize(colors=256)
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
        return None

# ================================================================================
#  VERİ ÇEKME
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
            st.sidebar.error(f"'{search_query}' için sonuç bulunamadı.")
            return None
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        return lines if len(lines) >= 2 else None
    except Exception as e:
        st.sidebar.error(f"API Hatası: {e}")
        return None

# ================================================================================
#  TLE AYRIŞTIRMA
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
#  YÖRÜNGE ELEMANLARI (TLE'den)
# ================================================================================
def get_orbital_elements(sat: EarthSatellite) -> dict:
    """TLE'den Kepler yörünge elemanlarını çıkarır."""
    try:
        model = sat.model
        # TLE epoch'tan elemanlar
        incl   = math.degrees(model.inclo)          # eğim (deg)
        raan   = math.degrees(model.nodeo)           # yükselen düğüm (deg)
        ecc    = model.ecco                          # dışmerkezlik
        argp   = math.degrees(model.argpo)           # perije argümanı (deg)
        mean_m = math.degrees(model.mo)              # ortalama anomali (deg)
        n_rpm  = model.no_kozai * (60.0 / (2*math.pi))  # rad/dk → devir/dk
        # Büyük yarı eksen: a = (GM/n^2)^(1/3), n rad/s
        GM     = 398600.4418  # km^3/s^2
        n_rads = model.no_kozai / 60.0  # rad/s
        a_km   = (GM / n_rads**2) ** (1/3)
        alt_km = a_km - 6371.0
        period_min = 2 * math.pi / model.no_kozai
        return {
            "Büyük Yarı Eksen a (km)":   round(a_km, 1),
            "Ortalama İrtifa (km)":       round(alt_km, 1),
            "Dışmerkezlik e":             f"{ecc:.6f}",
            "Eğim i (°)":                round(incl, 4),
            "Yükselen Düğüm (RAAN) (°)": round(raan, 4),
            "Perije Argümanı ω (°)":      round(argp, 4),
            "Ortalama Anomali M (°)":     round(mean_m, 4),
            "Yörünge Periyodu (dk)":      round(period_min, 2),
            "Ortalama Hareket n (dev/dk)": round(n_rpm, 6),
        }
    except Exception:
        return {}

# ================================================================================
#  APSİS FİLTRESİ (Bölüm 2.1 — Tez)
# ================================================================================
def apsis_filter(sats: list, threshold_km: float = 50.0) -> list:
    """
    Apsis (Apoje-Perije) Filtresi — Bölüm 2.1
    Yörünge yükseklik bantları örtüşmeyen çiftleri eleyerek O(N^2)
    karmaşıklığını azaltır.
    q1 > Q2 + D   →   fiziksel kesişim imkânsız → elenir
    """
    R_E = 6371.0
    GM  = 398600.4418

    def apsis(sat):
        try:
            n   = sat.model.no_kozai / 60.0  # rad/s
            a   = (GM / n**2) ** (1/3)
            e   = sat.model.ecco
            per = a * (1 - e) - R_E          # perije irtifası
            apo = a * (1 + e) - R_E          # apoje irtifası
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
#  FOSTER 1992 2D-Pc (Bölüm 3.1 — Tez)
# ================================================================================
def foster_2d_pc(miss_km: float, sigma_r: float, sigma_t: float,
                  sigma_n: float, hbr_km: float = 0.020) -> float:
    """
    Foster & Estes (1992) 2D-Pc Modeli — Bölüm 3.1
    Encounter plane'e iz düşürülmüş Gauss integrali.
    sigma_r: radyal (km), sigma_t: iz yönü (km), sigma_n: dik (km)
    Birleşik kovaryans: encounter plane'deki iki bileşen.
    """
    try:
        # Encounter plane bileşenleri (radyal + normal kullan)
        sig_x = math.sqrt(sigma_r**2 + sigma_r**2)   # combined radyal
        sig_y = math.sqrt(sigma_n**2 + sigma_n**2)   # combined normal
        if sig_x <= 0 or sig_y <= 0:
            return 0.0
        # 2D Gauss üzerinden daire integrali (sayısal)
        # Gauss dağılımı TCA ortalama miss noktasında (miss_km, 0) merkezli;
        # entegrasyon bölgesi HBR yarıçaplı daire orijin etrafında (gerçek çarpışma küresi).
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
    Chan (1997) izotropik model — hızlı fallback.
    Doğru formül: P(|X_rel| ≤ HBR) x ∈ N(miss, σ) için
    Pc = Φ((HBR - miss)/σ) + Φ((HBR + miss)/σ) - 1
    """
    if sigma_km <= 0:
        return 0.0
    pc = norm.cdf((hbr_km - miss_km) / sigma_km) + norm.cdf((hbr_km + miss_km) / sigma_km) - 1.0
    return max(float(pc), 0.0)

# Dışa açık alias
def collision_probability(miss_km, sigma_km, hbr_km=0.020):
    return collision_probability_isotropic(miss_km, sigma_km, hbr_km)

# ================================================================================
#  MAHALANOBİS MESAFESİ TESTİ (Bölüm 3.2 — Tez)
# ================================================================================
def mahalanobis_test(miss_km: float, sigma_km: float) -> dict:
    """
    2D-Pc geçerlilik testi (CARA metodolojisi — Bölüm 3.2).
    Mahalanobis mesafesi Md = miss / sigma.
    Md < 1.5 → doğrusal hareket varsayımı çöküyor → 3D-Pc gerekli.
    """
    if sigma_km <= 0:
        return {"Md": 999.0, "valid_2d": True, "label": "Geçerli"}
    Md = miss_km / sigma_km
    valid = Md >= 1.5
    if Md < 0.5:
        label = "Geçersiz — 3D-Pc / Monte Carlo gerekli"
    elif Md < 1.5:
        label = "Sınırda — 3D-Pc tavsiye edilir"
    else:
        label = "2D-Pc Geçerli"
    return {"Md": round(Md, 3), "valid_2d": valid, "label": label}

# ================================================================================
#  MAKSIMUM Pc ANALİZİ (Bölüm 4 — Tez)
# ================================================================================
def max_pc_analysis(miss_km: float, hbr_km: float = 0.020) -> float:
    """
    Max Pc — Bölüm 4 (CARA araç seti).
    Kovaryans çarpanı σ taranarak matematiksel maksimum Pc bulunur.
    En kötü senaryo: σ_opt = miss / sqrt(2) (Gauss zirve noktası).
    """
    sigma_opt = miss_km / math.sqrt(2.0) if miss_km > 0 else hbr_km
    return collision_probability_isotropic(miss_km, max(sigma_opt, 1e-6), hbr_km)

# ================================================================================
#  OLASILIK SEYRELMESİ TESPİTİ (Bölüm 4 — Tez)
# ================================================================================
def dilution_check(pc: float, sigma_km: float, miss_km: float) -> dict:
    """
    Probability Dilution (Olasılık Seyrelmesi) tespiti — Bölüm 4.
    Geniş kovaryans → küçük Pc → sahte güven.
    Uyarı: sigma > 5*miss_km ve pc < 1e-6
    """
    diluted = (sigma_km > 5.0 * miss_km) and (pc < 1e-6) and (miss_km < 100.0)
    if diluted:
        return {
            "diluted": True,
            "msg": "OLASILIK SEYRELMESİ TESPIT EDILDI: Genis kovaryans Pc degerini maskeliyor. "
                   "WSPRT veya Max-Pc analizi gereklidir.",
        }
    return {"diluted": False, "msg": "Normal"}

# ================================================================================
#  PARÇALANMA OLASILIĞI Pf (Bölüm 4 — Tez)
# ================================================================================
def fragmentation_probability(rel_vel_km_s: float,
                               mass_a_kg: float = 250.0,
                               mass_b_kg: float = 250.0) -> dict:
    """
    Çarpışma Sonucu (Collision Consequence) — Bölüm 4.
    NASA operasyonel rehberine göre kinetik enerji bazlı parçalanma riski.
    Özgül Enerji: E_c = 0.5 * m_b * v_rel^2 / m_a  (J/g)
    E_c > 40 J/g → Katastrofik parçalanma (Kessler katkısı)
    E_c > 0 J/g  → Hasar verici
    """
    v_ms = rel_vel_km_s * 1000.0
    E_c  = 0.5 * mass_b_kg * v_ms**2 / (mass_a_kg * 1000.0)  # J/g
    if E_c >= 40.0:
        pf_level = "KATASTROFIK"
        pf_color = "#ff2b4d"
        pf_desc  = "Tam parçalanma — Kessler katkısı muhtemel"
    elif E_c >= 10.0:
        pf_level = "CIDDI"
        pf_color = "#ff6b00"
        pf_desc  = "Operasyonel kayıp ve önemli enkaz"
    elif E_c >= 1.0:
        pf_level = "HASARLI"
        pf_color = "#ffaa00"
        pf_desc  = "Kısmi hasar veya subsystem arızası"
    else:
        pf_level = "DUSUK"
        pf_color = "#00ff9d"
        pf_desc  = "Küçük hasar — parçalanma düşük ihtimal"
    n_debris = int(0.1 * (mass_a_kg + mass_b_kg) * (rel_vel_km_s / 7.0))
    return {
        "E_c_J_per_g": round(E_c, 2),
        "level":       pf_level,
        "color":       pf_color,
        "desc":        pf_desc,
        "est_debris":  n_debris,
    }

# ================================================================================
#  RISK SEVİYESİ
# ================================================================================
def risk_level(pc: float) -> tuple:
    """NASA STD-8719.14 — 4 kademeli risk sınıflandırması."""
    if   pc > 1e-3: return "KRİTİK", "#ff2b4d"
    elif pc > 1e-4: return "YÜKSEK", "#ff6b00"
    elif pc > 1e-5: return "ORTA",   "#ffaa00"
    else:           return "DÜŞÜK",  "#00ff9d"

# ================================================================================
#  ANA YAKINSAMA ANALİZİ (APSİS FİLTRELİ)
# ================================================================================
def compute_conjunctions(sats: list, window_hrs: int, sigma_km: float) -> tuple:
    """
    Apsis filtresi + 5 dk adımlı TCA taraması + çoklu Pc metrikleri.
    Returns: (df_results, n_apsis_filtered, n_total_pairs)
    """
    now         = ts.now()
    step_m      = 5
    n_steps     = window_hrs * 60 // step_m
    n_total     = len(list(combinations(sats, 2)))

    # Apsis ön filtresi
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
        frag      = fragmentation_probability(rel_vel)
        sev, color = risk_level(pc_iso)

        results.append({
            "TCA (UTC)":           best_t.utc_strftime("%Y-%m-%d %H:%M:%S"),
            "Obje A":              s1.name,
            "Obje B":              s2.name,
            "Mesafe (km)":         round(min_d, 3),
            "Görecel Hız (km/s)":  round(rel_vel, 3),
            "Pc (izotropik)":      pc_iso,
            "Pc (Foster 2D)":      pc_foster,
            "Pc Max":              pc_max,
            "Pc (bilimsel)":       f"{pc_iso:.3e}",
            "Mahalanobis Md":      mah["Md"],
            "2D-Pc Geçerli":       mah["label"],
            "Dilüsyon":            dil["diluted"],
            "Dilüsyon Mesajı":     dil["msg"],
            "Ec (J/g)":            frag["E_c_J_per_g"],
            "Parçalanma Seviyesi": frag["level"],
            "Tahmini Enkaz":       frag["est_debris"],
            "Risk Seviyesi":       sev,
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
#  GRAFİKLER
# ================================================================================
DARK = dict(
    paper_bgcolor="#07090f",
    plot_bgcolor="#07090f",
    font=dict(family="Space Mono, monospace", color="#b8cfe0", size=11),
)

def fig_3d_orbits(sats):
    now    = ts.now()
    fig    = go.Figure()
    earth  = load_earth_texture(270)
    if earth:
        x, y, z, sc, cs = earth
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=sc, colorscale=cs,
            showscale=False, opacity=1.0, hoverinfo="skip", name="Dunya",
            lightposition=dict(x=200000, y=80000, z=120000),
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
        pos   = sat.at(times).position.km
        c     = colors[k % len(colors)]
        fig.add_trace(go.Scatter3d(x=pos[0], y=pos[1], z=pos[2], mode="lines",
            line=dict(color=c, width=2.5), name=sat.name, opacity=0.95))
        p0 = sat.at(now).position.km
        fig.add_trace(go.Scatter3d(x=[p0[0]], y=[p0[1]], z=[p0[2]], mode="markers",
            marker=dict(color=c, size=6, symbol="circle", line=dict(color="#ffffff", width=1)),
            name=f"{sat.name} (simdi)", showlegend=False))
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
        title=dict(text="Zemin Izi -- Anlik Konum ve 95dk Yorunge",
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
        line=dict(color="#00c8ff", width=1.5), name="Mesafe (km)",
        fill="tozeroy", fillcolor="rgba(0,200,255,.04)"))
    tca_i = int(np.argmin(dist_arr))
    fig.add_trace(go.Scatter(x=[t_axis[tca_i]], y=[dist_arr[tca_i]],
        mode="markers+text", marker=dict(color="#ff2b4d", size=8),
        text=[f" TCA: {dist_arr[tca_i]:.1f} km"], textposition="top right",
        textfont=dict(size=9, family="Space Mono", color="#ff2b4d"),
        name="TCA", showlegend=False))
    fig.update_layout(**DARK, height=280,
        xaxis=dict(title="Zaman (saat)", gridcolor="#1a2740", zeroline=False, tickfont=dict(size=9)),
        yaxis=dict(title="Mesafe (km)", gridcolor="#1a2740", zeroline=False, tickfont=dict(size=9)),
        title=dict(text="Mesafe Profili — TCA Analizi",
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
    """Uyduları yörünge elemanları scatter ile göster."""
    fig = go.Figure()
    colors = ["#00c8ff","#00ff9d","#ffaa00","#ff6b00","#c060ff","#ff2b4d"]
    for k, (name, elems) in enumerate(elems_list):
        if not elems:
            continue
        try:
            alt  = float(str(elems.get("Ortalama İrtifa (km)", 0)))
            incl = float(str(elems.get("Eğim i (°)", 0)))
            ecc  = float(str(elems.get("Dışmerkezlik e", "0")))
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
        xaxis=dict(title="Eğim i (°)", gridcolor="#1a2740", zeroline=False, tickfont=dict(size=9)),
        yaxis=dict(title="İrtifa (km)", gridcolor="#1a2740", zeroline=False, tickfont=dict(size=9)),
        title=dict(text="Yörünge Uzayı — İrtifa / Eğim Dağılımı",
            font=dict(size=11, family="Barlow Condensed", color="#00c8ff"), x=0.01),
        margin=dict(l=10, r=10, t=35, b=10),
        showlegend=False,
    )
    return fig

# ================================================================================
#  KENDİ UYDUSU İÇİN YAKINSAMA ANALİZİ
# ================================================================================
def compute_conjunctions_custom(my_sat, sats: list, window_hrs: int, sigma_km: float) -> pd.DataFrame:
    """
    Kullanıcının kendi uydusunu mevcut uydu filosuyla karşılaştırır.
    Apsis filtresi + 5 dk TCA taraması + tam Pc metrikleri.
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
        # Apsis filtresi
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
        frag      = fragmentation_probability(rel_vel)
        sev, color = risk_level(pc_iso)

        results.append({
            "TCA (UTC)":           best_t.utc_strftime("%Y-%m-%d %H:%M:%S"),
            "Obje A":              my_sat.name,
            "Obje B":              sat.name,
            "Mesafe (km)":         round(min_d, 3),
            "Görecel Hız (km/s)":  round(rel_vel, 3),
            "Pc (izotropik)":      pc_iso,
            "Pc (Foster 2D)":      pc_foster,
            "Pc Max":              pc_max,
            "Pc (bilimsel)":       f"{pc_iso:.3e}",
            "Mahalanobis Md":      mah["Md"],
            "2D-Pc Geçerli":       mah["label"],
            "Dilüsyon":            dil["diluted"],
            "Dilüsyon Mesajı":     dil["msg"],
            "Ec (J/g)":            frag["E_c_J_per_g"],
            "Parçalanma Seviyesi": frag["level"],
            "Tahmini Enkaz":       frag["est_debris"],
            "Risk Seviyesi":       sev,
            "_color":              color,
            "_dist_arr":           dist_arr,
            "_s1":                 my_sat,
            "_s2":                 sat,
        })

    return pd.DataFrame(results)


# ================================================================================
#  CANLI 3D ANİMASYON (İki Uydu — TCA Odaklı)
# ================================================================================
def fig_animated_conjunction(sat_a, sat_b, window_hrs: int = 6):
    """
    İki uyduyu gerçek zamanlı animasyonla gösteren 3D Plotly figürü.
    - Oynat / Durdur / 2× / 5× hız butonları
    - Zaman kaydırıcısı
    - TCA'ya atla butonu
    - Uydular arası mesafe çizgisi (renge göre tehlike seviyesi)
    - Yörünge izleri
    """
    now     = ts.now()
    step_min = 2                           # dakika adımı
    n_frames = min(window_hrs * 30, 360)  # max 360 kare (12 saat)
    trail_len = 25                         # izin uzunluğu (kare sayısı)
    orbit_pts = 100                        # tam yörünge nokta sayısı

    # Tam yörünge yolları (statik arka plan)
    orb_off = np.linspace(0, 96, orbit_pts) / 1440.0
    orb_a   = sat_a.at(ts.tt_jd(now.tt + orb_off)).position.km
    orb_b   = sat_b.at(ts.tt_jd(now.tt + orb_off)).position.km

    # Animasyon adımlarındaki konumlar (vektörize)
    anim_off = np.arange(n_frames) * step_min / 1440.0
    anim_jd  = now.tt + anim_off
    pos_a    = sat_a.at(ts.tt_jd(anim_jd)).position.km   # (3, n_frames)
    pos_b    = sat_b.at(ts.tt_jd(anim_jd)).position.km
    dists    = np.linalg.norm(pos_a - pos_b, axis=0)      # (n_frames,)
    tca_idx  = int(np.argmin(dists))
    tca_dist = float(dists[tca_idx])

    def dist_color(d):
        if d < 50:   return "#ff2b4d"
        if d < 200:  return "#ffaa00"
        return "rgba(100,180,255,0.55)"

    # ── Statik figür oluştur ──────────────────────────────────────────────────
    fig = go.Figure()

    earth = load_earth_texture(200)
    if earth:
        x, y, z, sc, cs = earth
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=sc, colorscale=cs,
            showscale=False, opacity=1.0, hoverinfo="skip",
            lightposition=dict(x=200000, y=80000, z=120000),
            lighting=dict(ambient=0.6, diffuse=0.9, specular=0.03, roughness=0.85),
            name="Dünya",
        ))
    else:
        r = 6371.0
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        fig.add_trace(go.Surface(
            x=r*np.cos(u)*np.sin(v), y=r*np.sin(u)*np.sin(v), z=r*np.cos(v),
            colorscale="Blues", opacity=0.4, showscale=False))

    # Tam yörünge yolları (soluk, statik)
    fig.add_trace(go.Scatter3d(x=orb_a[0], y=orb_a[1], z=orb_a[2], mode="lines",
        line=dict(color="rgba(0,200,255,0.12)", width=1.5), name=sat_a.name+" yörüngesi",
        showlegend=False))
    fig.add_trace(go.Scatter3d(x=orb_b[0], y=orb_b[1], z=orb_b[2], mode="lines",
        line=dict(color="rgba(255,107,0,0.12)", width=1.5), name=sat_b.name+" yörüngesi",
        showlegend=False))

    # TCA noktası (statik kırmızı işaret)
    mid_tca = (pos_a[:, tca_idx] + pos_b[:, tca_idx]) / 2
    fig.add_trace(go.Scatter3d(
        x=[mid_tca[0]], y=[mid_tca[1]], z=[mid_tca[2]],
        mode="markers+text",
        marker=dict(color="#ff2b4d", size=10, symbol="diamond",
                    line=dict(color="#ffffff", width=1)),
        text=[f"TCA {tca_dist:.1f} km"], textposition="top right",
        textfont=dict(color="#ff2b4d", size=9, family="Space Mono"),
        name="TCA Noktası",
    ))

    n_static = len(fig.data)  # statik iz sayısı — dinamik izler bundan sonra

    # İlk dinamik durum (kare 0)
    def make_dynamic_traces(i):
        t0 = max(0, i - trail_len)
        ta = pos_a[:, t0:i+1]
        tb = pos_b[:, t0:i+1]
        dc = dist_color(dists[i])
        return [
            go.Scatter3d(x=ta[0], y=ta[1], z=ta[2], mode="lines",
                line=dict(color="#00c8ff", width=2.5),
                name=sat_a.name, showlegend=True),
            go.Scatter3d(x=tb[0], y=tb[1], z=tb[2], mode="lines",
                line=dict(color="#ff6b00", width=2.5),
                name=sat_b.name, showlegend=True),
            go.Scatter3d(x=[pos_a[0,i]], y=[pos_a[1,i]], z=[pos_a[2,i]],
                mode="markers",
                marker=dict(color="#00c8ff", size=9, line=dict(color="#fff",width=1)),
                name=sat_a.name+" pos", showlegend=False),
            go.Scatter3d(x=[pos_b[0,i]], y=[pos_b[1,i]], z=[pos_b[2,i]],
                mode="markers",
                marker=dict(color="#ff6b00", size=9, line=dict(color="#fff",width=1)),
                name=sat_b.name+" pos", showlegend=False),
            go.Scatter3d(
                x=[pos_a[0,i], pos_b[0,i]],
                y=[pos_a[1,i], pos_b[1,i]],
                z=[pos_a[2,i], pos_b[2,i]],
                mode="lines+text",
                line=dict(color=dc, width=2, dash="dot"),
                text=["", f"  Δ {dists[i]:.1f} km"],
                textfont=dict(color=dc, size=9, family="Space Mono"),
                name=f"Mesafe", showlegend=False,
            ),
        ]

    for tr in make_dynamic_traces(0):
        fig.add_trace(tr)

    dyn_idx = list(range(n_static, n_static + 5))

    # ── Animasyon kareleri ────────────────────────────────────────────────────
    frames = []
    slider_steps = []
    for i in range(n_frames):
        t_utc = ts.tt_jd(anim_jd[i]).utc_strftime("%H:%M UTC")
        t_min = i * step_min
        title_txt = (f"T+{t_min:04d} dk  |  {t_utc}  |  "
                     f"Δ {dists[i]:.1f} km"
                     + ("  ⚠ TCA" if i == tca_idx else ""))
        frames.append(go.Frame(
            data=make_dynamic_traces(i),
            traces=dyn_idx,
            name=str(i),
            layout=go.Layout(title_text=title_txt),
        ))
        lbl = t_utc if i % max(1, n_frames//20) == 0 else ""
        slider_steps.append(dict(
            args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
            label=lbl, method="animate",
        ))

    fig.frames = frames

    # ── Layout + kontroller ───────────────────────────────────────────────────
    fig.update_layout(
        **DARK,
        height=640,
        margin=dict(l=0, r=0, t=52, b=10),
        title=dict(
            text=f"{sat_a.name}  ×  {sat_b.name} — TCA: {tca_dist:.2f} km  (T+{tca_idx*step_min} dk)",
            font=dict(family="Barlow Condensed", color="#00c8ff", size=14), x=0.01,
        ),
        scene=dict(
            bgcolor="#000408",
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.7, y=1.7, z=0.75), up=dict(x=0,y=0,z=1)),
        ),
        legend=dict(font=dict(size=8, family="Space Mono"),
                    bgcolor="rgba(0,4,8,.85)", bordercolor="#1a2740", borderwidth=1,
                    x=0.01, y=0.99, itemsizing="constant"),
        updatemenus=[dict(
            type="buttons", showactive=False,
            bgcolor="#0c1018", bordercolor="#1a2740",
            font=dict(family="Space Mono", size=9, color="#b8cfe0"),
            y=1.06, x=0.0, xanchor="left", pad=dict(r=4),
            buttons=[
                dict(label="▶ OYNAT", method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                     fromcurrent=True, mode="immediate")]),
                dict(label="⏸ DURDUR", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                dict(label="⏩ 2×", method="animate",
                     args=[None, dict(frame=dict(duration=40, redraw=True),
                                     fromcurrent=True, mode="immediate")]),
                dict(label="⏩⏩ 5×", method="animate",
                     args=[None, dict(frame=dict(duration=15, redraw=True),
                                     fromcurrent=True, mode="immediate")]),
                dict(label="⏮ TCA'ya Git", method="animate",
                     args=[[str(tca_idx)],
                           dict(frame=dict(duration=0, redraw=True), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=slider_steps, active=0,
            currentvalue=dict(prefix="⏱  ", font=dict(family="Space Mono",size=9,color="#4a6880")),
            pad=dict(t=46, b=0), len=0.92, x=0.04,
            bgcolor="#0c1018", bordercolor="#1a2740", tickcolor="#1a2740",
            font=dict(color="#4a6880", size=7),
        )],
    )
    return fig, tca_idx, tca_dist, dists, anim_jd


# ================================================================================
#  ARAYÜZ
# ================================================================================
st.set_page_config(page_title="LEO Yaknnsama Analiz Sistemi", page_icon="S", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(STYLE, unsafe_allow_html=True)

st.markdown("""
<div style="padding:20px 0 8px 0; border-bottom:1px solid #1a2740; margin-bottom:20px;">
  <div style="font-family:'Space Mono',monospace; font-size:.68rem;
              color:#4a6880; letter-spacing:.2em; text-transform:uppercase; margin-bottom:4px;">
    LEO UYDULARININ ÇARPIŞMA ANALİZİ SİSTEMİ (starlink/iss/oneweb)
  </div>
  <h1 style="margin:0; padding:0; font-size:1.7rem;">
    Alcak Dünya Yörüngesinde<br>
    <span style="color:#00c8ff;">Yakınsama Analizi &amp; Carpışma Riski Simulasyonu</span>
  </h1>
  <div style="font-family:'Barlow Condensed',sans-serif; font-size:.95rem;
              color:#4a6880; margin-top:6px; letter-spacing:.05em;">
    Uzay Bilimleri ve Teknolojileri Bitirme Ödevi · Space-Track GP Veri Tabani · Skyfield SGP4 Propagatoru
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("### KONTROL PANELİ")

# ─── BÖLÜM 1: OTOMATİK TLE İNDİRME ─────────────────────────────────────────
st.sidebar.markdown("""<div style="font-family:'Space Mono',monospace;font-size:.65rem;
    letter-spacing:.15em;color:#00c8ff;text-transform:uppercase;
    border-bottom:1px solid #1a2740;padding-bottom:4px;margin-bottom:8px;">
    1 — OTOMATİK TLE İNDİR</div>""", unsafe_allow_html=True)
st.sidebar.markdown("**Space-Track Kimlik Doğrulama**")
user_email  = st.sidebar.text_input("E-posta", placeholder="user@domain.com")
user_pass   = st.sidebar.text_input("Sifre", placeholder="........", type="password")
st.sidebar.markdown("**Hedef Uydu Kümesi** *(sadece LEO filolarına odaklanılmıştır)*")
search_term = st.sidebar.selectbox("Küme seçin",
    ["STARLINK", "ISS", "ONEWEB"], label_visibility="collapsed")
if st.sidebar.button("CANLI TLE VERİSİ İNDİR"):
    if user_email and user_pass:
        with st.spinner("Space-Track veritabanina baglaniliyor..."):
            key = "ISS" if search_term == "ISS" else search_term
            data = fetch_live_tles(user_email, user_pass, key)
            if data:
                st.session_state["tle_data"] = data
                st.session_state["loaded_group"] = search_term
                is3 = not (data[0].startswith("1 ") or data[0].startswith("2 "))
                count = len(data) // 3 if is3 else len(data) // 2
                st.sidebar.success(f"{count} uydu yüklendi.")
    else:
        st.sidebar.warning("Kimlik bilgisi gerekli.")

st.sidebar.markdown("---")

# ─── BÖLÜM 2: MANUEL TLE GİRİŞİ ─────────────────────────────────────────────
st.sidebar.markdown("""<div style="font-family:'Space Mono',monospace;font-size:.65rem;
    letter-spacing:.15em;color:#00ff9d;text-transform:uppercase;
    border-bottom:1px solid #1a2740;padding-bottom:4px;margin-bottom:8px;">
    2 — KENDİ UYDUNU GİR (TLE)</div>""", unsafe_allow_html=True)
st.sidebar.markdown("<small style='color:#4a6880;'>3 satır TLE (isim + satır1 + satır2)</small>",
                    unsafe_allow_html=True)
manual_tle_text = st.sidebar.text_area(
    "Manuel TLE",
    height=110,
    placeholder="MY-SAT\n1 99999U ...\n2 99999  ...",
    label_visibility="collapsed",
    key="manual_tle_input",
)
if st.sidebar.button("MANUEL TLE YÜKLE"):
    lines = [l.strip() for l in manual_tle_text.strip().split("\n") if l.strip()]
    if len(lines) >= 3:
        try:
            my_sat = EarthSatellite(lines[1], lines[2], lines[0], ts)
            st.session_state["my_sat"] = my_sat
            st.sidebar.success(f"✓ {my_sat.name} yüklendi.")
        except Exception as e:
            st.sidebar.error(f"TLE hatası: {e}")
    elif len(lines) == 2:
        try:
            my_sat = EarthSatellite(lines[0], lines[1], "CUSTOM-SAT", ts)
            st.session_state["my_sat"] = my_sat
            st.sidebar.success("✓ CUSTOM-SAT yüklendi.")
        except Exception as e:
            st.sidebar.error(f"TLE hatası: {e}")
    else:
        st.sidebar.warning("En az 2 TLE satırı girin.")

if "my_sat" in st.session_state:
    ms = st.session_state["my_sat"]
    st.sidebar.markdown(f"""<div style="font-family:'Space Mono',monospace;font-size:.65rem;
        color:#00ff9d;padding:6px 10px;background:rgba(0,255,157,.05);
        border:1px solid rgba(0,255,157,.2);border-radius:2px;margin-top:4px;">
        ✓ AKTİF: {ms.name}</div>""", unsafe_allow_html=True)
    if st.sidebar.button("Kendi Uydumu Sil"):
        del st.session_state["my_sat"]
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""<div style="font-family:'Space Mono',monospace;font-size:.65rem;
    letter-spacing:.15em;color:#4a6880;text-transform:uppercase;
    border-bottom:1px solid #1a2740;padding-bottom:4px;margin-bottom:8px;">
    3 — ANALİZ PARAMETRELERİ</div>""", unsafe_allow_html=True)
window_hrs  = st.sidebar.slider("Analiz penceresi (saat)", 1, 48, 24)
sigma_km    = st.sidebar.select_slider("Konum belirsizligi σ (km)",
    options=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0], value=0.5)
sat_limit   = st.sidebar.slider("Maksimum uydu sayisi", 5, 30, 15)
hbr_km      = st.sidebar.select_slider("Hard-Body Radius HBR (km)",
    options=[0.005, 0.010, 0.020, 0.050, 0.100], value=0.020)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model:** Chan 1997 + Foster 1992  
**Propagator:** SGP4/SDP4  
**Filtre:** Apsis + Mesafe  
**Veri:** Space-Track GP  
**TCA Adimi:** 5 dk  
**HBR:** Secim ile
""")

# VERİ KONTROLÜ
if "tle_data" not in st.session_state:
    st.info("Sol panelden Space-Track bilgilerinizi girerek veri indirin.")
    st.markdown("""
    <div class="info-panel">
      <b>Nasıl kullanılır?</b><br>
      1. <b>space-track.org</b> adresinden ücretsiz hesap oluşturun.<br>
      2. E-posta ve şifrenizi sol panele girin.<br>
      3. Uydu kümesini seçip <b>CANLI TLE VERİSİ İNDİR</b> butonuna tıklayın.<br>
      4. Tüm sekmeler aktif hale gelir.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

sats = parse_tles(st.session_state["tle_data"], limit=sat_limit)
if not sats:
    st.error("TLE ayristirma basarisiz.")
    st.stop()

# SEKMELERs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "GÖSTERGE PANELİ",
    "YAKINSAMA ANALİZİ",
    "KENDİ UYDUN",
    "CANLI SİMÜLASYON",
    "3B YÖRÜNGE & ZEMİN İZİ",
    "YÖRÜNGE ELEMANLARI",
    "METODOLOJİ",
])

# ── TAB 1: GÖSTERGE PANELİ ───────────────────────────────────────────────────
with tab1:
    with st.spinner("Apsis filtresi + yakinsama analizi..."):
        df, n_filtered, n_total = compute_conjunctions(sats, window_hrs, sigma_km)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(f"""<div style="font-family:'Space Mono',monospace; font-size:.65rem;
         color:#4a6880; text-align:right; margin-bottom:14px;">Son güncelleme: {now_str}</div>""",
        unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("İzlenen Uydu", len(sats))
    with c2:
        st.metric("Toplam Çift", n_total)
    with c3:
        st.metric("Apsis Filtresi Geçen", n_total - n_filtered)
    with c4:
        n_conj = len(df) if not df.empty else 0
        st.metric("Yakinsama Olayi (<500km)", n_conj)
    with c5:
        n_crit = len(df[df["Risk Seviyesi"] == "KRİTİK"]) if not df.empty else 0
        st.metric("Kritik Risk", n_crit)

    if n_filtered > 0:
        st.markdown(f"""<div class="info-panel">
        <b>Apsis Filtresi:</b> {n_filtered} çift yükseklik bantları örtüşmediği için
        yörünge ilerletmesine gerek kalmadan elendi — hesaplama süresi
        %{round(n_filtered/n_total*100,1)} oranında azaltıldı.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if df.empty:
        st.success(f"{window_hrs} saatlik pencerede 500 km altı yakınsama tespit edilmedi.")
    else:
        # Dilüsyon uyarısı
        n_dil = df["Dilüsyon"].sum() if not df.empty else 0
        if n_dil > 0:
            st.markdown(f"""<div class="warn-panel">
            <b>OLASILIK SEYRELMESİ UYARISI:</b> {int(n_dil)} olayda geniş kovaryans
            Pc değerini maskeliyor olabilir. Yakınsama Analizi sekmesinden Max-Pc
            değerlerini kontrol edin.
            </div>""", unsafe_allow_html=True)

        show_cols = ["TCA (UTC)", "Obje A", "Obje B", "Mesafe (km)",
                     "Görecel Hız (km/s)", "Pc (bilimsel)", "Pc Max",
                     "Mahalanobis Md", "Ec (J/g)", "Risk Seviyesi"]
        RISK_COLORS = {"KRİTİK":"#ff2b4d","YÜKSEK":"#ff6b00","ORTA":"#ffaa00","DÜŞÜK":"#00ff9d"}
        MONO = "font-family:'Space Mono',monospace; font-size:0.76rem;"

        df_show = df[show_cols].copy()
        styled = (
            df_show.style
            .map(lambda v: f"color:{RISK_COLORS.get(str(v),'#b8cfe0')};font-weight:bold;{MONO}",
                 subset=["Risk Seviyesi"])
            .map(lambda v: f"color:#00c8ff;{MONO}", subset=["Pc (bilimsel)"])
            .map(lambda v: f"color:#ff9060;{MONO}", subset=["Pc Max"])
            .map(lambda v: (f"color:#ff2b4d;{MONO}" if float(v) < 1.5 else f"color:#b8cfe0;{MONO}"),
                 subset=["Mahalanobis Md"])
            .map(lambda v: (f"color:#ff2b4d;{MONO}" if float(v) >= 40 else f"color:#b8cfe0;{MONO}"),
                 subset=["Ec (J/g)"])
            .format({"Mesafe (km)":"{:.3f}", "Görecel Hız (km/s)":"{:.3f}",
                     "Pc Max":"{:.3e}", "Mahalanobis Md":"{:.2f}", "Ec (J/g)":"{:.1f}"})
            .set_properties(**{"font-family":"Space Mono,monospace","font-size":"0.76rem"})
        )
        csv_bytes = df_show.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Raporu CSV Olarak Indir", data=csv_bytes,
            file_name=f"yakinsama_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
        st.dataframe(styled, use_container_width=True)

# ── TAB 2: YAKINSAMA ANALİZİ ─────────────────────────────────────────────────
with tab2:
    if df is None or df.empty:
        st.success("Seçili pencerede kritik yakınsama olayı yok.")
    else:
        st.markdown("**Detaylı İnceleme — Olay Seçin**")
        options = [f"{r['Obje A']}  <->  {r['Obje B']}  |  TCA {r['TCA (UTC)']}  |  {r['Mesafe (km)']} km"
                   for _, r in df.iterrows()]
        sel = st.selectbox("Yakinsama olayi", options, label_visibility="collapsed")
        idx = options.index(sel)
        row = df.iloc[idx]

        # Dilüsyon uyarısı
        if row["Dilüsyon"]:
            st.markdown(f"""<div class="crit-panel">
            <b>OLASILIK SEYRELMESİ:</b> {row["Dilüsyon Mesajı"]}
            </div>""", unsafe_allow_html=True)

        # Grafik + gauge
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.plotly_chart(fig_distance_profile(
                row["_dist_arr"], window_hrs, row["Mesafe (km)"], sigma_km),
                use_container_width=True)
        with col_r:
            st.plotly_chart(fig_risk_gauge(row["Pc (izotropik)"]), use_container_width=True)

        # Pc karşılaştırması
        st.markdown("**Çarpışma Olasılığı Model Karşılaştırması**")
        pc_cols = st.columns(3)
        with pc_cols[0]:
            st.metric("Chan 1997 (İzotropik)", f"{row['Pc (izotropik)']:.3e}")
        with pc_cols[1]:
            st.metric("Foster 1992 (2D-Pc)", f"{row['Pc (Foster 2D)']:.3e}")
        with pc_cols[2]:
            st.metric("Max Pc (En Kötü Senaryo)", f"{row['Pc Max']:.3e}")

        # Mahalanobis testi
        mah_color = "#ff2b4d" if row["2D-Pc Geçerli"] != "2D-Pc Geçerli" else "#00ff9d"
        st.markdown(f"""<div class="info-panel">
        <b>Mahalanobis Mesafesi Testi:</b> Md = {row['Mahalanobis Md']:.3f} — 
        <span style="color:{mah_color};">{row['2D-Pc Geçerli']}</span><br>
        <small>Md < 1.5 → doğrusal hareket varsayımı çöküyor → 3D-Pc gerekli (CARA metodolojisi)</small>
        </div>""", unsafe_allow_html=True)

        # Parçalanma analizi
        frag = fragmentation_probability(row["Görecel Hız (km/s)"])
        st.markdown("**Çarpışma Sonucu Analizi (Collision Consequence)**")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.metric("Özgül Kinetik Enerji (J/g)", f"{frag['E_c_J_per_g']:.1f}")
        with fc2:
            st.metric("Parçalanma Seviyesi", frag["level"])
        with fc3:
            st.metric("Tahmini Enkaz Nesnesi", frag["est_debris"])
        st.markdown(f"""<div class="info-panel" style="border-left-color:{frag['color']};">
        <b>{frag['level']}:</b> {frag['desc']}<br>
        <small>Ec ≥ 40 J/g → Katastrofik parçalanma (Kessler Sendromu katkısı)</small>
        </div>""", unsafe_allow_html=True)

        # Tam parametre tablosu
        st.markdown("**Tam Olay Parametreleri**")
        det = {
            "Obje A": row["Obje A"], "Obje B": row["Obje B"],
            "TCA (UTC)": row["TCA (UTC)"],
            "Iskala Mesafesi (km)": row["Mesafe (km)"],
            "Görecel Hız (km/s)": row["Görecel Hız (km/s)"],
            "Konum Belirsizligi sigma (km)": sigma_km,
            "Hard-Body Radius HBR (km)": hbr_km,
            "Pc — Chan 1997 Izotropik": f"{row['Pc (izotropik)']:.3e}",
            "Pc — Foster 1992 2D": f"{row['Pc (Foster 2D)']:.3e}",
            "Pc — Maksimum (En Kötü Senaryo)": f"{row['Pc Max']:.3e}",
            "Mahalanobis Mesafesi Md": row["Mahalanobis Md"],
            "2D-Pc Gecerliligi": row["2D-Pc Geçerli"],
            "Olasılık Seyrelmesi": "EVET" if row["Dilüsyon"] else "HAYIR",
            "Özgül Kinetik Enerji (J/g)": row["Ec (J/g)"],
            "Parçalanma Seviyesi": row["Parçalanma Seviyesi"],
            "Tahmini Enkaz Nesnesi": row["Tahmini Enkaz"],
            "Risk Seviyesi (NASA STD-8719.14)": row["Risk Seviyesi"],
        }
        df_det = pd.DataFrame(det.items(), columns=["Parametre", "Değer"])
        st.dataframe(df_det, use_container_width=True, hide_index=True)

# ── TAB 3: KENDİ UYDUN ───────────────────────────────────────────────────────
with tab3:
    st.markdown("## Kendi Uydunu Analiz Et")
    if "my_sat" not in st.session_state:
        st.markdown("""<div class="warn-panel">
        <b>Henüz kendi uydunuzu yüklemediniz.</b><br>
        Sol paneldeki <b>2 — KENDİ UYDUNU GİR (TLE)</b> bölümünden TLE verilerinizi girin
        ve <b>MANUEL TLE YÜKLE</b> butonuna tıklayın.
        </div>""", unsafe_allow_html=True)
    elif "tle_data" not in st.session_state:
        st.markdown("""<div class="warn-panel">
        <b>Filo verisi yüklenmemiş.</b><br>
        Sol panelden önce otomatik TLE indirme işlemini yapın; ardından kendi uydunuzla
        karşılaştırma yapılabilir.
        </div>""", unsafe_allow_html=True)
    else:
        my_sat = st.session_state["my_sat"]
        st.markdown(f"""<div class="info-panel">
        <b>Aktif uydu:</b> {my_sat.name}&nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Filo:</b> {st.session_state.get('loaded_group','—')}&nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Analiz penceresi:</b> {window_hrs} saat&nbsp;&nbsp;|&nbsp;&nbsp;
        <b>σ:</b> {sigma_km} km
        </div>""", unsafe_allow_html=True)

        with st.spinner(f"{my_sat.name} için yakınsama analizi çalışıyor..."):
            df_my = compute_conjunctions_custom(my_sat, sats, window_hrs, sigma_km)

        if df_my.empty:
            st.success(f"{window_hrs} saatlik pencerede {my_sat.name} için 500 km altı yakınsama yok.")
        else:
            n_crit_my = len(df_my[df_my["Risk Seviyesi"] == "KRİTİK"])
            n_high_my = len(df_my[df_my["Risk Seviyesi"] == "YÜKSEK"])

            c1m, c2m, c3m, c4m = st.columns(4)
            with c1m: st.metric("Toplam Yakınsama", len(df_my))
            with c2m: st.metric("Kritik Risk", n_crit_my)
            with c3m: st.metric("Yüksek Risk", n_high_my)
            with c4m: st.metric("Min. Mesafe (km)", f"{df_my['Mesafe (km)'].min():.2f}")

            st.markdown("**Yakınsamalar — Risk Tablosu**")
            RISK_COLORS = {"KRİTİK":"#ff2b4d","YÜKSEK":"#ff6b00","ORTA":"#ffaa00","DÜŞÜK":"#00ff9d"}
            MONO = "font-family:'Space Mono',monospace; font-size:0.76rem;"
            show_c = ["TCA (UTC)","Obje A","Obje B","Mesafe (km)","Görecel Hız (km/s)",
                      "Pc (bilimsel)","Pc Max","Mahalanobis Md","Ec (J/g)","Risk Seviyesi"]
            df_my_show = df_my[show_c].copy()
            styled_my = (
                df_my_show.style
                .map(lambda v: f"color:{RISK_COLORS.get(str(v),'#b8cfe0')};font-weight:bold;{MONO}",
                     subset=["Risk Seviyesi"])
                .map(lambda v: f"color:#00c8ff;{MONO}", subset=["Pc (bilimsel)"])
                .map(lambda v: f"color:#ff9060;{MONO}", subset=["Pc Max"])
                .map(lambda v: (f"color:#ff2b4d;{MONO}" if float(v) < 1.5 else f"color:#b8cfe0;{MONO}"),
                     subset=["Mahalanobis Md"])
                .format({"Mesafe (km)":"{:.3f}","Görecel Hız (km/s)":"{:.3f}",
                         "Pc Max":"{:.3e}","Mahalanobis Md":"{:.2f}","Ec (J/g)":"{:.1f}"})
                .set_properties(**{"font-family":"Space Mono,monospace","font-size":"0.76rem"})
            )
            csv_my = df_my_show.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Raporu CSV Olarak İndir", data=csv_my,
                file_name=f"kendi_uydu_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")
            st.dataframe(styled_my, use_container_width=True)

            # Seçili çift için detaylı analiz
            st.markdown("---")
            st.markdown("**Detaylı Çift Analizi — Olay Seçin**")
            opts_my = [f"{r['Obje B']}  |  TCA {r['TCA (UTC)']}  |  {r['Mesafe (km)']} km"
                       for _, r in df_my.iterrows()]
            sel_my  = st.selectbox("Olay seç", opts_my, label_visibility="collapsed", key="my_sel")
            idx_my  = opts_my.index(sel_my)
            row_my  = df_my.iloc[idx_my]

            if row_my["Dilüsyon"]:
                st.markdown(f"""<div class="crit-panel">
                <b>OLASILIK SEYRELMESİ:</b> {row_my["Dilüsyon Mesajı"]}</div>""",
                unsafe_allow_html=True)

            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.plotly_chart(
                    fig_distance_profile(row_my["_dist_arr"], window_hrs, row_my["Mesafe (km)"], sigma_km),
                    use_container_width=True)
            with col_r:
                st.plotly_chart(fig_risk_gauge(row_my["Pc (izotropik)"]), use_container_width=True)

            pc_c = st.columns(3)
            with pc_c[0]: st.metric("Chan 1997 (İzotropik)", f"{row_my['Pc (izotropik)']:.3e}")
            with pc_c[1]: st.metric("Foster 1992 (2D-Pc)",   f"{row_my['Pc (Foster 2D)']:.3e}")
            with pc_c[2]: st.metric("Max Pc",                 f"{row_my['Pc Max']:.3e}")

            mah_c = "#ff2b4d" if row_my["2D-Pc Geçerli"] != "2D-Pc Geçerli" else "#00ff9d"
            st.markdown(f"""<div class="info-panel">
            <b>Mahalanobis Testi:</b> Md = {row_my['Mahalanobis Md']:.3f} —
            <span style="color:{mah_c};">{row_my['2D-Pc Geçerli']}</span>
            </div>""", unsafe_allow_html=True)

            frag_my = fragmentation_probability(row_my["Görecel Hız (km/s)"])
            fc = st.columns(3)
            with fc[0]: st.metric("Ec (J/g)", f"{frag_my['E_c_J_per_g']:.1f}")
            with fc[1]: st.metric("Parçalanma", frag_my["level"])
            with fc[2]: st.metric("Tahmini Enkaz", frag_my["est_debris"])

            # Simülasyona gönder butonu
            st.markdown("---")
            if st.button("🔭 Bu Çifti Canlı Simülasyonda Göster", key="my_to_sim"):
                st.session_state["sim_sat_a"] = row_my["_s1"]
                st.session_state["sim_sat_b"] = row_my["_s2"]
                st.success("Çift 'CANLI SİMÜLASYON' sekmesine aktarıldı.")


# ── TAB 4: CANLI SİMÜLASYON ──────────────────────────────────────────────────
with tab4:
    st.markdown("## Canlı 3D Yörünge Simülasyonu")
    st.markdown("""<div class="info-panel">
    İki uydu arasındaki karşılaşmayı <b>gerçek zamanlı</b> animasyonla izleyin.
    Oynat / Durdur / Hız kontrolü ve <b>TCA'ya Git</b> butonu ile risk anını odaklayın.
    </div>""", unsafe_allow_html=True)

    # Uydu seçim kaynağı
    if "sim_sat_a" in st.session_state and "sim_sat_b" in st.session_state:
        default_a = st.session_state["sim_sat_a"].name
        default_b = st.session_state["sim_sat_b"].name
        st.markdown(f"""<div class="info-panel">
        <b>Seçili çift:</b> {default_a} × {default_b}<br>
        <small>Değiştirmek için aşağıdaki açılır menüleri kullanın.</small>
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
        sel_a = st.selectbox("Uydu A", sat_names_ext,
                             index=sat_names_ext.index(default_a) if default_a in sat_names_ext else 0,
                             key="sim_a")
    with sc2:
        sel_b = st.selectbox("Uydu B", sat_names_ext,
                             index=sat_names_ext.index(default_b) if default_b in sat_names_ext else min(1,len(sat_names_ext)-1),
                             key="sim_b")
    with sc3:
        sim_hrs = st.slider("Pencere (saat)", 1, 12, 6, key="sim_hrs")

    if sel_a == sel_b:
        st.warning("Farklı iki uydu seçin.")
    else:
        sat_obj_a = next(s for s in all_sats_ext if s.name == sel_a)
        sat_obj_b = next(s for s in all_sats_ext if s.name == sel_b)

        if st.button("▶ SİMÜLASYONU BAŞLAT", key="start_sim"):
            st.session_state["sim_sat_a"] = sat_obj_a
            st.session_state["sim_sat_b"] = sat_obj_b
            st.session_state["run_sim"]   = True

        if st.session_state.get("run_sim") and \
           "sim_sat_a" in st.session_state and "sim_sat_b" in st.session_state:
            sa = st.session_state["sim_sat_a"]
            sb = st.session_state["sim_sat_b"]
            with st.spinner("Yörüngeler hesaplanıyor ve animasyon oluşturuluyor..."):
                anim_fig, tca_i, tca_d, dists_arr, jd_arr = \
                    fig_animated_conjunction(sa, sb, sim_hrs)

            # TCA bilgisi
            tca_utc = ts.tt_jd(jd_arr[tca_i]).utc_strftime("%Y-%m-%d %H:%M:%S UTC")
            sev_sim, col_sim = risk_level(
                collision_probability_isotropic(tca_d, sigma_km))
            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1: st.metric("TCA Zamanı (UTC)", tca_utc)
            with tc2: st.metric("Min. Mesafe (km)", f"{tca_d:.3f}")
            with tc3: st.metric("TCA T+ (dk)", tca_i * 2)
            with tc4: st.metric("Risk", sev_sim)

            # 3D animasyon
            st.plotly_chart(anim_fig, use_container_width=True)

            # Mesafe profili (statik)
            st.markdown("**Mesafe Profili (Tam Pencere)**")
            step_m_sim = 2
            t_ax = np.arange(len(dists_arr)) * step_m_sim / 60.0
            fig_dp_sim = go.Figure()
            fig_dp_sim.add_hline(y=0.02, line=dict(color="#ff2b4d", dash="dot", width=1),
                                 annotation_text="HBR (20 m)")
            fig_dp_sim.add_trace(go.Scatter(x=t_ax, y=dists_arr, mode="lines",
                line=dict(color="#00c8ff", width=1.5), fill="tozeroy",
                fillcolor="rgba(0,200,255,.04)", name="Mesafe (km)"))
            fig_dp_sim.add_trace(go.Scatter(
                x=[t_ax[tca_i]], y=[dists_arr[tca_i]],
                mode="markers+text",
                marker=dict(color="#ff2b4d", size=10),
                text=[f" TCA {dists_arr[tca_i]:.1f} km"],
                textfont=dict(size=9, color="#ff2b4d", family="Space Mono"),
                name="TCA"))
            fig_dp_sim.update_layout(
                **DARK, height=240,
                xaxis=dict(title="Zaman (saat)", gridcolor="#1a2740", zeroline=False),
                yaxis=dict(title="Mesafe (km)", gridcolor="#1a2740", zeroline=False),
                title=dict(text=f"Mesafe Profili — {sa.name} × {sb.name}",
                    font=dict(size=11, family="Barlow Condensed", color="#00c8ff"), x=0.01),
                margin=dict(l=10, r=10, t=35, b=10),
                legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_dp_sim, use_container_width=True)


# ── TAB 5: 3B YÖRÜNGE & ZEMİN İZİ ───────────────────────────────────────────
with tab5:
    c1_3d, c2_3d = st.columns([3, 2])
    with c1_3d:
        st.markdown("**3 Boyutlu Yörünge Görünümü**")
        with st.spinner("Dünya dokusu yükleniyor..."):
            st.plotly_chart(fig_3d_orbits(sats), use_container_width=True, height=560)
    with c2_3d:
        st.markdown("**Zemin İzi Haritası**")
        with st.spinner("Hesaplanıyor..."):
            st.plotly_chart(fig_ground_tracks(sats), use_container_width=True)
        st.markdown("""<div style="font-family:'Space Mono',monospace; font-size:.65rem;
             color:#2a4060; line-height:2; margin-top:8px;">
          Her uydu için yaklasik 95 dakikalık iz gosterilmektedir.<br>
          Buyuk noktalar anlık konumu temsil eder.<br>
          Zemin izi SGP4/SDP4 propagatörü ile hesaplanmistir.
        </div>""", unsafe_allow_html=True)

# ── TAB 6: YÖRÜNGE ELEMANLARI ────────────────────────────────────────────────
with tab6:
    st.markdown("## Yörünge Elemanları ve Uzay Dağılımı")
    elems_list = [(sat.name, get_orbital_elements(sat)) for sat in sats]

    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.markdown("**Kepler Yörünge Elemanları Tablosu**")
        rows = []
        for name, elems in elems_list:
            if elems:
                rows.append({
                    "Uydu": name[:18],
                    "İrtifa (km)": elems.get("Ortalama İrtifa (km)", "-"),
                    "Eğim (°)": elems.get("Eğim i (°)", "-"),
                    "Dışmerkezlik": elems.get("Dışmerkezlik e", "-"),
                    "Periyot (dk)": elems.get("Yörünge Periyodu (dk)", "-"),
                })
        if rows:
            df_elems = pd.DataFrame(rows)
            st.dataframe(df_elems, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**İrtifa / Eğim Dağılımı** (nokta boyutu = dışmerkezlik)")
        st.plotly_chart(fig_orbital_elements_radar(elems_list), use_container_width=True)

    with st.expander("Seçili Uydu Detayı"):
        sel_sat = st.selectbox("Uydu seç", [s.name for s in sats], key="elem_sel")
        sel_elems = next((e for n, e in elems_list if n == sel_sat), {})
        if sel_elems:
            df_single = pd.DataFrame(sel_elems.items(), columns=["Eleman", "Değer"])
            st.dataframe(df_single, use_container_width=True, hide_index=True)

# ── TAB 7: METODOLOJİ ────────────────────────────────────────────────────────
with tab7:
    st.markdown("## Metodoloji ve Teorik Altyapı")
    st.markdown("""
    <div class="info-panel">
    <b>1. Yörünge Propagasyonu — SGP4/SDP4 (Skyfield)</b><br>
    TLE (Two-Line Element) verilerinin konum vektörlerine dönüştürülmesinde NORAD standardı
    <b>Basitleştirilmiş Genel Pertürbasyon-4 (SGP4)</b> modeli kullanılmaktadır.
    SGP4, yerçekimi harmoniklerini, atmosferik sürüklemeyi ve Güneş/Ay üçüncü cisim etkilerini
    ortalanmış bir kuvvet modeliyle yaklaşık olarak ele alır. Alçak yörüngeli (<2000 km) nesneler
    için SGP4; yüksek yörüngeli nesneler için SDP4 otomatik olarak devreye girer.<br><br>
    <b>Performans notu (Tez Bölüm 1):</b> Saf Python/Skyfield saniyede ~1M adım üretirken,
    Rust/Zig tabanlı <b>Astrora</b> (SIMD ile) 4.8–15M, SatKit (PyO3/Rust) ~3.4M hıza ulaşır.
    Büyük ölçekli operasyonel simülasyonlar için bu kütüphanelere geçiş önerilir.
    </div>

    <div class="info-panel">
    <b>2. Apsis Filtresi — Bölüm 2.1 (ESA/NASA standardı)</b><br>
    Analiz başlamadan önce tüm uydu çiftleri <b>Apsis (Apoje-Perije) Filtresi</b>'nden geçirilir.
    Birinci nesnenin <i>perije irtifası q₁</i>, ikinci nesnenin <i>apoje irtifasından Q₂</i>'den yüksekse,
    bu iki yörünge uzayda hiçbir zaman kesişemez. Matematiksel koşul:<br>
    &nbsp;&nbsp;<code>max(q₁, q₂) > min(Q₁, Q₂) + D_th</code><br>
    Bu filtrenin uygulanması O(N²) hesaplama yükünü, irtifa bantları örtüşmeyen tüm
    çiftleri eleyerek dramatik biçimde azaltır.
    </div>

    <div class="info-panel">
    <b>3. TCA Tespiti — 5 Dakika Adımlı Kaba Tarama</b><br>
    Apsis filtresini geçen çiftler için <b>5 dakikalık sabit zaman adımlarıyla</b>
    analiz penceresi boyunca Öklid mesafesi hesaplanır. En küçük mesafenin elde edildiği
    an <b>TCA (Time of Closest Approach — En Yakın Geçiş Zamanı)</b> olarak belirlenir.
    Daha hassas TCA için Brent yöntemi ile yerel minimizasyon uygulanabilir.
    </div>

    <div class="info-panel">
    <b>4. Çarpışma Olasılığı — İki Model</b><br>
    <b>4a. Chan (1997) İzotropik Model:</b> Konum belirsizliğinin her yöne eşit (küresel) dağıldığını
    varsayan basitleştirilmiş model. Yüksek hızda sonuç verir ancak gerçek asimetrik kovaryansı
    yansıtmaz. Formül: normal CDF tabanlı kapalı-form yaklaşımı.<br><br>
    <b>4b. Foster &amp; Estes (1992) 2D-Pc:</b> NASA Uzay Mekiği döneminden bu yana kullanılan
    endüstri standardı. Çarpışma entegrasyonu, <b>encounter plane</b>'e iz düşürülerek
    iki boyuta indirgenir. Birleşik kovaryans matrisi (Σ = Cₐ + C_b) oluşturulur ve
    Gauss dağılımının HBR dairesi üzerindeki 2D integrali hesaplanır:<br>
    &nbsp;&nbsp;<code>Pc = 1/(2π√detΣ) ∬_HBR exp(-½ rᵀΣ⁻¹r) dx dy</code>
    </div>

    <div class="info-panel">
    <b>5. Mahalanobis Mesafesi Testi — Bölüm 3.2 (CARA Metodolojisi)</b><br>
    2D-Pc'nin <i>"kısa süreli karşılaşma"</i> ve <i>"doğrusal hareket"</i> varsayımları,
    nesnelerin düşük bağıl hızlarla yaklaştığı durumlarda çöker. CARA metodolojisine göre
    <b>Mahalanobis mesafesi</b> (Md = miss / σ) bu geçerliliği test eder:<br>
    &nbsp;&nbsp;Md &lt; 0.5 → 2D-Pc <span style="color:#ff2b4d;">GEÇERSİZ</span> — 3D-Pc / Monte Carlo zorunlu<br>
    &nbsp;&nbsp;Md &lt; 1.5 → 2D-Pc <span style="color:#ffaa00;">SINIRDA</span> — 3D-Pc tavsiye edilir<br>
    &nbsp;&nbsp;Md ≥ 1.5 → 2D-Pc <span style="color:#00ff9d;">GEÇERLİ</span>
    </div>

    <div class="info-panel">
    <b>6. Olasılık Seyrelmesi (Probability Dilution) — Bölüm 4</b><br>
    Büyük konum belirsizliği (geniş kovaryans) durumunda Gauss dağılımı uzayda o denli
    yayılır ki HBR dairesi içine düşen yoğunluk sıfıra yaklaşır — Pc matematiksel olarak
    küçülür. Bu <b>"sahte güven"</b> (false confidence) problemi, gerçekte tehlikeli bir
    yakınlaşmayı güvenli zannettirabilir.<br><br>
    Çözüm araçları: <b>WSPRT</b> (Wald Sıralı Olasılık Oranı Testi) — arka plan riskiyle
    anlık riski oranlar; <b>Max-Pc Analizi</b> — kovaryans büyüklüğü iteratif olarak
    değiştirilerek o geometri için matematiksel en yüksek Pc bulunur.
    </div>

    <div class="info-panel">
    <b>7. Risk Sınıflandırması — NASA STD-8719.14</b><br>
    &nbsp;&nbsp;• <span style="color:#ff2b4d;">Pc &gt; 1×10⁻³ → KRİTİK</span> — Çarpışmadan kaçınma manevrası (CAM) zorunlu<br>
    &nbsp;&nbsp;• <span style="color:#ff6b00;">Pc &gt; 1×10⁻⁴ → YÜKSEK</span> — CAM değerlendirmesi gerekli<br>
    &nbsp;&nbsp;• <span style="color:#ffaa00;">Pc &gt; 1×10⁻⁵ → ORTA</span> — Artan izleme frekansı<br>
    &nbsp;&nbsp;• <span style="color:#00ff9d;">Pc ≤ 1×10⁻⁵ → DÜŞÜK</span> — Rutin izleme yeterli
    </div>

    <div class="info-panel">
    <b>8. Çarpışma Sonucu — Parçalanma Olasılığı Pf (Bölüm 4)</b><br>
    Yalnızca çarpışma ihtimali değil, olası felaketin büyüklüğü de risk hesabına katılmalıdır.
    <b>Özgül Kinetik Enerji:</b> Ec = ½ · m_b · v_rel² / m_a (J/g)<br>
    &nbsp;&nbsp;Ec ≥ 40 J/g → Katastrofik parçalanma — <b>Kessler Sendromu</b> katkısı<br>
    &nbsp;&nbsp;Ec ≥ 10 J/g → Ciddi hasar ve önemli enkaz bulutu<br>
    &nbsp;&nbsp;Ec ≥ 1 J/g  → Kısmi hasar<br>
    Kessler Sendromu: Çarpışmaların yeni çarpışmaları tetiklediği zincirleme reaksiyon.
    </div>

    <div class="info-panel">
    <b>9. Veri Kaynağı — Space-Track GP Sınıfı</b><br>
    TLE verileri <b>18. Uzay Savunma Filosu (ABD Uzay Kuvvetleri)</b> tarafından işletilen
    space-track.org üzerinden <b>GP (Genel Pertürbasyon)</b> uç noktasından çekilmektedir.
    Eski TLE/2LE formatı yerine JSON/CSV tabanlı <b>OMM (Orbit Mean-Elements Message)</b>
    formatına geçiş önerilir. API oran sınırı aşımından kaçınmak için toplu NORAD ID sorguları
    ve rastgele zamanlanmış güncelleme döngüleri kullanılmalıdır.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Referanslar")
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
