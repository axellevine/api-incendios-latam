import calendar
from datetime import date, datetime
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import unicodedata
from urllib.parse import urlparse
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# 1. Configuración de página en modo ancho total
st.set_page_config(
    page_title="NASA Hot Spots · Mapa Térmico Hexagonal LATAM",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
WORLD_GEOJSON_CACHE = BASE_DIR / "world_countries_110m.geojson"
DEFAULT_API_URL = os.getenv("INCENDIOS_API_URL", "http://127.0.0.1:8000/api/incendios-grid")

EARTH_RADIUS_M = 6_371_008.8
METERS_PER_DEG_LAT = (np.pi / 180.0) * EARTH_RADIUS_M  # ~111,195 m/grado
SENSOR_1TO1_RADIUS_DEG = 0.00558

LATAM_LAT_MIN, LATAM_LAT_MAX = -56.5, 34.0
LATAM_LON_MIN, LATAM_LON_MAX = -120.0, -32.0

ZOOM_TIERS: Dict[int, Dict[str, Any]] = {
    1: {
        "short": "1 · Macro (~300 km)",
        "label": "Nivel 1 · Macro-Regional (~300 km)",
        "radius_deg": 1.35,
        "grid_radius": 1.35,
        "grid_coverage": "globe",
        "block_size": 12,
        "camera_zoom": 2.9,
    },
    2: {
        "short": "2 · Regional (~110 km)",
        "label": "Nivel 2 · Regional (~110 km)",
        "radius_deg": 0.50,
        "grid_radius": 0.50,
        "grid_coverage": "globe",
        "block_size": 16,
        "camera_zoom": 4.3,
    },
    3: {
        "short": "3 · Sub-Regional (~40 km)",
        "label": "Nivel 3 · Sub-Regional (~40 km)",
        "radius_deg": 0.18,
        "grid_radius": 0.18,
        "grid_coverage": "latam",
        "block_size": 20,
        "camera_zoom": 5.7,
    },
    4: {
        "short": "4 · Local (~13 km)",
        "label": "Nivel 4 · Local / Municipal (~13 km)",
        "radius_deg": 0.06,
        "grid_radius": 0.06,
        "grid_coverage": "axial_clusters",
        "block_size": 24,
        "camera_zoom": 7.3,
    },
    5: {
        "short": "5 · Sensor 1:1 (~1 km)",
        "label": "Nivel 5 · Huella Física Sensor 1:1 (~1 km)",
        "radius_deg": SENSOR_1TO1_RADIUS_DEG,
        "grid_radius": SENSOR_1TO1_RADIUS_DEG,
        "grid_coverage": "axial_clusters",
        "block_size": 24,
        "camera_zoom": 9.6,
    },
}

THERMAL_PALETTES: Dict[str, Dict[str, Any]] = {
    "Térmico Clásico (Verde → Ámbar → Naranja → Rojo)": {
        "stops": np.array([0.0, 0.35, 0.68, 1.0]),
        "r": [46, 241, 230, 255],
        "g": [204, 196, 126, 28],
        "b": [113, 15, 34, 28],
    },
    "Térmico Infrarrojo (Ámbar → Carmesí → Incandescente)": {
        "stops": np.array([0.0, 0.25, 0.52, 0.78, 1.0]),
        "r": [74, 158, 224, 249, 255],
        "g": [25, 42, 92, 160, 236],
        "b": [54, 43, 24, 34, 130],
    },
    "Térmico Espectral (Índigo → Cian → Dorado → Rojo)": {
        "stops": np.array([0.0, 0.28, 0.58, 0.80, 1.0]),
        "r": [29, 14, 245, 249, 220],
        "g": [78, 165, 195, 105, 34],
        "b": [216, 233, 35, 22, 34],
    },
    "Magma Satelital (Violeta → Coral → Ámbar Claro)": {
        "stops": np.array([0.0, 0.33, 0.66, 0.85, 1.0]),
        "r": [59, 140, 222, 254, 252],
        "g": [15, 41, 73, 159, 244],
        "b": [112, 129, 104, 109, 175],
    },
}

MESES_NOMBRES = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}

MONTH_ALIASES: Dict[str, int] = {
    "enero": 1, "ene": 1, "january": 1, "jan": 1,
    "febrero": 2, "feb": 2, "february": 2,
    "marzo": 3, "mar": 3, "march": 3,
    "abril": 4, "abr": 4, "april": 4, "apr": 4,
    "mayo": 5, "may": 5,
    "junio": 6, "jun": 6, "june": 6,
    "julio": 7, "jul": 7, "july": 7,
    "agosto": 8, "ago": 8, "august": 8, "aug": 8,
    "septiembre": 9, "setiembre": 9, "sep": 9, "sept": 9, "september": 9,
    "octubre": 10, "oct": 10, "october": 10,
    "noviembre": 11, "nov": 11, "november": 11,
    "diciembre": 12, "dic": 12, "december": 12, "dec": 12,
}

EXPECTED_COLUMNS = [
    "pais",
    "anio",
    "fecha",
    "lat_grid",
    "lon_grid",
    "total_incendios",
    "frp_promedio",
    "brillo_promedio",
    "instrumento",
    "satelite",
    "mes",
    "nivel_riesgo",
    "color",
]

EXPORT_COLUMNS = [
    "pais",
    "fecha",
    "anio",
    "mes",
    "lat_grid",
    "lon_grid",
    "total_incendios",
    "frp_promedio",
    "brillo_promedio",
    "nivel_riesgo",
    "satelite",
    "instrumento",
]

COLUMN_ALIASES: Dict[str, List[str]] = {
    "pais": ["pais", "country", "region", "pais_nombre"],
    "anio": ["anio", "year", "ano"],
    "fecha": ["fecha", "date", "acq_date", "fecha_deteccion"],
    "lat_grid": ["lat_grid", "latitude", "lat", "y"],
    "lon_grid": ["lon_grid", "longitude", "lon", "lng", "x"],
    "total_incendios": ["total_incendios", "count", "total", "fires", "focos", "cantidad"],
    "frp_promedio": ["frp_promedio", "frp", "frp_mean", "potencia_frp"],
    "brillo_promedio": ["brillo_promedio", "brightness", "brillo", "brightness_mean", "bright_ti4"],
    "instrumento": ["instrumento", "instrument", "sensor"],
    "satelite": ["satelite", "satellite", "sat"],
    "mes": ["mes", "month"],
    "nivel_riesgo": ["nivel_riesgo", "risk", "riesgo", "risk_level"],
    "color": ["color", "hex_color"],
}

# 2. Estado canónico persistente entre páginas
DEFAULT_STATE: Dict[str, Any] = {
    "active_page": "🗺️ Grid de Incendios",
    "dock_visible": True,
    "api_url": DEFAULT_API_URL,
    "api_token": "",
    "api_limit": 60000,
    "allow_local_fallback": True,
    "strict_latam": True,
    "active_pais": "Todos",
    "active_riesgo": None,
    "active_sat": None,
    "active_inst": None,
    "frp_min": None,
    "frp_max": None,
    "brillo_min": None,
    "brillo_max": None,
    "temporal_mode": "mes_completo",
    "cal_mes": 8,
    "cal_anio": 2025,
    "cal_dia": 15,
    "smart_search_query": "",
    "last_parsed_entities": [],
    "drawn_polygon_2d": None,
    "drawn_filter_enabled": True,
    "draft_polygon_2d": None,
    "draft_vertices_2d": [],
    "draft_cleared": False,
    "palette_name": list(THERMAL_PALETTES.keys())[0],
    "smooth_pct": 35,
    "zoom_tier_num": 1,
    "show_all_hex": False,
    "sync_camera_zoom": True,
    "terrain_depth_3d": False,
    "projection_mode": "Plano (2D)",
    "show_world_map_3d": True,
    "show_graticule_3d": True,
    "show_base_grid": False,
}

for _k, _v in DEFAULT_STATE.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

is_map_page = st.session_state.active_page == "🗺️ Grid de Incendios"

# 3. Estilos CSS dinámicos según vista activa + Ventana Servidor arriba a la derecha
page_layout_css = (
    """
    .block-container {
        padding: 0rem !important;
        max-width: 100% !important;
        height: 100vh !important;
        overflow: hidden !important;
    }
    """
    if is_map_page
    else """
    .stApp {
        background: radial-gradient(circle at top left, #111827 0%, #090d16 65%, #05080f 100%) !important;
        color: #f8fafc !important;
    }
    .block-container {
        padding: 84px 36px 48px 36px !important;
        max-width: 1260px !important;
        margin: 0 auto !important;
        overflow-y: auto !important;
    }
    """
)

st.markdown(
    f"""
    <style>
    #MainMenu, footer, header, [data-testid="stSidebar"], [data-testid="collapsedControl"] {{
        display: none !important;
    }}

    {page_layout_css}

    /* Dock Superior Izquierdo (Navegación + Controles del Mapa) */
    .st-key-dock_superior_izquierdo {{
        position: fixed !important;
        top: 16px !important;
        left: 16px !important;
        z-index: 9999 !important;
        width: 385px !important;
        max-height: calc(100vh - 32px) !important;
        overflow-y: auto !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
        padding-right: 4px !important;
    }}

    .st-key-dock_superior_izquierdo::-webkit-scrollbar {{
        width: 5px;
    }}
    .st-key-dock_superior_izquierdo::-webkit-scrollbar-thumb {{
        background: rgba(255, 255, 255, 0.25);
        border-radius: 4px;
    }}

    /* Ventana Flotante Independiente de Conexión al Servidor (Arriba a la Derecha) */
    .st-key-dock_servidor_derecho {{
        position: fixed !important;
        top: 16px !important;
        right: 16px !important;
        z-index: 10000 !important;
        width: 350px !important;
        max-height: calc(100vh - 32px) !important;
        overflow-y: auto !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
    }}

    .st-key-dock_superior_izquierdo div[data-testid="stExpander"],
    .st-key-dock_servidor_derecho div[data-testid="stExpander"] {{
        background: rgba(17, 21, 30, 0.92) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.48) !important;
    }}

    .st-key-dock_superior_izquierdo div[data-testid="stExpander"] summary,
    .st-key-dock_servidor_derecho div[data-testid="stExpander"] summary {{
        font-weight: 600 !important;
        font-size: 14px !important;
        color: #f8fafc !important;
    }}

    .st-key-dock_superior_izquierdo button[kind="secondary"],
    .st-key-dock_servidor_derecho button[kind="secondary"],
    .st-key-dock_superior_izquierdo div[data-testid="stDownloadButton"] button {{
        background: rgba(17, 21, 30, 0.92) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
    }}

    .st-key-dock_superior_izquierdo div[data-testid="stDownloadButton"] button:hover {{
        border-color: #38bdf8 !important;
        background: rgba(30, 41, 59, 0.95) !important;
    }}

    /* Botón compacto de Periodo/Año con Calendario dentro de Filtro Activo */
    .st-key-btn_quick_date_modal button {{
        background: rgba(56, 189, 248, 0.12) !important;
        border: 1px solid rgba(56, 189, 248, 0.42) !important;
        color: #7dd3fc !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        min-height: 30px !important;
        height: 30px !important;
        padding: 0 8px !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }}

    .st-key-btn_quick_date_modal button:hover {{
        background: rgba(56, 189, 248, 0.24) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
    }}

    /* Tarjetas de la Página de Filtros y Exportación */
    .st-key-panel_buscador_inteligente,
    .st-key-panel_filtros_manuales,
    .st-key-panel_exportacion_filtros,
    .st-key-panel_preview_filtros {{
        background: rgba(17, 24, 39, 0.88) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 16px !important;
        padding: 20px 22px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35) !important;
        margin-bottom: 16px !important;
    }}

    /* Tarjeta oscura estilo calendario mensual (L M X J V S D) */
    .st-key-calendario_card,
    .st-key-calendario_dialog_card {{
        background: #181717 !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 14px !important;
        padding: 12px 10px 10px 10px !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
    }}

    .st-key-calendario_card div[data-testid="stHorizontalBlock"],
    .st-key-calendario_dialog_card div[data-testid="stHorizontalBlock"] {{
        gap: 3px !important;
        margin-bottom: 2px !important;
    }}

    .st-key-calendario_card div[data-testid="column"],
    .st-key-calendario_dialog_card div[data-testid="column"] {{
        min-width: 0 !important;
        padding: 0 !important;
    }}

    .st-key-calendario_card button,
    .st-key-calendario_dialog_card button {{
        width: 100% !important;
        min-height: 31px !important;
        height: 31px !important;
        padding: 0 !important;
        font-size: 13px !important;
        border-radius: 9px !important;
        line-height: 1 !important;
        box-shadow: none !important;
    }}

    .st-key-calendario_card button[kind="secondary"],
    .st-key-calendario_dialog_card button[kind="secondary"] {{
        background: transparent !important;
        border: none !important;
        color: #f8fafc !important;
        font-weight: 500 !important;
    }}

    .st-key-calendario_card button[kind="secondary"]:hover,
    .st-key-calendario_dialog_card button[kind="secondary"]:hover {{
        background: rgba(255, 255, 255, 0.11) !important;
    }}

    .st-key-calendario_card button:disabled,
    .st-key-calendario_dialog_card button:disabled {{
        background: transparent !important;
        border: none !important;
        color: #5c5e63 !important;
        opacity: 1 !important;
        font-weight: 400 !important;
    }}

    .st-key-calendario_card button[kind="primary"],
    .st-key-calendario_dialog_card button[kind="primary"] {{
        background: #e04856 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: 2px solid #181717 !important;
        box-shadow: 0 0 0 2px #e04856 !important;
        border-radius: 10px !important;
    }}

    /* Pantalla completa exclusiva para el mapa principal (sin capturar elementos dentro de stDialog) */
    section.main .stDeckGlJsonChart,
    section.main div[data-testid="stDeckGlJsonChart"] {{
        height: 100vh !important;
        width: 100vw !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        z-index: 1 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# 4. Carga del Mapamundi GeoJSON y Geometría Base para el Globo 3D / Contornos 2D
@st.cache_data
def load_world_map_geojson() -> Optional[Dict[str, Any]]:
    if WORLD_GEOJSON_CACHE.exists():
        try:
            with open(WORLD_GEOJSON_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    urls = [
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson",
        "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=12)
            resp.raise_for_status()
            geojson_data = resp.json()
            slim_features = [
                {"type": "Feature", "geometry": feat.get("geometry"), "properties": {}}
                for feat in geojson_data.get("features", [])
                if feat.get("geometry")
            ]
            slim_fc = {"type": "FeatureCollection", "features": slim_features}
            with open(WORLD_GEOJSON_CACHE, "w", encoding="utf-8") as f:
                json.dump(slim_fc, f)
            return slim_fc
        except Exception:
            continue
    return None


@st.cache_data
def build_globe_sphere_and_graticule() -> tuple[pd.DataFrame, pd.DataFrame]:
    lons = np.linspace(-180.0, 180.0, 73)
    lats = np.linspace(-89.5, 89.5, 37)
    ocean_quads = []
    for i in range(len(lons) - 1):
        for j in range(len(lats) - 1):
            ocean_quads.append(
                {
                    "polygon": [
                        [float(lons[i]), float(lats[j]), 0.0],
                        [float(lons[i + 1]), float(lats[j]), 0.0],
                        [float(lons[i + 1]), float(lats[j + 1]), 0.0],
                        [float(lons[i]), float(lats[j + 1]), 0.0],
                    ]
                }
            )

    graticule_paths = []
    for lon in range(-180, 180, 30):
        pts = [[float(lon), float(la), 300.0] for la in np.linspace(-89.5, 89.5, 90)]
        is_prime = lon == 0
        graticule_paths.append(
            {
                "path": pts,
                "color": [56, 189, 248, 85] if is_prime else [148, 163, 184, 32],
            }
        )
    for lat in range(-75, 90, 15):
        pts = [[float(lo), float(lat), 300.0] for lo in np.linspace(-180.0, 180.0, 120)]
        is_equator = lat == 0
        graticule_paths.append(
            {
                "path": pts,
                "color": [56, 189, 248, 85] if is_equator else [148, 163, 184, 32],
            }
        )

    return pd.DataFrame(ocean_quads), pd.DataFrame(graticule_paths)


# 5. Cliente de datos robusto para Servidores Backend Externos (/api/incendios-grid y /api/filtros-disponibles)
def _derive_base_api_url(grid_url: str) -> str:
    parsed = urlparse(grid_url.strip())
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return "http://127.0.0.1:8000"


def _build_api_headers(api_token: str) -> Dict[str, str]:
    headers = {"Cache-Control": "no-cache", "Accept": "application/json"}
    clean_tok = api_token.strip()
    if clean_tok:
        headers["Authorization"] = clean_tok if clean_tok.lower().startswith("bearer ") else f"Bearer {clean_tok}"
        headers["X-API-Key"] = clean_tok.replace("Bearer ", "").replace("bearer ", "").strip()
    return headers


@st.cache_data(ttl=20)
def fetch_api_catalogs(grid_url: str, api_token: str = "") -> Dict[str, Any]:
    base_url = _derive_base_api_url(grid_url)
    try:
        resp = requests.get(
            f"{base_url}/api/filtros-disponibles",
            headers=_build_api_headers(api_token),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _extract_records_from_payload(payload: Any) -> List[Dict[str, Any]]:
    """Extrae registros desde cualquier estructura JSON común en backends externos (Lista, Dict o GeoJSON)."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]

    if isinstance(payload, dict):
        if payload.get("type") == "FeatureCollection" and isinstance(payload.get("features"), list):
            extracted: List[Dict[str, Any]] = []
            for feat in payload["features"]:
                if not isinstance(feat, dict):
                    continue
                row = dict(feat.get("properties") or {})
                geom = feat.get("geometry") or {}
                coords = geom.get("coordinates")
                if geom.get("type") == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    row.setdefault("lon_grid", coords[0])
                    row.setdefault("lat_grid", coords[1])
                extracted.append(row)
            return extracted

        for key in ("data", "records", "grid", "results", "items", "rows"):
            val = payload.get(key)
            if isinstance(val, list):
                return [r for r in val if isinstance(r, dict)]

    return []


def _normalize_grid_dataframe(records: List[Dict[str, Any]], strict_latam_bounds: bool = True) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = pd.DataFrame(records)

    lower_cols = {c.lower().strip(): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical not in df.columns:
            for alias in aliases:
                if alias in lower_cols:
                    df[canonical] = df[lower_cols[alias]]
                    break

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["pais"] = df["pais"].fillna("LATAM").astype(str).str.strip()
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").fillna(2025).astype(int)
    df["fecha"] = df["fecha"].fillna("").astype(str).str.strip()
    df["lat_grid"] = pd.to_numeric(df["lat_grid"], errors="coerce")
    df["lon_grid"] = pd.to_numeric(df["lon_grid"], errors="coerce")
    df["total_incendios"] = pd.to_numeric(df["total_incendios"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    df["frp_promedio"] = pd.to_numeric(df["frp_promedio"], errors="coerce").fillna(0.0).clip(lower=0.0).round(1)
    df["brillo_promedio"] = pd.to_numeric(df["brillo_promedio"], errors="coerce").fillna(0.0).clip(lower=0.0).round(1)
    df["instrumento"] = df["instrumento"].fillna("MODIS").astype(str).str.strip()
    df["satelite"] = df["satelite"].fillna("Terra").astype(str).str.strip()
    df["mes"] = pd.to_numeric(df["mes"], errors="coerce").fillna(1).astype(int).clip(1, 12)
    df["nivel_riesgo"] = df["nivel_riesgo"].fillna("Bajo").astype(str)
    df["color"] = df["color"].fillna("#2ECC71").astype(str)

    df = df.dropna(subset=["lat_grid", "lon_grid"])

    if strict_latam_bounds:
        df = df[
            df["lat_grid"].between(LATAM_LAT_MIN, LATAM_LAT_MAX)
            & df["lon_grid"].between(LATAM_LON_MIN, LATAM_LON_MAX)
        ]
    else:
        df = df[df["lat_grid"].between(-85.0, 85.0) & df["lon_grid"].between(-180.0, 180.0)]

    return df.reset_index(drop=True)


@st.cache_data(ttl=15)
def fetch_incendios_grid(
    api_url: str,
    allow_local_fallback: bool,
    pais: Optional[str] = None,
    anio: Optional[int] = None,
    mes: Optional[int] = None,
    fecha: Optional[str] = None,
    strict_latam_bounds: bool = True,
    limit: int = 60000,
    api_token: str = "",
) -> tuple[pd.DataFrame, bool, str]:
    params: Dict[str, Any] = {"limit": int(limit)}
    if pais and pais != "Todos":
        params["pais"] = pais
    if fecha:
        params["fecha"] = fecha
    else:
        if anio:
            params["anio"] = int(anio)
        if mes:
            params["mes"] = int(mes)

    try:
        resp = requests.get(
            api_url.strip(),
            params=params,
            timeout=30,
            headers=_build_api_headers(api_token),
        )
        resp.raise_for_status()
        payload = resp.json()
        records = _extract_records_from_payload(payload)

        df = _normalize_grid_dataframe(records, strict_latam_bounds=strict_latam_bounds)
        if fecha:
            suffix = f" · {fecha}"
        elif mes:
            suffix = f" · {MESES_NOMBRES[mes]} {anio}"
        else:
            suffix = f" · Año {anio}"
        return df, True, f"🟢 API EN VIVO ({len(df):,} celdas{suffix})"
    except Exception as api_err:
        if not allow_local_fallback:
            return pd.DataFrame(columns=EXPECTED_COLUMNS), False, f"🔴 API OFFLINE: {api_err}"

        local_csv = BASE_DIR / "modis_2025_Argentina.csv"
        if not local_csv.exists():
            local_csv = Path.home() / "Downloads" / "modis_2025_Argentina.csv"

        if local_csv.exists():
            raw = pd.read_csv(local_csv)
            parsed_dates = pd.to_datetime(raw["acq_date"], format="mixed", dayfirst=False, errors="coerce")
            raw["mes"] = parsed_dates.dt.month.fillna(1).astype(int)
            raw["anio"] = parsed_dates.dt.year.fillna(2025).astype(int)
            raw["fecha"] = parsed_dates.dt.strftime("%Y-%m-%d").fillna("2025-01-01")
            raw["lat_grid"] = raw["latitude"].round(2)
            raw["lon_grid"] = raw["longitude"].round(2)

            grouped = (
                raw.groupby(["anio", "lat_grid", "lon_grid", "instrument", "satellite", "mes"], as_index=False)
                .agg(
                    total_incendios=("brightness", "count"),
                    frp_promedio=("frp", "mean"),
                    brillo_promedio=("brightness", "mean"),
                    fecha=("fecha", "max"),
                )
                .rename(columns={"instrument": "instrumento", "satellite": "satelite"})
            )
            grouped["pais"] = "Argentina"
            grouped["frp_promedio"] = grouped["frp_promedio"].round(1)
            grouped["brillo_promedio"] = grouped["brillo_promedio"].round(1)

            cond_alto = (grouped["total_incendios"] >= 10) | (grouped["brillo_promedio"] >= 355.0)
            cond_medio = (grouped["total_incendios"] >= 4) | (grouped["brillo_promedio"] >= 330.0)

            grouped["nivel_riesgo"] = np.select([cond_alto, cond_medio], ["Alto", "Medio"], default="Bajo")
            grouped["color"] = np.select([cond_alto, cond_medio], ["#FF0000", "#F39C12"], default="#2ECC71")

            df = _normalize_grid_dataframe(
                grouped.to_dict(orient="records"), strict_latam_bounds=strict_latam_bounds
            )
            if pais and pais != "Todos":
                df = df[df["pais"].str.lower() == pais.lower()].reset_index(drop=True)
            if fecha:
                df = df[df["fecha"] == fecha].reset_index(drop=True)
            elif mes:
                df = df[df["mes"] == mes].reset_index(drop=True)
            return df, False, f"🟠 MODO LOCAL (Error API: {api_err})"

        return pd.DataFrame(columns=EXPECTED_COLUMNS), False, f"🔴 Sin conexión y sin CSV local ({api_err})"


# 6. Motor de Reconocimiento Inteligente de Búsqueda
def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text.lower().strip()) if unicodedata.category(ch) != "Mn"
    )


def _extract_numeric_range(
    norm_query: str,
    metric_pattern: str,
    unit_pattern: str,
) -> tuple[Optional[float], Optional[float], List[tuple[int, int]]]:
    spans: List[tuple[int, int]] = []
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    num = r"(\d+(?:\.\d+)?)"

    m_between = re.search(
        rf"\b{metric_pattern}\s*(?:entre|de|desde)?\s*{num}\s*(?:y|a|hasta|-)\s*{num}\s*(?:{unit_pattern})?\b",
        norm_query,
    )
    if m_between:
        v1, v2 = float(m_between.group(1)), float(m_between.group(2))
        min_val, max_val = min(v1, v2), max(v1, v2)
        spans.append(m_between.span())
        return min_val, max_val, spans

    m_gt = re.search(
        rf"\b{metric_pattern}\s*(?:>=|>|mayor(?:\s+o\s+igual)?(?:\s+a|\s+que)?|superior(?:\s+a)?|sobre|desde|minimo)\s*{num}\s*(?:{unit_pattern})?",
        norm_query,
    )
    if m_gt:
        min_val = float(m_gt.group(1))
        spans.append(m_gt.span())

    m_lt = re.search(
        rf"\b{metric_pattern}\s*(?:<=|<|menor(?:\s+o\s+igual)?(?:\s+a|\s+que)?|inferior(?:\s+a)?|bajo|hasta|maximo)\s*{num}\s*(?:{unit_pattern})?",
        norm_query,
    )
    if m_lt:
        max_val = float(m_lt.group(1))
        spans.append(m_lt.span())

    if min_val is None and max_val is None:
        m_unit = re.search(rf"\b{num}\s*{unit_pattern}\b", norm_query)
        if m_unit:
            min_val = float(m_unit.group(1))
            spans.append(m_unit.span())

    return min_val, max_val, spans


def parse_and_apply_smart_search(
    raw_query: str,
    available_paises: List[str],
    available_anios: List[int],
    available_sats: List[str],
    available_insts: List[str],
) -> List[Dict[str, str]]:
    cleaned = raw_query.strip()
    st.session_state.frp_min = None
    st.session_state.frp_max = None
    st.session_state.brillo_min = None
    st.session_state.brillo_max = None

    if not cleaned:
        st.session_state.last_parsed_entities = []
        return []

    norm_query = _strip_accents(cleaned)
    consumed_spans: List[tuple[int, int]] = []
    detected: List[Dict[str, str]] = []

    def _mark_span(start: int, end: int) -> None:
        consumed_spans.append((start, end))

    frp_min, frp_max, frp_spans = _extract_numeric_range(
        norm_query,
        metric_pattern=r"(?:frp|potencia(?:\s+radiativa)?)",
        unit_pattern=r"(?:mw|megavatios?)",
    )
    if frp_min is not None or frp_max is not None:
        st.session_state.frp_min = frp_min
        st.session_state.frp_max = frp_max
        for sp in frp_spans:
            _mark_span(*sp)
        if frp_min is not None and frp_max is not None:
            frp_label = f"{frp_min:g} – {frp_max:g} MW"
        elif frp_min is not None:
            frp_label = f"≥ {frp_min:g} MW"
        else:
            frp_label = f"≤ {frp_max:g} MW"
        detected.append({"tipo": "FRP", "valor": frp_label, "color": "#fb923c"})

    brillo_min, brillo_max, brillo_spans = _extract_numeric_range(
        norm_query,
        metric_pattern=r"(?:brillo|brightness|temperatura)",
        unit_pattern=r"(?:k|kelvin)",
    )
    if brillo_min is not None or brillo_max is not None:
        st.session_state.brillo_min = brillo_min
        st.session_state.brillo_max = brillo_max
        for sp in brillo_spans:
            _mark_span(*sp)
        if brillo_min is not None and brillo_max is not None:
            brillo_label = f"{brillo_min:g} – {brillo_max:g} K"
        elif brillo_min is not None:
            brillo_label = f"≥ {brillo_min:g} K"
        else:
            brillo_label = f"≤ {brillo_max:g} K"
        detected.append({"tipo": "Brillo", "valor": brillo_label, "color": "#f87171"})

    iso_match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", norm_query)
    dmy_match = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2})\b", norm_query)
    exact_date_parsed: Optional[date] = None

    if iso_match:
        y_val, m_val, d_val = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
        try:
            exact_date_parsed = date(y_val, m_val, d_val)
            _mark_span(*iso_match.span())
        except ValueError:
            pass
    elif dmy_match:
        d_val, m_val, y_val = int(dmy_match.group(1)), int(dmy_match.group(2)), int(dmy_match.group(3))
        try:
            exact_date_parsed = date(y_val, m_val, d_val)
            _mark_span(*dmy_match.span())
        except ValueError:
            pass

    country_lookup: Dict[str, str] = {}
    for p in available_paises:
        if p != "Todos":
            country_lookup[_strip_accents(p)] = p
    extra_country_aliases = {
        "brazil": "Brasil",
        "peru": "Perú",
        "mexico": "México",
        "republica dominicana": "República Dominicana",
        "todos": "Todos",
        "toda latam": "Todos",
        "latam": "Todos",
    }
    for alias_k, canonical_p in extra_country_aliases.items():
        if canonical_p == "Todos":
            country_lookup[alias_k] = "Todos"
        else:
            for p in available_paises:
                if _strip_accents(p) == _strip_accents(canonical_p):
                    country_lookup[alias_k] = p

    for norm_c in sorted(country_lookup.keys(), key=len, reverse=True):
        m = re.search(rf"\b{re.escape(norm_c)}\b", norm_query)
        if m:
            matched_country = country_lookup[norm_c]
            st.session_state.active_pais = matched_country
            detected.append({"tipo": "País", "valor": matched_country, "color": "#38bdf8"})
            _mark_span(*m.span())
            break

    detected_year: Optional[int] = exact_date_parsed.year if exact_date_parsed else None
    if detected_year is None:
        for ym in re.finditer(r"\b(20[12]\d)\b", norm_query):
            if not any(ym.start() >= s and ym.end() <= e for s, e in consumed_spans):
                cand_year = int(ym.group(1))
                if cand_year in available_anios or 2015 <= cand_year <= 2030:
                    detected_year = cand_year
                    _mark_span(*ym.span())
                    break

    if detected_year is not None:
        st.session_state.cal_anio = detected_year
        detected.append({"tipo": "Año", "valor": str(detected_year), "color": "#a78bfa"})

    detected_month: Optional[int] = exact_date_parsed.month if exact_date_parsed else None
    if detected_month is None:
        for m_alias in sorted(MONTH_ALIASES.keys(), key=len, reverse=True):
            m = re.search(rf"\b{re.escape(m_alias)}\b", norm_query)
            if m and not any(m.start() >= s and m.end() <= e for s, e in consumed_spans):
                detected_month = MONTH_ALIASES[m_alias]
                _mark_span(*m.span())
                break

    if detected_month is not None:
        st.session_state.cal_mes = detected_month
        detected.append({"tipo": "Mes", "valor": MESES_NOMBRES[detected_month], "color": "#f472b6"})

    if exact_date_parsed is not None:
        st.session_state.cal_dia = exact_date_parsed.day
        st.session_state.temporal_mode = "fecha_exacta"
        detected.append({"tipo": "Fecha exacta", "valor": exact_date_parsed.isoformat(), "color": "#fb7185"})
    elif detected_month is not None:
        day_found = False
        for dm in re.finditer(r"\b([1-9]|[12]\d|3[01])\b", norm_query):
            if not any(dm.start() >= s and dm.end() <= e for s, e in consumed_spans):
                cand_day = int(dm.group(1))
                max_d = calendar.monthrange(st.session_state.cal_anio, detected_month)[1]
                if 1 <= cand_day <= max_d:
                    st.session_state.cal_dia = cand_day
                    st.session_state.temporal_mode = "fecha_exacta"
                    iso_str = f"{st.session_state.cal_anio:04d}-{detected_month:02d}-{cand_day:02d}"
                    detected.append({"tipo": "Fecha exacta", "valor": iso_str, "color": "#fb7185"})
                    _mark_span(*dm.span())
                    day_found = True
                    break
        if not day_found:
            st.session_state.temporal_mode = "mes_completo"
    elif re.search(r"\b(ano completo|anio completo|todo el ano|todo el anio|anual)\b", norm_query):
        st.session_state.temporal_mode = "anio_completo"
        detected.append({"tipo": "Periodo", "valor": "Año completo", "color": "#c084fc"})
    elif detected_year is not None and detected_month is None:
        st.session_state.temporal_mode = "anio_completo"
        detected.append({"tipo": "Periodo", "valor": "Año completo", "color": "#c084fc"})

    sat_catalog = available_sats if available_sats else ["Terra", "Aqua", "N20", "SNPP"]
    sat_aliases: Dict[str, List[str]] = {
        "terra": ["Terra"],
        "aqua": ["Aqua"],
        "n20": ["N20", "NOAA-20"],
        "noaa 20": ["N20", "NOAA-20"],
        "noaa-20": ["N20", "NOAA-20"],
        "n21": ["N21", "NOAA-21"],
        "noaa 21": ["N21", "NOAA-21"],
        "snpp": ["SNPP", "Suomi NPP"],
        "suomi": ["SNPP", "Suomi NPP"],
    }
    matched_sats: List[str] = []
    for s_val in sat_catalog:
        norm_s = _strip_accents(s_val)
        m = re.search(rf"\b{re.escape(norm_s)}\b", norm_query)
        if m:
            matched_sats.append(s_val)
            _mark_span(*m.span())
    for alias_s, targets in sat_aliases.items():
        m = re.search(rf"\b{re.escape(alias_s)}\b", norm_query)
        if m:
            for s_val in sat_catalog:
                if any(_strip_accents(t) == _strip_accents(s_val) for t in targets) and s_val not in matched_sats:
                    matched_sats.append(s_val)
            _mark_span(*m.span())

    if matched_sats:
        st.session_state.active_sat = matched_sats
        detected.append({"tipo": "Satélite", "valor": ", ".join(matched_sats), "color": "#34d399"})

    inst_catalog = available_insts if available_insts else ["MODIS", "VIIRS"]
    matched_insts: List[str] = []
    for i_val in inst_catalog:
        norm_i = _strip_accents(i_val)
        m = re.search(rf"\b{re.escape(norm_i)}\b", norm_query)
        if m:
            matched_insts.append(i_val)
            _mark_span(*m.span())

    if matched_insts:
        st.session_state.active_inst = matched_insts
        detected.append({"tipo": "Sensor", "valor": ", ".join(matched_insts), "color": "#2dd4bf"})

    risk_rules = [
        ("Alto", [r"\balto\b", r"\baltos\b", r"\bcritico\b", r"\bcriticos\b", r"\bgrave\b", r"\brojo\b"]),
        ("Medio", [r"\bmedio\b", r"\bmedios\b", r"\bmoderado\b", r"\bmoderados\b", r"\bnaranja\b", r"\bambar\b"]),
        ("Bajo", [r"\bbajo\b", r"\bbajos\b", r"\bleve\b", r"\bperiferico\b", r"\bverde\b"]),
    ]
    matched_risks: List[str] = []
    for canonical_risk, patterns in risk_rules:
        for pat in patterns:
            m = re.search(pat, norm_query)
            if m and not any(m.start() >= s and m.end() <= e for s, e in consumed_spans):
                if canonical_risk not in matched_risks:
                    matched_risks.append(canonical_risk)
                _mark_span(*m.span())

    if matched_risks:
        st.session_state.active_riesgo = matched_risks
        detected.append({"tipo": "Nivel de Riesgo", "valor": ", ".join(matched_risks), "color": "#fbbf24"})

    stopwords = {"de", "del", "la", "el", "los", "las", "en", "y", "con", "para", "por", "mes", "ano", "anio"}
    mask = [False] * len(norm_query)
    for st_idx, en_idx in consumed_spans:
        for idx in range(st_idx, min(en_idx, len(mask))):
            mask[idx] = True

    residual_chars = [ch if not mask[i] else " " for i, ch in enumerate(norm_query)]
    residual_tokens = [
        tok for tok in re.findall(r"[a-z0-9\-]+", "".join(residual_chars)) if tok not in stopwords and len(tok) > 1
    ]
    if residual_tokens:
        detected.append({"tipo": "No reconocido", "valor": " ".join(residual_tokens), "color": "#94a3b8"})

    st.session_state.last_parsed_entities = detected
    return detected


# 6B. Utilidades de Selección Espacial por Dibujo en 2D (Ray-Casting Vectorizado + Vista Previa Cartográfica SVG)
def points_in_polygon(lons: np.ndarray, lats: np.ndarray, polygon: List[List[float]]) -> np.ndarray:
    """Evalúa mediante Ray-Casting vectorizado qué puntos (lon, lat) caen dentro de un polígono cerrado."""
    if len(polygon) < 3 or len(lons) == 0:
        return np.ones(len(lons), dtype=bool)

    poly = np.asarray(polygon, dtype=np.float64)
    x = np.asarray(lons, dtype=np.float64)
    y = np.asarray(lats, dtype=np.float64)
    n = len(poly)
    inside = np.zeros(len(x), dtype=bool)

    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        cond_y = (y > min(p1y, p2y)) & (y <= max(p1y, p2y))
        cond_x = x <= max(p1x, p2x)
        mask = cond_y & cond_x
        if np.any(mask):
            if p1y != p2y:
                xinters = (y[mask] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            else:
                xinters = np.full(int(np.sum(mask)), p1x, dtype=np.float64)
            cross = (p1x == p2x) | (x[mask] <= xinters)
            idx = np.where(mask)[0]
            inside[idx[cross]] = ~inside[idx[cross]]
        p1x, p1y = p2x, p2y

    return inside


def build_circular_brush_polygon(
    center_lon: float, center_lat: float, radius_km: float, num_vertices: int = 36
) -> List[List[float]]:
    """Genera un polígono geodésico circular alrededor de un punto (pincel de radio fijo en km)."""
    deg_lat = radius_km / 111.195
    cos_lat = max(0.15, float(np.cos(np.deg2rad(center_lat))))
    deg_lon = radius_km / (111.195 * cos_lat)
    angles = np.linspace(0.0, 2.0 * np.pi, num_vertices, endpoint=False)
    ring = [
        [round(float(center_lon + deg_lon * np.cos(a)), 5), round(float(center_lat + deg_lat * np.sin(a)), 5)]
        for a in angles
    ]
    ring.append(ring[0])
    return ring


def _convex_hull_2d(pts: List[List[float]]) -> List[List[float]]:
    """Calcula la envolvente convexa (Monotone Chain) si una selección de puntos no trae contorno."""
    unique_pts = sorted({(round(float(p[0]), 5), round(float(p[1]), 5)) for p in pts})
    if len(unique_pts) < 3:
        return []

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[tuple[float, float]] = []
    for p in unique_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[tuple[float, float]] = []
    for p in reversed(unique_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return []
    ring = [[p[0], p[1]] for p in hull]
    ring.append(ring[0])
    return ring


def build_2d_cartographic_preview_svg(
    reference_df: pd.DataFrame,
    draft_poly: Optional[List[List[float]]] = None,
    open_vertices: Optional[List[List[float]]] = None,
    cursor_center: Optional[tuple[float, float]] = None,
) -> str:
    """Genera un mapa vectorial 2D nativo (SVG) con fronteras, focos térmicos y el trazo dibujado en vivo."""
    width, height = 780, 310
    pad = 24

    all_lons: List[float] = []
    all_lats: List[float] = []

    if not reference_df.empty:
        all_lons.extend([float(reference_df["lon_grid"].min()), float(reference_df["lon_grid"].max())])
        all_lats.extend([float(reference_df["lat_grid"].min()), float(reference_df["lat_grid"].max())])
    else:
        all_lons.extend([-115.0, -34.0])
        all_lats.extend([-55.0, 28.0])

    for pt_list in (draft_poly or [], open_vertices or []):
        for pt in pt_list:
            if len(pt) >= 2:
                all_lons.append(float(pt[0]))
                all_lats.append(float(pt[1]))

    if cursor_center is not None:
        all_lons.append(float(cursor_center[0]))
        all_lats.append(float(cursor_center[1]))

    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_min, lat_max = min(all_lats), max(all_lats)

    span_lon = max(lon_max - lon_min, 3.5)
    span_lat = max(lat_max - lat_min, 3.0)
    lon_min -= span_lon * 0.14
    lon_max += span_lon * 0.14
    lat_min -= span_lat * 0.14
    lat_max += span_lat * 0.14

    mid_lat = 0.5 * (lat_min + lat_max)
    cos_corr = max(0.35, float(np.cos(np.deg2rad(mid_lat))))
    inner_w = width - 2 * pad
    inner_h = height - 2 * pad

    deg_w = (lon_max - lon_min) * cos_corr
    deg_h = lat_max - lat_min
    target_ratio = inner_w / inner_h
    if (deg_w / deg_h) < target_ratio:
        needed_deg_w = deg_h * target_ratio
        extra_lon = ((needed_deg_w / cos_corr) - (lon_max - lon_min)) * 0.5
        lon_min -= extra_lon
        lon_max += extra_lon
    else:
        needed_deg_h = deg_w / target_ratio
        extra_lat = (needed_deg_h - deg_h) * 0.5
        lat_min -= extra_lat
        lat_max += extra_lat

    def _proj(lon_v: float, lat_v: float) -> tuple[float, float]:
        px = pad + ((lon_v - lon_min) / (lon_max - lon_min)) * inner_w
        py = pad + ((lat_max - lat_v) / (lat_max - lat_min)) * inner_h
        return round(px, 1), round(py, 1)

    svg_parts: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:auto;border-radius:12px;border:1px solid rgba(255,255,255,0.14);background:#0b1120;display:block;">'
    ]

    # Grilla de meridianos y paralelos
    for frac in (0.2, 0.4, 0.6, 0.8):
        gx = round(pad + frac * inner_w, 1)
        gy = round(pad + frac * inner_h, 1)
        g_lon = lon_min + frac * (lon_max - lon_min)
        g_lat = lat_max - frac * (lat_max - lat_min)
        svg_parts.append(
            f'<line x1="{gx}" y1="{pad}" x2="{gx}" y2="{height - pad}" stroke="rgba(148,163,184,0.12)" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<line x1="{pad}" y1="{gy}" x2="{width - pad}" y2="{gy}" stroke="rgba(148,163,184,0.12)" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{gx}" y="{height - 7}" fill="#64748b" font-size="10" text-anchor="middle">{g_lon:.1f}°</text>'
        )
        svg_parts.append(
            f'<text x="4" y="{gy + 3}" fill="#64748b" font-size="10">{g_lat:.1f}°</text>'
        )

    # Contornos geográficos desde el GeoJSON en caché
    world_gj = load_world_map_geojson()
    if world_gj:
        land_paths: List[str] = []
        for feat in world_gj.get("features", []):
            geom = feat.get("geometry") or {}
            gtype = geom.get("type")
            coords = geom.get("coordinates") or []
            rings = coords if gtype == "Polygon" else [r for poly in coords for r in poly] if gtype == "MultiPolygon" else []
            for ring in rings:
                if not ring:
                    continue
                xs = [pt[0] for pt in ring if len(pt) >= 2]
                ys = [pt[1] for pt in ring if len(pt) >= 2]
                if not xs or max(xs) < lon_min - 5 or min(xs) > lon_max + 5 or max(ys) < lat_min - 5 or min(ys) > lat_max + 5:
                    continue
                pts_str = []
                for pt in ring[::2]:
                    sx, sy = _proj(float(pt[0]), float(pt[1]))
                    pts_str.append(f"{sx},{sy}")
                if len(pts_str) >= 3:
                    land_paths.append(f"M {' L '.join(pts_str)} Z")
        if land_paths:
            svg_parts.append(
                f'<path d="{" ".join(land_paths)}" fill="#162033" stroke="#334155" stroke-width="1"/>'
            )

    # Focos térmicos filtrados (resaltando los que caen dentro del dibujo)
    if not reference_df.empty:
        sample_df = (
            reference_df.sort_values("total_incendios", ascending=False).head(3200)
            if len(reference_df) > 3200
            else reference_df
        )
        lons_s = sample_df["lon_grid"].to_numpy(dtype=np.float64)
        lats_s = sample_df["lat_grid"].to_numpy(dtype=np.float64)
        has_poly = draft_poly is not None and len(draft_poly) >= 3
        inside_mask = points_in_polygon(lons_s, lats_s, draft_poly) if has_poly else np.ones(len(lons_s), dtype=bool)

        outside_circles: List[str] = []
        inside_circles: List[str] = []
        for lo, la, is_in in zip(lons_s, lats_s, inside_mask):
            sx, sy = _proj(float(lo), float(la))
            if pad <= sx <= width - pad and pad <= sy <= height - pad:
                if is_in:
                    inside_circles.append(f'<circle cx="{sx}" cy="{sy}" r="2.6" fill="#fb923c" fill-opacity="0.88"/>')
                else:
                    outside_circles.append(f'<circle cx="{sx}" cy="{sy}" r="1.8" fill="#64748b" fill-opacity="0.35"/>')

        svg_parts.extend(outside_circles)
        svg_parts.extend(inside_circles)

    # Polígono / círculo cerrado trazado
    if draft_poly and len(draft_poly) >= 3:
        poly_pts = []
        for pt in draft_poly:
            sx, sy = _proj(float(pt[0]), float(pt[1]))
            poly_pts.append(f"{sx},{sy}")
        svg_parts.append(
            f'<polygon points="{" ".join(poly_pts)}" fill="rgba(56,189,248,0.22)" '
            f'stroke="#38bdf8" stroke-width="2.5"/>'
        )

    # Vértices abiertos si el usuario está trazando punto a punto
    if open_vertices and len(open_vertices) >= 1:
        v_pts = [_proj(float(p[0]), float(p[1])) for p in open_vertices]
        if len(v_pts) >= 2:
            line_str = " ".join(f"{vx},{vy}" for vx, vy in v_pts)
            svg_parts.append(
                f'<polyline points="{line_str}" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-dasharray="5,4"/>'
            )
        for idx, (vx, vy) in enumerate(v_pts):
            fill_v = "#22c55e" if idx == 0 else "#38bdf8"
            svg_parts.append(f'<circle cx="{vx}" cy="{vy}" r="4.5" fill="{fill_v}" stroke="#0f172a" stroke-width="1.5"/>')

    # Mira del centro del pincel
    if cursor_center is not None:
        cx, cy = _proj(float(cursor_center[0]), float(cursor_center[1]))
        svg_parts.append(
            f'<line x1="{cx - 7}" y1="{cy}" x2="{cx + 7}" y2="{cy}" stroke="#38bdf8" stroke-width="1.6"/>'
            f'<line x1="{cx}" y1="{cy - 7}" x2="{cx}" y2="{cy + 7}" stroke="#38bdf8" stroke-width="1.6"/>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


# 6C. Utilidades de Exportación de Registros Filtrados (CSV, Excel, GeoJSON) para Backend Externo / Local
def _slugify_filename_part(value: str) -> str:
    clean = unicodedata.normalize("NFD", str(value).strip())
    clean = "".join(ch for ch in clean if unicodedata.category(ch) != "Mn")
    clean = re.sub(r"[^A-Za-z0-9\-]+", "_", clean).strip("_")
    return clean or "LATAM"


def build_export_filename(
    pais: str,
    temporal_mode: str,
    fecha_iso: str,
    cal_mes: int,
    cal_anio: int,
    extension: str,
    has_drawn_zone: bool = False,
) -> str:
    pais_slug = "LATAM_Todos" if pais == "Todos" else _slugify_filename_part(pais)
    if temporal_mode == "fecha_exacta":
        period_slug = fecha_iso
    elif temporal_mode == "mes_completo":
        period_slug = f"{cal_anio:04d}-{cal_mes:02d}_{_slugify_filename_part(MESES_NOMBRES[cal_mes])}"
    else:
        period_slug = f"Anio_{cal_anio:04d}"
    zone_suffix = "_Zona2D" if has_drawn_zone else ""
    return f"hotspots_{pais_slug}_{period_slug}{zone_suffix}.{extension}"


def prepare_export_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS)
    cols = [c for c in EXPORT_COLUMNS if c in df.columns]
    return df[cols].copy().reset_index(drop=True)


def convert_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    export_df = prepare_export_dataframe(df)
    return export_df.to_csv(index=False).encode("utf-8-sig")


def _build_spreadsheetml_fallback(export_df: pd.DataFrame, summary_df: pd.DataFrame) -> bytes:
    """Genera un libro Excel XML (SpreadsheetML) puro sin dependencias externas si no hay openpyxl/xlsxwriter."""

    def _sheet_xml(sheet_name: str, frame: pd.DataFrame) -> str:
        rows_xml = []
        header_cells = "".join(
            f'<Cell ss:StyleID="Header"><Data ss:Type="String">{xml_escape(str(col))}</Data></Cell>'
            for col in frame.columns
        )
        rows_xml.append(f"<Row>{header_cells}</Row>")
        for row in frame.itertuples(index=False):
            cells = []
            for val in row:
                if isinstance(val, (int, float, np.integer, np.floating)) and not pd.isna(val):
                    cells.append(f'<Cell><Data ss:Type="Number">{val}</Data></Cell>')
                else:
                    cells.append(f'<Cell><Data ss:Type="String">{xml_escape(str(val))}</Data></Cell>')
            rows_xml.append(f"<Row>{''.join(cells)}</Row>")
        return f'<Worksheet ss:Name="{xml_escape(sheet_name)}"><Table>{"".join(rows_xml)}</Table></Worksheet>'

    xml_doc = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<?mso-application progid="Excel.Sheet"?>'
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" '
        'xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">'
        '<Styles><Style ss:ID="Header"><Font ss:Bold="1"/></Style></Styles>'
        f"{_sheet_xml('Registros_Filtrados', export_df)}"
        f"{_sheet_xml('Resumen_Consulta', summary_df)}"
        "</Workbook>"
    )
    return xml_doc.encode("utf-8")


def convert_df_to_excel_bytes(df: pd.DataFrame, resumen_meta: Dict[str, Any]) -> tuple[bytes, str, str]:
    """Retorna (bytes, extension, mime_type) garantizando exportación Excel en cualquier servidor."""
    export_df = prepare_export_dataframe(df)
    output = io.BytesIO()

    total_celdas = len(export_df)
    total_incendios = int(export_df["total_incendios"].sum()) if total_celdas > 0 else 0
    if total_celdas > 0 and total_incendios > 0:
        w = export_df["total_incendios"].to_numpy(dtype=np.float64)
        brillo_pond = round(float(np.average(export_df["brillo_promedio"], weights=w)), 2)
        frp_pond = round(float(np.average(export_df["frp_promedio"], weights=w)), 2)
    else:
        brillo_pond = 0.0
        frp_pond = 0.0

    summary_rows = [
        {"Parametro": "Fecha de Exportación", "Valor": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"Parametro": "Origen de Datos", "Valor": str(resumen_meta.get("origen", "Backend API"))},
        {"Parametro": "País / Región Filtrada", "Valor": str(resumen_meta.get("pais", "Todos"))},
        {"Parametro": "Periodo Temporal", "Valor": str(resumen_meta.get("periodo", ""))},
        {"Parametro": "Selección por Dibujo 2D", "Valor": str(resumen_meta.get("zona_2d", "No aplicada"))},
        {"Parametro": "Satélites Activos", "Valor": str(resumen_meta.get("satelites", ""))},
        {"Parametro": "Sensores / Instrumentos", "Valor": str(resumen_meta.get("sensores", ""))},
        {"Parametro": "Niveles de Riesgo", "Valor": str(resumen_meta.get("riesgo", ""))},
        {"Parametro": "Filtro FRP (MW)", "Valor": str(resumen_meta.get("frp_filtro", "Sin límite"))},
        {"Parametro": "Filtro Brillo (K)", "Valor": str(resumen_meta.get("brillo_filtro", "Sin límite"))},
        {"Parametro": "Total Celdas Exportadas", "Valor": total_celdas},
        {"Parametro": "Total Incendios / Focos", "Valor": total_incendios},
        {"Parametro": "Brillo Promedio Ponderado (K)", "Valor": brillo_pond},
        {"Parametro": "FRP Promedio Ponderado (MW)", "Valor": frp_pond},
    ]
    summary_df = pd.DataFrame(summary_rows)
    xlsx_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    try:
        import openpyxl  # noqa: F401
        from openpyxl.styles import Alignment, Font, PatternFill

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            export_df.to_excel(writer, sheet_name="Registros_Filtrados", index=False)
            summary_df.to_excel(writer, sheet_name="Resumen_Consulta", index=False)

            header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True)

            for sheet_name in ("Registros_Filtrados", "Resumen_Consulta"):
                ws = writer.sheets[sheet_name]
                ws.freeze_panes = "A2"
                for cell in ws[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                for col_cells in ws.columns:
                    max_len = max(len(str(c.value or "")) for c in col_cells)
                    ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 34)

        return output.getvalue(), "xlsx", xlsx_mime
    except ImportError:
        try:
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                export_df.to_excel(writer, sheet_name="Registros_Filtrados", index=False)
                summary_df.to_excel(writer, sheet_name="Resumen_Consulta", index=False)
            return output.getvalue(), "xlsx", xlsx_mime
        except ImportError:
            xml_bytes = _build_spreadsheetml_fallback(export_df, summary_df)
            return xml_bytes, "xls", "application/vnd.ms-excel"


def convert_df_to_geojson_bytes(
    df: pd.DataFrame,
    resumen_meta: Dict[str, Any],
    drawn_polygon: Optional[List[List[float]]] = None,
) -> bytes:
    export_df = prepare_export_dataframe(df)
    features: List[Dict[str, Any]] = []

    if drawn_polygon and len(drawn_polygon) >= 3:
        closed_ring = list(drawn_polygon)
        if closed_ring[0] != closed_ring[-1]:
            closed_ring.append(closed_ring[0])
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [closed_ring],
                },
                "properties": {
                    "tipo_geometria": "zona_seleccion_dibujo_2d",
                    "vertices": len(closed_ring),
                },
            }
        )

    for row in export_df.to_dict(orient="records"):
        lon_val = round(float(row["lon_grid"]), 5)
        lat_val = round(float(row["lat_grid"]), 5)
        props = {
            "tipo_geometria": "celda_incendio",
            "pais": str(row["pais"]),
            "fecha": str(row["fecha"]),
            "anio": int(row["anio"]),
            "mes": int(row["mes"]),
            "total_incendios": int(row["total_incendios"]),
            "frp_promedio_mw": float(row["frp_promedio"]),
            "brillo_promedio_k": float(row["brillo_promedio"]),
            "nivel_riesgo": str(row["nivel_riesgo"]),
            "satelite": str(row["satelite"]),
            "instrumento": str(row["instrumento"]),
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon_val, lat_val],
                },
                "properties": props,
            }
        )

    feature_collection = {
        "type": "FeatureCollection",
        "metadata": {
            "generado_en": datetime.now().isoformat(timespec="seconds"),
            "filtros": resumen_meta,
            "total_registros": len(export_df),
        },
        "features": features,
    }
    return json.dumps(feature_collection, ensure_ascii=False, indent=2).encode("utf-8")


# 7. Campo Térmico Continuo y Mallas Hexagonales
def _normalize_percentile(arr: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if len(clean) <= 1:
        return np.full(len(clean), 0.5, dtype=np.float64)
    p5, p98 = np.percentile(clean, [5, 98])
    if not np.isfinite(p5) or not np.isfinite(p98) or (p98 - p5) < 1e-6:
        return np.full(len(clean), 0.5, dtype=np.float64)
    return np.clip((clean - p5) / (p98 - p5), 0.0, 1.0)


def compute_smoothed_thermal_field(
    q_bins: np.ndarray,
    r_bins: np.ndarray,
    brightness: np.ndarray,
    frp: np.ndarray,
    counts: np.ndarray,
    smooth_weight: float = 0.35,
) -> np.ndarray:
    t_bright = _normalize_percentile(brightness)
    t_frp = _normalize_percentile(np.log1p(np.maximum(frp, 0.0)))
    t_count = _normalize_percentile(np.log1p(np.maximum(counts, 0.0)))

    raw_heat = np.nan_to_num(0.45 * t_bright + 0.30 * t_frp + 0.25 * t_count, nan=0.25)

    if smooth_weight <= 0.0 or len(raw_heat) <= 1:
        return np.clip(raw_heat, 0.0, 1.0)

    coord_to_heat = {(int(q), int(r)): float(h) for q, r, h in zip(q_bins, r_bins, raw_heat)}
    neighbor_offsets = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
    self_weight = 1.0 - smooth_weight

    smoothed = np.empty_like(raw_heat)
    for idx, (q, r, h) in enumerate(zip(q_bins, r_bins, raw_heat)):
        qi, ri = int(q), int(r)
        nbrs = [coord_to_heat[(qi + dq, ri + dr)] for dq, dr in neighbor_offsets if (qi + dq, ri + dr) in coord_to_heat]
        if nbrs:
            smoothed[idx] = self_weight * h + smooth_weight * (sum(nbrs) / len(nbrs))
        else:
            smoothed[idx] = h

    return np.clip(smoothed, 0.0, 1.0)


def thermal_values_to_rgba(
    heat_values: np.ndarray,
    palette_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(heat_values)
    if n == 0:
        empty = np.zeros((0, 4), dtype=np.uint8)
        return empty, empty

    pal = THERMAL_PALETTES.get(palette_name, next(iter(THERMAL_PALETTES.values())))
    stops = pal["stops"]
    t = np.clip(np.nan_to_num(heat_values, nan=0.0), 0.0, 1.0)

    r = np.interp(t, stops, pal["r"]).astype(np.uint8)
    g = np.interp(t, stops, pal["g"]).astype(np.uint8)
    b = np.interp(t, stops, pal["b"]).astype(np.uint8)
    a = np.interp(t, [0.0, 1.0], [185, 242]).astype(np.uint8)

    fill_rgba = np.column_stack([r, g, b, a])
    line_rgb = (fill_rgba[:, :3].astype(np.float32) * 0.80).astype(np.uint8)
    line_a = np.clip(fill_rgba[:, 3].astype(np.int32) + 12, 0, 255).astype(np.uint8)
    line_rgba = np.column_stack([line_rgb, line_a])

    return fill_rgba, line_rgba


def _lonlat_to_axial_bins(lons: np.ndarray, lats: np.ndarray, radius_deg: float) -> tuple[np.ndarray, np.ndarray]:
    q = (2.0 / 3.0) * lons / radius_deg
    r = (-lons / 3.0 + (np.sqrt(3.0) / 3.0) * lats) / radius_deg
    s = -q - r

    qi = np.round(q)
    ri = np.round(r)
    si = np.round(s)

    dq = np.abs(qi - q)
    dr = np.abs(ri - r)
    ds = np.abs(si - s)

    c1 = (dq > dr) & (dq > ds)
    c2 = (~c1) & (dr > ds)
    qi = np.where(c1, -ri - si, qi).astype(np.int64)
    ri = np.where(c2, -qi - si, ri).astype(np.int64)
    return qi, ri


@st.cache_data
def build_regional_or_globe_geodesic_grid(
    radius_deg: float,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
    lat_min: float = -82.0,
    lat_max: float = 82.0,
    is_globe: bool = False,
    globe_altitude_m: float = 1200.0,
) -> pd.DataFrame:
    dx = 1.5 * radius_deg
    dy = np.sqrt(3.0) * radius_deg

    q_min = int(np.floor(lon_min / dx))
    q_max = int(np.ceil(lon_max / dx))
    qs = np.arange(q_min, q_max + 1, dtype=np.int64)

    q_list = []
    r_list = []
    for q_val in qs:
        r_min = int(np.floor((lat_min / dy) - (q_val / 2.0)))
        r_max = int(np.ceil((lat_max / dy) - (q_val / 2.0)))
        rs = np.arange(r_min, r_max + 1, dtype=np.int64)
        q_list.append(np.full_like(rs, q_val))
        r_list.append(rs)

    q_arr = np.concatenate(q_list)
    r_arr = np.concatenate(r_list)

    c_lons = dx * q_arr.astype(np.float64)
    c_lats = dy * (r_arr.astype(np.float64) + q_arr.astype(np.float64) / 2.0)

    valid = (c_lons >= lon_min) & (c_lons <= lon_max) & (c_lats >= lat_min) & (c_lats <= lat_max)
    c_lons = c_lons[valid]
    c_lats = c_lats[valid]

    angles = np.deg2rad(np.arange(0, 360, 60))
    v_lons = c_lons[:, None] + radius_deg * np.cos(angles)[None, :]
    v_lats = np.clip(c_lats[:, None] + radius_deg * np.sin(angles)[None, :], -85.0, 85.0)

    if is_globe:
        v_alts = np.full_like(v_lons, float(globe_altitude_m))
        coords = np.stack([v_lons, v_lats, v_alts], axis=-1).round(5).tolist()
    else:
        coords = np.stack([v_lons, v_lats], axis=-1).round(5).tolist()

    return pd.DataFrame({"polygon": coords})


@st.cache_data
def build_matched_geodesic_grid(
    radius_deg: float,
    coverage_mode: str,
    data_lons: np.ndarray,
    data_lats: np.ndarray,
    center_lon: float,
    center_lat: float,
    block_size: int = 24,
    is_globe: bool = False,
    globe_altitude_m: float = 1200.0,
) -> pd.DataFrame:
    if coverage_mode == "globe":
        return build_regional_or_globe_geodesic_grid(
            radius_deg=radius_deg,
            lon_min=-180.0,
            lon_max=180.0,
            lat_min=-82.0,
            lat_max=82.0,
            is_globe=is_globe,
            globe_altitude_m=globe_altitude_m,
        )

    if coverage_mode == "latam":
        return build_regional_or_globe_geodesic_grid(
            radius_deg=radius_deg,
            lon_min=-125.0,
            lon_max=-30.0,
            lat_min=-60.0,
            lat_max=36.0,
            is_globe=is_globe,
            globe_altitude_m=globe_altitude_m,
        )

    if len(data_lons) > 0:
        all_lons = np.append(data_lons, center_lon)
        all_lats = np.append(data_lats, center_lat)
    else:
        all_lons = np.array([center_lon], dtype=np.float64)
        all_lats = np.array([center_lat], dtype=np.float64)

    qi, ri = _lonlat_to_axial_bins(all_lons, all_lats, radius_deg)
    bq = np.floor_divide(qi, block_size)
    br = np.floor_divide(ri, block_size)

    packed_blocks = (bq << 32) | (br & 0xFFFFFFFF)
    unique_blocks = np.unique(packed_blocks)

    expanded_set = set()
    for pb in unique_blocks.tolist():
        ubq = pb >> 32
        ubr = pb & 0xFFFFFFFF
        if ubr >= 0x80000000:
            ubr -= 0x100000000
        for dbq in (-1, 0, 1):
            for dbr in (-1, 0, 1):
                expanded_set.add((ubq + dbq, ubr + dbr))

    cq, cr = _lonlat_to_axial_bins(np.array([center_lon]), np.array([center_lat]), radius_deg)
    cbq, cbr = int(cq[0] // block_size), int(cr[0] // block_size)
    for dbq in range(-3, 4):
        for dbr in range(-2, 3):
            expanded_set.add((cbq + dbq, cbr + dbr))

    max_blocks = max(1, 85_000 // (block_size * block_size))
    blocks_list = list(expanded_set)
    if len(blocks_list) > max_blocks:
        blocks_list.sort(key=lambda b: (b[0] - cbq) ** 2 + (b[1] - cbr) ** 2)
        blocks_list = blocks_list[:max_blocks]

    local_offsets = np.arange(block_size, dtype=np.int64)
    lq, lr = np.meshgrid(local_offsets, local_offsets, indexing="ij")
    lq = lq.ravel()
    lr = lr.ravel()

    b_arr = np.array(blocks_list, dtype=np.int64)
    q_arr = (b_arr[:, 0:1] * block_size + lq[None, :]).ravel()
    r_arr = (b_arr[:, 1:2] * block_size + lr[None, :]).ravel()

    dx = 1.5 * radius_deg
    dy = np.sqrt(3.0) * radius_deg
    c_lons = dx * q_arr.astype(np.float64)
    c_lats = dy * (r_arr.astype(np.float64) + q_arr.astype(np.float64) / 2.0)

    valid_mask = (c_lats >= -82.0) & (c_lats <= 82.0) & (c_lons >= -180.0) & (c_lons <= 180.0)
    c_lons = c_lons[valid_mask]
    c_lats = c_lats[valid_mask]

    angles = np.deg2rad(np.arange(0, 360, 60))
    v_lons = c_lons[:, None] + radius_deg * np.cos(angles)[None, :]
    v_lats = np.clip(c_lats[:, None] + radius_deg * np.sin(angles)[None, :], -85.0, 85.0)

    if is_globe:
        v_alts = np.full_like(v_lons, float(globe_altitude_m))
        coords = np.stack([v_lons, v_lats, v_alts], axis=-1).round(5).tolist()
    else:
        coords = np.stack([v_lons, v_lats], axis=-1).round(5).tolist()

    return pd.DataFrame({"polygon": coords})


@st.cache_data
def build_clustered_hex_layer(
    df: pd.DataFrame,
    radius_deg: float,
    tier_label: str,
    palette_name: str,
    smooth_weight: float = 0.35,
    is_globe: bool = False,
    is_sensor_1to1: bool = False,
    globe_altitude_m: float = 1210.0,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    x = df["lon_grid"].to_numpy(dtype=np.float64)
    y = df["lat_grid"].to_numpy(dtype=np.float64)
    qi, ri = _lonlat_to_axial_bins(x, y, radius_deg)

    fires = df["total_incendios"].to_numpy(dtype=np.float64)
    work = pd.DataFrame(
        {
            "q_bin": qi,
            "r_bin": ri,
            "total_incendios": df["total_incendios"].to_numpy(dtype=np.int64),
            "w_brillo": df["brillo_promedio"].to_numpy(dtype=np.float64) * fires,
            "w_frp": df["frp_promedio"].to_numpy(dtype=np.float64) * fires,
            "celdas": 1,
            "instrumento": df["instrumento"].to_numpy(),
            "satelite": df["satelite"].to_numpy(),
            "pais": df["pais"].to_numpy(),
        }
    )

    agg = work.groupby(["q_bin", "r_bin"], as_index=False).agg(
        total_incendios=("total_incendios", "sum"),
        w_brillo=("w_brillo", "sum"),
        w_frp=("w_frp", "sum"),
        celdas_agrupadas=("celdas", "sum"),
        instrumento=("instrumento", "first"),
        satelite=("satelite", "first"),
        pais=("pais", "first"),
    )

    q_arr = agg["q_bin"].to_numpy(dtype=np.int64)
    r_arr = agg["r_bin"].to_numpy(dtype=np.int64)
    c_lons = radius_deg * 1.5 * q_arr.astype(np.float64)
    c_lats = np.clip(
        radius_deg * np.sqrt(3.0) * (r_arr.astype(np.float64) + q_arr.astype(np.float64) / 2.0),
        -84.0,
        84.0,
    )

    counts_arr = agg["total_incendios"].to_numpy(dtype=np.int64)
    brillo_arr = (agg["w_brillo"].to_numpy(dtype=np.float64) / counts_arr).round(1)
    frp_arr = (agg["w_frp"].to_numpy(dtype=np.float64) / counts_arr).round(1)

    heat_vals = compute_smoothed_thermal_field(
        q_arr, r_arr, brillo_arr, frp_arr, counts_arr.astype(np.float64), smooth_weight=smooth_weight
    )
    fill_rgba, line_rgba = thermal_values_to_rgba(heat_vals, palette_name=palette_name)

    nivel_termico = np.select(
        [heat_vals >= 0.72, heat_vals >= 0.42],
        ["Crítico / Alto", "Moderado / Medio"],
        default="Bajo / Periférico",
    )

    angles = np.deg2rad(np.arange(0, 360, 60))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    v_lons = c_lons[:, None] + radius_deg * cos_a[None, :]
    v_lats = np.clip(c_lats[:, None] + radius_deg * sin_a[None, :], -85.0, 85.0)

    if is_globe:
        v_alts = np.full_like(v_lons, float(globe_altitude_m))
        coords = np.stack([v_lons, v_lats, v_alts], axis=-1).round(5).tolist()
    else:
        coords = np.stack([v_lons, v_lats], axis=-1).round(5).tolist()

    km_per_deg = METERS_PER_DEG_LAT / 1000.0
    area_km2 = np.maximum(
        0.1, ((3.0 * np.sqrt(3.0) / 2.0) * ((radius_deg * km_per_deg) ** 2) * np.cos(np.deg2rad(c_lats)))
    ).round(2)
    diam_km = round(2.0 * radius_deg * km_per_deg, 2)

    max_count = max(int(counts_arr.max()), 1)
    max_elev = 40_000.0 if is_sensor_1to1 else 180_000.0
    elevations = ((counts_arr / max_count) * max_elev).round(0).tolist()
    n = len(agg)

    res_label = ["1 km × 1 km (Celda 1:1)"] * n if is_sensor_1to1 else [f"Clúster Ø ~{diam_km} km"] * n

    return pd.DataFrame(
        {
            "polygon": coords,
            "rgba": fill_rgba.tolist(),
            "line_rgba": line_rgba.tolist(),
            "elevation": elevations,
            "escala_vista": [tier_label] * n,
            "pais": agg["pais"].tolist(),
            "lat_grid": np.round(c_lats, 4).tolist(),
            "lon_grid": np.round(c_lons, 4).tolist(),
            "total_incendios": counts_arr.tolist(),
            "celdas_agrupadas": agg["celdas_agrupadas"].tolist(),
            "frp_promedio": frp_arr.tolist(),
            "brillo_promedio": brillo_arr.tolist(),
            "indice_termico": np.round(heat_vals * 100.0, 1).tolist(),
            "instrumento": agg["instrumento"].tolist(),
            "satelite": agg["satelite"].tolist(),
            "nivel_riesgo": nivel_termico.tolist(),
            "resolucion_m": res_label,
            "area_km2": area_km2.tolist(),
        }
    )


# 8. CONSULTA DE CATÁLOGOS Y DATOS SEGÚN ESTADO CANÓNICO
catalogs = fetch_api_catalogs(st.session_state.api_url, api_token=st.session_state.api_token)
paises_api = ["Todos"] + catalogs.get("paises", [])
anios_api = sorted(catalogs.get("anios", list(range(2018, 2026))), reverse=True)

if st.session_state.active_pais not in paises_api:
    st.session_state.active_pais = "Todos"
if st.session_state.cal_anio not in anios_api:
    st.session_state.cal_anio = anios_api[0]

max_dias_mes = calendar.monthrange(st.session_state.cal_anio, st.session_state.cal_mes)[1]
if st.session_state.cal_dia > max_dias_mes:
    st.session_state.cal_dia = max_dias_mes

fecha_iso_activa = f"{st.session_state.cal_anio:04d}-{st.session_state.cal_mes:02d}-{st.session_state.cal_dia:02d}"
query_fecha = fecha_iso_activa if st.session_state.temporal_mode == "fecha_exacta" else None
query_mes = st.session_state.cal_mes if st.session_state.temporal_mode == "mes_completo" else None

df_grid, is_live_api, connection_status = fetch_incendios_grid(
    st.session_state.api_url,
    st.session_state.allow_local_fallback,
    pais=st.session_state.active_pais,
    anio=st.session_state.cal_anio,
    mes=query_mes,
    fecha=query_fecha,
    strict_latam_bounds=st.session_state.strict_latam,
    limit=int(st.session_state.api_limit),
    api_token=st.session_state.api_token,
)

if not df_grid.empty:
    riesgos_disponibles = [r for r in ["Alto", "Medio", "Bajo"] if r in df_grid["nivel_riesgo"].unique()]
    if not riesgos_disponibles:
        riesgos_disponibles = sorted(df_grid["nivel_riesgo"].unique().tolist())
    sats_disponibles = sorted(df_grid["satelite"].unique().tolist())
    insts_disponibles = sorted(df_grid["instrumento"].unique().tolist())
else:
    riesgos_disponibles = ["Alto", "Medio", "Bajo"]
    sats_disponibles = catalogs.get("satelites", ["Terra", "Aqua", "N20", "SNPP"])
    insts_disponibles = catalogs.get("instrumentos", ["MODIS", "VIIRS"])

if st.session_state.active_riesgo is None:
    selected_riesgo = riesgos_disponibles
else:
    valid_r = [r for r in st.session_state.active_riesgo if r in riesgos_disponibles]
    selected_riesgo = valid_r if valid_r else riesgos_disponibles

if st.session_state.active_sat is None:
    selected_sat = sats_disponibles
else:
    valid_s = [s for s in st.session_state.active_sat if s in sats_disponibles]
    selected_sat = valid_s if valid_s else sats_disponibles

if st.session_state.active_inst is None:
    selected_inst = insts_disponibles
else:
    valid_i = [i for i in st.session_state.active_inst if i in insts_disponibles]
    selected_inst = valid_i if valid_i else insts_disponibles

if not df_grid.empty:
    base_filtered_df = df_grid[
        (df_grid["nivel_riesgo"].isin(selected_riesgo))
        & (df_grid["satelite"].isin(selected_sat))
        & (df_grid["instrumento"].isin(selected_inst))
    ].copy()
    if st.session_state.frp_min is not None:
        base_filtered_df = base_filtered_df[base_filtered_df["frp_promedio"] >= float(st.session_state.frp_min)]
    if st.session_state.frp_max is not None:
        base_filtered_df = base_filtered_df[base_filtered_df["frp_promedio"] <= float(st.session_state.frp_max)]
    if st.session_state.brillo_min is not None:
        base_filtered_df = base_filtered_df[base_filtered_df["brillo_promedio"] >= float(st.session_state.brillo_min)]
    if st.session_state.brillo_max is not None:
        base_filtered_df = base_filtered_df[base_filtered_df["brillo_promedio"] <= float(st.session_state.brillo_max)]
else:
    base_filtered_df = df_grid.copy()

has_active_2d_drawing = (
    st.session_state.drawn_polygon_2d is not None
    and len(st.session_state.drawn_polygon_2d) >= 3
    and st.session_state.drawn_filter_enabled
)

if has_active_2d_drawing and not base_filtered_df.empty:
    poly_mask = points_in_polygon(
        base_filtered_df["lon_grid"].to_numpy(dtype=np.float64),
        base_filtered_df["lat_grid"].to_numpy(dtype=np.float64),
        st.session_state.drawn_polygon_2d,
    )
    filtered_df = base_filtered_df[poly_mask].reset_index(drop=True)
else:
    filtered_df = base_filtered_df.reset_index(drop=True)

if st.session_state.temporal_mode == "fecha_exacta":
    periodo_txt = fecha_iso_activa
elif st.session_state.temporal_mode == "mes_completo":
    periodo_txt = f"{MESES_NOMBRES[st.session_state.cal_mes]} {st.session_state.cal_anio}"
else:
    periodo_txt = f"Año {st.session_state.cal_anio}"

frp_desc = "Sin límite"
if st.session_state.frp_min is not None or st.session_state.frp_max is not None:
    parts_f = []
    if st.session_state.frp_min is not None:
        parts_f.append(f"≥ {st.session_state.frp_min:g} MW")
    if st.session_state.frp_max is not None:
        parts_f.append(f"≤ {st.session_state.frp_max:g} MW")
    frp_desc = " y ".join(parts_f)

brillo_desc = "Sin límite"
if st.session_state.brillo_min is not None or st.session_state.brillo_max is not None:
    parts_b = []
    if st.session_state.brillo_min is not None:
        parts_b.append(f"≥ {st.session_state.brillo_min:g} K")
    if st.session_state.brillo_max is not None:
        parts_b.append(f"≤ {st.session_state.brillo_max:g} K")
    brillo_desc = " y ".join(parts_b)

zona_2d_desc = (
    f"Polígono cerrado ({len(st.session_state.drawn_polygon_2d)} vértices)"
    if has_active_2d_drawing
    else "No aplicada"
)

export_metadata: Dict[str, Any] = {
    "origen": f"Servidor Externo ({st.session_state.api_url})" if is_live_api else "Respaldo Local CSV",
    "pais": st.session_state.active_pais,
    "periodo": periodo_txt,
    "zona_2d": zona_2d_desc,
    "satelites": ", ".join(selected_sat),
    "sensores": ", ".join(selected_inst),
    "riesgo": ", ".join(selected_riesgo),
    "frp_filtro": frp_desc,
    "brillo_filtro": brillo_desc,
}

csv_payload = convert_df_to_csv_bytes(filtered_df) if not filtered_df.empty else b""
if not filtered_df.empty:
    excel_payload, excel_ext, excel_mime = convert_df_to_excel_bytes(filtered_df, export_metadata)
else:
    excel_payload, excel_ext, excel_mime = b"", "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

geojson_payload = (
    convert_df_to_geojson_bytes(
        filtered_df,
        export_metadata,
        drawn_polygon=st.session_state.drawn_polygon_2d if has_active_2d_drawing else None,
    )
    if not filtered_df.empty
    else b""
)

fname_csv = build_export_filename(
    st.session_state.active_pais,
    st.session_state.temporal_mode,
    fecha_iso_activa,
    st.session_state.cal_mes,
    st.session_state.cal_anio,
    "csv",
    has_drawn_zone=has_active_2d_drawing,
)
fname_excel = build_export_filename(
    st.session_state.active_pais,
    st.session_state.temporal_mode,
    fecha_iso_activa,
    st.session_state.cal_mes,
    st.session_state.cal_anio,
    excel_ext,
    has_drawn_zone=has_active_2d_drawing,
)
fname_geojson = build_export_filename(
    st.session_state.active_pais,
    st.session_state.temporal_mode,
    fecha_iso_activa,
    st.session_state.cal_mes,
    st.session_state.cal_anio,
    "geojson",
    has_drawn_zone=has_active_2d_drawing,
)


# 8B. DIÁLOGO MODAL DE CALENDARIO RÁPIDO
@st.dialog("📅 Cambiar Fecha o Periodo de Medición")
def open_quick_date_dialog(available_years: List[int]) -> None:
    c_mes, c_anio = st.columns([0.58, 0.42])
    with c_mes:
        dlg_mes = st.selectbox(
            "Mes:",
            options=list(range(1, 13)),
            index=st.session_state.cal_mes - 1,
            format_func=lambda m: MESES_NOMBRES[m],
            key="dlg_cal_mes",
        )
    with c_anio:
        dlg_anio = st.selectbox(
            "Año:",
            options=available_years,
            index=available_years.index(st.session_state.cal_anio),
            key="dlg_cal_anio",
        )

    if dlg_mes != st.session_state.cal_mes or dlg_anio != st.session_state.cal_anio:
        st.session_state.cal_mes = dlg_mes
        st.session_state.cal_anio = dlg_anio
        max_d = calendar.monthrange(dlg_anio, dlg_mes)[1]
        if st.session_state.cal_dia > max_d:
            st.session_state.cal_dia = max_d

    cal_obj = calendar.Calendar(firstweekday=0)
    semanas_mes = cal_obj.monthdatescalendar(st.session_state.cal_anio, st.session_state.cal_mes)

    with st.container(key="calendario_dialog_card"):
        st.markdown(
            """
            <div style="display:grid;grid-template-columns:repeat(7,1fr);text-align:center;
                        font-weight:700;font-size:13px;color:#f8fafc;padding:2px 0 8px 0;">
                <span>L</span><span>M</span><span>X</span><span>J</span><span>V</span><span>S</span><span>D</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for w_idx, semana in enumerate(semanas_mes):
            day_cols = st.columns(7)
            for d_idx, day_date in enumerate(semana):
                es_mes_actual = day_date.month == st.session_state.cal_mes
                es_dia_seleccionado = (
                    es_mes_actual
                    and st.session_state.temporal_mode == "fecha_exacta"
                    and day_date.day == st.session_state.cal_dia
                )
                with day_cols[d_idx]:
                    if not es_mes_actual:
                        st.button(
                            f"{day_date.day}",
                            key=f"dlg_cal_out_{w_idx}_{d_idx}_{day_date.isoformat()}",
                            disabled=True,
                            use_container_width=True,
                        )
                    else:
                        if st.button(
                            f"{day_date.day}",
                            key=f"dlg_cal_in_{st.session_state.cal_anio}_{st.session_state.cal_mes}_{day_date.day}",
                            type="primary" if es_dia_seleccionado else "secondary",
                            use_container_width=True,
                        ):
                            st.session_state.cal_dia = day_date.day
                            st.session_state.temporal_mode = "fecha_exacta"
                            st.rerun()

    col_opt_mes, col_opt_anio = st.columns(2)
    with col_opt_mes:
        if st.button(
            "📅 Mes completo",
            type="primary" if st.session_state.temporal_mode == "mes_completo" else "secondary",
            use_container_width=True,
            key="dlg_btn_mes_completo",
        ):
            st.session_state.temporal_mode = "mes_completo"
            st.rerun()
    with col_opt_anio:
        if st.button(
            "🗓️ Año completo",
            type="primary" if st.session_state.temporal_mode == "anio_completo" else "secondary",
            use_container_width=True,
            key="dlg_btn_anio_completo",
        ):
            st.session_state.temporal_mode = "anio_completo"
            st.rerun()

    if st.button("✅ Listo", type="primary", use_container_width=True, key="dlg_btn_close"):
        st.rerun()


# 8C. DIÁLOGO INTERACTIVO DE SELECCIÓN POR DIBUJO EN 2D (CON LIENZO EN VIVO SIN CERRAR EL MODAL)
@st.dialog("🖌️ Selección por Dibujo en 2D (Pincel / Lazo)", width="large")
def open_2d_drawing_dialog(reference_df: pd.DataFrame) -> None:
    if (
        st.session_state.draft_polygon_2d is None
        and not st.session_state.draft_cleared
        and st.session_state.drawn_polygon_2d is not None
    ):
        st.session_state.draft_polygon_2d = [list(pt) for pt in st.session_state.drawn_polygon_2d]

    if not reference_df.empty:
        default_lat = round(float(reference_df["lat_grid"].mean()), 2)
        default_lon = round(float(reference_df["lon_grid"].mean()), 2)
    else:
        default_lat, default_lon = -15.0, -60.0

    # Si Plotly está disponible, permitir también arrastre a mano alzada directo
    if HAS_PLOTLY:
        fig = go.Figure()
        world_gj = load_world_map_geojson()
        if world_gj:
            border_lons: List[Optional[float]] = []
            border_lats: List[Optional[float]] = []
            for feat in world_gj.get("features", []):
                geom = feat.get("geometry") or {}
                gtype = geom.get("type")
                coords = geom.get("coordinates") or []
                rings = coords if gtype == "Polygon" else [r for poly in coords for r in poly] if gtype == "MultiPolygon" else []
                for ring in rings:
                    for pt in ring:
                        if len(pt) >= 2 and LATAM_LON_MIN - 15 <= pt[0] <= LATAM_LON_MAX + 15 and LATAM_LAT_MIN - 10 <= pt[1] <= LATAM_LAT_MAX + 10:
                            border_lons.append(float(pt[0]))
                            border_lats.append(float(pt[1]))
                    border_lons.append(None)
                    border_lats.append(None)

            fig.add_trace(
                go.Scatter(
                    x=border_lons,
                    y=border_lats,
                    mode="lines",
                    line=dict(color="rgba(148, 163, 184, 0.35)", width=1),
                    hoverinfo="skip",
                )
            )

        if not reference_df.empty:
            plot_df = reference_df if len(reference_df) <= 10000 else reference_df.sample(10000, random_state=42)
            fig.add_trace(
                go.Scatter(
                    x=plot_df["lon_grid"],
                    y=plot_df["lat_grid"],
                    mode="markers",
                    marker=dict(size=6, color=plot_df["brillo_promedio"], colorscale="YlOrRd", opacity=0.82),
                    text=[
                        f"{p} · {f} focos · {b} K"
                        for p, f, b in zip(plot_df["pais"], plot_df["total_incendios"], plot_df["brillo_promedio"])
                    ],
                    hoverinfo="text",
                )
            )

        if st.session_state.draft_polygon_2d and len(st.session_state.draft_polygon_2d) >= 3:
            ring = list(st.session_state.draft_polygon_2d)
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in ring],
                    y=[p[1] for p in ring],
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(56, 189, 248, 0.22)",
                    line=dict(color="#38bdf8", width=2.5),
                )
            )

        fig.update_layout(
            dragmode="lasso",
            height=320,
            margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="#0b1120",
            plot_bgcolor="#0f172a",
            showlegend=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False, color="#94a3b8", scaleanchor="y", scaleratio=1),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False, color="#94a3b8"),
        )

        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode=("lasso", "box"),
            key="plotly_2d_brush_canvas",
        )

        if event and hasattr(event, "selection") and event.selection:
            sel = event.selection
            lasso_list = getattr(sel, "lasso", None) or (sel.get("lasso") if isinstance(sel, dict) else None)
            box_list = getattr(sel, "box", None) or (sel.get("box") if isinstance(sel, dict) else None)
            pts_list = getattr(sel, "points", None) or (sel.get("points") if isinstance(sel, dict) else None)

            new_ring: List[List[float]] = []
            if lasso_list and len(lasso_list) > 0:
                l0 = lasso_list[0]
                xs = l0.get("x", []) if isinstance(l0, dict) else getattr(l0, "x", [])
                ys = l0.get("y", []) if isinstance(l0, dict) else getattr(l0, "y", [])
                if len(xs) >= 3 and len(xs) == len(ys):
                    new_ring = [[round(float(xv), 5), round(float(yv), 5)] for xv, yv in zip(xs, ys)]
            elif box_list and len(box_list) > 0:
                b0 = box_list[0]
                xr = b0.get("x", []) if isinstance(b0, dict) else getattr(b0, "x", [])
                yr = b0.get("y", []) if isinstance(b0, dict) else getattr(b0, "y", [])
                if len(xr) >= 2 and len(yr) >= 2:
                    x0, x1 = min(float(xr[0]), float(xr[1])), max(float(xr[0]), float(xr[1]))
                    y0, y1 = min(float(yr[0]), float(yr[1])), max(float(yr[0]), float(yr[1]))
                    new_ring = [
                        [round(x0, 5), round(y0, 5)],
                        [round(x1, 5), round(y0, 5)],
                        [round(x1, 5), round(y1, 5)],
                        [round(x0, 5), round(y1, 5)],
                    ]
            elif pts_list and len(pts_list) >= 3:
                raw_pts = []
                for pt in pts_list:
                    px = pt.get("x") if isinstance(pt, dict) else getattr(pt, "x", None)
                    py = pt.get("y") if isinstance(pt, dict) else getattr(pt, "y", None)
                    if px is not None and py is not None:
                        raw_pts.append([float(px), float(py)])
                new_ring = _convex_hull_2d(raw_pts)

            if len(new_ring) >= 3:
                if new_ring[0] != new_ring[-1]:
                    new_ring.append(new_ring[0])
                st.session_state.draft_polygon_2d = new_ring
                st.session_state.draft_cleared = False

    # Controles de trazo circular y poligonal libre (sin st.rerun() para mantener abierto el modal)
    tab_circ, tab_poly = st.tabs(["🎯 Pincel Circular (Radio en km)", "✏️ Trazo Poligonal Libre (Vértices)"])

    with tab_circ:
        b1, b2, b3, b4 = st.columns([0.26, 0.26, 0.24, 0.24], vertical_alignment="bottom")
        with b1:
            brush_lat = st.number_input("Latitud centro:", value=default_lat, step=0.25, key="dlg_brush_lat")
        with b2:
            brush_lon = st.number_input("Longitud centro:", value=default_lon, step=0.25, key="dlg_brush_lon")
        with b3:
            brush_km = st.number_input(
                "Radio pincel (km):", min_value=5.0, max_value=2500.0, value=150.0, step=15.0, key="dlg_brush_km"
            )
        with b4:
            if st.button("🖌️ Trazar círculo", type="primary", use_container_width=True, key="dlg_btn_circle_brush"):
                st.session_state.draft_polygon_2d = build_circular_brush_polygon(brush_lon, brush_lat, brush_km)
                st.session_state.draft_vertices_2d = []
                st.session_state.draft_cleared = False

    with tab_poly:
        p1, p2, p3, p4 = st.columns([0.24, 0.24, 0.26, 0.26], vertical_alignment="bottom")
        with p1:
            v_lat = st.number_input("Latitud vértice:", value=default_lat, step=0.25, key="dlg_vert_lat")
        with p2:
            v_lon = st.number_input("Longitud vértice:", value=default_lon, step=0.25, key="dlg_vert_lon")
        with p3:
            if st.button("➕ Agregar punto", use_container_width=True, key="dlg_btn_add_vert"):
                verts = list(st.session_state.draft_vertices_2d)
                verts.append([round(float(v_lon), 5), round(float(v_lat), 5)])
                st.session_state.draft_vertices_2d = verts
                st.session_state.draft_cleared = False
                if len(verts) >= 3:
                    closed = list(verts) + [verts[0]]
                    st.session_state.draft_polygon_2d = closed
        with p4:
            if st.button(
                "↩️ Deshacer punto",
                disabled=len(st.session_state.draft_vertices_2d) == 0,
                use_container_width=True,
                key="dlg_btn_undo_vert",
            ):
                verts = list(st.session_state.draft_vertices_2d)
                verts.pop()
                st.session_state.draft_vertices_2d = verts
                if len(verts) >= 3:
                    st.session_state.draft_polygon_2d = list(verts) + [verts[0]]
                else:
                    st.session_state.draft_polygon_2d = None

    # Lienzo cartográfico 2D nativo siempre visible (muestra el trazo actualizado en vivo sin cerrar el modal)
    if not HAS_PLOTLY:
        svg_html = build_2d_cartographic_preview_svg(
            reference_df=reference_df,
            draft_poly=st.session_state.draft_polygon_2d,
            open_vertices=st.session_state.draft_vertices_2d,
            cursor_center=(float(brush_lon), float(brush_lat)),
        )
        st.markdown(svg_html, unsafe_allow_html=True)

    # Resumen en vivo de celdas y focos encerrados en el trazo actual
    draft_poly = st.session_state.draft_polygon_2d
    if draft_poly and len(draft_poly) >= 3:
        if not reference_df.empty:
            in_mask = points_in_polygon(
                reference_df["lon_grid"].to_numpy(dtype=np.float64),
                reference_df["lat_grid"].to_numpy(dtype=np.float64),
                draft_poly,
            )
            sel_count = int(np.sum(in_mask))
            sel_fires = int(reference_df.loc[in_mask, "total_incendios"].sum())
        else:
            sel_count, sel_fires = 0, 0
        st.success(
            f"Zona encerrada lista ({len(draft_poly)} vértices): **{sel_count:,}** celdas y **{sel_fires:,}** focos dentro del dibujo."
        )
    elif len(st.session_state.draft_vertices_2d) > 0:
        faltan = 3 - len(st.session_state.draft_vertices_2d)
        st.info(f"Trazo en progreso ({len(st.session_state.draft_vertices_2d)} puntos). Agrega **{faltan}** punto(s) más para cerrar la zona.")
    else:
        st.info("Ajusta las coordenadas y pulsa **«🖌️ Trazar círculo»** o agrega puntos en el polígono libre para ver la zona encerrada.")

    c_apply, c_clear, c_cancel = st.columns([0.46, 0.27, 0.27])
    with c_apply:
        if st.button(
            "✅ Aplicar zona dibujada al mapa",
            type="primary",
            disabled=not (draft_poly and len(draft_poly) >= 3),
            use_container_width=True,
            key="dlg_apply_2d_poly",
        ):
            st.session_state.drawn_polygon_2d = [list(pt) for pt in draft_poly]
            st.session_state.drawn_filter_enabled = True
            st.session_state.draft_polygon_2d = None
            st.session_state.draft_vertices_2d = []
            st.session_state.draft_cleared = False
            st.rerun()
    with c_clear:
        if st.button("🗑️ Limpiar trazo", use_container_width=True, key="dlg_clear_2d_poly"):
            st.session_state.draft_polygon_2d = None
            st.session_state.draft_vertices_2d = []
            st.session_state.drawn_polygon_2d = None
            st.session_state.draft_cleared = True
    with c_cancel:
        if st.button("Cerrar", use_container_width=True, key="dlg_close_2d_poly"):
            st.session_state.draft_polygon_2d = None
            st.session_state.draft_vertices_2d = []
            st.session_state.draft_cleared = False
            st.rerun()


# 9. VENTANA FLOTANTE INDEPENDIENTE DE CONEXIÓN AL SERVIDOR (ARRIBA A LA DERECHA)
with st.container(key="dock_servidor_derecho"):
    srv_badge = "🟢 API Conectada" if is_live_api else "🟠 Respaldo Local"
    with st.expander(f"🔌 Servidor Backend · {srv_badge}", expanded=False):
        new_api_url = st.text_input(
            "Endpoint Backend Externo:",
            value=st.session_state.api_url,
            key="w_api_endpoint",
        )
        if new_api_url != st.session_state.api_url:
            st.session_state.api_url = new_api_url
            st.rerun()

        new_api_token = st.text_input(
            "API Key / Bearer Token (opcional):",
            value=st.session_state.api_token,
            type="password",
            placeholder="Dejar vacío si el endpoint es público",
            key="w_api_token",
        )
        if new_api_token != st.session_state.api_token:
            st.session_state.api_token = new_api_token
            st.rerun()

        col_sync, col_fb = st.columns([0.52, 0.48])
        with col_sync:
            if st.button("🔄 Sincronizar", use_container_width=True, key="btn_sync_srv"):
                fetch_api_catalogs.clear()
                fetch_incendios_grid.clear()
                build_clustered_hex_layer.clear()
                build_matched_geodesic_grid.clear()
                build_regional_or_globe_geodesic_grid.clear()
                st.rerun()
        with col_fb:
            fb_val = st.checkbox(
                "Respaldo local",
                value=st.session_state.allow_local_fallback,
                key="w_allow_fb",
            )
            if fb_val != st.session_state.allow_local_fallback:
                st.session_state.allow_local_fallback = fb_val
                st.rerun()

        latam_val = st.checkbox(
            "Filtrar coordenadas fuera de LATAM",
            value=st.session_state.strict_latam,
            key="w_strict_latam",
        )
        if latam_val != st.session_state.strict_latam:
            st.session_state.strict_latam = latam_val
            st.rerun()

        if is_live_api:
            st.success(connection_status)
        else:
            st.warning(connection_status)


# 10. BARRA SUPERIOR IZQUIERDA (NAVEGACIÓN ENTRE GRID PRINCIPAL Y PÁGINA DE FILTROS)
with st.container(key="dock_superior_izquierdo"):
    nav_choice = st.radio(
        "Navegación:",
        options=["🗺️ Grid de Incendios", "🔍 Filtros y Buscador"],
        index=0 if is_map_page else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="w_nav_radio",
    )
    if nav_choice != st.session_state.active_page:
        st.session_state.active_page = nav_choice
        st.rerun()

    if is_map_page:
        col_toggle, col_badge = st.columns([0.66, 0.34])
        with col_toggle:
            btn_label = "👁️ Ocultar paneles" if st.session_state.dock_visible else "☰ Mostrar paneles"
            if st.button(btn_label, use_container_width=True, key="btn_toggle_dock"):
                st.session_state.dock_visible = not st.session_state.dock_visible
                st.rerun()
        with col_badge:
            total_fires_badge = int(filtered_df["total_incendios"].sum()) if not filtered_df.empty else 0
            badge_dot = "🟢" if is_live_api else "🟠"
            st.markdown(
                f"<div style='background:rgba(17,21,30,0.88);border:1px solid rgba(255,255,255,0.15);"
                f"border-radius:10px;padding:7px 6px;text-align:center;color:#fbbf24;font-size:12px;font-weight:700;'>"
                f"{badge_dot} {total_fires_badge:,}</div>",
                unsafe_allow_html=True,
            )

        if st.session_state.dock_visible:
            # 1) Tarjeta compacta del menú de filtros activos
            with st.expander(f"🔎 Filtro activo: {st.session_state.active_pais} · {periodo_txt}", expanded=False):
                col_pais_lbl, col_date_btn = st.columns([0.48, 0.52], vertical_alignment="center")
                with col_pais_lbl:
                    st.markdown(
                        f"<div style='font-size:13px;color:#cbd5e1;'><b>País:</b> {st.session_state.active_pais}</div>",
                        unsafe_allow_html=True,
                    )
                with col_date_btn:
                    with st.container(key="btn_quick_date_modal"):
                        if st.button(
                            f"📅 {periodo_txt}",
                            use_container_width=True,
                            help="Haz clic para cambiar únicamente la fecha o el periodo",
                            key="trigger_quick_date_dialog",
                        ):
                            open_quick_date_dialog(anios_api)

                numeric_tags = []
                if st.session_state.frp_min is not None or st.session_state.frp_max is not None:
                    f_lo = f"≥{st.session_state.frp_min:g}" if st.session_state.frp_min is not None else ""
                    f_hi = f"≤{st.session_state.frp_max:g}" if st.session_state.frp_max is not None else ""
                    numeric_tags.append(f"**FRP:** {' '.join(x for x in (f_lo, f_hi) if x)} MW")
                if st.session_state.brillo_min is not None or st.session_state.brillo_max is not None:
                    b_lo = f"≥{st.session_state.brillo_min:g}" if st.session_state.brillo_min is not None else ""
                    b_hi = f"≤{st.session_state.brillo_max:g}" if st.session_state.brillo_max is not None else ""
                    numeric_tags.append(f"**Brillo:** {' '.join(x for x in (b_lo, b_hi) if x)} K")
                if has_active_2d_drawing:
                    numeric_tags.append("**Zona 2D:** Dibujo activo")

                extra_line = f"\n\n{' · '.join(numeric_tags)}" if numeric_tags else ""
                st.caption(
                    f"**Satélites:** {', '.join(selected_sat)} · **Sensores:** {', '.join(selected_inst)}\n\n"
                    f"**Riesgo:** {', '.join(selected_riesgo)}{extra_line}"
                )
                if st.button("✏️ Abrir página de Filtros y Buscador", use_container_width=True, key="btn_go_filters"):
                    st.session_state.active_page = "🔍 Filtros y Buscador"
                    st.rerun()

            # 2) Módulo de Descarga justo debajo del menú de filtros (compatible con Servidor Externo)
            with st.expander(f"📥 Descargar Datos Filtrados ({len(filtered_df):,} celdas)", expanded=False):
                if not filtered_df.empty:
                    total_exp_fires = int(filtered_df["total_incendios"].sum())
                    src_txt = "servidor backend" if is_live_api else "respaldo local"
                    st.caption(
                        f"Exporta **{len(filtered_df):,}** celdas (**{total_exp_fires:,}** focos) desde **{src_txt}** "
                        f"para **{st.session_state.active_pais}** ({periodo_txt}):"
                    )
                    dl_c1, dl_c2, dl_c3 = st.columns(3)
                    with dl_c1:
                        st.download_button(
                            label="📄 CSV",
                            data=csv_payload,
                            file_name=fname_csv,
                            mime="text/csv",
                            use_container_width=True,
                            key="dock_dl_csv",
                        )
                    with dl_c2:
                        st.download_button(
                            label="📊 Excel",
                            data=excel_payload,
                            file_name=fname_excel,
                            mime=excel_mime,
                            use_container_width=True,
                            key="dock_dl_xlsx",
                        )
                    with dl_c3:
                        st.download_button(
                            label="🌍 GeoJSON",
                            data=geojson_payload,
                            file_name=fname_geojson,
                            mime="application/geo+json",
                            use_container_width=True,
                            key="dock_dl_geojson",
                        )
                else:
                    st.info("Sin registros filtrados para descargar.")

            # 3) Herramienta de Selección por Dibujo en 2D
            draw_badge = " · Activa" if has_active_2d_drawing else ""
            with st.expander(f"🖌️ Selección por Dibujo (2D){draw_badge}", expanded=has_active_2d_drawing):
                if st.session_state.projection_mode != "Plano (2D)":
                    st.caption("La selección por dibujo funciona sobre la vista **Plano (2D)**.")
                    if st.button("🗺️ Cambiar a Plano (2D) para dibujar", use_container_width=True, key="btn_switch_2d_draw"):
                        st.session_state.projection_mode = "Plano (2D)"
                        st.rerun()
                else:
                    if st.button(
                        "🖌️ Abrir pincel / lazo de dibujo 2D",
                        type="primary" if not has_active_2d_drawing else "secondary",
                        use_container_width=True,
                        key="btn_open_2d_brush_modal",
                    ):
                        st.session_state.draft_cleared = False
                        open_2d_drawing_dialog(base_filtered_df)

                    if st.session_state.drawn_polygon_2d is not None and len(st.session_state.drawn_polygon_2d) >= 3:
                        c_en, c_del = st.columns([0.62, 0.38])
                        with c_en:
                            en_val = st.checkbox(
                                "Filtrar por dibujo",
                                value=st.session_state.drawn_filter_enabled,
                                key="w_drawn_filter_en",
                            )
                            if en_val != st.session_state.drawn_filter_enabled:
                                st.session_state.drawn_filter_enabled = en_val
                                st.rerun()
                        with c_del:
                            if st.button("🗑️ Borrar", use_container_width=True, key="btn_clear_drawn_dock"):
                                st.session_state.drawn_polygon_2d = None
                                st.session_state.draft_polygon_2d = None
                                st.session_state.draft_vertices_2d = []
                                st.rerun()

            # 4) Controles visuales de Escala Térmica y Zoom Exclusivo / Multiescala
            with st.expander("🗺️ Escala Térmica y Zoom Exclusivo", expanded=True):
                palette_options = list(THERMAL_PALETTES.keys())
                sel_pal = st.selectbox(
                    "Paleta Térmica Continua:",
                    options=palette_options,
                    index=palette_options.index(st.session_state.palette_name),
                    key="w_palette_select",
                )
                st.session_state.palette_name = sel_pal

                sel_smooth = st.slider(
                    "Aproximación de color entre vecinos (%):",
                    min_value=0,
                    max_value=60,
                    value=st.session_state.smooth_pct,
                    step=5,
                    key="w_smooth_slider",
                )
                st.session_state.smooth_pct = sel_smooth

                sel_tier = st.select_slider(
                    "Nivel de Zoom / Escala Hexagonal (1 tamaño a la vez):",
                    options=[1, 2, 3, 4, 5],
                    value=st.session_state.zoom_tier_num,
                    format_func=lambda k: ZOOM_TIERS[k]["short"],
                    disabled=st.session_state.show_all_hex,
                    key="w_zoom_tier_slider",
                )
                st.session_state.zoom_tier_num = sel_tier

                sel_multi = st.checkbox(
                    "Mostrar todos los tamaños de hexágonos a la vez (Multiescala)",
                    value=st.session_state.show_all_hex,
                    help="Superpone simultáneamente los 5 niveles hexagonales (Macro ~300 km hasta Sensor 1:1).",
                    key="w_show_all_hex_cb",
                )
                if sel_multi != st.session_state.show_all_hex:
                    st.session_state.show_all_hex = sel_multi
                    st.rerun()

                sel_sync_cam = st.checkbox(
                    "Ajustar zoom de cámara automáticamente al nivel",
                    value=st.session_state.sync_camera_zoom,
                    key="w_sync_camera_cb",
                )
                st.session_state.sync_camera_zoom = sel_sync_cam

                sel_extrude = st.checkbox(
                    "Activar profundidad del terreno (Extrusión 3D)",
                    value=st.session_state.terrain_depth_3d,
                    help="Activa el relieve 3D por cantidad de incendios. Desactivado por defecto.",
                    key="w_extrude_cb",
                )
                st.session_state.terrain_depth_3d = sel_extrude

                proj_options = ["Plano (2D)", "Esfera 3D (Globo)"]
                sel_proj = st.radio(
                    "Proyección Cartográfica:",
                    options=proj_options,
                    index=proj_options.index(st.session_state.projection_mode),
                    key="w_proj_radio",
                )
                if sel_proj != st.session_state.projection_mode:
                    st.session_state.projection_mode = sel_proj
                    st.rerun()

                if st.session_state.projection_mode == "Esfera 3D (Globo)":
                    st.session_state.show_world_map_3d = st.checkbox(
                        "Mapamundi y Fronteras (3D)", value=st.session_state.show_world_map_3d, key="w_world_3d"
                    )
                    st.session_state.show_graticule_3d = st.checkbox(
                        "Paralelos y Meridianos (3D)", value=st.session_state.show_graticule_3d, key="w_grat_3d"
                    )

                st.session_state.show_base_grid = st.checkbox(
                    "Malla Geodésica de Referencia",
                    value=st.session_state.show_base_grid,
                    key="w_base_grid",
                )


# ==============================================================================
# PÁGINA A: INTERFAZ DEDICADA DE FILTROS Y BUSCADOR INTELIGENTE
# ==============================================================================
if not is_map_page:
    st.markdown("## 🔍 Buscador Inteligente y Filtros de Incendios")
    st.caption(
        "Escribe consultas combinadas en lenguaje natural (por ejemplo: **«Colombia agosto 2025 Terra FRP mayor a 50»**, "
        "**«Brasil VIIRS brillo > 350 15 de agosto»** o **«Argentina FRP entre 30 y 120»**) o ajusta manualmente cada parámetro."
    )

    with st.container(key="panel_buscador_inteligente"):
        col_q, col_btn_search, col_btn_reset = st.columns([0.66, 0.18, 0.16])
        with col_q:
            query_input = st.text_input(
                "Buscador inteligente multi-filtro:",
                value=st.session_state.smart_search_query,
                placeholder="Ej: Colombia agosto 2025 Terra FRP mayor a 50 brillo > 340...",
                key="w_smart_search_input",
            )
        with col_btn_search:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🔎 Reconocer y filtrar", type="primary", use_container_width=True, key="btn_run_smart_search"):
                st.session_state.smart_search_query = query_input
                parse_and_apply_smart_search(
                    query_input,
                    available_paises=paises_api,
                    available_anios=anios_api,
                    available_sats=sats_disponibles,
                    available_insts=insts_disponibles,
                )
                st.rerun()
        with col_btn_reset:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🧹 Limpiar todo", use_container_width=True, key="btn_clear_all_filters"):
                st.session_state.smart_search_query = ""
                st.session_state.last_parsed_entities = []
                st.session_state.active_pais = "Todos"
                st.session_state.active_riesgo = None
                st.session_state.active_sat = None
                st.session_state.active_inst = None
                st.session_state.frp_min = None
                st.session_state.frp_max = None
                st.session_state.brillo_min = None
                st.session_state.brillo_max = None
                st.session_state.drawn_polygon_2d = None
                st.session_state.draft_polygon_2d = None
                st.session_state.draft_vertices_2d = []
                st.session_state.temporal_mode = "mes_completo"
                st.rerun()

        if query_input != st.session_state.smart_search_query:
            st.session_state.smart_search_query = query_input
            parse_and_apply_smart_search(
                query_input,
                available_paises=paises_api,
                available_anios=anios_api,
                available_sats=sats_disponibles,
                available_insts=insts_disponibles,
            )
            st.rerun()

        if st.session_state.last_parsed_entities:
            badges_html = "".join(
                f"<span style='display:inline-block;margin:4px 6px 2px 0;padding:4px 10px;border-radius:999px;"
                f"background:rgba(255,255,255,0.07);border:1px solid {item['color']};color:{item['color']};"
                f"font-size:12px;font-weight:600;'>{item['tipo']}: {item['valor']}</span>"
                for item in st.session_state.last_parsed_entities
            )
            st.markdown(
                f"<div style='margin-top:6px;'><span style='font-size:12px;color:#94a3b8;margin-right:6px;'>"
                f"Entidades reconocidas:</span>{badges_html}</div>",
                unsafe_allow_html=True,
            )

    with st.container(key="panel_filtros_manuales"):
        col_left_filters, col_right_cal = st.columns([0.55, 0.45], gap="large")

        with col_left_filters:
            st.markdown("### 🛰️ Filtros Geográficos, Sensor y Umbrales")
            new_pais = st.selectbox(
                "País (API LATAM):",
                options=paises_api,
                index=paises_api.index(st.session_state.active_pais),
                key="w_page_pais_select",
            )
            if new_pais != st.session_state.active_pais:
                st.session_state.active_pais = new_pais
                st.rerun()

            new_riesgo = st.multiselect(
                "Nivel de Riesgo:",
                options=riesgos_disponibles,
                default=selected_riesgo,
                key="w_page_riesgo_multi",
            )
            if new_riesgo != selected_riesgo:
                st.session_state.active_riesgo = new_riesgo if new_riesgo else riesgos_disponibles
                st.rerun()

            c_sat, c_inst = st.columns(2)
            with c_sat:
                new_sat = st.multiselect(
                    "Satélite:",
                    options=sats_disponibles,
                    default=selected_sat,
                    key="w_page_sat_multi",
                )
                if new_sat != selected_sat:
                    st.session_state.active_sat = new_sat if new_sat else sats_disponibles
                    st.rerun()
            with c_inst:
                new_inst = st.multiselect(
                    "Instrumento / Sensor:",
                    options=insts_disponibles,
                    default=selected_inst,
                    key="w_page_inst_multi",
                )
                if new_inst != selected_inst:
                    st.session_state.active_inst = new_inst if new_inst else insts_disponibles
                    st.rerun()

            c_frp, c_brillo = st.columns(2)
            with c_frp:
                frp_min_in = st.number_input(
                    "FRP Mínimo (MW):",
                    min_value=0.0,
                    max_value=5000.0,
                    value=float(st.session_state.frp_min) if st.session_state.frp_min is not None else 0.0,
                    step=5.0,
                    key="w_page_frp_min",
                )
                if frp_min_in > 0.0:
                    st.session_state.frp_min = frp_min_in
                elif st.session_state.frp_min is not None and frp_min_in == 0.0:
                    st.session_state.frp_min = None

            with c_brillo:
                brillo_min_in = st.number_input(
                    "Brillo Mínimo (K):",
                    min_value=0.0,
                    max_value=600.0,
                    value=float(st.session_state.brillo_min) if st.session_state.brillo_min is not None else 0.0,
                    step=5.0,
                    key="w_page_brillo_min",
                )
                if brillo_min_in > 0.0:
                    st.session_state.brillo_min = brillo_min_in
                elif st.session_state.brillo_min is not None and brillo_min_in == 0.0:
                    st.session_state.brillo_min = None

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            c_go_map, c_draw_2d = st.columns([0.56, 0.44])
            with c_go_map:
                if st.button("✅ Aplicar filtros y ver en el Grid 2D/3D", type="primary", use_container_width=True):
                    st.session_state.active_page = "🗺️ Grid de Incendios"
                    st.rerun()
            with c_draw_2d:
                if st.button("🖌️ Selección por Dibujo 2D", use_container_width=True, key="btn_page_open_2d_draw"):
                    st.session_state.draft_cleared = False
                    open_2d_drawing_dialog(base_filtered_df)

        with col_right_cal:
            st.markdown("### 📅 Calendario de Mediciones")
            c_mes, c_anio = st.columns([0.58, 0.42])
            with c_mes:
                new_mes = st.selectbox(
                    "Mes:",
                    options=list(range(1, 13)),
                    index=st.session_state.cal_mes - 1,
                    format_func=lambda m: MESES_NOMBRES[m],
                    key="w_page_cal_mes",
                )
                if new_mes != st.session_state.cal_mes:
                    st.session_state.cal_mes = new_mes
                    st.rerun()
            with c_anio:
                new_anio = st.selectbox(
                    "Año:",
                    options=anios_api,
                    index=anios_api.index(st.session_state.cal_anio),
                    key="w_page_cal_anio",
                )
                if new_anio != st.session_state.cal_anio:
                    st.session_state.cal_anio = new_anio
                    st.rerun()

            cal_obj = calendar.Calendar(firstweekday=0)
            semanas_mes = cal_obj.monthdatescalendar(st.session_state.cal_anio, st.session_state.cal_mes)

            with st.container(key="calendario_card"):
                st.markdown(
                    """
                    <div style="display:grid;grid-template-columns:repeat(7,1fr);text-align:center;
                                font-weight:700;font-size:13px;color:#f8fafc;padding:2px 0 8px 0;">
                        <span>L</span><span>M</span><span>X</span><span>J</span><span>V</span><span>S</span><span>D</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for w_idx, semana in enumerate(semanas_mes):
                    day_cols = st.columns(7)
                    for d_idx, day_date in enumerate(semana):
                        es_mes_actual = day_date.month == st.session_state.cal_mes
                        es_dia_seleccionado = (
                            es_mes_actual
                            and st.session_state.temporal_mode == "fecha_exacta"
                            and day_date.day == st.session_state.cal_dia
                        )
                        with day_cols[d_idx]:
                            if not es_mes_actual:
                                st.button(
                                    f"{day_date.day}",
                                    key=f"p_cal_out_{w_idx}_{d_idx}_{day_date.isoformat()}",
                                    disabled=True,
                                    use_container_width=True,
                                )
                            else:
                                if st.button(
                                    f"{day_date.day}",
                                    key=f"p_cal_in_{st.session_state.cal_anio}_{st.session_state.cal_mes}_{day_date.day}",
                                    type="primary" if es_dia_seleccionado else "secondary",
                                    use_container_width=True,
                                ):
                                    st.session_state.cal_dia = day_date.day
                                    st.session_state.temporal_mode = "fecha_exacta"
                                    st.rerun()

            col_opt_mes, col_opt_anio = st.columns(2)
            with col_opt_mes:
                if st.button(
                    "📅 Mes completo",
                    type="primary" if st.session_state.temporal_mode == "mes_completo" else "secondary",
                    use_container_width=True,
                    key="p_btn_mes_completo",
                ):
                    st.session_state.temporal_mode = "mes_completo"
                    st.rerun()
            with col_opt_anio:
                if st.button(
                    "🗓️ Año completo",
                    type="primary" if st.session_state.temporal_mode == "anio_completo" else "secondary",
                    use_container_width=True,
                    key="p_btn_anio_completo",
                ):
                    st.session_state.temporal_mode = "anio_completo"
                    st.rerun()

            if st.session_state.temporal_mode == "fecha_exacta":
                st.caption(f"Filtro activo: Fecha exacta **{fecha_iso_activa}**")
            elif st.session_state.temporal_mode == "mes_completo":
                st.caption(
                    f"Filtro activo: Mes completo **{MESES_NOMBRES[st.session_state.cal_mes]} {st.session_state.cal_anio}**"
                )
            else:
                st.caption(f"Filtro activo: Año completo **{st.session_state.cal_anio}** (12 meses)")

    with st.container(key="panel_exportacion_filtros"):
        col_exp_info, col_exp_btns = st.columns([0.46, 0.54], vertical_alignment="center")
        with col_exp_info:
            st.markdown("### 📥 Exportar Registros Filtrados")
            if not filtered_df.empty:
                total_exp_fires = int(filtered_df["total_incendios"].sum())
                zone_note = " · **Con zona dibujada 2D**" if has_active_2d_drawing else ""
                st.caption(
                    f"Listos para descarga: **{len(filtered_df):,}** celdas activas (**{total_exp_fires:,}** focos) · "
                    f"Región: **{st.session_state.active_pais}** · Periodo: **{periodo_txt}**{zone_note}"
                )
            else:
                st.caption("No hay registros activos con la combinación actual de filtros.")

        with col_exp_btns:
            b_csv, b_xlsx, b_geo = st.columns(3)
            with b_csv:
                st.download_button(
                    label="📄 CSV (.csv)",
                    data=csv_payload,
                    file_name=fname_csv,
                    mime="text/csv",
                    disabled=filtered_df.empty,
                    use_container_width=True,
                    key="page_dl_csv",
                )
            with b_xlsx:
                st.download_button(
                    label=f"📊 Excel (.{excel_ext})",
                    data=excel_payload,
                    file_name=fname_excel,
                    mime=excel_mime,
                    disabled=filtered_df.empty,
                    use_container_width=True,
                    key="page_dl_xlsx",
                )
            with b_geo:
                st.download_button(
                    label="🌍 GeoJSON",
                    data=geojson_payload,
                    file_name=fname_geojson,
                    mime="application/geo+json",
                    disabled=filtered_df.empty,
                    use_container_width=True,
                    key="page_dl_geojson",
                )

    with st.container(key="panel_preview_filtros"):
        st.markdown("### 📊 Vista Previa de Resultados Filtrados")
        if not filtered_df.empty:
            preview_clusters = build_clustered_hex_layer(
                filtered_df,
                radius_deg=0.50,
                tier_label="Regional",
                palette_name=st.session_state.palette_name,
                smooth_weight=st.session_state.smooth_pct / 100.0,
                is_globe=False,
            )
            total_inc = int(filtered_df["total_incendios"].sum())
            weights = filtered_df["total_incendios"].to_numpy(dtype=np.float64)
            brillo_pond = np.average(filtered_df["brillo_promedio"], weights=weights)
            frp_pond = np.average(filtered_df["frp_promedio"], weights=weights)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Focos / Incendios", f"{total_inc:,}")
            k2.metric("Celdas Activas", f"{len(filtered_df):,}")
            k3.metric("Brillo Ponderado", f"{brillo_pond:.1f} K")
            k4.metric("FRP Ponderado", f"{frp_pond:.1f} MW")

            col_chart, col_tbl = st.columns([0.45, 0.55], gap="large")
            with col_chart:
                st.caption("Distribución Mensual de Incendios")
                mensual = (
                    df_grid[
                        (df_grid["nivel_riesgo"].isin(selected_riesgo))
                        & (df_grid["satelite"].isin(selected_sat))
                        & (df_grid["instrumento"].isin(selected_inst))
                    ]
                    .groupby("mes")["total_incendios"]
                    .sum()
                    .reindex(range(1, 13), fill_value=0)
                )
                st.bar_chart(mensual, height=220)

            with col_tbl:
                st.caption("Principales Clústeres Térmicos Regionales")
                st.dataframe(
                    preview_clusters[
                        ["pais", "lat_grid", "lon_grid", "total_incendios", "indice_termico", "brillo_promedio"]
                    ]
                    .sort_values("total_incendios", ascending=False)
                    .head(20),
                    use_container_width=True,
                    height=220,
                )
        else:
            st.warning("No se encontraron registros para la combinación actual de filtros o búsqueda.")


# ==============================================================================
# PÁGINA B: MAPA PRINCIPAL (GRID DE INCENDIOS 2D / ESFERA 3D)
# ==============================================================================
else:
    smooth_weight = st.session_state.smooth_pct / 100.0
    zoom_tier_num = st.session_state.zoom_tier_num
    active_tier = ZOOM_TIERS[zoom_tier_num]
    show_all_hex = st.session_state.show_all_hex
    sync_camera_zoom = st.session_state.sync_camera_zoom
    terrain_depth_3d = st.session_state.terrain_depth_3d
    is_globe = st.session_state.projection_mode == "Esfera 3D (Globo)"

    if not filtered_df.empty:
        center_lat = float(filtered_df["lat_grid"].mean())
        center_lon = float(filtered_df["lon_grid"].mean())
    elif has_active_2d_drawing:
        poly_arr = np.asarray(st.session_state.drawn_polygon_2d, dtype=np.float64)
        center_lon = float(poly_arr[:, 0].mean())
        center_lat = float(poly_arr[:, 1].mean())
    elif not df_grid.empty:
        center_lat = float(df_grid["lat_grid"].mean())
        center_lon = float(df_grid["lon_grid"].mean())
    else:
        center_lat, center_lon = -15.0, -60.0

    if show_all_hex:
        base_zoom = 3.8 if not is_globe else 2.2
    elif sync_camera_zoom:
        base_zoom = active_tier["camera_zoom"] if not is_globe else max(2.0, active_tier["camera_zoom"] - 0.8)
    else:
        base_zoom = 2.9 if not is_globe else 2.1

    active_layers = []

    if is_globe:
        ocean_df, graticule_df = build_globe_sphere_and_graticule()
        active_layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=ocean_df,
                get_polygon="polygon",
                filled=True,
                stroked=False,
                get_fill_color=[11, 19, 34, 255],
                pickable=False,
            )
        )

        if st.session_state.show_world_map_3d:
            world_geojson = load_world_map_geojson()
            if world_geojson:
                active_layers.append(
                    pdk.Layer(
                        "GeoJsonLayer",
                        data=world_geojson,
                        stroked=True,
                        filled=True,
                        get_fill_color=[27, 38, 56, 255],
                        get_line_color=[96, 122, 158, 210],
                        line_width_min_pixels=1,
                        pickable=False,
                    )
                )

        if st.session_state.show_graticule_3d:
            active_layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=graticule_df,
                    get_path="path",
                    get_color="color",
                    width_min_pixels=1,
                    pickable=False,
                )
            )

    if st.session_state.show_base_grid:
        if show_all_hex:
            normal_grid_df = build_regional_or_globe_geodesic_grid(
                radius_deg=1.35,
                is_globe=is_globe,
                globe_altitude_m=1200.0,
            )
            active_layers.append(
                pdk.Layer(
                    "PolygonLayer",
                    id="geodesic_grid_normal_multiscale",
                    data=normal_grid_df,
                    get_polygon="polygon",
                    filled=False,
                    stroked=True,
                    get_line_color=[0, 210, 255, 85],
                    line_width_min_pixels=1,
                    pickable=False,
                )
            )
        else:
            coverage_mode = active_tier["grid_coverage"]

            if coverage_mode != "globe":
                global_envelope_df = build_regional_or_globe_geodesic_grid(
                    radius_deg=1.35,
                    is_globe=is_globe,
                    globe_altitude_m=600.0,
                )
                active_layers.append(
                    pdk.Layer(
                        "PolygonLayer",
                        id="geodesic_grid_global_envelope",
                        data=global_envelope_df,
                        get_polygon="polygon",
                        filled=False,
                        stroked=True,
                        get_line_color=[0, 210, 255, 38],
                        line_width_min_pixels=1,
                        pickable=False,
                    )
                )

            data_lons_arr = filtered_df["lon_grid"].to_numpy(dtype=np.float64) if not filtered_df.empty else np.array([])
            data_lats_arr = filtered_df["lat_grid"].to_numpy(dtype=np.float64) if not filtered_df.empty else np.array([])

            matched_grid_df = build_matched_geodesic_grid(
                radius_deg=active_tier["grid_radius"],
                coverage_mode=coverage_mode,
                data_lons=data_lons_arr,
                data_lats=data_lats_arr,
                center_lon=center_lon,
                center_lat=center_lat,
                block_size=active_tier["block_size"],
                is_globe=is_globe,
                globe_altitude_m=1200.0,
            )
            active_layers.append(
                pdk.Layer(
                    "PolygonLayer",
                    id=f"geodesic_grid_tier_{zoom_tier_num}",
                    data=matched_grid_df,
                    get_polygon="polygon",
                    filled=False,
                    stroked=True,
                    get_line_color=[0, 210, 255, 95],
                    line_width_min_pixels=1,
                    pickable=False,
                )
            )

    def _make_hex_polygon_layer(layer_id: str, data_df: pd.DataFrame) -> pdk.Layer:
        layer_kwargs: Dict[str, Any] = {
            "id": layer_id,
            "data": data_df,
            "get_polygon": "polygon",
            "get_fill_color": "rgba",
            "get_line_color": "line_rgba",
            "line_width_min_pixels": 1,
            "filled": True,
            "stroked": True,
            "extruded": bool(terrain_depth_3d),
            "pickable": True,
            "auto_highlight": True,
        }
        if terrain_depth_3d:
            layer_kwargs["get_elevation"] = "elevation"
            if not is_globe:
                layer_kwargs["wireframe"] = True
        return pdk.Layer("PolygonLayer", **layer_kwargs)

    if not filtered_df.empty:
        if show_all_hex:
            for t_id in (1, 2, 3, 4, 5):
                tier_cfg = ZOOM_TIERS[t_id]
                df_tier = build_clustered_hex_layer(
                    filtered_df,
                    radius_deg=tier_cfg["radius_deg"],
                    tier_label=tier_cfg["label"],
                    palette_name=st.session_state.palette_name,
                    smooth_weight=smooth_weight,
                    is_globe=is_globe,
                    is_sensor_1to1=(t_id == 5),
                    globe_altitude_m=1210.0 + t_id * 5.0,
                )
                active_layers.append(_make_hex_polygon_layer(f"thermal_hex_multi_tier_{t_id}", df_tier))
        else:
            df_active_hex = build_clustered_hex_layer(
                filtered_df,
                radius_deg=active_tier["radius_deg"],
                tier_label=active_tier["label"],
                palette_name=st.session_state.palette_name,
                smooth_weight=smooth_weight,
                is_globe=is_globe,
                is_sensor_1to1=(zoom_tier_num == 5),
                globe_altitude_m=1210.0,
            )
            active_layers.append(_make_hex_polygon_layer(f"thermal_hex_single_tier_{zoom_tier_num}", df_active_hex))

    # Capa visual del polígono encerrado por dibujo en 2D
    if not is_globe and st.session_state.drawn_polygon_2d and len(st.session_state.drawn_polygon_2d) >= 3:
        drawn_ring = list(st.session_state.drawn_polygon_2d)
        if drawn_ring[0] != drawn_ring[-1]:
            drawn_ring.append(drawn_ring[0])
        drawn_zone_df = pd.DataFrame([{"polygon": drawn_ring}])
        active_layers.append(
            pdk.Layer(
                "PolygonLayer",
                id="drawn_2d_selection_zone",
                data=drawn_zone_df,
                get_polygon="polygon",
                filled=True,
                stroked=True,
                get_fill_color=[56, 189, 248, 36] if st.session_state.drawn_filter_enabled else [148, 163, 184, 18],
                get_line_color=[56, 189, 248, 240] if st.session_state.drawn_filter_enabled else [148, 163, 184, 140],
                line_width_min_pixels=2,
                pickable=False,
            )
        )

    pitch_angle = 42 if (terrain_depth_3d and not is_globe) else 0

    if is_globe:
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=base_zoom,
            min_zoom=0.8,
            max_zoom=16.0,
        )
        map_view = pdk.View(
            type="GlobeView",
            controller=True,
            clear_color=[5 / 255, 8 / 255, 15 / 255, 1.0],
        )
    else:
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=base_zoom,
            min_zoom=1.2,
            max_zoom=18.0,
            pitch=pitch_angle,
            bearing=0,
        )
        map_view = pdk.View(type="MapView", controller=True, repeat=False)

    deck = pdk.Deck(
        layers=active_layers,
        initial_view_state=view_state,
        views=[map_view],
        tooltip={
            "text": (
                "{escala_vista} · {pais}\n"
                "Intensidad Térmica: {indice_termico}% ({nivel_riesgo})\n"
                "Incendios Agrupados: {total_incendios} ({celdas_agrupadas} celdas)\n"
                "Brillo Prom.: {brillo_promedio} K | FRP Prom.: {frp_promedio} MW\n"
                "Cobertura: {resolucion_m} ({area_km2} km²) · Sensor: {instrumento} ({satelite})"
            )
        },
        map_style=None if is_globe else "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    )

    st.pydeck_chart(deck, use_container_width=True, height=900)