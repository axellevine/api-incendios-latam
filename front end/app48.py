import calendar
from datetime import date, datetime
import inspect
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import unicodedata
from urllib.parse import urlparse
from xml.sax.saxutils import escape as xml_escape

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

# 1. Configuración de página en modo ancho total
st.set_page_config(
    page_title="NASA Hot Spots · LATAM Hexagonal Thermal Map",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)
BASE_DIR = Path(__file__).resolve().parent
WORLD_GEOJSON_CACHE = BASE_DIR / "world_countries_110m.geojson"
DEFAULT_API_URL = os.getenv("INCENDIOS_API_URL", "http://127.0.0.1:8000/api/incendios-grid")
# Visible por defecto; puede ocultarse en despliegue público con ?admin=0 o SHOW_SERVER_PANEL=0
SHOW_SERVER_PANEL = (
    os.getenv("SHOW_SERVER_PANEL", "1") != "0"
    and st.query_params.get("admin", "1") != "0"
)
PAGE_MAP = "map"
PAGE_CAL = "analysis"
PAGE_FILT = "filters"
PAGES = [PAGE_MAP, PAGE_CAL, PAGE_FILT]

def _stretch(fn: Any) -> Dict[str, Any]:
    """Compatibilidad: `width="stretch"` en Streamlit recientes, `use_container_width` en versiones antiguas."""
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}

BTN = _stretch(st.button)
DL = _stretch(st.download_button)
DECK = _stretch(st.pydeck_chart)
DF = _stretch(st.dataframe)
ALT = _stretch(st.altair_chart)

EARTH_RADIUS_M = 6_371_008.8
METERS_PER_DEG_LAT = (np.pi / 180.0) * EARTH_RADIUS_M  # ~111,195 m/grado
SENSOR_1TO1_RADIUS_DEG = 0.00558
LATAM_LAT_MIN, LATAM_LAT_MAX = -56.5, 34.0
LATAM_LON_MIN, LATAM_LON_MAX = -120.0, -32.0

ZOOM_TIERS: Dict[int, Dict[str, Any]] = {
    1: {
        "short_en": "1 · Macro (~300 km)",
        "short_es": "1 · Macro (~300 km)",
        "label_en": "Tier 1 · Macro-Regional (~300 km)",
        "label_es": "Nivel 1 · Macro-Regional (~300 km)",
        "radius_deg": 1.35,
        "camera_zoom": 2.9,
    },
    2: {
        "short_en": "2 · Regional (~110 km)",
        "short_es": "2 · Regional (~110 km)",
        "label_en": "Tier 2 · Regional (~110 km)",
        "label_es": "Nivel 2 · Regional (~110 km)",
        "radius_deg": 0.50,
        "camera_zoom": 4.3,
    },
    3: {
        "short_en": "3 · Sub-Regional (~40 km)",
        "short_es": "3 · Sub-Regional (~40 km)",
        "label_en": "Tier 3 · Sub-Regional (~40 km)",
        "label_es": "Nivel 3 · Sub-Regional (~40 km)",
        "radius_deg": 0.18,
        "camera_zoom": 5.7,
    },
    4: {
        "short_en": "4 · Local (~13 km)",
        "short_es": "4 · Local (~13 km)",
        "label_en": "Tier 4 · Local / Municipal (~13 km)",
        "label_es": "Nivel 4 · Local / Municipal (~13 km)",
        "radius_deg": 0.06,
        "camera_zoom": 7.3,
    },
    5: {
        "short_en": "5 · Cell ~1 km",
        "short_es": "5 · Celda ~1 km",
        "label_en": "Tier 5 · Cell ~1 km",
        "label_es": "Nivel 5 · Celda ~1 km",
        "radius_deg": SENSOR_1TO1_RADIUS_DEG,
        "camera_zoom": 9.6,
    },
}

THERMAL_PALETTES: Dict[str, Dict[str, Any]] = {
    "inferno": {
        "name_en": "Inferno (colorblind-safe)",
        "name_es": "Inferno (apto para daltonismo)",
        "stops": np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        "r": [0, 87, 188, 249, 252],
        "g": [0, 16, 55, 142, 255],
        "b": [4, 110, 84, 9, 164],
    },
    "classic": {
        "name_en": "Classic Thermal (Green → Amber → Orange → Red)",
        "name_es": "Térmico Clásico (Verde → Ámbar → Naranja → Rojo)",
        "stops": np.array([0.0, 0.35, 0.68, 1.0]),
        "r": [46, 241, 230, 255],
        "g": [204, 196, 126, 28],
        "b": [113, 15, 34, 28],
    },
    "infrared": {
        "name_en": "Infrared Thermal (Amber → Crimson → Incandescent)",
        "name_es": "Térmico Infrarrojo (Ámbar → Carmesí → Incandescente)",
        "stops": np.array([0.0, 0.25, 0.52, 0.78, 1.0]),
        "r": [74, 158, 224, 249, 255],
        "g": [25, 42, 92, 160, 236],
        "b": [54, 43, 24, 34, 130],
    },
    "spectral": {
        "name_en": "Spectral Thermal (Indigo → Cyan → Gold → Red)",
        "name_es": "Térmico Espectral (Índigo → Cian → Dorado → Rojo)",
        "stops": np.array([0.0, 0.28, 0.58, 0.80, 1.0]),
        "r": [29, 14, 245, 249, 220],
        "g": [78, 165, 195, 105, 34],
        "b": [216, 233, 35, 22, 34],
    },
    "magma": {
        "name_en": "Satellite Magma (Violet → Coral → Light Amber)",
        "name_es": "Magma Satelital (Violeta → Coral → Ámbar Claro)",
        "stops": np.array([0.0, 0.33, 0.66, 0.85, 1.0]),
        "r": [59, 140, 222, 254, 252],
        "g": [15, 41, 73, 159, 244],
        "b": [112, 129, 104, 109, 175],
    },
}
MONTHS_EN = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}
MONTHS_ES = {
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
    "familia",
]
COLUMN_ALIASES: Dict[str, List[str]] = {
    "pais": ["pais", "country", "country_name", "region", "pais_nombre", "admin"],
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
# 2. Estado canónico persistente entre páginas e idiomas (predeterminado en inglés: "en")
DEFAULT_STATE: Dict[str, Any] = {
    "lang": "en",
    "active_page": PAGE_MAP,
    "dock_visible": True,
    "api_url": DEFAULT_API_URL,
    "api_limit": 120000,
    "allow_local_fallback": True,
    "strict_latam": True,
    "active_pais": "Todos",
    "known_paises": set(),
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
    "picked_lat_2d": None,
    "picked_lon_2d": None,
    "dialog_cam_lat": None,
    "dialog_cam_lon": None,
    "last_deck_selection_raw": "",
    "map_widget_nonce": 0,
    "last_draw_mode_2d": "circle",
    "auto_trace_on_click": True,
    "auto_uncross_polygon": True,
    "aoi_expander_open": False,
    "aoi_dialog_open": False,
    "palette_key": "inferno",
    "smooth_pct": 35,
    "zoom_tier_num": 1,
    "show_all_hex": False,
    "sync_camera_zoom": True,
    "terrain_depth_3d": False,
    "projection_mode": "2D",
    "show_world_map_3d": True,
    "show_graticule_3d": True,
    "show_base_grid": False,
    "w_adv_toggle": False,
    "search_nonce": 0,
    "cal_metric": "fires",
    "cal_series": "media",
    "cal_focus_year": None,
    "cal_range_w": None,
}

for _k, _v in DEFAULT_STATE.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.lang not in ("en", "es"):
    st.session_state.lang = "en"
if st.session_state.active_page not in PAGES:
    st.session_state.active_page = PAGE_MAP
if st.session_state.palette_key not in THERMAL_PALETTES:
    st.session_state.palette_key = "inferno"
if st.session_state.projection_mode not in ("2D", "3D"):
    st.session_state.projection_mode = "2D"
if st.session_state.last_draw_mode_2d not in ("circle", "polygon"):
    st.session_state.last_draw_mode_2d = "circle"
if st.session_state.cal_metric not in ("fires", "frp"):
    st.session_state.cal_metric = "fires"
if st.session_state.cal_series not in ("media", "modis", "viirs"):
    st.session_state.cal_series = "media"


def T(en: str, es: str) -> str:
    """Devuelve texto en inglés o español según el idioma activo."""
    return en if st.session_state.lang == "en" else es

def month_name(m: int) -> str:
    return (MONTHS_EN if st.session_state.lang == "en" else MONTHS_ES)[int(m)]

def country_label(c: str) -> str:
    return T("All", "Todos") if c == "Todos" else c

def risk_label(r: str) -> str:
    mapping = {
        "Alto": T("High", "Alto"),
        "Medio": T("Medium", "Medio"),
        "Bajo": T("Low", "Bajo"),
        "Sin clasificar": T("Unclassified", "Sin clasificar"),
    }
    return mapping.get(r, r)

is_map_page = st.session_state.active_page == PAGE_MAP
# 3. Estilos CSS dinámicos + Indicador flotante genérico de carga + Pestañas superiores
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
        padding: 86px 36px 48px 36px !important;
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

    @keyframes spinLoader {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}

    @keyframes pulseLoaderBorder {{
        0%, 100% {{ box-shadow: 0 14px 40px rgba(0, 0, 0, 0.72), 0 0 18px rgba(56, 189, 248, 0.25); }}
        50% {{ box-shadow: 0 14px 40px rgba(0, 0, 0, 0.78), 0 0 28px rgba(251, 146, 60, 0.35); }}
    }}

    /* Tarjeta flotante genérica de carga */
    .live-loading-card {{
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 100002;
        display: flex;
        align-items: center;
        gap: 14px;
        background: rgba(15, 23, 42, 0.96);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.55);
        border-radius: 14px;
        padding: 15px 24px;
        color: #f8fafc;
        pointer-events: none;
        min-width: 280px;
        max-width: 440px;
        animation: pulseLoaderBorder 1.6s ease-in-out infinite;
    }}

    .live-loading-spinner {{
        width: 22px;
        height: 22px;
        border: 3px solid rgba(148, 163, 184, 0.25);
        border-top-color: #38bdf8;
        border-right-color: #fb923c;
        border-radius: 50%;
        flex-shrink: 0;
        animation: spinLoader 0.8s linear infinite;
    }}

    .live-loading-title {{
        font-size: 14px;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.35;
    }}

    .live-loading-sub {{
        font-size: 12px;
        color: #94a3b8;
        margin-top: 2px;
        line-height: 1.3;
    }}

    /* Barra de desplazamiento inferior visible para gráficas temporales expandidas */
    div[data-testid="stAltairChart"],
    div[data-testid="stVegaLiteChart"] {{
        overflow-x: auto !important;
        overflow-y: hidden !important;
        padding-bottom: 8px !important;
        scrollbar-width: thin !important;
        scrollbar-color: #38bdf8 rgba(30, 41, 59, 0.75) !important;
    }}
    div[data-testid="stAltairChart"]::-webkit-scrollbar,
    div[data-testid="stVegaLiteChart"]::-webkit-scrollbar {{
        height: 10px !important;
    }}
    div[data-testid="stAltairChart"]::-webkit-scrollbar-track,
    div[data-testid="stVegaLiteChart"]::-webkit-scrollbar-track {{
        background: rgba(30, 41, 59, 0.75) !important;
        border-radius: 6px !important;
    }}
    div[data-testid="stAltairChart"]::-webkit-scrollbar-thumb,
    div[data-testid="stVegaLiteChart"]::-webkit-scrollbar-thumb {{
        background: #38bdf8 !important;
        border-radius: 6px !important;
    }}

    {page_layout_css}

    /* Dock Superior Izquierdo (Navegación tipo Pestañas + Controles del Mapa) */
    .st-key-dock_superior_izquierdo {{
        position: fixed !important;
        top: 16px !important;
        left: 16px !important;
        z-index: 9999 !important;
        width: 396px !important;
        max-height: calc(100vh - 32px) !important;
        overflow-y: auto !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
        padding-right: 4px !important;
    }}

    /* Estilizar las 3 opciones superiores como barra de pestañas */
    .st-key-dock_superior_izquierdo div[data-testid="stRadio"]:first-of-type div[role="radiogroup"] {{
        display: flex !important;
        flex-direction: row !important;
        width: 100% !important;
        gap: 4px !important;
        background: rgba(13, 18, 28, 0.94) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.48) !important;
    }}

    .st-key-dock_superior_izquierdo div[data-testid="stRadio"]:first-of-type div[role="radiogroup"] > label {{
        flex: 1 1 0% !important;
        margin: 0 !important;
        padding: 8px 6px !important;
        border-radius: 8px !important;
        border-bottom: 2px solid transparent !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        cursor: pointer !important;
        transition: all 0.18s ease !important;
        background: transparent !important;
    }}

    /* Ocultar el círculo de radio en la barra de navegación superior */
    .st-key-dock_superior_izquierdo div[data-testid="stRadio"]:first-of-type div[role="radiogroup"] > label > div:first-child {{
        display: none !important;
    }}

    .st-key-dock_superior_izquierdo div[data-testid="stRadio"]:first-of-type div[role="radiogroup"] > label p {{
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        margin: 0 !important;
        white-space: nowrap !important;
    }}

    .st-key-dock_superior_izquierdo div[data-testid="stRadio"]:first-of-type div[role="radiogroup"] > label:hover {{
        background: rgba(30, 41, 59, 0.72) !important;
    }}

    .st-key-dock_superior_izquierdo div[data-testid="stRadio"]:first-of-type div[role="radiogroup"] > label:hover p {{
        color: #e2e8f0 !important;
    }}

    /* Pestaña seleccionada */
    .st-key-dock_superior_izquierdo div[data-testid="stRadio"]:first-of-type div[role="radiogroup"] > label:has(input:checked) {{
        background: linear-gradient(180deg, rgba(56, 189, 248, 0.22) 0%, rgba(30, 41, 59, 0.96) 100%) !important;
        border-bottom: 2px solid #38bdf8 !important;
        box-shadow: 0 2px 10px rgba(56, 189, 248, 0.18) !important;
    }}

    .st-key-dock_superior_izquierdo div[data-testid="stRadio"]:first-of-type div[role="radiogroup"] > label:has(input:checked) p {{
        color: #f8fafc !important;
        font-weight: 700 !important;
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

    /* Pantalla completa exclusiva para el mapa principal */
    section.main .stDeckGlJsonChart,
    section.main div[data-testid="stDeckGlJsonChart"],
    [data-testid="stMain"] .stDeckGlJsonChart,
    [data-testid="stMain"] div[data-testid="stDeckGlJsonChart"] {{
        height: 100vh !important;
        width: 100vw !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        z-index: 1 !important;
    }}
    .map-legend {{
        position: fixed; right: 16px; bottom: 28px; z-index: 9999; width: 295px;
        background: rgba(17,21,30,0.90); border: 1px solid rgba(255,255,255,0.15); border-radius: 12px;
        padding: 10px 12px; color: #e5e7eb; font-size: 12px; line-height: 1.35;
        box-shadow: 0 8px 24px rgba(0,0,0,0.45);
    }}
    .map-legend .bar {{ height: 10px; border-radius: 6px; margin: 7px 0 2px 0; }}
    .map-legend .ends {{ display: flex; justify-content: space-between; color: #94a3b8; font-size: 11px; }}
    .map-legend .note {{ margin-top: 6px; color: #94a3b8; font-size: 11px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

loading_slot = st.empty()

def show_generic_loading() -> None:
    """Muestra un banner genérico flotante durante cualquier carga o procesamiento activo."""
    loading_slot.markdown(
        "<div class='live-loading-card'>"
        "<div class='live-loading-spinner'></div>"
        "<div>"
        f"<div class='live-loading-title'>{T('Loading...', 'Cargando...')}</div>"
        f"<div class='live-loading-sub'>{T('Processing data and updating view', 'Procesando datos y actualizando la vista')}</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

def clear_loading_status() -> None:
    """Oculta el banner genérico al completar el renderizado de la página."""
    loading_slot.empty()


show_generic_loading()


# 4. Carga del Mapamundi GeoJSON y Geometría Base para el Globo 3D
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


# 5. Cliente de datos robusto para Servidores Backend Externos
def _derive_base_api_url(grid_url: str) -> str:
    parsed = urlparse(grid_url.strip())
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return "http://127.0.0.1:8000"


def _build_api_headers() -> Dict[str, str]:
    return {"Cache-Control": "no-cache", "Accept": "application/json"}


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_api_catalogs_cached(grid_url: str) -> Dict[str, Any]:
    base_url = _derive_base_api_url(grid_url)
    resp = requests.get(
        f"{base_url}/api/filtros-disponibles",
        headers=_build_api_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) and data else {}


def fetch_api_catalogs(grid_url: str) -> Dict[str, Any]:
    """Consulta el catálogo del backend sin cachear fallos transitorios vacíos."""
    try:
        res = _fetch_api_catalogs_cached(grid_url)
        if not res:
            _fetch_api_catalogs_cached.clear()
        return res
    except Exception:
        return {}


def _extract_records_from_payload(payload: Any) -> List[Dict[str, Any]]:
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

def sensor_family(inst: Any, sat: Any) -> str:
    """Clasifica cada detección en la familia MODIS (1 km) o VIIRS (375 m)."""
    i = str(inst).upper()
    s = str(sat).upper()
    if "VIIRS" in i:
        return "VIIRS"
    if "MODIS" in i:
        return "MODIS"
    if any(t in s for t in ("TERRA", "AQUA")):
        return "MODIS"
    if any(t in s for t in ("SNPP", "NPP", "N20", "N21", "NOAA", "JPSS")):
        return "VIIRS"
    return "Otro"

def _apply_geo_bounds(df: pd.DataFrame, strict_latam_bounds: bool) -> pd.DataFrame:
    if df.empty:
        return df
    if strict_latam_bounds:
        mask = df["lat_grid"].between(LATAM_LAT_MIN, LATAM_LAT_MAX) & df["lon_grid"].between(LATAM_LON_MIN, LATAM_LON_MAX)
    else:
        mask = df["lat_grid"].between(-85.0, 85.0) & df["lon_grid"].between(-180.0, 180.0)
    return df[mask]

def _normalize_grid_dataframe(records: List[Dict[str, Any]], strict_latam_bounds: bool = True) -> pd.DataFrame:
    """Normaliza columnas SIN inventar datos: lo que falta queda vacío o la fila se descarta."""
    empty = pd.DataFrame(columns=EXPECTED_COLUMNS + ["familia"])
    if not records:
        return empty

    df = pd.DataFrame(records)
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical not in df.columns:
            for alias in aliases:
                if alias in lower_cols:
                    df[canonical] = df[lower_cols[alias]]
                    break

    count_provided = "total_incendios" in df.columns
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    fecha_dt = pd.to_datetime(df["fecha"], errors="coerce", format="mixed")
    df["fecha"] = fecha_dt.dt.strftime("%Y-%m-%d").fillna("")
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").fillna(fecha_dt.dt.year)
    df["mes"] = pd.to_numeric(df["mes"], errors="coerce").fillna(fecha_dt.dt.month)
    df["lat_grid"] = pd.to_numeric(df["lat_grid"], errors="coerce")
    df["lon_grid"] = pd.to_numeric(df["lon_grid"], errors="coerce")
    df["total_incendios"] = pd.to_numeric(df["total_incendios"], errors="coerce") if count_provided else 1.0
    df["frp_promedio"] = pd.to_numeric(df["frp_promedio"], errors="coerce").clip(lower=0.0).round(1)
    df["brillo_promedio"] = pd.to_numeric(df["brillo_promedio"], errors="coerce").clip(lower=0.0).round(1)
    df["pais"] = df["pais"].fillna("Sin dato").astype(str).str.strip()
    df["instrumento"] = df["instrumento"].fillna("Desconocido").astype(str).str.strip()
    df["satelite"] = df["satelite"].fillna("Desconocido").astype(str).str.strip()
    df["nivel_riesgo"] = df["nivel_riesgo"].fillna("Sin clasificar").astype(str)
    df["color"] = df["color"].fillna("#94A3B8").astype(str)

    df = df.dropna(subset=["lat_grid", "lon_grid", "anio", "mes", "total_incendios"])
    df = df[(df["total_incendios"] > 0) & df["mes"].between(1, 12)]
    df["anio"] = df["anio"].astype(int)
    df["mes"] = df["mes"].astype(int)
    df["total_incendios"] = df["total_incendios"].round().astype(int)

    if df.empty:
        return empty
    fam_map = {(i, s): sensor_family(i, s) for i, s in df[["instrumento", "satelite"]].drop_duplicates().itertuples(index=False)}
    df["familia"] = [fam_map[(i, s)] for i, s in zip(df["instrumento"], df["satelite"])]
    return _apply_geo_bounds(df, strict_latam_bounds).reset_index(drop=True)

def _local_signature() -> tuple:
    files: List[Path] = []
    data_dir = BASE_DIR / "datos_locales"
    if data_dir.is_dir():
        files += sorted(data_dir.glob("*.csv"))
    legacy = BASE_DIR / "modis_2025_Argentina.csv"
    if not legacy.exists():
        legacy = Path.home() / "Downloads" / "modis_2025_Argentina.csv"
    if legacy.exists() and legacy not in files:
        files.append(legacy)
    return tuple((str(p), p.stat().st_mtime) for p in files)


@st.cache_data(show_spinner=False)
def load_local_firms_grid(signature: tuple) -> pd.DataFrame:
    """Lee CSV de FIRMS (MODIS y/o VIIRS) y los agrega por DÍA, celda de 0.01° (~1 km), sensor y satélite."""
    frames = []
    for path_str, _mtime in signature:
        path = Path(path_str)
        try:
            raw = pd.read_csv(path)
        except Exception:
            continue
        raw.columns = [c.lower().strip() for c in raw.columns]
        if not {"latitude", "longitude", "acq_date"} <= set(raw.columns):
            continue
        nan_col = pd.Series(np.nan, index=raw.index)
        if "brightness" in raw.columns:
            bright = raw["brightness"]
        elif "bright_ti4" in raw.columns:
            bright = raw["bright_ti4"]
        else:
            bright = nan_col
        if "instrument" in raw.columns:
            inst = raw["instrument"].astype(str).str.strip()
        else:
            inst = pd.Series("VIIRS" if "bright_ti4" in raw.columns else "MODIS", index=raw.index)
        sat = raw["satellite"].astype(str).str.strip() if "satellite" in raw.columns else pd.Series("Desconocido", index=raw.index)
        pais = raw["country"].astype(str) if "country" in raw.columns else path.stem.split("_")[-1].capitalize()
        frames.append(
            pd.DataFrame(
                {
                    "fecha": pd.to_datetime(raw["acq_date"], errors="coerce", format="mixed"),
                    "lat_grid": raw["latitude"].round(2),
                    "lon_grid": raw["longitude"].round(2),
                    "instrumento": inst,
                    "satelite": sat,
                    "pais": pais,
                    "brillo": bright,
                    "frp": raw["frp"] if "frp" in raw.columns else nan_col,
                }
            ).dropna(subset=["fecha", "lat_grid", "lon_grid"])
        )
    if not frames:
        return pd.DataFrame(columns=EXPECTED_COLUMNS + ["familia"])

    allf = pd.concat(frames, ignore_index=True)
    g = allf.groupby(["fecha", "lat_grid", "lon_grid", "instrumento", "satelite", "pais"], as_index=False).agg(
        total_incendios=("frp", "size"),
        frp_promedio=("frp", "mean"),
        brillo_promedio=("brillo", "mean"),
    )
    g["fecha"] = g["fecha"].dt.strftime("%Y-%m-%d")
    fam = g["instrumento"].combine(g["satelite"], sensor_family)
    pct = g.groupby(fam)["frp_promedio"].rank(pct=True)
    g["nivel_riesgo"] = np.select([pct.isna(), pct >= 0.90, pct >= 0.60], ["Sin clasificar", "Alto", "Medio"], default="Bajo")
    g["color"] = g["nivel_riesgo"].map({"Alto": "#FF0000", "Medio": "#F39C12", "Bajo": "#2ECC71"}).fillna("#94A3B8")
    return _normalize_grid_dataframe(g.to_dict(orient="records"), strict_latam_bounds=False)

def local_catalogs() -> Dict[str, Any]:
    df = load_local_firms_grid(_local_signature())
    if df.empty:
        return {}
    return {
        "paises": sorted(df["pais"].unique().tolist()),
        "anios": sorted(int(a) for a in df["anio"].unique()),
        "satelites": sorted(df["satelite"].unique().tolist()),
        "instrumentos": sorted(df["instrumento"].unique().tolist()),
    }


@st.cache_data(ttl=15, show_spinner=False)
def fetch_incendios_grid(
    api_url: str,
    allow_local_fallback: bool,
    pais: Optional[str] = None,
    anio: Optional[int] = None,
    mes: Optional[int] = None,
    fecha: Optional[str] = None,
    strict_latam_bounds: bool = True,
    limit: int = 120000,
    lang: str = "en",
) -> tuple[pd.DataFrame, bool, str, bool]:
    """Devuelve (datos, es_api_en_vivo, estado, truncado_por_limite)."""
    is_en = lang == "en"
    months_lbl = MONTHS_EN if is_en else MONTHS_ES
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

    if fecha:
        suffix = f" · {fecha}"
    elif mes:
        suffix = f" · {months_lbl[mes]} {anio}"
    else:
        suffix = f" · {'Year' if is_en else 'Año'} {anio}"

    try:
        resp = requests.get(api_url.strip(), params=params, timeout=30, headers=_build_api_headers())
        resp.raise_for_status()
        records = _extract_records_from_payload(resp.json())
        df = _normalize_grid_dataframe(records, strict_latam_bounds=strict_latam_bounds)
        truncated = len(records) >= int(limit)
        status = (
            f"🟢 LIVE API ({len(df):,} cells{suffix})"
            if is_en
            else f"🟢 API EN VIVO ({len(df):,} celdas{suffix})"
        )
        dropped = len(records) - len(df)
        if dropped > 0:
            status += (
                f" · {dropped:,} records discarded (missing coords/date/count or out of bounds)"
                if is_en
                else f" · {dropped:,} registros descartados (sin coordenadas/fecha/conteo o fuera de rango)"
            )
        if truncated:
            status += (
                f" · ⚠️ response TRUNCATED at {int(limit):,} records: shorten period or raise limit"
                if is_en
                else f" · ⚠️ respuesta TRUNCADA en {int(limit):,} registros: reduce el periodo o sube el límite"
            )
        return df, True, status, truncated
    except Exception as api_err:
        empty = pd.DataFrame(columns=EXPECTED_COLUMNS + ["familia"])
        if not allow_local_fallback:
            return empty, False, f"🔴 API OFFLINE: {api_err}", False
        local = load_local_firms_grid(_local_signature())
        if local.empty:
            msg = (
                f"🔴 Offline and no local CSV found ({api_err})"
                if is_en
                else f"🔴 Sin conexión y sin CSV local ({api_err})"
            )
            return empty, False, msg, False
        df = _apply_geo_bounds(local, strict_latam_bounds)
        if pais and pais != "Todos":
            df = df[df["pais"].str.lower() == pais.lower()]
        if fecha:
            df = df[df["fecha"] == fecha]
        else:
            if anio:
                df = df[df["anio"] == int(anio)]
            if mes:
                df = df[df["mes"] == int(mes)]
        msg = (
            f"🟠 LOCAL MODE ({len(df):,} cells{suffix}) · API unavailable: {api_err}"
            if is_en
            else f"🟠 MODO LOCAL ({len(df):,} celdas{suffix}) · API no disponible: {api_err}"
        )
        return df.reset_index(drop=True), False, msg, False


@st.cache_data(ttl=60, show_spinner=False)
def fetch_history(
    api_url: str,
    allow_local_fallback: bool,
    pais: Optional[str],
    years: tuple,
    strict_latam_bounds: bool,
    limit: int,
    lang: str = "en",
) -> tuple[pd.DataFrame, bool, tuple]:
    """Carga varios años (una consulta por año). Devuelve (datos, hubo_api_en_vivo, años_truncados)."""
    frames, truncated_years, live_any = [], [], False
    for y in years:
        df_y, live, _status, trunc = fetch_incendios_grid(
            api_url,
            allow_local_fallback,
            pais=pais,
            anio=int(y),
            mes=None,
            fecha=None,
            strict_latam_bounds=strict_latam_bounds,
            limit=limit,
            lang=lang,
        )
        if not df_y.empty:
            frames.append(df_y)
        live_any = live_any or live
        if trunc:
            truncated_years.append(int(y))
    if frames:
        return pd.concat(frames, ignore_index=True), live_any, tuple(truncated_years)
    return pd.DataFrame(columns=EXPECTED_COLUMNS + ["familia"]), live_any, tuple(truncated_years)


# 6. Motor de Reconocimiento Inteligente de Búsqueda (Soporte EN + ES)
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
        rf"\b{metric_pattern}\s*(?:entre|de|desde|between|from)?\s*{num}\s*(?:y|a|hasta|and|to|-)\s*{num}\s*(?:{unit_pattern})?\b",
        norm_query,
    )
    if m_between:
        v1, v2 = float(m_between.group(1)), float(m_between.group(2))
        min_val, max_val = min(v1, v2), max(v1, v2)
        spans.append(m_between.span())
        return min_val, max_val, spans

    m_gt = re.search(
        rf"\b{metric_pattern}\s*(?:>=|>|mayor(?:\s+o\s+igual)?(?:\s+a|\s+que)?|superior(?:\s+a)?|sobre|desde|minimo|greater(?:\s+than)?|above|over|min(?:imum)?)\s*{num}\s*(?:{unit_pattern})?",
        norm_query,
    )
    if m_gt:
        min_val = float(m_gt.group(1))
        spans.append(m_gt.span())

    m_lt = re.search(
        rf"\b{metric_pattern}\s*(?:<=|<|menor(?:\s+o\s+igual)?(?:\s+a|\s+que)?|inferior(?:\s+a)?|bajo|hasta|maximo|less(?:\s+than)?|below|under|max(?:imum)?)\s*{num}\s*(?:{unit_pattern})?",
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
    st.session_state.active_sat = None
    st.session_state.active_inst = None
    st.session_state.active_riesgo = None
    st.session_state.active_pais = "Todos"

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
        metric_pattern=r"(?:frp|potencia(?:\s+radiativa)?|radiative\s+power|power)",
        unit_pattern=r"(?:mw|megavatios?|megawatts?)",
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
        metric_pattern=r"(?:brillo|brightness|temperatura|temperature)",
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
        detected.append({"tipo": T("Brightness", "Brillo"), "valor": brillo_label, "color": "#f87171"})

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
        "dominican republic": "República Dominicana",
        "todos": "Todos",
        "all": "Todos",
        "all latam": "Todos",
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
            detected.append({"tipo": T("Country", "País"), "valor": country_label(matched_country), "color": "#38bdf8"})
            _mark_span(*m.span())
            break

    if exact_date_parsed is not None and exact_date_parsed.year not in available_anios:
        detected.append({"tipo": T("Date out of range", "Fecha sin datos"), "valor": exact_date_parsed.isoformat(), "color": "#f87171"})
        exact_date_parsed = None

    detected_year: Optional[int] = exact_date_parsed.year if exact_date_parsed else None
    if detected_year is None:
        for ym in re.finditer(r"\b(20\d{2})\b", norm_query):
            if not any(ym.start() >= s and ym.end() <= e for s, e in consumed_spans):
                cand_year = int(ym.group(1))
                if 2000 <= cand_year <= date.today().year:
                    detected_year = cand_year
                    _mark_span(*ym.span())
                    break

    if detected_year is not None:
        if detected_year in available_anios:
            st.session_state.cal_anio = detected_year
            detected.append({"tipo": T("Year", "Año"), "valor": str(detected_year), "color": "#a78bfa"})
        else:
            detected.append({"tipo": T("Year out of range", "Año sin datos"), "valor": str(detected_year), "color": "#f87171"})
            detected_year = None

    detected_month: Optional[int] = exact_date_parsed.month if exact_date_parsed else None
    if detected_month is None:
        for m_alias in sorted(MONTH_ALIASES.keys(), key=len, reverse=True):
            m = re.search(rf"\b{re.escape(m_alias)}\b", norm_query)
            if m and not any(m.start() >= s and m.end() <= e for s, e in consumed_spans):
                detected_month = MONTH_ALIASES[m_alias]
                _mark_span(*m.span())
                break

    full_year_match = re.search(
        r"\b(ano completo|anio completo|todo el ano|todo el anio|anual|full year|whole year|annual)\b",
        norm_query,
    )
    if full_year_match:
        _mark_span(*full_year_match.span())

    if detected_month is not None:
        st.session_state.cal_mes = detected_month
        detected.append({"tipo": T("Month", "Mes"), "valor": month_name(detected_month), "color": "#f472b6"})

    if exact_date_parsed is not None:
        st.session_state.cal_dia = exact_date_parsed.day
        st.session_state.temporal_mode = "fecha_exacta"
        detected.append({"tipo": T("Exact date", "Fecha exacta"), "valor": exact_date_parsed.isoformat(), "color": "#fb7185"})
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
                    detected.append({"tipo": T("Exact date", "Fecha exacta"), "valor": iso_str, "color": "#fb7185"})
                    _mark_span(*dm.span())
                    day_found = True
                    break
        if not day_found:
            st.session_state.temporal_mode = "mes_completo"
    elif full_year_match:
        st.session_state.temporal_mode = "anio_completo"
        detected.append({"tipo": T("Period", "Periodo"), "valor": T("Full year", "Año completo"), "color": "#c084fc"})
    elif detected_year is not None and detected_month is None:
        st.session_state.temporal_mode = "anio_completo"
        detected.append({"tipo": T("Period", "Periodo"), "valor": T("Full year", "Año completo"), "color": "#c084fc"})

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
        detected.append({"tipo": T("Satellite", "Satélite"), "valor": ", ".join(matched_sats), "color": "#34d399"})

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
        ("Alto", [r"\balto\b", r"\baltos\b", r"\bcritico\b", r"\bcriticos\b", r"\bgrave\b", r"\brojo\b", r"\bhigh\b", r"\bcritical\b", r"\bred\b"]),
        ("Medio", [r"\bmedio\b", r"\bmedios\b", r"\bmoderado\b", r"\bmoderados\b", r"\bnaranja\b", r"\bambar\b", r"\bmedium\b", r"\bmoderate\b", r"\borange\b"]),
        ("Bajo", [r"\bbajo\b", r"\bbajos\b", r"\bleve\b", r"\bperiferico\b", r"\bverde\b", r"\blow\b", r"\bgreen\b"]),
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
        detected.append(
            {
                "tipo": T("Risk Level", "Nivel de Riesgo"),
                "valor": ", ".join(risk_label(r) for r in matched_risks),
                "color": "#fbbf24",
            }
        )

    stopwords = {
        "de", "del", "la", "el", "los", "las", "en", "y", "con", "para", "por", "mes", "ano", "anio",
        "the", "in", "and", "with", "for", "by", "of", "month", "year",
    }
    mask = [False] * len(norm_query)
    for st_idx, en_idx in consumed_spans:
        for idx in range(st_idx, min(en_idx, len(mask))):
            mask[idx] = True

    residual_chars = [ch if not mask[i] else " " for i, ch in enumerate(norm_query)]
    residual_tokens = [
        tok for tok in re.findall(r"[a-z0-9\-]+", "".join(residual_chars)) if tok not in stopwords and len(tok) > 1
    ]
    if residual_tokens:
        detected.append({"tipo": T("Unrecognized", "No reconocido"), "valor": " ".join(residual_tokens), "color": "#94a3b8"})

    st.session_state.last_parsed_entities = detected
    return detected


# 6B. Utilidades de Selección Espacial por Dibujo en 2D (Ray-Casting + Ordenamiento sin cruces + Malla Estática)
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


def _segments_properly_intersect(
    a: List[float], b: List[float], c: List[float], d: List[float]
) -> bool:
    """Comprueba si los segmentos AB y CD se cruzan estrictamente en su interior."""
    def _ccw(p1: List[float], p2: List[float], p3: List[float]) -> float:
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    d1 = _ccw(a, c, d)
    d2 = _ccw(b, c, d)
    d3 = _ccw(a, b, c)
    d4 = _ccw(a, b, d)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def sanitize_and_order_polygon_vertices(
    raw_vertices: List[List[float]], uncross: bool = True
) -> List[List[float]]:
    """Elimina duplicados y aplica 2-opt para deshacer cruces de líneas conservando el perímetro."""
    cleaned: List[List[float]] = []
    for pt in raw_vertices:
        lon_v, lat_v = round(float(pt[0]), 5), round(float(pt[1]), 5)
        if not any(abs(existing[0] - lon_v) < 1e-4 and abs(existing[1] - lat_v) < 1e-4 for existing in cleaned):
            cleaned.append([lon_v, lat_v])

    n = len(cleaned)
    if not uncross or n < 4:
        return cleaned

    pts = [list(p) for p in cleaned]
    for _ in range(60):
        found_crossing = False
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c = pts[j]
                d = pts[(j + 1) % n]
                if _segments_properly_intersect(a, b, c, d):
                    pts[i + 1 : j + 1] = list(reversed(pts[i + 1 : j + 1]))
                    found_crossing = True
                    break
            if found_crossing:
                break
        if not found_crossing:
            break

    return pts


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


@st.cache_data
def build_static_clickable_mesh() -> pd.DataFrame:
    """Construye una rejilla pickable 100% estática de alta precisión (0.25°)."""
    step = 0.25
    half = step * 0.5
    lats = np.arange(-58.0, 35.5, step)
    lons = np.arange(-122.0, -30.0, step)
    glon, glat = np.meshgrid(lons, lats)
    glon = glon.ravel()
    glat = glat.ravel()

    polys = np.stack(
        [
            np.column_stack([glon - half, glat - half]),
            np.column_stack([glon + half, glat - half]),
            np.column_stack([glon + half, glat + half]),
            np.column_stack([glon - half, glat + half]),
        ],
        axis=1,
    ).round(3).tolist()

    return pd.DataFrame(
        {
            "polygon": polys,
            "lat": np.round(glat, 3).tolist(),
            "lon": np.round(glon, 3).tolist(),
            "info": ["0.25° grid (~27 km)"] * len(glat),
        }
    )


def _sync_circle_from_inputs() -> None:
    lat_val = round(float(st.session_state.get("dlg_brush_lat", st.session_state.picked_lat_2d or -15.0)), 4)
    lon_val = round(float(st.session_state.get("dlg_brush_lon", st.session_state.picked_lon_2d or -60.0)), 4)
    rad_val = float(st.session_state.get("dlg_brush_km", 150.0))

    st.session_state.picked_lat_2d = lat_val
    st.session_state.picked_lon_2d = lon_val
    st.session_state["dlg_vert_lat"] = lat_val
    st.session_state["dlg_vert_lon"] = lon_val
    st.session_state.draft_polygon_2d = build_circular_brush_polygon(lon_val, lat_val, rad_val)
    st.session_state.draft_vertices_2d = []
    st.session_state.draft_cleared = False


def _sync_vertex_inputs() -> None:
    lat_val = round(float(st.session_state.get("dlg_vert_lat", st.session_state.picked_lat_2d or -15.0)), 4)
    lon_val = round(float(st.session_state.get("dlg_vert_lon", st.session_state.picked_lon_2d or -60.0)), 4)
    st.session_state.picked_lat_2d = lat_val
    st.session_state.picked_lon_2d = lon_val
    st.session_state["dlg_brush_lat"] = lat_val
    st.session_state["dlg_brush_lon"] = lon_val


def _wavg(values: Any, weights: Any) -> float:
    """Promedio ponderado que ignora valores faltantes (NaN) en vez de propagarlos."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m = np.isfinite(v) & np.isfinite(w)
    return float(np.average(v[m], weights=w[m])) if m.any() and w[m].sum() > 0 else float("nan")


# 6C. Utilidades de Exportación de Registros Filtrados (CSV, Excel, GeoJSON)
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
    pais_slug = "LATAM_All" if pais == "Todos" else _slugify_filename_part(pais)
    if temporal_mode == "fecha_exacta":
        period_slug = fecha_iso
    elif temporal_mode == "mes_completo":
        period_slug = f"{cal_anio:04d}-{cal_mes:02d}_{_slugify_filename_part(month_name(cal_mes))}"
    else:
        period_slug = f"Year_{cal_anio:04d}"
    zone_suffix = "_AOI2D" if has_drawn_zone else ""
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
        f"{_sheet_xml('Filtered_Records', export_df)}"
        f"{_sheet_xml('Query_Summary', summary_df)}"
        "</Workbook>"
    )
    return xml_doc.encode("utf-8")


def convert_df_to_excel_bytes(df: pd.DataFrame, resumen_meta: Dict[str, Any]) -> tuple[bytes, str, str]:
    export_df = prepare_export_dataframe(df)
    output = io.BytesIO()

    total_celdas = len(export_df)
    total_incendios = int(export_df["total_incendios"].sum()) if total_celdas > 0 else 0
    if total_celdas > 0 and total_incendios > 0:
        w = export_df["total_incendios"].to_numpy(dtype=np.float64)
        brillo_pond = round(_wavg(export_df["brillo_promedio"], w), 2)
        frp_pond = round(_wavg(export_df["frp_promedio"], w), 2)
    else:
        brillo_pond = 0.0
        frp_pond = 0.0

    summary_rows = [
        {"Parameter": "Export Timestamp", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"Parameter": "Data Source", "Value": str(resumen_meta.get("origen", "Backend API"))},
        {"Parameter": "Country / Region", "Value": str(resumen_meta.get("pais", "All"))},
        {"Parameter": "Time Period", "Value": str(resumen_meta.get("periodo", ""))},
        {"Parameter": "2D Area Selection", "Value": str(resumen_meta.get("zona_2d", "None"))},
        {"Parameter": "Satellites", "Value": str(resumen_meta.get("satelites", ""))},
        {"Parameter": "Sensors / Instruments", "Value": str(resumen_meta.get("sensores", ""))},
        {"Parameter": "Risk Levels", "Value": str(resumen_meta.get("riesgo", ""))},
        {"Parameter": "FRP Filter (MW)", "Value": str(resumen_meta.get("frp_filtro", "No limit"))},
        {"Parameter": "Brightness Filter (K)", "Value": str(resumen_meta.get("brillo_filtro", "No limit"))},
        {"Parameter": "Exported Cells", "Value": total_celdas},
        {"Parameter": "Total Hotspots", "Value": total_incendios},
        {"Parameter": "Weighted Mean Brightness (K)", "Value": brillo_pond},
        {"Parameter": "Weighted Mean FRP (MW)", "Value": frp_pond},
    ]
    summary_df = pd.DataFrame(summary_rows)
    xlsx_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    try:
        import openpyxl  # noqa: F401
        from openpyxl.styles import Alignment, Font, PatternFill

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            export_df.to_excel(writer, sheet_name="Filtered_Records", index=False)
            summary_df.to_excel(writer, sheet_name="Query_Summary", index=False)

            header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True)

            for sheet_name in ("Filtered_Records", "Query_Summary"):
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
                export_df.to_excel(writer, sheet_name="Filtered_Records", index=False)
                summary_df.to_excel(writer, sheet_name="Query_Summary", index=False)
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
                    "geometry_type": "aoi_2d_polygon",
                    "vertices": len(closed_ring),
                },
            }
        )

    for row in export_df.to_dict(orient="records"):
        lon_val = round(float(row["lon_grid"]), 5)
        lat_val = round(float(row["lat_grid"]), 5)
        props = {
            "geometry_type": "hotspot_cell",
            "country": str(row["pais"]),
            "date": str(row["fecha"]),
            "year": int(row["anio"]),
            "month": int(row["mes"]),
            "total_hotspots": int(row["total_incendios"]),
            "frp_mean_mw": float(row["frp_promedio"]),
            "brightness_mean_k": float(row["brillo_promedio"]),
            "risk_level": str(row["nivel_riesgo"]),
            "satellite": str(row["satelite"]),
            "instrument": str(row["instrumento"]),
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
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "filters": resumen_meta,
            "total_records": len(export_df),
        },
        "features": features,
    }
    return json.dumps(feature_collection, ensure_ascii=False, indent=2).encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=6)
def build_export_payload(fmt: str, df: pd.DataFrame, meta: Dict[str, Any], polygon: Optional[List[List[float]]]) -> tuple:
    if fmt == "CSV":
        return convert_df_to_csv_bytes(df), "csv", "text/csv"
    if fmt == "Excel":
        return convert_df_to_excel_bytes(df, meta)
    return convert_df_to_geojson_bytes(df, meta, polygon), "geojson", "application/geo+json"


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
    t_bright = _normalize_percentile(_fill_median(brightness))
    t_frp = _normalize_percentile(_fill_median(np.log1p(np.maximum(frp, 0.0))))
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
    palette_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(heat_values)
    if n == 0:
        empty = np.zeros((0, 4), dtype=np.uint8)
        return empty, empty

    pal = THERMAL_PALETTES.get(palette_key, THERMAL_PALETTES["inferno"])
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


_HEX_ANGLES = np.deg2rad(np.arange(0, 360, 60))


def _fill_median(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a)
    return np.where(finite, a, np.median(a[finite]))


def _hex_ring_coords(c_lons, c_lats, radius_deg, lon_scale, is_globe, altitude_m):
    v_lons = c_lons[:, None] + (radius_deg / lon_scale) * np.cos(_HEX_ANGLES)[None, :]
    v_lats = np.clip(c_lats[:, None] + radius_deg * np.sin(_HEX_ANGLES)[None, :], -85.0, 85.0)
    if is_globe:
        coords = np.stack([v_lons, v_lats, np.full_like(v_lons, float(altitude_m))], axis=-1)
    else:
        coords = np.stack([v_lons, v_lats], axis=-1)
    return coords.round(5).tolist()


def _lonlat_to_axial_bins(lons: np.ndarray, lats: np.ndarray, radius_deg: float, lon_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(lons, dtype=np.float64) * lon_scale
    y = np.asarray(lats, dtype=np.float64)
    q = (2.0 / 3.0) * x / radius_deg
    r = (-x / 3.0 + (np.sqrt(3.0) / 3.0) * y) / radius_deg
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
    lon_scale: float = 1.0,
) -> pd.DataFrame:
    dx = 1.5 * radius_deg
    dy = np.sqrt(3.0) * radius_deg

    qs = np.arange(int(np.floor(lon_min * lon_scale / dx)), int(np.ceil(lon_max * lon_scale / dx)) + 1, dtype=np.int64)
    q_list, r_list = [], []
    for q_val in qs:
        r_min = int(np.floor((lat_min / dy) - (q_val / 2.0)))
        r_max = int(np.ceil((lat_max / dy) - (q_val / 2.0)))
        rs = np.arange(r_min, r_max + 1, dtype=np.int64)
        q_list.append(np.full_like(rs, q_val))
        r_list.append(rs)

    q_arr = np.concatenate(q_list)
    r_arr = np.concatenate(r_list)
    c_lons = dx * q_arr.astype(np.float64) / lon_scale
    c_lats = dy * (r_arr.astype(np.float64) + q_arr.astype(np.float64) / 2.0)

    valid = (c_lons >= lon_min) & (c_lons <= lon_max) & (c_lats >= lat_min) & (c_lats <= lat_max)
    return pd.DataFrame({"polygon": _hex_ring_coords(c_lons[valid], c_lats[valid], radius_deg, lon_scale, is_globe, globe_altitude_m)})


def _fmt_or_na(arr: np.ndarray, suffix: str, na_txt: str = "n/a") -> List[str]:
    return [na_txt if not np.isfinite(v) else f"{v:.1f} {suffix}" for v in arr]


@st.cache_data
def build_clustered_hex_layer(
    df: pd.DataFrame,
    radius_deg: float,
    tier_label: str,
    palette_key: str,
    smooth_weight: float = 0.35,
    is_globe: bool = False,
    is_sensor_1to1: bool = False,
    globe_altitude_m: float = 1210.0,
    lon_scale: float = 1.0,
    lang: str = "en",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    is_en = lang == "en"
    x = df["lon_grid"].to_numpy(dtype=np.float64)
    y = df["lat_grid"].to_numpy(dtype=np.float64)
    qi, ri = _lonlat_to_axial_bins(x, y, radius_deg, lon_scale)

    fires = df["total_incendios"].to_numpy(dtype=np.float64)
    b = df["brillo_promedio"].to_numpy(dtype=np.float64)
    f = df["frp_promedio"].to_numpy(dtype=np.float64)
    fam = df["familia"].to_numpy() if "familia" in df.columns else np.full(len(df), "Otro")
    vb, vf = np.isfinite(b), np.isfinite(f)

    work = pd.DataFrame(
        {
            "q_bin": qi,
            "r_bin": ri,
            "total_incendios": fires,
            "w_brillo": np.where(vb, b * fires, 0.0),
            "n_brillo": np.where(vb, fires, 0.0),
            "w_frp": np.where(vf, f * fires, 0.0),
            "n_frp": np.where(vf, fires, 0.0),
            "celdas": 1,
            "n_modis": np.where(fam == "MODIS", fires, 0.0),
            "n_viirs": np.where(fam == "VIIRS", fires, 0.0),
            "instrumento": df["instrumento"].to_numpy(),
            "satelite": df["satelite"].to_numpy(),
            "pais": df["pais"].to_numpy(),
        }
    )
    keys = ["q_bin", "r_bin"]
    agg = work.groupby(keys).agg(
        total_incendios=("total_incendios", "sum"),
        w_brillo=("w_brillo", "sum"),
        n_brillo=("n_brillo", "sum"),
        w_frp=("w_frp", "sum"),
        n_frp=("n_frp", "sum"),
        celdas_agrupadas=("celdas", "sum"),
        n_modis=("n_modis", "sum"),
        n_viirs=("n_viirs", "sum"),
    )
    for col in ("instrumento", "satelite", "pais"):
        agg[col] = work[keys + [col]].drop_duplicates().sort_values(col).groupby(keys)[col].agg(" + ".join)
    agg = agg.reset_index()

    q_arr = agg["q_bin"].to_numpy(dtype=np.int64)
    r_arr = agg["r_bin"].to_numpy(dtype=np.int64)
    c_lons = radius_deg * 1.5 * q_arr.astype(np.float64) / lon_scale
    c_lats = np.clip(radius_deg * np.sqrt(3.0) * (r_arr.astype(np.float64) + q_arr.astype(np.float64) / 2.0), -84.0, 84.0)

    counts_arr = agg["total_incendios"].to_numpy(dtype=np.float64)
    nb = agg["n_brillo"].to_numpy(dtype=np.float64)
    nf = agg["n_frp"].to_numpy(dtype=np.float64)
    brillo_arr = np.where(nb > 0, agg["w_brillo"].to_numpy() / np.maximum(nb, 1e-9), np.nan).round(1)
    frp_arr = np.where(nf > 0, agg["w_frp"].to_numpy() / np.maximum(nf, 1e-9), np.nan).round(1)

    heat_vals = compute_smoothed_thermal_field(q_arr, r_arr, brillo_arr, frp_arr, counts_arr, smooth_weight=smooth_weight)
    fill_rgba, line_rgba = thermal_values_to_rgba(heat_vals, palette_key=palette_key)

    if is_en:
        nivel = np.select([heat_vals >= 0.72, heat_vals >= 0.42], ["High (relative)", "Medium (relative)"], default="Low (relative)")
    else:
        nivel = np.select([heat_vals >= 0.72, heat_vals >= 0.42], ["Alta (relativa)", "Media (relativa)"], default="Baja (relativa)")

    km_per_deg = METERS_PER_DEG_LAT / 1000.0
    area_km2 = np.maximum(
        0.1, (3.0 * np.sqrt(3.0) / 2.0) * ((radius_deg * km_per_deg) ** 2) * np.cos(np.deg2rad(c_lats)) / lon_scale
    ).round(2)
    diam_km = round(2.0 * radius_deg * km_per_deg, 2)

    max_count = max(float(counts_arr.max()), 1.0)
    max_elev = 40_000.0 if is_sensor_1to1 else 180_000.0
    n = len(agg)
    if is_en:
        res_label = ["Hexagon ≈1 km² (~1 km cells)"] * n if is_sensor_1to1 else [f"Cluster Ø ~{diam_km} km"] * n
        na_txt = "n/a"
    else:
        res_label = ["Hexágono ≈1 km² (celdas de ~1 km)"] * n if is_sensor_1to1 else [f"Clúster Ø ~{diam_km} km"] * n
        na_txt = "s/d"

    return pd.DataFrame(
        {
            "polygon": _hex_ring_coords(c_lons, c_lats, radius_deg, lon_scale, is_globe, globe_altitude_m),
            "rgba": fill_rgba.tolist(),
            "line_rgba": line_rgba.tolist(),
            "elevation": ((counts_arr / max_count) * max_elev).round(0).tolist(),
            "escala_vista": [tier_label] * n,
            "pais": agg["pais"].tolist(),
            "lat_grid": np.round(c_lats, 4).tolist(),
            "lon_grid": np.round(c_lons, 4).tolist(),
            "total_incendios": counts_arr.round().astype(np.int64).tolist(),
            "n_modis": agg["n_modis"].round().astype(np.int64).tolist(),
            "n_viirs": agg["n_viirs"].round().astype(np.int64).tolist(),
            "celdas_agrupadas": agg["celdas_agrupadas"].tolist(),
            "frp_promedio": np.nan_to_num(frp_arr, nan=0.0).tolist(),
            "brillo_promedio": np.nan_to_num(brillo_arr, nan=0.0).tolist(),
            "frp_txt": _fmt_or_na(frp_arr, "MW", na_txt=na_txt),
            "brillo_txt": _fmt_or_na(brillo_arr, "K", na_txt=na_txt),
            "indice_termico": np.round(heat_vals * 100.0, 1).tolist(),
            "instrumento": agg["instrumento"].tolist(),
            "satelite": agg["satelite"].tolist(),
            "nivel_intensidad": nivel.tolist(),
            "resolucion_m": res_label,
            "area_km2": area_km2.tolist(),
        }
    )


# 7B. ANÁLISIS TEMPORAL: CALENDARIO DE ACTIVIDAD, ARMONIZACIÓN MODIS↔VIIRS Y ANOMALÍAS
MIN_SOLAPE_PERIODOS = 8
MIN_ANIOS_REFERENCIA = 3
Z_CRITICO = 2.0

SERIE_MODIS = "modis"
SERIE_VIIRS_CONV = "viirs"
SERIE_MEDIA = "media"


def series_display_name(key: str) -> str:
    if key == SERIE_MODIS:
        return "MODIS"
    if key == SERIE_VIIRS_CONV:
        return T("VIIRS (MODIS scale)", "VIIRS (formato MODIS)")
    return T("Mean (MODIS & VIIRS)", "Media (MODIS y VIIRS)")


def prepare_history(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if df.empty:
        return df.assign(t_anio=pd.Series(dtype=int), t_idx=pd.Series(dtype=int)), "semanal"
    fdt = pd.to_datetime(df["fecha"], errors="coerce")
    if fdt.notna().mean() >= 0.5:
        out = df[fdt.notna()].copy()
        fdt = fdt[fdt.notna()]
        out["t_anio"] = fdt.dt.year.to_numpy()
        out["t_idx"] = ((fdt.dt.dayofyear.to_numpy() - 1) // 7 + 1).astype(int)
        return out, "semanal"
    out = df.copy()
    out["t_anio"] = out["anio"]
    out["t_idx"] = out["mes"]
    return out, "mensual"


def period_start(anio: Any, idx: Any, gran: str) -> pd.Timestamp:
    if gran == "semanal":
        return pd.Timestamp(year=int(anio), month=1, day=1) + pd.Timedelta(days=(int(idx) - 1) * 7)
    return pd.Timestamp(year=int(anio), month=int(idx), day=1)


def period_end(anio: Any, idx: Any, gran: str) -> pd.Timestamp:
    start = period_start(anio, idx, gran)
    if gran == "semanal":
        return min(start + pd.Timedelta(days=6), pd.Timestamp(year=int(anio), month=12, day=31))
    return start + pd.offsets.MonthEnd(0)


def aggregate_series(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["familia", "anio", "idx", "valor"])
    w = df["total_incendios"].to_numpy(dtype=np.float64)
    if metric == "frp":
        frp = df["frp_promedio"].to_numpy(dtype=np.float64)
        val = np.where(np.isfinite(frp), frp * w, 0.0)
    else:
        val = w
    tmp = pd.DataFrame({"familia": df["familia"].to_numpy(), "anio": df["t_anio"].to_numpy(), "idx": df["t_idx"].to_numpy(), "valor": val})
    tmp = tmp[tmp["familia"].isin(["MODIS", "VIIRS"])]
    return tmp.groupby(["familia", "anio", "idx"], as_index=False)["valor"].sum()


def fill_missing_periods(series: pd.DataFrame) -> pd.DataFrame:
    if series.empty:
        return series
    cut = series.groupby("anio")["idx"].max()
    frames = []
    for fam, g in series.groupby("familia"):
        first = int(g["anio"].min())
        rows = [(int(a), i) for a, c in cut.items() if int(a) >= first for i in range(1, int(c) + 1)]
        base = pd.DataFrame(rows, columns=["anio", "idx"])
        base["familia"] = fam
        merged = base.merge(g, on=["familia", "anio", "idx"], how="left")
        merged["valor"] = merged["valor"].fillna(0.0)
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def harmonize_to_modis(series: pd.DataFrame) -> tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    info: Dict[str, Any] = {"ok": False, "has_ratio": False, "ratio": None, "p25": None, "p75": None, "n_solape": 0}
    empty = pd.DataFrame(columns=["anio", "idx", "valor"])
    result_series: Dict[str, pd.DataFrame] = {
        SERIE_MODIS: empty,
        SERIE_VIIRS_CONV: empty,
        SERIE_MEDIA: empty,
    }
    if series.empty:
        return result_series, info

    wide = series.pivot_table(index=["anio", "idx"], columns="familia", values="valor", aggfunc="sum")
    if "MODIS" in wide.columns:
        result_series[SERIE_MODIS] = wide["MODIS"].dropna().rename("valor").reset_index()

    if not {"MODIS", "VIIRS"} <= set(wide.columns):
        return result_series, info

    both = wide.dropna(subset=["MODIS", "VIIRS"])
    both = both[(both["MODIS"] > 0) & (both["VIIRS"] > 0)]
    info["n_solape"] = int(len(both))
    if len(both) < 1:
        return result_series, info

    ratios = both["MODIS"] / both["VIIRS"]
    ratio = float(ratios.median())
    info.update(
        {
            "ok": len(both) >= MIN_SOLAPE_PERIODOS,
            "has_ratio": True,
            "ratio": ratio,
            "p25": float(ratios.quantile(0.25)),
            "p75": float(ratios.quantile(0.75)),
        }
    )

    viirs_conv = (wide["VIIRS"] * ratio).dropna()
    media = pd.concat([wide["MODIS"], wide["VIIRS"] * ratio], axis=1).mean(axis=1, skipna=True).dropna()

    result_series[SERIE_VIIRS_CONV] = viirs_conv.rename("valor").reset_index()
    result_series[SERIE_MEDIA] = media.rename("valor").reset_index()
    return result_series, info


def compute_anomalies(serie: pd.DataFrame, anio_foco: int) -> tuple[pd.DataFrame, str]:
    cols = ["idx", "observado", "esperado", "z", "n_ref", "critico"]
    piv = serie.pivot_table(index="idx", columns="anio", values="valor", aggfunc="sum")
    if anio_foco not in piv.columns:
        return pd.DataFrame(columns=cols), T("No data available for the evaluated year.", "No hay datos del año evaluado.")
    ref_cols = [c for c in piv.columns if c != anio_foco]
    if len(ref_cols) < MIN_ANIOS_REFERENCIA:
        return pd.DataFrame(columns=cols), T(
            f"At least {MIN_ANIOS_REFERENCIA} reference years are required in addition to the evaluated year.",
            f"Se necesitan al menos {MIN_ANIOS_REFERENCIA} años de referencia además del evaluado.",
        )
    ref = np.log1p(piv[ref_cols])
    mu = ref.mean(axis=1)
    sd = ref.std(axis=1, ddof=1).clip(lower=0.25)
    n_ref = ref.notna().sum(axis=1)
    x = piv[anio_foco]
    out = pd.DataFrame(
        {"idx": piv.index.to_numpy(), "observado": x.to_numpy(), "esperado": np.expm1(mu).to_numpy(),
         "z": ((np.log1p(x) - mu) / sd).to_numpy(), "n_ref": n_ref.to_numpy()}
    )
    out = out.dropna(subset=["observado", "z"])
    out = out[out["n_ref"] >= MIN_ANIOS_REFERENCIA].copy()
    out["critico"] = out["z"] >= Z_CRITICO
    return out.reset_index(drop=True), ""


# 8. CONSULTA DE CATÁLOGOS Y DATOS SEGÚN ESTADO CANÓNICO (con detección automática de países)
catalogs = fetch_api_catalogs(st.session_state.api_url)
if not catalogs and st.session_state.allow_local_fallback:
    catalogs = local_catalogs()

for p_cat in catalogs.get("paises", []):
    clean_cat = str(p_cat).strip()
    if clean_cat and clean_cat not in ("Todos", "Sin dato"):
        st.session_state.known_paises.add(clean_cat)

_anios_cat: List[int] = []
for _a in catalogs.get("anios", []):
    try:
        val_a = int(_a)
        if 2000 <= val_a <= 2025:
            _anios_cat.append(val_a)
    except (TypeError, ValueError):
        pass
anios_api = sorted(set(_anios_cat) or set(range(2018, 2026)), reverse=True)

if st.session_state.cal_anio not in anios_api:
    st.session_state.cal_anio = anios_api[0]

max_dias_mes = calendar.monthrange(st.session_state.cal_anio, st.session_state.cal_mes)[1]
if st.session_state.cal_dia > max_dias_mes:
    st.session_state.cal_dia = max_dias_mes

fecha_iso_activa = f"{st.session_state.cal_anio:04d}-{st.session_state.cal_mes:02d}-{st.session_state.cal_dia:02d}"
query_fecha = fecha_iso_activa if st.session_state.temporal_mode == "fecha_exacta" else None
query_mes = st.session_state.cal_mes if st.session_state.temporal_mode == "mes_completo" else None

df_grid, is_live_api, connection_status, was_truncated = fetch_incendios_grid(
    st.session_state.api_url,
    st.session_state.allow_local_fallback,
    pais=st.session_state.active_pais,
    anio=None if query_fecha else st.session_state.cal_anio,
    mes=query_mes,
    fecha=query_fecha,
    strict_latam_bounds=st.session_state.strict_latam,
    limit=int(st.session_state.api_limit),
    lang=st.session_state.lang,
)

# Extraer automáticamente los países reales presentes en los registros recibidos
if not df_grid.empty and "pais" in df_grid.columns:
    for p_row in df_grid["pais"].dropna().unique():
        clean_p = str(p_row).strip()
        if clean_p and clean_p not in ("Todos", "Sin dato"):
            st.session_state.known_paises.add(clean_p)

paises_api = ["Todos"] + sorted(st.session_state.known_paises)
if st.session_state.active_pais not in paises_api:
    st.session_state.active_pais = "Todos"

if not df_grid.empty:
    riesgos_disponibles = [r for r in ["Alto", "Medio", "Bajo"] if r in df_grid["nivel_riesgo"].unique()]
    if not riesgos_disponibles:
        riesgos_disponibles = sorted(df_grid["nivel_riesgo"].unique().tolist())
    insts_disponibles = sorted(df_grid["instrumento"].unique().tolist())
else:
    riesgos_disponibles = ["Alto", "Medio", "Bajo"]
    insts_disponibles = catalogs.get("instrumentos", ["MODIS", "VIIRS"])

preferred_inst = next((i for i in insts_disponibles if "VIIRS" in i.upper()), insts_disponibles[0] if insts_disponibles else "VIIRS")
if st.session_state.active_inst is None:
    selected_inst = [preferred_inst]
else:
    _valid_i = [i for i in st.session_state.active_inst if i in insts_disponibles]
    selected_inst = _valid_i if _valid_i else [preferred_inst]

if not df_grid.empty:
    sats_disponibles = sorted(df_grid.loc[df_grid["instrumento"].isin(selected_inst), "satelite"].unique().tolist())
    sats_catalogo = sorted(df_grid["satelite"].unique().tolist())
else:
    sats_disponibles = catalogs.get("satelites", [])
    sats_catalogo = sats_disponibles

if st.session_state.active_riesgo is None:
    selected_riesgo = riesgos_disponibles
else:
    _valid_r = [r for r in st.session_state.active_riesgo if r in riesgos_disponibles]
    selected_riesgo = _valid_r if _valid_r else riesgos_disponibles

if st.session_state.active_sat is None:
    selected_sat = sats_disponibles
else:
    _valid_s = [s for s in st.session_state.active_sat if s in sats_disponibles]
    selected_sat = _valid_s if _valid_s else sats_disponibles

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

fams_active = sorted(filtered_df["familia"].unique().tolist()) if not filtered_df.empty else []

if st.session_state.temporal_mode == "fecha_exacta":
    periodo_txt = fecha_iso_activa
elif st.session_state.temporal_mode == "mes_completo":
    periodo_txt = f"{month_name(st.session_state.cal_mes)} {st.session_state.cal_anio}"
else:
    periodo_txt = f"{T('Year', 'Año')} {st.session_state.cal_anio}"


def _range_desc(vmin: Optional[float], vmax: Optional[float], unit: str) -> str:
    parts = []
    if vmin is not None:
        parts.append(f"≥ {vmin:g} {unit}")
    if vmax is not None:
        parts.append(f"≤ {vmax:g} {unit}")
    joiner = T(" and ", " y ")
    return joiner.join(parts) if parts else T("No limit", "Sin límite")


frp_desc = _range_desc(st.session_state.frp_min, st.session_state.frp_max, "MW")
brillo_desc = _range_desc(st.session_state.brillo_min, st.session_state.brillo_max, "K")
zona_2d_desc = (
    T(f"Closed polygon ({len(st.session_state.drawn_polygon_2d)} vertices)", f"Polígono cerrado ({len(st.session_state.drawn_polygon_2d)} vértices)")
    if has_active_2d_drawing
    else T("Not applied", "No aplicada")
)

export_metadata: Dict[str, Any] = {
    "origen": f"External Server ({st.session_state.api_url})" if is_live_api else "Local CSV Backup",
    "pais": country_label(st.session_state.active_pais),
    "periodo": periodo_txt,
    "zona_2d": zona_2d_desc,
    "satelites": ", ".join(selected_sat),
    "sensores": ", ".join(selected_inst),
    "riesgo": ", ".join(risk_label(r) for r in selected_riesgo),
    "frp_filtro": frp_desc,
    "brillo_filtro": brillo_desc,
    "aviso": "MODIS & VIIRS combined" if len(fams_active) > 1 else "",
}


# 8C. DIÁLOGO INTERACTIVO DE SELECCIÓN POR DIBUJO EN 2D
def _on_dismiss_2d_dialog() -> None:
    st.session_state.aoi_dialog_open = False


def _rebuild_polygon_draft_from_vertices() -> None:
    ordered = sanitize_and_order_polygon_vertices(
        st.session_state.draft_vertices_2d,
        uncross=bool(st.session_state.auto_uncross_polygon),
    )
    st.session_state.draft_vertices_2d = ordered
    if len(ordered) >= 3:
        st.session_state.draft_polygon_2d = list(ordered) + [ordered[0]]
    else:
        st.session_state.draft_polygon_2d = None


_dialog_kwargs: Dict[str, Any] = {"width": "large"}
if "on_dismiss" in inspect.signature(st.dialog).parameters:
    _dialog_kwargs["on_dismiss"] = _on_dismiss_2d_dialog


@st.dialog(T("🖌️ 2D Drawing Selection (Brush / Polygon)", "🖌️ Selección por Dibujo en 2D (Pincel / Lazo)"), **_dialog_kwargs)
def open_2d_drawing_dialog(reference_df: pd.DataFrame) -> None:
    st.session_state.aoi_dialog_open = True
    st.session_state.aoi_expander_open = True

    if (
        st.session_state.draft_polygon_2d is None
        and not st.session_state.draft_cleared
        and st.session_state.drawn_polygon_2d is not None
    ):
        st.session_state.draft_polygon_2d = [list(pt) for pt in st.session_state.drawn_polygon_2d]

    if st.session_state.picked_lat_2d is None or st.session_state.picked_lon_2d is None:
        if not reference_df.empty:
            st.session_state.picked_lat_2d = round(float(reference_df["lat_grid"].mean()), 4)
            st.session_state.picked_lon_2d = round(float(reference_df["lon_grid"].mean()), 4)
        else:
            st.session_state.picked_lat_2d = -15.0
            st.session_state.picked_lon_2d = -60.0

    if st.session_state.dialog_cam_lat is None or st.session_state.dialog_cam_lon is None:
        st.session_state.dialog_cam_lat = float(st.session_state.picked_lat_2d)
        st.session_state.dialog_cam_lon = float(st.session_state.picked_lon_2d)

    for k_lat in ("dlg_brush_lat", "dlg_vert_lat"):
        if k_lat not in st.session_state:
            st.session_state[k_lat] = float(st.session_state.picked_lat_2d)
    for k_lon in ("dlg_brush_lon", "dlg_vert_lon"):
        if k_lon not in st.session_state:
            st.session_state[k_lon] = float(st.session_state.picked_lon_2d)
    if "dlg_brush_km" not in st.session_state:
        st.session_state["dlg_brush_km"] = 150.0

    draw_opts = ["circle", "polygon"]
    draw_mode = st.radio(
        T("2D draw mode:", "Modo de dibujo 2D:"),
        options=draw_opts,
        index=draw_opts.index(st.session_state.last_draw_mode_2d),
        format_func=lambda k: T("🎯 Circular Brush (Radius km)", "🎯 Pincel Circular (Radio km)") if k == "circle" else T("✏️ Click Polygon", "✏️ Polígono por Clics"),
        horizontal=True,
        label_visibility="collapsed",
        key="dlg_draw_mode_radio",
    )

    if draw_mode != st.session_state.last_draw_mode_2d:
        st.session_state.last_draw_mode_2d = draw_mode
        st.session_state.last_deck_selection_raw = ""
        if draw_mode == "polygon":
            st.session_state.draft_vertices_2d = []
            st.session_state.draft_polygon_2d = None
            st.session_state.draft_cleared = True

    map_key = f"dialog_2d_interactive_map_{st.session_state.map_widget_nonce}"
    prev_chart_state = st.session_state.get(map_key)
    if prev_chart_state is not None:
        sel_dict = (
            prev_chart_state.get("selection", {})
            if isinstance(prev_chart_state, dict)
            else getattr(prev_chart_state, "selection", {})
        )
        if not isinstance(sel_dict, dict):
            sel_dict = getattr(sel_dict, "__dict__", {})
        objs_map = sel_dict.get("objects", {}) if isinstance(sel_dict, dict) else {}

        if isinstance(objs_map, dict) and any(bool(v) for v in objs_map.values()):
            raw_sig = json.dumps(objs_map, sort_keys=True, default=str)
            if raw_sig != st.session_state.last_deck_selection_raw:
                st.session_state.last_deck_selection_raw = raw_sig
                picked_obj: Optional[Dict[str, Any]] = None
                for layer_priority in ("picker_hotspots_layer", "picker_static_grid_layer"):
                    items = objs_map.get(layer_priority)
                    if items and len(items) > 0 and isinstance(items[0], dict):
                        picked_obj = items[0]
                        break
                if picked_obj is None:
                    for items in objs_map.values():
                        if items and len(items) > 0 and isinstance(items[0], dict):
                            picked_obj = items[0]
                            break

                if picked_obj is not None and "lat" in picked_obj and "lon" in picked_obj:
                    new_lat = round(float(picked_obj["lat"]), 4)
                    new_lon = round(float(picked_obj["lon"]), 4)
                    st.session_state.picked_lat_2d = new_lat
                    st.session_state.picked_lon_2d = new_lon
                    st.session_state["dlg_brush_lat"] = new_lat
                    st.session_state["dlg_brush_lon"] = new_lon
                    st.session_state["dlg_vert_lat"] = new_lat
                    st.session_state["dlg_vert_lon"] = new_lon

                    if draw_mode == "polygon":
                        verts = list(st.session_state.draft_vertices_2d)
                        new_pt = [round(new_lon, 5), round(new_lat, 5)]
                        if not any(abs(v[0] - new_pt[0]) < 1e-4 and abs(v[1] - new_pt[1]) < 1e-4 for v in verts):
                            verts.append(new_pt)
                            st.session_state.draft_vertices_2d = verts
                            st.session_state.draft_cleared = False
                            _rebuild_polygon_draft_from_vertices()
                    elif st.session_state.auto_trace_on_click:
                        rad_km = float(st.session_state.get("dlg_brush_km", 150.0))
                        st.session_state.draft_polygon_2d = build_circular_brush_polygon(new_lon, new_lat, rad_km)
                        st.session_state.draft_vertices_2d = []
                        st.session_state.draft_cleared = False

    st.caption(
        T(
            "Pan and zoom the 2D map (`drag` + `scroll`) and **click anywhere** to add polygon vertices or center the circular brush.",
            "Mueve y acerca el mapa 2D con el cursor (`arrastrar` + `rueda`) y **haz clic en cualquier punto** para añadir vértices o centrar el pincel.",
        )
    )

    focus_lat = float(st.session_state.picked_lat_2d)
    focus_lon = float(st.session_state.picked_lon_2d)

    static_mesh_df = build_static_clickable_mesh()
    picker_layers: List[pdk.Layer] = [
        pdk.Layer(
            "PolygonLayer",
            id="picker_static_grid_layer",
            data=static_mesh_df,
            get_polygon="polygon",
            filled=True,
            stroked=True,
            get_fill_color=[255, 255, 255, 2],
            get_line_color=[148, 163, 184, 24],
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
            highlight_color=[56, 189, 248, 70],
        ),
    ]

    if not reference_df.empty:
        sample_pts = (
            reference_df.sort_values("total_incendios", ascending=False).head(6000)
            if len(reference_df) > 6000
            else reference_df
        )
        hotspots_picker_df = pd.DataFrame(
            {
                "lon": sample_pts["lon_grid"].round(4).tolist(),
                "lat": sample_pts["lat_grid"].round(4).tolist(),
                "info": [
                    T(
                        f"Active hotspot ({p}) · {f} hotspots · {b} K",
                        f"Foco activo ({p}) · {f} focos · {b} K",
                    )
                    for p, f, b in zip(
                        sample_pts["pais"], sample_pts["total_incendios"], sample_pts["brillo_promedio"]
                    )
                ],
            }
        )
        picker_layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id="picker_hotspots_layer",
                data=hotspots_picker_df,
                get_position="[lon, lat]",
                get_fill_color=[251, 146, 60, 210],
                get_line_color=[255, 237, 213, 180],
                get_radius=6500,
                radius_min_pixels=3,
                radius_max_pixels=7,
                pickable=True,
                auto_highlight=True,
                highlight_color=[255, 255, 255, 220],
            )
        )

    if st.session_state.draft_polygon_2d and len(st.session_state.draft_polygon_2d) >= 3:
        ring = list(st.session_state.draft_polygon_2d)
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        picker_layers.append(
            pdk.Layer(
                "PolygonLayer",
                id="picker_draft_zone_layer",
                data=pd.DataFrame([{"polygon": ring}]),
                get_polygon="polygon",
                filled=True,
                stroked=True,
                get_fill_color=[56, 189, 248, 60],
                get_line_color=[56, 189, 248, 255],
                line_width_min_pixels=3,
                pickable=False,
            )
        )

    verts = st.session_state.draft_vertices_2d
    if len(verts) > 0:
        if len(verts) >= 2:
            path_pts = list(verts) + ([verts[0]] if len(verts) >= 3 else [])
            picker_layers.append(
                pdk.Layer(
                    "PathLayer",
                    id="picker_open_poly_path",
                    data=pd.DataFrame([{"path": path_pts}]),
                    get_path="path",
                    get_color=[52, 211, 153, 255],
                    width_min_pixels=3,
                    pickable=False,
                )
            )
        nodes_df = pd.DataFrame(
            [{"lon": v[0], "lat": v[1], "label": f"P{idx + 1}"} for idx, v in enumerate(verts)]
        )
        picker_layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id="picker_open_poly_nodes",
                data=nodes_df,
                get_position="[lon, lat]",
                get_fill_color=[16, 185, 129, 255],
                get_line_color=[255, 255, 255, 255],
                stroked=True,
                line_width_min_pixels=2,
                radius_min_pixels=6,
                radius_max_pixels=9,
                pickable=False,
            )
        )
        picker_layers.append(
            pdk.Layer(
                "TextLayer",
                id="picker_open_poly_labels",
                data=nodes_df,
                get_position="[lon, lat]",
                get_text="label",
                get_size=13,
                get_color=[255, 255, 255, 255],
                get_pixel_offset=[0, -14],
                pickable=False,
            )
        )

    picker_layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            id="picker_selected_pin_layer",
            data=pd.DataFrame([{"lon": focus_lon, "lat": focus_lat}]),
            get_position="[lon, lat]",
            get_fill_color=[56, 189, 248, 255],
            get_line_color=[255, 255, 255, 255],
            stroked=True,
            line_width_min_pixels=2,
            radius_min_pixels=6,
            radius_max_pixels=9,
            pickable=False,
        )
    )

    picker_deck = pdk.Deck(
        layers=picker_layers,
        initial_view_state=pdk.ViewState(
            latitude=float(st.session_state.dialog_cam_lat),
            longitude=float(st.session_state.dialog_cam_lon),
            zoom=4.0,
            min_zoom=1.5,
            max_zoom=14.0,
            pitch=0,
            bearing=0,
        ),
        views=[pdk.View(type="MapView", controller=True, repeat=False)],
        tooltip={
            "text": T(
                "📍 Click to pick point:\nLatitude: {lat}° | Longitude: {lon}°\n{info}",
                "📍 Clic para elegir punto:\nLatitud: {lat}° | Longitud: {lon}°\n{info}",
            )
        },
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    )

    st.pydeck_chart(
        picker_deck,
        height=340,
        **DECK,
        on_select="rerun",
        selection_mode="single-object",
        key=map_key,
    )

    st.markdown(
        f"<div style='background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.45);"
        f"border-radius:10px;padding:8px 12px;color:#e0f2fe;font-size:13px;margin-bottom:6px;'>"
        f"📍 <b>{T('Last selected point:', 'Último punto seleccionado:')}</b> "
        f"{T('Latitude', 'Latitud')} <code>{st.session_state.picked_lat_2d:.4f}°</code> · "
        f"{T('Longitude', 'Longitud')} <code>{st.session_state.picked_lon_2d:.4f}°</code></div>",
        unsafe_allow_html=True,
    )

    if draw_mode == "circle":
        b1, b2, b3 = st.columns([0.34, 0.34, 0.32], vertical_alignment="bottom")
        with b1:
            st.number_input(
                T("Center latitude:", "Latitud centro:"),
                step=0.10,
                format="%.4f",
                key="dlg_brush_lat",
                on_change=_sync_circle_from_inputs,
            )
        with b2:
            st.number_input(
                T("Center longitude:", "Longitud centro:"),
                step=0.10,
                format="%.4f",
                key="dlg_brush_lon",
                on_change=_sync_circle_from_inputs,
            )
        with b3:
            st.number_input(
                T("Brush radius (km):", "Radio pincel (km):"),
                min_value=5.0,
                max_value=2500.0,
                step=15.0,
                key="dlg_brush_km",
                on_change=_sync_circle_from_inputs,
            )

        auto_tr = st.checkbox(
            T("⚡ Auto-draw circle on map click", "⚡ Trazar círculo automáticamente al hacer clic sobre el mapa"),
            value=st.session_state.auto_trace_on_click,
            key="w_auto_trace_cb",
        )
        st.session_state.auto_trace_on_click = auto_tr
    else:
        p1, p2, p3, p4 = st.columns([0.23, 0.23, 0.29, 0.25], vertical_alignment="bottom")
        with p1:
            v_lat = st.number_input(
                T("Manual latitude:", "Latitud manual:"),
                step=0.10,
                format="%.4f",
                key="dlg_vert_lat",
                on_change=_sync_vertex_inputs,
            )
        with p2:
            v_lon = st.number_input(
                T("Manual longitude:", "Longitud manual:"),
                step=0.10,
                format="%.4f",
                key="dlg_vert_lon",
                on_change=_sync_vertex_inputs,
            )
        already_in_verts = any(
            abs(v[0] - round(float(v_lon), 5)) < 1e-4 and abs(v[1] - round(float(v_lat), 5)) < 1e-4
            for v in st.session_state.draft_vertices_2d
        )
        with p3:
            if st.button(
                T("➕ Add coordinate", "➕ Añadir coordenada"),
                type="secondary",
                disabled=already_in_verts,
                help=T(
                    "Each map click adds a vertex automatically; use this button only for manual coordinates.",
                    "Cada clic en el mapa añade el vértice automáticamente; usa este botón solo si escribes coordenadas a mano.",
                ),
                **BTN,
                key="dlg_btn_add_vert",
            ):
                verts_new = list(st.session_state.draft_vertices_2d)
                verts_new.append([round(float(v_lon), 5), round(float(v_lat), 5)])
                st.session_state.draft_vertices_2d = verts_new
                st.session_state.picked_lat_2d = round(float(v_lat), 4)
                st.session_state.picked_lon_2d = round(float(v_lon), 4)
                st.session_state.draft_cleared = False
                _rebuild_polygon_draft_from_vertices()
                st.rerun()
        with p4:
            if st.button(
                T("↩️ Undo point", "↩️ Deshacer punto"),
                disabled=len(st.session_state.draft_vertices_2d) == 0,
                **BTN,
                key="dlg_btn_undo_vert",
            ):
                verts_new = list(st.session_state.draft_vertices_2d)
                verts_new.pop()
                st.session_state.draft_vertices_2d = verts_new
                if len(verts_new) >= 3:
                    _rebuild_polygon_draft_from_vertices()
                else:
                    st.session_state.draft_polygon_2d = None
                    st.session_state.draft_cleared = True
                st.session_state.last_deck_selection_raw = ""
                st.session_state.map_widget_nonce += 1
                st.rerun()

        uncross_cb = st.checkbox(
            T(
                "🧭 Auto-prevent line crossings (order perimeter without self-intersections)",
                "🧭 Evitar cruces de líneas automáticamente (ordenar perímetro sin intersecciones)",
            ),
            value=st.session_state.auto_uncross_polygon,
            key="w_auto_uncross_cb",
        )
        if uncross_cb != st.session_state.auto_uncross_polygon:
            st.session_state.auto_uncross_polygon = uncross_cb
            if len(st.session_state.draft_vertices_2d) >= 3:
                _rebuild_polygon_draft_from_vertices()
            st.rerun()

        if len(st.session_state.draft_vertices_2d) > 0:
            pts_summary = " → ".join(
                f"P{i + 1} ({v[1]:.2f}°, {v[0]:.2f}°)" for i, v in enumerate(st.session_state.draft_vertices_2d)
            )
            st.caption(f"**{T('Current perimeter:', 'Perímetro actual:')}** {pts_summary}")

    c_apply, c_clear, c_close = st.columns([0.46, 0.32, 0.22])
    apply_clicked = c_apply.button(
        T("✅ Apply zone to map", "✅ Aplicar zona al mapa"),
        type="primary",
        disabled=not (st.session_state.draft_polygon_2d and len(st.session_state.draft_polygon_2d) >= 3),
        **BTN,
        key="dlg_apply_2d_poly",
    )
    clear_clicked = c_clear.button(T("🗑️ Clear drawing", "🗑️ Limpiar trazo"), **BTN, key="dlg_clear_2d_poly")
    close_clicked = c_close.button(T("✖️ Close", "✖️ Cerrar"), **BTN, key="dlg_close_2d_poly")

    if apply_clicked and st.session_state.draft_polygon_2d:
        st.session_state.drawn_polygon_2d = [list(pt) for pt in st.session_state.draft_polygon_2d]
        st.session_state.drawn_filter_enabled = True
        st.session_state.draft_polygon_2d = None
        st.session_state.draft_vertices_2d = []
        st.session_state.draft_cleared = False
        st.session_state.last_deck_selection_raw = ""
        st.session_state.map_widget_nonce += 1
        st.session_state.aoi_dialog_open = False
        st.rerun()

    if clear_clicked:
        st.session_state.draft_polygon_2d = None
        st.session_state.draft_vertices_2d = []
        st.session_state.drawn_polygon_2d = None
        st.session_state.draft_cleared = True
        st.session_state.last_deck_selection_raw = ""
        st.session_state.map_widget_nonce += 1
        st.rerun()

    if close_clicked:
        st.session_state.aoi_dialog_open = False
        st.rerun()

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
            T(
                f"Enclosed zone ready ({len(draft_poly) - 1} vertices): **{sel_count:,}** cells and **{sel_fires:,}** hotspots inside.",
                f"Zona encerrada lista ({len(draft_poly) - 1} vértices): **{sel_count:,}** celdas y **{sel_fires:,}** focos dentro del dibujo.",
            )
        )
    elif len(st.session_state.draft_vertices_2d) > 0:
        faltan = 3 - len(st.session_state.draft_vertices_2d)
        st.info(
            T(
                f"Polygon in progress ({len(st.session_state.draft_vertices_2d)} points). Click **{faltan}** more point(s) on the map to close the zone.",
                f"Trazo poligonal en progreso ({len(st.session_state.draft_vertices_2d)} puntos). Haz clic en **{faltan}** punto(s) más del mapa para cerrar la zona.",
            )
        )
    else:
        st.info(
            T(
                "Click directly on the map above to place polygon vertices without duplicates.",
                "Haz clic directamente sobre el mapa superior para ir colocando los puntos del polígono sin duplicados.",
            )
        )


# 8D. COMPONENTES REUTILIZABLES (un solo código para dock, filtros y análisis)
def render_period_selector(prefix: str) -> None:
    modos = {
        "fecha_exacta": T("Day", "Día"),
        "mes_completo": T("Month", "Mes"),
        "anio_completo": T("Year", "Año"),
    }
    cur = st.session_state.temporal_mode
    sel = st.radio(
        T("Period", "Periodo"),
        list(modos),
        index=list(modos).index(cur),
        format_func=lambda k: modos[k],
        horizontal=True,
        label_visibility="collapsed",
        key=f"{prefix}_modo_{cur}",
    )
    a, m, d = st.session_state.cal_anio, st.session_state.cal_mes, st.session_state.cal_dia
    na, nm, nd = a, m, d
    if sel == "fecha_exacta":
        lo = date(min(anios_api), 1, 1)
        hi = min(date.today(), date(max(anios_api), 12, 31))
        cur_date = min(max(date(a, m, d), lo), hi)
        picked = st.date_input(
            T("Date", "Fecha"),
            value=cur_date,
            min_value=lo,
            max_value=hi,
            key=f"{prefix}_fecha_{cur_date.isoformat()}",
        )
        if picked is not None:
            na, nm, nd = picked.year, picked.month, picked.day
    else:
        cols = st.columns(2 if sel == "mes_completo" else 1)
        na = cols[0].selectbox(T("Year", "Año"), anios_api, index=anios_api.index(a), key=f"{prefix}_anio_{a}")
        if sel == "mes_completo":
            nm = cols[1].selectbox(
                T("Month", "Mes"),
                list(range(1, 13)),
                index=m - 1,
                format_func=month_name,
                key=f"{prefix}_mes_{m}",
            )
    nd = min(nd, calendar.monthrange(na, nm)[1])
    if sel != cur or (na, nm, nd) != (a, m, d):
        st.session_state.temporal_mode = sel
        st.session_state.cal_anio, st.session_state.cal_mes, st.session_state.cal_dia = na, nm, nd
        st.rerun()


def render_sensor_selector(prefix: str, show_joint_caption: bool = False) -> None:
    if len(insts_disponibles) < 2:
        return
    opts = insts_disponibles + ["BOTH"]
    current = selected_inst[0] if len(selected_inst) == 1 else "BOTH"
    sel = st.radio(
        "Sensor",
        opts,
        index=opts.index(current),
        format_func=lambda k: T("Both", "Ambos") if k == "BOTH" else k,
        horizontal=True,
        key=f"{prefix}_sensor_{current}",
    )
    if sel != current:
        st.session_state.active_inst = list(insts_disponibles) if sel == "BOTH" else [sel]
        st.session_state.active_sat = None
        st.rerun()
    if show_joint_caption and len(selected_inst) > 1:
        st.caption(
            T(
                "Showing combined MODIS + VIIRS detections grouped directly by the selected hexagon size.",
                "Mostrando detecciones conjuntas de MODIS + VIIRS agrupadas directamente por el tamaño de hexágono elegido.",
            )
        )


def render_aoi_controls(prefix: str) -> None:
    has_poly = st.session_state.drawn_polygon_2d is not None and len(st.session_state.drawn_polygon_2d) >= 3
    label = (
        T("✏️ Edit area of interest", "✏️ Editar área de interés")
        if has_poly
        else T("🖌️ Draw area of interest", "🖌️ Dibujar área de interés")
    )
    if st.button(label, type="secondary" if has_poly else "primary", key=f"{prefix}_aoi_open", **BTN):
        st.session_state.aoi_expander_open = True
        st.session_state.aoi_dialog_open = True
        st.session_state.draft_cleared = False
        st.session_state.last_deck_selection_raw = ""
        st.session_state.dialog_cam_lat = None
        st.session_state.dialog_cam_lon = None
        st.session_state.map_widget_nonce += 1
        st.rerun()
    if has_poly:
        c_en, c_del = st.columns([0.62, 0.38])
        enabled = st.session_state.drawn_filter_enabled
        en_val = c_en.checkbox(T("Filter by area", "Filtrar por área"), value=enabled, key=f"{prefix}_aoi_en_{enabled}")
        if en_val != enabled:
            st.session_state.aoi_expander_open = True
            st.session_state.drawn_filter_enabled = en_val
            st.rerun()
        if c_del.button(T("🗑️ Delete", "🗑️ Borrar"), key=f"{prefix}_aoi_del", **BTN):
            st.session_state.aoi_expander_open = True
            st.session_state.drawn_polygon_2d = None
            st.session_state.draft_polygon_2d = None
            st.session_state.draft_vertices_2d = []
            st.rerun()


def render_download_panel(prefix: str) -> None:
    if filtered_df.empty:
        st.info(T("No filtered records available for download.", "Sin registros filtrados para descargar."))
        return
    st.caption(
        f"**{len(filtered_df):,}** {T('cells', 'celdas')} (**{int(filtered_df['total_incendios'].sum()):,}** {T('hotspots', 'focos')}) · "
        f"{country_label(st.session_state.active_pais)} · {periodo_txt}"
    )
    fmt = st.radio(
        T("Format", "Formato"),
        ["CSV", "Excel", "GeoJSON"],
        horizontal=True,
        key=f"{prefix}_dl_fmt",
        label_visibility="collapsed",
    )
    polygon = st.session_state.drawn_polygon_2d if has_active_2d_drawing else None
    payload, ext, mime = build_export_payload(fmt, filtered_df, export_metadata, polygon)
    st.download_button(
        f"⬇️ {T('Download', 'Descargar')} {fmt}",
        data=payload,
        file_name=build_export_filename(
            st.session_state.active_pais,
            st.session_state.temporal_mode,
            fecha_iso_activa,
            st.session_state.cal_mes,
            st.session_state.cal_anio,
            ext,
            has_drawn_zone=has_active_2d_drawing,
        ),
        mime=mime,
        key=f"{prefix}_dl_btn",
        **DL,
    )


def render_map_legend() -> None:
    pal = THERMAL_PALETTES.get(st.session_state.palette_key, THERMAL_PALETTES["inferno"])
    stops = ", ".join(
        f"rgb({int(r)},{int(g)},{int(b)}) {int(s * 100)}%" for s, r, g, b in zip(pal["stops"], pal["r"], pal["g"], pal["b"])
    )
    st.markdown(
        f"<div class='map-legend'><b>{T('Relative intensity in current view', 'Intensidad relativa de esta vista')}</b>"
        f"<div class='bar' style='background:linear-gradient(90deg,{stops})'></div>"
        f"<div class='ends'><span>{T('low', 'baja')}</span><span>{T('high', 'alta')}</span></div>"
        f"<div class='note'>{T('Combines brightness, FRP, and hotspot count normalized <b>within the current view</b>: not comparable across periods or areas. Use 📊 Analysis for historical comparisons.', 'Combina brillo, FRP y focos, normalizados <b>dentro de la vista actual</b>: no es comparable entre periodos ni áreas. Para comparar en el tiempo usa 📊 Análisis.')}</div></div>",
        unsafe_allow_html=True,
    )


# 9. PANEL DE CONEXIÓN AL SERVIDOR BACKEND (Arriba a la Derecha, sin API Key)
if SHOW_SERVER_PANEL:
    with st.container(key="dock_servidor_derecho"):
        srv_badge = T("🟢 Live API", "🟢 API En Vivo") if is_live_api else T("🟠 Local Backup", "🟠 Respaldo Local")
        with st.expander(f"🔌 {T('Backend Server', 'Servidor Backend')} · {srv_badge}", expanded=False):
            new_api_url = st.text_input(
                T("External Backend Endpoint:", "Endpoint Backend Externo:"),
                value=st.session_state.api_url,
                key="w_api_endpoint",
            )
            col_sync, col_fb = st.columns([0.52, 0.48], vertical_alignment="center")
            sync = col_sync.button(T("🔄 Sync", "🔄 Sincronizar"), key="btn_sync_srv", **BTN)
            fb_val = col_fb.checkbox(
                T("Local backup", "Respaldo local"),
                value=st.session_state.allow_local_fallback,
                key="w_allow_fb",
            )
            latam_val = st.checkbox(
                T("Filter coordinates outside LATAM", "Filtrar coordenadas fuera de LATAM"),
                value=st.session_state.strict_latam,
                key="w_strict_latam",
            )
            limit_val = int(
                st.number_input(
                    T("Max records per query:", "Límite de registros por consulta:"),
                    min_value=1000,
                    max_value=500000,
                    step=10000,
                    value=int(st.session_state.api_limit),
                    key="w_api_limit",
                )
            )
            (st.success if is_live_api else st.warning)(connection_status)
            changed = (
                new_api_url != st.session_state.api_url
                or fb_val != st.session_state.allow_local_fallback
                or latam_val != st.session_state.strict_latam
                or limit_val != int(st.session_state.api_limit)
            )
            if changed or sync:
                st.session_state.api_url = new_api_url
                st.session_state.allow_local_fallback = fb_val
                st.session_state.strict_latam = latam_val
                st.session_state.api_limit = limit_val
                if sync:
                    for _fn in (
                        _fetch_api_catalogs_cached,
                        fetch_incendios_grid,
                        fetch_history,
                        build_clustered_hex_layer,
                        build_regional_or_globe_geodesic_grid,
                        load_local_firms_grid,
                    ):
                        _fn.clear()
                st.rerun()


# 10. BARRA SUPERIOR IZQUIERDA (PESTAÑAS + SELECTOR DE IDIOMA + CONTROLES DEL MAPA)
with st.container(key="dock_superior_izquierdo"):
    nav_choice = st.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(st.session_state.active_page),
        format_func=lambda p: (
            T("🗺️ Map", "🗺️ Mapa")
            if p == PAGE_MAP
            else T("📊 Analysis", "📊 Análisis")
            if p == PAGE_CAL
            else T("🔍 Filters", "🔍 Filtros")
        ),
        horizontal=True,
        label_visibility="collapsed",
        key="w_nav_radio",
    )
    if nav_choice != st.session_state.active_page:
        st.session_state.active_page = nav_choice
        st.rerun()

    if is_map_page:
        col_toggle, col_lang, col_badge = st.columns([0.48, 0.22, 0.30])
        if col_toggle.button(
            T("👁️ Hide", "👁️ Ocultar") if st.session_state.dock_visible else T("☰ Panels", "☰ Paneles"),
            key="btn_toggle_dock",
            **BTN,
        ):
            st.session_state.dock_visible = not st.session_state.dock_visible
            st.rerun()

        if col_lang.button(
            "🌐 ES" if st.session_state.lang == "en" else "🌐 EN",
            help=T("Switch interface language to Spanish", "Cambiar el idioma de la interfaz a inglés"),
            key="btn_toggle_lang_map",
            **BTN,
        ):
            st.session_state.lang = "es" if st.session_state.lang == "en" else "en"
            st.rerun()

        total_fires_badge = int(filtered_df["total_incendios"].sum()) if not filtered_df.empty else 0
        col_badge.markdown(
            f"<div style='background:rgba(17,21,30,0.88);border:1px solid rgba(255,255,255,0.15);"
            f"border-radius:10px;padding:7px 6px;text-align:center;color:#fbbf24;font-size:12px;font-weight:700;'>"
            f"{'🟢' if is_live_api else '🟠'} {total_fires_badge:,}</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.dock_visible:
            with st.expander(T("🌍 Countries & date", "🌍 Países y fecha"), expanded=False):
                st.caption(
                    f"{T('Active selection:', 'Selección activa:')} **{country_label(st.session_state.active_pais)}** · **{periodo_txt}**"
                )
                render_period_selector("dk")
                new_pais = st.selectbox(
                    T("Country", "País"),
                    paises_api,
                    index=paises_api.index(st.session_state.active_pais),
                    format_func=country_label,
                    key=f"dk_pais_{st.session_state.active_pais}",
                )
                if new_pais != st.session_state.active_pais:
                    st.session_state.active_pais = new_pais
                    st.rerun()
                st.caption(
                    T(
                        "More filters (FRP, brightness, satellite, risk) & natural-language search: **🔍 Filters** tab.",
                        "Más filtros (FRP, brillo, satélite, riesgo) y búsqueda en lenguaje natural: pestaña **🔍 Filtros**.",
                    )
                )
                if was_truncated:
                    st.warning(
                        T(
                            "Response truncated by record limit: displayed data is incomplete.",
                            "Respuesta truncada por el límite de registros: los datos están incompletos.",
                        )
                    )

            with st.expander(
                f"{T('📍 Area of interest', '📍 Área de interés')}{T(' · active', ' · activa') if has_active_2d_drawing else ''}",
                expanded=bool(st.session_state.aoi_expander_open or has_active_2d_drawing),
            ):
                render_aoi_controls("dk")

            with st.expander(
                f"📥 {T('Download', 'Descargar')} ({len(filtered_df):,} {T('cells', 'celdas')})",
                expanded=False,
            ):
                render_download_panel("dk")

            with st.expander(T("⚙️ Options", "⚙️ Opciones"), expanded=False):
                render_sensor_selector("opt", show_joint_caption=True)
                st.session_state.zoom_tier_num = st.select_slider(
                    T("Hexagon size", "Tamaño del hexágono"),
                    options=[1, 2, 3, 4, 5],
                    value=st.session_state.zoom_tier_num,
                    format_func=lambda k: ZOOM_TIERS[k]["short_en" if st.session_state.lang == "en" else "short_es"],
                    disabled=st.session_state.show_all_hex,
                    key="w_zoom_tier_slider",
                )
                palette_keys = list(THERMAL_PALETTES.keys())
                st.session_state.palette_key = st.selectbox(
                    T("Color palette", "Paleta"),
                    palette_keys,
                    index=palette_keys.index(st.session_state.palette_key),
                    format_func=lambda k: THERMAL_PALETTES[k]["name_en" if st.session_state.lang == "en" else "name_es"],
                    key="w_palette_select",
                )
                proj_options = ["2D", "3D"]
                sel_proj = st.radio(
                    T("Projection", "Proyección"),
                    proj_options,
                    index=proj_options.index(st.session_state.projection_mode),
                    format_func=lambda k: T("Flat (2D)", "Plano (2D)") if k == "2D" else T("3D Sphere (Globe)", "Esfera 3D (Globo)"),
                    horizontal=True,
                    key="w_proj_radio",
                )
                if sel_proj != st.session_state.projection_mode:
                    st.session_state.projection_mode = sel_proj
                    st.rerun()
                if st.toggle(T("Advanced options", "Opciones avanzadas"), key="w_adv_toggle"):
                    st.session_state.smooth_pct = st.slider(
                        T("Neighbor smoothing (%)", "Suavizado entre vecinos (%)"),
                        0,
                        60,
                        st.session_state.smooth_pct,
                        5,
                        key="w_smooth_slider",
                    )
                    st.session_state.terrain_depth_3d = st.checkbox(
                        T("3D extrusion by hotspots", "Extrusión 3D por focos"),
                        value=st.session_state.terrain_depth_3d,
                        key="w_extrude_cb",
                    )
                    st.session_state.sync_camera_zoom = st.checkbox(
                        T("Fit camera to hexagon size", "Ajustar cámara al tamaño"),
                        value=st.session_state.sync_camera_zoom,
                        key="w_sync_camera_cb",
                    )
                    st.session_state.show_base_grid = st.checkbox(
                        T("Reference mesh", "Malla de referencia"),
                        value=st.session_state.show_base_grid,
                        key="w_base_grid",
                    )
                    sel_multi = st.checkbox(
                        T("Show all sizes simultaneously", "Mostrar todos los tamaños a la vez"),
                        value=st.session_state.show_all_hex,
                        key="w_show_all_hex_cb",
                    )
                    if sel_multi != st.session_state.show_all_hex:
                        st.session_state.show_all_hex = sel_multi
                        st.rerun()
                    if st.session_state.projection_mode == "3D":
                        st.session_state.show_world_map_3d = st.checkbox(
                            T("Country borders (3D)", "Fronteras (3D)"),
                            value=st.session_state.show_world_map_3d,
                            key="w_world_3d",
                        )
                        st.session_state.show_graticule_3d = st.checkbox(
                            T("Parallels & meridians (3D)", "Paralelos y meridianos (3D)"),
                            value=st.session_state.show_graticule_3d,
                            key="w_grat_3d",
                        )
    else:
        if st.button(
            "🌐 Language: English → Cambiar a Español" if st.session_state.lang == "en" else "🌐 Idioma: Español → Switch to English",
            key="btn_toggle_lang_subpage",
            **BTN,
        ):
            st.session_state.lang = "es" if st.session_state.lang == "en" else "en"
            st.rerun()

if st.session_state.aoi_dialog_open:
    open_2d_drawing_dialog(base_filtered_df)


# ==============================================================================
# PÁGINA: FILTROS Y BÚSQUEDA
# ==============================================================================
def _num_filter(label: str, state_key: str, prefix: str) -> None:
    cur = st.session_state[state_key]
    val = st.number_input(
        label,
        min_value=0.0,
        max_value=5000.0,
        value=float(cur) if cur is not None else 0.0,
        step=5.0,
        key=f"{prefix}_{state_key}_{cur}",
        help=T("0 = no limit", "0 = sin límite"),
    )
    new = float(val) if val > 0 else None
    if new != cur:
        st.session_state[state_key] = new
        st.rerun()

def _run_smart_search(query: str) -> None:
    st.session_state.smart_search_query = query
    parse_and_apply_smart_search(
        query,
        available_paises=paises_api,
        available_anios=anios_api,
        available_sats=sats_catalogo,
        available_insts=insts_disponibles,
    )


def render_filters_page() -> None:
    st.markdown(f"## {T('🔍 Filters & Smart Search', '🔍 Filtros y búsqueda')}")
    st.caption(connection_status)
    st.caption(
        T(
            "Type a natural-language query and press Enter, e.g. **“Colombia August 2024 VIIRS FRP greater than 50”**, "
            "**“Argentina 2025-08-15 brightness > 350”**, or **“Brazil 2023 full year”**. You can also adjust filters manually below.",
            "Escribe una consulta y pulsa Enter, por ejemplo: **«Colombia agosto 2024 VIIRS FRP mayor a 50»**, "
            "**«Argentina 15/08/2025 brillo > 350»** o **«Brasil 2023 año completo»**. También puedes ajustar los filtros a mano.",
        )
    )

    with st.container(key="panel_buscador_inteligente"):
        c_q, c_clear = st.columns([0.84, 0.16], vertical_alignment="bottom")
        query_input = c_q.text_input(
            T("Search", "Búsqueda"),
            value="",
            placeholder=T(
                "Ex: Colombia August 2024 VIIRS FRP greater than 50",
                "Ej: Colombia agosto 2024 VIIRS FRP mayor a 50",
            ),
            key=f"pg_search_{st.session_state.search_nonce}",
            label_visibility="collapsed",
        )
        if c_clear.button(T("🧹 Clear all", "🧹 Limpiar todo"), key="pg_clear_all", **BTN):
            for _k in (
                "active_riesgo", "active_sat", "active_inst", "frp_min", "frp_max",
                "brillo_min", "brillo_max", "drawn_polygon_2d", "draft_polygon_2d",
            ):
                st.session_state[_k] = None
            st.session_state.active_pais = "Todos"
            st.session_state.draft_vertices_2d = []
            st.session_state.smart_search_query = ""
            st.session_state.last_parsed_entities = []
            st.session_state.temporal_mode = "mes_completo"
            st.session_state.search_nonce += 1
            st.rerun()
        if query_input and query_input != st.session_state.smart_search_query:
            _run_smart_search(query_input)
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
                f"{T('Parsed:', 'Entendí:')}</span>{badges_html}</div>",
                unsafe_allow_html=True,
            )

    with st.container(key="panel_filtros_manuales"):
        col_left, col_right = st.columns([0.55, 0.45], gap="large")
        with col_left:
            st.markdown(f"### {T('🛰️ Filters', '🛰️ Filtros')}")
            new_pais = st.selectbox(
                T("Country", "País"),
                paises_api,
                index=paises_api.index(st.session_state.active_pais),
                format_func=country_label,
                key=f"pg_pais_{st.session_state.active_pais}",
            )
            if new_pais != st.session_state.active_pais:
                st.session_state.active_pais = new_pais
                st.rerun()

            render_sensor_selector("pg")

            c_sat, c_risk = st.columns(2)
            new_sat = c_sat.multiselect(
                T("Satellite", "Satélite"),
                sats_disponibles,
                default=selected_sat,
                key="pg_sat_" + "|".join(selected_sat),
            )
            if new_sat != selected_sat:
                st.session_state.active_sat = new_sat or None
                st.rerun()
            new_risk = c_risk.multiselect(
                T("Intensity (source level)", "Intensidad (según la fuente)"),
                riesgos_disponibles,
                default=selected_riesgo,
                format_func=risk_label,
                key="pg_risk_" + "|".join(selected_riesgo),
            )
            if new_risk != selected_riesgo:
                st.session_state.active_riesgo = new_risk or None
                st.rerun()

            f1, f2 = st.columns(2)
            with f1:
                _num_filter(T("Minimum FRP (MW)", "FRP mínimo (MW)"), "frp_min", "pg")
                _num_filter(T("Minimum Brightness (K)", "Brillo mínimo (K)"), "brillo_min", "pg")
            with f2:
                _num_filter(T("Maximum FRP (MW)", "FRP máximo (MW)"), "frp_max", "pg")
                _num_filter(T("Maximum Brightness (K)", "Brillo máximo (K)"), "brillo_max", "pg")

        with col_right:
            st.markdown(f"### {T('📅 Period & Area', '📅 Periodo y área')}")
            render_period_selector("pg")
            render_aoi_controls("pg")

    with st.container(key="panel_exportacion_filtros"):
        st.markdown(f"### {T('📥 Export filtered records', '📥 Exportar registros filtrados')}")
        render_download_panel("pg")

    with st.container(key="panel_preview_filtros"):
        st.markdown(f"### {T('📋 Summary & Export Preview', '📋 Resumen y vista previa de exportación')}")
        if was_truncated:
            st.warning(
                T(
                    "The response hit the record limit: results are incomplete. Shorten the selected period.",
                    "La respuesta alcanzó el límite de registros: los resultados están incompletos. Reduce el periodo.",
                )
            )
        if filtered_df.empty:
            st.info(T("No records match the current filter combination.", "No hay registros para la combinación actual de filtros."))
            clear_loading_status()
            return

        weights = filtered_df["total_incendios"].to_numpy(dtype=np.float64)
        b_w = _wavg(filtered_df["brillo_promedio"], weights)
        f_w = _wavg(filtered_df["frp_promedio"], weights)
        na_str = T("n/a", "s/d")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(T("Hotspots / detections", "Focos / detecciones"), f"{int(weights.sum()):,}")
        k2.metric(T("Active cells", "Celdas activas"), f"{len(filtered_df):,}")
        k3.metric(T("Weighted brightness", "Brillo ponderado"), na_str if np.isnan(b_w) else f"{b_w:.1f} K")
        k4.metric(T("Weighted FRP", "FRP ponderado"), na_str if np.isnan(f_w) else f"{f_w:.1f} MW")

        st.caption(
            T(
                "Sample of exact records ready for download (sorted by hotspot count). Open **📊 Analysis** for time-series charts and regional rankings.",
                "Muestra de los registros exactos listos para descargar (ordenados por mayor número de focos). Para explorar gráficas temporales y ranking regional, abre **📊 Análisis**.",
            )
        )
        preview_df = prepare_export_dataframe(
            filtered_df.sort_values("total_incendios", ascending=False).head(25)
        )
        st.dataframe(preview_df, height=240, **DF)

    clear_loading_status()


# ==============================================================================
# PÁGINA: ANÁLISIS
# ==============================================================================
def add_period_start(df: pd.DataFrame, gran: str) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        d["inicio"] = pd.Series(dtype="datetime64[ns]")
        return d
    anio = d["anio"].astype(int)
    idx = d["idx"].astype(int)
    if gran == "semanal":
        d["inicio"] = pd.to_datetime({"year": anio, "month": 1, "day": 1}) + pd.to_timedelta((idx - 1) * 7, unit="D")
    else:
        d["inicio"] = pd.to_datetime({"year": anio, "month": idx, "day": 1})
    return d


def render_calendar_page() -> None:
    st.markdown(f"## {T('📊 Activity & Trend Analysis', '📊 Análisis de actividad y tendencias')}")
    st.caption(
        T(
            "Temporal and spatial exploration of the selected region: active period breakdown, harmonized MODIS/VIIRS trends, multi-year heatmap, and statistical anomaly detection.",
            "Exploración temporal y espacial del área elegida: distribución del periodo seleccionado, evolución comparada MODIS/VIIRS, mapa de calor multianual y detección estadística de anomalías.",
        )
    )

    with st.expander(
        f"{T('📌 Active period summary', '📌 Resumen del periodo activo')} ({country_label(st.session_state.active_pais)} · {periodo_txt})",
        expanded=True,
    ):
        if filtered_df.empty:
            st.info(T("No active records for the current map filters.", "Sin registros activos para los filtros seleccionados en el mapa."))
        else:
            w_act = filtered_df["total_incendios"].to_numpy(dtype=np.float64)
            b_act = _wavg(filtered_df["brillo_promedio"], w_act)
            f_act = _wavg(filtered_df["frp_promedio"], w_act)
            na_str = T("n/a", "s/d")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(T("Hotspots in active period", "Focos en periodo activo"), f"{int(w_act.sum()):,}")
            m2.metric(T("Active cells", "Celdas activas"), f"{len(filtered_df):,}")
            m3.metric(T("Weighted brightness", "Brillo ponderado"), na_str if np.isnan(b_act) else f"{b_act:.1f} K")
            m4.metric(T("Weighted FRP", "FRP ponderado"), na_str if np.isnan(f_act) else f"{f_act:.1f} MW")

            col_chart, col_tbl = st.columns([0.45, 0.55], gap="large")
            with col_chart:
                if st.session_state.temporal_mode == "anio_completo":
                    st.caption(T("Hotspots by month (active filters)", "Focos por mes (periodo y filtros activos)"))
                    dist = filtered_df.groupby("mes")["total_incendios"].sum().reindex(range(1, 13), fill_value=0)
                    dist.index = [month_name(m)[:3] for m in dist.index]
                elif st.session_state.temporal_mode == "mes_completo" and (filtered_df["fecha"] != "").any():
                    st.caption(T("Hotspots by day (active filters)", "Focos por día (periodo y filtros activos)"))
                    dist = filtered_df[filtered_df["fecha"] != ""].groupby("fecha")["total_incendios"].sum()
                else:
                    st.caption(T("Hotspots by satellite", "Focos por satélite"))
                    dist = filtered_df.groupby("satelite")["total_incendios"].sum()
                st.bar_chart(dist, height=230)

            with col_tbl:
                st.caption(T("Top regional clusters", "Principales clústeres regionales"))
                mean_lat = float(filtered_df["lat_grid"].mean())
                clusters = build_clustered_hex_layer(
                    filtered_df,
                    radius_deg=0.50,
                    tier_label="Regional",
                    palette_key=st.session_state.palette_key,
                    smooth_weight=st.session_state.smooth_pct / 100.0,
                    is_globe=False,
                    lon_scale=round(float(np.cos(np.deg2rad(mean_lat))), 2),
                    lang=st.session_state.lang,
                )
                st.dataframe(
                    clusters[["pais", "instrumento", "lat_grid", "lon_grid", "total_incendios", "indice_termico", "brillo_txt", "frp_txt"]]
                    .sort_values("total_incendios", ascending=False).head(20),
                    height=230,
                    **DF,
                )

    st.markdown("---")
    st.markdown(f"### {T('🗓️ Multi-Year Historical Analysis', '🗓️ Análisis histórico multianual')}")

    years_all = sorted(anios_api)
    if len(years_all) > 1:
        cur_rng = st.session_state.get("cal_range_w")
        if (
            not isinstance(cur_rng, (tuple, list))
            or len(cur_rng) != 2
            or cur_rng[0] not in years_all
            or cur_rng[1] not in years_all
            or cur_rng[0] > cur_rng[1]
        ):
            st.session_state.cal_range_w = (years_all[0], years_all[-1])
        else:
            st.session_state.cal_range_w = (int(cur_rng[0]), int(cur_rng[1]))

    series_opts = [SERIE_MEDIA, SERIE_MODIS, SERIE_VIIRS_CONV]
    if st.session_state.cal_series not in series_opts:
        st.session_state.cal_series = SERIE_MEDIA

    c1, c2, c3 = st.columns([0.4, 0.28, 0.32])
    with c1:
        if len(years_all) > 1:
            lo, hi = st.select_slider(
                T("Years to analyze", "Años a analizar"),
                options=years_all,
                value=st.session_state.cal_range_w,
                key="cal_range_w",
            )
        else:
            lo = hi = years_all[0]
            st.caption(f"{T('Available year:', 'Año disponible:')} {lo}")
    with c2:
        metric = st.radio(
            T("Metric", "Métrica"),
            ["fires", "frp"],
            format_func=lambda k: T("Hotspots", "Focos") if k == "fires" else T("Total FRP (MW)", "FRP total (MW)"),
            horizontal=True,
            key="cal_metric",
        )
    with c3:
        serie_sel = st.selectbox(
            T("Series for heatmap & anomalies", "Serie para mapa de calor y anomalías"),
            series_opts,
            format_func=series_display_name,
            key="cal_series",
        )

    zona = T("drawn area", "área dibujada") if has_active_2d_drawing else T("full coverage", "toda la cobertura")
    c_info, c_aoi = st.columns([0.62, 0.38], vertical_alignment="center")
    c_info.caption(f"📍 **{country_label(st.session_state.active_pais)}** · {zona} · {T('Range', 'Rango')} **{lo} – {hi}**")
    with c_aoi:
        render_aoi_controls("cal")

    years_sel = [y for y in years_all if lo <= y <= hi]
    hist, _live, trunc_years = fetch_history(
        st.session_state.api_url,
        st.session_state.allow_local_fallback,
        st.session_state.active_pais,
        tuple(years_sel),
        st.session_state.strict_latam,
        int(st.session_state.api_limit),
        lang=st.session_state.lang,
    )
    if has_active_2d_drawing and not hist.empty:
        mask = points_in_polygon(
            hist["lon_grid"].to_numpy(dtype=np.float64), hist["lat_grid"].to_numpy(dtype=np.float64), st.session_state.drawn_polygon_2d
        )
        hist = hist[mask]
    if trunc_years:
        st.warning(
            T(
                f"The API hit the record limit in {', '.join(map(str, trunc_years))}: those years may be incomplete and bias anomalies. Raise the limit or narrow the area.",
                f"La API devolvió exactamente el límite de registros en {', '.join(map(str, trunc_years))}: esos años pueden estar incompletos y sesgar anomalías. Sube el límite (panel de conexión) o acota el área.",
            )
        )
    if hist.empty:
        st.info(T("No data available for the selected country, area, and years.", "No hay datos para el país, área y años elegidos."))
        clear_loading_status()
        return

    hist_t, gran = prepare_history(hist)
    serie = fill_missing_periods(aggregate_series(hist_t, metric))
    if serie.empty:
        st.info(T("Data does not contain recognizable MODIS or VIIRS detections.", "Los datos no contienen detecciones MODIS ni VIIRS reconocibles."))
        clear_loading_status()
        return

    fams_present = sorted(serie["familia"].unique().tolist())
    harm_series, info = harmonize_to_modis(serie)
    xlab = T("Week of year", "Semana del año") if gran == "semanal" else T("Month", "Mes")

    if info["has_ratio"]:
        chosen = harm_series[serie_sel]
        chosen_name = series_display_name(serie_sel)
        if not info["ok"]:
            st.warning(
                T(
                    f"Low overlap between MODIS and VIIRS ({info['n_solape']} periods; recommended ≥ {MIN_SOLAPE_PERIODOS}). MODIS-scale conversion is preliminary.",
                    f"Solapamiento bajo entre MODIS y VIIRS ({info['n_solape']} periodos; recomendado ≥ {MIN_SOLAPE_PERIODOS}). La conversión a formato MODIS es preliminar.",
                )
            )
    else:
        fb_key = SERIE_MODIS if "MODIS" in fams_present else "VIIRS"
        chosen = (
            harm_series[SERIE_MODIS]
            if fb_key == SERIE_MODIS
            else serie[serie["familia"] == "VIIRS"][["anio", "idx", "valor"]]
        )
        chosen_name = "MODIS" if fb_key == SERIE_MODIS else "VIIRS"
        st.info(
            T(
                f"Only {chosen_name} data exists in this range: both sensors are needed for harmonization.",
                f"Solo hay datos de {chosen_name} en este rango: no hay ambos sensores para armonizar.",
            )
        )

    unit = T("hotspots", "focos") if metric == "fires" else "MW"
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Total · {chosen_name}", f"{chosen['valor'].sum():,.0f} {unit}")
    k2.metric(
        T("VIIRS → MODIS factor", "Factor VIIRS → MODIS"),
        f"×{info['ratio']:.3f}" if info["has_ratio"] else "—",
        help=T(
            "Median of MODIS÷VIIRS across periods where both sensors were active. Scales VIIRS to MODIS equivalence.",
            "Mediana de MODIS÷VIIRS en los periodos con ambos sensores activos. Multiplica VIIRS para expresarlo en escala MODIS.",
        ),
    )
    k3.metric(T("Overlapping periods", "Periodos solapados"), f"{info['n_solape']:,}")
    k4.metric(
        T("Temporal resolution", "Resolución temporal"),
        T("Weekly", "Semanal") if gran == "semanal" else T("Monthly", "Mensual"),
    )
    if info["has_ratio"]:
        st.caption(
            T(
                f"VIIRS → MODIS conversion factor: median ×{info['ratio']:.3f} (interquartile range ×{info['p25']:.3f} – ×{info['p75']:.3f}).",
                f"Factor de conversión VIIRS → MODIS: mediana ×{info['ratio']:.3f} (rango intercuartílico ×{info['p25']:.3f} – ×{info['p75']:.3f}).",
            )
        )

    st.markdown(f"### 🔥 {T('Activity Calendar', 'Calendario de actividad')} · {chosen_name}")
    st.caption(
        T(
            "Each cell is one period. The color scale is **identical across all years**, making colors directly comparable.",
            "Cada celda es un periodo. La escala de color es **la misma para todos los años**, así que los colores sí son comparables.",
        )
    )
    vmax = float(np.nanpercentile(chosen["valor"], 98)) if len(chosen) else 1.0
    vmax = vmax if vmax > 0 else 1.0
    heat = (
        alt.Chart(chosen)
        .mark_rect()
        .encode(
            x=alt.X("idx:O", title=xlab),
            y=alt.Y("anio:O", title=None, sort="descending"),
            color=alt.Color("valor:Q", title=unit, scale=alt.Scale(scheme="inferno", type="sqrt", domain=[0, vmax], clamp=True)),
            tooltip=[
                alt.Tooltip("anio:O", title=T("Year", "Año")),
                alt.Tooltip("idx:O", title=xlab),
                alt.Tooltip("valor:Q", title=unit, format=",.0f"),
            ],
        )
        .properties(height=max(120, 30 * chosen["anio"].nunique()))
    )
    st.altair_chart(heat, **ALT)

    st.markdown(f"### 📈 {T('Time Series by Sensor (MODIS scale)', 'Serie temporal por sensor (escala MODIS)')}")
    lbl_modis = series_display_name(SERIE_MODIS)
    lbl_viirs = series_display_name(SERIE_VIIRS_CONV)
    lbl_media = series_display_name(SERIE_MEDIA)

    chart_frames = []
    for s_key in (SERIE_MODIS, SERIE_VIIRS_CONV, SERIE_MEDIA):
        df_s = harm_series[s_key]
        if not df_s.empty:
            chart_frames.append(df_s.assign(serie=series_display_name(s_key))[["serie", "anio", "idx", "valor"]])

    if not chart_frames:
        chart_frames.append(
            serie.rename(columns={"familia": "serie"})[["serie", "anio", "idx", "valor"]]
        )

    raw_long = pd.concat(chart_frames, ignore_index=True)
    long_df = add_period_start(raw_long, gran)

    wide_hover = long_df.pivot_table(index=["anio", "idx", "inicio"], columns="serie", values="valor", aggfunc="first").reset_index()
    raw_viirs_period = add_period_start(serie[serie["familia"] == "VIIRS"][["anio", "idx", "valor"]], gran)
    if not raw_viirs_period.empty:
        wide_hover = wide_hover.merge(
            raw_viirs_period[["anio", "idx", "valor"]].rename(columns={"valor": "VIIRS_real"}),
            on=["anio", "idx"],
            how="left",
        )
    else:
        wide_hover["VIIRS_real"] = np.nan

    for col_s in (lbl_modis, lbl_viirs, lbl_media, "VIIRS_real"):
        if col_s not in wide_hover.columns:
            wide_hover[col_s] = np.nan

    wk_prefix = T("Wk ", "Sem ")
    wide_hover["momento_label"] = [
        f"{wk_prefix + str(int(i)) + ' · ' if gran == 'semanal' else month_name(int(i)) + ' '}{period_start(a, i, gran):%d/%m/%Y} – {period_end(a, i, gran):%d/%m/%Y}"
        for a, i in zip(wide_hover["anio"], wide_hover["idx"])
    ]
    wide_hover = wide_hover.sort_values("inicio").reset_index(drop=True)

    n_pts = max(len(wide_hover), 1)
    expanded_width = max(1250, n_pts * (22 if gran == "semanal" else 42))

    nearest_sel = alt.selection_point(
        fields=["inicio"],
        nearest=True,
        on="mouseover",
        empty=False,
        clear="mouseout",
    )

    series_domain = [lbl_modis, lbl_viirs, lbl_media]
    base_lines = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("inicio:T", title=None, axis=alt.Axis(format="%b %Y", tickCount="month", labelAngle=-35)),
            y=alt.Y("valor:Q", title=f"{unit} ({T('MODIS scale', 'escala MODIS')})"),
            color=alt.Color(
                "serie:N",
                title=T("Series", "Serie"),
                scale=alt.Scale(domain=series_domain, range=["#38bdf8", "#fb923c", "#f8fafc"]),
            ),
            strokeDash=alt.StrokeDash(
                "serie:N",
                title=T("Series", "Serie"),
                scale=alt.Scale(domain=series_domain, range=[[1, 0], [5, 3], [1, 0]]),
            ),
            strokeWidth=alt.condition(alt.datum.serie == lbl_media, alt.value(2.6), alt.value(1.8)),
        )
    )

    hover_points = (
        alt.Chart(long_df)
        .mark_circle(size=75, stroke="#0f172a", strokeWidth=1.5)
        .encode(
            x="inicio:T",
            y="valor:Q",
            color=alt.Color(
                "serie:N",
                scale=alt.Scale(domain=series_domain, range=["#38bdf8", "#fb923c", "#f8fafc"]),
                legend=None,
            ),
            opacity=alt.condition(nearest_sel, alt.value(1.0), alt.value(0.0)),
        )
    )

    hover_rule = (
        alt.Chart(wide_hover)
        .mark_rule(color="#94a3b8", strokeWidth=1.4, strokeDash=[4, 3])
        .encode(
            x="inicio:T",
            opacity=alt.condition(nearest_sel, alt.value(0.95), alt.value(0.0)),
            tooltip=[
                alt.Tooltip("momento_label:N", title=T("Exact period", "Momento exacto")),
                alt.Tooltip(f"{lbl_media}:Q", title=f"{T('Mean', 'Media')} ({unit})", format=",.1f"),
                alt.Tooltip(f"{lbl_modis}:Q", title=f"MODIS ({unit})", format=",.1f"),
                alt.Tooltip(f"{lbl_viirs}:Q", title=f"{lbl_viirs} ({unit})", format=",.1f"),
                alt.Tooltip("VIIRS_real:Q", title=f"{T('Raw unscaled VIIRS', 'VIIRS real sin escalar')} ({unit})", format=",.0f"),
            ],
        )
        .add_params(nearest_sel)
    )

    combined_ts_chart = (
        alt.layer(base_lines, hover_points, hover_rule)
        .properties(width=expanded_width, height=340)
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(combined_ts_chart, use_container_width=False)
    st.caption(
        T(
            "Scroll the bottom bar to pan through the expanded timeline. Hover over any point for exact values (**Mean**, **MODIS**, **VIIRS MODIS-scale**, and **Raw VIIRS**).",
            "Desliza la barra inferior para recorrer el histórico ampliado. Pasa el cursor sobre cualquier punto para ver el detalle exacto de ese periodo (**Media**, **MODIS**, **VIIRS formato MODIS** y **VIIRS real**).",
        )
    )

    st.markdown(f"### ⚠️ {T('Unusual Conditions & Critical Periods', 'Condiciones inusuales y periodos críticos')}")
    focus_opts = sorted(years_sel, reverse=True)
    if st.session_state.cal_focus_year not in focus_opts:
        st.session_state.cal_focus_year = focus_opts[0]
    anio_foco = st.selectbox(T("Year to evaluate", "Año a evaluar"), focus_opts, key="cal_focus_year")
    anom, anom_msg = compute_anomalies(chosen, int(anio_foco))
    if anom_msg:
        st.info(anom_msg)
    elif anom.empty:
        st.info(T("Not enough reference periods to evaluate anomalies.", "No hay periodos con referencia suficiente para evaluar."))
    else:
        crit = anom[anom["critico"]]
        st.caption(
            T(
                f"Each bar compares {anio_foco} against the same period across the other {anom['n_ref'].max()} years (z-score on log scale). Red: z ≥ {Z_CRITICO:g}, significantly above normal.",
                f"Cada barra compara el periodo de {anio_foco} con el mismo periodo de los otros {anom['n_ref'].max()} años (z en escala logarítmica). Rojo: z ≥ {Z_CRITICO:g}, es decir, claramente por encima de lo habitual.",
            )
        )
        bars = (
            alt.Chart(anom)
            .mark_bar()
            .encode(
                x=alt.X("idx:O", title=xlab),
                y=alt.Y("z:Q", title=T("Anomaly (z)", "Anomalía (z)")),
                color=alt.condition(alt.datum.z >= Z_CRITICO, alt.value("#ef4444"), alt.value("#64748b")),
                tooltip=[
                    alt.Tooltip("idx:O", title=xlab),
                    alt.Tooltip("observado:Q", title=T("Observed", "Observado"), format=",.0f"),
                    alt.Tooltip("esperado:Q", title=T("Expected", "Esperado"), format=",.0f"),
                    alt.Tooltip("z:Q", format=".2f"),
                ],
            )
        )
        rule = alt.Chart(pd.DataFrame({"y": [Z_CRITICO]})).mark_rule(strokeDash=[4, 4], color="#f87171").encode(y="y:Q")
        st.altair_chart(bars + rule, **ALT)
        if crit.empty:
            st.success(T(f"No periods in {anio_foco} exceeded the anomaly threshold.", f"Ningún periodo de {anio_foco} superó el umbral de anomalía."))
        else:
            st.warning(T(f"{len(crit)} critical period(s) in {anio_foco}.", f"{len(crit)} periodo(s) críticos en {anio_foco}."))
            tabla = crit.sort_values("z", ascending=False).copy()
            col_per = T("Period", "Periodo")
            col_obs = f"{T('Observed', 'Observado')} ({unit})"
            col_esp = f"{T('Expected', 'Esperado')} ({unit})"
            tabla[col_per] = [
                f"{period_start(anio_foco, i, gran):%d/%m} – {period_end(anio_foco, i, gran):%d/%m}" for i in tabla["idx"]
            ]
            tabla = tabla.rename(columns={"observado": col_obs, "esperado": col_esp, "z": "z"})
            st.dataframe(
                tabla[[col_per, col_obs, col_esp, "z"]].round(1).reset_index(drop=True),
                height=240,
                **DF,
            )

    raw_viirs_export = serie[serie["familia"] == "VIIRS"].assign(serie=T("VIIRS (unscaled)", "VIIRS (sin escalar)"))[
        ["serie", "anio", "idx", "valor"]
    ]
    export_series_df = pd.concat([long_df.drop(columns=["inicio"], errors="ignore"), raw_viirs_export], ignore_index=True)
    tidy = add_period_start(export_series_df, gran)
    tidy["periodo_inicio"] = tidy["inicio"].dt.strftime("%Y-%m-%d")
    st.download_button(
        T("⬇️ Download series (CSV)", "⬇️ Descargar series (CSV)"),
        data=tidy[["serie", "anio", "idx", "periodo_inicio", "valor"]].to_csv(index=False).encode("utf-8-sig"),
        file_name=f"series_{st.session_state.active_pais}_{lo}-{hi}_{metric}.csv".replace(" ", "_"),
        mime="text/csv",
        key="cal_dl_series",
    )
    with st.expander(T("ℹ️ Methodology & limitations", "ℹ️ Cómo se calcula y qué limitaciones tiene")):
        if st.session_state.lang == "en":
            st.markdown(
                f"""
- **Two distinct sensors.** MODIS (1 km, since 2000) and VIIRS (375 m, since 2012) measure differently: VIIRS detects smaller fires and may split one fire into multiple detections. **They are never summed raw** in historical analysis.
- **VIIRS (MODIS scale) & Mean.** Estimated via the median MODIS÷VIIRS ratio across overlapping active periods (minimum {MIN_SOLAPE_PERIODOS}). VIIRS×ratio expresses VIIRS on the MODIS scale, and **Mean (MODIS & VIIRS)** averages both when present.
- **Zeros.** The API only returns cells with active detections; periods without rows count as 0 from each sensor's first active year up to the last observed period of each year.
- **Anomaly score.** Compares `log(1+value)` of the evaluated year against the mean and std dev of the same period across reference years (minimum {MIN_ANIOS_REFERENCIA}; std dev floor 0.25). Critical threshold: z ≥ {Z_CRITICO:g}.
- **Scope.** Thermal detections ≠ burned area or future fire risk. Cloud cover, smoke, scan angle, and overpass times affect sensor visibility.
- **Resolution.** {'Weekly (7-day windows starting Jan 1)' if gran == 'semanal' else 'Monthly (source lacks daily timestamps)'}.
"""
            )
        else:
            st.markdown(
                f"""
- **Dos sensores distintos.** MODIS (1 km, desde 2000) y VIIRS (375 m, desde 2012) no cuentan lo mismo: VIIRS ve fuegos más pequeños y puede partir un mismo fuego en varias detecciones. **Nunca se suman sin escalar** en análisis histórico; cada sensor se agrega por separado.
- **VIIRS (formato MODIS) y Media.** Se estima la razón mediana MODIS÷VIIRS en los periodos donde ambos tuvieron actividad (mínimo {MIN_SOLAPE_PERIODOS}). VIIRS×razón expresa VIIRS en la escala de MODIS, y **Media (MODIS y VIIRS)** promedia ambas series cuando coinciden (o toma la disponible).
- **Ceros.** La API solo devuelve celdas con detección; un periodo sin filas cuenta como 0 desde el primer año con datos de ese sensor hasta el último periodo observado de cada año.
- **Anomalía.** Para cada periodo se compara `log(1+valor)` del año evaluado con la media y desviación del mismo periodo en los demás años (mínimo {MIN_ANIOS_REFERENCIA}); la desviación tiene un piso de 0.25 para no exagerar periodos casi constantes. Crítico: z ≥ {Z_CRITICO:g}.
- **Qué no es.** Detecciones térmicas ≠ superficie quemada ni riesgo futuro. Nubes, humo, ángulo de escaneo y horarios de paso afectan lo que ve cada sensor.
- **Cobertura.** Resolución {'semanal (semanas de 7 días desde el 1 de enero)' if gran == 'semanal' else 'mensual (la fuente no entrega fecha diaria)'}.
"""
            )

    clear_loading_status()


# ==============================================================================
# DESPACHO DE PÁGINAS
# ==============================================================================
if st.session_state.active_page == PAGE_FILT:
    render_filters_page()
elif st.session_state.active_page == PAGE_CAL:
    render_calendar_page()
else:
    smooth_weight = st.session_state.smooth_pct / 100.0
    zoom_tier_num = st.session_state.zoom_tier_num
    active_tier = ZOOM_TIERS[zoom_tier_num]
    active_tier_label = active_tier["label_en" if st.session_state.lang == "en" else "label_es"]
    show_all_hex = st.session_state.show_all_hex
    sync_camera_zoom = st.session_state.sync_camera_zoom
    terrain_depth_3d = st.session_state.terrain_depth_3d
    is_globe = st.session_state.projection_mode == "3D"

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
    lon_scale = round(float(np.cos(np.deg2rad(min(max(center_lat, -75.0), 75.0)))), 2)

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
        ref_grid_radius = ZOOM_TIERS[1]["radius_deg"] if (zoom_tier_num == 1 and not show_all_hex) else ZOOM_TIERS[2]["radius_deg"]
        global_ref_grid_df = build_regional_or_globe_geodesic_grid(
            radius_deg=ref_grid_radius,
            is_globe=is_globe,
            lon_scale=lon_scale,
            globe_altitude_m=1200.0,
        )
        active_layers.append(
            pdk.Layer(
                "PolygonLayer",
                id=f"geodesic_global_ref_grid_{ref_grid_radius}",
                data=global_ref_grid_df,
                get_polygon="polygon",
                filled=False,
                stroked=True,
                get_line_color=[0, 210, 255, 70],
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
                    tier_label=tier_cfg["label_en" if st.session_state.lang == "en" else "label_es"],
                    palette_key=st.session_state.palette_key,
                    smooth_weight=smooth_weight,
                    is_globe=is_globe,
                    lon_scale=lon_scale,
                    is_sensor_1to1=(t_id == 5),
                    globe_altitude_m=1210.0 + t_id * 5.0,
                    lang=st.session_state.lang,
                )
                active_layers.append(_make_hex_polygon_layer(f"thermal_hex_multi_tier_{t_id}", df_tier))
        else:
            df_active_hex = build_clustered_hex_layer(
                filtered_df,
                radius_deg=active_tier["radius_deg"],
                tier_label=active_tier_label,
                palette_key=st.session_state.palette_key,
                smooth_weight=smooth_weight,
                is_globe=is_globe,
                lon_scale=lon_scale,
                is_sensor_1to1=(zoom_tier_num == 5),
                globe_altitude_m=1210.0,
                lang=st.session_state.lang,
            )
            active_layers.append(_make_hex_polygon_layer(f"thermal_hex_single_tier_{zoom_tier_num}", df_active_hex))
            
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
    tooltip_text = (
        "{escala_vista} · {pais}\n"
        "Relative intensity (view): {indice_termico}% · {nivel_intensidad}\n"
        "Hotspots: {total_incendios} (MODIS {n_modis} · VIIRS {n_viirs}) · {celdas_agrupadas} cells\n"
        "Mean brightness: {brillo_txt} | Mean FRP: {frp_txt}\n"
        "Coverage: {resolucion_m} ({area_km2} km²) · Sensor: {instrumento} ({satelite})"
        if st.session_state.lang == "en"
        else "{escala_vista} · {pais}\n"
        "Intensidad relativa (vista): {indice_termico}% · {nivel_intensidad}\n"
        "Focos: {total_incendios} (MODIS {n_modis} · VIIRS {n_viirs}) · {celdas_agrupadas} celdas\n"
        "Brillo prom.: {brillo_txt} | FRP prom.: {frp_txt}\n"
        "Cobertura: {resolucion_m} ({area_km2} km²) · Sensor: {instrumento} ({satelite})"
    )
    deck = pdk.Deck(
        layers=active_layers,
        initial_view_state=view_state,
        views=[map_view],
        tooltip={"text": tooltip_text},
        map_style=None if is_globe else "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    )
    st.pydeck_chart(deck, height=900, **DECK)
    render_map_legend()
    clear_loading_status()

#muchas gracias por permitirnos particiopar en el evento, un 67 para la nasa, la utb, y la salle cartagena, y un 667 para el equipo