import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

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

# Límites geográficos reales de Latinoamérica y el Caribe (evita puntos corruptos en la Antártida)
LATAM_LAT_MIN, LATAM_LAT_MAX = -56.5, 34.0
LATAM_LON_MIN, LATAM_LON_MAX = -120.0, -32.0

# Niveles de agrupación hexagonal (1: más grande -> 5: huella física 1:1 del sensor)
ZOOM_TIERS: Dict[int, Dict[str, Any]] = {
    1: {
        "short": "1 · Macro (~300 km)",
        "label": "Nivel 1 · Macro-Regional (~300 km)",
        "radius_deg": 1.35,
        "camera_zoom": 2.9,
        "grid_radius": 9.0,
    },
    2: {
        "short": "2 · Regional (~110 km)",
        "label": "Nivel 2 · Regional (~110 km)",
        "radius_deg": 0.50,
        "camera_zoom": 4.3,
        "grid_radius": 4.5,
    },
    3: {
        "short": "3 · Sub-Regional (~40 km)",
        "label": "Nivel 3 · Sub-Regional (~40 km)",
        "radius_deg": 0.18,
        "camera_zoom": 5.7,
        "grid_radius": 2.25,
    },
    4: {
        "short": "4 · Local (~13 km)",
        "label": "Nivel 4 · Local / Municipal (~13 km)",
        "radius_deg": 0.06,
        "camera_zoom": 7.3,
        "grid_radius": 1.125,
    },
    5: {
        "short": "5 · Sensor 1:1 (1 km / 375 m)",
        "label": "Nivel 5 · Huella Física Sensor 1:1",
        "radius_deg": None,
        "camera_zoom": 9.6,
        "grid_radius": 0.50,
    },
}

# Paletas térmicas continuas (Verde → Ámbar → Naranja → Rojo por defecto)
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
    1: "1 · Enero",
    2: "2 · Febrero",
    3: "3 · Marzo",
    4: "4 · Abril",
    5: "5 · Mayo",
    6: "6 · Junio",
    7: "7 · Julio",
    8: "8 · Agosto",
    9: "9 · Septiembre",
    10: "10 · Octubre",
    11: "11 · Noviembre",
    12: "12 · Diciembre",
}

EXPECTED_COLUMNS = [
    "pais",
    "anio",
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

# 2. Estado inicial de sesión
if "dock_visible" not in st.session_state:
    st.session_state.dock_visible = True

# 3. Estilos CSS: Dock apilable en la esquina superior izquierda y mapa a pantalla completa
st.markdown(
    """
    <style>
    #MainMenu, footer, header, [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    .block-container {
        padding: 0rem !important;
        max-width: 100% !important;
        height: 100vh !important;
        overflow: hidden !important;
    }

    .st-key-dock_superior_izquierdo {
        position: fixed !important;
        top: 16px !important;
        left: 16px !important;
        z-index: 9999 !important;
        width: 375px !important;
        max-height: calc(100vh - 32px) !important;
        overflow-y: auto !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
        padding-right: 4px !important;
    }

    .st-key-dock_superior_izquierdo::-webkit-scrollbar {
        width: 5px;
    }
    .st-key-dock_superior_izquierdo::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.25);
        border-radius: 4px;
    }

    .st-key-dock_superior_izquierdo div[data-testid="stExpander"] {
        background: rgba(17, 21, 30, 0.90) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45) !important;
    }

    .st-key-dock_superior_izquierdo div[data-testid="stExpander"] summary {
        font-weight: 600 !important;
        font-size: 14px !important;
        color: #f8fafc !important;
    }

    .st-key-dock_superior_izquierdo button[kind="secondary"] {
        background: rgba(17, 21, 30, 0.92) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
    }

    .stDeckGlJsonChart, div[data-testid="stDeckGlJsonChart"], iframe {
        height: 100vh !important;
        width: 100vw !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        z-index: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# 4. Carga del Mapamundi GeoJSON y Geometría Base para el Globo 3D
@st.cache_data
def load_world_map_geojson() -> Optional[Dict[str, Any]]:
    """Descarga y cachea localmente el mapa mundial Natural Earth para el render 3D."""
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
    """Construye la malla oceánica esférica (quads de 5°) y los paralelos/meridianos para GlobeView."""
    lons = np.linspace(-180.0, 180.0, 73)
    lats = np.linspace(-89.5, 89.5, 37)
    ocean_quads = []
    for i in range(len(lons) - 1):
        for j in range(len(lats) - 1):
            ocean_quads.append(
                {
                    "polygon": [
                        [float(lons[i]), float(lats[j])],
                        [float(lons[i + 1]), float(lats[j])],
                        [float(lons[i + 1]), float(lats[j + 1])],
                        [float(lons[i]), float(lats[j + 1])],
                    ]
                }
            )

    graticule_paths = []
    for lon in range(-180, 180, 30):
        pts = [[float(lon), float(la)] for la in np.linspace(-89.5, 89.5, 90)]
        is_prime = lon == 0
        graticule_paths.append(
            {
                "path": pts,
                "color": [56, 189, 248, 85] if is_prime else [148, 163, 184, 32],
            }
        )
    for lat in range(-75, 90, 15):
        pts = [[float(lo), float(lat)] for lo in np.linspace(-180.0, 180.0, 120)]
        is_equator = lat == 0
        graticule_paths.append(
            {
                "path": pts,
                "color": [56, 189, 248, 85] if is_equator else [148, 163, 184, 32],
            }
        )

    return pd.DataFrame(ocean_quads), pd.DataFrame(graticule_paths)


# 5. Cliente de datos para /api/incendios-grid y /api/filtros-disponibles
def _derive_base_api_url(grid_url: str) -> str:
    parsed = urlparse(grid_url.strip())
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return "http://127.0.0.1:8000"


@st.cache_data(ttl=20)
def fetch_api_catalogs(grid_url: str) -> Dict[str, Any]:
    """Consulta /api/filtros-disponibles del backend LATAM si está en línea."""
    base_url = _derive_base_api_url(grid_url)
    try:
        resp = requests.get(f"{base_url}/api/filtros-disponibles", timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _normalize_grid_dataframe(records: List[Dict[str, Any]], strict_latam_bounds: bool = True) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = pd.DataFrame(records)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["pais"] = df["pais"].fillna("LATAM").astype(str).str.strip()
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").fillna(2025).astype(int)
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

    # Sanitización geográfica: elimina coordenadas corruptas (ej. N20 desplazado hacia la Antártida)
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
    strict_latam_bounds: bool = True,
    limit: int = 60000,
) -> tuple[pd.DataFrame, bool, str]:
    """Consulta /api/incendios-grid con soporte para pais, anio y limit."""
    params: Dict[str, Any] = {"limit": limit}
    if pais and pais != "Todos":
        params["pais"] = pais
    if anio:
        params["anio"] = int(anio)

    try:
        resp = requests.get(
            api_url.strip(),
            params=params,
            timeout=25,
            headers={"Cache-Control": "no-cache"},
        )
        resp.raise_for_status()
        payload = resp.json()

        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("data") or payload.get("records") or payload.get("grid") or []
        else:
            records = []

        df = _normalize_grid_dataframe(records, strict_latam_bounds=strict_latam_bounds)
        return df, True, f"🟢 API EN VIVO ({len(df):,} celdas válidas)"
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
            raw["lat_grid"] = raw["latitude"].round(2)
            raw["lon_grid"] = raw["longitude"].round(2)

            grouped = (
                raw.groupby(["anio", "lat_grid", "lon_grid", "instrument", "satellite", "mes"], as_index=False)
                .agg(
                    total_incendios=("brightness", "count"),
                    frp_promedio=("frp", "mean"),
                    brillo_promedio=("brightness", "mean"),
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
            return df, False, f"🟠 MODO LOCAL (Error API: {api_err})"

        return pd.DataFrame(columns=EXPECTED_COLUMNS), False, f"🔴 Sin conexión y sin CSV local ({api_err})"


# 6. Campo Térmico Continuo con Suavizado Vecinal Hexagonal
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
    a = np.interp(t, [0.0, 1.0], [180, 240]).astype(np.uint8)

    fill_rgba = np.column_stack([r, g, b, a])
    line_rgb = (fill_rgba[:, :3].astype(np.float32) * 0.82).astype(np.uint8)
    line_a = np.clip(fill_rgba[:, 3].astype(np.int32) + 15, 0, 255).astype(np.uint8)
    line_rgba = np.column_stack([line_rgb, line_a])

    return fill_rgba, line_rgba


def get_sensor_pixel_meters(instrument_series: pd.Series) -> np.ndarray:
    upper = instrument_series.str.upper()
    pixels = np.full(len(upper), 1000.0, dtype=np.float64)
    pixels[upper.str.contains("VIIRS", na=False)] = 375.0
    pixels[upper.str.contains("GOES", na=False)] = 2000.0
    pixels[upper.str.contains("ASTER", na=False)] = 90.0
    pixels[upper.str.contains("LANDSAT", na=False)] = 30.0
    return pixels


@st.cache_data
def build_grid_by_radius(radius_deg: float, is_globe: bool = False) -> pd.DataFrame:
    dx = radius_deg * 1.5
    dy = radius_deg * np.sqrt(3)

    col_start = int(np.floor(-180.0 / dx))
    col_end = int(np.ceil(180.0 / dx))
    row_start = int(np.floor(-70.0 / dy))
    row_end = int(np.ceil(80.0 / dy))

    angles = np.deg2rad(np.arange(0, 360, 60))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    polygons = []
    for col_idx in range(col_start, col_end + 1):
        lon = col_idx * dx
        lat_offset = (dy / 2.0) if (col_idx % 2 != 0) else 0.0
        for row_idx in range(row_start, row_end + 1):
            c_lat = row_idx * dy + lat_offset
            if c_lat > 80.0 or c_lat < -70.0:
                continue

            if is_globe:
                lon_scale = max(0.2, np.cos(np.deg2rad(c_lat)))
                v_lons = lon + (cos_a * radius_deg / lon_scale)
                v_lats = np.clip(c_lat + (sin_a * radius_deg), -85.0, 85.0)
                vertices = [[float(v_lons[i]), float(v_lats[i])] for i in range(6)]
            else:
                vertices = [
                    [float(lon + cos_a[i] * radius_deg), float(np.clip(c_lat + sin_a[i] * radius_deg, -85.0, 85.0))]
                    for i in range(6)
                ]
            polygons.append({"polygon": vertices})

    return pd.DataFrame(polygons)


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
def build_clustered_hex_layer(
    df: pd.DataFrame,
    radius_deg: float,
    tier_label: str,
    palette_name: str,
    smooth_weight: float = 0.35,
    is_globe: bool = False,
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

    if is_globe:
        lon_scale = np.maximum(0.2, np.cos(np.deg2rad(c_lats)))
        v_lons = c_lons[:, None] + (radius_deg / lon_scale[:, None]) * cos_a[None, :]
        v_lats = np.clip(c_lats[:, None] + radius_deg * sin_a[None, :], -85.0, 85.0)
    else:
        # En Web Mercator (2D) escalamos d_lat por cos(lat) para mantener hexágonos regulares sin estiramiento vertical
        lat_scale = np.maximum(0.2, np.cos(np.deg2rad(c_lats)))
        v_lons = c_lons[:, None] + radius_deg * cos_a[None, :]
        v_lats = np.clip(c_lats[:, None] + (radius_deg * lat_scale[:, None]) * sin_a[None, :], -85.0, 85.0)

    coords = np.stack([v_lons, v_lats], axis=-1).round(5).tolist()

    km_per_deg = METERS_PER_DEG_LAT / 1000.0
    area_km2 = np.maximum(
        0.1, ((3.0 * np.sqrt(3.0) / 2.0) * ((radius_deg * km_per_deg) ** 2) * np.cos(np.deg2rad(c_lats)))
    ).round(1)
    diam_km = round(2.0 * radius_deg * km_per_deg, 1)

    max_count = max(int(counts_arr.max()), 1)
    elevations = ((counts_arr / max_count) * 180_000.0).round(0).tolist()
    n = len(agg)

    return pd.DataFrame(
        {
            "polygon": coords,
            "rgba": fill_rgba.tolist(),
            "line_rgba": line_rgba.tolist(),
            "elevation": elevations,
            "escala_vista": [tier_label] * n,
            "pais": agg["pais"].tolist(),
            "lat_grid": np.round(c_lats, 3).tolist(),
            "lon_grid": np.round(c_lons, 3).tolist(),
            "total_incendios": counts_arr.tolist(),
            "celdas_agrupadas": agg["celdas_agrupadas"].tolist(),
            "frp_promedio": frp_arr.tolist(),
            "brillo_promedio": brillo_arr.tolist(),
            "indice_termico": np.round(heat_vals * 100.0, 1).tolist(),
            "instrumento": agg["instrumento"].tolist(),
            "satelite": agg["satelite"].tolist(),
            "nivel_riesgo": nivel_termico.tolist(),
            "resolucion_m": [f"Clúster Ø ~{diam_km} km"] * n,
            "area_km2": area_km2.tolist(),
        }
    )


@st.cache_data
def build_sensor_footprint_hexagons(
    df: pd.DataFrame,
    palette_name: str,
    smooth_weight: float = 0.35,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    lats = df["lat_grid"].to_numpy(dtype=np.float64)
    lons = df["lon_grid"].to_numpy(dtype=np.float64)
    pixel_m = get_sensor_pixel_meters(df["instrumento"])

    r_meters = pixel_m * np.sqrt(2.0 / (3.0 * np.sqrt(3.0)))
    r_lat_deg = r_meters / METERS_PER_DEG_LAT
    lon_scale = np.maximum(0.2, np.cos(np.deg2rad(lats)))
    r_lon_deg = r_lat_deg / lon_scale
    area_km2 = (pixel_m / 1000.0) ** 2
    resolucion_txt = [f"{int(w)} m × {int(w)} m (1:1)" for w in pixel_m]

    angles = np.deg2rad(np.arange(0, 360, 60))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    v_lons = lons[:, None] + (r_lon_deg[:, None] * cos_a[None, :])
    v_lats = np.clip(lats[:, None] + (r_lat_deg[:, None] * sin_a[None, :]), -85.0, 85.0)
    coords = np.stack([v_lons, v_lats], axis=-1).round(6).tolist()

    counts_arr = df["total_incendios"].to_numpy(dtype=np.int64)
    brillo_arr = df["brillo_promedio"].to_numpy(dtype=np.float64)
    frp_arr = df["frp_promedio"].to_numpy(dtype=np.float64)

    qi, ri = _lonlat_to_axial_bins(lons, lats, radius_deg=0.015)
    heat_vals = compute_smoothed_thermal_field(
        qi, ri, brillo_arr, frp_arr, counts_arr.astype(np.float64), smooth_weight=smooth_weight
    )
    fill_rgba, line_rgba = thermal_values_to_rgba(heat_vals, palette_name=palette_name)

    nivel_termico = np.select(
        [heat_vals >= 0.72, heat_vals >= 0.42],
        ["Crítico / Alto", "Moderado / Medio"],
        default="Bajo / Periférico",
    )

    max_count = max(int(counts_arr.max()), 1)
    elevations = ((counts_arr / max_count) * 40_000.0).round(0).tolist()
    n = len(df)

    return pd.DataFrame(
        {
            "polygon": coords,
            "rgba": fill_rgba.tolist(),
            "line_rgba": line_rgba.tolist(),
            "elevation": elevations,
            "escala_vista": ["Nivel 5 · Huella Física Sensor 1:1"] * n,
            "pais": df["pais"].tolist(),
            "lat_grid": df["lat_grid"].tolist(),
            "lon_grid": df["lon_grid"].tolist(),
            "total_incendios": counts_arr.tolist(),
            "celdas_agrupadas": [1] * n,
            "frp_promedio": df["frp_promedio"].tolist(),
            "brillo_promedio": df["brillo_promedio"].tolist(),
            "indice_termico": np.round(heat_vals * 100.0, 1).tolist(),
            "instrumento": df["instrumento"].tolist(),
            "satelite": df["satelite"].tolist(),
            "nivel_riesgo": nivel_termico.tolist(),
            "resolucion_m": resolucion_txt,
            "area_km2": np.round(area_km2, 3).tolist(),
        }
    )


# 7. DOCK APILADO Y OCULTABLE EN LA ESQUINA SUPERIOR IZQUIERDA
api_endpoint = st.session_state.get("api_endpoint", DEFAULT_API_URL)
allow_fallback = st.session_state.get("allow_local_fallback", True)
strict_latam = st.session_state.get("strict_latam_cb", True)
catalogs = fetch_api_catalogs(api_endpoint)

paises_api = ["Todos"] + catalogs.get("paises", [])
anios_api = sorted(catalogs.get("anios", [2025]), reverse=True)

selected_pais_api = st.session_state.get("pais_api_select", "Todos")
if selected_pais_api not in paises_api:
    selected_pais_api = "Todos"

selected_anio_api = st.session_state.get("anio_api_select", anios_api[0])
if selected_anio_api not in anios_api:
    selected_anio_api = anios_api[0]

df_grid, is_live_api, connection_status = fetch_incendios_grid(
    api_endpoint,
    allow_fallback,
    pais=selected_pais_api,
    anio=selected_anio_api,
    strict_latam_bounds=strict_latam,
    limit=60000,
)
filtered_df = df_grid.copy()

palette_options = list(THERMAL_PALETTES.keys())

with st.container(key="dock_superior_izquierdo"):
    col_toggle, col_badge = st.columns([0.66, 0.34])
    with col_toggle:
        btn_label = "👁️ Ocultar paneles" if st.session_state.dock_visible else "☰ Mostrar paneles (3)"
        if st.button(btn_label, use_container_width=True):
            st.session_state.dock_visible = not st.session_state.dock_visible
            st.rerun()
    with col_badge:
        total_fires_badge = int(df_grid["total_incendios"].sum()) if not df_grid.empty else 0
        badge_dot = "🟢" if is_live_api else "🟠"
        st.markdown(
            f"<div style='background:rgba(17,21,30,0.88);border:1px solid rgba(255,255,255,0.15);"
            f"border-radius:10px;padding:7px 6px;text-align:center;color:#fbbf24;font-size:12px;font-weight:700;'>"
            f"{badge_dot} {total_fires_badge:,}</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.dock_visible:
        # VENTANA 1: Conexión API LATAM y Filtros
        with st.expander("🔥 Grid de Incendios · Filtros", expanded=True):
            st.text_input("Endpoint Backend:", value=api_endpoint, key="api_endpoint")

            col_sync, col_fb = st.columns([0.52, 0.48])
            with col_sync:
                if st.button("🔄 Sincronizar ahora", use_container_width=True):
                    fetch_api_catalogs.clear()
                    fetch_incendios_grid.clear()
                    build_clustered_hex_layer.clear()
                    build_sensor_footprint_hexagons.clear()
                    st.rerun()
            with col_fb:
                st.checkbox("Respaldo local", value=True, key="allow_local_fallback")

            if is_live_api:
                st.success(connection_status)
            else:
                st.warning(connection_status)

            col_pais, col_anio = st.columns([0.62, 0.38])
            with col_pais:
                st.selectbox("País (API LATAM):", options=paises_api, key="pais_api_select")
            with col_anio:
                st.selectbox("Año:", options=anios_api, key="anio_api_select")

            st.checkbox(
                "Filtrar coordenadas fuera de LATAM (corrige anomalías N20)",
                value=True,
                key="strict_latam_cb",
            )

            if not df_grid.empty:
                modo_tiempo = st.radio(
                    "Filtro Temporal (`mes`):",
                    options=["Mes específico (Slider)", "Todo el año (1–12)"],
                    horizontal=True,
                    index=0,
                    key="modo_tiempo_radio",
                )
                if modo_tiempo == "Mes específico (Slider)":
                    meses_disponibles = sorted(df_grid["mes"].unique().tolist())
                    default_mes = 8 if 8 in meses_disponibles else meses_disponibles[0]
                    mes_seleccionado = st.select_slider(
                        "Mes:",
                        options=list(range(1, 13)),
                        value=default_mes,
                        format_func=lambda m: MESES_NOMBRES[m],
                        key="mes_slider",
                    )
                    filtered_df = filtered_df[filtered_df["mes"] == mes_seleccionado]

                riesgos_orden = [r for r in ["Alto", "Medio", "Bajo"] if r in df_grid["nivel_riesgo"].unique()]
                if not riesgos_orden:
                    riesgos_orden = sorted(df_grid["nivel_riesgo"].unique().tolist())
                selected_riesgo = st.multiselect(
                    "Nivel de Riesgo:", riesgos_orden, default=riesgos_orden, key="riesgo_multi"
                )

                sat_options = sorted(df_grid["satelite"].unique().tolist())
                selected_sat = st.multiselect("Satélite:", sat_options, default=sat_options, key="sat_multi")

                inst_options = sorted(df_grid["instrumento"].unique().tolist())
                selected_inst = st.multiselect(
                    "Instrumento / Sensor:", inst_options, default=inst_options, key="inst_multi"
                )

                filtered_df = filtered_df[
                    (filtered_df["nivel_riesgo"].isin(selected_riesgo))
                    & (filtered_df["satelite"].isin(selected_sat))
                    & (filtered_df["instrumento"].isin(selected_inst))
                ]

        # VENTANA 2: Escala Térmica y Tamaño Exclusivo / Multiescala
        with st.expander("🗺️ Escala Térmica y Zoom Exclusivo", expanded=True):
            palette_name = st.selectbox(
                "Paleta Térmica Continua:",
                options=palette_options,
                index=0,
                key="palette_select",
            )

            smooth_pct = st.slider(
                "Aproximación de color entre vecinos (%):",
                min_value=0,
                max_value=60,
                value=35,
                step=5,
                key="smooth_slider",
            )

            is_multiscale_active = st.session_state.get("show_all_hex_cb", False)

            zoom_tier_num = st.select_slider(
                "Nivel de Zoom / Escala Hexagonal (1 tamaño a la vez):",
                options=[1, 2, 3, 4, 5],
                value=1,
                format_func=lambda k: ZOOM_TIERS[k]["short"],
                disabled=is_multiscale_active,
                key="zoom_tier_slider",
            )

            show_all_hex = st.checkbox(
                "Mostrar todos los tamaños de hexágonos a la vez (Multiescala)",
                value=False,
                help="Superpone simultáneamente los 5 niveles hexagonales (Macro ~300 km hasta Sensor 1:1).",
                key="show_all_hex_cb",
            )

            sync_camera_zoom = st.checkbox(
                "Ajustar zoom de cámara automáticamente al nivel",
                value=True,
                key="sync_camera_cb",
            )

            # Desactivado por defecto para evitar conflictos de textura de profundidad (Z-fighting)
            terrain_depth_3d = st.checkbox(
                "Activar profundidad del terreno (Extrusión 3D)",
                value=False,
                help="Activa el relieve 3D por cantidad de incendios. Desactivado por defecto para evitar conflictos de profundidad.",
                key="extrude_cb",
            )

            projection_mode = st.radio(
                "Proyección Cartográfica:",
                options=["Plano (2D)", "Esfera 3D (Globo)"],
                index=0,
                key="proj_radio",
            )

            if projection_mode == "Esfera 3D (Globo)":
                show_world_map_3d = st.checkbox("Mapamundi y Fronteras (3D)", value=True, key="world_3d_cb")
                show_graticule_3d = st.checkbox("Paralelos y Meridianos (3D)", value=True, key="grat_3d_cb")
            else:
                show_world_map_3d = False
                show_graticule_3d = False

            show_base_grid = st.checkbox(
                "Malla Geodésica de Referencia",
                value=False,
                key="base_grid_cb",
            )

        # VENTANA 3: Métricas Ponderadas y Resumen de Clústeres
        with st.expander("📊 Análisis Térmico y Clústeres", expanded=False):
            if not filtered_df.empty:
                preview_clusters = build_clustered_hex_layer(
                    filtered_df,
                    radius_deg=0.50,
                    tier_label="Regional",
                    palette_name=palette_name,
                    smooth_weight=smooth_pct / 100.0,
                    is_globe=False,
                )
                total_inc = int(filtered_df["total_incendios"].sum())
                weights = filtered_df["total_incendios"].to_numpy(dtype=np.float64)
                brillo_pond = np.average(filtered_df["brillo_promedio"], weights=weights)

                c1, c2, c3 = st.columns(3)
                c1.metric("Incendios", f"{total_inc:,}")
                c2.metric("Clústeres Reg.", f"{len(preview_clusters):,}")
                c3.metric("Brillo Prom.", f"{brillo_pond:.1f} K")

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
                st.bar_chart(mensual, height=125)

                st.dataframe(
                    preview_clusters[
                        ["pais", "lat_grid", "lon_grid", "total_incendios", "indice_termico", "brillo_promedio"]
                    ]
                    .sort_values("total_incendios", ascending=False)
                    .head(15),
                    use_container_width=True,
                    height=145,
                )
            else:
                st.warning("Sin celdas para los filtros seleccionados.")
    else:
        palette_name = st.session_state.get("palette_select", palette_options[0])
        smooth_pct = st.session_state.get("smooth_slider", 35)
        zoom_tier_num = st.session_state.get("zoom_tier_slider", 1)
        show_all_hex = st.session_state.get("show_all_hex_cb", False)
        sync_camera_zoom = st.session_state.get("sync_camera_cb", True)
        terrain_depth_3d = st.session_state.get("extrude_cb", False)
        projection_mode = st.session_state.get("proj_radio", "Plano (2D)")
        show_world_map_3d = st.session_state.get("world_3d_cb", True)
        show_graticule_3d = st.session_state.get("grat_3d_cb", True)
        show_base_grid = st.session_state.get("base_grid_cb", False)

smooth_weight = smooth_pct / 100.0
active_tier = ZOOM_TIERS[zoom_tier_num]


# 8. RENDERIZADO EN PYDECK (SIN PROFUNDIDAD POR DEFECTO PARA EVITAR Z-FIGHTING)
is_globe = projection_mode == "Esfera 3D (Globo)"

if not filtered_df.empty:
    center_lat = float(filtered_df["lat_grid"].mean())
    center_lon = float(filtered_df["lon_grid"].mean())
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

    if show_world_map_3d:
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

    if show_graticule_3d:
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

if show_base_grid:
    if show_all_hex:
        for t_id in (1, 2, 3, 4):
            grid_df = build_grid_by_radius(ZOOM_TIERS[t_id]["grid_radius"], is_globe=is_globe)
            active_layers.append(
                pdk.Layer(
                    "PolygonLayer",
                    id=f"geodesic_grid_multi_{t_id}",
                    data=grid_df,
                    get_polygon="polygon",
                    filled=False,
                    stroked=True,
                    get_line_color=[0, 210, 255, 32 + t_id * 8],
                    line_width_min_pixels=1,
                    pickable=False,
                )
            )
    else:
        single_grid_df = build_grid_by_radius(active_tier["grid_radius"], is_globe=is_globe)
        active_layers.append(
            pdk.Layer(
                "PolygonLayer",
                id=f"geodesic_grid_tier_{zoom_tier_num}",
                data=single_grid_df,
                get_polygon="polygon",
                filled=False,
                stroked=True,
                get_line_color=[0, 210, 255, 45],
                line_width_min_pixels=1,
                pickable=False,
            )
        )


def _make_hex_polygon_layer(layer_id: str, data_df: pd.DataFrame) -> pdk.Layer:
    """Construye PolygonLayer plano o con profundidad según el interruptor del usuario."""
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
        layer_kwargs["wireframe"] = True
    return pdk.Layer("PolygonLayer", **layer_kwargs)


if not filtered_df.empty:
    if show_all_hex:
        for t_id in (1, 2, 3, 4):
            tier_cfg = ZOOM_TIERS[t_id]
            df_tier = build_clustered_hex_layer(
                filtered_df,
                radius_deg=tier_cfg["radius_deg"],
                tier_label=tier_cfg["label"],
                palette_name=palette_name,
                smooth_weight=smooth_weight,
                is_globe=is_globe,
            )
            active_layers.append(_make_hex_polygon_layer(f"thermal_hex_multi_tier_{t_id}", df_tier))

        df_sensor_all = build_sensor_footprint_hexagons(
            filtered_df,
            palette_name=palette_name,
            smooth_weight=smooth_weight,
        )
        active_layers.append(_make_hex_polygon_layer("thermal_hex_multi_tier_5_sensor", df_sensor_all))
    else:
        if active_tier["radius_deg"] is not None:
            df_active_hex = build_clustered_hex_layer(
                filtered_df,
                radius_deg=active_tier["radius_deg"],
                tier_label=active_tier["label"],
                palette_name=palette_name,
                smooth_weight=smooth_weight,
                is_globe=is_globe,
            )
        else:
            df_active_hex = build_sensor_footprint_hexagons(
                filtered_df,
                palette_name=palette_name,
                smooth_weight=smooth_weight,
            )

        active_layers.append(_make_hex_polygon_layer(f"thermal_hex_single_tier_{zoom_tier_num}", df_active_hex))

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