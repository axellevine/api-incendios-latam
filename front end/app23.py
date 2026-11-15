import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

# 1. Configuración de página en modo ancho total
st.set_page_config(
    page_title="NASA Hot Spots · Mapa Térmico Hexagonal",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
WORLD_GEOJSON_CACHE = BASE_DIR / "world_countries_110m.geojson"
DEFAULT_API_URL = os.getenv("INCENDIOS_API_URL", "http://127.0.0.1:8000/api/incendios-grid")

EARTH_RADIUS_M = 6_371_008.8
METERS_PER_DEG_LAT = (np.pi / 180.0) * EARTH_RADIUS_M  # ~111,195 m/grado

# Niveles de detalle (LOD) estrictamente excluyentes por rango de zoom
ZOOM_CLUSTER_TIERS = [
    {"key": "L1", "label": "Nivel 1 · Macro-Regional (~300 km)", "radius_deg": 1.35, "z_min": 0.0, "z_max": 3.199},
    {"key": "L2", "label": "Nivel 2 · Regional (~110 km)", "radius_deg": 0.50, "z_min": 3.2, "z_max": 4.799},
    {"key": "L3", "label": "Nivel 3 · Sub-Regional (~40 km)", "radius_deg": 0.18, "z_min": 4.8, "z_max": 6.499},
    {"key": "L4", "label": "Nivel 4 · Local / Municipal (~13 km)", "radius_deg": 0.06, "z_min": 6.5, "z_max": 8.499},
]

# Paletas térmicas continuas (sin choques de colores complementarios opuestos)
THERMAL_PALETTES: Dict[str, Dict[str, Any]] = {
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

# 2. Estado de visibilidad del Dock Superior Izquierdo
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
        width: 370px !important;
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


# 5. Cliente de datos para /api/incendios-grid + respaldo local opcional
def _normalize_grid_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = pd.DataFrame(records)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["lat_grid"] = pd.to_numeric(df["lat_grid"], errors="coerce")
    df["lon_grid"] = pd.to_numeric(df["lon_grid"], errors="coerce")
    df["total_incendios"] = pd.to_numeric(df["total_incendios"], errors="coerce").fillna(1).astype(int)
    df["frp_promedio"] = pd.to_numeric(df["frp_promedio"], errors="coerce").fillna(0.0).round(1)
    df["brillo_promedio"] = pd.to_numeric(df["brillo_promedio"], errors="coerce").fillna(0.0).round(1)
    df["instrumento"] = df["instrumento"].fillna("MODIS").astype(str).str.strip()
    df["satelite"] = df["satelite"].fillna("Terra").astype(str).str.strip()
    df["mes"] = pd.to_numeric(df["mes"], errors="coerce").fillna(1).astype(int).clip(1, 12)
    df["nivel_riesgo"] = df["nivel_riesgo"].fillna("Bajo").astype(str)
    df["color"] = df["color"].fillna("#2ECC71").astype(str)

    return df.dropna(subset=["lat_grid", "lon_grid"]).reset_index(drop=True)


@st.cache_data(ttl=15)
def fetch_incendios_grid(api_url: str, allow_local_fallback: bool) -> tuple[pd.DataFrame, bool, str]:
    """Consulta el endpoint /api/incendios-grid; retorna (df, is_live_api, status_msg)."""
    try:
        resp = requests.get(api_url, timeout=10, headers={"Cache-Control": "no-cache"})
        resp.raise_for_status()
        payload = resp.json()

        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("data") or payload.get("records") or payload.get("grid") or []
        else:
            records = []

        df = _normalize_grid_dataframe(records)
        return df, True, f"🟢 API EN VIVO ({len(df):,} celdas recibidas)"
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
            raw["lat_grid"] = raw["latitude"].round(2)
            raw["lon_grid"] = raw["longitude"].round(2)

            grouped = (
                raw.groupby(["lat_grid", "lon_grid", "instrument", "satellite", "mes"], as_index=False)
                .agg(
                    total_incendios=("brightness", "count"),
                    frp_promedio=("frp", "mean"),
                    brillo_promedio=("brightness", "mean"),
                )
                .rename(columns={"instrument": "instrumento", "satellite": "satelite"})
            )
            grouped["frp_promedio"] = grouped["frp_promedio"].round(1)
            grouped["brillo_promedio"] = grouped["brillo_promedio"].round(1)

            cond_alto = (grouped["total_incendios"] >= 10) | (grouped["brillo_promedio"] >= 355.0)
            cond_medio = (grouped["total_incendios"] >= 4) | (grouped["brillo_promedio"] >= 330.0)

            grouped["nivel_riesgo"] = np.select([cond_alto, cond_medio], ["Alto", "Medio"], default="Bajo")
            grouped["color"] = np.select([cond_alto, cond_medio], ["#FF0000", "#F39C12"], default="#2ECC71")

            df = _normalize_grid_dataframe(grouped.to_dict(orient="records"))
            return df, False, "🟠 MODO LOCAL (Backend apagado en puerto 8000)"

        return pd.DataFrame(columns=EXPECTED_COLUMNS), False, f"🔴 Sin conexión y sin CSV local ({api_err})"


# 6. Campo Térmico Continuo con Suavizado Vecinal Hexagonal
def _normalize_percentile(arr: np.ndarray) -> np.ndarray:
    if len(arr) <= 1:
        return np.full(len(arr), 0.5, dtype=np.float64)
    p5, p98 = np.percentile(arr, [5, 98])
    if p98 - p5 < 1e-6:
        return np.full(len(arr), 0.5, dtype=np.float64)
    return np.clip((arr - p5) / (p98 - p5), 0.0, 1.0)


def compute_smoothed_thermal_field(
    q_bins: np.ndarray,
    r_bins: np.ndarray,
    brightness: np.ndarray,
    frp: np.ndarray,
    counts: np.ndarray,
    smooth_weight: float = 0.35,
) -> np.ndarray:
    """
    Calcula un índice térmico compuesto [0, 1] y lo suaviza con los 6 vecinos hexagonales
    inmediatos para que los colores cercanos se aproximen como un mapa térmico continuo.
    """
    t_bright = _normalize_percentile(brightness)
    t_frp = _normalize_percentile(np.log1p(np.maximum(frp, 0.0)))
    t_count = _normalize_percentile(np.log1p(np.maximum(counts, 0.0)))

    raw_heat = 0.45 * t_bright + 0.30 * t_frp + 0.25 * t_count

    if smooth_weight <= 0.0 or len(raw_heat) <= 1:
        return raw_heat

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
    """Convierte el índice térmico suavizado [0, 1] a (fill_rgba, line_rgba) coherentes."""
    n = len(heat_values)
    if n == 0:
        empty = np.zeros((0, 4), dtype=np.uint8)
        return empty, empty

    pal = THERMAL_PALETTES.get(palette_name, next(iter(THERMAL_PALETTES.values())))
    stops = pal["stops"]
    t = np.clip(heat_values, 0.0, 1.0)

    r = np.interp(t, stops, pal["r"]).astype(np.uint8)
    g = np.interp(t, stops, pal["g"]).astype(np.uint8)
    b = np.interp(t, stops, pal["b"]).astype(np.uint8)
    a = np.interp(t, [0.0, 1.0], [165, 240]).astype(np.uint8)

    fill_rgba = np.column_stack([r, g, b, a])
    # Borde ligeramente más profundo en la misma gama cromática (nunca blanco)
    line_rgb = (fill_rgba[:, :3].astype(np.float32) * 0.82).astype(np.uint8)
    line_a = np.clip(fill_rgba[:, 3].astype(np.int32) + 15, 0, 255).astype(np.uint8)
    line_rgba = np.column_stack([line_rgb, line_a])

    return fill_rgba, line_rgba


def get_sensor_pixel_meters(instrument_series: pd.Series) -> np.ndarray:
    """Devuelve el ancho nominal de píxel en metros según el instrumento (MODIS, VIIRS, GOES, etc.)."""
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
                v_lats = c_lat + (sin_a * radius_deg)
                vertices = [[float(v_lons[i]), float(v_lats[i])] for i in range(6)]
            else:
                vertices = [
                    [float(lon + cos_a[i] * radius_deg), float(c_lat + sin_a[i] * radius_deg)]
                    for i in range(6)
                ]
            polygons.append({"polygon": vertices})

    return pd.DataFrame(polygons)


@st.cache_data
def load_multilevel_grids(is_globe: bool = False):
    return {
        "L1": build_grid_by_radius(9.0, is_globe),
        "L2": build_grid_by_radius(4.5, is_globe),
        "L3": build_grid_by_radius(2.25, is_globe),
        "L4": build_grid_by_radius(1.125, is_globe),
    }


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
) -> pd.DataFrame:
    """Agrupa celdas en una teselación hexagonal axial continua con gradiente térmico suavizado."""
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
        }
    )

    agg = work.groupby(["q_bin", "r_bin"], as_index=False).agg(
        total_incendios=("total_incendios", "sum"),
        w_brillo=("w_brillo", "sum"),
        w_frp=("w_frp", "sum"),
        celdas_agrupadas=("celdas", "sum"),
        instrumento=("instrumento", "first"),
        satelite=("satelite", "first"),
    )

    q_arr = agg["q_bin"].to_numpy(dtype=np.int64)
    r_arr = agg["r_bin"].to_numpy(dtype=np.int64)
    c_lons = radius_deg * 1.5 * q_arr.astype(np.float64)
    c_lats = radius_deg * np.sqrt(3.0) * (r_arr.astype(np.float64) + q_arr.astype(np.float64) / 2.0)

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
    v_lats = c_lats[:, None] + radius_deg * sin_a[None, :]
    coords = np.stack([v_lons, v_lats], axis=-1).round(5).tolist()

    km_per_deg = METERS_PER_DEG_LAT / 1000.0
    area_km2 = ((3.0 * np.sqrt(3.0) / 2.0) * ((radius_deg * km_per_deg) ** 2) * np.cos(np.deg2rad(c_lats))).round(1)
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
    """Nivel 5 (máximo zoom): huella física 1:1 del sensor con gradiente térmico suavizado."""
    if df.empty:
        return pd.DataFrame()

    lats = df["lat_grid"].to_numpy(dtype=np.float64)
    lons = df["lon_grid"].to_numpy(dtype=np.float64)
    pixel_m = get_sensor_pixel_meters(df["instrumento"])

    r_meters = pixel_m * np.sqrt(2.0 / (3.0 * np.sqrt(3.0)))
    r_lat_deg = r_meters / METERS_PER_DEG_LAT
    lon_scale = np.maximum(0.15, np.cos(np.deg2rad(lats)))
    r_lon_deg = r_lat_deg / lon_scale
    area_km2 = (pixel_m / 1000.0) ** 2
    resolucion_txt = [f"{int(w)} m × {int(w)} m (1:1)" for w in pixel_m]

    angles = np.deg2rad(np.arange(0, 360, 60))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    v_lons = lons[:, None] + (r_lon_deg[:, None] * cos_a[None, :])
    v_lats = lats[:, None] + (r_lat_deg[:, None] * sin_a[None, :])
    coords = np.stack([v_lons, v_lats], axis=-1).round(6).tolist()

    counts_arr = df["total_incendios"].to_numpy(dtype=np.int64)
    brillo_arr = df["brillo_promedio"].to_numpy(dtype=np.float64)
    frp_arr = df["frp_promedio"].to_numpy(dtype=np.float64)

    # Discretización axial fina (0.015°) para aproximar colores entre sensores vecinos
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
df_grid, is_live_api, connection_status = fetch_incendios_grid(api_endpoint, allow_fallback)
filtered_df = df_grid.copy()

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
        # VENTANA 1: Conexión API y Filtros
        with st.expander("🔥 Grid de Incendios · Filtros", expanded=True):
            st.text_input("Endpoint Backend:", value=api_endpoint, key="api_endpoint")

            col_sync, col_fb = st.columns([0.52, 0.48])
            with col_sync:
                if st.button("🔄 Sincronizar ahora", use_container_width=True):
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

            if not df_grid.empty:
                modo_tiempo = st.radio(
                    "Filtro Temporal (`mes`):",
                    options=["Mes específico (Slider)", "Todo el año (1–12)"],
                    horizontal=True,
                    index=0,
                )
                if modo_tiempo == "Mes específico (Slider)":
                    meses_disponibles = sorted(df_grid["mes"].unique().tolist())
                    default_mes = 8 if 8 in meses_disponibles else meses_disponibles[0]
                    mes_seleccionado = st.select_slider(
                        "Mes:",
                        options=list(range(1, 13)),
                        value=default_mes,
                        format_func=lambda m: MESES_NOMBRES[m],
                    )
                    filtered_df = filtered_df[filtered_df["mes"] == mes_seleccionado]

                riesgos_orden = [r for r in ["Alto", "Medio", "Bajo"] if r in df_grid["nivel_riesgo"].unique()]
                if not riesgos_orden:
                    riesgos_orden = sorted(df_grid["nivel_riesgo"].unique().tolist())
                selected_riesgo = st.multiselect("Nivel de Riesgo:", riesgos_orden, default=riesgos_orden)

                sat_options = sorted(df_grid["satelite"].unique().tolist())
                selected_sat = st.multiselect("Satélite:", sat_options, default=sat_options)

                inst_options = sorted(df_grid["instrumento"].unique().tolist())
                selected_inst = st.multiselect("Instrumento / Sensor:", inst_options, default=inst_options)

                filtered_df = filtered_df[
                    (filtered_df["nivel_riesgo"].isin(selected_riesgo))
                    & (filtered_df["satelite"].isin(selected_sat))
                    & (filtered_df["instrumento"].isin(selected_inst))
                ]

        # VENTANA 2: Escala Térmica Continua y Agrupación Exclusiva por Zoom
        with st.expander("🗺️ Escala Térmica y Zoom Exclusivo", expanded=False):
            palette_name = st.selectbox(
                "Paleta Térmica Continua:",
                options=list(THERMAL_PALETTES.keys()),
                index=0,
            )

            smooth_pct = st.slider(
                "Aproximación de color entre vecinos (%):",
                min_value=0,
                max_value=60,
                value=35,
                step=5,
                help="Suaviza la transición térmica entre hexágonos adyacentes.",
            )
            smooth_weight = smooth_pct / 100.0

            clustering_mode = st.selectbox(
                "Modo de Tamaño por Zoom:",
                options=[
                    "Exclusivo por Zoom (1 tamaño a la vez)",
                    "Fijo · Nivel 1 Macro (~300 km)",
                    "Fijo · Nivel 2 Regional (~110 km)",
                    "Fijo · Nivel 3 Sub-Regional (~40 km)",
                    "Fijo · Nivel 4 Local (~13 km)",
                    "Fijo · Nivel 5 Sensor 1:1 (MODIS 1 km / VIIRS 375 m)",
                ],
                index=0,
            )

            extrude_3d = st.checkbox("Extrusión 3D por Cantidad de Incendios", value=False)

            projection_mode = st.radio(
                "Proyección Cartográfica:",
                options=["Plano (2D)", "Esfera 3D (Globo)"],
                index=0,
            )

            if projection_mode == "Esfera 3D (Globo)":
                show_world_map_3d = st.checkbox("Mapamundi y Fronteras (3D)", value=True)
                show_graticule_3d = st.checkbox("Paralelos y Meridianos (3D)", value=True)
            else:
                show_world_map_3d = False
                show_graticule_3d = False

            show_base_grid = st.checkbox("Malla Geodésica de Referencia", value=False)

        # VENTANA 3: Métricas Ponderadas y Resumen de Clústeres
        with st.expander("📊 Análisis Térmico y Clústeres", expanded=False):
            if not filtered_df.empty:
                preview_clusters = build_clustered_hex_layer(
                    filtered_df,
                    radius_deg=0.50,
                    tier_label="Regional",
                    palette_name=palette_name,
                    smooth_weight=smooth_weight,
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
                        ["lat_grid", "lon_grid", "total_incendios", "indice_termico", "brillo_promedio", "nivel_riesgo"]
                    ]
                    .sort_values("total_incendios", ascending=False)
                    .head(15),
                    use_container_width=True,
                    height=145,
                )
            else:
                st.warning("Sin celdas para los filtros seleccionados.")
    else:
        palette_name = "Térmico Infrarrojo (Ámbar → Carmesí → Incandescente)"
        smooth_weight = 0.35
        clustering_mode = "Exclusivo por Zoom (1 tamaño a la vez)"
        extrude_3d = False
        projection_mode = "Plano (2D)"
        show_world_map_3d = True
        show_graticule_3d = True
        show_base_grid = False


# 8. RENDERIZADO EN PYDECK CON NIVELES EXCLUSIVOS Y GRADIENTE TÉRMICO SUAVIZADO
is_globe = projection_mode == "Esfera 3D (Globo)"

if not filtered_df.empty:
    center_lat = float(filtered_df["lat_grid"].mean())
    center_lon = float(filtered_df["lon_grid"].mean())
elif not df_grid.empty:
    center_lat = float(df_grid["lat_grid"].mean())
    center_lon = float(df_grid["lon_grid"].mean())
else:
    center_lat, center_lon = -31.5, -63.5

# Zoom inicial alejado para arrancar mostrando únicamente los hexágonos grandes (Nivel 1)
base_zoom = 2.9 if not is_globe else 2.1
active_layers = []

# Capas de fondo del Mapamundi para Esfera 3D (GlobeView)
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

# Malla geodésica opcional (desactivada por defecto para no mezclar tamaños)
if show_base_grid:
    grids_dict = load_multilevel_grids(is_globe=is_globe)
    level_specs = [
        ("L1", 0.0, 3.199, [0, 210, 255, 32]),
        ("L2", 3.2, 4.799, [0, 210, 255, 42]),
        ("L3", 4.8, 6.499, [0, 210, 255, 52]),
        ("L4", 6.5, 20.0, [0, 210, 255, 62]),
    ]
    for lvl_key, z_min, z_max, line_col in level_specs:
        active_layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=grids_dict[lvl_key],
                get_polygon="polygon",
                filled=False,
                stroked=True,
                get_line_color=line_col,
                line_width_min_pixels=1,
                zoom_min=z_min,
                zoom_max=z_max,
                pickable=False,
            )
        )

# Capas de Hexágonos Térmicos (exclusivamente 1 tamaño visible por nivel de zoom)
if not filtered_df.empty:
    if clustering_mode == "Exclusivo por Zoom (1 tamaño a la vez)":
        for idx, tier in enumerate(ZOOM_CLUSTER_TIERS):
            df_tier = build_clustered_hex_layer(
                filtered_df,
                radius_deg=tier["radius_deg"],
                tier_label=tier["label"],
                palette_name=palette_name,
                smooth_weight=smooth_weight,
            )
            active_layers.append(
                pdk.Layer(
                    "PolygonLayer",
                    id=f"thermal_hex_tier_{tier['key']}",
                    data=df_tier,
                    get_polygon="polygon",
                    get_fill_color="rgba",
                    get_line_color="line_rgba",
                    line_width_min_pixels=1,
                    filled=True,
                    stroked=True,
                    extruded=extrude_3d,
                    get_elevation="elevation",
                    zoom_min=tier["z_min"],
                    zoom_max=tier["z_max"],
                    pickable=True,
                    auto_highlight=True,
                )
            )

        # Nivel 5 (zoom >= 8.5): huella física exacta 1:1 del instrumento de medición
        df_sensor_1to1 = build_sensor_footprint_hexagons(
            filtered_df,
            palette_name=palette_name,
            smooth_weight=smooth_weight,
        )
        active_layers.append(
            pdk.Layer(
                "PolygonLayer",
                id="thermal_hex_tier_L5_sensor",
                data=df_sensor_1to1,
                get_polygon="polygon",
                get_fill_color="rgba",
                get_line_color="line_rgba",
                line_width_min_pixels=1,
                filled=True,
                stroked=True,
                extruded=extrude_3d,
                get_elevation="elevation",
                zoom_min=8.5,
                zoom_max=20.0,
                pickable=True,
                auto_highlight=True,
            )
        )
    else:
        fixed_map = {
            "Fijo · Nivel 1 Macro (~300 km)": (1.35, "Nivel 1 · Macro-Regional (~300 km)"),
            "Fijo · Nivel 2 Regional (~110 km)": (0.50, "Nivel 2 · Regional (~110 km)"),
            "Fijo · Nivel 3 Sub-Regional (~40 km)": (0.18, "Nivel 3 · Sub-Regional (~40 km)"),
            "Fijo · Nivel 4 Local (~13 km)": (0.06, "Nivel 4 · Local / Municipal (~13 km)"),
        }
        if clustering_mode in fixed_map:
            r_fixed, lbl_fixed = fixed_map[clustering_mode]
            df_fixed = build_clustered_hex_layer(
                filtered_df,
                radius_deg=r_fixed,
                tier_label=lbl_fixed,
                palette_name=palette_name,
                smooth_weight=smooth_weight,
            )
        else:
            df_fixed = build_sensor_footprint_hexagons(
                filtered_df,
                palette_name=palette_name,
                smooth_weight=smooth_weight,
            )

        active_layers.append(
            pdk.Layer(
                "PolygonLayer",
                id="thermal_hex_fixed",
                data=df_fixed,
                get_polygon="polygon",
                get_fill_color="rgba",
                get_line_color="line_rgba",
                line_width_min_pixels=1,
                filled=True,
                stroked=True,
                extruded=extrude_3d,
                get_elevation="elevation",
                pickable=True,
                auto_highlight=True,
            )
        )

pitch_angle = 40 if (extrude_3d and not is_globe) else 0

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
            "{escala_vista}\n"
            "Intensidad Térmica: {indice_termico}% ({nivel_riesgo})\n"
            "Incendios Agrupados: {total_incendios} ({celdas_agrupadas} celdas)\n"
            "Brillo Prom.: {brillo_promedio} K | FRP Prom.: {frp_promedio} MW\n"
            "Cobertura: {resolucion_m} ({area_km2} km²) · Sensor: {instrumento} ({satelite})"
        )
    },
    map_style=None if is_globe else "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
)

st.pydeck_chart(deck, use_container_width=True, height=900)