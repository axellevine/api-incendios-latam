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
    page_title="NASA Hot Spots · Grid 2D/3D",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
WORLD_GEOJSON_CACHE = BASE_DIR / "world_countries_110m.geojson"
DEFAULT_API_URL = os.getenv("INCENDIOS_API_URL", "http://127.0.0.1:8000/api/incendios-grid")

EARTH_RADIUS_M = 6_371_008.8
METERS_PER_DEG_LAT = (np.pi / 180.0) * EARTH_RADIUS_M  # ~111,195 m/grado

# Especificaciones físicas de resolución espacial por instrumento/sensor
SENSOR_SPECS: Dict[str, Dict[str, Any]] = {
    "MODIS": {"pixel_m": 1000.0, "label": "MODIS (1,000 m × 1,000 m · 1.00 km²)"},
    "VIIRS": {"pixel_m": 375.0, "label": "VIIRS (375 m × 375 m · 0.14 km²)"},
    "GOES": {"pixel_m": 2000.0, "label": "GOES-ABI (2,000 m × 2,000 m · 4.00 km²)"},
    "ASTER": {"pixel_m": 90.0, "label": "ASTER (90 m × 90 m · 0.008 km²)"},
    "LANDSAT": {"pixel_m": 30.0, "label": "Landsat (30 m × 30 m · 0.0009 km²)"},
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
        width: 365px !important;
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
            # Conservar solo geometría para reducir peso en memoria
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
                "color": [56, 189, 248, 95] if is_prime else [148, 163, 184, 38],
            }
        )
    for lat in range(-75, 90, 15):
        pts = [[float(lo), float(lat)] for lo in np.linspace(-180.0, 180.0, 120)]
        is_equator = lat == 0
        graticule_paths.append(
            {
                "path": pts,
                "color": [56, 189, 248, 95] if is_equator else [148, 163, 184, 38],
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


# 6. Conversión vectorizada de color hexadecimal y geometría hexagonal física por sensor
def hex_series_to_rgba(hex_series: pd.Series, alpha: int = 205) -> list[list[int]]:
    cleaned = hex_series.fillna("#2ECC71").astype(str).str.strip().str.lstrip("#")
    cleaned = cleaned.apply(lambda h: "".join(c * 2 for c in h) if len(h) == 3 else h)
    cleaned = cleaned.apply(lambda h: h if len(h) in (6, 8) else "2ECC71")

    r = cleaned.str.slice(0, 2).apply(lambda x: int(x, 16)).to_numpy(dtype=np.uint8)
    g = cleaned.str.slice(2, 4).apply(lambda x: int(x, 16)).to_numpy(dtype=np.uint8)
    b = cleaned.str.slice(4, 6).apply(lambda x: int(x, 16)).to_numpy(dtype=np.uint8)
    a = np.full(len(cleaned), alpha, dtype=np.uint8)

    return np.column_stack([r, g, b, a]).tolist()


def get_sensor_pixel_meters(instrument_series: pd.Series) -> np.ndarray:
    """Devuelve el ancho nominal de píxel en metros según el instrumento (MODIS, VIIRS, GOES, etc.)."""
    upper = instrument_series.str.upper()
    pixels = np.full(len(upper), 1000.0, dtype=np.float64)  # MODIS por defecto (1,000 m)
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


@st.cache_data
def build_sensor_footprint_hexagons(
    df: pd.DataFrame,
    footprint_mode: str = "Resolución del Sensor (1:1 Físico)",
    scale_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Construye hexágonos cuyo tamaño corresponde exactamente al área física medida
    por cada instrumento (MODIS 1 km, VIIRS 375 m, GOES 2 km) o a la celda 0.01° del grid.
    """
    if df.empty:
        return pd.DataFrame()

    lats = df["lat_grid"].to_numpy(dtype=np.float64)
    lons = df["lon_grid"].to_numpy(dtype=np.float64)
    pixel_m = get_sensor_pixel_meters(df["instrumento"])

    if footprint_mode == "Celda exacta Grid Backend (0.01° × 0.01°)":
        # Apotema = 0.005° -> circunradio R = 0.01 / sqrt(3) = 0.0057735° (sin solapamiento entre celdas)
        r_lat_deg = np.full(len(df), (0.01 / np.sqrt(3.0)) * scale_multiplier, dtype=np.float64)
        # En latitud/longitud de 0.01° x 0.01°:
        r_lon_deg = r_lat_deg.copy()
        area_km2 = ((0.01 * METERS_PER_DEG_LAT / 1000.0) ** 2) * np.cos(np.deg2rad(lats))
        resolucion_txt = np.full(len(df), "Grid 0.01° (~1.11 km)", dtype=object)
    else:
        # Área física exacta del píxel W x W: R_eq = W * sqrt(2 / (3 * sqrt(3))) = W * 0.62040324
        r_meters = pixel_m * np.sqrt(2.0 / (3.0 * np.sqrt(3.0))) * scale_multiplier
        r_lat_deg = r_meters / METERS_PER_DEG_LAT
        lon_scale = np.maximum(0.15, np.cos(np.deg2rad(lats)))
        r_lon_deg = r_lat_deg / lon_scale
        area_km2 = ((pixel_m / 1000.0) ** 2) * (scale_multiplier**2)
        resolucion_txt = [f"{int(w)} m × {int(w)} m" for w in pixel_m]

    angles = np.deg2rad(np.arange(0, 360, 60))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    v_lons = lons[:, None] + (r_lon_deg[:, None] * cos_a[None, :])
    v_lats = lats[:, None] + (r_lat_deg[:, None] * sin_a[None, :])
    coords = np.stack([v_lons, v_lats], axis=-1).round(6).tolist()

    rgba_colors = hex_series_to_rgba(df["color"], alpha=210)

    return pd.DataFrame(
        {
            "polygon": coords,
            "rgba": rgba_colors,
            "lat_grid": df["lat_grid"].tolist(),
            "lon_grid": df["lon_grid"].tolist(),
            "total_incendios": df["total_incendios"].tolist(),
            "frp_promedio": df["frp_promedio"].tolist(),
            "brillo_promedio": df["brillo_promedio"].tolist(),
            "instrumento": df["instrumento"].tolist(),
            "satelite": df["satelite"].tolist(),
            "mes": df["mes"].tolist(),
            "nivel_riesgo": df["nivel_riesgo"].tolist(),
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
            f"border-radius:10px;padding:7px 6px;text-align:center;color:#38bdf8;font-size:12px;font-weight:700;'>"
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

        # VENTANA 2: Proyección 2D/3D y Escala Física de Hexágonos por Sensor
        with st.expander("🗺️ Proyección y Cobertura del Sensor", expanded=False):
            projection_mode = st.radio(
                "Tipo de Visualización:",
                options=["Plano (2D)", "Esfera 3D (Globo)"],
                index=1,
            )
            if projection_mode == "Esfera 3D (Globo)":
                show_world_map_3d = st.checkbox("Mapamundi y Fronteras (3D)", value=True)
                show_graticule_3d = st.checkbox("Paralelos y Meridianos (3D)", value=True)
            else:
                show_world_map_3d = False
                show_graticule_3d = False

            show_base_grid = st.checkbox("Grid Geodésico Global", value=True)
            show_hex_readings = st.checkbox("Hexágonos de Medición", value=True)

            footprint_mode = st.radio(
                "Cobertura espacial de cada hexágono:",
                options=[
                    "Resolución del Sensor (1:1 Físico)",
                    "Celda exacta Grid Backend (0.01° × 0.01°)",
                ],
                index=0,
            )
            scale_multiplier = st.select_slider(
                "Multiplicador visual de escala:",
                options=[1.0, 2.0, 4.0, 8.0, 15.0],
                value=1.0,
                format_func=lambda x: f"{int(x)}x (Exacto 1:1)" if x == 1.0 else f"{int(x)}x (Ampliado)",
            )
            st.caption(
                "MODIS: 1,000 m × 1,000 m (1.00 km²) · VIIRS: 375 m × 375 m (0.14 km²) · GOES: 2,000 m × 2,000 m (4.00 km²)"
            )

        # VENTANA 3: Métricas Ponderadas y Superficie Cubierta
        with st.expander("📊 Análisis Térmico y Cobertura", expanded=False):
            if not filtered_df.empty:
                hex_preview = build_sensor_footprint_hexagons(filtered_df, footprint_mode, scale_multiplier)
                total_inc = int(filtered_df["total_incendios"].sum())
                total_area_km2 = float(hex_preview["area_km2"].sum())
                weights = filtered_df["total_incendios"].to_numpy(dtype=np.float64)
                brillo_pond = np.average(filtered_df["brillo_promedio"], weights=weights)

                c1, c2, c3 = st.columns(3)
                c1.metric("Incendios", f"{total_inc:,}")
                c2.metric("Área Sensor", f"{total_area_km2:,.1f} km²")
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
                    hex_preview[
                        ["lat_grid", "lon_grid", "instrumento", "resolucion_m", "area_km2", "total_incendios", "nivel_riesgo"]
                    ]
                    .sort_values("total_incendios", ascending=False)
                    .head(15),
                    use_container_width=True,
                    height=145,
                )
            else:
                st.warning("Sin celdas para los filtros seleccionados.")
    else:
        projection_mode = "Esfera 3D (Globo)"
        show_world_map_3d = True
        show_graticule_3d = True
        show_base_grid = True
        show_hex_readings = True
        footprint_mode = "Resolución del Sensor (1:1 Físico)"
        scale_multiplier = 1.0


# 8. RENDERIZADO EN PYDECK (2D O ESFERA 3D CON MAPAMUNDI Y HUELLA EXACTA DEL SENSOR)
is_globe = projection_mode == "Esfera 3D (Globo)"
grids_dict = load_multilevel_grids(is_globe=is_globe)

if not filtered_df.empty:
    center_lat = float(filtered_df["lat_grid"].mean())
    center_lon = float(filtered_df["lon_grid"].mean())
elif not df_grid.empty:
    center_lat = float(df_grid["lat_grid"].mean())
    center_lon = float(df_grid["lon_grid"].mean())
else:
    center_lat, center_lon = -31.5, -63.5

base_zoom = 3.8 if not is_globe else 2.2
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

# Grid Geodésico Global multinivel
if show_base_grid:
    level_specs = [
        ("L1", 0.0, 2.5, [0, 210, 255, 42]),
        ("L2", 2.5, 4.0, [0, 210, 255, 52]),
        ("L3", 4.0, 5.5, [0, 210, 255, 62]),
        ("L4", 5.5, 13.0, [0, 210, 255, 72]),
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

# Hexágonos de datos dimensionados según la resolución física del instrumento
if show_hex_readings and not filtered_df.empty:
    df_hex_sensor = build_sensor_footprint_hexagons(
        filtered_df,
        footprint_mode=footprint_mode,
        scale_multiplier=scale_multiplier,
    )
    active_layers.append(
        pdk.Layer(
            "PolygonLayer",
            data=df_hex_sensor,
            get_polygon="polygon",
            get_fill_color="rgba",
            get_line_color="rgba",
            line_width_min_pixels=2,
            filled=True,
            stroked=True,
            pickable=True,
            auto_highlight=True,
        )
    )

if is_globe:
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=base_zoom,
        min_zoom=0.8,
        max_zoom=10.0,
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
        max_zoom=13.0,
        pitch=0,
        bearing=0,
    )
    map_view = pdk.View(type="MapView", controller=True, repeat=False)

deck = pdk.Deck(
    layers=active_layers,
    initial_view_state=view_state,
    views=[map_view],
    tooltip={
        "text": (
            "Sensor: {instrumento} · {resolucion_m} ({area_km2} km²)\n"
            "Riesgo: {nivel_riesgo} (Mes {mes})\n"
            "Total Incendios: {total_incendios}\n"
            "Brillo Prom.: {brillo_promedio} K | FRP Prom.: {frp_promedio} MW\n"
            "Satélite: {satelite} | Celda: [{lat_grid}, {lon_grid}]"
        )
    },
    map_style=None if is_globe else "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
)

st.pydeck_chart(deck, use_container_width=True, height=900)