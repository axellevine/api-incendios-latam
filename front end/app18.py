import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

# 1. Configuración de página en modo ancho total
st.set_page_config(
    page_title="NASA Hot Spots · Grid Mensual",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_API_URL = os.getenv("INCENDIOS_API_URL", "http://localhost:8000/api/incendios-grid")

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

    /* Contenedor maestro apilado en la esquina superior izquierda */
    .st-key-dock_superior_izquierdo {
        position: fixed !important;
        top: 16px !important;
        left: 16px !important;
        z-index: 9999 !important;
        width: 350px !important;
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

    /* Ventanas flotantes apiladas */
    .st-key-dock_superior_izquierdo div[data-testid="stExpander"] {
        background: rgba(17, 21, 30, 0.88) !important;
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

    /* Botón maestro de ocultar/mostrar dock */
    .st-key-dock_superior_izquierdo button[kind="secondary"] {
        background: rgba(17, 21, 30, 0.92) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
    }

    /* Mapa / Globo ocupando el 100% del viewport */
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


# 4. Cliente de datos para /api/incendios-grid + respaldo local con el mismo esquema
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
    df["instrumento"] = df["instrumento"].fillna("MODIS").astype(str)
    df["satelite"] = df["satelite"].fillna("Terra").astype(str)
    df["mes"] = pd.to_numeric(df["mes"], errors="coerce").fillna(1).astype(int).clip(1, 12)
    df["nivel_riesgo"] = df["nivel_riesgo"].fillna("Bajo").astype(str)
    df["color"] = df["color"].fillna("#2ECC71").astype(str)

    return df.dropna(subset=["lat_grid", "lon_grid"]).reset_index(drop=True)


@st.cache_data(ttl=120)
def fetch_incendios_grid(api_url: str) -> tuple[pd.DataFrame, str]:
    """Consulta el endpoint /api/incendios-grid; si no responde, usa CSV local con el mismo esquema."""
    try:
        resp = requests.get(api_url, timeout=10)
        resp.raise_for_status()
        payload = resp.json()

        # Soporta tanto array directo [...] como objeto {"data": [...]} o {"records": [...]}
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("data") or payload.get("records") or payload.get("grid") or []
        else:
            records = []

        return _normalize_grid_dataframe(records), f"API Conectada ({api_url})"
    except Exception as api_err:
        # Respaldo local generando exactamente la misma estructura que el backend de tu compañero
        local_csv = Path(__file__).resolve().parent / "modis_2025_Argentina.csv"
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

            return _normalize_grid_dataframe(grouped.to_dict(orient="records")), f"Modo Local (API offline: {api_err})"

        return pd.DataFrame(columns=EXPECTED_COLUMNS), f"Sin conexión ({api_err})"


# 5. Conversión vectorizada de color hexadecimal (#FF0000) a RGBA y geometría hexagonal
def hex_series_to_rgba(hex_series: pd.Series, alpha: int = 200) -> list[list[int]]:
    cleaned = (
        hex_series.fillna("#2ECC71")
        .astype(str)
        .str.strip()
        .str.lstrip("#")
    )
    cleaned = cleaned.apply(lambda h: "".join(c * 2 for c in h) if len(h) == 3 else h)
    cleaned = cleaned.apply(lambda h: h if len(h) in (6, 8) else "2ECC71")

    r = cleaned.str.slice(0, 2).apply(lambda x: int(x, 16)).to_numpy(dtype=np.uint8)
    g = cleaned.str.slice(2, 4).apply(lambda x: int(x, 16)).to_numpy(dtype=np.uint8)
    b = cleaned.str.slice(4, 6).apply(lambda x: int(x, 16)).to_numpy(dtype=np.uint8)
    a = np.full(len(cleaned), alpha, dtype=np.uint8)

    return np.column_stack([r, g, b, a]).tolist()


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
def build_grid_cells_layer(df: pd.DataFrame, hex_radius: float, is_globe: bool = False) -> pd.DataFrame:
    """Genera polígonos hexagonales en las coordenadas lat_grid/lon_grid usando el color exacto del backend."""
    if df.empty:
        return pd.DataFrame()

    lats = df["lat_grid"].to_numpy(dtype=np.float64)
    lons = df["lon_grid"].to_numpy(dtype=np.float64)

    angles = np.deg2rad(np.arange(0, 360, 60))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    if is_globe:
        lon_scale = np.maximum(0.2, np.cos(np.deg2rad(lats)))[:, None]
        v_lons = lons[:, None] + (cos_a[None, :] * hex_radius / lon_scale)
    else:
        v_lons = lons[:, None] + (cos_a[None, :] * hex_radius)

    v_lats = lats[:, None] + (sin_a[None, :] * hex_radius)
    coords = np.stack([v_lons, v_lats], axis=-1).round(5).tolist()

    rgba_colors = hex_series_to_rgba(df["color"], alpha=205)

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
        }
    )


# 6. DOCK APILADO Y OCULTABLE EN LA ESQUINA SUPERIOR IZQUIERDA
api_endpoint = st.session_state.get("api_endpoint", DEFAULT_API_URL)
df_grid, connection_status = fetch_incendios_grid(api_endpoint)
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
        st.markdown(
            f"<div style='background:rgba(17,21,30,0.88);border:1px solid rgba(255,255,255,0.15);"
            f"border-radius:10px;padding:7px 6px;text-align:center;color:#38bdf8;font-size:12px;font-weight:700;'>"
            f"🔥 {total_fires_badge:,}</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.dock_visible:
        # VENTANA 1: Conexión API y Filtros del Esquema /api/incendios-grid
        with st.expander("🔥 Grid de Incendios · Filtros", expanded=True):
            new_url = st.text_input("Endpoint Backend:", value=api_endpoint, key="api_endpoint")
            st.caption(f"Estado: {connection_status}")

            if not df_grid.empty:
                # Control temporal por 'mes' (1 al 12)
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

                # Filtro por nivel_riesgo
                riesgos_orden = [r for r in ["Alto", "Medio", "Bajo"] if r in df_grid["nivel_riesgo"].unique()]
                if not riesgos_orden:
                    riesgos_orden = sorted(df_grid["nivel_riesgo"].unique().tolist())
                selected_riesgo = st.multiselect("Nivel de Riesgo:", riesgos_orden, default=riesgos_orden)

                # Filtro por satelite e instrumento
                sat_options = sorted(df_grid["satelite"].unique().tolist())
                selected_sat = st.multiselect("Satélite:", sat_options, default=sat_options)

                inst_options = sorted(df_grid["instrumento"].unique().tolist())
                selected_inst = st.multiselect("Instrumento:", inst_options, default=inst_options)

                filtered_df = filtered_df[
                    (filtered_df["nivel_riesgo"].isin(selected_riesgo))
                    & (filtered_df["satelite"].isin(selected_sat))
                    & (filtered_df["instrumento"].isin(selected_inst))
                ]

        # VENTANA 2: Controles de Proyección y Capas
        with st.expander("🗺️ Controles y Proyección", expanded=False):
            projection_mode = st.radio(
                "Tipo de Visualización:",
                options=["Plano (2D)", "Esfera 3D (Globo)"],
                index=0,
            )
            show_base_grid = st.checkbox("Grid Geodésico Global", value=True)
            show_hex_readings = st.checkbox("Celdas de Incendios (Backend)", value=True)

            zoom_mode = st.radio(
                "Comportamiento del Grid:",
                options=["Zoom Dinámico", "Zoom Manual"],
                index=0,
            )
            zoom_levels_list = [0, 50, 100, 150, 200, 250, 300, 400, 500]
            if zoom_mode == "Zoom Manual":
                selected_zoom = st.select_slider("Escala Manual:", options=zoom_levels_list, value=150)
            else:
                selected_zoom = 100

        # VENTANA 3: Métricas Ponderadas y Resumen de Celdas
        with st.expander("📊 Análisis Térmico por Celda", expanded=False):
            if not filtered_df.empty:
                total_inc = int(filtered_df["total_incendios"].sum())
                weights = filtered_df["total_incendios"].to_numpy(dtype=np.float64)
                brillo_pond = np.average(filtered_df["brillo_promedio"], weights=weights)
                frp_pond = np.average(filtered_df["frp_promedio"], weights=weights)

                c1, c2, c3 = st.columns(3)
                c1.metric("Incendios", f"{total_inc:,}")
                c2.metric("Brillo Prom.", f"{brillo_pond:.1f} K")
                c3.metric("FRP Prom.", f"{frp_pond:.1f}")

                # Evolución mensual usando df_grid filtrado por riesgo/satélite
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
                st.bar_chart(mensual, height=130)

                st.dataframe(
                    filtered_df[
                        ["lat_grid", "lon_grid", "total_incendios", "brillo_promedio", "frp_promedio", "nivel_riesgo"]
                    ]
                    .sort_values("total_incendios", ascending=False)
                    .head(15),
                    use_container_width=True,
                    height=145,
                )
            else:
                st.warning("Sin celdas para los filtros seleccionados.")
    else:
        projection_mode = "Plano (2D)"
        show_base_grid = True
        show_hex_readings = True
        zoom_mode = "Zoom Dinámico"
        selected_zoom = 100


# 7. RENDERIZADO EN PYDECK USANDO lat_grid, lon_grid Y color DEL BACKEND
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

if zoom_mode == "Zoom Dinámico":
    map_zoom = base_zoom

    if show_base_grid:
        level_specs = [
            ("L1", 0.0, 2.5, [0, 210, 255, 45]),
            ("L2", 2.5, 4.0, [0, 210, 255, 55]),
            ("L3", 4.0, 5.5, [0, 210, 255, 65]),
            ("L4", 5.5, 12.0, [0, 210, 255, 75]),
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

    if show_hex_readings and not filtered_df.empty:
        reading_specs = [
            (0.45, 0.0, 3.5),
            (0.22, 3.5, 5.0),
            (0.11, 5.0, 7.0),
            (0.05, 7.0, 12.0),
        ]
        for r_deg, z_min, z_max in reading_specs:
            df_r = build_grid_cells_layer(filtered_df, r_deg, is_globe=is_globe)
            active_layers.append(
                pdk.Layer(
                    "PolygonLayer",
                    data=df_r,
                    get_polygon="polygon",
                    get_fill_color="rgba",
                    get_line_color=[255, 255, 255, 95],
                    line_width_min_pixels=1,
                    filled=True,
                    stroked=True,
                    zoom_min=z_min,
                    zoom_max=z_max,
                    pickable=True,
                    auto_highlight=True,
                )
            )
else:
    map_zoom = base_zoom + (selected_zoom / 500.0) * 4.0

    if show_base_grid:
        manual_grid_radius = round(10.0 - (selected_zoom / 500.0) * 8.8, 2)
        df_manual_grid = build_grid_by_radius(manual_grid_radius, is_globe=is_globe)
        active_layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=df_manual_grid,
                get_polygon="polygon",
                filled=False,
                stroked=True,
                get_line_color=[0, 210, 255, 55],
                line_width_min_pixels=1,
                pickable=False,
            )
        )

    if show_hex_readings and not filtered_df.empty:
        manual_hex_radius = round(0.50 - (selected_zoom / 500.0) * 0.44, 3)
        df_manual_readings = build_grid_cells_layer(filtered_df, manual_hex_radius, is_globe=is_globe)
        active_layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=df_manual_readings,
                get_polygon="polygon",
                get_fill_color="rgba",
                get_line_color=[255, 255, 255, 95],
                line_width_min_pixels=1,
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
        zoom=map_zoom,
        min_zoom=0.8,
        max_zoom=7.0,
    )
    map_view = pdk.View(type="GlobeView", controller=True, clear_color=[10 / 255, 14 / 255, 23 / 255, 1.0])
else:
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=map_zoom,
        min_zoom=1.2,
        max_zoom=11.0,
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
            "Riesgo: {nivel_riesgo} (Mes {mes})\n"
            "Total Incendios: {total_incendios}\n"
            "Brillo Prom.: {brillo_promedio} K | FRP Prom.: {frp_promedio} MW\n"
            "Satélite: {satelite} ({instrumento})\n"
            "Celda: [{lat_grid}, {lon_grid}]"
        )
    },
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
)

st.pydeck_chart(deck, use_container_width=True, height=900)