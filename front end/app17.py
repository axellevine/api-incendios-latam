import json
import os
import datetime
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from backendeprueva import build_classified_json_payload

# 1. Configuración de página en modo ancho total
st.set_page_config(
    page_title="NASA Hot Spots - Argentina 2025",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
        width: 340px !important;
        max-height: calc(100vh - 32px) !important;
        overflow-y: auto !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
        padding-right: 4px !important;
    }

    /* Barra de scroll discreta para el dock */
    .st-key-dock_superior_izquierdo::-webkit-scrollbar {
        width: 5px;
    }
    .st-key-dock_superior_izquierdo::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.25);
        border-radius: 4px;
    }

    /* Estilo unificado tipo vidrio oscuro para cada ventana apilada */
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


# 4. Carga acoplada desde JSON del backend o CSV real de MODIS
@st.cache_data
def load_modis_dataset() -> pd.DataFrame:
    json_path = "modis_2025_argentina_classified.json"
    csv_path = "modis_2025_Argentina.csv"

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    elif os.path.exists(csv_path):
        payload = build_classified_json_payload(csv_path, json_path)
    else:
        st.error("No se encontró 'modis_2025_Argentina.csv' ni 'modis_2025_argentina_classified.json'.")
        return pd.DataFrame()

    df = pd.DataFrame(payload["records"])
    df["date_obj"] = pd.to_datetime(df["date"]).dt.date
    return df


df_data = load_modis_dataset()


# 5. Generadores geométricos vectorizados (2D Plano y Esfera 3D Corregida)
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
def build_readings_by_radius(df: pd.DataFrame, hex_radius: float, is_globe: bool = False) -> pd.DataFrame:
    """Construcción vectorizada en NumPy de los polígonos hexagonales para soportar >30k filas sin lag."""
    if df.empty:
        return pd.DataFrame()

    lats = df["lat"].to_numpy(dtype=np.float64)
    lons = df["lon"].to_numpy(dtype=np.float64)
    brightness = df["brightness"].to_numpy(dtype=np.float64)

    angles = np.deg2rad(np.arange(0, 360, 60))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    if is_globe:
        lon_scale = np.maximum(0.2, np.cos(np.deg2rad(lats)))[:, None]
        v_lons = lons[:, None] + (cos_a[None, :] * hex_radius / lon_scale)
    else:
        v_lons = lons[:, None] + (cos_a[None, :] * hex_radius)

    v_lats = lats[:, None] + (sin_a[None, :] * hex_radius)

    # Apilar coordenadas (N, 6, 2) -> lista de polígonos
    coords = np.stack([v_lons, v_lats], axis=-1).round(5).tolist()

    # Asignación vectorizada de colores según temperatura de brillo (K)
    colors = np.zeros((len(df), 4), dtype=np.uint8)
    colors[:] = [0, 200, 0, 170]  # Bajo (< 325 K)
    colors[brightness >= 325.0] = [173, 255, 47, 180]  # Moderado
    colors[brightness >= 355.0] = [255, 165, 0, 200]   # Alto
    colors[brightness >= 385.0] = [255, 0, 0, 215]     # Crítico

    return pd.DataFrame(
        {
            "polygon": coords,
            "color": colors.tolist(),
            "satellite": df["satellite"].tolist(),
            "brightness": df["brightness"].tolist(),
            "frp": df["frp"].tolist(),
            "severity": df["severity"].tolist(),
            "date": df["date"].tolist(),
            "time": df["time"].tolist(),
        }
    )


# 6. DOCK APILADO Y OCULTABLE EN LA ESQUINA SUPERIOR IZQUIERDA
filtered_df = df_data.copy()

with st.container(key="dock_superior_izquierdo"):
    col_toggle, col_badge = st.columns([0.68, 0.32])
    with col_toggle:
        btn_label = "👁️ Ocultar paneles" if st.session_state.dock_visible else "☰ Mostrar paneles (3)"
        if st.button(btn_label, use_container_width=True):
            st.session_state.dock_visible = not st.session_state.dock_visible
            st.rerun()
    with col_badge:
        st.markdown(
            f"<div style='background:rgba(17,21,30,0.88);border:1px solid rgba(255,255,255,0.15);"
            f"border-radius:10px;padding:7px 8px;text-align:center;color:#38bdf8;font-size:12px;font-weight:700;'>"
            f"🔥 {len(df_data):,}</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.dock_visible and not df_data.empty:
        # VENTANA 1: Filtros de Datos MODIS Argentina
        with st.expander("🔥 NASA Hot Spots · Filtros", expanded=True):
            sat_options = sorted(df_data["satellite"].unique().tolist())
            selected_sat = st.multiselect("Satélite MODIS:", sat_options, default=sat_options)

            sev_order = ["Crítico", "Alto", "Moderado", "Bajo"]
            selected_sev = st.multiselect("Severidad Térmica:", sev_order, default=sev_order)

            daynight_opt = st.radio(
                "Franja Horaria:",
                options=["Todos", "Diurno", "Nocturno"],
                horizontal=True,
                index=0,
            )

            min_conf = st.slider("Confianza mínima (%):", min_value=60, max_value=100, value=60, step=5)

            min_d, max_d = df_data["date_obj"].min(), df_data["date_obj"].max()
            date_range = st.date_input("Rango de fechas (2025):", value=(min_d, max_d), min_value=min_d, max_value=max_d)

        # Aplicar filtros
        filtered_df = df_data[
            (df_data["satellite"].isin(selected_sat))
            & (df_data["severity"].isin(selected_sev))
            & (df_data["confidence"] >= min_conf)
        ]
        if daynight_opt != "Todos":
            filtered_df = filtered_df[filtered_df["daynight_label"] == daynight_opt]
        if isinstance(date_range, tuple) and len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df["date_obj"] >= date_range[0]) & (filtered_df["date_obj"] <= date_range[1])
            ]

        # VENTANA 2: Controles de Proyección y Capas
        with st.expander("🗺️ Controles y Proyección", expanded=False):
            projection_mode = st.radio(
                "Tipo de Visualización:",
                options=["Plano (2D)", "Esfera 3D (Globo)"],
                index=0,
            )
            show_base_grid = st.checkbox("Grid Geodésico Global", value=True)
            show_hex_readings = st.checkbox("Hexágonos de Lecturas", value=True)

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

        # VENTANA 3: Análisis Térmico en Tiempo Real
        with st.expander("📊 Análisis Térmico (Argentina)", expanded=False):
            c1, c2 = st.columns(2)
            c1.metric("Focos activos", f"{len(filtered_df):,}")
            avg_b = f"{filtered_df['brightness'].mean():.1f} K" if not filtered_df.empty else "0 K"
            c2.metric("Brillo Prom.", avg_b)

            if not filtered_df.empty:
                trend_data = filtered_df.groupby("date").size().reset_index(name="Focos")
                st.line_chart(trend_data.set_index("date"), height=130)

                st.dataframe(
                    filtered_df[["date", "time", "satellite", "brightness", "frp", "severity"]].head(15),
                    use_container_width=True,
                    height=140,
                )
            else:
                st.warning("Sin registros para los filtros seleccionados.")
    else:
        # Valores por defecto cuando el dock está oculto
        projection_mode = "Plano (2D)"
        show_base_grid = True
        show_hex_readings = True
        zoom_mode = "Zoom Dinámico"
        selected_zoom = 100

# 7. RENDERIZADO SEGÚN EL MODO (2D o 3D ESFERA) CENTRADO EN ARGENTINA
is_globe = projection_mode == "Esfera 3D (Globo)"
grids_dict = load_multilevel_grids(is_globe=is_globe)

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
            df_r = build_readings_by_radius(filtered_df, r_deg, is_globe=is_globe)
            active_layers.append(
                pdk.Layer(
                    "PolygonLayer",
                    data=df_r,
                    get_polygon="polygon",
                    get_fill_color="color",
                    get_line_color=[255, 255, 255, 90],
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
        df_manual_readings = build_readings_by_radius(filtered_df, manual_hex_radius, is_globe=is_globe)
        active_layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=df_manual_readings,
                get_polygon="polygon",
                get_fill_color="color",
                get_line_color=[255, 255, 255, 90],
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
        "text": "Satélite: {satellite} ({severity})\nBrillo: {brightness} K | FRP: {frp} MW\nFecha: {date} {time} UTC"
    },
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
)

st.pydeck_chart(deck, use_container_width=True, height=900)