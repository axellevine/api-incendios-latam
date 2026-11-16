import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pydeck as pdk

# 1. Configuración de página en modo ancho total
st.set_page_config(
    page_title="NASA Hot Spots - Esfera y 2D Corregidos",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Estilos CSS para ventanas flotantes y pantalla completa
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    
    .block-container {
        padding: 0rem !important;
        max-width: 100% !important;
        height: 100vh !important;
        overflow: hidden !important;
    }

    .floating-card-left {
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 9999;
        background: rgba(20, 24, 33, 0.88);
        backdrop-filter: blur(12px);
        padding: 15px 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: white;
        width: 300px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    }

    .st-key-ventana_capas {
        position: fixed !important;
        bottom: 20px !important;
        left: 20px !important;
        z-index: 9999 !important;
        background: rgba(20, 24, 33, 0.88) !important;
        backdrop-filter: blur(12px) !important;
        padding: 15px 20px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: white !important;
        width: 300px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45) !important;
    }

    .st-key-ventana_derecha {
        position: fixed !important;
        top: 20px !important;
        right: 20px !important;
        z-index: 9999 !important;
        background: rgba(20, 24, 33, 0.88) !important;
        backdrop-filter: blur(12px) !important;
        padding: 18px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        width: 360px !important;
        max-height: 85vh !important;
        overflow-y: auto !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45) !important;
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
""", unsafe_allow_html=True)

# 3. Generación de datos simulados GLOBALES
@st.cache_data
def load_data():
    np.random.seed(42)
    n_rows = 1500
    
    regions = {
        "Colombia": (4.5, -74.0),
        "Brasil (Amazonía)": (-10.0, -55.0),
        "Estados Unidos (California)": (37.0, -119.5),
        "Canadá (Boreal)": (56.0, -106.0),
        "África Central (Congo)": (-2.0, 23.0),
        "Australia": (-25.0, 133.0),
        "Indonesia": (-2.5, 118.0),
        "Siberia (Rusia)": (61.0, 105.0),
        "Europa Mediterránea": (40.0, 15.0),
        "India": (21.0, 78.0)
    }
    
    region_names = list(regions.keys())
    assigned_regions = np.random.choice(region_names, n_rows)
    lats, lons = [], []
    
    for r in assigned_regions:
        base_lat, base_lon = regions[r]
        lats.append(np.clip(base_lat + np.random.normal(0, 8.0), -60.0, 75.0))
        lons.append(np.clip(base_lon + np.random.normal(0, 12.0), -170.0, 175.0))
        
    satellites = np.random.choice(["MODIS", "VIIRS"], n_rows, p=[0.4, 0.6])
    brightness = np.random.uniform(310, 460, n_rows)
    start_date = datetime.date(2026, 1, 1)
    dates = [start_date + datetime.timedelta(days=int(np.random.randint(0, 90))) for _ in range(n_rows)]
    
    df = pd.DataFrame({
        "lat": lats,
        "lon": lons,
        "country": assigned_regions,
        "satellite": satellites,
        "brightness": brightness,
        "date": dates
    })
    return df, regions

df_data, regions_coords = load_data()

# 4. Generadores geométricos con corrección de escala para Esfera 3D
@st.cache_data
def build_grid_by_radius(radius_deg, is_globe=False):
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
            
            # En esfera 3D corregimos la compresión de longitud debida al coseno de la latitud
            if is_globe:
                lon_scale = max(0.2, np.cos(np.deg2rad(c_lat)))
                v_lons = lon + (cos_a * radius_deg / lon_scale)
                v_lats = c_lat + (sin_a * radius_deg)
                vertices = [[float(v_lons[i]), float(v_lats[i])] for i in range(6)]
            else:
                vertices = [[float(lon + cos_a[i] * radius_deg), float(c_lat + sin_a[i] * radius_deg)] for i in range(6)]
                
            polygons.append({"polygon": vertices})
            
    return pd.DataFrame(polygons)

@st.cache_data
def load_multilevel_grids(is_globe=False):
    return {
        "L1": build_grid_by_radius(9.0, is_globe),
        "L2": build_grid_by_radius(4.5, is_globe),
        "L3": build_grid_by_radius(2.25, is_globe),
        "L4": build_grid_by_radius(1.125, is_globe),
    }

@st.cache_data
def build_readings_by_radius(df, hex_radius, is_globe=False):
    if df.empty:
        return pd.DataFrame()
        
    angles = np.deg2rad(np.arange(0, 360, 60))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    polygons = []
    for _, row in df.iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        brightness = float(row["brightness"])
        
        if is_globe:
            lon_scale = max(0.2, np.cos(np.deg2rad(lat)))
            v_lons = lon + (cos_a * hex_radius / lon_scale)
            v_lats = lat + (sin_a * hex_radius)
            vertices = [[float(v_lons[i]), float(v_lats[i])] for i in range(6)]
        else:
            vertices = [[lon + float(cos_a[i] * hex_radius), lat + float(sin_a[i] * hex_radius)] for i in range(6)]
            
        if brightness > 435:
            color = [255, 0, 0, 210]       # Rojo
        elif brightness > 410:
            color = [255, 128, 0, 200]     # Naranja
        elif brightness > 385:
            color = [255, 255, 0, 190]     # Amarillo
        elif brightness > 355:
            color = [173, 255, 47, 180]    # Amarillo verdoso
        else:
            color = [0, 200, 0, 170]       # Verde
            
        polygons.append({
            "polygon": vertices,
            "color": color,
            "satellite": row["satellite"],
            "brightness": round(brightness, 1),
            "country": row["country"],
            "date": str(row["date"])
        })
    return pd.DataFrame(polygons)

# ==========================================
# 5. FILTROS Y VENTANAS FLOTANTES
# ==========================================
with st.sidebar:
    st.header("⚙️ Filtros Globales")
    selected_country = st.selectbox("🌍 Región / País:", ["Todo el Mundo"] + list(regions_coords.keys()))
    selected_satellite = st.multiselect("🛰️ Satélite:", df_data["satellite"].unique(), default=df_data["satellite"].unique())
    min_date, max_date = df_data["date"].min(), df_data["date"].max()
    date_range = st.date_input("📅 Fechas:", value=(min_date, max_date), min_value=min_date, max_value=max_date)

filtered_df = df_data.copy()
if selected_country != "Todo el Mundo":
    filtered_df = filtered_df[filtered_df["country"] == selected_country]

if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["satellite"].isin(selected_satellite)) &
        (filtered_df["date"] >= date_range[0]) &
        (filtered_df["date"] <= date_range[1])
    ]

st.markdown("""
    <div class="floating-card-left">
        <h3 style="margin: 0; font-size: 18px;">🔥 NASA Hot Spots</h3>
        <p style="font-size: 11px; color: #aaa; margin: 4px 0 0 0;">Proyección: 2D Plano / Esfera 3D Corregida</p>
    </div>
""", unsafe_allow_html=True)

with st.container(key="ventana_capas"):
    st.markdown("<h4 style='margin:0 0 8px 0; font-size:14px;'>🗺️ Controles y Proyección</h4>", unsafe_allow_html=True)
    
    projection_mode = st.radio(
        "Tipo de Visualización:",
        options=["Plano (2D)", "Esfera 3D (Globo)"],
        index=0
    )
    
    st.markdown("<hr style='margin: 8px 0; border-color: rgba(255,255,255,0.15);'>", unsafe_allow_html=True)
    show_base_grid = st.checkbox("Grid Geodésico Global", value=True)
    show_hex_readings = st.checkbox("Hexágonos de Lecturas", value=True)
    
    st.markdown("<hr style='margin: 8px 0; border-color: rgba(255,255,255,0.15);'>", unsafe_allow_html=True)
    zoom_mode = st.radio(
        "Comportamiento del Grid:",
        options=["Zoom Dinámico", "Zoom Manual"],
        index=0
    )
    
    zoom_levels_list = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500]
    if zoom_mode == "Zoom Manual":
        selected_zoom = st.select_slider("Escala Manual:", options=zoom_levels_list, value=150)
    else:
        selected_zoom = 100

with st.container(key="ventana_derecha"):
    st.markdown("<h4 style='margin:0 0 10px 0; font-size:15px;'>📊 Análisis Térmico Global</h4>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    c1.metric("Lecturas", len(filtered_df))
    avg_b = f"{filtered_df['brightness'].mean():.1f} K" if not filtered_df.empty else "0 K"
    c2.metric("Brillo Prom.", avg_b)
    
    if not filtered_df.empty:
        trend_data = filtered_df.groupby("date").size().reset_index(name="Quemas")
        st.line_chart(trend_data.set_index("date"), height=140)
        
        with st.expander("📋 Ver registros"):
            st.dataframe(filtered_df[["date", "satellite", "brightness", "country"]].tail(10), use_container_width=True, height=150)
    else:
        st.warning("No hay datos para mostrar.")

# ==========================================
# 6. RENDERIZADO SEGÚN EL MODO (2D o 3D ESFERA)
# ==========================================
is_globe = (projection_mode == "Esfera 3D (Globo)")
grids_dict = load_multilevel_grids(is_globe=is_globe)

if selected_country != "Todo el Mundo" and selected_country in regions_coords:
    center_lat, center_lon = regions_coords[selected_country]
    base_zoom = 3.5 if not is_globe else 1.8
else:
    center_lat, center_lon = 15.0, 0.0
    base_zoom = 1.4 if not is_globe else 1.2

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
                    pickable=False
                )
            )
            
    if show_hex_readings and not filtered_df.empty:
        reading_specs = [
            (1.6, 0.0, 2.5),
            (0.9, 2.5, 4.0),
            (0.45, 4.0, 5.5),
            (0.22, 5.5, 12.0),
        ]
        for r_deg, z_min, z_max in reading_specs:
            df_r = build_readings_by_radius(filtered_df, r_deg, is_globe=is_globe)
            active_layers.append(
                pdk.Layer(
                    "PolygonLayer",
                    data=df_r,
                    get_polygon="polygon",
                    get_fill_color="color",
                    get_line_color=[255, 255, 255, 130],
                    line_width_min_pixels=1,
                    filled=True,
                    stroked=True,
                    zoom_min=z_min,
                    zoom_max=z_max,
                    pickable=True,
                    auto_highlight=True
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
                pickable=False
            )
        )
        
    if show_hex_readings and not filtered_df.empty:
        manual_hex_radius = round(1.8 - (selected_zoom / 500.0) * 1.55, 3)
        df_manual_readings = build_readings_by_radius(filtered_df, manual_hex_radius, is_globe=is_globe)
        active_layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=df_manual_readings,
                get_polygon="polygon",
                get_fill_color="color",
                get_line_color=[255, 255, 255, 130],
                line_width_min_pixels=1,
                filled=True,
                stroked=True,
                pickable=True,
                auto_highlight=True
            )
        )

if is_globe:
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=map_zoom,
        min_zoom=0.8,
        max_zoom=6.0
    )
    map_view = pdk.View(type="GlobeView", controller=True, clear_color=[10/255, 14/255, 23/255, 1.0])
else:
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=map_zoom,
        min_zoom=1.2,
        max_zoom=10.0,
        pitch=0,
        bearing=0
    )
    map_view = pdk.View(type="MapView", controller=True, repeat=False)

r = pdk.Deck(
    layers=active_layers,
    initial_view_state=view_state,
    views=[map_view],
    tooltip={"text": "Satélite: {satellite}\nBrillo: {brightness} K\nRegión: {country}\nFecha: {date}"},
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
)

st.pydeck_chart(r, use_container_width=True, height=900)