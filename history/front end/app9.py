import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pydeck as pdk

# 1. Configuración de página en modo ancho total
st.set_page_config(
    page_title="NASA Hot Spots - Mapa Geodésico 2D",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Estilos CSS para asegurar un único mapa contenedor y ventanas flotantes profesionales
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
        width: 290px;
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
        width: 290px !important;
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

    /* Asegurar renderizado único del visor cartográfico en pantalla completa */
    iframe[title="streamlit_pydeck.pydeck_chart"], .stDeckGlJsonChart {
        height: 100vh !important;
        width: 100vw !important;
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

# 4. Funciones de generación geométrica adaptadas a la escala de zoom
@st.cache_data
def generate_geodesic_grid(zoom_val):
    # Relacionar la escala de zoom personalizada con el tamaño de celda del grid
    # Zoom alto (500) -> celdas grandes (macro); Zoom bajo (0) -> celdas diminutas (micro)
    radius_deg = max(0.5, (500 - zoom_val) / 35.0)
    
    polygons = []
    dx = radius_deg * 1.5
    dy = radius_deg * np.sqrt(3)
    
    lats = np.arange(-65, 78, dy)
    lons = np.arange(-175, 178, dx)
    
    for col_idx, lon in enumerate(lons):
        lat_offset = (dy / 2.0) if (col_idx % 2 == 1) else 0.0
        for lat in lats:
            c_lat = lat + lat_offset
            if c_lat > 78 or c_lat < -65:
                continue
            vertices = []
            for i in range(6):
                angle_rad = np.deg2rad(60 * i)
                v_lon = lon + radius_deg * np.cos(angle_rad)
                v_lat = c_lat + radius_deg * np.sin(angle_rad)
                vertices.append([v_lon, v_lat])
            polygons.append({"polygon": vertices})
            
    return pd.DataFrame(polygons)

@st.cache_data
def create_hexagons_from_points(df, zoom_val):
    # Escalar el tamaño del hexágono de lectura según el nivel de zoom seleccionado
    hex_radius = max(0.2, (500 - zoom_val) / 120.0)
    polygons = []
    
    for _, row in df.iterrows():
        lat, lon = row["lat"], row["lon"]
        brightness = row["brightness"]
        
        vertices = []
        for i in range(6):
            angle_rad = np.deg2rad(60 * i)
            v_lon = lon + hex_radius * np.cos(angle_rad)
            v_lat = lat + hex_radius * np.sin(angle_rad)
            vertices.append([v_lon, v_lat])
            
        if brightness > 430:
            color = [255, 0, 0, 200]
        elif brightness > 400:
            color = [255, 140, 0, 190]
        elif brightness > 370:
            color = [255, 255, 0, 180]
        else:
            color = [0, 200, 255, 160]
            
        polygons.append({
            "polygon": vertices,
            "color": color,
            "satellite": row["satellite"],
            "brightness": brightness,
            "country": row["country"],
            "date": str(row["date"])
        })
    return pd.DataFrame(polygons)

# ==========================================
# 5. CONTROLES EN LA BARRA LATERAL Y VENTANAS
# ==========================================
with st.sidebar:
    st.header("⚙️ Filtros Globales")
    selected_country = st.selectbox("🌍 Región / País:", ["Todo el Mundo"] + list(regions_coords.keys()))
    selected_satellite = st.multiselect("🛰️ Satélite:", df_data["satellite"].unique(), default=df_data["satellite"].unique())
    min_date, max_date = df_data["date"].min(), df_data["date"].max()
    date_range = st.date_input("📅 Fechas:", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Filtrar datos
filtered_df = df_data.copy()
if selected_country != "Todo el Mundo":
    filtered_df = filtered_df[filtered_df["country"] == selected_country]

if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["satellite"].isin(selected_satellite)) &
        (filtered_df["date"] >= date_range[0]) &
        (filtered_df["date"] <= date_range[1])
    ]

# Ventana flotante superior izquierda
st.markdown("""
    <div class="floating-card-left">
        <h3 style="margin: 0; font-size: 18px;">🔥 NASA Hot Spots</h3>
        <p style="font-size: 11px; color: #aaa; margin: 4px 0 0 0;">Grid Geodésico & Lecturas 2D</p>
    </div>
""", unsafe_allow_html=True)

# Ventana flotante inferior izquierda (Capas y Zoom personalizado exacto)
with st.container(key="ventana_capas"):
    st.markdown("<h4 style='margin:0 0 8px 0; font-size:14px;'>🗺️ Controles y Zoom</h4>", unsafe_allow_html=True)
    show_base_grid = st.checkbox("Grid Geodésico Global", value=True)
    show_hex_readings = st.checkbox("Hexágonos de Lecturas", value=True)
    
    # Selector estricto con los niveles de zoom solicitados
    zoom_levels_list = [500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100, 75, 50, 25, 0]
    selected_zoom = st.select_slider("Nivel de Zoom Custom", options=zoom_levels_list, value=250)

# Ventana flotante superior derecha (Estadísticas)
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

# Generar datos geométricos basados en el zoom seleccionado
df_geodesic_grid = generate_geodesic_grid(selected_zoom)
filtered_hex_df = create_hexagons_from_points(filtered_df, selected_zoom)

# ==========================================
# 6. RENDERIZADO DEL MAPA 2D ÚNICO
# ==========================================
if selected_country != "Todo el Mundo" and selected_country in regions_coords:
    center_lat, center_lon = regions_coords[selected_country]
    map_zoom = 5.5
else:
    center_lat, center_lon = 15.0, 10.0
    map_zoom = 1.8

active_layers = []

# 1. Capa del Grid Geodésico Global independiente
if show_base_grid:
    base_grid_layer = pdk.Layer(
        "PolygonLayer",
        data=df_geodesic_grid,
        get_polygon="polygon",
        filled=False,
        stroked=True,
        get_line_color=[0, 210, 255, 45],
        line_width_min_pixels=1,
        pickable=False
    )
    active_layers.append(base_grid_layer)

# 2. Capa de Hexágonos individuales de lecturas satelitales
if show_hex_readings and not filtered_hex_df.empty:
    hex_readings_layer = pdk.Layer(
        "PolygonLayer",
        data=filtered_hex_df,
        get_polygon="polygon",
        get_fill_color="color",
        get_line_color=[255, 255, 255, 120],
        line_width_min_pixels=1,
        filled=True,
        stroked=True,
        pickable=True,
        auto_highlight=True
    )
    active_layers.append(hex_readings_layer)

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=map_zoom,
    pitch=0,  # 2D puro
    bearing=0
)

#