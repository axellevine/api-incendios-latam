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

# 2. Estilos CSS corrigiendo los selectores de PyDeck y asegurando un único mapa
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

    /* Selector universal para el canvas/iframe de PyDeck en pantalla completa */
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

# 4. Funciones geométricas optimizadas (evita sobrecarga de WebGL y adapta al zoom real)
@st.cache_data
def generate_geodesic_grid(zoom_val, center_lat, center_lon):
    # Mapeo lógico: zoom 0 (lejano -> celdas grandes de 10°), zoom 500 (cercano -> celdas finas de 1.2°)
    # Límite seguro >= 1.2° para que WebGL renderice fluido sin pantalla negra
    radius_deg = round(10.0 - (zoom_val / 500.0) * 8.8, 2)
    
    # Cuando hay acercamiento (zoom >= 250), recortamos la malla a la ventana visible para rendimiento instantáneo
    if zoom_val >= 250:
        span_lat = max(18.0, 75.0 - (zoom_val / 500.0) * 58.0)
        span_lon = max(28.0, 175.0 - (zoom_val / 500.0) * 145.0)
        lat_min = max(-70.0, center_lat - span_lat)
        lat_max = min(80.0, center_lat + span_lat)
        lon_min = max(-180.0, center_lon - span_lon)
        lon_max = min(180.0, center_lon + span_lon)
    else:
        lat_min, lat_max = -70.0, 80.0
        lon_min, lon_max = -180.0, 180.0

    dx = radius_deg * 1.5
    dy = radius_deg * np.sqrt(3)
    
    # Alinear índices a origen global (0,0) para que las subdivisiones encajen geométricamente
    col_start = int(np.floor(lon_min / dx))
    col_end = int(np.ceil(lon_max / dx))
    row_start = int(np.floor(lat_min / dy))
    row_end = int(np.ceil(lat_max / dy))

    angles = np.deg2rad(np.arange(0, 360, 60))
    cos_a = np.cos(angles) * radius_deg
    sin_a = np.sin(angles) * radius_deg

    polygons = []
    for col_idx in range(col_start, col_end + 1):
        lon = col_idx * dx
        lat_offset = (dy / 2.0) if (col_idx % 2 != 0) else 0.0
        for row_idx in range(row_start, row_end + 1):
            c_lat = row_idx * dy + lat_offset
            if c_lat > 80.0 or c_lat < -70.0:
                continue
            vertices = [[float(lon + cos_a[i]), float(c_lat + sin_a[i])] for i in range(6)]
            polygons.append({"polygon": vertices})
            
    return pd.DataFrame(polygons)

@st.cache_data
def create_hexagons_from_points(df, zoom_val):
    if df.empty:
        return pd.DataFrame()
        
    # El radio de cada lectura también se afina al hacer zoom (de 1.8° lejos a 0.25° cerca)
    hex_radius = round(1.8 - (zoom_val / 500.0) * 1.55, 3)
    angles = np.deg2rad(np.arange(0, 360, 60))
    cos_a = np.cos(angles) * hex_radius
    sin_a = np.sin(angles) * hex_radius

    polygons = []
    for _, row in df.iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        brightness = float(row["brightness"])
        
        vertices = [[lon + float(cos_a[i]), lat + float(sin_a[i])] for i in range(6)]
            
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
            "brightness": round(brightness, 1),
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
    
    # Lista ordenada de 0 (vista global) a 500 (máximo acercamiento)
    zoom_levels_list = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500]
    selected_zoom = st.select_slider("Nivel de Zoom Custom", options=zoom_levels_list, value=100)

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

# ==========================================
# 6. CÁLCULO DE VISTA Y RENDERIZADO DEL MAPA 2D ÚNICO
# ==========================================
if selected_country != "Todo el Mundo" and selected_country in regions_coords:
    center_lat, center_lon = regions_coords[selected_country]
    base_zoom = 3.5
else:
    center_lat, center_lon = 15.0, 0.0
    base_zoom = 1.3

# Convertir la escala personalizada (0 - 500) a zoom cartográfico real de PyDeck
map_zoom = base_zoom + (selected_zoom / 500.0) * 5.0

df_geodesic_grid = generate_geodesic_grid(selected_zoom, center_lat, center_lon) if show_base_grid else pd.DataFrame()
filtered_hex_df = create_hexagons_from_points(filtered_df, selected_zoom) if show_hex_readings else pd.DataFrame()

active_layers = []

# 1. Capa del Grid Geodésico Global independiente
if show_base_grid and not df_geodesic_grid.empty:
    base_grid_layer = pdk.Layer(
        "PolygonLayer",
        data=df_geodesic_grid,
        get_polygon="polygon",
        filled=False,
        stroked=True,
        get_line_color=[0, 210, 255, 55],
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
        get_line_color=[255, 255, 255, 130],
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
    min_zoom=1.2,
    max_zoom=10.0,
    pitch=0,
    bearing=0
)

# Instancia única del mapa global sin repetición horizontal de continentes
r = pdk.Deck(
    layers=active_layers,
    initial_view_state=view_state,
    views=[pdk.View(type="MapView", controller=True, repeat=False)],
    tooltip={"text": "Satélite: {satellite}\nBrillo: {brightness} K\nRegión: {country}\nFecha: {date}"},
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
)

st.pydeck_chart(r, use_container_width=True, height=900)