import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pydeck as pdk

# 1. Configuración de página en modo ancho total
st.set_page_config(
    page_title="NASA Hot Spots - Grid Térmico Global",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Estilos CSS para el diseño flotante y pantalla completa
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    
    .block-container {
        padding: 0rem !important;
        max-width: 100% !important;
        height: 100vh !important;
        overflow: hidden !important;
    }

    /* Tarjeta flotante superior izquierda */
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
        width: 280px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    }

    /* Tarjeta flotante superior derecha (Estadísticas y Tendencia) */
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

    /* Forzar mapa a ocupar toda la pantalla */
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
    n_rows = 4500  # Gran volumen de lecturas para cubrir todo el mapa mundial
    
    # Regiones y países globales principales con alta actividad térmica
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
        # Dispersión amplia para conectar los hexágonos en todo el continente/mapa
        lats.append(np.clip(base_lat + np.random.normal(0, 12.0), -60.0, 75.0))
        lons.append(np.clip(base_lon + np.random.normal(0, 18.0), -170.0, 175.0))
        
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

# ==========================================
# 4. CONTROLES EN LA BARRA LATERAL
# ==========================================
with st.sidebar:
    st.header("⚙️ Filtros Globales")
    selected_country = st.selectbox("🌍 Región / País:", ["Todo el Mundo"] + list(df_data["country"].unique()))
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

# ==========================================
# 5. VENTANAS FLOTANTES SOBRE EL MAPA
# ==========================================
st.markdown("""
    <div class="floating-card-left">
        <h3 style="margin: 0; font-size: 18px;">🔥 NASA Hot Spots</h3>
        <p style="font-size: 11px; color: #aaa; margin: 4px 0 0 0;">Grid Térmico Global (MODIS & VIIRS)</p>
    </div>
""", unsafe_allow_html=True)

with st.container(key="ventana_derecha"):
    st.markdown("<h4 style='margin:0 0 10px 0; font-size:15px;'>📊 Análisis Térmico Global</h4>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    c1.metric("Lecturas", len(filtered_df))
    avg_b = f"{filtered_df['brightness'].mean():.1f} K" if not filtered_df.empty else "0 K"
    c2.metric("Brillo Prom.", avg_b)
    
    if not filtered_df.empty:
        trend_data = filtered_df.groupby("date").size().reset_index(name="Quemas")
        st.line_chart(trend_data.set_index("date"), height=140)
        
        with st.expander("📋 Ver registros térmicos"):
            st.dataframe(filtered_df[["date", "satellite", "brightness", "country"]].tail(10), use_container_width=True, height=150)
    else:
        st.warning("Sin datos térmicos para mostrar.")

# ==========================================
# 6. MAPA DE FONDO CON GRID TÉRMICO EXTENDIDO
# ==========================================
if selected_country != "Todo el Mundo" and selected_country in regions_coords:
    center_lat, center_lon = regions_coords[selected_country]
    zoom_level = 4
    hex_radius = 45000
else:
    center_lat, center_lon = 15.0, 10.0  # Vista global centrada en el mapamundi
    zoom_level = 1.6
    hex_radius = 95000  # Radio amplio para cubrir continentes enteros como una malla continua

layer = pdk.Layer(
    "HexagonLayer",
    data=filtered_df,
    get_position="[lon, lat]",
    radius=hex_radius,
    elevation_scale=600,
    elevation_range=[0, 1500],
    pickable=True,
    extruded=True,
    coverage=0.95, # Hace que los hexágonos se toquen formando un grid casi continuo
    color_range=[
        [0, 100, 255, 170],   # Azul (Baja intensidad)
        [0, 200, 255, 170],   # Cyan
        [0, 255, 100, 170],   # Verde
        [255, 255, 0, 180],   # Amarillo
        [255, 140, 0, 190],   # Naranja
        [255, 0, 0, 210]      # Rojo intenso (Alta concentración térmica)
    ]
)

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=zoom_level,
    pitch=40,
    bearing=0
)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Celda Térmica Global\nDensidad de lecturas satelitales"},
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
)

st.pydeck_chart(r, use_container_width=True, height=900)