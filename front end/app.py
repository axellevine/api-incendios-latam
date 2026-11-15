import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pydeck as pdk

# 1. Configuración de página en modo ancho total
st.set_page_config(
    page_title="NASA Hot Spots - Mapa Interactivo",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. CSS para mapa a pantalla completa y ventanas flotantes reales
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

    /* Convertir el contenedor 'ventana_derecha' en una tarjeta flotante real */
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

    /* Ajustar el mapa de PyDeck para que ocupe el 100% de la pantalla de fondo */
    iframe[title="streamlit_pydeck.pydeck_chart"], .stDeckGlJsonChart {
        height: 100vh !important;
        width: 100vw !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Generación de datos simulados
@st.cache_data
def load_data():
    np.random.seed(42)
    n_rows = 700
    
    countries = ["Colombia", "Brasil", "Bolivia", "Perú", "Venezuela"]
    country_coords = {
        "Colombia": (4.5, -74.0),
        "Brasil": (-14.2, -51.9),
        "Bolivia": (-16.2, -63.5),
        "Perú": (-9.1, -75.0),
        "Venezuela": (6.4, -66.5)
    }
    
    assigned_countries = np.random.choice(countries, n_rows)
    lats, lons = [], []
    
    for c in assigned_countries:
        base_lat, base_lon = country_coords[c]
        lats.append(base_lat + np.random.uniform(-4.0, 4.0))
        lons.append(base_lon + np.random.uniform(-4.0, 4.0))
        
    satellites = np.random.choice(["MODIS", "VIIRS"], n_rows, p=[0.4, 0.6])
    brightness = np.random.uniform(310, 460, n_rows)
    start_date = datetime.date(2026, 1, 1)
    dates = [start_date + datetime.timedelta(days=int(np.random.randint(0, 90))) for _ in range(n_rows)]
    
    df = pd.DataFrame({
        "lat": lats,
        "lon": lons,
        "country": assigned_countries,
        "satellite": satellites,
        "brightness": brightness,
        "date": dates
    })
    return df, country_coords

df_data, country_coords = load_data()

# ==========================================
# 4. CONTROLES EN LA BARRA LATERAL
# ==========================================
with st.sidebar:
    st.header("⚙️ Filtros")
    selected_country = st.selectbox("🌍 País:", ["Todos"] + list(df_data["country"].unique()))
    selected_satellite = st.multiselect("🛰️ Satélite:", df_data["satellite"].unique(), default=df_data["satellite"].unique())
    min_date, max_date = df_data["date"].min(), df_data["date"].max()
    date_range = st.date_input("📅 Fechas:", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Filtrar datos
filtered_df = df_data.copy()
if selected_country != "Todos":
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
# Ventana Izquierda (Título)
st.markdown("""
    <div class="floating-card-left">
        <h3 style="margin: 0; font-size: 18px;">🔥 NASA Hot Spots</h3>
        <p style="font-size: 11px; color: #aaa; margin: 4px 0 0 0;">MODIS & VIIRS Harmonization</p>
    </div>
""", unsafe_allow_html=True)

# Ventana Derecha (Todo encerrado dentro del contenedor con key="ventana_derecha")
with st.container(key="ventana_derecha"):
    st.markdown("<h4 style='margin:0 0 10px 0; font-size:15px;'>📊 Tendencia General de Quemas</h4>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    c1.metric("Detectados", len(filtered_df))
    avg_b = f"{filtered_df['brightness'].mean():.1f} K" if not filtered_df.empty else "0 K"
    c2.metric("Brillo Prom.", avg_b)
    
    if not filtered_df.empty:
        trend_data = filtered_df.groupby("date").size().reset_index(name="Quemas")
        st.line_chart(trend_data.set_index("date"), height=140)
        
        with st.expander("📋 Ver tabla de datos"):
            st.dataframe(trend_data.tail(10), use_container_width=True, height=150)
    else:
        st.warning("Sin datos para mostrar.")

# ==========================================
# 6. MAPA DE FONDO (100% PANTALLA)
# ==========================================
if selected_country != "Todos" and selected_country in country_coords:
    center_lat, center_lon = country_coords[selected_country]
    zoom_level = 5
else:
    center_lat, center_lon = -10.0, -60.0
    zoom_level = 3

layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered_df,
    get_position="[lon, lat]",
    get_color="[255, 75, 75, 180]",
    get_radius="brightness * 15",
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=zoom_level,
    pitch=0,
    bearing=0
)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "País: {country}\nSatélite: {satellite}\nBrillo: {brightness} K\nFecha: {date}"},
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
)

st.pydeck_chart(r, use_container_width=True, height=900)