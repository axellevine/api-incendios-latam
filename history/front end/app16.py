import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS HUD (PANTALLA COMPLETA)
# ==============================================================================
st.set_page_config(
    page_title="NASA Hot Spots - Calendario Armonizado MODIS & VIIRS",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    
    .block-container {
        padding: 0rem !important;
        max-width: 100% !important;
        height: 100vh !important;
        overflow: hidden !important;
    }

    /* Panel superior izquierdo: Título y Filtros Globales (antes ocultos en sidebar) */
    .st-key-ventana_filtros {
        position: fixed !important;
        top: 16px !important;
        left: 16px !important;
        z-index: 9999 !important;
        background: rgba(16, 20, 29, 0.90) !important;
        backdrop-filter: blur(14px) !important;
        padding: 14px 16px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        color: white !important;
        width: 330px !important;
        max-height: 56vh !important;
        overflow-y: auto !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
    }

    /* Panel inferior izquierdo: Proyección 2D/3D, Capas y Escala del Grid */
    .st-key-ventana_capas {
        position: fixed !important;
        bottom: 16px !important;
        left: 16px !important;
        z-index: 9999 !important;
        background: rgba(16, 20, 29, 0.90) !important;
        backdrop-filter: blur(14px) !important;
        padding: 14px 16px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        color: white !important;
        width: 330px !important;
        max-height: 38vh !important;
        overflow-y: auto !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
    }

    /* Panel derecho: Calendario de Actividad de Quemas, Anomalías y CSV */
    .st-key-ventana_derecha {
        position: fixed !important;
        top: 16px !important;
        right: 16px !important;
        z-index: 9999 !important;
        background: rgba(16, 20, 29, 0.90) !important;
        backdrop-filter: blur(14px) !important;
        padding: 16px 18px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        color: white !important;
        width: 450px !important;
        max-height: 92vh !important;
        overflow-y: auto !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
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

MONTH_NAMES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}

# ==============================================================================
# 2. GENERACIÓN DE SERIE HISTÓRICA MULTIANUAL (2001–2026) Y ARMONIZACIÓN
# ==============================================================================
@st.cache_data
def load_historical_data():
    """
    Simula una serie histórica de 26 años (2001–2026) coherente con NASA FIRMS:
    - MODIS (Terra/Aqua): operativo desde 2001 (~1 km de resolución).
    - VIIRS (Suomi-NPP/NOAA-20): operativo desde 2012 (375 m, mayor sensibilidad).
    Incluye estacionalidad regional y una anomalía intensa en 2026.
    """
    rng = np.random.default_rng(42)
    regions = {
        "Colombia": (4.5, -74.0, [1, 2, 3, 8, 9]),
        "Brasil (Amazonía)": (-10.0, -55.0, [7, 8, 9, 10]),
        "Estados Unidos (California)": (37.0, -119.5, [7, 8, 9, 10]),
        "Canadá (Boreal)": (56.0, -106.0, [6, 7, 8]),
        "África Central (Congo)": (-2.0, 23.0, [6, 7, 8, 12, 1]),
        "Australia": (-25.0, 133.0, [10, 11, 12, 1, 2]),
        "Indonesia": (-2.5, 118.0, [8, 9, 10]),
        "Siberia (Rusia)": (61.0, 105.0, [5, 6, 7, 8]),
        "Europa Mediterránea": (40.0, 15.0, [7, 8, 9]),
        "India": (21.0, 78.0, [3, 4, 5]),
    }
    region_names = list(regions.keys())
    n_rows = 36000

    years = rng.integers(2001, 2027, size=n_rows)
    # Repunte anómalo en 2026 para demostrar detección de condiciones inusuales
    anomaly_mask = rng.random(n_rows) < 0.06
    years[anomaly_mask] = 2026

    reg_idx = rng.integers(0, len(region_names), size=n_rows)
    assigned_regions = np.array(region_names)[reg_idx]

    lats = np.empty(n_rows)
    lons = np.empty(n_rows)
    months = np.empty(n_rows, dtype=int)

    for i, r_name in enumerate(region_names):
        mask = (reg_idx == i)
        cnt = mask.sum()
        base_lat, base_lon, peak_months = regions[r_name]
        lats[mask] = np.clip(base_lat + rng.normal(0, 6.0, cnt), -60.0, 75.0)
        lons[mask] = np.clip(base_lon + rng.normal(0, 8.5, cnt), -170.0, 175.0)
        is_peak = rng.random(cnt) < 0.75
        months[mask] = np.where(
            is_peak,
            rng.choice(peak_months, size=cnt),
            rng.integers(1, 13, size=cnt)
        )

    # Disponibilidad realista: MODIS desde 2001; VIIRS desde 2012 con ~2.6x más detecciones
    satellites = np.where(
        years < 2012,
        "MODIS",
        np.where(rng.random(n_rows) < 0.72, "VIIRS", "MODIS")
    )

    days = rng.integers(1, 29, size=n_rows)
    dates = pd.to_datetime({"year": years, "month": months, "day": days})

    brightness = np.where(
        satellites == "VIIRS",
        rng.uniform(315, 458, size=n_rows),
        rng.uniform(310, 435, size=n_rows)
    )
    frp = np.round(np.exp(rng.normal(2.7, 0.85, size=n_rows)), 1)

    # Factor de armonización cruzada (normaliza resolución 375m de VIIRS a equivalente 1km MODIS)
    harmonized_weight = np.where(satellites == "VIIRS", 0.38, 1.0)

    df = pd.DataFrame({
        "lat": np.round(lats, 4),
        "lon": np.round(lons, 4),
        "country": assigned_regions,
        "satellite": satellites,
        "brightness": np.round(brightness, 1),
        "frp": frp,
        "harmonized_weight": harmonized_weight,
        "date": dates.dt.date,
        "year": years,
        "month": months,
    })
    return df, regions


# ==============================================================================
# 3. AGREGACIÓN ESPACIAL REAL EN CELDAS HEXAGONALES Y GRILLA GEODÉSICA
# ==============================================================================
@st.cache_data
def build_grid_by_radius(radius_deg, is_globe=False):
    dx = radius_deg * 1.5
    dy = radius_deg * np.sqrt(3.0)

    col_start = int(np.floor(-180.0 / dx))
    col_end = int(np.ceil(180.0 / dx))
    row_start = int(np.floor(-70.0 / dy))
    row_end = int(np.ceil(80.0 / dy))

    cols = np.arange(col_start, col_end + 1)
    rows = np.arange(row_start, row_end + 1)
    cc, rr = np.meshgrid(cols, rows, indexing="ij")
    cc = cc.ravel()
    rr = rr.ravel()

    lat_offsets = np.where(cc % 2 != 0, dy / 2.0, 0.0)
    c_lats = rr * dy + lat_offsets
    valid = (c_lats >= -70.0) & (c_lats <= 80.0)
    cc = cc[valid]
    c_lats = c_lats[valid]
    c_lons = cc * dx

    angles = np.deg2rad(np.arange(0, 360, 60))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    if is_globe:
        lon_scales = np.maximum(0.2, np.cos(np.deg2rad(c_lats)))[:, None]
        v_lons = c_lons[:, None] + (cos_a[None, :] * radius_deg / lon_scales)
        v_lats = c_lats[:, None] + (sin_a[None, :] * radius_deg)
    else:
        v_lons = c_lons[:, None] + (cos_a[None, :] * radius_deg)
        v_lats = c_lats[:, None] + (sin_a[None, :] * radius_deg)

    polygons = np.stack([v_lons, v_lats], axis=-1).tolist()
    return pd.DataFrame({"polygon": polygons})


def intensity_to_ylorrd(values):
    """Rampa perceptual accesible YlOrRd (daltonismo-segura) según percentiles de intensidad."""
    if len(values) == 0:
        return []
    q25, q50, q75, q90 = np.percentile(values, [25, 50, 75, 90])
    colors = []
    for v in values:
        if v >= q90:
            colors.append([128, 0, 38, 225])     # Rojo oscuro intenso
        elif v >= q75:
            colors.append([227, 26, 28, 210])    # Carmesí
        elif v >= q50:
            colors.append([253, 141, 60, 195])   # Naranja brillante
        elif v >= q25:
            colors.append([254, 204, 92, 180])   # Ámbar
        else:
            colors.append([255, 255, 178, 165])  # Amarillo pálido
    return colors


@st.cache_data
def aggregate_readings_into_hex_cells(df, radius_deg, is_globe=False, harmonize=True):
    """
    Agrupa los hotspots individuales dentro de celdas hexagonales exactas de la grilla.
    Elimina polígonos superpuestos y calcula métricas agregadas por celda.
    """
    if df.empty:
        return pd.DataFrame()

    dx = radius_deg * 1.5
    dy = radius_deg * np.sqrt(3.0)

    cols = np.round(df["lon"].to_numpy() / dx).astype(int)
    lat_offsets = np.where(cols % 2 != 0, dy / 2.0, 0.0)
    rows = np.round((df["lat"].to_numpy() - lat_offsets) / dy).astype(int)

    weights = df["harmonized_weight"].to_numpy() if harmonize else np.ones(len(df))

    work = pd.DataFrame({
        "col": cols,
        "row": rows,
        "weight": weights,
        "brightness": df["brightness"].to_numpy(),
        "frp": df["frp"].to_numpy(),
        "is_modis": (df["satellite"].to_numpy() == "MODIS").astype(int),
        "is_viirs": (df["satellite"].to_numpy() == "VIIRS").astype(int),
    })

    agg = work.groupby(["col", "row"], as_index=False).agg(
        raw_count=("weight", "size"),
        harmonized_events=("weight", "sum"),
        mean_brightness=("brightness", "mean"),
        total_frp=("frp", "sum"),
        modis_count=("is_modis", "sum"),
        viirs_count=("is_viirs", "sum"),
    )

    c_lons = agg["col"].to_numpy() * dx
    c_lats = agg["row"].to_numpy() * dy + np.where(agg["col"].to_numpy() % 2 != 0, dy / 2.0, 0.0)

    angles = np.deg2rad(np.arange(0, 360, 60))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    if is_globe:
        lon_scales = np.maximum(0.2, np.cos(np.deg2rad(c_lats)))[:, None]
        v_lons = c_lons[:, None] + (cos_a[None, :] * radius_deg / lon_scales)
        v_lats = c_lats[:, None] + (sin_a[None, :] * radius_deg)
    else:
        v_lons = c_lons[:, None] + (cos_a[None, :] * radius_deg)
        v_lats = c_lats[:, None] + (sin_a[None, :] * radius_deg)

    agg["polygon"] = np.stack([v_lons, v_lats], axis=-1).tolist()
    agg["harmonized_events"] = np.round(agg["harmonized_events"], 1)
    agg["mean_brightness"] = np.round(agg["mean_brightness"], 1)
    agg["total_frp"] = np.round(agg["total_frp"], 1)
    agg["color"] = intensity_to_ylorrd(agg["harmonized_events"].to_numpy())
    return agg


@st.cache_data
def load_multilevel_grids(is_globe=False):
    return {
        "L1": build_grid_by_radius(9.0, is_globe),
        "L2": build_grid_by_radius(4.5, is_globe),
        "L3": build_grid_by_radius(2.25, is_globe),
        "L4": build_grid_by_radius(1.125, is_globe),
    }


df_data, regions_coords = load_historical_data()

# ==============================================================================
# 4. VENTANA FLOTANTE SUPERIOR IZQUIERDA: FILTROS Y ARMONIZACIÓN (ACCESIBLES)
# ==============================================================================
with st.container(key="ventana_filtros"):
    st.markdown("<h3 style='margin:0; font-size:17px;'>🔥 NASA Hot Spots</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px; color:#94a3b8; margin:2px 0 8px 0;'>Armonización MODIS (2001+) & VIIRS (2012+)</p>", unsafe_allow_html=True)

    aoi_mode = st.selectbox(
        "🌍 Área de Interés (AOI):",
        ["Todo el Mundo", "Coordenadas Personalizadas (Lat/Lon)"] + list(regions_coords.keys()),
        index=3  # Inicia en Brasil (Amazonía)
    )

    custom_bbox = None
    if aoi_mode == "Coordenadas Personalizadas (Lat/Lon)":
        c_lat1, c_lat2 = st.columns(2)
        lat_min = c_lat1.number_input("Lat Mín:", min_value=-85.0, max_value=85.0, value=-15.0, step=1.0)
        lat_max = c_lat2.number_input("Lat Máx:", min_value=-85.0, max_value=85.0, value=12.0, step=1.0)
        c_lon1, c_lon2 = st.columns(2)
        lon_min = c_lon1.number_input("Lon Mín:", min_value=-180.0, max_value=180.0, value=-80.0, step=1.0)
        lon_max = c_lon2.number_input("Lon Máx:", min_value=-180.0, max_value=180.0, value=-48.0, step=1.0)
        custom_bbox = (min(lat_min, lat_max), max(lat_min, lat_max), min(lon_min, lon_max), max(lon_min, lon_max))

    year_range = st.slider(
        "📅 Rango Histórico (Años):",
        min_value=2001,
        max_value=2026,
        value=(2001, 2026)
    )

    harmonize_mode = st.toggle(
        "⚖️ Armonizar MODIS + VIIRS (Eliminar sesgo de sensor)",
        value=True,
        help="Normaliza la resolución de 375m de VIIRS a unidades equivalentes MODIS de 1km para mantener continuidad histórica desde 2001."
    )

    selected_satellites = st.multiselect(
        "🛰️ Sensores activos:",
        options=["MODIS", "VIIRS"],
        default=["MODIS", "VIIRS"]
    )

# Filtrado espacial y de sensores (conserva serie completa del AOI para calcular mediana histórica)
aoi_df = df_data[df_data["satellite"].isin(selected_satellites)].copy()
if aoi_mode == "Coordenadas Personalizadas (Lat/Lon)" and custom_bbox is not None:
    lat_lo, lat_hi, lon_lo, lon_hi = custom_bbox
    aoi_df = aoi_df[
        (aoi_df["lat"] >= lat_lo) & (aoi_df["lat"] <= lat_hi) &
        (aoi_df["lon"] >= lon_lo) & (aoi_df["lon"] <= lon_hi)
    ]
elif aoi_mode not in ["Todo el Mundo", "Coordenadas Personalizadas (Lat/Lon)"]:
    aoi_df = aoi_df[aoi_df["country"] == aoi_mode]

filtered_df = aoi_df[
    (aoi_df["year"] >= year_range[0]) &
    (aoi_df["year"] <= year_range[1])
].copy()

# ==============================================================================
# 5. VENTANA FLOTANTE INFERIOR IZQUIERDA: CONTROLES DE PROYECCIÓN Y GRILLA
# ==============================================================================
with st.container(key="ventana_capas"):
    st.markdown("<h4 style='margin:0 0 6px 0; font-size:14px;'>🗺️ Proyección y Capas Espaciales</h4>", unsafe_allow_html=True)

    c_proj, c_zoom = st.columns(2)
    projection_mode = c_proj.radio("Vista:", ["Plano (2D)", "Esfera 3D (Globo)"], index=0)
    zoom_mode = c_zoom.radio("Escala Hex:", ["Zoom Dinámico", "Zoom Manual"], index=0)

    c_chk1, c_chk2 = st.columns(2)
    show_base_grid = c_chk1.checkbox("Grilla Base", value=True)
    show_hex_readings = c_chk2.checkbox("Celdas Agregadas", value=True)

    zoom_levels_list = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    if zoom_mode == "Zoom Manual":
        selected_zoom = st.select_slider("Resolución Manual:", options=zoom_levels_list, value=200)
    else:
        selected_zoom = 150

# ==============================================================================
# 6. VENTANA FLOTANTE DERECHA: CALENDARIO DE QUEMAS, ANOMALÍAS Y EXPORTACIÓN CSV
# ==============================================================================
with st.container(key="ventana_derecha"):
    st.markdown("<h4 style='margin:0 0 8px 0; font-size:15px;'>📊 Calendario de Actividad de Quemas</h4>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No se encontraron focos térmicos para el área o filtros seleccionados.")
    else:
        filtered_df["metric_val"] = filtered_df["harmonized_weight"] if harmonize_mode else 1.0
        aoi_df["metric_val"] = aoi_df["harmonized_weight"] if harmonize_mode else 1.0

        # Matriz Año x Mes
        cal_df = (
            filtered_df.groupby(["year", "month"], as_index=False)
            .agg(
                eventos=("metric_val", "sum"),
                detecciones_crudas=("lat", "size"),
                frp_total=("frp", "sum"),
            )
        )
        cal_df["eventos"] = np.round(cal_df["eventos"], 1)
        cal_df["mes_nombre"] = cal_df["month"].map(MONTH_NAMES)

        # Mediana histórica mensual y percentiles (referencia climatológica del AOI)
        hist_monthly = (
            aoi_df.groupby(["year", "month"], as_index=False)["metric_val"].sum()
            .groupby("month", as_index=False)
            .agg(
                mediana_hist=("metric_val", "median"),
                p25_hist=("metric_val", lambda s: np.percentile(s, 25)),
                p75_hist=("metric_val", lambda s: np.percentile(s, 75)),
            )
        )

        available_years = sorted(cal_df["year"].unique(), reverse=True)
        focus_year = st.selectbox("🔎 Año de referencia para anomalía:", available_years, index=0)

        focus_df = cal_df[cal_df["year"] == focus_year].merge(hist_monthly, on="month", how="right").fillna(0)
        focus_df["mes_nombre"] = focus_df["month"].map(MONTH_NAMES)
        focus_df["anomalia_pct"] = np.where(
            focus_df["mediana_hist"] > 0,
            ((focus_df["eventos"] - focus_df["mediana_hist"]) / focus_df["mediana_hist"]) * 100.0,
            0.0
        )

        # KPIs principales
        total_events = cal_df["eventos"].sum()
        year_total = focus_df["eventos"].sum()
        median_annual_total = hist_monthly["mediana_hist"].sum()
        annual_anomaly = ((year_total - median_annual_total) / median_annual_total * 100.0) if median_annual_total > 0 else 0.0
        peak_row = focus_df.loc[focus_df["eventos"].idxmax()]

        k1, k2, k3 = st.columns(3)
        k1.metric("Índice Armonizado", f"{total_events:,.0f}")
        k2.metric(f"Anomalía {focus_year}", f"{annual_anomaly:+.1f}%", delta=f"{annual_anomaly:+.1f}% vs mediana", delta_color="inverse")
        k3.metric(f"Mes Pico ({focus_year})", f"{peak_row['mes_nombre']}", f"{peak_row['eventos']:.0f} idx")

        tab_cal, tab_anom, tab_csv = st.tabs(["🗓️ Matriz Año×Mes", "📈 vs Mediana Histórica", "📥 Exportar CSV"])

        with tab_cal:
            st.caption("Intensidad mensual armonizada (2001–2026). Permite identificar temporadas críticas y años extremos.")
            heatmap = (
                alt.Chart(alt.Data(values=cal_df.to_dict(orient="records")))
                .mark_rect(cornerRadius=2, stroke="#10141d", strokeWidth=1)
                .encode(
                    x=alt.X("month:O", title="Mes", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("year:O", title="Año", sort="descending"),
                    color=alt.Color(
                        "eventos:Q",
                        scale=alt.Scale(scheme="inferno"),
                        title="Actividad"
                    ),
                    tooltip=[
                        alt.Tooltip("year:O", title="Año"),
                        alt.Tooltip("mes_nombre:N", title="Mes"),
                        alt.Tooltip("eventos:Q", title="Índice Armonizado"),
                        alt.Tooltip("detecciones_crudas:Q", title="Detecciones Crudas"),
                    ]
                )
                .properties(height=310)
            )
            st.altair_chart(heatmap, use_container_width=True)

        with tab_anom:
            st.caption(f"Actividad mensual en {focus_year} (línea naranja) frente a mediana histórica y banda intercuartílica P25–P75.")
            base_chart = alt.Chart(alt.Data(values=focus_df.to_dict(orient="records")))

            band_iqr = base_chart.mark_area(opacity=0.28, color="#38bdf8").encode(
                x=alt.X("month:O", title="Mes", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("p25_hist:Q", title="Índice de Quemas"),
                y2="p75_hist:Q",
                tooltip=[
                    alt.Tooltip("mes_nombre:N", title="Mes"),
                    alt.Tooltip("p25_hist:Q", title="Percentil 25", format=".1f"),
                    alt.Tooltip("p75_hist:Q", title="Percentil 75", format=".1f"),
                ]
            )

            median_line = base_chart.mark_line(strokeDash=[5, 4], color="#94a3b8", strokeWidth=2).encode(
                x=alt.X("month:O"),
                y=alt.Y("mediana_hist:Q"),
                tooltip=[
                    alt.Tooltip("mes_nombre:N", title="Mes"),
                    alt.Tooltip("mediana_hist:Q", title="Mediana Histórica", format=".1f"),
                ]
            )

            current_line = base_chart.mark_line(point=True, color="#f97316", strokeWidth=3).encode(
                x=alt.X("month:O"),
                y=alt.Y("eventos:Q"),
                tooltip=[
                    alt.Tooltip("mes_nombre:N", title="Mes"),
                    alt.Tooltip("eventos:Q", title=f"Actividad {focus_year}", format=".1f"),
                    alt.Tooltip("mediana_hist:Q", title="Mediana Histórica", format=".1f"),
                    alt.Tooltip("anomalia_pct:Q", title="Anomalía (%)", format="+.1f"),
                ]
            )

            st.altair_chart((band_iqr + median_line + current_line).properties(height=260), use_container_width=True)

        with tab_csv:
            st.caption("Descarga los registros armonizados filtrados o la matriz del calendario de quemas en formato CSV.")
            export_cols = ["date", "year", "month", "lat", "lon", "country", "satellite", "brightness", "frp", "harmonized_weight"]
            csv_hotspots = filtered_df[export_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar Hotspots Armonizados (.CSV)",
                data=csv_hotspots,
                file_name=f"nasa_hotspots_armonizados_{year_range[0]}_{year_range[1]}.csv",
                mime="text/csv",
                use_container_width=True
            )

            csv_calendar = cal_df[["year", "month", "mes_nombre", "eventos", "detecciones_crudas", "frp_total"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar Calendario Año×Mes (.CSV)",
                data=csv_calendar,
                file_name=f"calendario_quemas_{year_range[0]}_{year_range[1]}.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.dataframe(
                filtered_df[["date", "satellite", "brightness", "frp", "harmonized_weight", "country"]].head(12),
                use_container_width=True,
                height=160
            )

# ==============================================================================
# 7. RENDERIZADO GEODÉSICO PYDECK (2D PLANO / 3D ESFERA CON CELDAS AGREGADAS)
# ==============================================================================
is_globe = (projection_mode == "Esfera 3D (Globo)")
grids_dict = load_multilevel_grids(is_globe=is_globe)

if aoi_mode == "Coordenadas Personalizadas (Lat/Lon)" and custom_bbox is not None:
    lat_lo, lat_hi, lon_lo, lon_hi = custom_bbox
    center_lat = (lat_lo + lat_hi) / 2.0
    center_lon = (lon_lo + lon_hi) / 2.0
    base_zoom = 3.2 if not is_globe else 1.8
elif aoi_mode in regions_coords:
    center_lat, center_lon = regions_coords[aoi_mode][0], regions_coords[aoi_mode][1]
    base_zoom = 3.3 if not is_globe else 1.8
else:
    center_lat, center_lon = 12.0, 0.0
    base_zoom = 1.4 if not is_globe else 1.2

active_layers = []

# Polígono visual del Área de Interés (AOI) cuando se ingresan coordenadas Lat/Lon
if aoi_mode == "Coordenadas Personalizadas (Lat/Lon)" and custom_bbox is not None:
    lat_lo, lat_hi, lon_lo, lon_hi = custom_bbox
    bbox_poly = pd.DataFrame({
        "polygon": [[
            [lon_lo, lat_lo],
            [lon_hi, lat_lo],
            [lon_hi, lat_hi],
            [lon_lo, lat_hi],
            [lon_lo, lat_lo],
        ]]
    })
    active_layers.append(
        pdk.Layer(
            "PolygonLayer",
            data=bbox_poly,
            get_polygon="polygon",
            filled=True,
            stroked=True,
            get_fill_color=[56, 189, 248, 25],
            get_line_color=[56, 189, 248, 220],
            line_width_min_pixels=2,
            pickable=False,
        )
    )

if zoom_mode == "Zoom Dinámico":
    map_zoom = base_zoom
    # Radios idénticos entre grilla y celdas agregadas para encajar sin solapamiento
    lod_specs = [
        ("L1", 9.0, 0.0, 2.5, [0, 210, 255, 40]),
        ("L2", 4.5, 2.5, 4.0, [0, 210, 255, 50]),
        ("L3", 2.25, 4.0, 5.5, [0, 210, 255, 60]),
        ("L4", 1.125, 5.5, 12.0, [0, 210, 255, 70]),
    ]
    for lvl_key, r_deg, z_min, z_max, line_col in lod_specs:
        if show_base_grid:
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
            df_cells = aggregate_readings_into_hex_cells(
                filtered_df, radius_deg=r_deg, is_globe=is_globe, harmonize=harmonize_mode
            )
            active_layers.append(
                pdk.Layer(
                    "PolygonLayer",
                    data=df_cells,
                    get_polygon="polygon",
                    get_fill_color="color",
                    get_line_color=[255, 255, 255, 140],
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
    map_zoom = base_zoom + (selected_zoom / 500.0) * 3.8
    manual_radius = round(9.0 - (selected_zoom / 500.0) * 7.875, 3)

    if show_base_grid:
        df_manual_grid = build_grid_by_radius(manual_radius, is_globe=is_globe)
        active_layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=df_manual_grid,
                get_polygon="polygon",
                filled=False,
                stroked=True,
                get_line_color=[0, 210, 255, 50],
                line_width_min_pixels=1,
                pickable=False
            )
        )

    if show_hex_readings and not filtered_df.empty:
        df_manual_cells = aggregate_readings_into_hex_cells(
            filtered_df, radius_deg=manual_radius, is_globe=is_globe, harmonize=harmonize_mode
        )
        active_layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=df_manual_cells,
                get_polygon="polygon",
                get_fill_color="color",
                get_line_color=[255, 255, 255, 140],
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

deck = pdk.Deck(
    layers=active_layers,
    initial_view_state=view_state,
    views=[map_view],
    tooltip={
        "text": (
            "Índice Armonizado: {harmonized_events}\n"
            "Detecciones Crudas: {raw_count} (MODIS: {modis_count} | VIIRS: {viirs_count})\n"
            "Brillo Promedio: {mean_brightness} K\n"
            "Potencia Radiativa (FRP): {total_frp} MW"
        )
    },
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
)

st.pydeck_chart(deck, use_container_width=True, height=900)