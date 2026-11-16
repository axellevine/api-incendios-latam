import gc
import glob
import os
import re
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import numpy as np
import pandas as pd
import uvicorn

app = FastAPI(title="API Mapa de Incendios LATAM/América", version="10.0")

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATOS = os.path.join(BASE_DIR, "Country_per_year")
CARPETA_CACHES = os.path.join(BASE_DIR, "Cachecountryperyear")

# Asegurarnos de que la carpeta de cachés exista
os.makedirs(CARPETA_CACHES, exist_ok=True)

PAISES_EXCLUIDOS = {
    "french guiana", "guyana francesa", "guyane", "french_guiana", "guyana_francesa",
}

NORMALIZACION_PAISES = {
    "brasil": "Brazil", "brazil": "Brazil", "peru": "Peru", "perú": "Peru",
    "mexico": "Mexico", "méxico": "Mexico", "panama": "Panama", "panamá": "Panama",
    "canada": "Canada", "canadá": "Canada", "usa": "United States", 
    "united states": "United States", "estados unidos": "United States",
    "haiti": "Haiti", "haití": "Haiti", "republica dominicana": "Dominican Republic",
    "república dominicana": "Dominican Republic", "bolivia": "Bolivia", 
    "colombia": "Colombia", "argentina": "Argentina", "chile": "Chile", 
    "ecuador": "Ecuador", "venezuela": "Venezuela", "paraguay": "Paraguay", 
    "uruguay": "Uruguay", "guatemala": "Guatemala", "honduras": "Honduras", 
    "el salvador": "El Salvador", "nicaragua": "Nicaragua", "costa rica": "Costa Rica", 
    "cuba": "Cuba"
}

TOKENS_RUIDO = {
    "modis", "viirs", "fire", "dl", "suomi", "noaa", "snpp", "npp", "terra", "aqua",
    "j1", "j2", "jpss1", "jpss2", "c61", "c6", "nrt", "archive", "final", "clean",
    "limpio", "v1", "v2", "ok", "copy", "copia", "data", "datos", "incendios",
    "fires", "filtrado", "filtrados", "filtered", "procesado", "processed", "csv",
    "latam", "america", "latina", "latinoamerica", "(1)", "(2)", "(3)", "(4)", 
    "(5)", "(6)", "(7)", "(8)", "(9)", " "
}

def detectar_separador(ruta_archivo: str) -> str:
    with open(ruta_archivo, "r", encoding="utf-8-sig", errors="ignore") as f:
        primera_linea = f.readline()
    return ";" if primera_linea.count(";") > primera_linea.count(",") else ","

def limpiar_numerico(serie: pd.Series) -> pd.Series:
    if serie.dtype == object:
        serie = serie.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(serie, errors="coerce").replace([np.inf, -np.inf], np.nan)

def normalizar_satelite(serie: pd.Series, sensor_defecto: str) -> pd.Series:
    s = serie.fillna("").astype(str).str.strip()
    return s.replace({
        "T": "Terra", "A": "Aqua", "N": "Suomi-NPP", "1": "NOAA-20", "2": "NOAA-21",
        "": "JPSS-1" if sensor_defecto == "VIIRS" else "Terra"
    })

def extraer_pais_y_anio(ruta_archivo: str) -> tuple[str, int]:
    rel_path = os.path.relpath(ruta_archivo, CARPETA_DATOS)
    partes_ruta = rel_path.split(os.sep)
    nombre_base = re.sub(r"\s*\(\d+\)$", "", os.path.splitext(partes_ruta[-1])[0]).strip()
    
    anio_fallback = 2025
    tokens_todos = []
    for segmento in partes_ruta[:-1] + [nombre_base]:
        tokens_todos.extend(re.split(r"[_\-\s]+", segmento))
        
    for tok in tokens_todos:
        if tok.isdigit() and 2018 <= int(tok) <= 2025:
            anio_fallback = int(tok)
            break

    tokens_archivo = [t for t in re.split(r"[_\-\s]+", nombre_base) if t and not t.isdigit() and t.lower() not in TOKENS_RUIDO]
    if tokens_archivo:
        pais_raw = " ".join(tokens_archivo).strip()
    else:
        pais_carpeta = "Desconocido"
        for carpeta in reversed(partes_ruta[:-1]):
            t_carp = [t for t in re.split(r"[_\-\s]+", carpeta) if t and not t.isdigit() and t.lower() not in TOKENS_RUIDO]
            if t_carp:
                pais_carpeta = " ".join(t_carp).strip()
                break
        pais_raw = pais_carpeta

    return NORMALIZACION_PAISES.get(pais_raw.lower(), pais_raw.title()), anio_fallback

def obtener_archivos_validos() -> list[str]:
    if not os.path.isdir(CARPETA_DATOS): return []
    return sorted([f for f in glob.glob(os.path.join(CARPETA_DATOS, "**", "*.csv"), recursive=True) if "venv" not in f.lower()])

def obtener_manifiesto_csv(archivos: list[str]) -> list[tuple[str, int, float]]:
    return [(os.path.relpath(f, BASE_DIR), os.stat(f).st_size, round(os.stat(f).st_mtime, 2)) for f in archivos]

def procesar_todos_los_csv(archivos_csv: list[str], manifiesto: list) -> dict:
    resumenes_diarios_archivos = []

    for archivo in archivos_csv:
        rel_display = os.path.relpath(archivo, CARPETA_DATOS)
        nombre_archivo = os.path.basename(archivo)
        pais, anio_archivo = extraer_pais_y_anio(archivo)

        if pais.lower() in PAISES_EXCLUIDOS: continue

        sep = detectar_separador(archivo)
        sensor_archivo = "VIIRS" if "viirs" in archivo.lower() or nombre_archivo.lower().startswith(("suomi", "noaa", "j1", "j2", "snpp")) else "MODIS"

        try:
            header_df = pd.read_csv(archivo, sep=sep, nrows=0)
        except: continue

        header_df.columns = header_df.columns.str.lower().str.strip()
        lat_col = next((c for c in header_df.columns if c in ["latitude", "lat", "y"] or c.startswith("lat")), None)
        lon_col = next((c for c in header_df.columns if c in ["longitude", "lon", "long", "x"] or c.startswith("lon")), None)
        if not lat_col or not lon_col or lat_col == lon_col: continue

        date_col = next((c for c in header_df.columns if "date" in c or "acq_date" in c or "time" in c), None)
        frp_col = next((c for c in header_df.columns if "frp" in c), None)
        bright_col = next((c for c in header_df.columns if c in ["brightness", "bright_ti4"] or "bright" in c), None)
        inst_col = next((c for c in header_df.columns if "instrument" in c or "inst" in c), None)
        sat_col = next((c for c in header_df.columns if "satellite" in c or "sat" in c), None)

        cols_to_use = [lat_col, lon_col]
        for col in [date_col, frp_col, bright_col, inst_col, sat_col]:
            if col and col not in cols_to_use: cols_to_use.append(col)

        rename_map = {lat_col: "lat", lon_col: "lon"}
        if date_col: rename_map[date_col] = "fecha"
        if frp_col: rename_map[frp_col] = "frp"
        if bright_col: rename_map[bright_col] = "bright"
        if inst_col: rename_map[inst_col] = "instrumento"
        if sat_col: rename_map[sat_col] = "satelite"

        print(f"Procesando: {rel_display} -> País: {pais} | Año: {anio_archivo}")
        chunks_archivo = []

        reader = pd.read_csv(archivo, sep=sep, usecols=lambda c: c.lower().strip() in cols_to_use, chunksize=250_000, low_memory=False, on_bad_lines="skip")
        
        for chunk in reader:
            chunk.columns = chunk.columns.str.lower().str.strip()
            chunk = chunk.rename(columns=rename_map).loc[:, ~chunk.columns.duplicated()]

            chunk["lat"] = limpiar_numerico(chunk["lat"]).astype("float32")
            chunk["lon"] = limpiar_numerico(chunk["lon"]).astype("float32")
            chunk = chunk.dropna(subset=["lat", "lon"])
            chunk = chunk[(chunk["lat"].between(-90.0, 90.0)) & (chunk["lon"].between(-180.0, 180.0))]
            if chunk.empty: continue

            chunk["frp"] = limpiar_numerico(chunk["frp"]).fillna(0.0).astype("float32") if "frp" in chunk.columns else 0.0
            chunk["bright"] = limpiar_numerico(chunk["bright"]).fillna(0.0).astype("float32") if "bright" in chunk.columns else 0.0

            if "instrumento" in chunk.columns:
                inst = chunk["instrumento"].fillna(sensor_archivo).astype(str).str.strip()
                chunk["instrumento"] = np.where(inst.str.upper().str.contains("VIIRS"), "VIIRS", "MODIS")
            else:
                chunk["instrumento"] = sensor_archivo

            chunk["satelite"] = normalizar_satelite(chunk["satelite"], sensor_archivo) if "satelite" in chunk.columns else ("JPSS-1" if sensor_archivo == "VIIRS" else "Terra")

            if "fecha" in chunk.columns:
                fechas_dt = pd.to_datetime(chunk["fecha"], format="mixed", errors="coerce")
                chunk["anio"] = fechas_dt.dt.year.fillna(anio_archivo).astype("int16")
                chunk["mes"] = fechas_dt.dt.month.fillna(1).astype("int8").clip(1, 12)
                chunk["fecha"] = fechas_dt.dt.strftime("%Y-%m-%d").fillna(f"{anio_archivo}-01-01")
            else:
                chunk["anio"] = anio_archivo
                chunk["mes"] = 1
                chunk["fecha"] = f"{anio_archivo}-01-01"

            chunk = chunk[(chunk["anio"] >= 2018) & (chunk["anio"] <= 2025)]
            if chunk.empty: continue

            chunk["lat_grid"], chunk["lon_grid"] = chunk["lat"].round(2), chunk["lon"].round(2)

            chunks_archivo.append(
                chunk.groupby(["anio", "mes", "fecha", "lat_grid", "lon_grid", "instrumento", "satelite"], observed=True)
                .agg(total_incendios=("lat", "count"), suma_frp=("frp", "sum"), suma_brillo=("bright", "sum")).reset_index()
            )

        if chunks_archivo:
            df_file = pd.concat(chunks_archivo, ignore_index=True).groupby(["anio", "mes", "fecha", "lat_grid", "lon_grid", "instrumento", "satelite"], observed=True).agg(total_incendios=("total_incendios", "sum"), suma_frp=("suma_frp", "sum"), suma_brillo=("suma_brillo", "sum")).reset_index()
            df_file["pais"] = pais
            resumenes_diarios_archivos.append(df_file)

        del chunks_archivo
        gc.collect()

    if not resumenes_diarios_archivos:
        raise ValueError(f"❌ No se encontraron registros válidos dentro de '{CARPETA_DATOS}'.")

    print("Consolidando bases y optimizando caché anual...")
    df_all = pd.concat(resumenes_diarios_archivos, ignore_index=True)
    del resumenes_diarios_archivos
    gc.collect()

    diario = df_all.groupby(["pais", "anio", "mes", "fecha", "lat_grid", "lon_grid", "instrumento", "satelite"], observed=True).agg(total_incendios=("total_incendios", "sum"), suma_frp=("suma_frp", "sum"), suma_brillo=("suma_brillo", "sum")).reset_index()
    del df_all
    gc.collect()

    mensual = diario.groupby(["pais", "anio", "mes", "lat_grid", "lon_grid", "instrumento", "satelite"], observed=True).agg(total_incendios=("total_incendios", "sum"), suma_frp=("suma_frp", "sum"), suma_brillo=("suma_brillo", "sum"), fecha=("fecha", "max")).reset_index()

    for tabla in (diario, mensual):
        tot_safe = tabla["total_incendios"].replace(0, 1)
        tabla["frp_promedio"] = (tabla["suma_frp"] / tot_safe).round(1).fillna(0.0).astype("float32")
        tabla["brillo_promedio"] = (tabla["suma_brillo"] / tot_safe).round(1).fillna(0.0).astype("float32")
        tabla.drop(columns=["suma_frp", "suma_brillo"], inplace=True)

        tabla["lat_grid"], tabla["lon_grid"] = tabla["lat_grid"].round(2).astype("float32"), tabla["lon_grid"].round(2).astype("float32")
        tabla["total_incendios"] = tabla["total_incendios"].astype("int32")
        tabla["anio"], tabla["mes"] = tabla["anio"].astype("int16"), tabla["mes"].astype("int8")
        
        tabla["fecha"] = tabla["fecha"].astype(str).str.replace("-", "", regex=False).astype("int32")

        for col_cat in ["pais", "instrumento", "satelite"]: tabla[col_cat] = tabla[col_cat].astype("category")

    diario = diario.sort_values("total_incendios", ascending=False).reset_index(drop=True)
    mensual = mensual.sort_values("total_incendios", ascending=False).reset_index(drop=True)

    anios_presentes = diario["anio"].unique()
    for anio_actual in anios_presentes:
        d_anio = diario[diario["anio"] == anio_actual]
        m_anio = mensual[mensual["anio"] == anio_actual]
        ruta = os.path.join(CARPETA_CACHES, f"cache_{anio_actual}.pkl.gz")
        pd.to_pickle({"manifiesto": manifiesto, "diario": d_anio, "mensual": m_anio}, ruta, compression="gzip")
        print(f"💾 Caché guardado: {ruta}")

    return {"diario": diario, "mensual": mensual}


# --- INICIALIZACIÓN INTELIGENTE ---
try:
    archivos_encontrados = obtener_archivos_validos()
    manifiesto_actual = obtener_manifiesto_csv(archivos_encontrados)
except:
    archivos_encontrados, manifiesto_actual = [], []

hay_csvs_locales = len(archivos_encontrados) > 0
archivos_cache = sorted(glob.glob(os.path.join(CARPETA_CACHES, "cache_20*.pkl.gz")))

requiere_actualizacion = True
cache_cargado = None

if archivos_cache:
    try:
        print(f"☁️ Uniendo {len(archivos_cache)} archivos de caché anuales desde 'Cachecountryperyear'...")
        diarios_list, mensuales_list = [], []
        manif_guardado = None
        
        for arch in archivos_cache:
            obj = pd.read_pickle(arch)
            diarios_list.append(obj["diario"])
            mensuales_list.append(obj["mensual"])
            manif_guardado = obj.get("manifiesto")
            
        cache_cargado = {
            "diario": pd.concat(diarios_list, ignore_index=True),
            "mensual": pd.concat(mensuales_list, ignore_index=True)
        }
        
        if not hay_csvs_locales:
            print("☁️ Modo Nube: Cachés unidos exitosamente.")
            requiere_actualizacion = False
        elif manif_guardado == manifiesto_actual:
            print("⚡ Cachés anuales sincronizados. Arrancando rápido...")
            requiere_actualizacion = False
    except Exception as e:
        print(f"⚠️ Error leyendo cachés: {e}")

if requiere_actualizacion:
    if not hay_csvs_locales and not archivos_cache:
        raise FileNotFoundError("❌ No se encontraron cachés anuales ni archivos CSV.")
    if archivos_cache and hay_csvs_locales:
        if input("\n⚠️ Detectados cambios en 'Country_per_year'. ¿Regenerar cachés? (s/n): ").strip().lower() == "s":
            cache_cargado = procesar_todos_los_csv(archivos_encontrados, manifiesto_actual)
    elif hay_csvs_locales:
        print("🔄 Creando cachés anuales por primera vez...")
        cache_cargado = procesar_todos_los_csv(archivos_encontrados, manifiesto_actual)

zonas_diario: pd.DataFrame = cache_cargado["diario"]
zonas_mensual: pd.DataFrame = cache_cargado["mensual"]

resumen_paises = zonas_mensual.groupby(["pais", "anio"], observed=True).agg(total_incendios=("total_incendios", "sum"), frp_promedio=("frp_promedio", "mean"), brillo_promedio=("brillo_promedio", "mean")).reset_index().fillna(0)
resumen_paises["pais"] = resumen_paises["pais"].astype(str)
for col in ["frp_promedio", "brillo_promedio"]: resumen_paises[col] = resumen_paises[col].astype("float64").round(1)

CACHE_PAISES = resumen_paises.to_dict(orient="records")
PAISES_CARGADOS = sorted(zonas_mensual["pais"].astype(str).unique().tolist())
ANIOS_CARGADOS = sorted(int(a) for a in zonas_mensual["anio"].unique().tolist())

print(f"✅ ¡Servidor listo! Celdas mensuales: {len(zonas_mensual):,} | Registros diarios: {len(zonas_diario):,}\n🌎 Países ({len(PAISES_CARGADOS)}): {PAISES_CARGADOS} | Años: {ANIOS_CARGADOS}")


def serializar_limpio(df_slice: pd.DataFrame) -> list[dict]:
    if df_slice.empty: return []
    out = df_slice.copy()

    for col in out.select_dtypes(include=["category"]).columns: out[col] = out[col].astype(str)
    
    if "fecha" in out.columns and pd.api.types.is_integer_dtype(out["fecha"]):
        f_str = out["fecha"].astype(str)
        out["fecha"] = f_str.str.slice(0, 4) + "-" + f_str.str.slice(4, 6) + "-" + f_str.str.slice(6, 8)

    if "total_incendios" in out.columns:
        t = out["total_incendios"].to_numpy()
        out["nivel_riesgo"] = np.select([t > 50, t > 15], ["Alto", "Medio"], default="Bajo")
        out["color"] = np.select([t > 75, t > 60, t > 50, t > 15], ["#FF0000", "#FF8000", "#FFFF00", "#80FF00"], default="#2ECC71")

    for col in ["lat_grid", "lon_grid"]:
        if col in out.columns: out[col] = out[col].astype("float64").round(2)
    for col in ["frp_promedio", "brillo_promedio"]:
        if col in out.columns: out[col] = out[col].astype("float64").round(1)

    return out.replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="records")


# --- ENDPOINTS ---
@app.get("/")
def home():
    return {
        "status": "API LATAM/América Optimizada activa 🚀",
        "paises_indexados": len(PAISES_CARGADOS),
        "paises": PAISES_CARGADOS, "anios": ANIOS_CARGADOS,
        "total_registros_diarios": len(zonas_diario)
    }

@app.get("/api/filtros-disponibles")
def get_filtros_disponibles():
    fechas = zonas_diario["fecha"].astype(str)
    f_min, f_max = (fechas.min(), fechas.max()) if not fechas.empty else ("20180101", "20251231")
    return {
        "paises": PAISES_CARGADOS, "anios": ANIOS_CARGADOS, "meses": list(range(1, 13)),
        "fecha_min": f"{f_min[:4]}-{f_min[4:6]}-{f_min[6:8]}", "fecha_max": f"{f_max[:4]}-{f_max[4:6]}-{f_max[6:8]}",
        "instrumentos": sorted(zonas_mensual["instrumento"].astype(str).unique().tolist()),
        "satelites": sorted(zonas_mensual["satelite"].astype(str).unique().tolist()),
    }

@app.get("/api/incendios-grid")
def get_mapa_incendios(pais: str = None, anio: int = None, mes: int = None, fecha: str = None, fecha_inicio: str = None, fecha_fin: str = None, instrumento: str = None, limit: int = Query(default=45000, le=200000)):
    usa_diario = bool(fecha or fecha_inicio or fecha_fin)
    df_out = zonas_diario if usa_diario else zonas_mensual

    if anio: df_out = df_out[df_out["anio"] == anio]
    if pais and pais.lower() != "todos": df_out = df_out[df_out["pais"].astype(str).str.lower() == pais.lower()]
    if mes and 1 <= mes <= 12: df_out = df_out[df_out["mes"] == mes]
    if instrumento and instrumento.lower() != "todos": df_out = df_out[df_out["instrumento"].astype(str).str.upper() == instrumento.upper()]
    
    if fecha: df_out = df_out[df_out["fecha"] == int(fecha.replace("-", ""))]
    if fecha_inicio: df_out = df_out[df_out["fecha"] >= int(fecha_inicio.replace("-", ""))]
    if fecha_fin: df_out = df_out[df_out["fecha"] <= int(fecha_fin.replace("-", ""))]

    if len(df_out) > limit:
        if (not pais or pais.lower() == "todos") and not anio and not mes:
            df_out = df_out.groupby(["pais", "anio"], group_keys=False, observed=True).head(max(1, limit // max(1, df_out.groupby(["pais", "anio"], observed=True).ngroups)))
        elif not pais or pais.lower() == "todos":
            df_out = df_out.groupby("pais", group_keys=False, observed=True).head(max(1, limit // max(1, df_out["pais"].nunique())))
        else:
            df_out = df_out.head(limit)

    return serializar_limpio(df_out)

@app.get("/api/incendios-por-pais")
def get_resumen_por_pais(anio: int = None):
    return [r for r in CACHE_PAISES if r["anio"] == anio] if anio else CACHE_PAISES

@app.get("/api/incendios-por-tiempo")
def get_resumen_temporal(granularidad: str = Query(default="mes", pattern="^(anio|mes|fecha)$"), pais: str = None, anio: int = None):
    df_t = zonas_diario if granularidad == "fecha" else zonas_mensual
    if pais and pais.lower() != "todos": df_t = df_t[df_t["pais"].astype(str).str.lower() == pais.lower()]
    if anio: df_t = df_t[df_t["anio"] == anio]

    cols_grupo = ["anio"] if granularidad == "anio" else (["anio", "mes"] if granularidad == "mes" else ["fecha"])
    if pais and pais.lower() != "todos": cols_grupo = ["pais"] + cols_grupo

    res = df_t.groupby(cols_grupo, observed=True).agg(total_incendios=("total_incendios", "sum"), frp_promedio=("frp_promedio", "mean"), brillo_promedio=("brillo_promedio", "mean")).reset_index().sort_values(cols_grupo)
    return serializar_limpio(res)

if __name__ == "__main__":
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=8000, reload=False)
