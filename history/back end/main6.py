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

# Memoria global optimizada para no cargar todo de golpe
CACHE_EN_MEMORIA = {
    "diario": pd.DataFrame(),
    "mensual": pd.DataFrame(),
    "anio_cargado": None,
    "resumen_paises": []
}
PAISES_CARGADOS = []
ANIOS_CARGADOS = []

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

def optimizar_memoria(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce el consumo de RAM convirtiendo tipos de datos pesados a ligeros."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in ["pais", "instrumento", "satelite"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().astype("category")
    return df


def cargar_anio_en_memoria(anio_solicitado: int):
    """Carga un año específico bajo demanda para no saturar los 512 MB de la nube."""
    global CACHE_EN_MEMORIA, PAISES_CARGADOS, ANIOS_CARGADOS
    
    # Si ya está cargado, no hacer nada
    if CACHE_EN_MEMORIA["anio_cargado"] == anio_solicitado:
        return
        
    ruta = os.path.join(CARPETA_CACHES, f"cache_{anio_solicitado}.pkl.gz")
    if not os.path.exists(ruta):
        return

    # Limpiar memoria anterior
    CACHE_EN_MEMORIA["diario"] = pd.DataFrame()
    CACHE_EN_MEMORIA["mensual"] = pd.DataFrame()
    gc.collect()

    try:
        obj = pd.read_pickle(ruta, compression="gzip")
        CACHE_EN_MEMORIA["diario"] = optimizar_memoria(obj["diario"])
        CACHE_EN_MEMORIA["mensual"] = optimizar_memoria(obj["mensual"])
        CACHE_EN_MEMORIA["anio_cargado"] = anio_solicitado
        
        # Actualizar metadatos
        PAISES_CARGADOS = sorted(CACHE_EN_MEMORIA["mensual"]["pais"].dropna().astype(str).unique().tolist())
        
        # Calcular resumen de países para ese año
        resumen = CACHE_EN_MEMORIA["mensual"].groupby(["pais", "anio"], observed=True).agg(
            total_incendios=("total_incendios", "sum"), 
            frp_promedio=("frp_promedio", "mean"), 
            brillo_promedio=("brillo_promedio", "mean")
        ).reset_index().fillna(0)
        resumen["pais"] = resumen["pais"].astype(str)
        for col in ["frp_promedio", "brillo_promedio"]: 
            resumen[col] = resumen[col].astype("float32").round(1)
            
        CACHE_EN_MEMORIA["resumen_paises"] = resumen.to_dict(orient="records")
        print(f"✅ Año {anio_solicitado} cargado en memoria RAM.")
    except Exception as e:
        print(f"⚠️ Error cargando año {anio_solicitado}: {e}")


def inicializar_api():
    """Detecta los años disponibles y carga solo el más reciente al arrancar."""
    global ANIOS_CARGADOS
    archivos_cache = sorted(glob.glob(os.path.join(CARPETA_CACHES, "cache_20*.pkl.gz")))
    
    if archivos_cache:
        # Extraer años disponibles leyendo los nombres de los archivos
        ANIOS_CARGADOS = sorted([int(re.search(r'cache_(\d{4})', f).group(1)) for f in archivos_cache])
        # Cargar solo el año más reciente al encender el servidor
        cargar_anio_en_memoria(ANIOS_CARGADOS[-1])
        print(f"🚀 Servidor encendido en modo nube. Años indexados: {ANIOS_CARGADOS}")
    else:
        print("❌ No se encontraron cachés anuales para iniciar.")

# Ejecutar inicialización al arrancar
inicializar_api()

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
        "anio_en_memoria": CACHE_EN_MEMORIA["anio_cargado"],
        "total_registros_diarios": len(CACHE_EN_MEMORIA["diario"])
    }

@app.get("/api/filtros-disponibles")
def get_filtros_disponibles():
    # Usar el año cargado para extraer meses e instrumentos
    fechas = CACHE_EN_MEMORIA["diario"]["fecha"].astype(str) if not CACHE_EN_MEMORIA["diario"].empty else pd.Series([])
    f_min, f_max = (fechas.min(), fechas.max()) if not fechas.empty else (f"{ANIOS_CARGADOS[0]}0101", f"{ANIOS_CARGADOS[-1]}1231")
    return {
        "paises": PAISES_CARGADOS, "anios": ANIOS_CARGADOS, "meses": list(range(1, 13)),
        "fecha_min": f"{f_min[:4]}-{f_min[4:6]}-{f_min[6:8]}", "fecha_max": f"{f_max[:4]}-{f_max[4:6]}-{f_max[6:8]}",
        "instrumentos": sorted(CACHE_EN_MEMORIA["mensual"]["instrumento"].dropna().astype(str).unique().tolist()) if not CACHE_EN_MEMORIA["mensual"].empty else [],
        "satelites": sorted(CACHE_EN_MEMORIA["mensual"]["satelite"].dropna().astype(str).unique().tolist()) if not CACHE_EN_MEMORIA["mensual"].empty else [],
    }

@app.get("/api/incendios-grid")
def get_mapa_incendios(pais: str = None, anio: int = None, mes: int = None, fecha: str = None, fecha_inicio: str = None, fecha_fin: str = None, instrumento: str = None, limit: int = Query(default=45000, le=200000)):
    # Si el frontend pide un año distinto, lo cargamos en RAM dinámicamente
    anio_query = anio
    if not anio_query and fecha:
        anio_query = int(fecha[:4])
    
    if anio_query and anio_query != CACHE_EN_MEMORIA["anio_cargado"]:
        cargar_anio_en_memoria(anio_query)

    usa_diario = bool(fecha or fecha_inicio or fecha_fin)
    df_out = CACHE_EN_MEMORIA["diario"] if usa_diario else CACHE_EN_MEMORIA["mensual"]

    if df_out.empty:
        return []

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
    if anio and anio != CACHE_EN_MEMORIA["anio_cargado"]:
        cargar_anio_en_memoria(anio)
    return CACHE_EN_MEMORIA["resumen_paises"]

@app.get("/api/incendios-por-tiempo")
def get_resumen_temporal(granularidad: str = Query(default="mes", pattern="^(anio|mes|fecha)$"), pais: str = None, anio: int = None):
    if anio and anio != CACHE_EN_MEMORIA["anio_cargado"]:
        cargar_anio_en_memoria(anio)
        
    df_t = CACHE_EN_MEMORIA["diario"] if granularidad == "fecha" else CACHE_EN_MEMORIA["mensual"]
    
    if df_t.empty:
        return []
        
    if pais and pais.lower() != "todos": df_t = df_t[df_t["pais"].astype(str).str.lower() == pais.lower()]
    if anio: df_t = df_t[df_t["anio"] == anio]

    cols_grupo = ["anio"] if granularidad == "anio" else (["anio", "mes"] if granularidad == "mes" else ["fecha"])
    if pais and pais.lower() != "todos": cols_grupo = ["pais"] + cols_grupo

    res = df_t.groupby(cols_grupo, observed=True).agg(total_incendios=("total_incendios", "sum"), frp_promedio=("frp_promedio", "mean"), brillo_promedio=("brillo_promedio", "mean")).reset_index().sort_values(cols_grupo)
    return serializar_limpio(res)

if __name__ == "__main__":
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=8000, reload=False)