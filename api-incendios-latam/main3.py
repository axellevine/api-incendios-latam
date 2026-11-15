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

app = FastAPI(title="API Mapa de Incendios LATAM/América (2018-2025)", version="8.0")

# Compresión GZip: reduce el peso de red hasta un 85% para no saturar el explorador
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS abierto para conexión local, Wi-Fi o túneles de VS Code / Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATOS = os.path.join(BASE_DIR, "Country_per_year")
PICKLE_CACHE = os.path.join(BASE_DIR, "cache_incendios_america_2018_2025.pkl")

PAISES_EXCLUIDOS = {
    "french guiana",
    "guyana francesa",
    "guyane",
    "french_guiana",
    "guyana_francesa",
}

NORMALIZACION_PAISES = {
    "brasil": "Brazil",
    "brazil": "Brazil",
    "peru": "Peru",
    "perú": "Peru",
    "mexico": "Mexico",
    "méxico": "Mexico",
    "panama": "Panama",
    "panamá": "Panama",
    "canada": "Canada",
    "canadá": "Canada",
    "usa": "United States",
    "united states": "United States",
    "estados unidos": "United States",
    "haiti": "Haiti",
    "haití": "Haiti",
    "republica dominicana": "Dominican Republic",
    "república dominicana": "Dominican Republic",
}

TOKENS_RUIDO = {
    "modis", "viirs", "fire", "dl", "suomi", "noaa", "j1", "j2", "jpss1", "jpss2",
    "c61", "c6", "nrt", "archive", "final", "clean", "limpio", "v1", "v2", "ok",
    "copy", "copia", "data", "datos", "incendios", "fires","filtrado", "filtered",
    "latam", "america", "america_latina", "latinoamerica","(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)", "(9)","snpp"," ",
}


def detectar_separador(ruta_archivo: str) -> str:
    """Detecta automáticamente si el CSV usa coma (,) o punto y coma (;)"""
    with open(ruta_archivo, "r", encoding="utf-8-sig", errors="ignore") as f:
        primera_linea = f.readline()
    return ";" if primera_linea.count(";") > primera_linea.count(",") else ","


def limpiar_numerico(serie: pd.Series) -> pd.Series:
    """Convierte comas decimales a puntos, fuerza numérico y limpia infinitos"""
    if serie.dtype == object:
        serie = serie.astype(str).str.strip().str.replace(",", ".", regex=False)
    num = pd.to_numeric(serie, errors="coerce")
    return num.replace([np.inf, -np.inf], np.nan)


def normalizar_satelite(serie: pd.Series, sensor_defecto: str) -> pd.Series:
    """Estandariza códigos abreviados de NASA FIRMS para el frontend"""
    s = serie.fillna("").astype(str).str.strip()
    mapa = {
        "T": "Terra",
        "A": "Aqua",
        "N": "Suomi-NPP",
        "1": "NOAA-20",
        "2": "NOAA-21",
        "": "JPSS-1" if sensor_defecto == "VIIRS" else "Terra",
    }
    return s.replace(mapa)


def extraer_pais_y_anio(ruta_archivo: str) -> tuple[str, int]:
    """
    Extrae el país y el año analizando tanto el nombre del archivo como
    las subcarpetas dentro de 'Country_per_year' (ej. Country_per_year/2024/Brazil.csv
    o Country_per_year/Brazil/2024.csv).
    """
    rel_path = os.path.relpath(ruta_archivo, CARPETA_DATOS)
    partes_ruta = rel_path.split(os.sep)
    carpetas_padre = partes_ruta[:-1]
    nombre_base = os.path.splitext(partes_ruta[-1])[0]
    nombre_base = re.sub(r"\s*\(\d+\)$", "", nombre_base).strip()

    # 1. Detectar año en las subcarpetas o en el nombre del archivo
    anio_fallback = 2025
    tokens_todos = []
    for segmento in carpetas_padre + [nombre_base]:
        tokens_todos.extend(re.split(r"[_\-\s]+", segmento))

    for tok in tokens_todos:
        if tok.isdigit() and 2018 <= int(tok) <= 2025:
            anio_fallback = int(tok)
            break

    # 2. Detectar país en el nombre del archivo (ignorando años y palabras técnicas)
    tokens_archivo = [
        t for t in re.split(r"[_\-\s]+", nombre_base)
        if t and not t.isdigit() and t.lower() not in TOKENS_RUIDO
    ]

    if tokens_archivo:
        pais_raw = " ".join(tokens_archivo).strip()
    else:
        # Si el archivo solo se llama ej. "2024.csv" o "modis_2024.csv", buscar el país en la subcarpeta
        pais_carpeta = None
        for carpeta in reversed(carpetas_padre):
            tokens_carp = [
                t for t in re.split(r"[_\-\s]+", carpeta)
                if t and not t.isdigit() and t.lower() not in TOKENS_RUIDO
            ]
            if tokens_carp:
                pais_carpeta = " ".join(tokens_carp).strip()
                break
        pais_raw = pais_carpeta if pais_carpeta else "Desconocido"

    pais_norm = NORMALIZACION_PAISES.get(pais_raw.lower(), pais_raw.title())
    return pais_norm, anio_fallback


def obtener_archivos_validos() -> list[str]:
    """Busca recursivamente TODOS los archivos CSV dentro de 'Country_per_year' y sus subcarpetas."""
    if not os.path.isdir(CARPETA_DATOS):
        return []

    candidatos = glob.glob(os.path.join(CARPETA_DATOS, "**", "*.csv"), recursive=True)
    validos = [
        f for f in candidatos
        if "venv" not in f.lower() and ".venv" not in f.lower()
    ]
    return sorted(validos)


def obtener_manifiesto_csv(archivos: list[str]) -> list[tuple[str, int, float]]:
    """Genera firma única (ruta relativa, tamaño, fecha de modificación) de cada CSV."""
    manifiesto = []
    for f in archivos:
        rel_path = os.path.relpath(f, BASE_DIR)
        st = os.stat(f)
        manifiesto.append((rel_path, st.st_size, round(st.st_mtime, 2)))
    return manifiesto


def asignar_riesgo_y_color(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula nivel de riesgo y color hexadecimal de forma vectorizada."""
    t = df["total_incendios"].to_numpy()
    df["nivel_riesgo"] = pd.Categorical(
        np.select([t > 50, t > 15], ["Alto", "Medio"], default="Bajo")
    )
    df["color"] = pd.Categorical(
        np.select(
            [t > 75, t > 60, t > 30, t < 30],
            ["#FF0000", "#FF8000", "#FFFF00", "#80FF00"],
            default="#2ECC71",
        )
    )
    return df


def procesar_todos_los_csv(archivos_csv: list[str], manifiesto: list) -> dict:
    """Procesamiento por bloques de 250,000 filas para mantener bajo el consumo de RAM."""
    resumenes_diarios_archivos = []

    for archivo in archivos_csv:
        rel_display = os.path.relpath(archivo, CARPETA_DATOS)
        nombre_archivo = os.path.basename(archivo)
        pais, anio_archivo = extraer_pais_y_anio(archivo)

        if pais.lower() in PAISES_EXCLUIDOS:
            print(f"⏭️ Excluyendo {rel_display} ({pais})")
            continue

        sep = detectar_separador(archivo)
        sensor_archivo = (
            "VIIRS"
            if "viirs" in archivo.lower() or nombre_archivo.lower().startswith(("suomi", "noaa", "j1", "j2"))
            else "MODIS"
        )

        try:
            header_df = pd.read_csv(archivo, sep=sep, nrows=0)
        except Exception as e:
            print(f"⚠️ No se pudo leer encabezado de {rel_display}: {e}")
            continue

        header_df.columns = header_df.columns.str.lower().str.strip()

        lat_col = next((c for c in header_df.columns if c in ["latitude", "lat", "y"] or c.startswith("lat")), None)
        lon_col = next((c for c in header_df.columns if c in ["longitude", "lon", "long", "x"] or c.startswith("lon")), None)
        if not lat_col or not lon_col or lat_col == lon_col:
            print(f"⚠️ Saltando {rel_display}: no contiene columnas de coordenadas válidas.")
            continue

        date_col = next((c for c in header_df.columns if "date" in c or "acq_date" in c or "time" in c), None)
        frp_col = next((c for c in header_df.columns if "frp" in c), None)
        bright_col = next((c for c in header_df.columns if c in ["brightness", "bright_ti4"] or "bright" in c), None)
        inst_col = next((c for c in header_df.columns if "instrument" in c or "inst" in c), None)
        sat_col = next((c for c in header_df.columns if "satellite" in c or "sat" in c), None)

        cols_to_use = [lat_col, lon_col]
        for col in [date_col, frp_col, bright_col, inst_col, sat_col]:
            if col and col not in cols_to_use:
                cols_to_use.append(col)

        rename_map = {lat_col: "lat", lon_col: "lon"}
        if date_col: rename_map[date_col] = "fecha"
        if frp_col: rename_map[frp_col] = "frp"
        if bright_col: rename_map[bright_col] = "bright"
        if inst_col: rename_map[inst_col] = "instrumento"
        if sat_col: rename_map[sat_col] = "satelite"

        print(f"Procesando: {rel_display} -> País: {pais} | Año base: {anio_archivo} | Sensor: {sensor_archivo}")
        chunks_archivo = []

        reader = pd.read_csv(
            archivo,
            sep=sep,
            usecols=lambda c: c.lower().strip() in cols_to_use,
            chunksize=250_000,
            low_memory=False,
            on_bad_lines="skip",
        )

        for chunk in reader:
            chunk.columns = chunk.columns.str.lower().str.strip()
            chunk = chunk.rename(columns=rename_map)
            chunk = chunk.loc[:, ~chunk.columns.duplicated()]

            chunk["lat"] = limpiar_numerico(chunk["lat"]).astype("float32")
            chunk["lon"] = limpiar_numerico(chunk["lon"]).astype("float32")
            chunk = chunk.dropna(subset=["lat", "lon"])
            chunk = chunk[(chunk["lat"].between(-90.0, 90.0)) & (chunk["lon"].between(-180.0, 180.0))]
            if chunk.empty:
                continue

            chunk["frp"] = limpiar_numerico(chunk["frp"]).fillna(0.0).astype("float32") if "frp" in chunk.columns else 0.0
            chunk["bright"] = limpiar_numerico(chunk["bright"]).fillna(0.0).astype("float32") if "bright" in chunk.columns else 0.0

            if "instrumento" in chunk.columns:
                inst = chunk["instrumento"].fillna(sensor_archivo).astype(str).str.strip()
                chunk["instrumento"] = np.where(inst.str.upper().str.contains("VIIRS"), "VIIRS", "MODIS")
            else:
                chunk["instrumento"] = sensor_archivo

            if "satelite" in chunk.columns:
                chunk["satelite"] = normalizar_satelite(chunk["satelite"], sensor_archivo)
            else:
                chunk["satelite"] = "JPSS-1" if sensor_archivo == "VIIRS" else "Terra"

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
            if chunk.empty:
                continue

            chunk["lat_grid"] = chunk["lat"].round(2)
            chunk["lon_grid"] = chunk["lon"].round(2)

            agg_chunk = (
                chunk.groupby(["anio", "mes", "fecha", "lat_grid", "lon_grid", "instrumento", "satelite"], observed=True)
                .agg(
                    total_incendios=("lat", "count"),
                    suma_frp=("frp", "sum"),
                    suma_brillo=("bright", "sum"),
                )
                .reset_index()
            )
            chunks_archivo.append(agg_chunk)

        if chunks_archivo:
            df_file = pd.concat(chunks_archivo, ignore_index=True)
            df_file = (
                df_file.groupby(["anio", "mes", "fecha", "lat_grid", "lon_grid", "instrumento", "satelite"], observed=True)
                .agg(
                    total_incendios=("total_incendios", "sum"),
                    suma_frp=("suma_frp", "sum"),
                    suma_brillo=("suma_brillo", "sum"),
                )
                .reset_index()
            )
            df_file["pais"] = pais
            resumenes_diarios_archivos.append(df_file)

        del chunks_archivo
        gc.collect()

    if not resumenes_diarios_archivos:
        raise ValueError(f"❌ No se encontraron registros válidos dentro de '{CARPETA_DATOS}'.")

    print("Consolidando base diaria y mensual de Country_per_year...")
    df_all = pd.concat(resumenes_diarios_archivos, ignore_index=True)
    del resumenes_diarios_archivos
    gc.collect()

    # 1. TABLA DIARIA EXACTA
    diario = (
        df_all.groupby(["pais", "anio", "mes", "fecha", "lat_grid", "lon_grid", "instrumento", "satelite"], observed=True)
        .agg(
            total_incendios=("total_incendios", "sum"),
            suma_frp=("suma_frp", "sum"),
            suma_brillo=("suma_brillo", "sum"),
        )
        .reset_index()
    )
    del df_all
    gc.collect()

    # 2. TABLA MENSUAL / ANUAL
    mensual = (
        diario.groupby(["pais", "anio", "mes", "lat_grid", "lon_grid", "instrumento", "satelite"], observed=True)
        .agg(
            total_incendios=("total_incendios", "sum"),
            suma_frp=("suma_frp", "sum"),
            suma_brillo=("suma_brillo", "sum"),
            fecha=("fecha", "max"),
        )
        .reset_index()
    )

    for tabla in (diario, mensual):
        tot_safe = tabla["total_incendios"].replace(0, 1)
        tabla["frp_promedio"] = (tabla["suma_frp"] / tot_safe).round(1).fillna(0.0).astype("float32")
        tabla["brillo_promedio"] = (tabla["suma_brillo"] / tot_safe).round(1).fillna(0.0).astype("float32")
        tabla.drop(columns=["suma_frp", "suma_brillo"], inplace=True)

        tabla["lat_grid"] = tabla["lat_grid"].round(2).astype("float32")
        tabla["lon_grid"] = tabla["lon_grid"].round(2).astype("float32")
        tabla["total_incendios"] = tabla["total_incendios"].astype("int32")
        tabla["anio"] = tabla["anio"].astype("int16")
        tabla["mes"] = tabla["mes"].astype("int8")

        for col_cat in ["pais", "fecha", "instrumento", "satelite"]:
            tabla[col_cat] = tabla[col_cat].astype("category")

        asignar_riesgo_y_color(tabla)

    diario = diario.sort_values("total_incendios", ascending=False).reset_index(drop=True)
    mensual = mensual.sort_values("total_incendios", ascending=False).reset_index(drop=True)

    payload_cache = {"manifiesto": manifiesto, "diario": diario, "mensual": mensual}
    pd.to_pickle(payload_cache, PICKLE_CACHE)
    print(f"💾 Caché actualizado y guardado en {PICKLE_CACHE}")
    return payload_cache


# --- INICIALIZACIÓN INTELIGENTE DEL SERVIDOR ---
try:
    archivos_encontrados = obtener_archivos_validos()
    manifiesto_actual = obtener_manifiesto_csv(archivos_encontrados)
except Exception:
    archivos_encontrados = []
    manifiesto_actual = []

cache_cargado = None
requiere_actualizacion = False
hay_csvs_locales = len(archivos_encontrados) > 0

if os.path.exists(PICKLE_CACHE):
    try:
        cache_obj = pd.read_pickle(PICKLE_CACHE)
        if isinstance(cache_obj, dict) and "diario" in cache_obj and "mensual" in cache_obj:
            if not hay_csvs_locales:
                print("☁️ Modo Nube / Sin carpeta Country_per_year: Cargando datos desde el caché...")
                cache_cargado = cache_obj
            elif cache_obj.get("manifiesto") == manifiesto_actual:
                print(f"⚡ Caché sincronizado ({len(archivos_encontrados)} archivos en Country_per_year). Arrancando rápido...")
                cache_cargado = cache_obj
            else:
                requiere_actualizacion = True
        else:
            print("⚠️ Caché corrupto o antiguo detectado.")
            requiere_actualizacion = True
    except Exception as e:
        print(f"⚠️ Error leyendo caché: {e}")
        requiere_actualizacion = True
else:
    requiere_actualizacion = True

if requiere_actualizacion:
    if not hay_csvs_locales:
        raise FileNotFoundError(
            f"❌ No se encontró el caché '{PICKLE_CACHE}' ni archivos CSV dentro de '{CARPETA_DATOS}'."
        )

    if os.path.exists(PICKLE_CACHE):
        respuesta = input(
            "\n⚠️ Se detectaron cambios en 'Country_per_year' o sus subcarpetas. "
            "¿Deseas regenerar el caché ahora? (s/n): "
        ).strip().lower()
        if respuesta != "s":
            print("⏭️ Omitiendo actualización. Usando el caché anterior...")
            cache_cargado = pd.read_pickle(PICKLE_CACHE)
        else:
            print("🔄 Regenerando caché con los archivos de Country_per_year...")
            cache_cargado = procesar_todos_los_csv(archivos_encontrados, manifiesto_actual)
    else:
        print(f"🔄 Creando caché por primera vez desde '{CARPETA_DATOS}'...")
        cache_cargado = procesar_todos_los_csv(archivos_encontrados, manifiesto_actual)

zonas_diario: pd.DataFrame = cache_cargado["diario"]
zonas_mensual: pd.DataFrame = cache_cargado["mensual"]

# Precalcular resúmenes globales
resumen_paises = (
    zonas_mensual.groupby(["pais", "anio"], observed=True)
    .agg(
        total_incendios=("total_incendios", "sum"),
        frp_promedio=("frp_promedio", "mean"),
        brillo_promedio=("brillo_promedio", "mean"),
    )
    .reset_index()
    .fillna(0)
)
resumen_paises["pais"] = resumen_paises["pais"].astype(str)
resumen_paises["frp_promedio"] = resumen_paises["frp_promedio"].astype("float64").round(1)
resumen_paises["brillo_promedio"] = resumen_paises["brillo_promedio"].astype("float64").round(1)

CACHE_PAISES = resumen_paises.to_dict(orient="records")
PAISES_CARGADOS = sorted(zonas_mensual["pais"].astype(str).unique().tolist())
ANIOS_CARGADOS = sorted(int(a) for a in zonas_mensual["anio"].unique().tolist())

print(
    f"✅ ¡Servidor listo! Celdas mensuales: {len(zonas_mensual):,} | Registros diarios: {len(zonas_diario):,}\n"
    f"🌎 Países ({len(PAISES_CARGADOS)}): {PAISES_CARGADOS} | Años: {ANIOS_CARGADOS}"
)


def serializar_limpio(df_slice: pd.DataFrame) -> list[dict]:
    """Prepara la respuesta JSON ligera limpiando tipos y redondeando decimales."""
    if df_slice.empty:
        return []
    out = df_slice.copy()
    for col in out.select_dtypes(include=["category"]).columns:
        out[col] = out[col].astype(str)
    for col in ["lat_grid", "lon_grid"]:
        if col in out.columns:
            out[col] = out[col].astype("float64").round(2)
    for col in ["frp_promedio", "brillo_promedio"]:
        if col in out.columns:
            out[col] = out[col].astype("float64").round(1)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out.to_dict(orient="records")


# --- ENDPOINTS REST API ---
@app.get("/")
def home():
    return {
        "status": "API LATAM/América Optimizada activa 🚀",
        "carpeta_origen": "Country_per_year",
        "paises_indexados": len(PAISES_CARGADOS),
        "paises": PAISES_CARGADOS,
        "anios": ANIOS_CARGADOS,
        "total_registros_diarios": len(zonas_diario),
        "endpoints": {
            "mapa": "/api/incendios-grid",
            "resumen": "/api/incendios-por-pais",
            "series": "/api/incendios-por-tiempo",
            "filtros": "/api/filtros-disponibles",
        },
    }


@app.get("/api/filtros-disponibles")
def get_filtros_disponibles():
    fechas_str = zonas_diario["fecha"].astype(str)
    return {
        "paises": PAISES_CARGADOS,
        "anios": ANIOS_CARGADOS,
        "meses": list(range(1, 13)),
        "fecha_min": fechas_str.min() if not fechas_str.empty else "2018-01-01",
        "fecha_max": fechas_str.max() if not fechas_str.empty else "2025-12-31",
        "instrumentos": sorted(zonas_mensual["instrumento"].astype(str).unique().tolist()),
        "satelites": sorted(zonas_mensual["satelite"].astype(str).unique().tolist()),
    }


@app.get("/api/incendios-grid")
def get_mapa_incendios(
    pais: str = None,
    anio: int = None,
    mes: int = None,
    fecha: str = None,
    fecha_inicio: str = None,
    fecha_fin: str = None,
    instrumento: str = None,
    limit: int = Query(default=45000, le=200000),
):
    usa_diario = bool(fecha or fecha_inicio or fecha_fin)
    df_out = zonas_diario if usa_diario else zonas_mensual

    if anio:
        df_out = df_out[df_out["anio"] == anio]
    if pais and pais.lower() != "todos":
        df_out = df_out[df_out["pais"].astype(str).str.lower() == pais.lower()]
    if mes and 1 <= mes <= 12:
        df_out = df_out[df_out["mes"] == mes]
    if instrumento and instrumento.lower() != "todos":
        df_out = df_out[df_out["instrumento"].astype(str).str.upper() == instrumento.upper()]
    if fecha:
        df_out = df_out[df_out["fecha"].astype(str) == fecha]
    if fecha_inicio:
        df_out = df_out[df_out["fecha"].astype(str) >= fecha_inicio]
    if fecha_fin:
        df_out = df_out[df_out["fecha"].astype(str) <= fecha_fin]

    if len(df_out) > limit:
        sin_pais = not pais or pais.lower() == "todos"
        if sin_pais and not anio and not mes:
            n_buckets = max(1, df_out.groupby(["pais", "anio"], observed=True).ngroups)
            por_bucket = max(1, limit // n_buckets)
            df_out = df_out.groupby(["pais", "anio"], group_keys=False, observed=True).head(por_bucket)
        elif sin_pais:
            n_paises = max(1, df_out["pais"].nunique())
            por_pais = max(1, limit // n_paises)
            df_out = df_out.groupby("pais", group_keys=False, observed=True).head(por_pais)
        else:
            df_out = df_out.head(limit)

    return serializar_limpio(df_out)


@app.get("/api/incendios-por-pais")
def get_resumen_por_pais(anio: int = None):
    if anio:
        return [r for r in CACHE_PAISES if r["anio"] == anio]
    return CACHE_PAISES


@app.get("/api/incendios-por-tiempo")
def get_resumen_temporal(
    granularidad: str = Query(default="mes", pattern="^(anio|mes|fecha)$"),
    pais: str = None,
    anio: int = None,
):
    df_t = zonas_diario if granularidad == "fecha" else zonas_mensual

    if pais and pais.lower() != "todos":
        df_t = df_t[df_t["pais"].astype(str).str.lower() == pais.lower()]
    if anio:
        df_t = df_t[df_t["anio"] == anio]

    cols_grupo = ["anio"] if granularidad == "anio" else (["anio", "mes"] if granularidad == "mes" else ["fecha"])
    if pais and pais.lower() != "todos":
        cols_grupo = ["pais"] + cols_grupo

    res = (
        df_t.groupby(cols_grupo, observed=True)
        .agg(
            total_incendios=("total_incendios", "sum"),
            frp_promedio=("frp_promedio", "mean"),
            brillo_promedio=("brillo_promedio", "mean"),
        )
        .reset_index()
        .sort_values(cols_grupo)
    )
    return serializar_limpio(res)


if __name__ == "__main__":
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=8000, reload=False)