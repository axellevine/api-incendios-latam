import gc
import glob
import io
import json
import os
import re
from typing import Iterator, List, Optional

from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uvicorn

app = FastAPI(title="API Mapa de Incendios LATAM/América", version="11.0-lowmem")

# Nota: No usamos GZipMiddleware sobre respuestas grandes porque Starlette
# almacena todo el cuerpo sin comprimir en RAM antes de comprimirlo.
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
RUTA_METADATA = os.path.join(CARPETA_CACHES, "metadata_catalogo.json")

os.makedirs(CARPETA_CACHES, exist_ok=True)

# Solo metadatos ligeros (~15 KB) residen en RAM; los datos se filtran directo desde Parquet en disco
CATALOGO = {
    "paises": [],
    "anios": [],
    "instrumentos": [],
    "satelites": [],
    "fecha_min": "",
    "fecha_max": "",
    "resumen_paises": [],
}

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
    "cuba": "Cuba",
}


def optimizar_tipos_df(df: pd.DataFrame, anio_fallback: int) -> pd.DataFrame:
    """Optimiza tipos numéricos sin perder precisión decimal de coordenadas, FRP o brillo."""
    out = df.copy()
    if "anio" not in out.columns:
        out["anio"] = np.int16(anio_fallback)
    if "mes" not in out.columns and "fecha" in out.columns and pd.api.types.is_integer_dtype(out["fecha"]):
        out["mes"] = ((out["fecha"] // 100) % 100).astype("int8")

    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = out[col].astype("float32")
    for col in ["anio", "mes", "total_incendios"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in ["pais", "instrumento", "satelite"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    return out


def convertir_pkl_a_parquet_si_falta() -> None:
    """Migra cachés .pkl.gz a Apache Parquet ordenado (uno por uno para no exceder 512 MB)."""
    archivos_pkl = sorted(glob.glob(os.path.join(CARPETA_CACHES, "cache_20*.pkl.gz")))
    for ruta_pkl in archivos_pkl:
        match = re.search(r"cache_(\d{4})", os.path.basename(ruta_pkl))
        if not match:
            continue
        anio = int(match.group(1))
        ruta_pq_diario = os.path.join(CARPETA_CACHES, f"cache_{anio}_diario.parquet")
        ruta_pq_mensual = os.path.join(CARPETA_CACHES, f"cache_{anio}_mensual.parquet")

        if os.path.exists(ruta_pq_diario) and os.path.exists(ruta_pq_mensual):
            continue

        try:
            obj = pd.read_pickle(ruta_pkl, compression="gzip")
            for tipo, ruta_dest in (("diario", ruta_pq_diario), ("mensual", ruta_pq_mensual)):
                if not os.path.exists(ruta_dest) and tipo in obj and not obj[tipo].empty:
                    df_opt = optimizar_tipos_df(obj[tipo], anio)
                    sort_cols = [c for c in ["pais", "mes", "fecha"] if c in df_opt.columns]
                    if sort_cols:
                        df_opt = df_opt.sort_values(sort_cols).reset_index(drop=True)
                    df_opt.to_parquet(ruta_dest, engine="pyarrow", compression="zstd", index=False)
                    del df_opt
            del obj
            gc.collect()
            print(f"✅ Caché Parquet listo para {anio}.")
        except Exception as e:
            print(f"⚠️ Error migrando {ruta_pkl} a Parquet: {e}")


def construir_o_cargar_catalogo(anios: List[int]) -> None:
    """Carga o construye un JSON ligero (~15 KB) con países, sensores, fechas y resumen anual."""
    global CATALOGO
    if not anios:
        return

    if os.path.exists(RUTA_METADATA):
        try:
            with open(RUTA_METADATA, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("anios") == anios and meta.get("paises"):
                CATALOGO = meta
                return
        except Exception:
            pass

    paises_set, inst_set, sat_set = set(), set(), set()
    fechas_min, fechas_max = [], []
    resumen_total = []

    for anio in anios:
        ruta_m = os.path.join(CARPETA_CACHES, f"cache_{anio}_mensual.parquet")
        ruta_d = os.path.join(CARPETA_CACHES, f"cache_{anio}_diario.parquet")

        if os.path.exists(ruta_m):
            cols_m = [c for c in ["pais", "anio", "instrumento", "satelite", "total_incendios", "frp_promedio", "brillo_promedio"] if c in pq.read_schema(ruta_m).names]
            df_m = pq.read_table(ruta_m, columns=cols_m).to_pandas()
            if not df_m.empty:
                paises_set.update(df_m["pais"].dropna().astype(str).unique())
                if "instrumento" in df_m.columns:
                    inst_set.update(df_m["instrumento"].dropna().astype(str).unique())
                if "satelite" in df_m.columns:
                    sat_set.update(df_m["satelite"].dropna().astype(str).unique())

                res = (
                    df_m.groupby(["pais", "anio"], observed=True)
                    .agg(
                        total_incendios=("total_incendios", "sum"),
                        frp_promedio=("frp_promedio", "mean"),
                        brillo_promedio=("brillo_promedio", "mean"),
                    )
                    .reset_index()
                    .fillna(0)
                )
                res["pais"] = res["pais"].astype(str)
                res["anio"] = res["anio"].astype(int)
                res["total_incendios"] = res["total_incendios"].astype(int)
                res["frp_promedio"] = res["frp_promedio"].astype("float64").round(1)
                res["brillo_promedio"] = res["brillo_promedio"].astype("float64").round(1)
                resumen_total.extend(res.to_dict(orient="records"))
            del df_m

        if os.path.exists(ruta_d) and "fecha" in pq.read_schema(ruta_d).names:
            df_f = pq.read_table(ruta_d, columns=["fecha"]).to_pandas()
            if not df_f.empty:
                f_str = df_f["fecha"].astype(str)
                fechas_min.append(f_str.min())
                fechas_max.append(f_str.max())
            del df_f

        gc.collect()

    f_min = min(fechas_min) if fechas_min else f"{anios[0]}0101"
    f_max = max(fechas_max) if fechas_max else f"{anios[-1]}1231"

    CATALOGO = {
        "paises": sorted(p for p in paises_set if p and p.lower() not in PAISES_EXCLUIDOS),
        "anios": anios,
        "instrumentos": sorted(inst_set),
        "satelites": sorted(sat_set),
        "fecha_min": f"{f_min[:4]}-{f_min[4:6]}-{f_min[6:8]}" if len(f_min) == 8 else f_min,
        "fecha_max": f"{f_max[:4]}-{f_max[4:6]}-{f_max[6:8]}" if len(f_max) == 8 else f_max,
        "resumen_paises": resumen_total,
    }

    try:
        with open(RUTA_METADATA, "w", encoding="utf-8") as f:
            json.dump(CATALOGO, f, ensure_ascii=False)
    except Exception:
        pass


def inicializar_api() -> None:
    convertir_pkl_a_parquet_si_falta()
    archivos_pq = sorted(glob.glob(os.path.join(CARPETA_CACHES, "cache_20*_mensual.parquet")))
    anios = sorted([int(re.search(r"cache_(\d{4})", os.path.basename(f)).group(1)) for f in archivos_pq])
    construir_o_cargar_catalogo(anios)
    print(f"🚀 API activa en modo bajo consumo (<100 MB RAM). Años indexados: {CATALOGO['anios']}")


inicializar_api()


def _resolver_nombre_pais(pais: Optional[str]) -> Optional[str]:
    if not pais or pais.lower() == "todos":
        return None
    norm = NORMALIZACION_PAISES.get(pais.strip().lower(), pais.strip())
    for p_cat in CATALOGO["paises"]:
        if p_cat.lower() == norm.lower():
            return p_cat
    return norm


def leer_parquet_filtrado(
    anio: int,
    tipo: str,
    pais: Optional[str] = None,
    mes: Optional[int] = None,
    fecha: Optional[str] = None,
    fecha_inicio: Optional[str] = None,
    fecha_fin: Optional[str] = None,
    instrumento: Optional[str] = None,
    columnas: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Lee únicamente las filas y columnas necesarias desde disco usando predicate pushdown."""
    ruta = os.path.join(CARPETA_CACHES, f"cache_{anio}_{tipo}.parquet")
    if not os.path.exists(ruta):
        return pd.DataFrame()

    schema_cols = set(pq.read_schema(ruta).names)
    filtros = []

    pais_real = _resolver_nombre_pais(pais)
    if pais_real and "pais" in schema_cols:
        filtros.append(("pais", "==", pais_real))
    if mes and 1 <= mes <= 12 and "mes" in schema_cols:
        filtros.append(("mes", "==", int(mes)))
    if instrumento and instrumento.lower() != "todos" and "instrumento" in schema_cols:
        filtros.append(("instrumento", "==", instrumento.upper()))

    if tipo == "diario" and "fecha" in schema_cols:
        if fecha:
            filtros.append(("fecha", "==", int(fecha.replace("-", ""))))
        if fecha_inicio:
            filtros.append(("fecha", ">=", int(fecha_inicio.replace("-", ""))))
        if fecha_fin:
            filtros.append(("fecha", "<=", int(fecha_fin.replace("-", ""))))

    cols_validas = [c for c in columnas if c in schema_cols] if columnas else None
    tabla = pq.read_table(ruta, columns=cols_validas, filters=filtros if filtros else None)
    df = tabla.to_pandas()
    del tabla
    return df


def preparar_salida(df_slice: pd.DataFrame) -> pd.DataFrame:
    if df_slice.empty:
        return df_slice
    out = df_slice.copy()

    for col in out.select_dtypes(include=["category"]).columns:
        out[col] = out[col].astype(str)

    if "fecha" in out.columns and pd.api.types.is_integer_dtype(out["fecha"]):
        f_str = out["fecha"].astype(str)
        out["fecha"] = f_str.str.slice(0, 4) + "-" + f_str.str.slice(4, 6) + "-" + f_str.str.slice(6, 8)

    if "total_incendios" in out.columns:
        t = out["total_incendios"].to_numpy()
        out["nivel_riesgo"] = np.select([t > 50, t > 15], ["Alto", "Medio"], default="Bajo")
        out["color"] = np.select(
            [t > 75, t > 60, t > 50, t > 15],
            ["#FF0000", "#FF8000", "#FFFF00", "#80FF00"],
            default="#2ECC71",
        )

    for col in ["lat_grid", "lon_grid"]:
        if col in out.columns:
            out[col] = np.round(out[col].astype("float64"), 2)
    for col in ["frp_promedio", "brillo_promedio"]:
        if col in out.columns:
            out[col] = np.round(out[col].astype("float64"), 1)

    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def _iter_json_records(df_out: pd.DataFrame, chunk_size: int = 4000) -> Iterator[str]:
    """Serializa el DataFrame en bloques pequeños para evitar picos de RAM al enviar JSON."""
    if df_out.empty:
        yield "[]"
        return

    cols = list(df_out.columns)
    n_rows = len(df_out)
    yield "["
    first = True
    for start in range(0, n_rows, chunk_size):
        sub = df_out.iloc[start : start + chunk_size]
        arrays = [sub[c].tolist() for c in cols]
        chunk_json = ",".join(
            json.dumps(dict(zip(cols, row_vals)), ensure_ascii=False, separators=(",", ":"))
            for row_vals in zip(*arrays)
        )
        if first:
            yield chunk_json
            first = False
        else:
            yield "," + chunk_json
    yield "]"
    del df_out
    gc.collect()


def construir_respuesta(df_out: pd.DataFrame, request: Request, formato: Optional[str] = None) -> Response:
    accept_header = request.headers.get("accept", "")
    quiere_parquet = (formato and formato.lower() == "parquet") or ("application/vnd.apache.parquet" in accept_header)

    if quiere_parquet:
        buf = io.BytesIO()
        df_out.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
        contenido = buf.getvalue()
        del df_out
        gc.collect()
        return Response(
            content=contenido,
            media_type="application/vnd.apache.parquet",
            headers={"X-Data-Format": "parquet"},
        )

    return StreamingResponse(_iter_json_records(df_out), media_type="application/json")


# --- ENDPOINTS ---
@app.get("/")
def home():
    return {
        "status": "API LATAM/América Optimizada (Parquet Stream) activa 🚀",
        "paises_indexados": len(CATALOGO["paises"]),
        "paises": CATALOGO["paises"],
        "anios": CATALOGO["anios"],
        "modo_memoria": "disk-pushdown-512mb",
    }


@app.get("/api/filtros-disponibles")
def get_filtros_disponibles():
    return {
        "paises": CATALOGO["paises"],
        "anios": CATALOGO["anios"],
        "meses": list(range(1, 13)),
        "fecha_min": CATALOGO["fecha_min"],
        "fecha_max": CATALOGO["fecha_max"],
        "instrumentos": CATALOGO["instrumentos"],
        "satelites": CATALOGO["satelites"],
    }


@app.get("/api/incendios-grid")
def get_mapa_incendios(
    request: Request,
    pais: str = None,
    anio: int = None,
    mes: int = None,
    fecha: str = None,
    fecha_inicio: str = None,
    fecha_fin: str = None,
    instrumento: str = None,
    formato: str = None,
    limit: int = Query(default=45000, le=200000),
):
    usa_diario = bool(fecha or fecha_inicio or fecha_fin)
    tipo = "diario" if usa_diario else "mensual"

    if anio:
        anios_objetivo = [int(anio)]
    elif fecha:
        anios_objetivo = [int(fecha[:4])]
    elif fecha_inicio or fecha_fin:
        y_ini = int(fecha_inicio[:4]) if fecha_inicio else min(CATALOGO["anios"] or [2018])
        y_fin = int(fecha_fin[:4]) if fecha_fin else max(CATALOGO["anios"] or [2025])
        anios_objetivo = [y for y in CATALOGO["anios"] if y_ini <= y <= y_fin]
    else:
        anios_objetivo = list(CATALOGO["anios"])

    if not anios_objetivo:
        return construir_respuesta(pd.DataFrame(), request, formato)

    partes = []
    cuota_por_anio = max(1, limit // len(anios_objetivo)) if len(anios_objetivo) > 1 else limit

    for y in anios_objetivo:
        df_y = leer_parquet_filtrado(
            anio=y,
            tipo=tipo,
            pais=pais,
            mes=mes,
            fecha=fecha,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            instrumento=instrumento,
        )
        if df_y.empty:
            continue
        if len(anios_objetivo) > 1 and len(df_y) > cuota_por_anio:
            df_y = df_y.head(cuota_por_anio)
        partes.append(df_y)

    if not partes:
        return construir_respuesta(pd.DataFrame(), request, formato)

    df_out = pd.concat(partes, ignore_index=True) if len(partes) > 1 else partes[0]
    del partes

    if len(df_out) > limit:
        if (not pais or pais.lower() == "todos") and not anio and not mes:
            n_grp = max(1, df_out.groupby(["pais", "anio"], observed=True).ngroups)
            df_out = df_out.groupby(["pais", "anio"], group_keys=False, observed=True).head(max(1, limit // n_grp))
        elif not pais or pais.lower() == "todos":
            n_grp = max(1, df_out["pais"].nunique())
            df_out = df_out.groupby("pais", group_keys=False, observed=True).head(max(1, limit // n_grp))
        else:
            df_out = df_out.head(limit)

    df_final = preparar_salida(df_out)
    del df_out
    return construir_respuesta(df_final, request, formato)


@app.get("/api/incendios-por-pais")
def get_resumen_por_pais(anio: int = None):
    resumen = CATALOGO["resumen_paises"]
    if anio:
        return [r for r in resumen if int(r.get("anio", 0)) == int(anio)]
    return resumen


@app.get("/api/incendios-por-tiempo")
def get_resumen_temporal(
    request: Request,
    granularidad: str = Query(default="mes", pattern="^(anio|mes|fecha)$"),
    pais: str = None,
    anio: int = None,
    formato: str = None,
):
    tipo = "diario" if granularidad == "fecha" else "mensual"
    anios_objetivo = [int(anio)] if anio else list(CATALOGO["anios"])

    cols_grupo = ["anio"] if granularidad == "anio" else (["anio", "mes"] if granularidad == "mes" else ["fecha"])
    if pais and pais.lower() != "todos":
        cols_grupo = ["pais"] + cols_grupo

    cols_lectura = list(dict.fromkeys(cols_grupo + ["total_incendios", "frp_promedio", "brillo_promedio"]))
    agregados = []

    for y in anios_objetivo:
        df_t = leer_parquet_filtrado(anio=y, tipo=tipo, pais=pais, columnas=cols_lectura)
        if df_t.empty:
            continue
        res_y = (
            df_t.groupby(cols_grupo, observed=True)
            .agg(
                total_incendios=("total_incendios", "sum"),
                frp_promedio=("frp_promedio", "mean"),
                brillo_promedio=("brillo_promedio", "mean"),
            )
            .reset_index()
        )
        agregados.append(res_y)
        del df_t
        gc.collect()

    if not agregados:
        return construir_respuesta(pd.DataFrame(), request, formato)

    res = pd.concat(agregados, ignore_index=True).sort_values(cols_grupo).reset_index(drop=True)
    del agregados
    df_final = preparar_salida(res)
    del res
    return construir_respuesta(df_final, request, formato)


if __name__ == "__main__":
    puerto = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=puerto, workers=1, reload=False)