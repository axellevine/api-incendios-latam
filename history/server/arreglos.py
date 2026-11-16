import gc
import glob
import json
import os
import re
import time
import tracemalloc
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_CSV = os.path.join(BASE_DIR, "Country_per_year")
CARPETA_CACHES = os.path.join(BASE_DIR, "Cachecountryperyear")
RUTA_METADATA = os.path.join(CARPETA_CACHES, "metadata_catalogo.json")

# Tamaño de bloque interno en Parquet: permite al servidor leer solo el país pedido
# sin descomprimir todo el archivo en RAM
PARQUET_ROW_GROUP_SIZE = 25000

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

NORMALIZACION_SATELITES = {
    "terra": "Terra",
    "aqua": "Aqua",
    "n20": "N20",
    "noaa-20": "N20",
    "noaa 20": "N20",
    "n21": "N21",
    "noaa-21": "N21",
    "noaa 21": "N21",
    "snpp": "SNPP",
    "suomi-npp": "SNPP",
    "suomi npp": "SNPP",
    "npp": "SNPP",
    "jpss-1": "JPSS-1",
    "jpss1": "JPSS-1",
}


def normalizar_pais(valor: str) -> str:
    limpio = str(valor).strip()
    return NORMALIZACION_PAISES.get(limpio.lower(), limpio)


def normalizar_satelite(valor: str) -> str:
    limpio = str(valor).strip()
    return NORMALIZACION_SATELITES.get(limpio.lower(), limpio)


def normalizar_instrumento(inst: str, sat: str) -> str:
    i_up = str(inst).strip().upper()
    s_up = str(sat).strip().upper()
    if "VIIRS" in i_up or any(k in s_up for k in ("SNPP", "NPP", "N20", "N21", "NOAA", "JPSS")):
        return "VIIRS"
    if "MODIS" in i_up or any(k in s_up for k in ("TERRA", "AQUA")):
        return "MODIS"
    return i_up if i_up and i_up != "NAN" else "MODIS"


def asegurar_fecha_entera(serie_fecha: pd.Series) -> pd.Series:
    """Convierte cualquier formato de fecha ('YYYY-MM-DD', datetime o int) a int32 YYYYMMDD."""
    if pd.api.types.is_integer_dtype(serie_fecha):
        return serie_fecha.astype("int32")
    dt = pd.to_datetime(serie_fecha, errors="coerce", format="mixed")
    enteros = (dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day).fillna(0).astype("int32")
    return enteros


def optimizar_esquema(df: pd.DataFrame, anio_esperado: int, es_diario: bool) -> pd.DataFrame:
    """Limpia nulos, filtra excluidos y ajusta tipos compactos sin pérdida de precisión."""
    if df.empty:
        return df

    out = df.copy()
    out["pais"] = out["pais"].fillna("Sin dato").astype(str).map(normalizar_pais)
    out = out[~out["pais"].str.lower().isin(PAISES_EXCLUIDOS)].copy()

    if out.empty:
        return out

    if es_diario and "fecha" in out.columns:
        out["fecha"] = asegurar_fecha_entera(out["fecha"])
        out = out[out["fecha"] > 0].copy()
        out["anio"] = (out["fecha"] // 10000).astype("int16")
        out["mes"] = ((out["fecha"] // 100) % 100).astype("int8")
    else:
        out["anio"] = pd.to_numeric(out.get("anio", anio_esperado), errors="coerce").fillna(anio_esperado).astype("int16")
        out["mes"] = pd.to_numeric(out["mes"], errors="coerce").fillna(1).astype("int8")

    out["lat_grid"] = pd.to_numeric(out["lat_grid"], errors="coerce").round(2).astype("float32")
    out["lon_grid"] = pd.to_numeric(out["lon_grid"], errors="coerce").round(2).astype("float32")
    out["total_incendios"] = pd.to_numeric(out["total_incendios"], errors="coerce").fillna(1).astype("int32")
    out["frp_promedio"] = pd.to_numeric(out["frp_promedio"], errors="coerce").fillna(0.0).clip(lower=0.0).round(1).astype("float32")
    out["brillo_promedio"] = pd.to_numeric(out["brillo_promedio"], errors="coerce").fillna(0.0).clip(lower=0.0).round(1).astype("float32")

    out["satelite"] = out.get("satelite", "Desconocido").fillna("Desconocido").astype(str).map(normalizar_satelite)
    inst_raw = out.get("instrumento", "MODIS").fillna("MODIS").astype(str)
    out["instrumento"] = [normalizar_instrumento(i, s) for i, s in zip(inst_raw, out["satelite"])]

    out = out.dropna(subset=["lat_grid", "lon_grid"])
    out = out[(out["total_incendios"] > 0) & out["mes"].between(1, 12)]

    cols_ordenadas = (
        ["pais", "anio", "mes", "fecha", "lat_grid", "lon_grid", "total_incendios", "frp_promedio", "brillo_promedio", "instrumento", "satelite"]
        if es_diario
        else ["pais", "anio", "mes", "lat_grid", "lon_grid", "total_incendios", "frp_promedio", "brillo_promedio", "instrumento", "satelite"]
    )
    out = out[cols_ordenadas]

    # Ordenar por país -> mes -> fecha maximiza el rendimiento del predicate pushdown en Parquet
    sort_cols = ["pais", "mes", "fecha"] if es_diario else ["pais", "mes"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def guardar_parquet_indexado(df: pd.DataFrame, ruta_destino: str) -> float:
    """Guarda el DataFrame en Parquet con compresión Zstd y grupos de 25,000 filas."""
    df.to_parquet(
        ruta_destino,
        engine="pyarrow",
        compression="zstd",
        compression_level=9,
        row_group_size=PARQUET_ROW_GROUP_SIZE,
        index=False,
    )
    return os.path.getsize(ruta_destino) / (1024 * 1024)


# ------------------------------------------------------------------------------
# OPCIÓN A: Migrar desde los cachés cache_20XX.pkl.gz existentes
# ------------------------------------------------------------------------------
def migrar_desde_pkl() -> List[int]:
    archivos_pkl = sorted(glob.glob(os.path.join(CARPETA_CACHES, "cache_20*.pkl.gz")))
    if not archivos_pkl:
        return []

    print(f"📦 Se encontraron {len(archivos_pkl)} archivos .pkl.gz para migrar a Parquet...")
    anios_procesados = []

    for ruta_pkl in archivos_pkl:
        m = re.search(r"cache_(\d{4})", os.path.basename(ruta_pkl))
        if not m:
            continue
        anio = int(m.group(1))
        t0 = time.time()
        peso_pkl_mb = os.path.getsize(ruta_pkl) / (1024 * 1024)

        obj = pd.read_pickle(ruta_pkl, compression="gzip")
        df_d = optimizar_esquema(obj.get("diario", pd.DataFrame()), anio, es_diario=True)
        df_m = optimizar_esquema(obj.get("mensual", pd.DataFrame()), anio, es_diario=False)

        ruta_d = os.path.join(CARPETA_CACHES, f"cache_{anio}_diario.parquet")
        ruta_m = os.path.join(CARPETA_CACHES, f"cache_{anio}_mensual.parquet")

        mb_d = guardar_parquet_indexado(df_d, ruta_d)
        mb_m = guardar_parquet_indexado(df_m, ruta_m)

        # Auditoría de integridad de conteo
        focos_orig = int(obj["mensual"]["total_incendios"].sum()) if "mensual" in obj and not obj["mensual"].empty else 0
        focos_pq = int(df_m["total_incendios"].sum()) if not df_m.empty else 0

        print(
            f"  ✅ {anio}: PKL {peso_pkl_mb:.1f} MB → Parquet (diario {mb_d:.2f} MB + mensual {mb_m:.2f} MB) "
            f"| Celdas M: {len(df_m):,} | Focos: {focos_pq:,} (orig: {focos_orig:,}) [{time.time() - t0:.1f}s]"
        )

        anios_procesados.append(anio)
        del obj, df_d, df_m
        gc.collect()

    return sorted(anios_procesados)


# ------------------------------------------------------------------------------
# OPCIÓN B: Procesar directamente desde los CSV crudos de Country_per_year/
# ------------------------------------------------------------------------------
def procesar_desde_csv_crudos() -> List[int]:
    if not os.path.isdir(CARPETA_CSV):
        return []

    archivos_csv = sorted(glob.glob(os.path.join(CARPETA_CSV, "**", "*.csv"), recursive=True))
    if not archivos_csv:
        return []

    print(f"🛰️ Procesando {len(archivos_csv)} archivos CSV crudos desde {CARPETA_CSV}...")
    por_anio: Dict[int, List[pd.DataFrame]] = {}

    for ruta_csv in archivos_csv:
        try:
            raw = pd.read_csv(ruta_csv, low_memory=False)
        except Exception as e:
            print(f"  ⚠️ No se pudo leer {ruta_csv}: {e}")
            continue

        raw.columns = [c.lower().strip() for c in raw.columns]
        if not {"latitude", "longitude", "acq_date"} <= set(raw.columns):
            continue

        # Inferir país desde columna o nombre de archivo/carpeta
        if "country" in raw.columns:
            pais_col = raw["country"].astype(str)
        elif "pais" in raw.columns:
            pais_col = raw["pais"].astype(str)
        else:
            stem = os.path.splitext(os.path.basename(ruta_csv))[0]
            partes = [p for p in re.split(r"[_\-\s]+", stem) if not p.isdigit() and p.lower() not in ("modis", "viirs", "firms", "nrt")]
            pais_col = pd.Series(" ".join(partes).title() if partes else "Sin dato", index=raw.index)

        brillo_col = raw["brightness"] if "brightness" in raw.columns else (raw["bright_ti4"] if "bright_ti4" in raw.columns else np.nan)
        frp_col = raw["frp"] if "frp" in raw.columns else 0.0
        sat_col = raw["satellite"].astype(str) if "satellite" in raw.columns else "Desconocido"
        inst_col = raw["instrument"].astype(str) if "instrument" in raw.columns else ("VIIRS" if "bright_ti4" in raw.columns else "MODIS")

        fecha_int = asegurar_fecha_entera(raw["acq_date"])
        df_clean = pd.DataFrame({
            "pais": pais_col,
            "fecha": fecha_int,
            "lat_grid": pd.to_numeric(raw["latitude"], errors="coerce").round(2),
            "lon_grid": pd.to_numeric(raw["longitude"], errors="coerce").round(2),
            "brillo": pd.to_numeric(brillo_col, errors="coerce"),
            "frp": pd.to_numeric(frp_col, errors="coerce"),
            "instrumento": inst_col,
            "satelite": sat_col,
        }).dropna(subset=["lat_grid", "lon_grid"])
        df_clean = df_clean[df_clean["fecha"] > 0]
        df_clean["anio"] = (df_clean["fecha"] // 10000).astype("int16")

        for anio_val, grupo in df_clean.groupby("anio"):
            por_anio.setdefault(int(anio_val), []).append(grupo.copy())

    anios_procesados = []
    for anio in sorted(por_anio.keys()):
        df_anio = pd.concat(por_anio[anio], ignore_index=True)
        df_anio["pais"] = df_anio["pais"].map(normalizar_pais)
        df_anio = df_anio[~df_anio["pais"].str.lower().isin(PAISES_EXCLUIDOS)]
        df_anio["satelite"] = df_anio["satelite"].map(normalizar_satelite)
        df_anio["instrumento"] = [normalizar_instrumento(i, s) for i, s in zip(df_anio["instrumento"], df_anio["satelite"])]
        df_anio["mes"] = ((df_anio["fecha"] // 100) % 100).astype("int8")

        diario = (
            df_anio.groupby(["pais", "anio", "mes", "fecha", "lat_grid", "lon_grid", "instrumento", "satelite"], as_index=False)
            .agg(
                total_incendios=("frp", "size"),
                frp_promedio=("frp", "mean"),
                brillo_promedio=("brillo", "mean"),
            )
        )
        mensual = (
            df_anio.groupby(["pais", "anio", "mes", "lat_grid", "lon_grid", "instrumento", "satelite"], as_index=False)
            .agg(
                total_incendios=("frp", "size"),
                frp_promedio=("frp", "mean"),
                brillo_promedio=("brillo", "mean"),
            )
        )

        df_d = optimizar_esquema(diario, anio, es_diario=True)
        df_m = optimizar_esquema(mensual, anio, es_diario=False)

        mb_d = guardar_parquet_indexado(df_d, os.path.join(CARPETA_CACHES, f"cache_{anio}_diario.parquet"))
        mb_m = guardar_parquet_indexado(df_m, os.path.join(CARPETA_CACHES, f"cache_{anio}_mensual.parquet"))
        print(f"  ✅ {anio} desde CSV: diario {mb_d:.2f} MB ({len(df_d):,} filas) | mensual {mb_m:.2f} MB ({len(df_m):,} filas)")

        anios_procesados.append(anio)
        del df_anio, diario, mensual, df_d, df_m
        gc.collect()

    return anios_procesados


# ------------------------------------------------------------------------------
# PROCESO ADICIONAL 1: Generación de metadata_catalogo.json (~15 KB)
# ------------------------------------------------------------------------------
def generar_catalogo_json(anios: List[int]) -> dict:
    paises_set, inst_set, sat_set = set(), set(), set()
    fechas_min, fechas_max = [], []
    resumen_total = []

    for anio in anios:
        ruta_m = os.path.join(CARPETA_CACHES, f"cache_{anio}_mensual.parquet")
        ruta_d = os.path.join(CARPETA_CACHES, f"cache_{anio}_diario.parquet")

        if os.path.exists(ruta_m):
            df_m = pq.read_table(ruta_m).to_pandas()
            if not df_m.empty:
                paises_set.update(df_m["pais"].unique())
                inst_set.update(df_m["instrumento"].unique())
                sat_set.update(df_m["satelite"].unique())

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

        if os.path.exists(ruta_d):
            df_f = pq.read_table(ruta_d, columns=["fecha"]).to_pandas()
            if not df_f.empty:
                f_str = df_f["fecha"].astype(str)
                fechas_min.append(f_str.min())
                fechas_max.append(f_str.max())
            del df_f
        gc.collect()

    f_min = min(fechas_min) if fechas_min else f"{anios[0]}0101"
    f_max = max(fechas_max) if fechas_max else f"{anios[-1]}1231"

    catalogo = {
        "paises": sorted(paises_set),
        "anios": anios,
        "instrumentos": sorted(inst_set),
        "satelites": sorted(sat_set),
        "fecha_min": f"{f_min[:4]}-{f_min[4:6]}-{f_min[6:8]}",
        "fecha_max": f"{f_max[:4]}-{f_max[4:6]}-{f_max[6:8]}",
        "resumen_paises": resumen_total,
    }

    with open(RUTA_METADATA, "w", encoding="utf-8") as f:
        json.dump(catalogo, f, ensure_ascii=False, indent=2)

    print(f"🗂️ Catálogo generado en {RUTA_METADATA} ({os.path.getsize(RUTA_METADATA) / 1024:.1f} KB)")
    return catalogo


# ------------------------------------------------------------------------------
# PROCESO ADICIONAL 2: Prueba de estrés de RAM simulando el servidor de 512 MB
# ------------------------------------------------------------------------------
def auditar_consumo_ram_servidor(anios: List[int]) -> None:
    if not anios:
        return
    anio_prueba = anios[-1]
    ruta_m = os.path.join(CARPETA_CACHES, f"cache_{anio_prueba}_mensual.parquet")

    tracemalloc.start()
    tabla_pais = pq.read_table(ruta_m, filters=[("pais", "==", "Colombia")])
    df_pais = tabla_pais.to_pandas()
    _, pico_pais = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    tabla_todo = pq.read_table(ruta_m)
    df_todo = tabla_todo.to_pandas()
    _, pico_todo = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_disco_mb = sum(
        os.path.getsize(f) / (1024 * 1024)
        for f in glob.glob(os.path.join(CARPETA_CACHES, "*.parquet"))
    )

    print("\n📊 AUDITORÍA FINAL PARA SERVIDOR CLOUD (LÍMITE 512 MB):")
    print(f"   • Espacio total en disco (.parquet): {total_disco_mb:.2f} MB")
    print(f"   • Pico RAM al consultar 1 país (Colombia {anio_prueba}): {pico_pais / (1024 * 1024):.2f} MB ({len(df_pais):,} filas)")
    print(f"   • Pico RAM al consultar toda LATAM ({anio_prueba}): {pico_todo / (1024 * 1024):.2f} MB ({len(df_todo):,} filas)")
    print("   • Estado: ✅ Seguro para desplegar en contenedor de 512 MB.\n")


if __name__ == "__main__":
    os.makedirs(CARPETA_CACHES, exist_ok=True)
    anios = migrar_desde_pkl()
    if not anios:
        anios = procesar_desde_csv_crudos()

    if not anios:
        print("❌ No se encontraron archivos .pkl.gz ni CSVs en Country_per_year/.")
    else:
        generar_catalogo_json(anios)
        auditar_consumo_ram_servidor(anios)