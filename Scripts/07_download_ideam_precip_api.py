import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BASE_URL = "https://www.datos.gov.co/resource/s54a-sgyg.json"
RAW_DIR = Path("Data/Raw/precipitacion_ideam_api")
PROCESSED_DIR = Path("Data/Processed")
REPORT_DIR = Path("Outputs")

SELECT_COLUMNS = [
    "codigoestacion",
    "fechaobservacion",
    "valorobservado",
    "nombreestacion",
    "departamento",
    "municipio",
    "latitud",
    "longitud",
]

# Caja espacial amplia para Bogotá y borde urbano/periurbano.
LAT_MIN, LAT_MAX = 4.4, 4.85
LON_MIN, LON_MAX = -74.25, -73.95
YEARS = range(2020, 2026)
PAGE_LIMIT = 50000
SLEEP_SECONDS = 0.2
FORCE_REDOWNLOAD = False


def fetch_page(where, offset):
    params = {
        "$select": ",".join(SELECT_COLUMNS),
        "$where": where,
        "$order": "fechaobservacion,codigoestacion",
        "$limit": PAGE_LIMIT,
        "$offset": offset,
    }
    url = f"{BASE_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def download_period(start, end):
    where = (
        f"fechaobservacion >= '{start}' "
        f"AND fechaobservacion < '{end}' "
        f"AND latitud between {LAT_MIN} and {LAT_MAX} "
        f"AND longitud between {LON_MIN} and {LON_MAX}"
    )

    chunks = []
    offset = 0
    while True:
        rows = fetch_page(where, offset)
        if not rows:
            break

        chunks.append(pd.DataFrame(rows))
        print(f"{start} a {end}: descargadas {offset + len(rows):,} filas", flush=True)

        if len(rows) < PAGE_LIMIT:
            break

        offset += PAGE_LIMIT
        time.sleep(SLEEP_SECONDS)

    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame(columns=SELECT_COLUMNS)


def download_year(year):
    raw_path = RAW_DIR / f"ideam_precip_bogota_bbox_{year}_raw.parquet"
    clean_path = RAW_DIR / f"ideam_precip_bogota_bbox_{year}_clean.parquet"
    if not FORCE_REDOWNLOAD and raw_path.exists() and clean_path.exists():
        print(f"{year}: usando archivos existentes", flush=True)
        return pd.read_parquet(clean_path), True

    chunks = []
    for month in range(1, 13):
        month_path = RAW_DIR / f"ideam_precip_bogota_bbox_{year}_{month:02d}_raw.parquet"
        if not FORCE_REDOWNLOAD and month_path.exists():
            df_month = pd.read_parquet(month_path)
            print(f"{year}-{month:02d}: usando archivo mensual existente", flush=True)
        else:
            start = f"{year}-{month:02d}-01T00:00:00"
            if month == 12:
                end = f"{year + 1}-01-01T00:00:00"
            else:
                end = f"{year}-{month + 1:02d}-01T00:00:00"
            df_month = download_period(start, end)
            df_month.to_parquet(month_path, index=False)
        chunks.append(df_month)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=SELECT_COLUMNS)

    return df, False


def clean_precip(df):
    df = df.copy()
    df["fechaobservacion"] = pd.to_datetime(df["fechaobservacion"], errors="coerce")
    df["valorobservado"] = pd.to_numeric(df["valorobservado"], errors="coerce")
    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")

    df = df.dropna(subset=["fechaobservacion", "valorobservado", "latitud", "longitud"])
    df = df[df["valorobservado"] >= 0].copy()
    return df


def hourly_aggregate(df):
    df = df.copy()
    df["Fecha_Hora"] = df["fechaobservacion"].dt.floor("h")

    hourly = (
        df.groupby(
            [
                "codigoestacion",
                "nombreestacion",
                "departamento",
                "municipio",
                "latitud",
                "longitud",
                "Fecha_Hora",
            ],
            dropna=False,
        )
        .agg(
            Precipitacion_IDEAM_mm_h=("valorobservado", "sum"),
            Precipitacion_IDEAM_obs_10min=("valorobservado", "count"),
        )
        .reset_index()
    )
    return hourly


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    yearly_clean = []
    summary_rows = []

    for year in YEARS:
        raw_path = RAW_DIR / f"ideam_precip_bogota_bbox_{year}_raw.parquet"
        clean_path = RAW_DIR / f"ideam_precip_bogota_bbox_{year}_clean.parquet"
        downloaded, already_clean = download_year(year)

        if already_clean:
            clean = downloaded
            raw_rows = None
        else:
            raw = downloaded
            raw.to_parquet(raw_path, index=False)
            clean = clean_precip(raw)
            clean.to_parquet(clean_path, index=False)
            raw_rows = int(len(raw))

        yearly_clean.append(clean)

        summary_rows.append(
            {
                "year": year,
                "raw_rows": raw_rows,
                "clean_rows": int(len(clean)),
                "stations": int(clean["codigoestacion"].nunique()) if not clean.empty else 0,
                "min_fecha": clean["fechaobservacion"].min().isoformat() if not clean.empty else None,
                "max_fecha": clean["fechaobservacion"].max().isoformat() if not clean.empty else None,
            }
        )

    all_clean = pd.concat(yearly_clean, ignore_index=True) if yearly_clean else pd.DataFrame()
    all_path = RAW_DIR / "ideam_precip_bogota_bbox_2020_2025_clean.parquet"
    all_clean.to_parquet(all_path, index=False)

    hourly = hourly_aggregate(all_clean)
    hourly_path = PROCESSED_DIR / "ideam_precip_bogota_hourly.parquet"
    hourly.to_parquet(hourly_path, index=False)

    stations = (
        all_clean[
            [
                "codigoestacion",
                "nombreestacion",
                "departamento",
                "municipio",
                "latitud",
                "longitud",
            ]
        ]
        .drop_duplicates()
        .sort_values(["nombreestacion", "codigoestacion"])
    )
    stations.to_csv(REPORT_DIR / "ideam_precip_bogota_stations.csv", index=False, encoding="utf-8")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORT_DIR / "ideam_precip_api_download_summary.csv", index=False, encoding="utf-8")

    print("Descarga IDEAM completada.")
    print(summary.to_string(index=False))
    print(f"Total limpio: {len(all_clean):,} filas")
    print(f"Total horario: {len(hourly):,} filas")
    print(f"Estaciones: {all_clean['codigoestacion'].nunique() if not all_clean.empty else 0}")
    print(f"Guardado horario: {hourly_path}")


if __name__ == "__main__":
    main()
