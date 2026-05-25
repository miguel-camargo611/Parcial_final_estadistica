from pathlib import Path

import pandas as pd
from scipy import stats


INPUT_PATH = Path("Data/Processed/analysis_env_complete.parquet")
OUTPUT_PATH = Path("Outputs/distribution_diagnostics_env_complete.csv")

RAW_TO_AUX_LOG = {
    "PM2.5": "log1p_PM2_5",
    "PM10": "log1p_PM10",
    "NO2": "log1p_NO2",
    "CO": "log1p_CO",
    "O3": "log1p_O3",
    "Temperatura": "Temperatura",
    "Humedad": "Humedad",
    "Viento": "Viento",
    "Lluvia": None,
    "Radiacion_Solar": "log1p_Radiacion_Solar",
}


def normality_row(df, variable, scale, col):
    x = df[col].dropna()
    shapiro_x = x.sample(5000, random_state=42) if len(x) > 5000 else x
    shapiro_stat, shapiro_p = stats.shapiro(shapiro_x)
    normaltest_stat, normaltest_p = stats.normaltest(x)
    return {
        "variable": variable,
        "scale": scale,
        "column": col,
        "n": len(x),
        "mean": x.mean(),
        "std": x.std(ddof=0),
        "skew": stats.skew(x, bias=False),
        "kurtosis_excess": stats.kurtosis(x, fisher=True, bias=False),
        "shapiro_p_sample_max_5000": shapiro_p,
        "dagostino_p": normaltest_p,
    }


def main():
    df = pd.read_parquet(INPUT_PATH)
    rows = []

    for raw_col, log_col in RAW_TO_AUX_LOG.items():
        rows.append(normality_row(df, raw_col, "raw_primary", raw_col))
        rows.append(normality_row(df, raw_col, "z_env_raw_primary", f"z_env_{raw_col}"))
        if log_col and log_col in df.columns:
            rows.append(normality_row(df, raw_col, "log1p_auxiliary", log_col))

    if "Precipitacion" in df.columns:
        rows.append(normality_row(df, "Precipitacion", "raw_auxiliary", "Precipitacion"))
        if "log1p_Precipitacion" in df.columns:
            rows.append(normality_row(df, "Precipitacion", "log1p_auxiliary", "log1p_Precipitacion"))

    diagnostics = pd.DataFrame(rows)
    diagnostics["normal_by_dagostino_0_05"] = diagnostics["dagostino_p"] >= 0.05
    diagnostics["normal_by_shapiro_0_05"] = diagnostics["shapiro_p_sample_max_5000"] >= 0.05
    diagnostics.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Diagnóstico guardado en {OUTPUT_PATH}")
    print(diagnostics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
