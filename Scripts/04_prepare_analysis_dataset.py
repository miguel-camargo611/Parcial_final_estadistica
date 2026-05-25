import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


MASTER_PATH = Path("Data/Processed/master_dataset.parquet")
OUTPUT_DIR = Path("Data/Processed")
REPORT_DIR = Path("Outputs")

RESPONSE = "Shannon_Index"
ENV_VARS = [
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "O3",
    "Temperatura",
    "Humedad",
    "Viento",
    "Lluvia",
    "Radiacion_Solar",
]
RAW_ENV_VARS = [
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "O3",
    "Temperatura",
    "Humedad",
    "Viento",
    "Precipitacion",
    "Radiacion_Solar",
]
EFFORT_VARS = ["DURATION MINUTES", "EFFORT DISTANCE KM"]
PHYSICALLY_NONNEGATIVE = [
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "O3",
    "Viento",
    "Precipitacion",
    "Radiacion_Solar",
    "DURATION MINUTES",
    "EFFORT DISTANCE KM",
]
ALWAYS_LOG1P = [
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "O3",
    "Precipitacion",
    "Radiacion_Solar",
    "DURATION MINUTES",
    "EFFORT DISTANCE KM",
]


def safe_col_name(name):
    return (
        name.replace(".", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(MASTER_PATH)
    initial_rows = len(df)

    df = df.dropna(subset=["Fecha_Hora", RESPONSE]).copy()
    df["Fecha_Hora"] = pd.to_datetime(df["Fecha_Hora"], errors="coerce")
    df["Anio"] = df["Fecha_Hora"].dt.year
    df["Mes"] = df["Fecha_Hora"].dt.month
    df["Hora"] = df["Fecha_Hora"].dt.hour

    negative_replacements = {}
    for col in PHYSICALLY_NONNEGATIVE:
        if col in df.columns:
            mask = df[col] < 0
            negative_replacements[col] = int(mask.sum())
            df.loc[mask, col] = np.nan

    for col in ALWAYS_LOG1P:
        if col in df.columns:
            df[f"log1p_{safe_col_name(col)}"] = np.log1p(df[col])

    if "Precipitacion" in df.columns:
        df["Lluvia"] = np.where(
            df["Precipitacion"].isna(),
            np.nan,
            (df["Precipitacion"] > 0).astype(int),
        )

    q1 = df[RESPONSE].quantile(0.25)
    q3 = df[RESPONSE].quantile(0.75)
    df["Diversidad_Grupo"] = np.select(
        [df[RESPONSE] <= q1, df[RESPONSE] >= q3],
        ["Baja_Q1", "Alta_Q3"],
        default="Media_Q2_Q3",
    )

    # Matriz principal: variables originales estandarizadas. Las columnas log1p
    # se conservan como diagnóstico auxiliar, pero no son la escala base.
    model_env_vars = ENV_VARS.copy()
    model_effort_vars = EFFORT_VARS.copy()
    env_complete_cols = [col for col in model_env_vars if col in df.columns]
    reg_complete_cols = env_complete_cols + [col for col in model_effort_vars if col in df.columns]

    env_scaler = StandardScaler()
    valid_env_scaler = df[env_complete_cols].notna().all(axis=1)
    z_env_values = pd.DataFrame(
        index=df.index,
        columns=[f"z_env_{col}" for col in env_complete_cols],
        dtype=float,
    )
    z_env_values.loc[valid_env_scaler, :] = env_scaler.fit_transform(
        df.loc[valid_env_scaler, env_complete_cols]
    )

    reg_scaler = StandardScaler()
    valid_reg_scaler = df[reg_complete_cols].notna().all(axis=1)
    z_reg_values = pd.DataFrame(
        index=df.index,
        columns=[f"z_reg_{col}" for col in reg_complete_cols],
        dtype=float,
    )
    z_reg_values.loc[valid_reg_scaler, :] = reg_scaler.fit_transform(
        df.loc[valid_reg_scaler, reg_complete_cols]
    )

    df = pd.concat([df, z_env_values, z_reg_values], axis=1)

    z_cols = list(z_env_values.columns) + list(z_reg_values.columns)
    outlier_counts = {}
    binary_z_cols = {"z_env_Lluvia", "z_reg_Lluvia"}
    for col in z_cols:
        flag_col = f"outlier_{col}"
        if col in binary_z_cols:
            df[flag_col] = False
        else:
            df[flag_col] = df[col].abs() > 3
        outlier_counts[col] = int(df[flag_col].sum())

    analysis_full = df
    analysis_env_complete = df.dropna(subset=env_complete_cols + [RESPONSE]).copy()
    analysis_regression_complete = df.dropna(subset=reg_complete_cols + [RESPONSE]).copy()

    no_radiation_env = [col for col in env_complete_cols if col != "Radiacion_Solar"]
    no_radiation_reg = [col for col in reg_complete_cols if col != "Radiacion_Solar"]
    analysis_env_reduced = df.dropna(subset=no_radiation_env + [RESPONSE]).copy()
    analysis_regression_reduced = df.dropna(subset=no_radiation_reg + [RESPONSE]).copy()

    analysis_full.to_parquet(OUTPUT_DIR / "analysis_dataset.parquet", index=False)
    analysis_env_complete.to_parquet(OUTPUT_DIR / "analysis_env_complete.parquet", index=False)
    analysis_regression_complete.to_parquet(OUTPUT_DIR / "analysis_regression_complete.parquet", index=False)
    analysis_env_reduced.to_parquet(OUTPUT_DIR / "analysis_env_reduced_no_radiation.parquet", index=False)
    analysis_regression_reduced.to_parquet(OUTPUT_DIR / "analysis_regression_reduced_no_radiation.parquet", index=False)

    summary = {
        "initial_master_rows": initial_rows,
        "analysis_rows": int(len(analysis_full)),
        "shannon_q1": float(q1),
        "shannon_q3": float(q3),
        "negative_values_set_to_nan": negative_replacements,
        "env_complete_rows_10_vars": int(len(analysis_env_complete)),
        "regression_complete_rows_10_vars": int(len(analysis_regression_complete)),
        "env_reduced_rows_no_radiation": int(len(analysis_env_reduced)),
        "regression_reduced_rows_no_radiation": int(len(analysis_regression_reduced)),
        "outlier_counts_abs_z_gt_3": outlier_counts,
        "missing_pct": (analysis_full[ENV_VARS + EFFORT_VARS + [RESPONSE]].isna().mean() * 100)
        .round(2)
        .to_dict(),
    }

    with open(REPORT_DIR / "analysis_preparation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    descriptive_cols = RAW_ENV_VARS + ["Lluvia"] + EFFORT_VARS + [RESPONSE, "Riqueza_Especies"]
    descriptive = analysis_full[descriptive_cols].describe().T
    descriptive.to_csv(REPORT_DIR / "analysis_descriptive_summary.csv", encoding="utf-8")

    print("Preparación analítica completada.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
