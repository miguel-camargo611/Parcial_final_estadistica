from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


DATA_PATH = Path("Data/Processed/analysis_env_complete.parquet")
OUTPUT_DIR = Path("Outputs")
FIG_DIR = Path("Figures/EDA")

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
CONTINUOUS_ENV_VARS = [col for col in ENV_VARS if col != "Lluvia"]
CORRELATION_ENV_VARS = CONTINUOUS_ENV_VARS
RAW_AUX_VARS = ["Precipitacion"]
ECO_VARS = [RESPONSE, "Riqueza_Especies"]
SPATIAL_VARS = ["Distance_to_Station_km"]
Z_ENV_VARS = [f"z_env_{col}" for col in ENV_VARS]
Z_CONTINUOUS_ENV_VARS = [f"z_env_{col}" for col in CONTINUOUS_ENV_VARS]


def safe_name(name):
    return (
        name.replace(".", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def save_descriptives(df):
    cols = ECO_VARS + SPATIAL_VARS + ENV_VARS + RAW_AUX_VARS
    desc = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["missing_n"] = df[cols].isna().sum()
    desc["missing_pct"] = (df[cols].isna().mean() * 100).round(2)
    desc["skew"] = df[cols].skew()
    desc["kurtosis"] = df[cols].kurtosis()
    desc.to_csv(OUTPUT_DIR / "eda_descriptive_env_complete.csv", encoding="utf-8")

    grouped = (
        df[df["Diversidad_Grupo"].isin(["Baja_Q1", "Alta_Q3"])]
        .groupby("Diversidad_Grupo")[cols]
        .agg(["count", "mean", "median", "std"])
    )
    grouped.to_csv(OUTPUT_DIR / "eda_q1_q3_group_summary.csv", encoding="utf-8")

    return desc, grouped


def save_correlations(df):
    corr_cols = ECO_VARS + SPATIAL_VARS + CORRELATION_ENV_VARS
    pearson = df[corr_cols].corr(method="pearson")
    spearman = df[corr_cols].corr(method="spearman")
    pearson.to_csv(OUTPUT_DIR / "eda_corr_pearson.csv", encoding="utf-8")
    spearman.to_csv(OUTPUT_DIR / "eda_corr_spearman.csv", encoding="utf-8")
    return pearson, spearman


def plot_shannon_distribution(df):
    q1 = df[RESPONSE].quantile(0.25)
    q3 = df[RESPONSE].quantile(0.75)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[RESPONSE], bins=40, kde=True, ax=ax, color="#376996")
    ax.axvline(q1, color="#c84630", linestyle="--", linewidth=1.8, label=f"Q1 = {q1:.2f}")
    ax.axvline(q3, color="#2a9d8f", linestyle="--", linewidth=1.8, label=f"Q3 = {q3:.2f}")
    ax.set_title("Distribución del índice de Shannon")
    ax.set_xlabel("Shannon_Index")
    ax.set_ylabel("Frecuencia")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_shannon_distribution.png", dpi=180)
    plt.close(fig)


def plot_environment_distributions(df):
    fig, axes = plt.subplots(5, 2, figsize=(12, 16))
    axes = axes.ravel()
    for ax, col in zip(axes, ENV_VARS):
        if col == "Lluvia":
            counts = (
                df[col]
                .map({0: "Sin lluvia", 1: "Con lluvia"})
                .value_counts()
                .reindex(["Sin lluvia", "Con lluvia"], fill_value=0)
            )
            sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#5f7f52")
            ax.set_ylim(0, counts.max() * 1.12)
            ax.bar_label(ax.containers[0], fmt="%.0f", fontsize=9)
        else:
            sns.histplot(df[col], bins=35, kde=True, ax=ax, color="#5f7f52")
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Frecuencia")
    fig.suptitle("Distribuciones de variables ambientales en escala original", y=1.0, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_environment_distributions_original.png", dpi=180)
    plt.close(fig)


def plot_correlation_heatmap(corr):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "Correlación de Pearson"},
        ax=ax,
    )
    ax.set_title("Matriz de correlación de Pearson")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_pearson_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def plot_shannon_scatter_grid(df):
    key_vars = ["PM2.5", "PM10", "NO2", "O3", "Temperatura", "Humedad", "Viento", "Radiacion_Solar"]
    sample = df.sample(min(len(df), 3500), random_state=42)

    fig, axes = plt.subplots(4, 2, figsize=(12, 15))
    axes = axes.ravel()
    for ax, col in zip(axes, key_vars):
        sns.regplot(
            data=sample,
            x=col,
            y=RESPONSE,
            lowess=True,
            scatter_kws={"s": 8, "alpha": 0.25, "color": "#3f5364"},
            line_kws={"color": "#c84630", "linewidth": 1.6},
            ax=ax,
        )
        rho, p_value = stats.spearmanr(df[col], df[RESPONSE], nan_policy="omit")
        ax.set_title(f"{col} vs Shannon (Spearman rho={rho:.2f})")
        ax.set_xlabel(col)
        ax.set_ylabel("Shannon_Index")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_shannon_vs_environment_scatter.png", dpi=180)
    plt.close(fig)


def plot_q1_q3_boxplots(df):
    groups = df[df["Diversidad_Grupo"].isin(["Baja_Q1", "Alta_Q3"])].copy()
    long = groups.melt(
        id_vars="Diversidad_Grupo",
        value_vars=Z_CONTINUOUS_ENV_VARS,
        var_name="Variable",
        value_name="Z",
    )
    long["Variable"] = long["Variable"].str.replace("z_env_", "", regex=False)

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(
        data=long,
        x="Variable",
        y="Z",
        hue="Diversidad_Grupo",
        palette={"Baja_Q1": "#c84630", "Alta_Q3": "#2a9d8f"},
        showfliers=False,
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("")
    ax.set_ylabel("Z-score")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_q1_q3_environment_boxplots_z.png", dpi=180)
    plt.close(fig)


def save_top_correlations(corr):
    top = (
        corr[RESPONSE]
        .drop(labels=[RESPONSE])
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .rename("pearson_with_shannon")
        .to_frame()
    )
    top.to_csv(OUTPUT_DIR / "eda_top_correlations_with_shannon.csv", encoding="utf-8")
    return top


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    print(f"EDA sobre {DATA_PATH}: {df.shape[0]} filas x {df.shape[1]} columnas")

    desc, grouped = save_descriptives(df)
    pearson, spearman = save_correlations(df)
    top_corr = save_top_correlations(pearson)

    plot_shannon_distribution(df)
    plot_environment_distributions(df)
    plot_correlation_heatmap(pearson)
    plot_shannon_scatter_grid(df)
    plot_q1_q3_boxplots(df)

    print("Descriptivas guardadas en Outputs/eda_descriptive_env_complete.csv")
    print("Resumen Q1/Q3 guardado en Outputs/eda_q1_q3_group_summary.csv")
    print("Correlaciones guardadas en Outputs/eda_corr_pearson.csv y eda_corr_spearman.csv")
    print("Figuras guardadas en Figures/EDA")
    print("\nCorrelaciones más fuertes con Shannon:")
    print(top_corr.head(10).round(3).to_string())


if __name__ == "__main__":
    main()
