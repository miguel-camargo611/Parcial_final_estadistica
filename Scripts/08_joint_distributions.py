from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.stats import chi2, multivariate_normal


DATA_PATH = Path("Data/Processed/analysis_env_complete.parquet")
OUTPUT_DIR = Path("Outputs")
FIG_DIR = Path("Figures/Joint_Distributions")

RESPONSE = "Shannon_Index"
BIVAR_RAW = ["Temperatura", "Humedad"]
BIVAR_Z = [f"z_env_{col}" for col in BIVAR_RAW]
POLLUTION_Z = ["z_env_PM2.5", "z_env_PM10", "z_env_NO2", "z_env_CO", "z_env_O3"]


def add_covariance_ellipse(ax, mean, cov, probability, **kwargs):
    radius = np.sqrt(chi2.ppf(probability, df=2))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * radius * np.sqrt(eigvals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, fill=False, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def prepare_data():
    df = pd.read_parquet(DATA_PATH).copy()
    df["Indice_Contaminacion_Z"] = df[POLLUTION_Z].mean(axis=1)
    q1 = df["Indice_Contaminacion_Z"].quantile(0.25)
    q3 = df["Indice_Contaminacion_Z"].quantile(0.75)
    df["Contaminacion_Grupo"] = np.select(
        [df["Indice_Contaminacion_Z"] <= q1, df["Indice_Contaminacion_Z"] >= q3],
        ["Baja_Q1", "Alta_Q3"],
        default="Media",
    )
    return df, q1, q3


def save_summaries(df, pollution_q1, pollution_q3):
    bivar = df[BIVAR_Z].agg(["count", "mean", "std", "min", "max"]).T
    bivar["raw_variable"] = BIVAR_RAW
    bivar.to_csv(OUTPUT_DIR / "joint_bivariate_summary.csv", encoding="utf-8")

    conditional = (
        df[df["Contaminacion_Grupo"].isin(["Baja_Q1", "Alta_Q3"])]
        .groupby("Contaminacion_Grupo")[[RESPONSE, "Indice_Contaminacion_Z"]]
        .agg(["count", "mean", "median", "std", "min", "max"])
    )
    conditional.to_csv(OUTPUT_DIR / "joint_conditional_pollution_summary.csv", encoding="utf-8")

    thresholds = pd.DataFrame(
        {
            "threshold": ["pollution_q1", "pollution_q3"],
            "value": [pollution_q1, pollution_q3],
        }
    )
    thresholds.to_csv(OUTPUT_DIR / "joint_pollution_thresholds.csv", index=False, encoding="utf-8")

    return bivar, conditional


def plot_kde_2d(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(
        data=df,
        x=BIVAR_Z[0],
        y=BIVAR_Z[1],
        fill=True,
        levels=12,
        thresh=0.03,
        cmap="YlGnBu",
        ax=ax,
    )
    sns.scatterplot(
        data=df.sample(min(len(df), 1500), random_state=42),
        x=BIVAR_Z[0],
        y=BIVAR_Z[1],
        s=10,
        alpha=0.18,
        color="#334155",
        edgecolor=None,
        ax=ax,
    )
    ax.set_title("KDE 2D: Temperatura y Humedad")
    ax.set_xlabel("Temperatura estandarizada")
    ax.set_ylabel("Humedad estandarizada")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_kde2d_temperatura_humedad.png", dpi=180)
    plt.close(fig)


def plot_density_ellipses(df):
    x = df[BIVAR_Z].to_numpy()
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df.sample(min(len(df), 1800), random_state=42),
        x=BIVAR_Z[0],
        y=BIVAR_Z[1],
        hue="Diversidad_Grupo",
        palette={"Baja_Q1": "#c84630", "Media_Q2_Q3": "#64748b", "Alta_Q3": "#2a9d8f"},
        s=12,
        alpha=0.28,
        linewidth=0,
        ax=ax,
    )
    for probability, color in [(0.50, "#1d4ed8"), (0.80, "#7c3aed"), (0.95, "#111827")]:
        add_covariance_ellipse(ax, mean, cov, probability, color=color, linewidth=2, label=f"{probability:.0%}")
    ax.set_title("Elipses de densidad empírica")
    ax.set_xlabel("Temperatura estandarizada")
    ax.set_ylabel("Humedad estandarizada")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_elipses_densidad_temperatura_humedad.png", dpi=180)
    plt.close(fig)


def plot_bivariate_normal(df):
    x = df[BIVAR_Z].to_numpy()
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)

    x_min, x_max = np.percentile(x[:, 0], [0.5, 99.5])
    y_min, y_max = np.percentile(x[:, 1], [0.5, 99.5])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 140), np.linspace(y_min, y_max, 140))
    grid = np.dstack((xx, yy))
    density = multivariate_normal(mean=mean, cov=cov).pdf(grid)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(
        data=df,
        x=BIVAR_Z[0],
        y=BIVAR_Z[1],
        levels=8,
        color="#c84630",
        linewidths=1.5,
        ax=ax,
        label="KDE observada",
    )
    ax.contour(xx, yy, density, levels=8, colors="#2563eb", linewidths=1.3)
    ax.set_title("Normal bivariada ajustada vs. densidad observada")
    ax.set_xlabel("Temperatura estandarizada")
    ax.set_ylabel("Humedad estandarizada")
    ax.plot([], [], color="#c84630", label="KDE observada")
    ax.plot([], [], color="#2563eb", label="Normal bivariada")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_normal_bivariada_temperatura_humedad.png", dpi=180)
    plt.close(fig)


def plot_conditional_shannon(df):
    conditional = df[df["Contaminacion_Grupo"].isin(["Baja_Q1", "Alta_Q3"])].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(
        data=conditional,
        x=RESPONSE,
        hue="Contaminacion_Grupo",
        hue_order=["Baja_Q1", "Alta_Q3"],
        palette={"Baja_Q1": "#2a9d8f", "Alta_Q3": "#c84630"},
        fill=True,
        common_norm=False,
        alpha=0.35,
        linewidth=2,
        ax=ax,
    )
    ax.set_title("Distribución condicional de Shannon por contaminación")
    ax.set_xlabel("Shannon_Index")
    ax.set_ylabel("Densidad")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_shannon_condicional_contaminacion.png", dpi=180)
    plt.close(fig)


def plot_scatter_shannon_gradient(df):
    """Scatter Temperatura vs Humedad coloreado por el valor de Shannon (gradiente continuo)."""
    sample = df.sample(min(len(df), 3000), random_state=42).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        sample[BIVAR_Z[0]],
        sample[BIVAR_Z[1]],
        c=sample[RESPONSE],
        cmap="RdYlGn",
        s=12,
        alpha=0.55,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Índice de Shannon ($H'$)", fontsize=10)
    ax.set_xlabel("Temperatura estandarizada", fontsize=11)
    ax.set_ylabel("Humedad estandarizada", fontsize=11)
    ax.set_title("Dispersión Temperatura-Humedad coloreada por Biodiversidad (Shannon)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_scatter_shannon_gradient.png", dpi=180)
    plt.close(fig)


def plot_pairplot_marginals(df):
    """Pairplot de variables clave con KDE en diagonal para mostrar distribuciones marginales."""
    VARS_PAIR = ["z_env_PM2.5", "z_env_Temperatura", "z_env_Humedad", "z_env_O3", RESPONSE]
    LABELS = {
        "z_env_PM2.5": "PM2.5 (Z)",
        "z_env_Temperatura": "Temp (Z)",
        "z_env_Humedad": "Humedad (Z)",
        "z_env_O3": "O3 (Z)",
        RESPONSE: "Shannon H'",
    }
    sample = df[VARS_PAIR].sample(min(len(df), 2500), random_state=42).rename(columns=LABELS)
    pg = sns.pairplot(
        sample,
        diag_kind="kde",
        plot_kws={"alpha": 0.18, "s": 10, "color": "#2563eb"},
        diag_kws={"color": "#2563eb", "fill": True, "alpha": 0.5},
    )
    pg.figure.suptitle(
        "Distribuciones Marginales (diagonal) y Bivariadas (off-diagonal)",
        y=1.02, fontsize=12
    )
    pg.figure.tight_layout()
    pg.figure.savefig(FIG_DIR / "06_pairplot_marginales.png", dpi=150, bbox_inches="tight")
    plt.close(pg.figure)




def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df, pollution_q1, pollution_q3 = prepare_data()
    bivar, conditional = save_summaries(df, pollution_q1, pollution_q3)

    plot_kde_2d(df)
    plot_density_ellipses(df)
    plot_bivariate_normal(df)
    plot_conditional_shannon(df)
    plot_scatter_shannon_gradient(df)
    plot_pairplot_marginals(df)

    print(f"Fase 3 ejecutada sobre {DATA_PATH}: {df.shape[0]} filas")
    print("\nPar bivariado principal:")
    print(bivar.round(3).to_string())
    print("\nUmbrales del índice de contaminación:")
    print(f"Q1 = {pollution_q1:.3f}, Q3 = {pollution_q3:.3f}")
    print("\nResumen condicional:")
    print(conditional.round(3).to_string())
    print(f"\nFiguras guardadas en {FIG_DIR}")


if __name__ == "__main__":
    main()
