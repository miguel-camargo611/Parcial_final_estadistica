from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import FactorAnalysis, PCA


DATA_PATH = Path("Data/Processed/analysis_env_complete.parquet")
OUTPUT_DIR = Path("Outputs")
FIG_DIR = Path("Figures/Dimensionality_Reduction")

GROUP_COL = "Diversidad_Grupo"
ENV_VARS = [
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "O3",
    "Temperatura",
    "Humedad",
    "Viento",
    "Radiacion_Solar",
]
Z_ENV_VARS = [f"z_env_{col}" for col in ENV_VARS]
N_COMPONENTS = 3


def clean_var(name):
    return name.replace("z_env_", "")


def varimax(loadings, gamma=1.0, max_iter=100, tol=1e-6):
    p, k = loadings.shape
    rotation = np.eye(k)
    old_singular_sum = 0
    for _ in range(max_iter):
        rotated = loadings @ rotation
        u, singular_values, vh = np.linalg.svd(
            loadings.T
            @ (rotated**3 - (gamma / p) * rotated @ np.diag(np.diag(rotated.T @ rotated)))
        )
        rotation = u @ vh
        singular_sum = singular_values.sum()
        if old_singular_sum and singular_sum < old_singular_sum * (1 + tol):
            break
        old_singular_sum = singular_sum
    return loadings @ rotation, rotation


def load_data():
    df = pd.read_parquet(DATA_PATH).copy()
    return df, df[Z_ENV_VARS].to_numpy()


def fit_pca(x):
    pca = PCA(n_components=len(Z_ENV_VARS), random_state=42)
    scores = pca.fit_transform(x)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    return pca, scores, loadings


def fit_ppca(x, n_components=N_COMPONENTS):
    ppca = PCA(n_components=n_components, random_state=42)
    ppca_scores = ppca.fit_transform(x)
    return ppca, ppca_scores


def fit_factor_analysis(x, n_components=N_COMPONENTS):
    fa = FactorAnalysis(n_components=n_components, random_state=42)
    scores = fa.fit_transform(x)
    unrotated_loadings = fa.components_.T
    rotated_loadings, rotation = varimax(unrotated_loadings)
    rotated_scores = scores @ rotation
    return fa, scores, unrotated_loadings, rotated_loadings, rotated_scores


def save_pca_outputs(pca, loadings):
    explained = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "eigenvalue": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained.to_csv(OUTPUT_DIR / "dimred_pca_explained_variance.csv", index=False, encoding="utf-8")

    loadings_df = pd.DataFrame(
        loadings[:, :N_COMPONENTS],
        index=[clean_var(col) for col in Z_ENV_VARS],
        columns=[f"PC{i + 1}" for i in range(N_COMPONENTS)],
    )
    loadings_df.to_csv(OUTPUT_DIR / "dimred_pca_loadings.csv", encoding="utf-8")
    return explained, loadings_df


def save_factor_outputs(fa, rotated_loadings):
    loadings_df = pd.DataFrame(
        rotated_loadings,
        index=[clean_var(col) for col in Z_ENV_VARS],
        columns=[f"Factor{i + 1}" for i in range(rotated_loadings.shape[1])],
    )
    loadings_df.to_csv(OUTPUT_DIR / "dimred_factor_analysis_varimax_loadings.csv", encoding="utf-8")

    uniqueness = pd.DataFrame(
        {
            "variable": [clean_var(col) for col in Z_ENV_VARS],
            "uniqueness": fa.noise_variance_,
            "communality": 1 - fa.noise_variance_,
        }
    )
    uniqueness.to_csv(OUTPUT_DIR / "dimred_factor_analysis_uniqueness.csv", index=False, encoding="utf-8")
    return loadings_df, uniqueness


def save_scores(df, pca_scores, fa_scores):
    scores = df[["SAMPLING EVENT IDENTIFIER", GROUP_COL, "Shannon_Index"]].copy()
    for idx in range(N_COMPONENTS):
        scores[f"PC{idx + 1}"] = pca_scores[:, idx]
        scores[f"Factor{idx + 1}"] = fa_scores[:, idx]
    scores.to_csv(OUTPUT_DIR / "dimred_scores.csv", index=False, encoding="utf-8")
    return scores


def save_comparison(pca, ppca, fa, explained):
    ppca_noise = ppca.noise_variance_
    comparison = pd.DataFrame(
        [
            {
                "method": "PCA",
                "n_components_reported": N_COMPONENTS,
                "assumption": "Orthogonal directions maximizing variance",
                "variance_or_noise_summary": f"PC1-PC{N_COMPONENTS} cumulative variance = {explained.loc[N_COMPONENTS - 1, 'cumulative_variance_ratio']:.3f}",
            },
            {
                "method": "PPCA",
                "n_components_reported": N_COMPONENTS,
                "assumption": "Latent Gaussian components plus isotropic residual noise",
                "variance_or_noise_summary": f"sigma2 noise estimate = {ppca_noise:.3f}",
            },
            {
                "method": "FA Varimax",
                "n_components_reported": N_COMPONENTS,
                "assumption": "Latent factors explain shared covariance; uniqueness per variable",
                "variance_or_noise_summary": f"mean uniqueness = {np.mean(fa.noise_variance_):.3f}",
            },
        ]
    )
    comparison.to_csv(OUTPUT_DIR / "dimred_method_comparison.csv", index=False, encoding="utf-8")
    return comparison


def plot_scree(explained):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        range(1, len(explained) + 1),
        explained["explained_variance_ratio"],
        marker="o",
        color="#2563eb",
        label="Varianza individual",
    )
    ax.plot(
        range(1, len(explained) + 1),
        explained["cumulative_variance_ratio"],
        marker="s",
        color="#c84630",
        label="Varianza acumulada",
    )
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Proporción de varianza")
    ax.set_title("Scree plot PCA")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_pca_scree_plot.png", dpi=180)
    plt.close(fig)


def plot_pca_biplot(scores, loadings_df):
    subset = scores[scores[GROUP_COL].isin(["Baja_Q1", "Alta_Q3"])].copy()
    subset = subset.sample(min(len(subset), 2200), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=subset,
        x="PC1",
        y="PC2",
        hue=GROUP_COL,
        palette={"Baja_Q1": "#c84630", "Alta_Q3": "#2a9d8f"},
        s=16,
        alpha=0.35,
        linewidth=0,
        ax=ax,
    )
    scale = 2.4
    for variable, row in loadings_df[["PC1", "PC2"]].iterrows():
        ax.arrow(0, 0, row["PC1"] * scale, row["PC2"] * scale, color="#111827", alpha=0.75, head_width=0.05)
        ax.text(row["PC1"] * scale * 1.08, row["PC2"] * scale * 1.08, variable, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_title("PCA biplot coloreado por diversidad")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_pca_biplot_q1_q3.png", dpi=180)
    plt.close(fig)


def plot_pca_3d(scores):
    # Generar proyección tridimensional coloreando por Alta y Baja diversidad
    subset = scores[scores[GROUP_COL].isin(["Baja_Q1", "Alta_Q3"])].copy()
    subset = subset.sample(min(len(subset), 2200), random_state=42)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Separar grupos para graficar
    for label, color in [("Baja_Q1", "#c84630"), ("Alta_Q3", "#2a9d8f")]:
        grp = subset[subset[GROUP_COL] == label]
        ax.scatter(
            grp["PC1"],
            grp["PC2"],
            grp["PC3"],
            c=color,
            label="Baja Diversidad (Q1)" if label == "Baja_Q1" else "Alta Diversidad (Q3)",
            s=22,
            alpha=0.4,
            edgecolors="none",
        )

    ax.set_xlabel("PC1 (Material Particulado)")
    ax.set_ylabel("PC2 (Clima / Radiacion)")
    ax.set_zlabel("PC3 (Gases de Combustión)")
    ax.set_title("Proyección Tridimensional PCA (Micro-hábitats de Aves)", pad=20, fontsize=12)
    ax.legend(frameon=True, facecolor="white", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_pca_3d_scatter.png", dpi=180)
    plt.close(fig)


def plot_loadings_heatmap(loadings_df, filename, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        loadings_df,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Carga"},
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=180)
    plt.close(fig)



def compute_reconstruction_error(x_scaled, pca, k_values=None):
    """Calcula el error de reconstrucción Frobenius ||X - X_k||_F^2 para k=1..N_COMPONENTS."""
    if k_values is None:
        k_values = list(range(1, N_COMPONENTS + 1))
    results = []
    for k in k_values:
        components_k = pca.components_[:k]
        scores_k = x_scaled @ components_k.T
        x_reconstructed = scores_k @ components_k
        frob_error = np.sum((x_scaled - x_reconstructed) ** 2)
        frob_pct = frob_error / np.sum(x_scaled ** 2)
        results.append({
            "k_componentes": k,
            "error_frobenius": frob_error,
            "fraccion_varianza_no_reconstruida": frob_pct,
            "varianza_explicada_acumulada": pca.explained_variance_ratio_[:k].sum(),
        })
    df_recon = pd.DataFrame(results)
    df_recon.to_csv(OUTPUT_DIR / "dimred_reconstruction_error.csv", index=False, encoding="utf-8")
    return df_recon


def compute_fa_covariance_residual(fa, x_scaled):
    """
    Calcula la matriz de correlación reproducida por FA (Lambda*Lambda' + Psi)
    y la compara con la matriz de correlación observada R.
    Devuelve el residual R - R_hat y el RMSE.
    """
    loadings = fa.components_.T  # shape (p, k) — cargas sin rotar
    psi = np.diag(fa.noise_variance_)   # Unicidades (matriz diagonal)
    R_hat = loadings @ loadings.T + psi  # Correlación reproducida
    R_obs = np.corrcoef(x_scaled, rowvar=False)  # Correlación observada
    residual = R_obs - R_hat
    rmse = np.sqrt(np.mean(residual[~np.eye(residual.shape[0], dtype=bool)] ** 2))
    # Guardar matrices
    pd.DataFrame(R_hat, index=ENV_VARS, columns=ENV_VARS).to_csv(
        OUTPUT_DIR / "dimred_fa_reproduced_correlation.csv", encoding="utf-8"
    )
    pd.DataFrame(residual, index=ENV_VARS, columns=ENV_VARS).to_csv(
        OUTPUT_DIR / "dimred_fa_residual_correlation.csv", encoding="utf-8"
    )
    return R_obs, R_hat, residual, rmse


def plot_fa_covariance_comparison(R_obs, R_hat, residual):
    """Heatmaps lado a lado: R observada, R reproducida FA, y residual."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    kw = dict(annot=True, fmt=".2f", cmap="vlag", center=0, linewidths=0.4,
              vmin=-1, vmax=1, cbar=False, ax=None)
    for ax, mat, title in zip(
        axes,
        [R_obs, R_hat, residual],
        ["R Observada", "R Reproducida (FA)", "Residual (R − R̂)"],
    ):
        kw["ax"] = ax
        sns.heatmap(
            pd.DataFrame(mat, index=ENV_VARS, columns=ENV_VARS),
            **kw,
        )
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Análisis de Covarianza: Reproducción FA vs Observada", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_fa_covariance_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df, x = load_data()
    pca, pca_scores, pca_loadings = fit_pca(x)
    ppca, ppca_scores = fit_ppca(x)
    fa, fa_scores, fa_loadings, fa_rotated_loadings, fa_rotated_scores = fit_factor_analysis(x)

    explained, pca_loadings_df = save_pca_outputs(pca, pca_loadings)
    fa_loadings_df, uniqueness = save_factor_outputs(fa, fa_rotated_loadings)
    scores = save_scores(df, pca_scores, fa_rotated_scores)
    comparison = save_comparison(pca, ppca, fa, explained)

    plot_scree(explained)
    plot_pca_biplot(scores, pca_loadings_df)
    plot_pca_3d(scores)
    plot_loadings_heatmap(pca_loadings_df, "03_pca_loadings_heatmap.png", "Cargas PCA")
    plot_loadings_heatmap(fa_loadings_df, "04_factor_analysis_varimax_loadings.png", "Cargas Factor Analysis Varimax")

    # Error de reconstrucción y análisis de covarianza
    df_recon = compute_reconstruction_error(x, pca)
    R_obs, R_hat, residual, rmse = compute_fa_covariance_residual(fa, x)
    plot_fa_covariance_comparison(R_obs, R_hat, residual)

    print(f"Reducción dimensional sobre {DATA_PATH}: {x.shape[0]} filas x {x.shape[1]} variables")
    print("\nVarianza explicada PCA:")
    print(explained.head(N_COMPONENTS).round(4).to_string(index=False))
    print("\nError de Reconstrucción PCA por componente:")
    print(df_recon.round(4).to_string(index=False))
    print("\nPPCA:")
    print(f"sigma2 estimado = {ppca.noise_variance_:.4f}")
    print("\nAnálisis de Covarianza Factor Analysis:")
    print(f"RMSE de la matriz residual de covarianza/correlación: {rmse:.4f}")
    print("\nComparación:")
    print(comparison.to_string(index=False))
    print("\nCargas FA Varimax:")
    print(fa_loadings_df.round(3).to_string())
    print(f"\nFiguras guardadas en {FIG_DIR}")


if __name__ == "__main__":
    main()
