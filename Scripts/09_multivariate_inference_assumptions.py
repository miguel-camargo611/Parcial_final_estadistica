from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.stats import chi2, f, norm


DATA_PATH = Path("Data/Processed/analysis_env_complete.parquet")
OUTPUT_DIR = Path("Outputs")
FIG_DIR = Path("Figures/Multivariate_Inference")

GROUP_COL = "Diversidad_Grupo"
LOW_GROUP = "Baja_Q1"
HIGH_GROUP = "Alta_Q3"

CONTINUOUS_ENV_VARS = [
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
Z_CONTINUOUS_ENV_VARS = [f"z_env_{col}" for col in CONTINUOUS_ENV_VARS]
BIVAR_Z = ["z_env_Temperatura", "z_env_Humedad"]


def mardia_test(x, chunk_size=512):
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    centered = x - x.mean(axis=0)
    inv_cov = np.linalg.pinv(np.cov(centered, rowvar=False, ddof=1))

    skew_sum = 0.0
    for start in range(0, n, chunk_size):
        block = centered[start : start + chunk_size]
        mahal_cross = block @ inv_cov @ centered.T
        skew_sum += np.sum(mahal_cross**3)

    mahal_diag = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
    skewness = skew_sum / (n**2)
    kurtosis = np.mean(mahal_diag**2)

    skew_stat = n * skewness / 6
    skew_df = p * (p + 1) * (p + 2) / 6
    skew_p = chi2.sf(skew_stat, skew_df)

    expected_kurtosis = p * (p + 2)
    kurtosis_var = 8 * p * (p + 2) / n
    kurtosis_z = (kurtosis - expected_kurtosis) / np.sqrt(kurtosis_var)
    kurtosis_p = 2 * norm.sf(abs(kurtosis_z))

    return {
        "n": n,
        "p": p,
        "mardia_skewness": skewness,
        "skew_chi2": skew_stat,
        "skew_df": skew_df,
        "skew_p_value": skew_p,
        "mardia_kurtosis": kurtosis,
        "expected_kurtosis": expected_kurtosis,
        "kurtosis_z": kurtosis_z,
        "kurtosis_p_value": kurtosis_p,
        "normal_skew_0_05": skew_p >= 0.05,
        "normal_kurtosis_0_05": kurtosis_p >= 0.05,
    }


def box_m_test(grouped_arrays):
    labels = list(grouped_arrays)
    arrays = [np.asarray(grouped_arrays[label], dtype=float) for label in labels]
    n_groups = len(arrays)
    p = arrays[0].shape[1]
    ns = np.array([arr.shape[0] for arr in arrays])
    covs = [np.cov(arr, rowvar=False, ddof=1) for arr in arrays]
    pooled = sum((n - 1) * cov for n, cov in zip(ns, covs)) / (ns.sum() - n_groups)

    sign_pooled, logdet_pooled = np.linalg.slogdet(pooled)
    if sign_pooled <= 0:
        raise ValueError("La matriz de covarianza combinada no es definida positiva.")

    logdets = []
    for cov in covs:
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("Una matriz de covarianza grupal no es definida positiva.")
        logdets.append(logdet)

    m_stat = (ns.sum() - n_groups) * logdet_pooled - sum((n - 1) * logdet for n, logdet in zip(ns, logdets))
    correction = (
        (2 * p**2 + 3 * p - 1)
        / (6 * (p + 1) * (n_groups - 1))
        * (sum(1 / (ns - 1)) - 1 / (ns.sum() - n_groups))
    )
    chi2_stat = (1 - correction) * m_stat
    df = (n_groups - 1) * p * (p + 1) / 2
    p_value = chi2.sf(chi2_stat, df)

    return {
        "groups": ", ".join(labels),
        "n_total": int(ns.sum()),
        "p": p,
        "box_m": m_stat,
        "chi2_corrected": chi2_stat,
        "df": df,
        "p_value": p_value,
        "equal_covariances_0_05": p_value >= 0.05,
    }


def hotellings_t2(x1, x2):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n1, p = x1.shape
    n2 = x2.shape[0]
    mean1 = x1.mean(axis=0)
    mean2 = x2.mean(axis=0)
    diff = mean1 - mean2
    s1 = np.cov(x1, rowvar=False, ddof=1)
    s2 = np.cov(x2, rowvar=False, ddof=1)
    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    t2 = (n1 * n2 / (n1 + n2)) * diff @ np.linalg.pinv(pooled) @ diff
    f_stat = ((n1 + n2 - p - 1) / (p * (n1 + n2 - 2))) * t2
    df1 = p
    df2 = n1 + n2 - p - 1
    p_value = f.sf(f_stat, df1, df2)
    return {
        "n_low": n1,
        "n_high": n2,
        "p": p,
        "hotelling_t2": t2,
        "f_stat": f_stat,
        "df1": df1,
        "df2": df2,
        "p_value": p_value,
        "reject_equal_centroids_0_05": p_value < 0.05,
    }


def permutation_centroid_test(x1, x2, n_permutations=9999, random_state=42):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    rng = np.random.default_rng(random_state)
    n1 = x1.shape[0]
    combined = np.vstack([x1, x2])

    observed_diff = x2.mean(axis=0) - x1.mean(axis=0)
    observed_distance = np.linalg.norm(observed_diff)

    permuted_distances = np.empty(n_permutations)
    for i in range(n_permutations):
        permutation = rng.permutation(combined.shape[0])
        perm_x1 = combined[permutation[:n1]]
        perm_x2 = combined[permutation[n1:]]
        permuted_distances[i] = np.linalg.norm(perm_x2.mean(axis=0) - perm_x1.mean(axis=0))

    p_value = (np.sum(permuted_distances >= observed_distance) + 1) / (n_permutations + 1)
    return {
        "n_low": x1.shape[0],
        "n_high": x2.shape[0],
        "p": x1.shape[1],
        "n_permutations": n_permutations,
        "observed_centroid_distance": observed_distance,
        "permutation_p_value": p_value,
        "reject_equal_centroids_0_05": p_value < 0.05,
    }, permuted_distances


def add_mean_confidence_ellipse(ax, mean, cov, n, probability, **kwargs):
    radius = np.sqrt(chi2.ppf(probability, df=2))
    eigvals, eigvecs = np.linalg.eigh(cov / n)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * radius * np.sqrt(eigvals)
    ax.add_patch(Ellipse(mean, width, height, angle=angle, fill=False, **kwargs))


def prepare_groups(df):
    groups = df[df[GROUP_COL].isin([LOW_GROUP, HIGH_GROUP])].copy()
    low = groups[groups[GROUP_COL] == LOW_GROUP][Z_CONTINUOUS_ENV_VARS].to_numpy()
    high = groups[groups[GROUP_COL] == HIGH_GROUP][Z_CONTINUOUS_ENV_VARS].to_numpy()
    return groups, low, high


def save_results(df, groups, low, high, n_permutations=9999):
    mardia_rows = []
    mardia_rows.append({"sample": "Completa", **mardia_test(df[Z_CONTINUOUS_ENV_VARS].to_numpy())})
    mardia_rows.append({"sample": LOW_GROUP, **mardia_test(low)})
    mardia_rows.append({"sample": HIGH_GROUP, **mardia_test(high)})
    mardia = pd.DataFrame(mardia_rows)
    mardia.to_csv(OUTPUT_DIR / "multivar_mardia_continuous.csv", index=False, encoding="utf-8")

    box_m = pd.DataFrame(
        [
            box_m_test(
                {
                    LOW_GROUP: low,
                    HIGH_GROUP: high,
                }
            )
        ]
    )
    box_m.to_csv(OUTPUT_DIR / "multivar_box_m_continuous.csv", index=False, encoding="utf-8")

    hotelling = pd.DataFrame([hotellings_t2(low, high)])
    hotelling.to_csv(OUTPUT_DIR / "multivar_hotelling_continuous.csv", index=False, encoding="utf-8")

    permutation_result, permuted_distances = permutation_centroid_test(
        low,
        high,
        n_permutations=n_permutations,
    )
    permutation = pd.DataFrame([permutation_result])
    permutation.to_csv(OUTPUT_DIR / "multivar_permutation_centroids_continuous.csv", index=False, encoding="utf-8")
    pd.DataFrame({"permuted_centroid_distance": permuted_distances}).to_csv(
        OUTPUT_DIR / "multivar_permutation_distances_continuous.csv",
        index=False,
        encoding="utf-8",
    )

    means = (
        groups.groupby(GROUP_COL)[Z_CONTINUOUS_ENV_VARS]
        .mean()
        .T.rename_axis("variable")
        .reset_index()
    )
    means["variable"] = means["variable"].str.replace("z_env_", "", regex=False)
    means["Alta_menos_Baja"] = means[HIGH_GROUP] - means[LOW_GROUP]
    means = means.sort_values("Alta_menos_Baja")
    means.to_csv(OUTPUT_DIR / "multivar_group_mean_differences_continuous.csv", index=False, encoding="utf-8")

    return mardia, box_m, hotelling, permutation, means, permuted_distances


def plot_permutation_distribution(permuted_distances, observed_distance):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(permuted_distances, bins=45, color="#64748b", edgecolor="white", ax=ax)
    ax.axvline(
        observed_distance,
        color="#c84630",
        linewidth=2.2,
        linestyle="--",
        label=f"Observado = {observed_distance:.3f}",
    )
    ax.set_title("Prueba permutacional de distancia entre centroides")
    ax.set_xlabel("Distancia entre centroides bajo permutación")
    ax.set_ylabel("Frecuencia")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_permutation_centroid_distance_continuous.png", dpi=180)
    plt.close(fig)


def plot_group_profile(means):
    long = means.melt(
        id_vars=["variable", "Alta_menos_Baja"],
        value_vars=[LOW_GROUP, HIGH_GROUP],
        var_name=GROUP_COL,
        value_name="mean_z",
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.pointplot(
        data=long,
        x="mean_z",
        y="variable",
        hue=GROUP_COL,
        palette={LOW_GROUP: "#c84630", HIGH_GROUP: "#2a9d8f"},
        dodge=0.35,
        linestyle="none",
        ax=ax,
    )
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Media estandarizada")
    ax.set_ylabel("")
    ax.set_title("Perfil ambiental medio por grupo de diversidad")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_group_mean_profile_continuous.png", dpi=180)
    plt.close(fig)


def plot_centroid_confidence_ellipses(groups):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    ax_full, ax_zoom = axes
    sample = groups.sample(min(len(groups), 1800), random_state=42)
    sns.scatterplot(
        data=sample,
        x=BIVAR_Z[0],
        y=BIVAR_Z[1],
        hue=GROUP_COL,
        palette={LOW_GROUP: "#c84630", HIGH_GROUP: "#2a9d8f"},
        s=14,
        alpha=0.25,
        linewidth=0,
        ax=ax_full,
    )
    centroid_rows = []
    for label, color in [(LOW_GROUP, "#c84630"), (HIGH_GROUP, "#2a9d8f")]:
        sub = groups[groups[GROUP_COL] == label][BIVAR_Z].to_numpy()
        mean = sub.mean(axis=0)
        cov = np.cov(sub, rowvar=False, ddof=1)
        centroid_rows.append((label, color, mean, cov, len(sub)))
        ax_full.scatter(mean[0], mean[1], s=85, color=color, edgecolor="black", linewidth=0.8, zorder=5)
        add_mean_confidence_ellipse(ax_zoom, mean, cov, len(sub), 0.95, color=color, linewidth=2.8)
        ax_zoom.scatter(mean[0], mean[1], s=85, color=color, edgecolor="black", linewidth=0.8, zorder=5, label=label)

    ax_full.set_title("Nube bivariada")
    ax_full.set_xlabel("Temperatura estandarizada")
    ax_full.set_ylabel("Humedad estandarizada")
    ax_full.legend(title="")

    means = np.vstack([row[2] for row in centroid_rows])
    x_pad = 0.18
    y_pad = 0.18
    ax_zoom.set_xlim(means[:, 0].min() - x_pad, means[:, 0].max() + x_pad)
    ax_zoom.set_ylim(means[:, 1].min() - y_pad, means[:, 1].max() + y_pad)
    ax_zoom.axhline(0, color="black", linewidth=0.6, alpha=0.25)
    ax_zoom.axvline(0, color="black", linewidth=0.6, alpha=0.25)
    ax_zoom.set_title("Zoom: IC 95% de centroides")
    ax_zoom.set_xlabel("Temperatura estandarizada")
    ax_zoom.set_ylabel("Humedad estandarizada")
    ax_zoom.legend(title="")
    fig.suptitle("Elipses de confianza 95% para centroides", y=0.99, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_centroid_confidence_ellipses_temp_humidity.png", dpi=180)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    groups, low, high = prepare_groups(df)
    mardia, box_m, hotelling, permutation, means, permuted_distances = save_results(df, groups, low, high)

    plot_group_profile(means)
    plot_centroid_confidence_ellipses(groups)
    plot_permutation_distribution(
        permuted_distances,
        permutation.loc[0, "observed_centroid_distance"],
    )

    print(f"Inferencia multivariable sobre variables continuas: {len(Z_CONTINUOUS_ENV_VARS)} variables")
    print(f"Grupos: {LOW_GROUP} n={len(low)}, {HIGH_GROUP} n={len(high)}")
    print("\nMardia:")
    print(mardia.round(4).to_string(index=False))
    print("\nBox's M:")
    print(box_m.round(4).to_string(index=False))
    print("\nHotelling T2:")
    print(hotelling.round(4).to_string(index=False))
    print("\nPrueba permutacional:")
    print(permutation.round(4).to_string(index=False))
    print("\nDiferencias medias estandarizadas mas grandes:")
    print(means.reindex(means["Alta_menos_Baja"].abs().sort_values(ascending=False).index).head(6).round(3).to_string(index=False))
    print(f"\nFiguras guardadas en {FIG_DIR}")


if __name__ == "__main__":
    main()
