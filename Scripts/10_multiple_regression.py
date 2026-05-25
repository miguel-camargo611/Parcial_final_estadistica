from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


DATA_PATH = Path("Data/Processed/analysis_regression_complete.parquet")
OUTPUT_DIR = Path("Outputs")
FIG_DIR = Path("Figures/Regression")

RESPONSE = "Shannon_Index"
PREDICTORS = [
    "z_reg_PM2.5",
    "z_reg_PM10",
    "z_reg_NO2",
    "z_reg_CO",
    "z_reg_O3",
    "z_reg_Temperatura",
    "z_reg_Humedad",
    "z_reg_Viento",
    "z_reg_Lluvia",
    "z_reg_Radiacion_Solar",
    "z_reg_DURATION MINUTES",
    "z_reg_EFFORT DISTANCE KM",
]
SENSITIVITY_PREDICTORS_NO_HUMIDITY = [col for col in PREDICTORS if col != "z_reg_Humedad"]
RAW_PREDICTOR_MAP = {
    "z_reg_PM2.5": "PM2.5",
    "z_reg_PM10": "PM10",
    "z_reg_NO2": "NO2",
    "z_reg_CO": "CO",
    "z_reg_O3": "O3",
    "z_reg_Temperatura": "Temperatura",
    "z_reg_Humedad": "Humedad",
    "z_reg_Viento": "Viento",
    "z_reg_Lluvia": "Lluvia",
    "z_reg_Radiacion_Solar": "Radiacion_Solar",
    "z_reg_DURATION MINUTES": "DURATION MINUTES",
    "z_reg_EFFORT DISTANCE KM": "EFFORT DISTANCE KM",
}


def clean_term(term):
    return (
        term.replace("z_reg_", "")
        .replace("DURATION MINUTES", "Duracion")
        .replace("EFFORT DISTANCE KM", "Distancia")
    )


def load_model_data():
    df = pd.read_parquet(DATA_PATH)
    cols = [RESPONSE] + PREDICTORS + list(RAW_PREDICTOR_MAP.values())
    return df[cols].dropna().copy()


def fit_model(df, predictors=None):
    predictors = predictors or PREDICTORS
    x = sm.add_constant(df[predictors], has_constant="add")
    y = df[RESPONSE]
    model = sm.OLS(y, x).fit()
    robust_model = model.get_robustcov_results(cov_type="HC3")
    return model, robust_model, x, y


def coefficient_table(model, robust_model):
    conf = model.conf_int()
    robust_conf = robust_model.conf_int()
    table = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "std_err": model.bse.values,
            "t": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_low": conf[0].values,
            "ci_high": conf[1].values,
            "robust_hc3_std_err": robust_model.bse,
            "robust_hc3_t": robust_model.tvalues,
            "robust_hc3_p_value": robust_model.pvalues,
            "robust_hc3_ci_low": robust_conf[:, 0],
            "robust_hc3_ci_high": robust_conf[:, 1],
        }
    )
    table["variable"] = table["term"].map(clean_term)
    table["abs_coef"] = table["coef"].abs()
    return table


def original_scale_coefficient_table(df, coefficients):
    rows = []
    for _, row in coefficients[coefficients["term"].isin(RAW_PREDICTOR_MAP)].iterrows():
        raw_col = RAW_PREDICTOR_MAP[row["term"]]
        raw_std = df[raw_col].std(ddof=0)
        rows.append(
            {
                "term": row["term"],
                "variable": clean_term(row["term"]),
                "raw_variable": raw_col,
                "raw_mean": df[raw_col].mean(),
                "raw_std_used_for_z": raw_std,
                "coef_per_1_original_unit": row["coef"] / raw_std,
                "ci_low_per_1_original_unit": row["ci_low"] / raw_std,
                "ci_high_per_1_original_unit": row["ci_high"] / raw_std,
                "p_value": row["p_value"],
                "robust_hc3_coef_per_1_original_unit": row["coef"] / raw_std,
                "robust_hc3_ci_low_per_1_original_unit": row["robust_hc3_ci_low"] / raw_std,
                "robust_hc3_ci_high_per_1_original_unit": row["robust_hc3_ci_high"] / raw_std,
                "robust_hc3_p_value": row["robust_hc3_p_value"],
                "standardized_coef_per_1_sd": row["coef"],
            }
        )
    return pd.DataFrame(rows)


def vif_table(x):
    predictors = x.drop(columns="const")
    rows = []
    for idx, col in enumerate(predictors.columns):
        rows.append(
            {
                "term": col,
                "variable": clean_term(col),
                "vif": variance_inflation_factor(predictors.to_numpy(), idx),
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def diagnostics_table(model, x):
    residuals = model.resid
    shapiro_sample = residuals.sample(5000, random_state=42) if len(residuals) > 5000 else residuals
    shapiro_stat, shapiro_p = stats.shapiro(shapiro_sample)
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals, x)
    return pd.DataFrame(
        [
            {
                "n": int(model.nobs),
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "aic": model.aic,
                "bic": model.bic,
                "f_stat": model.fvalue,
                "f_p_value": model.f_pvalue,
                "breusch_pagan_lm": bp_lm,
                "breusch_pagan_lm_p_value": bp_lm_p,
                "breusch_pagan_f": bp_f,
                "breusch_pagan_f_p_value": bp_f_p,
                "shapiro_w": shapiro_stat,
                "shapiro_p_value_sample_max_5000": shapiro_p,
                "durbin_watson": durbin_watson(residuals),
                "residual_mean": residuals.mean(),
                "residual_std": residuals.std(ddof=1),
                "residual_skew": stats.skew(residuals, bias=False),
                "residual_kurtosis_excess": stats.kurtosis(residuals, fisher=True, bias=False),
            }
        ]
    )


def save_results(model, robust_model, x):
    coefficients = coefficient_table(model, robust_model)
    coefficients.to_csv(OUTPUT_DIR / "regression_ols_coefficients.csv", index=False, encoding="utf-8")

    vifs = vif_table(x)
    vifs.to_csv(OUTPUT_DIR / "regression_vif.csv", index=False, encoding="utf-8")

    diagnostics = diagnostics_table(model, x)
    diagnostics.to_csv(OUTPUT_DIR / "regression_diagnostics.csv", index=False, encoding="utf-8")

    with open(OUTPUT_DIR / "regression_ols_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(model.summary().as_text())
        fh.write("\n\n")
        fh.write(robust_model.summary().as_text())

    return coefficients, vifs, diagnostics


def save_original_scale_results(df, coefficients):
    original_scale = original_scale_coefficient_table(df, coefficients)
    original_scale.to_csv(
        OUTPUT_DIR / "regression_ols_coefficients_original_scale.csv",
        index=False,
        encoding="utf-8",
    )
    return original_scale


def save_sensitivity_no_humidity(df):
    model, robust_model, x, y = fit_model(df, SENSITIVITY_PREDICTORS_NO_HUMIDITY)
    coefficients = coefficient_table(model, robust_model)
    coefficients.to_csv(
        OUTPUT_DIR / "regression_sensitivity_no_humidity_coefficients.csv",
        index=False,
        encoding="utf-8",
    )
    vifs = vif_table(x)
    vifs.to_csv(
        OUTPUT_DIR / "regression_sensitivity_no_humidity_vif.csv",
        index=False,
        encoding="utf-8",
    )
    diagnostics = diagnostics_table(model, x)
    diagnostics.to_csv(
        OUTPUT_DIR / "regression_sensitivity_no_humidity_diagnostics.csv",
        index=False,
        encoding="utf-8",
    )
    return model, robust_model, coefficients, vifs, diagnostics


def plot_coefficients(coefficients):
    plot_df = coefficients[coefficients["term"] != "const"].sort_values("coef")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#c84630" if coef < 0 else "#2a9d8f" for coef in plot_df["coef"]]
    ax.barh(plot_df["variable"], plot_df["coef"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coeficiente OLS estandarizado")
    ax.set_ylabel("")
    ax.set_title("Coeficientes del modelo de regresión múltiple")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_regression_coefficients.png", dpi=180)
    plt.close(fig)


def plot_residuals_vs_fitted(model):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        x=model.fittedvalues,
        y=model.resid,
        lowess=True,
        scatter_kws={"s": 10, "alpha": 0.25, "color": "#334155"},
        line_kws={"color": "#c84630", "linewidth": 2},
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Residuales vs valores ajustados")
    ax.set_xlabel("Valores ajustados")
    ax.set_ylabel("Residuales")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_residuals_vs_fitted.png", dpi=180)
    plt.close(fig)


def plot_qq(model):
    fig = sm.qqplot(model.resid, line="45", fit=True)
    fig.set_size_inches(7, 6)
    ax = fig.axes[0]
    ax.set_title("QQ-plot de residuales")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_residuals_qqplot.png", dpi=180)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_model_data()
    model, robust_model, x, y = fit_model(df)
    coefficients, vifs, diagnostics = save_results(model, robust_model, x)
    original_scale = save_original_scale_results(df, coefficients)
    sensitivity_model, sensitivity_robust, sensitivity_coefficients, sensitivity_vifs, sensitivity_diagnostics = (
        save_sensitivity_no_humidity(df)
    )

    plot_coefficients(coefficients)
    plot_residuals_vs_fitted(model)
    plot_qq(model)

    print(f"Regresión OLS sobre {DATA_PATH}: {len(df)} filas")
    print("\nDiagnósticos:")
    print(diagnostics.round(4).to_string(index=False))
    print("\nVIF:")
    print(vifs.round(3).to_string(index=False))
    print("\nCoeficientes principales por magnitud absoluta:")
    top = coefficients[coefficients["term"] != "const"].sort_values("abs_coef", ascending=False).head(8)
    print(top[["variable", "coef", "p_value", "robust_hc3_p_value"]].round(4).to_string(index=False))
    print("\nCoeficientes en escala original por unidad del predictor:")
    original_top = original_scale[original_scale["term"].isin(top["term"])]
    original_top = original_top.set_index("term").loc[top["term"]].reset_index()
    print(
        original_top[
            [
                "variable",
                "coef_per_1_original_unit",
                "robust_hc3_ci_low_per_1_original_unit",
                "robust_hc3_ci_high_per_1_original_unit",
                "robust_hc3_p_value",
            ]
        ]
        .round(5)
        .to_string(index=False)
    )
    print("\nSensibilidad sin Humedad:")
    print(sensitivity_diagnostics[["n", "r_squared", "adj_r_squared"]].round(4).to_string(index=False))
    print(sensitivity_vifs.round(3).to_string(index=False))
    print(f"\nFiguras guardadas en {FIG_DIR}")


if __name__ == "__main__":
    main()
