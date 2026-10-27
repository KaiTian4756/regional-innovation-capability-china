"""Structure diagnostics and fixed-effects panel regression.

Reads panel_with_scores.csv from this repository and writes outputs to
./verification_output/structure_diagnostics/ and ./verification_output/fe_verification/.

The recommended specification reported in Table 4 of the manuscript is
"province_year_fe_4x_recommended": outcome = High-tech output capability;
predictors = Scientific research talent, Technological investment,
Technology-oriented enterprises, Innovation environment; province and year
fixed effects; province-clustered standard errors.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT
STRUCTURE_DIR = DATA_DIR / "verification_output" / "structure_diagnostics"
FE_DIR = DATA_DIR / "verification_output" / "fe_verification"
PANEL_PATH = DATA_DIR / "panel_with_scores.csv"
REGRESSION_SUMMARY_PATH = DATA_DIR / "panel_regression_results.csv"

# Dimension-score columns; terminology follows Table 1 of the manuscript.
DIM_TALENT = "Scientific research talent capability"
DIM_INVEST = "Technological investment capability"
DIM_OUTPUT = "High-tech output capability"
DIM_ENTERPRISE = "Technology-oriented enterprise capability"
DIM_ENV = "Innovation environment capability"

DIMENSIONS = [DIM_TALENT, DIM_INVEST, DIM_OUTPUT, DIM_ENTERPRISE, DIM_ENV]
PREDICTORS_3 = [DIM_TALENT, DIM_INVEST, DIM_ENV]
PREDICTORS_4 = [DIM_TALENT, DIM_INVEST, DIM_ENTERPRISE, DIM_ENV]
OUTCOME = DIM_OUTPUT


def save_csv(frame: pd.DataFrame, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / name, index=False, encoding="utf-8-sig")


def standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    if np.any(std == 0):
        raise ValueError("Cannot standardize a zero-variance column")
    return (values - mean) / std


def pca_from_standardized(z: np.ndarray):
    corr = np.corrcoef(z, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    loadings = eigenvectors[:, order]
    for column in range(loadings.shape[1]):
        largest = int(np.argmax(np.abs(loadings[:, column])))
        if loadings[largest, column] < 0:
            loadings[:, column] *= -1
    shares = eigenvalues / eigenvalues.sum()
    scores = z @ loadings
    return corr, eigenvalues, shares, loadings, scores


def structure_diagnostic(panel: pd.DataFrame) -> dict:
    data = panel.loc[panel["Year"].astype(int) == 2020].copy()
    data = data.sort_values("Province").reset_index(drop=True)
    raw = data[DIMENSIONS].astype(float).to_numpy()
    z = standardize(raw)
    corr, eigenvalues, shares, loadings, scores = pca_from_standardized(z)

    corr_frame = pd.DataFrame(corr, index=DIMENSIONS, columns=DIMENSIONS)
    corr_frame.insert(0, "Dimension", corr_frame.index)
    save_csv(corr_frame.reset_index(drop=True), STRUCTURE_DIR, "correlation_matrix.csv")

    eigen_frame = pd.DataFrame({
        "Principal component": [f"PC{i + 1}" for i in range(len(eigenvalues))],
        "Eigenvalue": eigenvalues,
        "Variance explained": shares,
        "Cumulative variance explained": np.cumsum(shares),
    })
    save_csv(eigen_frame, STRUCTURE_DIR, "pca_eigenvalues.csv")

    loading_frame = pd.DataFrame(loadings, columns=[f"PC{i + 1}" for i in range(loadings.shape[1])])
    loading_frame.insert(0, "Dimension", DIMENSIONS)
    save_csv(loading_frame, STRUCTURE_DIR, "pca_loadings.csv")

    full_pc1_share = float(shares[0])
    influence_rows = []
    for index, region in enumerate(data["Province"].astype(str)):
        keep = np.arange(len(data)) != index
        loo_z = standardize(raw[keep])
        _, _, loo_shares, _, _ = pca_from_standardized(loo_z)
        influence_rows.append({
            "Province": region,
            "z distance": float(np.sqrt((z[index] ** 2).sum())),
            "PC1 score": float(scores[index, 0]),
            "|PC1 score|": float(abs(scores[index, 0])),
            "Full-sample PC1 explained variance": full_pc1_share,
            "PC1 explained variance after exclusion": float(loo_shares[0]),
            "Change in PC1 explained variance": float(loo_shares[0] - full_pc1_share),
        })
    influence_frame = pd.DataFrame(influence_rows).sort_values("z distance", ascending=False)
    save_csv(influence_frame, STRUCTURE_DIR, "province_outlier_influence.csv")

    diagnostics = {
        "n_regions_2020": int(len(data)),
        "unique_regions_2020": int(data["Province"].nunique()),
        "missing_values": int(data[DIMENSIONS].isna().sum().sum()),
        "max_abs_off_diagonal_correlation": float(np.max(np.abs(corr - np.eye(len(DIMENSIONS))))),
        "pc1_explained_variance": full_pc1_share,
        "pc2_cumulative_explained_variance": float(np.cumsum(shares)[1]),
        "top_z_distance_region": str(influence_frame.iloc[0]["Province"]),
        "top_z_distance": float(influence_frame.iloc[0]["z distance"]),
        "largest_pc1_leave_one_out_change_region": str(
            influence_frame.iloc[np.argmax(np.abs(influence_frame["Change in PC1 explained variance"].to_numpy()))]["Province"]
        ),
        "interpretation": (
            "This is a 2020 cross-sectional structure diagnostic. It does not establish causality "
            "or validate cluster generalization."
        ),
    }
    (STRUCTURE_DIR / "diagnostic_summary.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return diagnostics


def betacf(a: float, b: float, x: float) -> float:
    # Continued fraction for the regularized incomplete beta function.
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = max(abs(d), fpmin) * (1.0 if d >= 0 else -1.0)
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = max(abs(d), fpmin) * (1.0 if d >= 0 else -1.0)
        c = 1.0 + aa / c
        c = max(abs(c), fpmin) * (1.0 if c >= 0 else -1.0)
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = max(abs(d), fpmin) * (1.0 if d >= 0 else -1.0)
        c = 1.0 + aa / c
        c = max(abs(c), fpmin) * (1.0 if c >= 0 else -1.0)
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def two_sided_t_p(value: float, degrees_of_freedom: int) -> float:
    if not np.isfinite(value):
        return 0.0
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    return float(regularized_beta(x, degrees_of_freedom / 2.0, 0.5))


def build_design(data: pd.DataFrame, predictors: list[str], fixed_effects: str):
    n = len(data)
    columns = [np.ones(n)]
    names = ["Constant"]
    if fixed_effects in {"province", "province_year"}:
        provinces = sorted(data["Province"].astype(str).unique())
        for province in provinces[1:]:
            columns.append((data["Province"].astype(str).to_numpy() == province).astype(float))
            names.append(f"province FE:{province}")
    if fixed_effects == "province_year":
        years = sorted(data["Year"].astype(int).unique())
        for year in years[1:]:
            columns.append((data["Year"].astype(int).to_numpy() == year).astype(float))
            names.append(f"year FE:{year}")
    focus_start = len(columns)
    for predictor in predictors:
        columns.append(data[predictor].astype(float).to_numpy())
        names.append(predictor)
    design = np.column_stack(columns)
    return design, names, list(range(focus_start, len(names)))


def ols_with_cluster_se(data: pd.DataFrame, predictors: list[str], fixed_effects: str, model_name: str):
    y = data[OUTCOME].astype(float).to_numpy()
    design, names, focus_indices = build_design(data, predictors, fixed_effects)
    n, k = design.shape
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    xtx_inv = np.linalg.pinv(design.T @ design)
    sse = float(residuals @ residuals)
    dof_resid = n - k
    sigma2 = sse / dof_resid
    cov_conventional = xtx_inv * sigma2

    clusters = data["Province"].astype(str).to_numpy()
    unique_clusters = sorted(np.unique(clusters))
    meat = np.zeros((k, k), dtype=float)
    for cluster in unique_clusters:
        rows = clusters == cluster
        score = design[rows].T @ residuals[rows]
        meat += np.outer(score, score)
    correction = (len(unique_clusters) / (len(unique_clusters) - 1)) * ((n - 1) / dof_resid)
    cov_cluster = xtx_inv @ meat @ xtx_inv * correction

    fe_only_columns = [design[:, 0]]
    if fixed_effects in {"province", "province_year"}:
        fe_only_columns.extend(design[:, 1 : 1 + data["Province"].nunique() - 1].T)
    if fixed_effects == "province_year":
        start = 1 + data["Province"].nunique() - 1
        fe_only_columns.extend(design[:, start : start + data["Year"].nunique() - 1].T)
    fe_only = np.column_stack(fe_only_columns)
    fe_fit = fe_only @ np.linalg.lstsq(fe_only, y, rcond=None)[0]
    sst = float(((y - y.mean()) ** 2).sum())
    within_sst = float(((y - fe_fit) ** 2).sum())
    r2 = 1.0 - sse / sst if sst else float("nan")
    within_r2 = 1.0 - sse / within_sst if within_sst else float("nan")

    rows = []
    for predictor, index in zip(predictors, focus_indices):
        coefficient = float(beta[index])
        se_cluster = float(math.sqrt(max(cov_cluster[index, index], 0.0)))
        se_conventional = float(math.sqrt(max(cov_conventional[index, index], 0.0)))
        t_cluster = coefficient / se_cluster if se_cluster else float("inf")
        t_conventional = coefficient / se_conventional if se_conventional else float("inf")
        rows.append({
            "Model": model_name,
            "Variable": predictor,
            "Coefficient": coefficient,
            "Province-clustered SE": se_cluster,
            "Clustered t value": t_cluster,
            "Clustered p value (df=30)": two_sided_t_p(t_cluster, len(unique_clusters) - 1),
            "Conventional SE": se_conventional,
            "Conventional t value": t_conventional,
            "Conventional p value": two_sided_t_p(t_conventional, dof_resid),
        })
    summary = {
        "Model": model_name,
        "Fixed effects": fixed_effects,
        "Predictors": "; ".join(predictors),
        "Outcome": OUTCOME,
        "Observations": n,
        "Provinces": len(unique_clusters),
        "Years": int(data["Year"].nunique()),
        "Design matrix parameters": k,
        "Residual degrees of freedom": dof_resid,
        "Overall R2": r2,
        "Within R2": within_r2,
        "Province cluster correction": "G/(G-1) * (N-1)/(N-K)",
    }
    return pd.DataFrame(rows), summary


def fe_verification(panel: pd.DataFrame) -> dict:
    data = panel.copy()
    data["Year"] = data["Year"].astype(int)
    data = data.sort_values(["Province", "Year"]).reset_index(drop=True)
    coefficient_frames = []
    summaries = []
    models = [
        ("pooled_ols_3x", PREDICTORS_3, "none"),
        ("province_fe_3x_legacy", PREDICTORS_3, "province"),
        ("province_year_fe_3x", PREDICTORS_3, "province_year"),
        ("province_year_fe_4x_recommended", PREDICTORS_4, "province_year"),
    ]
    for model_name, predictors, fixed_effects in models:
        coefficients, summary = ols_with_cluster_se(data, predictors, fixed_effects, model_name)
        coefficient_frames.append(coefficients)
        summaries.append(summary)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    save_csv(coefficients, FE_DIR, "fe_coefficients.csv")
    save_csv(pd.DataFrame(summaries), FE_DIR, "fe_model_summary.csv")

    result = {
        "input": str(PANEL_PATH),
        "outcome": OUTCOME,
        "recommended_model": "province_year_fe_4x_recommended",
        "recommended_predictors": PREDICTORS_4,
        "all_models_use_province_clustered_covariance": True,
        "results_summary_file": str(REGRESSION_SUMMARY_PATH),
        "interpretation_boundary": (
            "These are association estimates in a 31-province, 2011-2020 panel; they do not establish causality."
        ),
    }
    (FE_DIR / "fe_verification_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def write_readmes(diagnostics: dict) -> None:
    (STRUCTURE_DIR / "README.md").write_text(
        f"""# Five-dimension structure diagnostic

Input: `panel_with_scores.csv`, 2020 cross-section, n={diagnostics['n_regions_2020']}.

Methods: Pearson correlation of the five dimension scores; PCA on 2020 z-scores;
leave-one-province-out PCA influence. This is a descriptive structure check, not a causal
analysis or a validation of cluster generalization.

Key values:

- Maximum absolute off-diagonal correlation: {diagnostics['max_abs_off_diagonal_correlation']:.4f}
- PC1 explained variance: {diagnostics['pc1_explained_variance']:.4f}
- PC1+PC2 cumulative explained variance: {diagnostics['pc2_cumulative_explained_variance']:.4f}
- Largest z-distance province: {diagnostics['top_z_distance_region']}
- Largest absolute PC1 leave-one-out change: {diagnostics['largest_pc1_leave_one_out_change_region']}

No source data files were overwritten.
""",
        encoding="utf-8",
    )
    (FE_DIR / "README.md").write_text(
        """# Fixed-effects verification

The script evaluates four models on `panel_with_scores.csv`:

1. pooled OLS with three predictors (scientific research talent, technological investment,
   innovation environment);
2. province fixed effects with the same three predictors (legacy specification);
3. province and year fixed effects with the same three predictors;
4. province and year fixed effects with four predictors, adding technology-oriented
   enterprise capability (the recommended specification reported in Table 4 of the manuscript).

All reported standard errors include a province-clustered sandwich correction. The model
coefficients are associations, not causal effects.

No source data files were overwritten.
""",
        encoding="utf-8",
    )


def main() -> None:
    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    FE_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH, encoding="utf-8-sig")
    diagnostics = structure_diagnostic(panel)
    fe_verification(panel)
    write_readmes(diagnostics)


if __name__ == "__main__":
    main()
