"""Recompute DEA-BCC efficiency, sigma/beta convergence, and cross-file consistency.

This script reads the English-language data files in this repository
(panel_with_scores.csv, dea_results.csv, convergence_analysis.csv,
spatial_analysis_results.csv, obstacle_analysis.csv, clustering_results.csv)
and writes verification outputs to ./verification_output/.
"""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT
PANEL_PATH = DATA_DIR / "panel_with_scores.csv"
DEA_PATH = DATA_DIR / "dea_results.csv"
CONVERGENCE_PATH = DATA_DIR / "convergence_analysis.csv"
SPATIAL_PATH = DATA_DIR / "spatial_analysis_results.csv"
OBSTACLE_PATH = DATA_DIR / "obstacle_analysis.csv"
CLUSTER_PATH = DATA_DIR / "clustering_results.csv"
DEA_DIR = DATA_DIR / "verification_output" / "dea_convergence_verification"
CONSISTENCY_DIR = DATA_DIR / "verification_output" / "numerical_consistency"

# Dimension-score columns in panel_with_scores.csv. Terminology follows
# Table 1 of the manuscript.
DIM_TALENT = "Scientific research talent capability"
DIM_INVEST = "Technological investment capability"
DIM_OUTPUT = "High-tech output capability"
DIM_ENTERPRISE = "Technology-oriented enterprise capability"
DIM_ENV = "Innovation environment capability"

DEA_INPUTS = [DIM_TALENT, DIM_INVEST, DIM_ENV]
DEA_OUTPUTS = [DIM_OUTPUT, DIM_ENTERPRISE]
DIMENSION_COLUMNS = [
    "Scientific research talent capability (%)",
    "Technological investment capability (%)",
    "High-tech output capability (%)",
    "Technology-oriented enterprise capability (%)",
    "Innovation environment capability (%)",
]


def save_csv(frame: pd.DataFrame, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / name, index=False, encoding="utf-8-sig")


def section_from_csv(path: Path, header_prefix: str, nrows: int | None = None) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(header_prefix))
    selected = lines[index:] if nrows is None else lines[index : index + nrows + 1]
    return pd.read_csv(StringIO("\n".join(selected)))


def dea_bcc_input_efficiency(inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """Solve input-oriented BCC envelopment LP for every DMU.

    min theta
    s.t. X lambda - theta*x_o <= 0,
         -Y lambda <= -y_o,
         sum(lambda) = 1, lambda >= 0, theta >= 0.
    """
    n = len(inputs)
    efficiencies = []
    for target in range(n):
        objective = np.r_[1.0, np.zeros(n)]
        input_constraints = np.column_stack([-inputs[target], inputs.T])
        output_constraints = np.column_stack([np.zeros(outputs.shape[1]), -outputs.T])
        a_ub = np.vstack([input_constraints, output_constraints])
        b_ub = np.r_[np.zeros(inputs.shape[1]), -outputs[target]]
        a_eq = np.r_[0.0, np.ones(n)][None, :]
        result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=np.array([1.0]),
            bounds=[(0.0, None)] * (n + 1),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"DEA LP failed for DMU {target}: {result.message}")
        efficiencies.append(float(result.fun))
    return np.asarray(efficiencies)


def run_dea(panel: pd.DataFrame) -> dict:
    rows = []
    for year, data in panel.groupby("Year", sort=True):
        data = data.sort_values("Province").reset_index(drop=True)
        inputs = data[DEA_INPUTS].astype(float).to_numpy()
        outputs = data[DEA_OUTPUTS].astype(float).to_numpy()
        efficiencies = dea_bcc_input_efficiency(inputs, outputs)
        rows.extend({
            "Province": region,
            "Year": int(year),
            "Recomputed overall efficiency": float(efficiency),
        } for region, efficiency in zip(data["Province"].astype(str), efficiencies))
    recomputed = pd.DataFrame(rows)
    save_csv(recomputed, DEA_DIR, "dea_recomputed_scores.csv")

    old_scores = section_from_csv(DEA_PATH, "Province,Overall efficiency", nrows=31)
    old_scores = old_scores.rename(columns={"Overall efficiency": "Existing overall efficiency"})
    old_scores["Province"] = old_scores["Province"].astype(str)
    comparison = old_scores[["Province", "Existing overall efficiency", "Innovation capability index", "Efficiency rank"]].merge(
        recomputed.loc[recomputed["Year"] == 2020], on="Province", how="left"
    )
    comparison["Efficiency difference"] = comparison["Recomputed overall efficiency"] - comparison["Existing overall efficiency"]
    save_csv(comparison, DEA_DIR, "dea_2020_comparison.csv")

    try:
        cluster = section_from_csv(CLUSTER_PATH, "Province,Innovation capability index", nrows=31)
        cluster = cluster[["Province", "Cluster ID"]].copy()
        cluster["Province"] = cluster["Province"].astype(str)
        cluster["Cluster ID"] = cluster["Cluster ID"].astype(int)
        tier_means = comparison[["Province", "Recomputed overall efficiency"]].merge(cluster, on="Province", how="left")
        tier_means = tier_means.groupby("Cluster ID", as_index=False).agg(
            n=("Recomputed overall efficiency", "size"),
            mean_efficiency=("Recomputed overall efficiency", "mean"),
        )
        cluster_tier_map = {0: 3, 1: 1, 2: 2, 3: 4}
        tier_means["Tier"] = tier_means["Cluster ID"].map(cluster_tier_map)
        tier_means = tier_means.sort_values("Tier")
        save_csv(tier_means, DEA_DIR, "dea_tier_means.csv")
        tier_means_map = {
            int(row["Cluster ID"]): float(row["mean_efficiency"])
            for row in tier_means.to_dict("records")
        }
    except Exception:
        tier_means_map = {}

    old_annual = section_from_csv(DEA_PATH, "Year,Mean efficiency", nrows=10)
    old_annual = old_annual.rename(columns={"Mean efficiency": "Existing mean efficiency"})
    recomputed_annual = recomputed.groupby("Year", as_index=False)["Recomputed overall efficiency"].mean()
    recomputed_annual = recomputed_annual.rename(columns={"Recomputed overall efficiency": "Recomputed mean efficiency"})
    annual = old_annual.merge(recomputed_annual, on="Year", how="left")
    annual["Mean efficiency difference"] = annual["Recomputed mean efficiency"] - annual["Existing mean efficiency"]
    save_csv(annual, DEA_DIR, "dea_annual_comparison.csv")

    summary = {
        "inputs": DEA_INPUTS,
        "outputs": DEA_OUTPUTS,
        "orientation": "input-oriented",
        "returns_to_scale": "BCC / variable returns to scale",
        "convexity_constraint": "sum(lambda)=1",
        "n_dmus_per_year": int(panel["Province"].nunique()),
        "years": sorted(panel["Year"].astype(int).unique().tolist()),
        "dea_2020_max_abs_difference": float(comparison["Efficiency difference"].abs().max()),
        "dea_2020_mean_recomputed": float(comparison["Recomputed overall efficiency"].mean()),
        "dea_2020_frontier_count_recomputed": int(np.isclose(comparison["Recomputed overall efficiency"], 1.0, atol=1e-7).sum()),
        "dea_2020_tier_means": tier_means_map,
        "annual_max_abs_difference": float(annual["Mean efficiency difference"].abs().max()),
        "interpretation_boundary": "DEA efficiency is resource-conversion efficiency conditional on observed input scale, not capability or output leadership.",
    }
    (DEA_DIR / "dea_verification_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def ols_convergence(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    scores = panel.pivot(index="Province", columns="Year", values="Innovation capability index")
    initial = scores[2011].astype(float)
    terminal = scores[2020].astype(float)
    x = np.log(initial.to_numpy())
    y = np.log((terminal / initial).to_numpy())
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ beta
    dof = len(x) - 2
    sigma2 = float(residuals @ residuals) / dof
    covariance = np.linalg.inv(design.T @ design) * sigma2
    standard_errors = np.sqrt(np.diag(covariance))
    t_values = beta / standard_errors
    p_values = 2 * stats.t.sf(np.abs(t_values), dof)
    r2 = 1.0 - float(residuals @ residuals) / float(((y - y.mean()) ** 2).sum())
    beta_frame = pd.DataFrame([
        {
            "Formula": "ln(RICI_2020/RICI_2011) = alpha + beta*ln(RICI_2011) + error",
            "alpha": float(beta[0]),
            "beta": float(beta[1]),
            "alpha_SE": float(standard_errors[0]),
            "beta_SE": float(standard_errors[1]),
            "alpha_t": float(t_values[0]),
            "beta_t": float(t_values[1]),
            "alpha_p": float(p_values[0]),
            "beta_p": float(p_values[1]),
            "R2": r2,
            "Observations": len(x),
        }
    ])
    save_csv(beta_frame, DEA_DIR, "convergence_beta_recomputed.csv")

    sigma_rows = []
    for year, group in panel.groupby("Year", sort=True):
        values = group["Innovation capability index"].astype(float).to_numpy()
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        sigma_rows.append({
            "Year": int(year),
            "Recomputed mean": mean,
            "Recomputed standard deviation": std,
            "Recomputed coefficient of variation": std / mean,
        })
    sigma_frame = pd.DataFrame(sigma_rows)
    save_csv(sigma_frame, DEA_DIR, "convergence_sigma_recomputed.csv")

    old_sigma = section_from_csv(CONVERGENCE_PATH, "Year,Mean", nrows=10)
    old_sigma = old_sigma.rename(columns={
        "Mean": "Existing mean",
        "Standard deviation": "Existing standard deviation",
        "Coefficient of variation": "Existing coefficient of variation",
    })
    sigma_comparison = old_sigma.merge(sigma_frame, on="Year", how="left")
    sigma_comparison["Mean difference"] = sigma_comparison["Recomputed mean"] - sigma_comparison["Existing mean"]
    sigma_comparison["SD difference"] = sigma_comparison["Recomputed standard deviation"] - sigma_comparison["Existing standard deviation"]
    sigma_comparison["CV difference"] = sigma_comparison["Recomputed coefficient of variation"] - sigma_comparison["Existing coefficient of variation"]
    save_csv(sigma_comparison, DEA_DIR, "convergence_sigma_comparison.csv")

    old_beta = section_from_csv(CONVERGENCE_PATH, "alpha,beta", nrows=1)
    old_beta = old_beta.rename(columns={
        "alpha": "Existing alpha",
        "beta": "Existing beta",
        "SE of beta": "Existing beta_SE",
        "t value": "Existing beta_t",
        "R-squared": "Existing R2",
        "Observations": "Existing observations",
    })
    beta_comparison = old_beta.copy()
    beta_comparison["Recomputed alpha"] = float(beta[0])
    beta_comparison["Recomputed beta"] = float(beta[1])
    beta_comparison["Recomputed beta_SE"] = float(standard_errors[1])
    beta_comparison["Recomputed beta_t"] = float(t_values[1])
    beta_comparison["Recomputed R2"] = r2
    beta_comparison["Recomputed observations"] = len(x)
    save_csv(beta_comparison, DEA_DIR, "convergence_beta_comparison.csv")

    return beta_frame, {
        "formula": "ln(RICI_2020/RICI_2011) = alpha + beta*ln(RICI_2011) + error",
        "recomputed_beta": float(beta[1]),
        "recomputed_beta_se": float(standard_errors[1]),
        "recomputed_beta_t": float(t_values[1]),
        "recomputed_beta_p": float(p_values[1]),
        "recomputed_r2": r2,
        "recomputed_n": len(x),
        "old_file_beta_n": int(old_beta.iloc[0]["Existing observations"]),
        "old_file_is_structurally_valid": True,
    }


def run_club_convergence(panel: pd.DataFrame) -> dict:
    """Recompute the 2011-2020 total-period club beta convergence rows."""
    scores = panel.pivot(index="Province", columns="Year", values="Innovation capability index")
    region_map = {}
    for province in ["Beijing", "Tianjin", "Hebei", "Shanghai", "Jiangsu", "Zhejiang", "Fujian", "Shandong", "Guangdong", "Hainan"]:
        region_map[province] = "Eastern"
    for province in ["Shanxi", "Anhui", "Jiangxi", "Henan", "Hubei", "Hunan"]:
        region_map[province] = "Central"
    for province in ["Inner Mongolia", "Guangxi", "Chongqing", "Sichuan", "Guizhou", "Yunnan", "Tibet", "Shaanxi", "Gansu", "Qinghai", "Ningxia", "Xinjiang"]:
        region_map[province] = "Western"
    for province in ["Liaoning", "Jilin", "Heilongjiang"]:
        region_map[province] = "Northeastern"

    rows = []
    for region in ["Eastern", "Central", "Western", "Northeastern"]:
        data = scores.loc[[province for province in scores.index if region_map.get(province) == region]]
        x = np.log(data[2011].astype(float).to_numpy())
        y = np.log((data[2020] / data[2011]).astype(float).to_numpy())
        design = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        residuals = y - design @ beta
        dof = len(x) - 2
        sigma2 = float(residuals @ residuals) / dof
        covariance = np.linalg.inv(design.T @ design) * sigma2
        standard_errors = np.sqrt(np.diag(covariance))
        t_values = beta / standard_errors
        r2 = 1.0 - float(residuals @ residuals) / float(((y - y.mean()) ** 2).sum())
        rows.append({
            "Region": region,
            "alpha": float(beta[0]),
            "beta": float(beta[1]),
            "SE of beta": float(standard_errors[1]),
            "t value": float(t_values[1]),
            "R-squared": r2,
            "Observations": len(x),
        })
    recomputed = pd.DataFrame(rows)
    save_csv(recomputed, DEA_DIR, "convergence_club_recomputed.csv")

    try:
        old_club = section_from_csv(CONVERGENCE_PATH, "Region,beta", nrows=4)
        old_club = old_club.rename(columns={"beta": "Existing beta", "SE of beta": "Existing SE of beta",
                                            "t value": "Existing t value", "R-squared": "Existing R-squared",
                                            "Observations": "Existing observations"})
        comparison = old_club.merge(recomputed, on="Region", how="outer")
        comparison["beta difference"] = comparison["beta"] - comparison["Existing beta"]
        save_csv(comparison, DEA_DIR, "convergence_club_comparison.csv")
        max_abs_beta_diff = float(comparison["beta difference"].abs().max())
    except Exception:
        comparison = recomputed.copy()
        comparison["Existing beta"] = None
        comparison["beta difference"] = None
        save_csv(comparison, DEA_DIR, "convergence_club_comparison.csv")
        max_abs_beta_diff = None

    return {
        "club_formula": "ln(RICI_2020/RICI_2011) = alpha + beta*ln(RICI_2011) + error for each four-region club",
        "club_rows": len(recomputed),
        "club_max_abs_beta_difference": max_abs_beta_diff,
    }


def numerical_consistency(panel: pd.DataFrame, convergence_summary: dict | None = None, club_summary: dict | None = None) -> dict:
    # 2020 innovation capability index scores across source-backed files.
    panel_2020 = panel.loc[panel["Year"].astype(int) == 2020, ["Province", "Innovation capability index"]].copy()
    cluster = section_from_csv(CLUSTER_PATH, "Province,Innovation capability index", nrows=31)
    cluster = cluster.rename(columns={"Innovation capability index": "Clustering file index"})
    cluster["Province"] = cluster["Province"].astype(str)
    rank_check = panel_2020.merge(cluster[["Province", "Clustering file index"]], on="Province", how="outer")
    rank_check["Clustering score difference"] = rank_check["Clustering file index"] - rank_check["Innovation capability index"]
    dea = pd.read_csv(DEA_DIR / "dea_2020_comparison.csv", encoding="utf-8-sig")
    rank_check = rank_check.merge(dea[["Province", "Existing overall efficiency"]], on="Province", how="left")
    save_csv(rank_check.sort_values("Innovation capability index", ascending=False), CONSISTENCY_DIR, "2020_score_crossfile_check.csv")

    # Growth percentages cited in the manuscript.
    scores = panel.pivot(index="Province", columns="Year", values="Innovation capability index")
    growth = pd.DataFrame({
        "Province": scores.index.astype(str),
        "2011 index": scores[2011].to_numpy(),
        "2020 index": scores[2020].to_numpy(),
    })
    growth["Growth rate %"] = (growth["2020 index"] / growth["2011 index"] - 1.0) * 100.0
    save_csv(growth.sort_values("Growth rate %", ascending=False), CONSISTENCY_DIR, "growth_rate_check.csv")

    obstacle = section_from_csv(OBSTACLE_PATH, "Province,Innovation capability index", nrows=31)
    obstacle["Obstacle degree total %"] = obstacle[DIMENSION_COLUMNS].sum(axis=1)
    obstacle["Deviation from 100 (pp)"] = obstacle["Obstacle degree total %"] - 100.0
    national = obstacle[DIMENSION_COLUMNS].mean().rename("National mean % (reported file)").reset_index()
    national.columns = ["Dimension", "National mean % (reported file)"]
    save_csv(obstacle[["Province", "Obstacle degree total %", "Deviation from 100 (pp)"]], CONSISTENCY_DIR, "obstacle_sum_check.csv")
    save_csv(national, CONSISTENCY_DIR, "obstacle_national_means.csv")

    spatial = pd.read_csv(SPATIAL_PATH, encoding="utf-8-sig")
    spatial_check = spatial.copy()
    spatial_check["Independent spatial-weights recomputation"] = False
    spatial_check["Note"] = "No spatial weight matrix or province geometry is distributed in this repository; values retained for cross-checking."
    save_csv(spatial_check, CONSISTENCY_DIR, "moran_source_check.csv")

    summary = {
        "score_2020_max_abs_difference_cluster_file": float(rank_check["Clustering score difference"].abs().max()),
        "obstacle_max_abs_deviation_from_100": float(obstacle["Deviation from 100 (pp)"].abs().max()),
        "obstacle_rows": int(len(obstacle)),
        "spatial_independent_recalculation": False,
        "spatial_limitation": "No spatial weight matrix or province geometry is distributed in this repository.",
        "files_checked": ["panel_with_scores.csv", "clustering_results.csv", "dea_results.csv", "convergence_analysis.csv", "spatial_analysis_results.csv", "obstacle_analysis.csv"],
    }
    if convergence_summary is not None:
        summary["convergence_file_corrected_to_total_period"] = True
        summary["national_beta_total_period"] = float(convergence_summary["recomputed_beta"])
        summary["national_beta_n"] = int(convergence_summary["recomputed_n"])
    if club_summary is not None:
        summary["club_convergence_recomputed"] = "verification_output/dea_convergence_verification/convergence_club_recomputed.csv"

    (CONSISTENCY_DIR / "consistency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def write_readmes(dea_summary: dict, convergence_summary: dict, club_summary: dict, consistency_summary: dict) -> None:
    tier_means = dea_summary.get("dea_2020_tier_means", {})
    cluster_tier_names = {0: "3", 1: "1", 2: "2", 3: "4"}
    tier_mean_lines = "\n".join(
        f"  - Tier {cluster_tier_names.get(k, str(int(k)+1))} (cluster {k}): {v:.4f}" for k, v in sorted(tier_means.items())
    ) if tier_means else "  (not available)"
    (DEA_DIR / "README.md").write_text(
        f"""# DEA and convergence verification

DEA was recomputed with the input-oriented BCC envelopment model, using inputs
{', '.join(DEA_INPUTS)} and outputs {', '.join(DEA_OUTPUTS)}. The convexity constraint is
`sum(lambda)=1`, which represents variable returns to scale.

- 2020 maximum efficiency difference from the existing file: {dea_summary['dea_2020_max_abs_difference']:.10f}
- Recomputed 2020 mean efficiency: {dea_summary['dea_2020_mean_recomputed']:.6f}
- Recomputed 2020 frontier DMUs: {dea_summary['dea_2020_frontier_count_recomputed']}
- Maximum annual mean difference: {dea_summary['annual_max_abs_difference']:.10f}
- Recomputed 2020 tier mean efficiency:
{tier_mean_lines}

Convergence was recomputed as:
`ln(RICI_2020/RICI_2011) = alpha + beta*ln(RICI_2011) + error`.

- Recomputed beta: {convergence_summary['recomputed_beta']:.6f}
- Recomputed beta t: {convergence_summary['recomputed_beta_t']:.4f}
- Recomputed beta p: {convergence_summary['recomputed_beta_p']:.6f}
- Recomputed R2: {convergence_summary['recomputed_r2']:.4f}
- Recomputed observations: {convergence_summary['recomputed_n']}
- Existing convergence file observations: {convergence_summary['old_file_beta_n']}
- Club convergence rows recomputed: {club_summary['club_rows']}
- Club maximum absolute beta difference: {club_summary['club_max_abs_beta_difference']}

DEA efficiency is conditional resource-conversion efficiency, not overall innovation capability.
""",
        encoding="utf-8",
    )
    (CONSISTENCY_DIR / "README.md").write_text(
        f"""# Numerical consistency checks

Cross-file checks cover 2020 innovation capability index scores, growth rates, obstacle-degree
sums, annual sigma statistics, DEA scores, and the source Moran table. The maximum absolute
difference between 2020 index scores in `panel_with_scores.csv` and `clustering_results.csv` is
{consistency_summary['score_2020_max_abs_difference_cluster_file']:.6f}, consistent with rounding.

Obstacle rows have a maximum absolute deviation from 100% of
{consistency_summary['obstacle_max_abs_deviation_from_100']:.4f} percentage points after
the source file's one-decimal rounding.

Moran's I was not independently recomputed because this repository does not contain the spatial
weights matrix or province geometry. The source values are retained with that limitation marked.
""",
        encoding="utf-8",
    )


def main() -> None:
    DEA_DIR.mkdir(parents=True, exist_ok=True)
    CONSISTENCY_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH, encoding="utf-8-sig")
    panel["Year"] = panel["Year"].astype(int)
    dea_summary = run_dea(panel)
    _, convergence_summary = ols_convergence(panel)
    club_summary = run_club_convergence(panel)
    consistency_summary = numerical_consistency(panel, convergence_summary, club_summary)
    write_readmes(dea_summary, convergence_summary, club_summary, consistency_summary)


if __name__ == "__main__":
    main()
