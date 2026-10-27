"""K-means clustering verification for the 2020 cross-section.

Reads panel_with_scores.csv and clustering_results.csv from this repository and
writes outputs to ./verification_output/kmeans_verification/.

K0 reproduces the reported k=4 tiers from the one-dimensional innovation
capability index; K1-K3 evaluate standardized five-dimension clustering for
k = 2..6 with initialization and leave-one-province-out stability checks.
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT
OUT_DIR = DATA_DIR / "verification_output" / "kmeans_verification"
PANEL_PATH = DATA_DIR / "panel_with_scores.csv"
OLD_PATH = DATA_DIR / "clustering_results.csv"

# Dimension-score columns; terminology follows Table 1 of the manuscript.
DIMENSIONS = [
    "Scientific research talent capability",
    "Technological investment capability",
    "High-tech output capability",
    "Technology-oriented enterprise capability",
    "Innovation environment capability",
]


def squared_distance_matrix(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


def kmeans_once(X: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 300):
    n = len(X)
    if k > n:
        raise ValueError("k cannot exceed the number of observations")

    centers = np.empty((k, X.shape[1]), dtype=float)
    first = int(rng.integers(n))
    centers[0] = X[first]
    closest_sq = ((X - centers[0]) ** 2).sum(axis=1)
    for idx in range(1, k):
        total = float(closest_sq.sum())
        if total == 0:
            choices = np.setdiff1d(np.arange(n), np.unique(np.argmax(closest_sq)))
            next_idx = int(rng.choice(choices))
        else:
            next_idx = int(rng.choice(n, p=closest_sq / total))
        centers[idx] = X[next_idx]
        closest_sq = np.minimum(closest_sq, ((X - centers[idx]) ** 2).sum(axis=1))

    labels = np.full(n, -1, dtype=int)
    for _ in range(max_iter):
        distances = squared_distance_matrix(X, centers)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = X[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
            else:
                # Keep all k clusters represented by moving an empty center to
                # the point currently farthest from its assigned center.
                farthest = int(np.argmax(distances[np.arange(n), labels]))
                centers[cluster] = X[farthest]

    distances = squared_distance_matrix(X, centers)
    labels = distances.argmin(axis=1)
    inertia = float(distances[np.arange(n), labels].sum())
    return labels, centers, inertia


def best_kmeans(X: np.ndarray, k: int, seeds: list[int]):
    runs = []
    for seed in seeds:
        labels, centers, inertia = kmeans_once(X, k, np.random.default_rng(seed))
        runs.append((labels, centers, inertia, seed))
    return min(runs, key=lambda item: item[2]), runs


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(squared_distance_matrix(X, X), 0.0))


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    distances = pairwise_distances(X)
    values = []
    for i in range(len(X)):
        own = labels[i]
        own_members = np.where(labels == own)[0]
        if len(own_members) <= 1:
            values.append(0.0)
            continue
        a = float(distances[i, own_members].sum() / (len(own_members) - 1))
        other_means = []
        for cluster in np.unique(labels):
            if cluster == own:
                continue
            members = np.where(labels == cluster)[0]
            other_means.append(float(distances[i, members].mean()))
        b = min(other_means)
        values.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return float(np.mean(values))


def calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    n = len(X)
    clusters = np.unique(labels)
    k = len(clusters)
    overall = X.mean(axis=0)
    between = 0.0
    within = 0.0
    for cluster in clusters:
        members = X[labels == cluster]
        center = members.mean(axis=0)
        between += len(members) * float(((center - overall) ** 2).sum())
        within += float(((members - center) ** 2).sum())
    if k <= 1 or n <= k or within == 0:
        return float("inf") if within == 0 else float("nan")
    return float((between / (k - 1)) / (within / (n - k)))


def davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    clusters = np.unique(labels)
    centers = []
    scatters = []
    for cluster in clusters:
        members = X[labels == cluster]
        center = members.mean(axis=0)
        centers.append(center)
        scatters.append(float(np.sqrt(((members - center) ** 2).sum(axis=1)).mean()))
    centers = np.asarray(centers)
    scatters = np.asarray(scatters)
    center_distances = np.sqrt(np.maximum(squared_distance_matrix(centers, centers), 0.0))
    ratios = np.zeros((len(clusters), len(clusters)), dtype=float)
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if i != j and center_distances[i, j] > 0:
                ratios[i, j] = (scatters[i] + scatters[j]) / center_distances[i, j]
    return float(np.max(ratios, axis=1).mean())


def comb2(values: np.ndarray | int) -> np.ndarray | float:
    values = np.asarray(values, dtype=float)
    return values * (values - 1) / 2


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("ARI inputs must have equal lengths")
    a_values = np.unique(labels_a)
    b_values = np.unique(labels_b)
    table = np.zeros((len(a_values), len(b_values)), dtype=float)
    for i, a in enumerate(a_values):
        for j, b in enumerate(b_values):
            table[i, j] = np.sum((labels_a == a) & (labels_b == b))
    sum_comb = comb2(table).sum()
    row_comb = comb2(table.sum(axis=1)).sum()
    col_comb = comb2(table.sum(axis=0)).sum()
    total_comb = float(comb2(len(labels_a)))
    expected = row_comb * col_comb / total_comb if total_comb else 0.0
    maximum = (row_comb + col_comb) / 2
    denominator = maximum - expected
    return float((sum_comb - expected) / denominator) if denominator else 1.0


def align_labels(reference: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    reference_values = sorted(np.unique(reference).tolist())
    target_values = sorted(np.unique(target).tolist())
    best = None
    for permutation in itertools.permutations(reference_values, len(target_values)):
        mapping = dict(zip(target_values, permutation))
        mapped = np.asarray([mapping[int(value)] for value in target], dtype=int)
        mismatches = int(np.sum(mapped != reference))
        if best is None or mismatches < best[0]:
            best = (mismatches, mapped, mapping)
    return best[1], best[2]


def save_csv(frame: pd.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH, encoding="utf-8-sig")
    panel_2020 = panel.loc[panel["Year"].astype(int) == 2020].copy()
    panel_2020 = panel_2020.sort_values("Province").reset_index(drop=True)
    X_raw = panel_2020[DIMENSIONS].astype(float).to_numpy()
    X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0, ddof=0)
    regions = panel_2020["Province"].astype(str).to_numpy()
    composite = panel_2020["Innovation capability index"].astype(float).to_numpy()

    old = pd.read_csv(OLD_PATH, skiprows=2, nrows=31, encoding="utf-8-sig")
    old = old.rename(columns={old.columns[0]: "Province"})
    old = old[["Province", "Innovation capability index", "Cluster ID", "Capability tier"]].copy()
    old["Province"] = old["Province"].astype(str)
    old = old.set_index("Province").loc[regions].reset_index()
    old_score_diff = float(np.max(np.abs(old["Innovation capability index"].astype(float).to_numpy() - composite)))

    one_d = composite.reshape(-1, 1)
    one_d_best, _ = best_kmeans(one_d, 4, list(range(100)))
    one_d_labels, _, one_d_inertia, one_d_seed = one_d_best
    k0 = pd.DataFrame({
        "Province": regions,
        "Existing cluster ID": old["Cluster ID"].astype(int),
        "Reproduced cluster ID": one_d_labels,
        "Existing capability tier": old["Capability tier"],
        "Innovation capability index": composite,
    })
    save_csv(k0, "k0_1d_k4_assignments.csv")

    seeds = list(range(100))
    metric_rows = []
    init_rows = []
    loo_rows = []
    assignments = {}
    profiles = {}
    for k in range(2, 7):
        best, runs = best_kmeans(X, k, seeds)
        labels, centers, inertia, best_seed = best
        assignments[k] = labels
        aris_to_best = [adjusted_rand_index(labels, run[0]) for run in runs]
        init_rows.extend({
            "k": k,
            "seed": run[3],
            "inertia": run[2],
            "ari_to_best": ari,
        } for run, ari in zip(runs, aris_to_best))

        loo_aris = []
        for omitted in range(len(X)):
            keep = np.arange(len(X)) != omitted
            loo_best, _ = best_kmeans(X[keep], k, list(range(50)))
            loo_labels = loo_best[0]
            loo_aris.append(adjusted_rand_index(labels[keep], loo_labels))
            loo_rows.append({
                "k": k,
                "omitted_region": regions[omitted],
                "ari_on_remaining_regions": loo_aris[-1],
                "loo_best_inertia": loo_best[2],
            })

        counts = np.bincount(labels, minlength=k)
        metric_rows.append({
            "k": k,
            "inertia": inertia,
            "silhouette": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz(X, labels),
            "davies_bouldin": davies_bouldin(X, labels),
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
            "best_seed": best_seed,
            "init_median_ari_to_best": float(np.median(aris_to_best)),
            "init_ari_ge_0_90_rate": float(np.mean(np.asarray(aris_to_best) >= 0.90)),
            "loo_median_ari": float(np.median(loo_aris)),
            "loo_min_ari": float(np.min(loo_aris)),
        })

        rows = []
        for cluster in range(k):
            members = labels == cluster
            row = {"k": k, "cluster": cluster, "n": int(members.sum())}
            for name, value in zip(DIMENSIONS, centers[cluster]):
                row[f"z_{name}"] = float(value)
            for name, value in zip(DIMENSIONS, X_raw[members].mean(axis=0)):
                row[f"mean_{name}"] = float(value)
            rows.append(row)
        profiles[k] = pd.DataFrame(rows)
        save_csv(profiles[k], f"k{k}_cluster_profiles.csv")
        save_csv(pd.DataFrame({"Province": regions, "cluster": labels}), f"k{k}_5d_assignments.csv")

    metrics = pd.DataFrame(metric_rows)
    metrics["silhouette_rank"] = metrics["silhouette"].rank(ascending=False, method="min").astype(int)
    metrics["calinski_harabasz_rank"] = metrics["calinski_harabasz"].rank(ascending=False, method="min").astype(int)
    metrics["davies_bouldin_rank"] = metrics["davies_bouldin"].rank(ascending=True, method="min").astype(int)
    metrics["top2_metric_count"] = (
        (metrics["silhouette_rank"] <= 2).astype(int)
        + (metrics["calinski_harabasz_rank"] <= 2).astype(int)
        + (metrics["davies_bouldin_rank"] <= 2).astype(int)
    )
    metrics["passes_planned_gate"] = (
        (metrics["min_cluster_size"] >= 3)
        & (metrics["init_ari_ge_0_90_rate"] >= 0.90)
        & (metrics["loo_median_ari"] >= 0.80)
        & (metrics["top2_metric_count"] >= 2)
    )
    eligible = metrics.loc[metrics["passes_planned_gate"]].sort_values(
        ["silhouette", "top2_metric_count"], ascending=[False, False]
    )
    recommended_k = int(eligible.iloc[0]["k"]) if len(eligible) else None
    save_csv(metrics, "k_metrics.csv")
    save_csv(pd.DataFrame(init_rows), "stability_initialization.csv")
    save_csv(pd.DataFrame(loo_rows), "stability_leave_one_out.csv")

    k4_labels = assignments[4]
    aligned_k4_labels, k4_mapping = align_labels(one_d_labels, k4_labels)
    comparison = pd.DataFrame({
        "Province": regions,
        "Existing cluster ID": old["Cluster ID"].astype(int),
        "Reproduced 1D cluster ID": one_d_labels,
        "5D cluster ID (k=4)": k4_labels,
        "5D cluster ID (k=4) aligned to 1D": aligned_k4_labels,
        "Membership changed (1D vs 5D)": one_d_labels != aligned_k4_labels,
    })
    save_csv(comparison, "k4_1d_vs_5d_comparison.csv")

    decision = {
        "inputs": {
            "panel_path": str(PANEL_PATH),
            "old_clustering_path": str(OLD_PATH),
            "n_regions_2020": int(len(regions)),
            "dimensions": DIMENSIONS,
            "standardization": "2020 cross-section z-score, population standard deviation",
        },
        "k0": {
            "one_d_best_seed": int(one_d_seed),
            "one_d_inertia": float(one_d_inertia),
            "old_score_max_abs_difference": old_score_diff,
            "old_vs_reproduced_ari": adjusted_rand_index(old["Cluster ID"].to_numpy(), one_d_labels),
            "old_vs_5d_k4_ari": adjusted_rand_index(old["Cluster ID"].to_numpy(), k4_labels),
        },
        "planned_gate": {
            "min_cluster_size": 3,
            "initialization_ari_ge_0_90_rate": 0.90,
            "leave_one_out_median_ari": 0.80,
            "top_two_internal_metrics": 2,
        },
        "recommended_k": recommended_k,
        "k4_label_mapping_to_1d": {str(key): int(value) for key, value in k4_mapping.items()},
        "interpretation_boundary": "2020 static five-dimension clustering; not evidence of catching-up dynamics",
    }
    (OUT_DIR / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# K-means verification output

This directory was generated from `panel_with_scores.csv` and the existing
`clustering_results.csv`. Existing files were not overwritten.

- K0: one-dimensional innovation capability index, k=4, 100 k-means++ initializations.
- K1: 2020 standardized five-dimension scores, k=2..6, 100 initializations per k.
- K2: initialization agreement and leave-one-province-out ARI stability (50 starts per fit).
- K3: k=4 one-dimensional versus five-dimensional membership comparison.

The planned gate is descriptive and pre-specified: minimum cluster size >=3, at least 90%
of starts with ARI >=0.90 to the best partition, leave-one-out median ARI >=0.80, and at
least two of three internal metrics in their top two. Recommended k under this gate:
{recommended_k if recommended_k is not None else 'none'}.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
