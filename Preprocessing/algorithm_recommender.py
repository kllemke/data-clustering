"""Recommend clustering algorithms based on detected data types.

Pipeline: analyze_dataset(df) -> match types to algorithms -> score/adjust
-> return ranked recommendations with reasons, risks, metrics and parameters.
"""
import json
import math
import re

import pandas as pd

from DataSetProfile import profile
from OutlierDetection import iqr_outliers
from TimePreprocessing import TimePreprocessing

# Priority when several types are detected (first = primary type)
TYPE_PRIORITY = ["time_series", "uncertain", "multimedia", "text", "categorical", "numerical"]
SECONDARY_TYPE_WEIGHT = 0.85   # algorithms matched only via a non-primary type

MEDIA_RE = re.compile(r"\.(?:jpe?g|png|gif|bmp|webp|tiff?|mp3|wav|flac|ogg|mp4|avi|mov|mkv)$", re.I)
UNCERTAIN_RE = re.compile(r"(^|_)(std|sd|stderr|err|error|uncertainty|lower|upper|ci)($|_)", re.I)

# ---------------------------------------------------------------------------
# Knowledge base: one entry per algorithm
# ---------------------------------------------------------------------------
def _k_range(c):
    return [2, min(10, max(2, int(math.sqrt(c["sample_size"] / 2))))]

ALGORITHMS = {
    "K-Means": {
        "reasons": ["Fast and scales linearly with the number of samples",
                    "Simple to interpret: every cluster has a centroid"],
        "risks": ["Requires the number of clusters k in advance",
                  "Assumes compact, roughly spherical clusters of similar size"],
        "metrics": ["silhouette_score", "davies_bouldin_index", "calinski_harabasz_index", "inertia (elbow method)"],
        "params": lambda c, t: {"n_clusters": f"try {_k_range(c)[0]}..{_k_range(c)[1]}, pick by silhouette/elbow",
                                "init": "k-means++", "n_init": 10},
    },
    "DBSCAN": {
        "reasons": ["Finds clusters of arbitrary shape without specifying their number",
                    "Labels outliers as noise instead of forcing them into a cluster"],
        "risks": ["Sensitive to parameter selection (eps, min_samples)",
                  "Performs poorly when clusters have very different densities"],
        "metrics": ["silhouette_score (non-noise points)", "davies_bouldin_index (non-noise points)",
                    "dbcv", "noise_ratio"],
        "params": lambda c, t: {"min_samples": max(4, 2 * c["dimensionality"]),
                                "eps": "read from the k-distance plot (k = min_samples), at the knee",
                                "metric": "cosine" if t in ("text", "multimedia") else "euclidean (standardize first)"},
    },
    "Hierarchical Clustering": {
        "reasons": ["Needs no fixed k up front; the dendrogram shows the structure at every level",
                    "Deterministic: the same data always gives the same result"],
        "risks": ["Memory and time grow quadratically with the number of samples",
                  "Merges can never be undone, so early mistakes persist"],
        "metrics": ["cophenetic_correlation", "silhouette_score", "davies_bouldin_index"],
        "params": lambda c, t: {"linkage": "average (cosine)" if t == "text" else "ward (euclidean)",
                                "n_clusters": "or distance_threshold, chosen from the dendrogram"},
    },
    "SOM": {
        "reasons": ["Projects the data onto a 2D map and preserves its neighbourhood structure",
                    "Gives a visual overview in addition to clusters"],
        "risks": ["Training time can be long and results depend on map size and learning rate",
                  "Clusters must be extracted from the map in a second step"],
        "metrics": ["quantization_error", "topographic_error"],
        "params": lambda c, t: {"map_size": f"about {max(3, round(math.sqrt(5 * math.sqrt(c['sample_size']))))}x"
                                            f"{max(3, round(math.sqrt(5 * math.sqrt(c['sample_size']))))}",
                                "sigma": 1.0, "learning_rate": 0.5},
    },
    "DTW-based K-Means": {
        "reasons": ["Dynamic Time Warping compares series by shape, tolerating shifts in time",
                    "Cluster centres (DTW barycenters) are themselves interpretable series"],
        "risks": ["DTW is expensive: roughly quadratic in series length per comparison",
                  "Series should be cleaned and normalized first, or scale dominates shape"],
        "metrics": ["silhouette_score (DTW distance)", "inertia (DTW)"],
        "params": lambda c, t: {"metric": "dtw", "n_clusters": f"try {_k_range(c)[0]}..{_k_range(c)[1]}",
                                "sakoe_chiba_radius": "about 10% of the series length"},
    },
    "GMM": {
        "reasons": ["Models clusters as ellipses, so it handles different sizes and correlations",
                    "Gives soft assignments (probabilities) instead of hard labels"],
        "risks": ["Assumes roughly Gaussian clusters",
                  "Many covariance parameters; can overfit or fail to converge on small data"],
        "metrics": ["bic", "aic", "silhouette_score", "log_likelihood"],
        "params": lambda c, t: {"n_components": f"try {_k_range(c)[0]}..{_k_range(c)[1]}, pick lowest BIC",
                                "covariance_type": "diag" if c["dimensionality"] > 10 else "full"},
    },
    "Spectral Clustering": {
        "reasons": ["Finds non-convex clusters that K-Means misses",
                    "Works on a similarity graph, so it follows the connectivity of the data"],
        "risks": ["Needs an eigendecomposition: slow and memory-hungry on large datasets",
                  "Results depend on the affinity / neighbourhood settings and on k"],
        "metrics": ["silhouette_score", "davies_bouldin_index", "eigengap (to choose k)"],
        "params": lambda c, t: {"affinity": "nearest_neighbors", "n_neighbors": 10,
                                "n_clusters": "from the eigengap heuristic"},
    },
    "K-Modes": {
        "reasons": ["Uses matching dissimilarity and modes, so no numeric encoding is needed",
                    "Same fast, simple workflow as K-Means"],
        "risks": ["Requires k in advance",
                  "Results depend on the initialisation; run several times"],
        "metrics": ["cost (sum of dissimilarities)", "silhouette_score (Hamming/Gower distance)"],
        "params": lambda c, t: {"n_clusters": f"try {_k_range(c)[0]}..{_k_range(c)[1]}, elbow on cost",
                                "init": "Cao", "n_init": 10},
    },
    "ROCK": {
        "reasons": ["Clusters by shared neighbours (links), which suits categorical attributes",
                    "More robust than pairwise similarity when categories overlap"],
        "risks": ["Quadratic cost in the number of samples",
                  "Needs a good similarity threshold theta"],
        "metrics": ["rock_criterion_function", "silhouette_score (Jaccard distance)"],
        "params": lambda c, t: {"theta": 0.5, "n_clusters": "chosen by the criterion function"},
    },
    "Categorical DBSCAN": {
        "reasons": ["Density-based: no fixed k and outliers become noise",
                    "Uses Hamming/Gower distance instead of Euclidean distance"],
        "risks": ["Distances on categorical data are coarse, so many ties occur",
                  "Sensitive to eps and min_samples"],
        "metrics": ["silhouette_score (Hamming/Gower distance)", "noise_ratio"],
        "params": lambda c, t: {"metric": "hamming or gower", "min_samples": max(4, 2 * c["dimensionality"]),
                                "eps": "read from the k-distance plot"},
    },
    "UK-Means": {
        "reasons": ["Takes the measurement uncertainty of each value into account",
                    "Keeps K-Means' speed and interpretability"],
        "risks": ["Needs reliable uncertainty estimates (std / intervals)",
                  "Still requires k in advance and assumes compact clusters"],
        "metrics": ["expected_within_cluster_cost", "silhouette_score (on expected values)"],
        "params": lambda c, t: {"n_clusters": f"try {_k_range(c)[0]}..{_k_range(c)[1]}",
                                "uncertainty_columns": c.get("uncertainty_columns", [])},
    },
    "FDBSCAN": {
        "reasons": ["Fuzzy distances model the uncertainty of each object in a density-based method",
                    "No fixed k and outliers become noise"],
        "risks": ["Fewer maintained implementations; may need a custom one",
                  "Sensitive to eps, min_samples and the probability threshold"],
        "metrics": ["silhouette_score (on expected values)", "noise_ratio"],
        "params": lambda c, t: {"min_samples": max(4, 2 * c["dimensionality"]),
                                "eps": "read from the k-distance plot", "probability_threshold": 0.5},
    },
    "Bayesian GMM": {
        "reasons": ["Probabilistic: every point gets cluster membership probabilities",
                    "A Dirichlet-process prior can switch off unneeded components"],
        "risks": ["Sensitive to the priors and to the upper bound on components",
                  "Assumes roughly Gaussian clusters"],
        "metrics": ["log_likelihood", "bic", "mean_assignment_entropy"],
        "params": lambda c, t: {"n_components": f"upper bound {_k_range(c)[1] + 5}",
                                "weight_concentration_prior_type": "dirichlet_process"},
    },
}

# ---------------------------------------------------------------------------
# Matching: data type -> [(algorithm, base score, type-specific reason)]
# ---------------------------------------------------------------------------
MATCHING = {
    "text": [
        ("K-Means", 0.85, "Works well on L2-normalized TF-IDF vectors (approximates spherical K-Means)"),
        ("Hierarchical Clustering", 0.75, "Shows topic hierarchy; cosine distance suits sparse text vectors"),
        ("DBSCAN", 0.70, "With cosine distance it separates dense topic groups and isolates off-topic documents"),
    ],
    "time_series": [
        ("DTW-based K-Means", 0.85, "Designed for series whose shapes match but are shifted or stretched in time"),
        ("DBSCAN", 0.80, "Can group series or windows of irregular shape and flags anomalous periods as noise"),
        ("SOM", 0.75, "Preserves the temporal structure of the series on a 2D map"),
    ],
    "numerical": [
        ("K-Means", 0.85, "Standard, efficient choice for continuous features"),
        ("GMM", 0.80, "Handles elliptical clusters and overlapping groups in continuous features"),
        ("Spectral Clustering", 0.70, "Captures non-convex structure in continuous features"),
    ],
    "categorical": [
        ("K-Modes", 0.85, "Built for categorical attributes (modes instead of means)"),
        ("ROCK", 0.75, "Link-based similarity is designed for categorical / boolean attributes"),
        ("Categorical DBSCAN", 0.70, "Density-based clustering with a categorical distance"),
    ],
    "multimedia": [
        ("K-Means", 0.85, "Run on embeddings (e.g. CNN / audio features); embeddings make Euclidean distance meaningful"),
        ("DBSCAN", 0.75, "On embeddings with cosine distance it finds near-duplicate groups and outliers"),
    ],
    "uncertain": [
        ("Bayesian GMM", 0.85, "Probabilistic method that represents uncertainty in the cluster assignment"),
        ("UK-Means", 0.80, "Uncertainty-aware variant of K-Means that uses per-value uncertainty"),
        ("FDBSCAN", 0.75, "Uncertainty-aware variant of DBSCAN using fuzzy distances"),
    ],
}

# ---------------------------------------------------------------------------
# Data-dependent adjustments
# ---------------------------------------------------------------------------
QUADRATIC = {"Hierarchical Clustering", "Spectral Clustering", "ROCK", "DTW-based K-Means"}
NOISE_SENSITIVE = {"K-Means", "GMM", "UK-Means", "DTW-based K-Means", "K-Modes"}
NOISE_ROBUST = {"DBSCAN", "Categorical DBSCAN", "FDBSCAN"}
HIGH_DIM_SENSITIVE = {"K-Means", "DBSCAN", "GMM"}
SMALL_SAMPLE_SENSITIVE = {"GMM", "Bayesian GMM", "SOM"}


def adjust(algo, c):
    """Return (score change, extra reasons, extra risks) for this dataset."""
    delta, reasons, risks = 0.0, [], []
    n, dim = c["sample_size"], c["dimensionality"]
    if algo in QUADRATIC and n > 10000:
        delta -= 0.15
        risks.append(f"With {n} rows the quadratic cost is high; consider sampling or an approximation")
    if algo in SMALL_SAMPLE_SENSITIVE and n < 50:
        delta -= 0.15
        risks.append(f"Only {n} rows: too few to fit this model reliably")
    if c["noise_level"] == "high":
        if algo in NOISE_ROBUST:
            delta += 0.05
            reasons.append("Noise level is high; this method treats outliers as noise")
        elif algo in NOISE_SENSITIVE:
            delta -= 0.10
            risks.append("High noise level: outliers will distort cluster centres")
    if algo in HIGH_DIM_SENSITIVE and dim > 20 and "text" not in c["detected_types"]:
        delta -= 0.05
        risks.append(f"{dim} dimensions: distances lose contrast; reduce dimensionality (PCA) first")
    return delta, reasons, risks


# ---------------------------------------------------------------------------
# Dataset analysis
# ---------------------------------------------------------------------------
def estimate_noise(df, numeric_cols):
    cols = [c for c in numeric_cols if df[c].nunique() > 1]
    if not cols:
        return "unknown"
    share = pd.concat([iqr_outliers(df[c]) for c in cols], axis=1).any(axis=1).mean()
    return "low" if share < 0.03 else "medium" if share < 0.10 else "high"


def analyze_dataset(df, data_types=None):
    """Detect data types and basic characteristics.
    Multimedia and uncertain data are detected heuristically (file extensions /
    column names); pass data_types=[...] to override detection."""
    p = profile(df)
    media = [c for c in p["text_columns"] + p["categorical_columns"]
             if df[c].dropna().astype(str).str.contains(MEDIA_RE).mean() >= 0.8]
    text = [c for c in p["text_columns"] if c not in media]
    categorical = [c for c in p["categorical_columns"] if c not in media]
    numeric = p["numerical_columns"]
    uncertainty = [c for c in numeric if UNCERTAIN_RE.search(c)]

    tp = TimePreprocessing()
    ts_cols = tp.identify_timestamp_columns(df)
    ts_likely = [c for c in ts_cols if tp.check_time_series(df, c)["likely_time_series"]]

    found = {
        "time_series": ts_likely,
        "uncertain": uncertainty,
        "multimedia": media,
        "text": text,
        "categorical": categorical,
        "numerical": [c for c in numeric if c not in uncertainty] or (numeric if not uncertainty else []),
    }
    detected = [t for t in TYPE_PRIORITY if found[t]]
    if data_types:
        unknown = set(data_types) - set(TYPE_PRIORITY)
        if unknown:
            raise ValueError(f"Unknown data types: {sorted(unknown)}")
        detected = [t for t in TYPE_PRIORITY if t in data_types]
    if not detected:
        raise ValueError("No supported data type detected.")

    return {
        "detected_types": detected,
        "characteristics": {
            "dimensionality": df.shape[1] - len(ts_cols),
            "sample_size": len(df),
            "temporal_patterns": bool(ts_likely),
            "noise_level": estimate_noise(df, numeric),
        },
        "columns": {t: found[t] for t in detected if found[t]},
        "_uncertainty_columns": uncertainty,
    }


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
def recommend(df, data_types=None, top_n=3):
    analysis = analyze_dataset(df, data_types)
    c = dict(analysis["characteristics"])
    c["detected_types"] = analysis["detected_types"]
    c["uncertainty_columns"] = analysis.pop("_uncertainty_columns")
    primary = analysis["detected_types"][0]

    # best match per algorithm over all detected types
    candidates = {}
    for t in analysis["detected_types"]:
        weight = 1.0 if t == primary else SECONDARY_TYPE_WEIGHT
        for algo, base, type_reason in MATCHING[t]:
            score = base * weight
            if algo not in candidates or score > candidates[algo]["score"]:
                candidates[algo] = {"score": score, "type": t, "type_reason": type_reason}

    recs = []
    for algo, m in candidates.items():
        kb = ALGORITHMS[algo]
        delta, extra_reasons, extra_risks = adjust(algo, c)
        risks = list(kb["risks"]) + extra_risks[:1]          # 2 fixed + at most 1 data-specific
        recs.append({
            "algorithm": algo,
            "score": round(min(1.0, max(0.0, m["score"] + delta)), 2),
            "why": [m["type_reason"]] + kb["reasons"] + extra_reasons,
            "risks": risks,
            "evaluation_metrics": kb["metrics"],
            "recommended_parameters": kb["params"](c, m["type"]),
        })
    recs.sort(key=lambda r: -r["score"])
    recs = recs[:top_n]
    for i, r in enumerate(recs, 1):
        r["rank"] = i
        r.update({k: r.pop(k) for k in ("score", "why", "risks", "evaluation_metrics", "recommended_parameters")})

    return {
        "recommendations": recs,
        "dataset_analysis": {k: analysis[k] for k in ("detected_types", "characteristics", "columns")},
    }


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    n = 200
    demo = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "sales": rng.normal(100, 10, n).cumsum(),
        "visitors": rng.normal(500, 50, n),
        "temperature": rng.normal(15, 5, n),
    })
    print(json.dumps(recommend(demo), indent=2, default=str))

    # Acceptance check: every data type gets recommendations with >=2 reasons,
    # 2-3 risks and at least 2 evaluation metrics
    for t in TYPE_PRIORITY:
        result = recommend(demo, data_types=[t], top_n=10)
        names = [r["algorithm"] for r in result["recommendations"]]
        assert names, t
        assert {a for a, *_ in MATCHING[t]} == set(names), (t, names)
        for r in result["recommendations"]:
            assert len(r["why"]) >= 2 and 2 <= len(r["risks"]) <= 3 and len(r["evaluation_metrics"]) >= 2, r
        print(f"{t:12s} OK -> {names}")