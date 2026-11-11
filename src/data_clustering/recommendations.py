"""Rule-based algorithm recommendations from detected data characteristics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping

import pandas as pd

@dataclass
class AlgorithmRecommendation:
    algorithm: str
    rationale: str
    preprocessing: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def recommend_algorithms(
    dataframe: pd.DataFrame | None = None, *, analysis: Mapping[str, Any] | None = None
) -> List[AlgorithmRecommendation]:
    """Suggest approaches from an existing analysis, or analyze a DataFrame once."""
    if analysis is None:
        if dataframe is None:
            raise TypeError("Provide a DataFrame or an analysis report.")
        # Import locally to keep the public DataFrame API without a circular import.
        from .analysis import analyze_dataframe
        return analyze_dataframe(dataframe).recommendations

    recommendations: List[AlgorithmRecommendation] = []
    columns = analysis.get("columns", {})
    time_profiles = analysis.get("time_series", [])
    text_columns = [
        name for name, profile in columns.items()
        if profile.get("text", {}).get("is_text")
        and (profile["text"]["average_word_count"] >= 3
             or profile["text"]["average_text_length"] >= 25)
    ]
    numeric_columns = [
        name for name, profile in columns.items() if profile["kind"] == "numerical"
    ]
    categorical_columns = [
        name for name, profile in columns.items()
        if profile["kind"] in {"categorical", "boolean"}
        and name not in text_columns and not profile.get("is_datetime")
        and not profile.get("multimedia")
    ]
    media_types = set((analysis.get("multimedia") or {}).get("file_type_counts", {}))

    if text_columns:
        recommendations.append(
            AlgorithmRecommendation(
                "TF-IDF + K-means",
                f"Natural-language text columns detected: {', '.join(text_columns)}.",
                ["Vectorize text with TF-IDF", "Normalize sparse vectors", "Choose cluster count"],
            )
        )
    if time_profiles:
        predictability = [
            score
            for profile in time_profiles
            for score in profile["predictability"].values()
        ]
        rationale = f"Time-indexed numerical signals detected in {len(time_profiles)} time column(s)."
        if predictability:
            rationale += f" Average predictability indicator: {sum(predictability) / len(predictability):.2f}."
        recommendations.append(
            AlgorithmRecommendation(
                "Time-series feature clustering (DTW or k-Shape)",
                rationale,
                ["Align or resample time intervals", "Handle missing observations", "Scale each signal"],
            )
        )
    if numeric_columns:
        recommendations.append(
            AlgorithmRecommendation(
                "K-means or Gaussian Mixture Model",
                f"Numerical features detected: {', '.join(numeric_columns)}.",
                ["Impute missing values", "Scale numerical features", "Validate cluster count and stability"],
            )
        )
    if categorical_columns:
        recommendations.append(
            AlgorithmRecommendation(
                "K-modes or hierarchical clustering with Gower distance",
                f"Categorical features detected: {', '.join(categorical_columns)}.",
                ["Encode categories or use a mixed-type distance", "Consider reducing high-cardinality categories"],
            )
        )
    if media_types:
        recommendations.append(
            AlgorithmRecommendation(
                "Embedding-based clustering (K-means or HDBSCAN)",
                f"References to multimedia types were detected: {', '.join(sorted(media_types))}.",
                ["Extract embeddings using a modality-specific pretrained model", "Normalize embedding vectors"],
            )
        )
    uncertainty = analysis.get("uncertainty", {})
    if uncertainty.get("confidence_score_columns") or uncertainty.get("confidence_interval_pairs"):
        recommendations.append(
            AlgorithmRecommendation(
                "Gaussian Mixture Model with uncertainty-aware review",
                "Confidence scores or interval bounds are present; preserve confidence information when evaluating assignments.",
                ["Validate score semantics and ranges", "Keep interval bounds paired", "Review low-confidence assignments"],
            )
        )
    if not recommendations:
        recommendations.append(
            AlgorithmRecommendation(
                "Manual feature review before clustering",
                "No supported clustering feature type was detected reliably.",
                ["Inspect column types and sample values", "Provide a tabular CSV, TSV, Excel, JSON, or Parquet dataset"],
            )
        )
    return recommendations
