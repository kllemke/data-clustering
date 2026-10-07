"""Rule-based algorithm recommendations from detected data characteristics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import pandas as pd

from .detection.categorical import analyze_categorical_column
from .detection.multimedia import analyze_multimedia_references
from .detection.text import analyze_text_column
from .detection.time_series_dataset import analyze_time_series_dataset
from .detection.numerical_uncertainty import analyze_uncertainty


@dataclass
class AlgorithmRecommendation:
    algorithm: str
    rationale: str
    preprocessing: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def recommend_algorithms(dataframe: pd.DataFrame) -> List[AlgorithmRecommendation]:
    """Suggest clustering approaches; recommendations do not train or run models."""
    recommendations: List[AlgorithmRecommendation] = []
    time_profiles = analyze_time_series_dataset(dataframe)
    text_columns = [
        str(column)
        for column in dataframe.columns
        if (
            (profile := analyze_text_column(dataframe[column])).is_text
            and (profile.average_word_count >= 3 or profile.average_text_length >= 25)
        )
    ]
    numeric_columns = [
        str(column) for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    categorical_columns = [
        str(column) for column in dataframe.columns if analyze_categorical_column(dataframe[column]).is_categorical
    ]
    media_types = {
        media_type
        for column in dataframe.columns
        for media_type in analyze_multimedia_references(dataframe[column]).file_type_counts
    }

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
            for score in profile.predictability.values()
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
    uncertainty = analyze_uncertainty(dataframe)
    if uncertainty.confidence_score_columns or uncertainty.confidence_interval_pairs:
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
