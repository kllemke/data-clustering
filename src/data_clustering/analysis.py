"""End-to-end file loading, data profiling, and recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd

from .detection.categorical import analyze_categorical_column
from .detection.multimedia import MultimediaProfile, analyze_multimedia_references, inspect_media_path
from .detection.numerical import analyze_numerical_column
from .detection.numerical_uncertainty import UncertaintyProfile, analyze_uncertainty
from .detection.text import analyze_text_column
from .detection.time_series_dataset import TimeSeriesDatasetProfile, analyze_time_series_dataset
from .recommendations import AlgorithmRecommendation, recommend_algorithms


TABLE_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm", ".json", ".parquet", ".pq"}
MEDIA_EXTENSIONS = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
    ".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav", ".wma",
    ".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv",
}


@dataclass
class DatasetAnalysis:
    source: str | None
    row_count: int
    column_count: int
    data_types: List[str]
    columns: Dict[str, Dict[str, Any]]
    uncertainty: UncertaintyProfile
    time_series: List[TimeSeriesDatasetProfile]
    recommendations: List[AlgorithmRecommendation]
    multimedia: MultimediaProfile | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "tabular",
            "source": self.source,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "data_types": self.data_types,
            "columns": self.columns,
            "uncertainty": self.uncertainty.to_dict(),
            "time_series": [profile.to_dict() for profile in self.time_series],
            "multimedia": self.multimedia.to_dict() if self.multimedia else None,
            "recommendations": [recommendation.to_dict() for recommendation in self.recommendations],
        }


@dataclass
class MediaAnalysis:
    source: str
    media: MultimediaProfile
    recommendations: List[AlgorithmRecommendation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "multimedia",
            "source": self.source,
            "data_types": ["multimedia"] if self.media.files else [],
            "media": self.media.to_dict(),
            "recommendations": [recommendation.to_dict() for recommendation in self.recommendations],
        }


def analyze_dataframe(
    dataframe: pd.DataFrame,
    *,
    source: str | None = None,
    ordinal_orders: Mapping[str, Iterable[Any]] | None = None,
) -> DatasetAnalysis:
    """Analyze a tabular dataset across supported data types."""
    profiles: Dict[str, Dict[str, Any]] = {}
    detected_types: set[str] = set()
    media_files = []
    for column in dataframe.columns:
        series = dataframe[column]
        text = analyze_text_column(series)
        numeric = analyze_numerical_column(series)
        categorical = analyze_categorical_column(
            series,
            ordinal_order=(ordinal_orders or {}).get(str(column)),
        )
        profiles[str(column)] = {
            "dtype": str(series.dtype),
            "text": text.to_dict(),
            "numerical": numeric.to_dict(),
            "categorical": categorical.to_dict(),
        }
        if text.is_text:
            detected_types.add("text")
        if numeric.is_numeric:
            detected_types.add("numerical")
        if categorical.is_categorical:
            detected_types.add("categorical")
        multimedia = analyze_multimedia_references(series)
        if multimedia.files:
            detected_types.add("multimedia")
            media_files.extend(multimedia.files)

    uncertainty = analyze_uncertainty(dataframe)
    if uncertainty.confidence_score_columns or uncertainty.confidence_interval_pairs:
        detected_types.add("uncertain/probabilistic")
    time_series = analyze_time_series_dataset(dataframe)
    if time_series:
        detected_types.add("time-series")

    multimedia_profile = None
    if media_files:
        media_counts: Dict[str, int] = {}
        for file_profile in media_files:
            media_counts[file_profile.media_type] = media_counts.get(file_profile.media_type, 0) + 1
        feature_dimensions = [
            item.feature_dimensions for item in media_files if item.feature_dimensions is not None
        ]
        multimedia_profile = MultimediaProfile(
            files=media_files,
            file_type_counts=media_counts,
            total_size_bytes=sum(item.size_bytes for item in media_files),
            feature_dimensions=max(feature_dimensions) if feature_dimensions else None,
        )

    recommendations = recommend_algorithms(dataframe)
    return DatasetAnalysis(
        source=source,
        row_count=int(dataframe.shape[0]),
        column_count=int(dataframe.shape[1]),
        data_types=sorted(detected_types),
        columns=profiles,
        uncertainty=uncertainty,
        time_series=time_series,
        recommendations=recommendations,
        multimedia=multimedia_profile,
    )


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.json_normalize(pd.read_json(path, typ="series").to_dict())
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported tabular format: {suffix or '(no extension)'}")


def analyze_file(path: str | Path) -> DatasetAnalysis | MediaAnalysis:
    """Analyze a supported tabular file, multimedia file, or multimedia directory.

    Supported tables are CSV, TSV, Excel, JSON, and Parquet. Directories are
    traversed for supported image, audio, and video files.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")

    if source.is_dir():
        media = inspect_media_path(source)
        return MediaAnalysis(
            source=str(source),
            media=media,
            recommendations=_media_recommendations(media),
        )
    suffix = source.suffix.lower()
    if suffix in TABLE_EXTENSIONS:
        dataframe = _load_table(source)
        return analyze_dataframe(dataframe, source=str(source))
    if suffix in MEDIA_EXTENSIONS:
        media = inspect_media_path(source)
        return MediaAnalysis(
            source=str(source),
            media=media,
            recommendations=_media_recommendations(media),
        )
    raise ValueError(
        f"Unsupported file type '{suffix or '(no extension)'}'. "
        "Supported inputs: CSV, TSV, TXT, Excel, JSON, Parquet, images, audio, and video."
    )


def _media_recommendations(media: MultimediaProfile) -> List[AlgorithmRecommendation]:
    if not media.files:
        return [
            AlgorithmRecommendation(
                "Manual multimedia feature extraction",
                "No supported image, audio, or video files were found in the input.",
                ["Provide supported media files", "Extract embeddings using a task-appropriate pretrained model"],
            )
        ]
    return [
        AlgorithmRecommendation(
            "Embedding-based clustering (K-means or HDBSCAN)",
            "Multimedia files require feature extraction before clustering; raw file bytes are not meaningful feature vectors.",
            ["Extract image, audio, or video embeddings with a suitable pretrained model", "Normalize embeddings", "Compare cluster quality"],
        )
    ]
