"""Data-clustering detection utilities."""

from .analysis import DatasetAnalysis, MediaAnalysis, analyze_dataframe, analyze_file, load_table
from .dataset_info import get_dataset_info, print_dataset_info, display_dataset_info
from .detection.column_types import identify_column_types
from .preprocessing import PreprocessingError, preprocess_dataset, print_preprocessing_report
from .recommendations import AlgorithmRecommendation, recommend_algorithms
from .detection.text import TextColumnProfile, analyze_text_column, detect_text_columns
from .detection.time_series import (TimeSeriesColumnProfile, TimeSeriesProfile, analyze_time_series,
                                    analyze_time_series_column, detect_time_series, detect_time_series_columns)

__all__ = [
    "TextColumnProfile",
    "analyze_text_column",
    "detect_text_columns",
    "TimeSeriesColumnProfile",
    "TimeSeriesProfile",
    "analyze_time_series",
    "analyze_time_series_column",
    "detect_time_series",
    "detect_time_series_columns",
    "DatasetAnalysis",
    "MediaAnalysis",
    "analyze_dataframe",
    "analyze_file",
    "AlgorithmRecommendation",
    "recommend_algorithms",
    "load_table",
    "get_dataset_info",
    "print_dataset_info",
    "display_dataset_info",
    "identify_column_types",
    "PreprocessingError",
    "preprocess_dataset",
    "print_preprocessing_report",
]
