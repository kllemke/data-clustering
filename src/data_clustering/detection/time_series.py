"""Utilities for detecting time-series candidate columns."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class TimeSeriesColumnProfile:
    """Summary of how strongly a column resembles a time-series signal."""

    column_name: str
    is_time_series: bool
    is_datetime_like: bool
    is_chronological: bool
    has_repeated_measurements: bool
    has_regular_intervals: bool
    datetime_ratio: float
    ordered_ratio: float
    interval_variation: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_name": self.column_name,
            "is_time_series": self.is_time_series,
            "is_datetime_like": self.is_datetime_like,
            "is_chronological": self.is_chronological,
            "has_repeated_measurements": self.has_repeated_measurements,
            "has_regular_intervals": self.has_regular_intervals,
            "datetime_ratio": round(self.datetime_ratio, 4),
            "ordered_ratio": round(self.ordered_ratio, 4),
            "interval_variation": round(self.interval_variation, 4),
            "reasons": self.reasons,
        }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _parse_datetime_values(series: pd.Series) -> pd.Series:
    """Attempt to coerce strings and objects into datetime values."""
    if series.empty:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce", utc=False)

    if pd.api.types.is_numeric_dtype(series):
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    parsed_values: List[pd.Timestamp | pd.NaT] = []
    for value in series:
        if pd.isna(value):
            parsed_values.append(pd.NaT)
            continue

        if isinstance(value, str):
            text = value.strip()
            if not text:
                parsed_values.append(pd.NaT)
                continue

            date_pattern = re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}", text)
            if date_pattern is None:
                parsed_values.append(pd.NaT)
                continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed_values.append(pd.to_datetime(value, errors="coerce", utc=False))

    return pd.Series(parsed_values, index=series.index, dtype="datetime64[ns]")


def analyze_time_series_column(series: pd.Series, *, min_datetime_ratio: float = 0.7, min_ordered_ratio: float = 0.7) -> TimeSeriesColumnProfile:
    """Analyze if a Series looks like a time-based or time-series column."""
    values = pd.Series(series, copy=True)
    column_name = values.name if values.name is not None else "unnamed"

    non_null = values.dropna()
    parsed = _parse_datetime_values(non_null)
    valid_datetime_mask = parsed.notna()
    datetime_values = parsed[valid_datetime_mask]
    datetime_ratio = _safe_ratio(len(datetime_values), len(non_null))

    is_datetime_like = datetime_ratio >= min_datetime_ratio

    ordered_ratio = 0.0
    is_chronological = False
    if not datetime_values.empty:
        sorted_values = datetime_values.sort_values(ignore_index=True)
        ordered_matches = sum(1 for left, right in zip(datetime_values.to_list(), sorted_values.to_list()) if left == right)
        ordered_ratio = _safe_ratio(ordered_matches, len(datetime_values))
        is_chronological = ordered_ratio >= min_ordered_ratio or datetime_values.is_monotonic_increasing

    has_repeated_measurements = False
    if not datetime_values.empty:
        duplicate_count = datetime_values.duplicated().sum()
        has_repeated_measurements = duplicate_count > 0

    interval_variation = 0.0
    has_regular_intervals = False
    if len(datetime_values) >= 2:
        diffs = datetime_values.sort_values().diff().dropna()
        if not diffs.empty:
            mean_diff = diffs.mean()
            if pd.notna(mean_diff) and mean_diff != 0:
                interval_variation = float(diffs.std() / mean_diff)
                has_regular_intervals = interval_variation < 0.5

    reasons: List[str] = []
    if not is_datetime_like:
        reasons.append("Too few values parse as date or timestamp values.")
    if not is_chronological:
        reasons.append("Values do not follow a clear chronological order.")
    if not has_repeated_measurements and not is_datetime_like:
        reasons.append("There are no repeated observations to suggest measurement over time.")
    if not has_regular_intervals and len(datetime_values) >= 3:
        reasons.append("Intervals are not regular enough for a typical time-series signal.")

    is_time_series = (
        is_datetime_like and is_chronological
    ) or (
        is_datetime_like and len(datetime_values) >= 2 and (has_repeated_measurements or has_regular_intervals)
    )

    return TimeSeriesColumnProfile(
        column_name=column_name,
        is_time_series=is_time_series,
        is_datetime_like=is_datetime_like,
        is_chronological=is_chronological,
        has_repeated_measurements=has_repeated_measurements,
        has_regular_intervals=has_regular_intervals,
        datetime_ratio=datetime_ratio,
        ordered_ratio=ordered_ratio,
        interval_variation=interval_variation,
        reasons=reasons,
    )


def detect_time_series_columns(dataframe: pd.DataFrame) -> Dict[str, TimeSeriesColumnProfile]:
    """Detect columns that look like time-based or time-series signals."""
    return {
        column_name: analyze_time_series_column(series)
        for column_name, series in dataframe.items()
    }


TimeSeriesProfile = TimeSeriesColumnProfile
analyze_time_series = analyze_time_series_column
detect_time_series = detect_time_series_columns
