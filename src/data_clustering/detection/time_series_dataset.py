"""Dataset-level time-series characteristics and predictability indicators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .time_series import analyze_time_series_column


@dataclass
class TimeSeriesDatasetProfile:
    time_column: str
    value_columns: List[str]
    frequency_seconds: float | None
    intervals_are_regular: bool
    trends: Dict[str, str]
    seasonal_lags: Dict[str, int | None]
    lag_one_autocorrelation: Dict[str, float | None]
    predictability: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_time_series_dataset(dataframe: pd.DataFrame) -> List[TimeSeriesDatasetProfile]:
    """Analyze numeric signals against detected time columns."""
    time_columns = [
        str(column)
        for column in dataframe.columns
        if analyze_time_series_column(dataframe[column]).is_datetime_like
    ]
    numeric_columns = [
        str(column)
        for column in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    profiles: List[TimeSeriesDatasetProfile] = []

    for time_column in time_columns:
        timestamps = pd.to_datetime(dataframe[time_column], errors="coerce").dropna()
        differences = timestamps.sort_values().diff().dropna().dt.total_seconds()
        frequency = float(differences.median()) if not differences.empty else None
        regular = bool(
            len(differences) >= 2
            and differences.std(ddof=0) <= max(1.0, abs(differences.mean()) * 0.1)
        )
        trends: Dict[str, str] = {}
        seasonal_lags: Dict[str, int | None] = {}
        autocorrelation: Dict[str, float | None] = {}
        predictability: Dict[str, float] = {}

        for value_column in numeric_columns:
            paired = pd.DataFrame(
                {"time": pd.to_datetime(dataframe[time_column], errors="coerce"), "value": dataframe[value_column]}
            ).dropna().sort_values("time")
            values = paired["value"].to_numpy(dtype=float)
            if len(values) < 3:
                continue
            centered = values - values.mean()
            lag_one = (
                float(np.corrcoef(centered[:-1], centered[1:])[0, 1])
                if len(values) > 2 and np.std(centered[:-1]) > 0 and np.std(centered[1:]) > 0
                else 0.0
            )
            lag_one = float(np.nan_to_num(lag_one))
            slope = float(np.polyfit(np.arange(len(values)), values, 1)[0])
            trends[value_column] = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            autocorrelation[value_column] = lag_one

            max_lag = min(len(values) // 2, 24)
            candidates: List[tuple[float, int]] = []
            for lag in range(2, max_lag + 1):
                left, right = centered[:-lag], centered[lag:]
                if np.std(left) > 0 and np.std(right) > 0:
                    candidates.append((float(np.corrcoef(left, right)[0, 1]), lag))
            seasonal_lags[value_column] = (
                max(candidates, key=lambda item: abs(item[0]))[1]
                if candidates and max(abs(item[0]) for item in candidates) >= 0.5
                else None
            )

            residual = values - np.polyval(np.polyfit(np.arange(len(values)), values, 1), np.arange(len(values)))
            variance = float(np.var(values))
            trend_fit = 1.0 - float(np.var(residual)) / variance if variance > 0 else 0.0
            predictability[value_column] = round(
                float(np.clip(0.5 * abs(lag_one) + 0.5 * max(0.0, trend_fit), 0.0, 1.0)), 4
            )

        profiles.append(
            TimeSeriesDatasetProfile(
                time_column=time_column,
                value_columns=list(trends),
                frequency_seconds=frequency,
                intervals_are_regular=regular,
                trends=trends,
                seasonal_lags=seasonal_lags,
                lag_one_autocorrelation=autocorrelation,
                predictability=predictability,
            )
        )
    return profiles
