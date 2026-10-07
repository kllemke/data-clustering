"""Profiling for numerical columns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import pandas as pd


@dataclass
class NumericalColumnProfile:
    column_name: str
    is_numeric: bool
    count: int
    missing_ratio: float
    mean: float | None
    standard_deviation: float | None
    minimum: float | None
    maximum: float | None
    unique_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_numerical_column(series: pd.Series) -> NumericalColumnProfile:
    """Compute basic distribution and missingness statistics for a column."""
    numeric = (
        pd.Series(dtype=float)
        if pd.api.types.is_datetime64_any_dtype(series)
        else pd.to_numeric(series, errors="coerce")
    )
    valid = numeric.dropna()
    missing_ratio = float(series.isna().mean()) if len(series) else 0.0

    def finite_or_none(value: Any) -> float | None:
        if pd.isna(value):
            return None
        return float(value)

    return NumericalColumnProfile(
        column_name=str(series.name or "unnamed"),
        is_numeric=bool(len(valid) > 0 and len(valid) / max(1, series.notna().sum()) >= 0.8),
        count=int(len(valid)),
        missing_ratio=missing_ratio,
        mean=finite_or_none(valid.mean()),
        standard_deviation=finite_or_none(valid.std()),
        minimum=finite_or_none(valid.min()),
        maximum=finite_or_none(valid.max()),
        unique_count=int(valid.nunique()),
    )
