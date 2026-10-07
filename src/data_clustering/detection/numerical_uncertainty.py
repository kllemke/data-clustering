"""Detection and profiling of uncertainty indicators."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd


@dataclass
class UncertaintyProfile:
    confidence_score_columns: List[str]
    confidence_interval_pairs: List[Tuple[str, str]]
    numeric_uncertainty: Dict[str, float]
    missing_ratios: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_uncertainty(dataframe: pd.DataFrame) -> UncertaintyProfile:
    """Find confidence-like fields, interval bounds, numeric spread, and missingness."""
    confidence_pattern = re.compile(r"confidence|probability|certainty|score|_prob$", re.IGNORECASE)
    lower_pattern = re.compile(r"^(.*?)(?:_lower|_low|_min|_lcl)$", re.IGNORECASE)
    upper_pattern = re.compile(r"^(.*?)(?:_upper|_high|_max|_ucl)$", re.IGNORECASE)
    lower_bounds: Dict[str, str] = {}
    upper_bounds: Dict[str, str] = {}
    confidence_columns: List[str] = []
    uncertainty: Dict[str, float] = {}
    missing: Dict[str, float] = {}

    for column in dataframe.columns:
        name = str(column)
        series = dataframe[column]
        missing[name] = float(series.isna().mean()) if len(series) else 0.0
        if confidence_pattern.search(name):
            confidence_columns.append(name)
        lower = lower_pattern.match(name)
        upper = upper_pattern.match(name)
        if lower:
            lower_bounds[lower.group(1).lower()] = name
        if upper:
            upper_bounds[upper.group(1).lower()] = name

        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) >= 2:
            spread = float(numeric.std(ddof=0))
            scale = float(abs(numeric.mean()))
            uncertainty[name] = spread / scale if scale > 1e-12 else spread

    interval_pairs = [
        (lower_bounds[key], upper_bounds[key])
        for key in sorted(lower_bounds.keys() & upper_bounds.keys())
    ]
    return UncertaintyProfile(confidence_columns, interval_pairs, uncertainty, missing)
