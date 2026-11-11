"""Profiling for nominal, ordinal, and hierarchical categories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List

import pandas as pd

from .column_types import classify_column


@dataclass
class CategoricalColumnProfile:
    column_name: str
    is_categorical: bool
    category_type: str
    cardinality: int
    cardinality_ratio: float
    missing_ratio: float
    frequencies: Dict[str, int]
    is_hierarchical: bool
    hierarchy_separator: str | None
    low_cardinality_numeric: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_categorical_column(
    series: pd.Series,
    *,
    ordinal_order: Iterable[Any] | None = None,
    max_cardinality_ratio: float = 0.5,
) -> CategoricalColumnProfile:
    """Analyze category distribution and optionally apply a known ordinal order."""
    values = series.dropna().astype(str)
    cardinality = int(values.nunique())
    ratio = cardinality / len(values) if len(values) else 0.0
    kind = classify_column(series)
    inferred_order = (
        _infer_ordinal_order(series)
        if ordinal_order is None and kind in {"categorical", "boolean"}
        else None
    )
    category_type = "ordinal" if ordinal_order is not None or inferred_order is not None else "nominal"
    hierarchy_separator = next(
        (separator for separator in ("/", ">", "|", "\\") if values.str.contains(separator, regex=False).any()),
        None,
    )
    if ordinal_order is not None:
        ordered_values = [str(value) for value in ordinal_order]
        observed = set(values)
        ordered_values = [value for value in ordered_values if value in observed]
        if len(ordered_values) < cardinality:
            ordered_values.extend(value for value in sorted(observed) if value not in ordered_values)
        inferred_order = ordered_values
    if inferred_order is not None:
        category_type = "ordinal"

    is_categorical = kind in {"categorical", "boolean"}

    return CategoricalColumnProfile(
        column_name=str(series.name or "unnamed"),
        is_categorical=is_categorical,
        category_type=category_type,
        cardinality=cardinality,
        cardinality_ratio=ratio,
        missing_ratio=float(series.isna().mean()) if len(series) else 0.0,
        frequencies={str(key): int(count) for key, count in values.value_counts().items()},
        is_hierarchical=hierarchy_separator is not None,
        hierarchy_separator=hierarchy_separator,
        low_cardinality_numeric=bool(
            kind == "numerical" and len(values) and ratio <= max_cardinality_ratio
        ),
    )


def _infer_ordinal_order(series: pd.Series) -> List[str] | None:
    """Infer ordinal categories from ordered pandas categoricals or common scales."""
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype) and dtype.ordered:
        return [str(value) for value in dtype.categories]

    observed = {str(value).strip().lower() for value in series.dropna().unique()}
    known_orders = (
        ("very low", "low", "medium", "high", "very high"),
        ("very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"),
        ("strongly disagree", "disagree", "neither agree nor disagree", "agree", "strongly agree"),
        ("small", "medium", "large", "extra large"),
        ("bronze", "silver", "gold", "platinum"),
    )
    for order in known_orders:
        if len(observed) >= 2 and observed.issubset(order):
            return [value for value in order if value in observed]
    return None
