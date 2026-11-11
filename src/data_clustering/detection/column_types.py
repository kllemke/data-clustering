"""Shared column classification for profiling and tabular preprocessing."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import (
    infer_dtype, is_bool_dtype, is_float_dtype, is_integer_dtype,
    is_object_dtype, is_string_dtype,
)


def classify_column(column: pd.Series) -> str:
    """Identify a basic kind without converting numeric-looking strings."""
    dtype = column.dtype
    if is_bool_dtype(dtype):
        return "boolean"
    if is_integer_dtype(dtype) or is_float_dtype(dtype):
        return "numerical"
    if is_object_dtype(dtype) or is_string_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
        inferred = infer_dtype(column.astype(object), skipna=True)
        if inferred in {"string", "empty", "integer", "floating", "mixed-integer-float", "boolean"}:
            if isinstance(dtype, pd.CategoricalDtype):
                return "categorical"
            if inferred == "boolean":
                return "boolean"
            if inferred in {"integer", "floating", "mixed-integer-float"}:
                return "numerical"
            return "categorical"
    return "unsupported"


def identify_column_types(dataframe: pd.DataFrame) -> dict[str, list[Any]]:
    """Group columns by the basic kind used by the tabular pipeline."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if not dataframe.columns.is_unique:
        raise ValueError("Column names must be unique.")
    groups = {f"{kind}_columns": [] for kind in (
        "numerical", "categorical", "boolean", "unsupported"
    )}
    for name, column in dataframe.items():
        groups[f"{classify_column(column)}_columns"].append(name)
    return groups
