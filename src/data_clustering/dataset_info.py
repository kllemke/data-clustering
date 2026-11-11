"""Utilities for collecting and printing basic dataset information."""

from __future__ import annotations

from typing import Any

import pandas as pd
from .detection.column_types import identify_column_types


PREVIEW_ROW_COUNT = 5


def get_dataset_info(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return a preview, pandas dtypes, dimensions and detected column groups.

    The result contains only data, so it can be reused by a terminal interface
    or a future web interface. Unsupported columns are reported, not removed.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    column_groups = identify_column_types(dataframe)
    row_count, column_count = dataframe.shape
    missing_by_column = dataframe.isna().sum()
    try:
        duplicate_rows = int(dataframe.duplicated().sum())
    except (TypeError, ValueError):
        # Nested unsupported values may not be hashable for duplicate detection.
        duplicate_rows = None

    dataset_info = {
        "first_rows": dataframe.head(PREVIEW_ROW_COUNT).copy(),
        "column_names": dataframe.columns.tolist(),
        "data_types": dataframe.dtypes.copy(),
        "row_count": row_count,
        "column_count": column_count,
        "missing_values_detected": int(missing_by_column.sum()),
        "missing_values_by_column": missing_by_column.to_dict(),
        "duplicate_rows_detected": duplicate_rows,
    }
    dataset_info.update(column_groups)
    return dataset_info


def print_dataset_info(dataset_info: dict[str, Any]) -> None:
    """Print previously collected dataset information in the terminal."""
    print("First 5 rows:")
    print(dataset_info["first_rows"].to_string(index=False))

    print("\nColumn names:")
    column_names = dataset_info["column_names"]
    if column_names:
        for column_name in column_names:
            print(f"- {column_name}")
    else:
        print("No columns")

    print("\nData types:")
    data_types = dataset_info["data_types"]
    if not data_types.empty:
        for column_name, data_type in data_types.items():
            print(f"- {column_name}: {data_type}")
    else:
        print("No data types")

    print("\nDetected column groups:")
    for label, key in (
        ("Numerical columns", "numerical_columns"),
        ("Categorical columns", "categorical_columns"),
        ("Boolean columns", "boolean_columns"),
        ("Unsupported columns", "unsupported_columns"),
    ):
        names = ", ".join(str(name) for name in dataset_info[key]) or "None"
        print(f"{label}: {names}")

    if dataset_info["unsupported_columns"]:
        print("Warning: unsupported columns must be converted or removed before preprocessing.")

    print(f"\nTotal rows: {dataset_info['row_count']}")
    print(f"Total columns: {dataset_info['column_count']}")


def display_dataset_info(dataframe: pd.DataFrame) -> None:
    """Collect and print basic information about a DataFrame."""
    dataset_info = get_dataset_info(dataframe)
    print_dataset_info(dataset_info)
