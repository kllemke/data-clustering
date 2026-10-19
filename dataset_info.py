"""Utilities for collecting and printing basic dataset information."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import (
    infer_dtype,
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
)


PREVIEW_ROW_COUNT = 5


def identify_column_types(dataframe: pd.DataFrame) -> dict[str, list[Any]]:
    """Group columns as numerical, categorical, boolean or unsupported.

    Object columns are checked for homogeneous scalar values. Numeric-looking
    strings remain categorical; this function does not convert any data.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    column_types = {
        "numerical_columns": [],
        "categorical_columns": [],
        "boolean_columns": [],
        "unsupported_columns": [],
    }

    for column_name, column in dataframe.items():
        data_type = column.dtype
        column_group = "unsupported_columns"

        # Check booleans before numerical types to keep them separate.
        if is_bool_dtype(data_type):
            column_group = "boolean_columns"
        elif is_integer_dtype(data_type) or is_float_dtype(data_type):
            column_group = "numerical_columns"
        elif (
            is_object_dtype(data_type)
            or is_string_dtype(data_type)
            or isinstance(data_type, pd.CategoricalDtype)
        ):
            inferred_type = infer_dtype(column.astype(object), skipna=True)
            scalar_types = {
                "string", "empty", "integer", "floating",
                "mixed-integer-float", "boolean",
            }
            if inferred_type in scalar_types:
                if isinstance(data_type, pd.CategoricalDtype):
                    column_group = "categorical_columns"
                elif inferred_type == "boolean":
                    column_group = "boolean_columns"
                elif inferred_type in {"integer", "floating", "mixed-integer-float"}:
                    column_group = "numerical_columns"
                else:
                    column_group = "categorical_columns"

        column_types[column_group].append(column_name)

    return column_types


def get_dataset_info(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return a preview, pandas dtypes, dimensions and detected column groups.

    The result contains only data, so it can be reused by a terminal interface
    or a future web interface. Unsupported columns are reported, not removed.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    row_count, column_count = dataframe.shape

    dataset_info = {
        "first_rows": dataframe.head(PREVIEW_ROW_COUNT).copy(),
        "column_names": dataframe.columns.tolist(),
        "data_types": dataframe.dtypes.copy(),
        "row_count": row_count,
        "column_count": column_count,
    }
    dataset_info.update(identify_column_types(dataframe))
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
