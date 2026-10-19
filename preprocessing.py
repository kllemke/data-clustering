"""Simple preprocessing utilities for clustering datasets.

The preprocessing logic returns structured data that can be reused by a
terminal application now and by a web interface later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataset_info import identify_column_types


class PreprocessingError(ValueError):
    """Raised when a dataset cannot be preprocessed safely."""


def detect_column_types(dataframe: pd.DataFrame) -> tuple[list[Any], list[Any]]:
    """Return numerical columns and categories, including booleans for encoding."""
    column_types = identify_column_types(dataframe)
    encoded_columns = (
        column_types["categorical_columns"] + column_types["boolean_columns"]
    )
    categorical_columns = [
        column for column in dataframe.columns if column in encoded_columns
    ]
    return column_types["numerical_columns"], categorical_columns


def _build_preprocessor(
    numerical_columns: list[Any],
    categorical_columns: list[Any],
) -> ColumnTransformer:
    """Build the scikit-learn transformer used by the public function."""
    transformers = []

    if numerical_columns:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numerical", numerical_pipeline, numerical_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def preprocess_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean and transform a DataFrame for clustering.

    Duplicate rows are removed. Missing numerical values are replaced with
    the median, numerical features are standardized, missing categorical
    values are replaced with the mode, and categorical features are one-hot
    encoded. Boolean columns use the categorical pipeline.

    Returns:
        A processed numerical DataFrame and a structured preprocessing report.

    Raises:
        PreprocessingError: If the input is invalid or cannot be transformed.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise PreprocessingError("Input must be a pandas DataFrame.")
    if dataframe.empty:
        raise PreprocessingError("The DataFrame must contain at least one row.")
    if dataframe.shape[1] == 0:
        raise PreprocessingError("The DataFrame must contain at least one column.")
    if not dataframe.columns.is_unique:
        raise PreprocessingError("Column names must be unique.")

    try:
        column_types = identify_column_types(dataframe)
        unsupported_columns = column_types["unsupported_columns"]
        if unsupported_columns:
            names = ", ".join(str(column) for column in unsupported_columns)
            raise PreprocessingError(
                f"Unsupported column types detected: {names}. "
                "Convert or remove these columns before preprocessing."
            )

        missing_values = int(dataframe.isna().sum().sum())
        duplicate_rows = int(dataframe.duplicated().sum())
        cleaned_dataframe = dataframe.drop_duplicates().copy()

        numerical_columns, categorical_columns = detect_column_types(
            cleaned_dataframe
        )
        if not numerical_columns and not categorical_columns:
            raise PreprocessingError("No supported columns were detected.")

        all_missing_columns = [
            column
            for column in numerical_columns + categorical_columns
            if cleaned_dataframe[column].isna().all()
        ]
        if all_missing_columns:
            names = ", ".join(str(column) for column in all_missing_columns)
            raise PreprocessingError(
                f"Cannot impute columns containing only missing values: {names}."
            )

        # Normalize pandas missing values for scikit-learn imputers.
        for column in numerical_columns:
            cleaned_dataframe[column] = pd.to_numeric(
                cleaned_dataframe[column], errors="raise"
            ).astype(float)
        for column in categorical_columns:
            values = cleaned_dataframe[column].astype(object)
            cleaned_dataframe[column] = values.where(values.notna(), np.nan)

        preprocessor = _build_preprocessor(
            numerical_columns,
            categorical_columns,
        )
        transformed_values = preprocessor.fit_transform(cleaned_dataframe)
        output_columns = preprocessor.get_feature_names_out().tolist()
        processed_dataframe = pd.DataFrame(
            transformed_values,
            columns=output_columns,
            index=cleaned_dataframe.index,
        )

        steps_performed = [
            "Detected missing values.",
            "Detected and removed duplicate rows.",
        ]
        if numerical_columns:
            steps_performed.extend(
                [
                    "Filled missing numerical values with the median.",
                    "Scaled numerical features with StandardScaler.",
                ]
            )
        if categorical_columns:
            steps_performed.extend(
                [
                    "Filled missing categorical values with the most frequent value.",
                    "Encoded categorical features with OneHotEncoder.",
                ]
            )

        report = {
            "missing_values_detected": missing_values,
            "duplicate_rows_detected": duplicate_rows,
            "duplicate_rows_removed": duplicate_rows,
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,
            "boolean_columns": column_types["boolean_columns"],
            "unsupported_columns": unsupported_columns,
            "numerical_imputation": (
                "Median" if numerical_columns else "Not applied"
            ),
            "categorical_imputation": (
                "Most frequent value" if categorical_columns else "Not applied"
            ),
            "scaling_method": (
                "StandardScaler" if numerical_columns else "Not applied"
            ),
            "encoding_method": (
                "OneHotEncoder" if categorical_columns else "Not applied"
            ),
            "input_shape": dataframe.shape,
            "output_shape": processed_dataframe.shape,
            "steps_performed": steps_performed,
        }
        return processed_dataframe, report
    except PreprocessingError:
        raise
    except Exception as error:
        raise PreprocessingError(f"Preprocessing failed: {error}") from error


def print_preprocessing_report(report: dict[str, Any]) -> None:
    """Print a preprocessing report in the terminal."""
    numerical_columns = ", ".join(
        str(column) for column in report["numerical_columns"]
    ) or "None"
    categorical_columns = ", ".join(
        str(column) for column in report["categorical_columns"]
    ) or "None"
    boolean_columns = ", ".join(
        str(column) for column in report["boolean_columns"]
    ) or "None"

    print("Preprocessing report")
    print(f"Missing values detected: {report['missing_values_detected']}")
    print(f"Duplicate rows detected: {report['duplicate_rows_detected']}")
    print(f"Duplicate rows removed: {report['duplicate_rows_removed']}")
    print(f"Numerical columns: {numerical_columns}")
    print(f"Categorical columns: {categorical_columns}")
    print(f"Boolean columns (one-hot encoded): {boolean_columns}")
    print(f"Numerical imputation: {report['numerical_imputation']}")
    print(f"Categorical imputation: {report['categorical_imputation']}")
    print(f"Scaling method: {report['scaling_method']}")
    print(f"Encoding method: {report['encoding_method']}")
    print(f"Input shape: {report['input_shape']}")
    print(f"Output shape: {report['output_shape']}")

    print("Steps performed:")
    for step in report["steps_performed"]:
        print(f"- {step}")
