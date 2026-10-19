"""Terminal demo and automated tests for the dataset tools.

Run the demo and all checks: py -3.14 test_dataset_processing.py
Run only the tests: py -3.14 -m pytest -q test_dataset_processing.py
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

from dataset_info import display_dataset_info, get_dataset_info, identify_column_types
from preprocessing import (
    PreprocessingError,
    detect_column_types,
    preprocess_dataset,
    print_preprocessing_report,
)


def build_sample_dataframe() -> pd.DataFrame:
    """Create a sample dataset with missing values and one duplicate row."""
    return pd.DataFrame(
        {
            "age": [20, 30, None, 40, 30],
            "score": [7.5, 9.0, 8.0, 6.0, 9.0],
            "city": ["Madrid", "Valencia", None, "Madrid", "Valencia"],
            "member": [True, False, True, False, False],
        }
    )


def run_checks(
    original_dataframe: pd.DataFrame,
    original_copy: pd.DataFrame,
    processed_dataframe: pd.DataFrame,
    report: dict[str, object],
) -> None:
    """Run the demo checks and print whether each one passed."""
    try:
        preprocess_dataset(pd.DataFrame())
    except PreprocessingError:
        invalid_input_was_handled = True
    else:
        invalid_input_was_handled = False

    checks = {
        "Two missing values were detected": (
            report["missing_values_detected"] == 2
        ),
        "One duplicate row was detected": (
            report["duplicate_rows_detected"] == 1
        ),
        "One duplicate row was removed": (
            report["duplicate_rows_removed"] == 1
        ),
        "The processed dataset has four rows": (
            processed_dataframe.shape[0] == 4
        ),
        "The processed dataset has no missing values": (
            not processed_dataframe.isna().any().any()
        ),
        "All processed features are numerical": all(
            is_numeric_dtype(data_type)
            for data_type in processed_dataframe.dtypes
        ),
        "The original dataset was not modified": original_dataframe.equals(
            original_copy
        ),
        "An empty dataset produces a controlled error": invalid_input_was_handled,
    }

    print("\nAUTOMATIC CHECKS")
    for description, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {description}")

    failed_checks = [
        description for description, passed in checks.items() if not passed
    ]
    if failed_checks:
        raise AssertionError(
            "Some checks failed: " + ", ".join(failed_checks)
        )


def run_demo() -> None:
    """Show the original data, processed data, report and demo checks."""
    dataframe = build_sample_dataframe()
    original_copy = dataframe.copy(deep=True)

    print("ORIGINAL DATASET")
    print("=" * 60)
    display_dataset_info(dataframe)

    try:
        processed_dataframe, report = preprocess_dataset(dataframe)
    except PreprocessingError as error:
        print(f"\n[ERROR] Preprocessing could not be completed: {error}")
        raise SystemExit(1) from error

    print("\nPROCESSED DATASET")
    print("=" * 60)
    print(processed_dataframe.to_string())

    print("\nPREPROCESSING DETAILS")
    print("=" * 60)
    print_preprocessing_report(report)

    run_checks(dataframe, original_copy, processed_dataframe, report)
    print("\nAll checks passed. The preprocessing works as expected.")


def test_dataset_info_classifies_columns_without_changing_the_input():
    dataframe = pd.DataFrame(
        {
            "age": pd.Series([20, pd.NA, 40], dtype="Int64"),
            "score": pd.Series([1.5, 2.5, pd.NA], dtype="Float64"),
            "city": pd.Series(["Madrid", "Valencia", None], dtype="string"),
            "rating": pd.Series([1, 2, 1], dtype="category"),
            "active": pd.Series([True, pd.NA, False], dtype="boolean"),
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        }
    )
    original = dataframe.copy(deep=True)

    info = get_dataset_info(dataframe)

    assert info["numerical_columns"] == ["age", "score"]
    assert info["categorical_columns"] == ["city", "rating"]
    assert info["boolean_columns"] == ["active"]
    assert info["unsupported_columns"] == ["date"]
    assert info["row_count"] == 3
    assert info["column_count"] == 6
    pd.testing.assert_frame_equal(info["first_rows"], original)
    pd.testing.assert_series_equal(info["data_types"], original.dtypes)
    pd.testing.assert_frame_equal(dataframe, original)


@pytest.mark.parametrize(
    "values",
    [
        pd.to_datetime(["2026-01-01", "2026-01-02"]),
        pd.to_timedelta([1, 2], unit="D"),
        [1 + 2j, 3 + 4j],
        [[1, 2], [3, 4]],
        [{"x": 1}, {"x": 2}],
        [1, "Madrid"],
    ],
    ids=["dates", "durations", "complex", "lists", "dictionaries", "mixed"],
)
def test_unsupported_values_are_flagged_before_preprocessing(values):
    dataframe = pd.DataFrame({"problem": values})

    assert identify_column_types(dataframe)["unsupported_columns"] == ["problem"]
    with pytest.raises(PreprocessingError, match="Unsupported column types detected: problem"):
        preprocess_dataset(dataframe)


def test_object_scalars_are_inferred_and_numeric_strings_are_preserved():
    dataframe = pd.DataFrame(
        {
            "number": pd.Series([1, pd.NA, 3], dtype=object),
            "active": pd.Series([True, None, False], dtype=object),
            "code": pd.Series(["001", "002", None], dtype=object),
        }
    )

    info = identify_column_types(dataframe)
    assert info["numerical_columns"] == ["number"]
    assert info["boolean_columns"] == ["active"]
    assert info["categorical_columns"] == ["code"]
    assert info["unsupported_columns"] == []

    processed, report = preprocess_dataset(dataframe)
    assert not processed.isna().any().any()
    assert "categorical__code_001" in processed.columns
    assert report["boolean_columns"] == ["active"]


def test_preprocessing_still_imputes_scales_and_encodes_booleans():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, None, 40, 30],
            "city": ["Madrid", "Valencia", None, "Madrid", "Valencia"],
            "active": pd.Series([True, False, pd.NA, False, False], dtype="boolean"),
        }
    )
    original = dataframe.copy(deep=True)

    assert detect_column_types(dataframe) == (["age"], ["city", "active"])
    processed, report = preprocess_dataset(dataframe)

    assert processed.shape == (4, 5)
    assert not processed.isna().any().any()
    assert abs(processed["numerical__age"].mean()) < 1e-12
    assert processed["numerical__age"].std(ddof=0) == pytest.approx(1.0)
    assert {"categorical__active_False", "categorical__active_True"} <= set(processed.columns)
    assert report["duplicate_rows_removed"] == 1
    assert report["boolean_columns"] == ["active"]
    assert report["unsupported_columns"] == []
    pd.testing.assert_frame_equal(dataframe, original)


def test_terminal_output_shows_the_groups_and_unsupported_warning(capsys):
    dataframe = pd.DataFrame(
        {
            "age": range(6),
            "active": [True, False] * 3,
            "date": pd.date_range("2026-01-01", periods=6),
        }
    )
    display_dataset_info(dataframe)

    output = capsys.readouterr().out
    assert "Numerical columns: age" in output
    assert "Boolean columns: active" in output
    assert "Unsupported columns: date" in output
    assert "Warning:" in output
    assert "Total rows: 6" in output
    assert len(get_dataset_info(dataframe)["first_rows"]) == 5


def test_empty_dataframes_can_still_be_inspected():
    info = get_dataset_info(pd.DataFrame())
    assert all(not info[key] for key in (
        "numerical_columns", "categorical_columns", "boolean_columns", "unsupported_columns"
    ))
    assert info["row_count"] == info["column_count"] == 0


def test_invalid_inputs_raise_a_clear_error():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        identify_column_types([])
    with pytest.raises(TypeError, match="pandas DataFrame"):
        get_dataset_info(None)


def test_terminal_demo_runs_all_checks(capsys):
    run_demo()

    output = capsys.readouterr().out
    assert "ORIGINAL DATASET" in output
    assert "PROCESSED DATASET" in output
    assert "Preprocessing report" in output
    assert "Output shape: (4, 6)" in output
    assert output.count("[PASS]") == 8
    assert "[FAIL]" not in output


if __name__ == "__main__":
    run_demo()
    print("\nRUNNING ALL AUTOMATED TESTS")
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
