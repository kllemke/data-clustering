import pandas as pd

from data_clustering.detection.time_series import analyze_time_series_column, detect_time_series_columns


def test_analyze_time_series_column_detects_datetime_data():
    series = pd.Series(
        [
            "2024-01-01 00:00:00",
            "2024-01-02 00:00:00",
            "2024-01-03 00:00:00",
            "2024-01-04 00:00:00",
        ],
        name="event_time",
    )

    profile = analyze_time_series_column(series)

    assert profile.is_time_series is True
    assert profile.is_datetime_like is True
    assert profile.is_chronological is True
    assert profile.datetime_ratio > 0.7


def test_analyze_time_series_column_rejects_non_time_columns():
    series = pd.Series([10, 12, 14, 18, 9], name="temperature")

    profile = analyze_time_series_column(series)

    assert profile.is_time_series is False
    assert profile.is_datetime_like is False


def test_detect_time_series_columns_flags_time_columns_in_dataframe():
    dataframe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "value": [12.1, 14.3, 15.0, 16.8],
            "category": ["A", "B", "A", "B"],
        }
    )

    profiles = detect_time_series_columns(dataframe)

    assert profiles["timestamp"].is_time_series is True
    assert profiles["value"].is_time_series is False
    assert profiles["category"].is_time_series is False
