import pandas as pd

from data_clustering.detection.text import analyze_text_column, detect_text_columns


def test_analyze_text_column_detects_natural_language():
    series = pd.Series(
        [
            "The project team delivered the launch plan on time.",
            "Customer feedback highlighted a need for clearer onboarding.",
            "The dashboard will be updated next week.",
            None,
            "",
        ],
        name="description",
    )

    profile = analyze_text_column(series)

    assert profile.is_text is True
    assert profile.string_ratio > 0.7
    assert profile.average_text_length > 3
    assert profile.language_like_score > 0.35


def test_analyze_text_column_rejects_numeric_identifier_strings():
    series = pd.Series(["001", "002", "003", "004", None], name="customer_id")

    profile = analyze_text_column(series)

    assert profile.is_text is False
    assert profile.language_like_score < 0.35


def test_detect_text_columns_identifies_text_and_non_text_columns():
    dataframe = pd.DataFrame(
        {
            "description": [
                "This is a summary of the new onboarding flow.",
                "Users can access the portal after authentication.",
                "The dataset was refreshed earlier this morning.",
            ],
            "age": [29, 41, 36],
            "status": ["active", "inactive", "active"],
            "zipcode": ["01234", "56789", "99999"],
        }
    )

    profiles = detect_text_columns(dataframe)

    assert profiles["description"].is_text is True
    assert profiles["age"].is_text is False
    assert profiles["status"].is_text is True
    assert profiles["zipcode"].is_text is False
