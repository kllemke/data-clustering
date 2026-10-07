from app import _column_rows


def test_column_summary_maps_detected_data_types():
    rows = _column_rows(
        {
            "columns": {
                "description": {
                    "dtype": "object",
                    "text": {"is_text": True, "empty_ratio": 0.1},
                    "numerical": {"is_numeric": False, "missing_ratio": 0.0},
                    "categorical": {"is_categorical": False, "missing_ratio": 0.0},
                },
                "amount": {
                    "dtype": "int64",
                    "text": {"is_text": False, "empty_ratio": 0.0},
                    "numerical": {"is_numeric": True, "missing_ratio": 0.2},
                    "categorical": {"is_categorical": False, "missing_ratio": 0.2},
                },
            }
        }
    )

    assert rows[0]["Detected as"] == "text"
    assert rows[0]["Missing"] == 0.1
    assert rows[1]["Detected as"] == "numerical"
    assert rows[1]["Missing"] == 0.2
