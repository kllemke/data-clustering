from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

import app as frontend
from app import _column_rows


def test_column_summary_maps_detected_data_types():
    rows = _column_rows(
        {
            "columns": {
                "description": {
                    "dtype": "object",
                    "missing_ratio": 0.1,
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


def test_upload_is_loaded_once_and_special_columns_are_not_selected(monkeypatch):
    original_loader = frontend.load_table
    loaded_paths = []

    def count_load(path):
        loaded_paths.append(path)
        return original_loader(path)

    monkeypatch.setattr(frontend, "load_table", count_load)
    dataframe, report = frontend._analyze_upload(
        b'age,date,notes,asset\n20,2026-01-01,"A long description of the first observation.",photo.jpg\n'
        b'30,2026-01-02,"Another long description for a second observation.",photo.jpg\n',
        "sample.csv",
    )
    assert len(loaded_paths) == 1
    assert report["source"] == "sample.csv"
    assert frontend._default_preprocessing_columns(dataframe, report) == ["age"]


def _uploaded_csv(content):
    file = BytesIO(content)
    file.name = "sample.csv"
    file.size = len(content)
    return file


def _button(application, label):
    return next(button for button in application.button if button.label == label)


def test_frontend_preprocesses_and_keeps_results_until_the_input_changes(monkeypatch):
    upload = {"file": _uploaded_csv(
        b'age,score,city,member,date,notes,asset\n'
        b'20,7.5,Madrid,True,2026-01-01,"A detailed description of this observation.",photo.jpg\n'
        b'30,9,Valencia,False,2026-01-02,"A detailed description of this observation.",photo.jpg\n'
        b',8,,True,2026-01-03,"A detailed description of this observation.",photo.jpg\n'
        b'40,6,Madrid,False,2026-01-04,"A detailed description of this observation.",photo.jpg\n'
        b'30,9,Valencia,False,2026-01-02,"A detailed description of this observation.",photo.jpg\n'
    )}
    # AppTest has no file-uploader interaction; provide bytes at that boundary.
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload["file"])
    application = AppTest.from_file(str(Path(frontend.__file__)), default_timeout=20).run()
    _button(application, "Analyze file").click().run()
    assert not application.exception
    assert application.multiselect[0].value == ["age", "score", "city", "member"]
    original = application.session_state["original_dataframe"].copy(deep=True)
    original_report = application.session_state["analysis_report"].copy()

    _button(application, "Preprocess dataset").click().run()
    assert not application.exception
    processed = application.session_state["processed_dataframe"]
    assert processed.shape == (4, 6)
    assert not processed.isna().any().any()
    assert application.session_state["preprocessing_report"]["excluded_columns"] == ["date", "notes", "asset"]
    pd.testing.assert_frame_equal(application.session_state["original_dataframe"], original)
    assert application.session_state["analysis_report"] == original_report
    downloads = application.get("download_button")
    assert any(button.label == "Download processed CSV" for button in downloads)

    application.run()
    pd.testing.assert_frame_equal(application.session_state["processed_dataframe"], processed)
    application.multiselect[0].set_value(["age"]).run()
    assert "processed_dataframe" not in application.session_state
    _button(application, "Preprocess dataset").click().run()
    assert application.session_state["processed_dataframe"].shape == (4, 1)
    application.multiselect[0].set_value([]).run()
    assert _button(application, "Preprocess dataset").disabled

    # Even an upload with the same name must invalidate old results when bytes change.
    upload["file"] = _uploaded_csv(b"age\n99\n")
    application.run()
    assert not application.exception
    assert "analysis_report" not in application.session_state
    assert "processed_dataframe" not in application.session_state


def test_frontend_displays_preprocessing_errors_without_losing_the_analysis(monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: _uploaded_csv(b"age,city\n,Madrid\n,Valencia\n"))
    application = AppTest.from_file(str(Path(frontend.__file__)), default_timeout=20).run()
    _button(application, "Analyze file").click().run()
    _button(application, "Preprocess dataset").click().run()
    assert not application.exception
    assert any("only missing values: age" in error.value for error in application.error)
    assert "analysis_report" in application.session_state
    assert "processed_dataframe" not in application.session_state
