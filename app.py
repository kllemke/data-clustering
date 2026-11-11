"""Streamlit frontend for interactive dataset analysis."""

from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from data_clustering.analysis import TABLE_EXTENSIONS, analyze_dataframe, analyze_file, load_table
from data_clustering.dataset_info import PREVIEW_ROW_COUNT
from data_clustering.preprocessing import PreprocessingError, preprocess_dataset


SUPPORTED_TYPES = [
    "csv", "tsv", "txt", "xlsx", "xls", "xlsm", "json", "parquet", "pq",
    "bmp", "gif", "jpeg", "jpg", "png", "tif", "tiff", "webp",
    "aac", "flac", "m4a", "mp3", "ogg", "wav", "wma",
    "avi", "mkv", "mov", "mp4", "mpeg", "mpg", "webm", "wmv",
]


def _column_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for name, profile in report.get("columns", {}).items():
        detection_flags = {
            "text": "is_text",
            "numerical": "is_numeric",
            "categorical": "is_categorical",
        }
        types = [
            data_type
            for data_type, flag in detection_flags.items()
            if profile.get(data_type, {}).get(flag, False)
        ]
        kind = profile.get("kind")
        if kind in {"boolean", "unsupported"}:
            types.insert(0, kind)
        if profile.get("is_datetime"):
            types.append("datetime")
        rows.append(
            {
                "Column": name,
                "Data type": profile.get("dtype", "unknown"),
                "Basic kind": kind or "unknown",
                "Detected as": ", ".join(types) if types else "unclassified",
                "Missing": round(profile.get("missing_ratio", max(
                    profile.get("numerical", {}).get("missing_ratio", 0.0),
                    profile.get("categorical", {}).get("missing_ratio", 0.0),
                )), 3),
                "Blank strings": profile.get("blank_count", 0),
            }
        )
    return rows


def _render_recommendations(report: Dict[str, Any]) -> None:
    recommendations = report.get("recommendations", [])
    if not recommendations:
        st.info("No algorithm recommendations are available for this input.")
        return
    for index, recommendation in enumerate(recommendations, start=1):
        with st.container(border=True):
            st.subheader(f"{index}. {recommendation['algorithm']}")
            st.write(recommendation["rationale"])
            steps = recommendation.get("preprocessing", [])
            if steps:
                st.markdown("**Suggested preparation**")
                for step in steps:
                    st.markdown(f"- {step}")


def _render_tabular_report(report: Dict[str, Any], dataframe: pd.DataFrame) -> None:
    st.subheader("Dataset overview")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Rows", f"{report['row_count']:,}")
    metric_columns[1].metric("Columns", f"{report['column_count']:,}")
    metric_columns[2].metric("Detected types", len(report.get("data_types", [])))
    metric_columns[3].metric("Missing values", report["missing_values_detected"])
    duplicates = report["duplicate_rows_detected"]
    metric_columns[4].metric("Duplicate rows", duplicates if duplicates is not None else "Unavailable")
    st.subheader("First 5 rows (original dataset)")
    st.dataframe(dataframe.head(PREVIEW_ROW_COUNT), use_container_width=True, hide_index=True)

    detected = report.get("data_types", [])
    if detected:
        st.markdown("**Detected data types**")
        st.write(" · ".join(f"`{data_type}`" for data_type in detected))
    else:
        st.info("No specialized data types were detected.")

    overview_tab, timeseries_tab, recommendations_tab, json_tab = st.tabs(
        ["Columns", "Time series", "Recommendations", "JSON report"]
    )
    with overview_tab:
        rows = _column_rows(report)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            for name, profile in report.get("columns", {}).items():
                with st.expander(f"Details: {name}"):
                    st.json(profile)
        uncertainty = report.get("uncertainty", {})
        st.markdown("**Uncertainty and missing values**")
        st.json(uncertainty)
        multimedia = report.get("multimedia")
        if multimedia:
            st.markdown("**Media references found in columns**")
            st.json(multimedia)
    with timeseries_tab:
        profiles = report.get("time_series", [])
        if not profiles:
            st.info("No time-series columns were detected.")
        for profile in profiles:
            with st.expander(f"Time column: {profile['time_column']}", expanded=True):
                st.json(profile)
    with recommendations_tab:
        _render_recommendations(report)
    with json_tab:
        st.json(report)


def _render_media_report(report: Dict[str, Any]) -> None:
    media = report["media"]
    st.subheader("Multimedia overview")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Supported files", len(media.get("files", [])))
    metric_columns[1].metric("Total size", f"{media.get('total_size_bytes', 0):,} bytes")
    metric_columns[2].metric(
        "Feature dimensions",
        f"{media['feature_dimensions']:,}" if media.get("feature_dimensions") else "Not extracted",
    )
    st.markdown("**File types**")
    st.json(media.get("file_type_counts", {}))
    file_rows = [
        {
            "File": Path(item["path"]).name,
            "Type": item["media_type"],
            "MIME type": item.get("mime_type") or "unknown",
            "Size (bytes)": item["size_bytes"],
            "Feature dimensions": item.get("feature_dimensions"),
        }
        for item in media.get("files", [])
    ]
    if file_rows:
        st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)
    for item in media.get("files", []):
        with st.expander(f"Metadata: {Path(item['path']).name}"):
            st.json(item.get("metadata", {}))
    st.subheader("Algorithm recommendations")
    _render_recommendations(report)
    with st.expander("JSON report"):
        st.json(report)


def _analyze_upload(content: bytes, name: str) -> tuple[pd.DataFrame | None, Dict[str, Any]]:
    """Load an uploaded table once, or keep the existing multimedia analysis."""
    suffix = Path(name).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="data-clustering-") as temporary_directory:
        path = Path(temporary_directory) / f"uploaded{suffix}"
        path.write_bytes(content)
        if suffix in TABLE_EXTENSIONS:
            dataframe = load_table(path)
            return dataframe, analyze_dataframe(dataframe, source=name).to_dict()
        report = analyze_file(path).to_dict()
        report["source"] = name
        return None, report


def _clear_analysis() -> None:
    """Discard results and widget choices when the uploaded file changes."""
    for key in (
        "analysis_report", "original_dataframe", "processed_dataframe",
        "preprocessing_report", "preprocessing_columns", "processed_columns",
    ):
        st.session_state.pop(key, None)


def _default_preprocessing_columns(dataframe: pd.DataFrame, report: Dict[str, Any]) -> list:
    """Suggest tabular features without automatically encoding special modalities."""
    defaults = []
    for column in dataframe.columns:
        profile = report["columns"][str(column)]
        text = profile.get("text", {})
        free_text = text.get("is_text", False) and (
            text.get("average_word_count", 0) >= 3 or text.get("average_text_length", 0) >= 25
        )
        if (profile["kind"] != "unsupported" and not free_text
                and not profile.get("is_datetime") and not profile.get("multimedia")):
            defaults.append(column)
    return defaults


def _render_preprocessing(dataframe: pd.DataFrame, report: Dict[str, Any]) -> None:
    st.subheader("Tabular preprocessing")
    st.caption(
        "This pipeline removes duplicate rows, imputes median/mode, applies StandardScaler "
        "and One-Hot encoding. It does not execute clustering or specialized text/time/media preparation. "
        "Its output is not suitable for every recommended algorithm."
    )
    compatible = [column for column in dataframe.columns
                  if report["columns"][str(column)]["kind"] != "unsupported"]
    selected = st.multiselect(
        "Columns to preprocess", options=compatible,
        default=_default_preprocessing_columns(dataframe, report), key="preprocessing_columns",
        help="Dates, free text and media references are not selected by default. "
             "Selecting a string column explicitly encodes its whole values as categories.",
    )
    excluded = [str(column) for column in dataframe.columns if column not in selected]
    st.write("Excluded columns: " + (", ".join(excluded) or "None"))
    if not selected:
        st.info("Select at least one compatible column to preprocess.")

    # A previous output must never be shown as the result of a new selection.
    if tuple(selected) != st.session_state.get("processed_columns"):
        st.session_state.pop("processed_dataframe", None)
        st.session_state.pop("preprocessing_report", None)

    if st.button("Preprocess dataset", disabled=not selected):
        try:
            processed, preprocessing_report = preprocess_dataset(dataframe[selected])
        except PreprocessingError as error:
            st.error(str(error))
        else:
            preprocessing_report["excluded_columns"] = excluded
            st.session_state["processed_dataframe"] = processed
            st.session_state["preprocessing_report"] = preprocessing_report
            st.session_state["processed_columns"] = tuple(selected)

    if "processed_dataframe" in st.session_state:
        processed = st.session_state["processed_dataframe"]
        preprocessing_report = st.session_state["preprocessing_report"]
        st.subheader("Processed dataset")
        st.write(f"Rows: {processed.shape[0]} | Features: {processed.shape[1]}")
        st.dataframe(processed.head(PREVIEW_ROW_COUNT), use_container_width=True, hide_index=True)
        st.markdown("**Steps performed**")
        for step in preprocessing_report["steps_performed"]:
            st.write(f"- {step}")
        with st.expander("Preprocessing report"):
            st.json(preprocessing_report)
        st.download_button(
            "Download processed CSV", data=processed.to_csv(index=False).encode("utf-8"),
            file_name="dataset_preprocessed.csv", mime="text/csv",
        )


def main() -> None:
    st.set_page_config(
        page_title="Data Clustering | Dataset Analyzer",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Dataset Analyzer")
    st.write(
        "Upload a supported data or media file to inspect its characteristics "
        "and get clustering algorithm recommendations."
    )
    st.caption(
        "Analysis is heuristic. Recommendations are suggestions only; "
        "the application does not train clustering models."
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=SUPPORTED_TYPES,
        help="Supported: CSV, TSV, Excel, JSON, Parquet, images, audio, and video.",
    )
    content = uploaded_file.getvalue() if uploaded_file is not None else b""
    signature = (uploaded_file.name, hashlib.sha256(content).hexdigest()) if uploaded_file is not None else None
    if signature != st.session_state.get("upload_signature"):
        _clear_analysis()
        st.session_state["upload_signature"] = signature
    if uploaded_file is None:
        st.info("Choose a file to get started.")
        return

    st.caption(f"Selected: **{uploaded_file.name}** · {uploaded_file.size:,} bytes")
    if st.button("Analyze file", type="primary"):
        _clear_analysis()
        try:
            with st.spinner("Analyzing uploaded file..."):
                dataframe, report = _analyze_upload(content, uploaded_file.name)
            st.session_state["original_dataframe"] = dataframe
            st.session_state["analysis_report"] = report
        except (FileNotFoundError, ValueError, ImportError, OSError) as error:
            st.error(f"Could not analyze this file: {error}")
            return

    if "analysis_report" not in st.session_state:
        return

    report = st.session_state["analysis_report"]
    st.success("Analysis complete.")
    if report["kind"] == "tabular":
        dataframe = st.session_state["original_dataframe"]
        _render_tabular_report(report, dataframe)
        _render_preprocessing(dataframe, report)
    else:
        _render_media_report(report)

    st.download_button(
        "Download JSON report",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name=f"{Path(uploaded_file.name).stem}-analysis.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
