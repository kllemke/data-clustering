"""Streamlit frontend for interactive dataset analysis."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from data_clustering.analysis import analyze_file


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
        rows.append(
            {
                "Column": name,
                "Data type": profile.get("dtype", "unknown"),
                "Detected as": ", ".join(types) if types else "unclassified",
                "Missing": round(
                    max(
                        profile.get("text", {}).get("empty_ratio", 0.0),
                        profile.get("numerical", {}).get("missing_ratio", 0.0),
                        profile.get("categorical", {}).get("missing_ratio", 0.0),
                    ),
                    3,
                ),
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


def _render_tabular_report(report: Dict[str, Any]) -> None:
    st.subheader("Dataset overview")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Rows", f"{report['row_count']:,}")
    metric_columns[1].metric("Columns", f"{report['column_count']:,}")
    metric_columns[2].metric("Detected types", len(report.get("data_types", [])))

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
    if uploaded_file is None:
        st.info("Choose a file to get started.")
        return

    st.caption(f"Selected: **{uploaded_file.name}** · {uploaded_file.size:,} bytes")
    if not st.button("Analyze file", type="primary"):
        return

    try:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.TemporaryDirectory(prefix="data-clustering-") as temporary_directory:
            temporary_path = Path(temporary_directory) / f"uploaded{suffix}"
            temporary_path.write_bytes(uploaded_file.getvalue())
            with st.spinner("Analyzing uploaded file..."):
                analysis = analyze_file(temporary_path)
                report = analysis.to_dict()
    except (FileNotFoundError, ValueError, ImportError, OSError) as error:
        st.error(f"Could not analyze this file: {error}")
        return
    except Exception as error:
        st.error(f"Analysis failed: {error}")
        raise

    st.success("Analysis complete.")
    if report["kind"] == "tabular":
        _render_tabular_report(report)
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
