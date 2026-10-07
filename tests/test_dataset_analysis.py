import json

import pandas as pd
import pytest

from data_clustering import analyze_dataframe, analyze_file
from data_clustering.cli import main
from data_clustering.detection.categorical import analyze_categorical_column
from data_clustering.detection.multimedia import inspect_media_path
from data_clustering.detection.numerical_uncertainty import analyze_uncertainty
from data_clustering.detection.text import analyze_text_column


def test_analyze_dataframe_profiles_types_uncertainty_and_recommendations():
    dataframe = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=8, freq="D"),
            "measurement": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "confidence_score": [0.9, 0.8, 0.95, 0.85, 0.88, 0.9, 0.92, 0.91],
            "measurement_lower": [0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8],
            "measurement_upper": [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],
            "segment": ["north", "north", "south", "south", "north", "south", "north", "south"],
            "description": [
                "An observation from the northern region.",
                "An observation from the southern region.",
            ] * 4,
            "asset": ["photo.jpg"] * 8,
        }
    )

    result = analyze_dataframe(dataframe)
    output = result.to_dict()

    assert {"text", "time-series", "numerical", "categorical", "uncertain/probabilistic", "multimedia"} <= set(
        result.data_types
    )
    assert result.row_count == 8
    assert result.time_series[0].trends["measurement"] == "increasing"
    assert result.uncertainty.confidence_score_columns == ["confidence_score"]
    assert ("measurement_lower", "measurement_upper") in result.uncertainty.confidence_interval_pairs
    assert result.multimedia.file_type_counts == {"image": 1}
    assert any("TF-IDF" in item.algorithm for item in result.recommendations)
    assert output["recommendations"]


def test_categorical_profile_reports_hierarchy_and_ordinal_order():
    profile = analyze_categorical_column(
        pd.Series(["region/north", "region/south", None], name="location"),
        ordinal_order=["low", "medium", "high"],
    )

    assert profile.is_categorical
    assert profile.is_hierarchical
    assert profile.hierarchy_separator == "/"
    assert profile.category_type == "ordinal"
    assert profile.cardinality == 2
    assert profile.missing_ratio == pytest.approx(1 / 3)


def test_categorical_profile_infers_known_ordinal_scale():
    profile = analyze_categorical_column(pd.Series(["low", "medium", "high"], name="rating"))

    assert profile.category_type == "ordinal"


def test_analyze_dataframe_accepts_explicit_ordinal_order():
    result = analyze_dataframe(
        pd.DataFrame({"rating": ["bronze", "silver", "gold"]}),
        ordinal_orders={"rating": ["bronze", "silver", "gold"]},
    )

    assert result.columns["rating"]["categorical"]["category_type"] == "ordinal"


def test_uncertainty_profile_quantifies_missingness_and_spread():
    dataframe = pd.DataFrame(
        {
            "probability": [0.2, 0.5, None],
            "value_lower": [1, 2, 3],
            "value_upper": [3, 4, 5],
        }
    )

    profile = analyze_uncertainty(dataframe)

    assert "probability" in profile.confidence_score_columns
    assert profile.confidence_interval_pairs == [("value_lower", "value_upper")]
    assert profile.missing_ratios["probability"] == pytest.approx(1 / 3)
    assert profile.numeric_uncertainty["probability"] > 0


def test_text_profile_includes_complexity_and_keywords():
    profile = analyze_text_column(
        pd.Series(
            [
                "Clustering helps reveal repeated customer behavior.",
                "Customer behavior can indicate useful product clusters.",
            ],
            name="notes",
        )
    )

    assert profile.average_word_count > 3
    assert 0 < profile.lexical_diversity <= 1
    assert ("customer", 2) in profile.keywords


def test_analyze_file_reads_csv_and_json(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"value": [1, 2, 3], "group": ["a", "b", "a"]}).to_csv(csv_path, index=False)

    csv_result = analyze_file(csv_path)
    assert csv_result.row_count == 3
    assert csv_result.source == str(csv_path)

    json_path = tmp_path / "sample.json"
    json_path.write_text(json.dumps([{"value": 4}, {"value": 5}]), encoding="utf-8")
    json_result = analyze_file(json_path)
    assert json_result.row_count == 2


def test_analyze_file_reads_excel_and_parquet(tmp_path):
    dataframe = pd.DataFrame({"measurement": [1, 2, 3]})
    excel_path = tmp_path / "sample.xlsx"
    parquet_path = tmp_path / "sample.parquet"
    dataframe.to_excel(excel_path, index=False)
    assert analyze_file(excel_path).row_count == 3

    try:
        import pyarrow.parquet
    except ImportError:
        pytest.skip("Parquet support is unavailable in this Python environment.")
    dataframe.to_parquet(parquet_path, index=False)
    assert analyze_file(parquet_path).row_count == 3


def test_analyze_file_inspects_image_metadata_and_rejects_unknown_format(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "tiny.png"
    image_module.new("RGB", (3, 2), color="red").save(image_path)

    result = analyze_file(image_path)

    assert result.to_dict()["kind"] == "multimedia"
    image = result.media.files[0]
    assert image.media_type == "image"
    assert image.metadata["width"] == 3
    assert image.metadata["height"] == 2
    assert image.feature_dimensions == 18

    unknown_path = tmp_path / "unknown.xyz"
    unknown_path.write_text("not supported", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        analyze_file(unknown_path)


def test_media_directory_identifies_audio_and_video(tmp_path, monkeypatch):
    from data_clustering.detection import multimedia

    monkeypatch.setattr(multimedia.shutil, "which", lambda _: None)
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "sound.wav").write_bytes(b"audio")
    (media_dir / "clip.mp4").write_bytes(b"video")

    profile = inspect_media_path(media_dir)

    assert profile.file_type_counts == {"audio": 1, "video": 1}
    assert all(item.metadata["metadata_status"].startswith("Install FFmpeg") for item in profile.files)


def test_analyze_file_raises_for_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        analyze_file(tmp_path / "missing.csv")


def test_cli_writes_json_analysis(tmp_path, capsys):
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"value": [1, 2, 3]}).to_csv(csv_path, index=False)

    assert main([str(csv_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["kind"] == "tabular"
    assert output["row_count"] == 3
