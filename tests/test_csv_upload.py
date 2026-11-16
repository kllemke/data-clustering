"""Tests für Ticket #9: CSV-Upload."""

import io

import pytest

from cluster_to_purpose.csv_upload import CSVUploadError, load_csv

NUMERIC_ROWS = "\n".join(f"{i},{i * 1.5},{i % 3}" for i in range(20))


def write(tmp_path, name, content, encoding="utf-8"):
    path = tmp_path / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding=encoding)
    return path


# --- Gültige Dateien -------------------------------------------------------- #
def test_comma_separated_with_header(tmp_path):
    path = write(tmp_path, "data.csv", "a,b,c\n" + NUMERIC_ROWS)
    result = load_csv(path)
    assert result.delimiter == ","
    assert result.has_header
    assert list(result.df.columns) == ["a", "b", "c"]
    assert result.n_rows == 20
    assert result.df["b"].dtype.kind == "f"


def test_german_semicolon_with_decimal_comma(tmp_path):
    rows = "\n".join(f"{i};{i},5;Berlin" for i in range(15))
    path = write(tmp_path, "de.csv", "Nr;Wert;Ort\n" + rows)
    result = load_csv(path)
    assert result.delimiter == ";"
    assert result.decimal == ","
    assert result.df["Wert"].iloc[1] == pytest.approx(1.5)


def test_tab_separated(tmp_path):
    rows = "\n".join(f"{i}\t{i * 2}" for i in range(12))
    result = load_csv(write(tmp_path, "data.tsv", "x\ty\n" + rows))
    assert result.delimiter == "\t"
    assert result.n_cols == 2


def test_windows_encoding_with_umlauts(tmp_path):
    rows = "\n".join(f"Müller;{i}" for i in range(12))
    path = write(tmp_path, "win.csv", "Name;Größe\n" + rows, encoding="cp1252")
    result = load_csv(path)
    assert result.encoding == "cp1252"
    assert "Größe" in result.df.columns
    assert result.df["Name"].iloc[0] == "Müller"
    assert any("UTF-8" in w for w in result.warnings)


def test_utf8_with_bom(tmp_path):
    path = write(tmp_path, "bom.csv", ("a,b,c\n" + NUMERIC_ROWS).encode("utf-8-sig"))
    result = load_csv(path)
    assert result.encoding == "utf-8"
    assert result.df.columns[0] == "a"


def test_without_header(tmp_path):
    result = load_csv(write(tmp_path, "noheader.csv", NUMERIC_ROWS))
    assert not result.has_header
    assert list(result.df.columns) == ["spalte_1", "spalte_2", "spalte_3"]
    assert result.n_rows == 20


def test_categorical_header_detected(tmp_path):
    rows = "\n".join(["m;ja", "w;nein", "m;nein", "w;ja"] * 3)
    result = load_csv(write(tmp_path, "umfrage.csv", "geschlecht;raucher\n" + rows))
    assert result.has_header
    assert list(result.df.columns) == ["geschlecht", "raucher"]


def test_quoted_fields_with_delimiter_inside(tmp_path):
    rows = "\n".join(f'{i},"Berlin, Spandau",{i}' for i in range(12))
    result = load_csv(write(tmp_path, "quoted.csv", "id,ort,wert\n" + rows))
    assert result.n_cols == 3
    assert result.df["ort"].iloc[0] == "Berlin, Spandau"


def test_pandas_index_column_gets_name(tmp_path):
    rows = "\n".join(f"{i},{i},{i * 2}" for i in range(12))
    result = load_csv(write(tmp_path, "index.csv", ",a,b\n" + rows))
    assert result.df.columns[0] == "spalte_1"


def test_empty_rows_and_columns_removed(tmp_path):
    rows = "\n".join(f"{i},,{i}" for i in range(12))
    result = load_csv(write(tmp_path, "gaps.csv", "a,leer,b\n" + rows + "\n,,\n"))
    assert list(result.df.columns) == ["a", "b"]
    assert result.n_rows == 12


def test_file_like_object_like_streamlit():
    buffer = io.BytesIO(("a,b,c\n" + NUMERIC_ROWS).encode())
    buffer.name = "upload.csv"
    result = load_csv(buffer)
    assert result.filename == "upload.csv"
    assert result.preview(3).shape == (3, 3)


def test_bytes_input():
    result = load_csv(("a,b,c\n" + NUMERIC_ROWS).encode(), filename="x.csv")
    assert result.n_rows == 20


def test_few_rows_gives_warning(tmp_path):
    result = load_csv(write(tmp_path, "small.csv", "a,b\n1,2\n3,4\n"))
    assert any("Datenzeilen" in w for w in result.warnings)


# --- Ungültige Dateien ------------------------------------------------------ #
@pytest.mark.parametrize(
    "name, content, message",
    [
        ("empty.csv", b"", "leer"),
        ("blank.csv", "   \n\n", "leer"),
        ("header_only.csv", "a,b,c\n", "keine Datenzeilen"),
        ("data.xlsx", "a,b\n1,2", "Dateityp"),
        ("excel.csv", b"PK\x03\x04rest", "Excel"),
        ("binary.csv", b"\x00\x01\x02\x03", "keine Textdatei"),
        ("broken.csv", "a,b\n1,2\n3,4,5,6\n", "gleich viele Spalten"),
    ],
)
def test_invalid_files(tmp_path, name, content, message):
    with pytest.raises(CSVUploadError, match=message):
        load_csv(write(tmp_path, name, content))


def test_file_too_large(tmp_path):
    path = write(tmp_path, "big.csv", "a,b\n" + "1,2\n" * 1000)
    with pytest.raises(CSVUploadError, match="größer als"):
        load_csv(path, max_size_mb=0.001)


def test_missing_file(tmp_path):
    with pytest.raises(CSVUploadError, match="nicht gefunden"):
        load_csv(tmp_path / "gibt_es_nicht.csv")
