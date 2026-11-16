"""
Ticket #9 – CSV-Upload.

Liest eine CSV-Datei robust ein und gibt einen pandas-DataFrame plus Metadaten
zurück. Der Code hängt von keinem UI-Framework ab: load_csv() akzeptiert einen
Dateipfad, Bytes oder ein Datei-Objekt (z. B. Streamlit-UploadedFile oder
Flask-FileStorage).

Erkannt werden automatisch:
- Trennzeichen (, ; Tab |)
- Encoding (UTF-8, mit oder ohne BOM, Windows-1252, Latin-1)
- Kopfzeile ja/nein
- deutsches Dezimalkomma (1,5) bei Semikolon- oder Tab-getrennten Dateien

Alle Fehler werden als CSVUploadError mit einer verständlichen deutschen
Meldung geworfen, die das UI direkt anzeigen kann.
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".csv", ".txt", ".tsv"}
CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]
CANDIDATE_ENCODINGS = ["utf-8-sig", "cp1252", "latin-1"]
SNIFF_LINES = 50          # so viele Zeilen werden für die Erkennung angeschaut
MIN_ROWS_WARNING = 10     # darunter ist Clustering kaum sinnvoll

_NUMBER_RE = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")
_GERMAN_DECIMAL_RE = re.compile(r"^[+-]?\d+,\d+$")
_DOT_DECIMAL_RE = re.compile(r"^[+-]?\d+\.\d+$")

Source = Union[str, Path, bytes, bytearray, BinaryIO]


class CSVUploadError(Exception):
    """Fehler, dessen Meldung man dem Nutzer direkt anzeigen kann."""


@dataclass
class UploadResult:
    """Ergebnis eines erfolgreichen Uploads, wird an das Profiling (#14) übergeben."""

    df: pd.DataFrame
    filename: str
    delimiter: str
    encoding: str
    decimal: str
    has_header: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def n_cols(self) -> int:
        return self.df.shape[1]

    def preview(self, n: int = 5) -> pd.DataFrame:
        """Die ersten n Zeilen für die Vorschau im UI."""
        return self.df.head(n)

    def summary(self) -> dict:
        delimiter_names = {",": "Komma", ";": "Semikolon", "\t": "Tab", "|": "Pipe"}
        return {
            "filename": self.filename,
            "rows": self.n_rows,
            "columns": self.n_cols,
            "delimiter": delimiter_names.get(self.delimiter, self.delimiter),
            "encoding": self.encoding,
            "decimal": self.decimal,
            "has_header": self.has_header,
            "column_names": list(self.df.columns),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# Öffentliche Funktion
# --------------------------------------------------------------------------- #
def load_csv(
    source: Source,
    filename: str | None = None,
    *,
    max_size_mb: float = MAX_FILE_SIZE_MB,
    delimiter: str | None = None,
    has_header: bool | None = None,
) -> UploadResult:
    """
    Liest eine CSV ein.

    source       Pfad, Bytes oder Datei-Objekt mit .read()
    filename     optionaler Dateiname (für die Endungsprüfung bei Bytes)
    max_size_mb  maximale Dateigröße
    delimiter    Trennzeichen erzwingen (sonst automatisch)
    has_header   Kopfzeile erzwingen (sonst automatisch)
    """
    raw, name = _read_bytes(source, filename, max_size_mb)
    _check_file(raw, name, max_size_mb)

    warnings: list[str] = []
    text, encoding = _decode(raw)
    if encoding != "utf-8":
        warnings.append(f"Die Datei ist nicht UTF-8-kodiert und wurde als {encoding} gelesen.")
    if not text.strip():
        raise CSVUploadError("Die Datei ist leer.")

    if delimiter is None:
        delimiter, single_column = _detect_delimiter(text)
        if single_column:
            warnings.append(
                "Es wurde nur eine Spalte erkannt. Prüfe, ob das Trennzeichen stimmt."
            )

    rows = _sample_rows(text, delimiter)
    if has_header is None:
        has_header = _detect_header(rows)
    decimal = _detect_decimal(rows[1:] if has_header else rows, delimiter)

    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            header=0 if has_header else None,
            decimal=decimal,
            skip_blank_lines=True,
        )
    except pd.errors.EmptyDataError as exc:
        raise CSVUploadError("Die Datei enthält keine Daten.") from exc
    except pd.errors.ParserError as exc:
        raise CSVUploadError(
            "Die CSV ist fehlerhaft: Nicht alle Zeilen haben gleich viele Spalten. "
            f"Details: {_short_parser_message(exc)}"
        ) from exc

    df = _clean(df, has_header, warnings)

    if df.empty or df.shape[1] == 0:
        raise CSVUploadError("Die Datei enthält keine Datenzeilen (nur eine Kopfzeile oder leere Zeilen).")
    if len(df) < MIN_ROWS_WARNING:
        warnings.append(
            f"Nur {len(df)} Datenzeilen. Für ein aussagekräftiges Clustering sind mehr Daten nötig."
        )

    return UploadResult(
        df=df,
        filename=name,
        delimiter=delimiter,
        encoding=encoding,
        decimal=decimal,
        has_header=has_header,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Einlesen und Prüfen
# --------------------------------------------------------------------------- #
def _read_bytes(source: Source, filename: str | None, max_size_mb: float) -> tuple[bytes, str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise CSVUploadError(f"Datei nicht gefunden: {path}")
        if path.stat().st_size > max_size_mb * 1024 * 1024:
            raise CSVUploadError(f"Die Datei ist größer als {max_size_mb:g} MB.")
        return path.read_bytes(), filename or path.name

    if isinstance(source, (bytes, bytearray)):
        return bytes(source), filename or "upload.csv"

    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        name = filename or getattr(source, "name", None) or getattr(source, "filename", None) or "upload.csv"
        return data, Path(str(name)).name

    raise CSVUploadError("Nicht unterstützter Eingabetyp. Erwartet: Pfad, Bytes oder Datei-Objekt.")


def _check_file(raw: bytes, name: str, max_size_mb: float) -> None:
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise CSVUploadError(
            f"Dateityp '{ext or 'ohne Endung'}' wird nicht unterstützt. Erlaubt: "
            + ", ".join(sorted(ALLOWED_EXTENSIONS))
        )
    if len(raw) == 0:
        raise CSVUploadError("Die Datei ist leer.")
    if len(raw) > max_size_mb * 1024 * 1024:
        raise CSVUploadError(f"Die Datei ist größer als {max_size_mb:g} MB.")
    if raw.startswith(b"PK\x03\x04"):
        raise CSVUploadError(
            "Das ist eine Excel- oder ZIP-Datei, keine CSV. Bitte in Excel über "
            "'Speichern unter → CSV' exportieren."
        )
    if b"\x00" in raw[:4096]:
        raise CSVUploadError("Die Datei ist keine Textdatei (Binärdaten erkannt).")


def _decode(raw: bytes) -> tuple[str, str]:
    for enc in CANDIDATE_ENCODINGS:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        return text, "utf-8" if enc == "utf-8-sig" else enc
    # latin-1 dekodiert jedes Byte, hier kommt man praktisch nie an
    raise CSVUploadError("Das Encoding der Datei konnte nicht erkannt werden.")


# --------------------------------------------------------------------------- #
# Erkennung von Trennzeichen, Kopfzeile, Dezimalzeichen
# --------------------------------------------------------------------------- #
def _sample_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()][:SNIFF_LINES]


def _sample_rows(text: str, delimiter: str) -> list[list[str]]:
    return list(csv.reader(_sample_lines(text), delimiter=delimiter))


def _detect_delimiter(text: str) -> tuple[str, bool]:
    """
    Wählt das Trennzeichen, bei dem die Zeilen am gleichmäßigsten in mehr als
    eine Spalte zerfallen. csv.reader beachtet dabei Anführungszeichen.
    Rückgabe: (Trennzeichen, nur_eine_Spalte)
    """
    lines = _sample_lines(text)
    best, best_score = ",", (0.0, 0)
    for cand in CANDIDATE_DELIMITERS:
        counts = [len(row) for row in csv.reader(lines, delimiter=cand)]
        if not counts:
            continue
        mode, freq = Counter(counts).most_common(1)[0]
        if mode < 2:
            continue
        score = (round(freq / len(counts), 3), mode)  # erst Gleichmäßigkeit, dann Spaltenzahl
        if score > best_score:
            best, best_score = cand, score
    return best, best_score == (0.0, 0)


def _is_number(value: str) -> bool:
    return bool(_NUMBER_RE.match(value.strip()))


def _detect_header(rows: list[list[str]]) -> bool:
    """
    Heuristik:
    - Enthält die erste Zeile eine Zahl → keine Kopfzeile.
    - Enthalten die übrigen Zeilen Zahlen, die erste aber nicht → Kopfzeile.
    - Nur Text: Kopfzeile, wenn die Werte der ersten Zeile eindeutig sind und
      in ihrer Spalte nicht noch einmal vorkommen.
    """
    if len(rows) < 2:
        return True
    first, rest = rows[0], rows[1:]
    non_empty = [c.strip() for c in first if c.strip()]
    if not non_empty:
        return False
    if any(_is_number(c) for c in non_empty):
        return False
    if any(_is_number(c) for row in rest for c in row if c.strip()):
        return True
    if len(set(non_empty)) != len(non_empty):
        return False
    for i, value in enumerate(first):
        value = value.strip()
        if value and any(i < len(r) and r[i].strip() == value for r in rest):
            return False
    return True


def _detect_decimal(rows: list[list[str]], delimiter: str) -> str:
    if delimiter == ",":
        return "."
    cells = [c.strip() for row in rows for c in row]
    german = sum(bool(_GERMAN_DECIMAL_RE.match(c)) for c in cells)
    dot = sum(bool(_DOT_DECIMAL_RE.match(c)) for c in cells)
    return "," if german > 0 and german >= dot else "."


# --------------------------------------------------------------------------- #
# Aufräumen
# --------------------------------------------------------------------------- #
def _clean(df: pd.DataFrame, has_header: bool, warnings: list[str]) -> pd.DataFrame:
    if not has_header:
        df.columns = [f"spalte_{i + 1}" for i in range(df.shape[1])]
        warnings.append("Keine Kopfzeile erkannt. Spalten wurden spalte_1, spalte_2 … benannt.")
    else:
        names, renamed = [], 0
        for i, col in enumerate(df.columns):
            col = str(col).strip()
            if not col or col.startswith("Unnamed:"):
                col = f"spalte_{i + 1}"
                renamed += 1
            names.append(col)
        if renamed:
            warnings.append(f"{renamed} Spalte(n) ohne Namen wurden automatisch benannt.")
        df.columns = _make_unique(names, warnings)

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        warnings.append(f"{len(empty_cols)} komplett leere Spalte(n) entfernt: {', '.join(map(str, empty_cols))}")

    n_before = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    if len(df) < n_before:
        warnings.append(f"{n_before - len(df)} komplett leere Zeile(n) entfernt.")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    return df


def _make_unique(names: list[str], warnings: list[str]) -> list[str]:
    seen: Counter[str] = Counter()
    result = []
    for name in names:
        seen[name] += 1
        result.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    duplicates = [n for n, c in seen.items() if c > 1]
    if duplicates:
        warnings.append(f"Doppelte Spaltennamen umbenannt: {', '.join(duplicates)}")
    return result


def _short_parser_message(exc: Exception) -> str:
    msg = str(exc).strip().splitlines()[-1] if str(exc).strip() else ""
    return msg.replace("Error tokenizing data. C error: ", "")


# --------------------------------------------------------------------------- #
# Kommandozeile:  python -m cluster_to_purpose.csv_upload datei.csv
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Aufruf: python -m cluster_to_purpose.csv_upload <datei.csv>")
        sys.exit(2)
    try:
        result = load_csv(sys.argv[1])
    except CSVUploadError as err:
        print(f"Fehler: {err}")
        sys.exit(1)
    for key, value in result.summary().items():
        print(f"{key:>13}: {value}")
    print()
    print(result.preview().to_string())
