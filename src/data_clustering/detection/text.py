"""Utilities for detecting natural-language text columns."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class TextColumnProfile:
    """Summary of a column's text-likeness."""

    column_name: str
    is_text: bool
    string_ratio: float
    average_text_length: float
    unique_ratio: float
    empty_ratio: float
    language_like_score: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_name": self.column_name,
            "is_text": self.is_text,
            "string_ratio": round(self.string_ratio, 4),
            "average_text_length": round(self.average_text_length, 4),
            "unique_ratio": round(self.unique_ratio, 4),
            "empty_ratio": round(self.empty_ratio, 4),
            "language_like_score": round(self.language_like_score, 4),
            "reasons": self.reasons,
        }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _language_like_score(values: pd.Series) -> float:
    """Score how strongly a set of strings resembles natural language."""
    if values.empty:
        return 0.0

    text_scores: List[float] = []
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue

        letters = sum(character.isalpha() for character in text)
        alpha_ratio = _safe_ratio(letters, len(text))
        digit_ratio = _safe_ratio(sum(character.isdigit() for character in text), len(text))
        whitespace_ratio = _safe_ratio(sum(character.isspace() for character in text), len(text))
        words = re.findall(r"\b[\w']+\b", text)
        has_word_structure = 1.0 if len(words) > 1 or whitespace_ratio > 0.1 or any(character in string.punctuation for character in text) else 0.0

        score = 0.6 * alpha_ratio + 0.4 * has_word_structure
        if digit_ratio > 0.3 and has_word_structure == 0.0:
            score *= 0.5

        text_scores.append(score)

    if not text_scores:
        return 0.0
    return float(sum(text_scores) / len(text_scores))


def analyze_text_column(series: pd.Series, *, min_string_ratio: float = 0.7, min_language_score: float = 0.35) -> TextColumnProfile:
    """Analyze whether a Series contains natural-language text."""
    values = pd.Series(series, copy=True)
    column_name = values.name if values.name is not None else "unnamed"

    non_null = values.dropna()
    string_values = non_null[non_null.map(lambda value: isinstance(value, str) and bool(str(value).strip()))]

    string_ratio = _safe_ratio(len(string_values), len(non_null)) if len(non_null) else 0.0
    empty_ratio = _safe_ratio(len(values[values.isna() | values.map(lambda item: isinstance(item, str) and not str(item).strip())]), len(values)) if len(values) else 0.0

    average_text_length = float(string_values.map(len).mean()) if not string_values.empty else 0.0
    unique_ratio = _safe_ratio(string_values.nunique(dropna=True), len(string_values)) if len(string_values) else 0.0
    language_like_score = _language_like_score(string_values)

    reasons: List[str] = []
    if string_ratio < min_string_ratio:
        reasons.append("Too few values are string-like to suggest text.")
    if average_text_length < 3:
        reasons.append("Average text length is too short to suggest natural-language text.")
    if language_like_score < min_language_score:
        reasons.append("Content is not sufficiently language-like.")
    if empty_ratio > 0.75:
        reasons.append("Too many values are empty or missing.")

    is_text = (
        string_ratio >= min_string_ratio
        and average_text_length >= 3
        and language_like_score >= min_language_score
        and empty_ratio <= 0.75
    )

    return TextColumnProfile(
        column_name=column_name,
        is_text=is_text,
        string_ratio=string_ratio,
        average_text_length=average_text_length,
        unique_ratio=unique_ratio,
        empty_ratio=empty_ratio,
        language_like_score=language_like_score,
        reasons=reasons,
    )


def detect_text_columns(dataframe: pd.DataFrame) -> Dict[str, TextColumnProfile]:
    """Detect all natural-language text columns in a DataFrame."""
    return {
        column_name: analyze_text_column(series)
        for column_name, series in dataframe.items()
    }
