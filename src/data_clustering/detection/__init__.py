"""Built-in data-type detection modules."""

from .text import TextColumnProfile, analyze_text_column, detect_text_columns

__all__ = ["TextColumnProfile", "analyze_text_column", "detect_text_columns"]
