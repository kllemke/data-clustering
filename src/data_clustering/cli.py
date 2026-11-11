"""Command-line entry point for dataset analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .analysis import analyze_file


def main(argv: Sequence[str] | None = None) -> int:
    """Analyze an input file and write its JSON report to standard output."""
    parser = argparse.ArgumentParser(description="Profile a dataset and recommend clustering algorithms.")
    parser.add_argument("path", type=Path, help="Table file, media file, or directory of media files")
    arguments = parser.parse_args(argv)
    try:
        report = analyze_file(arguments.path)
    except (FileNotFoundError, ValueError, ImportError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0
