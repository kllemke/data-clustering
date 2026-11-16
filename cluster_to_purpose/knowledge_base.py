"""
Wissensbasis laden und Regeln gegen ein Datenprofil abgleichen.

match_rules() ist eine bewusst einfache Referenz-Implementierung, mit der die
Wissensbasis in Ticket #8 getestet wird. Die eigentliche Empfehlungs-Engine mit
Ranking und Begründung baut Ticket #22. Sie kann match_rules() wiederverwenden.

Datenprofil (Vertrag mit #14, #18, #19):
    {
        "data_type": "numeric" | "categorical" | "mixed" | "time_series"
                     | "text" | "stream" | "uncertain",
        "n_rows": int,            # Anzahl Objekte (Zeilen bzw. Zeitreihen)
        "n_features": int,        # Anzahl Merkmale bzw. Zeitreihenlänge
        "outlier_ratio": float,   # Anteil Ausreißer, 0..1 (aus #19)
    }
Fehlt ein Wert im Profil, wird die zugehörige Bedingung nicht geprüft.
"""

from __future__ import annotations

import json
import operator
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_KB_PATH = DATA_DIR / "knowledge_base.json"
DEFAULT_CASES_PATH = DATA_DIR / "kb_test_cases.json"

DATA_TYPES = {"numeric", "categorical", "mixed", "time_series", "text", "stream", "uncertain"}

# Bedingungsname -> (Feld im Profil, Vergleich "Profilwert OP Grenzwert")
CONDITION_CHECKS = {
    "min_rows": ("n_rows", operator.ge),
    "max_rows": ("n_rows", operator.le),
    "min_features": ("n_features", operator.ge),
    "max_features": ("n_features", operator.le),
    "min_outlier_ratio": ("outlier_ratio", operator.ge),
    "max_outlier_ratio": ("outlier_ratio", operator.le),
}


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_knowledge_base(path: str | Path = DEFAULT_KB_PATH) -> list[dict]:
    return load_json(path)["rules"]


def rule_applies(rule: dict, profile: dict) -> bool:
    """True, wenn Datentyp und alle prüfbaren Bedingungen der Regel passen."""
    if profile.get("data_type") not in rule.get("data_types", []):
        return False
    for name, limit in rule.get("conditions", {}).items():
        field, compare = CONDITION_CHECKS[name]
        value = profile.get(field)
        if value is not None and not compare(value, limit):
            return False
    return True


def match_rules(profile: dict, rules: list[dict]) -> dict:
    """
    Liefert passende Empfehlungen und Warnungen ("avoid"-Regeln).
    Eine passende avoid-Regel schlägt eine Empfehlung für denselben Algorithmus.
    Passt keine Empfehlung, ist der Status "uncertain" statt einer Notlösung.
    """
    applicable = [r for r in rules if rule_applies(r, profile)]
    avoid = [r for r in applicable if r["verdict"] == "avoid"]
    avoided = {r["algorithm"] for r in avoid}
    recommended = [
        r for r in applicable if r["verdict"] == "recommend" and r["algorithm"] not in avoided
    ]
    if recommended:
        status, message = "ok", f"{len(recommended)} passende Algorithmen gefunden."
    else:
        status = "uncertain"
        message = (
            "Für dieses Datenprofil gibt es keine gesicherte Empfehlung. "
            "Prüfe die Vorverarbeitung oder wähle den Algorithmus manuell."
        )
    return {"status": status, "message": message, "recommended": recommended, "avoid": avoid}
