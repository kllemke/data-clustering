"""
Ticket #8 – Review und Validierung der Wissensbasis.

Drei Prüfungen:
1. Struktur:    Hat jede Regel alle Pflichtfelder mit gültigen Werten?
2. Fachlich:    Bekannte Fehlkombinationen (z. B. K-Means für kategoriale Daten),
                fehlende Vorverarbeitung bei Zeitreihen und Text, unpassende Metriken.
3. Konsistenz:  Doppelte Regeln, Widersprüche zwischen recommend und avoid,
                Datentypen ohne Empfehlung.
Dazu kommen die Testfälle aus data/kb_test_cases.json.

Aufruf:
    python -m cluster_to_purpose.kb_validation
    python -m cluster_to_purpose.kb_validation --report kb_review.md
Exit-Code 1, wenn Fehler gefunden wurden. So kann man es auch in GitHub Actions nutzen.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path

from .knowledge_base import (
    CONDITION_CHECKS,
    DATA_TYPES,
    DEFAULT_CASES_PATH,
    DEFAULT_KB_PATH,
    load_json,
    match_rules,
)

VERDICTS = {"recommend", "avoid"}

REQUIRED_FIELDS = {
    "recommend": {
        "id": str, "algorithm": str, "verdict": str, "data_types": list,
        "conditions": dict, "preconditions": list, "rationale": str,
        "risks": list, "evaluation": list, "source": str,
    },
    "avoid": {
        "id": str, "algorithm": str, "verdict": str, "data_types": list,
        "conditions": dict, "rationale": str, "source": str,
    },
}

METRICS = {
    "silhouette", "davies_bouldin", "calinski_harabasz", "bic", "dbcv", "noise_ratio",
    "cophenetic_correlation", "within_cluster_cost", "quantization_error",
    "topographic_error", "windowed_silhouette", "adjusted_rand",
}
CENTROID_METRICS = {"silhouette", "davies_bouldin", "calinski_harabasz"}

# Fachwissen: Für diese Datentypen darf ein Algorithmus NICHT empfohlen werden
FORBIDDEN_DATA_TYPES = {
    "K-Means": {"categorical", "mixed"},
    "GMM": {"categorical", "mixed"},
    "Agglomerative (Ward)": {"categorical", "mixed"},
    "K-Modes": {"numeric", "time_series"},
}
DENSITY_BASED = {"DBSCAN", "HDBSCAN", "DenStream"}
OUTLIER_SENSITIVE = {"K-Means", "GMM", "Agglomerative (Ward)"}
NEEDS_PREPROCESSING = {"time_series", "text"}
MIN_RATIONALE_CHARS = 30


@dataclass
class Issue:
    severity: str   # "error" oder "warning"
    rule_id: str
    check: str
    message: str


# --------------------------------------------------------------------------- #
# 1. Struktur
# --------------------------------------------------------------------------- #
def check_structure(rules: list[dict]) -> list[Issue]:
    issues: list[Issue] = []
    seen_ids: set[str] = set()
    for i, rule in enumerate(rules):
        rid = str(rule.get("id", f"#{i + 1}"))
        if rid in seen_ids:
            issues.append(Issue("error", rid, "Struktur", "Regel-ID ist doppelt vergeben."))
        seen_ids.add(rid)

        verdict = rule.get("verdict")
        if verdict not in VERDICTS:
            issues.append(Issue("error", rid, "Struktur", f"verdict muss 'recommend' oder 'avoid' sein, ist {verdict!r}."))
            continue

        for name, typ in REQUIRED_FIELDS[verdict].items():
            if name not in rule:
                issues.append(Issue("error", rid, "Struktur", f"Pflichtfeld '{name}' fehlt."))
            elif not isinstance(rule[name], typ):
                issues.append(Issue("error", rid, "Struktur", f"Feld '{name}' muss vom Typ {typ.__name__} sein."))

        for dt in rule.get("data_types", []):
            if dt not in DATA_TYPES:
                issues.append(Issue("error", rid, "Struktur", f"Unbekannter Datentyp '{dt}'."))
        if isinstance(rule.get("data_types"), list) and not rule["data_types"]:
            issues.append(Issue("error", rid, "Struktur", "data_types ist leer."))

        issues += _check_conditions(rid, rule.get("conditions", {}))

        if verdict == "recommend":
            if isinstance(rule.get("risks"), list) and not rule["risks"]:
                issues.append(Issue("error", rid, "Struktur", "Eine Empfehlung braucht mindestens ein Risiko."))
            if isinstance(rule.get("evaluation"), list) and not rule["evaluation"]:
                issues.append(Issue("error", rid, "Struktur", "Eine Empfehlung braucht mindestens eine Bewertungsmetrik."))
            for metric in rule.get("evaluation", []):
                if metric not in METRICS:
                    issues.append(Issue("error", rid, "Struktur", f"Unbekannte Metrik '{metric}'."))

        if len(str(rule.get("rationale", ""))) < MIN_RATIONALE_CHARS:
            issues.append(Issue("warning", rid, "Struktur", "Begründung ist sehr kurz."))
        if not str(rule.get("source", "")).strip():
            issues.append(Issue("error", rid, "Quelle", "Keine Quelle angegeben."))
        if not str(rule.get("paper_section", "")).strip():
            issues.append(Issue("warning", rid, "Quelle", "Abschnitt im Kurs-Paper noch nicht eingetragen."))
    return issues


def _check_conditions(rid: str, conditions: dict) -> list[Issue]:
    issues = []
    for name, value in conditions.items():
        if name not in CONDITION_CHECKS:
            issues.append(Issue("error", rid, "Struktur", f"Unbekannte Bedingung '{name}'."))
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            issues.append(Issue("error", rid, "Struktur", f"Bedingung '{name}' muss eine Zahl sein."))
        elif value < 0 or ("ratio" in name and value > 1):
            issues.append(Issue("error", rid, "Struktur", f"Bedingung '{name}' hat einen ungültigen Wert ({value})."))
    for lo, hi in (("min_rows", "max_rows"), ("min_features", "max_features"),
                   ("min_outlier_ratio", "max_outlier_ratio")):
        if lo in conditions and hi in conditions and _num(conditions[lo]) > _num(conditions[hi]):
            issues.append(Issue("error", rid, "Struktur", f"{lo} ist größer als {hi}."))
    return issues


def _num(value) -> float:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


# --------------------------------------------------------------------------- #
# 2. Fachliche Prüfung
# --------------------------------------------------------------------------- #
def check_domain(rules: list[dict]) -> list[Issue]:
    issues: list[Issue] = []
    for rule in rules:
        if rule.get("verdict") != "recommend":
            continue
        rid, algo = rule.get("id", "?"), rule.get("algorithm", "")
        types = set(rule.get("data_types", []))
        conditions = rule.get("conditions", {})
        metrics = set(rule.get("evaluation", []))

        bad = types & FORBIDDEN_DATA_TYPES.get(algo, set())
        if bad:
            issues.append(Issue("error", rid, "Fachlich",
                                f"{algo} wird für {', '.join(sorted(bad))} empfohlen, ist dafür aber ungeeignet."))

        needs = types & NEEDS_PREPROCESSING
        if needs and not rule.get("preconditions"):
            issues.append(Issue("error", rid, "Fachlich",
                                f"Für {', '.join(sorted(needs))} fehlt die nötige Vorverarbeitung "
                                "(z. B. DTW-Distanz, Merkmalsextraktion, TF-IDF)."))

        if algo in DENSITY_BASED and metrics and metrics <= CENTROID_METRICS:
            issues.append(Issue("warning", rid, "Fachlich",
                                "Nur zentroidbasierte Metriken. Diese bevorzugen kugelförmige Cluster "
                                "und bewerten dichtebasierte Ergebnisse zu schlecht. DBCV ergänzen."))

        if algo in OUTLIER_SENSITIVE and "max_outlier_ratio" not in conditions:
            issues.append(Issue("warning", rid, "Fachlich",
                                f"{algo} ist ausreißerempfindlich, aber die Regel hat keine Grenze max_outlier_ratio."))

        if algo in DENSITY_BASED and "numeric" in types and "max_features" not in conditions:
            issues.append(Issue("warning", rid, "Fachlich",
                                f"{algo} wird ohne Grenze max_features empfohlen. Bei vielen Dimensionen versagt die Dichteschätzung."))
    return issues


# --------------------------------------------------------------------------- #
# 3. Konsistenz
# --------------------------------------------------------------------------- #
def _ranges_overlap(a: dict, b: dict) -> bool:
    """True, wenn es ein Datenprofil gibt, das beide Bedingungssätze erfüllt."""
    for field in ("rows", "features", "outlier_ratio"):
        lo = max(a.get(f"min_{field}", float("-inf")), b.get(f"min_{field}", float("-inf")))
        hi = min(a.get(f"max_{field}", float("inf")), b.get(f"max_{field}", float("inf")))
        if lo > hi:
            return False
    return True


def check_consistency(rules: list[dict]) -> list[Issue]:
    issues: list[Issue] = []
    valid = [r for r in rules if r.get("verdict") in VERDICTS]
    for a, b in combinations(valid, 2):
        if a.get("algorithm") != b.get("algorithm"):
            continue
        shared = set(a.get("data_types", [])) & set(b.get("data_types", []))
        if not shared or not _ranges_overlap(a.get("conditions", {}), b.get("conditions", {})):
            continue
        pair = f"{a.get('id')}/{b.get('id')}"
        if a["verdict"] != b["verdict"]:
            issues.append(Issue("error", pair, "Konsistenz",
                                f"Widerspruch: {a['algorithm']} wird für {', '.join(sorted(shared))} "
                                "unter denselben Bedingungen empfohlen und abgelehnt."))
        else:
            issues.append(Issue("warning", pair, "Konsistenz",
                                f"Doppelte Regel für {a['algorithm']} ({', '.join(sorted(shared))}). Zusammenführen."))

    covered = {dt for r in valid if r["verdict"] == "recommend" for dt in r.get("data_types", [])}
    for dt in sorted(DATA_TYPES - covered):
        issues.append(Issue("warning", "-", "Abdeckung", f"Für den Datentyp '{dt}' gibt es keine Empfehlung."))
    return issues


def validate_rules(rules: list[dict]) -> list[Issue]:
    return check_structure(rules) + check_domain(rules) + check_consistency(rules)


# --------------------------------------------------------------------------- #
# Testfälle
# --------------------------------------------------------------------------- #
@dataclass
class CaseResult:
    case_id: str
    description: str
    passed: bool
    recommended: list[str]
    avoided: list[str]
    problems: list[str]


def run_case(case: dict, rules: list[dict]) -> CaseResult:
    result = match_rules(case["profile"], rules)
    rec = sorted({r["algorithm"] for r in result["recommended"]})
    avo = sorted({r["algorithm"] for r in result["avoid"]})
    problems = []

    expected_status = case.get("expect_status")
    if expected_status and result["status"] != expected_status:
        problems.append(f"Status {result['status']!r} statt {expected_status!r}")
    if case.get("expect_any") and not set(case["expect_any"]) & set(rec):
        problems.append(f"keiner von {case['expect_any']} empfohlen")
    wrong = set(case.get("expect_none", [])) & set(rec)
    if wrong:
        problems.append(f"fälschlich empfohlen: {sorted(wrong)}")
    missing_avoid = set(case.get("expect_avoid", [])) - set(avo)
    if missing_avoid:
        problems.append(f"Warnung fehlt für: {sorted(missing_avoid)}")

    return CaseResult(case["id"], case.get("description", ""), not problems, rec, avo, problems)


def run_cases(cases: list[dict], rules: list[dict]) -> list[CaseResult]:
    return [run_case(c, rules) for c in cases]


# --------------------------------------------------------------------------- #
# Review-Bericht
# --------------------------------------------------------------------------- #
def to_markdown(rules: list[dict], issues: list[Issue], results: list[CaseResult]) -> str:
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    passed = sum(r.passed for r in results)

    lines = [
        "# Review der Wissensbasis (Ticket #8)",
        "",
        f"Stand: {date.today().isoformat()} · {len(rules)} Regeln · "
        f"{len(errors)} Fehler · {len(warnings)} Warnungen · Testfälle {passed}/{len(results)} bestanden",
        "",
        "## Gefundene Probleme",
        "",
    ]
    if issues:
        lines += ["| Schwere | Regel | Prüfung | Befund |", "| --- | --- | --- | --- |"]
        for i in sorted(issues, key=lambda x: (x.severity != "error", x.rule_id)):
            label = "Fehler" if i.severity == "error" else "Warnung"
            lines.append(f"| {label} | {i.rule_id} | {i.check} | {i.message} |")
    else:
        lines.append("Keine Probleme gefunden.")

    lines += ["", "## Testfälle", "",
              "| Fall | Beschreibung | Ergebnis | Empfohlen | Gewarnt |", "| --- | --- | --- | --- | --- |"]
    for r in results:
        status = "bestanden" if r.passed else "FEHLER: " + "; ".join(r.problems)
        lines.append(f"| {r.case_id} | {r.description} | {status} | "
                     f"{', '.join(r.recommended) or '–'} | {', '.join(r.avoided) or '–'} |")

    lines += ["", "## Regeln im Überblick", "",
              "| ID | Algorithmus | Urteil | Datentypen | Quelle |", "| --- | --- | --- | --- | --- |"]
    for rule in rules:
        lines.append(f"| {rule.get('id')} | {rule.get('algorithm')} | {rule.get('verdict')} | "
                     f"{', '.join(rule.get('data_types', []))} | {rule.get('source', '')} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wissensbasis prüfen (Ticket #8)")
    parser.add_argument("--kb", default=DEFAULT_KB_PATH, type=Path)
    parser.add_argument("--cases", default=DEFAULT_CASES_PATH, type=Path)
    parser.add_argument("--report", type=Path, help="Review-Bericht als Markdown speichern")
    args = parser.parse_args(argv)

    rules = load_json(args.kb)["rules"]
    issues = validate_rules(rules)
    results = run_cases(load_json(args.cases)["cases"], rules)

    for i in issues:
        print(f"[{i.severity.upper():7}] {i.rule_id:8} {i.check:11} {i.message}")
    for r in results:
        print(f"[{'OK' if r.passed else 'FAIL':7}] {r.case_id:8} {r.description}"
              + ("" if r.passed else "  -> " + "; ".join(r.problems)))

    if args.report:
        args.report.write_text(to_markdown(rules, issues, results), encoding="utf-8")
        print(f"\nBericht gespeichert: {args.report}")

    n_errors = sum(i.severity == "error" for i in issues) + sum(not r.passed for r in results)
    print(f"\n{n_errors} Fehler insgesamt.")
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
