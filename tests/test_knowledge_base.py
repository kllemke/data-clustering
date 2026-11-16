"""Tests für Ticket #8: Wissensbasis prüfen."""

import copy

import pytest

from cluster_to_purpose.kb_validation import (
    check_consistency,
    check_domain,
    check_structure,
    run_case,
    validate_rules,
)
from cluster_to_purpose.knowledge_base import (
    DEFAULT_CASES_PATH,
    load_json,
    load_knowledge_base,
    match_rules,
)

RULES = load_knowledge_base()
CASES = load_json(DEFAULT_CASES_PATH)["cases"]


def errors(issues):
    return [i for i in issues if i.severity == "error"]


def rule_by_id(rule_id):
    return copy.deepcopy(next(r for r in RULES if r["id"] == rule_id))


# --- Die echte Wissensbasis ------------------------------------------------- #
def test_knowledge_base_has_no_errors():
    found = errors(validate_rules(RULES))
    assert not found, "\n".join(f"{i.rule_id}: {i.message}" for i in found)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_expected_recommendations(case):
    result = run_case(case, RULES)
    assert result.passed, f"{case['id']}: {'; '.join(result.problems)}"


def test_every_recommendation_explains_itself():
    for rule in RULES:
        if rule["verdict"] == "recommend":
            assert rule["rationale"] and rule["risks"] and rule["evaluation"], rule["id"]


def test_fallback_returns_message_instead_of_guess():
    result = match_rules({"data_type": "numeric", "n_rows": 3}, RULES)
    assert result["status"] == "uncertain"
    assert result["recommended"] == []
    assert "keine gesicherte Empfehlung" in result["message"]


def test_avoid_rule_overrides_recommendation():
    rules = [rule_by_id("R01"), rule_by_id("A02")]
    rules[0]["conditions"].pop("max_outlier_ratio")
    result = match_rules({"data_type": "numeric", "n_rows": 500, "n_features": 3, "outlier_ratio": 0.2}, rules)
    assert [r["algorithm"] for r in result["recommended"]] == []
    assert [r["algorithm"] for r in result["avoid"]] == ["K-Means"]


# --- Der Validator erkennt kaputte Regeln ---------------------------------- #
def test_detects_missing_field():
    rule = rule_by_id("R01")
    del rule["risks"]
    assert any("risks" in i.message for i in errors(check_structure([rule])))


def test_detects_duplicate_id():
    assert any("doppelt" in i.message for i in errors(check_structure([rule_by_id("R01"), rule_by_id("R01")])))


def test_detects_unknown_condition_and_bad_range():
    rule = rule_by_id("R01")
    rule["conditions"] = {"min_rows": 100, "max_rows": 10, "max_colums": 5}
    messages = [i.message for i in errors(check_structure([rule]))]
    assert any("Unbekannte Bedingung" in m for m in messages)
    assert any("min_rows ist größer als max_rows" in m for m in messages)


def test_detects_kmeans_for_categorical():
    rule = rule_by_id("R01")
    rule["data_types"] = ["categorical"]
    assert any("ungeeignet" in i.message for i in errors(check_domain([rule])))


def test_detects_time_series_without_preprocessing():
    rule = rule_by_id("R11")
    rule["preconditions"] = []
    assert any("Vorverarbeitung" in i.message for i in errors(check_domain([rule])))


def test_warns_density_algorithm_with_only_centroid_metrics():
    rule = rule_by_id("R03")
    rule["evaluation"] = ["silhouette", "davies_bouldin"]
    assert any("DBCV" in i.message for i in check_domain([rule]))


def test_detects_contradiction():
    recommend = rule_by_id("R01")
    avoid = rule_by_id("A02")
    avoid["conditions"] = {"min_outlier_ratio": 0.01}   # überlappt mit max 0.05
    assert any("Widerspruch" in i.message for i in errors(check_consistency([recommend, avoid])))


def test_no_contradiction_when_ranges_do_not_overlap():
    found = check_consistency([rule_by_id("R01"), rule_by_id("A02")])
    assert not any("Widerspruch" in i.message for i in found)
