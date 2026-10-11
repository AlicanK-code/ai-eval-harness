import json

import pytest

from evalharness.dataset import Case, DatasetError, health_report, load_cases


def write(tmp_path, rows):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_duplicate_ids_rejected(tmp_path):
    row = {"id": "a", "input": {"q": "x"}, "reference": "y"}
    with pytest.raises(DatasetError, match="duplicate"):
        load_cases(write(tmp_path, [row, row]))


def test_missing_fields_rejected(tmp_path):
    with pytest.raises(DatasetError, match="missing"):
        load_cases(write(tmp_path, [{"id": "a", "input": {}}]))


def test_health_report_flags_problems():
    cases = [
        Case("a", {"q": "What is the capital of France? Paris, obviously."}, "Paris"),
        Case("b", {"q": "what is the capital of France?  paris, obviously"}, "Paris"),
        Case("c", {"other": "x"}, "y"),
    ]
    report = health_report(cases, "{q}")
    assert report["missing_template_fields"] == {"c": ["q"]}
    assert report["near_duplicates"] == [("a", "b")]
    assert set(report["reference_in_prompt"]) == {"a", "b"}
    assert report["majority_reference_share"] == pytest.approx(2 / 3, abs=1e-3)
