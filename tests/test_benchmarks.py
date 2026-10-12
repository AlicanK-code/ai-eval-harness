"""Benchmark adapters and the mixed-format grader, tested offline with fake dataset rows."""

import json
import random
from types import SimpleNamespace

import pytest

from evalharness import benchmarks
from evalharness.dataset import Case, health_report
from evalharness.graders import GradeContext
from evalharness.graders.answer_match import AnswerMatch, final_answer


def ctx(reference, answer_type, **meta):
    return GradeContext(Case("c", {}, reference, meta={"answer_type": answer_type, **meta}), prompt="Q")


class TestAnswerMatch:
    g = AnswerMatch("correct", judge_provider="mock")

    @pytest.mark.parametrize("out", ["ANSWER: C", "answer: (C)", "**Answer:** C. Because...", "Final answer: c"])
    def test_choice_formats(self, out):
        assert self.g.grade("working mentions (A) and B\n" + out, ctx("C", "choice")).passed

    def test_only_the_last_answer_line_counts(self):
        assert not self.g.grade("ANSWER: C\nOn reflection...\nANSWER: D", ctx("C", "choice")).passed

    def test_label_and_numeric(self):
        assert self.g.grade("ANSWER: Not enough info.", ctx("NOT ENOUGH INFO", "label")).passed
        assert not self.g.grade("ANSWER: SUPPORT", ctx("CONTRADICT", "label")).passed
        assert self.g.grade("ANSWER: 1,250", ctx("1250", "numeric")).passed
        assert not self.g.grade("ANSWER: 1251", ctx("1250", "numeric")).passed

    def test_missing_answer_line_is_unanswered(self):
        grade = self.g.grade("I believe it's C", ctx("C", "choice"))
        assert not grade.passed and not grade.answered

    def test_free_answers_go_to_the_judge_with_the_short_task(self):
        grade = self.g.grade("Reasoning...\nANSWER: news articles", ctx("news articles", "free", judge_task="Q only"))
        assert grade.passed and grade.judge_model == "mock-judge"

    def test_free_answer_can_span_lines(self):
        assert final_answer("x\nANSWER: first\nsecond", to_end=True) == "first\nsecond"
        assert final_answer("x\nANSWER: first\nsecond") == "first"


class FakeTable(list):
    @property
    def column_names(self):
        return list(self[0]) if self else []

    def select_columns(self, cols):
        return FakeTable({c: row[c] for c in cols} for row in self)


class FakeDatasets:
    def __init__(self, rows):
        self.rows = FakeTable(rows)

    def load_dataset(self, *args, **kwargs):
        return self.rows


def with_rows(monkeypatch, rows):
    monkeypatch.setattr(benchmarks, "_datasets", lambda: FakeDatasets(rows))


def test_arc_maps_numbered_options_to_letters(monkeypatch):
    with_rows(monkeypatch, [{"id": "q1", "question": "Pick", "answerKey": "3",
                             "choices": {"label": ["1", "2", "3", "4"], "text": ["a", "b", "c", "d"]}}])
    (case,) = benchmarks.arc_challenge(1, random.Random(0))
    assert case["reference"] == "C" and "(C) c" in case["input"]["question"]


def test_mmlu_pro_strips_padding_and_stratifies(monkeypatch):
    rows = [{"question_id": i, "question": "Q", "options": ["x", "y", "N/A", "N/A"], "answer": "B",
             "category": cat} for i, cat in enumerate(["math", "physics", "law", "math", "physics"])]
    with_rows(monkeypatch, rows)
    cases = benchmarks.mmlu_pro_stem(2, random.Random(0))
    assert {c["meta"]["subject"] for c in cases} == {"math", "physics"}  # law is excluded
    assert "N/A" not in cases[0]["input"]["question"]


def gpqa_row(i):
    return {"Question": f"Q{i}", "Correct Answer": "right", "Incorrect Answer 1": "w1",
            "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3", "Record ID": f"rec{i}",
            "High-level domain": "Physics"}


def test_gpqa_shuffles_options_deterministically(monkeypatch):
    with_rows(monkeypatch, [gpqa_row(i) for i in range(40)])
    cases = benchmarks.gpqa_diamond(40, random.Random(0))
    for c in cases:
        letter = c["reference"]
        assert f"({letter}) right" in c["input"]["question"]
    assert len({c["reference"] for c in cases}) > 1  # the answer isn't always (A)
    again = benchmarks.gpqa_diamond(40, random.Random(0))
    assert [c["reference"] for c in cases] == [c["reference"] for c in again]


def test_hle_keeps_text_only_and_routes_answer_types(monkeypatch):
    rows = [
        {"id": "1", "question": "MC?", "image": "", "answer": "B", "answer_type": "multipleChoice", "category": "Math"},
        {"id": "2", "question": "Short?", "image": "", "answer": "42 J", "answer_type": "exactMatch", "category": "Physics"},
        {"id": "3", "question": "Pic?", "image": "data:image/png;base64,xx", "answer": "A",
         "answer_type": "multipleChoice", "category": "Math"},
    ]
    with_rows(monkeypatch, rows)
    cases = {c["id"]: c for c in benchmarks.hle_text(2, random.Random(0))}
    assert set(cases) == {"hle-1", "hle-2"}
    assert cases["hle-1"]["meta"]["answer_type"] == "choice"
    assert cases["hle-2"]["meta"]["answer_type"] == "free"


def qasper_paper(answers_per_question):
    qs = list(answers_per_question)
    return {
        "title": "T", "abstract": "A",
        "full_text": {"section_name": ["Intro"], "paragraphs": [["Body text."]]},
        "qas": {
            "question": qs,
            "question_id": [f"{q}-id-0000000000" for q in qs],
            "answers": [{"answer": [dict(unanswerable=a == "U", yes_no=None, extractive_spans=[] if a == "U" else [a],
                                         free_form_answer="") for a in answers_per_question[q]]} for q in qs],
        },
    }


def test_qasper_skips_disagreements_and_keeps_unanswerable_share(monkeypatch):
    paper = qasper_paper({
        "agreed-unanswerable": ["U", "U"],
        "disputed": ["U", "span"],
        "answerable-1": ["span one"],
        "answerable-2": ["span two", "other span"],
        "answerable-3": ["span three"],
        "answerable-4": ["span four"],
    })
    with_rows(monkeypatch, [paper])
    cases = benchmarks.qasper(5, random.Random(0))
    ids = {c["id"].split("-id")[0] for c in cases}
    assert "qasper-disputed" not in ids
    assert sum(c["reference"] == "Unanswerable" for c in cases) == 1
    multi = next(c for c in cases if "span two" in c["reference"])
    assert multi["reference"].startswith("Accepted answers (any one):")


def test_offered_options_are_not_flagged_as_leaks():
    cases = [
        Case("a", {"q": "Claim x. Answer SUPPORT or CONTRADICT."}, "SUPPORT"),
        Case("b", {"q": "Claim y. Answer SUPPORT or CONTRADICT."}, "CONTRADICT"),
        Case("c", {"q": "The capital of France is Paris. What is it?"}, "Paris"),
    ]
    assert health_report(cases, "{q}")["reference_in_prompt"] == ["c"]


def test_build_writes_cases_and_lock(tmp_path, monkeypatch):
    (tmp_path / "mix.toml").write_text(
        'seed = 1\n[[sources]]\nadapter = "gpqa_diamond"\nn = 3\ntier = "3-hard"\n'
        '[[sources]]\nadapter = "hle_text"\nn = 2\ntier = "4-frontier"\n', encoding="utf-8")
    with_rows(monkeypatch, [gpqa_row(i) for i in range(5)])
    monkeypatch.setattr(benchmarks, "_revision", lambda repo: "abc123")
    monkeypatch.setitem(benchmarks.ADAPTERS, "hle_text", benchmarks.Source(
        "hle_text", "hle", "cais/hle", "MIT", True, lambda n, rng: (_ for _ in ()).throw(RuntimeError("gated"))))

    with pytest.raises(benchmarks.FetchError, match="gated"):
        benchmarks.build(tmp_path, log=lambda _: None)

    benchmarks.build(tmp_path, skip_unavailable=True, log=lambda _: None)
    rows = [json.loads(line) for line in (tmp_path / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3 and rows[0]["tags"] == ["3-hard", "gpqa-diamond"]
    lock = json.loads((tmp_path / "sources.lock.json").read_text(encoding="utf-8"))
    assert lock["sources"][0]["revision"] == "abc123" and lock["sources"][1]["skipped"]
