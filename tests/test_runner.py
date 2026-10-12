import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evalharness import runner
from evalharness.analysis import compare, summarize
from evalharness.cli import main
from evalharness.config import load_suite
from evalharness.providers import Completion, ProviderError, Usage
from evalharness.providers.anthropic_provider import AnthropicProvider
from evalharness.runner import execute, prepare_run, read_jsonl

SUITES = Path(__file__).parent.parent / "suites"


def mock_run(tmp_path, model, suite="arithmetic", reps=2):
    cfg = load_suite(SUITES / suite)
    cfg.model.provider, cfg.model.model, cfg.run.reps = "mock", model, reps
    run_dir = tmp_path / model
    prepare_run(cfg, run_dir)
    execute(run_dir, log=lambda _: None)
    return run_dir


def test_bundled_suites_validate():
    # Oracle answers pass every grader, empty answers fail every grader, no dataset issues.
    assert main(["validate", *map(str, sorted(SUITES.iterdir()))]) == 0


def test_oracle_and_null_baselines(tmp_path):
    assert summarize(mock_run(tmp_path, "mock-oracle"))["headline"]["mean"] == 1.0
    null = summarize(mock_run(tmp_path, "mock-null"))
    assert null["headline"]["mean"] == 0.0
    assert null["headline"]["answered_rate"] == 0.0  # "no answer", not "wrong answer"


def test_faults_are_separated_from_scores(tmp_path):
    run_dir = mock_run(tmp_path, "mock-strong", reps=6)
    rows = read_jsonl(run_dir / "results.jsonl")
    s = summarize(run_dir)
    # Transient errors were retried and the attempt count recorded.
    assert any(r["attempts"] > 1 for r in rows)
    # Truncated/refused rows exist but don't enter the headline mean; strict mode counts them as 0.
    assert {"truncated", "refused"} & set(s["counts"]["status"])
    assert s["headline"]["strict_mean"] < s["headline"]["mean"]
    assert all(r["grades"] == {} for r in rows if r["status"] == "refused")
    # Headline recomputes from raw rows.
    assert s["counts"]["rows"] == len(rows) == 16 * 6


def test_mock_runs_are_deterministic(tmp_path):
    a = summarize(mock_run(tmp_path / "1", "mock-weak"))
    b = summarize(mock_run(tmp_path / "2", "mock-weak"))
    assert a["headline"] == b["headline"]


def test_compare_detects_the_better_model(tmp_path):
    result = compare(mock_run(tmp_path, "mock-weak", "triage", 3), mock_run(tmp_path, "mock-strong", "triage", 3))
    assert result["significant"] and result["diff"]["mean_diff"] > 0


class FlakyProvider:
    """Fails case ar-001 with a non-retryable error until `healed` is set."""

    name, model, retry_base_delay = "flaky", "mock-flaky", 0.0
    healed = False

    def complete(self, request):
        if request.trace["case_id"] == "ar-001" and not FlakyProvider.healed:
            raise ProviderError("http_500", "boom", retryable=False)
        return Completion("ANSWER: 48", "end_turn", self.model, Usage(10, 5), 0.1)


def test_errors_go_to_sidecar_and_resume_retries_them(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "make_provider", lambda *a, **k: FlakyProvider())
    cfg = load_suite(SUITES / "arithmetic")
    cfg.run.reps = 1
    run_dir = tmp_path / "run"
    prepare_run(cfg, run_dir)

    counts = execute(run_dir, log=lambda _: None)
    assert counts == {"results": 15, "errors": 1}
    assert "ar-001" not in {r["case_id"] for r in read_jsonl(run_dir / "results.jsonl")}
    assert summarize(run_dir)["counts"]["unresolved_errors"] == {"http_500": 1}

    FlakyProvider.healed = True
    counts = execute(run_dir, log=lambda _: None)
    assert counts == {"results": 1, "errors": 0}  # only the failed attempt re-ran
    assert summarize(run_dir)["counts"]["unresolved_errors"] == {}


def test_resume_refuses_changed_dataset(tmp_path):
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    for name in ("suite.toml", "cases.jsonl"):
        (suite_dir / name).write_text((SUITES / "triage" / name).read_text(encoding="utf-8"), encoding="utf-8")
    cfg = load_suite(suite_dir)
    cfg.model.provider, cfg.model.model = "mock", "mock-strong"
    prepare_run(cfg, tmp_path / "run")
    with (suite_dir / "cases.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "new", "input": {"ticket": "x"}, "reference": "routine"}) + "\n")
    with pytest.raises(RuntimeError, match="changed"):
        execute(tmp_path / "run", log=lambda _: None)


def fake_message(model, stop_reason="end_turn", text="hi"):
    return SimpleNamespace(
        model=model,
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=3, cache_read_input_tokens=None,
                              cache_creation_input_tokens=None),
    )


def anthropic_with(message, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    provider = AnthropicProvider("claude-opus-5-5")
    calls = []

    class FakeStream:
        def __enter__(self):
            return SimpleNamespace(get_final_message=lambda: message)

        def __exit__(self, *exc):
            return False

    def stream(**kwargs):
        calls.append(kwargs)
        return FakeStream()

    client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    provider.client = SimpleNamespace(with_options=lambda **_: client)
    return provider, calls


def test_anthropic_provider_request_and_parsing(monkeypatch):
    from evalharness.providers import CompletionRequest

    provider, calls = anthropic_with(fake_message("claude-opus-5-5"), monkeypatch)
    out = provider.complete(CompletionRequest(prompt="p", system="s", effort="low", json_schema={"type": "object"}))
    assert out.text == "hi" and out.usage.cache_read_input_tokens == 0
    assert calls[0]["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert "thinking" not in calls[0] and "temperature" not in calls[0]


def test_anthropic_provider_rejects_wrong_served_model(monkeypatch):
    from evalharness.providers import CompletionRequest

    provider, _ = anthropic_with(fake_message("claude-opus-5"), monkeypatch)
    with pytest.raises(ProviderError) as e:
        provider.complete(CompletionRequest(prompt="p"))
    assert e.value.kind == "model_mismatch" and not e.value.retryable


def test_anthropic_provider_reports_refusal_category(monkeypatch):
    from evalharness.providers import CompletionRequest

    provider, _ = anthropic_with(fake_message("claude-opus-5-5", "refusal", ""), monkeypatch)
    out = provider.complete(CompletionRequest(prompt="p"))
    assert out.stop_reason == "refusal" and out.refusal_category == "cyber"


def test_select_cases_per_group():
    from evalharness.dataset import Case
    from evalharness.runner import select_cases

    cases = [Case(f"{src}{i}", {}, "x", tags=("tier", src)) for src in "abc" for i in range(3)]
    assert [c.id for c in select_cases(cases, per_group=1)] == ["a0", "b0", "c0"]
    assert [c.id for c in select_cases(cases, limit=2, per_group=2)] == ["a0", "a1"]


class BrokeProvider:
    name, model, retry_base_delay = "broke", "mock-broke", 0.0

    def complete(self, request):
        raise ProviderError("billing", "Your credit balance is too low", retryable=False)


def test_account_errors_stop_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "make_provider", lambda *a, **k: BrokeProvider())
    cfg = load_suite(SUITES / "arithmetic")
    cfg.run.concurrency = 1
    prepare_run(cfg, tmp_path / "run")
    counts = execute(tmp_path / "run", log=lambda _: None)
    assert counts["results"] == 0
    assert counts["errors"] < 16 * cfg.run.reps  # queued requests were never sent


def test_known_issues_are_excluded_but_reported(tmp_path):
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    for name in ("suite.toml", "cases.jsonl"):
        (suite_dir / name).write_text((SUITES / "triage" / name).read_text(encoding="utf-8"), encoding="utf-8")
    cfg = load_suite(suite_dir)
    cfg.model.provider, cfg.model.model, cfg.run.reps = "mock", "mock-null", 1
    prepare_run(cfg, tmp_path / "run")
    execute(tmp_path / "run", log=lambda _: None)
    assert summarize(tmp_path / "run")["counts"]["rows"] == 14

    (suite_dir / "known_issues.toml").write_text(
        '[[issues]]\ncase_id = "tr-001"\nkind = "wrong key"\nreason = "test"\n', encoding="utf-8")
    s = summarize(tmp_path / "run")
    assert s["counts"]["rows"] == 13
    assert "tr-001" not in {c["case_id"] for c in s["cases"]}
    assert s["excluded"] == [{"case_id": "tr-001", "kind": "wrong key", "reason": "test", "score": 0.0}]


def test_system_prompt_override_and_split_scores(tmp_path, monkeypatch):
    seen = []

    class Recorder:
        name, model, retry_base_delay = "rec", "mock-rec", 0.0

        def complete(self, request):
            seen.append(request.system)
            ok = request.trace["case_id"] in {"a", "b"}
            return Completion("ANSWER: 1" if ok else "ANSWER: 2", "end_turn", self.model, Usage(5, 5), 0.1)

    monkeypatch.setattr(runner, "make_provider", lambda *a, **k: Recorder())
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    (suite_dir / "suite.toml").write_text(
        '[suite]\nname = "s"\ndataset = "cases.jsonl"\nprompt_template = "{q}"\nsystem_prompt = "old"\n'
        '[run]\nreps = 1\n[[graders]]\ntype = "numeric"\nname = "n"\nextract = \'ANSWER:\\s*(.+)\'\n',
        encoding="utf-8")
    cases = [("a", "train"), ("b", "test"), ("c", "test")]
    (suite_dir / "cases.jsonl").write_text("\n".join(
        json.dumps({"id": cid, "input": {"q": "?"}, "reference": 1, "tags": ["t", "src", split]})
        for cid, split in cases), encoding="utf-8")
    prompt_file = tmp_path / "v1.txt"
    prompt_file.write_text("new prompt\n", encoding="utf-8")

    assert main(["run", str(suite_dir), "--system-prompt", str(prompt_file), "--out", str(tmp_path / "run")]) == 0
    assert set(seen) == {"new prompt"}
    s = summarize(tmp_path / "run")
    assert s["by_split"]["train"]["mean"] == 1.0 and s["by_split"]["test"]["mean"] == 0.5
    assert s["system_prompt_file"] == str(prompt_file)


def test_reviewed_cases_stay_in_scores(tmp_path):
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    for name in ("suite.toml", "cases.jsonl"):
        (suite_dir / name).write_text((SUITES / "triage" / name).read_text(encoding="utf-8"), encoding="utf-8")
    (suite_dir / "known_issues.toml").write_text(
        '[[issues]]\ncase_id = "tr-001"\nkind = "reviewed"\nexclude = false\nreason = "fine"\n', encoding="utf-8")
    cfg = load_suite(suite_dir)
    cfg.model.provider, cfg.model.model, cfg.run.reps = "mock", "mock-oracle", 1
    prepare_run(cfg, tmp_path / "run")
    execute(tmp_path / "run", log=lambda _: None)
    s = summarize(tmp_path / "run")
    assert s["counts"]["rows"] == 14 and s["excluded"] == []


def test_regrade_remarks_saved_answers_without_model_calls(tmp_path, monkeypatch):
    run_dir = mock_run(tmp_path, "mock-strong", reps=1)
    before = read_jsonl(run_dir / "results.jsonl")

    def no_model(*a, **k):
        raise AssertionError("regrade must not call the model")

    monkeypatch.setattr(runner, "make_provider", no_model)
    # Make the grader stricter: only exact "ANSWER: <ref>" lines now pass.
    meta = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    meta["suite"]["graders"][0]["abs_tol"] = 0.0
    meta["suite"]["graders"][0]["extract"] = r"ANSWER:\s*(\d+)\s*$"
    (run_dir / "run_config.json").write_text(json.dumps(meta), encoding="utf-8")

    record = runner.regrade(run_dir, log=lambda _: None)
    after = read_jsonl(run_dir / "results.jsonl")
    assert [r["output"] for r in after] == [r["output"] for r in before]
    assert record["changed_verdicts"] > 0
    assert (run_dir / record["previous_results"]).exists()
    assert json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["regrades"][0] == record


def test_excluded_questions_still_count_toward_cost(tmp_path):
    run_dir = mock_run(tmp_path, "mock-oracle", reps=1)
    full = summarize(run_dir)["cost"]["tokens"]["input_tokens"]
    source = Path(json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["suite"]["source"])
    issues = source.parent / "known_issues.toml"
    assert not issues.exists()
    try:
        issues.write_text('[[issues]]\ncase_id = "ar-001"\nreason = "test"\n', encoding="utf-8")
        s = summarize(run_dir)
        assert s["counts"]["rows"] == 15 and s["cost"]["tokens"]["input_tokens"] == full
    finally:
        issues.unlink()
