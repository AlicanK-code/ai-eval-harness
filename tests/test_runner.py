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

    def create(**kwargs):
        calls.append(kwargs)
        return message

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
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
