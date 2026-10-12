"""Execute a suite: (case x rep) -> model call -> graders -> results.jsonl / errors.jsonl.

Design rules (each one prevents a way evals quietly lie):
  * Only attempts that produced a scorable output go to results.jsonl. Infra failures
    (rate limits after retries, timeouts, auth, wrong served model, grader crashes) go to
    errors.jsonl with a failure class and are retried on --resume.
  * Truncated and refused responses are recorded with their own status, never averaged
    in as "wrong".
  * Every row keeps the full prompt, output, grader details, usage, attempts and latency,
    so any surprising score can be traced without re-running.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .config import ModelConfig, RunConfig, SuiteConfig
from .dataset import Case, file_sha256, load_cases, render_prompt
from .graders import GradeContext, Grader, GraderError, build_graders
from .pricing import cost_usd
from .providers import CompletionRequest, Provider, ProviderError, complete_with_retries, make_provider

# Account-level failures: every other request in the run would fail identically.
FATAL_ERRORS = {"auth", "billing"}

STATUS_BY_STOP_REASON = {
    "end_turn": "ok",
    "stop_sequence": "ok",
    "max_tokens": "truncated",
    "refusal": "refused",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class RunContext:
    suite: SuiteConfig
    cases: list[Case]
    provider: Provider
    graders: list[Grader]


def run_one(ctx: RunContext, case: Case, rep: int) -> tuple[str, dict[str, Any]]:
    suite = ctx.suite
    prompt = render_prompt(suite.prompt_template, case)
    request = CompletionRequest(
        prompt=prompt,
        system=suite.system_prompt,
        max_tokens=suite.model.max_tokens,
        effort=suite.model.effort,
        timeout_s=suite.run.timeout_s,
        trace={"case_id": case.id, "rep": rep},
    )
    base = {"case_id": case.id, "rep": rep, "tags": list(case.tags), "timestamp": _now()}
    try:
        completion, attempts = complete_with_retries(ctx.provider, request, suite.run.max_attempts)
    except ProviderError as e:
        return "error", {
            **base,
            "stage": "model",
            "error_class": e.kind,
            "message": str(e)[:500],
            "attempts": getattr(e, "attempts", 1),
        }

    status = STATUS_BY_STOP_REASON.get(completion.stop_reason, "incomplete")
    grades: dict[str, Any] = {}
    judge_cost = 0.0
    if status != "refused":
        grade_ctx = GradeContext(case=case, prompt=prompt)
        for grader in ctx.graders:
            try:
                grade = grader.grade(completion.text, grade_ctx)
            except GraderError as e:
                return "error", {
                    **base,
                    "stage": "grader",
                    "grader": grader.name,
                    "error_class": "grader_error",
                    "message": str(e)[:500],
                    "attempts": attempts,
                }
            grades[grader.name] = grade.to_dict()
            judge_cost += cost_usd(grade.judge_model, grade.judge_usage) or 0.0

    usage = completion.usage.to_dict()
    return "result", {
        **base,
        "status": status,
        "stop_reason": completion.stop_reason,
        "refusal_category": completion.refusal_category,
        "attempts": attempts,
        "requested_model": ctx.provider.model,
        "served_model": completion.model,
        "prompt": prompt,
        "output": completion.text,
        "grades": grades,
        "usage": usage,
        "cost_usd": cost_usd(completion.model, usage),
        "judge_cost_usd": round(judge_cost, 6),
        "latency_s": round(completion.latency_s, 3),
    }


def suite_from_snapshot(snap: dict[str, Any]) -> SuiteConfig:
    return SuiteConfig(
        name=snap["name"],
        description=snap["description"],
        dataset=Path(snap["dataset"]),
        prompt_template=snap["prompt_template"],
        system_prompt=snap["system_prompt"],
        model=ModelConfig(**snap["model"]),
        run=RunConfig(**snap["run"]),
        graders=snap["graders"],
        mock_answer_template=snap["mock_answer_template"],
        source=Path(snap["source"]) if snap.get("source") else None,
    )


def default_run_dir(suite: SuiteConfig, root: Path = Path("runs")) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / f"{suite.name}__{suite.model.model}__{stamp}"


def select_cases(cases: list[Case], limit: int | None = None, per_group: int | None = None) -> list[Case]:
    """Optionally trim the case list for smoke tests: the first `limit` cases, and/or the
    first `per_group` cases of each group (second tag if present, else first tag)."""
    if per_group:
        counts: dict[str, int] = {}
        kept = []
        for c in cases:
            group = c.tags[1] if len(c.tags) > 1 else c.primary_tag
            if counts.get(group, 0) < per_group:
                counts[group] = counts.get(group, 0) + 1
                kept.append(c)
        cases = kept
    return cases[:limit] if limit else cases


def prepare_run(
    suite: SuiteConfig, run_dir: Path, limit: int | None = None, per_group: int | None = None
) -> None:
    """Create the run directory and freeze the exact config + dataset hash used."""
    run_dir.mkdir(parents=True, exist_ok=False)
    meta = {
        "harness_version": __version__,
        "created_at": _now(),
        "dataset_sha256": file_sha256(suite.dataset),
        "limit": limit,
        "per_group": per_group,
        "suite": suite.snapshot(),
    }
    (run_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_run_meta(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))


def execute(run_dir: Path, *, log: Callable[[str], None] = print) -> dict[str, int]:
    """Run (or resume) every pending (case, rep) in run_dir. Returns counts for this pass."""
    meta = load_run_meta(run_dir)
    suite = suite_from_snapshot(meta["suite"])
    if file_sha256(suite.dataset) != meta["dataset_sha256"]:
        raise RuntimeError(
            f"{suite.dataset} changed since this run started; scores would not be comparable. "
            "Start a new run instead of resuming."
        )
    cases = select_cases(load_cases(suite.dataset), meta.get("limit"), meta.get("per_group"))

    for spec in suite.graders:
        if spec.get("type") in ("llm_judge", "answer_match") and spec.get("judge_model") == suite.model.model:
            log(f"warning: grader {spec.get('name')!r} uses the model under test as its own judge "
                "(self-preference bias). Prefer a different judge model.")

    provider = make_provider(
        suite.model.provider,
        suite.model.model,
        references={c.id: c.reference for c in cases},
        answer_template=suite.mock_answer_template,
    )
    ctx = RunContext(suite=suite, cases=cases, provider=provider, graders=build_graders(suite.graders))

    results_path, errors_path = run_dir / "results.jsonl", run_dir / "errors.jsonl"
    done = {(r["case_id"], r["rep"]) for r in read_jsonl(results_path)}
    pending = [(c, rep) for c in cases for rep in range(suite.run.reps) if (c.id, rep) not in done]
    total = len(cases) * suite.run.reps
    log(f"{suite.name}: {len(cases)} cases x {suite.run.reps} reps on {suite.model.provider}:"
        f"{suite.model.model} - {len(pending)} pending, {len(done)} already done")

    counts = {"results": 0, "errors": 0}
    started = time.perf_counter()
    # Rows are written from this thread only, so the JSONL files never interleave.
    with (
        ThreadPoolExecutor(max_workers=suite.run.concurrency) as pool,
        results_path.open("a", encoding="utf-8") as results_f,
        errors_path.open("a", encoding="utf-8") as errors_f,
    ):
        futures = {pool.submit(run_one, ctx, case, rep): (case, rep) for case, rep in pending}
        aborted = None
        for i, future in enumerate(as_completed(futures), 1):
            if future.cancelled():
                continue
            try:
                kind, row = future.result()
            except Exception as e:  # a bug in the harness must not masquerade as a score
                case, rep = futures[future]
                kind, row = "error", {
                    "case_id": case.id, "rep": rep, "tags": list(case.tags), "timestamp": _now(),
                    "stage": "harness", "error_class": type(e).__name__, "message": str(e)[:500],
                }
            if kind == "error" and row.get("error_class") in FATAL_ERRORS and not aborted:
                # Every remaining request would fail the same way; don't send them.
                aborted = row["error_class"]
                cancelled = sum(f.cancel() for f in futures)
                log(f"  stopping: {row['error_class']} error ({row.get('message', '')[:160]}). "
                    f"{cancelled} queued requests not sent; fix it and use --resume.")
            target = results_f if kind == "result" else errors_f
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            target.flush()
            counts["results" if kind == "result" else "errors"] += 1
            if i % 10 == 0 or i == len(futures):
                log(f"  [{len(done) + i}/{total}] {counts['errors']} errors, "
                    f"{time.perf_counter() - started:.1f}s elapsed")
    return counts
