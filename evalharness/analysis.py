"""Turn raw result rows into a summary, and compare two runs. Everything is recomputed
from results.jsonl / errors.jsonl; no aggregate is trusted from elsewhere."""

from __future__ import annotations

import json
import math
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .runner import load_run_meta, read_jsonl
from .stats import bootstrap_ci, paired_diff, percentile


SPLITS = ("train", "test")


def in_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Rows whose tags (after the tier and source) mark them as part of `split`."""
    return [r for r in rows if split in r["tags"][2:]]


def _primary_name(meta: dict[str, Any]) -> str:
    graders = meta["suite"]["graders"]
    g = next((g for g in graders if g.get("primary")), graders[0])
    return g.get("name", g["type"])


def per_case_scores(rows: list[dict[str, Any]], grader: str, *, strict: bool = False) -> dict[str, float]:
    """Mean score per case over its reps.

    Default: only status == "ok" rows count (truncated/refused are reported separately).
    strict=True: every recorded row counts, and non-ok rows score 0.
    """
    by_case: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            by_case[r["case_id"]].append(r["grades"][grader]["score"])
        elif strict:
            by_case[r["case_id"]].append(0.0)
    return {cid: mean(v) for cid, v in by_case.items()}


def _metric(rows: list[dict[str, Any]], grader: str, seed: int) -> dict[str, Any]:
    ok = [r for r in rows if r["status"] == "ok"]
    scores = per_case_scores(rows, grader)
    passes: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        passes[r["case_id"]].append(float(r["grades"][grader]["passed"]))
    pass_case = [mean(v) for v in passes.values()]
    # Sort by case id: rows arrive in thread-completion order, and the seeded bootstrap
    # must see the same sequence every time to give the same interval.
    lo, hi = bootstrap_ci([scores[k] for k in sorted(scores)], seed=seed)
    strict = per_case_scores(rows, grader, strict=True)
    graded = [r for r in rows if grader in r.get("grades", {})]
    return {
        "mean": mean(scores.values()) if scores else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "pass_rate": mean(pass_case) if pass_case else float("nan"),
        "strict_mean": mean(strict.values()) if strict else float("nan"),
        "answered_rate": mean(float(r["grades"][grader]["answered"]) for r in graded) if graded else float("nan"),
        "n_cases": len(scores),
    }


def _breakdown(rows: list[dict[str, Any]], grader: str, position: int) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["tags"][position] if len(r["tags"]) > position else "untagged"].append(r)
    out = {}
    for tag, tag_rows in sorted(groups.items()):
        s = per_case_scores(tag_rows, grader)
        out[tag] = {"mean": mean(s.values()) if s else float("nan"), "n_cases": len({r["case_id"] for r in tag_rows})}
    return out


def load_known_issues(suite_dir: Path) -> dict[str, dict[str, Any]]:
    """Every entry in a suite's known_issues.toml, keyed by case id. An entry with
    `exclude = false` records a case that was reviewed and kept; any other entry marks a
    problem in the case itself (a wrong or ambiguous answer key) and is left out of scores."""
    path = suite_dir / "known_issues.toml"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return {i["case_id"]: i for i in tomllib.load(f).get("issues", [])}


def known_issues(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The excluded cases for a run. Read at analysis time, so a problem found after a run
    applies to that run's report too."""
    source = meta["suite"].get("source")
    if not source:
        return {}
    return {cid: i for cid, i in load_known_issues(Path(source).parent).items() if i.get("exclude", True)}


def scored_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(rows that count toward scores, rows excluded as known issues)."""
    issues = known_issues(load_run_meta(run_dir))
    rows = read_jsonl(run_dir / "results.jsonl")
    return [r for r in rows if r["case_id"] not in issues], [r for r in rows if r["case_id"] in issues]


def summarize(run_dir: Path) -> dict[str, Any]:
    meta = load_run_meta(run_dir)
    suite = meta["suite"]
    issues = known_issues(meta)
    rows, flagged = scored_rows(run_dir)
    errors = [e for e in read_jsonl(run_dir / "errors.jsonl") if e["case_id"] not in issues]
    done = {(r["case_id"], r["rep"]) for r in rows}
    unresolved = [e for e in errors if (e["case_id"], e["rep"]) not in done]
    seed = suite["run"]["seed"]
    primary = _primary_name(meta)

    grader_names = [g.get("name", g["type"]) for g in suite["graders"]]
    metrics = {name: _metric(rows, name, seed) for name in grader_names}
    head = metrics[primary]

    tags = _breakdown(rows, primary, position=0)
    splits = {s: _metric(split_rows, primary, seed) for s in SPLITS if (split_rows := in_split(rows, s))}
    # Second tag: the benchmark source in stem-research, a category in triage.
    subtags = _breakdown([r for r in rows if len(r["tags"]) > 1], primary, position=1)

    ok = [r for r in rows if r["status"] == "ok"]
    latencies = [r["latency_s"] for r in ok]
    # Spend counts every answer actually paid for, including excluded questions.
    spent = rows + flagged
    model_costs = [r["cost_usd"] for r in spent]
    tokens = Counter()
    for r in spent:
        tokens.update(r["usage"])

    cases: dict[str, dict[str, Any]] = {}
    for r in sorted(rows, key=lambda r: (r["case_id"], r["rep"])):
        c = cases.setdefault(r["case_id"], {"case_id": r["case_id"], "tags": r["tags"], "reps": []})
        g = r["grades"].get(primary)
        c["reps"].append({
            "rep": r["rep"],
            "status": r["status"],
            "score": g["score"] if g else None,
            "passed": g["passed"] if g else None,
            "attempts": r["attempts"],
            "latency_s": r["latency_s"],
            "output": r["output"],
            "grades": r["grades"],
        })
    case_means = per_case_scores(rows, primary)
    for cid, c in cases.items():
        c["mean"] = case_means.get(cid)
        c["prompt"] = next(r["prompt"] for r in rows if r["case_id"] == cid)

    expected = None
    if rows or errors:
        n_cases = len({r["case_id"] for r in rows} | {e["case_id"] for e in errors})
        expected = n_cases * suite["run"]["reps"]

    return {
        "run_dir": str(run_dir),
        "suite": suite["name"],
        "description": suite["description"],
        "provider": suite["model"]["provider"],
        "model": suite["model"]["model"],
        "effort": suite["model"]["effort"],
        "reps": suite["run"]["reps"],
        "created_at": meta["created_at"],
        "dataset_sha256": meta["dataset_sha256"],
        "primary_grader": primary,
        "headline": head,
        # Half-width of the 95% CI: differences smaller than this are not distinguishable.
        "noise_floor": (head["ci_high"] - head["ci_low"]) / 2 if rows else float("nan"),
        "metrics": metrics,
        "by_tag": tags,
        "by_subtag": subtags,
        "by_split": splits,
        "system_prompt_file": suite.get("extra", {}).get("system_prompt_file"),
        "counts": {
            "rows": len(rows),
            "expected_rows": expected,
            "status": dict(Counter(r["status"] for r in rows)),
            "unresolved_errors": dict(Counter(e["error_class"] for e in unresolved)),
            "retried_rows": sum(1 for r in rows if r["attempts"] > 1),
            "total_attempts": sum(r["attempts"] for r in rows),
            "refusal_rate": sum(r["status"] == "refused" for r in rows) / len(rows) if rows else 0.0,
            "truncation_rate": sum(r["status"] == "truncated" for r in rows) / len(rows) if rows else 0.0,
        },
        "cost": {
            "model_usd": None if any(c is None for c in model_costs) else sum(model_costs),
            "judge_usd": sum(r.get("judge_cost_usd") or 0.0 for r in spent),
            "tokens": dict(tokens),
        },
        "latency_s": {
            "p50": percentile(latencies, 50) if latencies else float("nan"),
            "p95": percentile(latencies, 95) if latencies else float("nan"),
        },
        "cases": list(cases.values()),
        # Left out of every number above; shown with this run's score so nothing is hidden.
        "excluded": [
            {
                "case_id": cid,
                "kind": issue.get("kind", ""),
                "reason": issue.get("reason", ""),
                "score": per_case_scores([r for r in flagged if r["case_id"] == cid], primary).get(cid),
            }
            for cid, issue in sorted(issues.items())
        ],
    }


def compare(run_a: Path, run_b: Path) -> dict[str, Any]:
    """Paired comparison of B against A on the cases both runs scored."""
    sa, sb = summarize(run_a), summarize(run_b)
    if load_run_meta(run_a)["dataset_sha256"] != load_run_meta(run_b)["dataset_sha256"]:
        raise ValueError("runs used different datasets; a paired comparison would be meaningless")
    primary = sa["primary_grader"]
    rows_a, _ = scored_rows(run_a)
    rows_b, _ = scored_rows(run_b)
    a, b = per_case_scores(rows_a, primary), per_case_scores(rows_b, primary)
    diff = paired_diff(a, b, seed=0)
    shared = sorted(set(a) & set(b))
    flips = {
        "improved": [k for k in shared if b[k] - a[k] >= 0.5],
        "regressed": [k for k in shared if a[k] - b[k] >= 0.5],
    }
    significant = _significant(diff)
    by_split = {}
    for split in SPLITS:
        ra, rb = in_split(rows_a, split), in_split(rows_b, split)
        if ra and rb:
            d = paired_diff(per_case_scores(ra, primary), per_case_scores(rb, primary), seed=0)
            by_split[split] = {**d, "significant": _significant(d)}
    return {"a": sa, "b": sb, "primary_grader": primary, "diff": diff, "significant": significant,
            "flips": flips, "by_split": by_split}


def _significant(d: dict[str, float]) -> bool:
    return not math.isnan(d["ci_low"]) and (d["ci_low"] > 0 or d["ci_high"] < 0)


def write_summary(run_dir: Path) -> dict[str, Any]:
    summary = summarize(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
