"""Command line: validate, run, report, compare."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .analysis import compare, write_summary
from .config import SuiteConfig, load_suite
from .dataset import health_report, load_cases, render_prompt
from .graders import GradeContext, build_graders
from .report import console_compare, console_summary, render_compare, render_run
from .runner import default_run_dir, execute, prepare_run


def _use_mock_judges(suite: SuiteConfig) -> None:
    for spec in suite.graders:
        if spec.get("type") in ("llm_judge", "answer_match"):
            spec["judge_provider"] = "mock"
            spec["judge_model"] = "mock-judge"


def cmd_validate(args: argparse.Namespace) -> int:
    """Dataset health checks plus an oracle/null smoke test of every grader - no model calls."""
    failed = False
    for path in args.suites:
        suite = load_suite(Path(path))
        _use_mock_judges(suite)  # validation must be free and deterministic
        if not suite.dataset.exists() and (suite.dataset.parent / "mix.toml").exists():
            print(f"\n== {suite.name} ==\n  skipped: built from public benchmarks; run `evalh fetch {path}` first")
            continue
        cases = load_cases(suite.dataset)
        health = health_report(cases, suite.prompt_template)
        graders = build_graders(suite.graders)
        print(f"\n== {suite.name} ({len(cases)} cases) ==")
        print(f"  tags: {health['tags']}")
        print(f"  prompt chars: {health['prompt_chars']}")
        print(f"  majority-reference share: {health['majority_reference_share']:.0%}")
        problems = []
        if health["missing_template_fields"]:
            problems.append(f"cases missing template fields: {health['missing_template_fields']}")
        if health["near_duplicates"]:
            problems.append(f"near-duplicate prompts: {health['near_duplicates']}")
        if health["reference_in_prompt"]:
            problems.append(f"reference answer appears in prompt: {health['reference_in_prompt']}")

        # Oracle: the reference, in the format the prompt asks for, must pass every grader.
        # Null: an empty response must fail every grader. Either failing means a broken
        # case or grader, not a hard task.
        for case in cases:
            ctx = GradeContext(case=case, prompt=render_prompt(suite.prompt_template, case))
            ref = case.reference
            rendered = json.dumps(ref) if isinstance(ref, (dict, list)) else str(ref)
            oracle = suite.mock_answer_template.replace("{answer}", rendered)
            for g in graders:
                if not g.grade(oracle, ctx).passed:
                    problems.append(f"oracle answer fails grader {g.name!r} on case {case.id}")
                if g.grade("", ctx).passed:
                    problems.append(f"empty answer passes grader {g.name!r} on case {case.id}")
        for p in problems:
            print(f"  PROBLEM: {p}")
        print("  OK" if not problems else f"  {len(problems)} problem(s)")
        failed |= bool(problems)
    return 1 if failed else 0


def _finish(run_dir: Path) -> None:
    summary = write_summary(run_dir)
    (run_dir / "report.html").write_text(render_run(summary), encoding="utf-8")
    print()
    print(console_summary(summary))
    print(f"\n  report: {run_dir / 'report.html'}")


def cmd_run(args: argparse.Namespace) -> int:
    if args.resume:
        run_dir = Path(args.resume)
    else:
        if not args.suite:
            print("error: give a suite path, or --resume RUN_DIR", file=sys.stderr)
            return 2
        suite = load_suite(Path(args.suite))
        if args.provider:
            suite.model.provider = args.provider
        if args.model:
            suite.model.model = args.model
        elif suite.model.provider == "mock" and not suite.model.model.startswith("mock"):
            suite.model.model = "mock-strong"
        if args.effort:
            suite.model.effort = None if args.effort == "none" else args.effort
        if args.reps:
            suite.run.reps = args.reps
        if args.concurrency:
            suite.run.concurrency = args.concurrency
        if args.judge_provider == "mock" or (suite.model.provider == "mock" and args.judge_provider is None):
            _use_mock_judges(suite)
        run_dir = Path(args.out) if args.out else default_run_dir(suite, Path(args.runs_root))
        prepare_run(suite, run_dir, limit=args.limit, per_group=args.per_group)
    counts = execute(run_dir)
    _finish(run_dir)
    return 0 if counts["errors"] == 0 else 3


def cmd_report(args: argparse.Namespace) -> int:
    _finish(Path(args.run_dir))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from .benchmarks import FetchError, build

    try:
        build(Path(args.suite), skip_unavailable=args.skip_unavailable)
    except FetchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return cmd_validate(argparse.Namespace(suites=[args.suite]))


def cmd_compare(args: argparse.Namespace) -> int:
    result = compare(Path(args.run_a), Path(args.run_b))
    print(console_compare(result))
    if args.html:
        Path(args.html).write_text(render_compare(result), encoding="utf-8")
        print(f"\n  report: {args.html}")
    return 0


def load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=value lines from a git-ignored .env file, so secrets such as
    ANTHROPIC_API_KEY never need to be typed into a shell or committed. Variables
    already set in the environment win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        if value:
            os.environ.setdefault(key.strip(), value)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="evalh", description="Run, grade, and compare LLM evals.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check datasets and graders without calling a model")
    p.add_argument("suites", nargs="+", help="suite directories or suite.toml files")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("fetch", help="build a suite's cases from public benchmarks (see its mix.toml)")
    p.add_argument("suite", help="suite directory containing mix.toml")
    p.add_argument("--skip-unavailable", action="store_true",
                   help="leave out sources that can't be downloaded (e.g. gated datasets) instead of failing")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("run", help="run a suite (or resume a run)")
    p.add_argument("suite", nargs="?", help="suite directory or suite.toml")
    p.add_argument("--resume", metavar="RUN_DIR", help="continue an interrupted run; retries its errors")
    p.add_argument("--provider", choices=["anthropic", "mock"])
    p.add_argument("--model", help="e.g. claude-opus-5-5, or mock-strong / mock-weak / mock-oracle / mock-null")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max", "none"])
    p.add_argument("--reps", type=int)
    p.add_argument("--concurrency", type=int)
    p.add_argument("--limit", type=int, help="only the first N cases (for smoke tests)")
    p.add_argument("--per-group", type=int,
                   help="only the first N cases from each source/tag group (a smoke test that covers every group)")
    p.add_argument("--judge-provider", choices=["anthropic", "mock"],
                   help="override llm_judge graders (defaults to mock when --provider mock)")
    p.add_argument("--out", help="run directory (default: runs/<suite>__<model>__<timestamp>)")
    p.add_argument("--runs-root", default="runs")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="rebuild summary.json and report.html for a run")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("compare", help="paired comparison of two runs of the same suite")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--html", help="also write an HTML comparison report here")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
