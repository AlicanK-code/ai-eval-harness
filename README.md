# evalharness

A small evaluation harness for LLM applications that is careful about what its numbers mean.

You write test cases and a grading rule. It runs a model on every case several times, then scores each output. It reports a mean with a confidence interval, plus cost and latency. It can also tell you whether a change between two runs is real or just noise.

```
$ evalh run suites/arithmetic --provider mock --model mock-strong

arithmetic  |  mock:mock-strong  |  3 reps
  correct            95.8%  (95% CI 89.6% to 100.0%, noise floor +/-5.2%)
  pass rate          95.8%   strict mean 93.8%
  status             ok=47, refused=1
  unresolved errors  0
  cost               model $0.0000, judge $0.0000
  latency            p50 1.87s, p95 2.35s

$ evalh compare runs/triage-mock-weak runs/triage-mock-strong

A mock:mock-weak            57.1%
B mock:mock-strong          92.9%
B - A = +35.7 pts  (paired 95% CI +21.4 to +50.0, n=14 cases) -> SIGNIFICANT
improved: tr-001, tr-004, tr-007, tr-011
```

Each run also writes a self-contained `report.html`. It has summary cards, a per-tag breakdown, and a filterable table of cases. You can expand any case to see the prompt, every rep's output, and the grader's reasoning.

## Why this exists

Writing an eval loop is easy. Getting numbers you can trust is harder, because evals tend to fail quietly:

| Failure mode | What this harness does about it |
|---|---|
| A rate-limit or timeout gets scored as a wrong answer | Attempts that produced no output go to `errors.jsonl` with a failure class. They never enter `results.jsonl`, and `--resume` retries them. |
| A response cut off at `max_tokens` counts as "wrong" | It is recorded as `status: truncated`. Refusals are recorded as `status: refused` with their category. Headline scores use `ok` rows only; a separate **strict** score counts every non-`ok` row as 0. |
| An empty response passes as the answer "none" | Every grade records `answered`. "No answer" and "wrong answer" are kept apart. |
| A difference of 3 points is reported as a win | Reps are averaged per case, then bootstrapped over cases (reps of the same case are correlated). Each run reports its **noise floor**, and `compare` uses a **paired** bootstrap. |
| The provider served a different model than requested | The served model is checked on every call; a mismatch fails the attempt. Server-side refusal fallbacks are deliberately left off, since they would swap the model under test. |
| The grader itself is broken | `evalh validate` sends each case's reference answer (the **oracle**) through every grader and it must pass. An empty response (the **null**) must fail. CI runs this. |
| The LLM judge is biased or easily manipulated | Atomic yes/no criteria with structured output. The candidate is passed as untrusted data and can't close its own tag. Empty answers fail without a judge call. You get a warning when the judge is the model under test. |
| "It scored 80% last week" and nobody knows on what | Each run directory freezes the config and a SHA-256 of the dataset, and keeps every prompt, output, grade, token count and attempt. Resuming after the dataset changes is refused. |
| Cost estimated from string length | Tokens come from the API's `usage` block and are priced per served model, including cache reads and writes. Judge cost is reported separately. |

## Quick start

No API key is needed. The mock provider is deterministic, and its "models" behave like real ones: they get some answers wrong, hit rate limits, truncate output, and refuse.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"

evalh validate suites/*                                         # dataset + grader health checks
evalh run suites/triage --provider mock --model mock-weak   --out runs/weak
evalh run suites/triage --provider mock --model mock-strong --out runs/strong
evalh compare runs/weak runs/strong --html runs/compare.html
pytest
```

Mock models: `mock-oracle` (always right), `mock-null` (always empty), `mock-strong`, `mock-weak`.

### Running against Claude

```bash
export ANTHROPIC_API_KEY=sk-ant-...
evalh run suites/arithmetic --limit 3 --reps 1      # smoke test first: 3 calls
evalh run suites/arithmetic                          # full run: 16 cases x 3 reps
evalh run suites/arithmetic --effort high            # same suite, different effort
evalh compare runs/<first> runs/<second>
```

The suites default to `claude-opus-5-5` at `medium` effort. The summarization judge uses `claude-sonnet-5-5`, a different model, to avoid self-preference bias. Override with `--model` and `--effort`. If a run is interrupted, or some attempts error out, `evalh run --resume runs/<dir>` picks up where it stopped.

## Bundled suites

| Suite | Tests | Grader | Notes |
|---|---|---|---|
| `arithmetic` | 16 multi-step word problems | `numeric`: only the `ANSWER:` line counts | Synthetic numbers, so answers can't come from memory |
| `extraction` | 10 messy support emails → JSON | `json_match`: partial credit per field | Negations, corrected order ids, European number formats, a message with no order |
| `triage` | 14 tickets → `escalate` / `routine` | `exact_match` with label extraction | Balanced 7/7, so "always escalate" scores 50%. The first tag is the label, so the per-tag view gives per-class recall |
| `summarization` | 6 synthetic news passages | `llm_judge` (3 atomic criteria) + `token_f1` | Shows how a judge and word overlap disagree on good paraphrases |

## Writing your own suite

A suite is a directory holding a `suite.toml` and a `cases.jsonl`:

```toml
[suite]
name = "my-suite"
dataset = "cases.jsonl"
system_prompt = "Answer with one word."
prompt_template = "Question: {question}"     # fields come from each case's "input"

[model]
provider = "anthropic"
model = "claude-opus-5-5"
effort = "medium"

[run]
reps = 3

[[graders]]
type = "exact_match"            # exact_match | numeric | json_match | token_f1 | llm_judge
primary = true
```

```json
{"id": "q-001", "input": {"question": "..."}, "reference": "...", "tags": ["category"]}
```

Run `evalh validate` before spending money. It flags duplicate prompts, missing template fields, reference answers leaking into the prompt, label imbalance, and any case where the reference answer itself fails the grader.

## Project layout

```
evalharness/
  providers/      anthropic_provider.py (Claude via the official SDK), mock.py, retry/backoff
  graders/        deterministic.py (exact, numeric, JSON, token F1), llm_judge.py
  dataset.py      loading, prompt rendering, tier-1 health checks
  runner.py       concurrency, retries, results/errors sidecars, resume, frozen config
  analysis.py     summaries and paired comparisons, recomputed from raw rows
  stats.py        bootstrap CIs (dependency-free)
  report.py       HTML + console reports
suites/           four example suites
tests/            graders, stats, dataset checks, runner fault handling, provider wiring
```

## Things I learned building it

- **Most surprising eval results are eval bugs.** The oracle and null checks catch wiring mistakes in seconds. Writing the tests caught one in this harness: the bootstrap CI changed between identical runs, because rows arrive in thread-completion order.
- **Bootstrap over cases, not rows.** Three reps of one case are not three independent samples. Resampling rows gives a falsely narrow interval.
- **Pairing matters.** Two models share the same hard cases. A paired comparison cancels per-case difficulty and detects much smaller differences than comparing two separate intervals.
- **Keep the plumbing out of the score.** Retries, truncations, refusals and grader crashes each get their own column, so none of them can quietly move the headline number.

## License

MIT
