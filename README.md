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
| `stem-research` | 110 questions from 7 public benchmarks, from grade-school science to frontier research | `answer_match`: letter, label or number checked exactly; free-text answers by an LLM judge | Built on demand with `evalh fetch` (see below). Tagged by difficulty tier and source |

## Public benchmarks: the `stem-research` suite

This suite samples a fixed, seeded mix of questions from established benchmarks. Each question is tagged with a difficulty tier, so the report shows how a model's score falls as the questions get harder:

| Tier | Source | Questions | What it tests | Licence |
|---|---|---|---|---|
| 1-easy | [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) | 15 | Grade-school science | CC BY-SA 4.0 |
| 2-medium | [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) (6 STEM subjects) | 20 | University STEM, 10 options per question | MIT |
| 2-medium | [SciFact](https://huggingface.co/datasets/allenai/scifact) | 15 | Does an abstract support or contradict a claim? | CC BY-NC 2.0 |
| 3-hard | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), levels 4–5 | 15 | Competition maths with numeric answers | MIT (source MATH dataset) |
| 3-hard | [QASPER](https://huggingface.co/datasets/allenai/qasper) | 10 | Questions about full research papers, including ones the paper can't answer | CC BY 4.0 |
| 3-hard | [GPQA Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) | 25 | PhD-level biology, chemistry and physics | CC BY 4.0, gated |
| 4-frontier | [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle), text-only | 10 | Expert questions at the edge of current models | MIT, gated |

The questions are **not in this repository**. GPQA asks that its examples never be published online, and HLE asks that it not be redistributed, because both exist to stay out of AI training data. `evalh fetch` downloads them under each dataset's own terms and writes `cases.jsonl` and a `sources.lock.json` that records the exact dataset revisions. Both files are git-ignored.

```bash
pip install -e ".[benchmarks]"
hf auth login                                        # once; then accept the terms on the GPQA and HLE pages
evalh fetch suites/stem-research                     # all 110 questions
evalh fetch suites/stem-research --skip-unavailable  # or just the 75 open ones, without logging in
evalh run suites/stem-research --limit 3 --reps 1    # smoke test against Claude
```

A few choices worth knowing about:

- **Mixed answer formats, one score.** Each question records how its answer is checked: an option letter, a fixed label, a number, or free text. Only free text goes to the LLM judge, which sees the question and the final answer, not the model's reasoning.
- **Only the final `ANSWER:` line is graded,** so a letter or number in the working can't pass by accident. A response without one is recorded as unanswered.
- **Ambiguous questions are dropped at build time.** QASPER questions where the human annotators disagreed (one says the paper answers it, another says it doesn't) have no fair reference answer, so they're skipped.
- **Faulty answer keys are excluded, not hidden.** Cases listed in a suite's `known_issues.toml` (a wrong or ambiguous key) are left out of every score, but each report lists them with the reason and the run's answers. Two are listed for `stem-research`, both found because every model "failed" them on every try.
- **Public benchmarks may be in training data.** High scores on the easier tiers partly measure memory. GPQA and HLE were designed to resist this, and the handwritten suites in this repository can't have been seen at all.

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
type = "exact_match"            # exact_match | numeric | json_match | token_f1 | llm_judge | answer_match
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
  graders/        deterministic.py (exact, numeric, JSON, token F1), llm_judge.py, answer_match.py
  benchmarks.py   adapters that build a suite from public Hugging Face datasets
  dataset.py      loading, prompt rendering, tier-1 health checks
  runner.py       concurrency, retries, results/errors sidecars, resume, frozen config
  analysis.py     summaries and paired comparisons, recomputed from raw rows
  stats.py        bootstrap CIs (dependency-free)
  report.py       HTML + console reports
suites/           four handwritten suites and one built from public benchmarks
tests/            graders, stats, dataset checks, runner fault handling, provider wiring, benchmark adapters
```

## Things I learned building it

- **Most surprising eval results are eval bugs.** The oracle and null checks catch wiring mistakes in seconds. Writing the tests caught one in this harness: the bootstrap CI changed between identical runs, because rows arrive in thread-completion order.
- **Bootstrap over cases, not rows.** Three reps of one case are not three independent samples. Resampling rows gives a falsely narrow interval.
- **Pairing matters.** Two models share the same hard cases. A paired comparison cancels per-case difficulty and detects much smaller differences than comparing two separate intervals.
- **Keep the plumbing out of the score.** Retries, truncations, refusals and grader crashes each get their own column, so none of them can quietly move the headline number.

## License

MIT
