# Prompt tuning: Claude Haiku 5.5 on `stem-hard`

Can a better system prompt alone make Claude Haiku 5.5 measurably better at hard science questions?

## Setup

| Will change | Held fixed |
|---|---|
| The system prompt, one idea per round (`prompts/vN.txt`) | Model `claude-haiku-5-5`, effort `medium`, `max_tokens` 64000 |
| | No tools: no code execution, no web search |
| | Questions, answer keys and graders (`suites/stem-hard`) |
| | Judge for free-text answers: `claude-sonnet-5-5` |

**Questions.** `suites/stem-hard`: 150 questions from GPQA Diamond (70), Humanity's Last Exam text-only (40), MATH-500 levels 4–5 (25) and QASPER (15), built with `evalh fetch suites/stem-hard`. They're downloaded on demand and never committed, at the datasets' request.

**Train / test split.** Fixed once, at random within each source: about a third train (51 questions), two thirds test (99). Only train answers are ever read when proposing a change. The test score is the headline, because the tuning process never saw those questions.

## Each round

1. A fresh analyzer reads Haiku's answers to the **train** questions only, names the most common mistake, and proposes one change to the prompt. It describes the behaviour to fix and never copies question content into the prompt.
2. The change is reviewed and approved before it runs.
3. Haiku answers all 150 questions, 3 tries each, with the new prompt.
4. The test score is compared with the previous best using a paired bootstrap over questions. A change is kept only if it helps beyond noise.

Run a round with:

```bash
evalh run suites/stem-hard --system-prompt experiments/prompt-tuning/prompts/vN.txt --out runs/tune-haiku/vN
evalh compare runs/tune-haiku/v0 runs/tune-haiku/vN
```

## Files

- `prompts/vN.txt`: the exact prompt used in round N (`v0` is the baseline)
- `rounds/vN.md`: what changed in round N and why, with its result
- `results.md`: the running results table and summary
