# Round 4: fix the grading, then re-score every round

**Status:** done 12 Oct 2026. No model reruns; saved answers were re-marked with `evalh regrade`, for $1.01 of judge calls.

## What changed

1. **The judge's bug (a harness fix).** When a reference lists several accepted answers, the judge sometimes compared an answer against the most detailed one and failed it for missing details, or for adding correct extra ones. Its instruction now says to compare against each accepted answer separately, never to require details that appear only in a longer one, and to allow extra correct detail. This changed 6, 6 and 2 verdicts in the three runs, all on QASPER.
2. **Faulty questions excluded.** 11 are listed in `suites/stem-hard/known_issues.toml`:
   - 7 wrong answer keys: 3 train, 4 test;
   - 4 questions that can't be answered as written: all test.
   The rule, fixed before re-scoring: exclude only high-confidence findings with a concrete, checkable argument. Six medium- or low-confidence findings are recorded but stay in the score. Two keys were verified by hand: one by re-running the named algorithm, one by applying the standard notation to the given values.
3. **The test keys were audited only after tuning stopped,** by a checker that judged keys and never proposed prompt changes. Only questions Haiku failed at least once were audited.

## Corrected results

| Round | Test before | **Test after** | Train before | Train after |
|---|---|---|---|---|
| v0, baseline | 68.0% | **74.8%** [65.9, 83.3] | 70.6% | 77.1% |
| v1, red-flags rule | 68.0% | **74.7%** | 68.6% | 75.0% |
| v0, high effort | 70.4% | **76.6%** | 71.2% | 77.1% |

Corrected paired comparisons on test (90 questions):

| Comparison | Change | 95% CI | Significant? |
|---|---|---|---|
| v1 vs v0 | −0.4 pts | [−2.6, +1.9] | no |
| High effort vs v0 | +1.5 pts | [−1.1, +4.4] | no |

By source, for the v0 baseline after the fix:

| Source | Score |
|---|---|
| GPQA | 83.8% |
| HLE | 31.2% (was 21.8% on test) |
| MATH | 100% |
| QASPER | 93.3% |

## Reading

- **The grading fix moved Haiku's score by +6.8 points on test,** more than any prompt or effort change. About a fifth of the apparent failures were the test's fault, not Haiku's.
- **The conclusions about tuning don't change.** The red-flags prompt did nothing, and high effort is a small, non-significant gain (now +1.5, down from +2.0) at 71% more cost. The v0 prompt at medium effort stays the recommended setup.
