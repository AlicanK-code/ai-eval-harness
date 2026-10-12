# Results

Scores after the round 4 grading fix: the judge bug fixed, and 11 faulty questions excluded (see `rounds/r4-grading-fix.md`). The numbers before the fix are kept in each round's file.

| Round | Change | Test (90 questions) | Train (48) | Cost | Outcome |
|---|---|---|---|---|---|
| v0 | baseline prompt, medium effort | **74.8%** [65.9, 83.3] | 77.1% | $0.80 | **recommended setup** |
| v1 | resolve red flags instead of caveating them | 74.7% (−0.4 pts [−2.6, +1.9]) | 75.0% | $0.83 | reverted, no effect |
| r2 | failure analysis, no run | | | $0 | 1 of 20 failures prompt-fixable |
| r3 | v0 at high effort (not a prompt change) | 76.6% (+1.5 pts [−1.1, +4.4]) | 77.1% | $1.15 | not significant, 71% more cost |
| r4 | grading fix and re-score (no model calls) | v0: 68.0% → 74.8% | 70.6% → 77.1% | $1.01 | **biggest change of all** |

The noise floor for a paired comparison is about ±3 points on test. Total spend on this experiment: about $3.80.

## Findings

1. **A better system prompt didn't help Haiku on hard questions.** Only 1 of the 20 failing train questions showed a habit a prompt could fix, and the rule targeting it changed neither the habit nor the score.
2. **Thinking longer helped only slightly.** High effort was +1.5 points on test, not significant, for 71% more cost and a slower tail of answers.
3. **The test itself was the largest source of error.** Fixing one judge bug and excluding 11 questions with wrong keys or missing information raised Haiku's measured test score by 6.8 points, more than every tuning change combined.
4. **What's left is mostly capability.** After the fix, HLE is still about 31%. The remaining failures are mostly specialised knowledge or derivations Haiku can't do, which no instruction supplies.
