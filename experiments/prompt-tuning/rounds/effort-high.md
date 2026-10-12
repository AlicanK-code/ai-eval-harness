# Round 3: high effort (same v0 prompt)

**Status:** run 12 Oct 2026. Outside the original "system prompt only" scope, by choice after the round 2 analysis. Only the effort setting changed, from `medium` to `high`.

## Result

| | v0, medium effort | v0, high effort | Change (paired 95% CI) |
|---|---|---|---|
| **Test** (99 questions) | 68.0% | 70.4% | +2.0 pts [−0.3, +5.1], not significant |
| Train (51 questions) | 70.6% | 71.2% | +0.7 pts [−5.2, +6.5], not significant |
| Haiku cost | $0.48 | $0.83 | +71% |
| Mean output tokens | 1,941 | 3,479 | 1.8× |
| Slowest 5% of answers | 33 s | 52 s | |
| Truncated at the 64,000-token limit | 0 | 3 | |

Test results by source (paired):

| Source | Change |
|---|---|
| GPQA | +2.2 [−0.7, +6.7] |
| HLE | +1.3 [−5.1, +9.0] |
| MATH | +0.0 (already 100%) |
| QASPER | +6.7 [+0.0, +16.7] (10 questions) |

## Reading

- **The direction is positive but not established.** The test interval only just reaches below zero, so the gain may be real but small, about 2 points.
- **It isn't cheap.** That gain costs about 71% more per run and makes the slowest answers about half as long again.
- **Three answers ran out of room.** They hit the 64,000-token output limit before giving an answer: two tries at one HLE question and one at a GPQA question, all test cases. Those are recorded as truncated, not wrong, and left out of the headline mean.
- **The questions the round 2 analysis classed as knowledge limits mostly stayed wrong.** Thinking longer doesn't supply a missing fact.
