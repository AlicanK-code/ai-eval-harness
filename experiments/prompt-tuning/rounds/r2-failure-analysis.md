# Round 2: why does Haiku still fail? (train split only)

**Status:** analysis only; no model run. Question text from GPQA and HLE isn't reproduced here, at those datasets' request.

A fresh analyzer read every train question Haiku failed at least once, under both v0 and v1 (6 answers per question), and assigned each one root cause. 20 of the 51 train questions failed at least once.

| Root cause | Questions | Max train points recoverable (v0) |
|---|---|---|
| Prompt-fixable habit | 1 | 2.0 |
| Knowledge or capability limit | 6 | 11.8 |
| **Suspect answer key or grading** | **8** | **13.7** (7.8 at high confidence) |
| Run-to-run variance | 5 | 2.0 |
| Refusal or infrastructure | 0 | 0 |

## Findings

- **A system prompt has almost nothing left to fix.** Only one question shows a habit an instruction could plausibly change: dismissing a real error in an option as loose wording. That's worth at most 2 points, below the ±3.1-point noise floor, and v1 already targeted it.
- **The largest cause is the grading, not the model.**
  - **Answer keys:** four high-confidence problems, three HLE keys (`hle-671f0b0c…`, `hle-67325a61…`, `hle-671ee933…`) and one GPQA key (`gpqa-recf6ayQ…`, medium confidence). Two more HLE questions (`hle-6710b2cb…`, `hle-66fd89bd…`) are underspecified.
  - **Hand check:** the `hle-67325a61…` key was verified by hand. Its expected answer follows from a sign error in the normal force.
  - **Judge:** the free-text judge twice failed QASPER answers that matched an accepted answer, by treating "any one accepted answer" as "the most complete one" (`qasper-5206b6f4…`, and one rep of `qasper-cf0b7d8a…`).
- **Knowledge limits** are mostly recall of one specific paper, study or chart. No instruction can supply that. Retrieval could.
- **v0 vs v1 is noise.** The whole three-answer gap between them sits in the variance bucket. With the eight suspect questions removed, both prompts score about 81% on train.

## Conclusion

Prompt iteration on Haiku is exhausted for this question set. The levers that could still move the measured score are fixing the grading (answer keys and the judge's handling of multiple accepted answers), higher effort or majority voting for the variance cases, and retrieval for the knowledge cases.
