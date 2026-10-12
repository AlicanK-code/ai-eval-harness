"""One grader for a suite that mixes answer formats.

Each case says how its answer should be checked, in case.meta["answer_type"]:
  choice   a letter (A-J) for multiple-choice questions
  label    one of a fixed set of words, e.g. SUPPORT / CONTRADICT / NOT ENOUGH INFO
  numeric  a number, compared with a relative tolerance
  free     a short free-text answer, checked for equivalence by an LLM judge

Only the model's final "ANSWER:" line is graded, so a letter or number that appears
in the working can't pass by accident. A response with no such line is recorded as
unanswered rather than wrong.
"""

from __future__ import annotations

import math
import re

from .base import Grade, GradeContext, normalise_text
from .deterministic import _NUMBER, _to_float
from .llm_judge import LLMJudge

_ANSWER_LINE = re.compile(r"^[\s>*#_]*(?:final\s+)?answer\s*[:：]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_LETTER = re.compile(r"^\(?\[?([A-J])\]?\)?(?:[.):\s]|$)")

EQUIVALENCE = {
    "id": "equivalent",
    "description": (
        "The candidate's final answer means the same as the reference answer. When the reference "
        "lists several accepted answers separated by '|', compare the candidate with each one on "
        "its own: it is correct if it matches any single one, and it never needs details that "
        "appear only in a longer accepted answer. Extra correct detail is fine; a different value, "
        "a claim that contradicts the reference, or a hedge between different answers is not. "
        "Differences in wording, formatting, notation or units of expression don't matter."
    ),
}


def final_answer(output: str, *, to_end: bool = False) -> str | None:
    """Text after the last "ANSWER:" marker: that line only, or (for free-text answers,
    which may run over several lines) everything after it."""
    matches = list(_ANSWER_LINE.finditer(output))
    if not matches:
        return None
    m = matches[-1]
    text = output[m.start(1):] if to_end else m.group(1)
    return text.replace("**", "").strip().strip("`$").strip()


class AnswerMatch:
    def __init__(
        self,
        name: str,
        primary: bool = False,
        judge_provider: str = "anthropic",
        judge_model: str = "claude-sonnet-5-5",
        judge_effort: str | None = "low",
        rel_tol: float = 1e-4,
    ):
        self.name, self.primary, self.rel_tol = name, primary, rel_tol
        self.judge = LLMJudge(
            name,
            [EQUIVALENCE],
            judge_provider=judge_provider,
            judge_model=judge_model,
            judge_effort=judge_effort,
        )

    def grade(self, output: str, ctx: GradeContext) -> Grade:
        kind = ctx.case.meta.get("answer_type", "free")
        answer = final_answer(output, to_end=kind == "free")
        if not answer:
            return Grade(0.0, False, answered=False, details={"answer_type": kind, "reason": "no ANSWER line"})
        reference = str(ctx.case.reference)
        details = {"answer_type": kind, "extracted": answer[:300]}

        if kind == "choice":
            m = _LETTER.match(answer.upper())
            if not m:
                return Grade(0.0, False, answered=False, details={**details, "reason": "no option letter"})
            ok = m.group(1) == reference.upper()
        elif kind == "label":
            got, want = normalise_text(answer), normalise_text(reference)
            ok = got == want or got.startswith(want + " ")
        elif kind == "numeric":
            numbers = re.findall(_NUMBER, answer)
            if not numbers:
                return Grade(0.0, False, answered=False, details={**details, "reason": "no number"})
            got_n, want_n = _to_float(numbers[-1]), float(reference)
            ok = got_n is not None and math.isclose(got_n, want_n, rel_tol=self.rel_tol, abs_tol=1e-9)
        else:
            # The judge sees the question (not the full source document) and only the final
            # answer, which keeps it cheap and stops long reasoning from swaying it.
            task = ctx.case.meta.get("judge_task", ctx.prompt)
            grade = self.judge.grade(f"ANSWER: {answer}", GradeContext(case=ctx.case, prompt=task))
            grade.details = {**details, **grade.details}
            return grade
        return Grade(float(ok), ok, details=details)
