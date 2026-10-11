from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..dataset import Case


@dataclass
class GradeContext:
    case: Case
    prompt: str  # the rendered prompt the model saw


@dataclass
class Grade:
    score: float  # 0.0 - 1.0
    passed: bool
    # False when the output contained no extractable answer at all. "No answer" is kept
    # distinct from "wrong answer" so an empty or crashed response can't pass as a negative.
    answered: bool = True
    details: dict[str, Any] = field(default_factory=dict)
    # Set by graders that call a model (LLM judge), so judge cost is accounted separately.
    judge_model: str | None = None
    judge_usage: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "score": round(self.score, 4),
            "passed": self.passed,
            "answered": self.answered,
            "details": self.details,
        }
        if self.judge_model:
            d["judge_model"] = self.judge_model
            d["judge_usage"] = self.judge_usage
        return d


class GraderError(Exception):
    """The grader itself failed (e.g. the judge API errored). Never scored as a model failure."""


class Grader(Protocol):
    name: str
    primary: bool

    def grade(self, output: str, ctx: GradeContext) -> Grade: ...


_FENCE = re.compile(r"^```[\w-]*\s*|\s*```$")


def normalise_text(text: str) -> str:
    """Forgiving comparison form: no fences/markdown emphasis, casefolded, collapsed spaces,
    no surrounding quotes or trailing punctuation."""
    text = _FENCE.sub("", text.strip())
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"\s+", " ", text).strip().strip("\"'").rstrip(".!;:,").strip()
    return text.casefold()
