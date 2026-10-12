from __future__ import annotations

from typing import Any

from .answer_match import AnswerMatch
from .base import Grade, GradeContext, Grader, GraderError
from .deterministic import ExactMatch, JsonMatch, Numeric, TokenF1
from .llm_judge import LLMJudge

REGISTRY = {
    "exact_match": ExactMatch,
    "numeric": Numeric,
    "json_match": JsonMatch,
    "token_f1": TokenF1,
    "llm_judge": LLMJudge,
    "answer_match": AnswerMatch,
}


def build_graders(specs: list[dict[str, Any]]) -> list[Grader]:
    graders = []
    for spec in specs:
        spec = dict(spec)
        kind = spec.pop("type", None)
        if kind not in REGISTRY:
            raise ValueError(f"unknown grader type {kind!r}; choose from {sorted(REGISTRY)}")
        spec.setdefault("name", kind)
        graders.append(REGISTRY[kind](**spec))
    return graders


def primary_grader(graders: list[Grader]) -> Grader:
    return next(g for g in graders if g.primary)


__all__ = ["Grade", "GradeContext", "Grader", "GraderError", "build_graders", "primary_grader"]
