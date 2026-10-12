"""Graders that need no model: exact match, numeric, JSON fields, token F1."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

from .base import Grade, GradeContext, normalise_text


def _extract(output: str, pattern: str | None) -> str | None:
    """Apply an optional extraction regex; the LAST match wins (final answers come last)."""
    if not pattern:
        return output
    matches = list(re.finditer(pattern, output, flags=re.IGNORECASE | re.MULTILINE))
    if not matches:
        return None
    m = matches[-1]
    return m.group(1) if m.groups() else m.group(0)


class ExactMatch:
    """Normalised string equality. `reference` may be a string or a list of accepted forms."""

    def __init__(self, name: str, primary: bool = False, extract: str | None = None):
        self.name, self.primary, self.extract = name, primary, extract

    def grade(self, output: str, ctx: GradeContext) -> Grade:
        accepted = ctx.case.reference if isinstance(ctx.case.reference, list) else [ctx.case.reference]
        got = _extract(output, self.extract)
        if got is None or not got.strip():
            return Grade(0.0, False, answered=False, details={"reason": "no answer found"})
        norm = normalise_text(got)
        ok = any(norm == normalise_text(str(a)) for a in accepted)
        return Grade(float(ok), ok, details={"extracted": got.strip()[:200]})


_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


class Numeric:
    """Compare the final number in the output to the reference, within tolerance."""

    def __init__(
        self,
        name: str,
        primary: bool = False,
        extract: str | None = None,
        abs_tol: float = 1e-6,
        rel_tol: float = 1e-6,
    ):
        self.name, self.primary = name, primary
        self.extract = extract
        self.abs_tol, self.rel_tol = abs_tol, rel_tol

    def grade(self, output: str, ctx: GradeContext) -> Grade:
        region = _extract(output, self.extract)
        numbers = re.findall(_NUMBER, region or "")
        if not numbers:
            return Grade(0.0, False, answered=False, details={"reason": "no number found"})
        got = _to_float(numbers[-1])
        want = float(ctx.case.reference)
        ok = got is not None and math.isclose(got, want, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        return Grade(float(ok), ok, details={"extracted": numbers[-1], "expected": want})


def extract_json_object(text: str) -> Any | None:
    """Find the first parseable JSON object in text (tolerates prose and code fences)."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def _values_equal(got: Any, want: Any) -> bool:
    if isinstance(want, bool) or isinstance(got, bool):
        return got is want
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(want, (int, float)) and isinstance(got, str):
        parsed = _to_float(got)
        return parsed is not None and math.isclose(parsed, want, rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(want, str) and isinstance(got, str):
        return normalise_text(got) == normalise_text(want)
    if isinstance(want, list) and isinstance(got, list):
        # Order-insensitive: extraction order is rarely what's being tested.
        norm = lambda xs: sorted(normalise_text(str(x)) for x in xs)  # noqa: E731
        return norm(got) == norm(want)
    return got == want


class JsonMatch:
    """Field-level comparison of a JSON object. Partial credit = fraction of fields correct."""

    def __init__(self, name: str, primary: bool = False, allow_extra_keys: bool = True):
        self.name, self.primary, self.allow_extra_keys = name, primary, allow_extra_keys

    def grade(self, output: str, ctx: GradeContext) -> Grade:
        want = ctx.case.reference
        got = extract_json_object(output)
        if not isinstance(got, dict):
            return Grade(0.0, False, answered=False, details={"reason": "no JSON object found"})
        field_ok = {k: _values_equal(got.get(k), v) if k in got else False for k, v in want.items()}
        extra = sorted(set(got) - set(want))
        score = sum(field_ok.values()) / len(field_ok) if field_ok else 1.0
        ok = all(field_ok.values()) and (self.allow_extra_keys or not extra)
        wrong = {k: {"expected": want[k], "got": got.get(k)} for k, good in field_ok.items() if not good}
        return Grade(score, ok, details={"wrong_fields": wrong, "extra_keys": extra})


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.lower())  # runs of letters/digits, in any script


def token_f1(candidate: str, reference: str) -> float:
    c, r = _tokens(candidate), _tokens(reference)
    if not c or not r:
        return 0.0
    overlap = sum((Counter(c) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(c), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


class TokenF1:
    """Bag-of-words F1 against the reference. A cheap, reference-bound similarity signal -
    good as a secondary metric, weak as a headline for open-ended outputs."""

    def __init__(self, name: str, primary: bool = False, threshold: float = 0.5):
        self.name, self.primary, self.threshold = name, primary, threshold

    def grade(self, output: str, ctx: GradeContext) -> Grade:
        if not output.strip():
            return Grade(0.0, False, answered=False, details={"reason": "empty output"})
        f1 = token_f1(output, str(ctx.case.reference))
        return Grade(f1, f1 >= self.threshold, details={"f1": round(f1, 4)})
