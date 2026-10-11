"""A deterministic, offline provider for demos, CI, and harness self-tests.

It knows each case's reference answer and decides - from a stable hash of
(model, case, rep, attempt) - whether to answer correctly, answer wrongly, or
simulate a failure mode (rate limit, truncation, refusal). Same inputs, same outputs,
every run, so tests can assert exact numbers.

Models:
  mock-oracle  always returns the reference          (harness should score ~100%)
  mock-null    always returns an empty string        (harness should score ~0%)
  mock-strong  mostly right, occasional faults
  mock-weak    often wrong, occasional faults
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .base import Completion, CompletionRequest, ProviderError, Usage

PROFILES: dict[str, dict[str, float]] = {
    # base_accuracy, fault rates, and a fake latency range (seconds).
    "mock-oracle": {"acc": 1.0, "transient": 0.0, "truncate": 0.0, "refuse": 0.0, "lat_lo": 0.2, "lat_hi": 0.4},
    "mock-null": {"acc": 0.0, "transient": 0.0, "truncate": 0.0, "refuse": 0.0, "lat_lo": 0.1, "lat_hi": 0.2},
    "mock-strong": {"acc": 0.95, "transient": 0.05, "truncate": 0.02, "refuse": 0.015, "lat_lo": 0.8, "lat_hi": 2.4},
    "mock-weak": {"acc": 0.72, "transient": 0.05, "truncate": 0.02, "refuse": 0.015, "lat_lo": 0.3, "lat_hi": 0.9},
}


def unit_hash(*parts: Any) -> float:
    """Map arbitrary parts to a stable float in [0, 1)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def corrupt(reference: Any, salt: str, labels: list[str] | None = None) -> Any:
    """Produce a plausible-looking wrong answer for a reference."""
    if labels and reference in labels:
        others = [label for label in labels if label != reference]
        return others[int(unit_hash(salt, "label") * len(others))]
    if isinstance(reference, dict):
        wrong = dict(reference)
        if wrong:
            keys = sorted(wrong)
            key = keys[int(unit_hash(salt, "key") * len(keys))]
            value = wrong[key]
            if isinstance(value, bool):
                wrong[key] = not value
            elif isinstance(value, (int, float)):
                wrong[key] = value + 1
            else:
                wrong[key] = None
        return wrong
    text = str(reference)
    try:
        number = float(text.replace(",", ""))
        delta = 1 + int(unit_hash(salt, "delta") * 9)
        wrong_number = number + delta
        return str(int(wrong_number)) if wrong_number.is_integer() else str(wrong_number)
    except ValueError:
        pass
    words = text.split()
    if len(words) > 12:
        return " ".join(words[: len(words) // 4])
    return "unsure"


class MockProvider:
    name = "mock"
    retry_base_delay = 0.0

    def __init__(self, model: str, references: dict[str, Any], answer_template: str = "{answer}"):
        if model not in PROFILES:
            raise ValueError(f"unknown mock model {model!r}; choose from {sorted(PROFILES)}")
        self.model = model
        self.profile = PROFILES[model]
        self.references = references
        self.answer_template = answer_template
        # Classification-style suites: a wrong answer should be another valid label.
        short = sorted({r for r in references.values() if isinstance(r, str) and len(r.split()) <= 3})
        self.labels = short if 2 <= len(short) <= 10 else None

    def complete(self, request: CompletionRequest) -> Completion:
        case_id = request.trace.get("case_id", "")
        rep = request.trace.get("rep", 0)
        attempt = request.trace.get("attempt", 1)
        p = self.profile
        key = (self.model, case_id, rep)

        if unit_hash(*key, attempt, "transient") < p["transient"]:
            raise ProviderError("rate_limit", "mock 429: simulated rate limit", retryable=True)

        latency = p["lat_lo"] + unit_hash(*key, "latency") * (p["lat_hi"] - p["lat_lo"])
        reference = self.references.get(case_id)

        if self.model == "mock-null":
            text = ""
        else:
            # Some cases are hard for every model; that shared difficulty is what makes
            # paired comparisons between mock models behave like real ones.
            difficulty = unit_hash(case_id, "difficulty") * 0.35
            correct = unit_hash(*key, "correct") < p["acc"] - difficulty * (1 - p["acc"]) * 3
            if self.model == "mock-oracle":
                correct = True
            answer = reference if correct else corrupt(reference, f"{key}", self.labels)
            rendered = json.dumps(answer) if isinstance(answer, (dict, list)) else str(answer)
            text = self.answer_template.replace("{answer}", rendered)

        stop_reason = "end_turn"
        category = None
        if unit_hash(*key, "refuse") < p["refuse"]:
            text, stop_reason, category = "", "refusal", "mock"
        elif unit_hash(*key, "truncate") < p["truncate"]:
            text, stop_reason = text[: max(1, len(text) // 2)], "max_tokens"

        return Completion(
            text=text,
            stop_reason=stop_reason,
            model=self.model,
            usage=Usage(
                input_tokens=max(1, len((request.system or "") + request.prompt) // 4),
                output_tokens=max(1, len(text) // 4),
            ),
            latency_s=round(latency, 3),
            refusal_category=category,
        )
