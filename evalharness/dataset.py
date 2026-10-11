"""Eval cases: loading, prompt rendering, and tier-1 health checks."""

from __future__ import annotations

import hashlib
import json
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    id: str
    input: dict[str, Any]
    reference: Any
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def primary_tag(self) -> str:
        return self.tags[0] if self.tags else "untagged"


class DatasetError(ValueError):
    pass


def load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path}:{lineno}: invalid JSON ({e})") from e
            missing = [k for k in ("id", "input", "reference") if k not in row]
            if missing:
                raise DatasetError(f"{path}:{lineno}: missing field(s) {missing}")
            if not isinstance(row["input"], dict):
                raise DatasetError(f"{path}:{lineno}: 'input' must be an object")
            cases.append(
                Case(
                    id=str(row["id"]),
                    input=row["input"],
                    reference=row["reference"],
                    tags=tuple(row.get("tags", [])),
                )
            )
    ids = Counter(c.id for c in cases)
    dupes = [i for i, n in ids.items() if n > 1]
    if dupes:
        raise DatasetError(f"{path}: duplicate case ids {dupes}")
    return cases


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def template_fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def render_prompt(template: str, case: Case) -> str:
    return template.format(**case.input)


def _normalise_for_dupes(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def health_report(cases: list[Case], prompt_template: str) -> dict[str, Any]:
    """Cheap, exhaustive checks over the full case set (eval-audit tier 1)."""
    needed = template_fields(prompt_template)
    missing_fields = {c.id: sorted(needed - c.input.keys()) for c in cases if needed - c.input.keys()}

    rendered = {}
    for c in cases:
        if c.id not in missing_fields:
            rendered[c.id] = _normalise_for_dupes(render_prompt(prompt_template, c))
    seen: dict[str, str] = {}
    near_dupes = []
    for cid, text in rendered.items():
        if text in seen:
            near_dupes.append((seen[text], cid))
        else:
            seen[text] = cid

    # Label leakage: the reference appearing verbatim in the prompt the model sees.
    leaks = []
    for c in cases:
        ref = c.reference
        if isinstance(ref, str) and len(ref) >= 4 and c.id in rendered:
            if _normalise_for_dupes(ref) in rendered[c.id]:
                leaks.append(c.id)

    lengths = sorted(len(t) for t in rendered.values()) or [0]
    tag_counts = Counter(c.primary_tag for c in cases)
    ref_counts = Counter(json.dumps(c.reference, sort_keys=True) for c in cases)
    majority_ref, majority_n = ref_counts.most_common(1)[0] if cases else ("", 0)
    return {
        "n_cases": len(cases),
        "tags": dict(sorted(tag_counts.items())),
        "missing_template_fields": missing_fields,
        "near_duplicates": near_dupes,
        "reference_in_prompt": leaks,
        "prompt_chars": {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1]},
        # For classification-style sets: what does always answering the commonest label score?
        "majority_reference_share": round(majority_n / len(cases), 3) if cases else 0.0,
        "majority_reference": json.loads(majority_ref) if cases else None,
    }
