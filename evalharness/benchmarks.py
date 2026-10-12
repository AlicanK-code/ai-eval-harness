"""Build a suite's cases.jsonl from public benchmarks on Hugging Face.

A suite that draws on public benchmarks ships a mix.toml instead of its cases:

    seed = 7
    [[sources]]
    adapter = "gpqa_diamond"
    n = 25
    tier = "3-hard"

`evalh fetch <suite>` downloads each source, samples `n` questions with a fixed seed,
formats them consistently, and writes cases.jsonl plus a sources.lock.json recording
exactly which dataset revisions were used. Neither file is committed: several of these
benchmarks ask that their questions never be republished, so they're downloaded on
demand under each dataset's own terms.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

LETTERS = "ABCDEFGHIJ"


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Source:
    name: str
    tag: str  # second tag on every case, used for the per-source breakdown
    repo: str
    license: str
    gated: bool
    load: Callable[[int, random.Random], list[dict[str, Any]]]


ADAPTERS: dict[str, Source] = {}


def adapter(name: str, *, tag: str, repo: str, license: str, gated: bool = False):
    def register(fn):
        ADAPTERS[name] = Source(name, tag, repo, license, gated, fn)
        return fn

    return register


def _datasets():
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        import datasets
    except ImportError as e:
        raise FetchError('the benchmark loader needs extra packages: pip install -e ".[benchmarks]"') from e
    datasets.logging.set_verbosity_error()
    datasets.disable_progress_bars()
    return datasets


def _choices(question: str, options: list[str]) -> str:
    lines = [f"({LETTERS[i]}) {str(opt).strip()}" for i, opt in enumerate(options)]
    return f"{question.strip()}\n\n" + "\n".join(lines)


def _case(case_id: str, question: str, reference: Any, answer_type: str, **meta: Any) -> dict[str, Any]:
    return {
        "id": case_id,
        "input": {"question": question},
        "reference": reference,
        "meta": {"answer_type": answer_type, **meta},
    }


def _sample_stratified(groups: dict[str, list], n: int, rng: random.Random) -> list:
    """Round-robin across groups (sorted, each shuffled) so every group is represented."""
    pools = {k: rng.sample(v, len(v)) for k, v in sorted(groups.items())}
    picked: list = []
    while len(picked) < n and any(pools.values()):
        for key in list(pools):
            if pools[key] and len(picked) < n:
                picked.append(pools[key].pop())
    if len(picked) < n:
        raise FetchError(f"only {len(picked)} eligible questions, {n} requested")
    return picked


def _require(row: dict, keys: list[str], repo: str) -> None:
    missing = [k for k in keys if k not in row]
    if missing:
        raise FetchError(f"{repo} no longer has columns {missing}; the adapter needs updating")


@adapter("arc_challenge", tag="arc-challenge", repo="allenai/ai2_arc", license="CC BY-SA 4.0")
def arc_challenge(n: int, rng: random.Random) -> list[dict[str, Any]]:
    ds = _datasets().load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = list(ds)
    _require(rows[0], ["id", "question", "choices", "answerKey"], "allenai/ai2_arc")
    cases = []
    for row in rng.sample(rows, n):
        labels = row["choices"]["label"]
        letter = LETTERS[labels.index(row["answerKey"])]  # some items label options 1-4
        cases.append(_case(f"arc-{row['id']}", _choices(row["question"], row["choices"]["text"]), letter, "choice"))
    return cases


MMLU_STEM = ["biology", "chemistry", "computer science", "engineering", "math", "physics"]


@adapter("mmlu_pro_stem", tag="mmlu-pro", repo="TIGER-Lab/MMLU-Pro", license="MIT")
def mmlu_pro_stem(n: int, rng: random.Random) -> list[dict[str, Any]]:
    ds = _datasets().load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    groups: dict[str, list] = defaultdict(list)
    for row in ds:
        if row["category"] in MMLU_STEM:
            groups[row["category"]].append(row)
    cases = []
    for row in _sample_stratified(groups, n, rng):
        options = list(row["options"])
        while options and options[-1].strip() == "N/A":  # padding added by the dataset
            options.pop()
        cases.append(_case(
            f"mmlu-pro-{row['question_id']}", _choices(row["question"], options), row["answer"], "choice",
            subject=row["category"],
        ))
    return cases


_PLAIN_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")


@adapter("math500_hard", tag="math-500", repo="HuggingFaceH4/MATH-500",
         license="not stated on the card; the source MATH dataset is MIT")
def math500_hard(n: int, rng: random.Random) -> list[dict[str, Any]]:
    ds = _datasets().load_dataset("HuggingFaceH4/MATH-500", split="test")
    # Levels 4-5 only, and only problems whose answer is a plain number so grading is exact.
    # (That skews toward arithmetic-answer topics; algebraic answers would need a judge.)
    groups: dict[str, list] = defaultdict(list)
    for row in ds:
        if row["level"] >= 4 and _PLAIN_NUMBER.match(row["answer"].strip()):
            groups[f"level-{row['level']}"].append(row)
    cases = []
    for row in _sample_stratified(groups, n, rng):
        slug = re.sub(r"[^a-z0-9]+", "-", row["unique_id"].lower()).strip("-")
        question = row["problem"].strip() + "\n\nGive the final answer as a plain number."
        cases.append(_case(f"math500-{slug}", question, row["answer"].strip(), "numeric",
                           subject=row["subject"], level=row["level"]))
    return cases


@adapter("gpqa_diamond", tag="gpqa-diamond", repo="Idavidrein/gpqa", license="CC BY 4.0", gated=True)
def gpqa_diamond(n: int, rng: random.Random) -> list[dict[str, Any]]:
    ds = _datasets().load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rows = list(ds)
    cols = ["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Record ID"]
    _require(rows[0], cols, "Idavidrein/gpqa")
    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        groups[row.get("High-level domain") or "other"].append(row)
    cases = []
    for row in _sample_stratified(groups, n, rng):
        options = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        # Shuffle per question with its own seed, so the correct option isn't always (A).
        order = random.Random(f"gpqa-{row['Record ID']}").sample(range(4), 4)
        shuffled = [options[i] for i in order]
        letter = LETTERS[order.index(0)]
        cases.append(_case(f"gpqa-{row['Record ID']}", _choices(row["Question"], shuffled), letter, "choice",
                           subject=row.get("High-level domain")))
    return cases


@adapter("hle_text", tag="hle", repo="cais/hle", license="MIT", gated=True)
def hle_text(n: int, rng: random.Random) -> list[dict[str, Any]]:
    ds = _datasets().load_dataset("cais/hle", split="test")
    cols = ["id", "question", "image", "answer", "answer_type", "category"]
    _require({c: None for c in ds.column_names}, cols, "cais/hle")
    # Select only text columns: iterating the image-preview column would decode every image.
    rows = list(ds.select_columns(cols))
    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        if not row["image"]:  # text-only questions
            groups[row["category"]].append(row)
    cases = []
    for row in _sample_stratified(groups, n, rng):
        answer = row["answer"].strip()
        if row["answer_type"] == "multipleChoice" and re.fullmatch(r"[A-J]", answer):
            kind = "choice"
        else:
            kind = "free"
        cases.append(_case(f"hle-{row['id']}", row["question"].strip(), answer, kind,
                           subject=row["category"], judge_task=row["question"].strip()))
    return cases


SCIFACT_LABELS = {"SUPPORT": "SUPPORT", "CONTRADICT": "CONTRADICT", "": "NOT ENOUGH INFO"}


@adapter("scifact", tag="scifact", repo="allenai/scifact", license="CC BY-NC 2.0")
def scifact(n: int, rng: random.Random) -> list[dict[str, Any]]:
    datasets = _datasets()
    from huggingface_hub import hf_hub_download

    def parquet(path: str):
        local = hf_hub_download("allenai/scifact", path, repo_type="dataset", revision="refs/convert/parquet")
        return datasets.load_dataset("parquet", data_files=local, split="train")

    corpus = {str(r["doc_id"]): r for r in parquet("corpus/train/0000.parquet")}
    # The test split has no public labels, so questions come from the labelled validation split.
    groups: dict[str, list] = defaultdict(list)
    seen = set()
    for row in parquet("claims/validation/0000.parquet"):
        if row["id"] in seen:
            continue
        doc_id = row["evidence_doc_id"] or (str(row["cited_doc_ids"][0]) if row["cited_doc_ids"] else "")
        if doc_id not in corpus:
            continue
        seen.add(row["id"])
        groups[SCIFACT_LABELS[row["evidence_label"]]].append((row, corpus[doc_id]))
    cases = []
    for claim, doc in _sample_stratified(groups, n, rng):
        abstract = doc["abstract"]
        if isinstance(abstract, str):  # the parquet conversion stores the sentence list as text
            abstract = ast.literal_eval(abstract)
        title = str(doc["title"]).strip("'\"")
        question = (
            f"Claim: {claim['claim']}\n\n"
            f"Abstract — {title}\n{' '.join(s.strip() for s in abstract)}\n\n"
            "Based only on this abstract, does it SUPPORT the claim, CONTRADICT it, or give NOT ENOUGH INFO? "
            "Answer with exactly one of: SUPPORT, CONTRADICT, NOT ENOUGH INFO."
        )
        label = SCIFACT_LABELS[claim["evidence_label"]]
        cases.append(_case(f"scifact-{claim['id']}", question, label, "label"))
    return cases


def _qasper_answer(ans: dict[str, Any]) -> str:
    if ans["unanswerable"]:
        return "Unanswerable"
    if ans["yes_no"] is not None:
        return "Yes" if ans["yes_no"] else "No"
    spans = [s.strip() for s in ans["extractive_spans"] if s.strip()]
    return "; ".join(spans) if spans else ans["free_form_answer"].strip()


def _qasper_paper(paper: dict[str, Any]) -> str:
    parts = [f"Title: {paper['title']}", f"Abstract: {paper['abstract']}"]
    for name, paragraphs in zip(paper["full_text"]["section_name"], paper["full_text"]["paragraphs"]):
        body = "\n".join(p.strip() for p in paragraphs if p and p.strip())
        if body:
            parts.append(f"## {name or 'Section'}\n{body}")
    return "\n\n".join(parts)


@adapter("qasper", tag="qasper", repo="allenai/qasper", license="CC BY 4.0")
def qasper(n: int, rng: random.Random) -> list[dict[str, Any]]:
    ds = _datasets().load_dataset("allenai/qasper", split="test", revision="refs/convert/parquet")
    groups: dict[str, list] = defaultdict(list)
    for paper in ds:
        qas = paper["qas"]
        for question, qid, answers in zip(qas["question"], qas["question_id"], qas["answers"]):
            rendered = [_qasper_answer(a) for a in answers["answer"]]
            rendered = list(dict.fromkeys(r for r in rendered if r))
            if not rendered:
                continue
            # Skip questions the human annotators disagreed on (one says the paper answers
            # it, another says it doesn't; or yes vs no): there's no fair reference answer.
            if len(rendered) > 1 and ("Unanswerable" in rendered or {"Yes", "No"} <= set(rendered)):
                continue
            kind = "unanswerable" if rendered[0] == "Unanswerable" else "answerable"
            groups[kind].append((paper, question.strip(), qid, rendered))
    # About one in five questions has no answer in the paper; keep that share so the
    # suite rewards saying so instead of inventing an answer.
    n_unanswerable = max(1, n // 5)
    picked = (rng.sample(groups["unanswerable"], n_unanswerable)
              + rng.sample(groups["answerable"], n - n_unanswerable))
    cases = []
    for paper, question, qid, rendered in picked:
        reference = rendered[0] if len(rendered) == 1 else "Accepted answers (any one): " + " | ".join(rendered)
        prompt = (
            f"<paper>\n{_qasper_paper(paper)}\n</paper>\n\n"
            f"Answer this question about the paper above: {question}\n"
            'Answer briefly. If the paper does not contain the answer, answer "Unanswerable".'
        )
        cases.append(_case(f"qasper-{qid[:12]}", prompt, reference, "free", answer_in_context=True,
                           judge_task=f"Question about an NLP research paper: {question}"))
    return cases


def _revision(repo: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo).sha
    except Exception:
        return None


def build(suite_dir: Path, *, skip_unavailable: bool = False, log: Callable[[str], None] = print) -> Path:
    mix_path = suite_dir / "mix.toml"
    if not mix_path.exists():
        raise FetchError(f"{suite_dir} has no mix.toml")
    with mix_path.open("rb") as f:
        spec = tomllib.load(f)
    seed = spec.get("seed", 0)
    rows: list[dict[str, Any]] = []
    lock: dict[str, Any] = {"seed": seed, "sources": []}

    for src in spec["sources"]:
        if src["adapter"] not in ADAPTERS:
            raise FetchError(f"unknown adapter {src['adapter']!r}; choose from {sorted(ADAPTERS)}")
        source = ADAPTERS[src["adapter"]]
        log(f"  {source.name}: fetching {src['n']} from {source.repo}")
        try:
            cases = source.load(src["n"], random.Random(f"{seed}:{source.name}"))
        except Exception as e:
            hint = ""
            if source.gated and any(w in str(e).lower() for w in ("gated", "authenticat", "401", "403")):
                hint = (f" - this dataset is gated: accept its terms at https://huggingface.co/datasets/"
                        f"{source.repo} and log in with `hf auth login`")
            message = f"{source.name} failed: {type(e).__name__}: {str(e)[:300]}{hint}"
            if not skip_unavailable:
                raise FetchError(message) from e
            log(f"  SKIPPED {message}")
            lock["sources"].append({"adapter": source.name, "skipped": True, "reason": message})
            continue
        for case in cases:
            case["tags"] = [src["tier"], source.tag]
            case["meta"]["source"] = source.repo
        rows.extend(cases)
        lock["sources"].append({
            "adapter": source.name, "repo": source.repo, "revision": _revision(source.repo),
            "license": source.license, "n": len(cases), "tier": src["tier"],
        })

    out = suite_dir / "cases.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (suite_dir / "sources.lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    log(f"  wrote {len(rows)} cases to {out}")
    return out
