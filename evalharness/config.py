"""Suite configuration, loaded from a TOML file next to the suite's cases."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    provider: str = "anthropic"
    model: str = "claude-opus-5-5"
    max_tokens: int = 16000
    # Claude Opus 5.5 always thinks adaptively; effort is the depth control (its default is
    # "medium"). Set it explicitly so runs are comparable.
    effort: str | None = "medium"


@dataclass
class RunConfig:
    reps: int = 3
    concurrency: int = 4
    timeout_s: float = 300.0
    max_attempts: int = 4
    seed: int = 0


@dataclass
class SuiteConfig:
    name: str
    description: str
    dataset: Path
    prompt_template: str
    system_prompt: str | None
    model: ModelConfig
    run: RunConfig
    graders: list[dict[str, Any]]
    mock_answer_template: str = "{answer}"
    source: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        data = asdict(self)
        data["dataset"] = str(self.dataset)
        data["source"] = str(self.source) if self.source else None
        return data


class ConfigError(ValueError):
    pass


def load_suite(path: Path) -> SuiteConfig:
    """Load a suite from a directory (containing suite.toml) or a .toml path."""
    path = Path(path)
    if path.is_dir():
        path = path / "suite.toml"
    if not path.exists():
        raise ConfigError(f"no suite config at {path}")
    with path.open("rb") as f:
        raw = tomllib.load(f)

    suite = raw.get("suite", {})
    for key in ("name", "dataset", "prompt_template"):
        if key not in suite:
            raise ConfigError(f"{path}: [suite] is missing '{key}'")
    graders = raw.get("graders", [])
    if not graders:
        raise ConfigError(f"{path}: define at least one [[graders]] entry")
    primaries = [g for g in graders if g.get("primary")]
    if len(primaries) > 1:
        raise ConfigError(f"{path}: only one grader may be primary")
    if not primaries:
        graders[0]["primary"] = True
    names = [g.get("name", g.get("type")) for g in graders]
    if len(set(names)) != len(names):
        raise ConfigError(f"{path}: grader names must be unique (got {names})")

    return SuiteConfig(
        name=suite["name"],
        description=suite.get("description", ""),
        dataset=(path.parent / suite["dataset"]).resolve(),
        prompt_template=suite["prompt_template"],
        system_prompt=suite.get("system_prompt"),
        model=ModelConfig(**raw.get("model", {})),
        run=RunConfig(**raw.get("run", {})),
        graders=graders,
        mock_answer_template=raw.get("mock", {}).get("answer_template", "{answer}"),
        source=path.resolve(),
    )
