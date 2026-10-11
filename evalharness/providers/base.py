"""Provider interface: one request in, one completion (or a classified error) out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CompletionRequest:
    prompt: str
    system: str | None = None
    max_tokens: int = 16000
    effort: str | None = None
    # When set, the provider must return text that parses as JSON matching this schema.
    json_schema: dict[str, Any] | None = None
    timeout_s: float = 300.0
    # Opaque hints for the mock provider (case id, rep, attempt). Real providers ignore it.
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
        }


@dataclass
class Completion:
    text: str
    # Normalised stop reason: "end_turn", "max_tokens", "refusal", or the provider's raw value.
    stop_reason: str
    model: str  # the model that actually served the request
    usage: Usage
    latency_s: float
    refusal_category: str | None = None


class ProviderError(Exception):
    """A failure that produced no scorable output.

    `kind` is recorded in errors.jsonl so plumbing failures are never scored as model failures.
    """

    def __init__(self, kind: str, message: str, retryable: bool):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


class Provider(Protocol):
    name: str
    model: str
    # Base delay (seconds) for the runner's jittered exponential backoff.
    retry_base_delay: float

    def complete(self, request: CompletionRequest) -> Completion: ...
