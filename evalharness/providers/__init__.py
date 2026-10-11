from __future__ import annotations

import dataclasses
import random
import time
from typing import Any

from .base import Completion, CompletionRequest, Provider, ProviderError, Usage


def complete_with_retries(
    provider: Provider, request: CompletionRequest, max_attempts: int
) -> tuple[Completion, int]:
    """Call the provider, retrying transient errors with jittered exponential backoff.

    Returns (completion, attempts). Re-raises the last ProviderError when retries run out
    or the error is not retryable. Latency on the completion covers only the final call.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            req = dataclasses.replace(request, trace={**request.trace, "attempt": attempt})
            return provider.complete(req), attempt
        except ProviderError as e:
            if not e.retryable or attempt == max_attempts:
                e.attempts = attempt  # type: ignore[attr-defined]
                raise
            delay = min(provider.retry_base_delay * 2 ** (attempt - 1), 60.0)
            time.sleep(delay + random.uniform(0, delay / 2))
    raise AssertionError("unreachable")


def make_provider(
    name: str,
    model: str,
    *,
    references: dict[str, Any] | None = None,
    answer_template: str = "{answer}",
) -> Provider:
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model)
    if name == "mock":
        from .mock import MockProvider

        return MockProvider(model, references or {}, answer_template)
    raise ValueError(f"unknown provider {name!r} (expected 'anthropic' or 'mock')")


__all__ = [
    "Completion",
    "CompletionRequest",
    "Provider",
    "ProviderError",
    "Usage",
    "complete_with_retries",
    "make_provider",
]
