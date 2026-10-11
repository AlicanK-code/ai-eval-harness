"""Per-model token prices (USD per million tokens). Check https://claude.com/pricing before
quoting costs; unknown models report cost as None rather than guessing."""

from __future__ import annotations

PRICES: dict[str, dict[str, float]] = {
    # model:            input, output, cache read, cache write (5-minute)
    "claude-fable-5-1": {"input": 10.00, "output": 50.00, "cache_read": 0.25, "cache_write": 12.50},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
}


def cost_usd(model: str | None, usage: dict[str, int] | None) -> float | None:
    if not model or not usage:
        return None
    price = next((p for name, p in PRICES.items() if model.startswith(name)), None)
    if price is None:
        return 0.0 if model.startswith("mock") else None
    total = (
        usage.get("input_tokens", 0) * price["input"]
        + usage.get("output_tokens", 0) * price["output"]
        + usage.get("cache_read_input_tokens", 0) * price["cache_read"]
        + usage.get("cache_creation_input_tokens", 0) * price["cache_write"]
    )
    return total / 1_000_000
