"""Claude via the official Anthropic Python SDK."""

from __future__ import annotations

import time

import anthropic

from .base import Completion, CompletionRequest, ProviderError, Usage


class AnthropicProvider:
    name = "anthropic"
    retry_base_delay = 2.0

    def __init__(self, model: str):
        self.model = model
        # The runner owns retries so every attempt is counted and latency covers only the
        # final successful request. Credentials resolve from the environment
        # (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile).
        self.client = anthropic.Anthropic(max_retries=0)

    def complete(self, request: CompletionRequest) -> Completion:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            kwargs["system"] = request.system
        output_config: dict = {}
        if request.effort:
            output_config["effort"] = request.effort
        if request.json_schema:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        if output_config:
            kwargs["output_config"] = output_config

        # Deliberately no server-side refusal fallback: a fallback would silently swap the
        # model under test. Refusals are recorded as their own outcome instead.
        start = time.perf_counter()
        try:
            response = self.client.with_options(timeout=request.timeout_s).messages.create(**kwargs)
        except anthropic.APITimeoutError as e:
            raise ProviderError("timeout", str(e), retryable=True) from e
        except anthropic.RateLimitError as e:
            raise ProviderError("rate_limit", e.message, retryable=True) from e
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise ProviderError("auth", e.message, retryable=False) from e
        except anthropic.BadRequestError as e:
            raise ProviderError("bad_request", e.message, retryable=False) from e
        except anthropic.NotFoundError as e:
            raise ProviderError("not_found", e.message, retryable=False) from e
        except anthropic.APIStatusError as e:
            retryable = e.status_code >= 500 or e.status_code in (408, 409)
            raise ProviderError(f"http_{e.status_code}", e.message, retryable=retryable) from e
        except anthropic.APIConnectionError as e:
            raise ProviderError("connection", str(e), retryable=True) from e
        latency = time.perf_counter() - start

        # A score served by a different model measures nothing - fail loudly.
        if not response.model.startswith(self.model):
            raise ProviderError(
                "model_mismatch",
                f"requested {self.model}, served by {response.model}",
                retryable=False,
            )

        text = "".join(b.text for b in response.content if b.type == "text")
        u = response.usage
        category = None
        if response.stop_reason == "refusal" and getattr(response, "stop_details", None):
            category = response.stop_details.category
        return Completion(
            text=text,
            stop_reason=response.stop_reason or "unknown",
            model=response.model,
            usage=Usage(
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cache_read_input_tokens=u.cache_read_input_tokens or 0,
                cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
            ),
            latency_s=latency,
            refusal_category=category,
        )
