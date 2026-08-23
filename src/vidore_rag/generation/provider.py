from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any, Protocol, cast

from vidore_rag.generation.models import (
    ProviderConfig,
    ProviderRequest,
    ProviderResponse,
    TokenUsage,
)


class LLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def generate(self, request: ProviderRequest) -> ProviderResponse: ...


class _ResponsesAPI(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _OpenAIClient(Protocol):
    responses: _ResponsesAPI


class OpenAIResponsesProvider(LLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        if config.provider != "openai":
            raise ValueError("OpenAIResponsesProvider requires provider='openai'")
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"missing API key environment variable {config.api_key_env}; "
                "set it in your shell, never in a committed config file"
            )
        try:
            openai_module = import_module("openai")
        except ImportError as exc:
            raise RuntimeError(
                "install the generation dependencies: pip install -e '.[generation]'"
            ) from exc
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": config.timeout_seconds,
            "max_retries": config.max_retries,
        }
        if config.base_url is not None:
            client_kwargs["base_url"] = config.base_url
        self._client = cast(_OpenAIClient, openai_module.OpenAI(**client_kwargs))
        self._config = config

    @property
    def name(self) -> str:
        return self._config.provider

    @property
    def model(self) -> str:
        return self._config.model

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        started = time.perf_counter()
        response = self._client.responses.create(
            model=self.model,
            instructions=request.instructions,
            input=request.prompt,
            max_output_tokens=request.max_output_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "cited_document_answer",
                    "strict": True,
                    "schema": request.output_schema,
                }
            },
            store=False,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        usage = response.usage
        if usage is None:
            raise RuntimeError("provider response did not include token usage")
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        token_usage = TokenUsage(
            input_tokens=int(usage.input_tokens),
            cached_input_tokens=int(getattr(input_details, "cached_tokens", 0) or 0),
            output_tokens=int(usage.output_tokens),
            reasoning_tokens=int(getattr(output_details, "reasoning_tokens", 0) or 0),
            total_tokens=int(usage.total_tokens),
        )
        return ProviderResponse(
            provider=self.name,
            model=str(response.model),
            response_id=str(response.id),
            output_text=str(response.output_text),
            usage=token_usage,
            latency_ms=latency_ms,
            time_to_response_ms=latency_ms,
        )


class RateLimitedProvider(LLMProvider):
    def __init__(self, provider: LLMProvider, *, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._provider = provider
        self._minimum_interval = 1 / requests_per_second
        self._lock = threading.Lock()
        self._next_request_at = 0.0

    @property
    def name(self) -> str:
        return self._provider.name

    @property
    def model(self) -> str:
        return self._provider.model

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        with self._lock:
            now = time.monotonic()
            delay = max(self._next_request_at - now, 0)
            self._next_request_at = max(self._next_request_at, now) + self._minimum_interval
        if delay:
            time.sleep(delay)
        return self._provider.generate(request)


def build_provider(config: ProviderConfig) -> LLMProvider:
    if config.provider == "openai":
        return OpenAIResponsesProvider(config)
    raise ValueError(f"unsupported generation provider: {config.provider!r}")
