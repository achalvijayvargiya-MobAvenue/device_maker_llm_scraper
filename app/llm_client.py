"""
Async OpenAI client wrapper.

Responsibilities:
- Single place for all OpenAI API calls.
- Enforces temperature=0, configurable model/timeout.
- Tracks token usage and latency per call.
- Exposes a clean interface for the extractor / retry handler.
"""

from __future__ import annotations

import time
from typing import Any

from openai import AsyncOpenAI, APIStatusError, APIConnectionError, RateLimitError

from app.config import get_settings
from app.logger import get_logger

logger = get_logger(__name__)


class LLMResponse:
    """Lightweight value object returned from a single LLM call."""

    __slots__ = (
        "content",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "model",
        "finish_reason",
    )

    def __init__(
        self,
        content: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model: str,
        finish_reason: str,
    ) -> None:
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.model = model
        self.finish_reason = finish_reason

    def __repr__(self) -> str:
        return (
            f"LLMResponse(tokens={self.input_tokens}+{self.output_tokens}, "
            f"latency={self.latency_ms:.0f}ms, finish={self.finish_reason!r})"
        )


class LLMClient:
    """
    Async wrapper around OpenAI's chat completions endpoint.

    Usage
    -----
    async with LLMClient() as client:
        response = await client.complete(messages)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: AsyncOpenAI | None = None

    async def __aenter__(self) -> "LLMClient":
        self._client = AsyncOpenAI(
            api_key=self._settings.openai_api_key,
            timeout=self._settings.openai_timeout,
            max_retries=0,  # Retries managed externally via tenacity
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        request_id: str = "",
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Send a chat completion request and return an LLMResponse.

        Parameters
        ----------
        messages:
            OpenAI messages list (system + user, optionally assistant turn).
        request_id:
            Optional trace ID for logging correlation.

        Returns
        -------
        LLMResponse

        Raises
        ------
        RateLimitError, APIConnectionError, APIStatusError:
            Propagated so the retry handler can act on them.
        """
        if self._client is None:
            raise RuntimeError("LLMClient must be used as an async context manager.")

        settings = self._settings
        t0 = time.perf_counter()

        logger.debug(
            '{"event":"llm_request","request_id":"%s","model":"%s","messages":%d}',
            request_id,
            settings.openai_model,
            len(messages),
        )

        response = await self._client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            response_format={"type": "json_object"} if json_mode else {"type": "text"},
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        choice = response.choices[0]
        usage = response.usage

        result = LLMResponse(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            model=response.model,
            finish_reason=choice.finish_reason or "unknown",
        )

        logger.info(
            '{"event":"llm_response","request_id":"%s","input_tokens":%d,'
            '"output_tokens":%d,"latency_ms":%.1f,"finish":"%s"}',
            request_id,
            result.input_tokens,
            result.output_tokens,
            result.latency_ms,
            result.finish_reason,
        )

        return result
