"""
Retry and resilience layer built on tenacity.

Provides:
- Decorated async retry wrapper for OpenAI calls.
- Rate-limit-aware exponential backoff.
- JSON repair retry for malformed LLM output.
- Structured logging on every attempt.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, TypeVar

from openai import APIConnectionError, APIStatusError, RateLimitError
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    RetryError,
)

from app.config import get_settings
from app.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Exceptions that warrant a retry
_RETRYABLE = (RateLimitError, APIConnectionError, asyncio.TimeoutError)


def _is_retryable_api_error(exc: BaseException) -> bool:
    """Return True for 429 / 500 / 502 / 503 HTTP errors."""
    if isinstance(exc, _RETRYABLE):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in (429, 500, 502, 503, 529)
    return False


async def with_retry(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],
    *,
    request_id: str = "",
) -> T:
    """
    Execute an async coroutine factory with exponential backoff retry.

    Parameters
    ----------
    coro_factory:
        A zero-argument callable that returns a fresh coroutine on each call.
        Must be a factory (not an awaitable) so that a new coroutine object is
        created on each retry attempt.
    request_id:
        Optional trace ID for log correlation.

    Returns
    -------
    T
        The return value of the successful coroutine execution.

    Raises
    ------
    RetryError:
        When all retry attempts are exhausted.
    """
    settings = get_settings()

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(_RETRYABLE)
        | retry_if_exception_type(APIStatusError),
        stop=stop_after_attempt(settings.openai_max_retries + 1),
        wait=wait_exponential(multiplier=1, min=2, max=60) + wait_random(0, 2),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    ):
        with attempt:
            try:
                return await coro_factory()
            except APIStatusError as exc:
                if not _is_retryable_api_error(exc):
                    logger.error(
                        "Non-retryable API error %s for request_id=%s",
                        exc.status_code,
                        request_id,
                    )
                    raise
                logger.warning(
                    "Retryable API error %s for request_id=%s (attempt %d)",
                    exc.status_code,
                    request_id,
                    attempt.retry_state.attempt_number,
                )
                raise  # Let tenacity handle it

    # Unreachable but satisfies type checker
    raise RuntimeError("Retry loop exited without result")


async def extract_json_with_repair(
    llm_call: Callable[[list[dict[str, Any]]], Coroutine[Any, Any, str]],
    repair_call: Callable[
        [str, str], Coroutine[Any, Any, str]
    ],
    messages: list[dict[str, Any]],
    *,
    max_repair_attempts: int = 2,
    request_id: str = "",
) -> list[dict[str, Any]]:
    """
    Call the LLM, parse JSON, and attempt self-repair on parse failure.

    Parameters
    ----------
    llm_call:
        Coroutine factory that accepts messages and returns raw LLM text.
    repair_call:
        Coroutine factory that accepts (malformed_text, error_detail) and
        returns corrected LLM text.
    messages:
        Initial messages payload.
    max_repair_attempts:
        Number of times to ask the LLM to fix its own output.
    request_id:
        Trace ID for logging.

    Returns
    -------
    list[dict]
        Parsed JSON array from LLM response.

    Raises
    ------
    ValueError:
        When JSON cannot be parsed after all repair attempts.
    """
    raw_text = await llm_call(messages)

    for attempt in range(max_repair_attempts + 1):
        cleaned = _strip_markdown_fences(raw_text)
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, list):
                if isinstance(parsed, dict):
                    # Wrapped array: {"items": [...]} or {"results": [...]}
                    for v in parsed.values():
                        if isinstance(v, list):
                            return v
                    # Single device spec returned as a plain object — wrap it
                    if "device_model" in parsed or "device_manufacturer" in parsed:
                        return [parsed]
                raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            error_detail = str(exc)
            if attempt >= max_repair_attempts:
                logger.error(
                    "JSON parse failed after %d repair attempts for request_id=%s: %s",
                    max_repair_attempts,
                    request_id,
                    error_detail,
                )
                raise ValueError(
                    f"Could not parse LLM JSON after {max_repair_attempts} repairs: "
                    f"{error_detail}"
                ) from exc

            logger.warning(
                "JSON parse error (attempt %d/%d) for request_id=%s: %s",
                attempt + 1,
                max_repair_attempts,
                request_id,
                error_detail,
            )
            raw_text = await repair_call(raw_text, error_detail)

    # Should not be reached
    raise ValueError("Repair loop exhausted")


def _strip_markdown_fences(text: str) -> str:
    """
    Remove markdown code fences if the LLM wrapped its JSON in them.

    Handles patterns like:
    ```json\n...\n```
    ```\n...\n```
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (``` or ```json) and last line (```)
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    return text
