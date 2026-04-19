"""LLM client wrapper for AI Reviewer.

This module provides an async-compatible wrapper around the OpenAI SDK
with support for OpenRouter, Ollama, vLLM, and other OpenAI-compatible APIs.

Features:
- Async chat completions with retry logic (exponential backoff)
- Token tracking integration
- Budget enforcement
- Error handling for common API issues (429, 503, etc.)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError, APIStatusError

from .token_tracker import TokenTracker, TokenUsage, BudgetExceededError

logger = logging.getLogger(__name__)


@dataclass
class ChatCompletionRequest:
    """Represents a chat completion request.

    Attributes:
        messages: List of message dicts with 'role' and 'content'.
        model: Model identifier to use.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.
        budget_limit: Optional budget limit for this request.
    """

    messages: list[dict[str, str]]
    model: str
    max_tokens: int | None = None
    temperature: float | None = None
    budget_limit: float | None = None


@dataclass
class ChatCompletionResponse:
    """Represents a chat completion response.

    Attributes:
        content: Response text content.
        model: Model that generated the response.
        usage: Token usage information.
        finish_reason: Reason for completion (stop, length, etc.).
    """

    content: str
    model: str
    usage: TokenUsage
    finish_reason: str


class LLMClientError(Exception):
    """Base exception for LLM client errors."""

    pass


class RetryExhaustedError(LLMClientError):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, last_error: Exception | None = None) -> None:
        super().__init__(message)
        self.last_error = last_error


# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _extract_usage(usage_data: Any) -> TokenUsage:
    """Extract token usage from OpenAI response usage object.

    Args:
        usage_data: Usage object from OpenAI response.

    Returns:
        TokenUsage with token counts.
    """
    if usage_data is None:
        return TokenUsage()

    return TokenUsage(
        input_tokens=getattr(usage_data, "prompt_tokens", 0),
        output_tokens=getattr(usage_data, "completion_tokens", 0),
        cached_tokens=getattr(usage_data, "prompt_tokens_details", {}).get("cached_tokens", 0)
        if hasattr(usage_data, "prompt_tokens_details") and usage_data.prompt_tokens_details
        else 0,
    )


class LLMClient:
    """Async LLM client wrapper for OpenAI-compatible APIs.

    This class provides a unified interface for calling various LLM providers
    through the OpenAI SDK, with built-in retry logic, token tracking, and
    budget enforcement.

    Attributes:
        base_url: API endpoint URL.
        api_key: Authentication key.
        model: Default model to use.
        tracker: TokenTracker instance for usage monitoring.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        budget_limit_usd: float | None = None,
        timeout: int = 60,
    ) -> None:
        """Initialize the LLM client.

        Args:
            base_url: Base URL for the API endpoint.
            api_key: API key for authentication.
            model: Default model identifier.
            budget_limit_usd: Optional budget limit in USD.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

        # Initialize OpenAI async client
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=float(timeout),
        )

        # Initialize token tracker
        self.tracker = TokenTracker(
            model_name=model,
            budget_limit_usd=budget_limit_usd,
        )

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        budget_limit: float | None = None,
        response_format: dict[str, str] | None = None,
    ) -> ChatCompletionResponse:
        """Send a chat completion request with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model to use (overrides default if provided).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            budget_limit: Optional per-request budget limit.
            response_format: Optional response format spec (e.g., {"type": "json_object"}).

        Returns:
            ChatCompletionResponse with content and usage info.

        Raises:
            RetryExhaustedError: If all retries fail.
            BudgetExceededError: If budget limit is exceeded.
            LLMClientError: For other client errors.
        """
        model_to_use = model or self.model
        effective_budget = budget_limit or self.tracker.budget_limit_usd

        # Update tracker budget if specified
        if effective_budget is not None:
            self.tracker.budget_limit_usd = effective_budget

        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                return await self._make_request(
                    messages=messages,
                    model=model_to_use,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                )
            except RateLimitError as e:
                last_error = e
                logger.warning(
                    "Rate limit hit (attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    str(e),
                )
                await self._handle_retry_delay(attempt, is_rate_limit=True)
            except APIConnectionError as e:
                last_error = e
                logger.warning(
                    "Connection error (attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    str(e),
                )
                await self._handle_retry_delay(attempt)
            except APIStatusError as e:
                if e.status_code in RETRYABLE_STATUS_CODES:
                    last_error = e
                    logger.warning(
                        "API error %d (attempt %d/%d): %s",
                        e.status_code,
                        attempt + 1,
                        MAX_RETRIES,
                        str(e),
                    )
                    await self._handle_retry_delay(attempt)
                else:
                    # Non-retryable error
                    raise LLMClientError(f"API error: {e.status_code} - {e.message}") from e
            except BudgetExceededError:
                raise
            except Exception as e:
                last_error = e
                logger.warning(
                    "Unexpected error (attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    str(e),
                )
                await self._handle_retry_delay(attempt)

        # All retries exhausted
        raise RetryExhaustedError(
            f"Failed after {MAX_RETRIES} attempts",
            last_error=last_error,
        )

    async def _make_request(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, str] | None = None,
    ) -> ChatCompletionResponse:
        """Make the actual API request.

        Args:
            messages: List of message dicts.
            model: Model to use.
            max_tokens: Maximum tokens.
            temperature: Sampling temperature.
            response_format: Optional response format.

        Returns:
            ChatCompletionResponse with result.

        Raises:
            BudgetExceededError: If budget would be exceeded.
            APIError: For API errors.
        """
        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if temperature is not None:
            kwargs["temperature"] = temperature

        # Enable JSON mode if requested
        if response_format is not None:
            kwargs["response_format"] = response_format

        # Make the API call
        response = await self._client.chat.completions.create(**kwargs)

        # Extract response data
        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason or "unknown"

        # Extract and record usage
        usage = _extract_usage(response.usage)

        # Record usage and check budget
        recorded_usage = self.tracker.record_usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
        )

        # Log metrics (without prompt/response content)
        self.tracker.log_metrics(recorded_usage)

        logger.info(
            "Chat completion: model=%s, finish_reason=%s, tokens=%d",
            model,
            finish_reason,
            recorded_usage.total_tokens,
        )

        return ChatCompletionResponse(
            content=content,
            model=model,
            usage=recorded_usage,
            finish_reason=finish_reason,
        )

    async def _handle_retry_delay(self, attempt: int, is_rate_limit: bool = False) -> None:
        """Wait before retrying with exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed).
            is_rate_limit: Whether this is a rate limit retry.
        """
        # Exponential backoff with jitter
        delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)

        # Add some jitter to avoid thundering herd
        import random

        jitter = random.uniform(0, 0.1 * delay)
        total_delay = delay + jitter

        logger.debug("Retrying in %.2f seconds (attempt %d)", total_delay, attempt + 1)
        await asyncio.sleep(total_delay)

    def get_usage_summary(self) -> dict[str, object]:
        """Get current usage summary.

        Returns:
            Dictionary with usage statistics.
        """
        return self.tracker.get_summary()

    async def close(self) -> None:
        """Close the client and release resources.

        Should be called when done using the client.
        """
        await self._client.close()

    async def __aenter__(self) -> LLMClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
