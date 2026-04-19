"""Tests for llm_client module."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ai_reviewer.llm_client import (
    LLMClient,
    LLMClientError,
    RetryExhaustedError,
    ChatCompletionResponse,
    _extract_usage,
    MAX_RETRIES,
)
from ai_reviewer.token_tracker import TokenUsage, BudgetExceededError


class TestExtractUsage:
    """Tests for _extract_usage helper function."""

    def test_none_usage(self) -> None:
        """Test extraction from None usage data."""
        result = _extract_usage(None)
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.cached_tokens == 0

    def test_basic_usage(self) -> None:
        """Test extraction from basic usage object."""
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.prompt_tokens_details = None
        
        result = _extract_usage(mock_usage)
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cached_tokens == 0

    def test_usage_with_cached_tokens(self) -> None:
        """Test extraction with cached tokens."""
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.prompt_tokens_details = {"cached_tokens": 25}
        
        result = _extract_usage(mock_usage)
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cached_tokens == 25


class TestLLMClientInitialization:
    """Tests for LLMClient initialization."""

    def test_basic_init(self) -> None:
        """Test basic client initialization."""
        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
        )
        assert client.base_url == "https://api.example.com/v1"
        assert client.api_key == "test-key"
        assert client.model == "gpt-4"
        assert client.timeout == 60

    def test_init_with_budget(self) -> None:
        """Test initialization with budget limit."""
        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
            budget_limit_usd=10.0,
        )
        assert client.tracker.budget_limit_usd == 10.0

    def test_init_with_timeout(self) -> None:
        """Test initialization with custom timeout."""
        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4",
            timeout=120,
        )
        assert client.timeout == 120

    def test_base_url_trailing_slash_removed(self) -> None:
        """Test that trailing slash is removed from base_url."""
        client = LLMClient(
            base_url="https://api.example.com/v1/",
            api_key="test-key",
            model="gpt-4",
        )
        assert client.base_url == "https://api.example.com/v1"


class TestLLMClientChatCompletion:
    """Tests for LLMClient.chat_completion method."""

    @pytest.fixture
    def mock_openai_client(self) -> MagicMock:
        """Create a mock OpenAI client."""
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_successful_request(self, mock_openai_client: MagicMock) -> None:
        """Test successful chat completion request."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Test response"),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=50,
            prompt_tokens_details=None,
        )
        mock_openai_client.chat.completions.create.return_value = mock_response

        with patch("ai_reviewer.llm_client.AsyncOpenAI", return_value=mock_openai_client):
            client = LLMClient(
                base_url="https://api.example.com/v1",
                api_key="test-key",
                model="gpt-4",
            )

            response = await client.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert isinstance(response, ChatCompletionResponse)
            assert response.content == "Test response"
            assert response.model == "gpt-4"
            assert response.finish_reason == "stop"
            assert response.usage.input_tokens == 100
            assert response.usage.output_tokens == 50

    @pytest.mark.asyncio
    async def test_request_with_parameters(self, mock_openai_client: MagicMock) -> None:
        """Test request with custom parameters."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Response"),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=50,
            completion_tokens=30,
            prompt_tokens_details=None,
        )
        mock_openai_client.chat.completions.create.return_value = mock_response

        with patch("ai_reviewer.llm_client.AsyncOpenAI", return_value=mock_openai_client):
            client = LLMClient(
                base_url="https://api.example.com/v1",
                api_key="test-key",
                model="gpt-4",
            )

            await client.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="custom-model",
                max_tokens=100,
                temperature=0.7,
                response_format={"type": "json_object"},
            )

            # Verify parameters were passed correctly
            call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == "custom-model"
            assert call_kwargs["max_tokens"] == 100
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, mock_openai_client: MagicMock) -> None:
        """Test retry logic on rate limit error (429)."""
        from openai import RateLimitError

        # First two calls fail with rate limit, third succeeds
        rate_limit_error = RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Success after retry"),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=50,
            completion_tokens=30,
            prompt_tokens_details=None,
        )

        mock_openai_client.chat.completions.create.side_effect = [
            rate_limit_error,
            rate_limit_error,
            mock_response,
        ]

        with patch("ai_reviewer.llm_client.AsyncOpenAI", return_value=mock_openai_client):
            client = LLMClient(
                base_url="https://api.example.com/v1",
                api_key="test-key",
                model="gpt-4",
            )

            response = await client.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert response.content == "Success after retry"
            # Verify it was called 3 times (2 failures + 1 success)
            assert mock_openai_client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_503_error(self, mock_openai_client: MagicMock) -> None:
        """Test retry logic on 503 Service Unavailable error."""
        from openai import APIStatusError

        # Create 503 error using APIStatusError which is the correct exception type
        server_error = APIStatusError(
            message="Service unavailable",
            response=MagicMock(status_code=503, request=MagicMock()),
            body=None,
        )

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Recovered"),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=50,
            completion_tokens=30,
            prompt_tokens_details=None,
        )

        mock_openai_client.chat.completions.create.side_effect = [
            server_error,
            mock_response,
        ]

        with patch("ai_reviewer.llm_client.AsyncOpenAI", return_value=mock_openai_client):
            client = LLMClient(
                base_url="https://api.example.com/v1",
                api_key="test-key",
                model="gpt-4",
            )

            response = await client.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert response.content == "Recovered"

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, mock_openai_client: MagicMock) -> None:
        """Test RetryExhaustedError when all retries fail."""
        from openai import RateLimitError

        # All calls fail with rate limit
        rate_limit_error = RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )

        mock_openai_client.chat.completions.create.side_effect = rate_limit_error

        with patch("ai_reviewer.llm_client.AsyncOpenAI", return_value=mock_openai_client):
            client = LLMClient(
                base_url="https://api.example.com/v1",
                api_key="test-key",
                model="gpt-4",
            )

            with pytest.raises(RetryExhaustedError) as exc_info:
                await client.chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                )

            assert "Failed after" in str(exc_info.value)
            assert exc_info.value.last_error is rate_limit_error
            # Should have tried MAX_RETRIES times
            assert mock_openai_client.chat.completions.create.call_count == MAX_RETRIES

    @pytest.mark.asyncio
    async def test_non_retryable_error(self, mock_openai_client: MagicMock) -> None:
        """Test that non-retryable errors are raised immediately."""
        from openai import APIStatusError

        # 400 Bad Request - not retryable
        bad_request_error = APIStatusError(
            message="Bad request",
            response=MagicMock(status_code=400, request=MagicMock()),
            body=None,
        )

        mock_openai_client.chat.completions.create.side_effect = bad_request_error

        with patch("ai_reviewer.llm_client.AsyncOpenAI", return_value=mock_openai_client):
            client = LLMClient(
                base_url="https://api.example.com/v1",
                api_key="test-key",
                model="gpt-4",
            )

            with pytest.raises(LLMClientError) as exc_info:
                await client.chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                )

            assert "API error: 400" in str(exc_info.value)
            # Should only be called once (no retries)
            assert mock_openai_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_budget_exceeded_raises_immediately(self, mock_openai_client: MagicMock) -> None:
        """Test that BudgetExceededError is not caught by retry logic."""
        from ai_reviewer.token_tracker import BudgetExceededError

        # Set up mock to succeed
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Response"),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=1000000,  # Large number to trigger budget exceeded
            completion_tokens=500000,
            prompt_tokens_details=None,
        )
        mock_openai_client.chat.completions.create.return_value = mock_response

        with patch("ai_reviewer.llm_client.AsyncOpenAI", return_value=mock_openai_client):
            client = LLMClient(
                base_url="https://api.example.com/v1",
                api_key="test-key",
                model="gpt-4",
                budget_limit_usd=0.001,  # Very small budget
            )

            with pytest.raises(BudgetExceededError):
                await client.chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                )


class TestLLMClientContextManager:
    """Tests for LLMClient async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Test using LLMClient as async context manager."""
        mock_openai_client = MagicMock()
        mock_openai_client.close = AsyncMock()

        with patch("ai_reviewer.llm_client.AsyncOpenAI", return_value=mock_openai_client):
            async with LLMClient(
                base_url="https://api.example.com/v1",
                api_key="test-key",
                model="gpt-4",
            ) as client:
                assert client is not None

            # Verify close was called
            mock_openai_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_manual_close(self) -> None:
        """Test manual close method."""
        mock_openai_client = MagicMock()
        mock_openai_client.close = AsyncMock()

        with patch("ai_reviewer.llm_client.AsyncOpenAI", return_value=mock_openai_client):
            client = LLMClient(
                base_url="https://api.example.com/v1",
                api_key="test-key",
                model="gpt-4",
            )
            await client.close()

            # Verify close was called
            mock_openai_client.close.assert_called_once()


class TestLLMClientUsageSummary:
    """Tests for get_usage_summary method."""

    @pytest.mark.asyncio
    async def test_get_usage_summary(self) -> None:
        """Test getting usage summary after requests."""
        mock_openai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Response"),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=50,
            prompt_tokens_details=None,
        )
        # Make create an async function that returns the mock response
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("ai_reviewer.llm_client.AsyncOpenAI", return_value=mock_openai_client):
            client = LLMClient(
                base_url="https://api.example.com/v1",
                api_key="test-key",
                model="gpt-4",
                budget_limit_usd=10.0,
            )

            # Make a request
            await client.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
            )

            summary = client.get_usage_summary()

            assert summary["model"] == "gpt-4"
            assert summary["calls"] == 1
            assert summary["input_tokens"] == 100
            assert summary["output_tokens"] == 50
            assert summary["budget_limit_usd"] == 10.0
