"""Tests for token_tracker module."""

import pytest

from ai_reviewer.token_tracker import (
    TokenTracker,
    TokenUsage,
    ModelPricing,
    BudgetExceededError,
    DEFAULT_MODEL_PRICES,
)


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_default_values(self) -> None:
        """Test default initialization values."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cached_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_values(self) -> None:
        """Test initialization with custom values."""
        usage = TokenUsage(input_tokens=100, output_tokens=50, cached_tokens=10)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 10
        assert usage.total_tokens == 150

    def test_addition(self) -> None:
        """Test adding two TokenUsage objects."""
        usage1 = TokenUsage(input_tokens=100, output_tokens=50, cached_tokens=10)
        usage2 = TokenUsage(input_tokens=200, output_tokens=75, cached_tokens=20)
        result = usage1 + usage2
        assert result.input_tokens == 300
        assert result.output_tokens == 125
        assert result.cached_tokens == 30
        assert result.total_tokens == 425


class TestModelPricing:
    """Tests for ModelPricing dataclass."""

    def test_creation(self) -> None:
        """Test ModelPricing creation."""
        pricing = ModelPricing(input_price_usd=0.001, output_price_usd=0.003)
        assert pricing.input_price_usd == 0.001
        assert pricing.output_price_usd == 0.003


class TestTokenTracker:
    """Tests for TokenTracker class."""

    def test_initialization(self) -> None:
        """Test TokenTracker initialization."""
        tracker = TokenTracker(model_name="gpt-4")
        assert tracker.model_name == "gpt-4"
        assert tracker.budget_limit_usd is None
        assert tracker.calls_count == 0
        assert tracker.total_usage.input_tokens == 0
        assert tracker.total_usage.output_tokens == 0

    def test_initialization_with_budget(self) -> None:
        """Test TokenTracker initialization with budget limit."""
        tracker = TokenTracker(model_name="gpt-4", budget_limit_usd=10.0)
        assert tracker.budget_limit_usd == 10.0

    def test_initialization_with_custom_pricing(self) -> None:
        """Test TokenTracker initialization with custom pricing."""
        custom_pricing = ModelPricing(input_price_usd=0.005, output_price_usd=0.015)
        tracker = TokenTracker(
            model_name="custom-model",
            custom_pricing=custom_pricing,
        )
        assert tracker.pricing.input_price_usd == 0.005
        assert tracker.pricing.output_price_usd == 0.015

    def test_get_model_pricing_gpt4o(self) -> None:
        """Test pricing lookup for GPT-4o."""
        pricing = TokenTracker._get_model_pricing("gpt-4o")
        assert pricing.input_price_usd == 0.005
        assert pricing.output_price_usd == 0.015

    def test_get_model_pricing_qwen(self) -> None:
        """Test pricing lookup for Qwen models."""
        pricing = TokenTracker._get_model_pricing("qwen/qwen3.5-flash")
        assert pricing.input_price_usd == 0.0002
        assert pricing.output_price_usd == 0.0006

    def test_get_model_pricing_default(self) -> None:
        """Test pricing lookup falls back to default."""
        pricing = TokenTracker._get_model_pricing("unknown-model-xyz")
        assert pricing.input_price_usd == 0.001
        assert pricing.output_price_usd == 0.003

    def test_record_usage(self) -> None:
        """Test recording token usage."""
        tracker = TokenTracker(model_name="gpt-4")
        usage = tracker.record_usage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert tracker.calls_count == 1
        assert tracker.total_usage.input_tokens == 100
        assert tracker.total_usage.output_tokens == 50

    def test_record_usage_multiple_calls(self) -> None:
        """Test recording multiple usages accumulates correctly."""
        tracker = TokenTracker(model_name="gpt-4")
        tracker.record_usage(input_tokens=100, output_tokens=50)
        tracker.record_usage(input_tokens=200, output_tokens=75)
        assert tracker.calls_count == 2
        assert tracker.total_usage.input_tokens == 300
        assert tracker.total_usage.output_tokens == 125

    def test_estimated_cost_usd(self) -> None:
        """Test cost calculation."""
        # Use a model with known pricing
        tracker = TokenTracker(model_name="gpt-4o")
        tracker.record_usage(input_tokens=1000, output_tokens=500)
        # gpt-4o: $0.005/1K input, $0.015/1K output
        expected_cost = (1000 / 1000) * 0.005 + (500 / 1000) * 0.015
        expected_cost = 0.005 + 0.0075
        assert abs(tracker.estimated_cost_usd - expected_cost) < 0.0001

    def test_get_summary(self) -> None:
        """Test getting usage summary."""
        tracker = TokenTracker(model_name="gpt-4", budget_limit_usd=10.0)
        tracker.record_usage(input_tokens=1000, output_tokens=500, cached_tokens=100)
        summary = tracker.get_summary()
        assert summary["model"] == "gpt-4"
        assert summary["calls"] == 1
        assert summary["input_tokens"] == 1000
        assert summary["output_tokens"] == 500
        assert summary["cached_tokens"] == 100
        assert summary["total_tokens"] == 1500
        assert "estimated_cost_usd" in summary
        assert summary["budget_limit_usd"] == 10.0
        assert summary["budget_remaining_usd"] is not None

    def test_budget_warning_at_80_percent(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test budget warning is logged at 80% threshold."""
        import logging

        # Set up tracker with small budget
        tracker = TokenTracker(model_name="gpt-4o", budget_limit_usd=0.01)
        
        # Configure logger for the token_tracker module
        logger = logging.getLogger("ai_reviewer.token_tracker")
        logger.setLevel(logging.WARNING)
        logger.addHandler(caplog.handler)

        with caplog.at_level(logging.WARNING):
            # Record usage that exceeds 80% of budget
            # 80% of $0.01 = $0.008
            # At gpt-4o prices ($0.005/1K input): need >1600 input tokens to exceed 80%
            # Using 2000 tokens = $0.01 which is 100% of budget
            tracker.record_usage(input_tokens=2000, output_tokens=0)

        # Check warning was logged
        assert "Budget warning" in caplog.text

    def test_budget_exceeded_error(self) -> None:
        """Test BudgetExceededError is raised when limit exceeded."""
        tracker = TokenTracker(model_name="gpt-4o", budget_limit_usd=0.001)
        
        # First usage within budget
        tracker.record_usage(input_tokens=100, output_tokens=0)
        
        # Second usage that exceeds budget
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.record_usage(input_tokens=10000, output_tokens=0)
        
        assert "Budget exceeded" in str(exc_info.value)
        assert exc_info.value.current_cost > 0.001
        assert exc_info.value.limit == 0.001

    def test_record_usage_with_cached_tokens(self) -> None:
        """Test recording usage with cached tokens."""
        tracker = TokenTracker(model_name="gpt-4")
        usage = tracker.record_usage(input_tokens=100, output_tokens=50, cached_tokens=25)
        assert usage.cached_tokens == 25
        assert tracker.total_usage.cached_tokens == 25


class TestBudgetExceededError:
    """Tests for BudgetExceededError exception."""

    def test_exception_attributes(self) -> None:
        """Test exception stores correct attributes."""
        error = BudgetExceededError(
            message="Budget exceeded",
            current_cost=1.5,
            limit=1.0,
        )
        assert str(error) == "Budget exceeded"
        assert error.current_cost == 1.5
        assert error.limit == 1.0
