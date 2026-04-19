"""Token tracking and budget management for AI Reviewer.

This module provides:
- Token counting (input, output, cached)
- Cost calculation based on model pricing
- Budget limit enforcement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


# Default model pricing (per 1K tokens)
# Format: {model_pattern: (input_price_usd, output_price_usd)}
# Patterns are matched as substrings in model names
DEFAULT_MODEL_PRICES: dict[str, tuple[float, float]] = {
    # OpenAI models
    "gpt-4o": (0.005, 0.015),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    # Anthropic models
    "claude-3-opus": (0.015, 0.075),
    "claude-3-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
    # Qwen models (OpenRouter)
    "qwen/qwen3.5-flash": (0.0002, 0.0006),
    "qwen/qwen3.5": (0.0004, 0.0012),
    "qwen/": (0.0002, 0.0006),  # Default for qwen models
    # Mistral models
    "mistral/": (0.0002, 0.0006),
    # Llama models
    "meta-llama/": (0.0002, 0.0006),
    # Default fallback
    "default": (0.001, 0.003),
}


@dataclass
class TokenUsage:
    """Represents token usage from a single API call.

    Attributes:
        input_tokens: Number of tokens in the request.
        output_tokens: Number of tokens in the response.
        cached_tokens: Number of tokens served from cache (if supported).
        total_tokens: Total tokens consumed (input + output).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Add two TokenUsage objects together."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass
class ModelPricing:
    """Pricing information for a model.

    Attributes:
        input_price_usd: Price per 1K input tokens in USD.
        output_price_usd: Price per 1K output tokens in USD.
    """

    input_price_usd: float
    output_price_usd: float


class TokenTracker:
    """Tracks token usage and costs across multiple API calls.

    This class maintains cumulative token counts and calculates costs
    based on model-specific pricing. It also enforces budget limits.

    Attributes:
        model_name: Name of the model being used.
        budget_limit_usd: Maximum allowed spend in USD.
        pricing: Model pricing information.
        total_usage: Cumulative token usage.
        calls_count: Number of API calls made.
    """

    def __init__(
        self,
        model_name: str,
        budget_limit_usd: float | None = None,
        custom_pricing: ModelPricing | None = None,
    ) -> None:
        """Initialize the token tracker.

        Args:
            model_name: Name of the model for pricing lookup.
            budget_limit_usd: Optional budget limit in USD.
            custom_pricing: Optional custom pricing override.
        """
        self.model_name = model_name
        self.budget_limit_usd = budget_limit_usd
        self.pricing = custom_pricing or self._get_model_pricing(model_name)
        self.total_usage = TokenUsage()
        self.calls_count = 0
        self._call_history: list[TokenUsage] = []

    @staticmethod
    def _get_model_pricing(model_name: str) -> ModelPricing:
        """Get pricing for a model by name matching.

        Args:
            model_name: Model identifier string.

        Returns:
            ModelPricing with input/output prices.
        """
        model_lower = model_name.lower()

        # Try to match specific patterns first (longer/more specific first)
        sorted_patterns = sorted(
            DEFAULT_MODEL_PRICES.keys(),
            key=lambda x: len(x),
            reverse=True,
        )

        for pattern in sorted_patterns:
            if pattern == "default":
                continue
            if pattern in model_lower:
                prices = DEFAULT_MODEL_PRICES[pattern]
                return ModelPricing(
                    input_price_usd=prices[0],
                    output_price_usd=prices[1],
                )

        # Fallback to default pricing
        default_prices = DEFAULT_MODEL_PRICES["default"]
        return ModelPricing(
            input_price_usd=default_prices[0],
            output_price_usd=default_prices[1],
        )

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> TokenUsage:
        """Record token usage from an API call.

        Args:
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens generated.
            cached_tokens: Number of tokens served from cache.

        Returns:
            TokenUsage object for this call.

        Raises:
            BudgetExceededError: If recording this usage exceeds budget.
        """
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )

        self.total_usage += usage
        self.calls_count += 1
        self._call_history.append(usage)

        # Check budget limit
        if self.budget_limit_usd is not None:
            current_cost = self.estimated_cost_usd
            if current_cost > self.budget_limit_usd * 0.8:  # 80% threshold
                logger.warning(
                    "Budget warning: %.4f USD spent (%.1f%% of %.4f USD limit)",
                    current_cost,
                    (current_cost / self.budget_limit_usd) * 100,
                    self.budget_limit_usd,
                )
            if current_cost > self.budget_limit_usd:
                raise BudgetExceededError(
                    f"Budget exceeded: {current_cost:.4f} USD > {self.budget_limit_usd:.4f} USD limit",
                    current_cost=current_cost,
                    limit=self.budget_limit_usd,
                )

        return usage

    @property
    def estimated_cost_usd(self) -> float:
        """Calculate estimated cost in USD based on token usage.

        Returns:
            Estimated cost in USD.
        """
        input_cost = (self.total_usage.input_tokens / 1000) * self.pricing.input_price_usd
        output_cost = (self.total_usage.output_tokens / 1000) * self.pricing.output_price_usd
        return input_cost + output_cost

    def get_summary(self) -> dict[str, object]:
        """Get a summary of token usage and costs.

        Returns:
            Dictionary with usage statistics.
        """
        return {
            "model": self.model_name,
            "calls": self.calls_count,
            "input_tokens": self.total_usage.input_tokens,
            "output_tokens": self.total_usage.output_tokens,
            "cached_tokens": self.total_usage.cached_tokens,
            "total_tokens": self.total_usage.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "budget_limit_usd": self.budget_limit_usd,
            "budget_remaining_usd": (
                self.budget_limit_usd - self.estimated_cost_usd
                if self.budget_limit_usd
                else None
            ),
        }

    def log_metrics(self, usage: TokenUsage, include_details: bool = False) -> None:
        """Log metrics for an API call.

        Note: Does NOT log prompt/response content for security.

        Args:
            usage: TokenUsage from the call.
            include_details: Whether to include detailed breakdown.
        """
        logger.info(
            "API call: input=%d, output=%d, cached=%d, total=%d, cost=%.6f USD",
            usage.input_tokens,
            usage.output_tokens,
            usage.cached_tokens,
            usage.total_tokens,
            self._calculate_call_cost(usage),
        )

        if include_details:
            logger.debug("Full usage summary: %s", self.get_summary())

    def _calculate_call_cost(self, usage: TokenUsage) -> float:
        """Calculate cost for a single call.

        Args:
            usage: TokenUsage for the call.

        Returns:
            Cost in USD.
        """
        input_cost = (usage.input_tokens / 1000) * self.pricing.input_price_usd
        output_cost = (usage.output_tokens / 1000) * self.pricing.output_price_usd
        return input_cost + output_cost


class BudgetExceededError(Exception):
    """Raised when token usage exceeds the budget limit.

    Attributes:
        message: Error message.
        current_cost: Current accumulated cost in USD.
        limit: Budget limit in USD.
    """

    def __init__(
        self,
        message: str,
        current_cost: float,
        limit: float,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Error message.
            current_cost: Current accumulated cost.
            limit: Budget limit that was exceeded.
        """
        super().__init__(message)
        self.current_cost = current_cost
        self.limit = limit
