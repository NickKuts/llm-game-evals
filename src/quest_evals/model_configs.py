"""
Model configurations for quest evaluation benchmarks.

Simple lookup of model IDs and their optimal parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReasoningField(Enum):
    """
    Where to extract reasoning tokens from in the API response.

    Based on OpenRouter docs: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens

    - NONE: Model doesn't support/return reasoning tokens
    - REASONING: Simple string field at message.reasoning (legacy format)
    - REASONING_DETAILS: Structured array at message.reasoning_details (preferred format)
      Contains objects with types: reasoning.text, reasoning.summary, reasoning.encrypted
    """

    NONE = "none"  # No reasoning support
    REASONING = "reasoning"  # message.reasoning string field
    REASONING_DETAILS = "reasoning_details"  # message.reasoning_details array


@dataclass
class ModelConfig:
    """Configuration for a benchmark model."""

    model_id: str
    display_name: str
    extra_body: dict[str, Any] = field(default_factory=dict)
    reasoning_field: ReasoningField = ReasoningField.NONE
    reasoning_may_be_hidden: bool = False  # True for models that do reasoning internally


# =============================================================================
# MODEL CONFIGURATIONS
#
# reasoning_field values based on OpenRouter docs:
# - REASONING_DETAILS: Anthropic (Claude 3.7+), OpenAI (o1/o3/GPT-5), Gemini, xAI
# - REASONING: DeepSeek, some open-source models (legacy format)
# - NONE: Models without reasoning support or with reasoning disabled
# =============================================================================

MODELS: dict[str, ModelConfig] = {
    # Anthropic
    "claude-opus-4.5": ModelConfig(
        model_id="anthropic/claude-opus-4.5",
        display_name="Claude 4.5 Opus",
        extra_body={"reasoning": {"max_tokens": 8000}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    "claude-sonnet-4.5": ModelConfig(
        model_id="anthropic/claude-sonnet-4.5",
        display_name="Claude 4.5 Sonnet",
        extra_body={"reasoning": {"max_tokens": 8000}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    "claude-opus-4": ModelConfig(
        model_id="anthropic/claude-opus-4",
        display_name="Claude 4 Opus",
        extra_body={"reasoning": {"max_tokens": 8000}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    "claude-sonnet-4": ModelConfig(
        model_id="anthropic/claude-sonnet-4",
        display_name="Claude 4 Sonnet",
        extra_body={"reasoning": {"max_tokens": 8000}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    "claude-3.7-sonnet": ModelConfig(
        model_id="anthropic/claude-3.7-sonnet",
        display_name="Claude 3.7 Sonnet",
        extra_body={"reasoning": {"max_tokens": 8000}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    "claude-haiku-4.5": ModelConfig(
        model_id="anthropic/claude-haiku-4.5",
        display_name="Claude 4.5 Haiku",
    ),
    # OpenAI
    "gpt-5.2-high": ModelConfig(
        model_id="openai/gpt-5.2",
        display_name="GPT-5.2 (high reasoning)",
        extra_body={"reasoning": {"effort": "high"}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    "gpt-5.2": ModelConfig(
        model_id="openai/gpt-5.2",
        display_name="GPT-5.2",
    ),
    "gpt-5.1-codex": ModelConfig(
        model_id="openai/gpt-5.1-codex",
        display_name="GPT-5.1-Codex",
        extra_body={"reasoning": {"effort": "medium"}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    "gpt-5": ModelConfig(
        model_id="openai/gpt-5",
        display_name="GPT-5",
        extra_body={"reasoning": {"effort": "medium"}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    "gpt-5-mini": ModelConfig(
        model_id="openai/gpt-5-mini",
        display_name="GPT-5 Mini",
        extra_body={"reasoning": {"effort": "medium"}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    "gpt-5-nano": ModelConfig(
        model_id="openai/gpt-5-nano",
        display_name="GPT-5 Nano",
        extra_body={"reasoning": {"effort": "medium"}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    "o3": ModelConfig(
        model_id="openai/o3",
        display_name="o3",
        extra_body={"reasoning": {"effort": "medium"}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
        reasoning_may_be_hidden=True,
    ),
    "o4-mini": ModelConfig(
        model_id="openai/o4-mini",
        display_name="o4-mini",
        extra_body={"reasoning": {"effort": "medium"}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    # Google
    "gemini-3-pro": ModelConfig(
        model_id="google/gemini-3-pro-preview",
        display_name="Gemini 3 Pro Preview",
        extra_body={"reasoning": {"effort": "medium"}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    "gemini-3-flash": ModelConfig(
        model_id="google/gemini-3-flash-preview",
        display_name="Gemini 3 Flash Preview",
    ),
    "gemini-2.5-pro": ModelConfig(
        model_id="google/gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        extra_body={"reasoning": {"max_tokens": 8000}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    "gemini-2.5-flash": ModelConfig(
        model_id="google/gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        extra_body={"reasoning": {"max_tokens": 4000}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    # DeepSeek
    "deepseek-v3.2": ModelConfig(
        model_id="deepseek/deepseek-v3.2",
        display_name="DeepSeek V3.2 Reasoner",
        extra_body={"reasoning": {"enabled": True}},
        reasoning_field=ReasoningField.REASONING,
    ),
    # Other providers
    "kimi-k2-thinking": ModelConfig(
        model_id="moonshotai/kimi-k2-thinking",
        display_name="Kimi K2 Thinking",
        extra_body={"reasoning": {"enabled": True}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    "minimax-m2": ModelConfig(
        model_id="minimax/minimax-m2",
        display_name="MiniMax M2",
        extra_body={"reasoning": {"enabled": True}},
        reasoning_field=ReasoningField.REASONING_DETAILS,
    ),
    "devstral-small": ModelConfig(
        model_id="mistralai/devstral-small-2512",
        display_name="Devstral Small",
    ),
    # Free tier
    "devstral:free": ModelConfig(
        model_id="mistralai/devstral-2512:free",
        display_name="Devstral 2 (free)",
    ),
    "mimo:free": ModelConfig(
        model_id="xiaomi/mimo-v2-flash:free",
        display_name="MiMo-V2-Flash (free)",
        extra_body={"reasoning": {"enabled": False}},
    ),
    "kat-coder:free": ModelConfig(
        model_id="kwaipilot/kat-coder-pro:free",
        display_name="KAT-Coder-Pro (free)",
    ),
    "olmo-think:free": ModelConfig(
        model_id="allenai/olmo-3.1-32b-think:free",
        display_name="OLMo 3.1 32B Think (free)",
        extra_body={"reasoning": {"enabled": True}},
        reasoning_field=ReasoningField.REASONING,
    ),
    "nemotron:free": ModelConfig(
        model_id="nvidia/nemotron-3-nano-30b-a3b:free",
        display_name="Nemotron 3 Nano (free)",
    ),
}


def get_config(name: str) -> ModelConfig:
    """Get model config by shortcut name."""
    if name in MODELS:
        return MODELS[name]

    # Try to find by model_id
    for config in MODELS.values():
        if config.model_id == name:
            return config

    raise ValueError(f"Unknown model: {name}. Available: {', '.join(MODELS.keys())}")


def list_models() -> None:
    """Print available models."""
    print(f"{'Shortcut':<20} {'Model ID':<45} {'Reasoning':<10}")
    print("-" * 75)
    for name, cfg in MODELS.items():
        reasoning = cfg.reasoning_field.value if cfg.reasoning_field != ReasoningField.NONE else "-"
        print(f"{name:<20} {cfg.model_id:<45} {reasoning:<10}")


if __name__ == "__main__":
    list_models()
