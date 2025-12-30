"""
Integration tests for OpenRouter model configurations.

These tests verify that:
1. Each model config uses valid OpenRouter API parameters
2. Reasoning tokens are correctly requested and returned
3. The response format matches expectations

Run with: pytest tests/test_model_configs_integration.py -v
Run single model: pytest tests/test_model_configs_integration.py -v -k "claude_sonnet_4"

Requires OPENROUTER_API_KEY environment variable.
"""

import os
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from quest_evals.llm import extract_reasoning
from quest_evals.model_configs import MODELS, ModelConfig, ReasoningField

# Marker for tests requiring API key
requires_api_key = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set"
)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
TEST_PROMPT = "What is 2+2? Reply with just the number."
TEST_SYSTEM = "You are a helpful assistant. Be concise."


# =============================================================================
# MODEL LISTS FROM SOURCE
# Uses the actual model configs from model_configs.py
# =============================================================================

# Get all models as (shortcut, config) pairs
ALL_MODELS = [(name, cfg) for name, cfg in MODELS.items()]

# Models that expect reasoning tokens
REASONING_MODELS = [
    (name, cfg) for name, cfg in MODELS.items() if cfg.reasoning_field != ReasoningField.NONE
]

# Models that don't expect reasoning tokens
NON_REASONING_MODELS = [
    (name, cfg) for name, cfg in MODELS.items() if cfg.reasoning_field == ReasoningField.NONE
]


# =============================================================================
# API TEST HELPER
# =============================================================================


@dataclass
class APITestResult:
    """Result of a model integration test."""

    success: bool
    model_id: str
    response_text: str | None = None
    reasoning_content: str | None = None
    reasoning_field_used: ReasoningField | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    raw_message: dict | None = None


def call_openrouter(
    model_id: str,
    extra_body: dict[str, Any],
    reasoning_field: ReasoningField,
    timeout: float = 60.0,
) -> APITestResult:
    """Make a test call to OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")

    payload = {
        "model": model_id,
        "max_tokens": 500,
        "messages": [
            {"role": "system", "content": TEST_SYSTEM},
            {"role": "user", "content": TEST_PROMPT},
        ],
        **extra_body,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/quest-evals",
        "X-Title": "Quest Evals Integration Test",
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                OPENROUTER_API_URL,
                json=payload,
                headers=headers,
            )

            # Check for HTTP errors
            if response.status_code != 200:
                return APITestResult(
                    success=False,
                    model_id=model_id,
                    error=f"HTTP {response.status_code}: {response.text[:500]}",
                )

            data = response.json()

            # Check for API errors
            if "error" in data:
                return APITestResult(
                    success=False,
                    model_id=model_id,
                    error=f"API Error: {data['error']}",
                )

            # Extract response
            message = data.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or ""

            # Use the same extraction logic as the app
            reasoning = extract_reasoning(message, reasoning_field)

            # Token usage
            usage = data.get("usage", {})

            return APITestResult(
                success=True,
                model_id=model_id,
                response_text=content,
                reasoning_content=reasoning or None,
                reasoning_field_used=reasoning_field,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                raw_message=message,
            )

    except httpx.TimeoutException:
        return APITestResult(
            success=False,
            model_id=model_id,
            error=f"Timeout after {timeout}s",
        )
    except Exception as e:
        return APITestResult(
            success=False,
            model_id=model_id,
            error=f"Exception: {type(e).__name__}: {e}",
        )


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def api_key():
    """Ensure API key is available."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return key


# =============================================================================
# TESTS
# =============================================================================


@requires_api_key
class TestModelConfigs:
    """Test each model configuration."""

    @pytest.mark.parametrize(
        "name,config",
        ALL_MODELS,
        ids=[name for name, _ in ALL_MODELS],
    )
    def test_model_api_call(self, name: str, config: ModelConfig):
        """Test that the model responds correctly to a simple prompt."""
        result = call_openrouter(
            model_id=config.model_id,
            extra_body=config.extra_body,
            reasoning_field=config.reasoning_field,
            timeout=90.0,
        )

        # Basic assertions
        assert result.success, f"API call failed: {result.error}"
        assert result.response_text, f"No response text from {config.model_id}"

        # Check if response is sensible (should contain "4" for 2+2)
        assert "4" in result.response_text, (
            f"Expected '4' in response, got: {result.response_text[:200]}"
        )

    @pytest.mark.parametrize(
        "name,config",
        REASONING_MODELS,
        ids=[name for name, _ in REASONING_MODELS],
    )
    def test_reasoning_tokens_returned(self, name: str, config: ModelConfig):
        """Test that models configured for reasoning actually return reasoning tokens."""
        result = call_openrouter(
            model_id=config.model_id,
            extra_body=config.extra_body,
            reasoning_field=config.reasoning_field,
            timeout=120.0,
        )

        assert result.success, f"API call failed: {result.error}"

        # Check reasoning was returned
        has_reasoning = bool(result.reasoning_content)

        # Some models (like o3) may do reasoning internally but not return it
        if config.reasoning_may_be_hidden and not has_reasoning:
            pytest.skip(f"Model {name} did reasoning but didn't return tokens (expected per docs)")
        else:
            assert has_reasoning, (
                f"Expected reasoning tokens from {name} "
                f"(field: {config.reasoning_field.value}), got none. "
                f"Response: {result.response_text[:200]}"
            )

    @pytest.mark.parametrize(
        "name,config",
        NON_REASONING_MODELS,
        ids=[name for name, _ in NON_REASONING_MODELS],
    )
    def test_no_unexpected_reasoning(self, name: str, config: ModelConfig):
        """Test that models without reasoning config don't return reasoning tokens."""
        result = call_openrouter(
            model_id=config.model_id,
            extra_body=config.extra_body,
            reasoning_field=config.reasoning_field,
            timeout=60.0,
        )

        assert result.success, f"API call failed: {result.error}"

        # With NONE field, extract_reasoning should return empty string
        assert not result.reasoning_content, (
            f"Model {name} unexpectedly returned reasoning: {result.reasoning_content[:100]}"
        )


class TestReasoningParameterFormats:
    """Test that reasoning parameter formats are valid per OpenRouter docs."""

    def test_effort_param_valid_values(self):
        """Test that effort-based models use valid effort levels."""
        valid_efforts = {"xhigh", "high", "medium", "low", "minimal", "none"}

        for name, config in MODELS.items():
            reasoning = config.extra_body.get("reasoning", {})
            if "effort" in reasoning:
                effort = reasoning["effort"]
                assert effort in valid_efforts, (
                    f"{name}: invalid effort '{effort}', must be one of {valid_efforts}"
                )

    def test_max_tokens_param_valid_range(self):
        """Test that max_tokens reasoning is within valid range."""
        # Per OpenRouter docs: Anthropic min=1024, max=32000
        MIN_REASONING_TOKENS = 1024
        MAX_REASONING_TOKENS = 32000

        for name, config in MODELS.items():
            reasoning = config.extra_body.get("reasoning", {})
            if "max_tokens" in reasoning:
                max_tokens = reasoning["max_tokens"]
                assert MIN_REASONING_TOKENS <= max_tokens <= MAX_REASONING_TOKENS, (
                    f"{name}: max_tokens {max_tokens} outside valid range "
                    f"[{MIN_REASONING_TOKENS}, {MAX_REASONING_TOKENS}]"
                )

    def test_enabled_param_is_boolean(self):
        """Test that enabled reasoning param is boolean."""
        for name, config in MODELS.items():
            reasoning = config.extra_body.get("reasoning", {})
            if "enabled" in reasoning:
                enabled = reasoning["enabled"]
                assert isinstance(enabled, bool), (
                    f"{name}: 'enabled' should be bool, got {type(enabled)}"
                )

    def test_verbosity_param_valid_values(self):
        """Test that verbosity param uses valid values."""
        valid_verbosity = {"low", "medium", "high"}

        for name, config in MODELS.items():
            verbosity = config.extra_body.get("verbosity")
            if verbosity is not None:
                assert verbosity in valid_verbosity, (
                    f"{name}: invalid verbosity '{verbosity}', must be one of {valid_verbosity}"
                )

    def test_reasoning_field_matches_extra_body(self):
        """Test that reasoning_field is consistent with extra_body params."""
        for name, config in MODELS.items():
            has_reasoning_params = bool(config.extra_body.get("reasoning", {}))
            expects_reasoning = config.reasoning_field != ReasoningField.NONE

            # If we have reasoning params, we should expect reasoning output
            # (unless explicitly disabled)
            reasoning = config.extra_body.get("reasoning", {})
            explicitly_disabled = reasoning.get("enabled") is False

            if has_reasoning_params and not explicitly_disabled:
                assert expects_reasoning, (
                    f"{name}: has reasoning params but reasoning_field is NONE"
                )


# =============================================================================
# QUICK SMOKE TEST (run with: pytest -k "smoke")
# =============================================================================


@requires_api_key
class TestSmoke:
    """Quick smoke test with a single cheap model."""

    @pytest.mark.smoke
    def test_openrouter_connectivity(self):
        """Test basic OpenRouter API connectivity."""
        result = call_openrouter(
            model_id="mistralai/devstral-2512:free",
            extra_body={},
            reasoning_field=ReasoningField.NONE,
            timeout=30.0,
        )

        assert result.success, f"Smoke test failed: {result.error}"
        assert result.response_text, "No response from smoke test"


# =============================================================================
# APP LLM EXTRACTION VERIFICATION
# Verifies quest_evals.llm extracts reasoning the same way as the test
# =============================================================================


class TestExtractReasoning:
    """Test the extract_reasoning function directly."""

    def test_extract_none(self):
        """Test NONE field returns empty string."""
        msg = {"content": "4", "reasoning": "some reasoning"}
        assert extract_reasoning(msg, ReasoningField.NONE) == ""

    def test_extract_reasoning_field(self):
        """Test REASONING field extracts from message.reasoning."""
        msg = {"content": "4", "reasoning": "Let me think... 2+2=4"}
        assert extract_reasoning(msg, ReasoningField.REASONING) == "Let me think... 2+2=4"

    def test_extract_reasoning_content_field(self):
        """Test REASONING field also checks reasoning_content."""
        msg = {"content": "4", "reasoning_content": "Calculation: 2+2=4"}
        assert extract_reasoning(msg, ReasoningField.REASONING) == "Calculation: 2+2=4"

    def test_extract_reasoning_details(self):
        """Test REASONING_DETAILS field extracts from array."""
        msg = {
            "content": "4",
            "reasoning_details": [
                {"type": "reasoning.text", "text": "Step 1: Add 2+2"},
                {"type": "reasoning.text", "text": "Step 2: Result is 4"},
            ],
        }
        result = extract_reasoning(msg, ReasoningField.REASONING_DETAILS)
        assert "Step 1" in result
        assert "Step 2" in result

    def test_extract_reasoning_summary(self):
        """Test REASONING_DETAILS extracts summary type."""
        msg = {
            "content": "4",
            "reasoning_details": [
                {"type": "reasoning.summary", "summary": "Simple arithmetic"},
            ],
        }
        assert extract_reasoning(msg, ReasoningField.REASONING_DETAILS) == "Simple arithmetic"

    def test_extract_reasoning_encrypted_skipped(self):
        """Test REASONING_DETAILS skips encrypted type."""
        msg = {
            "content": "4",
            "reasoning_details": [
                {"type": "reasoning.encrypted", "data": "xxx"},
                {"type": "reasoning.text", "text": "Visible reasoning"},
            ],
        }
        result = extract_reasoning(msg, ReasoningField.REASONING_DETAILS)
        assert "Visible reasoning" in result
        assert "xxx" not in result

    def test_extract_empty_details(self):
        """Test REASONING_DETAILS with empty array returns empty string."""
        msg = {"content": "4", "reasoning_details": []}
        assert extract_reasoning(msg, ReasoningField.REASONING_DETAILS) == ""

    def test_extract_missing_field(self):
        """Test extraction when expected field is missing."""
        msg = {"content": "4"}
        assert extract_reasoning(msg, ReasoningField.REASONING) == ""
        assert extract_reasoning(msg, ReasoningField.REASONING_DETAILS) == ""


@requires_api_key
class TestAppLLMExtraction:
    """Verify the app's LLM class extracts reasoning correctly."""

    @pytest.mark.parametrize(
        "name,config",
        REASONING_MODELS[:3],  # Test first 3 reasoning models
        ids=[name for name, _ in REASONING_MODELS[:3]],
    )
    def test_app_extracts_reasoning_same_as_test(self, name: str, config: ModelConfig):
        """
        Verify that quest_evals.llm.OpenRouterLLM extracts reasoning tokens
        the same way our test does.
        """
        from quest_evals.llm import OpenRouterLLM

        # Call via the app's LLM class
        llm = OpenRouterLLM(
            model=config.model_id,
            max_tokens=500,
            extra_body=config.extra_body,
            reasoning_field=config.reasoning_field,
        )

        response = llm.complete(
            prompt=TEST_PROMPT,
            system=TEST_SYSTEM,
        )

        # Also call via our test helper for comparison
        test_result = call_openrouter(
            model_id=config.model_id,
            extra_body=config.extra_body,
            reasoning_field=config.reasoning_field,
            timeout=120.0,
        )

        # Both should succeed
        assert test_result.success, f"Test call failed: {test_result.error}"
        assert response.text, "App returned empty response"

        # If test found reasoning, app should too
        test_has_reasoning = bool(test_result.reasoning_content)
        app_has_reasoning = bool(response.reasoning)

        if test_has_reasoning:
            assert app_has_reasoning, (
                f"Test found reasoning but app didn't extract it.\n"
                f"Model: {name}\n"
                f"Field: {config.reasoning_field.value}\n"
                f"App response.reasoning: {response.reasoning[:100] if response.reasoning else '(empty)'}"
            )


# =============================================================================
# REASONING DETAILS PRESERVATION TESTS
# Per OpenRouter docs: reasoning_details must be passed back unmodified
# https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
# =============================================================================


class TestReasoningDetailsPreservation:
    """Test that reasoning_details are correctly preserved for multi-turn conversations."""

    def test_llm_response_includes_reasoning_details(self):
        """Test that LLMResponse dataclass has reasoning_details field."""
        from quest_evals.llm import LLMResponse

        response = LLMResponse(
            text="4",
            input_tokens=10,
            output_tokens=5,
            reasoning="some reasoning",
            reasoning_details=[{"type": "reasoning.text", "text": "step 1"}],
        )
        assert response.reasoning_details is not None
        assert len(response.reasoning_details) == 1
        assert response.reasoning_details[0]["type"] == "reasoning.text"

    def test_reasoning_details_preserved_in_history(self):
        """Test that conversation history correctly includes reasoning_details."""
        # Simulate what runner.py does
        conversation_history = []

        # User message
        conversation_history.append({"role": "user", "content": "What is 2+2?"})

        # Assistant message with reasoning_details
        reasoning_details = [
            {"type": "reasoning.text", "text": "Let me calculate 2+2"},
            {"type": "reasoning.summary", "summary": "Simple addition"},
        ]
        assistant_msg: dict = {"role": "assistant", "content": "4"}
        if reasoning_details:
            assistant_msg["reasoning_details"] = reasoning_details
        conversation_history.append(assistant_msg)

        # Verify structure
        assert len(conversation_history) == 2
        assert conversation_history[1]["role"] == "assistant"
        assert conversation_history[1]["content"] == "4"
        assert "reasoning_details" in conversation_history[1]
        assert len(conversation_history[1]["reasoning_details"]) == 2
        # Verify it's the exact same object (unmodified)
        assert conversation_history[1]["reasoning_details"] is reasoning_details

    def test_reasoning_details_not_added_when_none(self):
        """Test that reasoning_details is not added when not present."""
        conversation_history = []

        # Assistant message without reasoning_details
        reasoning_details = None
        assistant_msg: dict = {"role": "assistant", "content": "4"}
        if reasoning_details:
            assistant_msg["reasoning_details"] = reasoning_details
        conversation_history.append(assistant_msg)

        # Verify no reasoning_details key
        assert "reasoning_details" not in conversation_history[0]

    def test_history_truncation_preserves_reasoning_details(self):
        """Test that history truncation doesn't lose reasoning_details."""
        conversation_history = []

        # Fill with messages
        for i in range(35):  # 70 messages total
            conversation_history.append({"role": "user", "content": f"Question {i}"})
            assistant_msg: dict = {"role": "assistant", "content": f"Answer {i}"}
            if i % 2 == 0:  # Add reasoning_details to even messages
                assistant_msg["reasoning_details"] = [
                    {"type": "reasoning.text", "text": f"Reasoning {i}"}
                ]
            conversation_history.append(assistant_msg)

        # Truncate like runner.py does
        if len(conversation_history) > 60:
            conversation_history = conversation_history[-60:]

        assert len(conversation_history) == 60

        # Verify reasoning_details still present in truncated history
        messages_with_details = [
            m
            for m in conversation_history
            if m.get("role") == "assistant" and "reasoning_details" in m
        ]
        assert len(messages_with_details) > 0  # Some should remain


@requires_api_key
class TestReasoningDetailsAPI:
    """Integration tests for reasoning_details with actual API."""

    @pytest.mark.parametrize(
        "name,config",
        [m for m in REASONING_MODELS if m[1].reasoning_field == ReasoningField.REASONING_DETAILS][
            :2
        ],
        ids=[
            m[0]
            for m in REASONING_MODELS
            if m[1].reasoning_field == ReasoningField.REASONING_DETAILS
        ][:2],
    )
    def test_app_returns_reasoning_details(self, name: str, config: ModelConfig):
        """Test that OpenRouterLLM returns raw reasoning_details array."""
        from quest_evals.llm import OpenRouterLLM

        llm = OpenRouterLLM(
            model=config.model_id,
            max_tokens=500,
            extra_body=config.extra_body,
            reasoning_field=config.reasoning_field,
        )

        response = llm.complete(
            prompt=TEST_PROMPT,
            system=TEST_SYSTEM,
        )

        # Models with REASONING_DETAILS field should return the raw array
        if config.reasoning_field == ReasoningField.REASONING_DETAILS:
            # May or may not have reasoning_details depending on model
            # But if it has reasoning text, it should have details
            if response.reasoning:
                assert response.reasoning_details is not None, (
                    f"{name}: has reasoning text but no reasoning_details array"
                )
                assert isinstance(response.reasoning_details, list), (
                    f"{name}: reasoning_details should be a list"
                )

    def test_multi_turn_with_reasoning_details(self):
        """Test a simple multi-turn conversation preserving reasoning_details."""
        from quest_evals.llm import OpenRouterLLM

        # Use a cheap model that supports reasoning
        llm = OpenRouterLLM(
            model="openai/gpt-5-nano",
            max_tokens=500,
            extra_body={"reasoning": {"effort": "medium"}},
            reasoning_field=ReasoningField.REASONING_DETAILS,
        )

        # First turn
        response1 = llm.complete(
            prompt="What is 2+2?",
            system="Be concise.",
        )

        # Build history with reasoning_details
        history = [
            {"role": "user", "content": "What is 2+2?"},
        ]
        assistant_msg: dict = {"role": "assistant", "content": response1.text}
        if response1.reasoning_details:
            assistant_msg["reasoning_details"] = response1.reasoning_details
        history.append(assistant_msg)

        # Second turn using history
        response2 = llm.complete(
            prompt="Now multiply that by 3",
            system="Be concise.",
            history=history,
        )

        # Both should succeed
        assert response1.text, "First response empty"
        assert response2.text, "Second response empty"

        # Second response should reference the first
        # (This is a weak test but validates the flow works)
        assert "12" in response2.text or "twelve" in response2.text.lower(), (
            f"Expected 12 (4*3), got: {response2.text}"
        )


# =============================================================================
# MAIN (for running outside pytest)
# =============================================================================

if __name__ == "__main__":
    """Run a quick test of all models when executed directly."""
    print("=" * 70)
    print("OpenRouter Model Configuration Integration Test")
    print("=" * 70)

    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set")
        exit(1)

    results = []
    for name, config in ALL_MODELS:
        print(f"\nTesting {name} ({config.model_id})...")
        result = call_openrouter(
            model_id=config.model_id,
            extra_body=config.extra_body,
            reasoning_field=config.reasoning_field,
            timeout=90.0,
        )
        results.append((name, config, result))

        if result.success:
            reasoning_status = "✓" if result.reasoning_content else "✗"
            expects = config.reasoning_field != ReasoningField.NONE
            print(f"  ✓ Response: {result.response_text[:50]}...")
            print(f"  Reasoning: {reasoning_status} (expected: {'yes' if expects else 'no'})")
            print(f"  Tokens: {result.input_tokens} in, {result.output_tokens} out")
        else:
            print(f"  ✗ Error: {result.error}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successes = sum(1 for _, _, r in results if r.success)
    failures = len(results) - successes

    print(f"Total: {len(results)} | Success: {successes} | Failed: {failures}")

    # List failures
    if failures:
        print("\nFailed models:")
        for name, config, result in results:
            if not result.success:
                print(f"  - {name}: {result.error}")

    # Check reasoning expectations
    print("\nReasoning token mismatches:")
    for name, config, result in results:
        if result.success:
            got_reasoning = bool(result.reasoning_content)
            expects_reasoning = config.reasoning_field != ReasoningField.NONE
            if expects_reasoning and not got_reasoning:
                print(f"  ⚠ {name}: expected reasoning but got none")
            elif not expects_reasoning and got_reasoning:
                print(f"  ⚠ {name}: got unexpected reasoning")
