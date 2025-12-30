"""LLM interface for quest evaluation using OpenRouter."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from .model_configs import ReasoningField

# Load environment variables from .env
load_dotenv()


@dataclass
class LLMResponse:
    """Response from an LLM."""

    text: str
    input_tokens: int
    output_tokens: int
    reasoning: str = ""  # Extracted reasoning text (for display/logging)
    reasoning_details: list | None = (
        None  # Raw reasoning_details array (for multi-turn preservation)
    )


class LLMInterface(ABC):
    """Abstract interface for LLM providers."""

    @property
    @abstractmethod
    def model(self) -> str:
        """The model identifier being used."""
        pass

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Get a completion from the LLM."""
        pass


def extract_reasoning(message: dict, reasoning_field: ReasoningField) -> str:
    """
    Extract reasoning tokens from API response based on model config.

    Args:
        message: The message dict from API response (choices[0].message)
        reasoning_field: Which field to extract reasoning from (from ModelConfig)

    Returns:
        Extracted reasoning text, or empty string if not available
    """
    if reasoning_field == ReasoningField.NONE:
        return ""

    if reasoning_field == ReasoningField.REASONING:
        # Simple string field: message.reasoning or message.reasoning_content
        return message.get("reasoning") or message.get("reasoning_content") or ""

    if reasoning_field == ReasoningField.REASONING_DETAILS:
        # Structured array: message.reasoning_details[]
        # Contains objects with types: reasoning.text, reasoning.summary, reasoning.encrypted
        reasoning_details = message.get("reasoning_details", [])
        if not reasoning_details:
            return ""

        reasoning_parts = []
        for detail in reasoning_details:
            if isinstance(detail, dict):
                detail_type = detail.get("type", "")
                if detail_type == "reasoning.text":
                    reasoning_parts.append(detail.get("text", ""))
                elif detail_type == "reasoning.summary":
                    reasoning_parts.append(detail.get("summary", ""))
                # reasoning.encrypted is intentionally skipped (redacted content)

        return "\n".join(reasoning_parts) if reasoning_parts else ""

    return ""


class OpenRouterLLM(LLMInterface):
    """OpenRouter API implementation for cross-LLM evaluation."""

    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4",
        max_tokens: int = 16000,  # High enough for reasoning + response
        api_key: str | None = None,
        extra_body: dict | None = None,
        reasoning_field: ReasoningField = ReasoningField.NONE,
    ):
        """
        Initialize OpenRouter LLM.

        Args:
            model: OpenRouter model ID (e.g., "anthropic/claude-opus-4.5")
            max_tokens: Max tokens for completion
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            extra_body: Extra parameters for the model (reasoning, verbosity, etc.)
            reasoning_field: Where to extract reasoning from in response
        """
        self._model = model
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.extra_body = extra_body or {}
        self.reasoning_field = reasoning_field

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY in .env or pass api_key parameter."
            )

    @property
    def model(self) -> str:
        """The model identifier being used."""
        return self._model

    @property
    def model_name(self) -> str:
        """Human-readable model name including config."""
        return self._model

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> LLMResponse:
        """Get a completion from OpenRouter with retry logic.

        Args:
            prompt: Current user prompt
            system: System prompt
            history: Previous conversation turns [{"role": "user/assistant", "content": "..."}]
        """
        import time

        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        # Add conversation history
        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            **self.extra_body,  # Model-specific params (reasoning, verbosity, etc.)
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/quest-evals",
            "X-Title": "Quest Evals",
        }

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=120.0) as client:
                    response = client.post(
                        self.OPENROUTER_API_URL,
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()

                # Extract response
                message = data["choices"][0]["message"]
                text = message.get("content") or ""

                # Extract reasoning based on configured field type
                reasoning = extract_reasoning(message, self.reasoning_field)

                # Preserve raw reasoning_details for multi-turn conversations
                # Per OpenRouter docs: must pass back unmodified for reasoning continuity
                # https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
                reasoning_details = message.get("reasoning_details")

                # Debug output
                if os.getenv("DEBUG_LLM"):
                    print(f"DEBUG: model = {self.model}")
                    print(f"DEBUG: reasoning_field = {self.reasoning_field.value}")
                    print(f"DEBUG: message keys = {list(message.keys())}")
                    print(f"DEBUG: content = {text[:200] if text else '(empty)'}")
                    print(f"DEBUG: reasoning = {reasoning[:200] if reasoning else '(empty)'}")
                    print(
                        f"DEBUG: reasoning_details = {len(reasoning_details) if reasoning_details else 0} items"
                    )

                # If content is empty but we have reasoning, the model didn't output an action
                # This happens with some thinking models - format as our expected structure
                if not text and reasoning:
                    # Model did reasoning but no content - default to action 1
                    text = f"REASONING: {reasoning}\n\nACTION: 1"

                # Extract token usage
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                return LLMResponse(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning=reasoning,
                    reasoning_details=reasoning_details,
                )
            except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2**attempt  # exponential backoff
                    time.sleep(wait)
                    continue
                raise

        raise last_error  # type: ignore


SYSTEM_PROMPT = """You are a Space Ranger playing an interactive text quest from the game Space Rangers 2.

CONTEXT:
- You are a skilled ranger who has been given an important mission
- The quest involves reading situations, tracking information, and making strategic choices
- Your goal is to COMPLETE THE QUEST SUCCESSFULLY (reach a winning ending)
- Failing the quest or dying means you lose
- IMPORTANT: The quest has some randomness (dice rolls, random events), so not all failures are due to bad choices. Focus on LOGICAL mistakes you can learn from, not random bad luck.

RESPONSE FORMAT (you MUST follow this exactly):
```
REASONING: <your analysis of the current situation, what you remember, what you've learned, and why you're making this choice - be thorough>

ACTION: <single number only, e.g. 1 or 2 or 3>
```

GUIDELINES:
1. In REASONING, analyze the situation thoroughly:
   - What is the current state/puzzle?
   - What have you learned from previous steps?
   - What information is important (codes, passwords, patterns)?
   - Why is this choice the best option?
2. In ACTION, write ONLY the choice number
3. NEVER choose options that abandon the mission or lead to obvious failure
4. Pay close attention to puzzles, patterns, and numerical information
5. Remember everything - your memory of past events is crucial

You will receive the current situation and your previous actions. Think carefully before choosing."""


def format_system_prompt_with_notes(failure_notes: list[str] | None = None) -> str:
    """Format system prompt, optionally including failure notes from previous attempts."""
    if not failure_notes:
        return SYSTEM_PROMPT

    notes_section = "\n\nPREVIOUS ATTEMPT NOTES (learn from these mistakes):\n"
    for i, note in enumerate(failure_notes, 1):
        notes_section += f"\nAttempt {i} failure reflection:\n{note}\n"
    notes_section += "\nUse these insights to make better choices this time. Avoid repeating the same logical mistakes."

    return SYSTEM_PROMPT + notes_section


REFLECTION_PROMPT = """The quest has ended in failure.

Final status: {status}
Steps taken: {steps}

Please reflect on what went wrong. Focus ONLY on LOGICAL mistakes in your decision-making, NOT on random bad luck (dice rolls, random events are part of the game).

Write a brief reflection (max {max_words} words) that captures:
1. What key strategic mistakes did you make?
2. What should you do differently next time?
3. Any important information you discovered (codes, patterns, optimal paths)?

Be concise and actionable. Future attempts will read this note."""


def format_prompt(
    text: str,
    choices: list[tuple[int, str]],  # (choice_number, choice_text)
    params: list[str],
) -> str:
    """Format the game state as a prompt for the LLM.

    Args:
        text: The current location/situation text
        choices: List of (number, text) tuples for available choices
        params: List of visible parameter strings

    Returns:
        Formatted prompt string
    """
    parts = []

    parts.append("=== CURRENT SITUATION ===")
    parts.append(text)
    parts.append("")

    # Filter out empty params (visual separators in original game)
    visible_params = [p for p in params if p and p.strip()]
    if visible_params:
        parts.append("=== STATUS ===")
        for param in visible_params:
            parts.append(f"- {param}")
        parts.append("")

    parts.append("=== AVAILABLE CHOICES ===")
    for num, choice_text in choices:
        parts.append(f"{num}. {choice_text}")
    parts.append("")

    parts.append("Provide your REASONING and then your ACTION choice:")

    return "\n".join(parts)


@dataclass
class ParsedResponse:
    """Parsed LLM response with reasoning and action."""

    reasoning: str
    action: int | None
    raw_response: str


def parse_response(response: str, num_choices: int) -> ParsedResponse:
    """Parse the LLM's response to extract reasoning and action.

    Args:
        response: The LLM's response text
        num_choices: Number of valid choices

    Returns:
        ParsedResponse with reasoning and action number
    """
    raw = response.strip()
    reasoning = ""
    action = None

    # Try to extract REASONING section
    reasoning_match = re.search(r"REASONING:\s*(.+?)(?=ACTION:|$)", raw, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Try to extract ACTION section
    action_match = re.search(r"ACTION:\s*(\d+)", raw, re.IGNORECASE)
    if action_match:
        action_num = int(action_match.group(1))
        if 1 <= action_num <= num_choices:
            action = action_num

    # Fallback: just look for any number if no ACTION found
    if action is None:
        num_match = re.search(r"\d+", raw)
        if num_match:
            action_num = int(num_match.group())
            if 1 <= action_num <= num_choices:
                action = action_num

    # If no reasoning found, use the whole response minus action
    if not reasoning:
        reasoning = re.sub(r"ACTION:\s*\d+", "", raw).strip()
        if not reasoning:
            reasoning = "(no reasoning provided)"

    return ParsedResponse(reasoning=reasoning, action=action, raw_response=raw)


def parse_choice(response: str, num_choices: int) -> int | None:
    """Parse the LLM's response to extract a choice number (legacy compatibility).

    Args:
        response: The LLM's response text
        num_choices: Number of valid choices

    Returns:
        The choice number (1-indexed) or None if invalid
    """
    return parse_response(response, num_choices).action
