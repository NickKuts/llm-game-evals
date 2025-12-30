"""
Per-step checkpoint system for quest evaluations.

Provides full state recovery - can resume from any point without data loss.
Checkpoints are saved after every LLM call to ensure no work is lost.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class StepCheckpoint:
    """Checkpoint data for a single step."""

    step: int
    location_text: str
    params: list[str]
    choices: list[str]
    llm_response: str
    reasoning: str
    chosen_index: int
    input_tokens: int
    output_tokens: int
    was_auto: bool = False
    thinking_tokens: str = ""


@dataclass
class AttemptCheckpoint:
    """Checkpoint data for an attempt (possibly in-progress)."""

    attempt_num: int
    completed: bool = False
    game_status: str = "RUNNING"  # GameStatus enum value
    steps: list[StepCheckpoint] = field(default_factory=list)
    conversation_history: list[dict] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    llm_calls: int = 0
    error: str | None = None
    failure_reflection: str | None = None


@dataclass
class RunCheckpoint:
    """Checkpoint data for a run (possibly in-progress)."""

    run_num: int
    completed: bool = False
    success: bool = False
    successful_attempt: int | None = None
    attempts: list[AttemptCheckpoint] = field(default_factory=list)
    failure_notes: list[str] = field(default_factory=list)  # Reflections for next attempt
    total_input_tokens: int = 0
    total_output_tokens: int = 0


@dataclass
class EvalCheckpoint:
    """Full evaluation checkpoint - can restore complete state."""

    # Version for future compatibility
    version: str = "2.0"

    # Metadata
    timestamp: str = ""
    model: str = ""
    quest_file: str = ""
    quest_name: str = ""

    # Config (to verify we're resuming same evaluation)
    num_runs: int = 1
    max_attempts: int = 1
    max_steps: int = 500
    max_llm_calls: int = 60

    # State
    runs: list[RunCheckpoint] = field(default_factory=list)
    current_run: int = 1  # 1-indexed
    current_attempt: int = 1  # 1-indexed
    current_step: int = 0  # 0-indexed

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


def _get_checkpoint_path(results_dir: Path, model: str, quest_name: str) -> Path:
    """Get checkpoint file path for a model/quest combination."""
    model_clean = model.replace("/", "_").replace(":", "_")
    quest_clean = re.sub(r'[<>:"/\\|?*\s]', "_", quest_name)[:40]
    return results_dir / "runs" / f".checkpoint_{model_clean}_{quest_clean}.json"


def _to_dict(obj: Any) -> Any:
    """Convert dataclass (possibly nested) to dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _from_dict(cls, data: dict) -> Any:
    """Reconstruct a dataclass from a dict."""
    if data is None:
        return None

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key not in field_types:
            continue

        field_type = field_types[key]
        type_str = str(field_type)

        # Handle list[StepCheckpoint], list[AttemptCheckpoint], etc.
        if "list[StepCheckpoint]" in type_str and isinstance(value, list):
            kwargs[key] = [_from_dict(StepCheckpoint, item) for item in value]
        elif "list[AttemptCheckpoint]" in type_str and isinstance(value, list):
            kwargs[key] = [_from_dict(AttemptCheckpoint, item) for item in value]
        elif "list[RunCheckpoint]" in type_str and isinstance(value, list):
            kwargs[key] = [_from_dict(RunCheckpoint, item) for item in value]
        else:
            kwargs[key] = value

    return cls(**kwargs)


def save_checkpoint(checkpoint: EvalCheckpoint, results_dir: Path) -> Path:
    """
    Save checkpoint atomically (write to temp, then rename).

    Returns the checkpoint path.
    """
    results_dir = Path(results_dir)
    (results_dir / "runs").mkdir(parents=True, exist_ok=True)

    # Update timestamp
    checkpoint.timestamp = datetime.now(UTC).isoformat()

    path = _get_checkpoint_path(results_dir, checkpoint.model, checkpoint.quest_name)
    temp_path = path.with_suffix(".tmp")

    # Write atomically
    data = _to_dict(checkpoint)
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=2)
    temp_path.rename(path)

    return path


def load_checkpoint(results_dir: Path, model: str, quest_name: str) -> EvalCheckpoint | None:
    """
    Load checkpoint if it exists and is valid.

    Returns None if no checkpoint or if corrupt.
    """
    path = _get_checkpoint_path(Path(results_dir), model, quest_name)

    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)

        # Check version compatibility
        version = data.get("version", "1.0")
        if version.startswith("1."):
            # Old format - can't restore, return None
            return None

        return _from_dict(EvalCheckpoint, data)

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"Warning: Corrupt checkpoint file, ignoring: {e}")
        return None


def delete_checkpoint(results_dir: Path, model: str, quest_name: str) -> bool:
    """Delete checkpoint file. Returns True if deleted."""
    path = _get_checkpoint_path(Path(results_dir), model, quest_name)
    if path.exists():
        path.unlink()
        return True
    return False


def config_matches(checkpoint: EvalCheckpoint, **config) -> bool:
    """Check if checkpoint config matches the requested config."""
    return (
        checkpoint.num_runs == config.get("num_runs", 1)
        and checkpoint.max_attempts == config.get("max_attempts", 1)
        and checkpoint.max_steps == config.get("max_steps", 500)
        and checkpoint.max_llm_calls == config.get("max_llm_calls", 60)
    )


# =============================================================================
# CHECKPOINT BUILDER - Helper class for managing checkpoint state
# =============================================================================


class CheckpointManager:
    """
    Manages checkpoint state during evaluation.

    Usage:
        mgr = CheckpointManager(results_dir, model, quest_name, quest_file, config)
        mgr.start_run(run_num)
        mgr.start_attempt(attempt_num)
        mgr.record_step(step_data)
        mgr.save()  # Called after each step
        mgr.complete_attempt(result)
        mgr.complete_run(result)
    """

    def __init__(
        self,
        results_dir: Path | None,
        model: str,
        quest_name: str,
        quest_file: str,
        num_runs: int = 1,
        max_attempts: int = 1,
        max_steps: int = 500,
        max_llm_calls: int = 60,
    ):
        self.results_dir = Path(results_dir) if results_dir else None
        self.enabled = results_dir is not None

        self.checkpoint = EvalCheckpoint(
            model=model,
            quest_name=quest_name,
            quest_file=quest_file,
            num_runs=num_runs,
            max_attempts=max_attempts,
            max_steps=max_steps,
            max_llm_calls=max_llm_calls,
        )

        self._current_run: RunCheckpoint | None = None
        self._current_attempt: AttemptCheckpoint | None = None

    def try_resume(self) -> EvalCheckpoint | None:
        """
        Try to load existing checkpoint for resume.

        Returns the checkpoint if valid and config matches, None otherwise.
        """
        if not self.enabled:
            return None

        existing = load_checkpoint(
            self.results_dir, self.checkpoint.model, self.checkpoint.quest_name
        )

        if existing and config_matches(
            existing,
            num_runs=self.checkpoint.num_runs,
            max_attempts=self.checkpoint.max_attempts,
            max_steps=self.checkpoint.max_steps,
            max_llm_calls=self.checkpoint.max_llm_calls,
        ):
            self.checkpoint = existing
            return existing

        return None

    def start_run(self, run_num: int, failure_notes: list[str] = None):
        """Start a new run."""
        self._current_run = RunCheckpoint(
            run_num=run_num,
            failure_notes=failure_notes or [],
        )
        self.checkpoint.current_run = run_num
        self.checkpoint.current_attempt = 1
        self.checkpoint.current_step = 0

    def start_attempt(self, attempt_num: int, conversation_history: list[dict] = None):
        """Start a new attempt within the current run."""
        self._current_attempt = AttemptCheckpoint(
            attempt_num=attempt_num,
            conversation_history=conversation_history or [],
        )
        self.checkpoint.current_attempt = attempt_num
        self.checkpoint.current_step = 0

    def record_step(
        self,
        step: int,
        location_text: str,
        params: list[str],
        choices: list[str],
        llm_response: str,
        reasoning: str,
        chosen_index: int,
        input_tokens: int,
        output_tokens: int,
        was_auto: bool = False,
        thinking_tokens: str = "",
    ):
        """Record a completed step."""
        if self._current_attempt is None:
            return

        step_data = StepCheckpoint(
            step=step,
            location_text=location_text,
            params=params,
            choices=choices,
            llm_response=llm_response,
            reasoning=reasoning,
            chosen_index=chosen_index,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            was_auto=was_auto,
            thinking_tokens=thinking_tokens,
        )

        self._current_attempt.steps.append(step_data)
        self._current_attempt.total_input_tokens += input_tokens
        self._current_attempt.total_output_tokens += output_tokens
        if not was_auto:
            self._current_attempt.llm_calls += 1

        self.checkpoint.current_step = step + 1

    def update_conversation(self, conversation_history: list[dict]):
        """Update conversation history (for reasoning_details preservation)."""
        if self._current_attempt:
            self._current_attempt.conversation_history = conversation_history.copy()

    def complete_attempt(
        self,
        completed: bool,
        game_status: str,
        error: str | None = None,
        failure_reflection: str | None = None,
    ):
        """Mark current attempt as complete."""
        if self._current_attempt is None:
            return

        self._current_attempt.completed = True
        self._current_attempt.game_status = game_status
        self._current_attempt.error = error
        self._current_attempt.failure_reflection = failure_reflection

        if self._current_run:
            self._current_run.attempts.append(self._current_attempt)
            self._current_run.total_input_tokens += self._current_attempt.total_input_tokens
            self._current_run.total_output_tokens += self._current_attempt.total_output_tokens

            # Add reflection to failure notes for next attempt
            if failure_reflection:
                self._current_run.failure_notes.append(failure_reflection)

        self._current_attempt = None

    def complete_run(self, success: bool, successful_attempt: int | None = None):
        """Mark current run as complete."""
        if self._current_run is None:
            return

        self._current_run.completed = True
        self._current_run.success = success
        self._current_run.successful_attempt = successful_attempt

        self.checkpoint.runs.append(self._current_run)
        self._current_run = None

    def save(self):
        """Save checkpoint to disk."""
        if not self.enabled:
            return

        # Include in-progress run/attempt
        checkpoint_copy = EvalCheckpoint(
            version=self.checkpoint.version,
            timestamp=self.checkpoint.timestamp,
            model=self.checkpoint.model,
            quest_file=self.checkpoint.quest_file,
            quest_name=self.checkpoint.quest_name,
            num_runs=self.checkpoint.num_runs,
            max_attempts=self.checkpoint.max_attempts,
            max_steps=self.checkpoint.max_steps,
            max_llm_calls=self.checkpoint.max_llm_calls,
            runs=self.checkpoint.runs.copy(),
            current_run=self.checkpoint.current_run,
            current_attempt=self.checkpoint.current_attempt,
            current_step=self.checkpoint.current_step,
        )

        # Add in-progress run with in-progress attempt
        if self._current_run:
            run_copy = RunCheckpoint(
                run_num=self._current_run.run_num,
                completed=False,
                success=False,
                attempts=self._current_run.attempts.copy(),
                failure_notes=self._current_run.failure_notes.copy(),
                total_input_tokens=self._current_run.total_input_tokens,
                total_output_tokens=self._current_run.total_output_tokens,
            )

            if self._current_attempt:
                run_copy.attempts.append(self._current_attempt)

            checkpoint_copy.runs.append(run_copy)

        save_checkpoint(checkpoint_copy, self.results_dir)

    def delete(self):
        """Delete checkpoint file (call after successful completion)."""
        if self.enabled:
            delete_checkpoint(self.results_dir, self.checkpoint.model, self.checkpoint.quest_name)

    def get_resume_point(self) -> tuple[int, int, int]:
        """
        Get the point to resume from.

        Returns (run_num, attempt_num, step_num) - all 1-indexed except step (0-indexed).
        """
        return (
            self.checkpoint.current_run,
            self.checkpoint.current_attempt,
            self.checkpoint.current_step,
        )

    def get_completed_runs(self) -> list[RunCheckpoint]:
        """Get list of fully completed runs."""
        return [r for r in self.checkpoint.runs if r.completed]

    def get_failure_notes(self) -> list[str]:
        """Get failure notes for the current run."""
        if self._current_run:
            return self._current_run.failure_notes

        # Check last incomplete run in checkpoint
        for run in reversed(self.checkpoint.runs):
            if not run.completed:
                return run.failure_notes

        return []

    def get_conversation_history(self) -> list[dict]:
        """Get conversation history from last incomplete attempt."""
        # Check in-progress attempt
        if self._current_attempt:
            return self._current_attempt.conversation_history

        # Check checkpoint for incomplete attempt
        for run in reversed(self.checkpoint.runs):
            if not run.completed and run.attempts:
                last_attempt = run.attempts[-1]
                if not last_attempt.completed:
                    return last_attempt.conversation_history

        return []
