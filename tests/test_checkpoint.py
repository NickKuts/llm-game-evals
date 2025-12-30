"""
Unit tests for the checkpoint system.

Tests cover:
1. Serialization/deserialization of checkpoint data
2. Atomic file operations
3. Resume from various states (mid-step, mid-attempt, mid-run)
4. Config validation
5. Corruption handling
"""

import json
import tempfile
from pathlib import Path

import pytest

from quest_evals.checkpoint import (
    AttemptCheckpoint,
    CheckpointManager,
    EvalCheckpoint,
    RunCheckpoint,
    StepCheckpoint,
    _from_dict,
    _to_dict,
    config_matches,
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_results_dir():
    """Create a temporary directory for checkpoint files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_step():
    """Create a sample step checkpoint."""
    return StepCheckpoint(
        step=0,
        location_text="You are in a dark room.",
        params=["Health: 100", "Gold: 50"],
        choices=["Go left", "Go right", "Wait"],
        llm_response="I'll go left.\n\nAction: 1",
        reasoning="The left path seems safer.",
        chosen_index=1,
        input_tokens=150,
        output_tokens=50,
        was_auto=False,
        thinking_tokens="Let me think about this...",
    )


@pytest.fixture
def sample_attempt(sample_step):
    """Create a sample attempt checkpoint."""
    return AttemptCheckpoint(
        attempt_num=1,
        completed=True,
        game_status="WIN",
        steps=[sample_step],
        conversation_history=[
            {"role": "user", "content": "Choose wisely"},
            {"role": "assistant", "content": "I choose left"},
        ],
        total_input_tokens=150,
        total_output_tokens=50,
        llm_calls=1,
        error=None,
        failure_reflection=None,
    )


@pytest.fixture
def sample_run(sample_attempt):
    """Create a sample run checkpoint."""
    return RunCheckpoint(
        run_num=1,
        completed=True,
        success=True,
        successful_attempt=1,
        attempts=[sample_attempt],
        failure_notes=[],
        total_input_tokens=150,
        total_output_tokens=50,
    )


@pytest.fixture
def sample_checkpoint(sample_run):
    """Create a sample full checkpoint."""
    return EvalCheckpoint(
        version="2.0",
        model="anthropic/claude-sonnet-4",
        quest_file="assets/json/Muzon_eng.json",
        quest_name="Muzon Quest",
        num_runs=3,
        max_attempts=3,
        max_steps=500,
        max_llm_calls=60,
        runs=[sample_run],
        current_run=2,
        current_attempt=1,
        current_step=0,
    )


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestSerialization:
    """Test checkpoint serialization and deserialization."""

    def test_step_roundtrip(self, sample_step):
        """Test StepCheckpoint serialization roundtrip."""
        data = _to_dict(sample_step)
        restored = _from_dict(StepCheckpoint, data)

        assert restored.step == sample_step.step
        assert restored.location_text == sample_step.location_text
        assert restored.params == sample_step.params
        assert restored.choices == sample_step.choices
        assert restored.llm_response == sample_step.llm_response
        assert restored.reasoning == sample_step.reasoning
        assert restored.chosen_index == sample_step.chosen_index
        assert restored.input_tokens == sample_step.input_tokens
        assert restored.output_tokens == sample_step.output_tokens
        assert restored.was_auto == sample_step.was_auto
        assert restored.thinking_tokens == sample_step.thinking_tokens

    def test_attempt_roundtrip(self, sample_attempt):
        """Test AttemptCheckpoint serialization roundtrip."""
        data = _to_dict(sample_attempt)
        restored = _from_dict(AttemptCheckpoint, data)

        assert restored.attempt_num == sample_attempt.attempt_num
        assert restored.completed == sample_attempt.completed
        assert restored.game_status == sample_attempt.game_status
        assert len(restored.steps) == len(sample_attempt.steps)
        assert restored.steps[0].step == sample_attempt.steps[0].step
        assert restored.conversation_history == sample_attempt.conversation_history
        assert restored.total_input_tokens == sample_attempt.total_input_tokens
        assert restored.llm_calls == sample_attempt.llm_calls

    def test_run_roundtrip(self, sample_run):
        """Test RunCheckpoint serialization roundtrip."""
        data = _to_dict(sample_run)
        restored = _from_dict(RunCheckpoint, data)

        assert restored.run_num == sample_run.run_num
        assert restored.completed == sample_run.completed
        assert restored.success == sample_run.success
        assert restored.successful_attempt == sample_run.successful_attempt
        assert len(restored.attempts) == len(sample_run.attempts)
        assert restored.failure_notes == sample_run.failure_notes

    def test_full_checkpoint_roundtrip(self, sample_checkpoint):
        """Test full EvalCheckpoint serialization roundtrip."""
        data = _to_dict(sample_checkpoint)
        restored = _from_dict(EvalCheckpoint, data)

        assert restored.version == sample_checkpoint.version
        assert restored.model == sample_checkpoint.model
        assert restored.quest_file == sample_checkpoint.quest_file
        assert restored.quest_name == sample_checkpoint.quest_name
        assert restored.num_runs == sample_checkpoint.num_runs
        assert restored.max_attempts == sample_checkpoint.max_attempts
        assert restored.current_run == sample_checkpoint.current_run
        assert len(restored.runs) == len(sample_checkpoint.runs)

    def test_json_serializable(self, sample_checkpoint):
        """Test that checkpoint can be serialized to JSON."""
        data = _to_dict(sample_checkpoint)
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = _from_dict(EvalCheckpoint, restored_data)

        assert restored.model == sample_checkpoint.model
        assert len(restored.runs) == len(sample_checkpoint.runs)


# =============================================================================
# FILE OPERATIONS TESTS
# =============================================================================


class TestFileOperations:
    """Test checkpoint file save/load/delete."""

    def test_save_and_load(self, temp_results_dir, sample_checkpoint):
        """Test saving and loading a checkpoint."""
        path = save_checkpoint(sample_checkpoint, temp_results_dir)

        assert path.exists()
        assert path.suffix == ".json"

        loaded = load_checkpoint(
            temp_results_dir, sample_checkpoint.model, sample_checkpoint.quest_name
        )

        assert loaded is not None
        assert loaded.model == sample_checkpoint.model
        assert loaded.quest_name == sample_checkpoint.quest_name
        assert len(loaded.runs) == len(sample_checkpoint.runs)

    def test_atomic_write(self, temp_results_dir, sample_checkpoint):
        """Test that writes are atomic (no .tmp file left)."""
        save_checkpoint(sample_checkpoint, temp_results_dir)

        # Check no temp files exist
        tmp_files = list((temp_results_dir / "runs").glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_delete_checkpoint(self, temp_results_dir, sample_checkpoint):
        """Test checkpoint deletion."""
        save_checkpoint(sample_checkpoint, temp_results_dir)

        deleted = delete_checkpoint(
            temp_results_dir, sample_checkpoint.model, sample_checkpoint.quest_name
        )
        assert deleted is True

        # Try to load - should return None
        loaded = load_checkpoint(
            temp_results_dir, sample_checkpoint.model, sample_checkpoint.quest_name
        )
        assert loaded is None

    def test_load_nonexistent(self, temp_results_dir):
        """Test loading a nonexistent checkpoint returns None."""
        loaded = load_checkpoint(temp_results_dir, "nonexistent/model", "Nonexistent Quest")
        assert loaded is None

    def test_load_corrupt_file(self, temp_results_dir, sample_checkpoint):
        """Test that corrupt checkpoint files are handled gracefully."""
        path = save_checkpoint(sample_checkpoint, temp_results_dir)

        # Corrupt the file
        with open(path, "w") as f:
            f.write("{ invalid json }")

        loaded = load_checkpoint(
            temp_results_dir, sample_checkpoint.model, sample_checkpoint.quest_name
        )
        assert loaded is None

    def test_creates_runs_directory(self, temp_results_dir, sample_checkpoint):
        """Test that save creates the runs subdirectory."""
        runs_dir = temp_results_dir / "runs"
        assert not runs_dir.exists()

        save_checkpoint(sample_checkpoint, temp_results_dir)

        assert runs_dir.exists()
        assert runs_dir.is_dir()


# =============================================================================
# CONFIG VALIDATION TESTS
# =============================================================================


class TestConfigValidation:
    """Test checkpoint config matching."""

    def test_config_matches_same(self, sample_checkpoint):
        """Test that identical config matches."""
        assert config_matches(
            sample_checkpoint,
            num_runs=3,
            max_attempts=3,
            max_steps=500,
            max_llm_calls=60,
        )

    def test_config_mismatch_runs(self, sample_checkpoint):
        """Test that different num_runs doesn't match."""
        assert not config_matches(
            sample_checkpoint,
            num_runs=5,  # Different
            max_attempts=3,
            max_steps=500,
            max_llm_calls=60,
        )

    def test_config_mismatch_attempts(self, sample_checkpoint):
        """Test that different max_attempts doesn't match."""
        assert not config_matches(
            sample_checkpoint,
            num_runs=3,
            max_attempts=1,  # Different
            max_steps=500,
            max_llm_calls=60,
        )

    def test_config_uses_defaults(self, sample_checkpoint):
        """Test that missing config values use defaults."""
        # sample_checkpoint has max_steps=500, which matches the default
        sample_checkpoint.num_runs = 1
        sample_checkpoint.max_attempts = 1

        assert config_matches(sample_checkpoint)  # All defaults


# =============================================================================
# CHECKPOINT MANAGER TESTS
# =============================================================================


class TestCheckpointManager:
    """Test the CheckpointManager helper class."""

    def test_disabled_when_no_dir(self):
        """Test manager is disabled when no results_dir provided."""
        mgr = CheckpointManager(
            results_dir=None,
            model="test/model",
            quest_name="Test Quest",
            quest_file="test.json",
        )
        assert mgr.enabled is False

    def test_enabled_with_dir(self, temp_results_dir):
        """Test manager is enabled with results_dir."""
        mgr = CheckpointManager(
            results_dir=temp_results_dir,
            model="test/model",
            quest_name="Test Quest",
            quest_file="test.json",
        )
        assert mgr.enabled is True

    def test_run_lifecycle(self, temp_results_dir):
        """Test complete run lifecycle."""
        mgr = CheckpointManager(
            results_dir=temp_results_dir,
            model="test/model",
            quest_name="Test Quest",
            quest_file="test.json",
            num_runs=2,
            max_attempts=2,
        )

        # Start run 1
        mgr.start_run(1)
        assert mgr.checkpoint.current_run == 1

        # Start attempt 1
        mgr.start_attempt(1)
        assert mgr.checkpoint.current_attempt == 1

        # Record a step
        mgr.record_step(
            step=0,
            location_text="Test location",
            params=["Health: 100"],
            choices=["Choice A", "Choice B"],
            llm_response="I choose A",
            reasoning="A seems better",
            chosen_index=1,
            input_tokens=100,
            output_tokens=50,
        )
        assert mgr.checkpoint.current_step == 1

        # Save checkpoint
        mgr.save()
        assert (temp_results_dir / "runs").exists()

        # Complete attempt
        mgr.complete_attempt(completed=True, game_status="WIN")
        assert mgr._current_attempt is None

        # Complete run
        mgr.complete_run(success=True, successful_attempt=1)
        assert mgr._current_run is None
        assert len(mgr.checkpoint.runs) == 1

    def test_step_recording(self, temp_results_dir):
        """Test that steps are recorded correctly."""
        mgr = CheckpointManager(
            results_dir=temp_results_dir,
            model="test/model",
            quest_name="Test Quest",
            quest_file="test.json",
        )

        mgr.start_run(1)
        mgr.start_attempt(1)

        # Record multiple steps
        for i in range(5):
            mgr.record_step(
                step=i,
                location_text=f"Location {i}",
                params=[],
                choices=["Continue"],
                llm_response="Continue",
                reasoning="Only option",
                chosen_index=1,
                input_tokens=10,
                output_tokens=5,
                was_auto=(i % 2 == 0),
            )

        assert len(mgr._current_attempt.steps) == 5
        assert mgr._current_attempt.total_input_tokens == 50
        assert mgr._current_attempt.total_output_tokens == 25
        # Only non-auto steps count as LLM calls
        assert mgr._current_attempt.llm_calls == 2  # Steps 1, 3 (0-indexed)

    def test_resume_from_checkpoint(self, temp_results_dir):
        """Test resuming from a saved checkpoint."""
        # Create and save initial state
        mgr1 = CheckpointManager(
            results_dir=temp_results_dir,
            model="test/model",
            quest_name="Test Quest",
            quest_file="test.json",
            num_runs=3,
            max_attempts=2,
        )

        mgr1.start_run(1)
        mgr1.start_attempt(1)
        mgr1.record_step(
            step=0,
            location_text="Step 1",
            params=[],
            choices=["A"],
            llm_response="A",
            reasoning="",
            chosen_index=1,
            input_tokens=10,
            output_tokens=5,
        )
        mgr1.record_step(
            step=1,
            location_text="Step 2",
            params=[],
            choices=["B"],
            llm_response="B",
            reasoning="",
            chosen_index=1,
            input_tokens=10,
            output_tokens=5,
        )
        mgr1.save()

        # Create new manager and resume
        mgr2 = CheckpointManager(
            results_dir=temp_results_dir,
            model="test/model",
            quest_name="Test Quest",
            quest_file="test.json",
            num_runs=3,
            max_attempts=2,
        )

        resumed = mgr2.try_resume()
        assert resumed is not None
        assert resumed.current_run == 1
        assert resumed.current_step == 2

        # Check the in-progress run was restored
        assert len(resumed.runs) == 1
        incomplete_run = resumed.runs[0]
        assert not incomplete_run.completed
        assert len(incomplete_run.attempts) == 1
        assert len(incomplete_run.attempts[0].steps) == 2

    def test_resume_config_mismatch(self, temp_results_dir):
        """Test that resume fails if config doesn't match."""
        # Create checkpoint with specific config
        mgr1 = CheckpointManager(
            results_dir=temp_results_dir,
            model="test/model",
            quest_name="Test Quest",
            quest_file="test.json",
            num_runs=3,
            max_attempts=2,
        )
        mgr1.start_run(1)
        mgr1.start_attempt(1)
        mgr1.save()

        # Try to resume with different config
        mgr2 = CheckpointManager(
            results_dir=temp_results_dir,
            model="test/model",
            quest_name="Test Quest",
            quest_file="test.json",
            num_runs=5,  # Different!
            max_attempts=2,
        )

        resumed = mgr2.try_resume()
        assert resumed is None  # Should not resume

    def test_conversation_history_preserved(self, temp_results_dir):
        """Test that conversation history with reasoning_details is preserved."""
        mgr = CheckpointManager(
            results_dir=temp_results_dir,
            model="test/model",
            quest_name="Test Quest",
            quest_file="test.json",
        )

        mgr.start_run(1)
        mgr.start_attempt(1)

        # Update conversation with reasoning_details
        history = [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning_details": [{"type": "reasoning.text", "text": "I thought about it"}],
            },
        ]
        mgr.update_conversation(history)
        mgr.save()

        # Load and verify
        loaded = load_checkpoint(temp_results_dir, "test/model", "Test Quest")
        assert loaded is not None
        last_run = loaded.runs[-1]
        last_attempt = last_run.attempts[-1]
        assert len(last_attempt.conversation_history) == 2
        assert "reasoning_details" in last_attempt.conversation_history[1]

    def test_failure_notes_accumulate(self, temp_results_dir):
        """Test that failure notes accumulate across attempts."""
        mgr = CheckpointManager(
            results_dir=temp_results_dir,
            model="test/model",
            quest_name="Test Quest",
            quest_file="test.json",
            max_attempts=3,
        )

        mgr.start_run(1)

        # Attempt 1 fails
        mgr.start_attempt(1)
        mgr.complete_attempt(
            completed=True, game_status="FAIL", failure_reflection="I should have gone left"
        )

        # Check failure notes
        assert len(mgr._current_run.failure_notes) == 1
        assert "left" in mgr._current_run.failure_notes[0]

        # Attempt 2 fails
        mgr.start_attempt(2)
        mgr.complete_attempt(
            completed=True, game_status="DEAD", failure_reflection="I should have avoided the trap"
        )

        # Check both notes
        assert len(mgr._current_run.failure_notes) == 2
        assert mgr.get_failure_notes() == mgr._current_run.failure_notes


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_checkpoint(self, temp_results_dir):
        """Test saving and loading an empty checkpoint."""
        checkpoint = EvalCheckpoint(
            model="test/model",
            quest_name="Test",
            quest_file="test.json",
        )

        save_checkpoint(checkpoint, temp_results_dir)
        loaded = load_checkpoint(temp_results_dir, "test/model", "Test")

        assert loaded is not None
        assert loaded.runs == []
        assert loaded.current_run == 1

    def test_special_characters_in_names(self, temp_results_dir):
        """Test handling of special characters in quest/model names."""
        checkpoint = EvalCheckpoint(
            model="anthropic/claude-sonnet-4:beta",
            quest_name="Quest <with> special/chars",
            quest_file="test.json",
        )

        path = save_checkpoint(checkpoint, temp_results_dir)
        assert path.exists()

        loaded = load_checkpoint(temp_results_dir, checkpoint.model, checkpoint.quest_name)
        assert loaded is not None

    def test_large_step_history(self, temp_results_dir):
        """Test checkpoint with many steps."""
        mgr = CheckpointManager(
            results_dir=temp_results_dir,
            model="test/model",
            quest_name="Test Quest",
            quest_file="test.json",
        )

        mgr.start_run(1)
        mgr.start_attempt(1)

        # Record 100 steps
        for i in range(100):
            mgr.record_step(
                step=i,
                location_text=f"Location {i} with some longer text to simulate real content",
                params=[f"Param{j}: {i + j}" for j in range(5)],
                choices=[f"Choice {c}" for c in range(4)],
                llm_response=f"Response for step {i}",
                reasoning=f"Reasoning for step {i}",
                chosen_index=1,
                input_tokens=100 + i,
                output_tokens=50 + i,
            )

        mgr.save()

        # Load and verify
        loaded = load_checkpoint(temp_results_dir, "test/model", "Test Quest")
        assert loaded is not None
        assert len(loaded.runs[-1].attempts[-1].steps) == 100

    def test_unicode_content(self, temp_results_dir):
        """Test checkpoint with unicode content."""
        mgr = CheckpointManager(
            results_dir=temp_results_dir,
            model="test/model",
            quest_name="Test Quest",
            quest_file="test.json",
        )

        mgr.start_run(1)
        mgr.start_attempt(1)
        mgr.record_step(
            step=0,
            location_text="Привет! 你好! 🎮 Émojis work!",
            params=["金: 100"],
            choices=["继续", "Продолжить"],
            llm_response="选择1",
            reasoning="日本語テスト",
            chosen_index=1,
            input_tokens=50,
            output_tokens=25,
        )
        mgr.save()

        loaded = load_checkpoint(temp_results_dir, "test/model", "Test Quest")
        assert loaded is not None
        step = loaded.runs[-1].attempts[-1].steps[0]
        assert "Привет" in step.location_text
        assert "🎮" in step.location_text

    def test_disabled_manager_operations(self):
        """Test that disabled manager operations are no-ops."""
        mgr = CheckpointManager(
            results_dir=None,  # Disabled
            model="test/model",
            quest_name="Test Quest",
            quest_file="test.json",
        )

        # These should not raise
        mgr.start_run(1)
        mgr.start_attempt(1)
        mgr.record_step(
            step=0,
            location_text="",
            params=[],
            choices=[],
            llm_response="",
            reasoning="",
            chosen_index=1,
            input_tokens=0,
            output_tokens=0,
        )
        mgr.save()  # Should be a no-op
        mgr.delete()  # Should be a no-op

        # Manager state should still be empty
        assert len(mgr.checkpoint.runs) == 0
