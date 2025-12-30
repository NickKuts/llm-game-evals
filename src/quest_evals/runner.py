"""Quest evaluation runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .checkpoint import CheckpointManager
from .llm import (
    REFLECTION_PROMPT,
    SYSTEM_PROMPT,
    LLMInterface,
    format_prompt,
    format_system_prompt_with_notes,
    parse_response,
)
from .loader import load_quest as _load_quest
from .models import Quest
from .player import GameStatus, QuestPlayer


# ANSI colors for progress display
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    MAGENTA = "\033[35m"


def _print_overall_progress(
    run: int,
    total_runs: int,
    attempt: int,
    total_attempts: int,
    successful_runs: int = 0,
    model: str = "",
    quest: str = "",
) -> None:
    """Print a tqdm-style overall progress header."""
    # Overall progress bar (runs completed)
    total_items = total_runs * total_attempts
    completed_items = (run - 1) * total_attempts + (attempt - 1)
    pct = completed_items / total_items if total_items > 0 else 0
    bar_width = 30
    filled = int(bar_width * pct)
    bar = (
        "━" * filled + "╺" + "─" * (bar_width - filled - 1)
        if filled < bar_width
        else "━" * bar_width
    )

    # Status emoji
    status = f"{C.GREEN}✓{successful_runs}{C.RESET}" if successful_runs > 0 else ""

    print(f"\n{C.MAGENTA}{'─' * 70}{C.RESET}")
    print(
        f"{C.BOLD}{C.MAGENTA}📊 PROGRESS:{C.RESET} [{bar}] {pct * 100:5.1f}% │ "
        f"{C.CYAN}Run {run}/{total_runs}{C.RESET} │ "
        f"{C.YELLOW}Attempt {attempt}/{total_attempts}{C.RESET} {status}"
    )
    if model:
        print(f"   {C.DIM}Model: {model} │ Quest: {quest}{C.RESET}")
    print(f"{C.MAGENTA}{'─' * 70}{C.RESET}\n", flush=True)


@dataclass
class StepRecord:
    """Record of a single step in the game."""

    step: int
    location_text: str
    params: list[str]
    choices: list[str]
    llm_response: str
    reasoning: str  # LLM's reasoning extracted from response
    chosen_index: int
    input_tokens: int
    output_tokens: int
    was_auto: bool = False  # True if auto-selected (single choice)
    thinking_tokens: str = ""  # API's internal thinking (separate from content)


@dataclass
class EvalResult:
    """Result of a quest evaluation run."""

    quest_name: str
    completed: bool
    game_status: GameStatus
    steps: int
    llm_calls: int  # Actual LLM API calls made
    total_input_tokens: int
    total_output_tokens: int
    history: list[StepRecord] = field(default_factory=list)
    error: str | None = None
    attempt: int = 1  # Which attempt this is (1-indexed)
    failure_reflection: str | None = None  # Reflection on failure (if failed)


def load_quest(path: Path) -> Quest:
    """Load a quest from a file (.qm, .qmm, or .json)."""
    return _load_quest(path)


def _print_progress(
    step: int,
    llm_calls: int,
    max_llm_calls: int,
    total_tokens: int,
    text: str,
    choices: list[str],
    chosen: int,
    was_auto: bool,
    reasoning: str = "",
    status: str = "RUNNING",
) -> None:
    """Print a compact progress line."""
    # Progress bar for LLM calls
    bar_width = 20
    filled = int(bar_width * llm_calls / max_llm_calls)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Truncate text for display
    text_preview = (
        text.replace("\n", " ")[:60] + "..." if len(text) > 60 else text.replace("\n", " ")
    )

    # Choice info
    if was_auto:
        choice_info = f"{C.DIM}auto{C.RESET}"
    else:
        choice_info = f"{C.CYAN}→{chosen}{C.RESET}"

    # Format choices compactly
    choices_preview = " | ".join(c[:20] + "..." if len(c) > 20 else c for c in choices[:4])
    if len(choices) > 4:
        choices_preview += f" (+{len(choices) - 4})"

    # Reasoning preview
    reasoning_preview = (
        reasoning.replace("\n", " ")[:80] + "..."
        if len(reasoning) > 80
        else reasoning.replace("\n", " ")
    )

    print(
        f"{C.BOLD}Step {step + 1:3d}{C.RESET} │ "
        f"{C.YELLOW}LLM [{bar}] {llm_calls}/{max_llm_calls}{C.RESET} │ "
        f"{C.DIM}{total_tokens:,} tok{C.RESET} │ "
        f"{choice_info}"
    )
    print(f"        │ {C.DIM}{text_preview}{C.RESET}")
    print(f"        │ {C.GREEN}Choices:{C.RESET} {choices_preview}")
    if reasoning and not was_auto:
        print(f"        │ {C.BLUE}Reasoning:{C.RESET} {reasoning_preview}")
    print(flush=True)


def run_quest(
    quest: Quest,
    llm: LLMInterface,
    max_steps: int = 500,
    max_llm_calls: int = 60,
    verbose: bool = False,
    show_progress: bool = True,
    failure_notes: list[str] | None = None,
    attempt: int = 1,
    reflection_max_words: int = 200,
    checkpoint_mgr: CheckpointManager | None = None,
) -> EvalResult:
    """Run a quest with an LLM player.

    Args:
        quest: The quest to play
        llm: The LLM interface to use
        max_steps: Maximum number of game steps before timeout
        max_llm_calls: Maximum number of LLM API calls (default 60)
        verbose: Whether to print detailed output
        show_progress: Whether to show live progress (default True)
        failure_notes: List of reflection notes from previous failed attempts
        attempt: Which attempt this is (1-indexed)
        reflection_max_words: Max words for failure reflection
        checkpoint_mgr: Optional checkpoint manager for per-step saving

    Returns:
        EvalResult with the outcome
    """
    player = QuestPlayer(quest)
    step_history: list[StepRecord] = []
    conversation_history: list[dict[str, str]] = []  # For LLM memory
    total_input = 0
    total_output = 0
    llm_calls = 0

    # Build system prompt with any failure notes from previous attempts
    system_prompt = format_system_prompt_with_notes(failure_notes)

    quest_name = quest.task_text[:50] if quest.task_text else "Unknown Quest"

    if show_progress and not verbose:
        attempt_str = f" (Attempt {attempt})" if attempt > 1 else ""
        print(f"\n{C.BOLD}{C.CYAN}═══ Playing: {quest_name}...{attempt_str} ═══{C.RESET}\n")
        if failure_notes:
            print(
                f"{C.DIM}Using {len(failure_notes)} failure note(s) from previous attempts{C.RESET}\n"
            )

    for step in range(max_steps):
        state = player.get_state()

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Step {step + 1} | LLM calls: {llm_calls}/{max_llm_calls}")
            print(f"Status: {state.game_status.name}")
            print(f"\n{state.text[:300]}..." if len(state.text) > 300 else f"\n{state.text}")
            if state.params_state:
                print("\nParams:", state.params_state)
            print(f"\nChoices ({len(state.choices)}):")
            for i, c in enumerate(state.choices, 1):
                status = "" if c.active else " [inactive]"
                print(f"  {i}. {c.text}{status}")

        # Check if game is over
        if state.game_status != GameStatus.RUNNING:
            if show_progress and not verbose:
                if state.game_status == GameStatus.WIN:
                    print(f"{C.BRIGHT_GREEN}{C.BOLD}✓ QUEST COMPLETED!{C.RESET}")
                elif state.game_status == GameStatus.FAIL:
                    print(f"{C.RED}{C.BOLD}✗ Quest Failed{C.RESET}")
                elif state.game_status == GameStatus.DEAD:
                    print(f"{C.RED}{C.BOLD}💀 You Died{C.RESET}")
            return EvalResult(
                quest_name=quest_name,
                completed=(state.game_status == GameStatus.WIN),
                game_status=state.game_status,
                steps=step,
                llm_calls=llm_calls,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                history=step_history,
            )

        # Get active choices
        active_choices = [(i + 1, c.text) for i, c in enumerate(state.choices) if c.active]

        if not active_choices:
            return EvalResult(
                quest_name=quest_name,
                completed=False,
                game_status=state.game_status,
                steps=step,
                llm_calls=llm_calls,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                history=step_history,
                error="No active choices available",
            )

        # If only one choice, take it automatically (save tokens)
        if len(active_choices) == 1:
            chosen_idx = active_choices[0][0]
            response_text = str(chosen_idx)
            reasoning = "(auto-selected single choice)"
            thinking_tokens = ""
            input_tokens = 0
            output_tokens = 0
            was_auto = True
            if verbose:
                print(f"\n[Auto-selecting only choice: {chosen_idx}]")
        else:
            # Check LLM call limit
            if llm_calls >= max_llm_calls:
                if show_progress and not verbose:
                    print(f"{C.YELLOW}{C.BOLD}⚠ LLM call limit reached ({max_llm_calls}){C.RESET}")
                return EvalResult(
                    quest_name=quest_name,
                    completed=False,
                    game_status=state.game_status,
                    steps=step,
                    llm_calls=llm_calls,
                    total_input_tokens=total_input,
                    total_output_tokens=total_output,
                    history=step_history,
                    error=f"Max LLM calls ({max_llm_calls}) exceeded",
                )

            # Format prompt and get LLM response
            prompt = format_prompt(state.text, active_choices, state.params_state)

            if verbose:
                print(f"\n[Calling LLM... ({llm_calls + 1}/{max_llm_calls})]")

            response = llm.complete(prompt, system=system_prompt, history=conversation_history)
            response_text = response.text.strip()
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            thinking_tokens = response.reasoning  # API's internal thinking (if any)
            reasoning_details = response.reasoning_details  # Raw array for multi-turn preservation
            was_auto = False
            llm_calls += 1

            total_input += input_tokens
            total_output += output_tokens

            # Parse response to get reasoning and action
            parsed = parse_response(response_text, len(state.choices))
            reasoning = parsed.reasoning
            chosen_idx = parsed.action

            if verbose:
                print(f"LLM reasoning: {reasoning[:200]}...")
                print(f"LLM action: {chosen_idx}")
                print(f"Tokens: {input_tokens} in, {output_tokens} out")

            if chosen_idx is None:
                # Invalid response - try first active choice as fallback
                chosen_idx = active_choices[0][0]
                if verbose:
                    print(f"[Invalid response, falling back to choice {chosen_idx}]")

            # Add FULL context to conversation history for memory
            # Include the complete situation so LLM remembers everything
            full_context = f"=== STEP {step + 1} ===\n"
            full_context += f"SITUATION:\n{state.text}\n\n"
            visible_params = [p for p in state.params_state if p and p.strip()]
            if visible_params:
                full_context += f"STATUS: {', '.join(visible_params)}\n\n"
            full_context += "CHOICES:\n"
            for num, choice_text in active_choices:
                full_context += f"  {num}. {choice_text}\n"

            conversation_history.append({"role": "user", "content": full_context})

            # Build assistant message with reasoning_details for multi-turn preservation
            # Per OpenRouter docs: must pass back unmodified for reasoning continuity
            # https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
            assistant_msg: dict = {"role": "assistant", "content": response_text}
            if reasoning_details:
                assistant_msg["reasoning_details"] = reasoning_details
            conversation_history.append(assistant_msg)

            # Keep more history (last 30 turns = 60 messages) for better memory
            if len(conversation_history) > 60:
                conversation_history = conversation_history[-60:]

        # Record step
        step_history.append(
            StepRecord(
                step=step,
                location_text=state.text,
                params=state.params_state,
                choices=[c.text for c in state.choices],
                llm_response=response_text,
                reasoning=reasoning,
                chosen_index=chosen_idx,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                was_auto=was_auto,
                thinking_tokens=thinking_tokens,
            )
        )

        # Save checkpoint after each step (per-step checkpointing)
        if checkpoint_mgr:
            checkpoint_mgr.record_step(
                step=step,
                location_text=state.text,
                params=state.params_state,
                choices=[c.text for c in state.choices],
                llm_response=response_text,
                reasoning=reasoning,
                chosen_index=chosen_idx,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                was_auto=was_auto,
                thinking_tokens=thinking_tokens,
            )
            checkpoint_mgr.update_conversation(conversation_history)
            checkpoint_mgr.save()

        # Show progress
        if show_progress and not verbose:
            _print_progress(
                step=step,
                llm_calls=llm_calls,
                max_llm_calls=max_llm_calls,
                total_tokens=total_input + total_output,
                text=state.text,
                choices=[c.text for c in state.choices if c.active],
                chosen=chosen_idx,
                was_auto=was_auto,
                reasoning=reasoning,
            )

        # Perform the jump
        choice = state.choices[chosen_idx - 1]
        player.perform_jump(choice.jump_id)

    # Ran out of steps
    return EvalResult(
        quest_name=quest_name,
        completed=False,
        game_status=player.get_state().game_status,
        steps=max_steps,
        llm_calls=llm_calls,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        history=step_history,
        error="Max steps exceeded",
        attempt=attempt,
    )


def _get_reflection(
    llm: LLMInterface,
    result: EvalResult,
    conversation_history: list[dict[str, str]],
    max_words: int = 200,
    show_progress: bool = True,
) -> tuple[str, int, int]:
    """Get the LLM's reflection on why the quest failed.

    Returns:
        Tuple of (reflection_text, input_tokens, output_tokens)
    """
    # Include final game text for better context
    final_text = ""
    if result.history:
        final_text = f"\n\nFinal game text:\n{result.history[-1].location_text[:500]}"

    prompt = (
        REFLECTION_PROMPT.format(
            status=result.game_status.name,
            steps=result.steps,
            max_words=max_words,
        )
        + final_text
    )

    if show_progress:
        print(f"\n{C.DIM}Generating failure reflection...{C.RESET}")

    # Use conversation history so model remembers what happened
    response = llm.complete(prompt, system=SYSTEM_PROMPT, history=conversation_history[-20:])
    reflection = response.text.strip()

    # Truncate if too long (rough word count)
    words = reflection.split()
    if len(words) > max_words:
        reflection = " ".join(words[:max_words]) + "..."

    if show_progress:
        print(
            f"{C.YELLOW}Reflection:{C.RESET} {reflection[:200]}{'...' if len(reflection) > 200 else ''}\n"
        )

    return reflection, response.input_tokens, response.output_tokens


@dataclass
class MultiAttemptResult:
    """Result of multiple quest attempts."""

    quest_name: str
    model: str
    attempts: list[EvalResult]
    success: bool  # True if any attempt succeeded
    successful_attempt: int | None = None  # Which attempt succeeded (1-indexed)
    total_tokens_in: int = 0
    total_tokens_out: int = 0


def run_quest_with_retries(
    quest: Quest,
    llm: LLMInterface,
    max_attempts: int = 3,
    max_steps: int = 500,
    max_llm_calls: int = 60,
    verbose: bool = False,
    show_progress: bool = True,
    reflection_max_words: int = 200,
    # For progress display context
    run_num: int = 1,
    total_runs: int = 1,
    successful_runs_so_far: int = 0,
    checkpoint_mgr: CheckpointManager | None = None,
) -> MultiAttemptResult:
    """Run a quest multiple times, learning from failures.

    Args:
        quest: The quest to play
        llm: The LLM interface to use
        max_attempts: Maximum number of attempts
        max_steps: Maximum game steps per attempt
        max_llm_calls: Maximum LLM calls per attempt
        verbose: Detailed output
        show_progress: Show progress
        reflection_max_words: Max words for failure reflections
        checkpoint_mgr: Optional checkpoint manager for per-step saving

    Returns:
        MultiAttemptResult with all attempts
    """
    quest_name = quest.task_text[:50] if quest.task_text else "Unknown Quest"
    attempts: list[EvalResult] = []
    failure_notes: list[str] = []
    conversation_history: list[dict[str, str]] = []
    total_in = 0
    total_out = 0

    for attempt_num in range(1, max_attempts + 1):
        if show_progress:
            _print_overall_progress(
                run=run_num,
                total_runs=total_runs,
                attempt=attempt_num,
                total_attempts=max_attempts,
                successful_runs=successful_runs_so_far,
                model=llm.model,
                quest=quest_name,
            )

        # Start attempt in checkpoint
        if checkpoint_mgr:
            checkpoint_mgr.start_attempt(attempt_num)

        # Run the quest
        result = run_quest(
            quest=quest,
            llm=llm,
            max_steps=max_steps,
            max_llm_calls=max_llm_calls,
            verbose=verbose,
            show_progress=show_progress,
            failure_notes=failure_notes if failure_notes else None,
            attempt=attempt_num,
            reflection_max_words=reflection_max_words,
            checkpoint_mgr=checkpoint_mgr,
        )

        total_in += result.total_input_tokens
        total_out += result.total_output_tokens

        # Build conversation history from this attempt for reflection
        conversation_history = []
        for step_record in result.history[-10:]:  # Last 10 steps
            conversation_history.append(
                {
                    "role": "user",
                    "content": f"Step {step_record.step + 1}: {step_record.location_text[:500]}",
                }
            )
            conversation_history.append({"role": "assistant", "content": step_record.llm_response})

        # If succeeded, we're done!
        if result.completed:
            result.attempt = attempt_num
            attempts.append(result)

            # Complete attempt in checkpoint
            if checkpoint_mgr:
                checkpoint_mgr.complete_attempt(
                    completed=True,
                    game_status=result.game_status.name,
                )

            if show_progress:
                print(f"\n{C.BRIGHT_GREEN}{C.BOLD}✓ SUCCESS on attempt {attempt_num}!{C.RESET}")

            return MultiAttemptResult(
                quest_name=quest_name,
                model=llm.model,
                attempts=attempts,
                success=True,
                successful_attempt=attempt_num,
                total_tokens_in=total_in,
                total_tokens_out=total_out,
            )

        # Failed - get reflection if not last attempt
        reflection = None
        if attempt_num < max_attempts:
            try:
                reflection, ref_tokens_in, ref_tokens_out = _get_reflection(
                    llm=llm,
                    result=result,
                    conversation_history=conversation_history,
                    max_words=reflection_max_words,
                    show_progress=show_progress,
                )
                result.failure_reflection = reflection
                failure_notes.append(reflection)
                total_in += ref_tokens_in
                total_out += ref_tokens_out
            except Exception as e:
                # Reflection failed - must retry the whole run to avoid corrupting benchmark
                if show_progress:
                    print(f"{C.RED}Reflection call failed: {e}{C.RESET}")
                    print(f"{C.YELLOW}Restarting run to avoid corrupting benchmark...{C.RESET}")
                raise RuntimeError(f"Reflection failed, run must be retried: {e}")

        # Complete attempt in checkpoint
        if checkpoint_mgr:
            checkpoint_mgr.complete_attempt(
                completed=True,
                game_status=result.game_status.name,
                error=result.error,
                failure_reflection=reflection,
            )

        result.attempt = attempt_num
        attempts.append(result)

    # All attempts failed
    if show_progress:
        print(f"\n{C.RED}{C.BOLD}✗ All {max_attempts} attempts failed{C.RESET}")

    return MultiAttemptResult(
        quest_name=quest_name,
        model=llm.model,
        attempts=attempts,
        success=False,
        total_tokens_in=total_in,
        total_tokens_out=total_out,
    )


@dataclass
class MultiRunResult:
    """Result of multiple independent runs (each with potentially multiple attempts)."""

    quest_name: str
    quest_file: str
    model: str
    runs: list[MultiAttemptResult]
    total_runs: int
    successful_runs: int
    success_rate: float
    total_tokens_in: int = 0
    total_tokens_out: int = 0


def run_evaluation(
    quest: Quest,
    llm: LLMInterface,
    quest_file: str,
    num_runs: int = 1,
    max_attempts: int = 1,
    max_steps: int = 500,
    max_llm_calls: int = 60,
    verbose: bool = False,
    show_progress: bool = True,
    reflection_max_words: int = 200,
    results_dir: Path | None = None,
    resume: bool = False,
) -> MultiRunResult:
    """Run a complete evaluation with multiple independent runs.

    Each run is completely independent (fresh start, no memory from previous runs).
    Within each run, attempts share reflection notes from failures.

    Checkpointing: When results_dir is provided, saves checkpoint after every step.
    Can resume from any point (run, attempt, or step) without data loss.

    Args:
        quest: The quest to evaluate
        llm: The LLM to use
        quest_file: Path to quest file (for logging)
        num_runs: Number of independent runs
        max_attempts: Max attempts per run (with reflection)
        max_steps: Max game steps per attempt
        max_llm_calls: Max LLM calls per attempt
        verbose: Detailed output
        show_progress: Show progress
        reflection_max_words: Max words for reflections
        results_dir: Directory for results/checkpoints (enables checkpointing)
        resume: If True, resume from last checkpoint if available

    Returns:
        MultiRunResult with all runs and aggregate stats
    """
    quest_name = quest.task_text[:50] if quest.task_text else "Unknown Quest"
    runs: list[MultiAttemptResult] = []
    total_in = 0
    total_out = 0
    successful_runs = 0
    start_run = 1

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(
        results_dir=results_dir if results_dir else None,
        model=llm.model,
        quest_name=quest_name,
        quest_file=quest_file,
        num_runs=num_runs,
        max_attempts=max_attempts,
        max_steps=max_steps,
        max_llm_calls=max_llm_calls,
    )

    # Try to resume from checkpoint
    if resume and results_dir:
        resumed = checkpoint_mgr.try_resume()
        if resumed:
            start_run, _, _ = checkpoint_mgr.get_resume_point()
            completed = checkpoint_mgr.get_completed_runs()

            if show_progress:
                print(
                    f"{C.YELLOW}Resuming from checkpoint: {len(completed)}/{num_runs} runs completed{C.RESET}"
                )

            # Restore completed runs
            for run_ckpt in completed:
                run_result = _reconstruct_run_from_full_checkpoint(run_ckpt)
                runs.append(run_result)
                total_in += run_result.total_tokens_in
                total_out += run_result.total_tokens_out
                if run_result.success:
                    successful_runs += 1

    MAX_RUN_RETRIES = 3  # Retry runs if reflection fails

    for run_num in range(start_run, num_runs + 1):
        run_result = None

        # Start tracking this run in checkpoint
        checkpoint_mgr.start_run(run_num)

        for retry in range(MAX_RUN_RETRIES):
            if show_progress:
                print(f"\n{C.BOLD}{C.CYAN}{'#' * 60}{C.RESET}")
                retry_note = f" (retry {retry + 1})" if retry > 0 else ""
                print(f"{C.BOLD}{C.CYAN}# RUN {run_num}/{num_runs}{retry_note}{C.RESET}")
                print(f"{C.BOLD}{C.CYAN}{'#' * 60}{C.RESET}")

            try:
                if max_attempts == 1:
                    # Single attempt per run
                    if show_progress:
                        _print_overall_progress(
                            run=run_num,
                            total_runs=num_runs,
                            attempt=1,
                            total_attempts=1,
                            successful_runs=successful_runs,
                            model=llm.model,
                            quest=quest_name,
                        )

                    # Start attempt in checkpoint
                    checkpoint_mgr.start_attempt(1)

                    result = run_quest(
                        quest=quest,
                        llm=llm,
                        max_steps=max_steps,
                        max_llm_calls=max_llm_calls,
                        verbose=verbose,
                        show_progress=show_progress,
                        checkpoint_mgr=checkpoint_mgr,
                    )

                    # Complete attempt in checkpoint
                    checkpoint_mgr.complete_attempt(
                        completed=True,
                        game_status=result.game_status.name,
                        error=result.error,
                    )

                    run_result = MultiAttemptResult(
                        quest_name=quest_name,
                        model=llm.model,
                        attempts=[result],
                        success=result.completed,
                        successful_attempt=1 if result.completed else None,
                        total_tokens_in=result.total_input_tokens,
                        total_tokens_out=result.total_output_tokens,
                    )
                else:
                    # Multiple attempts with reflection
                    run_result = run_quest_with_retries(
                        quest=quest,
                        llm=llm,
                        max_attempts=max_attempts,
                        max_steps=max_steps,
                        max_llm_calls=max_llm_calls,
                        verbose=verbose,
                        show_progress=show_progress,
                        reflection_max_words=reflection_max_words,
                        run_num=run_num,
                        total_runs=num_runs,
                        successful_runs_so_far=successful_runs,
                        checkpoint_mgr=checkpoint_mgr,
                    )

                break  # Success, exit retry loop

            except RuntimeError as e:
                # Reflection or other recoverable error - retry the run
                if retry < MAX_RUN_RETRIES - 1:
                    if show_progress:
                        print(
                            f"{C.YELLOW}Run failed, retrying... ({retry + 1}/{MAX_RUN_RETRIES}){C.RESET}"
                        )
                    continue
                else:
                    # Max retries exceeded
                    if show_progress:
                        print(
                            f"{C.RED}Run {run_num} failed after {MAX_RUN_RETRIES} retries: {e}{C.RESET}"
                        )
                    raise

        if run_result is None:
            raise RuntimeError(f"Run {run_num} failed to produce a result")

        # Complete run in checkpoint
        checkpoint_mgr.complete_run(
            success=run_result.success,
            successful_attempt=run_result.successful_attempt,
        )
        checkpoint_mgr.save()

        runs.append(run_result)
        total_in += run_result.total_tokens_in
        total_out += run_result.total_tokens_out

        if run_result.success:
            successful_runs += 1

        if show_progress and results_dir:
            print(f"{C.DIM}Checkpoint saved: {len(runs)}/{num_runs} runs{C.RESET}")

    success_rate = successful_runs / num_runs if num_runs > 0 else 0.0

    # Delete checkpoint on successful completion
    checkpoint_mgr.delete()

    if show_progress:
        print(f"\n{C.BOLD}{'#' * 60}{C.RESET}")
        print(f"{C.BOLD}EVALUATION COMPLETE{C.RESET}")
        print(f"{'#' * 60}")
        print(f"Success rate: {successful_runs}/{num_runs} ({success_rate:.0%})")

    return MultiRunResult(
        quest_name=quest_name,
        quest_file=quest_file,
        model=llm.model,
        runs=runs,
        total_runs=num_runs,
        successful_runs=successful_runs,
        success_rate=success_rate,
        total_tokens_in=total_in,
        total_tokens_out=total_out,
    )


def _reconstruct_run_from_full_checkpoint(run_ckpt) -> MultiAttemptResult:
    """Reconstruct MultiAttemptResult from full checkpoint data."""

    # Build EvalResult list from checkpoint attempts
    attempts = []
    for attempt_ckpt in run_ckpt.attempts:
        # Convert steps to StepRecord
        history = []
        for step_ckpt in attempt_ckpt.steps:
            history.append(
                StepRecord(
                    step=step_ckpt.step,
                    location_text=step_ckpt.location_text,
                    params=step_ckpt.params,
                    choices=step_ckpt.choices,
                    llm_response=step_ckpt.llm_response,
                    reasoning=step_ckpt.reasoning,
                    chosen_index=step_ckpt.chosen_index,
                    input_tokens=step_ckpt.input_tokens,
                    output_tokens=step_ckpt.output_tokens,
                    was_auto=step_ckpt.was_auto,
                    thinking_tokens=step_ckpt.thinking_tokens,
                )
            )

        # Map string game status back to enum
        game_status = (
            GameStatus[attempt_ckpt.game_status] if attempt_ckpt.game_status else GameStatus.RUNNING
        )

        attempts.append(
            EvalResult(
                quest_name=run_ckpt.run_num,  # Placeholder, will be overwritten
                completed=(game_status == GameStatus.WIN),
                game_status=game_status,
                steps=len(history),
                llm_calls=attempt_ckpt.llm_calls,
                total_input_tokens=attempt_ckpt.total_input_tokens,
                total_output_tokens=attempt_ckpt.total_output_tokens,
                history=history,
                error=attempt_ckpt.error,
                attempt=attempt_ckpt.attempt_num,
                failure_reflection=attempt_ckpt.failure_reflection,
            )
        )

    return MultiAttemptResult(
        quest_name="",  # Will be set by caller
        model="",  # Will be set by caller
        attempts=attempts,
        success=run_ckpt.success,
        successful_attempt=run_ckpt.successful_attempt,
        total_tokens_in=run_ckpt.total_input_tokens,
        total_tokens_out=run_ckpt.total_output_tokens,
    )
