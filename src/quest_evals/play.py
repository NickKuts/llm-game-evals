"""Interactive CLI for playing quests manually."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .player import GameStatus, QuestPlayer
from .qm_parser import parse_qm


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"


def colored(text: str, *colors: str) -> str:
    """Wrap text in ANSI color codes."""
    return "".join(colors) + text + Colors.RESET


def print_header(text: str) -> None:
    """Print a section header."""
    print()
    print(colored(f"═══ {text} ═══", Colors.BOLD, Colors.CYAN))


def print_separator() -> None:
    """Print a separator line."""
    print(colored("─" * 60, Colors.DIM))


def print_text(text: str) -> None:
    """Print game text with formatting."""
    # Clean up and wrap text nicely
    text = text.strip()
    if text:
        print()
        print(colored(text, Colors.WHITE))


def print_params(params: list[str]) -> None:
    """Print parameter status."""
    if params:
        print()
        print(colored("Status:", Colors.BOLD, Colors.YELLOW))
        for param in params:
            print(colored(f"  • {param}", Colors.YELLOW))


def print_choices(choices: list[tuple[int, str, bool]]) -> None:
    """Print available choices."""
    print()
    print(colored("Choices:", Colors.BOLD, Colors.GREEN))
    for num, text, active in choices:
        if active:
            print(colored(f"  [{num}] ", Colors.BRIGHT_GREEN) + colored(text, Colors.WHITE))
        else:
            print(
                colored(f"  [{num}] {text}", Colors.DIM)
                + colored(" (unavailable)", Colors.DIM, Colors.RED)
            )


def print_status(status: GameStatus) -> None:
    """Print game status."""
    if status == GameStatus.WIN:
        print()
        print(colored("╔════════════════════════════════════╗", Colors.BRIGHT_GREEN, Colors.BOLD))
        print(colored("║       🎉 QUEST COMPLETED! 🎉       ║", Colors.BRIGHT_GREEN, Colors.BOLD))
        print(colored("╚════════════════════════════════════╝", Colors.BRIGHT_GREEN, Colors.BOLD))
    elif status == GameStatus.FAIL:
        print()
        print(colored("╔════════════════════════════════════╗", Colors.BRIGHT_RED, Colors.BOLD))
        print(colored("║         ❌ QUEST FAILED ❌          ║", Colors.BRIGHT_RED, Colors.BOLD))
        print(colored("╚════════════════════════════════════╝", Colors.BRIGHT_RED, Colors.BOLD))
    elif status == GameStatus.DEAD:
        print()
        print(colored("╔════════════════════════════════════╗", Colors.BRIGHT_RED, Colors.BOLD))
        print(colored("║           💀 YOU DIED 💀           ║", Colors.BRIGHT_RED, Colors.BOLD))
        print(colored("╚════════════════════════════════════╝", Colors.BRIGHT_RED, Colors.BOLD))


def get_player_choice(num_choices: int, active_indices: list[int]) -> int:
    """Get player's choice input."""
    while True:
        try:
            prompt = colored("\nYour choice: ", Colors.BOLD, Colors.MAGENTA)
            raw = input(prompt).strip()

            if raw.lower() in ("q", "quit", "exit"):
                print(colored("\nQuitting game...", Colors.DIM))
                sys.exit(0)

            choice = int(raw)
            if choice < 1 or choice > num_choices:
                print(colored(f"Please enter a number between 1 and {num_choices}", Colors.RED))
                continue

            if choice not in active_indices:
                print(colored("That choice is not available right now", Colors.RED))
                continue

            return choice

        except ValueError:
            print(colored("Please enter a valid number (or 'q' to quit)", Colors.RED))


def play_quest(quest_path: Path) -> None:
    """Play a quest interactively."""
    # Load quest
    print(colored(f"\nLoading quest: {quest_path.name}", Colors.DIM))

    with open(quest_path, "rb") as f:
        data = f.read()

    quest = parse_qm(data)
    player = QuestPlayer(quest)

    print(
        colored(f"Loaded: {len(quest.locations)} locations, {len(quest.jumps)} jumps", Colors.DIM)
    )
    print_separator()

    step = 0
    while True:
        state = player.get_state()
        step += 1

        # Print header
        print_header(f"Step {step}")

        # Print current text
        print_text(state.text)

        # Print parameters
        print_params(state.params_state)

        # Check for game end
        if state.game_status != GameStatus.RUNNING:
            print_status(state.game_status)
            break

        # No choices means we're stuck (shouldn't happen normally)
        if not state.choices:
            print(colored("\nNo choices available - game ended", Colors.RED))
            break

        # Print choices
        choices = [(i + 1, c.text, c.active) for i, c in enumerate(state.choices)]
        print_choices(choices)

        # Get active choice indices
        active_indices = [i + 1 for i, c in enumerate(state.choices) if c.active]

        if not active_indices:
            print(colored("\nNo active choices - game ended", Colors.RED))
            break

        # Auto-advance if only one choice (like "Next" or "I agree")
        if len(active_indices) == 1:
            choice = state.choices[active_indices[0] - 1]
            if choice.text in ("Next", "I agree", "Go back to ship"):
                print(colored(f"\n[Auto: {choice.text}]", Colors.DIM))
                player.perform_jump(choice.jump_id)
                continue

        # Get player choice
        choice_num = get_player_choice(len(state.choices), active_indices)
        choice = state.choices[choice_num - 1]

        # Perform the jump
        player.perform_jump(choice.jump_id)

    print()
    print(colored(f"Game finished in {step} steps", Colors.DIM))


def main():
    parser = argparse.ArgumentParser(description="Play Space Rangers 2 quests interactively")
    parser.add_argument(
        "quest_file",
        type=Path,
        help="Path to the .qm or .qmm quest file",
    )

    args = parser.parse_args()

    if not args.quest_file.exists():
        print(
            colored(f"Error: Quest file not found: {args.quest_file}", Colors.RED), file=sys.stderr
        )
        sys.exit(1)

    try:
        play_quest(args.quest_file)
    except KeyboardInterrupt:
        print(colored("\n\nGame interrupted", Colors.DIM))
        sys.exit(0)
    except Exception as e:
        print(colored(f"\nError: {e}", Colors.RED), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
