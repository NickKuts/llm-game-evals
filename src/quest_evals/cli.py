"""Command-line interface for quest evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .llm import OpenRouterLLM
from .model_configs import MODELS, ReasoningField, get_config, list_models
from .results import ResultLogger
from .runner import (
    load_quest,
    run_evaluation,
)

# Default paths
DEFAULT_QUESTS_DIR = Path("assets/json")

# Model shortcuts (from model_configs.py)
SUGGESTED_MODELS = list(MODELS.keys())[:6]  # Top 6 for help text


def list_quests(quests_dir: Path) -> list[str]:
    """List available quests."""
    quests = []
    for f in sorted(quests_dir.glob("*.json")):
        quests.append(f.stem)
    return quests


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM performance on Space Rangers 2 quests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s Muzon_eng                    # Run Muzon quest with default model
  %(prog)s Muzon_eng -m gpt-4o-mini     # Specify model shortcut
  %(prog)s assets/json/Muzon_eng.json   # Full path also works
  %(prog)s --list                       # List available quests

Model shortcuts: """
        + ", ".join(SUGGESTED_MODELS),
    )
    parser.add_argument(
        "quest",
        type=str,
        nargs="?",
        help="Quest name (e.g. Muzon_eng) or path to .json/.qm file",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available quests",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model configurations",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-5-mini",
        help=f"Model shortcut or OpenRouter ID. Shortcuts: {', '.join(SUGGESTED_MODELS)}",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum game steps before timeout (default: 500)",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=60,
        help="Maximum LLM API calls (default: 60)",
    )
    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=1,
        help="Number of independent runs to check consistency (default: 1)",
    )
    parser.add_argument(
        "--attempts",
        "-a",
        type=int,
        default=1,
        help="Number of attempts per run with reflection on failure (default: 1)",
    )
    parser.add_argument(
        "--reflection-words",
        type=int,
        default=200,
        help="Max words for failure reflections (default: 200)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print very detailed output (full text)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="No progress output, only final result",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Don't save results to disk",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default="results",
        help="Directory for results (default: results/)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint if available",
    )
    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        list_models()
        sys.exit(0)

    # Handle --list
    if args.list:
        quests = list_quests(DEFAULT_QUESTS_DIR)
        print(f"Available quests ({len(quests)}):")
        for q in quests:
            print(f"  {q}")
        sys.exit(0)

    # Require quest argument
    if not args.quest:
        parser.print_help()
        sys.exit(1)

    # Resolve quest path - accept name or full path
    quest_input = args.quest
    if quest_input.endswith(".json"):
        quest_path = Path(quest_input)
    else:
        # Treat as quest name, look in assets/json/
        quest_path = DEFAULT_QUESTS_DIR / f"{quest_input}.json"

    if not quest_path.exists():
        print(f"Error: Quest not found: {quest_path}", file=sys.stderr)
        print("Use --list to see available quests", file=sys.stderr)
        sys.exit(1)

    # Resolve model config
    try:
        model_config = get_config(args.model)
        model_id = model_config.model_id
        extra_body = model_config.extra_body
        model_display = model_config.display_name
        reasoning_field = model_config.reasoning_field
    except ValueError:
        # Not a shortcut, use as raw model ID
        model_id = args.model
        extra_body = {}
        model_display = args.model
        reasoning_field = ReasoningField.NONE

    print(f"Loading quest: {quest_path.stem}")
    try:
        quest = load_quest(quest_path)
    except Exception as e:
        print(f"Error loading quest: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Quest loaded: {len(quest.locations)} locations, {len(quest.jumps)} jumps")
    print(
        f"Task: {quest.task_text[:100]}..."
        if len(quest.task_text) > 100
        else f"Task: {quest.task_text}"
    )
    print()

    print(f"Using model: {model_display} (via OpenRouter)")
    if extra_body:
        print(f"  Config: {extra_body}")
    try:
        llm = OpenRouterLLM(
            model=model_id,
            extra_body=extra_body,
            reasoning_field=reasoning_field,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Build config string
    config_parts = [f"max {args.max_steps} steps", f"{args.max_llm_calls} LLM calls"]
    if args.runs > 1:
        config_parts.append(f"{args.runs} runs")
    if args.attempts > 1:
        config_parts.append(f"{args.attempts} attempts/run")
    print(f"Running evaluation ({', '.join(config_parts)})...")
    print("=" * 60)

    # Run the evaluation
    result = run_evaluation(
        quest=quest,
        llm=llm,
        quest_file=str(quest_path),
        num_runs=args.runs,
        max_attempts=args.attempts,
        max_steps=args.max_steps,
        max_llm_calls=args.max_llm_calls,
        verbose=args.verbose,
        show_progress=not args.quiet,
        reflection_max_words=args.reflection_words,
        results_dir=args.results_dir if not args.no_log else None,
        resume=args.resume,
    )

    # Print final summary
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Quest: {result.quest_name}")
    print(f"Model: {result.model}")
    print(f"Success rate: {result.successful_runs}/{result.total_runs} ({result.success_rate:.0%})")
    print(f"Total tokens: {result.total_tokens_in:,} in, {result.total_tokens_out:,} out")

    # Print per-run summary
    if len(result.runs) > 1 or args.attempts > 1:
        print()
        print("Runs:")
        for run_idx, run in enumerate(result.runs, 1):
            if run.success:
                status = f"✓ SUCCESS (attempt {run.successful_attempt})"
            else:
                status = f"✗ FAILED ({len(run.attempts)} attempts)"
            print(f"  Run {run_idx}: {status}")

            # Show attempt details if multiple attempts
            if args.attempts > 1:
                for attempt in run.attempts:
                    a_status = "✓" if attempt.completed else "✗"
                    print(
                        f"    Attempt {attempt.attempt}: {a_status} {attempt.game_status.name} ({attempt.steps} steps)"
                    )
                    if attempt.failure_reflection:
                        first_line = attempt.failure_reflection.split("\n")[0][:60]
                        print(f"      → {first_line}...")

    # Log results
    if not args.no_log:
        logger = ResultLogger(args.results_dir)
        eval_id = logger.log_evaluation_result(result)
        print()
        print(f"Results saved: {args.results_dir}/runs/{eval_id}/")


if __name__ == "__main__":
    main()
