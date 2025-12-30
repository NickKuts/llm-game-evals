"""Result logging and storage for quest evaluations."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .player import GameStatus
from .runner import EvalResult, MultiAttemptResult, MultiRunResult, StepRecord


def game_status_to_str(status: GameStatus) -> str:
    """Convert GameStatus enum to string."""
    return status.name


def step_record_to_dict(record: StepRecord) -> dict[str, Any]:
    """Convert StepRecord to serializable dict."""
    data = {
        "step": record.step,
        "location_text": record.location_text,
        "params": record.params,
        "choices": record.choices,
        "llm_response": record.llm_response,
        "reasoning": record.reasoning,
        "chosen_index": record.chosen_index,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "was_auto": record.was_auto,
    }
    # Only include thinking_tokens if present (avoid clutter for non-thinking models)
    if record.thinking_tokens:
        data["thinking_tokens"] = record.thinking_tokens
    return data


def eval_result_to_dict(
    result: EvalResult,
    model: str,
    quest_file: str,
    run_id: str,
) -> dict[str, Any]:
    """Convert EvalResult to a fully serializable dict for logging."""
    data = {
        # Metadata
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model,
        "quest_file": quest_file,
        # Results
        "quest_name": result.quest_name,
        "completed": result.completed,
        "game_status": game_status_to_str(result.game_status),
        "steps": result.steps,
        "llm_calls": result.llm_calls,
        "total_input_tokens": result.total_input_tokens,
        "total_output_tokens": result.total_output_tokens,
        "error": result.error,
        "attempt": result.attempt,
        # Full trajectory
        "history": [step_record_to_dict(s) for s in result.history],
    }

    # Only include failure_reflection if present
    if result.failure_reflection:
        data["failure_reflection"] = result.failure_reflection

    return data


def generate_run_id(model: str) -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    # Clean model name for filename
    model_clean = model.replace("/", "_").replace(":", "_")
    return f"{timestamp}_{model_clean}"


class ResultLogger:
    """Logs evaluation results to JSONL files."""

    def __init__(self, results_dir: str | Path = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirs for organization
        (self.results_dir / "runs").mkdir(exist_ok=True)
        (self.results_dir / "summaries").mkdir(exist_ok=True)

    def get_runs_file(self) -> Path:
        """Get the main JSONL file for all runs."""
        return self.results_dir / "runs" / "all_runs.jsonl"

    def log_result(
        self,
        result: EvalResult,
        model: str,
        quest_file: str,
        run_id: str | None = None,
    ) -> str:
        """Log a single evaluation result.

        Returns:
            The run_id used
        """
        if run_id is None:
            run_id = generate_run_id(model)

        record = eval_result_to_dict(result, model, quest_file, run_id)

        # Append to JSONL file
        runs_file = self.get_runs_file()
        with open(runs_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Also save individual run file for easy inspection
        run_file = self.results_dir / "runs" / f"{run_id}.json"
        with open(run_file, "w") as f:
            json.dump(record, f, indent=2)

        return run_id

    def log_multi_attempt_result(
        self,
        result: MultiAttemptResult,
        quest_file: str,
    ) -> str:
        """Log a multi-attempt evaluation result.

        Creates a folder with:
        - summary.json: Overall result
        - attempt_1.json, attempt_2.json, etc.: Individual attempts

        Returns:
            The run_id used (folder name)
        """
        run_id = generate_run_id(result.model)

        # Create run folder
        run_dir = self.results_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary = {
            "run_id": run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "model": result.model,
            "quest_file": quest_file,
            "quest_name": result.quest_name,
            "success": result.success,
            "successful_attempt": result.successful_attempt,
            "total_attempts": len(result.attempts),
            "total_tokens_in": result.total_tokens_in,
            "total_tokens_out": result.total_tokens_out,
            "attempts_summary": [
                {
                    "attempt": a.attempt,
                    "completed": a.completed,
                    "game_status": game_status_to_str(a.game_status),
                    "steps": a.steps,
                    "llm_calls": a.llm_calls,
                    "failure_reflection": a.failure_reflection,
                }
                for a in result.attempts
            ],
        }

        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save individual attempts
        for attempt in result.attempts:
            attempt_data = eval_result_to_dict(
                result=attempt,
                model=result.model,
                quest_file=quest_file,
                run_id=f"{run_id}_attempt{attempt.attempt}",
            )
            with open(run_dir / f"attempt_{attempt.attempt}.json", "w") as f:
                json.dump(attempt_data, f, indent=2)

        # Also log to the main JSONL for aggregate stats
        runs_file = self.get_runs_file()
        with open(runs_file, "a") as f:
            f.write(json.dumps(summary) + "\n")

        return run_id

    def log_evaluation_result(
        self,
        result: MultiRunResult,
    ) -> str:
        """Log a full evaluation result with multiple runs.

        Creates a folder structure:
        - eval_id/
          - summary.json (overall stats)
          - run_1/
            - summary.json
            - attempt_1.json
            - attempt_2.json
          - run_2/
            - ...

        Returns:
            The evaluation ID (folder name)
        """
        # Generate eval ID with quest name for easy identification
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        model_clean = result.model.replace("/", "_").replace(":", "_")
        quest_clean = Path(result.quest_file).stem
        eval_id = f"{timestamp}_{model_clean}_{quest_clean}"

        # Create evaluation folder
        eval_dir = self.results_dir / "runs" / eval_id
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Save overall summary
        overall_summary = {
            "eval_id": eval_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "model": result.model,
            "quest_file": result.quest_file,
            "quest_name": result.quest_name,
            "total_runs": result.total_runs,
            "successful_runs": result.successful_runs,
            "success_rate": result.success_rate,
            "total_tokens_in": result.total_tokens_in,
            "total_tokens_out": result.total_tokens_out,
            "runs_summary": [],
        }

        # Process each run
        for run_idx, run in enumerate(result.runs, 1):
            run_dir = eval_dir / f"run_{run_idx}"
            run_dir.mkdir(exist_ok=True)

            # Run summary
            run_summary = {
                "run": run_idx,
                "success": run.success,
                "successful_attempt": run.successful_attempt,
                "total_attempts": len(run.attempts),
                "total_tokens_in": run.total_tokens_in,
                "total_tokens_out": run.total_tokens_out,
                "attempts_summary": [
                    {
                        "attempt": a.attempt,
                        "completed": a.completed,
                        "game_status": game_status_to_str(a.game_status),
                        "steps": a.steps,
                        "llm_calls": a.llm_calls,
                        "failure_reflection": a.failure_reflection,
                    }
                    for a in run.attempts
                ],
            }

            with open(run_dir / "summary.json", "w") as f:
                json.dump(run_summary, f, indent=2)

            # Save individual attempts
            for attempt in run.attempts:
                attempt_data = eval_result_to_dict(
                    result=attempt,
                    model=result.model,
                    quest_file=result.quest_file,
                    run_id=f"{eval_id}_run{run_idx}_attempt{attempt.attempt}",
                )
                with open(run_dir / f"attempt_{attempt.attempt}.json", "w") as f:
                    json.dump(attempt_data, f, indent=2)

            # Add to overall summary
            overall_summary["runs_summary"].append(
                {
                    "run": run_idx,
                    "success": run.success,
                    "successful_attempt": run.successful_attempt,
                    "total_attempts": len(run.attempts),
                }
            )

        # Save overall summary
        with open(eval_dir / "summary.json", "w") as f:
            json.dump(overall_summary, f, indent=2)

        # Also append to JSONL for aggregate tracking
        runs_file = self.get_runs_file()
        with open(runs_file, "a") as f:
            f.write(json.dumps(overall_summary) + "\n")

        return eval_id

    def load_all_runs(self) -> list[dict[str, Any]]:
        """Load all runs from the JSONL file."""
        runs_file = self.get_runs_file()
        if not runs_file.exists():
            return []

        runs = []
        with open(runs_file) as f:
            for line in f:
                if line.strip():
                    runs.append(json.loads(line))
        return runs

    def generate_summary(self) -> dict[str, Any]:
        """Generate a summary of all runs."""
        runs = self.load_all_runs()

        if not runs:
            return {"total_runs": 0}

        # Group by model
        by_model: dict[str, list[dict]] = {}
        for run in runs:
            model = run["model"]
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(run)

        # Calculate stats per model
        model_stats = {}
        for model, model_runs in by_model.items():
            completed = sum(1 for r in model_runs if r["completed"])
            total_steps = sum(r["steps"] for r in model_runs)
            total_input = sum(r["total_input_tokens"] for r in model_runs)
            total_output = sum(r["total_output_tokens"] for r in model_runs)

            model_stats[model] = {
                "total_runs": len(model_runs),
                "completed": completed,
                "completion_rate": completed / len(model_runs) if model_runs else 0,
                "avg_steps": total_steps / len(model_runs) if model_runs else 0,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "status_counts": {},
            }

            # Count statuses
            for run in model_runs:
                status = run["game_status"]
                model_stats[model]["status_counts"][status] = (
                    model_stats[model]["status_counts"].get(status, 0) + 1
                )

        summary = {
            "generated_at": datetime.now(UTC).isoformat(),
            "total_runs": len(runs),
            "models": model_stats,
        }

        # Save summary
        summary_file = self.results_dir / "summaries" / "latest.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def print_summary(self) -> None:
        """Print a formatted summary to console."""
        summary = self.generate_summary()

        print(f"\n{'=' * 60}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total runs: {summary['total_runs']}")

        for model, stats in summary.get("models", {}).items():
            print(f"\n{model}:")
            print(f"  Runs: {stats['total_runs']}")
            print(f"  Completion rate: {stats['completion_rate']:.1%}")
            print(f"  Avg steps: {stats['avg_steps']:.1f}")
            print(f"  Tokens: {stats['total_input_tokens']} in, {stats['total_output_tokens']} out")
            print(f"  Outcomes: {stats['status_counts']}")
