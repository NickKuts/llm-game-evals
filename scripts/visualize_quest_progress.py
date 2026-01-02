#!/usr/bin/env python3
"""
Quest Progress Visualizer for Muzon (Music Festival) Quest

Creates elegant, scalable line charts showing progress toward quest completion.
- Comparison chart: Best attempt from each model
- Per-model charts: All trajectories for each model
"""

import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
from collections import defaultdict

# Color palette optimized for distinction (colorblind-friendly)
MODEL_COLORS = [
    "#2563eb",  # Blue
    "#dc2626",  # Red
    "#16a34a",  # Green
    "#9333ea",  # Purple
    "#ea580c",  # Orange
    "#0891b2",  # Cyan
    "#c026d3",  # Magenta
    "#65a30d",  # Lime
    "#0d9488",  # Teal
    "#e11d48",  # Rose
]

# Lighter versions for multiple trajectories
TRAJECTORY_ALPHAS = [0.9, 0.6, 0.4, 0.3, 0.25, 0.2]

MILESTONE_MARKERS = {
    "guitarist": {"symbol": "G", "order": 1},
    "keyboardist": {"symbol": "K", "order": 2},
    "bass_player": {"symbol": "B", "order": 3},
    "drummer": {"symbol": "D", "order": 4},
    "tattoo": {"symbol": "t", "order": 5},
    "piercing": {"symbol": "p", "order": 6},
    "etiquette": {"symbol": "e", "order": 7},
    "song": {"symbol": "s", "order": 8},
    "win": {"symbol": "W", "order": 0},
}

MILESTONE_WEIGHTS = {
    "guitarist": 15,
    "keyboardist": 15,
    "bass_player": 15,
    "drummer": 15,
    "tattoo": 5,
    "piercing": 5,
    "etiquette": 5,
    "song": 5,
    "win": 5,
}


def detect_band_status(params: list) -> dict:
    """Parse band member status from params."""
    status = {"guitarist": False, "keyboardist": False, "bass_player": False, "drummer": False}
    params_text = " ".join(str(p) for p in params).lower()

    if "guitarist" in params_text and "no guitarist" not in params_text:
        status["guitarist"] = True
    if "keyboardist" in params_text and "no keyboardist" not in params_text:
        status["keyboardist"] = True
    if "bass-player" in params_text and "no bass-player" not in params_text:
        status["bass_player"] = True
    if "drummer" in params_text and "no drummer" not in params_text:
        status["drummer"] = True

    return status


def detect_inventory_items(params: list) -> set:
    """Detect items in inventory from params."""
    items = set()
    params_text = " ".join(str(p) for p in params).lower()

    if "dragon tattoo" in params_text or "guitar tattoo" in params_text:
        items.add("tattoo")
    if "piercing" in params_text:
        items.add("piercing")
    if "etiquette" in params_text or "politeness" in params_text:
        items.add("etiquette")

    return items


def get_clean_model_name(model_name: str) -> str:
    """Clean up model name for display."""
    name = model_name.replace("anthropic/", "").replace("openai/", "").replace("google/", "")
    name = name.replace("moonshotai/", "")
    name = name.replace("-preview", "").replace("-thinking", "")

    # Capitalize nicely
    parts = name.split("-")
    cleaned = " ".join(p.capitalize() for p in parts)
    return cleaned


def parse_attempt(attempt_file: Path) -> dict:
    """Parse an attempt file and extract progress data."""
    with open(attempt_file) as f:
        data = json.load(f)

    history = data.get("history", [])
    model_name = data.get("model", "Unknown")
    display_name = get_clean_model_name(model_name)

    progress_data = {
        "model": model_name,
        "display_name": display_name,
        "file": str(attempt_file),
        "steps": [],
        "progress": [],
        "milestones": [],
        "final_status": data.get("game_status", "UNKNOWN"),
        "total_steps": len(history)
    }

    achieved = set()
    current_progress = 0
    prev_band = {"guitarist": False, "keyboardist": False, "bass_player": False, "drummer": False}
    prev_items = set()

    for step_data in history:
        step_num = step_data.get("step", 0)
        params = step_data.get("params", [])

        # Check band status
        band_status = detect_band_status(params)
        for member in ["guitarist", "keyboardist", "bass_player", "drummer"]:
            if band_status[member] and not prev_band[member] and member not in achieved:
                achieved.add(member)
                current_progress += MILESTONE_WEIGHTS[member]
                progress_data["milestones"].append((step_num, member, current_progress))
        prev_band = band_status.copy()

        # Check inventory items
        current_items = detect_inventory_items(params)
        for item in current_items - prev_items:
            if item not in achieved and item in MILESTONE_WEIGHTS:
                achieved.add(item)
                current_progress += MILESTONE_WEIGHTS[item]
                progress_data["milestones"].append((step_num, item, current_progress))
        prev_items = current_items

        progress_data["steps"].append(step_num)
        progress_data["progress"].append(current_progress)

    if data.get("game_status") == "WIN":
        progress_data["progress"][-1] = 100
        if "win" not in achieved:
            progress_data["milestones"].append((progress_data["steps"][-1], "win", 100))

    return progress_data


def find_all_attempts(results_dir: Path, quest_name: str = "Muzon") -> dict:
    """Find all attempts grouped by model."""
    model_attempts = defaultdict(list)

    for result_dir in sorted(results_dir.glob(f"*{quest_name}*")):
        # Find all run directories
        for run_dir in sorted(result_dir.glob("run_*")):
            # Find all attempt files
            for attempt_file in sorted(run_dir.glob("attempt_*.json")):
                try:
                    progress_data = parse_attempt(attempt_file)
                    model_attempts[progress_data["model"]].append(progress_data)
                except Exception as e:
                    print(f"Error loading {attempt_file}: {e}")

    return model_attempts


def get_best_attempt(attempts: list) -> dict:
    """Get the best attempt (highest final progress, then fewest steps)."""
    return max(attempts, key=lambda x: (x["progress"][-1], -x["total_steps"]))


def create_comparison_chart(
    progress_data_list: list[dict],
    output_file: str,
    title: str = "Muzon Quest - Best Attempts Comparison"
):
    """Create an elegant multi-model progress visualization."""

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafa')

    max_steps = 0

    # Sort by final progress (descending) for legend order
    progress_data_list = sorted(progress_data_list, key=lambda x: -x["progress"][-1])

    for idx, data in enumerate(progress_data_list):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        steps = data["steps"]
        progress = data["progress"]
        max_steps = max(max_steps, max(steps))

        # Main progress line
        ax.plot(steps, progress,
                linewidth=2.5,
                color=color,
                alpha=0.85,
                solid_capstyle='round',
                label=data["display_name"],
                zorder=3)

        # Add milestone markers
        for step, milestone_id, prog in data["milestones"]:
            marker_info = MILESTONE_MARKERS.get(milestone_id, {"symbol": "?", "order": 99})

            ax.plot(step, prog, 'o',
                    markersize=8,
                    color=color,
                    markeredgecolor='white',
                    markeredgewidth=1.5,
                    zorder=5)

            if milestone_id in ["guitarist", "keyboardist", "bass_player", "drummer"]:
                ax.annotate(marker_info["symbol"],
                           xy=(step, prog),
                           fontsize=6,
                           fontweight='bold',
                           color='white',
                           ha='center',
                           va='center',
                           zorder=6)

        # Mark endpoint
        final_step = steps[-1]
        final_progress = progress[-1]

        if data["final_status"] == "WIN":
            ax.plot(final_step, final_progress,
                    marker='*', markersize=18,
                    color=color, markeredgecolor='white', markeredgewidth=1.5,
                    zorder=7)
        else:
            ax.plot(final_step, final_progress,
                    marker='X', markersize=10,
                    color=color, markeredgecolor='white', markeredgewidth=1.5,
                    zorder=7)

    # Reference lines
    ax.axhline(y=100, color='#22c55e', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
    ax.axhline(y=60, color='#a855f7', linestyle=':', linewidth=1.5, alpha=0.4, zorder=1)

    ax.text(max_steps * 1.02, 100, 'WIN', fontsize=9, color='#22c55e', fontweight='bold', va='center')
    ax.text(max_steps * 1.02, 60, 'Full\nBand', fontsize=8, color='#a855f7', va='center', alpha=0.7)

    # Styling
    ax.set_xlabel('Game Steps', fontsize=12, fontweight='medium', color='#374151', labelpad=10)
    ax.set_ylabel('Quest Progress (%)', fontsize=12, fontweight='medium', color='#374151', labelpad=10)
    ax.set_title(title, fontsize=16, fontweight='bold', color='#1f2937', pad=20)

    ax.set_xlim(-2, max_steps * 1.08)
    ax.set_ylim(-2, 110)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#d1d5db')
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color('#e5e7eb')
        spine.set_linewidth(1)

    # Model legend
    legend1 = ax.legend(loc='upper left', fontsize=9, framealpha=0.95,
                        edgecolor='#e5e7eb', title='Models', title_fontsize=10)
    ax.add_artist(legend1)

    # Milestone legend
    milestone_legend = [
        Line2D([0], [0], marker='o', color='#6b7280', markersize=8,
               markeredgecolor='white', markeredgewidth=1, linestyle='None',
               label='G=Guitarist  K=Keyboardist'),
        Line2D([0], [0], marker='o', color='#6b7280', markersize=8,
               markeredgecolor='white', markeredgewidth=1, linestyle='None',
               label='B=Bass       D=Drummer'),
        Line2D([0], [0], marker='X', color='#6b7280', markersize=8,
               markeredgecolor='white', markeredgewidth=1, linestyle='None',
               label='X=Failed     ★=Won'),
    ]

    legend2 = ax.legend(handles=milestone_legend, loc='lower right', fontsize=8,
                        framealpha=0.95, edgecolor='#e5e7eb', title='Milestones',
                        title_fontsize=9, handletextpad=0.5)
    ax.add_artist(legend1)

    # Summary table
    summary_lines = []
    for data in progress_data_list:
        achieved_band = sum(1 for m in ["guitarist", "keyboardist", "bass_player", "drummer"]
                          if any(x[1] == m for x in data["milestones"]))
        status = "W" if data["final_status"] == "WIN" else "F"
        name = data['display_name'][:14]
        summary_lines.append(f"{name:14} {achieved_band}/4 {data['progress'][-1]:3}% {status}")

    summary_text = "Model          Band Prog\n" + "-" * 26 + "\n" + "\n".join(summary_lines)

    ax.text(0.99, 0.50, summary_text,
            transform=ax.transAxes,
            fontsize=8,
            fontfamily='monospace',
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#d1d5db', alpha=0.95, linewidth=1),
            zorder=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Saved: {output_file}")
    return output_file


def create_model_trajectories_chart(
    model_name: str,
    attempts: list[dict],
    output_file: str,
    color: str = "#2563eb"
):
    """Create a chart showing all trajectories for a single model."""

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafa')

    max_steps = 0
    display_name = attempts[0]["display_name"]

    # Sort attempts by progress (best first)
    attempts = sorted(attempts, key=lambda x: -x["progress"][-1])

    for idx, data in enumerate(attempts):
        steps = data["steps"]
        progress = data["progress"]
        max_steps = max(max_steps, max(steps))

        # Vary alpha for different attempts
        alpha = TRAJECTORY_ALPHAS[min(idx, len(TRAJECTORY_ALPHAS) - 1)]
        linewidth = 3.0 if idx == 0 else 2.0

        # Extract run/attempt info from filename
        parts = data["file"].split("/")
        run_info = ""
        for p in parts:
            if p.startswith("run_"):
                run_num = p.replace("run_", "")
            if p.startswith("attempt_"):
                attempt_num = p.replace("attempt_", "").replace(".json", "")
                run_info = f"R{run_num}A{attempt_num}"

        label = f"{run_info}: {data['progress'][-1]}%"
        if data["final_status"] == "WIN":
            label += " ★"

        # Main progress line
        ax.plot(steps, progress,
                linewidth=linewidth,
                color=color,
                alpha=alpha,
                solid_capstyle='round',
                label=label,
                zorder=3 + (len(attempts) - idx))

        # Add milestone markers (only for best attempt to reduce clutter)
        if idx == 0:
            for step, milestone_id, prog in data["milestones"]:
                marker_info = MILESTONE_MARKERS.get(milestone_id, {"symbol": "?", "order": 99})

                ax.plot(step, prog, 'o',
                        markersize=10,
                        color=color,
                        markeredgecolor='white',
                        markeredgewidth=2,
                        zorder=5)

                if milestone_id in ["guitarist", "keyboardist", "bass_player", "drummer"]:
                    ax.annotate(marker_info["symbol"],
                               xy=(step, prog),
                               fontsize=7,
                               fontweight='bold',
                               color='white',
                               ha='center',
                               va='center',
                               zorder=6)

        # Mark endpoint
        final_step = steps[-1]
        final_progress = progress[-1]
        marker_size = 15 if idx == 0 else 8

        if data["final_status"] == "WIN":
            ax.plot(final_step, final_progress,
                    marker='*', markersize=marker_size + 5,
                    color=color, markeredgecolor='white', markeredgewidth=1.5,
                    alpha=alpha, zorder=7)
        else:
            ax.plot(final_step, final_progress,
                    marker='X', markersize=marker_size,
                    color=color, markeredgecolor='white', markeredgewidth=1.5,
                    alpha=alpha, zorder=7)

    # Reference lines
    ax.axhline(y=100, color='#22c55e', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
    ax.axhline(y=60, color='#a855f7', linestyle=':', linewidth=1.5, alpha=0.4, zorder=1)

    ax.text(max_steps * 1.02, 100, 'WIN', fontsize=9, color='#22c55e', fontweight='bold', va='center')
    ax.text(max_steps * 1.02, 60, 'Full\nBand', fontsize=8, color='#a855f7', va='center', alpha=0.7)

    # Styling
    ax.set_xlabel('Game Steps', fontsize=12, fontweight='medium', color='#374151', labelpad=10)
    ax.set_ylabel('Quest Progress (%)', fontsize=12, fontweight='medium', color='#374151', labelpad=10)
    ax.set_title(f'Muzon Quest - {display_name} (All Attempts)',
                 fontsize=16, fontweight='bold', color='#1f2937', pad=20)

    ax.set_xlim(-2, max_steps * 1.08)
    ax.set_ylim(-2, 110)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#d1d5db')
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color('#e5e7eb')
        spine.set_linewidth(1)

    # Legend showing all attempts
    legend1 = ax.legend(loc='upper left', fontsize=9, framealpha=0.95,
                        edgecolor='#e5e7eb', title='Attempts (Run/Attempt: Progress)',
                        title_fontsize=10)

    # Stats box
    wins = sum(1 for a in attempts if a["final_status"] == "WIN")
    best_progress = max(a["progress"][-1] for a in attempts)
    avg_progress = sum(a["progress"][-1] for a in attempts) / len(attempts)

    stats_text = f"Total Attempts: {len(attempts)}\n"
    stats_text += f"Wins: {wins}\n"
    stats_text += f"Best: {best_progress}%\n"
    stats_text += f"Average: {avg_progress:.0f}%"

    ax.text(0.99, 0.35, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            fontfamily='monospace',
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#d1d5db', alpha=0.95, linewidth=1),
            zorder=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Saved: {output_file}")
    return output_file


def main():
    results_dir = Path("results/runs")
    output_dir = Path("results/charts")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Finding all Muzon attempts...")
    model_attempts = find_all_attempts(results_dir, "Muzon")

    if not model_attempts:
        print("No Muzon results found!")
        return

    print(f"\nFound {len(model_attempts)} models:")
    for model, attempts in model_attempts.items():
        best = get_best_attempt(attempts)
        print(f"  {best['display_name']}: {len(attempts)} attempts, best={best['progress'][-1]}%")

    # 1. Create comparison chart with best attempts
    print("\n1. Creating best attempts comparison chart...")
    best_attempts = [get_best_attempt(attempts) for attempts in model_attempts.values()]
    create_comparison_chart(best_attempts, str(output_dir / "muzon_best_comparison.png"))

    # 2. Create per-model trajectory charts
    print("\n2. Creating per-model trajectory charts...")
    for idx, (model, attempts) in enumerate(sorted(model_attempts.items())):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        display_name = attempts[0]["display_name"]
        safe_name = display_name.lower().replace(" ", "_").replace(".", "")
        output_file = output_dir / f"muzon_trajectories_{safe_name}.png"
        create_model_trajectories_chart(model, attempts, str(output_file), color)

    print("\nDone!")


if __name__ == "__main__":
    main()
