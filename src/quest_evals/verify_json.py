"""Verification script to ensure JSON conversion preserves quest behavior."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from .models import Quest
from .player import GameStatus, QuestPlayer
from .qm_parser import parse_qm
from .quest_json import dict_to_quest, quest_to_dict, save_quest_json


# ANSI colors
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    RED = "\033[91m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"


def compare_quests(q1: Quest, q2: Quest) -> list[str]:
    """Compare two Quest objects and return list of differences."""
    diffs = []

    # Compare basic properties
    if q1.giving_race != q2.giving_race:
        diffs.append(f"giving_race: {q1.giving_race} vs {q2.giving_race}")
    if q1.when_done != q2.when_done:
        diffs.append(f"when_done: {q1.when_done} vs {q2.when_done}")
    if q1.planet_race != q2.planet_race:
        diffs.append(f"planet_race: {q1.planet_race} vs {q2.planet_race}")
    if q1.player_career != q2.player_career:
        diffs.append(f"player_career: {q1.player_career} vs {q2.player_career}")
    if q1.player_race != q2.player_race:
        diffs.append(f"player_race: {q1.player_race} vs {q2.player_race}")
    if q1.default_jump_count_limit != q2.default_jump_count_limit:
        diffs.append(
            f"default_jump_count_limit: {q1.default_jump_count_limit} vs {q2.default_jump_count_limit}"
        )
    if q1.hardness != q2.hardness:
        diffs.append(f"hardness: {q1.hardness} vs {q2.hardness}")
    if q1.params_count != q2.params_count:
        diffs.append(f"params_count: {q1.params_count} vs {q2.params_count}")
    if q1.task_text != q2.task_text:
        diffs.append("task_text differs")
    if q1.success_text != q2.success_text:
        diffs.append("success_text differs")

    # Compare params
    if len(q1.params) != len(q2.params):
        diffs.append(f"param count: {len(q1.params)} vs {len(q2.params)}")
    else:
        for i, (p1, p2) in enumerate(zip(q1.params, q2.params, strict=False)):
            if p1.name != p2.name:
                diffs.append(f"param[{i}].name: '{p1.name}' vs '{p2.name}'")
            if p1.min_val != p2.min_val:
                diffs.append(f"param[{i}].min_val: {p1.min_val} vs {p2.min_val}")
            if p1.max_val != p2.max_val:
                diffs.append(f"param[{i}].max_val: {p1.max_val} vs {p2.max_val}")
            if p1.active != p2.active:
                diffs.append(f"param[{i}].active: {p1.active} vs {p2.active}")
            if p1.param_type != p2.param_type:
                diffs.append(f"param[{i}].param_type: {p1.param_type} vs {p2.param_type}")
            if p1.crit_type != p2.crit_type:
                diffs.append(f"param[{i}].crit_type: {p1.crit_type} vs {p2.crit_type}")
            if p1.starting != p2.starting:
                diffs.append(f"param[{i}].starting: '{p1.starting}' vs '{p2.starting}'")

    # Compare locations
    if len(q1.locations) != len(q2.locations):
        diffs.append(f"location count: {len(q1.locations)} vs {len(q2.locations)}")
    else:
        for i, (l1, l2) in enumerate(zip(q1.locations, q2.locations, strict=False)):
            if l1.id != l2.id:
                diffs.append(f"loc[{i}].id: {l1.id} vs {l2.id}")
            if l1.is_starting != l2.is_starting:
                diffs.append(f"loc[{i}].is_starting: {l1.is_starting} vs {l2.is_starting}")
            if l1.is_success != l2.is_success:
                diffs.append(f"loc[{i}].is_success: {l1.is_success} vs {l2.is_success}")
            if l1.is_fail != l2.is_fail:
                diffs.append(f"loc[{i}].is_fail: {l1.is_fail} vs {l2.is_fail}")
            if l1.is_fail_deadly != l2.is_fail_deadly:
                diffs.append(f"loc[{i}].is_fail_deadly: {l1.is_fail_deadly} vs {l2.is_fail_deadly}")
            # Compare texts (filter empty)
            t1 = [t for t in l1.texts if t]
            t2 = [t for t in l2.texts if t]
            if t1 != t2:
                diffs.append(f"loc[{i}].texts differ")

    # Compare jumps
    if len(q1.jumps) != len(q2.jumps):
        diffs.append(f"jump count: {len(q1.jumps)} vs {len(q2.jumps)}")
    else:
        for i, (j1, j2) in enumerate(zip(q1.jumps, q2.jumps, strict=False)):
            if j1.id != j2.id:
                diffs.append(f"jump[{i}].id: {j1.id} vs {j2.id}")
            if j1.from_location_id != j2.from_location_id:
                diffs.append(f"jump[{i}].from: {j1.from_location_id} vs {j2.from_location_id}")
            if j1.to_location_id != j2.to_location_id:
                diffs.append(f"jump[{i}].to: {j1.to_location_id} vs {j2.to_location_id}")
            if j1.text != j2.text:
                diffs.append(f"jump[{i}].text: '{j1.text[:30]}...' vs '{j2.text[:30]}...'")
            if j1.formula_to_pass != j2.formula_to_pass:
                diffs.append(f"jump[{i}].formula: '{j1.formula_to_pass}' vs '{j2.formula_to_pass}'")

    return diffs


def compare_game_states(player1: QuestPlayer, player2: QuestPlayer, step: int) -> list[str]:
    """Compare the current game states of two players."""
    diffs = []
    s1 = player1.get_state()
    s2 = player2.get_state()

    if s1.game_status != s2.game_status:
        diffs.append(f"Step {step}: game_status {s1.game_status.name} vs {s2.game_status.name}")

    if s1.text != s2.text:
        # Find first difference position for better debugging
        for i, (c1, c2) in enumerate(zip(s1.text, s2.text, strict=False)):
            if c1 != c2:
                context = s1.text[max(0, i - 20) : i + 20].replace("\n", " ")
                diffs.append(f"Step {step}: text differs at char {i} near '{context}'")
                break
        else:
            # Lengths differ
            diffs.append(f"Step {step}: text length differs ({len(s1.text)} vs {len(s2.text)})")

    if s1.params_state != s2.params_state:
        diffs.append(f"Step {step}: params_state differs")

    if len(s1.choices) != len(s2.choices):
        diffs.append(f"Step {step}: choice count {len(s1.choices)} vs {len(s2.choices)}")
    else:
        for i, (c1, c2) in enumerate(zip(s1.choices, s2.choices, strict=False)):
            if c1.text != c2.text:
                diffs.append(f"Step {step}: choice[{i}].text differs")
            if c1.active != c2.active:
                diffs.append(f"Step {step}: choice[{i}].active {c1.active} vs {c2.active}")
            if c1.jump_id != c2.jump_id:
                diffs.append(f"Step {step}: choice[{i}].jump_id {c1.jump_id} vs {c2.jump_id}")

    return diffs


def verify_quest_file(qm_path: Path, max_steps: int = 50) -> tuple[bool, list[str]]:
    """Verify a single quest file by comparing QM and JSON behavior.

    Returns (success, list of differences/errors).
    """
    errors = []

    # Load from QM
    try:
        with open(qm_path, "rb") as f:
            qm_data = f.read()
        quest_from_qm = parse_qm(qm_data)
    except Exception as e:
        return False, [f"Failed to parse QM: {e}"]

    # Convert to dict and back
    try:
        quest_dict = quest_to_dict(quest_from_qm)
        quest_from_json = dict_to_quest(quest_dict)
    except Exception as e:
        return False, [f"Failed in dict conversion: {e}"]

    # Compare static quest structures
    struct_diffs = compare_quests(quest_from_qm, quest_from_json)
    if struct_diffs:
        errors.extend([f"Structure: {d}" for d in struct_diffs])

    # Run QM player first and record states/choices
    seed = 12345
    random.seed(seed)
    player_qm = QuestPlayer(quest_from_qm)

    qm_states = []  # List of (text, params_state, choices)
    qm_jumps = []  # List of jump_ids taken

    for step in range(max_steps):
        state = player_qm.get_state()
        qm_states.append(
            (state.text, state.params_state, [(c.text, c.jump_id, c.active) for c in state.choices])
        )

        if state.game_status != GameStatus.RUNNING:
            break

        active_choices = [(i, c) for i, c in enumerate(state.choices) if c.active]
        if not active_choices:
            break

        jump_id = active_choices[0][1].jump_id
        qm_jumps.append(jump_id)

        try:
            player_qm.perform_jump(jump_id)
        except Exception as e:
            errors.append(f"QM Step {step}: jump failed: {e}")
            break

    # Now run JSON player with same seed and compare
    random.seed(seed)
    player_json = QuestPlayer(quest_from_json)

    for step in range(len(qm_states)):
        state = player_json.get_state()
        qm_text, qm_params, qm_choices = qm_states[step]

        # Compare text
        if state.text != qm_text:
            diff_pos = next(
                (i for i, (a, b) in enumerate(zip(state.text, qm_text, strict=False)) if a != b),
                min(len(state.text), len(qm_text)),
            )
            context = qm_text[max(0, diff_pos - 10) : diff_pos + 30] if qm_text else ""
            errors.append(f"Step {step}: text differs at char {diff_pos} near '{context}'")

        # Compare params_state
        if state.params_state != qm_params:
            errors.append(f"Step {step}: params_state differs")

        # Compare choices
        json_choices = [(c.text, c.jump_id, c.active) for c in state.choices]
        if len(json_choices) != len(qm_choices):
            errors.append(f"Step {step}: choice count {len(qm_choices)} vs {len(json_choices)}")
        else:
            for i, (qc, jc) in enumerate(zip(qm_choices, json_choices, strict=False)):
                if qc[0] != jc[0]:
                    errors.append(f"Step {step}: choice[{i}].text differs")
                if qc[1] != jc[1]:
                    errors.append(f"Step {step}: choice[{i}].jump_id {qc[1]} vs {jc[1]}")
                if qc[2] != jc[2]:
                    errors.append(f"Step {step}: choice[{i}].active {qc[2]} vs {jc[2]}")

        # Take same jump if we have one
        if step < len(qm_jumps):
            jump_id = qm_jumps[step]
            # Find corresponding jump in JSON player's choices
            json_jump = None
            for c in state.choices:
                if c.active:
                    json_jump = c.jump_id
                    break

            if json_jump is None:
                errors.append(f"Step {step}: no active choices in JSON player")
                break

            try:
                player_json.perform_jump(json_jump)
            except Exception as e:
                errors.append(f"JSON Step {step}: jump failed: {e}")
                break

    return len(errors) == 0, errors


def verify_all_quests(assets_dir: Path, max_steps: int = 50) -> dict[str, tuple[bool, list[str]]]:
    """Verify all .qm files in the assets directory."""
    results = {}

    for qm_path in sorted(assets_dir.glob("*.qm")):
        print(f"\n{C.BOLD}Verifying {qm_path.name}...{C.RESET}")
        success, errors = verify_quest_file(qm_path, max_steps)
        results[qm_path.name] = (success, errors)

        if success:
            print(f"  {C.GREEN}✓ PASS{C.RESET} - {max_steps} steps verified")
        else:
            print(f"  {C.RED}✗ FAIL{C.RESET} - {len(errors)} issue(s)")
            for err in errors[:5]:  # Show first 5 errors
                print(f"    {C.DIM}- {err}{C.RESET}")
            if len(errors) > 5:
                print(f"    {C.DIM}... and {len(errors) - 5} more{C.RESET}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify JSON conversion preserves quest behavior")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("assets"),
        help="Path to quest file or directory (default: assets/)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Maximum steps to verify per quest (default: 50)",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Also save JSON files alongside QM files",
    )

    args = parser.parse_args()

    print(f"{C.BOLD}{C.CYAN}═══ Quest JSON Verification ═══{C.RESET}")

    if args.path.is_file():
        success, errors = verify_quest_file(args.path, args.steps)

        if success:
            print(f"\n{C.GREEN}✓ PASS{C.RESET} - Quest behavior preserved")
        else:
            print(f"\n{C.RED}✗ FAIL{C.RESET} - {len(errors)} issue(s):")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)

        if args.convert:
            json_path = args.path.with_suffix(".json")
            with open(args.path, "rb") as f:
                quest = parse_qm(f.read())
            save_quest_json(quest, json_path)
            print(f"\n{C.DIM}Saved: {json_path}{C.RESET}")

    elif args.path.is_dir():
        results = verify_all_quests(args.path, args.steps)

        passed = sum(1 for s, _ in results.values() if s)
        total = len(results)

        print(f"\n{C.BOLD}═══ Summary ═══{C.RESET}")
        print(f"Passed: {passed}/{total}")

        if passed == total:
            print(f"{C.GREEN}All quests verified successfully!{C.RESET}")
        else:
            print(f"{C.RED}Some quests have issues.{C.RESET}")
            sys.exit(1)

        if args.convert:
            print(f"\n{C.DIM}Converting to JSON...{C.RESET}")
            for qm_path in args.path.glob("*.qm"):
                json_path = qm_path.with_suffix(".json")
                with open(qm_path, "rb") as f:
                    quest = parse_qm(f.read())
                save_quest_json(quest, json_path)
                print(f"  {json_path.name}")
    else:
        print(f"{C.RED}Error: Path not found: {args.path}{C.RESET}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
