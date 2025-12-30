"""JSON format for Space Rangers quests.

This module provides conversion between the binary QM/QMM format and a
human-readable JSON format that preserves all quest behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    Jump,
    JumpParameterCondition,
    Location,
    ParamCritType,
    ParameterChange,
    ParameterShowingType,
    ParamShowingInfo,
    ParamType,
    QMParam,
    Quest,
    QuestStrings,
    WhenDone,
)
from .qm_parser import parse_qm

# Schema version for future compatibility
SCHEMA_VERSION = "1.0"


def quest_to_dict(quest: Quest) -> dict[str, Any]:
    """Convert a Quest object to a JSON-serializable dictionary."""
    return {
        "$schema": "quest-evals/quest-schema-v1",
        "version": SCHEMA_VERSION,
        "meta": {
            "giving_race": quest.giving_race,
            "when_done": quest.when_done.name.lower(),
            "planet_race": quest.planet_race,
            "player_career": quest.player_career,
            "player_race": quest.player_race,
            "default_jump_count_limit": quest.default_jump_count_limit,
            "hardness": quest.hardness,
            "screen_size": [quest.screen_size_x, quest.screen_size_y],
            "version_info": {
                "major": quest.major_version,
                "minor": quest.minor_version,
                "changelog": quest.changelog,
            }
            if quest.major_version is not None
            else None,
        },
        "strings": {
            "to_star": quest.strings.to_star,
            "to_planet": quest.strings.to_planet,
            "from_star": quest.strings.from_star,
            "from_planet": quest.strings.from_planet,
            "ranger": quest.strings.ranger,
            "date": quest.strings.date,
            "money": quest.strings.money,
            "parsec": quest.strings.parsec,
            "artefact": quest.strings.artefact,
        },
        "task_text": quest.task_text,
        "success_text": quest.success_text,
        "params": [_param_to_dict(p) for p in quest.params],
        "locations": [_location_to_dict(loc) for loc in quest.locations],
        "jumps": [_jump_to_dict(j, quest.params) for j in quest.jumps],
    }


def _param_to_dict(param: QMParam) -> dict[str, Any]:
    """Convert a parameter to dict."""
    result: dict[str, Any] = {
        "name": param.name,
        "min": param.min_val,
        "max": param.max_val,
        "active": param.active,
    }

    # Only include non-default values to keep JSON clean
    if param.param_type != ParamType.NORMAL:
        result["type"] = param.param_type.name.lower()
    if param.crit_type != ParamCritType.MAX:
        result["crit_type"] = param.crit_type.name.lower()
    if param.show_when_zero:
        result["show_when_zero"] = True
    if param.is_money:
        result["is_money"] = True
    if param.starting:
        result["starting"] = param.starting
    if param.crit_value_string:
        result["crit_text"] = param.crit_value_string
    if param.showing_info:
        result["display_ranges"] = [
            {"from": si.from_val, "to": si.to_val, "text": si.text} for si in param.showing_info
        ]
    if param.img:
        result["img"] = param.img
    if param.sound:
        result["sound"] = param.sound
    if param.track:
        result["track"] = param.track

    return result


def _param_change_to_dict(change: ParameterChange, param_idx: int) -> dict[str, Any] | None:
    """Convert a parameter change to dict, or None if it's a no-op."""
    # Check if this is a meaningful change
    has_change = (
        change.change != 0
        or change.is_change_percentage
        or change.is_change_value
        or change.is_change_formula
        or change.showing_type != ParameterShowingType.NO_CHANGE
        or change.crit_text
    )

    if not has_change:
        return None

    result: dict[str, Any] = {"param": param_idx}

    if change.is_change_formula and change.changing_formula:
        result["formula"] = change.changing_formula
    elif change.is_change_value:
        result["set"] = change.change
    elif change.is_change_percentage:
        result["percent"] = change.change
    elif change.change != 0:
        result["add"] = change.change

    if change.showing_type == ParameterShowingType.SHOW:
        result["show"] = True
    elif change.showing_type == ParameterShowingType.HIDE:
        result["hide"] = True

    if change.crit_text:
        result["crit_text"] = change.crit_text
    if change.img:
        result["img"] = change.img
    if change.sound:
        result["sound"] = change.sound
    if change.track:
        result["track"] = change.track

    return result


def _condition_to_dict(
    cond: JumpParameterCondition, param: QMParam, param_idx: int
) -> dict[str, Any] | None:
    """Convert a jump condition to dict, or None if it's the default (full range)."""
    # Check if this is a meaningful condition (not just the full param range)
    is_default = (
        cond.must_from == param.min_val
        and cond.must_to == param.max_val
        and not cond.must_equal_values
        and not cond.must_mod_values
    )

    if is_default:
        return None

    result: dict[str, Any] = {"param": param_idx}

    # Range condition
    if cond.must_from != param.min_val or cond.must_to != param.max_val:
        result["range"] = [cond.must_from, cond.must_to]

    # Equal values
    if cond.must_equal_values:
        if cond.must_equal_values_equal:
            result["equals"] = cond.must_equal_values
        else:
            result["not_equals"] = cond.must_equal_values

    # Mod values
    if cond.must_mod_values:
        if cond.must_mod_values_mod:
            result["divisible_by"] = cond.must_mod_values
        else:
            result["not_divisible_by"] = cond.must_mod_values

    return result


def _location_to_dict(loc: Location) -> dict[str, Any]:
    """Convert a location to dict."""
    result: dict[str, Any] = {
        "id": loc.id,
    }

    # Type flags - note: deadly is a subtype of fail (both flags can be set)
    if loc.is_starting:
        result["type"] = "starting"
    elif loc.is_success:
        result["type"] = "success"
    elif loc.is_fail_deadly:
        # Deadly implies fail, check deadly first
        result["type"] = "deadly"
    elif loc.is_fail:
        result["type"] = "fail"
    elif loc.is_empty:
        result["type"] = "empty"

    # Texts - preserve all texts including empty ones for index-based selection
    # Only optimize to single "text" if there's exactly one text total
    if loc.texts:
        if len(loc.texts) == 1 and loc.texts[0]:
            result["text"] = loc.texts[0]
        elif any(t for t in loc.texts):  # At least one non-empty
            result["texts"] = loc.texts  # Preserve full array including empty strings
            if loc.is_text_by_formula and loc.text_select_formula:
                result["text_select_formula"] = loc.text_select_formula

    # Optional fields
    if loc.day_passed:
        result["day_passed"] = True
    if loc.max_visits:
        result["max_visits"] = loc.max_visits

    # Param changes - only include non-trivial ones
    changes = []
    for i, change in enumerate(loc.params_changes):
        change_dict = _param_change_to_dict(change, i)
        if change_dict:
            changes.append(change_dict)
    if changes:
        result["param_changes"] = changes

    # Position (optional, for editor)
    if loc.loc_x != 50 or loc.loc_y != 50:
        result["position"] = [loc.loc_x, loc.loc_y]

    return result


def _jump_to_dict(jump: Jump, params: list[QMParam]) -> dict[str, Any]:
    """Convert a jump to dict."""
    result: dict[str, Any] = {
        "id": jump.id,
        "from": jump.from_location_id,
        "to": jump.to_location_id,
    }

    # Text
    if jump.text:
        result["text"] = jump.text
    if jump.description:
        result["description"] = jump.description

    # Priority/probability
    if jump.prio != 1.0:
        result["priority"] = jump.prio

    # Flags
    if jump.day_passed:
        result["day_passed"] = True
    if jump.always_show:
        result["always_show"] = True
    if jump.jumping_count_limit:
        result["max_uses"] = jump.jumping_count_limit
    if jump.showing_order != 0:
        result["order"] = jump.showing_order

    # Formula condition
    if jump.formula_to_pass:
        result["condition_formula"] = jump.formula_to_pass

    # Param conditions - only include non-default ones
    conditions = []
    for i, cond in enumerate(jump.params_conditions):
        if i < len(params):
            cond_dict = _condition_to_dict(cond, params[i], i)
            if cond_dict:
                conditions.append(cond_dict)
    if conditions:
        result["conditions"] = conditions

    # Param changes
    changes = []
    for i, change in enumerate(jump.params_changes):
        change_dict = _param_change_to_dict(change, i)
        if change_dict:
            changes.append(change_dict)
    if changes:
        result["param_changes"] = changes

    # Media
    if jump.img:
        result["img"] = jump.img
    if jump.sound:
        result["sound"] = jump.sound
    if jump.track:
        result["track"] = jump.track

    return result


def dict_to_quest(data: dict[str, Any]) -> Quest:
    """Convert a dictionary back to a Quest object."""
    meta = data.get("meta", {})
    strings_data = data.get("strings", {})

    # Parse when_done enum
    when_done_str = meta.get("when_done", "on_return")
    when_done = WhenDone.ON_RETURN if when_done_str == "on_return" else WhenDone.ON_FINISH

    # Build strings
    strings = QuestStrings(
        to_star=strings_data.get("to_star", ""),
        to_planet=strings_data.get("to_planet", ""),
        from_star=strings_data.get("from_star", ""),
        from_planet=strings_data.get("from_planet", ""),
        ranger=strings_data.get("ranger", ""),
        date=strings_data.get("date", ""),
        money=strings_data.get("money", ""),
        parsec=strings_data.get("parsec"),
        artefact=strings_data.get("artefact"),
    )

    # Parse params
    params_data = data.get("params", [])
    params = [_dict_to_param(p) for p in params_data]
    params_count = len(params)

    # Parse locations
    locations_data = data.get("locations", [])
    locations = [_dict_to_location(loc, params_count) for loc in locations_data]

    # Parse jumps
    jumps_data = data.get("jumps", [])
    jumps = [_dict_to_jump(j, params) for j in jumps_data]

    # Screen size
    screen_size = meta.get("screen_size", [200, 200])

    # Version info
    version_info = meta.get("version_info")
    major_version = version_info.get("major") if version_info else None
    minor_version = version_info.get("minor") if version_info else None
    changelog = version_info.get("changelog") if version_info else None

    return Quest(
        giving_race=meta.get("giving_race", 0),
        when_done=when_done,
        planet_race=meta.get("planet_race", 0),
        player_career=meta.get("player_career", 0),
        player_race=meta.get("player_race", 0),
        default_jump_count_limit=meta.get("default_jump_count_limit", 0),
        hardness=meta.get("hardness", 0),
        params_count=params_count,
        screen_size_x=screen_size[0],
        screen_size_y=screen_size[1],
        major_version=major_version,
        minor_version=minor_version,
        changelog=changelog,
        strings=strings,
        locations_count=len(locations),
        jumps_count=len(jumps),
        success_text=data.get("success_text", ""),
        task_text=data.get("task_text", ""),
        params=params,
        locations=locations,
        jumps=jumps,
    )


def _dict_to_param(data: dict[str, Any]) -> QMParam:
    """Convert dict back to QMParam."""
    # Parse type enum
    type_str = data.get("type", "normal")
    param_type = ParamType[type_str.upper()]

    # Parse crit_type enum
    crit_type_str = data.get("crit_type", "max")
    crit_type = ParamCritType[crit_type_str.upper()]

    # Parse display ranges
    display_ranges = data.get("display_ranges", [])
    showing_info = [
        ParamShowingInfo(from_val=r["from"], to_val=r["to"], text=r["text"]) for r in display_ranges
    ]

    return QMParam(
        name=data.get("name", ""),
        min_val=data.get("min", 0),
        max_val=data.get("max", 0),
        param_type=param_type,
        crit_type=crit_type,
        show_when_zero=data.get("show_when_zero", False),
        active=data.get("active", False),
        is_money=data.get("is_money", False),
        starting=data.get("starting", ""),
        crit_value_string=data.get("crit_text", ""),
        showing_info=showing_info,
        img=data.get("img"),
        sound=data.get("sound"),
        track=data.get("track"),
    )


def _dict_to_location(data: dict[str, Any], params_count: int) -> Location:
    """Convert dict back to Location."""
    loc_type = data.get("type", "")

    # Handle texts
    if "text" in data:
        texts = [data["text"]]
    else:
        texts = data.get("texts", [])

    # Build param changes array
    params_changes = [ParameterChange() for _ in range(params_count)]
    for change_data in data.get("param_changes", []):
        idx = change_data["param"]
        params_changes[idx] = _dict_to_param_change(change_data)

    # Position
    position = data.get("position", [50, 50])

    # Note: deadly implies fail (both flags should be set)
    is_deadly = loc_type == "deadly"

    return Location(
        id=data["id"],
        day_passed=data.get("day_passed", False),
        is_starting=(loc_type == "starting"),
        is_success=(loc_type == "success"),
        is_fail=(loc_type == "fail" or is_deadly),
        is_fail_deadly=is_deadly,
        is_empty=(loc_type == "empty"),
        max_visits=data.get("max_visits", 0),
        loc_x=position[0],
        loc_y=position[1],
        params_changes=params_changes,
        texts=texts,
        is_text_by_formula=("text_select_formula" in data),
        text_select_formula=data.get("text_select_formula", ""),
    )


def _dict_to_param_change(data: dict[str, Any]) -> ParameterChange:
    """Convert dict back to ParameterChange."""
    # Determine change type
    is_formula = "formula" in data
    is_value = "set" in data
    is_percent = "percent" in data

    if is_formula:
        change = 0
        formula = data["formula"]
    elif is_value:
        change = data["set"]
        formula = ""
    elif is_percent:
        change = data["percent"]
        formula = ""
    else:
        change = data.get("add", 0)
        formula = ""

    # Showing type
    if data.get("show"):
        showing_type = ParameterShowingType.SHOW
    elif data.get("hide"):
        showing_type = ParameterShowingType.HIDE
    else:
        showing_type = ParameterShowingType.NO_CHANGE

    return ParameterChange(
        change=change,
        is_change_percentage=is_percent,
        is_change_value=is_value,
        is_change_formula=is_formula,
        changing_formula=formula,
        showing_type=showing_type,
        crit_text=data.get("crit_text", ""),
        img=data.get("img"),
        sound=data.get("sound"),
        track=data.get("track"),
    )


def _dict_to_jump(data: dict[str, Any], params: list[QMParam]) -> Jump:
    """Convert dict back to Jump."""
    params_count = len(params)

    # Build conditions array with defaults (must_equal_values_equal and must_mod_values_mod
    # default to True in the QM binary format)
    params_conditions = [
        JumpParameterCondition(
            must_from=params[i].min_val,
            must_to=params[i].max_val,
            must_equal_values_equal=True,
            must_mod_values_mod=True,
        )
        for i in range(params_count)
    ]

    for cond_data in data.get("conditions", []):
        idx = cond_data["param"]
        params_conditions[idx] = _dict_to_condition(cond_data, params[idx])

    # Build changes array
    params_changes = [ParameterChange() for _ in range(params_count)]
    for change_data in data.get("param_changes", []):
        idx = change_data["param"]
        params_changes[idx] = _dict_to_param_change(change_data)

    return Jump(
        id=data["id"],
        from_location_id=data["from"],
        to_location_id=data["to"],
        prio=data.get("priority", 1.0),
        day_passed=data.get("day_passed", False),
        always_show=data.get("always_show", False),
        jumping_count_limit=data.get("max_uses", 0),
        showing_order=data.get("order", 0),
        params_changes=params_changes,
        params_conditions=params_conditions,
        formula_to_pass=data.get("condition_formula", ""),
        text=data.get("text", ""),
        description=data.get("description", ""),
        img=data.get("img"),
        sound=data.get("sound"),
        track=data.get("track"),
    )


def _dict_to_condition(data: dict[str, Any], param: QMParam) -> JumpParameterCondition:
    """Convert dict back to JumpParameterCondition."""
    # Range
    range_val = data.get("range", [param.min_val, param.max_val])

    # Equal values
    # Equal values - default must_equal_values_equal to True (matches QM binary default)
    if "equals" in data:
        must_equal_values = data["equals"]
        must_equal_values_equal = True
    elif "not_equals" in data:
        must_equal_values = data["not_equals"]
        must_equal_values_equal = False
    else:
        must_equal_values = []
        must_equal_values_equal = True  # Default matches QM binary format

    # Mod values - default must_mod_values_mod to True (matches QM binary default)
    if "divisible_by" in data:
        must_mod_values = data["divisible_by"]
        must_mod_values_mod = True
    elif "not_divisible_by" in data:
        must_mod_values = data["not_divisible_by"]
        must_mod_values_mod = False
    else:
        must_mod_values = []
        must_mod_values_mod = True  # Default matches QM binary format

    return JumpParameterCondition(
        must_from=range_val[0],
        must_to=range_val[1],
        must_equal_values=must_equal_values,
        must_equal_values_equal=must_equal_values_equal,
        must_mod_values=must_mod_values,
        must_mod_values_mod=must_mod_values_mod,
    )


def save_quest_json(quest: Quest, path: Path) -> None:
    """Save a quest to a JSON file."""
    data = quest_to_dict(quest)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_quest_json(path: Path) -> Quest:
    """Load a quest from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return dict_to_quest(data)


def convert_qm_to_json(qm_path: Path, json_path: Path) -> None:
    """Convert a QM/QMM file to JSON."""
    with open(qm_path, "rb") as f:
        data = f.read()
    quest = parse_qm(data)
    save_quest_json(quest, json_path)
