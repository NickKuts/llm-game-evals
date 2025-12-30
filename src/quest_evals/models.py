"""Quest data models for Space Rangers 2 quests."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class ParamType(IntEnum):
    """Parameter type determining quest outcome when critical value is reached."""

    NORMAL = 0
    FAIL = 1
    SUCCESS = 2
    DEADLY = 3


class ParamCritType(IntEnum):
    """Which boundary triggers the critical condition."""

    MAX = 0
    MIN = 1


class ParameterShowingType(IntEnum):
    """How parameter visibility changes."""

    NO_CHANGE = 0x00
    SHOW = 0x01
    HIDE = 0x02


class ParameterChangeType(IntEnum):
    """How a parameter value is modified."""

    VALUE = 0x00
    SUM = 0x01
    PERCENTAGE = 0x02
    FORMULA = 0x03


class LocationType(IntEnum):
    """Location type in the quest graph."""

    ORDINARY = 0x00
    STARTING = 0x01
    EMPTY = 0x02
    SUCCESS = 0x03
    FAIL = 0x04
    DEADLY = 0x05


class WhenDone(IntEnum):
    """When the quest is considered complete."""

    ON_RETURN = 0
    ON_FINISH = 1


@dataclass
class QuestStrings:
    """Localized strings for quest display."""

    to_star: str = ""
    parsec: str | None = None
    artefact: str | None = None
    to_planet: str = ""
    date: str = ""
    money: str = ""
    from_planet: str = ""
    from_star: str = ""
    ranger: str = ""


@dataclass
class ParamShowingInfo:
    """Defines how a parameter is displayed for a value range."""

    from_val: int
    to_val: int
    text: str


@dataclass
class QMParam:
    """Quest parameter definition."""

    min_val: int
    max_val: int
    param_type: ParamType
    show_when_zero: bool
    crit_type: ParamCritType
    active: bool
    is_money: bool
    name: str
    showing_info: list[ParamShowingInfo] = field(default_factory=list)
    starting: str = ""
    crit_value_string: str = ""
    img: str | None = None
    sound: str | None = None
    track: str | None = None


@dataclass
class ParameterChange:
    """Defines how a parameter changes at a location or jump."""

    change: int = 0
    is_change_percentage: bool = False
    is_change_value: bool = False
    is_change_formula: bool = False
    changing_formula: str = ""
    showing_type: ParameterShowingType = ParameterShowingType.NO_CHANGE
    crit_text: str = ""
    img: str | None = None
    sound: str | None = None
    track: str | None = None


@dataclass
class JumpParameterCondition:
    """Condition that must be met for a jump to be available."""

    must_from: int
    must_to: int
    must_equal_values: list[int] = field(default_factory=list)
    must_equal_values_equal: bool = False
    must_mod_values: list[int] = field(default_factory=list)
    must_mod_values_mod: bool = False


@dataclass
class Location:
    """A location (node) in the quest graph."""

    id: int
    day_passed: bool
    is_starting: bool
    is_success: bool
    is_fail: bool
    is_fail_deadly: bool
    is_empty: bool
    params_changes: list[ParameterChange]
    texts: list[str]
    is_text_by_formula: bool = False
    text_select_formula: str = ""
    max_visits: int = 0
    loc_x: int = 50
    loc_y: int = 50


@dataclass
class Jump:
    """A jump (edge) between locations in the quest graph."""

    id: int
    from_location_id: int
    to_location_id: int
    prio: float
    day_passed: bool
    always_show: bool
    jumping_count_limit: int
    showing_order: int
    params_changes: list[ParameterChange]
    params_conditions: list[JumpParameterCondition]
    formula_to_pass: str
    text: str
    description: str
    img: str | None = None
    sound: str | None = None
    track: str | None = None


@dataclass
class Quest:
    """Complete quest definition."""

    # Base info
    giving_race: int = 0
    when_done: WhenDone = WhenDone.ON_RETURN
    planet_race: int = 0
    player_career: int = 0
    player_race: int = 0
    default_jump_count_limit: int = 0
    hardness: int = 0
    params_count: int = 0
    screen_size_x: int = 200
    screen_size_y: int = 200

    # Version info (QMM only)
    major_version: int | None = None
    minor_version: int | None = None
    changelog: str | None = None

    # Strings
    strings: QuestStrings = field(default_factory=QuestStrings)

    # Counts
    locations_count: int = 0
    jumps_count: int = 0

    # Text
    success_text: str = ""
    task_text: str = ""

    # Data
    params: list[QMParam] = field(default_factory=list)
    locations: list[Location] = field(default_factory=list)
    jumps: list[Jump] = field(default_factory=list)
