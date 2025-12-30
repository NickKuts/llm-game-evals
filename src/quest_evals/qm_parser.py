"""Binary parser for Space Rangers QM/QMM quest files.

This module parses the binary .qm and .qmm formats used by Space Rangers 2.
For the recommended JSON format, see quest_json.py.
"""

from __future__ import annotations

import struct

from .models import (
    Jump,
    JumpParameterCondition,
    Location,
    LocationType,
    ParamCritType,
    ParameterChange,
    ParameterChangeType,
    ParameterShowingType,
    ParamShowingInfo,
    ParamType,
    QMParam,
    Quest,
    QuestStrings,
    WhenDone,
)

# Re-export models for backward compatibility
__all__ = [
    "parse_qm",
    "Quest",
    "QuestStrings",
    "QMParam",
    "ParamShowingInfo",
    "ParameterChange",
    "JumpParameterCondition",
    "Location",
    "Jump",
    "ParamType",
    "ParamCritType",
    "ParameterShowingType",
    "ParameterChangeType",
    "LocationType",
    "WhenDone",
]


class BinaryReader:
    """Binary reader for little-endian data."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def int32(self) -> int:
        val = struct.unpack_from("<i", self.data, self.pos)[0]
        self.pos += 4
        return val

    def uint32(self) -> int:
        val = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return val

    def float64(self) -> float:
        val = struct.unpack_from("<d", self.data, self.pos)[0]
        self.pos += 8
        return val

    def byte(self) -> int:
        val = self.data[self.pos]
        self.pos += 1
        return val

    def read_string(self) -> str:
        has_string = self.int32()
        if has_string:
            str_len = self.int32()
            raw = self.data[self.pos : self.pos + str_len * 2]
            self.pos += str_len * 2
            return raw.decode("utf-16-le")
        return ""

    def seek(self, offset: int) -> None:
        self.pos += offset

    def remaining(self) -> int:
        return len(self.data) - self.pos


# Magic header values
HEADER_QMM_V6 = 0x423A35D6
HEADER_QMM_V7 = 0x423A35D7
HEADER_QM_24 = 0x423A35D2
HEADER_QM_48 = 0x423A35D3
HEADER_QM_96 = 0x423A35D4


def parse_qm(data: bytes) -> Quest:
    """Parse a QM/QMM quest file."""
    r = BinaryReader(data)
    quest = Quest()

    header = r.uint32()
    is_qmm = header in (HEADER_QMM_V6, HEADER_QMM_V7)

    # Parse base header
    if is_qmm:
        if header == HEADER_QMM_V7:
            quest.major_version = r.int32()
            quest.minor_version = r.int32()
            quest.changelog = r.read_string()

        quest.giving_race = r.byte()
        quest.when_done = WhenDone(r.byte())
        quest.planet_race = r.byte()
        quest.player_career = r.byte()
        quest.player_race = r.byte()
        _reputation_change = r.int32()

        quest.screen_size_x = r.int32()
        quest.screen_size_y = r.int32()
        _width_size = r.int32()
        _height_size = r.int32()
        quest.default_jump_count_limit = r.int32()
        quest.hardness = r.int32()
        quest.params_count = r.int32()
    else:
        # Old QM format
        if header == HEADER_QM_24:
            quest.params_count = 24
        elif header == HEADER_QM_48:
            quest.params_count = 48
        elif header == HEADER_QM_96:
            quest.params_count = 96
        else:
            raise ValueError(f"Unknown header: 0x{header:08x}")

        r.int32()  # skip
        quest.giving_race = r.byte()
        quest.when_done = WhenDone(r.byte())
        r.int32()  # skip
        quest.planet_race = r.byte()
        r.int32()  # skip
        quest.player_career = r.byte()
        r.int32()  # skip
        quest.player_race = r.byte()
        _reputation_change = r.int32()
        for _ in range(5):
            r.int32()  # skip
        quest.default_jump_count_limit = r.int32()
        quest.hardness = r.int32()

    # Parse parameters
    for _ in range(quest.params_count):
        if is_qmm:
            param = _parse_param_qmm(r)
        else:
            param = _parse_param_qm(r)
        quest.params.append(param)

    # Parse base2 (strings, counts, texts)
    quest.strings.to_star = r.read_string()
    if not is_qmm:
        quest.strings.parsec = r.read_string()
        quest.strings.artefact = r.read_string()
    quest.strings.to_planet = r.read_string()
    quest.strings.date = r.read_string()
    quest.strings.money = r.read_string()
    quest.strings.from_planet = r.read_string()
    quest.strings.from_star = r.read_string()
    quest.strings.ranger = r.read_string()

    quest.locations_count = r.int32()
    quest.jumps_count = r.int32()

    quest.success_text = r.read_string()
    quest.task_text = r.read_string()

    if not is_qmm:
        _unknown_text = r.read_string()

    # Parse locations
    for _ in range(quest.locations_count):
        if is_qmm:
            loc = _parse_location_qmm(r, quest.params_count)
        else:
            loc = _parse_location_qm(r, quest.params_count)
        quest.locations.append(loc)

    # Parse jumps
    for _ in range(quest.jumps_count):
        if is_qmm:
            jump = _parse_jump_qmm(r, quest.params_count, quest.params)
        else:
            jump = _parse_jump_qm(r, quest.params_count)
        quest.jumps.append(jump)

    if r.remaining() != 0:
        raise ValueError(f"Not at end of file, {r.remaining()} bytes remaining")

    return quest


def _parse_param_qm(r: BinaryReader) -> QMParam:
    min_val = r.int32()
    max_val = r.int32()
    r.int32()  # skip middle value
    param_type = ParamType(r.byte())
    r.int32()  # skip
    show_when_zero = bool(r.byte())
    crit_type = ParamCritType(r.byte())
    active = bool(r.byte())
    showing_ranges_count = r.int32()
    is_money = bool(r.byte())
    name = r.read_string()

    showing_info = []
    for _ in range(showing_ranges_count):
        from_val = r.int32()
        to_val = r.int32()
        text = r.read_string()
        showing_info.append(ParamShowingInfo(from_val, to_val, text))

    crit_value_string = r.read_string()
    starting = r.read_string()

    return QMParam(
        min_val=min_val,
        max_val=max_val,
        param_type=param_type,
        show_when_zero=show_when_zero,
        crit_type=crit_type,
        active=active,
        is_money=is_money,
        name=name,
        showing_info=showing_info,
        starting=starting,
        crit_value_string=crit_value_string,
    )


def _parse_param_qmm(r: BinaryReader) -> QMParam:
    min_val = r.int32()
    max_val = r.int32()
    param_type = ParamType(r.byte())
    r.seek(3)  # unknown bytes
    show_when_zero = bool(r.byte())
    crit_type = ParamCritType(r.byte())
    active = bool(r.byte())
    showing_ranges_count = r.int32()
    is_money = bool(r.byte())
    name = r.read_string()

    showing_info = []
    for _ in range(showing_ranges_count):
        from_val = r.int32()
        to_val = r.int32()
        text = r.read_string()
        showing_info.append(ParamShowingInfo(from_val, to_val, text))

    crit_value_string = r.read_string()
    img = r.read_string() or None
    sound = r.read_string() or None
    track = r.read_string() or None
    starting = r.read_string()

    return QMParam(
        min_val=min_val,
        max_val=max_val,
        param_type=param_type,
        show_when_zero=show_when_zero,
        crit_type=crit_type,
        active=active,
        is_money=is_money,
        name=name,
        showing_info=showing_info,
        starting=starting,
        crit_value_string=crit_value_string,
        img=img,
        sound=sound,
        track=track,
    )


def _parse_location_qm(r: BinaryReader, params_count: int) -> Location:
    day_passed = bool(r.int32())
    r.seek(8)  # skip coordinates
    loc_id = r.int32()
    is_starting = bool(r.byte())
    is_success = bool(r.byte())
    is_fail = bool(r.byte())
    is_fail_deadly = bool(r.byte())
    is_empty = bool(r.byte())

    params_changes = []
    for _ in range(params_count):
        r.seek(12)
        change = r.int32()
        showing_type = ParameterShowingType(r.byte())
        r.seek(4)
        is_change_percentage = bool(r.byte())
        is_change_value = bool(r.byte())
        is_change_formula = bool(r.byte())
        changing_formula = r.read_string()
        r.seek(10)
        crit_text = r.read_string()

        params_changes.append(
            ParameterChange(
                change=change,
                showing_type=showing_type,
                is_change_percentage=is_change_percentage,
                is_change_value=is_change_value,
                is_change_formula=is_change_formula,
                changing_formula=changing_formula,
                crit_text=crit_text,
            )
        )

    texts = []
    for _ in range(10):  # LOCATION_TEXTS = 10
        texts.append(r.read_string())

    is_text_by_formula = bool(r.byte())
    r.seek(4)
    r.read_string()  # skip
    r.read_string()  # skip
    text_select_formula = r.read_string()

    return Location(
        id=loc_id,
        day_passed=day_passed,
        is_starting=is_starting,
        is_success=is_success,
        is_fail=is_fail,
        is_fail_deadly=is_fail_deadly,
        is_empty=is_empty,
        params_changes=params_changes,
        texts=texts,
        is_text_by_formula=is_text_by_formula,
        text_select_formula=text_select_formula,
    )


def _parse_location_qmm(r: BinaryReader, params_count: int) -> Location:
    day_passed = bool(r.int32())
    loc_x = r.int32()
    loc_y = r.int32()
    loc_id = r.int32()
    max_visits = r.int32()

    loc_type = LocationType(r.byte())
    is_starting = loc_type == LocationType.STARTING
    is_success = loc_type == LocationType.SUCCESS
    is_fail = loc_type == LocationType.FAIL
    is_fail_deadly = loc_type == LocationType.DEADLY
    is_empty = loc_type == LocationType.EMPTY

    # Initialize default params changes
    params_changes = [ParameterChange() for _ in range(params_count)]

    affected_params_count = r.int32()
    for _ in range(affected_params_count):
        param_n = r.int32() - 1  # 1-indexed in file
        change = r.int32()
        showing_type = ParameterShowingType(r.byte())
        change_type = ParameterChangeType(r.byte())
        changing_formula = r.read_string()
        crit_text = r.read_string()
        img = r.read_string() or None
        sound = r.read_string() or None
        track = r.read_string() or None

        params_changes[param_n] = ParameterChange(
            change=change,
            showing_type=showing_type,
            is_change_percentage=(change_type == ParameterChangeType.PERCENTAGE),
            is_change_value=(change_type == ParameterChangeType.VALUE),
            is_change_formula=(change_type == ParameterChangeType.FORMULA),
            changing_formula=changing_formula,
            crit_text=crit_text,
            img=img,
            sound=sound,
            track=track,
        )

    location_texts_count = r.int32()
    texts = []
    for _ in range(location_texts_count):
        text = r.read_string()
        texts.append(text)
        _img = r.read_string()
        _sound = r.read_string()
        _track = r.read_string()

    is_text_by_formula = bool(r.byte())
    text_select_formula = r.read_string()

    return Location(
        id=loc_id,
        day_passed=day_passed,
        is_starting=is_starting,
        is_success=is_success,
        is_fail=is_fail,
        is_fail_deadly=is_fail_deadly,
        is_empty=is_empty,
        params_changes=params_changes,
        texts=texts,
        is_text_by_formula=is_text_by_formula,
        text_select_formula=text_select_formula,
        max_visits=max_visits,
        loc_x=loc_x,
        loc_y=loc_y,
    )


def _parse_jump_qm(r: BinaryReader, params_count: int) -> Jump:
    prio = r.float64()
    day_passed = bool(r.int32())
    jump_id = r.int32()
    from_location_id = r.int32()
    to_location_id = r.int32()
    r.seek(1)
    always_show = bool(r.byte())
    jumping_count_limit = r.int32()
    showing_order = r.int32()

    params_changes = []
    params_conditions = []

    for _ in range(params_count):
        r.seek(4)
        must_from = r.int32()
        must_to = r.int32()
        change = r.int32()
        showing_type = ParameterShowingType(r.int32())
        r.seek(1)
        is_change_percentage = bool(r.byte())
        is_change_value = bool(r.byte())
        is_change_formula = bool(r.byte())
        changing_formula = r.read_string()

        must_equal_values_count = r.int32()
        must_equal_values_equal = bool(r.byte())
        must_equal_values = [r.int32() for _ in range(must_equal_values_count)]

        must_mod_values_count = r.int32()
        must_mod_values_mod = bool(r.byte())
        must_mod_values = [r.int32() for _ in range(must_mod_values_count)]

        crit_text = r.read_string()

        params_changes.append(
            ParameterChange(
                change=change,
                showing_type=showing_type,
                is_change_percentage=is_change_percentage,
                is_change_value=is_change_value,
                is_change_formula=is_change_formula,
                changing_formula=changing_formula,
                crit_text=crit_text,
            )
        )
        params_conditions.append(
            JumpParameterCondition(
                must_from=must_from,
                must_to=must_to,
                must_equal_values=must_equal_values,
                must_equal_values_equal=must_equal_values_equal,
                must_mod_values=must_mod_values,
                must_mod_values_mod=must_mod_values_mod,
            )
        )

    formula_to_pass = r.read_string()
    text = r.read_string()
    description = r.read_string()

    return Jump(
        id=jump_id,
        from_location_id=from_location_id,
        to_location_id=to_location_id,
        prio=prio,
        day_passed=day_passed,
        always_show=always_show,
        jumping_count_limit=jumping_count_limit,
        showing_order=showing_order,
        params_changes=params_changes,
        params_conditions=params_conditions,
        formula_to_pass=formula_to_pass,
        text=text,
        description=description,
    )


def _parse_jump_qmm(r: BinaryReader, params_count: int, quest_params: list[QMParam]) -> Jump:
    prio = r.float64()
    day_passed = bool(r.int32())
    jump_id = r.int32()
    from_location_id = r.int32()
    to_location_id = r.int32()
    always_show = bool(r.byte())
    jumping_count_limit = r.int32()
    showing_order = r.int32()

    # Initialize defaults
    params_changes = [ParameterChange() for _ in range(params_count)]
    params_conditions = [
        JumpParameterCondition(
            must_from=quest_params[i].min_val,
            must_to=quest_params[i].max_val,
        )
        for i in range(params_count)
    ]

    # Parse affected conditions
    affected_conditions_count = r.int32()
    for _ in range(affected_conditions_count):
        param_id = r.int32() - 1
        must_from = r.int32()
        must_to = r.int32()

        must_equal_values_count = r.int32()
        must_equal_values_equal = bool(r.byte())
        must_equal_values = [r.int32() for _ in range(must_equal_values_count)]

        must_mod_values_count = r.int32()
        must_mod_values_mod = bool(r.byte())
        must_mod_values = [r.int32() for _ in range(must_mod_values_count)]

        params_conditions[param_id] = JumpParameterCondition(
            must_from=must_from,
            must_to=must_to,
            must_equal_values=must_equal_values,
            must_equal_values_equal=must_equal_values_equal,
            must_mod_values=must_mod_values,
            must_mod_values_mod=must_mod_values_mod,
        )

    # Parse affected changes
    affected_change_count = r.int32()
    for _ in range(affected_change_count):
        param_id = r.int32() - 1
        change = r.int32()
        showing_type = ParameterShowingType(r.byte())
        change_type = ParameterChangeType(r.byte())
        changing_formula = r.read_string()
        crit_text = r.read_string()
        img = r.read_string() or None
        sound = r.read_string() or None
        track = r.read_string() or None

        params_changes[param_id] = ParameterChange(
            change=change,
            showing_type=showing_type,
            is_change_percentage=(change_type == ParameterChangeType.PERCENTAGE),
            is_change_value=(change_type == ParameterChangeType.VALUE),
            is_change_formula=(change_type == ParameterChangeType.FORMULA),
            changing_formula=changing_formula,
            crit_text=crit_text,
            img=img,
            sound=sound,
            track=track,
        )

    formula_to_pass = r.read_string()
    text = r.read_string()
    description = r.read_string()
    img = r.read_string() or None
    sound = r.read_string() or None
    track = r.read_string() or None

    return Jump(
        id=jump_id,
        from_location_id=from_location_id,
        to_location_id=to_location_id,
        prio=prio,
        day_passed=day_passed,
        always_show=always_show,
        jumping_count_limit=jumping_count_limit,
        showing_order=showing_order,
        params_changes=params_changes,
        params_conditions=params_conditions,
        formula_to_pass=formula_to_pass,
        text=text,
        description=description,
        img=img,
        sound=sound,
        track=track,
    )
