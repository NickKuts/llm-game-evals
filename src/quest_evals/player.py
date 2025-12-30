"""Quest player - state machine for playing Space Rangers quests."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from enum import Enum, auto

from .formula import parse
from .models import (
    Jump,
    Location,
    ParamCritType,
    ParameterChange,
    ParameterShowingType,
    ParamType,
    Quest,
)

# Special jump IDs
JUMP_I_AGREE = -1
JUMP_NEXT = -2
JUMP_GO_BACK_TO_SHIP = -3

DEFAULT_DAYS_TO_PASS = 35
DEFAULT_MONEY = 2000


class GameStatus(Enum):
    RUNNING = auto()
    WIN = auto()
    FAIL = auto()
    DEAD = auto()


class PlayerState(Enum):
    STARTING = auto()
    LOCATION = auto()
    JUMP = auto()
    JUMP_AND_NEXT_CRIT = auto()
    CRIT_ON_LOCATION = auto()
    CRIT_ON_LOCATION_LAST = auto()
    CRIT_ON_JUMP = auto()
    RETURNED_ENDING = auto()


@dataclass
class Choice:
    """A choice available to the player."""

    text: str
    jump_id: int
    active: bool


@dataclass
class GameState:
    """Current state of the game."""

    text: str
    params_state: list[str]
    choices: list[Choice]
    game_status: GameStatus


@dataclass
class InternalState:
    """Internal state of the player."""

    state: PlayerState = PlayerState.STARTING
    location_id: int = 0
    last_jump_id: int | None = None
    possible_jumps: list[tuple[int, bool]] = field(default_factory=list)  # (jump_id, active)
    param_values: list[int] = field(default_factory=list)
    param_show: list[bool] = field(default_factory=list)
    jumped_count: dict[int, int] = field(default_factory=dict)
    location_visit_count: dict[int, int] = field(default_factory=dict)
    days_passed: int = 0
    crit_param_id: int | None = None


class QuestPlayer:
    """State machine for playing a quest."""

    TEXTS_EN = {
        "i_agree": "I agree",
        "next": "Next",
        "go_back": "Go back to ship",
    }

    def __init__(self, quest: Quest):
        self.quest = quest
        self._state: InternalState = InternalState()

        # Set up money parameter if present
        for i, param in enumerate(quest.params):
            if param.is_money:
                money = min(DEFAULT_MONEY, param.max_val)
                quest.params[i].starting = f"[{money}]"

        self._start()

    def _start(self) -> None:
        """Initialize the game state."""
        # Find starting location
        start_loc = next((loc for loc in self.quest.locations if loc.is_starting), None)
        if not start_loc:
            raise ValueError("No starting location found!")

        # Initialize parameter values
        param_values = []
        for param in self.quest.params:
            if param.active and param.starting:
                val = parse(param.starting)
            else:
                val = 0
            param_values.append(val)

        self._state = InternalState(
            state=PlayerState.STARTING,
            location_id=start_loc.id,
            param_values=param_values,
            param_show=[True] * self.quest.params_count,
        )

    def _substitute(self, text: str, diamond_index: int | None = None) -> str:
        """Perform variable substitution in text."""
        if diamond_index is not None:
            text = text.replace("<>", f"[p{diamond_index + 1}]")

        # Replace formula expressions {expr}
        def replace_formula(match: re.Match) -> str:
            formula = match.group(1)
            result = parse(formula, self._state.param_values)
            return str(result)

        text = re.sub(r"\{([^}]+)\}", replace_formula, text)

        # Replace player variables
        replacements = {
            "<Ranger>": "Ranger",
            "<Player>": "Player",
            "<FromPlanet>": "Earth",
            "<FromStar>": "Solar",
            "<ToPlanet>": "Bonnasis",
            "<ToStar>": "Procyon",
            "<Money>": "65535",
            "<Day>": str(DEFAULT_DAYS_TO_PASS - self._state.days_passed),
            "<Date>": self._format_date(DEFAULT_DAYS_TO_PASS),
            "<CurDate>": self._format_date(self._state.days_passed),
        }

        for key, value in replacements.items():
            text = text.replace(key, value)

        # Replace parameter references [pN]
        for i, val in enumerate(self._state.param_values):
            text = text.replace(f"[p{i + 1}]", str(val))

        # Remove color tags
        text = text.replace("<clr>", "").replace("<clrEnd>", "")

        return text

    def _format_date(self, days_offset: int) -> str:
        """Format a date as it would appear in the game."""
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        # Simple date calculation (game uses fictional dates)
        day = (days_offset % 30) + 1
        month = (days_offset // 30) % 12
        year = 3000 + (days_offset // 365)
        return f"{day} {months[month]} {year}"

    def _get_params_state(self) -> list[str]:
        """Get the displayable parameter state."""
        result = []
        for i, param in enumerate(self.quest.params):
            if not self._state.param_show[i] or not param.active:
                continue

            val = self._state.param_values[i]
            if val == 0 and not param.show_when_zero:
                continue

            for info in param.showing_info:
                if info.from_val <= val <= info.to_val:
                    text = self._substitute(info.text, i)
                    result.append(text)
                    break

        return result

    def _get_location(self, loc_id: int) -> Location:
        """Get a location by ID."""
        for loc in self.quest.locations:
            if loc.id == loc_id:
                return loc
        raise ValueError(f"Location {loc_id} not found")

    def _get_jump(self, jump_id: int) -> Jump:
        """Get a jump by ID."""
        for jump in self.quest.jumps:
            if jump.id == jump_id:
                return jump
        raise ValueError(f"Jump {jump_id} not found")

    def _calculate_location_text_id(self, location: Location) -> int:
        """Determine which text to show for a location."""
        texts_with_content = [(i, text) for i, text in enumerate(location.texts) if text]

        if not texts_with_content:
            return 0

        if location.is_text_by_formula and location.text_select_formula:
            text_id = parse(location.text_select_formula, self._state.param_values) - 1
            if 0 <= text_id < len(location.texts) and location.texts[text_id]:
                return text_id
            return 0

        # Cycle through texts based on visit count
        visit_count = self._state.location_visit_count.get(location.id, 0)
        idx = visit_count % len(texts_with_content)
        return texts_with_content[idx][0]

    def get_state(self) -> GameState:
        """Get the current visible game state."""
        state = self._state

        if state.state == PlayerState.STARTING:
            return GameState(
                text=self._substitute(self.quest.task_text),
                params_state=[],
                choices=[Choice(self.TEXTS_EN["i_agree"], JUMP_I_AGREE, True)],
                game_status=GameStatus.RUNNING,
            )

        if state.state == PlayerState.JUMP:
            jump = self._get_jump(state.last_jump_id)
            return GameState(
                text=self._substitute(jump.description),
                params_state=self._get_params_state(),
                choices=[Choice(self.TEXTS_EN["next"], JUMP_NEXT, True)],
                game_status=GameStatus.RUNNING,
            )

        if state.state in (PlayerState.LOCATION, PlayerState.CRIT_ON_LOCATION):
            location = self._get_location(state.location_id)
            text_id = self._calculate_location_text_id(location)
            text = location.texts[text_id] if text_id < len(location.texts) else ""

            # Handle empty location with jump description
            if location.is_empty and state.last_jump_id is not None:
                last_jump = self._get_jump(state.last_jump_id)
                if last_jump.description:
                    text = last_jump.description

            # Determine choices
            if state.state == PlayerState.CRIT_ON_LOCATION:
                choices = [Choice(self.TEXTS_EN["next"], JUMP_NEXT, True)]
            elif location.is_fail or location.is_fail_deadly:
                choices = []
            elif location.is_success:
                choices = [Choice(self.TEXTS_EN["go_back"], JUMP_GO_BACK_TO_SHIP, True)]
            else:
                choices = []
                for jump_id, active in state.possible_jumps:
                    jump = self._get_jump(jump_id)
                    jump_text = self._substitute(jump.text) if jump.text else self.TEXTS_EN["next"]
                    choices.append(Choice(jump_text, jump_id, active))

            game_status = GameStatus.RUNNING
            if location.is_fail_deadly:
                game_status = GameStatus.DEAD
            elif location.is_fail:
                game_status = GameStatus.FAIL

            return GameState(
                text=self._substitute(text),
                params_state=self._get_params_state(),
                choices=choices,
                game_status=game_status,
            )

        if state.state in (PlayerState.CRIT_ON_JUMP, PlayerState.CRIT_ON_LOCATION_LAST):
            crit_id = state.crit_param_id
            param = self.quest.params[crit_id]

            if state.state == PlayerState.CRIT_ON_JUMP:
                jump = self._get_jump(state.last_jump_id)
                crit_text = jump.params_changes[crit_id].crit_text or param.crit_value_string
            else:
                location = self._get_location(state.location_id)
                crit_text = location.params_changes[crit_id].crit_text or param.crit_value_string

            if param.param_type == ParamType.SUCCESS:
                choices = [Choice(self.TEXTS_EN["go_back"], JUMP_GO_BACK_TO_SHIP, True)]
                game_status = GameStatus.RUNNING
            elif param.param_type == ParamType.FAIL:
                choices = []
                game_status = GameStatus.FAIL
            else:  # DEADLY
                choices = []
                game_status = GameStatus.DEAD

            return GameState(
                text=self._substitute(crit_text),
                params_state=self._get_params_state(),
                choices=choices,
                game_status=game_status,
            )

        if state.state == PlayerState.JUMP_AND_NEXT_CRIT:
            jump = self._get_jump(state.last_jump_id)
            return GameState(
                text=self._substitute(jump.description),
                params_state=self._get_params_state(),
                choices=[Choice(self.TEXTS_EN["next"], JUMP_NEXT, True)],
                game_status=GameStatus.RUNNING,
            )

        if state.state == PlayerState.RETURNED_ENDING:
            return GameState(
                text=self._substitute(self.quest.success_text),
                params_state=[],
                choices=[],
                game_status=GameStatus.WIN,
            )

        raise ValueError(f"Unknown state: {state.state}")

    def _apply_param_changes(self, changes: list[ParameterChange]) -> list[int]:
        """Apply parameter changes and return list of triggered critical params."""
        old_values = self._state.param_values.copy()
        new_values = self._state.param_values.copy()
        crit_params = []

        for i, change in enumerate(changes):
            param = self.quest.params[i]

            # Handle showing type
            if change.showing_type == ParameterShowingType.SHOW:
                self._state.param_show[i] = True
            elif change.showing_type == ParameterShowingType.HIDE:
                self._state.param_show[i] = False

            # Apply change
            if change.is_change_value:
                new_values[i] = change.change
            elif change.is_change_percentage:
                new_values[i] = round(old_values[i] * (100 + change.change) / 100)
            elif change.is_change_formula and change.changing_formula:
                new_values[i] = parse(change.changing_formula, old_values)
            else:
                new_values[i] = old_values[i] + change.change

            # Clamp to bounds
            new_values[i] = max(param.min_val, min(param.max_val, new_values[i]))

            # Check for critical parameter
            if new_values[i] != old_values[i] and param.param_type != ParamType.NORMAL:
                is_crit = (
                    param.crit_type == ParamCritType.MAX and new_values[i] == param.max_val
                ) or (param.crit_type == ParamCritType.MIN and new_values[i] == param.min_val)
                if is_crit:
                    crit_params.append(i)

        self._state.param_values = new_values
        return crit_params

    def _check_jump_conditions(self, jump: Jump) -> bool:
        """Check if a jump's conditions are met."""
        for i, cond in enumerate(jump.params_conditions):
            if not self.quest.params[i].active:
                continue

            val = self._state.param_values[i]

            # Range check
            if val < cond.must_from or val > cond.must_to:
                return False

            # Equal values check
            if cond.must_equal_values:
                is_equal = val in cond.must_equal_values
                if cond.must_equal_values_equal and not is_equal:
                    return False
                if not cond.must_equal_values_equal and is_equal:
                    return False

            # Mod values check
            if cond.must_mod_values:
                is_mod = any(val % m == 0 for m in cond.must_mod_values)
                if cond.must_mod_values_mod and not is_mod:
                    return False
                if not cond.must_mod_values_mod and is_mod:
                    return False

        # Formula check
        if jump.formula_to_pass:
            if parse(jump.formula_to_pass, self._state.param_values) == 0:
                return False

        # Jump count limit check
        if jump.jumping_count_limit:
            if self._state.jumped_count.get(jump.id, 0) >= jump.jumping_count_limit:
                return False

        return True

    def _calculate_location(self) -> None:
        """Calculate the state for the current location."""
        loc_id = self._state.location_id
        location = self._get_location(loc_id)

        # Update visit count
        self._state.location_visit_count[loc_id] = (
            self._state.location_visit_count.get(loc_id, 0) + 1
        )

        # Apply day passage
        if location.day_passed:
            self._state.days_passed += 1

        # Apply parameter changes
        crit_params = self._apply_param_changes(location.params_changes)

        # Find available jumps from this location
        all_jumps = [j for j in self.quest.jumps if j.from_location_id == loc_id]

        # Filter and evaluate jumps
        possible_jumps: list[tuple[Jump, bool]] = []
        for jump in all_jumps:
            # Check destination location visit limit
            to_loc = self._get_location(jump.to_location_id)
            if to_loc.max_visits:
                visits = self._state.location_visit_count.get(jump.to_location_id, 0)
                if visits >= to_loc.max_visits:
                    continue

            # TGE behavior: Filter out jumps leading to locations where all outgoing
            # jumps have exhausted their jumpingCountLimit (dead-end prevention)
            jumps_from_dest = [
                j for j in self.quest.jumps if j.from_location_id == jump.to_location_id
            ]
            if jumps_from_dest:
                all_exhausted = all(
                    j.jumping_count_limit
                    and self._state.jumped_count.get(j.id, 0) >= j.jumping_count_limit
                    for j in jumps_from_dest
                )
                if all_exhausted:
                    continue

            active = self._check_jump_conditions(jump)
            possible_jumps.append((jump, active))

        # Sort by showing order
        possible_jumps.sort(key=lambda x: (x[0].showing_order, random.random()))

        # Handle duplicate texts and probability
        seen_texts: set[str] = set()
        final_jumps: list[tuple[int, bool]] = []

        for jump, active in possible_jumps:
            text = jump.text

            if text in seen_texts:
                continue

            same_text_jumps = [(j, a) for j, a in possible_jumps if j.text == text]

            if len(same_text_jumps) == 1:
                # Single jump with this text - apply probability
                if active and jump.prio < 1:
                    active = random.random() < jump.prio

                if active or jump.always_show:
                    final_jumps.append((jump.id, active))
                    seen_texts.add(text)
            else:
                # Multiple jumps with same text - weighted random
                active_jumps = [(j, a) for j, a in same_text_jumps if a]
                if active_jumps:
                    max_prio = max(j.prio for j, _ in active_jumps)
                    candidates = [(j, a) for j, a in active_jumps if j.prio * 100 >= max_prio]
                    total_prio = sum(j.prio for j, _ in candidates)

                    rnd = random.random() * total_prio
                    for j, _a in candidates:
                        if j.prio >= rnd:
                            final_jumps.append((j.id, True))
                            seen_texts.add(text)
                            break
                        rnd -= j.prio
                else:
                    # No active jumps - check always_show
                    for j, _a in same_text_jumps:
                        if j.always_show:
                            final_jumps.append((j.id, False))
                            seen_texts.add(text)
                            break

        # Filter out empty text jumps if there are non-empty ones
        non_empty = [(jid, a) for jid, a in final_jumps if self._get_jump(jid).text]
        if non_empty:
            final_jumps = non_empty
        else:
            # Keep only one active empty-text jump
            active_empty = [(jid, a) for jid, a in final_jumps if a]
            final_jumps = active_empty[:1] if active_empty else []

        self._state.possible_jumps = final_jumps

        # Handle critical parameter
        if crit_params:
            crit_id = crit_params[0]
            param = self.quest.params[crit_id]

            # Check if we should ignore crit due to available active choices
            has_active_choices = any(a for _, a in self._state.possible_jumps)
            is_fail_or_deadly = param.param_type in (ParamType.FAIL, ParamType.DEADLY)

            if not (is_fail_or_deadly and has_active_choices):
                last_jump = None
                if self._state.last_jump_id is not None:
                    last_jump = self._get_jump(self._state.last_jump_id)

                if location.is_empty and last_jump and last_jump.description:
                    self._state.state = PlayerState.CRIT_ON_LOCATION
                else:
                    self._state.state = PlayerState.CRIT_ON_LOCATION_LAST

                self._state.crit_param_id = crit_id
                return

        # Auto-jump if only one choice with empty text and empty location
        game_state = self.get_state()
        if len(game_state.choices) == 1:
            choice = game_state.choices[0]
            if choice.active and choice.jump_id not in (
                JUMP_I_AGREE,
                JUMP_NEXT,
                JUMP_GO_BACK_TO_SHIP,
            ):
                jump = self._get_jump(choice.jump_id)
                last_jump = None
                if self._state.last_jump_id is not None:
                    last_jump = self._get_jump(self._state.last_jump_id)

                if not jump.text and location.is_empty:
                    if not last_jump or not last_jump.description:
                        if not game_state.text:
                            self.perform_jump(jump.id)

    def perform_jump(self, jump_id: int) -> None:
        """Perform a jump (player choice)."""
        if jump_id == JUMP_GO_BACK_TO_SHIP:
            self._state.state = PlayerState.RETURNED_ENDING
            return

        state = self._state

        if state.state == PlayerState.STARTING:
            state.state = PlayerState.LOCATION
            self._calculate_location()

        elif state.state == PlayerState.JUMP:
            jump = self._get_jump(state.last_jump_id)
            state.location_id = jump.to_location_id
            state.state = PlayerState.LOCATION
            self._calculate_location()

        elif state.state == PlayerState.LOCATION:
            # Validate jump is available
            if not any(jid == jump_id for jid, _ in state.possible_jumps):
                raise ValueError(f"Jump {jump_id} not available")

            jump = self._get_jump(jump_id)
            state.last_jump_id = jump_id

            if jump.day_passed:
                state.days_passed += 1

            state.jumped_count[jump_id] = state.jumped_count.get(jump_id, 0) + 1

            crit_params = self._apply_param_changes(jump.params_changes)
            next_location = self._get_location(jump.to_location_id)

            if not jump.description:
                if crit_params:
                    state.state = PlayerState.CRIT_ON_JUMP
                    state.crit_param_id = crit_params[0]
                else:
                    state.location_id = next_location.id
                    state.state = PlayerState.LOCATION
                    self._calculate_location()
            else:
                if crit_params:
                    state.state = PlayerState.JUMP_AND_NEXT_CRIT
                    state.crit_param_id = crit_params[0]
                elif next_location.is_empty:
                    state.location_id = next_location.id
                    state.state = PlayerState.LOCATION
                    self._calculate_location()
                else:
                    state.state = PlayerState.JUMP

        elif state.state == PlayerState.JUMP_AND_NEXT_CRIT:
            state.state = PlayerState.CRIT_ON_JUMP

        elif state.state == PlayerState.CRIT_ON_LOCATION:
            state.state = PlayerState.CRIT_ON_LOCATION_LAST

        else:
            raise ValueError(f"Cannot perform jump in state {state.state}")

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.get_state().game_status != GameStatus.RUNNING

    def get_active_choices(self) -> list[Choice]:
        """Get only the active (selectable) choices."""
        return [c for c in self.get_state().choices if c.active]
