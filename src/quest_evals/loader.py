"""Unified quest loader for QM binary and JSON formats."""

from __future__ import annotations

from pathlib import Path

from .models import Quest
from .qm_parser import parse_qm
from .quest_json import dict_to_quest, load_quest_json, quest_to_dict


def load_quest(path: str | Path) -> Quest:
    """Load a quest from either a .qm/.qmm binary file or .json file.

    Args:
        path: Path to quest file (.qm, .qmm, or .json)

    Returns:
        Parsed Quest object

    Raises:
        ValueError: If file extension is not recognized
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".qm", ".qmm"):
        with open(path, "rb") as f:
            return parse_qm(f.read())
    elif ext == ".json":
        return load_quest_json(path)
    else:
        raise ValueError(f"Unknown quest file extension: {ext}")


def convert_to_json(quest: Quest) -> dict:
    """Convert a Quest to JSON-serializable dict.

    This is the format that should be shipped with the evaluation code.
    """
    return quest_to_dict(quest)


def load_from_dict(data: dict) -> Quest:
    """Load a Quest from a JSON dict."""
    return dict_to_quest(data)
