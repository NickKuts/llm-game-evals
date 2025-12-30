"""CLI tool for converting QM/QMM quest files to JSON format.

Usage:
    python -m quest_evals.convert assets/*.qm
    python -m quest_evals.convert input.qm -o output.json
    python -m quest_evals.convert assets/ --output-dir quests/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .qm_parser import parse_qm
from .quest_json import save_quest_json


def convert_file(qm_path: Path, json_path: Path, verbose: bool = True) -> bool:
    """Convert a single QM file to JSON.

    Returns True on success, False on error.
    """
    try:
        with open(qm_path, "rb") as f:
            quest = parse_qm(f.read())

        save_quest_json(quest, json_path)

        if verbose:
            print(f"  ✓ {qm_path.name} → {json_path.name}")
        return True

    except Exception as e:
        if verbose:
            print(f"  ✗ {qm_path.name}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Space Rangers QM/QMM quest files to JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s assets/Codebox_eng.qm              # Convert single file
  %(prog)s assets/*.qm                        # Convert multiple files
  %(prog)s assets/ -o quests/                 # Convert directory
  %(prog)s input.qm -o output.json            # Specify output name
        """,
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="Input QM/QMM file(s) or directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON file or directory (default: same location as input)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    # Collect all input files
    input_files: list[Path] = []
    for inp in args.input:
        if inp.is_dir():
            input_files.extend(inp.glob("*.qm"))
            input_files.extend(inp.glob("*.qmm"))
        elif inp.suffix.lower() in (".qm", ".qmm"):
            input_files.append(inp)
        else:
            print(f"Skipping non-QM file: {inp}", file=sys.stderr)

    if not input_files:
        print("No QM/QMM files found", file=sys.stderr)
        sys.exit(1)

    # Determine output
    if args.output:
        if len(input_files) == 1 and args.output.suffix == ".json":
            # Single file with explicit output name
            output_files = [args.output]
        elif args.output.is_dir() or len(input_files) > 1:
            # Output directory
            output_dir = args.output
            output_dir.mkdir(parents=True, exist_ok=True)
            output_files = [output_dir / f.with_suffix(".json").name for f in input_files]
        else:
            # Treat as directory
            output_dir = args.output
            output_dir.mkdir(parents=True, exist_ok=True)
            output_files = [output_dir / f.with_suffix(".json").name for f in input_files]
    else:
        # Same location as input
        output_files = [f.with_suffix(".json") for f in input_files]

    if not args.quiet:
        print(f"Converting {len(input_files)} file(s)...")

    success_count = 0
    for qm_path, json_path in zip(input_files, output_files, strict=False):
        if convert_file(qm_path, json_path, verbose=not args.quiet):
            success_count += 1

    if not args.quiet:
        print(f"\nConverted {success_count}/{len(input_files)} files")

    if success_count < len(input_files):
        sys.exit(1)


if __name__ == "__main__":
    main()
