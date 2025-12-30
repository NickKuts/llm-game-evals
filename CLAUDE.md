# CLAUDE.md

This file provides guidance to Claude Code (and other AI assistants) when working with this repository.

## Project Overview

quest-evals is an LLM benchmark using Space Rangers 2 text quests. It parses .qm/.qmm quest files and runs them interactively with an LLM to evaluate performance on interactive fiction tasks.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Download quest files (from GitLab)
./download_quests.sh

# Run evaluation on a quest
python -m quest_evals.cli Muzon_eng --model claude-sonnet-4.5 --runs 3

# Play a quest manually (human mode)
python -m quest_evals.play assets/json/Muzon_eng.json

# List available quests
python -m quest_evals.cli --list
```

## Project Structure

```
src/quest_evals/
  __init__.py        # Package exports
  qm_parser.py       # Binary QM/QMM file parser
  quest_json.py      # JSON quest format converter
  formula.py         # Formula expression evaluator
  player.py          # Quest state machine
  llm.py             # OpenRouter LLM interface
  runner.py          # Evaluation orchestrator
  checkpoint.py      # Per-step checkpointing for resume
  model_configs.py   # Model definitions (Claude, GPT, Gemini, etc.)
  results.py         # Result logging and aggregation
  cli.py             # Main CLI entry point
  play.py            # Human play mode
  convert.py         # QM to JSON converter CLI
  verify_json.py     # JSON conversion verification

assets/              # Quest files (gitignored, use download_quests.sh)
results/             # Evaluation outputs (gitignored)
tests/               # Pytest tests
```

## Architecture

1. **qm_parser.py**: Parses binary .qm/.qmm quest files into Python dataclasses
2. **quest_json.py**: Converts between binary and human-readable JSON format
3. **formula.py**: Evaluates in-game formulas (arithmetic, conditionals, random ranges)
4. **player.py**: State machine that tracks game state, evaluates conditions, handles transitions
5. **llm.py**: OpenRouter API interface supporting 20+ models
6. **runner.py**: Runs quests with LLM, handles retries and failure reflection
7. **checkpoint.py**: Saves state after each LLM call for resumability

## Key Concepts

- **Parameters**: Variables tracked during quest (resources, progress flags)
- **Locations**: Nodes in the quest graph with text and parameter changes
- **Jumps**: Transitions between locations with conditions and effects
- **Critical parameters**: Params that trigger win/lose when hitting min/max

## Environment

Requires `OPENROUTER_API_KEY` environment variable (get one at https://openrouter.ai/keys).

```bash
cp .env.example .env
# Edit .env with your API key
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check src/ tests/

# Run formatter
ruff format src/ tests/

# Run tests
pytest -v
```
