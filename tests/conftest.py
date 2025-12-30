"""
Pytest configuration for quest-evals tests.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "smoke: quick smoke test for CI/CD")
    config.addinivalue_line("markers", "integration: integration tests requiring API keys")
