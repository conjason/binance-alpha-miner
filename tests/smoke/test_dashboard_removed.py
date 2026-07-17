"""
Smoke test: verify that the dashboard/ directory has been removed.
Requirements: 1.1
"""
from pathlib import Path

# Project root is two levels up from this file (tests/smoke/test_dashboard_removed.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_dashboard_directory_does_not_exist():
    """Assert that the dashboard/ directory no longer exists in the project root."""
    dashboard_dir = PROJECT_ROOT / "dashboard"
    assert not dashboard_dir.exists(), (
        f"dashboard/ directory still exists at {dashboard_dir}. "
        "Task 1.1 requires it to be deleted."
    )
