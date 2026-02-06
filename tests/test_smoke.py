"""Smoke tests for the template package."""

from python_workspace_template import __version__


def test_package_version() -> None:
    """Ensure the base package is importable."""
    assert __version__ == "0.1.0"
