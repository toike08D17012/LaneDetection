"""Smoke tests for the lane_detection package."""

from lane_detection import __version__


def test_package_version() -> None:
    """Ensure the lane_detection package is importable and version is set."""
    assert __version__ == "0.1.0"
