"""Smoke test: package is importable and has a version."""

from proof_of_replica import __version__


def test_version_is_set():
    assert __version__ == "0.1.0"
