"""Smoke tests for the initial package scaffold."""

import gridfm_trustlayer


def test_import() -> None:
    """The package should import and expose a version string."""
    assert gridfm_trustlayer.__version__ == "0.0.1"
