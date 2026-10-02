"""Smoke tests for the package scaffold."""

from importlib.metadata import version

import gridfm_trustlayer


def test_import() -> None:
    """The package should import and expose a version string."""
    assert gridfm_trustlayer.__version__


def test_version_matches_installed_metadata() -> None:
    """`_version.py` is the single source of the version used by the build."""
    assert gridfm_trustlayer.__version__ == version("gridfm-trustlayer")
