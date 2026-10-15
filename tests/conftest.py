"""Shared test setup."""

from __future__ import annotations

import pytest

from gridfm_trustlayer.cli.main import DEBUG_VARIABLE


@pytest.fixture(autouse=True)
def _cli_errors_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the CLI raise its errors, so that tests can match them.

    Users get a one-line message instead (see `test_cli_reports_errors_in_one_line`).
    """

    monkeypatch.setenv(DEBUG_VARIABLE, "1")
