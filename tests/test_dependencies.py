"""The CI job for the lowest supported versions pins exactly the declared minimums."""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - pytest installs tomli on Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def _release(version: str) -> tuple[int, ...]:
    """`2.3` and `2.3.0` are the same release."""

    numbers = [int(part) for part in version.split(".")]
    while numbers and numbers[-1] == 0:
        numbers.pop()
    return tuple(numbers)


def test_minimum_versions_file_pins_the_declared_minimums() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {}
    for requirement in project["project"]["dependencies"]:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)>=([0-9.]+)", requirement)
        assert match, f"{requirement!r}: give each dependency a '>=' minimum"
        declared[match.group(1).lower()] = _release(match.group(2))

    pinned = {}
    lines = (ROOT / ".github" / "minimum-versions.txt").read_text(encoding="utf-8")
    for line in lines.splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.split("==")
            pinned[name.strip().lower()] = _release(version.strip())

    assert pinned == declared
