"""
Unit tests for pyproject.toml packaging metadata (TASK-31, rail 1).

Guards against essentia-tensorflow's dependency pin drifting, or being
"helpfully" promoted from the optional `embeddings` extra into core
[project.dependencies]. The pinned dev-line build
(essentia-tensorflow==2.1b6.dev1389) publishes no linux-aarch64 wheel for
any CPython version, and it bundles a TensorFlow C library -- core (GUI/CLI)
users must never be forced to carry either cost just to run Playchitect
without ever touching the embeddings feature. See prd.json TASK-31 / the
TASK-19 note for the full rationale; the pin decision itself is settled and
is not relitigated here, only guarded against regression.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"
_EXPECTED_ESSENTIA_SPEC = "essentia-tensorflow==2.1b6.dev1389"


def _load_pyproject() -> dict[str, Any]:
    with open(_PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


class TestEssentiaPin:
    """Rail 1: essentia-tensorflow must be exactly pinned inside the optional extra."""

    def test_exact_pin_present_in_embeddings_extra(self) -> None:
        data = _load_pyproject()
        embeddings_extra = data["project"]["optional-dependencies"]["embeddings"]
        assert _EXPECTED_ESSENTIA_SPEC in embeddings_extra

    def test_embeddings_extra_has_no_stray_unpinned_essentia_entry(self) -> None:
        """Guard against a second, unpinned essentia-tensorflow entry sneaking
        in alongside the pinned one (e.g. during a careless merge) -- pip
        would silently prefer whichever resolves, defeating the pin."""
        data = _load_pyproject()
        embeddings_extra = data["project"]["optional-dependencies"]["embeddings"]
        essentia_entries = [dep for dep in embeddings_extra if "essentia" in dep.lower()]
        assert essentia_entries == [_EXPECTED_ESSENTIA_SPEC]

    def test_core_dependencies_do_not_mention_essentia(self) -> None:
        """Guardrail against essentia-tensorflow being 'helpfully' promoted to
        a core dependency. That build publishes no linux-aarch64 wheel for
        any CPython version, and core (GUI/CLI) users must not be forced to
        carry a bundled TensorFlow C library just to run Playchitect."""
        data = _load_pyproject()
        core_deps = data["project"]["dependencies"]
        essentia_in_core = [dep for dep in core_deps if "essentia" in dep.lower()]
        assert essentia_in_core == []

    def test_no_other_optional_dependency_group_carries_essentia(self) -> None:
        """essentia-tensorflow must live in exactly one place: the
        `embeddings` extra -- not duplicated into `gui`, `dev`, etc."""
        data = _load_pyproject()
        optional_deps = data["project"]["optional-dependencies"]
        groups_with_essentia = [
            group
            for group, deps in optional_deps.items()
            if any("essentia" in dep.lower() for dep in deps)
        ]
        assert groups_with_essentia == ["embeddings"]

    def test_embeddings_extra_still_exists_as_a_list_of_strings(self) -> None:
        """Sanity check on the parse itself: catches a badly-formed TOML edit
        (e.g. turning the extra into a table) that would otherwise make the
        other assertions in this class pass vacuously."""
        data = _load_pyproject()
        embeddings_extra = data["project"]["optional-dependencies"]["embeddings"]
        assert isinstance(embeddings_extra, list)
        assert all(isinstance(dep, str) for dep in embeddings_extra)
        assert len(embeddings_extra) >= 1
