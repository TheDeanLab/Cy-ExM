from __future__ import annotations

from pathlib import Path

import pytest

from cyexm.errors import CacheValidationError
from cyexm.io.atomic import atomic_write


def test_atomic_write_preserves_compound_extension(tmp_path: Path) -> None:
    destination = tmp_path / "warp.nii.gz"
    seen: list[Path] = []

    def writer(path: Path) -> None:
        seen.append(path)
        path.write_bytes(b"warp")

    atomic_write(destination, writer, lambda path: path.read_bytes() == b"warp")

    assert destination.read_bytes() == b"warp"
    assert seen[0].parent == destination.parent
    assert seen[0].name.endswith(".nii.gz")
    assert seen[0] != destination


def test_failed_write_keeps_previous_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "transform.mat"
    destination.write_bytes(b"old")

    def writer(path: Path) -> None:
        path.write_bytes(b"partial")
        raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        atomic_write(destination, writer, lambda path: True)

    assert destination.read_bytes() == b"old"
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_failed_validation_keeps_previous_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "transform.mat"
    destination.write_bytes(b"old")

    def writer(path: Path) -> None:
        path.write_bytes(b"invalid")

    with pytest.raises(CacheValidationError, match="failed validation"):
        atomic_write(destination, writer, lambda path: False)

    assert destination.read_bytes() == b"old"
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_missing_temporary_artifact_fails_validation(tmp_path: Path) -> None:
    destination = tmp_path / "transform.mat"

    with pytest.raises(CacheValidationError, match="failed validation"):
        atomic_write(destination, lambda path: None, lambda path: True)

    assert not destination.exists()
    assert list(tmp_path.glob(".*.tmp*")) == []
