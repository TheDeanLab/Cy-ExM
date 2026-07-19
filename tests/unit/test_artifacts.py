from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import cyexm.io.atomic as atomic_module
from cyexm.errors import CacheValidationError
from cyexm.registration.artifacts import (
    ArtifactStore,
    CompletionRecord,
    FileIdentity,
    StageRequest,
)
from cyexm.spatial import ImageGeometry


def make_request(source: Path, accuracy: str = "low") -> StageRequest:
    return StageRequest.for_files(
        stage="affine",
        inputs=(source,),
        geometry=ImageGeometry(
            shape=(8, 16, 16),
            origin=(0.0, 0.0, 0.0),
            spacing=(0.15, 0.15, 0.4),
            direction=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        ),
        method="TRSAA",
        accuracy=accuracy,
        native_threads={"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": 1},
        upstream=None,
        parameters={"metric": "mattes", "sampling": [0.2, 0.1]},
    )


def write_bytes(value: bytes):
    def writer(path: Path) -> None:
        path.write_bytes(value)

    return writer


def assert_json_primitives(value: Any) -> None:
    if value is None or isinstance(value, str | int | float | bool):
        return
    if isinstance(value, list):
        for item in value:
            assert_json_primitives(item)
        return
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    for item in value.values():
        assert_json_primitives(item)


def test_file_identity_captures_resolved_path_size_and_mtime(tmp_path: Path) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    alias = tmp_path / "moving-link.tif"
    alias.symlink_to(source)

    identity = FileIdentity.from_path(alias)
    stat = source.stat()

    assert identity.path == str(source.resolve(strict=True))
    assert identity.st_size == stat.st_size
    assert identity.st_mtime_ns == stat.st_mtime_ns


def test_file_identity_requires_an_existing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FileIdentity.from_path(tmp_path / "missing.tif")


def test_stage_request_snapshots_input_identities(tmp_path: Path) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    captured = make_request(source)
    identity = captured.inputs[0]

    source.write_bytes(b"changed-source")

    assert captured.inputs[0] == identity
    assert make_request(source).inputs[0] != identity


def test_completion_record_is_canonical_json_with_fresh_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    artifact = tmp_path / "linear_transform_2.mat"
    stage_request = make_request(source)
    store = ArtifactStore()

    store.commit(
        artifact,
        stage_request,
        writer=write_bytes(b"affine"),
        validator=lambda path: path.read_bytes() == b"affine",
    )

    record_path = store.record_path(artifact)
    serialized = record_path.read_text(encoding="utf-8")
    payload = json.loads(serialized)
    record = CompletionRecord.read(record_path)

    assert serialized.endswith("\n")
    assert serialized == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert_json_primitives(payload)
    assert payload["schema_version"] == 1
    assert payload["request"] == stage_request.to_dict()
    assert payload["artifact"] == FileIdentity.from_path(artifact).to_dict()
    assert record == CompletionRecord(
        schema_version=1,
        request=stage_request,
        artifact=FileIdentity.from_path(artifact),
    )


def test_commit_then_reuse(tmp_path: Path) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    artifact = tmp_path / "linear_transform_2.mat"
    store = ArtifactStore()
    store.commit(
        artifact,
        make_request(source),
        writer=write_bytes(b"affine"),
        validator=lambda path: path.read_bytes() == b"affine",
    )

    value = store.reuse(artifact, make_request(source), Path.read_bytes)

    assert value == b"affine"


def test_absent_artifact_and_record_are_a_cache_miss(tmp_path: Path) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")

    value = ArtifactStore().reuse(
        tmp_path / "linear_transform_2.mat",
        make_request(source),
        Path.read_bytes,
    )

    assert value is None


def test_modified_artifact_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    artifact = tmp_path / "linear_transform_2.mat"
    store = ArtifactStore()
    store.commit(
        artifact,
        make_request(source),
        writer=write_bytes(b"affine"),
        validator=lambda path: True,
    )
    artifact.write_bytes(b"corrupt-longer")

    with pytest.raises(CacheValidationError, match="identity"):
        store.reuse(artifact, make_request(source), Path.read_bytes)


def test_mismatched_request_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    artifact = tmp_path / "linear_transform_2.mat"
    store = ArtifactStore()
    store.commit(
        artifact,
        make_request(source, "low"),
        writer=write_bytes(b"affine"),
        validator=lambda path: True,
    )

    with pytest.raises(CacheValidationError, match="request"):
        store.reuse(artifact, make_request(source, "high"), Path.read_bytes)


def test_changed_input_file_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    artifact = tmp_path / "linear_transform_2.mat"
    store = ArtifactStore()
    store.commit(
        artifact,
        make_request(source),
        writer=write_bytes(b"affine"),
        validator=lambda path: True,
    )
    source.write_bytes(b"changed-source")

    with pytest.raises(CacheValidationError, match="request"):
        store.reuse(artifact, make_request(source), Path.read_bytes)


def test_bare_artifact_is_an_incomplete_cache_error(tmp_path: Path) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    artifact = tmp_path / "linear_transform_2.mat"
    artifact.write_bytes(b"unexplained")

    with pytest.raises(CacheValidationError, match="Incomplete cache entry"):
        ArtifactStore().reuse(artifact, make_request(source), Path.read_bytes)


def test_record_without_artifact_is_an_incomplete_cache_error(tmp_path: Path) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    artifact = tmp_path / "linear_transform_2.mat"
    store = ArtifactStore()
    store.commit(
        artifact,
        make_request(source),
        writer=write_bytes(b"affine"),
        validator=lambda path: True,
    )
    artifact.unlink()

    with pytest.raises(CacheValidationError, match="Incomplete cache entry"):
        store.reuse(artifact, make_request(source), Path.read_bytes)


def test_unreadable_artifact_is_a_cache_validation_error(tmp_path: Path) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    artifact = tmp_path / "linear_transform_2.mat"
    store = ArtifactStore()
    store.commit(
        artifact,
        make_request(source),
        writer=write_bytes(b"affine"),
        validator=lambda path: True,
    )

    def unreadable(path: Path) -> bytes:
        raise ValueError(f"cannot read {path}")

    with pytest.raises(CacheValidationError, match="unreadable") as error:
        store.reuse(artifact, make_request(source), unreadable)

    assert isinstance(error.value.__cause__, ValueError)


def test_malformed_completion_record_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    artifact = tmp_path / "linear_transform_2.mat"
    store = ArtifactStore()
    store.commit(
        artifact,
        make_request(source),
        writer=write_bytes(b"affine"),
        validator=lambda path: True,
    )
    store.record_path(artifact).write_text("{not-json", encoding="utf-8")

    with pytest.raises(CacheValidationError, match="completion record"):
        store.reuse(artifact, make_request(source), Path.read_bytes)


def test_failed_replacement_leaves_previous_cache_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    artifact = tmp_path / "linear_transform_2.mat"
    store = ArtifactStore()
    stage_request = make_request(source)
    store.commit(
        artifact,
        stage_request,
        writer=write_bytes(b"old-affine"),
        validator=lambda path: True,
    )
    previous_record = store.record_path(artifact).read_bytes()

    def fail_replace(source_path: Path, destination_path: Path) -> None:
        raise OSError(f"cannot replace {source_path} with {destination_path}")

    monkeypatch.setattr(atomic_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="cannot replace"):
        store.commit(
            artifact,
            stage_request,
            writer=write_bytes(b"new-affine"),
            validator=lambda path: True,
        )

    assert artifact.read_bytes() == b"old-affine"
    assert store.record_path(artifact).read_bytes() == previous_record
    assert store.reuse(artifact, stage_request, Path.read_bytes) == b"old-affine"
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_mark_stale_is_an_explicit_miss_without_deleting_artifact(
    tmp_path: Path,
) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    artifact = tmp_path / "linear_transform_2.mat"
    store = ArtifactStore()
    stage_request = make_request(source)
    store.commit(
        artifact,
        stage_request,
        writer=write_bytes(b"affine"),
        validator=lambda path: True,
    )

    store.mark_stale(artifact)

    assert artifact.read_bytes() == b"affine"
    assert not store.record_path(artifact).exists()
    assert store.stale_path(artifact).is_file()
    assert store.reuse(artifact, stage_request, Path.read_bytes) is None


def test_successful_replacement_removes_stale_marker(tmp_path: Path) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    artifact = tmp_path / "linear_transform_2.mat"
    store = ArtifactStore()
    stage_request = make_request(source)
    store.commit(
        artifact,
        stage_request,
        writer=write_bytes(b"old-affine"),
        validator=lambda path: True,
    )
    store.mark_stale(artifact)

    store.commit(
        artifact,
        stage_request,
        writer=write_bytes(b"new-affine"),
        validator=lambda path: True,
    )

    assert not store.stale_path(artifact).exists()
    assert store.record_path(artifact).is_file()
    assert store.reuse(artifact, stage_request, Path.read_bytes) == b"new-affine"
