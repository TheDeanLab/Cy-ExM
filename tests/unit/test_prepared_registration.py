from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fakes import FakeBackend, FakeImage, fake_image

import cyexm.registration.whole_image as whole_image
from cyexm.errors import CacheValidationError, RegistrationStageError
from cyexm.registration.artifacts import (
    ArtifactStore,
    CompletionRecord,
    FileIdentity,
    StageRequest,
)
from cyexm.registration.chunked import ChunkInfo
from cyexm.registration.prepared import (
    PreparedManifest,
    PreparedRoundPaths,
    commit_prepared_round,
    read_prepared_region,
)
from cyexm.registration.whole_image import ImageRegistration
from cyexm.runtime import NATIVE_THREAD_ENV_VARS, ResolvedNativeThreads
from cyexm.spatial import ImageGeometry

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def resolved_native_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> ResolvedNativeThreads:
    values = {name: index + 1 for index, name in enumerate(NATIVE_THREAD_ENV_VARS)}
    resolved = ResolvedNativeThreads(
        values=values,
        sources={name: "explicit" for name in values},
    )
    monkeypatch.setattr(
        whole_image, "configure_native_threads", lambda config: resolved
    )
    return resolved


def _registration(
    tmp_path: Path,
    *,
    backend: FakeBackend | None = None,
    force_override: bool = False,
) -> ImageRegistration:
    fixed = tmp_path / "fixed.tif"
    moving = tmp_path / "moving.tif"
    fixed.write_bytes(b"fixed")
    moving.write_bytes(b"moving")
    return ImageRegistration(
        fixed,
        moving,
        tmp_path / "output",
        imaging_round=3,
        pixel_size=(2.0, 3.0, 4.0),
        enable_logging=False,
        force_override=force_override,
        _backend=backend or FakeBackend(),
    )


def _chunk() -> ChunkInfo:
    return ChunkInfo(
        chunk_id=0,
        z_start=1,
        z_end=4,
        y_start=1,
        y_end=5,
        x_start=2,
        x_end=6,
        z_start_ext=1,
        z_end_ext=5,
        y_start_ext=0,
        y_end_ext=6,
        x_start_ext=1,
        x_end_ext=7,
    )


def test_prepared_paths_use_exact_internal_bundle_names(tmp_path: Path) -> None:
    paths = PreparedRoundPaths.for_round(tmp_path, 7)

    assert paths.directory == tmp_path / "round_7_prepared"
    assert paths.fixed_scalar == paths.directory / "fixed_scalar.npy"
    assert paths.moving_linear_scalar == paths.directory / "moving_linear_scalar.npy"
    assert paths.fixed_mask == paths.directory / "fixed_mask.npy"
    assert paths.moving_linear_mask == paths.directory / "moving_linear_mask.npy"
    assert paths.manifest == paths.directory / "manifest.json"


def test_manifest_transport_rejects_unknown_keys(tmp_path: Path) -> None:
    manifest = commit_prepared_round(_registration(tmp_path))
    payload = manifest.to_dict()
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="keys must be exactly"):
        PreparedManifest.from_dict(payload)


def test_commit_writes_exact_float32_c_arrays_with_completion_records(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    manifest = commit_prepared_round(_registration(tmp_path, backend=backend))
    paths = PreparedRoundPaths.for_round(tmp_path / "output", 3)

    assert set(json.loads(paths.manifest.read_text())) == set(manifest.to_dict())
    for path in paths.array_paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        assert array.dtype == np.dtype(np.float32)
        assert array.shape == manifest.geometry.shape
        assert array.flags.c_contiguous
        assert np.isfinite(array).all()
        assert CompletionRecord.read(ArtifactStore().record_path(path)).artifact == (
            FileIdentity.from_path(path)
        )
    assert CompletionRecord.read(
        ArtifactStore().record_path(paths.manifest)
    ).artifact == FileIdentity.from_path(paths.manifest)
    assert backend.nonlinear_calls == 0


class _SourceMutatingStore(ArtifactStore):
    def __init__(self, source: Path) -> None:
        self.source = source

    def commit(
        self,
        artifact: Path,
        request: StageRequest,
        writer: Callable[[Path], None],
        validator: Callable[[Path], bool],
    ) -> None:
        super().commit(artifact, request, writer, validator)
        if artifact.name == "fixed_scalar.npy":
            self.source.write_bytes(b"mutated-during-prepared-commit")


def test_source_mutation_during_array_commit_stales_partial_bundle(
    tmp_path: Path,
) -> None:
    registration = _registration(tmp_path)
    fixed = Path(registration.fixed_image_path)
    registration._artifact_store = _SourceMutatingStore(fixed)
    paths = PreparedRoundPaths.for_round(tmp_path / "output", 3)

    with pytest.raises(RegistrationStageError, match="identity changed"):
        commit_prepared_round(registration)

    assert not paths.manifest.exists()
    assert ArtifactStore().stale_path(paths.fixed_scalar).is_file()


class _PreparedArrayMutatingStore(ArtifactStore):
    def __init__(self) -> None:
        self.fixed_scalar: Path | None = None

    def commit(
        self,
        artifact: Path,
        request: StageRequest,
        writer: Callable[[Path], None],
        validator: Callable[[Path], bool],
    ) -> None:
        super().commit(artifact, request, writer, validator)
        if artifact.name == "fixed_scalar.npy":
            self.fixed_scalar = artifact
        elif artifact.name == "moving_linear_mask.npy":
            if self.fixed_scalar is None:
                raise RuntimeError("fixed scalar was not committed first")
            array = np.load(self.fixed_scalar, allow_pickle=False)
            np.save(
                self.fixed_scalar,
                np.asarray(array) + np.float32(1.0),
                allow_pickle=False,
            )


def test_array_mutation_before_manifest_capture_fails_closed_and_stales(
    tmp_path: Path,
) -> None:
    registration = _registration(tmp_path)
    registration._artifact_store = _PreparedArrayMutatingStore()
    paths = PreparedRoundPaths.for_round(tmp_path / "output", 3)

    with pytest.raises(CacheValidationError, match="fixed scalar identity changed"):
        commit_prepared_round(registration)

    assert not paths.manifest.exists()
    assert ArtifactStore().stale_path(paths.fixed_scalar).is_file()


class _PostCommitArtifactMutatingStore(ArtifactStore):
    def commit(
        self,
        artifact: Path,
        request: StageRequest,
        writer: Callable[[Path], None],
        validator: Callable[[Path], bool],
    ) -> None:
        super().commit(artifact, request, writer, validator)
        if artifact.name == "fixed_scalar.npy":
            array = np.load(artifact, allow_pickle=False)
            np.save(
                artifact,
                np.asarray(array) + np.float32(1.0),
                allow_pickle=False,
            )


def test_post_commit_array_mutation_cannot_replace_completion_identity(
    tmp_path: Path,
) -> None:
    registration = _registration(tmp_path)
    registration._artifact_store = _PostCommitArtifactMutatingStore()
    paths = PreparedRoundPaths.for_round(tmp_path / "output", 3)

    with pytest.raises(CacheValidationError, match="completion identity mismatch"):
        commit_prepared_round(registration)

    assert not paths.manifest.exists()
    assert ArtifactStore().stale_path(paths.fixed_scalar).is_file()


def test_commit_rejects_nonfinite_prepared_values_without_manifest(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        moving_image=fake_image(
            data=np.full((8, 8, 8), np.float32(np.nan), dtype=np.float32)
        )
    )
    paths = PreparedRoundPaths.for_round(tmp_path / "output", 3)

    with pytest.raises(ValueError, match="finite"):
        commit_prepared_round(_registration(tmp_path, backend=backend))

    assert not paths.manifest.exists()
    assert not ArtifactStore().record_path(paths.manifest).exists()


def test_stale_partial_arrays_are_reused_and_manifest_is_committed(
    tmp_path: Path,
) -> None:
    registration = _registration(tmp_path)
    first = commit_prepared_round(registration)
    paths = PreparedRoundPaths.for_round(tmp_path / "output", 3)
    identities = tuple(FileIdentity.from_path(path) for path in paths.array_paths)
    paths.manifest.unlink()
    ArtifactStore().record_path(paths.manifest).unlink()

    second = commit_prepared_round(registration)

    assert second == first
    assert (
        tuple(FileIdentity.from_path(path) for path in paths.array_paths) == identities
    )
    assert paths.manifest.is_file()


def test_force_replaces_prepared_arrays(tmp_path: Path) -> None:
    commit_prepared_round(_registration(tmp_path))
    paths = PreparedRoundPaths.for_round(tmp_path / "output", 3)
    before = tuple(FileIdentity.from_path(path) for path in paths.array_paths)

    commit_prepared_round(_registration(tmp_path, force_override=True))

    after = tuple(FileIdentity.from_path(path) for path in paths.array_paths)
    assert after != before


def test_regional_read_copies_owned_regions_and_uses_direction_aware_origin(
    tmp_path: Path,
) -> None:
    direction = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    fixed = fake_image(
        data=np.arange(8**3, dtype=np.float32).reshape(8, 8, 8),
        origin=(10.0, 20.0, 30.0),
        spacing=(2.0, 3.0, 4.0),
        direction=direction,
    )
    backend = FakeBackend(fixed_image=fixed)
    manifest = commit_prepared_round(_registration(tmp_path, backend=backend))
    chunk = _chunk()

    region = read_prepared_region(manifest, chunk)

    persisted = np.load(manifest.moving_linear_scalar.path, mmap_mode="r")
    np.testing.assert_array_equal(
        region.moving_linear_scalar,
        persisted[chunk.extended_slices],
    )
    assert type(region.moving_linear_scalar) is np.ndarray
    assert region.moving_linear_scalar.flags.owndata
    assert region.moving_linear_scalar.flags.c_contiguous
    assert region.geometry.shape == chunk.extended_shape
    assert region.geometry.spacing == manifest.geometry.spacing
    assert region.geometry.direction == direction
    assert region.geometry.origin == (10.0, 22.0, 34.0)


def test_regional_read_rechecks_all_array_generations_after_copying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cyexm.registration.prepared as prepared

    manifest = commit_prepared_round(_registration(tmp_path))
    original_copy_region = prepared._copy_region

    def mutate_after_later_copy(
        np_runtime: Any,
        identity: FileIdentity,
        role: str,
        geometry: ImageGeometry,
        chunk: ChunkInfo,
    ) -> object:
        copied = original_copy_region(
            np_runtime,
            identity,
            role,
            geometry,
            chunk,
        )
        if role == "moving linear scalar":
            fixed_path = Path(manifest.fixed_scalar.path)
            fixed = np.load(fixed_path, allow_pickle=False)
            np.save(
                fixed_path,
                np.asarray(fixed) + np.float32(1.0),
                allow_pickle=False,
            )
        return copied

    monkeypatch.setattr(prepared, "_copy_region", mutate_after_later_copy)

    with pytest.raises(CacheValidationError, match="fixed scalar identity mismatch"):
        read_prepared_region(manifest, _chunk())


class _NearlyMismatchedAffineBackend(FakeBackend):
    def linear(self, **kwargs: object) -> tuple[object, object | None, object]:
        moving, moving_mask, affine = super().linear(**kwargs)
        assert isinstance(moving, FakeImage)
        moving.origin = (np.float64(5e-8), 0.0, 0.0)
        return moving, moving_mask, affine


def test_prepared_inputs_require_exact_fixed_geometry(tmp_path: Path) -> None:
    backend = _NearlyMismatchedAffineBackend()

    with pytest.raises(ValueError, match="moving scalar geometry"):
        commit_prepared_round(_registration(tmp_path, backend=backend))


def test_regional_read_rejects_identity_changes_before_mapping(tmp_path: Path) -> None:
    manifest = commit_prepared_round(_registration(tmp_path))
    chunk = _chunk()
    scalar_path = Path(manifest.fixed_scalar.path)

    np.save(scalar_path, np.zeros((8, 8, 8), dtype=np.float64), allow_pickle=False)
    with pytest.raises(CacheValidationError, match="identity"):
        read_prepared_region(manifest, chunk)


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        (np.zeros((8, 8, 8), dtype=np.float64), "float32"),
        (np.zeros((7, 8, 8), dtype=np.float32), "shape"),
    ),
)
def test_regional_read_rejects_array_contract_mismatches(
    tmp_path: Path,
    replacement: np.ndarray,
    message: str,
) -> None:
    manifest = commit_prepared_round(_registration(tmp_path))
    chunk = _chunk()
    scalar_path = Path(manifest.fixed_scalar.path)
    np.save(scalar_path, replacement, allow_pickle=False)
    changed = PreparedManifest.from_dict(manifest.to_dict())
    object.__setattr__(changed, "fixed_scalar", FileIdentity.from_path(scalar_path))
    with pytest.raises(ValueError, match=message):
        read_prepared_region(changed, chunk)
