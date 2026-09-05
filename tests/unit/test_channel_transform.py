from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from fakes import FakeBackend, fake_image

import cyexm.io.atomic as atomic_module
import cyexm.registration as registration_namespace
import cyexm.registration.artifacts as artifacts_module
import cyexm.registration.channels as channels_module
from cyexm.errors import CacheValidationError, RegistrationStageError
from cyexm.registration.artifacts import (
    ArtifactStore,
    CompletionRecord,
    FileIdentity,
)
from cyexm.registration.channels import transform_channel
from cyexm.runtime import NATIVE_THREAD_ENV_VARS, NativeThreadConfig
from cyexm.spatial import AXIS_CONVENTION_VERSION, PixelSize

pytestmark = pytest.mark.unit


def source_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    fixed = tmp_path / "fixed.tif"
    moving = tmp_path / "moving.tif"
    affine = tmp_path / "linear.mat"
    warp = tmp_path / "warp.nii.gz"
    fixed.write_bytes(b"fixed")
    moving.write_bytes(b"moving")
    affine.write_bytes(b"affine")
    warp.write_bytes(b"warp")
    return fixed, moving, affine, warp


def run_transform(
    tmp_path: Path,
    *,
    backend: FakeBackend | None = None,
    store: ArtifactStore | None = None,
    fixed: Path | None = None,
    moving: Path | None = None,
    affine: Path | None = None,
    warp: Path | None = None,
    output: Path | None = None,
    pixel_size: object = (0.324, 0.324, 0.9),
    interpolation: object = "linear",
    force_override: object = False,
) -> Path:
    sources = source_files(tmp_path) if fixed is None else None
    return transform_channel(
        fixed_image_path=fixed if fixed is not None else sources[0],  # type: ignore[index]
        moving_image_path=moving if moving is not None else sources[1],  # type: ignore[index]
        affine_path=affine if affine is not None else sources[2],  # type: ignore[index]
        warp_path=warp if warp is not None else sources[3],  # type: ignore[index]
        output_path=output or (tmp_path / "registered-channel.tif"),
        pixel_size=cast("PixelSize", pixel_size),
        interpolation=cast("str", interpolation),
        native_threads=NativeThreadConfig(1, 1),
        force_override=force_override,  # type: ignore[arg-type]
        _backend=backend or FakeBackend(),
        _artifact_store=store,
    )


def test_transform_channel_commits_transform_identity(tmp_path: Path) -> None:
    fixed, moving, affine, warp = source_files(tmp_path)
    output = tmp_path / "registered-channel.tif"
    backend = FakeBackend()

    result = transform_channel(
        fixed_image_path=fixed,
        moving_image_path=moving,
        affine_path=affine,
        warp_path=warp,
        output_path=output,
        pixel_size=(0.324, 0.324, 0.9),
        interpolation="linear",
        native_threads=NativeThreadConfig(1, 1),
        _backend=backend,
    )

    record = CompletionRecord.read(output.with_name(output.name + ".complete.json"))
    assert result == output.resolve()
    assert backend.transform_lists == [(warp.resolve(), affine.resolve())]
    assert backend.transform_inversions == [(False, False)]
    assert backend.transform_interpolations == ["linear"]
    assert record.request.stage == "channel-transform-v1"
    assert record.request.inputs == tuple(
        FileIdentity.from_path(path) for path in (fixed, moving, affine, warp)
    )
    assert record.request.upstream == FileIdentity.from_path(warp)
    assert record.request.method == "warp-affine"
    assert record.request.accuracy == "not-applicable"
    assert record.request.parameters == {
        "pixel_size_xyz_um": [0.324, 0.324, 0.9],
        "interpolation": "linear",
        "transform_order": ["warp", "affine"],
        "whichtoinvert": [False, False],
        "axis_convention": AXIS_CONVENTION_VERSION,
        "output_format": "ome-compatible-tiff-uint16-v1",
        "postprocessing": "clip-uint16-no-normalization-v1",
    }
    assert record.request.geometry == backend.geometry(backend.clipped_tiff_images[0])
    assert output.read_bytes().startswith(b"tiff-fake:")
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_transform_channel_reuses_exact_readable_artifact_without_transform(
    tmp_path: Path,
) -> None:
    sources = source_files(tmp_path)
    output = tmp_path / "registered-channel.tif"
    run_transform(
        tmp_path,
        backend=FakeBackend(),
        fixed=sources[0],
        moving=sources[1],
        affine=sources[2],
        warp=sources[3],
        output=output,
    )
    cached_identity = FileIdentity.from_path(output)
    cached_record = output.with_name(output.name + ".complete.json").read_bytes()
    resumed = FakeBackend()

    result = run_transform(
        tmp_path,
        backend=resumed,
        fixed=sources[0],
        moving=sources[1],
        affine=sources[2],
        warp=sources[3],
        output=output,
    )

    assert result == output.resolve()
    assert resumed.transform_lists == []
    assert resumed.clipped_tiff_writes == []
    assert FileIdentity.from_path(output) == cached_identity
    assert (
        output.with_name(output.name + ".complete.json").read_bytes() == cached_record
    )


@pytest.mark.parametrize(
    "source_index",
    [
        pytest.param(0, id="fixed"),
        pytest.param(1, id="moving"),
        pytest.param(2, id="affine"),
        pytest.param(3, id="warp"),
    ],
)
def test_transform_channel_rejects_changed_input_identity_without_recomputing(
    tmp_path: Path,
    source_index: int,
) -> None:
    sources = source_files(tmp_path)
    output = tmp_path / "registered-channel.tif"
    run_transform(
        tmp_path,
        fixed=sources[0],
        moving=sources[1],
        affine=sources[2],
        warp=sources[3],
        output=output,
    )
    sources[source_index].write_bytes(b"scientifically-changed-input")
    resumed = FakeBackend()

    with pytest.raises(CacheValidationError, match="request"):
        run_transform(
            tmp_path,
            backend=resumed,
            fixed=sources[0],
            moving=sources[1],
            affine=sources[2],
            warp=sources[3],
            output=output,
        )

    assert resumed.transform_lists == []


@pytest.mark.parametrize(
    ("pixel_size", "interpolation"),
    [
        pytest.param(
            (0.4, 0.5, 1.2),
            "linear",
            id="voxel-spacing",
        ),
        pytest.param(
            (0.324, 0.324, 0.9),
            "nearestNeighbor",
            id="interpolation",
        ),
    ],
)
def test_transform_channel_rejects_changed_scientific_policy_without_force(
    tmp_path: Path,
    pixel_size: PixelSize,
    interpolation: str,
) -> None:
    sources = source_files(tmp_path)
    output = tmp_path / "registered-channel.tif"
    run_transform(
        tmp_path,
        fixed=sources[0],
        moving=sources[1],
        affine=sources[2],
        warp=sources[3],
        output=output,
    )
    resumed = FakeBackend()

    with pytest.raises(CacheValidationError, match="request"):
        run_transform(
            tmp_path,
            backend=resumed,
            fixed=sources[0],
            moving=sources[1],
            affine=sources[2],
            warp=sources[3],
            output=output,
            pixel_size=pixel_size,
            interpolation=interpolation,
        )

    assert resumed.transform_lists == []
    assert resumed.transform_interpolations == []


@pytest.mark.parametrize("interpolation", ["linear", "nearestNeighbor"])
def test_transform_channel_forwards_each_supported_interpolation_exactly(
    tmp_path: Path,
    interpolation: str,
) -> None:
    sources = source_files(tmp_path)
    backend = FakeBackend()

    run_transform(
        tmp_path,
        backend=backend,
        fixed=sources[0],
        moving=sources[1],
        affine=sources[2],
        warp=sources[3],
        output=tmp_path / f"channel-{interpolation}.tif",
        interpolation=interpolation,
    )

    assert backend.transform_interpolations == [interpolation]


def test_transform_channel_rejects_changed_postprocessing_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sources = source_files(tmp_path)
    output = tmp_path / "registered-channel.tif"
    run_transform(
        tmp_path,
        fixed=sources[0],
        moving=sources[1],
        affine=sources[2],
        warp=sources[3],
        output=output,
    )
    from cyexm.registration import _request_policy

    monkeypatch.setattr(
        _request_policy,
        "CHANNEL_POSTPROCESSING_POLICY",
        "clip-uint16-no-normalization-v2",
    )

    with pytest.raises(CacheValidationError, match="request"):
        run_transform(
            tmp_path,
            fixed=sources[0],
            moving=sources[1],
            affine=sources[2],
            warp=sources[3],
            output=output,
        )


@pytest.mark.parametrize("damage", ["data-without-record", "record-without-data"])
def test_transform_channel_rejects_incomplete_cache_entries(
    tmp_path: Path,
    damage: str,
) -> None:
    sources = source_files(tmp_path)
    output = tmp_path / "registered-channel.tif"
    store = ArtifactStore()
    run_transform(
        tmp_path,
        store=store,
        fixed=sources[0],
        moving=sources[1],
        affine=sources[2],
        warp=sources[3],
        output=output,
    )
    if damage == "data-without-record":
        store.record_path(output).unlink()
    else:
        output.unlink()

    with pytest.raises(CacheValidationError, match="Incomplete cache entry"):
        run_transform(
            tmp_path,
            store=store,
            fixed=sources[0],
            moving=sources[1],
            affine=sources[2],
            warp=sources[3],
            output=output,
        )


def test_transform_channel_rejects_identity_valid_but_unreadable_cached_tiff(
    tmp_path: Path,
) -> None:
    sources = source_files(tmp_path)
    output = tmp_path / "registered-channel.tif"
    store = ArtifactStore()
    run_transform(
        tmp_path,
        store=store,
        fixed=sources[0],
        moving=sources[1],
        affine=sources[2],
        warp=sources[3],
        output=output,
    )
    record_path = store.record_path(output)
    original = CompletionRecord.read(record_path)
    output.write_bytes(b"corrupt-tiff")
    repaired_identity = CompletionRecord(
        schema_version=1,
        request=original.request,
        artifact=FileIdentity.from_path(output),
    )
    record_path.write_text(repaired_identity.to_json(), encoding="utf-8")

    with pytest.raises(CacheValidationError, match="unreadable"):
        run_transform(
            tmp_path,
            backend=FakeBackend(),
            store=store,
            fixed=sources[0],
            moving=sources[1],
            affine=sources[2],
            warp=sources[3],
            output=output,
        )


def test_transform_channel_force_marks_old_record_stale_and_recomputes(
    tmp_path: Path,
) -> None:
    sources = source_files(tmp_path)
    output = tmp_path / "registered-channel.tif"

    class TrackingStore(ArtifactStore):
        def __init__(self) -> None:
            self.staled: list[Path] = []

        def mark_stale(self, artifact: Path) -> None:
            self.staled.append(artifact)
            super().mark_stale(artifact)

    store = TrackingStore()
    run_transform(
        tmp_path,
        store=store,
        fixed=sources[0],
        moving=sources[1],
        affine=sources[2],
        warp=sources[3],
        output=output,
    )
    replacement = FakeBackend()

    run_transform(
        tmp_path,
        backend=replacement,
        store=store,
        fixed=sources[0],
        moving=sources[1],
        affine=sources[2],
        warp=sources[3],
        output=output,
        force_override=True,
    )

    assert store.staled == [output.resolve()]
    assert replacement.transform_lists == [(sources[3].resolve(), sources[2].resolve())]
    assert store.record_path(output).is_file()
    assert not store.stale_path(output).exists()


def test_transform_channel_stales_committed_output_when_an_input_mutates(
    tmp_path: Path,
) -> None:
    fixed, moving, affine, warp = source_files(tmp_path)
    output = tmp_path / "registered-channel.tif"
    store = ArtifactStore()

    class MutatingBackend(FakeBackend):
        def write_clipped_tiff(self, image: object, path: Path) -> None:
            super().write_clipped_tiff(image, path)
            moving.write_bytes(b"moving-mutated-during-output-commit")

    with pytest.raises(RegistrationStageError, match="identity changed"):
        run_transform(
            tmp_path,
            backend=MutatingBackend(),
            store=store,
            fixed=fixed,
            moving=moving,
            affine=affine,
            warp=warp,
            output=output,
        )

    assert output.is_file()
    assert not store.record_path(output).exists()
    assert store.stale_path(output).is_file()
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_transform_channel_writer_failure_leaves_no_partial_artifact(
    tmp_path: Path,
) -> None:
    sources = source_files(tmp_path)
    output = tmp_path / "registered-channel.tif"
    backend = FakeBackend(fail_tiff_role="registered-channel")

    with pytest.raises(RegistrationStageError, match="commit failed") as error:
        run_transform(
            tmp_path,
            backend=backend,
            fixed=sources[0],
            moving=sources[1],
            affine=sources[2],
            warp=sources[3],
            output=output,
        )

    assert isinstance(error.value.__cause__, RuntimeError)
    assert not output.exists()
    assert not ArtifactStore().record_path(output).exists()
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_transform_channel_completion_record_failure_removes_uncommitted_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sources = source_files(tmp_path)
    output = tmp_path / "registered-channel.tif"
    original_atomic_write = atomic_module.atomic_write
    calls = 0

    def fail_record_commit(
        destination: Path,
        writer: object,
        validator: object,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            original_atomic_write(
                destination,
                writer,  # type: ignore[arg-type]
                lambda path: False,
            )
            return
        original_atomic_write(
            destination,
            writer,  # type: ignore[arg-type]
            validator,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(artifacts_module, "atomic_write", fail_record_commit)

    with pytest.raises(RegistrationStageError, match="commit failed") as error:
        run_transform(
            tmp_path,
            fixed=sources[0],
            moving=sources[1],
            affine=sources[2],
            warp=sources[3],
            output=output,
        )

    assert isinstance(error.value.__cause__, CacheValidationError)
    assert calls == 2
    assert not output.exists()
    assert not ArtifactStore().record_path(output).exists()
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_transform_channel_resamples_unequal_moving_shape_with_fixed_reference(
    tmp_path: Path,
) -> None:
    sources = source_files(tmp_path)
    fixed_image = fake_image(data=np.ones((8, 8, 8), dtype=np.float32))
    moving_image = fake_image(data=np.ones((7, 9, 6), dtype=np.float32))

    class FixedReferenceBackend(FakeBackend):
        transform_arguments: list[tuple[object, object]]

        def __init__(self) -> None:
            super().__init__(fixed_image=fixed_image, moving_image=moving_image)
            self.transform_arguments = []

        def apply_transform_chain(self, **kwargs: object) -> object:
            self.transform_arguments.append((kwargs["fixed"], kwargs["moving"]))
            super().apply_transform_chain(**kwargs)  # type: ignore[arg-type]
            fixed = kwargs["fixed"]
            assert hasattr(fixed, "clone")
            return fixed.clone()  # type: ignore[union-attr]

    backend = FixedReferenceBackend()

    output = run_transform(
        tmp_path,
        backend=backend,
        fixed=sources[0],
        moving=sources[1],
        affine=sources[2],
        warp=sources[3],
    )

    assert output.is_file()
    assert len(backend.transform_arguments) == 1
    fixed_argument, moving_argument = backend.transform_arguments[0]
    assert backend.geometry(fixed_argument).shape == (8, 8, 8)
    assert backend.geometry(moving_argument).shape == (7, 9, 6)
    assert backend.geometry(backend.clipped_tiff_images[0]) == backend.geometry(
        fixed_argument
    )


def test_transform_channel_rejects_result_that_loses_fixed_geometry(
    tmp_path: Path,
) -> None:
    sources = source_files(tmp_path)

    class WrongGeometryBackend(FakeBackend):
        def apply_transform_chain(self, **kwargs: object) -> object:
            transformed = super().apply_transform_chain(**kwargs)  # type: ignore[arg-type]
            transformed.origin = (10.0, 20.0, 30.0)  # type: ignore[attr-defined]
            return transformed

    backend = WrongGeometryBackend()

    with pytest.raises(RegistrationStageError, match="result geometry"):
        run_transform(
            tmp_path,
            backend=backend,
            fixed=sources[0],
            moving=sources[1],
            affine=sources[2],
            warp=sources[3],
        )

    assert backend.clipped_tiff_writes == []


@pytest.mark.parametrize(
    ("keyword", "value", "error_type"),
    [
        pytest.param("pixel_size", (1.0, 2.0), ValueError, id="short-spacing"),
        pytest.param("pixel_size", (1.0, True, 3.0), TypeError, id="boolean-spacing"),
        pytest.param("pixel_size", (1.0, 2.0, 0.0), ValueError, id="zero-spacing"),
        pytest.param(
            "pixel_size", (1.0, 2.0, float("nan")), ValueError, id="nan-spacing"
        ),
        pytest.param("interpolation", 1, TypeError, id="non-string-interpolation"),
        pytest.param(
            "interpolation", "bspline", ValueError, id="unknown-interpolation"
        ),
        pytest.param("force_override", 1, TypeError, id="non-boolean-force"),
    ],
)
def test_transform_channel_validates_scalar_policy_before_configuring_threads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    keyword: str,
    value: object,
    error_type: type[Exception],
) -> None:
    configured = False

    def unexpected_configuration(config: NativeThreadConfig | None) -> object:
        nonlocal configured
        configured = True
        raise AssertionError("thread configuration must follow input validation")

    monkeypatch.setattr(
        channels_module,
        "configure_native_threads",
        unexpected_configuration,
    )
    sources = source_files(tmp_path)
    arguments: dict[str, object] = {
        "fixed_image_path": sources[0],
        "moving_image_path": sources[1],
        "affine_path": sources[2],
        "warp_path": sources[3],
        "output_path": tmp_path / "channel.tif",
        "pixel_size": (1.0, 2.0, 3.0),
        "interpolation": "linear",
        "native_threads": NativeThreadConfig(1, 1),
        "force_override": False,
        "_backend": FakeBackend(),
    }
    arguments[keyword] = value

    with pytest.raises(error_type):
        transform_channel(**arguments)  # type: ignore[arg-type]

    assert configured is False


def test_transform_channel_validates_all_files_before_configuring_threads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured = False

    def unexpected_configuration(config: NativeThreadConfig | None) -> object:
        nonlocal configured
        configured = True
        raise AssertionError("thread configuration must follow file validation")

    monkeypatch.setattr(
        channels_module,
        "configure_native_threads",
        unexpected_configuration,
    )
    fixed, moving, affine, _ = source_files(tmp_path)

    with pytest.raises(FileNotFoundError):
        transform_channel(
            fixed_image_path=fixed,
            moving_image_path=moving,
            affine_path=affine,
            warp_path=tmp_path / "missing-warp.nii.gz",
            output_path=tmp_path / "channel.tif",
            pixel_size=(1.0, 1.0, 1.0),
            interpolation="linear",
            native_threads=NativeThreadConfig(1, 1),
            _backend=FakeBackend(),
        )

    assert configured is False


def test_transform_channel_configures_threads_before_production_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    sources = source_files(tmp_path)
    backend = FakeBackend()

    class Resolved:
        def __init__(self) -> None:
            self.values = {name: 1 for name in NATIVE_THREAD_ENV_VARS}

        def describe(self) -> str:
            return "test native thread policy"

    def configure(config: NativeThreadConfig | None) -> Resolved:
        assert config == NativeThreadConfig(1, 1)
        events.append("configure")
        return Resolved()

    def production(config: NativeThreadConfig | None) -> FakeBackend:
        assert config == NativeThreadConfig(1, 1)
        events.append("backend")
        return backend

    monkeypatch.setattr(channels_module, "configure_native_threads", configure)
    monkeypatch.setattr(channels_module, "_production_backend", production)

    transform_channel(
        fixed_image_path=sources[0],
        moving_image_path=sources[1],
        affine_path=sources[2],
        warp_path=sources[3],
        output_path=tmp_path / "channel.tif",
        pixel_size=(1.0, 1.0, 1.0),
        interpolation="linear",
        native_threads=NativeThreadConfig(1, 1),
    )

    assert events == ["configure", "backend"]


def test_transform_channel_logs_the_resolved_native_thread_policy(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sources = source_files(tmp_path)

    with caplog.at_level(logging.INFO, logger="cyexm.registration.channels"):
        run_transform(
            tmp_path,
            fixed=sources[0],
            moving=sources[1],
            affine=sources[2],
            warp=sources[3],
        )

    for name in NATIVE_THREAD_ENV_VARS:
        assert f"{name}=1 (explicit)" in caplog.text


def test_transform_channel_rejects_a_non_file_input_before_configuring_threads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured = False
    fixed, moving, affine, warp = source_files(tmp_path)
    fixed.unlink()
    fixed.mkdir()

    def unexpected_configuration(config: NativeThreadConfig | None) -> object:
        nonlocal configured
        configured = True
        raise AssertionError("thread configuration must follow file validation")

    monkeypatch.setattr(
        channels_module,
        "configure_native_threads",
        unexpected_configuration,
    )

    with pytest.raises(ValueError, match="regular file"):
        transform_channel(
            fixed_image_path=fixed,
            moving_image_path=moving,
            affine_path=affine,
            warp_path=warp,
            output_path=tmp_path / "channel.tif",
            pixel_size=(1.0, 1.0, 1.0),
            interpolation="linear",
            native_threads=NativeThreadConfig(1, 1),
            _backend=FakeBackend(),
        )

    assert configured is False


def test_transform_channel_rejects_an_output_that_aliases_an_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured = False
    fixed, moving, affine, warp = source_files(tmp_path)

    def unexpected_configuration(config: NativeThreadConfig | None) -> object:
        nonlocal configured
        configured = True
        raise AssertionError("thread configuration must follow path validation")

    monkeypatch.setattr(
        channels_module,
        "configure_native_threads",
        unexpected_configuration,
    )

    with pytest.raises(ValueError, match="must not alias"):
        transform_channel(
            fixed_image_path=fixed,
            moving_image_path=moving,
            affine_path=affine,
            warp_path=warp,
            output_path=moving,
            pixel_size=(1.0, 1.0, 1.0),
            interpolation="linear",
            native_threads=NativeThreadConfig(1, 1),
            _backend=FakeBackend(),
        )

    assert configured is False


def test_force_rejects_input_symlink_to_completion_record_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixed, moving, _, warp = source_files(tmp_path)
    output = tmp_path / "channel.tif"
    store = ArtifactStore()
    record = store.record_path(output)
    record.write_bytes(b"affine-input-owned-by-caller")
    affine = tmp_path / "affine-link.mat"
    affine.symlink_to(record)
    input_paths = (fixed, moving, affine, warp)
    identities_before = tuple(FileIdentity.from_path(path) for path in input_paths)
    bytes_before = tuple(path.read_bytes() for path in input_paths)
    events: list[str] = []

    def configure(config: NativeThreadConfig | None) -> object:
        del config
        events.append("configure")
        raise AssertionError("thread configuration must not run")

    def production(config: NativeThreadConfig | None) -> FakeBackend:
        del config
        events.append("backend")
        raise AssertionError("production backend must not be constructed")

    monkeypatch.setattr(channels_module, "configure_native_threads", configure)
    monkeypatch.setattr(channels_module, "_production_backend", production)

    with pytest.raises(ValueError, match="output management path"):
        transform_channel(
            fixed_image_path=fixed,
            moving_image_path=moving,
            affine_path=affine,
            warp_path=warp,
            output_path=output,
            pixel_size=(1.0, 1.0, 1.0),
            interpolation="linear",
            native_threads=NativeThreadConfig(1, 1),
            force_override=True,
            _artifact_store=store,
        )

    assert events == []
    assert affine.is_symlink()
    assert (
        tuple(FileIdentity.from_path(path) for path in input_paths) == identities_before
    )
    assert tuple(path.read_bytes() for path in input_paths) == bytes_before
    assert record.is_file()
    assert not store.stale_path(output).exists()


def test_resume_rejects_input_symlink_to_stale_path_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixed, moving, affine, _ = source_files(tmp_path)
    output = tmp_path / "channel.tif"
    store = ArtifactStore()
    stale = store.stale_path(output)
    stale.write_bytes(b"warp-input-owned-by-caller")
    warp = tmp_path / "warp-link.nii.gz"
    warp.symlink_to(stale)
    input_paths = (fixed, moving, affine, warp)
    identities_before = tuple(FileIdentity.from_path(path) for path in input_paths)
    bytes_before = tuple(path.read_bytes() for path in input_paths)
    events: list[str] = []

    def configure(config: NativeThreadConfig | None) -> object:
        del config
        events.append("configure")
        raise AssertionError("thread configuration must not run")

    def production(config: NativeThreadConfig | None) -> FakeBackend:
        del config
        events.append("backend")
        raise AssertionError("production backend must not be constructed")

    monkeypatch.setattr(channels_module, "configure_native_threads", configure)
    monkeypatch.setattr(channels_module, "_production_backend", production)

    with pytest.raises(ValueError, match="output management path"):
        transform_channel(
            fixed_image_path=fixed,
            moving_image_path=moving,
            affine_path=affine,
            warp_path=warp,
            output_path=output,
            pixel_size=(1.0, 1.0, 1.0),
            interpolation="linear",
            native_threads=NativeThreadConfig(1, 1),
            _artifact_store=store,
        )

    assert events == []
    assert warp.is_symlink()
    assert (
        tuple(FileIdentity.from_path(path) for path in input_paths) == identities_before
    )
    assert tuple(path.read_bytes() for path in input_paths) == bytes_before
    assert stale.is_file()
    assert not store.record_path(output).exists()


def test_transform_channel_has_only_the_two_documented_test_injections() -> None:
    signature = inspect.signature(transform_channel)

    assert tuple(signature.parameters) == (
        "fixed_image_path",
        "moving_image_path",
        "affine_path",
        "warp_path",
        "output_path",
        "pixel_size",
        "interpolation",
        "native_threads",
        "force_override",
        "_backend",
        "_artifact_store",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )


def test_channels_module_exports_only_transform_channel() -> None:
    assert channels_module.__all__ == ["transform_channel"]


def test_registration_namespace_exports_the_accepted_channel_operation() -> None:
    assert registration_namespace.transform_channel is transform_channel
    assert registration_namespace.__all__ == [
        "ChunkedImageRegistration",
        "EstimatedXYRotationPolicy",
        "EstimatedXYSimilarityPolicy",
        "ExplicitXYRotationPolicy",
        "ImageRegistration",
        "InitialAlignmentPolicy",
        "register_round",
        "transform_channel",
    ]
