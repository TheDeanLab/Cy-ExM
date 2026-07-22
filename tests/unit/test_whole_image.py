from __future__ import annotations

import inspect
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fakes import FakeBackend, FakeImage, fake_image

import cyexm.registration.whole_image as whole_image
from cyexm.errors import CacheValidationError, RegistrationStageError
from cyexm.registration._request_policy import nonlinear_parameters
from cyexm.registration.artifacts import (
    ArtifactStore,
    CompletionRecord,
    FileIdentity,
    StageRequest,
)
from cyexm.registration.whole_image import (
    ImageRegistration,
    RegistrationPaths,
    register_round,
)
from cyexm.runtime import (
    NATIVE_THREAD_ENV_VARS,
    NativeThreadConfig,
    ResolvedNativeThreads,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def resolved_native_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> ResolvedNativeThreads:
    values = {name: index + 1 for index, name in enumerate(NATIVE_THREAD_ENV_VARS)}
    sources = {
        name: ("explicit" if index < 2 else "environment")
        for index, name in enumerate(NATIVE_THREAD_ENV_VARS)
    }
    resolved = ResolvedNativeThreads(values=values, sources=sources)
    monkeypatch.setattr(
        whole_image,
        "configure_native_threads",
        lambda config: resolved,
    )
    return resolved


def write_stub(path: Path, contents: bytes = b"stub-image") -> Path:
    path.write_bytes(contents)
    return path


def make_registration(
    tmp_path: Path,
    *,
    backend: FakeBackend,
    artifact_store: ArtifactStore | None = None,
    **kwargs: Any,
) -> ImageRegistration:
    fixed = tmp_path / "fixed.tif"
    moving = tmp_path / "moving.tif"
    if not fixed.exists():
        write_stub(fixed, b"fixed-input")
    if not moving.exists():
        write_stub(moving, b"moving-input")
    return ImageRegistration(
        fixed,
        moving,
        tmp_path / "output",
        2,
        _backend=backend,
        _artifact_store=artifact_store or ArtifactStore(),
        **kwargs,
    )


def round_paths(tmp_path: Path) -> RegistrationPaths:
    return RegistrationPaths.for_round(tmp_path / "output", 2)


def completion_request(path: Path) -> object:
    store = ArtifactStore()
    return CompletionRecord.read(store.record_path(path)).request


def test_nonlinear_request_policy_resolves_metric_sampling() -> None:
    assert nonlinear_parameters("high", "CC", None)["syn_sampling"] == 4
    assert nonlinear_parameters("high", "CC", 2)["syn_sampling"] == 2
    assert nonlinear_parameters("high", "mattes", None)["syn_sampling"] == 32


def test_whole_image_cc_policy_is_normalized_forwarded_and_recorded(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    registration = make_registration(
        tmp_path,
        backend=backend,
        nonlinear_metric="CC",
    )

    assert registration.nonlinear_metric == "CC"
    assert registration.nonlinear_radius == 4

    registration.register()

    assert backend.nonlinear_kwargs[0]["metric"] == "CC"
    assert backend.nonlinear_kwargs[0]["radius"] == 4
    request = completion_request(round_paths(tmp_path).warp)
    assert request.parameters["registration_parameters"]["syn_metric"] == "CC"
    assert request.parameters["registration_parameters"]["syn_sampling"] == 4


class RejectingValidatorStore(ArtifactStore):
    def __init__(self, target_name: str) -> None:
        self.target_name = target_name

    def commit(
        self,
        artifact: Path,
        request: StageRequest,
        writer: Callable[[Path], None],
        validator: Callable[[Path], bool],
    ) -> None:
        if artifact.name == self.target_name:

            def reject(candidate: Path) -> bool:
                del candidate
                return False

            validator = reject
        super().commit(artifact, request, writer, validator)


class SourceMutatingStore(ArtifactStore):
    def __init__(
        self,
        source: Path,
        target_name: str,
        operation: str,
        *,
        raise_after_mutation: bool = False,
        raise_before_commit: bool = False,
    ) -> None:
        self.source = source
        self.target_name = target_name
        self.operation = operation
        self.raise_after_mutation = raise_after_mutation
        self.raise_before_commit = raise_before_commit
        self.failure = RuntimeError("injected artifact-store failure")
        self.changed = False

    def _mutate_source(self, artifact: Path, operation: str) -> bool:
        if (
            not self.changed
            and artifact.name == self.target_name
            and operation == self.operation
        ):
            self.source.write_bytes(b"source-mutated-inside-artifact-store")
            self.changed = True
            return True
        return False

    def reuse(
        self,
        artifact: Path,
        request: StageRequest,
        reader: Callable[[Path], Any],
    ) -> Any | None:
        value = super().reuse(artifact, request, reader)
        mutated = self._mutate_source(artifact, "reuse")
        if mutated and self.raise_after_mutation:
            raise self.failure
        return value

    def commit(
        self,
        artifact: Path,
        request: StageRequest,
        writer: Callable[[Path], None],
        validator: Callable[[Path], bool],
    ) -> None:
        if (
            self.raise_before_commit
            and artifact.name == self.target_name
            and self.operation == "commit"
        ):
            self._mutate_source(artifact, "commit")
            raise self.failure
        super().commit(artifact, request, writer, validator)
        mutated = self._mutate_source(artifact, "commit")
        if mutated and self.raise_after_mutation:
            raise self.failure


class StableRaisingStore(ArtifactStore):
    def __init__(self, target_name: str, operation: str, failure: RuntimeError) -> None:
        self.target_name = target_name
        self.operation = operation
        self.failure = failure

    def reuse(
        self,
        artifact: Path,
        request: StageRequest,
        reader: Callable[[Path], Any],
    ) -> Any | None:
        if artifact.name == self.target_name and self.operation == "reuse":
            raise self.failure
        return super().reuse(artifact, request, reader)

    def commit(
        self,
        artifact: Path,
        request: StageRequest,
        writer: Callable[[Path], None],
        validator: Callable[[Path], bool],
    ) -> None:
        if artifact.name == self.target_name and self.operation == "commit":
            raise self.failure
        super().commit(artifact, request, writer, validator)


def restore_file_identity(path: Path, contents: bytes, identity: FileIdentity) -> None:
    path.write_bytes(contents)
    os.utime(path, ns=(path.stat().st_atime_ns, identity.st_mtime_ns))
    assert FileIdentity.from_path(path) == identity


class GeometryMismatchingBackend(FakeBackend):
    def __init__(self, target_name: str, component: str = "origin") -> None:
        super().__init__()
        self.target_name = target_name
        self.component = component

    def load_image(
        self,
        path: Path,
        pixel_size_xyz: tuple[float, float, float],
    ) -> object:
        loaded = super().load_image(path, pixel_size_xyz)
        if self.target_name not in path.name:
            return loaded
        assert isinstance(loaded, FakeImage)
        if self.component == "shape":
            loaded.data = np.pad(loaded.data, ((0, 1), (0, 0), (0, 0)))
        elif self.component == "origin":
            loaded.origin = (loaded.origin[0] + 1.0, *loaded.origin[1:])
        elif self.component == "spacing":
            loaded.spacing = (loaded.spacing[0] + 0.01, *loaded.spacing[1:])
        elif self.component == "direction":
            loaded.direction = (
                (-loaded.direction[0][0], *loaded.direction[0][1:]),
                loaded.direction[1],
                loaded.direction[2],
            )
        else:
            raise AssertionError(f"unknown geometry component: {self.component}")
        return loaded


def test_registration_defaults_are_legacy_compatible(tmp_path: Path) -> None:
    fixed = write_stub(tmp_path / "fixed.tif")
    moving = write_stub(tmp_path / "moving.tif")

    registration = ImageRegistration(fixed, moving, tmp_path / "result")

    assert registration.pixel_size == (0.15, 0.15, 0.15)
    assert registration.imaging_round == 0
    assert registration.linear_type == "TRSAA"
    assert registration.linear_accuracy == "low"
    assert registration.nonlinear_type == "SyNOnly"
    assert registration.nonlinear_accuracy == "high"
    assert registration.crop is False
    assert registration.force_override is False


def test_registration_paths_are_round_scoped(tmp_path: Path) -> None:
    paths = RegistrationPaths.for_round(tmp_path, 3)

    assert paths.output == tmp_path
    assert paths.affine.name == "linear_transform_3.mat"
    assert paths.warp.name == "nonlinear_warp_3.nii.gz"
    assert paths.registered.name == "nonlinear_registered_3.tif"
    assert paths.fixed_reference.name == "fixed_reference.tif"
    assert paths.pre_metrics.name == "image_metrics_pre-registration.txt"
    assert paths.linear_metrics.name == "image_metrics_linearly-registered.txt"
    assert paths.nonlinear_metrics.name == "image_metrics_nonlinear-registered.txt"
    assert paths.fixed_crop.name == "fixed_crop_indices_3.json"
    assert paths.moving_crop.name == "moving_crop_indices_3.json"


def test_constructor_is_lazy_about_filesystem_threads_and_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "not-created"
    backend = object()

    def unexpected_configuration(config: NativeThreadConfig | None) -> object:
        del config
        raise AssertionError("native threads were configured during construction")

    monkeypatch.setattr(
        "cyexm.registration.whole_image.configure_native_threads",
        unexpected_configuration,
    )

    registration = ImageRegistration(
        tmp_path / "missing-fixed.tif",
        tmp_path / "missing-moving.tif",
        output,
        _backend=backend,
    )

    assert registration._backend is backend
    assert not output.exists()


def test_new_configuration_parameters_are_keyword_only() -> None:
    signature = inspect.signature(ImageRegistration)

    for name in (
        "native_threads",
        "linear_type",
        "linear_accuracy",
        "nonlinear_type",
        "nonlinear_accuracy",
        "_backend",
        "_artifact_store",
    ):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_register_round_exposes_the_legacy_positional_prefix() -> None:
    signature = inspect.signature(register_round)

    assert tuple(signature.parameters)[:6] == (
        "fixed_image_path",
        "moving_image_path",
        "save_directory",
        "imaging_round",
        "crop",
        "enable_logging",
    )


def test_constructor_stores_custom_scalar_configuration(tmp_path: Path) -> None:
    native_threads = NativeThreadConfig(itk_threads=2, blas_threads=3)

    registration = ImageRegistration(
        tmp_path / "fixed.tif",
        tmp_path / "moving.tif",
        tmp_path / "output",
        7,
        (0.2, 0.3, 0.4),
        True,
        False,
        True,
        native_threads=native_threads,
        linear_type="Rigid",
        linear_accuracy="high",
        nonlinear_type="SyN",
        nonlinear_accuracy="low",
        nonlinear_metric="CC",
        nonlinear_radius=2,
    )

    assert registration.imaging_round == 7
    assert registration.pixel_size == (0.2, 0.3, 0.4)
    assert registration.crop is True
    assert registration.enable_logging is False
    assert registration.force_override is True
    assert registration.native_threads is native_threads
    assert registration.linear_type == "Rigid"
    assert registration.linear_accuracy == "high"
    assert registration.nonlinear_type == "SyN"
    assert registration.nonlinear_accuracy == "low"
    assert registration.prepared_fixed is False


def test_prepared_fixed_mode_crops_only_moving_and_omits_fixed_duplicates(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    registration = make_registration(
        tmp_path,
        backend=backend,
        crop=True,
        prepared_fixed=True,
    )

    registration.register()

    paths = round_paths(tmp_path)
    assert backend.crop_bounds == [(2, 6, 1, 5, 2, 6)]
    assert not paths.fixed_reference.exists()
    assert not ArtifactStore().record_path(paths.fixed_reference).exists()
    assert not paths.fixed_crop.exists()
    assert not ArtifactStore().record_path(paths.fixed_crop).exists()
    assert paths.moving_crop.is_file()


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        pytest.param("imaging_round", True, id="boolean-round"),
        pytest.param("imaging_round", -1, id="negative-round"),
        pytest.param("pixel_size", (0.1, 0.2), id="short-spacing"),
        pytest.param("pixel_size", (0.1, 0.2, 0.0), id="zero-spacing"),
        pytest.param("pixel_size", (0.1, 0.2, float("nan")), id="nan-spacing"),
        pytest.param("crop", 1, id="non-boolean-crop"),
        pytest.param("enable_logging", 1, id="non-boolean-logging"),
        pytest.param("force_override", 1, id="non-boolean-force"),
        pytest.param("linear_type", "SyNOnly", id="linear-method"),
        pytest.param("linear_accuracy", "unknown", id="linear-accuracy"),
        pytest.param("nonlinear_type", "Rigid", id="nonlinear-method"),
        pytest.param("nonlinear_accuracy", "unknown", id="nonlinear-accuracy"),
    ],
)
def test_constructor_rejects_invalid_scalar_configuration(
    tmp_path: Path,
    keyword: str,
    value: object,
) -> None:
    kwargs = {keyword: value}

    with pytest.raises((TypeError, ValueError)):
        ImageRegistration(
            tmp_path / "fixed.tif",
            tmp_path / "moving.tif",
            tmp_path / "output",
            **kwargs,
        )


def test_constructor_uses_named_logger_without_adding_root_handlers(
    tmp_path: Path,
) -> None:
    root = logging.getLogger()
    handlers_before = tuple(root.handlers)

    registration = ImageRegistration(
        tmp_path / "fixed.tif",
        tmp_path / "moving.tif",
        tmp_path / "output",
    )

    assert registration._log.name == "cyexm.registration.whole_image"
    assert tuple(root.handlers) == handlers_before


def test_registration_package_exports_the_stable_whole_image_api() -> None:
    import cyexm.registration as registration_package

    assert registration_package.ImageRegistration is ImageRegistration
    assert registration_package.register_round is register_round


def test_register_round_forwards_every_option_and_registers_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cyexm.registration.whole_image as whole_image

    received: dict[str, Any] = {}
    register_calls = 0

    class RecordingRegistration:
        def __init__(self, *args: object, **kwargs: object) -> None:
            received["args"] = args
            received["kwargs"] = kwargs

        def register(self) -> None:
            nonlocal register_calls
            register_calls += 1

    monkeypatch.setattr(whole_image, "ImageRegistration", RecordingRegistration)
    native_threads = NativeThreadConfig(itk_threads=2, blas_threads=3)
    fixed = tmp_path / "fixed.tif"
    moving = tmp_path / "moving.tif"
    output = tmp_path / "output"

    whole_image.register_round(
        fixed,
        moving,
        output,
        4,
        True,
        False,
        pixel_size=(0.2, 0.3, 0.4),
        force_override=True,
        native_threads=native_threads,
        linear_type="Rigid",
        linear_accuracy="medium",
        nonlinear_type="SyN",
        nonlinear_accuracy="low",
        nonlinear_metric="CC",
        nonlinear_radius=2,
    )

    assert received["args"] == (
        fixed,
        moving,
        output,
        4,
        (0.2, 0.3, 0.4),
        True,
        False,
        True,
    )
    assert received["kwargs"] == {
        "native_threads": native_threads,
        "linear_type": "Rigid",
        "linear_accuracy": "medium",
        "nonlinear_type": "SyN",
        "nonlinear_accuracy": "low",
        "nonlinear_metric": "CC",
        "nonlinear_radius": 2,
    }
    assert register_calls == 1


def test_register_resolves_threads_before_loading_any_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resolved_native_threads: ResolvedNativeThreads,
) -> None:
    events: list[str] = []
    requested = NativeThreadConfig(itk_threads=2, blas_threads=3)

    def configure(config: NativeThreadConfig | None) -> ResolvedNativeThreads:
        assert config is requested
        events.append("configure")
        return resolved_native_threads

    original_from_path = FileIdentity.from_path

    def record_identity(cls: type[FileIdentity], path: Path) -> FileIdentity:
        del cls
        events.append(f"identity:{path.name}")
        return original_from_path(path)

    class TrackingBackend(FakeBackend):
        def load_image(self, path: Path, pixel_size_xyz: tuple[float, float, float]):
            events.append(f"load:{path.name}")
            return super().load_image(path, pixel_size_xyz)

    monkeypatch.setattr(whole_image, "configure_native_threads", configure)
    monkeypatch.setattr(FileIdentity, "from_path", classmethod(record_identity))
    backend = TrackingBackend()

    make_registration(tmp_path, backend=backend, native_threads=requested).register()

    assert events[:7] == [
        "configure",
        "identity:fixed.tif",
        "identity:moving.tif",
        "load:fixed.tif",
        "load:moving.tif",
        "identity:fixed.tif",
        "identity:moving.tif",
    ]


def test_orchestration_cache_calls_are_confined_to_guard_helpers() -> None:
    source = inspect.getsource(ImageRegistration)

    assert source.count("self._store().reuse(") == 1
    assert source.count("store.commit(") == 1
    assert "self._store().commit(" not in source


def test_successful_registration_commits_every_artifact_in_stage_order(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    store = ArtifactStore()

    make_registration(tmp_path, backend=backend, artifact_store=store).register()

    paths = round_paths(tmp_path)
    artifacts = (
        paths.fixed_reference,
        paths.pre_metrics,
        paths.affine,
        paths.linear_metrics,
        paths.warp,
        paths.nonlinear_metrics,
        paths.registered,
    )
    assert all(path.is_file() for path in artifacts)
    assert all(store.record_path(path).is_file() for path in artifacts)
    assert list(paths.output.glob(".*.tmp*")) == []
    assert backend.linear_calls == 1
    assert backend.nonlinear_calls == 1
    assert len(backend.metric_calls) == 3

    first_metrics = backend.events.index("metrics")
    linear = backend.events.index("linear")
    affine_write = backend.events.index("write-affine")
    second_metrics = backend.events.index("metrics", first_metrics + 1)
    nonlinear = backend.events.index("nonlinear")
    third_metrics = backend.events.index("metrics", second_metrics + 1)
    assert first_metrics < linear < affine_write < second_metrics < nonlinear
    assert nonlinear < third_metrics


def test_stage_records_capture_inputs_geometry_threads_policy_and_upstream(
    tmp_path: Path,
    resolved_native_threads: ResolvedNativeThreads,
) -> None:
    backend = FakeBackend()
    registration = make_registration(tmp_path, backend=backend)
    expected_inputs = (
        FileIdentity.from_path(tmp_path / "fixed.tif"),
        FileIdentity.from_path(tmp_path / "moving.tif"),
    )
    registration.register()
    paths = round_paths(tmp_path)

    records = {
        "fixed": completion_request(paths.fixed_reference),
        "pre": completion_request(paths.pre_metrics),
        "affine": completion_request(paths.affine),
        "linear": completion_request(paths.linear_metrics),
        "warp": completion_request(paths.warp),
        "nonlinear": completion_request(paths.nonlinear_metrics),
        "final": completion_request(paths.registered),
    }

    for request in records.values():
        assert request.inputs == expected_inputs
        assert request.geometry.spacing == (0.15, 0.15, 0.15)
        assert request.native_threads == resolved_native_threads.values
        assert (
            request.parameters["axis_convention_version"] == "external-zyx-ants-xyz-v1"
        )
        assert "native_thread_sources" not in request.parameters

    assert records["affine"].method == "TRSAA"
    assert records["affine"].accuracy == "low"
    assert records["affine"].parameters["registration_parameters"] == {
        "aff_metric": "mattes",
        "aff_random_sampling_rate": 0.5,
        "aff_sampling": 32,
        "mask_all_stages": True,
        "reg_iterations": [100, 50, 25, 10],
        "verbose": True,
    }
    assert records["warp"].method == "SyNOnly"
    assert records["warp"].accuracy == "high"
    assert records["warp"].parameters["registration_parameters"] == {
        "aff_random_sampling_rate": 1.0,
        "flow_sigma": 3.0,
        "initial_transform": "Identity",
        "mask_all_stages": True,
        "reg_iterations": [100, 70, 50],
        "singleprecision": False,
        "syn_metric": "mattes",
        "syn_sampling": 32,
        "total_sigma": 0.5,
        "verbose": True,
    }
    assert records["pre"].upstream is None
    assert records["affine"].upstream is None
    assert records["linear"].upstream == FileIdentity.from_path(paths.affine)
    assert records["warp"].upstream == FileIdentity.from_path(paths.affine)
    assert records["nonlinear"].upstream == FileIdentity.from_path(paths.warp)
    assert records["final"].upstream == FileIdentity.from_path(paths.warp)


def test_affine_survives_nonlinear_failure_and_is_reused(tmp_path: Path) -> None:
    backend = FakeBackend(fail_nonlinear=True)

    with pytest.raises(RegistrationStageError, match="nonlinear") as error:
        make_registration(tmp_path, backend=backend).register()

    assert isinstance(error.value.__cause__, RuntimeError)
    paths = round_paths(tmp_path)
    store = ArtifactStore()
    assert paths.affine.is_file()
    assert store.record_path(paths.affine).is_file()
    assert not paths.warp.exists()
    assert not store.record_path(paths.warp).exists()
    assert not paths.nonlinear_metrics.exists()
    assert not store.record_path(paths.nonlinear_metrics).exists()
    assert not paths.registered.exists()
    assert not store.record_path(paths.registered).exists()

    resumed = FakeBackend()
    make_registration(tmp_path, backend=resumed).register()

    assert resumed.linear_calls == 0
    assert resumed.read_affine_calls == 1
    assert resumed.nonlinear_calls == 1


def test_complete_registration_is_reused_without_reestimating_or_rewriting(
    tmp_path: Path,
) -> None:
    make_registration(tmp_path, backend=FakeBackend()).register()
    resumed = FakeBackend()

    make_registration(tmp_path, backend=resumed).register()

    paths = round_paths(tmp_path)
    assert resumed.linear_calls == 0
    assert resumed.nonlinear_calls == 0
    assert resumed.metric_calls == []
    assert resumed.tiff_writes == []
    assert resumed.affine_interpolations == ["linear", "nearestNeighbor"]
    assert resumed.warp_applications == [
        (paths.warp, "linear"),
        (paths.warp, "nearestNeighbor"),
    ]


@pytest.mark.parametrize(
    ("target_name", "stage"),
    [
        pytest.param("fixed_reference", "fixed reference", id="fixed-reference"),
        pytest.param(
            "nonlinear_registered_2",
            "registered image",
            id="registered-image",
        ),
    ],
)
def test_fresh_whole_tiff_validator_rejects_reopened_geometry_mismatch(
    tmp_path: Path,
    target_name: str,
    stage: str,
) -> None:
    backend = GeometryMismatchingBackend(target_name)

    with pytest.raises(RegistrationStageError, match=stage):
        make_registration(tmp_path, backend=backend).register()


@pytest.mark.parametrize(
    "target_name",
    [
        pytest.param("fixed_reference", id="fixed-reference"),
        pytest.param("nonlinear_registered_2", id="registered-image"),
    ],
)
@pytest.mark.parametrize(
    "component",
    [
        pytest.param("shape", id="shape"),
        pytest.param("origin", id="origin"),
        pytest.param("spacing", id="spacing"),
        pytest.param("direction", id="direction"),
    ],
)
def test_cached_whole_tiff_reader_rejects_every_geometry_mismatch(
    tmp_path: Path,
    target_name: str,
    component: str,
) -> None:
    make_registration(tmp_path, backend=FakeBackend()).register()
    resumed = GeometryMismatchingBackend(target_name, component)

    with pytest.raises(CacheValidationError, match="unreadable"):
        make_registration(tmp_path, backend=resumed).register()


def test_equal_effective_threads_reuse_cache_across_resolution_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {name: index + 1 for index, name in enumerate(NATIVE_THREAD_ENV_VARS)}
    resolutions = iter(
        (
            ResolvedNativeThreads(
                values=values,
                sources={name: "default" for name in NATIVE_THREAD_ENV_VARS},
            ),
            ResolvedNativeThreads(
                values=values,
                sources={name: "environment" for name in NATIVE_THREAD_ENV_VARS},
            ),
        )
    )
    monkeypatch.setattr(
        whole_image,
        "configure_native_threads",
        lambda config: next(resolutions),
    )
    backend = FakeBackend()
    registration = make_registration(tmp_path, backend=backend)

    registration.register()
    registration.register()

    assert backend.linear_calls == 1
    assert backend.nonlinear_calls == 1
    assert len(backend.metric_calls) == 3
    assert len(backend.tiff_writes) == 2


def test_fresh_and_reused_warp_use_the_same_canonical_nonlinear_scalar(
    tmp_path: Path,
) -> None:
    fresh = FakeBackend()
    make_registration(tmp_path, backend=fresh).register()
    paths = round_paths(tmp_path)
    fresh_metric_scalar = fresh.metric_calls[-1][1]
    fresh_final_scalar = fresh.tiff_images[-1]

    store = ArtifactStore()
    store.mark_stale(paths.nonlinear_metrics)
    store.mark_stale(paths.registered)
    resumed = FakeBackend()
    make_registration(tmp_path, backend=resumed).register()
    resumed_metric_scalar = resumed.metric_calls[-1][1]
    resumed_final_scalar = resumed.tiff_images[-1]

    assert isinstance(fresh_metric_scalar, FakeImage)
    assert isinstance(resumed_metric_scalar, FakeImage)
    assert fresh_metric_scalar is fresh_final_scalar
    assert resumed_metric_scalar is resumed_final_scalar
    np.testing.assert_array_equal(
        fresh_metric_scalar.data,
        resumed_metric_scalar.data,
    )
    np.testing.assert_array_equal(
        fresh_metric_scalar.data,
        fresh.moving_image.data + 1_010.0,
    )
    assert not np.array_equal(
        fresh_metric_scalar.data,
        fresh.nonlinear_candidates[0].data,
    )
    assert len(fresh.histogram_match_calls) == 1
    assert len(resumed.histogram_match_calls) == 1
    assert (
        fresh.histogram_match_calls[0][1] is fresh.nonlinear_kwargs[0]["moving_image"]
    )
    assert resumed.nonlinear_calls == 0


def test_linear_failure_is_contextualized_without_committing_affine(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(fail_linear=True)

    with pytest.raises(RegistrationStageError, match="affine") as error:
        make_registration(tmp_path, backend=backend).register()

    paths = round_paths(tmp_path)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert not paths.affine.exists()
    assert not ArtifactStore().record_path(paths.affine).exists()
    assert backend.nonlinear_calls == 0


@pytest.mark.parametrize(
    ("target_name", "registration_kwargs", "stage"),
    [
        pytest.param(
            "fixed_reference.tif", {}, "fixed reference", id="fixed-reference"
        ),
        pytest.param(
            "fixed_crop_indices_2.json",
            {"crop": True},
            "fixed crop",
            id="crop",
        ),
        pytest.param(
            "image_metrics_pre-registration.txt",
            {},
            "pre-registration metrics",
            id="metrics",
        ),
        pytest.param("linear_transform_2.mat", {}, "affine", id="affine"),
        pytest.param("nonlinear_warp_2.nii.gz", {}, "nonlinear", id="nonlinear"),
        pytest.param(
            "nonlinear_registered_2.tif",
            {},
            "registered image",
            id="final-image",
        ),
    ],
)
def test_fresh_validator_failure_is_a_contextual_runtime_stage_error(
    tmp_path: Path,
    target_name: str,
    registration_kwargs: dict[str, object],
    stage: str,
) -> None:
    store = RejectingValidatorStore(target_name)

    with pytest.raises(RegistrationStageError, match=stage) as error:
        make_registration(
            tmp_path,
            backend=FakeBackend(),
            artifact_store=store,
            **registration_kwargs,
        ).register()

    assert isinstance(error.value.__cause__, CacheValidationError)


@pytest.mark.parametrize(
    ("to_array_call", "stage"),
    [
        pytest.param(1, "fixed reference", id="fixed-reference"),
        pytest.param(2, "registered image", id="final-image"),
    ],
)
def test_tiff_range_failure_is_contextualized_by_persistence_stage(
    tmp_path: Path,
    to_array_call: int,
    stage: str,
) -> None:
    backend = FakeBackend(fail_to_array_call=to_array_call)

    with pytest.raises(RegistrationStageError, match=stage) as error:
        make_registration(tmp_path, backend=backend).register()

    assert isinstance(error.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    ("tiff_role", "stage"),
    [
        pytest.param("fixed_reference", "fixed reference", id="fixed-reference"),
        pytest.param("nonlinear_registered_2", "registered image", id="final-image"),
    ],
)
def test_tiff_writer_failure_is_contextualized_by_persistence_stage(
    tmp_path: Path,
    tiff_role: str,
    stage: str,
) -> None:
    backend = FakeBackend(fail_tiff_role=tiff_role)

    with pytest.raises(RegistrationStageError, match=stage) as error:
        make_registration(tmp_path, backend=backend).register()

    assert isinstance(error.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param("linear", id="scalar-application"),
        pytest.param("nearestNeighbor", id="mask-application"),
        pytest.param("threshold", id="mask-thresholding"),
    ],
)
def test_cached_affine_reconstruction_failure_is_contextualized(
    tmp_path: Path,
    failure: str,
) -> None:
    make_registration(tmp_path, backend=FakeBackend()).register()
    paths = round_paths(tmp_path)
    store = ArtifactStore()
    record_before = store.record_path(paths.affine).read_bytes()
    resumed = FakeBackend(
        fail_affine_interpolation=(
            failure if failure in {"linear", "nearestNeighbor"} else None
        ),
        fail_threshold_mask=failure == "threshold",
    )

    with pytest.raises(RegistrationStageError, match="cached affine") as error:
        make_registration(tmp_path, backend=resumed).register()

    assert isinstance(error.value.__cause__, RuntimeError)
    assert store.record_path(paths.affine).read_bytes() == record_before


def test_estimated_affine_result_geometry_must_match_fixed_reference(
    tmp_path: Path,
) -> None:
    class WrongEstimatedGeometryBackend(FakeBackend):
        def linear(self, **kwargs: object) -> tuple[object, object | None, object]:
            scalar, mask, transform = super().linear(**kwargs)
            assert isinstance(scalar, FakeImage)
            assert isinstance(mask, FakeImage)
            scalar.origin = (10.0, 20.0, 30.0)
            mask.origin = scalar.origin
            return scalar, mask, transform

    backend = WrongEstimatedGeometryBackend()

    with pytest.raises(RegistrationStageError, match="affine result geometry"):
        make_registration(tmp_path, backend=backend).register()

    paths = round_paths(tmp_path)
    assert backend.nonlinear_calls == 0
    assert not paths.warp.exists()
    assert not paths.registered.exists()


def test_cached_affine_result_geometry_must_match_fixed_reference(
    tmp_path: Path,
) -> None:
    make_registration(tmp_path, backend=FakeBackend()).register()

    class WrongCachedGeometryBackend(FakeBackend):
        def apply_affine(self, **kwargs: object) -> object:
            result = super().apply_affine(**kwargs)  # type: ignore[arg-type]
            assert isinstance(result, FakeImage)
            result.origin = (10.0, 20.0, 30.0)
            return result

    backend = WrongCachedGeometryBackend()

    with pytest.raises(RegistrationStageError, match="affine result geometry"):
        make_registration(tmp_path, backend=backend).register()

    assert backend.nonlinear_calls == 0


def test_crop_application_failure_is_contextualized_after_bounds_commit(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(fail_crop_image=True)

    with pytest.raises(RegistrationStageError, match="fixed crop") as error:
        make_registration(tmp_path, backend=backend, crop=True).register()

    paths = round_paths(tmp_path)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert ArtifactStore().record_path(paths.fixed_crop).is_file()


def test_first_run_log_reports_every_effective_thread_value_and_source(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    resolved_native_threads: ResolvedNativeThreads,
) -> None:
    with caplog.at_level(logging.INFO, logger="cyexm.registration.whole_image"):
        make_registration(tmp_path, backend=FakeBackend()).register()

    for name in NATIVE_THREAD_ENV_VARS:
        assert (
            f"{name}={resolved_native_threads.values[name]} "
            f"({resolved_native_threads.sources[name]})"
        ) in caplog.text


@pytest.mark.parametrize("damage", ["corrupt-artifact", "missing-record"])
def test_affine_cache_corruption_or_incompleteness_is_not_recomputed(
    tmp_path: Path,
    damage: str,
) -> None:
    make_registration(tmp_path, backend=FakeBackend()).register()
    paths = round_paths(tmp_path)
    store = ArtifactStore()
    if damage == "corrupt-artifact":
        paths.affine.write_bytes(b"corrupt-affine")
    else:
        store.record_path(paths.affine).unlink()
    resumed = FakeBackend()

    with pytest.raises(CacheValidationError):
        make_registration(tmp_path, backend=resumed).register()

    assert resumed.linear_calls == 0


def test_changed_input_identity_rejects_stale_cache_without_force(
    tmp_path: Path,
) -> None:
    make_registration(tmp_path, backend=FakeBackend()).register()
    (tmp_path / "moving.tif").write_bytes(b"changed-moving-input")
    resumed = FakeBackend()

    with pytest.raises(CacheValidationError, match="request"):
        make_registration(tmp_path, backend=resumed).register()

    assert resumed.linear_calls == 0


def test_force_recomputes_after_input_identity_changes(tmp_path: Path) -> None:
    make_registration(tmp_path, backend=FakeBackend()).register()
    (tmp_path / "moving.tif").write_bytes(b"changed-moving-input")
    replacement = FakeBackend(
        affine_bytes=b"affine-for-changed-input",
        warp_bytes=b"warp-for-changed-input",
    )

    make_registration(
        tmp_path,
        backend=replacement,
        force_override=True,
    ).register()

    assert replacement.linear_calls == 1
    assert replacement.nonlinear_calls == 1
    assert round_paths(tmp_path).affine.read_bytes() == b"affine-for-changed-input"


@pytest.mark.parametrize(
    "changed_configuration",
    [
        pytest.param({"pixel_size": (0.2, 0.2, 0.2)}, id="spacing"),
        pytest.param({"linear_type": "Rigid"}, id="linear-method"),
        pytest.param({"linear_accuracy": "high"}, id="linear-accuracy"),
        pytest.param({"nonlinear_type": "SyN"}, id="nonlinear-method"),
        pytest.param({"nonlinear_accuracy": "low"}, id="nonlinear-accuracy"),
    ],
)
def test_changed_scientific_configuration_rejects_stale_reuse(
    tmp_path: Path,
    changed_configuration: dict[str, object],
) -> None:
    make_registration(tmp_path, backend=FakeBackend()).register()

    with pytest.raises(CacheValidationError, match="request"):
        make_registration(
            tmp_path,
            backend=FakeBackend(),
            **changed_configuration,
        ).register()


def test_changed_affine_identity_rejects_downstream_warp_reuse(
    tmp_path: Path,
) -> None:
    make_registration(tmp_path, backend=FakeBackend()).register()
    paths = round_paths(tmp_path)
    store = ArtifactStore()
    affine_request = CompletionRecord.read(store.record_path(paths.affine)).request
    store.commit(
        paths.affine,
        affine_request,
        writer=lambda path: path.write_bytes(b"affine-replaced"),
        validator=lambda path: path.read_bytes() == b"affine-replaced",
    )
    store.mark_stale(paths.linear_metrics)
    resumed = FakeBackend()

    with pytest.raises(CacheValidationError, match="request"):
        make_registration(tmp_path, backend=resumed).register()

    assert resumed.linear_calls == 0
    assert resumed.nonlinear_calls == 0


def test_force_replacement_invalidates_descendants_only_after_affine_commit(
    tmp_path: Path,
) -> None:
    make_registration(
        tmp_path,
        backend=FakeBackend(
            affine_bytes=b"affine-old",
            warp_bytes=b"warp-old",
        ),
    ).register()
    paths = round_paths(tmp_path)
    store = ArtifactStore()
    old_warp = paths.warp.read_bytes()
    old_final = paths.registered.read_bytes()
    replacement = FakeBackend(
        fail_nonlinear=True,
        affine_bytes=b"affine-new",
        warp_bytes=b"warp-new",
    )

    with pytest.raises(RegistrationStageError, match="nonlinear"):
        make_registration(
            tmp_path,
            backend=replacement,
            force_override=True,
        ).register()

    assert paths.affine.read_bytes() == b"affine-new"
    assert store.record_path(paths.affine).is_file()
    assert replacement.linear_calls == 1
    assert paths.warp.read_bytes() == old_warp
    assert paths.registered.read_bytes() == old_final
    for stale in (paths.warp, paths.nonlinear_metrics, paths.registered):
        assert not store.record_path(stale).exists()
        assert store.stale_path(stale).is_file()
    assert store.record_path(paths.linear_metrics).is_file()
    assert not store.stale_path(paths.linear_metrics).exists()


def test_failed_force_affine_writer_preserves_previous_reusable_pair(
    tmp_path: Path,
) -> None:
    make_registration(
        tmp_path,
        backend=FakeBackend(affine_bytes=b"affine-old", warp_bytes=b"warp-old"),
    ).register()
    paths = round_paths(tmp_path)
    store = ArtifactStore()
    old_affine = paths.affine.read_bytes()
    old_record = store.record_path(paths.affine).read_bytes()
    downstream_records = {
        path: store.record_path(path).read_bytes()
        for path in (
            paths.linear_metrics,
            paths.warp,
            paths.nonlinear_metrics,
            paths.registered,
        )
    }

    with pytest.raises(RegistrationStageError, match="affine") as error:
        make_registration(
            tmp_path,
            backend=FakeBackend(
                fail_affine_write=True,
                affine_bytes=b"affine-new",
            ),
            force_override=True,
        ).register()

    assert isinstance(error.value.__cause__, RuntimeError)
    assert paths.affine.read_bytes() == old_affine
    assert store.record_path(paths.affine).read_bytes() == old_record
    assert all(
        store.record_path(path).read_bytes() == record
        for path, record in downstream_records.items()
    )
    assert all(not store.stale_path(path).exists() for path in downstream_records)
    assert list(paths.output.glob(".*.tmp*")) == []

    resumed = FakeBackend()
    make_registration(tmp_path, backend=resumed).register()
    assert resumed.linear_calls == 0
    assert resumed.read_affine_calls == 1


def test_successful_force_replacement_clears_all_stale_markers(
    tmp_path: Path,
) -> None:
    make_registration(
        tmp_path,
        backend=FakeBackend(affine_bytes=b"affine-old", warp_bytes=b"warp-old"),
    ).register()
    paths = round_paths(tmp_path)
    store = ArtifactStore()

    make_registration(
        tmp_path,
        backend=FakeBackend(affine_bytes=b"affine-new", warp_bytes=b"warp-new"),
        force_override=True,
    ).register()

    assert paths.affine.read_bytes() == b"affine-new"
    assert paths.warp.read_bytes() == b"warp-new"
    for artifact in (
        paths.linear_metrics,
        paths.warp,
        paths.nonlinear_metrics,
        paths.registered,
    ):
        assert store.record_path(artifact).is_file()
        assert not store.stale_path(artifact).exists()


def test_crop_bounds_are_canonical_exclusive_zyx_artifacts(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()

    make_registration(tmp_path, backend=backend, crop=True).register()

    paths = round_paths(tmp_path)
    store = ArtifactStore()
    expected_fixed = (1, 5, 2, 6, 1, 5)
    expected_moving = (2, 6, 1, 5, 2, 6)
    assert backend.crop_bounds == [expected_fixed, expected_moving]
    assert paths.fixed_crop.read_text(encoding="utf-8") == "[1,5,2,6,1,5]\n"
    assert paths.moving_crop.read_text(encoding="utf-8") == "[2,6,1,5,2,6]\n"
    assert store.record_path(paths.fixed_crop).is_file()
    assert store.record_path(paths.moving_crop).is_file()
    affine_request = CompletionRecord.read(store.record_path(paths.affine)).request
    assert affine_request.geometry.shape == (4, 4, 4)
    assert affine_request.geometry.origin == (0.15, 0.3, 0.15)


def test_crop_bounds_are_reused_only_for_matching_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_registration(tmp_path, backend=FakeBackend(), crop=True).register()

    def unexpected_estimation(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("crop bounds were re-estimated")

    monkeypatch.setattr(
        "cyexm.registration.crop.robust_bounding_box", unexpected_estimation
    )
    resumed = FakeBackend()
    make_registration(tmp_path, backend=resumed, crop=True).register()

    assert resumed.crop_bounds == [
        (1, 5, 2, 6, 1, 5),
        (2, 6, 1, 5, 2, 6),
    ]

    (tmp_path / "moving.tif").write_bytes(b"changed-moving-input")
    with pytest.raises(CacheValidationError, match="request"):
        make_registration(tmp_path, backend=FakeBackend(), crop=True).register()


@pytest.mark.parametrize(
    "invalid_bounds",
    [
        pytest.param([1, 5, 2, 6, 1, True], id="boolean"),
        pytest.param([1, 9, 2, 6, 1, 5], id="outside-image"),
        pytest.param([1, 5, 2, 2, 1, 5], id="empty-axis"),
    ],
)
def test_invalid_completed_crop_bounds_are_cache_errors(
    tmp_path: Path,
    invalid_bounds: list[object],
) -> None:
    make_registration(tmp_path, backend=FakeBackend(), crop=True).register()
    paths = round_paths(tmp_path)
    store = ArtifactStore()
    request = CompletionRecord.read(store.record_path(paths.fixed_crop)).request
    serialized = json.dumps(invalid_bounds, separators=(",", ":")) + "\n"
    store.commit(
        paths.fixed_crop,
        request,
        writer=lambda path: path.write_text(serialized, encoding="utf-8"),
        validator=lambda path: path.read_text(encoding="utf-8") == serialized,
    )

    with pytest.raises(CacheValidationError, match="unreadable"):
        make_registration(tmp_path, backend=FakeBackend(), crop=True).register()


def test_nonlinear_hook_leaves_warp_lifecycle_to_parent(tmp_path: Path) -> None:
    backend = FakeBackend()
    fixed = write_stub(tmp_path / "fixed.tif", b"fixed-input")
    moving = write_stub(tmp_path / "moving.tif", b"moving-input")

    class HookedRegistration(ImageRegistration):
        hook_calls = 0
        candidate: FakeImage | None = None

        def _estimate_nonlinear(
            self,
            backend: object,
            fixed: object,
            moving_linear: object,
            fixed_mask: object,
            moving_linear_mask: object,
        ) -> tuple[object, Path]:
            del fixed, fixed_mask, moving_linear_mask
            assert isinstance(moving_linear, FakeImage)
            self.hook_calls += 1
            temporary = tmp_path / "hook-warp.nii.gz"
            temporary.write_bytes(b"warp-hook")
            self.candidate = moving_linear.clone()
            self.candidate.data += 500.0
            return self.candidate, temporary

    registration = HookedRegistration(
        fixed,
        moving,
        tmp_path / "output",
        2,
        _backend=backend,
        _artifact_store=ArtifactStore(),
    )

    registration.register()

    paths = round_paths(tmp_path)
    assert registration.hook_calls == 1
    assert paths.warp.read_bytes() == b"warp-hook"
    assert backend.warp_applications == [
        (paths.warp, "linear"),
        (paths.warp, "nearestNeighbor"),
    ]
    assert backend.metric_calls[-1][1] is not registration.candidate
    assert paths.nonlinear_metrics.is_file()
    assert paths.registered.is_file()


def test_temporary_warp_is_validated_and_masks_use_only_committed_path(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()

    make_registration(tmp_path, backend=backend).register()

    paths = round_paths(tmp_path)
    assert backend.temporary_warps == [tmp_path / "temporary-warp-1.nii.gz"]
    assert not backend.temporary_warps[0].exists()
    assert all(
        path != backend.temporary_warps[0] for path, _ in backend.warp_applications
    )
    assert backend.warp_applications == [
        (paths.warp, "linear"),
        (paths.warp, "nearestNeighbor"),
    ]
    assert len(backend.read_warp_paths) == 1
    candidate = backend.read_warp_paths[0]
    assert candidate.parent == paths.output
    assert candidate != paths.warp
    assert candidate.name.startswith(".nonlinear_warp_2")


def test_candidate_cleanup_oserror_does_not_fail_successful_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()
    candidate = tmp_path / "temporary-warp-1.nii.gz"
    original_unlink = Path.unlink

    def fail_candidate_unlink(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        if path == candidate:
            raise OSError("injected candidate cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_candidate_unlink)
    try:
        make_registration(tmp_path, backend=backend).register()

        assert round_paths(tmp_path).registered.is_file()
        assert candidate.is_file()
    finally:
        original_unlink(candidate, missing_ok=True)


def test_candidate_cleanup_oserror_does_not_mask_primary_stage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(fail_warp_interpolation="linear")
    candidate = tmp_path / "temporary-warp-1.nii.gz"
    original_unlink = Path.unlink

    def fail_candidate_unlink(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        if path == candidate:
            raise OSError("injected candidate cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_candidate_unlink)
    try:
        with pytest.raises(
            RegistrationStageError,
            match="nonlinear registration failed",
        ) as error:
            make_registration(tmp_path, backend=backend).register()

        assert isinstance(error.value.__cause__, RuntimeError)
        assert "injected linear warp application failure" in str(error.value.__cause__)
    finally:
        original_unlink(candidate, missing_ok=True)


@pytest.mark.parametrize(
    ("failure", "force_after_existing"),
    [
        pytest.param("linear", False, id="scalar-application"),
        pytest.param("histogram", False, id="histogram-matching"),
        pytest.param("nearestNeighbor", False, id="mask-application"),
        pytest.param("threshold", False, id="mask-thresholding"),
        pytest.param("histogram", True, id="force-after-affine-replacement"),
    ],
)
def test_postcommit_nonlinear_reconstruction_failure_stales_new_warp_record(
    tmp_path: Path,
    failure: str,
    force_after_existing: bool,
) -> None:
    if force_after_existing:
        make_registration(tmp_path, backend=FakeBackend()).register()
    backend = FakeBackend(
        affine_bytes=(
            b"affine-replacement" if force_after_existing else b"affine-fake"
        ),
        warp_bytes=b"warp-replacement",
        fail_warp_interpolation=(
            failure if failure in {"linear", "nearestNeighbor"} else None
        ),
        fail_histogram_match=failure == "histogram",
        fail_threshold_mask=failure == "threshold",
    )

    with pytest.raises(RegistrationStageError, match="nonlinear") as error:
        make_registration(
            tmp_path,
            backend=backend,
            force_override=force_after_existing,
        ).register()

    paths = round_paths(tmp_path)
    store = ArtifactStore()
    assert isinstance(error.value.__cause__, RuntimeError)
    assert paths.warp.read_bytes() == b"warp-replacement"
    assert not store.record_path(paths.warp).exists()
    assert store.stale_path(paths.warp).is_file()
    stale_record = CompletionRecord.read(store.stale_path(paths.warp))
    assert store.reuse(paths.warp, stale_record.request, backend.read_warp) is None
    if force_after_existing:
        assert paths.affine.read_bytes() == b"affine-replacement"
        assert store.record_path(paths.affine).is_file()


def test_reused_warp_runtime_failure_preserves_prior_completion_record(
    tmp_path: Path,
) -> None:
    make_registration(tmp_path, backend=FakeBackend()).register()
    paths = round_paths(tmp_path)
    store = ArtifactStore()
    record_before = store.record_path(paths.warp).read_bytes()
    resumed = FakeBackend(fail_warp_interpolation="linear")

    with pytest.raises(RegistrationStageError, match="nonlinear") as error:
        make_registration(tmp_path, backend=resumed).register()

    assert isinstance(error.value.__cause__, RuntimeError)
    assert store.record_path(paths.warp).read_bytes() == record_before
    assert not store.stale_path(paths.warp).exists()


def test_transformed_masks_are_thresholded_after_cached_transforms(
    tmp_path: Path,
) -> None:
    initial = FakeBackend()
    make_registration(tmp_path, backend=initial).register()
    resumed = FakeBackend()

    make_registration(tmp_path, backend=resumed).register()

    assert initial.threshold_mask_calls == 1
    assert resumed.threshold_mask_calls == 2


@pytest.mark.parametrize(
    "artifact_name",
    [
        pytest.param("fixed_reference", id="fixed-reference"),
        pytest.param("pre_metrics", id="pre-metrics"),
        pytest.param("affine", id="affine"),
        pytest.param("linear_metrics", id="linear-metrics"),
        pytest.param("warp", id="warp"),
        pytest.param("nonlinear_metrics", id="nonlinear-metrics"),
        pytest.param("registered", id="final-image"),
    ],
)
def test_corrupt_cached_artifacts_remain_cache_validation_errors(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    make_registration(tmp_path, backend=FakeBackend()).register()
    artifact = getattr(round_paths(tmp_path), artifact_name)
    artifact.write_bytes(b"corrupt-downstream-artifact")
    resumed = FakeBackend()

    with pytest.raises(CacheValidationError):
        make_registration(tmp_path, backend=resumed).register()

    assert resumed.nonlinear_calls == 0


def test_midrun_source_mutation_aborts_without_mislabeled_affine(
    tmp_path: Path,
) -> None:
    moving_path = tmp_path / "moving.tif"

    class InputChangingBackend(FakeBackend):
        changed = False

        def calculate_metrics(
            self, *args: object, **kwargs: object
        ) -> dict[str, float]:
            metrics = super().calculate_metrics(*args, **kwargs)
            if not self.changed:
                moving_path.write_bytes(b"moving-input-changed-during-run")
                self.changed = True
            return metrics

    backend = InputChangingBackend()
    registration = make_registration(tmp_path, backend=backend)
    original_contents = moving_path.read_bytes()
    original_identity = FileIdentity.from_path(moving_path)

    with pytest.raises(
        RegistrationStageError, match="source input identity changed"
    ) as error:
        registration.register()

    paths = round_paths(tmp_path)
    store = ArtifactStore()
    fixed_reference_request = CompletionRecord.read(
        store.record_path(paths.fixed_reference)
    ).request

    assert isinstance(error.value.__cause__, CacheValidationError)
    assert fixed_reference_request.inputs[1] == original_identity
    assert not paths.pre_metrics.exists()
    assert not store.record_path(paths.pre_metrics).exists()
    assert not paths.affine.exists()
    assert not store.record_path(paths.affine).exists()

    restore_file_identity(moving_path, original_contents, original_identity)
    registration.register()

    assert CompletionRecord.read(store.record_path(paths.affine)).request.inputs[1] == (
        original_identity
    )


def test_force_run_checks_snapshot_before_starting_next_expensive_stage(
    tmp_path: Path,
) -> None:
    moving_path = tmp_path / "moving.tif"

    class BetweenStagesMutationRegistration(ImageRegistration):
        changed = False

        def _commit_metrics(self, stage: str, *args: object, **kwargs: object) -> None:
            super()._commit_metrics(stage, *args, **kwargs)  # type: ignore[arg-type]
            if stage == "pre-registration" and not self.changed:
                moving_path.write_bytes(b"moving-input-changed-between-stages")
                self.changed = True

    fixed_path = write_stub(tmp_path / "fixed.tif", b"fixed-input")
    write_stub(moving_path, b"moving-input")
    backend = FakeBackend()
    registration = BetweenStagesMutationRegistration(
        fixed_path,
        moving_path,
        tmp_path / "output",
        2,
        force_override=True,
        _backend=backend,
        _artifact_store=ArtifactStore(),
    )

    with pytest.raises(
        RegistrationStageError, match="source input identity changed"
    ) as error:
        registration.register()

    assert isinstance(error.value.__cause__, CacheValidationError)
    assert backend.linear_calls == 0


@pytest.mark.parametrize("role", ["fixed", "moving"])
@pytest.mark.parametrize("mutation", ["change", "remove"])
def test_source_mutation_during_either_load_aborts_before_artifacts(
    tmp_path: Path,
    role: str,
    mutation: str,
) -> None:
    class LoadMutatingBackend(FakeBackend):
        def load_image(
            self,
            path: Path,
            pixel_size_xyz: tuple[float, float, float],
        ) -> object:
            loaded = super().load_image(path, pixel_size_xyz)
            if path.name == f"{role}.tif":
                if mutation == "change":
                    path.write_bytes(b"source-mutated-during-load")
                else:
                    path.unlink()
            return loaded

    backend = LoadMutatingBackend()

    with pytest.raises(
        RegistrationStageError, match="source input identity changed"
    ) as error:
        make_registration(tmp_path, backend=backend).register()

    assert isinstance(error.value.__cause__, CacheValidationError)
    assert backend.linear_calls == 0
    assert not any((tmp_path / "output").glob("*.complete.json"))


def test_source_mutation_after_commit_stales_new_record_and_restarts(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    registration = make_registration(tmp_path, backend=backend)
    moving_path = tmp_path / "moving.tif"
    original_contents = moving_path.read_bytes()
    original_identity = FileIdentity.from_path(moving_path)
    paths = round_paths(tmp_path)
    store = SourceMutatingStore(moving_path, paths.affine.name, "commit")
    registration._artifact_store = store

    with pytest.raises(
        RegistrationStageError, match="source input identity changed"
    ) as error:
        registration.register()

    assert isinstance(error.value.__cause__, CacheValidationError)
    assert paths.affine.is_file()
    assert not store.record_path(paths.affine).exists()
    assert store.stale_path(paths.affine).is_file()
    stale = CompletionRecord.read(store.stale_path(paths.affine))
    assert stale.request.inputs[1] == original_identity
    assert not paths.linear_metrics.exists()

    restore_file_identity(moving_path, original_contents, original_identity)
    registration.register()

    assert backend.linear_calls == 2
    assert store.record_path(paths.affine).is_file()
    assert not store.stale_path(paths.affine).exists()


def test_source_mutation_after_reuse_aborts_without_staling_old_record(
    tmp_path: Path,
) -> None:
    make_registration(tmp_path, backend=FakeBackend()).register()
    paths = round_paths(tmp_path)
    moving_path = tmp_path / "moving.tif"
    original_contents = moving_path.read_bytes()
    original_identity = FileIdentity.from_path(moving_path)
    old_record = ArtifactStore().record_path(paths.affine).read_bytes()
    store = SourceMutatingStore(moving_path, paths.affine.name, "reuse")
    backend = FakeBackend()
    registration = make_registration(
        tmp_path,
        backend=backend,
        artifact_store=store,
    )

    with pytest.raises(
        RegistrationStageError, match="source input identity changed"
    ) as error:
        registration.register()

    assert isinstance(error.value.__cause__, CacheValidationError)
    assert store.record_path(paths.affine).read_bytes() == old_record
    assert not store.stale_path(paths.affine).exists()
    assert backend.linear_calls == 0

    restore_file_identity(moving_path, original_contents, original_identity)
    registration.register()

    assert backend.linear_calls == 0
    assert backend.nonlinear_calls == 0


def test_source_mutation_then_reuse_error_still_runs_post_guard(
    tmp_path: Path,
) -> None:
    make_registration(tmp_path, backend=FakeBackend()).register()
    paths = round_paths(tmp_path)
    moving_path = tmp_path / "moving.tif"
    old_record = ArtifactStore().record_path(paths.affine).read_bytes()
    store = SourceMutatingStore(
        moving_path,
        paths.affine.name,
        "reuse",
        raise_after_mutation=True,
    )

    with pytest.raises(
        RegistrationStageError, match="source input identity changed"
    ) as error:
        make_registration(
            tmp_path,
            backend=FakeBackend(),
            artifact_store=store,
        ).register()

    assert isinstance(error.value.__cause__, CacheValidationError)
    assert error.value.__context__ is store.failure
    assert store.record_path(paths.affine).read_bytes() == old_record
    assert not store.stale_path(paths.affine).exists()


def test_source_mutation_then_late_commit_error_stales_promoted_record(
    tmp_path: Path,
) -> None:
    paths = round_paths(tmp_path)
    moving_path = tmp_path / "moving.tif"
    store = SourceMutatingStore(
        moving_path,
        paths.affine.name,
        "commit",
        raise_after_mutation=True,
    )

    with pytest.raises(
        RegistrationStageError, match="source input identity changed"
    ) as error:
        make_registration(
            tmp_path,
            backend=FakeBackend(),
            artifact_store=store,
        ).register()

    assert isinstance(error.value.__cause__, CacheValidationError)
    assert error.value.__context__ is store.failure
    assert paths.affine.is_file()
    assert not store.record_path(paths.affine).exists()
    assert store.stale_path(paths.affine).is_file()


def test_source_mutation_before_failed_commit_preserves_preexisting_record(
    tmp_path: Path,
) -> None:
    make_registration(tmp_path, backend=FakeBackend()).register()
    paths = round_paths(tmp_path)
    moving_path = tmp_path / "moving.tif"
    old_record = ArtifactStore().record_path(paths.affine).read_bytes()
    store = SourceMutatingStore(
        moving_path,
        paths.affine.name,
        "commit",
        raise_before_commit=True,
    )

    with pytest.raises(
        RegistrationStageError, match="source input identity changed"
    ) as error:
        make_registration(
            tmp_path,
            backend=FakeBackend(),
            artifact_store=store,
            force_override=True,
        ).register()

    assert isinstance(error.value.__cause__, CacheValidationError)
    assert error.value.__context__ is store.failure
    assert store.record_path(paths.affine).read_bytes() == old_record
    assert not store.stale_path(paths.affine).exists()


def test_stable_source_reuse_error_is_re_raised_unchanged(tmp_path: Path) -> None:
    make_registration(tmp_path, backend=FakeBackend()).register()
    paths = round_paths(tmp_path)
    failure = RuntimeError("stable reuse failure")
    store = StableRaisingStore(paths.affine.name, "reuse", failure)

    with pytest.raises(RuntimeError) as error:
        make_registration(
            tmp_path,
            backend=FakeBackend(),
            artifact_store=store,
        ).register()

    assert error.value is failure
    assert store.record_path(paths.affine).is_file()
    assert not store.stale_path(paths.affine).exists()


def test_stable_source_commit_error_retains_stage_cause_identity(
    tmp_path: Path,
) -> None:
    paths = round_paths(tmp_path)
    failure = RuntimeError("stable commit failure")
    store = StableRaisingStore(paths.affine.name, "commit", failure)

    with pytest.raises(RegistrationStageError, match="affine") as error:
        make_registration(
            tmp_path,
            backend=FakeBackend(),
            artifact_store=store,
        ).register()

    assert error.value.__cause__ is failure
    assert not store.record_path(paths.affine).exists()
    assert not store.stale_path(paths.affine).exists()


def test_register_creates_output_only_after_thread_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resolved_native_threads: ResolvedNativeThreads,
) -> None:
    output = tmp_path / "output"
    events: list[str] = []

    def configure(config: NativeThreadConfig | None) -> ResolvedNativeThreads:
        del config
        assert not output.exists()
        events.append("configure")
        return resolved_native_threads

    monkeypatch.setattr(whole_image, "configure_native_threads", configure)
    registration = ImageRegistration(
        tmp_path / "missing-fixed.tif",
        tmp_path / "missing-moving.tif",
        output,
        _backend=FakeBackend(),
    )

    with pytest.raises(FileNotFoundError):
        registration.register()

    assert events == ["configure"]
    assert output.is_dir()


def test_registration_does_not_add_root_logging_handlers(tmp_path: Path) -> None:
    root = logging.getLogger()
    handlers_before = tuple(root.handlers)

    make_registration(tmp_path, backend=FakeBackend()).register()

    assert tuple(root.handlers) == handlers_before


def test_crop_completion_records_capture_inputs_and_effective_threads(
    tmp_path: Path,
    resolved_native_threads: ResolvedNativeThreads,
) -> None:
    registration = make_registration(tmp_path, backend=FakeBackend(), crop=True)
    expected_inputs = (
        FileIdentity.from_path(tmp_path / "fixed.tif"),
        FileIdentity.from_path(tmp_path / "moving.tif"),
    )
    registration.register()
    paths = round_paths(tmp_path)

    for path, role in (
        (paths.fixed_crop, "fixed"),
        (paths.moving_crop, "moving"),
    ):
        request = completion_request(path)
        assert request.inputs == expected_inputs
        assert request.method == "robust-bounding-box"
        assert request.accuracy == "legacy-robust"
        assert request.parameters["image_role"] == role
        assert (
            request.parameters["axis_convention_version"] == "external-zyx-ants-xyz-v1"
        )
        assert request.native_threads == resolved_native_threads.values
        assert "native_thread_sources" not in request.parameters


def test_pre_registration_metrics_resample_unequal_moving_grid_to_fixed(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        fixed_image=fake_image(data=np.ones((6, 8, 10), dtype=np.float32)),
        moving_image=fake_image(data=np.ones((7, 9, 11), dtype=np.float32)),
    )

    make_registration(tmp_path, backend=backend, prepared_fixed=True).register()

    assert len(backend.resample_calls) == 2
    assert tuple(call[2] for call in backend.resample_calls) == (
        "linear",
        "nearestNeighbor",
    )
    fixed, moving, fixed_mask, moving_mask = backend.metric_calls[0]
    assert backend.geometry(moving) == backend.geometry(fixed)
    assert backend.geometry(moving_mask) == backend.geometry(fixed_mask)
