from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeBackend

import cyexm.registration.whole_image as whole_image
from cyexm.errors import CacheValidationError, RegistrationStageError
from cyexm.registration.artifacts import (
    ArtifactStore,
    CompletionRecord,
    FileIdentity,
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

    class TrackingBackend(FakeBackend):
        def load_image(self, path: Path, pixel_size_xyz: tuple[float, float, float]):
            events.append(f"load:{path.name}")
            return super().load_image(path, pixel_size_xyz)

    monkeypatch.setattr(whole_image, "configure_native_threads", configure)
    backend = TrackingBackend()

    make_registration(tmp_path, backend=backend, native_threads=requested).register()

    assert events[:3] == ["configure", "load:fixed.tif", "load:moving.tif"]


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
    make_registration(tmp_path, backend=backend).register()
    paths = round_paths(tmp_path)
    expected_inputs = (
        FileIdentity.from_path(tmp_path / "fixed.tif"),
        FileIdentity.from_path(tmp_path / "moving.tif"),
    )

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
        assert request.parameters["native_thread_sources"] == dict(
            resolved_native_threads.sources
        )

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

    monkeypatch.setattr(whole_image, "robust_bounding_box", unexpected_estimation)
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

        def _estimate_nonlinear(
            self,
            backend: object,
            fixed: object,
            moving_linear: object,
            fixed_mask: object,
            moving_linear_mask: object,
        ) -> tuple[object, Path]:
            del fixed, fixed_mask, moving_linear_mask
            self.hook_calls += 1
            temporary = tmp_path / "hook-warp.nii.gz"
            temporary.write_bytes(b"warp-hook")
            return moving_linear, temporary

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
    assert backend.warp_applications == [(paths.warp, "nearestNeighbor")]
    assert paths.nonlinear_metrics.is_file()
    assert paths.registered.is_file()


def test_temporary_warp_is_validated_and_masks_use_only_committed_path(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()

    make_registration(tmp_path, backend=backend).register()

    paths = round_paths(tmp_path)
    assert backend.temporary_warps == [tmp_path / "temporary-warp-1.nii.gz"]
    assert all(
        path != backend.temporary_warps[0] for path, _ in backend.warp_applications
    )
    assert backend.warp_applications == [(paths.warp, "nearestNeighbor")]
    assert len(backend.read_warp_paths) == 1
    candidate = backend.read_warp_paths[0]
    assert candidate.parent == paths.output
    assert candidate != paths.warp
    assert candidate.name.startswith(".nonlinear_warp_2")


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
        pytest.param("pre_metrics", id="metrics"),
        pytest.param("warp", id="warp"),
        pytest.param("registered", id="final-image"),
    ],
)
def test_corrupt_downstream_artifacts_remain_cache_errors(
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


def test_stage_requests_snapshot_input_identities_at_each_call(tmp_path: Path) -> None:
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

    make_registration(tmp_path, backend=InputChangingBackend()).register()
    paths = round_paths(tmp_path)
    store = ArtifactStore()
    fixed_reference_request = CompletionRecord.read(
        store.record_path(paths.fixed_reference)
    ).request
    affine_request = CompletionRecord.read(store.record_path(paths.affine)).request

    assert fixed_reference_request.inputs[1] != affine_request.inputs[1]
    assert affine_request.inputs[1] == FileIdentity.from_path(moving_path)


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


def test_crop_completion_records_capture_both_inputs_and_thread_sources(
    tmp_path: Path,
    resolved_native_threads: ResolvedNativeThreads,
) -> None:
    make_registration(tmp_path, backend=FakeBackend(), crop=True).register()
    paths = round_paths(tmp_path)
    expected_inputs = (
        FileIdentity.from_path(tmp_path / "fixed.tif"),
        FileIdentity.from_path(tmp_path / "moving.tif"),
    )

    for path, role in (
        (paths.fixed_crop, "fixed"),
        (paths.moving_crop, "moving"),
    ):
        request = completion_request(path)
        assert request.inputs == expected_inputs
        assert request.method == "robust-bounding-box"
        assert request.accuracy == "legacy-robust"
        assert request.parameters["image_role"] == role
        assert request.native_threads == resolved_native_threads.values
        assert request.parameters["native_thread_sources"] == dict(
            resolved_native_threads.sources
        )
