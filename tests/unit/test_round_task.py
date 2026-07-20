from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest

from cyexm.config.models import (
    NativeThreadSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.registration._request_policy import (
    affine_parameters,
    canonical_compact_json,
    channel_transform_parameters,
    chunk_grid,
    chunk_manifest_parameters,
    chunk_manifest_payload,
    common_parameters,
    crop_geometry,
    crop_parameters,
    fixed_reference_parameters,
    local_chunk_parameters,
    nonlinear_stage_parameters,
    registered_image_parameters,
    scalar_policy,
)
from cyexm.registration.artifacts import CompletionRecord, FileIdentity, StageRequest
from cyexm.runtime import (
    NATIVE_THREAD_ENV_VARS,
    NativeThreadConfig,
    ResolvedNativeThreads,
)
from cyexm.spatial import TIFF_GEOMETRY_ATOL, ImageGeometry
from cyexm.workflow.models import ChannelTask, RoundTask
from cyexm.workflow.round_task import run_round_task

pytestmark = pytest.mark.unit

_FULL_FIXED_GEOMETRY = ImageGeometry(
    shape=(6, 7, 8),
    origin=(1.23456789, -2.34567891, 3.45678912),
    spacing=(0.31, 0.27, 0.93),
    direction=(
        (0.813797681, -0.440969611, 0.378522306),
        (0.469846310, 0.882564119, 0.018028311),
        (-0.342020143, 0.163175911, 0.925416578),
    ),
)
_FULL_MOVING_GEOMETRY = replace(
    _FULL_FIXED_GEOMETRY,
    origin=(4.56789123, -5.67891234, 6.78912345),
)
_FIXED_CROP_BOUNDS = (1, 5, 1, 6, 2, 7)
_MOVING_CROP_BOUNDS = (0, 4, 2, 7, 1, 6)
_GEOMETRY = _FULL_FIXED_GEOMETRY
ITK_THREAD_ENV_VAR, *BLAS_THREAD_ENV_VARS = NATIVE_THREAD_ENV_VARS


def _native_thread_values(task: RoundTask) -> dict[str, int]:
    return {
        ITK_THREAD_ENV_VAR: task.native_threads.itk,
        **{name: task.native_threads.blas for name in BLAS_THREAD_ENV_VARS},
    }


def _write_completion(
    path: Path,
    request: StageRequest,
    *,
    contents: bytes | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"artifact:{path.name}".encode() if contents is None else contents)
    record = CompletionRecord(
        schema_version=1,
        request=request,
        artifact=FileIdentity.from_path(path),
    )
    path.with_name(f"{path.name}.complete.json").write_text(
        record.to_json(), encoding="utf-8"
    )


def _ensure_completion(
    path: Path,
    request: StageRequest,
    *,
    contents: bytes | None = None,
) -> None:
    record_path = path.with_name(f"{path.name}.complete.json")
    if path.is_file() and record_path.is_file():
        record = CompletionRecord.read(record_path)
        expected_contents = path.read_bytes() if contents is None else contents
        if (
            path.read_bytes() == expected_contents
            and record.request == request
            and record.artifact == FileIdentity.from_path(path)
        ):
            return
    _write_completion(path, request, contents=contents)


def _effective_geometries(task: RoundTask) -> tuple[ImageGeometry, ImageGeometry]:
    if not task.registration.crop:
        return _FULL_FIXED_GEOMETRY, _FULL_MOVING_GEOMETRY
    return (
        crop_geometry(_FULL_FIXED_GEOMETRY, _FIXED_CROP_BOUNDS),
        crop_geometry(_FULL_MOVING_GEOMETRY, _MOVING_CROP_BOUNDS),
    )


def _commit_registration_chain(task: RoundTask, *, affine_only: bool = False) -> None:
    paths = _registration_paths(task)
    inputs = (
        FileIdentity.from_path(task.fixed_image),
        FileIdentity.from_path(task.moving_image),
    )
    common = common_parameters(
        crop=task.registration.crop,
        imaging_round=task.round,
        pixel_size_xyz=task.voxel_spacing_xyz,
    )
    fixed_geometry, moving_geometry = _effective_geometries(task)
    if task.registration.crop:
        crop_cases: tuple[
            tuple[
                Path,
                Literal["fixed", "moving"],
                ImageGeometry,
                tuple[int, int, int, int, int, int],
            ],
            ...,
        ] = (
            (
                paths.fixed_crop,
                "fixed",
                _FULL_FIXED_GEOMETRY,
                _FIXED_CROP_BOUNDS,
            ),
            (
                paths.moving_crop,
                "moving",
                _FULL_MOVING_GEOMETRY,
                _MOVING_CROP_BOUNDS,
            ),
        )
        for path, role, geometry, bounds in crop_cases:
            crop_request = StageRequest.for_identities(
                stage=f"crop:{role}",
                inputs=inputs,
                geometry=geometry,
                method="robust-bounding-box",
                accuracy="legacy-robust",
                native_threads=_native_thread_values(task),
                upstream=None,
                parameters={**common, **crop_parameters(role)},
            )
            _ensure_completion(
                path,
                crop_request,
                contents=canonical_compact_json(list(bounds)).encode(),
            )
    fixed_reference_request = StageRequest.for_identities(
        stage="fixed-reference",
        inputs=inputs,
        geometry=fixed_geometry,
        method="uint16-tiff",
        accuracy="exact",
        native_threads=_native_thread_values(task),
        upstream=None,
        parameters={
            **common,
            **fixed_reference_parameters(0.0, 100.0),
        },
    )
    _ensure_completion(paths.fixed_reference, fixed_reference_request)
    affine_request = StageRequest.for_identities(
        stage="affine",
        inputs=inputs,
        geometry=fixed_geometry,
        method=task.registration.linear.type,
        accuracy=task.registration.linear.accuracy,
        native_threads=_native_thread_values(task),
        upstream=None,
        parameters={
            **common,
            **affine_parameters(
                moving_geometry=moving_geometry,
                linear_type=task.registration.linear.type,
                accuracy=task.registration.linear.accuracy,
            ),
        },
    )
    _ensure_completion(paths.affine, affine_request)
    if affine_only:
        return

    affine_identity = FileIdentity.from_path(paths.affine)
    selected_scalar_policy = scalar_policy(task.strategy)
    nonlinear_evidence: dict[str, object] = {}
    if task.strategy == "serial_chunked":
        assert task.chunk_size_zyx is not None
        assert task.overlap_fraction is not None
        chunk_directory = task.registration_directory / f"round_{task.round}_chunks"
        chunks = chunk_grid(
            fixed_geometry.shape,
            task.chunk_size_zyx,
            task.overlap_fraction,
        )
        local_identities: list[FileIdentity] = []
        for chunk in chunks:
            local_warp = chunk_directory / f"chunk_{chunk['chunk_id']:04d}_warp.nii.gz"
            local_geometry = crop_geometry(
                fixed_geometry,
                (
                    chunk["z_start_ext"],
                    chunk["z_end_ext"],
                    chunk["y_start_ext"],
                    chunk["y_end_ext"],
                    chunk["x_start_ext"],
                    chunk["x_end_ext"],
                ),
            )
            local_request = StageRequest.for_identities(
                stage=f"nonlinear-chunk:{chunk['chunk_id']}",
                inputs=inputs,
                geometry=local_geometry,
                method=task.registration.nonlinear.type,
                accuracy=task.registration.nonlinear.accuracy,
                native_threads=_native_thread_values(task),
                upstream=affine_identity,
                parameters=local_chunk_parameters(
                    chunk=chunk,
                    chunks=chunks,
                    chunk_size_zyx=task.chunk_size_zyx,
                    fixed_mask_geometry=local_geometry,
                    image_shape_zyx=fixed_geometry.shape,
                    moving_geometry=local_geometry,
                    moving_mask_geometry=local_geometry,
                    overlap_fraction=task.overlap_fraction,
                    nonlinear_accuracy=task.registration.nonlinear.accuracy,
                    scalar_postprocessing_policy=selected_scalar_policy,
                ),
            )
            _ensure_completion(local_warp, local_request)
            local_identities.append(FileIdentity.from_path(local_warp))

        manifest = chunk_directory / "manifest.json"
        local_dicts = [identity.to_dict() for identity in local_identities]
        manifest_payload = chunk_manifest_payload(
            round_number=task.round,
            image_shape_zyx=fixed_geometry.shape,
            chunk_size_zyx=task.chunk_size_zyx,
            overlap_fraction=task.overlap_fraction,
            chunks=chunks,
            local_warps=local_dicts,
        )
        manifest_request = StageRequest.for_identities(
            stage="nonlinear-chunk-manifest",
            inputs=inputs,
            geometry=fixed_geometry,
            method="serial-chunk-grid",
            accuracy="exact",
            native_threads=_native_thread_values(task),
            upstream=affine_identity,
            parameters=chunk_manifest_parameters(
                chunk_size_zyx=task.chunk_size_zyx,
                image_shape_zyx=fixed_geometry.shape,
                local_warps=local_dicts,
                nonlinear_accuracy=task.registration.nonlinear.accuracy,
                nonlinear_type=task.registration.nonlinear.type,
                overlap_fraction=task.overlap_fraction,
            ),
        )
        _ensure_completion(
            manifest,
            manifest_request,
            contents=canonical_compact_json(manifest_payload).encode(),
        )
        nonlinear_evidence = {
            "chunk_grid": [dict(chunk) for chunk in chunks],
            "chunk_manifest": FileIdentity.from_path(manifest).to_dict(),
            "chunk_size_zyx": list(task.chunk_size_zyx),
            "local_warps": local_dicts,
            "overlap_fraction": task.overlap_fraction,
        }
    warp_request = StageRequest.for_identities(
        stage="nonlinear",
        inputs=inputs,
        geometry=fixed_geometry,
        method=task.registration.nonlinear.type,
        accuracy=task.registration.nonlinear.accuracy,
        native_threads=_native_thread_values(task),
        upstream=affine_identity,
        parameters={
            **common,
            **nonlinear_stage_parameters(
                moving_geometry=fixed_geometry,
                registration_mode=(
                    "whole-image"
                    if task.strategy == "whole_image"
                    else "serial-chunked"
                ),
                nonlinear_accuracy=task.registration.nonlinear.accuracy,
                scalar_postprocessing_policy=selected_scalar_policy,
                evidence=nonlinear_evidence,
            ),
        },
    )
    _ensure_completion(paths.warp, warp_request)
    registered_request = StageRequest.for_identities(
        stage="registered-image",
        inputs=inputs,
        geometry=fixed_geometry,
        method=task.registration.nonlinear.type,
        accuracy=task.registration.nonlinear.accuracy,
        native_threads=_native_thread_values(task),
        upstream=FileIdentity.from_path(paths.warp),
        parameters={
            **common,
            **registered_image_parameters(
                minimum=0.0,
                maximum=100.0,
                scalar_postprocessing_policy=selected_scalar_policy,
            ),
        },
    )
    _ensure_completion(paths.registered, registered_request)


def _commit_channel(task: RoundTask, channel: ChannelTask) -> None:
    paths = _registration_paths(task)
    request = StageRequest.for_files(
        stage="channel-transform-v1",
        inputs=(task.fixed_image, channel.source, paths.affine, paths.warp),
        geometry=_FULL_FIXED_GEOMETRY,
        method="warp-affine",
        accuracy="not-applicable",
        native_threads=_native_thread_values(task),
        upstream=FileIdentity.from_path(paths.warp),
        parameters=channel_transform_parameters(
            pixel_size_xyz=task.voxel_spacing_xyz,
            interpolation=channel.interpolation,
        ),
    )
    _ensure_completion(channel.output, request)


def _task(
    tmp_path: Path,
    *,
    strategy: str = "whole_image",
    channel_names: tuple[str, ...] = ("b.tif", "a.tif"),
    force: bool = False,
    crop: bool = True,
) -> RoundTask:
    fixed = tmp_path / "inputs" / "fixed.tif"
    moving = tmp_path / "inputs" / "registration.tif"
    fixed.parent.mkdir(parents=True, exist_ok=True)
    fixed.write_bytes(b"fixed")
    moving.write_bytes(b"moving")
    channels = []
    for name in channel_names:
        source = tmp_path / "inputs" / name
        source.write_bytes(name.encode())
        channels.append(
            ChannelTask(
                source=source,
                output=tmp_path / "output" / "channels" / name,
                interpolation=(
                    "nearestNeighbor" if name.startswith("label") else "linear"
                ),
            )
        )
    return RoundTask(
        dataset_id="dataset-1",
        round=2,
        strategy=strategy,  # type: ignore[arg-type]
        fixed_image=fixed,
        moving_image=moving,
        registration_directory=tmp_path / "output" / "registration",
        log_path=tmp_path / "output" / "logs" / "round-002.log",
        voxel_spacing_xyz=(0.31, 0.27, 0.93),
        registration=RegistrationSpec(
            crop=crop,
            linear=RegistrationStageSpec(type="TRSAA", accuracy="low"),
            nonlinear=RegistrationStageSpec(type="SyNOnly", accuracy="high"),
        ),
        native_threads=NativeThreadSpec(itk=2, blas=3),
        channels=tuple(channels),
        chunk_size_zyx=(4, 8, 12) if strategy == "serial_chunked" else None,
        overlap_fraction=0.25 if strategy == "serial_chunked" else None,
        force=force,
    )


def _resolved(task: RoundTask) -> ResolvedNativeThreads:
    values = _native_thread_values(task)
    return ResolvedNativeThreads(
        values=values,
        sources={name: "explicit" for name in values},
    )


def _registration_paths(task: RoundTask) -> SimpleNamespace:
    directory = task.registration_directory
    return SimpleNamespace(
        output=directory,
        affine=directory / f"linear_transform_{task.round}.mat",
        warp=directory / f"nonlinear_warp_{task.round}.nii.gz",
        registered=directory / f"nonlinear_registered_{task.round}.tif",
        fixed_reference=directory / "fixed_reference.tif",
        pre_metrics=directory / "image_metrics_pre-registration.txt",
        linear_metrics=directory / "image_metrics_linearly-registered.txt",
        nonlinear_metrics=directory / "image_metrics_nonlinear-registered.txt",
        fixed_crop=directory / f"fixed_crop_indices_{task.round}.json",
        moving_crop=directory / f"moving_crop_indices_{task.round}.json",
    )


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    task: RoundTask,
    calls: list[tuple[str, str]],
    constructor_calls: list[tuple[str, dict[str, object]]],
    *,
    registration_failure: BaseException | None = None,
    channel_failure: str | None = None,
    after_registration: Callable[[RoundTask], None] | None = None,
    after_channel: Callable[[RoundTask, ChannelTask], None] | None = None,
) -> None:
    from cyexm.workflow import round_task as module

    class Paths:
        @classmethod
        def for_round(cls, output: Path, round_number: int) -> SimpleNamespace:
            assert output == task.registration_directory
            assert round_number == task.round
            return _registration_paths(task)

    def registration_class(strategy: str) -> type[object]:
        class Registration:
            def __init__(self, **kwargs: object) -> None:
                constructor_calls.append((strategy, kwargs))

            def register(self) -> None:
                calls.append(("register", strategy))
                if registration_failure is not None:
                    _commit_registration_chain(task, affine_only=True)
                    raise registration_failure
                _commit_registration_chain(task)
                if after_registration is not None:
                    after_registration(task)

        return Registration

    def transform_channel(**kwargs: object) -> Path:
        source = Path(kwargs["moving_image_path"])  # type: ignore[arg-type]
        output = Path(kwargs["output_path"])  # type: ignore[arg-type]
        calls.append(("channel", source.name))
        if source.name == channel_failure:
            raise RuntimeError(f"failed {source.name}")
        channel = next(item for item in task.channels if item.output == output)
        _commit_channel(task, channel)
        if after_channel is not None:
            after_channel(task, channel)
        return output

    monkeypatch.setattr(
        module, "configure_native_threads", lambda config: _resolved(task)
    )
    geometry_by_path = {
        task.fixed_image: _FULL_FIXED_GEOMETRY,
        task.moving_image: _FULL_MOVING_GEOMETRY,
        _registration_paths(task).fixed_reference: _effective_geometries(task)[0],
    }
    monkeypatch.setattr(
        module,
        "_inspect_image_geometry",
        lambda path, spacing: geometry_by_path[Path(path)],
    )
    monkeypatch.setattr(
        module,
        "_runtime",
        lambda: (
            registration_class("whole_image"),
            registration_class("serial_chunked"),
            Paths,
            transform_channel,
        ),
    )


def _replace_completion_request(
    path: Path, mutation: Callable[[StageRequest], StageRequest]
) -> None:
    record_path = path.with_name(f"{path.name}.complete.json")
    record = CompletionRecord.read(record_path)
    changed = replace(record, request=mutation(record.request))
    record_path.write_text(changed.to_json(), encoding="utf-8")


def _perturb_geometry(
    geometry: ImageGeometry,
    field: str,
    delta: float,
) -> ImageGeometry:
    if field == "origin":
        return replace(
            geometry,
            origin=(geometry.origin[0] + delta, *geometry.origin[1:]),
        )
    if field == "spacing":
        return replace(
            geometry,
            spacing=(geometry.spacing[0] + delta, *geometry.spacing[1:]),
        )
    if field == "direction":
        first_row = geometry.direction[0]
        return replace(
            geometry,
            direction=(
                (first_row[0] + delta, *first_row[1:]),
                *geometry.direction[1:],
            ),
        )
    raise ValueError(f"unsupported geometry field: {field}")


def _perturb_request_geometry(
    request: StageRequest,
    field: str,
    delta: float,
) -> StageRequest:
    parameters: dict[str, Any] = dict(request.parameters)
    for name in (
        "fixed_mask_geometry",
        "moving_geometry",
        "moving_mask_geometry",
    ):
        value = parameters.get(name)
        if isinstance(value, Mapping):
            geometry = ImageGeometry.from_dict(dict(value))
            parameters[name] = _perturb_geometry(geometry, field, delta).to_dict()
    return replace(
        request,
        geometry=_perturb_geometry(request.geometry, field, delta),
        parameters=parameters,
    )


def test_round_task_registers_then_transforms_channels_in_manifest_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, channel_names=("b.tif", "a.tif"))
    _install_runtime(monkeypatch, task, calls, constructors)

    result = run_round_task(task)

    assert calls == [
        ("register", "whole_image"),
        ("channel", "b.tif"),
        ("channel", "a.tif"),
    ]
    assert result.status == "completed"
    assert result.round == task.round
    assert result.dataset_id == task.dataset_id
    assert result.failure is None
    assert result.artifact_paths == (
        str(task.registration_directory / "linear_transform_2.mat"),
        str(task.registration_directory / "nonlinear_warp_2.nii.gz"),
        str(task.registration_directory / "nonlinear_registered_2.tif"),
        *(str(channel.output) for channel in task.channels),
    )


def test_round_task_uses_only_initial_post_registration_and_final_full_scans(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cyexm.workflow import round_task as module

    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, channel_names=("a.tif", "b.tif", "c.tif"))
    _install_runtime(monkeypatch, task, calls, constructors)
    original = module._completed_artifacts
    full_scans = 0

    def count_full_scan(*args: object, **kwargs: object) -> Any:
        nonlocal full_scans
        full_scans += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "_completed_artifacts", count_full_scan)

    result = run_round_task(task)

    assert result.status == "completed"
    assert full_scans == 3


@pytest.mark.parametrize("role", ["affine", "warp"])
@pytest.mark.parametrize("change", ["delete", "mutate"])
def test_channel_validation_rejects_changed_transform_completion_evidence(
    role: str,
    change: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, channel_names=("a.tif", "b.tif"))

    def change_transform_evidence(task: RoundTask, channel: ChannelTask) -> None:
        if channel != task.channels[0]:
            return
        path = Path(getattr(_registration_paths(task), role))
        if change == "delete":
            path.with_name(f"{path.name}.complete.json").unlink()
        else:
            _replace_completion_request(
                path,
                lambda request: replace(request, stage=f"changed-{request.stage}"),
            )

    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        after_channel=change_transform_evidence,
    )

    result = run_round_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "channel:a.tif"
    assert calls == [("register", "whole_image"), ("channel", "a.tif")]
    assert not task.channels[1].output.exists()


@pytest.mark.parametrize("role", ["affine", "warp"])
def test_channel_validation_uses_snapshots_accepted_by_registration_scan(
    role: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cyexm.workflow import round_task as module

    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, channel_names=("a.tif", "b.tif"))
    _install_runtime(monkeypatch, task, calls, constructors)
    original = module._completed_artifacts
    full_scans = 0

    def mutate_after_post_registration_scan(*args: object, **kwargs: object) -> Any:
        nonlocal full_scans
        evidence = original(*args, **kwargs)  # type: ignore[arg-type]
        full_scans += 1
        if full_scans == 2:
            path = Path(getattr(_registration_paths(task), role))
            _replace_completion_request(
                path,
                lambda request: replace(request, stage=f"changed-{request.stage}"),
            )
        return evidence

    monkeypatch.setattr(
        module,
        "_completed_artifacts",
        mutate_after_post_registration_scan,
    )

    result = run_round_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "channel:a.tif"
    assert calls == [("register", "whole_image"), ("channel", "a.tif")]
    assert not task.channels[1].output.exists()


@pytest.mark.parametrize("strategy", ["whole_image", "serial_chunked"])
def test_round_task_passes_exact_strategy_constructor_arguments(
    strategy: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, strategy=strategy, channel_names=("a.tif",), force=True)
    _install_runtime(monkeypatch, task, calls, constructors)

    result = run_round_task(task)

    assert result.status == "completed"
    assert len(constructors) == 1
    selected, arguments = constructors[0]
    assert selected == strategy
    expected: dict[str, object] = {
        "fixed_image_path": task.fixed_image,
        "moving_image_path": task.moving_image,
        "save_directory": task.registration_directory,
        "imaging_round": task.round,
        "pixel_size": task.voxel_spacing_xyz,
        "crop": task.registration.crop,
        "enable_logging": True,
        "force_override": True,
        "native_threads": NativeThreadConfig(itk_threads=2, blas_threads=3),
        "linear_type": task.registration.linear.type,
        "linear_accuracy": task.registration.linear.accuracy,
        "nonlinear_type": task.registration.nonlinear.type,
        "nonlinear_accuracy": task.registration.nonlinear.accuracy,
    }
    if strategy == "serial_chunked":
        expected["chunk_size"] = task.chunk_size_zyx
        expected["overlap_fraction"] = task.overlap_fraction
    assert arguments == expected
    thread_config = arguments["native_threads"]
    assert isinstance(thread_config, NativeThreadConfig)
    assert thread_config.itk_threads == task.native_threads.itk
    assert thread_config.blas_threads == task.native_threads.blas


def test_round_task_passes_force_threads_spacing_and_interpolation_to_channels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cyexm.workflow import round_task as module

    task = _task(tmp_path, channel_names=("label.tif",), force=True)
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    channel_arguments: list[dict[str, object]] = []
    _install_runtime(monkeypatch, task, calls, constructors)
    image, chunked, paths, transform = module._runtime()

    def capture(**kwargs: object) -> Path:
        channel_arguments.append(kwargs)
        return transform(**kwargs)

    monkeypatch.setattr(module, "_runtime", lambda: (image, chunked, paths, capture))

    run_round_task(task)

    assert channel_arguments == [
        {
            "fixed_image_path": task.fixed_image,
            "moving_image_path": task.channels[0].source,
            "affine_path": task.registration_directory / "linear_transform_2.mat",
            "warp_path": task.registration_directory / "nonlinear_warp_2.nii.gz",
            "output_path": task.channels[0].output,
            "pixel_size": task.voxel_spacing_xyz,
            "interpolation": "nearestNeighbor",
            "native_threads": constructors[0][1]["native_threads"],
            "force_override": True,
        }
    ]


def test_round_task_configures_threads_before_loading_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cyexm.workflow import round_task as module

    task = _task(tmp_path, channel_names=())
    order: list[str] = []
    original_runtime: Callable[[], tuple[Any, Any, Any, Any]] | None = None
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    _install_runtime(monkeypatch, task, calls, constructors)
    original_runtime = module._runtime

    def configure(config: NativeThreadConfig) -> ResolvedNativeThreads:
        order.append("threads")
        assert config.itk_threads == 2
        assert config.blas_threads == 3
        return _resolved(task)

    def load_runtime() -> tuple[Any, Any, Any, Any]:
        order.append("runtime")
        assert original_runtime is not None
        return original_runtime()

    monkeypatch.setattr(module, "configure_native_threads", configure)
    monkeypatch.setattr(module, "_runtime", load_runtime)

    run_round_task(task)

    assert order == ["threads", "runtime"]


def test_round_task_creates_output_directories_and_log_only_when_invoked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path)
    _install_runtime(monkeypatch, task, calls, constructors)
    assert not task.registration_directory.exists()
    assert not task.log_path.parent.exists()
    assert not task.channels[0].output.parent.exists()

    run_round_task(task)

    assert task.registration_directory.is_dir()
    assert task.log_path.is_file()
    assert task.channels[0].output.parent.is_dir()


def test_round_task_records_process_scheduler_and_expanded_thread_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cyexm.workflow import round_task as module

    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, channel_names=())
    _install_runtime(monkeypatch, task, calls, constructors)
    monkeypatch.setattr(module.socket, "gethostname", lambda: "node-42")
    monkeypatch.setattr(module.os, "getpid", lambda: 4321)
    monkeypatch.setenv("SLURM_JOB_ID", "9988")
    monkeypatch.setitem(
        sys.modules,
        "distributed",
        SimpleNamespace(
            get_worker=lambda: SimpleNamespace(
                get_current_task=lambda: "cyexm-round-dataset-1-2"
            )
        ),
    )

    result = run_round_task(task)

    assert result.hostname == "node-42"
    assert result.pid == 4321
    assert result.slurm_job_id == "9988"
    assert result.dask_task_key == "cyexm-round-dataset-1-2"
    assert result.native_threads == (
        (ITK_THREAD_ENV_VAR, 2),
        *((name, 3) for name in BLAS_THREAD_ENV_VARS),
    )
    assert result.started_at.endswith("Z")
    assert result.ended_at.endswith("Z")
    assert type(result.duration_seconds) is float
    assert result.duration_seconds >= 0.0


def test_round_task_does_not_import_distributed_to_find_a_task_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, channel_names=())
    _install_runtime(monkeypatch, task, calls, constructors)
    monkeypatch.delitem(sys.modules, "distributed", raising=False)

    result = run_round_task(task)

    assert result.dask_task_key is None
    assert "distributed" not in sys.modules


def test_registration_failure_returns_structured_partial_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path)
    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        registration_failure=RuntimeError("injected registration failure"),
    )

    result = run_round_task(task)

    affine = task.registration_directory / "linear_transform_2.mat"
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "registration"
    assert result.failure.exception_type == "RuntimeError"
    assert result.failure.message == "injected registration failure"
    assert result.artifact_paths == (str(affine),)
    assert calls == [("register", "whole_image")]
    assert affine.is_file()
    assert task.log_path.is_file()
    log_text = task.log_path.read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in log_text
    assert "RuntimeError: injected registration failure" in log_text


def test_middle_channel_failure_stops_later_channels_and_preserves_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, channel_names=("a.tif", "b.tif", "c.tif"))
    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        channel_failure="b.tif",
    )

    result = run_round_task(task)

    assert calls == [
        ("register", "whole_image"),
        ("channel", "a.tif"),
        ("channel", "b.tif"),
    ]
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "channel:b.tif"
    assert str(task.channels[0].output) in result.artifact_paths
    assert str(task.channels[1].output) not in result.artifact_paths
    assert str(task.channels[2].output) not in result.artifact_paths
    assert task.channels[0].output.is_file()
    assert not task.channels[2].output.exists()


def test_round_task_reports_artifacts_reused_from_before_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, channel_names=("a.tif",))
    _commit_registration_chain(task)
    _commit_channel(task, task.channels[0])
    _install_runtime(monkeypatch, task, calls, constructors)

    result = run_round_task(task)

    assert result.status == "completed"
    assert result.reused_artifacts == result.artifact_paths


@pytest.mark.parametrize("failure", [KeyboardInterrupt("stop"), SystemExit(7)])
def test_round_task_propagates_process_control_exceptions(
    failure: BaseException, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path)
    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        registration_failure=failure,
    )

    with pytest.raises(type(failure)):
        run_round_task(task)


def test_round_task_rejects_registration_without_authoritative_completion_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cyexm.workflow import round_task as module

    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path)
    _install_runtime(monkeypatch, task, calls, constructors)
    _whole, chunked, paths, transform = module._runtime()

    class InvalidRegistration:
        def __init__(self, **kwargs: object) -> None:
            constructors.append(("whole_image", kwargs))

        def register(self) -> None:
            calls.append(("register", "whole_image"))
            registration_paths = _registration_paths(task)
            _commit_registration_chain(task)
            registration_paths.affine.with_name(
                f"{registration_paths.affine.name}.complete.json"
            ).unlink()
            registration_paths.registered.unlink()
            registration_paths.registered.with_name(
                f"{registration_paths.registered.name}.complete.json"
            ).unlink()

    monkeypatch.setattr(
        module, "_runtime", lambda: (InvalidRegistration, chunked, paths, transform)
    )

    result = run_round_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "registration"
    assert result.artifact_paths == ()
    assert all(kind != "channel" for kind, _ in calls)


def test_force_run_does_not_report_preexisting_artifacts_as_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, channel_names=("a.tif",), force=True)
    _commit_registration_chain(task)
    _commit_channel(task, task.channels[0])
    _install_runtime(monkeypatch, task, calls, constructors)

    result = run_round_task(task)

    assert result.status == "completed"
    assert result.reused_artifacts == ()


def test_round_task_evidence_does_not_report_blank_scheduler_identifiers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, channel_names=())
    _install_runtime(monkeypatch, task, calls, constructors)
    monkeypatch.setenv("SLURM_JOB_ID", "")
    monkeypatch.setitem(
        sys.modules,
        "distributed",
        SimpleNamespace(
            get_worker=lambda: SimpleNamespace(get_current_task=lambda: "")
        ),
    )

    result = run_round_task(task)

    assert result.slurm_job_id is None
    assert result.dask_task_key is None
    assert os.environ["SLURM_JOB_ID"] == ""


@pytest.mark.parametrize(
    ("role", "mutation"),
    [
        ("affine", lambda request: replace(request, stage="not-affine")),
        ("affine", lambda request: replace(request, inputs=())),
        (
            "affine",
            lambda request: replace(
                request,
                native_threads={
                    **request.native_threads,
                    ITK_THREAD_ENV_VAR: 99,
                },
            ),
        ),
        (
            "warp",
            lambda request: replace(
                request,
                parameters={
                    **request.parameters,
                    "scalar_postprocessing_policy": "wrong-policy",
                },
            ),
        ),
        ("registered", lambda request: replace(request, upstream=None)),
    ],
)
def test_round_task_rejects_semantically_invalid_registration_completion(
    role: str,
    mutation: Callable[[StageRequest], StageRequest],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, channel_names=("a.tif",))

    def corrupt(task: RoundTask) -> None:
        path = getattr(_registration_paths(task), role)
        _replace_completion_request(path, mutation)

    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        after_registration=corrupt,
    )

    result = run_round_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "registration"
    assert str(getattr(_registration_paths(task), role)) not in result.artifact_paths
    assert all(kind != "channel" for kind, _ in calls)


def test_round_task_rejects_mismatched_serial_chunk_policy_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(
        tmp_path,
        strategy="serial_chunked",
        channel_names=("a.tif",),
    )

    def corrupt(task: RoundTask) -> None:
        _replace_completion_request(
            _registration_paths(task).warp,
            lambda request: replace(
                request,
                parameters={**request.parameters, "overlap_fraction": 0.5},
            ),
        )

    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        after_registration=corrupt,
    )

    result = run_round_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "registration"
    assert str(_registration_paths(task).warp) not in result.artifact_paths


def test_round_task_rejects_semantically_invalid_channel_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, channel_names=("a.tif",))

    def corrupt(_task: RoundTask, channel: ChannelTask) -> None:
        _replace_completion_request(
            channel.output,
            lambda request: replace(
                request,
                parameters={**request.parameters, "interpolation": "nearestNeighbor"},
            ),
        )

    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        after_channel=corrupt,
    )

    result = run_round_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "channel:a.tif"
    assert str(task.channels[0].output) not in result.artifact_paths


def test_round_task_compares_all_registration_paths_from_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cyexm.workflow import round_task as module

    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, channel_names=("a.tif",))
    _install_runtime(monkeypatch, task, calls, constructors)
    whole, chunked, _paths, transform = module._runtime()

    class DivergentPaths:
        @classmethod
        def for_round(cls, output: Path, round_number: int) -> SimpleNamespace:
            paths = _registration_paths(task)
            paths.registered = output / f"wrong_registered_{round_number}.tif"
            return paths

    monkeypatch.setattr(
        module,
        "_runtime",
        lambda: (whole, chunked, DivergentPaths, transform),
    )

    result = run_round_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "registration"
    assert all(kind != "channel" for kind, _ in calls)


def test_hostname_failure_becomes_a_structured_round_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cyexm.workflow import round_task as module

    task = _task(tmp_path, channel_names=())
    monkeypatch.setattr(
        module.socket,
        "gethostname",
        lambda: (_ for _ in ()).throw(RuntimeError("hostname unavailable")),
    )

    result = run_round_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.message == "hostname unavailable"
    assert result.hostname == "unknown"


def test_completion_prescan_failure_becomes_a_structured_round_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cyexm.workflow import round_task as module

    task = _task(tmp_path, channel_names=())
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    _install_runtime(monkeypatch, task, calls, constructors)
    original = module._completion_snapshot
    snapshot_calls = 0

    def fail_once(path: Path) -> object:
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls == 1:
            raise RuntimeError("prescan unavailable")
        return original(path)

    monkeypatch.setattr(module, "_completion_snapshot", fail_once)

    result = run_round_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.message == "prescan unavailable"


def test_unprintable_already_loaded_dask_key_is_treated_as_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, channel_names=())
    _install_runtime(monkeypatch, task, calls, constructors)

    class UnprintableKey:
        def __str__(self) -> str:
            raise RuntimeError("cannot stringify key")

    monkeypatch.setitem(
        sys.modules,
        "distributed",
        SimpleNamespace(
            get_worker=lambda: SimpleNamespace(
                get_current_task=lambda: UnprintableKey()
            )
        ),
    )

    result = run_round_task(task)

    assert result.status == "completed"
    assert result.dask_task_key is None


def test_serial_chunk_completion_requires_local_and_manifest_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, strategy="serial_chunked", channel_names=())

    def remove_records(task: RoundTask) -> None:
        chunk_directory = task.registration_directory / f"round_{task.round}_chunks"
        local_warp = chunk_directory / "chunk_0000_warp.nii.gz"
        manifest = chunk_directory / "manifest.json"
        for path in (local_warp, manifest):
            path.with_name(f"{path.name}.complete.json").unlink()

    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        after_registration=remove_records,
    )

    result = run_round_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "registration"


def test_serial_chunk_completion_rejects_a_noncanonical_derived_grid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, strategy="serial_chunked", channel_names=())

    def corrupt(task: RoundTask) -> None:
        path = _registration_paths(task).warp

        def mutation(request: StageRequest) -> StageRequest:
            grid = [dict(item) for item in request.parameters["chunk_grid"]]  # type: ignore[union-attr]
            grid[0]["x_end"] = 3
            return replace(
                request,
                parameters={**request.parameters, "chunk_grid": grid},
            )

        _replace_completion_request(path, mutation)

    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        after_registration=corrupt,
    )

    result = run_round_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "registration"


@pytest.mark.parametrize("field", ["origin", "spacing", "direction"])
def test_registration_completion_accepts_tiff_geometry_roundoff(
    field: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, crop=False, channel_names=())

    def perturb(task: RoundTask) -> None:
        paths = _registration_paths(task)
        for path in (
            paths.fixed_reference,
            paths.affine,
            paths.warp,
            paths.registered,
        ):
            _replace_completion_request(
                path,
                lambda request: _perturb_request_geometry(
                    request,
                    field,
                    TIFF_GEOMETRY_ATOL / 2.0,
                ),
            )

    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        after_registration=perturb,
    )

    result = run_round_task(task)

    assert result.status == "completed"


@pytest.mark.parametrize("field", ["origin", "spacing", "direction"])
def test_channel_completion_accepts_tiff_geometry_roundoff(
    field: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, crop=False, channel_names=("a.tif",))

    def perturb(_task: RoundTask, channel: ChannelTask) -> None:
        _replace_completion_request(
            channel.output,
            lambda request: _perturb_request_geometry(
                request,
                field,
                TIFF_GEOMETRY_ATOL / 2.0,
            ),
        )

    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        after_channel=perturb,
    )

    result = run_round_task(task)

    assert result.status == "completed"


def test_serial_local_completion_accepts_nested_tiff_geometry_roundoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(
        tmp_path,
        strategy="serial_chunked",
        crop=True,
        channel_names=(),
    )

    def perturb(task: RoundTask) -> None:
        local_warp = (
            task.registration_directory
            / f"round_{task.round}_chunks"
            / "chunk_0000_warp.nii.gz"
        )

        def mutate(request: StageRequest) -> StageRequest:
            changed = request
            for field in ("origin", "spacing", "direction"):
                changed = _perturb_request_geometry(
                    changed,
                    field,
                    TIFF_GEOMETRY_ATOL / 2.0,
                )
            return changed

        _replace_completion_request(local_warp, mutate)

    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        after_registration=perturb,
    )

    result = run_round_task(task)

    assert result.status == "completed"


@pytest.mark.parametrize("field", ["origin", "spacing", "direction"])
def test_serial_local_completion_rejects_geometry_beyond_tiff_tolerance(
    field: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(
        tmp_path,
        strategy="serial_chunked",
        crop=True,
        channel_names=(),
    )

    def perturb(task: RoundTask) -> None:
        local_warp = (
            task.registration_directory
            / f"round_{task.round}_chunks"
            / "chunk_0000_warp.nii.gz"
        )
        _replace_completion_request(
            local_warp,
            lambda request: _perturb_request_geometry(
                request,
                field,
                2.0 * TIFF_GEOMETRY_ATOL,
            ),
        )

    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        after_registration=perturb,
    )

    result = run_round_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "registration"


def test_serial_completion_rejects_tiny_non_geometry_float_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, strategy="serial_chunked", channel_names=())

    def perturb(task: RoundTask) -> None:
        path = _registration_paths(task).warp

        def mutate(request: StageRequest) -> StageRequest:
            overlap = request.parameters["overlap_fraction"]
            assert isinstance(overlap, float)
            return replace(
                request,
                parameters={
                    **request.parameters,
                    "overlap_fraction": overlap + TIFF_GEOMETRY_ATOL / 2.0,
                },
            )

        _replace_completion_request(path, mutate)

    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        after_registration=perturb,
    )

    result = run_round_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "registration"


@pytest.mark.parametrize("field", ["shape", "origin", "spacing", "direction"])
def test_registration_completion_rejects_wrong_full_geometry_provenance(
    field: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, crop=False, channel_names=())
    changes = {
        "shape": (9, 9, 9),
        "origin": (
            _GEOMETRY.origin[0] + 2.0 * TIFF_GEOMETRY_ATOL,
            *_GEOMETRY.origin[1:],
        ),
        "spacing": (
            _GEOMETRY.spacing[0] + 2.0 * TIFF_GEOMETRY_ATOL,
            *_GEOMETRY.spacing[1:],
        ),
        "direction": (
            (
                _GEOMETRY.direction[0][0] + 2.0 * TIFF_GEOMETRY_ATOL,
                *_GEOMETRY.direction[0][1:],
            ),
            *_GEOMETRY.direction[1:],
        ),
    }
    wrong_geometry = replace(_GEOMETRY, **{field: changes[field]})

    def corrupt(task: RoundTask) -> None:
        paths = _registration_paths(task)
        for path in (paths.affine, paths.warp, paths.registered):
            _replace_completion_request(
                path,
                lambda request: replace(
                    request,
                    geometry=wrong_geometry,
                    parameters={
                        **request.parameters,
                        **(
                            {"moving_geometry": wrong_geometry.to_dict()}
                            if "moving_geometry" in request.parameters
                            else {}
                        ),
                    },
                ),
            )

    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        after_registration=corrupt,
    )

    result = run_round_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "registration"


@pytest.mark.parametrize("field", ["shape", "origin", "spacing", "direction"])
def test_channel_completion_rejects_wrong_fixed_geometry_provenance(
    field: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, crop=False, channel_names=("a.tif",))
    changes = {
        "shape": (9, 9, 9),
        "origin": (
            _GEOMETRY.origin[0] + 2.0 * TIFF_GEOMETRY_ATOL,
            *_GEOMETRY.origin[1:],
        ),
        "spacing": (
            _GEOMETRY.spacing[0] + 2.0 * TIFF_GEOMETRY_ATOL,
            *_GEOMETRY.spacing[1:],
        ),
        "direction": (
            (
                _GEOMETRY.direction[0][0] + 2.0 * TIFF_GEOMETRY_ATOL,
                *_GEOMETRY.direction[0][1:],
            ),
            *_GEOMETRY.direction[1:],
        ),
    }
    wrong_geometry = replace(_GEOMETRY, **{field: changes[field]})

    def corrupt(_task: RoundTask, channel: ChannelTask) -> None:
        _replace_completion_request(
            channel.output,
            lambda request: replace(request, geometry=wrong_geometry),
        )

    _install_runtime(
        monkeypatch,
        task,
        calls,
        constructors,
        after_channel=corrupt,
    )

    result = run_round_task(task)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.stage == "channel:a.tif"


@pytest.mark.parametrize("crop", [False, True])
def test_round_task_accepts_authoritative_geometry_for_each_crop_mode(
    crop: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    constructors: list[tuple[str, dict[str, object]]] = []
    task = _task(tmp_path, crop=crop, channel_names=())
    _install_runtime(monkeypatch, task, calls, constructors)

    result = run_round_task(task)

    assert result.status == "completed"
