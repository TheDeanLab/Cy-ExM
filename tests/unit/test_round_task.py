from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cyexm.config.models import (
    NativeThreadSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.registration.artifacts import CompletionRecord, FileIdentity, StageRequest
from cyexm.runtime import (
    NATIVE_THREAD_ENV_VARS,
    NativeThreadConfig,
    ResolvedNativeThreads,
)
from cyexm.spatial import ImageGeometry
from cyexm.workflow.models import ChannelTask, RoundTask
from cyexm.workflow.round_task import run_round_task

pytestmark = pytest.mark.unit

_GEOMETRY = ImageGeometry(
    shape=(2, 3, 4),
    origin=(0.0, 0.0, 0.0),
    spacing=(0.3, 0.3, 0.9),
    direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
)
ITK_THREAD_ENV_VAR, *BLAS_THREAD_ENV_VARS = NATIVE_THREAD_ENV_VARS
_WHOLE_SCALAR_POLICY = "whole-global-histogram-match-v1"
_CHUNK_SCALAR_POLICY = "serial-chunk-local-histogram-match-core-stitch-v1"


def _native_thread_values(task: RoundTask) -> dict[str, int]:
    return {
        ITK_THREAD_ENV_VAR: task.native_threads.itk,
        **{name: task.native_threads.blas for name in BLAS_THREAD_ENV_VARS},
    }


def _write_completion(path: Path, request: StageRequest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"artifact:{path.name}".encode())
    record = CompletionRecord(
        schema_version=1,
        request=request,
        artifact=FileIdentity.from_path(path),
    )
    path.with_name(f"{path.name}.complete.json").write_text(
        record.to_json(), encoding="utf-8"
    )


def _ensure_completion(path: Path, request: StageRequest) -> None:
    record_path = path.with_name(f"{path.name}.complete.json")
    if path.is_file() and record_path.is_file():
        record = CompletionRecord.read(record_path)
        if record.request == request and record.artifact == FileIdentity.from_path(
            path
        ):
            return
    _write_completion(path, request)


def _common_parameters(task: RoundTask) -> dict[str, object]:
    return {
        "axis_convention_version": "external-zyx-ants-xyz-v1",
        "crop": task.registration.crop,
        "imaging_round": task.round,
        "pixel_size_xyz": list(task.voxel_spacing_xyz),
    }


def _linear_parameters(task: RoundTask) -> dict[str, object]:
    parameters: dict[str, object] = {
        "aff_metric": "mattes",
        "aff_random_sampling_rate": 0.5,
        "aff_sampling": 32,
        "mask_all_stages": True,
        "verbose": True,
    }
    iterations = {
        "high": [1000, 500, 250, 100],
        "medium": [500, 250, 100, 50],
        "low": [100, 50, 25, 10],
    }
    accuracy = task.registration.linear.accuracy
    if task.registration.linear.type == "TRSAA" and accuracy != "dry run":
        parameters["reg_iterations"] = iterations[accuracy]
    if accuracy == "dry run":
        parameters.update(
            effective_registration_type="Rigid",
            aff_iterations=[1],
            aff_smoothing_sigmas=[0],
            aff_shrink_factors=[8],
            aff_random_sampling_rate=0.1,
        )
    return parameters


def _nonlinear_parameters(task: RoundTask) -> dict[str, object]:
    iterations = {
        "high": [100, 70, 50],
        "medium": [50, 35, 25],
        "low": [5, 5, 5],
        "dry run": [1, 1, 1],
    }
    return {
        "aff_random_sampling_rate": 1.0,
        "flow_sigma": 3.0,
        "initial_transform": "Identity",
        "mask_all_stages": True,
        "reg_iterations": iterations[task.registration.nonlinear.accuracy],
        "singleprecision": False,
        "syn_metric": "mattes",
        "syn_sampling": 32,
        "total_sigma": 0.5,
        "verbose": True,
    }


def _commit_registration_chain(task: RoundTask, *, affine_only: bool = False) -> None:
    paths = _registration_paths(task)
    inputs = (
        FileIdentity.from_path(task.fixed_image),
        FileIdentity.from_path(task.moving_image),
    )
    common = _common_parameters(task)
    affine_request = StageRequest.for_identities(
        stage="affine",
        inputs=inputs,
        geometry=_GEOMETRY,
        method=task.registration.linear.type,
        accuracy=task.registration.linear.accuracy,
        native_threads=_native_thread_values(task),
        upstream=None,
        parameters={
            **common,
            "image_interpolation": "linear",
            "mask_interpolation": "nearestNeighbor",
            "moving_geometry": _GEOMETRY.to_dict(),
            "registration_parameters": _linear_parameters(task),
        },
    )
    _ensure_completion(paths.affine, affine_request)
    if affine_only:
        return

    affine_identity = FileIdentity.from_path(paths.affine)
    scalar_policy = (
        _WHOLE_SCALAR_POLICY if task.strategy == "whole_image" else _CHUNK_SCALAR_POLICY
    )
    nonlinear_evidence: dict[str, object] = {}
    if task.strategy == "serial_chunked":
        chunk_directory = task.registration_directory / f"round_{task.round}_chunks"
        local_warp = chunk_directory / "chunk_0000_warp.nii.gz"
        manifest = chunk_directory / "manifest.json"
        local_warp.parent.mkdir(parents=True, exist_ok=True)
        if not local_warp.exists():
            local_warp.write_bytes(b"local-warp")
        if not manifest.exists():
            manifest.write_bytes(b"chunk-manifest")
        chunk = {
            "chunk_id": 0,
            "z_start": 0,
            "z_end": 2,
            "y_start": 0,
            "y_end": 3,
            "x_start": 0,
            "x_end": 4,
            "z_start_ext": 0,
            "z_end_ext": 2,
            "y_start_ext": 0,
            "y_end_ext": 3,
            "x_start_ext": 0,
            "x_end_ext": 4,
        }
        nonlinear_evidence = {
            "chunk_grid": [chunk],
            "chunk_manifest": FileIdentity.from_path(manifest).to_dict(),
            "chunk_size_zyx": list(task.chunk_size_zyx or ()),
            "local_warps": [FileIdentity.from_path(local_warp).to_dict()],
            "overlap_fraction": task.overlap_fraction,
        }
    warp_request = StageRequest.for_identities(
        stage="nonlinear",
        inputs=inputs,
        geometry=_GEOMETRY,
        method=task.registration.nonlinear.type,
        accuracy=task.registration.nonlinear.accuracy,
        native_threads=_native_thread_values(task),
        upstream=affine_identity,
        parameters={
            **common,
            "image_interpolation": "linear",
            "mask_interpolation": "nearestNeighbor",
            "moving_geometry": _GEOMETRY.to_dict(),
            "registration_mode": (
                "whole-image" if task.strategy == "whole_image" else "serial-chunked"
            ),
            "registration_parameters": _nonlinear_parameters(task),
            "scalar_postprocessing_policy": scalar_policy,
            **nonlinear_evidence,
        },
    )
    _ensure_completion(paths.warp, warp_request)
    registered_request = StageRequest.for_identities(
        stage="registered-image",
        inputs=inputs,
        geometry=_GEOMETRY,
        method=task.registration.nonlinear.type,
        accuracy=task.registration.nonlinear.accuracy,
        native_threads=_native_thread_values(task),
        upstream=FileIdentity.from_path(paths.warp),
        parameters={
            **common,
            "format": "uint16-tiff",
            "minimum": 0.0,
            "maximum": 100.0,
            "scalar_postprocessing_policy": scalar_policy,
        },
    )
    _ensure_completion(paths.registered, registered_request)


def _commit_channel(task: RoundTask, channel: ChannelTask) -> None:
    paths = _registration_paths(task)
    request = StageRequest.for_files(
        stage="channel-transform-v1",
        inputs=(task.fixed_image, channel.source, paths.affine, paths.warp),
        geometry=_GEOMETRY,
        method="warp-affine",
        accuracy="not-applicable",
        native_threads=_native_thread_values(task),
        upstream=FileIdentity.from_path(paths.warp),
        parameters={
            "pixel_size_xyz_um": list(task.voxel_spacing_xyz),
            "interpolation": channel.interpolation,
            "transform_order": ["warp", "affine"],
            "whichtoinvert": [False, False],
            "axis_convention": "external-zyx-ants-xyz-v1",
            "output_format": "ome-compatible-tiff-uint16-v1",
            "postprocessing": "clip-uint16-no-normalization-v1",
        },
    )
    _ensure_completion(channel.output, request)


def _task(
    tmp_path: Path,
    *,
    strategy: str = "whole_image",
    channel_names: tuple[str, ...] = ("b.tif", "a.tif"),
    force: bool = False,
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
        voxel_spacing_xyz=(0.3, 0.3, 0.9),
        registration=RegistrationSpec(
            crop=True,
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
        affine=directory / f"linear_transform_{task.round}.mat",
        warp=directory / f"nonlinear_warp_{task.round}.nii.gz",
        registered=directory / f"nonlinear_registered_{task.round}.tif",
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
            return SimpleNamespace(
                affine=paths.affine,
                warp=paths.warp,
                registered=output / f"wrong_registered_{round_number}.tif",
            )

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
    original = module._completion_snapshot
    calls = 0

    def fail_once(path: Path) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
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
