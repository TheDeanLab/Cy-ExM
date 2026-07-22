from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from cyexm.config.models import (
    ExecutionPaths,
    ExecutionProfile,
    LocalDaskSpec,
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.execution.dask import LocalDaskExecutor
from cyexm.workflow.models import (
    ChannelTask,
    ReferenceChannelTask,
    ReferenceTask,
    RoundResult,
    RoundTask,
)

from .local_dask_helpers import (
    independent_failure_runner,
    placement_runner,
    retry_once_runner,
)

pytestmark = pytest.mark.integration

type _Runner = Callable[[ReferenceTask | RoundTask], RoundResult]


def _profile(
    tmp_path: Path,
    *,
    workers: int = 2,
    retries: int = 0,
) -> ExecutionProfile:
    return ExecutionProfile(
        schema_version=1,
        executor="local_dask",
        paths=ExecutionPaths(
            dataset_root=(tmp_path / "dataset").resolve(),
            output_root=(tmp_path / "output").resolve(),
            worker_local_directory=(tmp_path / "worker-scratch").resolve(),
        ),
        native_threads=NativeThreadSpec(itk=1, blas=1),
        local_dask=LocalDaskSpec(
            workers=workers,
            # Two worker threads make the one-slot resource constraint observable:
            # tasks on one worker still must not overlap.
            threads_per_worker=2,
            processes=True,
            memory_limit="1GB",
            startup_timeout="30s",
            task_retries=retries,
        ),
        slurm=None,
        source_path=(tmp_path / "execution.yaml").resolve(),
    )


def _task(tmp_path: Path, round_number: int) -> RoundTask:
    round_root = (tmp_path / "output" / f"round-{round_number:03d}").resolve()
    moving = (tmp_path / "dataset" / f"moving-{round_number}.tif").resolve()
    return RoundTask(
        dataset_id="bounded-local-dask",
        round=round_number,
        strategy="whole_image",
        fixed_image=(tmp_path / "dataset" / "fixed.tif").resolve(),
        moving_image=moving,
        registration_directory=round_root / "registration",
        log_path=round_root / "logs" / f"round-{round_number:03d}.log",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        registration=RegistrationSpec(
            crop=False,
            linear=RegistrationStageSpec(type="TRSAA", accuracy="dry run"),
            nonlinear=NonlinearRegistrationStageSpec(
                type="SyNOnly", accuracy="dry run", metric="mattes", radius=None
            ),
        ),
        native_threads=NativeThreadSpec(itk=1, blas=1),
        channels=(
            ChannelTask(
                source=moving,
                output=round_root / "channels" / moving.name,
                interpolation="linear",
            ),
        ),
        chunk_size_zyx=None,
        overlap_fraction=None,
    )


def _reference_task(tmp_path: Path) -> ReferenceTask:
    round_root = (tmp_path / "output" / "round-001").resolve()
    source = (tmp_path / "dataset" / "reference-source.tif").resolve()
    output = (tmp_path / "dataset" / "fixed.tif").resolve()
    return ReferenceTask(
        dataset_id="bounded-local-dask",
        round=1,
        registration_channel_source=source,
        registration_channel_output=output,
        registration_directory=round_root / "registration",
        crop_bounds_path=None,
        log_path=round_root / "logs" / "round-001.log",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        crop=False,
        native_threads=NativeThreadSpec(itk=1, blas=1),
        channels=(ReferenceChannelTask(source=source, output=output),),
    )


def _observations(task: RoundTask) -> tuple[dict[str, object], ...]:
    evidence_directory = task.registration_directory.parent / "synthetic-evidence"
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(evidence_directory.glob("attempt-*.json"))
    )


def _cluster_owned_child_pids() -> set[int]:
    import psutil

    owned: set[int] = set()
    for process in psutil.Process().children(recursive=True):
        if not process.is_running():
            continue
        try:
            command = " ".join(process.cmdline())
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            owned.add(process.pid)
            continue
        # Python's process-wide multiprocessing resource tracker starts on the
        # first spawned worker and intentionally lives until the pytest process
        # exits. It is not owned by an individual LocalCluster invocation.
        if "multiprocessing.resource_tracker" not in command:
            owned.add(process.pid)
    return owned


def _active_dask_scratch_entries(scratch: Path) -> tuple[Path, ...]:
    if not scratch.exists():
        return ()
    return tuple(
        path
        for path in scratch.rglob("*")
        if path.name.endswith(".dirlock")
        or path.name.startswith(("worker-", "nanny-", "scheduler-"))
    )


def _execute_and_assert_cleanup(
    profile: ExecutionProfile,
    runner: _Runner,
    tasks: tuple[RoundTask, ...],
) -> tuple[RoundResult, ...]:
    before = _cluster_owned_child_pids()
    results = LocalDaskExecutor(profile, _runner=runner).execute(
        _reference_task(profile.source_path.parent), tasks
    )
    scratch = profile.paths.worker_local_directory
    assert scratch is not None

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        remaining_children = _cluster_owned_child_pids() - before
        active_scratch = _active_dask_scratch_entries(scratch)
        if not remaining_children and not active_scratch:
            break
        time.sleep(0.05)

    assert _cluster_owned_child_pids() - before == set()
    assert _active_dask_scratch_entries(scratch) == ()
    assert tuple((result.round, result.status) for result in results[:1]) == (
        (1, "completed"),
    )
    return results[1:]


def _overlap(first: dict[str, object], second: dict[str, object]) -> bool:
    return int(first["start_monotonic_ns"]) < int(second["end_monotonic_ns"]) and int(
        second["start_monotonic_ns"]
    ) < int(first["end_monotonic_ns"])


def test_local_cluster_enforces_one_round_slot_and_preserves_result_order(
    tmp_path: Path,
) -> None:
    tasks = tuple(_task(tmp_path, round_number) for round_number in range(2, 6))

    results = _execute_and_assert_cleanup(
        _profile(tmp_path),
        placement_runner,
        tasks,
    )
    observations = tuple(_observations(task)[0] for task in tasks)

    assert tuple(result.round for result in results) == (2, 3, 4, 5)
    assert tuple(
        int(observation["round"])
        for observation in sorted(
            observations,
            key=lambda item: int(item["end_monotonic_ns"]),
        )
    ) != (2, 3, 4, 5)
    assert len({str(item["worker_address"]) for item in observations}) == 2
    assert len({int(item["pid"]) for item in observations}) == 2

    for index, first in enumerate(observations):
        for second in observations[index + 1 :]:
            if first["worker_address"] == second["worker_address"]:
                assert not _overlap(first, second)

    assert any(
        first["worker_address"] != second["worker_address"] and _overlap(first, second)
        for index, first in enumerate(observations)
        for second in observations[index + 1 :]
    )
    assert {(result.hostname, result.pid) for result in results} == {
        (str(item["hostname"]), int(item["pid"])) for item in observations
    }


def test_local_cluster_reports_one_failed_attempt_without_stopping_other_rounds(
    tmp_path: Path,
) -> None:
    tasks = tuple(_task(tmp_path, round_number) for round_number in (2, 3, 4))

    results = _execute_and_assert_cleanup(
        _profile(tmp_path, retries=0),
        independent_failure_runner,
        tasks,
    )

    assert tuple((result.round, result.status) for result in results) == (
        (2, "completed"),
        (3, "failed"),
        (4, "completed"),
    )
    assert tuple(len(_observations(task)) for task in tasks) == (1, 1, 1)
    failed_observation = _observations(tasks[1])[0]
    assert failed_observation["outcome"] == "raised"
    assert results[1].failure is not None
    assert results[1].failure.exception_type == "RuntimeError"
    assert results[1].failure.message == "injected failure for round 3"


def test_local_cluster_retries_once_after_atomic_marker_failure(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, 2)

    result = _execute_and_assert_cleanup(
        _profile(tmp_path, retries=1),
        retry_once_runner,
        (task,),
    )[0]
    observations = _observations(task)

    assert result.status == "completed"
    assert tuple(int(item["attempt"]) for item in observations) == (1, 2)
    assert tuple(str(item["outcome"]) for item in observations) == (
        "raised",
        "completed",
    )
    marker = task.registration_directory.parent / "synthetic-evidence/retry.marker"
    assert marker.is_file()
    assert marker.read_text(encoding="utf-8").startswith("first attempt from pid ")
