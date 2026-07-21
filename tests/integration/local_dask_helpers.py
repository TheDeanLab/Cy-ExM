"""Picklable synthetic runners for real local-Dask integration tests."""

from __future__ import annotations

import json
import os
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

from cyexm.workflow.models import RoundResult, RoundTask

_FAILING_ROUND = 3


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _evidence_directory(task: RoundTask) -> Path:
    return task.registration_directory.parent / "synthetic-evidence"


def _reserve_attempt(task: RoundTask) -> tuple[int, Path]:
    evidence_directory = _evidence_directory(task)
    evidence_directory.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 100):
        reservation = evidence_directory / f"attempt-{attempt:03d}.reserve"
        try:
            with reservation.open("x", encoding="utf-8") as stream:
                stream.write(f"{os.getpid()}\n")
        except FileExistsError:
            continue
        return attempt, evidence_directory
    raise RuntimeError(f"could not reserve a synthetic attempt for round {task.round}")


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    encoded = json.dumps(value, allow_nan=False, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _worker_observation(task: RoundTask, attempt: int) -> dict[str, object]:
    from distributed import get_worker

    worker = get_worker()
    return {
        "attempt": attempt,
        "dask_task_key": str(worker.get_current_task()),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "round": task.round,
        "start_monotonic_ns": time.monotonic_ns(),
        "started_at": _utc_timestamp(),
        "worker_address": str(worker.address),
    }


def _planned_artifacts(task: RoundTask) -> tuple[str, ...]:
    return (
        str(task.registration_directory / f"linear_transform_{task.round}.mat"),
        str(task.registration_directory / f"nonlinear_warp_{task.round}.nii.gz"),
        str(task.registration_directory / f"nonlinear_registered_{task.round}.tif"),
        *(str(channel.output) for channel in task.channels),
    )


def _finish_observation(
    evidence_directory: Path,
    observation: dict[str, object],
    *,
    outcome: str,
) -> dict[str, object]:
    ended_at = _utc_timestamp()
    end_monotonic_ns = time.monotonic_ns()
    finished = {
        **observation,
        "end_monotonic_ns": end_monotonic_ns,
        "ended_at": ended_at,
        "outcome": outcome,
    }
    attempt = int(observation["attempt"])
    _atomic_write_json(
        evidence_directory / f"attempt-{attempt:03d}.json",
        finished,
    )
    return finished


def _completed_result(task: RoundTask, observation: dict[str, object]) -> RoundResult:
    duration_seconds = (
        int(observation["end_monotonic_ns"]) - int(observation["start_monotonic_ns"])
    ) / 1_000_000_000.0
    return RoundResult(
        dataset_id=task.dataset_id,
        round=task.round,
        status="completed",
        started_at=str(observation["started_at"]),
        ended_at=str(observation["ended_at"]),
        duration_seconds=duration_seconds,
        hostname=str(observation["hostname"]),
        pid=int(observation["pid"]),
        dask_task_key=str(observation["dask_task_key"]),
        slurm_job_id=None,
        strategy=task.strategy,
        native_threads=(
            ("itk", task.native_threads.itk),
            ("blas", task.native_threads.blas),
        ),
        artifact_paths=_planned_artifacts(task),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=None,
    )


def placement_runner(task: RoundTask) -> RoundResult:
    """Record placement and hold a worker's sole round slot for a bounded interval."""
    attempt, evidence_directory = _reserve_attempt(task)
    observation = _worker_observation(task, attempt)
    # The lowest round deliberately outlives all three short tasks. This proves
    # executor result ordering independently of actual completion order.
    time.sleep(0.8 if task.round == 2 else 0.2)
    finished = _finish_observation(
        evidence_directory,
        observation,
        outcome="completed",
    )
    return _completed_result(task, finished)


def independent_failure_runner(task: RoundTask) -> RoundResult:
    """Raise for one configured round while allowing independent rounds to finish."""
    attempt, evidence_directory = _reserve_attempt(task)
    observation = _worker_observation(task, attempt)
    time.sleep(0.1)
    if task.round == _FAILING_ROUND:
        _finish_observation(evidence_directory, observation, outcome="raised")
        raise RuntimeError(f"injected failure for round {task.round}")
    finished = _finish_observation(
        evidence_directory,
        observation,
        outcome="completed",
    )
    return _completed_result(task, finished)


def retry_once_runner(task: RoundTask) -> RoundResult:
    """Fail the atomic marker-creating attempt, then complete the retry."""
    attempt, evidence_directory = _reserve_attempt(task)
    observation = _worker_observation(task, attempt)
    marker = evidence_directory / "retry.marker"
    try:
        with marker.open("x", encoding="utf-8") as stream:
            stream.write(f"first attempt from pid {os.getpid()}\n")
    except FileExistsError:
        finished = _finish_observation(
            evidence_directory,
            observation,
            outcome="completed",
        )
        return _completed_result(task, finished)

    _finish_observation(evidence_directory, observation, outcome="raised")
    raise RuntimeError(f"injected first-attempt failure for round {task.round}")


__all__ = [
    "independent_failure_runner",
    "placement_runner",
    "retry_once_runner",
]
