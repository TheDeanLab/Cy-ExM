"""Bounded orchestration helpers for real-SLURM workflow acceptance."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from cyexm.config.models import DatasetManifest, ExecutionProfile
from cyexm.execution.slurm import SlurmExecutor
from cyexm.io.atomic import atomic_write
from cyexm.workflow.api import run_workflow
from cyexm.workflow.models import (
    ReferenceTask,
    RoundResult,
    RoundTask,
    WorkflowPlan,
    WorkflowTask,
)
from hpc.environment import (
    collect_queue_snapshot,
    wait_for_queue_empty,
    write_evidence_json,
)

_PICKLE_LOCK = threading.RLock()
_PICKLE_DEPTH = 0
_PICKLE_WAS_REGISTERED = False


class HpcWorkflowAcceptanceError(RuntimeError):
    """A Task 5 workflow acceptance invariant was not satisfied."""


@dataclass(frozen=True, slots=True)
class Task5OutputRoots:
    """Explicit roots exclusively owned by one Task 5 acceptance run."""

    base: Path
    parallel: Path
    sequential: Path
    failure: Path
    cancellation: Path

    @property
    def scenario_roots(self) -> tuple[Path, ...]:
        return (
            self.parallel,
            self.sequential,
            self.failure,
            self.cancellation,
        )


def bounded_acceptance_manifest(manifest: DatasetManifest) -> DatasetManifest:
    """Select exactly one whole-image and one serial-chunked moving round."""
    selected = []
    for strategy in ("whole_image", "serial_chunked"):
        matching = sorted(
            (item for item in manifest.rounds if item.strategy == strategy),
            key=lambda item: item.round,
        )
        if not matching:
            raise HpcWorkflowAcceptanceError(
                f"approved dataset contains no {strategy} moving round"
            )
        selected.append(matching[0])
    return replace(
        manifest, rounds=tuple(sorted(selected, key=lambda item: item.round))
    )


def task5_output_roots(profile: ExecutionProfile) -> Task5OutputRoots:
    """Describe Task 5's named roots without mutating the caller's results root."""
    base = profile.paths.output_root / "cyexm-phase-2e-task-5"
    return Task5OutputRoots(
        base=base,
        parallel=base / "parallel",
        sequential=base / "sequential-reference",
        failure=base / "failure-resume",
        cancellation=base / "cancellation-resume",
    )


def _slurm_walltime_seconds(value: object) -> int:
    if type(value) is not str or not value:
        raise HpcWorkflowAcceptanceError(
            f"approved SLURM walltime must be a nonempty string; got {value!r}"
        )
    minute_only = False
    if "-" in value:
        if value.count("-") != 1:
            raise HpcWorkflowAcceptanceError(
                f"approved SLURM walltime is malformed: {value!r}"
            )
        days_text, clock = value.split("-", 1)
        parts = clock.split(":")
        if len(parts) not in {1, 2, 3}:
            raise HpcWorkflowAcceptanceError(
                f"approved SLURM walltime is malformed: {value!r}"
            )
        parts.extend(["0"] * (3 - len(parts)))
        hours_text, minutes_text, seconds_text = parts
    else:
        days_text = "0"
        parts = value.split(":")
        minute_only = len(parts) in {1, 2}
        if len(parts) == 1:
            hours_text, minutes_text, seconds_text = "0", parts[0], "0"
        elif len(parts) == 2:
            hours_text, minutes_text, seconds_text = "0", parts[0], parts[1]
        elif len(parts) == 3:
            hours_text, minutes_text, seconds_text = parts
        else:
            raise HpcWorkflowAcceptanceError(
                f"approved SLURM walltime is malformed: {value!r}"
            )
    fields = (days_text, hours_text, minutes_text, seconds_text)
    if any(not field.isdigit() for field in fields):
        raise HpcWorkflowAcceptanceError(
            f"approved SLURM walltime is malformed: {value!r}"
        )
    days, hours, minutes, seconds = (int(field, 10) for field in fields)
    if minute_only:
        if seconds > 59:
            raise HpcWorkflowAcceptanceError(
                f"approved SLURM walltime is malformed: {value!r}"
            )
        total = minutes * 60 + seconds
    elif minutes > 59 or seconds > 59 or (days > 0 and hours > 23):
        raise HpcWorkflowAcceptanceError(
            f"approved SLURM walltime is malformed: {value!r}"
        )
    else:
        total = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    if total < 1:
        raise HpcWorkflowAcceptanceError(
            f"approved SLURM walltime must be positive: {value!r}"
        )
    return total


def coordination_timeout_seconds(profile: ExecutionProfile) -> float:
    """Bound worker coordination by the exact approved SLURM job walltime."""
    if profile.executor != "slurm" or profile.slurm is None:
        raise HpcWorkflowAcceptanceError(
            "coordination timeout requires an approved SLURM profile"
        )
    return float(_slurm_walltime_seconds(profile.slurm.walltime))


def prepare_task5_output_roots(roots: Task5OutputRoots) -> dict[str, object]:
    """Create only new named Task 5 roots below the caller's general result root."""
    try:
        general = roots.base.parent.resolve(strict=True)
    except OSError as error:
        raise HpcWorkflowAcceptanceError(
            f"general results root must be an existing directory: {roots.base.parent}"
        ) from error
    if not general.is_dir():
        raise HpcWorkflowAcceptanceError(
            f"general results root must be an existing directory: {general}"
        )
    if roots.base.parent.resolve(strict=False) != general:
        raise HpcWorkflowAcceptanceError("Task 5 base must be a direct results child")
    if roots.base.exists() or roots.base.is_symlink():
        raise HpcWorkflowAcceptanceError(
            f"Task 5 owned base must not exist before acceptance: {roots.base}"
        )
    expected_scenarios = {
        roots.base / "parallel",
        roots.base / "sequential-reference",
        roots.base / "failure-resume",
        roots.base / "cancellation-resume",
    }
    if set(roots.scenario_roots) != expected_scenarios:
        raise HpcWorkflowAcceptanceError("Task 5 scenario roots are not canonical")

    entries_before = sorted(path.name for path in general.iterdir())
    roots.base.mkdir()
    for root in roots.scenario_roots:
        root.mkdir()
    return {
        "schema_version": 1,
        "general_results_root": str(general),
        "entries_before": entries_before,
        "owned_base": str(roots.base.resolve(strict=True)),
        "scenario_roots": [
            str(root.resolve(strict=True)) for root in roots.scenario_roots
        ],
    }


def validate_task5_output_ownership(
    roots: Task5OutputRoots,
    ownership: Mapping[str, object],
) -> None:
    """Prove the caller's general result root changed only by the owned base."""
    general = roots.base.parent.resolve(strict=True)
    before = ownership.get("entries_before")
    if not isinstance(before, list) or any(type(item) is not str for item in before):
        raise HpcWorkflowAcceptanceError("Task 5 ownership evidence is malformed")
    if ownership.get("general_results_root") != str(general):
        raise HpcWorkflowAcceptanceError("Task 5 general results identity changed")
    expected_entries = sorted([*before, roots.base.name])
    observed_entries = sorted(path.name for path in general.iterdir())
    if observed_entries != expected_entries:
        raise HpcWorkflowAcceptanceError(
            "Task 5 modified the general results root outside its owned base: "
            f"expected={expected_entries!r}, observed={observed_entries!r}"
        )
    if roots.base.is_symlink() or not roots.base.is_dir():
        raise HpcWorkflowAcceptanceError("Task 5 owned base is not a real directory")
    if sorted(path.name for path in roots.base.iterdir()) != sorted(
        path.name for path in roots.scenario_roots
    ):
        raise HpcWorkflowAcceptanceError(
            "Task 5 wrote outside its explicitly owned scenario roots"
        )
    expected_resolved = [
        str(root.resolve(strict=True)) for root in roots.scenario_roots
    ]
    if ownership.get("owned_base") != str(roots.base.resolve(strict=True)) or (
        ownership.get("scenario_roots") != expected_resolved
    ):
        raise HpcWorkflowAcceptanceError("Task 5 owned path identities changed")


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise HpcWorkflowAcceptanceError(
            f"round timestamp is not strict ISO-8601: {value!r}"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HpcWorkflowAcceptanceError(
            f"round timestamp must be timezone-aware: {value!r}"
        )
    return parsed


def parallel_placement_evidence(
    results: tuple[RoundResult, ...],
) -> dict[str, object]:
    """Prove a completed reference barrier and two overlapping moving rounds."""
    references = tuple(
        result for result in results if result.strategy == "reference-preparation"
    )
    moving = tuple(
        result for result in results if result.strategy != "reference-preparation"
    )
    if (
        len(references) != 1
        or references[0].status != "completed"
        or len(moving) != 2
        or any(result.status != "completed" for result in moving)
    ):
        raise HpcWorkflowAcceptanceError(
            "parallel placement requires one completed reference result and "
            "exactly two completed moving-round results"
        )
    if {result.strategy for result in moving} != {"whole_image", "serial_chunked"}:
        raise HpcWorkflowAcceptanceError(
            "parallel placement requires whole_image and serial_chunked rounds"
        )
    job_ids = [result.slurm_job_id for result in moving]
    if any(type(job_id) is not str or not job_id for job_id in job_ids) or len(
        set(job_ids)
    ) != len(moving):
        raise HpcWorkflowAcceptanceError(
            "parallel rounds must report one unique SLURM job ID each"
        )
    hostnames = [result.hostname for result in moving]
    if any(type(host) is not str or not host for host in hostnames) or len(
        set(hostnames)
    ) != len(moving):
        raise HpcWorkflowAcceptanceError(
            "parallel rounds must report one unique hostname each"
        )
    latest_start = max(_timestamp(result.started_at) for result in moving)
    earliest_end = min(_timestamp(result.ended_at) for result in moving)
    if latest_start >= earliest_end:
        raise HpcWorkflowAcceptanceError(
            "parallel round timestamps do not prove concurrent execution"
        )
    return {
        "rounds": [result.round for result in moving],
        "job_ids": job_ids,
        "hostnames": hostnames,
        "overlap_proven": True,
        "strategies": [result.strategy for result in moving],
    }


def numeric_comparison_evidence(
    expected: object,
    observed: object,
    *,
    location: str,
    require_close: bool,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict[str, object]:
    """Record numeric differences, optionally enforcing strict closeness."""
    import numpy as np

    expected_array = np.asarray(expected)
    observed_array = np.asarray(observed)
    if expected_array.shape != observed_array.shape:
        raise HpcWorkflowAcceptanceError(
            f"{location} shape changed: "
            f"{expected_array.shape!r} != {observed_array.shape!r}"
        )
    if expected_array.dtype != observed_array.dtype:
        raise HpcWorkflowAcceptanceError(
            f"{location} dtype changed: "
            f"{expected_array.dtype!r} != {observed_array.dtype!r}"
        )
    if not np.all(np.isfinite(expected_array)) or not np.all(
        np.isfinite(observed_array)
    ):
        raise HpcWorkflowAcceptanceError(f"{location} contains non-finite values")
    if require_close:
        np.testing.assert_allclose(
            observed_array,
            expected_array,
            rtol=rtol,
            atol=atol,
        )
    difference = np.abs(
        observed_array.astype(np.float64) - expected_array.astype(np.float64)
    )
    return {
        "shape": list(expected_array.shape),
        "reference_dtype": str(expected_array.dtype),
        "candidate_dtype": str(observed_array.dtype),
        "array_equal": bool(np.array_equal(observed_array, expected_array)),
        "close_required": require_close,
        "maximum_absolute_difference": float(difference.max(initial=0.0)),
        "mean_absolute_difference": float(difference.mean()),
        "p99_9_absolute_difference": float(np.percentile(difference, 99.9)),
        "rtol": rtol,
        "atol": atol,
    }


def metric_comparison_evidence(
    reference: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    location: str,
    absolute_tolerance: float = 0.01,
) -> dict[str, object]:
    """Require finite canonical stage metrics within one absolute bound."""
    import math

    expected_names = {"Correlation", "MattesMutualInformation"}
    if set(reference) != expected_names or set(candidate) != expected_names:
        raise HpcWorkflowAcceptanceError(f"{location} metric schema changed")
    differences: dict[str, float] = {}
    for name in sorted(expected_names):
        reference_value = reference[name]
        candidate_value = candidate[name]
        if not math.isfinite(reference_value) or not math.isfinite(candidate_value):
            raise HpcWorkflowAcceptanceError(f"{location} {name} is non-finite")
        difference = abs(candidate_value - reference_value)
        if difference > absolute_tolerance:
            raise HpcWorkflowAcceptanceError(
                f"{location} {name} difference {difference} exceeds "
                f"{absolute_tolerance}"
            )
        differences[name] = difference
    return {
        "reference": dict(reference),
        "candidate": dict(candidate),
        "absolute_differences": differences,
        "absolute_tolerance": absolute_tolerance,
    }


def _scientific_comparison_evidence(
    reference: WorkflowPlan,
    candidate: WorkflowPlan,
    *,
    require_close: bool,
    metric_absolute_tolerance: float,
) -> dict[str, object]:
    """Compare exact structure plus strict or stochastic numerical evidence."""
    import numpy as np
    import tifffile

    from cyexm.io.tiff import read_tiff_geometry
    from cyexm.metrics import read_metrics
    from cyexm.registration.transforms import import_affine_transform
    from cyexm.registration.whole_image import RegistrationPaths

    relative_tolerance = 1e-4
    absolute_tolerance = 1e-5
    if reference.scientific_fingerprint != candidate.scientific_fingerprint:
        raise HpcWorkflowAcceptanceError(
            "sequential and SLURM plans have different scientific fingerprints"
        )
    if len(reference.tasks) != len(candidate.tasks):
        raise HpcWorkflowAcceptanceError(
            "sequential and SLURM plans contain different round counts"
        )

    def tiff_evidence(
        expected_path: Path,
        observed_path: Path,
        *,
        location: str,
    ) -> dict[str, object]:
        expected = tifffile.imread(expected_path)
        observed = tifffile.imread(observed_path)
        if expected.dtype != observed.dtype:
            raise HpcWorkflowAcceptanceError(
                f"{location} dtype changed: {expected.dtype!r} != {observed.dtype!r}"
            )
        if expected.shape != observed.shape:
            raise HpcWorkflowAcceptanceError(
                f"{location} shape changed: {expected.shape!r} != {observed.shape!r}"
            )
        expected_geometry = read_tiff_geometry(expected_path, expected.shape)
        observed_geometry = read_tiff_geometry(observed_path, observed.shape)
        if expected_geometry is None or observed_geometry is None:
            raise HpcWorkflowAcceptanceError(
                f"{location} lacks strict TIFF geometry metadata"
            )
        if expected_geometry != observed_geometry:
            raise HpcWorkflowAcceptanceError(f"{location} geometry changed")
        return {
            "reference": str(expected_path),
            "candidate": str(observed_path),
            "dtype": str(expected.dtype),
            "shape": list(expected.shape),
            "geometry": expected_geometry.to_dict(),
            "numeric": numeric_comparison_evidence(
                expected,
                observed,
                location=location,
                require_close=require_close,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            ),
        }

    rounds: list[dict[str, object]] = []
    for expected_task, observed_task in zip(
        reference.tasks,
        candidate.tasks,
        strict=True,
    ):
        if (
            expected_task.round != observed_task.round
            or expected_task.strategy != observed_task.strategy
            or len(expected_task.channels) != len(observed_task.channels)
        ):
            raise HpcWorkflowAcceptanceError(
                "sequential and SLURM round task structure changed"
            )
        round_number = expected_task.round
        expected_paths = RegistrationPaths.for_round(
            expected_task.registration_directory,
            round_number,
        )
        observed_paths = RegistrationPaths.for_round(
            observed_task.registration_directory,
            round_number,
        )
        expected_affine = cast(Any, import_affine_transform(expected_paths.affine))
        observed_affine = cast(Any, import_affine_transform(observed_paths.affine))
        for attribute in ("transform_type", "dimension"):
            if getattr(expected_affine, attribute) != getattr(
                observed_affine, attribute
            ):
                raise HpcWorkflowAcceptanceError(
                    f"round {round_number} affine {attribute} changed"
                )
        expected_fixed = np.asarray(expected_affine.fixed_parameters)
        observed_fixed = np.asarray(observed_affine.fixed_parameters)
        np.testing.assert_array_equal(observed_fixed, expected_fixed)

        import ants

        expected_warp = ants.image_read(str(expected_paths.warp))
        observed_warp = ants.image_read(str(observed_paths.warp))
        warp_geometry = {
            "shape": list(expected_warp.shape),
            "origin": list(expected_warp.origin),
            "spacing": list(expected_warp.spacing),
            "direction": np.asarray(expected_warp.direction).tolist(),
            "has_components": bool(expected_warp.has_components),
        }
        observed_warp_geometry = {
            "shape": list(observed_warp.shape),
            "origin": list(observed_warp.origin),
            "spacing": list(observed_warp.spacing),
            "direction": np.asarray(observed_warp.direction).tolist(),
            "has_components": bool(observed_warp.has_components),
        }
        if observed_warp_geometry != warp_geometry:
            raise HpcWorkflowAcceptanceError(
                f"round {round_number} warp geometry changed"
            )

        stage_metrics: dict[str, dict[str, object]] = {}
        metric_paths = (
            (
                "pre-registration",
                expected_paths.pre_metrics,
                observed_paths.pre_metrics,
            ),
            (
                "linearly-registered",
                expected_paths.linear_metrics,
                observed_paths.linear_metrics,
            ),
            (
                "nonlinear-registered",
                expected_paths.nonlinear_metrics,
                observed_paths.nonlinear_metrics,
            ),
        )
        reference_metrics: dict[str, dict[str, float]] = {}
        candidate_metrics: dict[str, dict[str, float]] = {}
        for stage, reference_path, candidate_path in metric_paths:
            reference_stage = read_metrics(reference_path)
            candidate_stage = read_metrics(candidate_path)
            reference_metrics[stage] = reference_stage
            candidate_metrics[stage] = candidate_stage
            stage_metrics[stage] = metric_comparison_evidence(
                reference_stage,
                candidate_stage,
                location=f"round {round_number} {stage} metrics",
                absolute_tolerance=metric_absolute_tolerance,
            )
        for execution_name, values in (
            ("reference", reference_metrics),
            ("candidate", candidate_metrics),
        ):
            for metric_name in ("Correlation", "MattesMutualInformation"):
                if (
                    values["nonlinear-registered"][metric_name]
                    < values["pre-registration"][metric_name]
                ):
                    raise HpcWorkflowAcceptanceError(
                        f"round {round_number} {execution_name} nonlinear "
                        f"{metric_name} degraded below pre-registration"
                    )

        channels: list[dict[str, object]] = []
        for index, (expected_channel, observed_channel) in enumerate(
            zip(expected_task.channels, observed_task.channels, strict=True)
        ):
            if (
                expected_channel.source != observed_channel.source
                or expected_channel.interpolation != observed_channel.interpolation
            ):
                raise HpcWorkflowAcceptanceError(
                    f"round {round_number} channel {index} contract changed"
                )
            channels.append(
                {
                    "source": str(expected_channel.source),
                    "interpolation": expected_channel.interpolation,
                    "comparison": tiff_evidence(
                        expected_channel.output,
                        observed_channel.output,
                        location=f"round {round_number} channel {index}",
                    ),
                }
            )
        rounds.append(
            {
                "round": round_number,
                "strategy": expected_task.strategy,
                "affine": {
                    "transform_type": expected_affine.transform_type,
                    "dimension": expected_affine.dimension,
                    "fixed_parameters_exact": True,
                    "parameters": numeric_comparison_evidence(
                        expected_affine.parameters,
                        observed_affine.parameters,
                        location=f"round {round_number} affine parameters",
                        require_close=require_close,
                        rtol=relative_tolerance,
                        atol=absolute_tolerance,
                    ),
                },
                "warp": {
                    "geometry": warp_geometry,
                    "numeric": numeric_comparison_evidence(
                        expected_warp.numpy(),
                        observed_warp.numpy(),
                        location=f"round {round_number} nonlinear warp",
                        require_close=require_close,
                        rtol=relative_tolerance,
                        atol=absolute_tolerance,
                    ),
                },
                "registered": tiff_evidence(
                    expected_paths.registered,
                    observed_paths.registered,
                    location=f"round {round_number} registered image",
                ),
                "metrics": stage_metrics,
                "channels": channels,
            }
        )
    return {
        "schema_version": 1,
        "scientific_fingerprint": reference.scientific_fingerprint,
        "rtol": relative_tolerance,
        "atol": absolute_tolerance,
        "comparison_policy": (
            "strict-deterministic-numeric-parity"
            if require_close
            else "stochastic-scientific-equivalence"
        ),
        "metric_absolute_tolerance": metric_absolute_tolerance,
        "voxel_differences_are_characterization": not require_close,
        "rounds": rounds,
    }


def scientific_equivalence_evidence(
    reference: WorkflowPlan,
    candidate: WorkflowPlan,
) -> dict[str, object]:
    """Compare independent multithreaded estimates by structure and quality."""
    return _scientific_comparison_evidence(
        reference,
        candidate,
        require_close=False,
        metric_absolute_tolerance=0.01,
    )


def scientific_parity_evidence(
    reference: WorkflowPlan,
    candidate: WorkflowPlan,
) -> dict[str, object]:
    """Require strict numerical parity under deterministic conditions."""
    return _scientific_comparison_evidence(
        reference,
        candidate,
        require_close=True,
        metric_absolute_tolerance=0.0,
    )


def bounded_slurm_profile(
    profile: ExecutionProfile,
    output_root: Path,
) -> ExecutionProfile:
    """Derive an approved SLURM profile bounded to exactly two single-process jobs."""
    spec = profile.slurm
    if profile.executor != "slurm" or spec is None:
        raise HpcWorkflowAcceptanceError("Task 5 requires an approved SLURM profile")
    if spec.workers < 2:
        raise HpcWorkflowAcceptanceError(
            "Task 5 requires an approved profile with at least two workers"
        )
    if spec.processes_per_worker != 1:
        raise HpcWorkflowAcceptanceError(
            "Task 5 requires exactly one process per SLURM worker job"
        )
    return replace(
        profile,
        paths=replace(profile.paths, output_root=output_root),
        slurm=replace(spec, workers=2, task_retries=0),
    )


def sequential_reference_profile(
    profile: ExecutionProfile,
    output_root: Path,
) -> ExecutionProfile:
    """Derive the scientifically identical sequential reference execution profile."""
    return replace(
        profile,
        executor="sequential",
        paths=replace(
            profile.paths,
            output_root=output_root,
            worker_local_directory=None,
        ),
        local_dask=None,
        slurm=None,
    )


@contextmanager
def workflow_module_by_value() -> Iterator[None]:
    """Serialize only this helper by value and restore its exact prior state."""
    import cloudpickle

    global _PICKLE_DEPTH, _PICKLE_WAS_REGISTERED

    module = sys.modules[__name__]
    registry = getattr(cloudpickle.cloudpickle, "_PICKLE_BY_VALUE_MODULES", None)
    if not isinstance(registry, set):
        raise HpcWorkflowAcceptanceError(
            "cloudpickle does not expose registration state needed for exact cleanup"
        )
    with _PICKLE_LOCK:
        if _PICKLE_DEPTH == 0:
            _PICKLE_WAS_REGISTERED = module.__name__ in registry
        if module.__name__ not in registry:
            cloudpickle.register_pickle_by_value(module)
        _PICKLE_DEPTH += 1
    try:
        yield
    finally:
        with _PICKLE_LOCK:
            _PICKLE_DEPTH -= 1
            if _PICKLE_DEPTH < 0:
                _PICKLE_DEPTH = 0
                raise HpcWorkflowAcceptanceError(
                    "cloudpickle registration nesting was corrupted"
                )
            if _PICKLE_DEPTH == 0:
                if _PICKLE_WAS_REGISTERED:
                    if module.__name__ not in registry:
                        cloudpickle.register_pickle_by_value(module)
                elif module.__name__ in registry:
                    cloudpickle.unregister_pickle_by_value(module)
                _PICKLE_WAS_REGISTERED = False


def _owned_job_ids(evidence: object) -> tuple[str, ...]:
    if not isinstance(evidence, Mapping):
        return ()
    observed = evidence.get("observed")
    if not isinstance(observed, Mapping):
        return ()
    probes = observed.get("worker_probes")
    if not isinstance(probes, list):
        return ()
    return tuple(
        sorted(
            {
                job_id
                for probe in probes
                if isinstance(probe, Mapping)
                and type(job_id := probe.get("slurm_job_id")) is str
                and job_id
            }
        )
    )


def _error_payload(error: BaseException) -> dict[str, str]:
    return {"exception_type": type(error).__name__, "message": str(error)}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_marker(path: Path, value: Mapping[str, object]) -> None:
    import json

    encoded = json.dumps(value, allow_nan=False, sort_keys=True) + "\n"

    def writer(temporary: Path) -> None:
        temporary.write_text(encoded, encoding="utf-8")

    def validator(temporary: Path) -> bool:
        try:
            return temporary.read_text(encoding="utf-8") == encoded
        except (OSError, UnicodeError):
            return False

    atomic_write(path, writer, validator)


def _wait_for_committed_marker(
    path: Path,
    *,
    expected_round: int,
    timeout_seconds: float,
    _monotonic: Callable[[], float],
    _sleep: Callable[[float], None],
) -> dict[str, object]:
    import json

    if timeout_seconds <= 0.0:
        raise HpcWorkflowAcceptanceError("marker timeout must be positive and bounded")
    deadline = _monotonic() + timeout_seconds
    while True:
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise HpcWorkflowAcceptanceError(
                    f"committed marker is malformed: {path}: {error}"
                ) from error
            expected_keys = {
                "schema_version",
                "round",
                "status",
                "result_ended_at",
                "marker_written_at",
            }
            if not isinstance(payload, dict) or set(payload) != expected_keys:
                raise HpcWorkflowAcceptanceError(
                    f"committed marker has invalid schema: {payload!r}"
                )
            if (
                payload["schema_version"] != 1
                or payload["round"] != expected_round
                or payload["status"] != "completed"
                or type(payload["result_ended_at"]) is not str
                or type(payload["marker_written_at"]) is not str
            ):
                raise HpcWorkflowAcceptanceError(
                    "committed marker does not prove round "
                    f"{expected_round}: {payload!r}"
                )
            if _timestamp(payload["result_ended_at"]) > _timestamp(
                payload["marker_written_at"]
            ):
                raise HpcWorkflowAcceptanceError(
                    "committed marker timestamp precedes the completed round"
                )
            return cast(dict[str, object], payload)
        now = _monotonic()
        if now >= deadline:
            raise HpcWorkflowAcceptanceError(
                f"timed out waiting for committed round {expected_round}: {path}"
            )
        _sleep(min(0.25, max(0.0, deadline - now)))


def _production_round_task(task: RoundTask) -> RoundResult:
    from cyexm.workflow.round_task import run_round_task

    return run_round_task(task)


def _production_workflow_task(task: WorkflowTask) -> RoundResult:
    from cyexm.workflow.task import run_workflow_task

    return run_workflow_task(task)


def _commit_round_marker(
    path: Path,
    result: RoundResult,
    expected_round: int,
    *,
    utc_now: Callable[[], str],
) -> None:
    if result.round != expected_round or result.status != "completed":
        raise HpcWorkflowAcceptanceError(
            f"round {expected_round} did not commit before coordination: "
            f"round={result.round}, status={result.status!r}"
        )
    marker_written_at = utc_now()
    if _timestamp(result.ended_at) > _timestamp(marker_written_at):
        raise HpcWorkflowAcceptanceError(
            "commit marker timestamp precedes the completed round"
        )
    _write_marker(
        path,
        {
            "schema_version": 1,
            "round": expected_round,
            "status": "completed",
            "result_ended_at": result.ended_at,
            "marker_written_at": marker_written_at,
        },
    )


def validate_injected_failure_record(
    record: Mapping[str, object],
    *,
    successful_round: int,
    failing_round: int,
    committed_marker: Path,
) -> dict[str, object]:
    """Require the exact post-commit injected failure, not a coordination timeout."""
    rounds = record.get("rounds")
    if record.get("status") != "failed" or not isinstance(rounds, list):
        raise HpcWorkflowAcceptanceError(
            "injected failure run must have failed workflow status and round results"
        )
    by_round = {item.get("round"): item for item in rounds if isinstance(item, Mapping)}
    successful = by_round.get(successful_round)
    failed = by_round.get(failing_round)
    if not isinstance(successful, Mapping) or not isinstance(failed, Mapping):
        raise HpcWorkflowAcceptanceError(
            "injected failure run is missing the selected round results"
        )
    if successful.get("status") != "completed" or successful.get("failure") is not None:
        raise HpcWorkflowAcceptanceError(
            "peer round did not complete before the injected failure"
        )
    if failed.get("status") != "failed":
        raise HpcWorkflowAcceptanceError("selected injected round did not fail")

    marker = _wait_for_committed_marker(
        committed_marker,
        expected_round=successful_round,
        timeout_seconds=1.0,
        _monotonic=time.monotonic,
        _sleep=time.sleep,
    )
    if successful.get("ended_at") != marker["result_ended_at"]:
        raise HpcWorkflowAcceptanceError(
            "committed marker does not identify the completed peer result"
        )
    expected_failure = {
        "stage": "round",
        "exception_type": "RuntimeError",
        "message": (
            f"injected Task 5 failure for round {failing_round} after committed "
            f"round {successful_round} at {marker['marker_written_at']}"
        ),
    }
    if failed.get("failure") != expected_failure:
        raise HpcWorkflowAcceptanceError(
            "failed round does not identify the intended injected RuntimeError "
            f"after peer commit: {failed.get('failure')!r}"
        )
    return {
        "schema_version": 1,
        "successful_round": successful_round,
        "failing_round": failing_round,
        "committed_marker": marker,
        "failure": expected_failure,
        "ordering_proven": True,
    }


def run_failure_task(
    task: WorkflowTask,
    *,
    successful_round: int,
    failing_round: int,
    committed_marker: Path,
    timeout_seconds: float,
    _run_round_task: Callable[[RoundTask], RoundResult] = _production_round_task,
    _run_workflow_task: Callable[
        [WorkflowTask], RoundResult
    ] = _production_workflow_task,
    _monotonic: Callable[[], float] = time.monotonic,
    _sleep: Callable[[float], None] = time.sleep,
    _utc_now: Callable[[], str] = _utc_now,
) -> RoundResult:
    """Commit one real round before deterministically failing its independent peer."""
    if type(task) is ReferenceTask:
        return _run_workflow_task(task)
    if task.round == successful_round:
        result = _run_round_task(task)
        _commit_round_marker(
            committed_marker,
            result,
            successful_round,
            utc_now=_utc_now,
        )
        return result
    if task.round == failing_round:
        marker = _wait_for_committed_marker(
            committed_marker,
            expected_round=successful_round,
            timeout_seconds=timeout_seconds,
            _monotonic=_monotonic,
            _sleep=_sleep,
        )
        raise RuntimeError(
            f"injected Task 5 failure for round {failing_round} after committed "
            f"round {successful_round} at {marker['marker_written_at']}"
        )
    raise HpcWorkflowAcceptanceError(
        f"unexpected Task 5 failure task round {task.round}"
    )


def run_cancellation_task(
    task: WorkflowTask,
    *,
    successful_round: int,
    blocked_round: int,
    committed_marker: Path,
    ready_marker: Path,
    timeout_seconds: float,
    _run_round_task: Callable[[RoundTask], RoundResult] = _production_round_task,
    _run_workflow_task: Callable[
        [WorkflowTask], RoundResult
    ] = _production_workflow_task,
    _monotonic: Callable[[], float] = time.monotonic,
    _sleep: Callable[[float], None] = time.sleep,
    _utc_now: Callable[[], str] = _utc_now,
) -> RoundResult:
    """Commit one real round while its peer stays cancellably unfinished."""
    if type(task) is ReferenceTask:
        return _run_workflow_task(task)
    if task.round == successful_round:
        result = _run_round_task(task)
        _commit_round_marker(
            committed_marker,
            result,
            successful_round,
            utc_now=_utc_now,
        )
        return result
    if task.round != blocked_round:
        raise HpcWorkflowAcceptanceError(
            f"unexpected Task 5 cancellation task round {task.round}"
        )
    _wait_for_committed_marker(
        committed_marker,
        expected_round=successful_round,
        timeout_seconds=timeout_seconds,
        _monotonic=_monotonic,
        _sleep=_sleep,
    )
    _write_marker(
        ready_marker,
        {
            "blocked_round": blocked_round,
            "committed_round": successful_round,
            "status": "waiting-for-coordinator-interruption",
        },
    )
    deadline = _monotonic() + timeout_seconds
    while True:
        now = _monotonic()
        if now >= deadline:
            raise HpcWorkflowAcceptanceError(
                f"cancellation task round {blocked_round} exceeded its bounded wait"
            )
        _sleep(min(0.25, max(0.0, deadline - now)))


def execute_cancellation_acceptance(
    plan: Any,
    *,
    runner: Callable[[Any], Any],
    evidence_dir: Path,
    operation: str,
    user: str,
    ready_marker: Path,
    successful_round: int,
    blocked_round: int,
    timeout_seconds: float,
    _execute_owned: Callable[..., Any] | None = None,
    _send_signal: Callable[[int, int], None] = os.kill,
    _monotonic: Callable[[], float] = time.monotonic,
    _sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Interrupt the coordinator only after a committed peer and blocked future."""
    if timeout_seconds <= 0.0:
        raise HpcWorkflowAcceptanceError(
            "cancellation timeout must be positive and bounded"
        )
    stopped = threading.Event()
    trigger: dict[str, object] = {
        "blocked_round": blocked_round,
        "committed_round": successful_round,
        "signal": "SIGINT",
        "status": "waiting",
        "timeout_seconds": timeout_seconds,
    }

    def trigger_when_ready() -> None:
        deadline = _monotonic() + timeout_seconds
        while not stopped.is_set():
            if ready_marker.is_file():
                try:
                    payload = json.loads(ready_marker.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    trigger["status"] = "failed"
                    trigger["error"] = _error_payload(error)
                    return
                expected = {
                    "blocked_round": blocked_round,
                    "committed_round": successful_round,
                    "status": "waiting-for-coordinator-interruption",
                }
                if payload != expected:
                    trigger["status"] = "failed"
                    trigger["error"] = {
                        "exception_type": "HpcWorkflowAcceptanceError",
                        "message": f"unexpected cancellation marker: {payload!r}",
                    }
                    return
                trigger["status"] = "signalled"
                try:
                    _send_signal(os.getpid(), signal.SIGINT)
                except BaseException as error:
                    trigger["status"] = "failed"
                    trigger["error"] = _error_payload(error)
                return
            if _monotonic() >= deadline:
                trigger["status"] = "failed"
                trigger["error"] = {
                    "exception_type": "HpcWorkflowAcceptanceError",
                    "message": "timed out waiting for cancellation-ready marker",
                }
                return
            _sleep(0.1)

    def interrupt_handler(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt("Task 5 coordinator interruption")

    previous_handler = signal.getsignal(signal.SIGINT)
    trigger_thread = threading.Thread(
        target=trigger_when_ready,
        name="cyexm-task5-cancellation-trigger",
        daemon=True,
    )
    execution_error: BaseException | None = None
    exit_code: int | None = None
    owned_executor = _execute_owned or execute_owned_slurm_workflow
    signal.signal(signal.SIGINT, interrupt_handler)
    trigger_thread.start()
    try:
        try:
            owned_executor(
                plan,
                runner=runner,
                evidence_dir=evidence_dir,
                operation=operation,
                user=user,
            )
        except KeyboardInterrupt:
            exit_code = 130
        except BaseException as error:
            execution_error = error
    finally:
        stopped.set()
        trigger_thread.join(timeout=min(5.0, timeout_seconds))
        signal.signal(signal.SIGINT, previous_handler)

    if trigger_thread.is_alive():
        trigger["status"] = "failed"
        trigger["error"] = {
            "exception_type": "HpcWorkflowAcceptanceError",
            "message": "cancellation trigger thread did not stop",
        }
    elif trigger.get("status") == "signalled" and exit_code == 130:
        trigger["status"] = "interrupted"
    trigger["exit_code"] = exit_code
    write_evidence_json(evidence_dir / f"{operation}-trigger.json", trigger)

    if trigger.get("status") != "interrupted" or exit_code != 130:
        raise HpcWorkflowAcceptanceError(
            "coordinator cancellation did not complete with exit semantics 130"
        ) from execution_error
    if execution_error is not None:
        raise execution_error
    return exit_code


def execute_owned_slurm_workflow(
    plan: Any,
    *,
    runner: Callable[[Any], Any],
    evidence_dir: Path,
    operation: str,
    user: str,
    _executor_ctor: Callable[..., Any] = SlurmExecutor,
    _run_workflow: Callable[..., Any] = run_workflow,
    _collect_queue: Callable[..., dict[str, object]] = collect_queue_snapshot,
    _wait_for_queue: Callable[..., dict[str, object]] = wait_for_queue_empty,
    _pickle_context: Callable[
        [], AbstractContextManager[Any]
    ] = workflow_module_by_value,
) -> Any:
    """Run one owned workflow and durably prove current-user named-job cleanup."""
    if not operation or not operation.replace("-", "").isalnum():
        raise HpcWorkflowAcceptanceError(
            f"operation must be a nonempty filesystem-safe label; got {operation!r}"
        )
    spec = plan.execution.slurm
    if plan.execution.executor != "slurm" or spec is None:
        raise HpcWorkflowAcceptanceError("owned workflow requires a SLURM plan")

    before = _collect_queue(spec.job_name, user=user)
    write_evidence_json(evidence_dir / f"{operation}-queue-before.json", before)
    if before.get("jobs") != []:
        raise HpcWorkflowAcceptanceError(
            "refusing Task 5 submission while current-user jobs already use "
            f"the approved harness name {spec.job_name!r}"
        )

    executor = _executor_ctor(plan.execution, _runner=runner)
    result: Any = None
    execution_error: BaseException | None = None
    try:
        with _pickle_context():
            result = _run_workflow(
                plan,
                _executor_factory=lambda _profile: executor,
            )
    except BaseException as error:
        execution_error = error

    try:
        execution_evidence: object = executor.execution_evidence
    except BaseException as evidence_error:
        execution_evidence = {"evidence_error": _error_payload(evidence_error)}
    owned_ids = _owned_job_ids(execution_evidence)
    cleanup = _wait_for_queue(
        spec.job_name,
        user=user,
        job_ids=owned_ids,
    )
    write_evidence_json(
        evidence_dir / f"{operation}-slurm-execution.json",
        execution_evidence,
    )
    write_evidence_json(evidence_dir / f"{operation}-queue-after.json", cleanup)
    outcome_status = (
        "cancelled"
        if isinstance(execution_error, KeyboardInterrupt)
        else "failed"
        if execution_error is not None
        else "completed"
    )
    outcome: dict[str, object] = {
        "schema_version": 1,
        "operation": operation,
        "status": outcome_status,
        "owned_job_ids": list(owned_ids),
        "cleanup_outcome": cleanup.get("outcome"),
        "workflow_result": (result.to_dict() if hasattr(result, "to_dict") else None),
    }
    if execution_error is not None:
        outcome["error"] = _error_payload(execution_error)
    write_evidence_json(evidence_dir / f"{operation}-outcome.json", outcome)

    if cleanup.get("outcome") != "empty":
        raise HpcWorkflowAcceptanceError(
            f"owned SLURM cleanup failed for {operation}; inspect queue evidence"
        ) from execution_error
    if execution_error is not None:
        raise execution_error
    return result


__all__ = [
    "HpcWorkflowAcceptanceError",
    "Task5OutputRoots",
    "bounded_acceptance_manifest",
    "bounded_slurm_profile",
    "coordination_timeout_seconds",
    "execute_cancellation_acceptance",
    "execute_owned_slurm_workflow",
    "metric_comparison_evidence",
    "numeric_comparison_evidence",
    "parallel_placement_evidence",
    "prepare_task5_output_roots",
    "run_cancellation_task",
    "run_failure_task",
    "scientific_equivalence_evidence",
    "scientific_parity_evidence",
    "sequential_reference_profile",
    "task5_output_roots",
    "validate_injected_failure_record",
    "validate_task5_output_ownership",
    "workflow_module_by_value",
]
