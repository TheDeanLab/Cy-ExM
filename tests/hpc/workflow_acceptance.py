"""Bounded orchestration helpers for real-SLURM workflow acceptance."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

from cyexm.config.models import (
    ChannelSpec,
    DatasetManifest,
    ExecutionProfile,
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    ReferenceChannelSpec,
    ReferenceSpec,
    RegistrationSpec,
    RegistrationStageSpec,
    RoundSpec,
)
from cyexm.execution.slurm import SlurmExecutor
from cyexm.io.atomic import atomic_write
from cyexm.registration.artifacts import FileIdentity
from cyexm.runtime.threads import BLAS_THREAD_ENV_VARS, ITK_THREAD_ENV_VAR
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
    deterministic_inputs: Path
    deterministic_parallel: Path
    deterministic_sequential: Path

    @property
    def scenario_roots(self) -> tuple[Path, ...]:
        return (
            self.parallel,
            self.sequential,
            self.failure,
            self.cancellation,
            self.deterministic_inputs,
            self.deterministic_parallel,
            self.deterministic_sequential,
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
        deterministic_inputs=base / "deterministic-inputs",
        deterministic_parallel=base / "deterministic-parallel",
        deterministic_sequential=base / "deterministic-sequential",
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
        roots.base / "deterministic-inputs",
        roots.base / "deterministic-parallel",
        roots.base / "deterministic-sequential",
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

    relative_tolerance = 0.0 if require_close else 1e-4
    absolute_tolerance = 0.0 if require_close else 1e-5
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


def _validated_scientific_comparison_evidence(
    value: object,
    *,
    require_close: bool,
    metric_absolute_tolerance: float,
) -> dict[str, object]:
    """Strictly decode the complete scientific-comparison evidence schema."""
    import math

    from cyexm.spatial import ImageGeometry

    def exact_object(node: object, keys: set[str], location: str) -> dict[str, object]:
        if not isinstance(node, dict) or set(node) != keys:
            raise HpcWorkflowAcceptanceError(
                f"scientific comparison {location} has an invalid exact schema"
            )
        return cast(dict[str, object], node)

    def finite_number(node: object, location: str) -> float:
        if isinstance(node, bool) or not isinstance(node, int | float):
            raise HpcWorkflowAcceptanceError(
                f"scientific comparison {location} is not a finite number"
            )
        number = float(node)
        if not math.isfinite(number):
            raise HpcWorkflowAcceptanceError(
                f"scientific comparison {location} is not finite"
            )
        return number

    def numeric(
        node: object, location: str, *, expected_shape: object | None = None
    ) -> dict[str, object]:
        payload = exact_object(
            node,
            {
                "shape",
                "reference_dtype",
                "candidate_dtype",
                "array_equal",
                "close_required",
                "maximum_absolute_difference",
                "mean_absolute_difference",
                "p99_9_absolute_difference",
                "rtol",
                "atol",
            },
            location,
        )
        shape = payload["shape"]
        if (
            not isinstance(shape, list)
            or any(type(item) is not int or item < 1 for item in shape)
            or (expected_shape is not None and shape != expected_shape)
            or not isinstance(payload["reference_dtype"], str)
            or not payload["reference_dtype"]
            or payload["reference_dtype"] != payload["candidate_dtype"]
            or type(payload["array_equal"]) is not bool
            or payload["close_required"] is not require_close
        ):
            raise HpcWorkflowAcceptanceError(
                f"scientific comparison {location} shape/dtype contract changed"
            )
        for name in (
            "maximum_absolute_difference",
            "mean_absolute_difference",
            "p99_9_absolute_difference",
        ):
            if finite_number(payload[name], f"{location}.{name}") < 0.0:
                raise HpcWorkflowAcceptanceError(
                    f"scientific comparison {location}.{name} is negative"
                )
        if require_close:
            if (
                payload["rtol"] != 0.0
                or payload["atol"] != 0.0
                or payload["array_equal"] is not True
                or payload["maximum_absolute_difference"] != 0.0
            ):
                raise HpcWorkflowAcceptanceError(
                    f"scientific comparison {location} is not exactly equal"
                )
        elif payload["rtol"] != 1e-4 or payload["atol"] != 1e-5:
            raise HpcWorkflowAcceptanceError(
                f"scientific comparison {location} tolerance changed"
            )
        return payload

    def geometry(node: object, location: str) -> dict[str, object]:
        payload = exact_object(
            node, {"shape", "origin", "spacing", "direction"}, location
        )
        try:
            ImageGeometry.from_dict(payload)
        except (TypeError, ValueError) as error:
            raise HpcWorkflowAcceptanceError(
                f"scientific comparison {location} is malformed"
            ) from error
        return payload

    def tiff(node: object, location: str) -> None:
        payload = exact_object(
            node,
            {"reference", "candidate", "dtype", "shape", "geometry", "numeric"},
            location,
        )
        if (
            not isinstance(payload["reference"], str)
            or not payload["reference"]
            or not isinstance(payload["candidate"], str)
            or not payload["candidate"]
            or not isinstance(payload["dtype"], str)
            or not payload["dtype"]
        ):
            raise HpcWorkflowAcceptanceError(
                f"scientific comparison {location} artifact identity is malformed"
            )
        image_geometry = geometry(payload["geometry"], f"{location}.geometry")
        if payload["shape"] != image_geometry["shape"]:
            raise HpcWorkflowAcceptanceError(
                f"scientific comparison {location} shape differs from geometry"
            )
        comparison = numeric(
            payload["numeric"], f"{location}.numeric", expected_shape=payload["shape"]
        )
        if comparison["reference_dtype"] != payload["dtype"]:
            raise HpcWorkflowAcceptanceError(
                f"scientific comparison {location} dtype differs from numeric data"
            )

    expected_policy = (
        "strict-deterministic-numeric-parity"
        if require_close
        else "stochastic-scientific-equivalence"
    )
    payload = exact_object(
        value,
        {
            "schema_version",
            "scientific_fingerprint",
            "rtol",
            "atol",
            "comparison_policy",
            "metric_absolute_tolerance",
            "voxel_differences_are_characterization",
            "rounds",
        },
        "root",
    )
    if (
        payload["schema_version"] != 1
        or not isinstance(payload["scientific_fingerprint"], str)
        or not payload["scientific_fingerprint"]
        or payload["comparison_policy"] != expected_policy
        or payload["rtol"] != (0.0 if require_close else 1e-4)
        or payload["atol"] != (0.0 if require_close else 1e-5)
        or payload["metric_absolute_tolerance"] != metric_absolute_tolerance
        or payload["voxel_differences_are_characterization"] is not (not require_close)
    ):
        raise HpcWorkflowAcceptanceError(
            "scientific comparison policy or tolerance contract changed"
        )
    rounds = payload["rounds"]
    if not isinstance(rounds, list) or not rounds:
        raise HpcWorkflowAcceptanceError(
            "scientific comparison contains no round evidence"
        )
    observed_rounds: list[int] = []
    for round_index, raw_round in enumerate(rounds):
        location = f"rounds[{round_index}]"
        round_payload = exact_object(
            raw_round,
            {
                "round",
                "strategy",
                "affine",
                "warp",
                "registered",
                "metrics",
                "channels",
            },
            location,
        )
        round_number = round_payload["round"]
        if (
            type(round_number) is not int
            or round_number < 1
            or round_payload["strategy"] not in {"whole_image", "serial_chunked"}
        ):
            raise HpcWorkflowAcceptanceError(
                f"scientific comparison {location} identity is malformed"
            )
        observed_rounds.append(round_number)
        affine = exact_object(
            round_payload["affine"],
            {"transform_type", "dimension", "fixed_parameters_exact", "parameters"},
            f"{location}.affine",
        )
        if (
            not isinstance(affine["transform_type"], str)
            or not affine["transform_type"]
            or affine["dimension"] != 3
            or affine["fixed_parameters_exact"] is not True
        ):
            raise HpcWorkflowAcceptanceError(
                f"scientific comparison {location} affine metadata changed"
            )
        numeric(affine["parameters"], f"{location}.affine.parameters")
        warp = exact_object(
            round_payload["warp"], {"geometry", "numeric"}, f"{location}.warp"
        )
        warp_geometry = exact_object(
            warp["geometry"],
            {"shape", "origin", "spacing", "direction", "has_components"},
            f"{location}.warp.geometry",
        )
        if (
            not isinstance(warp_geometry["shape"], list)
            or any(
                type(item) is not int or item < 1
                for item in cast(list[object], warp_geometry["shape"])
            )
            or warp_geometry["has_components"] is not True
        ):
            raise HpcWorkflowAcceptanceError(
                f"scientific comparison {location} warp geometry changed"
            )
        geometry(
            {
                name: warp_geometry[name]
                for name in ("shape", "origin", "spacing", "direction")
            },
            f"{location}.warp.geometry",
        )
        numeric(warp["numeric"], f"{location}.warp.numeric")
        tiff(round_payload["registered"], f"{location}.registered")
        metrics = exact_object(
            round_payload["metrics"],
            {"pre-registration", "linearly-registered", "nonlinear-registered"},
            f"{location}.metrics",
        )
        stage_values: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
        for stage, raw_metric in metrics.items():
            metric = exact_object(
                raw_metric,
                {
                    "reference",
                    "candidate",
                    "absolute_differences",
                    "absolute_tolerance",
                },
                f"{location}.metrics.{stage}",
            )
            reference = exact_object(
                metric["reference"],
                {"Correlation", "MattesMutualInformation"},
                f"{location}.metrics.{stage}.reference",
            )
            candidate = exact_object(
                metric["candidate"],
                {"Correlation", "MattesMutualInformation"},
                f"{location}.metrics.{stage}.candidate",
            )
            differences = exact_object(
                metric["absolute_differences"],
                {"Correlation", "MattesMutualInformation"},
                f"{location}.metrics.{stage}.absolute_differences",
            )
            if metric["absolute_tolerance"] != metric_absolute_tolerance:
                raise HpcWorkflowAcceptanceError(
                    f"scientific comparison {location} metric tolerance changed"
                )
            reference_numbers: dict[str, float] = {}
            candidate_numbers: dict[str, float] = {}
            for name in ("Correlation", "MattesMutualInformation"):
                reference_numbers[name] = finite_number(
                    reference[name], f"{location}.{stage}.reference.{name}"
                )
                candidate_numbers[name] = finite_number(
                    candidate[name], f"{location}.{stage}.candidate.{name}"
                )
                observed_difference = finite_number(
                    differences[name], f"{location}.{stage}.difference.{name}"
                )
                if (
                    abs(
                        observed_difference
                        - abs(candidate_numbers[name] - reference_numbers[name])
                    )
                    > 1e-12
                    or observed_difference > metric_absolute_tolerance
                ):
                    raise HpcWorkflowAcceptanceError(
                        f"scientific comparison {location} metric evidence is forged"
                    )
            stage_values[stage] = (reference_numbers, candidate_numbers)
        for execution_index in (0, 1):
            for name in ("Correlation", "MattesMutualInformation"):
                if (
                    stage_values["nonlinear-registered"][execution_index][name]
                    < stage_values["pre-registration"][execution_index][name]
                ):
                    raise HpcWorkflowAcceptanceError(
                        f"scientific comparison {location} nonlinear metric degraded"
                    )
        channels = round_payload["channels"]
        if not isinstance(channels, list) or not channels:
            raise HpcWorkflowAcceptanceError(
                f"scientific comparison {location} contains no channels"
            )
        for channel_index, raw_channel in enumerate(channels):
            channel = exact_object(
                raw_channel,
                {"source", "interpolation", "comparison"},
                f"{location}.channels[{channel_index}]",
            )
            if (
                not isinstance(channel["source"], str)
                or not channel["source"]
                or channel["interpolation"] not in {"linear", "nearestNeighbor"}
            ):
                raise HpcWorkflowAcceptanceError(
                    f"scientific comparison {location} channel contract changed"
                )
            tiff(
                channel["comparison"],
                f"{location}.channels[{channel_index}].comparison",
            )
    if observed_rounds != sorted(set(observed_rounds)):
        raise HpcWorkflowAcceptanceError(
            "scientific comparison rounds are duplicate or noncanonical"
        )
    return payload


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


def deterministic_slurm_profile(
    profile: ExecutionProfile,
    output_root: Path,
) -> ExecutionProfile:
    """Derive the bounded one-thread SLURM profile for strict parity."""
    spec = profile.slurm
    if profile.executor != "slurm" or spec is None:
        raise HpcWorkflowAcceptanceError(
            "deterministic parity requires an approved SLURM profile"
        )
    if spec.workers < 2 or spec.processes_per_worker != 1:
        raise HpcWorkflowAcceptanceError(
            "deterministic parity requires two single-process workers"
        )
    job_name = f"{spec.job_name}-det"
    if "\n" in job_name or "\r" in job_name or len(job_name) > 128:
        raise HpcWorkflowAcceptanceError(
            "deterministic parity job name must be one line and at most 128 characters"
        )
    return replace(
        profile,
        paths=replace(profile.paths, output_root=output_root),
        native_threads=NativeThreadSpec(itk=1, blas=1),
        slurm=replace(
            spec,
            workers=2,
            task_retries=0,
            job_name=job_name,
        ),
    )


def deterministic_sequential_profile(
    profile: ExecutionProfile,
    output_root: Path,
) -> ExecutionProfile:
    """Derive the one-thread sequential profile for strict parity."""
    return replace(
        profile,
        executor="sequential",
        paths=replace(
            profile.paths,
            output_root=output_root,
            worker_local_directory=None,
        ),
        native_threads=NativeThreadSpec(itk=1, blas=1),
        local_dask=None,
        slurm=None,
    )


def deterministic_acceptance_manifest(input_root: Path) -> DatasetManifest:
    """Build the bounded synthetic manifest used only by strict parity."""
    if not input_root.is_dir() or input_root.is_symlink():
        raise HpcWorkflowAcceptanceError(
            "deterministic input root must be an existing real directory"
        )
    return DatasetManifest(
        schema_version=1,
        dataset_id="phase2e-deterministic-parity",
        voxel_spacing_um=(0.4, 0.4, 1.0),
        reference=ReferenceSpec(
            round=1,
            registration_channel=PurePosixPath("fixed.tif"),
            channels=(ReferenceChannelSpec(PurePosixPath("fixed.tif")),),
        ),
        registration=RegistrationSpec(
            crop=False,
            linear=RegistrationStageSpec(type="TRSAA", accuracy="dry run"),
            nonlinear=NonlinearRegistrationStageSpec(
                type="SyNOnly",
                accuracy="dry run",
                metric="mattes",
                radius=None,
            ),
        ),
        rounds=(
            RoundSpec(
                round=2,
                strategy="whole_image",
                registration_channel=PurePosixPath("moving-whole.tif"),
                channels=(ChannelSpec(PurePosixPath("moving-whole.tif"), "linear"),),
            ),
            RoundSpec(
                round=3,
                strategy="serial_chunked",
                registration_channel=PurePosixPath("moving-chunked.tif"),
                channels=(ChannelSpec(PurePosixPath("moving-chunked.tif"), "linear"),),
                chunk_size_zyx=(8, 16, 16),
                overlap_fraction=0.25,
            ),
        ),
    )


def write_deterministic_inputs(input_root: Path) -> dict[str, object]:
    """Write three small deterministic scalar TIFFs into one owned root."""
    if input_root.is_symlink() or not input_root.is_dir():
        raise HpcWorkflowAcceptanceError(
            "deterministic input root must be an existing empty real directory"
        )
    if any(input_root.iterdir()):
        raise HpcWorkflowAcceptanceError(
            "deterministic input root must be an existing empty real directory"
        )

    import numpy as np
    import tifffile

    z, y, x = np.mgrid[:16, :32, :32]
    fixed = (
        45_000.0
        * np.exp(
            -(
                ((z - 7.0) / 2.8) ** 2
                + ((y - 15.0) / 5.5) ** 2
                + ((x - 17.0) / 6.0) ** 2
            )
        )
    ).astype(np.float32)
    fixed += (
        18_000.0
        * np.exp(
            -(
                ((z - 10.0) / 2.0) ** 2
                + ((y - 22.0) / 3.5) ** 2
                + ((x - 9.0) / 4.0) ** 2
            )
        )
    ).astype(np.float32)
    texture = (
        np.random.default_rng(20260722)
        .normal(
            loc=0.0,
            scale=25.0,
            size=fixed.shape,
        )
        .astype(np.float32)
    )
    fixed = np.maximum(fixed + texture, 0.0).astype(np.float32)
    arrays = {
        "fixed.tif": fixed,
        "moving-whole.tif": np.roll(fixed, shift=(0, 1, -1), axis=(0, 1, 2)),
        "moving-chunked.tif": np.roll(fixed, shift=(1, -1, 1), axis=(0, 1, 2)),
    }
    identities: dict[str, object] = {}
    for name, array in arrays.items():
        destination = input_root / name

        def writer(temporary: Path, *, value: Any = array) -> None:
            tifffile.imwrite(
                temporary,
                value,
                metadata={"axes": "ZYX"},
                photometric="minisblack",
            )

        def validator(temporary: Path, *, value: Any = array) -> bool:
            try:
                observed = tifffile.imread(temporary)
            except Exception:
                return False
            return bool(
                observed.dtype == np.dtype(np.float32)
                and observed.shape == (16, 32, 32)
                and np.array_equal(observed, value)
            )

        atomic_write(destination, writer, validator)
        identities[name] = FileIdentity.from_path(destination).to_dict()
    return {
        "schema_version": 1,
        "shape_zyx": [16, 32, 32],
        "dtype": "float32",
        "files": identities,
    }


def _write_text_evidence(path: Path, value: str) -> None:
    def writer(temporary: Path) -> None:
        temporary.write_text(value, encoding="utf-8")

    def validator(temporary: Path) -> bool:
        try:
            return temporary.read_text(encoding="utf-8") == value
        except (OSError, UnicodeError):
            return False

    atomic_write(path, writer, validator)


def _validated_deterministic_parity_evidence(
    value: object,
    *,
    require_fine_internal: bool = False,
) -> dict[str, object]:
    expected_keys = {
        "schema_version",
        "status",
        "seed",
        "native_threads",
        "inputs",
        "parallel",
        "sequential",
        "strict_parity",
        "final_queue",
    }
    if require_fine_internal:
        expected_keys.add("strict_internal_parity")
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise HpcWorkflowAcceptanceError(
            "deterministic parity evidence has an invalid exact schema"
        )
    expected_threads = {
        ITK_THREAD_ENV_VAR: 1,
        **{name: 1 for name in BLAS_THREAD_ENV_VARS},
    }
    final_queue = value.get("final_queue")
    parallel = value.get("parallel")
    sequential = value.get("sequential")
    strict_parity = value.get("strict_parity")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "completed"
        or value.get("seed") != "20260722"
        or value.get("native_threads") != expected_threads
        or not isinstance(value.get("inputs"), dict)
        or not isinstance(parallel, dict)
        or parallel.get("status") != "completed"
        or not isinstance(sequential, dict)
        or sequential.get("status") != "completed"
        or not isinstance(strict_parity, dict)
        or not isinstance(final_queue, dict)
        or final_queue.get("jobs") != []
    ):
        raise HpcWorkflowAcceptanceError(
            "deterministic parity evidence does not prove strict one-thread parity"
        )
    if (
        strict_parity.get("comparison_policy") != "strict-deterministic-numeric-parity"
        or strict_parity.get("rtol") != 0.0
        or strict_parity.get("atol") != 0.0
        or strict_parity.get("metric_absolute_tolerance") != 0.0
        or strict_parity.get("voxel_differences_are_characterization") is not False
    ):
        raise HpcWorkflowAcceptanceError(
            "deterministic parity evidence does not require exact numerical equality"
        )
    rounds = strict_parity.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise HpcWorkflowAcceptanceError(
            "deterministic parity evidence contains no compared rounds"
        )

    def require_exact_numeric(node: object, location: str) -> None:
        if (
            not isinstance(node, Mapping)
            or node.get("array_equal") is not True
            or node.get("close_required") is not True
            or node.get("maximum_absolute_difference") != 0.0
            or node.get("rtol") != 0.0
            or node.get("atol") != 0.0
        ):
            raise HpcWorkflowAcceptanceError(
                f"deterministic parity {location} is not exactly equal"
            )

    for round_index, round_evidence in enumerate(rounds):
        if not isinstance(round_evidence, Mapping):
            raise HpcWorkflowAcceptanceError(
                "deterministic parity round evidence is malformed"
            )
        affine = round_evidence.get("affine")
        warp = round_evidence.get("warp")
        registered = round_evidence.get("registered")
        if not all(isinstance(item, Mapping) for item in (affine, warp, registered)):
            raise HpcWorkflowAcceptanceError(
                "deterministic parity artifact evidence is malformed"
            )
        require_exact_numeric(
            cast(Mapping[str, object], affine).get("parameters"),
            f"round {round_index} affine",
        )
        require_exact_numeric(
            cast(Mapping[str, object], warp).get("numeric"),
            f"round {round_index} warp",
        )
        require_exact_numeric(
            cast(Mapping[str, object], registered).get("numeric"),
            f"round {round_index} registered image",
        )
        metrics = round_evidence.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            raise HpcWorkflowAcceptanceError(
                "deterministic parity metric evidence is malformed"
            )
        for stage, metric in metrics.items():
            differences = (
                metric.get("absolute_differences")
                if isinstance(metric, Mapping)
                else None
            )
            if (
                not isinstance(metric, Mapping)
                or metric.get("absolute_tolerance") != 0.0
                or not isinstance(differences, Mapping)
                or not differences
                or any(value != 0.0 for value in differences.values())
            ):
                raise HpcWorkflowAcceptanceError(
                    f"deterministic parity {stage} metrics are not exactly equal"
                )
        channels = round_evidence.get("channels")
        if not isinstance(channels, list) or not channels:
            raise HpcWorkflowAcceptanceError(
                "deterministic parity channel evidence is malformed"
            )
        for channel_index, channel in enumerate(channels):
            comparison = (
                channel.get("comparison") if isinstance(channel, Mapping) else None
            )
            numeric = (
                comparison.get("numeric") if isinstance(comparison, Mapping) else None
            )
            require_exact_numeric(
                numeric,
                f"round {round_index} channel {channel_index}",
            )
    if require_fine_internal:
        internal = value.get("strict_internal_parity")
        if (
            not isinstance(internal, Mapping)
            or set(internal)
            != {
                "schema_version",
                "local_warp_exact",
                "diagnostic_exact",
                "rounds",
            }
            or internal.get("schema_version") != 1
            or internal.get("local_warp_exact") is not True
            or internal.get("diagnostic_exact") is not True
        ):
            raise HpcWorkflowAcceptanceError(
                "deterministic fine internal parity schema or summary is invalid"
            )
        internal_rounds = internal.get("rounds")
        if not isinstance(internal_rounds, list) or not internal_rounds:
            raise HpcWorkflowAcceptanceError(
                "deterministic fine internal parity contains no chunk rounds"
            )
        seen_rounds: list[int] = []
        for round_evidence in internal_rounds:
            if (
                not isinstance(round_evidence, Mapping)
                or set(round_evidence) != {"round", "local_warps", "diagnostics"}
                or type(round_evidence.get("round")) is not int
            ):
                raise HpcWorkflowAcceptanceError(
                    "deterministic fine internal round evidence is malformed"
                )
            seen_rounds.append(cast(int, round_evidence["round"]))
            for role in ("local_warps", "diagnostics"):
                artifacts = round_evidence.get(role)
                if not isinstance(artifacts, list) or not artifacts:
                    raise HpcWorkflowAcceptanceError(
                        f"deterministic fine internal {role} evidence is empty"
                    )
                names: list[str] = []
                for artifact in artifacts:
                    if (
                        not isinstance(artifact, Mapping)
                        or set(artifact) != {"name", "shape", "dtype", "exact"}
                        or type(artifact.get("name")) is not str
                        or type(artifact.get("dtype")) is not str
                        or not isinstance(artifact.get("shape"), list)
                        or not artifact["shape"]
                        or any(
                            type(item) is not int or item < 1
                            for item in cast(list[object], artifact["shape"])
                        )
                        or artifact.get("exact") is not True
                    ):
                        raise HpcWorkflowAcceptanceError(
                            f"deterministic fine internal {role} artifact is malformed"
                        )
                    names.append(cast(str, artifact["name"]))
                if names != sorted(set(names)):
                    raise HpcWorkflowAcceptanceError(
                        f"deterministic fine internal {role} names are noncanonical"
                    )
        if seen_rounds != sorted(set(seen_rounds)):
            raise HpcWorkflowAcceptanceError(
                "deterministic fine internal rounds are noncanonical"
            )
    return cast(dict[str, object], value)


def execute_deterministic_parity_subprocess(
    *,
    execution_path: Path,
    fine_execution_path: Path | None = None,
    owned_root: Path,
    evidence_dir: Path,
    timeout_seconds: float,
    _run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    _environ: Mapping[str, str] = os.environ,
) -> dict[str, object]:
    """Run strict parity in a fresh process before scientific imports occur."""
    if timeout_seconds <= 0.0:
        raise HpcWorkflowAcceptanceError(
            "deterministic parity subprocess timeout must be positive"
        )
    if not execution_path.is_file():
        raise HpcWorkflowAcceptanceError(
            f"deterministic parity execution profile is not a file: {execution_path}"
        )
    if fine_execution_path is not None and not fine_execution_path.is_file():
        raise HpcWorkflowAcceptanceError(
            f"deterministic fine execution profile is not a file: {fine_execution_path}"
        )
    if owned_root.is_symlink() or not owned_root.is_dir():
        raise HpcWorkflowAcceptanceError(
            f"deterministic parity owned root is not a real directory: {owned_root}"
        )
    if evidence_dir.is_symlink() or not evidence_dir.is_dir():
        raise HpcWorkflowAcceptanceError(
            "deterministic parity evidence directory is not a real directory: "
            f"{evidence_dir}"
        )

    repository_root = Path(__file__).resolve().parents[2]
    tests_root = repository_root / "tests"
    evidence_path = evidence_dir / "deterministic-executor-parity.json"
    command = (
        sys.executable,
        "-m",
        "hpc.deterministic_parity",
        "--execution",
        str(execution_path),
        *(
            ()
            if fine_execution_path is None
            else ("--fine-execution", str(fine_execution_path))
        ),
        "--owned-root",
        str(owned_root),
        "--evidence",
        str(evidence_path),
    )
    environment = dict(_environ)
    environment[ITK_THREAD_ENV_VAR] = "1"
    environment.update({name: "1" for name in BLAS_THREAD_ENV_VARS})
    environment["ANTS_RANDOM_SEED"] = "20260722"
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tests_root), inherited_pythonpath)
        if inherited_pythonpath
        else (str(tests_root),)
    )
    try:
        completed = _run(
            command,
            cwd=repository_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        _write_text_evidence(
            evidence_dir / "deterministic-coordinator.stdout.txt", stdout
        )
        _write_text_evidence(
            evidence_dir / "deterministic-coordinator.stderr.txt", stderr
        )
        raise HpcWorkflowAcceptanceError(
            "deterministic parity subprocess exceeded its bounded timeout"
        ) from error

    _write_text_evidence(
        evidence_dir / "deterministic-coordinator.stdout.txt", completed.stdout
    )
    _write_text_evidence(
        evidence_dir / "deterministic-coordinator.stderr.txt", completed.stderr
    )
    if completed.returncode != 0:
        raise HpcWorkflowAcceptanceError(
            "deterministic parity subprocess failed with exit code "
            f"{completed.returncode}; inspect captured coordinator output"
        )
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HpcWorkflowAcceptanceError(
            "deterministic parity subprocess did not write valid evidence"
        ) from error
    return _validated_deterministic_parity_evidence(
        evidence,
        require_fine_internal=fine_execution_path is not None,
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

    return cast(RoundResult, run_workflow_task(task))


def _production_fine_task(task: Any) -> Any:
    from cyexm.workflow.fine_task import run_fine_task

    return run_fine_task(task)


def _profiled_task_call(
    task: Any,
    *,
    timing_root: Path,
    key: str,
    operation: Callable[[Any], Any],
) -> Any:
    import resource

    from cyexm.registration.backend import AntsRegistrationBackend

    nonlinear_calls: list[dict[str, object]] = []
    original = AntsRegistrationBackend.nonlinear

    def timed_nonlinear(self: Any, **kwargs: object) -> tuple[object, Path]:
        started_at = _utc_now()
        started = time.monotonic()
        try:
            return original(self, **kwargs)
        finally:
            ended = time.monotonic()
            nonlinear_calls.append(
                {
                    "started_at": started_at,
                    "ended_at": _utc_now(),
                    "duration_seconds": float(max(0.0, ended - started)),
                    "monotonic_started_seconds": float(started),
                    "monotonic_ended_seconds": float(ended),
                }
            )

    started_at = _utc_now()
    started = time.monotonic()
    status = "completed"
    AntsRegistrationBackend.nonlinear = timed_nonlinear
    try:
        return operation(task)
    except BaseException:
        status = "failed"
        raise
    finally:
        AntsRegistrationBackend.nonlinear = original
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_memory_bytes = int(peak_rss * 1024)
        _write_marker(
            timing_root / f"{key.replace(':', '-')}.json",
            {
                "schema_version": 1,
                "task_key": key,
                "status": status,
                "started_at": started_at,
                "ended_at": _utc_now(),
                "duration_seconds": float(max(0.0, time.monotonic() - started)),
                "peak_memory_bytes": peak_memory_bytes,
                "nonlinear_calls": nonlinear_calls,
            },
        )


def run_profiled_workflow_task(
    task: WorkflowTask,
    *,
    timing_root: Path,
    _run_workflow_task: Callable[
        [WorkflowTask], RoundResult
    ] = _production_workflow_task,
) -> RoundResult:
    """Run one coarse task with acceptance-only kernel timing and peak RSS."""
    key = (
        f"round-{task.round:03d}:reference:main"
        if type(task) is ReferenceTask
        else f"round-{task.round:03d}:coarse:main"
    )
    return cast(
        RoundResult,
        _profiled_task_call(
            task,
            timing_root=timing_root,
            key=key,
            operation=_run_workflow_task,
        ),
    )


def run_profiled_fine_task(
    task: Any,
    *,
    timing_root: Path,
    _run_fine_task: Callable[[Any], Any] = _production_fine_task,
) -> Any:
    """Run one fine task with acceptance-only kernel timing and peak RSS."""
    key = getattr(getattr(task, "key", None), "encoded", None)
    if type(key) is not str:
        raise HpcWorkflowAcceptanceError(
            "profiled fine task lacks an exact task identity"
        )
    return _profiled_task_call(
        task,
        timing_root=timing_root,
        key=key,
        operation=_run_fine_task,
    )


def run_injected_fine_failure_task(
    task: Any,
    *,
    target_key: str,
    peer_key: str,
    failure_marker: Path,
    settled_marker: Path,
    timeout_seconds: float = 1_800.0,
    _run_fine_task: Callable[[Any], Any] = _production_fine_task,
) -> Any:
    """Inject one exact-key failure once after one exact peer durably settles."""
    from cyexm.workflow.fine_models import FineTaskKey

    target = FineTaskKey.from_encoded(target_key)
    peer = FineTaskKey.from_encoded(peer_key)
    if target == peer or target.stage not in {"chunk", "channel"}:
        raise HpcWorkflowAcceptanceError("fine failure keys must be distinct peers")
    key = getattr(task, "key", None)
    encoded = getattr(key, "encoded", None)
    is_target = encoded == target_key
    if is_target:
        if failure_marker.exists():
            return _run_fine_task(task)
        if not settled_marker.is_file():
            deadline = time.monotonic() + timeout_seconds
            while not settled_marker.is_file():
                if time.monotonic() >= deadline:
                    raise HpcWorkflowAcceptanceError(
                        "fine failure injection timed out waiting for a settled sibling"
                    )
                time.sleep(0.25)
        peer_record = json.loads(settled_marker.read_text(encoding="utf-8"))
        if (
            not isinstance(peer_record, dict)
            or peer_record.get("task_key") != peer_key
            or peer_record.get("status") not in {"completed", "reused"}
            or not peer_record.get("artifacts")
        ):
            raise HpcWorkflowAcceptanceError(
                "fine failure settled-peer marker is not exact committed evidence"
            )
        _write_marker(
            failure_marker,
            {
                "schema_version": 1,
                "task_key": target_key,
                "settled_peer_key": peer_key,
                "status": "injected",
                "injected_at": _utc_now(),
            },
        )
        raise RuntimeError(f"injected fine acceptance failure for {target_key}")
    result = _run_fine_task(task)
    if encoded == peer_key and getattr(result, "status", None) in {
        "completed",
        "reused",
    }:
        _write_fine_peer_commit(settled_marker, peer_key, result)
    return result


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_fine_peer_commit(path: Path, peer_key: str, result: Any) -> None:
    artifacts = []
    for raw in getattr(result, "artifact_paths", ()):
        artifact = Path(raw).resolve(strict=True)
        if not artifact.is_file():
            raise HpcWorkflowAcceptanceError(
                "fine committed peer artifact is not a real file"
            )
        artifacts.append(
            {
                "identity": FileIdentity.from_path(artifact).to_dict(),
                "sha256": _file_digest(artifact),
            }
        )
    if not artifacts:
        raise HpcWorkflowAcceptanceError(
            "fine committed peer must persist at least one artifact"
        )
    _write_marker(
        path,
        {
            "schema_version": 1,
            "task_key": peer_key,
            "status": getattr(result, "status", None),
            "artifacts": artifacts,
            "committed_at": _utc_now(),
        },
    )


def run_fine_cancellation_task(
    task: Any,
    *,
    target_key: str,
    peer_key: str,
    committed_marker: Path,
    ready_marker: Path,
    timeout_seconds: float,
    _run_fine_task: Callable[[Any], Any] = _production_fine_task,
    _monotonic: Callable[[], float] = time.monotonic,
    _sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Commit one exact peer, then block one distinct exact task for cancellation."""
    from cyexm.workflow.fine_models import FineTaskKey

    target = FineTaskKey.from_encoded(target_key)
    peer = FineTaskKey.from_encoded(peer_key)
    if (
        target == peer
        or target.round == peer.round
        or target.stage not in {"chunk", "channel"}
        or peer.stage != target.stage
        or timeout_seconds <= 0.0
    ):
        raise HpcWorkflowAcceptanceError(
            "fine cancellation requires distinct cross-round exact peer keys"
        )
    key = getattr(task, "key", None)
    encoded = getattr(key, "encoded", None)
    if encoded == peer_key:
        result = _run_fine_task(task)
        if getattr(result, "status", None) not in {"completed", "reused"}:
            raise HpcWorkflowAcceptanceError(
                "fine cancellation peer did not commit successfully"
            )
        _write_fine_peer_commit(committed_marker, peer_key, result)
        return result
    if encoded != target_key:
        return _run_fine_task(task)
    deadline = _monotonic() + timeout_seconds
    while not committed_marker.is_file():
        if _monotonic() >= deadline:
            raise HpcWorkflowAcceptanceError(
                "fine cancellation timed out waiting for committed exact peer"
            )
        _sleep(0.25)
    committed = json.loads(committed_marker.read_text(encoding="utf-8"))
    if (
        not isinstance(committed, dict)
        or committed.get("task_key") != peer_key
        or committed.get("status") not in {"completed", "reused"}
        or not committed.get("artifacts")
    ):
        raise HpcWorkflowAcceptanceError(
            "fine cancellation peer marker lacks committed artifact evidence"
        )
    _write_marker(
        ready_marker,
        {
            "blocked_task_key": target_key,
            "committed_peer_key": peer_key,
            "status": "waiting-for-coordinator-interruption",
        },
    )
    while _monotonic() < deadline:
        _sleep(min(0.25, max(0.0, deadline - _monotonic())))
    raise HpcWorkflowAcceptanceError(
        f"fine cancellation task {encoded} exceeded its bounded wait"
    )


def run_fine_worker_loss_task(
    task: Any,
    *,
    target_key: str,
    loss_marker: Path,
    _run_fine_task: Callable[[Any], Any] = _production_fine_task,
    _worker_identity: Callable[[], dict[str, object]] | None = None,
    _utc_now: Callable[[], str] = _utc_now,
    _exit: Callable[[int], Any] = os._exit,
) -> Any:
    """Terminate one worker exactly once, then let the replacement retry succeed."""
    from cyexm.workflow.fine_models import FineTaskKey

    target = FineTaskKey.from_encoded(target_key)
    if target.stage not in {"chunk", "channel"}:
        raise HpcWorkflowAcceptanceError(
            "fine worker-loss stage must be chunk or channel"
        )
    key = getattr(task, "key", None)
    if getattr(key, "encoded", None) != target_key or loss_marker.exists():
        return _run_fine_task(task)

    def production_identity() -> dict[str, object]:
        from distributed import get_worker

        return {
            "worker_address": get_worker().address,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }

    identity = (_worker_identity or production_identity)()
    if (
        set(identity) != {"worker_address", "hostname", "pid", "slurm_job_id"}
        or any(
            type(identity[name]) is not str or not identity[name]
            for name in ("worker_address", "hostname", "slurm_job_id")
        )
        or type(identity["pid"]) is not int
        or cast(int, identity["pid"]) < 1
    ):
        raise HpcWorkflowAcceptanceError(
            "fine worker-loss marker requires one exact worker identity"
        )
    _write_marker(
        loss_marker,
        {
            "schema_version": 1,
            "task_key": target_key,
            **identity,
            "status": "worker-loss-injected",
            "injected_at": _utc_now(),
        },
    )
    _exit(75)
    raise AssertionError("worker-loss exit unexpectedly returned")


def run_seeded_workflow_task(
    task: WorkflowTask,
    *,
    seed: str = "20260722",
    _run_workflow_task: Callable[[WorkflowTask], RoundResult] = (
        _production_workflow_task
    ),
    _environ: MutableMapping[str, str] = os.environ,
) -> RoundResult:
    """Run one acceptance task under a worker-local fixed ANTs seed."""
    if type(seed) is not str or not seed.isdecimal():
        raise HpcWorkflowAcceptanceError(
            "deterministic parity ANTs seed must contain decimal digits"
        )
    previous = _environ.get("ANTS_RANDOM_SEED")
    _environ["ANTS_RANDOM_SEED"] = seed
    try:
        return _run_workflow_task(task)
    finally:
        if previous is None:
            _environ.pop("ANTS_RANDOM_SEED", None)
        else:
            _environ["ANTS_RANDOM_SEED"] = previous


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
    task = cast(RoundTask, task)
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
    task = cast(RoundTask, task)
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
    committed_peer_key: str | None = None,
    blocked_task_key: str | None = None,
    timeout_seconds: float,
    fine_runner: Callable[[Any], Any] | None = None,
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
    if (committed_peer_key is None) != (blocked_task_key is None):
        raise HpcWorkflowAcceptanceError(
            "exact cancellation task keys must be supplied together"
        )
    exact_keys = committed_peer_key is not None and blocked_task_key is not None
    trigger: dict[str, object] = {
        **(
            {
                "blocked_task_key": blocked_task_key,
                "committed_peer_key": committed_peer_key,
            }
            if exact_keys
            else {
                "blocked_round": blocked_round,
                "committed_round": successful_round,
            }
        ),
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
                    **(
                        {
                            "blocked_task_key": blocked_task_key,
                            "committed_peer_key": committed_peer_key,
                        }
                        if exact_keys
                        else {
                            "blocked_round": blocked_round,
                            "committed_round": successful_round,
                        }
                    ),
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
                fine_runner=fine_runner,
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
    fine_runner: Callable[[Any], Any] | None = None,
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

    executor_kwargs: dict[str, object] = {"_runner": runner}
    if fine_runner is not None:
        executor_kwargs["_fine_runner"] = fine_runner
    executor = _executor_ctor(plan.execution, **executor_kwargs)
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
    "deterministic_acceptance_manifest",
    "deterministic_sequential_profile",
    "deterministic_slurm_profile",
    "execute_cancellation_acceptance",
    "execute_deterministic_parity_subprocess",
    "execute_owned_slurm_workflow",
    "metric_comparison_evidence",
    "numeric_comparison_evidence",
    "parallel_placement_evidence",
    "prepare_task5_output_roots",
    "run_cancellation_task",
    "run_failure_task",
    "run_fine_cancellation_task",
    "run_fine_worker_loss_task",
    "run_injected_fine_failure_task",
    "run_profiled_fine_task",
    "run_profiled_workflow_task",
    "run_seeded_workflow_task",
    "scientific_equivalence_evidence",
    "scientific_parity_evidence",
    "sequential_reference_profile",
    "task5_output_roots",
    "validate_injected_failure_record",
    "validate_task5_output_ownership",
    "workflow_module_by_value",
    "write_deterministic_inputs",
]
