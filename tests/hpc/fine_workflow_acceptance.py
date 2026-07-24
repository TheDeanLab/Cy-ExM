"""Strict portable builders and validators for fine-SLURM acceptance evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import cast

from cyexm.config.models import DatasetManifest, ExecutionProfile
from cyexm.workflow.models import WorkflowPlan
from hpc.workflow_acceptance import HpcWorkflowAcceptanceError


def _timestamp(value: str) -> datetime:
    if not value.endswith("Z"):
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance timestamp must use canonical UTC Z notation: {value!r}"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance timestamp is not strict ISO-8601: {value!r}"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance timestamp must be UTC: {value!r}"
        )
    return parsed


_FINE_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "identities",
        "scientific",
        "fixed_grid",
        "placement",
        "durability",
        "performance",
        "owned_root",
        "evidence_files",
    }
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fine_acceptance_profiles(
    coarse: ExecutionProfile,
    fine: ExecutionProfile,
) -> tuple[ExecutionProfile, ExecutionProfile]:
    """Validate two exact operator-approved coarse and fine SLURM profiles.

    Parameters
    ----------
    coarse, fine
        Independently loaded operator-approved execution profiles.

    Returns
    -------
    tuple of ExecutionProfile
        The unchanged approved profiles.
    """
    if any(
        profile.executor != "slurm" or profile.slurm is None
        for profile in (coarse, fine)
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance requires two approved SLURM profiles"
        )
    if any(
        profile.slurm is None
        or profile.slurm.workers < 2
        or profile.slurm.processes_per_worker != 1
        for profile in (coarse, fine)
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance requires at least two single-process workers"
        )
    if coarse.parallelism.mode != "coarse" or fine.parallelism.mode != "fine_grained":
        raise HpcWorkflowAcceptanceError(
            "fine acceptance profiles must explicitly select coarse and fine_grained"
        )
    coarse_slurm = coarse.slurm
    fine_slurm = fine.slurm
    assert coarse_slurm is not None and fine_slurm is not None
    coarse_scratch = coarse.paths.worker_local_directory
    fine_scratch = fine.paths.worker_local_directory
    if coarse_scratch is None or fine_scratch is None:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance profiles require explicit worker scratch roots"
        )
    coarse_output = coarse.paths.output_root.resolve(strict=False)
    fine_output = fine.paths.output_root.resolve(strict=False)
    coarse_scratch_resolved = coarse_scratch.resolve(strict=False)
    fine_scratch_resolved = fine_scratch.resolve(strict=False)

    def disjoint(left: Path, right: Path) -> bool:
        return not left.is_relative_to(right) and not right.is_relative_to(left)

    if (
        not disjoint(coarse_output, fine_output)
        or not disjoint(coarse_scratch_resolved, fine_scratch_resolved)
        or coarse_slurm.job_name == fine_slurm.job_name
    ):
        raise HpcWorkflowAcceptanceError(
            "coarse and fine output/scratch roots must be canonically disjoint "
            "and job names must be distinct"
        )
    normalized_fine = replace(
        fine,
        paths=replace(
            fine.paths,
            output_root=coarse.paths.output_root,
            worker_local_directory=coarse_scratch,
        ),
        slurm=replace(fine_slurm, job_name=coarse_slurm.job_name),
        source_path=coarse.source_path,
        parallelism=coarse.parallelism,
    )
    if normalized_fine != coarse:
        raise HpcWorkflowAcceptanceError(
            "coarse and fine acceptance profiles must be otherwise identical"
        )
    return coarse, fine


def fine_acceptance_manifest(
    manifest: DatasetManifest,
) -> tuple[DatasetManifest, dict[str, str]]:
    """Select and preflight the exact two-round acceptance task identities."""
    rounds = tuple(
        sorted(
            (item for item in manifest.rounds if item.strategy == "serial_chunked"),
            key=lambda item: item.round,
        )[:2]
    )
    if len(rounds) != 2:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance requires two serial-chunked rounds before submission"
        )
    dependent_indices: list[int] = []
    for item in rounds:
        indices = [
            index
            for index, channel in enumerate(item.channels)
            if channel.file != item.registration_channel
        ]
        if not indices:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance requires an actual dependent channel in each "
                "selected round before submission"
            )
        dependent_indices.append(indices[0])
    first, second = rounds
    targets = {
        "chunk_target": f"round-{first.round:03d}:chunk:0000",
        "chunk_peer": f"round-{second.round:03d}:chunk:0000",
        "channel_target": (
            f"round-{first.round:03d}:channel:{dependent_indices[0]:04d}"
        ),
        "channel_peer": (
            f"round-{second.round:03d}:channel:{dependent_indices[1]:04d}"
        ),
        "replacement_target": f"round-{first.round:03d}:chunk:0000",
        "liveness_peer": f"round-{second.round:03d}:chunk:0000",
    }
    if len(set(targets.values())) < 4:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance exact task prerequisites are not distinct"
        )
    return replace(manifest, rounds=rounds), targets


def fine_internal_parity_evidence(
    coarse: WorkflowPlan,
    fine: WorkflowPlan,
) -> dict[str, object]:
    """Require exact local warp and scalar-core parity for serial chunk rounds."""
    import numpy as np

    if coarse.scientific_fingerprint != fine.scientific_fingerprint:
        raise HpcWorkflowAcceptanceError(
            "fine internal parity plans changed scientific identity"
        )
    rounds: list[dict[str, object]] = []
    for coarse_task, fine_task in zip(coarse.tasks, fine.tasks, strict=True):
        if coarse_task.round != fine_task.round or (
            coarse_task.strategy != fine_task.strategy
        ):
            raise HpcWorkflowAcceptanceError(
                "fine internal parity round structures differ"
            )
        if coarse_task.strategy != "serial_chunked":
            continue
        coarse_root = (
            coarse_task.registration_directory / f"round_{coarse_task.round}_chunks"
        )
        fine_root = fine_task.registration_directory / f"round_{fine_task.round}_chunks"
        coarse_warps = sorted(coarse_root.glob("chunk_*_warp.nii.gz"))
        fine_warps = sorted(fine_root.glob("chunk_*_warp.nii.gz"))
        coarse_cores = sorted(coarse_root.glob("chunk_*_registered_core.npy"))
        fine_cores = sorted(fine_root.glob("chunk_*_registered_core.npy"))
        if (
            not coarse_warps
            or not coarse_cores
            or [item.name for item in coarse_warps]
            != [item.name for item in fine_warps]
            or [item.name for item in coarse_cores]
            != [item.name for item in fine_cores]
        ):
            raise HpcWorkflowAcceptanceError(
                "fine internal parity chunk artifact structures differ"
            )
        local_warps: list[dict[str, object]] = []
        import ants

        for coarse_path, fine_path in zip(coarse_warps, fine_warps, strict=True):
            coarse_warp = ants.image_read(str(coarse_path))
            fine_warp = ants.image_read(str(fine_path))
            coarse_array = np.asarray(coarse_warp.numpy())
            fine_array = np.asarray(fine_warp.numpy())
            if (
                coarse_array.dtype != fine_array.dtype
                or not np.all(np.isfinite(coarse_array))
                or not np.all(np.isfinite(fine_array))
                or not np.array_equal(coarse_array, fine_array)
                or tuple(coarse_warp.shape) != tuple(fine_warp.shape)
                or tuple(coarse_warp.origin) != tuple(fine_warp.origin)
                or tuple(coarse_warp.spacing) != tuple(fine_warp.spacing)
                or not np.array_equal(
                    np.asarray(coarse_warp.direction),
                    np.asarray(fine_warp.direction),
                )
                or bool(coarse_warp.has_components) != bool(fine_warp.has_components)
            ):
                raise HpcWorkflowAcceptanceError(
                    f"fine internal local warp differs: {coarse_path.name}"
                )
            local_warps.append(
                {
                    "name": coarse_path.name,
                    "shape": list(coarse_array.shape),
                    "dtype": str(coarse_array.dtype),
                    "exact": True,
                }
            )
        diagnostics: list[dict[str, object]] = []
        for coarse_path, fine_path in zip(coarse_cores, fine_cores, strict=True):
            coarse_core = np.load(coarse_path, allow_pickle=False)
            fine_core = np.load(fine_path, allow_pickle=False)
            if (
                coarse_core.dtype != fine_core.dtype
                or not np.all(np.isfinite(coarse_core))
                or not np.all(np.isfinite(fine_core))
                or not np.array_equal(coarse_core, fine_core)
            ):
                raise HpcWorkflowAcceptanceError(
                    f"fine internal scalar diagnostic differs: {coarse_path.name}"
                )
            diagnostics.append(
                {
                    "name": coarse_path.name,
                    "shape": list(coarse_core.shape),
                    "dtype": str(coarse_core.dtype),
                    "exact": True,
                }
            )
        rounds.append(
            {
                "round": coarse_task.round,
                "local_warps": local_warps,
                "diagnostics": diagnostics,
            }
        )
    if not rounds:
        raise HpcWorkflowAcceptanceError(
            "fine internal parity contains no serial chunk round"
        )
    return {
        "schema_version": 1,
        "local_warp_exact": True,
        "diagnostic_exact": True,
        "rounds": rounds,
    }


def _fine_object(
    value: object,
    *,
    keys: frozenset[str],
    location: str,
) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} must be an exact object"
        )
    result = cast(dict[str, object], value)
    if frozenset(result) != keys:
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} schema must be exactly {sorted(keys)!r}"
        )
    return result


def _fine_array(value: object, location: str) -> list[object]:
    if type(value) is not list:
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} must be an exact array"
        )
    return cast(list[object], value)


def _fine_number(
    value: object,
    location: str,
    *,
    minimum: float | None = 0.0,
) -> float:
    import math

    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} must be a finite number"
            + ("" if minimum is None else f" >= {minimum}")
        )
    return float(value)


def _fine_nonempty_string(value: object, location: str) -> str:
    if type(value) is not str or not value:
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} must be a nonempty exact string"
        )
    return value


def _fine_file_identity(value: object, location: str) -> dict[str, object]:
    identity = _fine_object(
        value,
        keys=frozenset({"path", "st_size", "st_mtime_ns"}),
        location=location,
    )
    _fine_nonempty_string(identity["path"], f"{location}.path")
    for name in ("st_size", "st_mtime_ns"):
        if type(identity[name]) is not int or cast(int, identity[name]) < 0:
            raise HpcWorkflowAcceptanceError(
                f"fine acceptance {location}.{name} must be a non-negative integer"
            )
    return identity


def _verify_file_identity_and_sha256(
    identity: dict[str, object],
    digest: object,
    location: str,
) -> None:
    claimed = _fine_sha256(digest, f"{location}.sha256")
    try:
        path = Path(cast(str, identity["path"])).resolve(strict=True)
        stat = path.stat()
    except OSError as error:
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} file is missing"
        ) from error
    if (
        not path.is_file()
        or path.is_symlink()
        or str(path) != identity["path"]
        or stat.st_size != identity["st_size"]
        or stat.st_mtime_ns != identity["st_mtime_ns"]
        or _file_sha256(path) != claimed
    ):
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} identity or checksum is stale or forged"
        )


def _fine_sha256(value: object, location: str) -> str:
    encoded = _fine_nonempty_string(value, location)
    if len(encoded) != 64 or any(
        character not in "0123456789abcdef" for character in encoded
    ):
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} must be a lowercase SHA-256 digest"
        )
    return encoded


def _fine_relative_path(value: object, location: str) -> str:
    encoded = _fine_nonempty_string(value, location)
    path = PurePosixPath(encoded)
    if path.is_absolute() or encoded != path.as_posix() or ".." in path.parts:
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} must be a canonical relative path"
        )
    return encoded


def _evidence_reference(
    value: object,
    *,
    owned_root: Path,
    location: str,
) -> tuple[str, str, object]:
    reference = _fine_object(
        value,
        keys=frozenset({"relative_path", "sha256"}),
        location=location,
    )
    relative = _fine_relative_path(
        reference["relative_path"], f"{location}.relative_path"
    )
    claimed = _fine_sha256(reference["sha256"], f"{location}.sha256")
    try:
        path = (owned_root / PurePosixPath(relative)).resolve(strict=True)
        path.relative_to(owned_root)
        observed = _file_sha256(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} is missing, unsafe, or invalid JSON"
        ) from error
    if not path.is_file() or path.is_symlink() or observed != claimed:
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} checksum is stale or forged"
        )
    return relative, claimed, payload


def _derive_primary_evidence(value: object) -> tuple[dict[str, object], float]:
    """Derive placement, fairness, cap, and fine timing from primary raw evidence."""
    if not isinstance(value, Mapping) or not isinstance(
        graph := value.get("fine_graph"), Mapping
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance primary evidence lacks a fine graph"
        )
    placements = graph.get("placements")
    scale_events = graph.get("scale_events")
    cap = graph.get("worker_cap")
    if (
        not isinstance(placements, list)
        or not isinstance(scale_events, list)
        or type(cap) is not int
        or cap < 2
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance primary graph placement/scale schema is malformed"
        )

    def accepted(item: object) -> bool:
        return (
            isinstance(item, Mapping)
            and item.get("placement_accepted") is True
            and item.get("status") in {"completed", "reused"}
        )

    task_summaries: list[dict[str, object]] = []
    preparations: list[Mapping[str, object]] = []
    assemblies: list[Mapping[str, object]] = []
    for raw in placements:
        if not accepted(raw):
            continue
        item = cast(Mapping[str, object], raw)
        key = item.get("task_key")
        if not isinstance(key, str):
            raise HpcWorkflowAcceptanceError(
                "fine acceptance primary placement task key is malformed"
            )
        if ":preparation:" in key:
            preparations.append(item)
            continue
        if ":assembly:" in key:
            assemblies.append(item)
            continue
        parts = key.split(":")
        if len(parts) != 3 or parts[1] not in {"chunk", "channel"}:
            continue
        try:
            round_number = int(parts[0].removeprefix("round-"))
            submitted = _timestamp(cast(str, item["coordinator_submitted_at"]))
            ready = _timestamp(cast(str, item["coordinator_ready_at"]))
            duration = _fine_number(
                item["duration_seconds"], f"primary placement {key}.duration"
            )
            job_id = _fine_nonempty_string(
                item["slurm_job_id"], f"primary placement {key}.slurm_job_id"
            )
            hostname = _fine_nonempty_string(
                item["hostname"], f"primary placement {key}.hostname"
            )
            worker = _fine_nonempty_string(
                item["worker_address"], f"primary placement {key}.worker_address"
            )
        except (KeyError, TypeError, ValueError) as error:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance primary placement record is malformed"
            ) from error
        if ready < submitted:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance primary placement readiness predates submission"
            )
        task_summaries.append(
            {
                "task_key": key,
                "kind": parts[1],
                "round": round_number,
                "job_id": job_id,
                "hostname": hostname,
                "worker": worker,
                "coordinator_submitted_at": cast(str, item["coordinator_submitted_at"]),
                "coordinator_ready_at": cast(str, item["coordinator_ready_at"]),
                "duration_seconds": duration,
            }
        )
    if not task_summaries:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance primary graph has no accepted granular tasks"
        )
    if (
        len({item["job_id"] for item in task_summaries}) < 2
        or len({item["hostname"] for item in task_summaries}) < 2
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance primary graph lacks two jobs and hosts"
        )
    ready_by_round: dict[int, datetime] = {}
    for item in preparations:
        try:
            key = cast(str, item["task_key"])
            round_number = int(key.split(":", 1)[0].removeprefix("round-"))
            ready = _timestamp(cast(str, item["coordinator_ready_at"]))
        except (KeyError, TypeError, ValueError) as error:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance primary preparation record is malformed"
            ) from error
        ready_by_round[round_number] = ready
    if len(ready_by_round) < 2:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance primary lacks two coordinator-ready preparations"
        )
    both_ready = max(ready_by_round.values())
    fair_candidates = sorted(
        (
            item
            for item in task_summaries
            if item["kind"] == "chunk"
            and _timestamp(cast(str, item["coordinator_submitted_at"])) >= both_ready
        ),
        key=lambda item: cast(str, item["coordinator_submitted_at"]),
    )
    if len(fair_candidates) < 2 or (
        fair_candidates[0]["round"] == fair_candidates[1]["round"]
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance primary lacks fair cross-round chunk dispatch"
        )
    active: set[str] = set()
    maximum_active = 0
    requests: list[int] = []
    for raw_event in scale_events:
        if not isinstance(raw_event, Mapping):
            raise HpcWorkflowAcceptanceError(
                "fine acceptance primary scale event is malformed"
            )
        requested = raw_event.get("requested")
        activated = raw_event.get("activated")
        departed = raw_event.get("departed")
        if (
            type(requested) is not int
            or requested < 1
            or not isinstance(activated, list)
            or not isinstance(departed, list)
            or any(not isinstance(item, str) or not item for item in activated)
            or any(not isinstance(item, str) or not item for item in departed)
        ):
            raise HpcWorkflowAcceptanceError(
                "fine acceptance primary scale event fields are malformed"
            )
        active.difference_update(cast(list[str], departed))
        active.update(cast(list[str], activated))
        maximum_active = max(maximum_active, len(active))
        requests.append(requested)
    if maximum_active < 2 or maximum_active > cap:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance primary activated worker maximum violates cap"
        )
    chunk_attempts = [item for item in task_summaries if item["kind"] == "chunk"]
    if not assemblies:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance primary has no accepted assembly readiness"
        )
    try:
        fine_wall = (
            max(
                _timestamp(cast(str, item["coordinator_ready_at"]))
                for item in assemblies
            )
            - min(
                _timestamp(cast(str, item["coordinator_submitted_at"]))
                for item in chunk_attempts
            )
        ).total_seconds()
    except (KeyError, TypeError, ValueError) as error:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance primary timing boundaries are malformed"
        ) from error
    if fine_wall <= 0.0:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance primary fine upper bound is not positive"
        )
    return (
        {
            "worker_cap": cap,
            "maximum_connected_workers": maximum_active,
            "scale_events": requests,
            "fair_ready_rounds": [
                cast(int, fair_candidates[0]["round"]),
                cast(int, fair_candidates[1]["round"]),
            ],
            "fair_dispatch_order": [
                cast(str, fair_candidates[0]["task_key"]),
                cast(str, fair_candidates[1]["task_key"]),
            ],
            "tasks": task_summaries,
        },
        fine_wall,
    )


def _coarse_nonlinear_lower_bound(
    records: Sequence[tuple[str, object]],
) -> tuple[float, str]:
    """Return the maximum per-round serial nonlinear interval."""
    round_totals: list[float] = []
    policies: set[str] = set()
    seen_rounds: set[int] = set()
    for relative, raw in records:
        expected_name = PurePosixPath(relative).name
        if not (
            expected_name.startswith("round-")
            and expected_name.endswith("-coarse-main.json")
            and isinstance(raw, Mapping)
            and raw.get("schema_version") == 1
            and raw.get("status") == "completed"
        ):
            raise HpcWorkflowAcceptanceError(
                "fine acceptance coarse timing record is not a stable completed "
                "round-XXX-coarse-main record"
            )
        try:
            round_token = expected_name.removeprefix("round-").removesuffix(
                "-coarse-main.json"
            )
            round_number = int(round_token)
        except ValueError as error:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance coarse timing filename has no round identity"
            ) from error
        if round_number in seen_rounds:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance coarse timing repeats a round record"
            )
        seen_rounds.add(round_number)
        if raw.get("task_key") != f"round-{round_number:03d}:coarse:main":
            raise HpcWorkflowAcceptanceError(
                "fine acceptance coarse timing task identity differs from filename"
            )
        calls = raw.get("nonlinear_calls")
        if not isinstance(calls, list) or not calls:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance coarse timing round has no nonlinear calls"
            )
        durations: list[float] = []
        boundaries: list[tuple[float, float]] = []
        for index, call in enumerate(calls):
            if not isinstance(call, Mapping):
                raise HpcWorkflowAcceptanceError(
                    "fine acceptance coarse timing call is malformed"
                )
            duration = _fine_number(
                call.get("duration_seconds"),
                f"coarse timing round {round_number} call {index}.duration",
            )
            durations.append(duration)
            has_start = "monotonic_started_seconds" in call
            has_end = "monotonic_ended_seconds" in call
            if has_start != has_end:
                raise HpcWorkflowAcceptanceError(
                    "fine acceptance coarse timing call has partial monotonic bounds"
                )
            if has_start:
                started = _fine_number(
                    call["monotonic_started_seconds"],
                    f"coarse timing round {round_number} call {index}.start",
                )
                ended = _fine_number(
                    call["monotonic_ended_seconds"],
                    f"coarse timing round {round_number} call {index}.end",
                )
                if ended < started or duration > ended - started + 1e-9:
                    raise HpcWorkflowAcceptanceError(
                        "fine acceptance coarse timing monotonic interval is invalid"
                    )
                boundaries.append((started, ended))
        if boundaries and len(boundaries) != len(calls):
            raise HpcWorkflowAcceptanceError(
                "fine acceptance coarse timing mixes monotonic and duration-only calls"
            )
        if boundaries:
            if any(
                current[0] < previous[1] for previous, current in pairwise(boundaries)
            ):
                raise HpcWorkflowAcceptanceError(
                    "fine acceptance coarse timing calls are not monotonic and ordered"
                )
            total = max(end for _, end in boundaries) - min(
                start for start, _ in boundaries
            )
            policies.add("per-round-monotonic-kernel-span")
        else:
            total = sum(durations)
            policies.add("per-round-summed-kernel-durations")
        if total <= 0.0:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance coarse timing lower bound is not positive"
            )
        round_totals.append(total)
    if not round_totals or len(policies) != 1:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance coarse timing records use no single lower-bound policy"
        )
    return max(round_totals), policies.pop()


def _validated_owned_outcome(
    value: object,
    *,
    operation: str,
    status: str,
) -> tuple[str, ...]:
    """Decode the exact outcome schema emitted by the owned workflow helper."""
    base_keys = {
        "schema_version",
        "operation",
        "status",
        "owned_job_ids",
        "cleanup_outcome",
        "workflow_result",
    }
    expected_keys = base_keys | ({"error"} if status == "cancelled" else set())
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {operation} outcome has an invalid exact schema"
        )
    if (
        value.get("schema_version") != 1
        or value.get("operation") != operation
        or value.get("status") != status
        or value.get("cleanup_outcome") != "empty"
    ):
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {operation} cleanup outcome is not successful"
        )
    owned = value.get("owned_job_ids")
    if (
        not isinstance(owned, list)
        or not owned
        or any(type(item) is not str or not item for item in owned)
        or owned != sorted(set(owned))
    ):
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {operation} owned job IDs are not canonical"
        )
    if status == "completed":
        if not isinstance(value.get("workflow_result"), Mapping):
            raise HpcWorkflowAcceptanceError(
                f"fine acceptance {operation} lacks its completed workflow result"
            )
    else:
        error = value.get("error")
        if (
            value.get("workflow_result") is not None
            or not isinstance(error, dict)
            or set(error) != {"exception_type", "message"}
            or error.get("exception_type") != "KeyboardInterrupt"
            or not isinstance(error.get("message"), str)
            or not error["message"]
        ):
            raise HpcWorkflowAcceptanceError(
                f"fine acceptance {operation} cancellation outcome is malformed"
            )
    return tuple(cast(list[str], owned))


def _validate_empty_owned_queue(value: object, *, operation: str) -> None:
    if not isinstance(value, Mapping) or value.get("jobs") != []:
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {operation} queue is not empty after cleanup"
        )


def _validate_executor_cleanup(value: object, *, operation: str) -> None:
    if not isinstance(value, Mapping):
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {operation} execution evidence lacks cleanup"
        )
    expected_order = ["client", "cluster", "configuration"]
    if value.get("order") != expected_order or set(value) != {
        *expected_order,
        "order",
    }:
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {operation} executor cleanup order is invalid"
        )
    for name in expected_order:
        if value.get(name) != {"attempted": True, "succeeded": True}:
            raise HpcWorkflowAcceptanceError(
                f"fine acceptance {operation} executor {name} cleanup failed"
            )


def _maximum_metric_difference(value: object) -> float:
    differences: list[float] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key == "absolute_differences" and isinstance(child, Mapping):
                    differences.extend(
                        _fine_number(
                            difference,
                            "Tier 1 evidence metric absolute difference",
                        )
                        for difference in child.values()
                    )
                else:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    if not differences:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance Tier 1 evidence contains no metric differences"
        )
    return max(differences)


def _fine_artifacts(value: object, location: str) -> tuple[dict[str, object], ...]:
    artifacts: list[dict[str, object]] = []
    paths: list[str] = []
    for index, raw in enumerate(_fine_array(value, location)):
        artifact = _fine_object(
            raw,
            keys=frozenset({"relative_path", "sha256"}),
            location=f"{location}[{index}]",
        )
        paths.append(
            _fine_relative_path(
                artifact["relative_path"], f"{location}[{index}].relative_path"
            )
        )
        _fine_sha256(artifact["sha256"], f"{location}[{index}].sha256")
        artifacts.append(artifact)
    if not artifacts or paths != sorted(set(paths)):
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} must be nonempty, unique, and canonical"
        )
    return tuple(artifacts)


def _fine_exact_strings(value: object, location: str) -> tuple[str, ...]:
    values = tuple(
        _fine_nonempty_string(item, f"{location}[{index}]")
        for index, item in enumerate(_fine_array(value, location))
    )
    if len(values) != len(set(values)):
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} contains duplicate values"
        )
    return values


def _validate_fine_resume(
    value: object,
    location: str,
    expected_kind: str,
    *,
    owned_root: Path,
) -> tuple[tuple[str, str], ...]:
    resume = _fine_object(
        value,
        keys=frozenset(
            {
                "injected_task_key",
                "failure_status",
                "failed_attempts",
                "settled_peer_key",
                "preserved_artifacts",
                "resume_reused_keys",
                "resume_recomputed_keys",
                "ordinary_artifact_store_reuse",
                "failure_evidence_file",
                "resume_evidence_file",
                "failure_marker_file",
                "settled_peer_file",
            }
        ),
        location=location,
    )
    injected = _fine_nonempty_string(
        resume["injected_task_key"], f"{location}.injected_task_key"
    )
    if f":{expected_kind}:" not in injected or resume["failure_status"] != "failed":
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} lacks the exact injected "
            f"{expected_kind} failure"
        )
    settled_peer = _fine_nonempty_string(
        resume["settled_peer_key"], f"{location}.settled_peer_key"
    )
    reused = _fine_exact_strings(
        resume["resume_reused_keys"], f"{location}.resume_reused_keys"
    )
    recomputed = _fine_exact_strings(
        resume["resume_recomputed_keys"], f"{location}.resume_recomputed_keys"
    )
    if (
        resume["failed_attempts"] != 1
        or settled_peer == injected
        or f":{expected_kind}:" not in settled_peer
        or injected.split(":", 1)[0] == settled_peer.split(":", 1)[0]
        or set(reused) & set(recomputed)
        or reused != (settled_peer,)
        or recomputed != (injected,)
        or resume["ordinary_artifact_store_reuse"] is not True
    ):
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} does not prove exact ArtifactStore resume"
        )
    references: list[tuple[str, str]] = []
    payloads: dict[str, object] = {}
    for name in (
        "failure_evidence_file",
        "resume_evidence_file",
        "failure_marker_file",
        "settled_peer_file",
    ):
        relative, digest, payload = _evidence_reference(
            resume[name],
            owned_root=owned_root,
            location=f"{location}.{name}",
        )
        references.append((relative, digest))
        payloads[name] = payload
    failure_evidence = payloads["failure_evidence_file"]
    resume_evidence = payloads["resume_evidence_file"]
    marker = payloads["failure_marker_file"]
    settled_marker = payloads["settled_peer_file"]
    try:
        failure_graph = cast(Mapping[str, object], failure_evidence)["fine_graph"]
        failed_placements = [
            item
            for item in cast(
                list[dict[str, object]],
                cast(Mapping[str, object], failure_graph)["placements"],
            )
            if item.get("task_key") == injected and item.get("status") == "failed"
        ]
        failure_results = cast(
            list[dict[str, object]],
            cast(Mapping[str, object], failure_graph)["task_results"],
        )
        resume_results = cast(
            list[dict[str, object]],
            cast(
                Mapping[str, object],
                cast(Mapping[str, object], resume_evidence)["fine_graph"],
            )["task_results"],
        )
    except (KeyError, TypeError) as error:
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} raw graph evidence is malformed"
        ) from error
    resume_by_key = {
        cast(str, item.get("key")): item.get("status") for item in resume_results
    }
    if (
        len(failed_placements) != 1
        or failed_placements[0].get("attempt") != 1
        or not any(
            item.get("key") == settled_peer
            and item.get("status") in {"completed", "reused"}
            for item in failure_results
        )
        or not isinstance(marker, Mapping)
        or marker.get("task_key") != injected
        or marker.get("settled_peer_key") != settled_peer
        or marker.get("status") != "injected"
        or not isinstance(settled_marker, Mapping)
        or settled_marker.get("task_key") != settled_peer
        or settled_marker.get("status") not in {"completed", "reused"}
        or not settled_marker.get("artifacts")
        or resume_by_key.get(injected) != "completed"
        or resume_by_key.get(settled_peer) != "reused"
    ):
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} does not match checksum-bound exact "
            "failure, peer settlement, and resume evidence"
        )
    _fine_artifacts(resume["preserved_artifacts"], f"{location}.preserved_artifacts")
    return tuple(references)


def _validate_duration_summary(
    value: object,
    *,
    location: str,
    expected_count: int,
) -> None:
    summary = _fine_object(
        value,
        keys=frozenset({"count", "minimum", "median", "maximum"}),
        location=location,
    )
    if type(summary["count"]) is not int or summary["count"] != expected_count:
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location}.count does not match task evidence"
        )
    minimum = _fine_number(summary["minimum"], f"{location}.minimum")
    median = _fine_number(summary["median"], f"{location}.median")
    maximum = _fine_number(summary["maximum"], f"{location}.maximum")
    if not minimum <= median <= maximum:
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} is not an ordered duration distribution"
        )


def _fine_worker_identity(value: object, location: str) -> dict[str, object]:
    identity = _fine_object(
        value,
        keys=frozenset({"worker_address", "hostname", "pid", "slurm_job_id"}),
        location=location,
    )
    for name in ("worker_address", "hostname", "slurm_job_id"):
        _fine_nonempty_string(identity[name], f"{location}.{name}")
    if type(identity["pid"]) is not int or cast(int, identity["pid"]) < 1:
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location}.pid must be a positive integer"
        )
    return identity


def _validate_worker_replacement(
    value: object, *, owned_root: Path
) -> tuple[tuple[str, str], ...]:
    location = "durability.worker_replacement"
    replacement = _fine_object(
        value,
        keys=frozenset(
            {
                "loss_marker",
                "evidence_file",
                "outcome_file",
                "queue_after_file",
                "loss_marker_file",
                "failed_attempt",
                "recovery_mode",
                "departed_worker",
                "replacement_worker",
                "replacement_liveness_path",
                "retry",
                "liveness",
                "owned_job_ids",
                "cleanup_outcome",
            }
        ),
        location=location,
    )
    evidence_relative, evidence_sha, raw_evidence = _evidence_reference(
        replacement["evidence_file"],
        owned_root=owned_root,
        location=f"{location}.evidence_file",
    )
    queue_relative, queue_sha, raw_queue = _evidence_reference(
        replacement["queue_after_file"],
        owned_root=owned_root,
        location=f"{location}.queue_after_file",
    )
    outcome_relative, outcome_sha, raw_outcome = _evidence_reference(
        replacement["outcome_file"],
        owned_root=owned_root,
        location=f"{location}.outcome_file",
    )
    marker_relative, marker_sha, raw_marker = _evidence_reference(
        replacement["loss_marker_file"],
        owned_root=owned_root,
        location=f"{location}.loss_marker_file",
    )
    loss = _fine_object(
        replacement["loss_marker"],
        keys=frozenset(
            {"task_key", "worker_address", "hostname", "pid", "slurm_job_id"}
        ),
        location=f"{location}.loss_marker",
    )
    loss_key = _fine_nonempty_string(
        loss["task_key"], f"{location}.loss_marker.task_key"
    )
    loss_identity = _fine_worker_identity(
        {
            name: loss[name]
            for name in ("worker_address", "hostname", "pid", "slurm_job_id")
        },
        f"{location}.loss_marker.identity",
    )
    if (
        not isinstance(raw_marker, Mapping)
        or raw_marker.get("schema_version") != 1
        or raw_marker.get("status") != "worker-loss-injected"
        or raw_marker.get("task_key") != loss_key
        or {
            name: raw_marker.get(name)
            for name in ("worker_address", "hostname", "pid", "slurm_job_id")
        }
        != loss_identity
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance loss-marker summary is not checksum-bound to raw marker"
        )
    scheduler_reschedule = replacement["failed_attempt"] is None
    recovery_mode = _fine_nonempty_string(
        replacement["recovery_mode"], f"{location}.recovery_mode"
    )
    failed = (
        None
        if scheduler_reschedule
        else _fine_object(
            replacement["failed_attempt"],
            keys=frozenset({"task_key", "attempt", "status"}),
            location=f"{location}.failed_attempt",
        )
    )
    departed = _fine_worker_identity(
        replacement["departed_worker"], f"{location}.departed_worker"
    )
    activated = _fine_worker_identity(
        replacement["replacement_worker"], f"{location}.replacement_worker"
    )
    retry = _fine_object(
        replacement["retry"],
        keys=frozenset(
            {
                "task_key",
                "attempt",
                "status",
                "worker_address",
                "hostname",
                "pid",
                "slurm_job_id",
            }
        ),
        location=f"{location}.retry",
    )
    retry_identity = _fine_worker_identity(
        {
            name: retry[name]
            for name in ("worker_address", "hostname", "pid", "slurm_job_id")
        },
        f"{location}.retry.identity",
    )
    liveness_path = replacement["replacement_liveness_path"]
    if liveness_path not in {"retry", "later-task"}:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance replacement liveness path is not canonical"
        )
    liveness: dict[str, object] | None = None
    liveness_identity: dict[str, object] | None = None
    if liveness_path == "retry":
        if replacement["liveness"] is not None:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance retry liveness path must not claim another task"
            )
    else:
        liveness = _fine_object(
            replacement["liveness"],
            keys=frozenset(
                {
                    "task_key",
                    "status",
                    "worker_address",
                    "hostname",
                    "pid",
                    "slurm_job_id",
                }
            ),
            location=f"{location}.liveness",
        )
        liveness_identity = _fine_worker_identity(
            {
                name: liveness[name]
                for name in ("worker_address", "hostname", "pid", "slurm_job_id")
            },
            f"{location}.liveness.identity",
        )
    owned = _fine_exact_strings(
        replacement["owned_job_ids"], f"{location}.owned_job_ids"
    )
    outcome_owned = _validated_owned_outcome(
        raw_outcome,
        operation="fine-worker-replacement",
        status="completed",
    )
    expected_owned = {
        cast(str, departed["slurm_job_id"]),
        cast(str, activated["slurm_job_id"]),
        cast(str, retry_identity["slurm_job_id"]),
    }
    if liveness_identity is not None:
        expected_owned.add(cast(str, liveness_identity["slurm_job_id"]))
    recovered_attempt = 1 if scheduler_reschedule else 2
    if (
        recovery_mode
        != ("scheduler-reschedule" if scheduler_reschedule else "coordinator-retry")
        or (
            failed is not None
            and (
                failed["task_key"] != loss_key
                or failed["attempt"] != 1
                or failed["status"] != "failed"
            )
        )
        or departed != loss_identity
        or activated == departed
        or retry["task_key"] != loss_key
        or retry["attempt"] != recovered_attempt
        or retry["status"] != "completed"
        or tuple(owned) != outcome_owned
        or replacement["cleanup_outcome"] != "empty"
        or not expected_owned <= set(owned)
        or (liveness_path == "retry" and retry_identity != activated)
        or (
            liveness_path == "later-task"
            and (
                liveness is None
                or liveness_identity != activated
                or liveness["task_key"] == loss_key
                or liveness["status"] not in {"completed", "reused"}
            )
        )
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance worker loss, departure, replacement activation, "
            "exact retry, liveness, or cleanup lineage is inconsistent"
        )
    if not isinstance(raw_evidence, Mapping) or not isinstance(raw_queue, Mapping):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance replacement raw evidence must be canonical objects"
        )
    graph = raw_evidence.get("fine_graph")
    if not isinstance(graph, Mapping):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance replacement evidence lacks a fine graph"
        )
    events = graph.get("scale_events")
    attempts = graph.get("attempt_counts")
    placements = graph.get("placements")
    cleanup = raw_evidence.get("cleanup")
    if not all(isinstance(item, list) for item in (events, attempts, placements)):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance replacement evidence lacks canonical event records"
        )

    def raw_identity(item: object) -> dict[str, object]:
        if not isinstance(item, Mapping):
            raise HpcWorkflowAcceptanceError(
                "fine acceptance replacement probe/placement is malformed"
            )
        return _fine_worker_identity(
            {
                name: item.get(name)
                for name in ("worker_address", "hostname", "pid", "slurm_job_id")
            },
            f"{location}.raw_identity",
        )

    activation_index: int | None = None
    departure_index: int | None = None
    for index, event in enumerate(cast(list[object], events)):
        if not isinstance(event, Mapping):
            raise HpcWorkflowAcceptanceError(
                "fine acceptance replacement scale event is malformed"
            )
        activated_probes = event.get("activated_probes")
        if not isinstance(activated_probes, list):
            raise HpcWorkflowAcceptanceError(
                "fine acceptance replacement scale probes are malformed"
            )
        identities = [raw_identity(item) for item in activated_probes]
        if loss_identity in identities and activation_index is None:
            activation_index = index
        if (
            activation_index is not None
            and index > activation_index
            and cast(str, loss_identity["worker_address"])
            in cast(list[object], event.get("departed", []))
            and activated in identities
        ):
            departure_index = index
            break
    attempt_by_key = {
        cast(str, item["task_key"]): item["attempts"]
        for item in cast(list[dict[str, object]], attempts)
        if isinstance(item, dict)
        and type(item.get("task_key")) is str
        and type(item.get("attempts")) is int
    }
    target_placements = [
        item
        for item in cast(list[dict[str, object]], placements)
        if isinstance(item, dict) and item.get("task_key") == loss_key
    ]
    liveness_placements = (
        [
            item
            for item in cast(list[dict[str, object]], placements)
            if isinstance(item, dict)
            and liveness is not None
            and item.get("task_key") == liveness["task_key"]
        ]
        if liveness_path == "later-task"
        else []
    )
    recovered_placement = (
        target_placements[-1] if len(target_placements) == recovered_attempt else None
    )
    scheduler_reschedule_spans_loss = False
    if scheduler_reschedule and recovered_placement is not None:
        submitted = recovered_placement.get("coordinator_submitted_at")
        ready = recovered_placement.get("coordinator_ready_at")
        injected = (
            raw_marker.get("injected_at") if isinstance(raw_marker, Mapping) else None
        )
        if all(type(value) is str for value in (submitted, ready, injected)):
            scheduler_reschedule_spans_loss = (
                _timestamp(cast(str, submitted))
                <= _timestamp(cast(str, injected))
                <= _timestamp(cast(str, ready))
            )
    if (
        activation_index is None
        or departure_index is None
        or attempt_by_key.get(loss_key) != recovered_attempt
        or len(target_placements) != recovered_attempt
        or [item.get("attempt") for item in target_placements]
        != list(range(1, recovered_attempt + 1))
        or (
            not scheduler_reschedule
            and (
                target_placements[0].get("status") != "failed"
                or target_placements[0].get("placement_accepted") is not False
            )
        )
        or recovered_placement is None
        or recovered_placement.get("status") != "completed"
        or recovered_placement.get("placement_accepted") is not True
        or raw_identity(recovered_placement) != retry_identity
        or (scheduler_reschedule and not scheduler_reschedule_spans_loss)
        or (liveness_path == "retry" and raw_identity(recovered_placement) != activated)
        or (
            liveness_path == "later-task"
            and (
                not liveness_placements
                or liveness_placements[-1].get("status") not in {"completed", "reused"}
                or liveness_placements[-1].get("placement_accepted") is not True
                or raw_identity(liveness_placements[-1]) != activated
                or cast(str, liveness_placements[-1].get("coordinator_ready_at", ""))
                <= cast(str, recovered_placement.get("coordinator_ready_at", ""))
            )
        )
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance replacement report is not derived from canonical "
            "probe, scale, attempt, placement, liveness, and queue evidence"
        )
    _validate_empty_owned_queue(
        raw_queue,
        operation="fine-worker-replacement",
    )
    _validate_executor_cleanup(
        cleanup,
        operation="fine-worker-replacement",
    )
    return (
        (evidence_relative, evidence_sha),
        (outcome_relative, outcome_sha),
        (queue_relative, queue_sha),
        (marker_relative, marker_sha),
    )


def validate_fine_acceptance_report(
    value: object,
    *,
    evidence_root: Path,
) -> dict[str, object]:
    """Decode and fail closed on one complete fine-SLURM acceptance report.

    This portable validator performs no scheduler contact. It accepts only a complete
    exact-schema report whose external evidence files are confined to and checksum-
    verified beneath the caller-owned evidence root.
    """
    report = _fine_object(value, keys=_FINE_REPORT_KEYS, location="report")
    if report["schema_version"] != 1 or report["status"] != "completed":
        raise HpcWorkflowAcceptanceError(
            "fine acceptance report must be schema 1 and completed"
        )
    try:
        owned_root = Path(
            _fine_nonempty_string(report["owned_root"], "owned_root")
        ).resolve(strict=True)
        expected_root = evidence_root.resolve(strict=True)
    except OSError as error:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance owned root must be an existing path"
        ) from error
    if (
        owned_root != expected_root
        or not owned_root.is_dir()
        or owned_root.is_symlink()
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance owned root does not match the caller-owned evidence root"
        )

    identities = _fine_object(
        report["identities"],
        keys=frozenset(
            {
                "git_commit",
                "package_versions",
                "scientific_fingerprint",
                "coarse_execution_fingerprint",
                "fine_execution_fingerprint",
                "coarse_profile",
                "fine_profile",
                "coarse_profile_sha256",
                "fine_profile_sha256",
                "inputs",
                "input_checksums",
            }
        ),
        location="identities",
    )
    commit = _fine_nonempty_string(identities["git_commit"], "identities.git_commit")
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance commit identity must be one exact lowercase Git commit"
        )
    for name in (
        "scientific_fingerprint",
        "coarse_execution_fingerprint",
        "fine_execution_fingerprint",
    ):
        _fine_sha256(identities[name], f"identities.{name}")
    packages = identities["package_versions"]
    if (
        type(packages) is not dict
        or not packages
        or any(
            type(name) is not str or not name or type(version) is not str or not version
            for name, version in cast(dict[object, object], packages).items()
        )
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance package identities are malformed"
        )
    coarse_profile_identity = _fine_file_identity(
        identities["coarse_profile"], "identities.coarse_profile"
    )
    fine_profile_identity = _fine_file_identity(
        identities["fine_profile"], "identities.fine_profile"
    )
    _verify_file_identity_and_sha256(
        coarse_profile_identity,
        identities["coarse_profile_sha256"],
        "identities.coarse_profile",
    )
    _verify_file_identity_and_sha256(
        fine_profile_identity,
        identities["fine_profile_sha256"],
        "identities.fine_profile",
    )
    inputs = _fine_array(identities["inputs"], "identities.inputs")
    if not inputs:
        raise HpcWorkflowAcceptanceError("fine acceptance input identities are empty")
    input_paths: list[str] = []
    for index, item in enumerate(inputs):
        input_paths.append(
            cast(
                str,
                _fine_file_identity(item, f"identities.inputs[{index}]")["path"],
            )
        )
    if input_paths != sorted(set(input_paths)):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance input identities must be unique and canonical"
        )
    input_checksums = _fine_array(
        identities["input_checksums"], "identities.input_checksums"
    )
    checksum_paths: list[str] = []
    for index, raw in enumerate(input_checksums):
        checksum = _fine_object(
            raw,
            keys=frozenset({"path", "sha256"}),
            location=f"identities.input_checksums[{index}]",
        )
        checksum_paths.append(
            _fine_nonempty_string(
                checksum["path"],
                f"identities.input_checksums[{index}].path",
            )
        )
        _fine_sha256(
            checksum["sha256"],
            f"identities.input_checksums[{index}].sha256",
        )
    if checksum_paths != input_paths:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance input checksums do not exactly cover input identities"
        )
    checksum_by_path = {
        cast(str, item["path"]): item["sha256"]
        for item in cast(list[dict[str, object]], input_checksums)
    }
    for index, item in enumerate(inputs):
        identity = _fine_file_identity(item, f"identities.inputs[{index}]")
        _verify_file_identity_and_sha256(
            identity,
            checksum_by_path[cast(str, identity["path"])],
            f"identities.inputs[{index}]",
        )

    scientific = _fine_object(
        report["scientific"],
        keys=frozenset(
            {
                "coarse_fingerprint",
                "fine_fingerprint",
                "public_artifacts_coarse",
                "public_artifacts_fine",
                "tier1",
                "tier2",
            }
        ),
        location="scientific",
    )
    fingerprints = (
        scientific["coarse_fingerprint"],
        scientific["fine_fingerprint"],
        identities["scientific_fingerprint"],
    )
    if len(set(fingerprints)) != 1:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance coarse and fine scientific fingerprints differ"
        )
    coarse_artifacts = _fine_artifacts(
        scientific["public_artifacts_coarse"],
        "scientific.public_artifacts_coarse",
    )
    fine_artifacts = _fine_artifacts(
        scientific["public_artifacts_fine"],
        "scientific.public_artifacts_fine",
    )
    if tuple(item["relative_path"] for item in coarse_artifacts) != tuple(
        item["relative_path"] for item in fine_artifacts
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance public artifact structure differs"
        )
    tier1 = _fine_object(
        scientific["tier1"],
        keys=frozenset(
            {
                "evidence_file",
                "metric_absolute_tolerance",
                "maximum_metric_absolute_difference",
                "structure_equal",
            }
        ),
        location="scientific.tier1",
    )
    maximum_difference = _fine_number(
        tier1["maximum_metric_absolute_difference"],
        "scientific.tier1.maximum_metric_absolute_difference",
    )
    tier1_reference, tier1_sha256, tier1_payload = _evidence_reference(
        tier1["evidence_file"],
        owned_root=owned_root,
        location="scientific.tier1.evidence_file",
    )
    from hpc.workflow_acceptance import _validated_scientific_comparison_evidence

    validated_tier1 = _validated_scientific_comparison_evidence(
        tier1_payload,
        require_close=False,
        metric_absolute_tolerance=0.01,
    )
    if abs(_maximum_metric_difference(tier1_payload) - maximum_difference) > 1e-12:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance Tier 1 summary does not match full equivalence evidence"
        )
    if (
        tier1["metric_absolute_tolerance"] != 0.01
        or maximum_difference > 0.01
        or tier1["structure_equal"] is not True
        or validated_tier1["scientific_fingerprint"]
        != identities["scientific_fingerprint"]
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance Tier 1 metric or structure gate failed"
        )
    tier2 = _fine_object(
        scientific["tier2"],
        keys=frozenset(
            {
                "evidence_file",
                "seed",
                "native_threads",
                "affine_exact",
                "local_warp_exact",
                "full_warp_exact",
                "diagnostic_exact",
                "metrics_exact",
                "channels",
            }
        ),
        location="scientific.tier2",
    )
    if (
        tier2["seed"] != "20260722"
        or tier2["native_threads"] != 1
        or any(
            tier2[name] is not True
            for name in (
                "affine_exact",
                "local_warp_exact",
                "full_warp_exact",
                "diagnostic_exact",
                "metrics_exact",
            )
        )
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance Tier 2 exact seeded one-thread gate failed"
        )
    tier2_reference, tier2_sha256, tier2_payload = _evidence_reference(
        tier2["evidence_file"],
        owned_root=owned_root,
        location="scientific.tier2.evidence_file",
    )
    from hpc.workflow_acceptance import _validated_deterministic_parity_evidence

    deterministic = _validated_deterministic_parity_evidence(
        tier2_payload,
        require_fine_internal=True,
    )
    strict = cast(dict[str, object], deterministic["strict_parity"])
    strict_rounds = cast(list[dict[str, object]], strict["rounds"])
    expected_channel_paths: list[str] = []
    for round_item in strict_rounds:
        round_number = round_item.get("round")
        raw_channels = round_item.get("channels")
        if type(round_number) is not int or not isinstance(raw_channels, list):
            raise HpcWorkflowAcceptanceError(
                "fine acceptance Tier 2 raw round/channel identities are malformed"
            )
        for channel_index, raw_channel in enumerate(raw_channels):
            if not isinstance(raw_channel, Mapping) or not isinstance(
                source := raw_channel.get("source"), str
            ):
                raise HpcWorkflowAcceptanceError(
                    "fine acceptance Tier 2 raw channel source is malformed"
                )
            expected_channel_paths.append(
                f"round-{round_number:03d}/channel-{channel_index:04d}/"
                f"{PurePosixPath(source).name}"
            )
    channels = _fine_array(tier2["channels"], "scientific.tier2.channels")
    if not channels:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance Tier 2 contains no channel comparisons"
        )
    channel_paths: list[str] = []
    for index, raw in enumerate(channels):
        channel = _fine_object(
            raw,
            keys=frozenset({"relative_path", "exact"}),
            location=f"scientific.tier2.channels[{index}]",
        )
        channel_paths.append(
            _fine_relative_path(
                channel["relative_path"],
                f"scientific.tier2.channels[{index}].relative_path",
            )
        )
        if channel["exact"] is not True:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance Tier 2 channel comparison is not exact"
            )
    if channel_paths != sorted(set(channel_paths)):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance Tier 2 channels are duplicate or noncanonical"
        )
    if channel_paths != sorted(expected_channel_paths):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance Tier 2 channels do not match raw deterministic identities"
        )

    fixed = _fine_object(
        report["fixed_grid"],
        keys=frozenset(
            {
                "shape_zyx",
                "origin_xyz",
                "spacing_xyz",
                "direction",
                "prepared_paths",
                "owned_roots",
            }
        ),
        location="fixed_grid",
    )
    shape = _fine_array(fixed["shape_zyx"], "fixed_grid.shape_zyx")
    origin = _fine_array(fixed["origin_xyz"], "fixed_grid.origin_xyz")
    spacing = _fine_array(fixed["spacing_xyz"], "fixed_grid.spacing_xyz")
    direction = _fine_array(fixed["direction"], "fixed_grid.direction")
    if (
        len(shape) != 3
        or any(type(item) is not int or cast(int, item) < 1 for item in shape)
        or len(origin) != 3
        or len(spacing) != 3
        or any(
            not isinstance(_fine_number(item, "fixed-grid origin", minimum=None), float)
            for item in origin
        )
        or any(
            _fine_number(item, "fixed-grid spacing", minimum=1e-300) <= 0
            for item in spacing
        )
        or len(direction) != 3
        or any(
            len(row := _fine_array(item, "fixed-grid direction row")) != 3
            or any(
                not isinstance(
                    _fine_number(
                        component,
                        "fixed-grid direction component",
                        minimum=None,
                    ),
                    float,
                )
                for component in row
            )
            for item in direction
        )
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance fixed-grid geometry is malformed"
        )
    prepared_paths = _fine_exact_strings(
        fixed["prepared_paths"], "fixed_grid.prepared_paths"
    )
    if not prepared_paths:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance fixed-grid prepared paths are empty"
        )
    owned_roots = tuple(
        Path(encoded).resolve(strict=True)
        for encoded in _fine_exact_strings(
            fixed["owned_roots"], "fixed_grid.owned_roots"
        )
    )
    if not owned_roots or any(
        not root.is_dir() or root.is_symlink() for root in owned_roots
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance fixed-grid owned roots are malformed"
        )
    for encoded in prepared_paths:
        path = Path(encoded)
        try:
            resolved = path.resolve(strict=True)
        except (OSError, ValueError) as error:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance prepared path is missing"
            ) from error
        if not any(resolved.is_relative_to(root) for root in owned_roots):
            raise HpcWorkflowAcceptanceError(
                "fine acceptance prepared path escapes every owned root"
            )
    fixed_geometry = {
        "shape": shape,
        "origin": origin,
        "spacing": spacing,
        "direction": direction,
    }
    for round_item in cast(list[dict[str, object]], validated_tier1["rounds"]):
        registered = cast(dict[str, object], round_item["registered"])
        comparisons = [
            registered,
            *[
                cast(dict[str, object], channel["comparison"])
                for channel in cast(list[dict[str, object]], round_item["channels"])
            ],
        ]
        if any(item["geometry"] != fixed_geometry for item in comparisons):
            raise HpcWorkflowAcceptanceError(
                "fine acceptance scientific comparison geometry differs from "
                "the declared fixed grid"
            )

    placement = _fine_object(
        report["placement"],
        keys=frozenset(
            {
                "evidence_file",
                "worker_cap",
                "maximum_connected_workers",
                "scale_events",
                "fair_ready_rounds",
                "fair_dispatch_order",
                "tasks",
            }
        ),
        location="placement",
    )
    primary_reference, primary_sha256, primary_payload = _evidence_reference(
        placement["evidence_file"],
        owned_root=owned_root,
        location="placement.evidence_file",
    )
    derived_placement, derived_fine_wall = _derive_primary_evidence(primary_payload)
    placement_without_reference = {
        name: value for name, value in placement.items() if name != "evidence_file"
    }
    if placement_without_reference != derived_placement:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance primary placement summary differs from raw evidence"
        )
    cap = placement["worker_cap"]
    maximum_workers = placement["maximum_connected_workers"]
    if (
        type(cap) is not int
        or cap < 2
        or type(maximum_workers) is not int
        or maximum_workers < 2
        or maximum_workers > cap
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance placement exceeds or fails to reach the worker cap"
        )
    scale_events = _fine_array(placement["scale_events"], "placement.scale_events")
    if not scale_events or any(
        type(item) is not int or cast(int, item) < 1 for item in scale_events
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance demand scale events are not canonical"
        )
    fair_rounds = _fine_array(
        placement["fair_ready_rounds"], "placement.fair_ready_rounds"
    )
    if len(fair_rounds) < 2 or any(type(item) is not int for item in fair_rounds):
        raise HpcWorkflowAcceptanceError("fine acceptance lacks two fair ready rounds")
    fair_order = _fine_exact_strings(
        placement["fair_dispatch_order"], "placement.fair_dispatch_order"
    )
    try:
        dispatched_rounds = tuple(
            int(key.split(":", 1)[0].removeprefix("round-")) for key in fair_order
        )
    except ValueError as error:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance fair dispatch keys are malformed"
        ) from error
    if len(dispatched_rounds) < len(fair_rounds) or tuple(
        dispatched_rounds[: len(fair_rounds)]
    ) != tuple(fair_rounds):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance does not prove round-robin fair placement"
        )
    tasks = _fine_array(placement["tasks"], "placement.tasks")
    task_keys: list[str] = []
    by_kind: dict[str, list[dict[str, object]]] = {"chunk": [], "channel": []}
    task_schema = frozenset(
        {
            "task_key",
            "kind",
            "round",
            "job_id",
            "hostname",
            "worker",
            "coordinator_submitted_at",
            "coordinator_ready_at",
            "duration_seconds",
        }
    )
    for index, raw in enumerate(tasks):
        task = _fine_object(raw, keys=task_schema, location=f"placement.tasks[{index}]")
        key = _fine_nonempty_string(
            task["task_key"], f"placement.tasks[{index}].task_key"
        )
        kind = _fine_nonempty_string(task["kind"], f"placement.tasks[{index}].kind")
        if kind not in by_kind or f":{kind}:" not in key:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance placement task kind is forged"
            )
        if type(task["round"]) is not int or cast(int, task["round"]) < 1:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance placement task round is malformed"
            )
        for name in ("job_id", "hostname", "worker"):
            _fine_nonempty_string(task[name], f"placement.tasks[{index}].{name}")
        submitted = _timestamp(cast(str, task["coordinator_submitted_at"]))
        ready = _timestamp(cast(str, task["coordinator_ready_at"]))
        _fine_number(
            task["duration_seconds"],
            f"placement.tasks[{index}].duration_seconds",
        )
        if ready < submitted:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance coordinator observed task readiness before submission"
            )
        task_keys.append(key)
        by_kind[kind].append(task)
    if len(task_keys) != len(set(task_keys)):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance placement contains duplicate task identities"
        )
    for kind, kind_tasks in by_kind.items():
        concurrent = False
        for left_index, left in enumerate(kind_tasks):
            for right in kind_tasks[left_index + 1 :]:
                if (
                    left["job_id"] != right["job_id"]
                    and left["hostname"] != right["hostname"]
                    and max(
                        _timestamp(cast(str, left["coordinator_submitted_at"])),
                        _timestamp(cast(str, right["coordinator_submitted_at"])),
                    )
                    < min(
                        _timestamp(cast(str, left["coordinator_ready_at"])),
                        _timestamp(cast(str, right["coordinator_ready_at"])),
                    )
                ):
                    concurrent = True
        if not concurrent:
            raise HpcWorkflowAcceptanceError(
                f"fine acceptance does not prove two concurrent {kind} jobs and hosts"
            )

    durability = _fine_object(
        report["durability"],
        keys=frozenset(
            {
                "chunk_failure_resume",
                "channel_failure_resume",
                "cancellation_resume",
                "worker_replacement",
            }
        ),
        location="durability",
    )
    chunk_resume_references = _validate_fine_resume(
        durability["chunk_failure_resume"],
        "durability.chunk_failure_resume",
        "chunk",
        owned_root=owned_root,
    )
    channel_resume_references = _validate_fine_resume(
        durability["channel_failure_resume"],
        "durability.channel_failure_resume",
        "channel",
        owned_root=owned_root,
    )
    cancellation = _fine_object(
        durability["cancellation_resume"],
        keys=frozenset(
            {
                "status",
                "structured_cancellation",
                "blocked_task_key",
                "committed_peer_key",
                "committed_peer_file",
                "ready_marker_file",
                "trigger_file",
                "execution_evidence_file",
                "outcome_file",
                "queue_after_file",
                "preserved_artifacts",
                "resume_reused_keys",
                "resume_recomputed_keys",
                "owned_job_ids",
                "cleanup_outcome",
            }
        ),
        location="durability.cancellation_resume",
    )
    owned_jobs = _fine_exact_strings(
        cancellation["owned_job_ids"],
        "durability.cancellation_resume.owned_job_ids",
    )
    blocked_task_key = _fine_nonempty_string(
        cancellation["blocked_task_key"],
        "durability.cancellation_resume.blocked_task_key",
    )
    committed_peer_key = _fine_nonempty_string(
        cancellation["committed_peer_key"],
        "durability.cancellation_resume.committed_peer_key",
    )
    reused_keys = _fine_exact_strings(
        cancellation["resume_reused_keys"],
        "durability.cancellation_resume.resume_reused_keys",
    )
    recomputed_keys = _fine_exact_strings(
        cancellation["resume_recomputed_keys"],
        "durability.cancellation_resume.resume_recomputed_keys",
    )
    cancellation_references: list[tuple[str, str]] = []
    cancellation_payloads: dict[str, object] = {}
    for name in (
        "committed_peer_file",
        "ready_marker_file",
        "trigger_file",
        "execution_evidence_file",
        "outcome_file",
        "queue_after_file",
    ):
        relative, digest, payload = _evidence_reference(
            cancellation[name],
            owned_root=owned_root,
            location=f"durability.cancellation_resume.{name}",
        )
        cancellation_references.append((relative, digest))
        cancellation_payloads[name] = payload
    committed_payload = cancellation_payloads["committed_peer_file"]
    ready_payload = cancellation_payloads["ready_marker_file"]
    trigger_payload = cancellation_payloads["trigger_file"]
    cancellation_execution = cancellation_payloads["execution_evidence_file"]
    cancellation_outcome = cancellation_payloads["outcome_file"]
    cancellation_queue_payload = cancellation_payloads["queue_after_file"]
    outcome_owned = _validated_owned_outcome(
        cancellation_outcome,
        operation="fine-cancellation",
        status="cancelled",
    )
    if (
        cancellation["status"] != "cancelled"
        or cancellation["structured_cancellation"] is not True
        or blocked_task_key == committed_peer_key
        or blocked_task_key.split(":", 1)[0] == committed_peer_key.split(":", 1)[0]
        or committed_peer_key not in reused_keys
        or blocked_task_key not in recomputed_keys
        or not owned_jobs
        or tuple(owned_jobs) != outcome_owned
        or cancellation["cleanup_outcome"] != "empty"
        or not reused_keys
        or not recomputed_keys
        or not isinstance(committed_payload, Mapping)
        or committed_payload.get("task_key") != committed_peer_key
        or committed_payload.get("status") not in {"completed", "reused"}
        or not committed_payload.get("artifacts")
        or ready_payload
        != {
            "blocked_task_key": blocked_task_key,
            "committed_peer_key": committed_peer_key,
            "status": "waiting-for-coordinator-interruption",
        }
        or not isinstance(trigger_payload, Mapping)
        or trigger_payload.get("blocked_task_key") != blocked_task_key
        or trigger_payload.get("committed_peer_key") != committed_peer_key
        or trigger_payload.get("status") != "interrupted"
        or trigger_payload.get("signal") != "SIGINT"
        or trigger_payload.get("exit_code") != 130
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance cancellation, replacement, liveness, resume, "
            "or cleanup gate failed"
        )
    _validate_empty_owned_queue(
        cancellation_queue_payload,
        operation="fine-cancellation",
    )
    if (
        not isinstance(cancellation_execution, Mapping)
        or not isinstance(
            cancellation_graph := cancellation_execution.get("fine_graph"),
            Mapping,
        )
        or cancellation_graph.get("cancelled") is not True
        or cancellation["structured_cancellation"] is not True
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance cancellation execution evidence is not structured"
        )
    raw_cancellation_placements = cancellation_graph.get("placements")
    if not isinstance(raw_cancellation_placements, list):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance cancellation placements are malformed"
        )
    placement_job_ids = {
        cast(str, item["slurm_job_id"])
        for item in raw_cancellation_placements
        if isinstance(item, Mapping)
        and isinstance(item.get("slurm_job_id"), str)
        and item["slurm_job_id"]
    }
    if not placement_job_ids <= set(owned_jobs):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance cancellation placement job is absent from owned outcome"
        )
    _validate_executor_cleanup(
        cancellation_execution.get("cleanup"),
        operation="fine-cancellation",
    )
    _fine_artifacts(
        cancellation["preserved_artifacts"],
        "durability.cancellation_resume.preserved_artifacts",
    )
    replacement_references = _validate_worker_replacement(
        durability["worker_replacement"], owned_root=owned_root
    )

    performance = _fine_object(
        report["performance"],
        keys=frozenset(
            {
                "coarse_nonlinear_wall_seconds",
                "coarse_timing_policy",
                "coarse_timing_files",
                "fine_nonlinear_wall_seconds",
                "clock_policy",
                "end_to_end_wall_seconds",
                "task_seconds",
                "node_hours",
                "peak_memory_bytes",
                "startup_overhead_seconds",
                "worker_utilization",
                "chunk_duration_seconds",
                "channel_duration_seconds",
            }
        ),
        location="performance",
    )
    coarse_wall = _fine_number(
        performance["coarse_nonlinear_wall_seconds"],
        "performance.coarse_nonlinear_wall_seconds",
        minimum=1e-300,
    )
    fine_wall = _fine_number(
        performance["fine_nonlinear_wall_seconds"],
        "performance.fine_nonlinear_wall_seconds",
        minimum=1e-300,
    )
    timing_references: list[tuple[str, str]] = []
    timing_records: list[tuple[str, object]] = []
    for index, reference in enumerate(
        _fine_array(
            performance["coarse_timing_files"],
            "performance.coarse_timing_files",
        )
    ):
        relative, digest, payload = _evidence_reference(
            reference,
            owned_root=owned_root,
            location=f"performance.coarse_timing_files[{index}]",
        )
        timing_references.append((relative, digest))
        timing_records.append((relative, payload))
    derived_coarse_wall, derived_coarse_policy = _coarse_nonlinear_lower_bound(
        timing_records
    )
    if (
        abs(coarse_wall - derived_coarse_wall) > 1e-9
        or performance["coarse_timing_policy"] != derived_coarse_policy
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance coarse timing summary differs from all per-round calls"
        )
    if abs(fine_wall - derived_fine_wall) > 1e-9:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance fine timing summary differs from primary evidence"
        )
    if performance["clock_policy"] != (
        "coordinator-fine-upper-bound-vs-worker-monotonic-coarse-lower-bound"
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance performance clock policy is not skew-safe"
        )
    if fine_wall > 0.8 * coarse_wall:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance does not satisfy the 20% nonlinear wall-time reduction"
        )
    for name in (
        "end_to_end_wall_seconds",
        "task_seconds",
        "node_hours",
        "peak_memory_bytes",
        "startup_overhead_seconds",
    ):
        _fine_number(performance[name], f"performance.{name}")
    utilization = _fine_number(
        performance["worker_utilization"], "performance.worker_utilization"
    )
    if utilization > 1.0:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance worker utilization exceeds one"
        )
    _validate_duration_summary(
        performance["chunk_duration_seconds"],
        location="performance.chunk_duration_seconds",
        expected_count=len(by_kind["chunk"]),
    )
    _validate_duration_summary(
        performance["channel_duration_seconds"],
        location="performance.channel_duration_seconds",
        expected_count=len(by_kind["channel"]),
    )

    evidence_files = _fine_array(report["evidence_files"], "evidence_files")
    if not evidence_files:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance contains no evidence checksums"
        )
    evidence_paths: list[str] = []
    evidence_digests: dict[str, str] = {}
    for index, raw in enumerate(evidence_files):
        evidence = _fine_object(
            raw,
            keys=frozenset({"relative_path", "sha256"}),
            location=f"evidence_files[{index}]",
        )
        relative = _fine_relative_path(
            evidence["relative_path"], f"evidence_files[{index}].relative_path"
        )
        claimed = _fine_sha256(evidence["sha256"], f"evidence_files[{index}].sha256")
        path = owned_root / PurePosixPath(relative)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(owned_root)
        except (OSError, ValueError) as error:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance evidence file escapes or is missing from owned root"
            ) from error
        if not resolved.is_file() or resolved.is_symlink():
            raise HpcWorkflowAcceptanceError(
                "fine acceptance evidence checksum target is not a real file"
            )
        observed = _file_sha256(resolved)
        if observed != claimed:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance evidence checksum is forged or stale"
            )
        evidence_paths.append(relative)
        evidence_digests[relative] = claimed
    if evidence_paths != sorted(set(evidence_paths)):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance evidence checksums contain duplicate or noncanonical paths"
        )
    for relative, digest in (
        (tier1_reference, tier1_sha256),
        (tier2_reference, tier2_sha256),
        (primary_reference, primary_sha256),
        *timing_references,
        *replacement_references,
        *cancellation_references,
        *chunk_resume_references,
        *channel_resume_references,
    ):
        if evidence_digests.get(relative) != digest:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance Tier evidence is not bound into evidence checksums"
            )
    return json.loads(json.dumps(report, allow_nan=False, sort_keys=True))


def fine_evidence_checksums(
    evidence_root: Path,
    paths: tuple[Path, ...],
) -> list[dict[str, str]]:
    """Build canonical SHA-256 evidence entries below one caller-owned root."""
    try:
        root = evidence_root.resolve(strict=True)
    except OSError as error:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance evidence root must exist"
        ) from error
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        try:
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(root).as_posix()
        except (OSError, ValueError) as error:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance checksum input escapes or is missing from owned root"
            ) from error
        if (
            not resolved.is_file()
            or resolved.is_symlink()
            or relative in seen
            or relative != PurePosixPath(relative).as_posix()
        ):
            raise HpcWorkflowAcceptanceError(
                "fine acceptance checksum inputs must be unique canonical real files"
            )
        seen.add(relative)
        entries.append(
            {
                "relative_path": relative,
                "sha256": _file_sha256(resolved),
            }
        )
    if not entries:
        raise HpcWorkflowAcceptanceError(
            "fine acceptance requires at least one evidence checksum"
        )
    return sorted(entries, key=lambda item: item["relative_path"])


def build_fine_acceptance_report(
    *,
    identities: Mapping[str, object],
    scientific: Mapping[str, object],
    fixed_grid: Mapping[str, object],
    placement: Mapping[str, object],
    durability: Mapping[str, object],
    performance: Mapping[str, object],
    evidence_root: Path,
    evidence_paths: tuple[Path, ...],
) -> dict[str, object]:
    """Build and immediately validate one complete external acceptance report."""
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "identities": dict(identities),
        "scientific": dict(scientific),
        "fixed_grid": dict(fixed_grid),
        "placement": dict(placement),
        "durability": dict(durability),
        "performance": dict(performance),
        "owned_root": str(evidence_root.resolve(strict=True)),
        "evidence_files": fine_evidence_checksums(evidence_root, evidence_paths),
    }
    return validate_fine_acceptance_report(report, evidence_root=evidence_root)


__all__ = [
    "build_fine_acceptance_report",
    "fine_acceptance_manifest",
    "fine_acceptance_profiles",
    "fine_evidence_checksums",
    "fine_internal_parity_evidence",
    "validate_fine_acceptance_report",
]
