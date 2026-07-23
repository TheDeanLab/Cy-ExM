"""Strict portable builders and validators for fine-SLURM acceptance evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast

from cyexm.config.models import ExecutionProfile
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
    if coarse.paths.output_root == fine.paths.output_root:
        raise HpcWorkflowAcceptanceError(
            "coarse and fine acceptance roots must be distinct"
        )
    return coarse, fine


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


def _validate_fine_resume(value: object, location: str, expected_kind: str) -> None:
    resume = _fine_object(
        value,
        keys=frozenset(
            {
                "injected_task_key",
                "failure_status",
                "settled_sibling_keys",
                "preserved_artifacts",
                "resume_reused_keys",
                "resume_recomputed_keys",
                "ordinary_artifact_store_reuse",
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
    settled = _fine_exact_strings(
        resume["settled_sibling_keys"], f"{location}.settled_sibling_keys"
    )
    reused = _fine_exact_strings(
        resume["resume_reused_keys"], f"{location}.resume_reused_keys"
    )
    recomputed = _fine_exact_strings(
        resume["resume_recomputed_keys"], f"{location}.resume_recomputed_keys"
    )
    if (
        not settled
        or injected not in recomputed
        or set(reused) & set(recomputed)
        or not set(settled) <= set(reused)
        or resume["ordinary_artifact_store_reuse"] is not True
    ):
        raise HpcWorkflowAcceptanceError(
            f"fine acceptance {location} does not prove exact ArtifactStore resume"
        )
    _fine_artifacts(resume["preserved_artifacts"], f"{location}.preserved_artifacts")


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
    if (
        tier1["metric_absolute_tolerance"] != 0.01
        or maximum_difference > 0.01
        or tier1["structure_equal"] is not True
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance Tier 1 metric or structure gate failed"
        )
    tier2 = _fine_object(
        scientific["tier2"],
        keys=frozenset(
            {
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

    placement = _fine_object(
        report["placement"],
        keys=frozenset(
            {
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
    scale_requests = [cast(int, item) for item in scale_events]
    if (
        not scale_events
        or any(type(item) is not int or cast(int, item) < 1 for item in scale_events)
        or scale_requests != sorted(set(scale_requests))
        or cast(int, scale_events[-1]) != maximum_workers
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
            "started_at",
            "ended_at",
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
        started = _timestamp(cast(str, task["started_at"]))
        ended = _timestamp(cast(str, task["ended_at"]))
        duration = _fine_number(
            task["duration_seconds"],
            f"placement.tasks[{index}].duration_seconds",
        )
        if ended < started or abs((ended - started).total_seconds() - duration) > 1e-6:
            raise HpcWorkflowAcceptanceError(
                "fine acceptance placement task timestamps contradict duration"
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
                        _timestamp(cast(str, left["started_at"])),
                        _timestamp(cast(str, right["started_at"])),
                    )
                    < min(
                        _timestamp(cast(str, left["ended_at"])),
                        _timestamp(cast(str, right["ended_at"])),
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
            }
        ),
        location="durability",
    )
    _validate_fine_resume(
        durability["chunk_failure_resume"],
        "durability.chunk_failure_resume",
        "chunk",
    )
    _validate_fine_resume(
        durability["channel_failure_resume"],
        "durability.channel_failure_resume",
        "channel",
    )
    cancellation = _fine_object(
        durability["cancellation_resume"],
        keys=frozenset(
            {
                "status",
                "structured_cancellation",
                "preserved_artifacts",
                "resume_reused_keys",
                "resume_recomputed_keys",
                "replacement_events",
                "liveness_proven",
                "owned_job_ids",
                "remaining_owned_job_ids",
            }
        ),
        location="durability.cancellation_resume",
    )
    owned_jobs = _fine_exact_strings(
        cancellation["owned_job_ids"],
        "durability.cancellation_resume.owned_job_ids",
    )
    remaining_jobs = _fine_exact_strings(
        cancellation["remaining_owned_job_ids"],
        "durability.cancellation_resume.remaining_owned_job_ids",
    )
    if (
        cancellation["status"] != "cancelled"
        or cancellation["structured_cancellation"] is not True
        or not owned_jobs
        or remaining_jobs
        or type(cancellation["replacement_events"]) is not int
        or cast(int, cancellation["replacement_events"]) < 1
        or cancellation["liveness_proven"] is not True
        or not _fine_exact_strings(
            cancellation["resume_reused_keys"],
            "durability.cancellation_resume.resume_reused_keys",
        )
        or not _fine_exact_strings(
            cancellation["resume_recomputed_keys"],
            "durability.cancellation_resume.resume_recomputed_keys",
        )
    ):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance cancellation, replacement, liveness, resume, "
            "or cleanup gate failed"
        )
    _fine_artifacts(
        cancellation["preserved_artifacts"],
        "durability.cancellation_resume.preserved_artifacts",
    )

    performance = _fine_object(
        report["performance"],
        keys=frozenset(
            {
                "coarse_nonlinear_wall_seconds",
                "fine_nonlinear_wall_seconds",
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
    if evidence_paths != sorted(set(evidence_paths)):
        raise HpcWorkflowAcceptanceError(
            "fine acceptance evidence checksums contain duplicate or noncanonical paths"
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
    "fine_acceptance_profiles",
    "fine_evidence_checksums",
    "fine_internal_parity_evidence",
    "validate_fine_acceptance_report",
]
