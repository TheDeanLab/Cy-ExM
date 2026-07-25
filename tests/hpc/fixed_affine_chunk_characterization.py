"""Explicit real-SLURM fixed-affine chunk-size characterization helpers."""

from __future__ import annotations

import getpass
import json
import math
import os
import shutil
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import cast

from cyexm.chunking import compute_chunk_grid
from cyexm.config._validation import built_in_int, require_exact_keys
from cyexm.config._yaml import load_yaml_mapping
from cyexm.config.models import DatasetManifest, ExecutionProfile
from cyexm.errors import ConfigurationValidationError
from cyexm.execution.slurm import SlurmExecutor
from cyexm.registration.artifacts import (
    ArtifactStore,
    CompletionRecord,
    FileIdentity,
)
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.spatial import ImageGeometry
from cyexm.workflow.fine_models import (
    FineTaskResult,
    FineWorkflowTask,
    RoundPreparationTask,
)
from cyexm.workflow.models import (
    ChannelTask,
    ReferenceChannelTask,
    ReferenceTask,
    RoundResult,
    RoundTask,
    WorkflowTask,
    WorkflowTaskResult,
)
from hpc.environment import (
    collect_queue_snapshot,
    wait_for_queue_empty,
    write_evidence_json,
)

APPROVED_CHUNK_SIZES = {
    "1125": (512, 1125, 1125),
    "512": (512, 512, 512),
    "256": (512, 256, 256),
    "128": (512, 128, 128),
}
_EXPECTED_KEYS = frozenset(
    {
        "schema_version",
        "source_workflow_root",
        "condition_order",
        "chunk_sizes_zyx",
        "overlap_fraction",
        "rounds",
        "dim_roi_yx",
        "bright_roi_yx",
        "search_radius_px",
    }
)


class FixedAffineCharacterizationError(ValueError):
    """Raised when fixed-affine characterization evidence is invalid."""


@dataclass(frozen=True, slots=True)
class RoiYX:
    """One half-open YX region of interest."""

    y_start: int
    y_end: int
    x_start: int
    x_end: int

    def __post_init__(self) -> None:
        values = (self.y_start, self.y_end, self.x_start, self.x_end)
        if any(type(value) is not int or value < 0 for value in values):
            raise FixedAffineCharacterizationError(
                "ROI bounds must be nonnegative built-in integers"
            )
        if self.y_start >= self.y_end or self.x_start >= self.x_end:
            raise FixedAffineCharacterizationError(
                "ROI half-open bounds must be nonempty"
            )


@dataclass(frozen=True, slots=True)
class FixedAffineCharacterizationSpec:
    """Normalized operator-owned characterization configuration."""

    schema_version: int
    source_workflow_root: Path
    condition_order: tuple[str, ...]
    chunk_sizes_zyx: dict[str, tuple[int, int, int]]
    overlap_fraction: float
    rounds: tuple[int, ...]
    dim_roi_yx: RoiYX
    bright_roi_yx: RoiYX
    search_radius_px: int


@dataclass(frozen=True, slots=True)
class SourceRoundState:
    """Validated immutable affine-stage state for one moving round."""

    round: int
    fixed_reference: Path
    moving_image: Path
    moving_crop: Path
    pre_metrics: Path
    affine: Path
    linear_metrics: Path
    fixed_geometry: ImageGeometry
    moving_crop_record: CompletionRecord
    pre_metrics_record: CompletionRecord
    affine_record: CompletionRecord
    linear_metrics_record: CompletionRecord


@dataclass(frozen=True, slots=True)
class ValidatedFixedAffineSource:
    """Validated read-only radius-4 source workflow."""

    workflow_root: Path
    fixed_reference: Path
    fixed_reference_record: CompletionRecord
    rounds: tuple[SourceRoundState, ...]


@dataclass(frozen=True, slots=True)
class SharedPreparedRound:
    """One caller-owned prepared seed reused by every chunk-size condition."""

    task: RoundTask
    source: SourceRoundState
    manifest: object
    result: FineTaskResult


@dataclass(frozen=True, slots=True)
class CharacterizationCondition:
    """One exact caller-owned nonlinear chunk-size condition."""

    label: str
    chunk_size_zyx: tuple[int, int, int]
    profile: ExecutionProfile
    reference_task: ReferenceTask
    round_tasks: tuple[RoundTask, ...]
    condition_root: Path
    dataset_root: Path
    shared_root: Path
    fixed_geometry: ImageGeometry


@dataclass(frozen=True, slots=True)
class ReadOnlyReferenceRunner:
    """Validate the accepted reference barrier without writing source state."""

    source: ValidatedFixedAffineSource

    def __call__(self, task: WorkflowTask) -> WorkflowTaskResult:
        if type(task) is not ReferenceTask:
            from cyexm.workflow.task import run_workflow_task

            return run_workflow_task(task)
        started = time.monotonic()
        started_at = _utc_now()
        if (
            task.registration_channel_output != self.source.fixed_reference
            or task.channels
            != (
                ReferenceChannelTask(
                    source=self.source.fixed_reference,
                    output=self.source.fixed_reference,
                ),
            )
            or task.crop
            or task.crop_bounds_path is not None
        ):
            raise FixedAffineCharacterizationError(
                "reference barrier does not name the validated fixed source"
            )
        current_record = _record_for(self.source.fixed_reference, "fixed reference")
        if current_record != self.source.fixed_reference_record:
            raise FixedAffineCharacterizationError(
                "fixed reference evidence changed before the read-only barrier"
            )
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        task.log_path.write_text(
            f"{started_at} validated read-only fixed reference\n",
            encoding="utf-8",
        )
        threads = tuple(
            (
                name,
                (
                    task.native_threads.itk
                    if name == NATIVE_THREAD_ENV_VARS[0]
                    else task.native_threads.blas
                ),
            )
            for name in NATIVE_THREAD_ENV_VARS
        )
        artifact_paths = (str(self.source.fixed_reference),)
        return RoundResult(
            dataset_id=task.dataset_id,
            round=task.round,
            status="reused",
            started_at=started_at,
            ended_at=_utc_now(),
            duration_seconds=time.monotonic() - started,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            dask_task_key=_dask_task_key(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            strategy="reference-preparation",
            native_threads=threads,
            artifact_paths=artifact_paths,
            reused_artifacts=artifact_paths,
            log_path=str(task.log_path),
            failure=None,
        )


@dataclass(frozen=True, slots=True)
class FixedAffineFineRunner:
    """Clone one shared preparation, then delegate ordinary fine-stage work."""

    shared_rounds: tuple[SharedPreparedRound, ...]

    def __post_init__(self) -> None:
        if any(type(item) is not SharedPreparedRound for item in self.shared_rounds):
            raise TypeError("shared_rounds must contain exact shared prepared values")
        if len({item.task.round for item in self.shared_rounds}) != len(
            self.shared_rounds
        ):
            raise ValueError("shared prepared rounds must be unique")

    def __call__(self, task: FineWorkflowTask) -> FineTaskResult:
        if type(task) is RoundPreparationTask:
            matching = tuple(
                item for item in self.shared_rounds if item.task.round == task.round
            )
            if len(matching) != 1:
                raise FixedAffineCharacterizationError(
                    f"missing unique shared preparation for round {task.round}"
                )
            return clone_shared_prepared_state(task, matching[0])
        from cyexm.workflow.fine_task import run_fine_task

        return run_fine_task(task)


def _exact_list(value: object, location: str) -> list[object]:
    if type(value) is not list:
        raise FixedAffineCharacterizationError(f"{location} must be an exact list")
    return cast(list[object], value)


def _int_tuple(
    value: object,
    location: str,
    *,
    length: int,
    minimum: int,
) -> tuple[int, ...]:
    items = _exact_list(value, location)
    if len(items) != length:
        raise FixedAffineCharacterizationError(
            f"{location} must contain exactly {length} integers"
        )
    try:
        return tuple(
            built_in_int(item, f"{location}[{index}]", minimum=minimum)
            for index, item in enumerate(items)
        )
    except ConfigurationValidationError as error:
        raise FixedAffineCharacterizationError(str(error)) from error


def _roi(value: object, location: str) -> RoiYX:
    values = _int_tuple(value, location, length=4, minimum=0)
    return RoiYX(*values)


def load_characterization_spec(
    path: str | Path,
) -> FixedAffineCharacterizationSpec:
    """Load one exact operator-owned fixed-affine characterization file."""
    try:
        mapping = require_exact_keys(
            load_yaml_mapping(path),
            _EXPECTED_KEYS,
            "fixed-affine characterization",
        )
    except ConfigurationValidationError as error:
        raise FixedAffineCharacterizationError(str(error)) from error

    schema_version = mapping["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise FixedAffineCharacterizationError("schema_version must equal 1")

    raw_source = mapping["source_workflow_root"]
    if type(raw_source) is not str or not raw_source:
        raise FixedAffineCharacterizationError(
            "source_workflow_root must be a nonempty string"
        )
    source_path = Path(raw_source)
    if source_path.is_symlink():
        raise FixedAffineCharacterizationError(
            "source_workflow_root must not be a symlink"
        )
    try:
        source_root = source_path.resolve(strict=True)
    except OSError as error:
        raise FixedAffineCharacterizationError(
            "source_workflow_root must be an existing directory"
        ) from error
    if not source_root.is_dir() or not (source_root / "scientific-plan.json").is_file():
        raise FixedAffineCharacterizationError(
            "source_workflow_root must contain scientific-plan.json"
        )

    order_items = _exact_list(mapping["condition_order"], "condition_order")
    if any(type(item) is not str for item in order_items):
        raise FixedAffineCharacterizationError(
            "condition_order must contain exact strings"
        )
    condition_order = tuple(cast(list[str], order_items))

    raw_sizes = mapping["chunk_sizes_zyx"]
    if type(raw_sizes) is not dict or any(
        type(key) is not str for key in cast(dict[object, object], raw_sizes)
    ):
        raise FixedAffineCharacterizationError(
            "chunk_sizes_zyx must be an exact string-keyed mapping"
        )
    sizes = {
        key: cast(
            tuple[int, int, int],
            _int_tuple(value, f"chunk_sizes_zyx[{key!r}]", length=3, minimum=1),
        )
        for key, value in cast(dict[str, object], raw_sizes).items()
    }
    if sizes != APPROVED_CHUNK_SIZES or condition_order != (
        "1125",
        "512",
        "256",
        "128",
    ):
        raise FixedAffineCharacterizationError(
            "configuration must equal the approved chunk sizes and condition order"
        )

    overlap = mapping["overlap_fraction"]
    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, int | float)
        or not math.isfinite(float(overlap))
        or float(overlap) != 0.15
    ):
        raise FixedAffineCharacterizationError(
            "overlap_fraction must equal the approved value 0.15"
        )
    rounds = _int_tuple(mapping["rounds"], "rounds", length=3, minimum=0)
    if rounds != (2, 3, 4):
        raise FixedAffineCharacterizationError("rounds must equal [2, 3, 4]")
    search_radius = mapping["search_radius_px"]
    if type(search_radius) is not int or search_radius != 30:
        raise FixedAffineCharacterizationError("search_radius_px must equal 30")

    return FixedAffineCharacterizationSpec(
        schema_version=1,
        source_workflow_root=source_root,
        condition_order=condition_order,
        chunk_sizes_zyx=sizes,
        overlap_fraction=0.15,
        rounds=cast(tuple[int, int, int], rounds),
        dim_roi_yx=_roi(mapping["dim_roi_yx"], "dim_roi_yx"),
        bright_roi_yx=_roi(mapping["bright_roi_yx"], "bright_roi_yx"),
        search_radius_px=search_radius,
    )


def _json_mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise FixedAffineCharacterizationError(f"{location} must be a JSON object")
    return cast(Mapping[str, object], value)


def _record_for(path: Path, role: str) -> CompletionRecord:
    store = ArtifactStore()
    try:
        record = CompletionRecord.read(store.record_path(path))
        current = FileIdentity.from_path(path)
    except Exception as error:
        raise FixedAffineCharacterizationError(
            f"{role} completion evidence is unreadable"
        ) from error
    if record.artifact != current:
        raise FixedAffineCharacterizationError(f"{role} identity mismatch")
    for identity in record.request.inputs:
        try:
            observed = FileIdentity.from_path(Path(identity.path))
        except OSError as error:
            raise FixedAffineCharacterizationError(
                f"{role} input identity is unreadable"
            ) from error
        if observed != identity:
            raise FixedAffineCharacterizationError(f"{role} input identity mismatch")
    if record.request.upstream is not None:
        upstream = record.request.upstream
        try:
            observed_upstream = FileIdentity.from_path(Path(upstream.path))
        except OSError as error:
            raise FixedAffineCharacterizationError(
                f"{role} upstream identity is unreadable"
            ) from error
        if observed_upstream != upstream:
            raise FixedAffineCharacterizationError(f"{role} upstream identity mismatch")
    return record


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def clone_accepted_artifact(
    source: Path,
    source_record: CompletionRecord,
    destination: Path,
    *,
    upstream: FileIdentity | None | object = ...,
) -> CompletionRecord:
    """Atomically clone one accepted artifact into caller-owned state."""
    try:
        observed_source = FileIdentity.from_path(source)
    except OSError as error:
        raise FixedAffineCharacterizationError(
            "accepted source artifact is unreadable"
        ) from error
    if observed_source != source_record.artifact:
        raise FixedAffineCharacterizationError(
            "accepted source artifact identity mismatch"
        )
    source_hash = _file_sha256(source)
    request = (
        source_record.request
        if upstream is ...
        else replace(
            source_record.request,
            upstream=cast(FileIdentity | None, upstream),
        )
    )
    store = ArtifactStore()
    if destination.exists() or store.record_path(destination).exists():
        try:
            reused_hash = store.reuse(destination, request, _file_sha256)
        except Exception as error:
            raise FixedAffineCharacterizationError(
                f"destination cache is incompatible: {destination}"
            ) from error
        if reused_hash != source_hash:
            raise FixedAffineCharacterizationError(
                f"destination bytes differ from accepted source: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        store.commit(
            destination,
            request,
            writer=lambda temporary: shutil.copy2(source, temporary),
            validator=lambda candidate: _file_sha256(candidate) == source_hash,
        )
    try:
        observed_after = FileIdentity.from_path(source)
        cloned_record = CompletionRecord.read(store.record_path(destination))
    except Exception as error:
        raise FixedAffineCharacterizationError(
            "artifact identity changed during accepted clone"
        ) from error
    if observed_after != source_record.artifact:
        raise FixedAffineCharacterizationError(
            "accepted source artifact identity changed during clone"
        )
    if (
        cloned_record.request != request
        or cloned_record.artifact != FileIdentity.from_path(destination)
        or _file_sha256(destination) != source_hash
    ):
        raise FixedAffineCharacterizationError(
            "cloned artifact completion evidence is invalid"
        )
    return cloned_record


def materialize_shared_prepared_state(
    task: RoundTask,
    source: SourceRoundState,
) -> SharedPreparedRound:
    """Create or reuse one prepared seed without estimating a new affine."""
    from cyexm.registration.prepared import PreparedManifest
    from cyexm.workflow._round_provenance import RegistrationArtifactPaths
    from cyexm.workflow.fine_task import run_fine_task

    if type(task) is not RoundTask or type(source) is not SourceRoundState:
        raise TypeError("shared preparation requires exact task and source values")
    if (
        task.round != source.round
        or task.fixed_image.resolve(strict=True) != source.fixed_reference
        or task.moving_image.resolve(strict=True) != source.moving_image
        or task.registration.linear.type != source.affine_record.request.method
        or task.registration.linear.accuracy != source.affine_record.request.accuracy
        or not task.prepared_fixed
        or not task.registration.crop
    ):
        raise FixedAffineCharacterizationError(
            "shared preparation task does not match the accepted affine source"
        )
    expected_threads = {
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": task.native_threads.itk,
        "MKL_NUM_THREADS": task.native_threads.blas,
        "NUMEXPR_NUM_THREADS": task.native_threads.blas,
        "OMP_NUM_THREADS": task.native_threads.blas,
        "OPENBLAS_NUM_THREADS": task.native_threads.blas,
        "VECLIB_MAXIMUM_THREADS": task.native_threads.blas,
    }
    if source.affine_record.request.native_threads != expected_threads:
        raise FixedAffineCharacterizationError(
            "shared preparation thread policy does not match the source affine"
        )
    paths = RegistrationArtifactPaths.for_round(task.registration_directory, task.round)
    clone_accepted_artifact(
        source.moving_crop,
        source.moving_crop_record,
        paths.moving_crop,
    )
    clone_accepted_artifact(
        source.pre_metrics,
        source.pre_metrics_record,
        paths.pre_metrics,
    )
    affine_record = clone_accepted_artifact(
        source.affine,
        source.affine_record,
        paths.affine,
    )
    clone_accepted_artifact(
        source.linear_metrics,
        source.linear_metrics_record,
        paths.linear_metrics,
        upstream=affine_record.artifact,
    )
    result = run_fine_task(RoundPreparationTask.create(task))
    if result.status not in {"completed", "reused"}:
        message = "" if result.failure is None else result.failure.message
        raise FixedAffineCharacterizationError(
            f"shared prepared seed failed: {message}"
        )
    descriptor = result.to_dict()["prepared_descriptor"]
    if not isinstance(descriptor, Mapping):
        raise FixedAffineCharacterizationError(
            "shared prepared seed omitted its descriptor"
        )
    manifest = PreparedManifest.from_dict(descriptor)
    if (
        manifest.affine != FileIdentity.from_path(paths.affine)
        or manifest.geometry != source.fixed_geometry
    ):
        raise FixedAffineCharacterizationError(
            "shared prepared seed does not preserve accepted affine geometry"
        )
    return SharedPreparedRound(
        task=task,
        source=source,
        manifest=manifest,
        result=result,
    )


def _copy_with_request(
    source: Path,
    destination: Path,
    request: object,
) -> CompletionRecord:
    if not hasattr(request, "stage"):
        raise TypeError("prepared clone request must be a StageRequest")
    source_hash = _file_sha256(source)
    store = ArtifactStore()
    if destination.exists() or store.record_path(destination).exists():
        try:
            reused = store.reuse(destination, request, _file_sha256)  # type: ignore[arg-type]
        except Exception as error:
            raise FixedAffineCharacterizationError(
                f"prepared destination cache is incompatible: {destination}"
            ) from error
        if reused != source_hash:
            raise FixedAffineCharacterizationError(
                f"prepared destination bytes differ: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        store.commit(
            destination,
            request,  # type: ignore[arg-type]
            writer=lambda temporary: shutil.copy2(source, temporary),
            validator=lambda candidate: _file_sha256(candidate) == source_hash,
        )
    return CompletionRecord.read(store.record_path(destination))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _dask_task_key() -> str | None:
    try:
        from distributed import get_worker

        value = get_worker().get_current_task()
    except Exception:
        return None
    return None if value is None else str(value)


def clone_shared_prepared_state(
    task: RoundPreparationTask,
    shared: SharedPreparedRound,
) -> FineTaskResult:
    """Clone one shared prepared seed into a condition without affine application."""
    from cyexm.registration._request_policy import (
        PREPARED_MASK_POLICY,
        PREPARED_SCALAR_POLICY,
        canonical_compact_json,
        common_parameters,
    )
    from cyexm.registration.artifacts import StageRequest
    from cyexm.registration.prepared import PreparedManifest, PreparedRoundPaths
    from cyexm.workflow._round_provenance import RegistrationArtifactPaths

    if (
        type(task) is not RoundPreparationTask
        or type(shared) is not SharedPreparedRound
    ):
        raise TypeError("prepared clone requires exact preparation and shared values")
    round_task = task.round_task
    shared_task = shared.task
    if (
        round_task.round != shared_task.round
        or round_task.fixed_image.resolve(strict=True)
        != shared_task.fixed_image.resolve(strict=True)
        or round_task.moving_image.resolve(strict=True)
        != shared_task.moving_image.resolve(strict=True)
        or round_task.registration != shared_task.registration
        or round_task.native_threads != shared_task.native_threads
        or round_task.voxel_spacing_xyz != shared_task.voxel_spacing_xyz
        or not round_task.prepared_fixed
    ):
        raise FixedAffineCharacterizationError(
            "condition preparation does not match the shared affine state"
        )
    started = time.monotonic()
    started_at = _utc_now()
    source_registration = RegistrationArtifactPaths.for_round(
        shared_task.registration_directory, shared_task.round
    )
    destination_registration = RegistrationArtifactPaths.for_round(
        round_task.registration_directory, round_task.round
    )
    source_store = ArtifactStore()
    crop_record = clone_accepted_artifact(
        source_registration.moving_crop,
        CompletionRecord.read(
            source_store.record_path(source_registration.moving_crop)
        ),
        destination_registration.moving_crop,
    )
    clone_accepted_artifact(
        source_registration.pre_metrics,
        CompletionRecord.read(
            source_store.record_path(source_registration.pre_metrics)
        ),
        destination_registration.pre_metrics,
    )
    affine_record = clone_accepted_artifact(
        source_registration.affine,
        CompletionRecord.read(source_store.record_path(source_registration.affine)),
        destination_registration.affine,
    )
    clone_accepted_artifact(
        source_registration.linear_metrics,
        CompletionRecord.read(
            source_store.record_path(source_registration.linear_metrics)
        ),
        destination_registration.linear_metrics,
        upstream=affine_record.artifact,
    )

    source_prepared = PreparedRoundPaths.for_round(
        shared_task.registration_directory, shared_task.round
    )
    destination_prepared = PreparedRoundPaths.for_round(
        round_task.registration_directory, round_task.round
    )
    destination_prepared.directory.mkdir(parents=True, exist_ok=True)
    fixed = FileIdentity.from_path(round_task.fixed_image)
    moving = FileIdentity.from_path(round_task.moving_image)
    threads = {
        name: (
            round_task.native_threads.itk
            if name == NATIVE_THREAD_ENV_VARS[0]
            else round_task.native_threads.blas
        )
        for name in NATIVE_THREAD_ENV_VARS
    }
    common = common_parameters(
        crop=round_task.registration.crop,
        imaging_round=round_task.round,
        pixel_size_xyz=round_task.voxel_spacing_xyz,
    )
    destination_array_identities: list[FileIdentity] = []
    for role, source_path, destination_path in zip(
        (
            "fixed_scalar",
            "moving_linear_scalar",
            "fixed_mask",
            "moving_linear_mask",
        ),
        source_prepared.array_paths,
        destination_prepared.array_paths,
        strict=True,
    ):
        parameters = {
            **common,
            "format_version": shared.manifest.format_version,  # type: ignore[attr-defined]
            "prepared_role": role,
            "scalar_policy": PREPARED_SCALAR_POLICY,
            "mask_policy": PREPARED_MASK_POLICY,
            "ants_pixeltype": shared.manifest.pixeltypes[role],  # type: ignore[attr-defined]
        }
        request = StageRequest.for_identities(
            stage=f"prepared:{role}",
            inputs=(fixed, moving),
            geometry=shared.manifest.geometry,  # type: ignore[attr-defined]
            method=round_task.registration.linear.type,
            accuracy=round_task.registration.linear.accuracy,
            native_threads=threads,
            upstream=affine_record.artifact,
            parameters=parameters,
        )
        record = _copy_with_request(source_path, destination_path, request)
        destination_array_identities.append(record.artifact)

    source_manifest = cast(PreparedManifest, shared.manifest)
    destination_manifest = PreparedManifest(
        format_version=source_manifest.format_version,
        fixed_scalar=destination_array_identities[0],
        moving_linear_scalar=destination_array_identities[1],
        fixed_mask=destination_array_identities[2],
        moving_linear_mask=destination_array_identities[3],
        native_fixed=fixed,
        native_moving=moving,
        moving_crop=crop_record.artifact,
        affine=affine_record.artifact,
        geometry=source_manifest.geometry,
        scalar_policy=source_manifest.scalar_policy,
        mask_policy=source_manifest.mask_policy,
        native_threads=dict(sorted(threads.items())),
        pixeltypes=source_manifest.pixeltypes,
    )
    manifest_request = StageRequest.for_identities(
        stage="prepared:manifest",
        inputs=(
            fixed,
            moving,
            *destination_manifest.array_identities,
            crop_record.artifact,
        ),
        geometry=destination_manifest.geometry,
        method=round_task.registration.linear.type,
        accuracy=round_task.registration.linear.accuracy,
        native_threads=threads,
        upstream=affine_record.artifact,
        parameters={
            "format_version": destination_manifest.format_version,
            "scalar_policy": destination_manifest.scalar_policy,
            "mask_policy": destination_manifest.mask_policy,
            "pixeltypes": destination_manifest.pixeltypes,
        },
    )
    manifest_text = canonical_compact_json(destination_manifest.to_dict())
    destination_store = ArtifactStore()
    existing = (
        destination_store.reuse(
            destination_prepared.manifest,
            manifest_request,
            lambda path: path.read_text(encoding="utf-8"),
        )
        if (
            destination_prepared.manifest.exists()
            or destination_store.record_path(destination_prepared.manifest).exists()
        )
        else None
    )
    if existing is None:
        destination_store.commit(
            destination_prepared.manifest,
            manifest_request,
            writer=lambda path: path.write_text(manifest_text, encoding="utf-8"),
            validator=lambda path: path.read_text(encoding="utf-8") == manifest_text,
        )
    elif existing != manifest_text:
        raise FixedAffineCharacterizationError(
            "condition prepared manifest differs from the shared seed"
        )
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    task.log_path.write_text(
        f"{started_at} cloned shared fixed-affine prepared state\n",
        encoding="utf-8",
    )
    ended_at = _utc_now()
    return FineTaskResult(
        task=task,
        status="completed",
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=time.monotonic() - started,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        dask_task_key=_dask_task_key(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        native_threads=tuple((name, threads[name]) for name in NATIVE_THREAD_ENV_VARS),
        artifact_paths=tuple(str(path) for path in task.output_paths),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=None,
        prepared_descriptor=destination_manifest.to_dict(),
    )


def build_condition(
    *,
    label: str,
    chunk_size_zyx: tuple[int, int, int],
    manifest: DatasetManifest,
    profile: ExecutionProfile,
    source: ValidatedFixedAffineSource,
) -> CharacterizationCondition:
    """Build one fixed-affine condition without inspecting or mutating a scheduler."""
    if (
        label not in APPROVED_CHUNK_SIZES
        or APPROVED_CHUNK_SIZES[label] != chunk_size_zyx
    ):
        raise FixedAffineCharacterizationError(
            "condition label and chunk size must match the approved sweep"
        )
    if (
        manifest.registration.linear.type != "TRSAA"
        or manifest.registration.linear.accuracy != "high"
        or manifest.registration.nonlinear.type != "SyNOnly"
        or manifest.registration.nonlinear.accuracy != "high"
        or manifest.registration.nonlinear.metric != "CC"
        or manifest.registration.nonlinear.radius != 4
        or not manifest.registration.crop
    ):
        raise FixedAffineCharacterizationError(
            "condition manifest must preserve TRSAA/high and CC SyN radius 4"
        )
    if (
        profile.executor != "slurm"
        or profile.parallelism.mode != "fine_grained"
        or profile.slurm is None
        or profile.paths.worker_local_directory is None
    ):
        raise FixedAffineCharacterizationError(
            "condition requires a fine-grained SLURM profile"
        )
    condition_root = profile.paths.output_root / f"chunk-{label}"
    scratch_root = profile.paths.worker_local_directory / f"chunk-{label}"
    job_name = f"{profile.slurm.job_name}-c{label}"
    condition_profile = replace(
        profile,
        paths=replace(
            profile.paths,
            output_root=condition_root,
            worker_local_directory=scratch_root,
        ),
        slurm=replace(profile.slurm, job_name=job_name),
    )
    dataset_id = f"fixed-affine-chunk-{label}"
    dataset_root = condition_root / dataset_id
    reference_directory = dataset_root / "rounds/round-001/registration"
    reference_task = ReferenceTask(
        dataset_id=dataset_id,
        round=1,
        registration_channel_source=source.fixed_reference,
        registration_channel_output=source.fixed_reference,
        registration_directory=reference_directory,
        crop_bounds_path=None,
        log_path=dataset_root / "rounds/round-001/logs/reference.log",
        voxel_spacing_xyz=manifest.voxel_spacing_um,
        crop=False,
        native_threads=profile.native_threads,
        channels=(
            ReferenceChannelTask(
                source=source.fixed_reference,
                output=source.fixed_reference,
            ),
        ),
    )
    manifest_rounds = {item.round: item for item in manifest.rounds}
    source_rounds = {item.round: item for item in source.rounds}
    if set(manifest_rounds) != {2, 3, 4} or set(source_rounds) != {2, 3, 4}:
        raise FixedAffineCharacterizationError(
            "condition requires moving rounds 2, 3, and 4"
        )
    tasks: list[RoundTask] = []
    for round_number in (2, 3, 4):
        source_round = source_rounds[round_number]
        round_root = dataset_root / f"rounds/round-{round_number:03d}"
        channel_output = round_root / "channels" / source_round.moving_image.name
        tasks.append(
            RoundTask(
                dataset_id=dataset_id,
                round=round_number,
                strategy="serial_chunked",
                fixed_image=source.fixed_reference,
                moving_image=source_round.moving_image,
                registration_directory=round_root / "registration",
                log_path=round_root / "logs" / f"round-{round_number:03d}.log",
                voxel_spacing_xyz=manifest.voxel_spacing_um,
                registration=manifest.registration,
                native_threads=profile.native_threads,
                channels=(
                    ChannelTask(
                        source=source_round.moving_image,
                        output=channel_output,
                        interpolation="linear",
                    ),
                ),
                chunk_size_zyx=chunk_size_zyx,
                overlap_fraction=0.15,
                prepared_fixed=True,
            )
        )
    return CharacterizationCondition(
        label=label,
        chunk_size_zyx=chunk_size_zyx,
        profile=condition_profile,
        reference_task=reference_task,
        round_tasks=tuple(tasks),
        condition_root=condition_root,
        dataset_root=dataset_root,
        shared_root=profile.paths.output_root / "shared-prepared",
        fixed_geometry=source.rounds[0].fixed_geometry,
    )


def build_shared_preparation_tasks(
    condition: CharacterizationCondition,
) -> tuple[RoundTask, ...]:
    """Build the one caller-owned fixed-grid preparation seed per moving round."""
    tasks: list[RoundTask] = []
    for task in condition.round_tasks:
        round_root = condition.shared_root / f"round-{task.round:03d}"
        tasks.append(
            replace(
                task,
                dataset_id="fixed-affine-shared-prepared",
                registration_directory=round_root / "registration",
                log_path=round_root / "logs" / "preparation.log",
                channels=(),
            )
        )
    return tuple(tasks)


def expected_chunk_task_count(condition: CharacterizationCondition) -> int:
    """Return the exact nonlinear chunk count across the condition's rounds."""
    total = 0
    for task in condition.round_tasks:
        if task.chunk_size_zyx is None or task.overlap_fraction is None:
            raise FixedAffineCharacterizationError(
                "characterization round lacks chunk geometry"
            )
        total += len(
            compute_chunk_grid(
                condition.fixed_geometry.shape,
                task.chunk_size_zyx,
                task.overlap_fraction,
            )
        )
    return total


def _owned_job_ids(
    results: tuple[RoundResult, ...],
    fine_results: tuple[FineTaskResult, ...],
    evidence: Mapping[str, object],
) -> tuple[str, ...]:
    values: set[str] = {
        value
        for value in (
            *(result.slurm_job_id for result in results),
            *(result.slurm_job_id for result in fine_results),
        )
        if isinstance(value, str) and value
    }
    observed = evidence.get("observed")
    if isinstance(observed, Mapping):
        probes = observed.get("worker_probes")
        if isinstance(probes, list):
            for probe in probes:
                if isinstance(probe, Mapping):
                    value = probe.get("slurm_job_id")
                    if isinstance(value, str) and value:
                        values.add(value)
    return tuple(sorted(values))


def execute_chunk_condition(
    *,
    condition: CharacterizationCondition,
    source: ValidatedFixedAffineSource,
    shared_rounds: tuple[SharedPreparedRound, ...],
    evidence_dir: Path,
    git_commit: str,
    package_versions: Mapping[str, str],
) -> dict[str, object]:
    """Execute one owned fine-SLURM condition and durably record its outcome."""
    spec = condition.profile.slurm
    if spec is None:
        raise FixedAffineCharacterizationError("condition lacks SLURM settings")
    if validate_fixed_affine_source(source.workflow_root) != source:
        raise FixedAffineCharacterizationError(
            "validated source state changed before condition execution"
        )
    condition.condition_root.mkdir(parents=True, exist_ok=True)
    scratch = condition.profile.paths.worker_local_directory
    if scratch is None:
        raise FixedAffineCharacterizationError("condition lacks worker scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    condition_evidence = evidence_dir / f"chunk-{condition.label}"
    condition_evidence.mkdir(parents=True, exist_ok=False)
    user = getpass.getuser()
    queue_before = collect_queue_snapshot(spec.job_name, user=user)
    write_evidence_json(condition_evidence / "queue-before.json", queue_before)
    if queue_before.get("jobs") != []:
        raise FixedAffineCharacterizationError(
            f"refusing to submit while {spec.job_name!r} jobs already exist"
        )

    executor = SlurmExecutor(
        condition.profile,
        _runner=ReadOnlyReferenceRunner(source),
        _fine_runner=FixedAffineFineRunner(shared_rounds),
    )
    results: tuple[RoundResult, ...] = ()
    execution_error: BaseException | None = None
    started = time.monotonic()
    started_at = _utc_now()
    try:
        results = executor.execute(condition.reference_task, condition.round_tasks)
    except BaseException as error:
        execution_error = error
    finally:
        elapsed = time.monotonic() - started
        execution_evidence = dict(executor.execution_evidence)
        fine_results = executor.fine_task_results
        owned_ids = _owned_job_ids(results, fine_results, execution_evidence)
        queue_after = wait_for_queue_empty(
            spec.job_name,
            user=user,
            job_ids=owned_ids,
        )
        source_unchanged = validate_fixed_affine_source(source.workflow_root) == source
        expected_chunks = expected_chunk_task_count(condition)
        observed_chunks = sum(result.key.stage == "chunk" for result in fine_results)
        summary: dict[str, object] = {
            "schema_version": 1,
            "condition": condition.label,
            "chunk_size_zyx": list(condition.chunk_size_zyx),
            "overlap_fraction": 0.15,
            "cc_radius": 4,
            "started_at": started_at,
            "ended_at": _utc_now(),
            "elapsed_seconds": elapsed,
            "git_commit": git_commit,
            "package_versions": dict(sorted(package_versions.items())),
            "source_workflow_root": str(source.workflow_root),
            "source_unchanged": source_unchanged,
            "expected_chunk_tasks": expected_chunks,
            "observed_chunk_results": observed_chunks,
            "fine_task_count": len(executor.fine_tasks),
            "fine_result_count": len(fine_results),
            "fine_tasks": [task.to_dict() for task in executor.fine_tasks],
            "fine_results": [result.to_dict() for result in fine_results],
            "round_results": [result.to_dict() for result in results],
            "owned_job_ids": list(owned_ids),
            "queue_after": queue_after,
            "execution_error": (
                None
                if execution_error is None
                else {
                    "type": type(execution_error).__name__,
                    "message": str(execution_error),
                }
            ),
        }
        write_evidence_json(
            condition_evidence / "slurm-execution-evidence.json",
            execution_evidence,
        )
        write_evidence_json(condition_evidence / "condition-summary.json", summary)

    if queue_after.get("outcome") != "empty":
        raise FixedAffineCharacterizationError(
            f"owned queue cleanup failed for condition {condition.label}"
        ) from execution_error
    if not source_unchanged:
        raise FixedAffineCharacterizationError(
            f"source identity changed during condition {condition.label}"
        ) from execution_error
    if execution_error is not None:
        raise execution_error
    if (
        len(results) != 4
        or any(result.status not in {"completed", "reused"} for result in results)
        or observed_chunks != expected_chunks
        or len(fine_results) != expected_chunks + 12
        or any(result.status not in {"completed", "reused"} for result in fine_results)
    ):
        raise FixedAffineCharacterizationError(
            f"condition {condition.label} produced an incomplete fine graph"
        )
    return summary


def _integer(value: object, location: str) -> int:
    if type(value) is not int:
        raise FixedAffineCharacterizationError(f"{location} must be an integer")
    return value


def _string(value: object, location: str) -> str:
    if type(value) is not str or not value:
        raise FixedAffineCharacterizationError(f"{location} must be a nonempty string")
    return value


def validate_fixed_affine_source(
    workflow_root: str | Path,
) -> ValidatedFixedAffineSource:
    """Validate one immutable completed CC-radius-4 affine source workflow."""
    raw_root = Path(workflow_root)
    if raw_root.is_symlink():
        raise FixedAffineCharacterizationError(
            "source workflow root must not be a symlink"
        )
    try:
        root = raw_root.resolve(strict=True)
        plan_payload = json.loads(
            (root / "scientific-plan.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise FixedAffineCharacterizationError(
            "source scientific plan is unreadable"
        ) from error
    plan = _json_mapping(plan_payload, "scientific plan")
    registration = _json_mapping(plan.get("registration"), "registration")
    linear = _json_mapping(registration.get("linear"), "registration.linear")
    nonlinear = _json_mapping(registration.get("nonlinear"), "registration.nonlinear")
    if (
        linear.get("type") != "TRSAA"
        or linear.get("accuracy") != "high"
        or nonlinear.get("type") != "SyNOnly"
        or nonlinear.get("accuracy") != "high"
        or nonlinear.get("metric") != "CC"
        or nonlinear.get("radius") != 4
    ):
        raise FixedAffineCharacterizationError(
            "source must use TRSAA/high and SyNOnly/high with CC radius 4"
        )
    if registration.get("crop") is not True:
        raise FixedAffineCharacterizationError("source registration must enable crop")
    reference = _json_mapping(plan.get("reference"), "reference")
    fixed_relative = Path(
        _string(reference.get("registration_output"), "reference.registration_output")
    )
    if fixed_relative.is_absolute() or ".." in fixed_relative.parts:
        raise FixedAffineCharacterizationError(
            "reference.registration_output must stay below the source root"
        )
    try:
        fixed = (root / fixed_relative).resolve(strict=True)
    except OSError as error:
        raise FixedAffineCharacterizationError(
            "fixed reference output is unreadable"
        ) from error
    if not fixed.is_relative_to(root):
        raise FixedAffineCharacterizationError(
            "fixed reference output escapes the source root"
        )
    fixed_record = _record_for(fixed, "fixed reference")

    raw_rounds = plan.get("rounds")
    if type(raw_rounds) is not list:
        raise FixedAffineCharacterizationError("scientific plan rounds must be a list")
    by_round: dict[int, Mapping[str, object]] = {}
    for index, item in enumerate(cast(list[object], raw_rounds)):
        value = _json_mapping(item, f"rounds[{index}]")
        round_number = _integer(value.get("round"), f"rounds[{index}].round")
        if round_number in by_round:
            raise FixedAffineCharacterizationError(
                "scientific plan moving rounds must be unique"
            )
        by_round[round_number] = value
    if set(by_round) != {2, 3, 4}:
        raise FixedAffineCharacterizationError(
            "source scientific plan must contain rounds 2, 3, and 4"
        )

    expected_threads = {
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": 30,
        "MKL_NUM_THREADS": 30,
        "NUMEXPR_NUM_THREADS": 30,
        "OMP_NUM_THREADS": 30,
        "OPENBLAS_NUM_THREADS": 30,
        "VECLIB_MAXIMUM_THREADS": 30,
    }
    source_rounds: list[SourceRoundState] = []
    for round_number in (2, 3, 4):
        item = by_round[round_number]
        moving = Path(
            _string(
                item.get("registration_channel"),
                f"rounds[{round_number}].registration_channel",
            )
        ).resolve(strict=True)
        registration_root = root / f"rounds/round-{round_number:03d}/registration"
        crop = registration_root / f"moving_crop_indices_{round_number}.json"
        pre = registration_root / "image_metrics_pre-registration.txt"
        affine = registration_root / f"linear_transform_{round_number}.mat"
        linear_metrics = registration_root / "image_metrics_linearly-registered.txt"
        crop_record = _record_for(crop, f"round {round_number} moving crop")
        pre_record = _record_for(pre, f"round {round_number} pre metrics")
        affine_record = _record_for(affine, f"round {round_number} affine")
        linear_record = _record_for(
            linear_metrics, f"round {round_number} linear metrics"
        )
        if (
            affine_record.request.method != "TRSAA"
            or affine_record.request.accuracy != "high"
            or affine_record.request.native_threads != expected_threads
            or affine_record.request.parameters.get("prepared_fixed") is not True
            or affine_record.request.inputs
            != (FileIdentity.from_path(fixed), FileIdentity.from_path(moving))
        ):
            raise FixedAffineCharacterizationError(
                f"round {round_number} affine request violates the accepted policy"
            )
        if linear_record.request.upstream != affine_record.artifact:
            raise FixedAffineCharacterizationError(
                f"round {round_number} linear metrics do not name the affine"
            )
        for role, record in (
            ("moving crop", crop_record),
            ("pre metrics", pre_record),
            ("linear metrics", linear_record),
        ):
            if record.request.native_threads != expected_threads:
                raise FixedAffineCharacterizationError(
                    f"round {round_number} {role} thread policy mismatch"
                )
        source_rounds.append(
            SourceRoundState(
                round=round_number,
                fixed_reference=fixed,
                moving_image=moving,
                moving_crop=crop,
                pre_metrics=pre,
                affine=affine,
                linear_metrics=linear_metrics,
                fixed_geometry=affine_record.request.geometry,
                moving_crop_record=crop_record,
                pre_metrics_record=pre_record,
                affine_record=affine_record,
                linear_metrics_record=linear_record,
            )
        )
    if len({item.fixed_geometry for item in source_rounds}) != 1:
        raise FixedAffineCharacterizationError(
            "source rounds must share one fixed affine geometry"
        )
    return ValidatedFixedAffineSource(
        workflow_root=root,
        fixed_reference=fixed,
        fixed_reference_record=fixed_record,
        rounds=tuple(source_rounds),
    )


__all__ = [
    "APPROVED_CHUNK_SIZES",
    "CharacterizationCondition",
    "FixedAffineCharacterizationError",
    "FixedAffineCharacterizationSpec",
    "FixedAffineFineRunner",
    "ReadOnlyReferenceRunner",
    "RoiYX",
    "SharedPreparedRound",
    "SourceRoundState",
    "ValidatedFixedAffineSource",
    "build_condition",
    "build_shared_preparation_tasks",
    "clone_accepted_artifact",
    "clone_shared_prepared_state",
    "execute_chunk_condition",
    "expected_chunk_task_count",
    "load_characterization_spec",
    "materialize_shared_prepared_state",
    "validate_fixed_affine_source",
]
