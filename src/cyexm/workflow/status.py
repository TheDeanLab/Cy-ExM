from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from cyexm.config.models import (
    NativeThreadSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.errors import WorkflowPlanError
from cyexm.registration._request_policy import (
    CHANNEL_POSTPROCESSING_POLICY,
    OUTPUT_FORMAT_POLICY,
)
from cyexm.registration.artifacts import CompletionRecord, FileIdentity
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.spatial import AXIS_CONVENTION_VERSION, ImageGeometry, spacing_compatible
from cyexm.workflow._round_provenance import (
    ArtifactSnapshot,
    RegistrationArtifactPaths,
    SourceGeometries,
    completion_evidence,
    completion_snapshot,
)
from cyexm.workflow.layout import WorkflowLayout
from cyexm.workflow.models import (
    ChannelTask,
    RoundStatus,
    RoundTask,
    WorkflowStatus,
    canonical_json,
    sha256_fingerprint,
)

_DATASET_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SCIENTIFIC_KEYS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "voxel_spacing_xyz_um",
        "reference",
        "registration",
        "sources",
        "rounds",
        "policies",
    }
)
_REFERENCE_KEYS = frozenset({"round", "channel"})
_REGISTRATION_KEYS = frozenset({"crop", "linear", "nonlinear"})
_STAGE_KEYS = frozenset({"type", "accuracy"})
_SOURCE_KEYS = frozenset(
    {
        "path",
        "shape_zyx",
        "dtype",
        "axes",
        "geometry",
        "st_size",
        "st_mtime_ns",
    }
)
_ROUND_KEYS = frozenset(
    {
        "round",
        "strategy",
        "registration_channel",
        "registration_manifest_channel",
        "chunk_size_zyx",
        "overlap_fraction",
        "channels",
    }
)
_CHANNEL_KEYS = frozenset({"file", "manifest_file", "output", "interpolation"})
_POLICY_KEYS = frozenset({"axis_convention", "output_format", "postprocessing"})
_LINEAR_TYPES = frozenset({"Translation", "Rigid", "Similarity", "Affine", "TRSAA"})
_NONLINEAR_TYPES = frozenset({"Elastic", "SyNOnly", "SyN"})
_ACCURACIES = frozenset({"high", "medium", "low", "dry run"})
type ArtifactRole = Literal["affine", "warp", "registered", "channel"]


@dataclass(frozen=True, slots=True)
class _SourceContract:
    path: Path
    identity: FileIdentity
    shape_zyx: tuple[int, int, int]
    dtype: str
    axes: str
    geometry: ImageGeometry


@dataclass(frozen=True, slots=True)
class _ChannelContract:
    source: _SourceContract
    manifest_file: PurePosixPath
    output: Path
    interpolation: Literal["linear", "nearestNeighbor"]


@dataclass(frozen=True, slots=True)
class _ArtifactExpectation:
    path: Path
    role: ArtifactRole
    channel: _ChannelContract | None = None


@dataclass(frozen=True, slots=True)
class _RoundContract:
    round: int
    strategy: Literal["whole_image", "serial_chunked"]
    registration_source: _SourceContract
    registration_manifest_channel: PurePosixPath
    chunk_size_zyx: tuple[int, int, int] | None
    overlap_fraction: float | None
    channels: tuple[_ChannelContract, ...]
    artifacts: tuple[_ArtifactExpectation, ...]


@dataclass(frozen=True, slots=True)
class _RegistrationContract:
    crop: bool
    linear_type: str
    linear_accuracy: str
    nonlinear_type: str
    nonlinear_accuracy: str


@dataclass(frozen=True, slots=True)
class _ScientificSnapshot:
    fingerprint: str
    dataset_id: str
    voxel_spacing_xyz: tuple[float, float, float]
    reference_round: int
    fixed_source: _SourceContract
    registration: _RegistrationContract
    sources: tuple[_SourceContract, ...]
    rounds: tuple[_RoundContract, ...]

    @property
    def expected_by_round(
        self,
    ) -> tuple[tuple[int, tuple[_ArtifactExpectation, ...]], ...]:
        return tuple(
            (round_contract.round, round_contract.artifacts)
            for round_contract in self.rounds
        )


@dataclass(frozen=True, slots=True)
class _ArtifactEvidence:
    state: Literal["incomplete", "invalid", "recorded"]
    identity: FileIdentity | None = None
    record: CompletionRecord | None = None


def _validate_dataset_id(dataset_id: object) -> str:
    if (
        not isinstance(dataset_id, str)
        or _DATASET_ID_PATTERN.fullmatch(dataset_id) is None
    ):
        raise WorkflowPlanError(
            f"dataset_id must match [A-Za-z0-9][A-Za-z0-9._-]*; got {dataset_id!r}"
        )
    return dataset_id


def _safe_workflow_layout(output_root: str | Path, dataset_id: str) -> WorkflowLayout:
    _validate_dataset_id(dataset_id)
    output = Path(output_root).resolve(strict=False)
    layout = WorkflowLayout.for_dataset(output_root, dataset_id)
    resolved = layout.root.resolve(strict=False)
    if not resolved.is_relative_to(output):
        raise WorkflowPlanError(
            f"Dataset workflow path resolves outside the output root {output}; "
            f"got {resolved}"
        )
    return layout


def _safe_descendant(path: Path, *, root: Path, location: str) -> Path:
    resolved_root = root.resolve(strict=False)
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(resolved_root):
        raise WorkflowPlanError(
            f"{location} resolves outside workflow output {resolved_root}; "
            f"got {resolved}"
        )
    return path


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"finite JSON number required; got {value!r}")
    return result


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_json(text: str, location: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_float=_finite_json_float,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise WorkflowPlanError(
            f"{location} is not strict finite JSON: {error}"
        ) from error


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise WorkflowPlanError(f"{location} must be a JSON object")
    return value


def _exact_mapping(
    value: object,
    keys: frozenset[str],
    location: str,
) -> Mapping[str, object]:
    mapping = _mapping(value, location)
    if frozenset(mapping) != keys:
        raise WorkflowPlanError(
            f"{location} keys must be exactly {sorted(keys)!r}; got {sorted(mapping)!r}"
        )
    return mapping


def _list(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise WorkflowPlanError(f"{location} must be a JSON array; got {value!r}")
    return value


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise WorkflowPlanError(
            f"{location} must be a nonempty single-line string; got {value!r}"
        )
    return value


def _integer(value: object, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise WorkflowPlanError(
            f"{location} must be a built-in integer >= {minimum}; got {value!r}"
        )
    return value


def _finite_number(value: object, location: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WorkflowPlanError(f"{location} must be a finite number; got {value!r}")
    try:
        result = float(value)
    except OverflowError as error:
        raise WorkflowPlanError(
            f"{location} must be a finite number; got {value!r}"
        ) from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise WorkflowPlanError(
            f"{location} must be a {qualifier} number; got {value!r}"
        )
    return result


def _triple_float(value: object, location: str) -> tuple[float, float, float]:
    values = _list(value, location)
    if len(values) != 3:
        raise WorkflowPlanError(f"{location} must contain exactly three values")
    return (
        _finite_number(values[0], f"{location}[0]", positive=True),
        _finite_number(values[1], f"{location}[1]", positive=True),
        _finite_number(values[2], f"{location}[2]", positive=True),
    )


def _triple_int(value: object, location: str) -> tuple[int, int, int]:
    values = _list(value, location)
    if len(values) != 3:
        raise WorkflowPlanError(f"{location} must contain exactly three values")
    return (
        _integer(values[0], f"{location}[0]", minimum=1),
        _integer(values[1], f"{location}[1]", minimum=1),
        _integer(values[2], f"{location}[2]", minimum=1),
    )


def _relative_path(value: object, location: str) -> PurePosixPath:
    text = _string(value, location)
    raw_parts = text.split("/")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in raw_parts):
        raise WorkflowPlanError(
            f"{location} must be a safe relative POSIX path; got {value!r}"
        )
    return path


def _absolute_path(value: object, location: str) -> Path:
    text = _string(value, location)
    path = Path(text)
    if not path.is_absolute() or str(path.resolve(strict=False)) != text:
        raise WorkflowPlanError(
            f"{location} must be a canonical absolute path; got {value!r}"
        )
    return path


def _choice(value: object, choices: frozenset[str], location: str) -> str:
    text = _string(value, location)
    if text not in choices:
        raise WorkflowPlanError(
            f"{location} must be one of {sorted(choices)!r}; got {value!r}"
        )
    return text


def _default_geometry(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> ImageGeometry:
    return ImageGeometry(
        shape=shape,
        origin=(0.0, 0.0, 0.0),
        spacing=spacing,
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )


def _parse_source(
    value: object,
    *,
    spacing: tuple[float, float, float],
    index: int,
) -> _SourceContract:
    location = f"scientific plan sources[{index}]"
    mapping = _exact_mapping(value, _SOURCE_KEYS, location)
    path = _absolute_path(mapping["path"], f"{location}.path")
    shape = _triple_int(mapping["shape_zyx"], f"{location}.shape_zyx")
    dtype = _string(mapping["dtype"], f"{location}.dtype")
    axes = _choice(mapping["axes"], frozenset({"ZYX", "QYX", "YX"}), f"{location}.axes")
    if axes == "YX" and shape[0] != 1:
        raise WorkflowPlanError(
            f"{location}.shape_zyx must have singleton Z for YX axes"
        )
    raw_geometry = mapping["geometry"]
    if raw_geometry is None:
        geometry = _default_geometry(shape, spacing)
    else:
        try:
            geometry = ImageGeometry.from_dict(
                _mapping(raw_geometry, f"{location}.geometry")
            )
        except (TypeError, ValueError) as error:
            raise WorkflowPlanError(
                f"{location}.geometry is invalid: {error}"
            ) from error
        if geometry.shape != shape or not spacing_compatible(geometry.spacing, spacing):
            raise WorkflowPlanError(
                f"{location}.geometry must match shape and voxel spacing"
            )
    identity = FileIdentity(
        path=str(path),
        st_size=_integer(mapping["st_size"], f"{location}.st_size"),
        st_mtime_ns=_integer(mapping["st_mtime_ns"], f"{location}.st_mtime_ns"),
    )
    return _SourceContract(path, identity, shape, dtype, axes, geometry)


def _parse_registration(value: object) -> _RegistrationContract:
    mapping = _exact_mapping(value, _REGISTRATION_KEYS, "scientific plan registration")
    crop = mapping["crop"]
    if type(crop) is not bool:
        raise WorkflowPlanError(
            f"scientific plan registration.crop must be boolean; got {crop!r}"
        )
    linear = _exact_mapping(
        mapping["linear"], _STAGE_KEYS, "scientific plan registration.linear"
    )
    nonlinear = _exact_mapping(
        mapping["nonlinear"], _STAGE_KEYS, "scientific plan registration.nonlinear"
    )
    return _RegistrationContract(
        crop=crop,
        linear_type=_choice(
            linear["type"], _LINEAR_TYPES, "scientific plan registration.linear.type"
        ),
        linear_accuracy=_choice(
            linear["accuracy"],
            _ACCURACIES,
            "scientific plan registration.linear.accuracy",
        ),
        nonlinear_type=_choice(
            nonlinear["type"],
            _NONLINEAR_TYPES,
            "scientific plan registration.nonlinear.type",
        ),
        nonlinear_accuracy=_choice(
            nonlinear["accuracy"],
            _ACCURACIES,
            "scientific plan registration.nonlinear.accuracy",
        ),
    )


def _parse_channel(
    value: object,
    *,
    layout: WorkflowLayout,
    round_number: int,
    sources: Mapping[Path, _SourceContract],
    index: int,
) -> _ChannelContract:
    location = f"scientific plan rounds[{round_number}].channels[{index}]"
    mapping = _exact_mapping(value, _CHANNEL_KEYS, location)
    source_path = _absolute_path(mapping["file"], f"{location}.file")
    try:
        source = sources[source_path]
    except KeyError as error:
        raise WorkflowPlanError(
            f"{location}.file must reference a declared source; got {source_path}"
        ) from error
    manifest_file = _relative_path(
        mapping["manifest_file"], f"{location}.manifest_file"
    )
    relative_output = _relative_path(mapping["output"], f"{location}.output")
    output = layout.root.joinpath(*relative_output.parts)
    expected_output = layout.round(round_number).channel(manifest_file)
    if output != expected_output:
        raise WorkflowPlanError(
            f"{location}.output must be "
            f"{expected_output.relative_to(layout.root).as_posix()!r}; "
            f"got {relative_output.as_posix()!r}"
        )
    _safe_descendant(output, root=layout.root, location=f"{location}.output")
    interpolation = _choice(
        mapping["interpolation"],
        frozenset({"linear", "nearestNeighbor"}),
        f"{location}.interpolation",
    )
    return _ChannelContract(
        source=source,
        manifest_file=manifest_file,
        output=output,
        interpolation=interpolation,  # type: ignore[arg-type]
    )


def _parse_round(
    value: object,
    *,
    layout: WorkflowLayout,
    sources: Mapping[Path, _SourceContract],
    index: int,
) -> _RoundContract:
    location = f"scientific plan rounds[{index}]"
    mapping = _exact_mapping(value, _ROUND_KEYS, location)
    round_number = _integer(mapping["round"], f"{location}.round")
    strategy = _choice(
        mapping["strategy"],
        frozenset({"whole_image", "serial_chunked"}),
        f"{location}.strategy",
    )
    registration_path = _absolute_path(
        mapping["registration_channel"], f"{location}.registration_channel"
    )
    try:
        registration_source = sources[registration_path]
    except KeyError as error:
        raise WorkflowPlanError(
            f"{location}.registration_channel must reference a declared source; "
            f"got {registration_path}"
        ) from error
    registration_manifest = _relative_path(
        mapping["registration_manifest_channel"],
        f"{location}.registration_manifest_channel",
    )
    channels = tuple(
        _parse_channel(
            channel,
            layout=layout,
            round_number=round_number,
            sources=sources,
            index=channel_index,
        )
        for channel_index, channel in enumerate(
            _list(mapping["channels"], f"{location}.channels")
        )
    )
    if not channels:
        raise WorkflowPlanError(f"{location}.channels must not be empty")
    channel_files = tuple(channel.manifest_file for channel in channels)
    if len(set(channel_files)) != len(channel_files):
        raise WorkflowPlanError(
            f"{location} channel manifest files must be unique; got {channel_files!r}"
        )
    channel_outputs = tuple(
        channel.output.resolve(strict=False) for channel in channels
    )
    if len(set(channel_outputs)) != len(channel_outputs):
        raise WorkflowPlanError(
            f"{location} channel outputs must be unique; got {channel_outputs!r}"
        )
    matches = tuple(
        channel
        for channel in channels
        if channel.manifest_file == registration_manifest
    )
    if len(matches) != 1 or matches[0].source != registration_source:
        raise WorkflowPlanError(
            f"{location}.registration_manifest_channel must identify the "
            "registration source"
        )
    if strategy == "whole_image":
        if (
            mapping["chunk_size_zyx"] is not None
            or mapping["overlap_fraction"] is not None
        ):
            raise WorkflowPlanError(f"{location} whole_image chunk values must be null")
        chunk_size = None
        overlap = None
    else:
        chunk_size = _triple_int(
            mapping["chunk_size_zyx"], f"{location}.chunk_size_zyx"
        )
        overlap = _finite_number(
            mapping["overlap_fraction"], f"{location}.overlap_fraction"
        )
        if not 0.0 <= overlap < 1.0:
            raise WorkflowPlanError(
                f"{location}.overlap_fraction must satisfy 0 <= value < 1"
            )
    round_layout = layout.round(round_number)
    artifacts = (
        _ArtifactExpectation(
            _safe_descendant(
                round_layout.registration / f"linear_transform_{round_number}.mat",
                root=layout.root,
                location=f"{location} affine",
            ),
            "affine",
        ),
        _ArtifactExpectation(
            _safe_descendant(
                round_layout.registration / f"nonlinear_warp_{round_number}.nii.gz",
                root=layout.root,
                location=f"{location} warp",
            ),
            "warp",
        ),
        _ArtifactExpectation(
            _safe_descendant(
                round_layout.registration / f"nonlinear_registered_{round_number}.tif",
                root=layout.root,
                location=f"{location} registered image",
            ),
            "registered",
        ),
        *(
            _ArtifactExpectation(channel.output, "channel", channel)
            for channel in channels
        ),
    )
    return _RoundContract(
        round=round_number,
        strategy=strategy,  # type: ignore[arg-type]
        registration_source=registration_source,
        registration_manifest_channel=registration_manifest,
        chunk_size_zyx=chunk_size,
        overlap_fraction=overlap,
        channels=channels,
        artifacts=artifacts,
    )


def _parse_scientific_snapshot(
    payload: object,
    *,
    layout: WorkflowLayout,
    dataset_id: str,
) -> _ScientificSnapshot:
    scientific = _exact_mapping(payload, _SCIENTIFIC_KEYS, "scientific plan")
    schema_version = scientific["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise WorkflowPlanError(
            "scientific plan schema_version must be the built-in integer 1; "
            f"got {schema_version!r}"
        )
    if scientific["dataset_id"] != dataset_id:
        raise WorkflowPlanError(
            f"scientific plan dataset_id must be {dataset_id!r}; "
            f"got {scientific['dataset_id']!r}"
        )
    spacing = _triple_float(
        scientific["voxel_spacing_xyz_um"], "scientific plan voxel_spacing_xyz_um"
    )
    registration = _parse_registration(scientific["registration"])
    sources = tuple(
        _parse_source(source, spacing=spacing, index=index)
        for index, source in enumerate(
            _list(scientific["sources"], "scientific plan sources")
        )
    )
    if not sources:
        raise WorkflowPlanError("scientific plan sources must not be empty")
    if tuple(source.path for source in sources) != tuple(
        sorted((source.path for source in sources), key=str)
    ):
        raise WorkflowPlanError(
            "scientific plan sources must be in canonical path order"
        )
    sources_by_path = {source.path: source for source in sources}
    if len(sources_by_path) != len(sources):
        raise WorkflowPlanError("scientific plan source paths must be unique")
    reference = _exact_mapping(
        scientific["reference"], _REFERENCE_KEYS, "scientific plan reference"
    )
    reference_round = _integer(reference["round"], "scientific plan reference.round")
    fixed_path = _absolute_path(
        reference["channel"], "scientific plan reference.channel"
    )
    try:
        fixed_source = sources_by_path[fixed_path]
    except KeyError as error:
        raise WorkflowPlanError(
            "scientific plan reference.channel must reference a declared source"
        ) from error
    rounds = tuple(
        _parse_round(round_value, layout=layout, sources=sources_by_path, index=index)
        for index, round_value in enumerate(
            _list(scientific["rounds"], "scientific plan rounds")
        )
    )
    round_numbers = tuple(round_contract.round for round_contract in rounds)
    if not rounds or round_numbers != tuple(sorted(set(round_numbers))):
        raise WorkflowPlanError(
            "scientific plan rounds must be nonempty, unique, and sorted; "
            f"got {round_numbers!r}"
        )
    if reference_round in round_numbers:
        raise WorkflowPlanError(
            f"scientific plan rounds must not reuse reference round {reference_round}"
        )
    used_sources = {
        fixed_source.path,
        *(
            channel.source.path
            for round_contract in rounds
            for channel in round_contract.channels
        ),
    }
    unused_sources = tuple(
        source.path for source in sources if source.path not in used_sources
    )
    if unused_sources:
        raise WorkflowPlanError(
            f"scientific plan declares unused sources: {unused_sources!r}"
        )
    incompatible_shapes = tuple(
        (source.path, source.shape_zyx)
        for source in sources
        if source.shape_zyx != fixed_source.shape_zyx
    )
    if incompatible_shapes:
        raise WorkflowPlanError(
            "scientific plan source shape must match the reference shape "
            f"{fixed_source.shape_zyx!r}; got {incompatible_shapes!r}"
        )
    artifact_paths = tuple(
        expectation.path.resolve(strict=False)
        for round_contract in rounds
        for expectation in round_contract.artifacts
    )
    if len(set(artifact_paths)) != len(artifact_paths):
        raise WorkflowPlanError(
            "scientific plan final artifact outputs must be unique; "
            f"got {artifact_paths!r}"
        )
    policies = _exact_mapping(
        scientific["policies"], _POLICY_KEYS, "scientific plan policies"
    )
    expected_policies = {
        "axis_convention": AXIS_CONVENTION_VERSION,
        "output_format": OUTPUT_FORMAT_POLICY,
        "postprocessing": CHANNEL_POSTPROCESSING_POLICY,
    }
    if canonical_json(policies) != canonical_json(expected_policies):
        raise WorkflowPlanError(
            f"scientific plan policies must be exactly {expected_policies!r}; "
            f"got {dict(policies)!r}"
        )
    encoded = canonical_json(scientific)
    return _ScientificSnapshot(
        fingerprint=sha256_fingerprint(encoded),
        dataset_id=dataset_id,
        voxel_spacing_xyz=spacing,
        reference_round=reference_round,
        fixed_source=fixed_source,
        registration=registration,
        sources=sources,
        rounds=rounds,
    )


def _snapshot_from_text(
    text: str,
    *,
    layout: WorkflowLayout,
    dataset_id: str,
) -> _ScientificSnapshot:
    return _parse_scientific_snapshot(
        _decode_json(text, "scientific plan"),
        layout=layout,
        dataset_id=dataset_id,
    )


def _read_scientific_snapshot(
    plan_path: Path,
    dataset_id: str,
) -> _ScientificSnapshot | None:
    layout = WorkflowLayout(plan_path.parent)
    _safe_descendant(plan_path, root=layout.root, location="scientific-plan.json")
    if not plan_path.exists() and not plan_path.is_symlink():
        return None
    try:
        text = plan_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise WorkflowPlanError(
            f"Scientific plan is unreadable at {plan_path}: {error}"
        ) from error
    return _snapshot_from_text(text, layout=layout, dataset_id=dataset_id)


def _read_artifact_evidence(
    expectation: _ArtifactExpectation,
    *,
    layout: WorkflowLayout,
) -> _ArtifactEvidence:
    artifact = _safe_descendant(
        expectation.path,
        root=layout.root,
        location=f"{expectation.role} artifact",
    )
    record_path = _safe_descendant(
        artifact.with_name(f"{artifact.name}.complete.json"),
        root=layout.root,
        location=f"{expectation.role} completion record",
    )
    artifact_present = artifact.exists() or artifact.is_symlink()
    record_present = record_path.exists() or record_path.is_symlink()
    if not artifact_present and not record_present:
        return _ArtifactEvidence("incomplete")
    if artifact_present and not record_present:
        return _ArtifactEvidence("incomplete")
    if not artifact_present or not artifact.is_file() or not record_path.is_file():
        return _ArtifactEvidence("invalid")
    try:
        record_text = record_path.read_text(encoding="utf-8")
        _decode_json(record_text, f"completion record {record_path}")
        record = CompletionRecord.read(record_path)
        identity = FileIdentity.from_path(artifact)
    except Exception:
        return _ArtifactEvidence("invalid")
    if record.to_json() != record_text or record.artifact != identity:
        return _ArtifactEvidence("invalid")
    return _ArtifactEvidence("recorded", identity, record)


def _valid_native_threads(value: Mapping[str, int]) -> bool:
    if frozenset(value) != frozenset(NATIVE_THREAD_ENV_VARS):
        return False
    if any(type(item) is not int or item < 1 for item in value.values()):
        return False
    blas_values = tuple(value[name] for name in NATIVE_THREAD_ENV_VARS[1:])
    return len(set(blas_values)) == 1


def _native_threads_from_evidence(
    evidences: tuple[_ArtifactEvidence, ...],
) -> NativeThreadSpec:
    for evidence in evidences:
        if evidence.record is None:
            continue
        values = evidence.record.request.native_threads
        if _valid_native_threads(values):
            return NativeThreadSpec(
                itk=values[NATIVE_THREAD_ENV_VARS[0]],
                blas=values[NATIVE_THREAD_ENV_VARS[1]],
            )
    return NativeThreadSpec(itk=1, blas=1)


def _task_from_snapshot(
    snapshot: _ScientificSnapshot,
    round_contract: _RoundContract,
    layout: WorkflowLayout,
    native_threads: NativeThreadSpec,
) -> RoundTask:
    round_layout = layout.round(round_contract.round)
    return RoundTask(
        dataset_id=snapshot.dataset_id,
        round=round_contract.round,
        strategy=round_contract.strategy,
        fixed_image=snapshot.fixed_source.path,
        moving_image=round_contract.registration_source.path,
        registration_directory=round_layout.registration,
        log_path=round_layout.log,
        voxel_spacing_xyz=snapshot.voxel_spacing_xyz,
        registration=RegistrationSpec(
            crop=snapshot.registration.crop,
            linear=RegistrationStageSpec(
                type=snapshot.registration.linear_type,
                accuracy=snapshot.registration.linear_accuracy,
            ),
            nonlinear=RegistrationStageSpec(
                type=snapshot.registration.nonlinear_type,
                accuracy=snapshot.registration.nonlinear_accuracy,
            ),
        ),
        native_threads=native_threads,
        channels=tuple(
            ChannelTask(
                source=channel.source.path,
                output=channel.output,
                interpolation=channel.interpolation,
            )
            for channel in round_contract.channels
        ),
        chunk_size_zyx=round_contract.chunk_size_zyx,
        overlap_fraction=round_contract.overlap_fraction,
    )


def _safe_completion_snapshot(
    path: Path,
    *,
    layout: WorkflowLayout,
) -> ArtifactSnapshot | None:
    artifact = _safe_descendant(
        path,
        root=layout.root,
        location="authoritative round artifact",
    )
    _safe_descendant(
        artifact.with_name(f"{artifact.name}.complete.json"),
        root=layout.root,
        location="authoritative round completion record",
    )
    return completion_snapshot(artifact)


def _round_status(
    snapshot: _ScientificSnapshot,
    round_contract: _RoundContract,
    *,
    layout: WorkflowLayout,
    task: RoundTask | None = None,
) -> RoundStatus:
    evidences = tuple(
        _read_artifact_evidence(expectation, layout=layout)
        for expectation in round_contract.artifacts
    )
    if task is None:
        task = _task_from_snapshot(
            snapshot,
            round_contract,
            layout,
            _native_threads_from_evidence(evidences),
        )
    paths = RegistrationArtifactPaths.for_round(
        task.registration_directory,
        task.round,
    )

    def snapshot_reader(path: Path) -> ArtifactSnapshot | None:
        return _safe_completion_snapshot(path, layout=layout)

    def geometry_reader(
        path: Path,
        _spacing: tuple[float, float, float],
    ) -> ImageGeometry:
        evidence = snapshot_reader(path)
        if evidence is None:
            raise ValueError(f"Missing fixed-reference completion evidence: {path}")
        return evidence[0].request.geometry

    authoritative = completion_evidence(
        task,
        paths,
        SourceGeometries(
            fixed=snapshot.fixed_source.geometry,
            moving=round_contract.registration_source.geometry,
        ),
        snapshot_reader=snapshot_reader,
        geometry_reader=geometry_reader,
    )
    accepted = frozenset(authoritative.artifacts)
    complete: list[str] = []
    incomplete: list[str] = []
    invalid: list[str] = []
    for expectation, evidence in zip(round_contract.artifacts, evidences, strict=True):
        path = str(expectation.path)
        if evidence.state == "incomplete":
            incomplete.append(path)
        elif evidence.state == "recorded" and expectation.path in accepted:
            complete.append(path)
        else:
            invalid.append(path)
    if invalid:
        status = "invalid"
    elif len(complete) == len(round_contract.artifacts):
        status = "complete"
    elif complete:
        status = "partial"
    else:
        status = "pending"
    return RoundStatus(
        round=round_contract.round,
        status=status,
        complete_artifacts=tuple(complete),
        incomplete_artifacts=tuple(incomplete),
        invalid_artifacts=tuple(invalid),
    )


def _inspect_workflow_status(
    output_root: str | Path,
    dataset_id: str,
    *,
    _tasks: tuple[RoundTask, ...] | None = None,
) -> WorkflowStatus:
    """Inspect authoritative completion provenance without changing workflow state."""
    layout = _safe_workflow_layout(output_root, dataset_id)
    _safe_descendant(
        layout.plan_path, root=layout.root, location="scientific-plan.json"
    )
    snapshot = _read_scientific_snapshot(layout.plan_path, dataset_id)
    if snapshot is None:
        return WorkflowStatus(dataset_id, None, (), False)
    tasks_by_round: dict[int, RoundTask] = {}
    if _tasks is not None:
        tasks_by_round = {task.round: task for task in _tasks}
        expected_rounds = tuple(item.round for item in snapshot.rounds)
        if len(tasks_by_round) != len(_tasks) or set(tasks_by_round) != set(
            expected_rounds
        ):
            actual_rounds = tuple(task.round for task in _tasks)
            raise WorkflowPlanError(
                "Status task rounds must match the persisted scientific plan; "
                f"expected={expected_rounds!r}, got={actual_rounds!r}"
            )
    rounds = tuple(
        _round_status(
            snapshot,
            round_contract,
            layout=layout,
            task=tasks_by_round.get(round_contract.round),
        )
        for round_contract in snapshot.rounds
    )
    return WorkflowStatus(
        dataset_id=dataset_id,
        scientific_fingerprint=snapshot.fingerprint,
        rounds=rounds,
        complete=bool(rounds) and all(item.status == "complete" for item in rounds),
    )


def inspect_workflow_status(
    output_root: str | Path,
    dataset_id: str,
) -> WorkflowStatus:
    """Inspect authoritative completion provenance without changing workflow state."""
    return _inspect_workflow_status(output_root, dataset_id)


__all__ = ["inspect_workflow_status"]
