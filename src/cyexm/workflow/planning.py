from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from cyexm._nonlinear_metric import MATTES_BINS
from cyexm.config.dataset import _validate_dataset_manifest
from cyexm.config.execution import _validate_execution_profile
from cyexm.config.models import (
    DatasetManifest,
    ExecutionPaths,
    ExecutionProfile,
    NonlinearRegistrationStageSpec,
    RoundSpec,
)
from cyexm.errors import (
    ConfigurationValidationError,
    InputValidationError,
    WorkflowPlanError,
)
from cyexm.registration._request_policy import (
    CHANNEL_POSTPROCESSING_POLICY,
    OUTPUT_FORMAT_POLICY,
)
from cyexm.spatial import (
    AXIS_CONVENTION_VERSION,
    ImageGeometry,
    geometry_compatible,
    spacing_compatible,
)
from cyexm.workflow.layout import RoundLayout, WorkflowLayout
from cyexm.workflow.models import (
    ChannelTask,
    ImageHeader,
    ReferenceChannelTask,
    ReferenceTask,
    RoundTask,
    WorkflowPlan,
    canonical_json,
    sha256_fingerprint,
)


@dataclass(frozen=True, slots=True)
class _InputReference:
    yaml_path: str
    relative_path: PurePosixPath


@dataclass(frozen=True, slots=True)
class _FinalArtifact:
    path: Path
    role: str


def _nonlinear_mapping(
    stage: NonlinearRegistrationStageSpec,
) -> dict[str, object]:
    if stage.metric == "CC":
        return {
            "type": stage.type,
            "accuracy": stage.accuracy,
            "metric": "CC",
            "radius": stage.radius,
        }
    return {
        "type": stage.type,
        "accuracy": stage.accuracy,
        "metric": "mattes",
        "bins": MATTES_BINS,
    }


def _input_references(manifest: DatasetManifest) -> tuple[_InputReference, ...]:
    references = [
        _InputReference(
            yaml_path=f"reference.channels[{index}].file",
            relative_path=channel.file,
        )
        for index, channel in enumerate(manifest.reference.channels)
    ]
    for round_index, round_spec in enumerate(manifest.rounds):
        references.extend(
            _InputReference(
                yaml_path=f"rounds[{round_index}].channels[{channel_index}].file",
                relative_path=channel.file,
            )
            for channel_index, channel in enumerate(round_spec.channels)
        )
    return tuple(references)


def _resolve_root(path: Path, role: str, *, strict: bool) -> Path:
    try:
        resolved = path.resolve(strict=strict)
    except (OSError, RuntimeError) as error:
        contract = "an existing directory" if strict else "a safe path"
        raise WorkflowPlanError(
            f"{role} must resolve to {contract}; got {path}"
        ) from error
    if strict and not resolved.is_dir():
        raise WorkflowPlanError(
            f"{role} must resolve to an existing directory; got {resolved}"
        )
    if not strict and resolved.exists() and not resolved.is_dir():
        raise WorkflowPlanError(
            f"{role} must resolve to a directory when it exists; got {resolved}"
        )
    return resolved


def _resolved_roots(execution: ExecutionProfile) -> tuple[Path, Path]:
    dataset_root = _resolve_root(
        execution.paths.dataset_root,
        "paths.dataset_root",
        strict=True,
    )
    output_root = _resolve_root(
        execution.paths.output_root,
        "paths.output_root",
        strict=False,
    )
    if dataset_root.is_relative_to(output_root) or output_root.is_relative_to(
        dataset_root
    ):
        raise WorkflowPlanError(
            "dataset and output roots must be disjoint after resolution; "
            f"got {dataset_root} and {output_root}"
        )
    return dataset_root, output_root


def _resolve_inputs(
    manifest: DatasetManifest,
    dataset_root: Path,
) -> tuple[dict[PurePosixPath, Path], dict[Path, tuple[str, ...]]]:
    resolved_by_relative: dict[PurePosixPath, Path] = {}
    yaml_paths_by_resolved: dict[Path, list[str]] = {}
    errors: list[str] = []
    for reference in _input_references(manifest):
        cached = resolved_by_relative.get(reference.relative_path)
        if cached is not None:
            yaml_paths_by_resolved[cached].append(reference.yaml_path)
            continue
        candidate = dataset_root.joinpath(*reference.relative_path.parts)
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            errors.append(
                f"{reference.yaml_path}: input must resolve to an existing file; "
                f"got {candidate} ({type(error).__name__}: {error})"
            )
            continue
        if not resolved.is_file():
            errors.append(
                f"{reference.yaml_path}: input must resolve to a file; got {resolved}"
            )
            continue
        if not resolved.is_relative_to(dataset_root):
            errors.append(
                f"{reference.yaml_path}: input resolves outside the dataset root "
                f"{dataset_root}; got {resolved}"
            )
            continue
        resolved_by_relative[reference.relative_path] = resolved
        yaml_paths_by_resolved.setdefault(resolved, []).append(reference.yaml_path)
    if errors:
        details = "\n".join(f"- {message}" for message in errors)
        raise WorkflowPlanError(f"Workflow input validation failed:\n{details}")
    return resolved_by_relative, {
        path: tuple(yaml_paths) for path, yaml_paths in yaml_paths_by_resolved.items()
    }


def _shape_from_series(series: object, shaped_metadata: object) -> tuple[int, ...]:
    raw_shape = tuple(int(value) for value in series.shape)  # type: ignore[attr-defined]
    if len(raw_shape) == 3:
        return raw_shape
    metadata = (
        shaped_metadata[0]
        if isinstance(shaped_metadata, tuple | list) and shaped_metadata
        else None
    )
    if (
        getattr(series, "axes", None) == "YX"
        and isinstance(metadata, Mapping)
        and metadata.get("axes") == "ZYX"
    ):
        metadata_shape = metadata.get("shape")
        if (
            isinstance(metadata_shape, list)
            and len(metadata_shape) == 3
            and all(type(value) is int for value in metadata_shape)
        ):
            return tuple(metadata_shape)
    return raw_shape


def _inspect_tiff_header_in_process(path: Path) -> ImageHeader:
    """Inspect one TIFF's scalar layout and strict geometry without voxel loading."""
    try:
        import tifffile

        stat_before = path.stat()
        identity_before = (
            stat_before.st_dev,
            stat_before.st_ino,
            stat_before.st_size,
            stat_before.st_mtime_ns,
        )
        with tifffile.TiffFile(path) as handle:
            series = handle.series[0]
            shape = _shape_from_series(series, handle.shaped_metadata)
            axes = str(series.axes)
            dtype = str(series.dtype)
        from cyexm.io.tiff import read_tiff_geometry

        geometry = read_tiff_geometry(path, shape)
        stat_after = path.stat()
        identity_after = (
            stat_after.st_dev,
            stat_after.st_ino,
            stat_after.st_size,
            stat_after.st_mtime_ns,
        )
    except (IndexError, InputValidationError, OSError, TypeError, ValueError) as error:
        raise WorkflowPlanError(
            f"TIFF must have a readable scalar three-dimensional layout: {path}; "
            f"{error}"
        ) from error
    if identity_before != identity_after:
        raise WorkflowPlanError(
            f"File changed during TIFF header inspection: {path}; "
            f"before={identity_before!r}, after={identity_after!r}"
        )
    if len(shape) != 3 or axes not in {"ZYX", "QYX", "YX"}:
        raise WorkflowPlanError(
            f"TIFF must have a readable scalar three-dimensional layout: {path}; "
            f"got shape={shape!r}, axes={axes!r}"
        )
    return ImageHeader(
        path=path,
        shape_zyx=(shape[0], shape[1], shape[2]),
        dtype=dtype,
        axes=axes,
        geometry=geometry,
        st_size=stat_after.st_size,
        st_mtime_ns=stat_after.st_mtime_ns,
    )


def _header_from_probe(payload: object, expected_path: Path) -> ImageHeader:
    expected_keys = {
        "path",
        "shape_zyx",
        "dtype",
        "axes",
        "geometry",
        "st_size",
        "st_mtime_ns",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError("header probe returned an invalid object")
    raw_path = payload["path"]
    if not isinstance(raw_path, str) or Path(raw_path) != expected_path:
        raise ValueError("header probe returned the wrong path")
    raw_shape = payload["shape_zyx"]
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 3
        or any(type(value) is not int or value < 1 for value in raw_shape)
    ):
        raise ValueError("header probe returned an invalid shape")
    dtype = payload["dtype"]
    axes = payload["axes"]
    st_size = payload["st_size"]
    st_mtime_ns = payload["st_mtime_ns"]
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("header probe returned an invalid dtype")
    if not isinstance(axes, str) or not axes:
        raise ValueError("header probe returned invalid axes")
    if type(st_size) is not int or st_size < 0:
        raise ValueError("header probe returned an invalid file size")
    if type(st_mtime_ns) is not int or st_mtime_ns < 0:
        raise ValueError("header probe returned an invalid modification time")
    raw_geometry = payload["geometry"]
    geometry = None
    if raw_geometry is not None:
        if not isinstance(raw_geometry, Mapping):
            raise ValueError("header probe returned invalid geometry")
        geometry = ImageGeometry.from_dict(raw_geometry)
    return ImageHeader(
        path=expected_path,
        shape_zyx=(raw_shape[0], raw_shape[1], raw_shape[2]),
        dtype=dtype,
        axes=axes,
        geometry=geometry,
        st_size=st_size,
        st_mtime_ns=st_mtime_ns,
    )


def _inspect_tiff_header(path: Path) -> ImageHeader:
    """Inspect a TIFF without importing scientific runtimes in the caller."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "cyexm.workflow._header_probe", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise WorkflowPlanError(
            f"TIFF header inspection process could not start for {path}: {error}"
        ) from error
    except UnicodeError as error:
        raise WorkflowPlanError(
            f"TIFF header inspection returned invalid output for {path}: {error}"
        ) from error
    if completed.returncode != 0:
        detail = " ".join(completed.stderr.splitlines()) or (
            f"probe exited with status {completed.returncode}"
        )
        raise WorkflowPlanError(f"TIFF header inspection failed for {path}: {detail}")
    try:
        payload = json.loads(
            completed.stdout,
            object_pairs_hook=_strict_json_object,
            parse_float=_finite_json_float,
            parse_constant=_reject_json_constant,
        )
        return _header_from_probe(payload, path)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise WorkflowPlanError(
            f"TIFF header inspection returned invalid data for {path}: {error}"
        ) from error


def _header_contract_error(
    header: ImageHeader,
    expected_spacing: tuple[float, float, float],
) -> str | None:
    shape = header.shape_zyx
    if (
        type(shape) is not tuple
        or len(shape) != 3
        or any(type(value) is not int or value <= 0 for value in shape)
        or header.axes not in {"ZYX", "QYX", "YX"}
        or (header.axes == "YX" and shape[0] != 1)
        or not isinstance(header.dtype, str)
        or not header.dtype
    ):
        return (
            "header must describe a scalar three-dimensional TIFF; "
            f"got shape={shape!r}, dtype={header.dtype!r}, axes={header.axes!r}"
        )
    if header.geometry is not None:
        if header.geometry.shape != shape:
            return (
                "embedded geometry shape must match the TIFF shape; "
                f"got geometry={header.geometry.shape!r}, TIFF={shape!r}"
            )
        if not spacing_compatible(header.geometry.spacing, expected_spacing):
            return (
                "embedded spacing must match manifest voxel_spacing_um; "
                f"got embedded={header.geometry.spacing!r}, "
                f"manifest={expected_spacing!r}"
            )
    return None


def _inspect_headers(
    paths: Sequence[Path],
    yaml_paths_by_resolved: Mapping[Path, tuple[str, ...]],
    expected_spacing: tuple[float, float, float],
) -> dict[Path, ImageHeader]:
    headers: dict[Path, ImageHeader] = {}
    errors: list[str] = []
    for path in paths:
        labels = ", ".join(yaml_paths_by_resolved[path])
        try:
            header = _inspect_tiff_header(path)
        except WorkflowPlanError as error:
            errors.append(f"{labels}: {path}: {error}")
            continue
        if header.path != path:
            errors.append(
                f"{labels}: inspected header path must equal resolved input {path}; "
                f"got {header.path}"
            )
            continue
        contract_error = _header_contract_error(header, expected_spacing)
        if contract_error is not None:
            errors.append(f"{labels}: {path}: {contract_error}")
            continue
        headers[path] = header

    if errors:
        details = "\n".join(f"- {message}" for message in errors)
        raise WorkflowPlanError(f"Workflow TIFF header validation failed:\n{details}")
    return headers


def _final_artifacts(
    manifest: DatasetManifest,
    reference_task: ReferenceTask,
    tasks: tuple[RoundTask, ...],
) -> tuple[_FinalArtifact, ...]:
    artifacts = [
        _FinalArtifact(
            channel.output,
            f"reference.channels[{index}].file output",
        )
        for index, channel in enumerate(reference_task.channels)
    ]
    if reference_task.crop_bounds_path is not None:
        artifacts.append(
            _FinalArtifact(reference_task.crop_bounds_path, "reference crop bounds")
        )
    round_indexes = {
        round_spec.round: round_index
        for round_index, round_spec in enumerate(manifest.rounds)
    }
    for task in tasks:
        round_label = f"rounds[{round_indexes[task.round]}]"
        artifacts.extend(
            (
                _FinalArtifact(
                    task.registration_directory / f"linear_transform_{task.round}.mat",
                    f"{round_label} registration affine",
                ),
                _FinalArtifact(
                    task.registration_directory / f"nonlinear_warp_{task.round}.nii.gz",
                    f"{round_label} registration warp",
                ),
                _FinalArtifact(
                    task.registration_directory
                    / f"nonlinear_registered_{task.round}.tif",
                    f"{round_label} registration diagnostic",
                ),
            )
        )
        artifacts.extend(
            _FinalArtifact(
                channel.output,
                f"{round_label}.channels[{channel_index}].file output",
            )
            for channel_index, channel in enumerate(task.channels)
        )
    return tuple(artifacts)


def _validate_output_paths(
    layout: WorkflowLayout,
    output_root: Path,
    source_paths: frozenset[Path],
    reference_task: ReferenceTask,
    tasks: tuple[RoundTask, ...],
    artifacts: tuple[_FinalArtifact, ...],
) -> None:
    errors: list[str] = []
    artifacts_by_resolved_path: dict[Path, list[_FinalArtifact]] = {}
    planned_locations = [
        ("dataset output directory", layout.root),
        ("scientific plan", layout.plan_path),
    ]
    planned_locations.extend(
        (
            ("reference registration directory", reference_task.registration_directory),
            ("reference log", reference_task.log_path),
        )
    )
    for task in tasks:
        planned_locations.extend(
            (
                (
                    f"round {task.round} registration directory",
                    task.registration_directory,
                ),
                (f"round {task.round} log", task.log_path),
            )
        )
    for artifact in artifacts:
        resolved = artifact.path.resolve(strict=False)
        artifacts_by_resolved_path.setdefault(resolved, []).append(artifact)
        planned_locations.append((artifact.role, artifact.path))
        if resolved in source_paths:
            errors.append(f"{artifact.role} collides with source input {resolved}")
    for destination, aliases in artifacts_by_resolved_path.items():
        if len(aliases) > 1:
            roles = ", ".join(artifact.role for artifact in aliases)
            channel_detail = (
                " (channel output collision)"
                if all("channels[" in artifact.role for artifact in aliases)
                else ""
            )
            errors.append(
                f"final artifact collision{channel_detail} for {destination}: {roles}"
            )
    for role, location in planned_locations:
        resolved = location.resolve(strict=False)
        if not resolved.is_relative_to(output_root):
            errors.append(
                f"{role} resolves outside the output root {output_root}; got {resolved}"
            )
    if errors:
        details = "\n".join(f"- {message}" for message in errors)
        raise WorkflowPlanError(f"Workflow output validation failed:\n{details}")


def _channel_tasks(
    round_spec: RoundSpec,
    round_layout: RoundLayout,
    resolved_inputs: Mapping[PurePosixPath, Path],
) -> tuple[ChannelTask, ...]:
    return tuple(
        ChannelTask(
            source=resolved_inputs[channel.file],
            output=round_layout.channel(channel.file),
            interpolation=channel.interpolation,
        )
        for channel in round_spec.channels
    )


def _round_tasks(
    manifest: DatasetManifest,
    execution: ExecutionProfile,
    layout: WorkflowLayout,
    resolved_inputs: Mapping[PurePosixPath, Path],
    fixed_image: Path,
) -> tuple[RoundTask, ...]:
    tasks = []
    for round_spec in sorted(manifest.rounds, key=lambda item: item.round):
        round_layout = layout.round(round_spec.round)
        tasks.append(
            RoundTask(
                dataset_id=manifest.dataset_id,
                round=round_spec.round,
                strategy=round_spec.strategy,
                fixed_image=fixed_image,
                moving_image=resolved_inputs[round_spec.registration_channel],
                registration_directory=round_layout.registration,
                log_path=round_layout.log,
                voxel_spacing_xyz=manifest.voxel_spacing_um,
                registration=manifest.registration,
                native_threads=execution.native_threads,
                channels=_channel_tasks(
                    round_spec,
                    round_layout,
                    resolved_inputs,
                ),
                chunk_size_zyx=round_spec.chunk_size_zyx,
                overlap_fraction=round_spec.overlap_fraction,
            )
        )
    return tuple(tasks)


def _effective_geometry(
    header: ImageHeader,
    spacing: tuple[float, float, float],
) -> ImageGeometry:
    return header.geometry or ImageGeometry(
        shape=header.shape_zyx,
        origin=(0.0, 0.0, 0.0),
        spacing=spacing,
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )


def _reference_task(
    manifest: DatasetManifest,
    execution: ExecutionProfile,
    layout: WorkflowLayout,
    resolved_inputs: Mapping[PurePosixPath, Path],
    headers: Mapping[Path, ImageHeader],
) -> ReferenceTask:
    reference_layout = layout.round(manifest.reference.round)
    registration_source = resolved_inputs[manifest.reference.registration_channel]
    expected_geometry = _effective_geometry(
        headers[registration_source], manifest.voxel_spacing_um
    )
    errors: list[str] = []
    channels: list[ReferenceChannelTask] = []
    for index, channel in enumerate(manifest.reference.channels):
        source = resolved_inputs[channel.file]
        actual_geometry = _effective_geometry(
            headers[source], manifest.voxel_spacing_um
        )
        if not geometry_compatible(actual_geometry, expected_geometry):
            errors.append(
                f"reference.channels[{index}].file geometry must match "
                "reference.registration_channel; "
                f"got {actual_geometry.to_dict()!r}, expected "
                f"{expected_geometry.to_dict()!r}"
            )
        channels.append(
            ReferenceChannelTask(
                source=source,
                output=reference_layout.channel(channel.file),
            )
        )
    if errors:
        raise WorkflowPlanError(
            "Reference channel geometry validation failed:\n"
            + "\n".join(f"- {message}" for message in errors)
        )
    registration_output = reference_layout.channel(
        manifest.reference.registration_channel
    )
    return ReferenceTask(
        dataset_id=manifest.dataset_id,
        round=manifest.reference.round,
        registration_channel_source=registration_source,
        registration_channel_output=registration_output,
        registration_directory=reference_layout.registration,
        crop_bounds_path=(
            reference_layout.reference_crop if manifest.registration.crop else None
        ),
        log_path=reference_layout.log,
        voxel_spacing_xyz=manifest.voxel_spacing_um,
        crop=manifest.registration.crop,
        native_threads=execution.native_threads,
        channels=tuple(channels),
    )


def _header_mapping(header: ImageHeader) -> dict[str, object]:
    return {
        "path": str(header.path),
        "shape_zyx": list(header.shape_zyx),
        "dtype": header.dtype,
        "axes": header.axes,
        "geometry": (None if header.geometry is None else header.geometry.to_dict()),
        "st_size": header.st_size,
        "st_mtime_ns": header.st_mtime_ns,
    }


def _scientific_mapping(
    manifest: DatasetManifest,
    resolved_inputs: Mapping[PurePosixPath, Path],
    headers: Mapping[Path, ImageHeader],
    layout: WorkflowLayout,
    reference_task: ReferenceTask,
    tasks: tuple[RoundTask, ...],
) -> dict[str, object]:
    tasks_by_round = {task.round: task for task in tasks}
    return {
        "schema_version": manifest.schema_version,
        "dataset_id": manifest.dataset_id,
        "voxel_spacing_xyz_um": list(manifest.voxel_spacing_um),
        "reference": {
            "round": manifest.reference.round,
            "registration_channel": str(reference_task.registration_channel_source),
            "registration_manifest_file": (
                manifest.reference.registration_channel.as_posix()
            ),
            "registration_output": (
                reference_task.registration_channel_output.relative_to(
                    layout.root
                ).as_posix()
            ),
            "crop_output": (
                None
                if reference_task.crop_bounds_path is None
                else reference_task.crop_bounds_path.relative_to(layout.root).as_posix()
            ),
            "channels": [
                {
                    "file": str(channel_task.source),
                    "manifest_file": channel.file.as_posix(),
                    "output": channel_task.output.relative_to(layout.root).as_posix(),
                }
                for channel, channel_task in zip(
                    manifest.reference.channels,
                    reference_task.channels,
                    strict=True,
                )
            ],
        },
        "registration": {
            "crop": manifest.registration.crop,
            "linear": {
                "type": manifest.registration.linear.type,
                "accuracy": manifest.registration.linear.accuracy,
            },
            "nonlinear": _nonlinear_mapping(manifest.registration.nonlinear),
        },
        "sources": [
            _header_mapping(headers[path]) for path in sorted(headers, key=str)
        ],
        "rounds": [
            {
                "round": round_spec.round,
                "strategy": round_spec.strategy,
                "registration_channel": str(
                    resolved_inputs[round_spec.registration_channel]
                ),
                "registration_manifest_channel": (
                    round_spec.registration_channel.as_posix()
                ),
                "chunk_size_zyx": (
                    None
                    if round_spec.chunk_size_zyx is None
                    else list(round_spec.chunk_size_zyx)
                ),
                "overlap_fraction": round_spec.overlap_fraction,
                "channels": [
                    {
                        "file": str(resolved_inputs[channel.file]),
                        "manifest_file": channel.file.as_posix(),
                        "output": channel_task.output.relative_to(
                            layout.root
                        ).as_posix(),
                        "interpolation": channel.interpolation,
                    }
                    for channel, channel_task in zip(
                        round_spec.channels,
                        tasks_by_round[round_spec.round].channels,
                        strict=True,
                    )
                ],
            }
            for round_spec in sorted(manifest.rounds, key=lambda item: item.round)
        ],
        "policies": {
            "axis_convention": AXIS_CONVENTION_VERSION,
            "output_format": OUTPUT_FORMAT_POLICY,
            "postprocessing": CHANNEL_POSTPROCESSING_POLICY,
        },
    }


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


def _schema_differences(
    actual: object,
    expected: object,
    path: str = "",
) -> list[str]:
    location = path or "<root>"
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{location}: expected object, got {type(actual).__name__}"]
        differences: list[str] = []
        actual_keys = set(actual)
        expected_keys = set(expected)
        for key in sorted(expected_keys - actual_keys):
            child = f"{path}.{key}" if path else key
            differences.append(f"{child}: missing key")
        for key in sorted(actual_keys - expected_keys):
            child = f"{path}.{key}" if path else key
            differences.append(f"{child}: unexpected key")
        for key in sorted(actual_keys & expected_keys):
            child = f"{path}.{key}" if path else key
            differences.extend(_schema_differences(actual[key], expected[key], child))
        return differences
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{location}: expected array, got {type(actual).__name__}"]
        differences = []
        for index, actual_item in enumerate(actual):
            if index < len(expected):
                expected_item = expected[index]
            elif expected:
                expected_item = expected[-1]
            else:
                continue
            differences.extend(
                _schema_differences(
                    actual_item,
                    expected_item,
                    f"{path}[{index}]",
                )
            )
        return differences
    if expected is None:
        return [] if actual is None else [f"{location}: expected null"]
    if type(actual) is not type(expected):
        return [
            f"{location}: expected {type(expected).__name__}, "
            f"got {type(actual).__name__}"
        ]
    return []


def _value_differences(
    actual: object,
    expected: object,
    path: str = "",
) -> list[str]:
    location = path or "<root>"
    if isinstance(actual, dict) and isinstance(expected, dict):
        differences: list[str] = []
        for key in sorted(set(actual) | set(expected)):
            child = f"{path}.{key}" if path else key
            if key not in actual:
                differences.append(
                    f"{child}: existing=<missing>, planned={expected[key]!r}"
                )
            elif key not in expected:
                differences.append(
                    f"{child}: existing={actual[key]!r}, planned=<missing>"
                )
            else:
                differences.extend(
                    _value_differences(actual[key], expected[key], child)
                )
        return differences
    if isinstance(actual, list) and isinstance(expected, list):
        differences = []
        for index in range(max(len(actual), len(expected))):
            child = f"{path}[{index}]"
            if index >= len(actual):
                differences.append(
                    f"{child}: existing=<missing>, planned={expected[index]!r}"
                )
            elif index >= len(expected):
                differences.append(
                    f"{child}: existing={actual[index]!r}, planned=<missing>"
                )
            else:
                differences.extend(
                    _value_differences(actual[index], expected[index], child)
                )
        return differences
    if canonical_json(actual) != canonical_json(expected):
        return [f"{location}: existing={actual!r}, planned={expected!r}"]
    return []


def _validate_existing_plan(
    plan_path: Path,
    scientific_mapping: dict[str, object],
    scientific_fingerprint: str,
) -> None:
    if not plan_path.exists() and not plan_path.is_symlink():
        return
    try:
        text = plan_path.read_text(encoding="utf-8")
        existing = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_float=_finite_json_float,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise WorkflowPlanError(
            f"Existing scientific plan is not strict JSON at {plan_path}: {error}; "
            "select a new output root"
        ) from error
    schema_differences = _schema_differences(existing, scientific_mapping)
    if schema_differences:
        raise WorkflowPlanError(
            f"Existing scientific plan does not match the exact schema at {plan_path}: "
            + "; ".join(schema_differences)
            + "; select a new output root"
        )
    existing_json = canonical_json(existing)
    existing_fingerprint = sha256_fingerprint(existing_json)
    if existing_fingerprint != scientific_fingerprint:
        differences = _value_differences(existing, scientific_mapping)
        raise WorkflowPlanError(
            f"Existing scientific-plan fingerprint mismatch at {plan_path}: "
            + "; ".join(differences)
            + "; select a new output root"
        )


def build_workflow_plan(
    manifest: DatasetManifest,
    execution: ExecutionProfile,
) -> WorkflowPlan:
    """Validate inputs and construct a deterministic plan without filesystem writes."""
    try:
        _validate_dataset_manifest(manifest)
    except ConfigurationValidationError as error:
        raise WorkflowPlanError(f"Dataset manifest is invalid: {error}") from error
    try:
        _validate_execution_profile(execution)
    except ConfigurationValidationError as error:
        raise WorkflowPlanError(f"Execution profile is invalid: {error}") from error
    dataset_root, output_root = _resolved_roots(execution)
    normalized_execution = replace(
        execution,
        paths=ExecutionPaths(
            dataset_root=dataset_root,
            output_root=output_root,
            worker_local_directory=execution.paths.worker_local_directory,
        ),
    )
    resolved_inputs, yaml_paths_by_resolved = _resolve_inputs(manifest, dataset_root)
    unique_paths = tuple(yaml_paths_by_resolved)
    headers = _inspect_headers(
        unique_paths,
        yaml_paths_by_resolved,
        manifest.voxel_spacing_um,
    )
    layout = WorkflowLayout.for_dataset(output_root, manifest.dataset_id)
    reference_task = _reference_task(
        manifest,
        normalized_execution,
        layout,
        resolved_inputs,
        headers,
    )
    tasks = _round_tasks(
        manifest,
        normalized_execution,
        layout,
        resolved_inputs,
        reference_task.registration_channel_output,
    )
    artifacts = _final_artifacts(manifest, reference_task, tasks)
    _validate_output_paths(
        layout,
        output_root,
        frozenset(resolved_inputs.values()),
        reference_task,
        tasks,
        artifacts,
    )
    scientific_mapping = _scientific_mapping(
        manifest,
        resolved_inputs,
        headers,
        layout,
        reference_task,
        tasks,
    )
    scientific_json = canonical_json(scientific_mapping)
    scientific_fingerprint = sha256_fingerprint(scientific_json)
    _validate_existing_plan(
        layout.plan_path,
        scientific_mapping,
        scientific_fingerprint,
    )
    return WorkflowPlan(
        schema_version=manifest.schema_version,
        dataset_id=manifest.dataset_id,
        manifest=manifest,
        execution=normalized_execution,
        reference_task=reference_task,
        fixed_image=reference_task.registration_channel_output,
        tasks=tasks,
        expected_artifacts=tuple(artifact.path for artifact in artifacts),
        scientific_json=scientific_json,
        scientific_fingerprint=scientific_fingerprint,
    )


__all__ = ("build_workflow_plan",)
