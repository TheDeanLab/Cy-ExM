from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from cyexm.config.models import (
    DatasetManifest,
    ExecutionPaths,
    ExecutionProfile,
    RoundSpec,
)
from cyexm.errors import InputValidationError, WorkflowPlanError
from cyexm.spatial import AXIS_CONVENTION_VERSION, TIFF_GEOMETRY_ATOL
from cyexm.workflow.layout import RoundLayout, WorkflowLayout
from cyexm.workflow.models import (
    ChannelTask,
    ImageHeader,
    RoundTask,
    WorkflowPlan,
    canonical_json,
    sha256_fingerprint,
)

_OUTPUT_FORMAT_POLICY = "ome-compatible-tiff-uint16-v1"
_POSTPROCESSING_POLICY = "clip-uint16-no-normalization-v1"


@dataclass(frozen=True, slots=True)
class _InputReference:
    yaml_path: str
    relative_path: PurePosixPath


def _input_references(manifest: DatasetManifest) -> tuple[_InputReference, ...]:
    references = [
        _InputReference(
            yaml_path="reference.channel",
            relative_path=manifest.reference.channel,
        )
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


def _inspect_tiff_header(path: Path) -> ImageHeader:
    """Inspect one TIFF's scalar layout and strict geometry without voxel loading."""
    try:
        import tifffile

        with tifffile.TiffFile(path) as handle:
            series = handle.series[0]
            shape = _shape_from_series(series, handle.shaped_metadata)
            axes = str(series.axes)
            dtype = str(series.dtype)
        from cyexm.io.tiff import read_tiff_geometry

        geometry = read_tiff_geometry(path, shape)
        stat = path.stat()
    except (IndexError, InputValidationError, OSError, TypeError, ValueError) as error:
        raise WorkflowPlanError(
            f"TIFF must have a readable scalar three-dimensional layout: {path}; "
            f"{error}"
        ) from error
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
        st_size=stat.st_size,
        st_mtime_ns=stat.st_mtime_ns,
    )


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
        if any(
            not math.isclose(
                embedded,
                requested,
                rel_tol=0.0,
                abs_tol=TIFF_GEOMETRY_ATOL,
            )
            for embedded, requested in zip(
                header.geometry.spacing,
                expected_spacing,
                strict=True,
            )
        ):
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

    reference_path = paths[0]
    reference_header = headers.get(reference_path)
    if reference_header is not None:
        for path in paths[1:]:
            header = headers.get(path)
            if header is not None and header.shape_zyx != reference_header.shape_zyx:
                labels = ", ".join(yaml_paths_by_resolved[path])
                errors.append(
                    f"{labels}: {path}: shape {header.shape_zyx!r} must match "
                    f"reference shape {reference_header.shape_zyx!r} at "
                    f"{reference_path}"
                )
    if errors:
        details = "\n".join(f"- {message}" for message in errors)
        raise WorkflowPlanError(f"Workflow TIFF header validation failed:\n{details}")
    return headers


def _validate_output_paths(
    manifest: DatasetManifest,
    layout: WorkflowLayout,
    output_root: Path,
    resolved_inputs: Mapping[PurePosixPath, Path],
) -> None:
    source_paths = frozenset(resolved_inputs.values())
    errors: list[str] = []
    outputs: dict[Path, list[str]] = {}
    planned_locations = [
        ("dataset output directory", layout.root),
        ("scientific plan", layout.plan_path),
    ]
    for round_index, round_spec in enumerate(manifest.rounds):
        round_layout = layout.round(round_spec.round)
        planned_locations.extend(
            (
                (
                    f"rounds[{round_index}] registration directory",
                    round_layout.registration,
                ),
                (f"rounds[{round_index}] log", round_layout.log),
            )
        )
        for channel_index, channel in enumerate(round_spec.channels):
            yaml_path = f"rounds[{round_index}].channels[{channel_index}].file"
            destination = round_layout.channel(channel.file).resolve(strict=False)
            planned_locations.append((f"{yaml_path} output", destination))
            outputs.setdefault(destination, []).append(yaml_path)
            if destination in source_paths:
                errors.append(
                    f"{yaml_path}: channel output collides with source input "
                    f"{destination}"
                )
    for destination, yaml_paths in outputs.items():
        if len(yaml_paths) > 1:
            errors.append(
                f"channel output collision for {destination}: {', '.join(yaml_paths)}"
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
) -> tuple[RoundTask, ...]:
    fixed_image = resolved_inputs[manifest.reference.channel]
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
) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "dataset_id": manifest.dataset_id,
        "voxel_spacing_xyz_um": list(manifest.voxel_spacing_um),
        "reference": {
            "round": manifest.reference.round,
            "channel": str(resolved_inputs[manifest.reference.channel]),
        },
        "registration": {
            "crop": manifest.registration.crop,
            "linear": {
                "type": manifest.registration.linear.type,
                "accuracy": manifest.registration.linear.accuracy,
            },
            "nonlinear": {
                "type": manifest.registration.nonlinear.type,
                "accuracy": manifest.registration.nonlinear.accuracy,
            },
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
                "chunk_size_zyx": (
                    None
                    if round_spec.chunk_size_zyx is None
                    else list(round_spec.chunk_size_zyx)
                ),
                "overlap_fraction": round_spec.overlap_fraction,
                "channels": [
                    {
                        "file": str(resolved_inputs[channel.file]),
                        "interpolation": channel.interpolation,
                    }
                    for channel in round_spec.channels
                ],
            }
            for round_spec in sorted(manifest.rounds, key=lambda item: item.round)
        ],
        "policies": {
            "axis_convention": AXIS_CONVENTION_VERSION,
            "output_format": _OUTPUT_FORMAT_POLICY,
            "postprocessing": _POSTPROCESSING_POLICY,
        },
    }


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


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
        if len(actual) != len(expected):
            differences.append(
                f"{location}: expected {len(expected)} items, got {len(actual)}"
            )
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=False)
        ):
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
    if actual != expected:
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


def _expected_artifacts(tasks: tuple[RoundTask, ...]) -> tuple[Path, ...]:
    return tuple(
        artifact
        for task in tasks
        for artifact in (
            task.registration_directory / f"linear_transform_{task.round}.mat",
            task.registration_directory / f"nonlinear_warp_{task.round}.nii.gz",
            *(channel.output for channel in task.channels),
        )
    )


def build_workflow_plan(
    manifest: DatasetManifest,
    execution: ExecutionProfile,
) -> WorkflowPlan:
    """Validate inputs and construct a deterministic plan without filesystem writes."""
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
    _validate_output_paths(manifest, layout, output_root, resolved_inputs)
    tasks = _round_tasks(
        manifest,
        normalized_execution,
        layout,
        resolved_inputs,
    )
    scientific_mapping = _scientific_mapping(manifest, resolved_inputs, headers)
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
        fixed_image=resolved_inputs[manifest.reference.channel],
        tasks=tasks,
        expected_artifacts=_expected_artifacts(tasks),
        scientific_json=scientific_json,
        scientific_fingerprint=scientific_fingerprint,
    )


__all__ = ("build_workflow_plan",)
