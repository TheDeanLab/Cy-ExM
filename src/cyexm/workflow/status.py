from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from cyexm.errors import WorkflowPlanError
from cyexm.registration.artifacts import CompletionRecord, FileIdentity
from cyexm.workflow.layout import WorkflowLayout
from cyexm.workflow.models import (
    RoundStatus,
    WorkflowStatus,
    canonical_json,
    sha256_fingerprint,
)

_DATASET_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


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


@dataclass(frozen=True, slots=True)
class _ScientificSnapshot:
    fingerprint: str
    expected_by_round: tuple[tuple[int, tuple[Path, ...]], ...]


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


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise WorkflowPlanError(f"{location} must be a JSON object")
    return value


def _built_in_round(value: object, location: str) -> int:
    if type(value) is not int or value < 0:
        raise WorkflowPlanError(
            f"{location} must be a built-in non-negative integer; got {value!r}"
        )
    return value


def _channel_output(
    value: object,
    *,
    layout: WorkflowLayout,
    location: str,
) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise WorkflowPlanError(
            f"{location} must be a safe relative POSIX path; got {value!r}"
        )
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
        or path == PurePosixPath(".")
    ):
        raise WorkflowPlanError(
            f"{location} must be a safe relative POSIX path; got {value!r}"
        )
    destination = layout.root.joinpath(*path.parts)
    return _safe_artifact_path(destination, layout=layout, location=location)


def _safe_artifact_path(
    destination: Path,
    *,
    layout: WorkflowLayout,
    location: str,
) -> Path:
    root = layout.root.resolve(strict=False)
    resolved = destination.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise WorkflowPlanError(
            f"{location} resolves outside workflow output {root}; got {resolved}"
        )
    return destination


def _round_artifacts(
    value: object,
    *,
    layout: WorkflowLayout,
    index: int,
) -> tuple[int, tuple[Path, ...]]:
    round_mapping = _mapping(value, f"scientific plan rounds[{index}]")
    round_number = _built_in_round(
        round_mapping.get("round"), f"scientific plan rounds[{index}].round"
    )
    channels = round_mapping.get("channels")
    if not isinstance(channels, list):
        raise WorkflowPlanError(
            f"scientific plan rounds[{index}].channels must be a JSON array"
        )
    channel_outputs = tuple(
        _channel_output(
            _mapping(
                channel,
                f"scientific plan rounds[{index}].channels[{channel_index}]",
            ).get("output"),
            layout=layout,
            location=(
                f"scientific plan rounds[{index}].channels[{channel_index}].output"
            ),
        )
        for channel_index, channel in enumerate(channels)
    )
    round_layout = layout.round(round_number)
    registration_artifacts = tuple(
        _safe_artifact_path(
            path,
            layout=layout,
            location=f"scientific plan round {round_number} registration artifact",
        )
        for path in (
            round_layout.registration / f"linear_transform_{round_number}.mat",
            round_layout.registration / f"nonlinear_warp_{round_number}.nii.gz",
            round_layout.registration / f"nonlinear_registered_{round_number}.tif",
        )
    )
    return (
        round_number,
        (
            *registration_artifacts,
            *channel_outputs,
        ),
    )


def _read_scientific_snapshot(
    plan_path: Path,
    dataset_id: str,
) -> _ScientificSnapshot | None:
    if not plan_path.exists() and not plan_path.is_symlink():
        return None
    try:
        payload = json.loads(
            plan_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_float=_finite_json_float,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise WorkflowPlanError(
            f"Scientific plan is not strict JSON at {plan_path}: {error}"
        ) from error
    scientific = _mapping(payload, "scientific plan")
    schema_version = scientific.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise WorkflowPlanError(
            "scientific plan schema_version must be the built-in integer 1; "
            f"got {schema_version!r}"
        )
    if scientific.get("dataset_id") != dataset_id:
        raise WorkflowPlanError(
            f"scientific plan dataset_id must be {dataset_id!r}; "
            f"got {scientific.get('dataset_id')!r}"
        )
    raw_rounds = scientific.get("rounds")
    if not isinstance(raw_rounds, list):
        raise WorkflowPlanError("scientific plan rounds must be a JSON array")
    layout = WorkflowLayout(plan_path.parent)
    expected_by_round = tuple(
        sorted(
            (
                _round_artifacts(round_value, layout=layout, index=index)
                for index, round_value in enumerate(raw_rounds)
            ),
            key=lambda item: item[0],
        )
    )
    round_numbers = tuple(round_number for round_number, _ in expected_by_round)
    if len(set(round_numbers)) != len(round_numbers):
        raise WorkflowPlanError(
            f"scientific plan round numbers must be unique; got {round_numbers!r}"
        )
    encoded = canonical_json(scientific)
    return _ScientificSnapshot(
        fingerprint=sha256_fingerprint(encoded),
        expected_by_round=expected_by_round,
    )


def _artifact_state(artifact: Path) -> str:
    record_path = artifact.with_name(f"{artifact.name}.complete.json")
    artifact_present = artifact.exists() or artifact.is_symlink()
    record_present = record_path.exists() or record_path.is_symlink()
    if not artifact_present and record_present:
        return "invalid"
    if not artifact_present or not record_present:
        return "incomplete"
    if not artifact.is_file() or not record_path.is_file():
        return "invalid"
    try:
        record = CompletionRecord.read(record_path)
        identity = FileIdentity.from_path(artifact)
    except Exception:
        return "invalid"
    return "complete" if record.artifact == identity else "invalid"


def _round_status(round_number: int, artifacts: tuple[Path, ...]) -> RoundStatus:
    complete: list[str] = []
    incomplete: list[str] = []
    invalid: list[str] = []
    for artifact in artifacts:
        state = _artifact_state(artifact)
        destination = {
            "complete": complete,
            "incomplete": incomplete,
            "invalid": invalid,
        }[state]
        destination.append(str(artifact))
    if invalid:
        status = "invalid"
    elif len(complete) == len(artifacts):
        status = "complete"
    elif complete:
        status = "partial"
    else:
        status = "pending"
    return RoundStatus(
        round=round_number,
        status=status,
        complete_artifacts=tuple(complete),
        incomplete_artifacts=tuple(incomplete),
        invalid_artifacts=tuple(invalid),
    )


def inspect_workflow_status(
    output_root: str | Path,
    dataset_id: str,
) -> WorkflowStatus:
    """Inspect authoritative completion records without changing workflow state."""
    layout = _safe_workflow_layout(output_root, dataset_id)
    snapshot = _read_scientific_snapshot(layout.plan_path, dataset_id)
    if snapshot is None:
        return WorkflowStatus(
            dataset_id=dataset_id,
            scientific_fingerprint=None,
            rounds=(),
            complete=False,
        )
    rounds = tuple(
        _round_status(round_number, artifacts)
        for round_number, artifacts in snapshot.expected_by_round
    )
    return WorkflowStatus(
        dataset_id=dataset_id,
        scientific_fingerprint=snapshot.fingerprint,
        rounds=rounds,
        complete=bool(rounds) and all(item.status == "complete" for item in rounds),
    )


__all__ = ["inspect_workflow_status"]
