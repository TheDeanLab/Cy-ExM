from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyexm.errors import WorkflowPlanError
from cyexm.registration.artifacts import (
    ArtifactStore,
    CompletionRecord,
    FileIdentity,
    StageRequest,
)
from cyexm.spatial import ImageGeometry
from cyexm.workflow import inspect_workflow_status
from cyexm.workflow.layout import WorkflowLayout
from cyexm.workflow.models import canonical_json, sha256_fingerprint

pytestmark = pytest.mark.unit


def _scientific_mapping(rounds: tuple[int, ...] = (2,)) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": "dataset-1",
        "rounds": [
            {
                "round": round_number,
                "channels": [
                    {
                        "output": (
                            f"rounds/round-{round_number:03d}/channels/"
                            f"round_{round_number}_488.tif"
                        )
                    }
                ],
            }
            for round_number in rounds
        ],
    }


def _write_plan(output_root: Path, rounds: tuple[int, ...] = (2,)) -> str:
    scientific_json = canonical_json(_scientific_mapping(rounds))
    path = WorkflowLayout.for_dataset(output_root, "dataset-1").plan_path
    path.parent.mkdir(parents=True)
    path.write_text(scientific_json + "\n", encoding="utf-8")
    return sha256_fingerprint(scientific_json)


def _expected_artifacts(output_root: Path, round_number: int) -> tuple[Path, ...]:
    round_layout = WorkflowLayout.for_dataset(output_root, "dataset-1").round(
        round_number
    )
    return (
        round_layout.registration / f"linear_transform_{round_number}.mat",
        round_layout.registration / f"nonlinear_warp_{round_number}.nii.gz",
        round_layout.registration / f"nonlinear_registered_{round_number}.tif",
        round_layout.channels / f"round_{round_number}_488.tif",
    )


def _commit_artifact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"artifact")
    geometry = ImageGeometry(
        shape=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    record = CompletionRecord(
        schema_version=1,
        request=StageRequest.for_identities(
            stage="test",
            inputs=(),
            geometry=geometry,
            method="test",
            accuracy="test",
            native_threads={},
            upstream=None,
            parameters={},
        ),
        artifact=FileIdentity.from_path(path),
    )
    ArtifactStore().record_path(path).write_text(record.to_json(), encoding="utf-8")


def test_status_missing_plan_is_pending_without_creating_files(tmp_path: Path) -> None:
    output_root = tmp_path / "output"

    status = inspect_workflow_status(output_root, "dataset-1")

    assert status.dataset_id == "dataset-1"
    assert status.scientific_fingerprint is None
    assert status.rounds == ()
    assert not status.complete
    assert not output_root.exists()


def test_status_rejects_an_unsafe_dataset_identifier_without_reading_outside_root(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "scientific-plan.json"
    marker.write_text("do not read", encoding="utf-8")

    with pytest.raises(WorkflowPlanError, match="dataset_id"):
        inspect_workflow_status(output_root, "../outside")

    assert marker.read_text(encoding="utf-8") == "do not read"


def test_status_rejects_boolean_scientific_schema_version(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    scientific = _scientific_mapping()
    scientific["schema_version"] = True
    plan_path = WorkflowLayout.for_dataset(output_root, "dataset-1").plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(canonical_json(scientific), encoding="utf-8")

    with pytest.raises(WorkflowPlanError, match="schema_version"):
        inspect_workflow_status(output_root, "dataset-1")


def test_status_rejects_dataset_output_symlink_escape(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output_root / "dataset-1").symlink_to(outside, target_is_directory=True)
    marker = outside / "scientific-plan.json"
    marker.write_text(canonical_json(_scientific_mapping()), encoding="utf-8")

    with pytest.raises(WorkflowPlanError, match="outside the output root"):
        inspect_workflow_status(output_root, "dataset-1")


def test_status_rejects_round_artifact_symlink_escape(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    registration = (
        WorkflowLayout.for_dataset(output_root, "dataset-1").round(2).registration
    )
    registration.parent.mkdir(parents=True)
    outside = tmp_path / "outside-registration"
    outside.mkdir()
    registration.symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkflowPlanError, match="outside workflow output"):
        inspect_workflow_status(output_root, "dataset-1")


def test_status_classifies_a_round_with_no_artifacts_as_pending(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    fingerprint = _write_plan(output_root)

    status = inspect_workflow_status(output_root, "dataset-1")

    assert status.scientific_fingerprint == fingerprint
    assert status.rounds[0].status == "pending"
    assert status.rounds[0].complete_artifacts == ()
    assert status.rounds[0].incomplete_artifacts == tuple(
        str(path) for path in _expected_artifacts(output_root, 2)
    )
    assert status.rounds[0].invalid_artifacts == ()


def test_status_classifies_mixed_complete_and_missing_artifacts_as_partial(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifacts = _expected_artifacts(output_root, 2)
    _commit_artifact(artifacts[0])

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[0]

    assert round_status.status == "partial"
    assert round_status.complete_artifacts == (str(artifacts[0]),)
    assert round_status.incomplete_artifacts == tuple(
        str(path) for path in artifacts[1:]
    )


def test_status_requires_every_artifact_completion_record(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifacts = _expected_artifacts(output_root, 2)
    for artifact in artifacts:
        _commit_artifact(artifact)

    status = inspect_workflow_status(output_root, "dataset-1")

    assert status.rounds[0].status == "complete"
    assert status.rounds[0].complete_artifacts == tuple(map(str, artifacts))
    assert status.rounds[0].incomplete_artifacts == ()
    assert status.rounds[0].invalid_artifacts == ()
    assert status.complete


def test_status_marks_a_corrupt_completion_record_invalid(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifact = _expected_artifacts(output_root, 2)[0]
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"artifact")
    record = ArtifactStore().record_path(artifact)
    record.write_text("{not-json", encoding="utf-8")

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[0]

    assert round_status.status == "invalid"
    assert round_status.invalid_artifacts == (str(artifact),)


def test_status_treats_an_artifact_without_a_record_as_incomplete(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifact = _expected_artifacts(output_root, 2)[0]
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"uncommitted")

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[0]

    assert round_status.status == "pending"
    assert str(artifact) in round_status.incomplete_artifacts


def test_status_marks_a_completion_record_without_its_artifact_invalid(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifact = _expected_artifacts(output_root, 2)[0]
    _commit_artifact(artifact)
    artifact.unlink()

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[0]

    assert round_status.status == "invalid"
    assert round_status.invalid_artifacts == (str(artifact),)


def test_status_ignores_missing_or_corrupt_operational_run_summaries(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifacts = _expected_artifacts(output_root, 2)
    for artifact in artifacts:
        _commit_artifact(artifact)
    runs = WorkflowLayout.for_dataset(output_root, "dataset-1").workflow_runs
    runs.mkdir()
    (runs / "corrupt.json").write_text("not JSON", encoding="utf-8")

    status = inspect_workflow_status(output_root, "dataset-1")

    assert status.complete
    assert status.rounds[0].status == "complete"


def test_status_marks_identity_mismatch_invalid(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifact = _expected_artifacts(output_root, 2)[0]
    _commit_artifact(artifact)
    artifact.write_bytes(b"changed after commit")

    status = inspect_workflow_status(output_root, "dataset-1")

    assert status.rounds[0].status == "invalid"
    assert status.rounds[0].invalid_artifacts == (str(artifact),)


def test_status_serializes_as_json_compatible_values(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root, rounds=(3, 2))

    payload = inspect_workflow_status(output_root, "dataset-1").to_dict()

    assert [item["round"] for item in payload["rounds"]] == [2, 3]  # type: ignore[index]
    json.dumps(payload, allow_nan=False)
