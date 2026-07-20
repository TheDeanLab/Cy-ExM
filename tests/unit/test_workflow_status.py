from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from cyexm.errors import WorkflowPlanError
from cyexm.registration.artifacts import (
    ArtifactStore,
    CompletionRecord,
    FileIdentity,
    StageRequest,
)
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.spatial import TIFF_GEOMETRY_ATOL, ImageGeometry
from cyexm.workflow import inspect_workflow_status
from cyexm.workflow.layout import WorkflowLayout
from cyexm.workflow.models import canonical_json, sha256_fingerprint

pytestmark = pytest.mark.unit


def _scientific_mapping(
    output_root: Path,
    rounds: tuple[int, ...] = (2,),
) -> dict[str, object]:
    dataset_root = output_root.parent / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sources = [dataset_root / "fixed.tif"]
    sources.extend(
        dataset_root / f"round_{round_number}_488.tif" for round_number in rounds
    )
    for source in sources:
        if not source.exists():
            source.write_bytes(f"source:{source.name}".encode())

    def source_mapping(path: Path) -> dict[str, object]:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "shape_zyx": [4, 8, 8],
            "dtype": "uint16",
            "axes": "ZYX",
            "geometry": None,
            "st_size": stat.st_size,
            "st_mtime_ns": stat.st_mtime_ns,
        }

    return {
        "schema_version": 1,
        "dataset_id": "dataset-1",
        "voxel_spacing_xyz_um": [1.0, 1.0, 1.0],
        "reference": {
            "round": 1,
            "channel": str((dataset_root / "fixed.tif").resolve()),
        },
        "registration": {
            "crop": False,
            "linear": {"type": "TRSAA", "accuracy": "high"},
            "nonlinear": {"type": "SyNOnly", "accuracy": "high"},
        },
        "sources": [source_mapping(path) for path in sorted(sources)],
        "rounds": [
            {
                "round": round_number,
                "strategy": "whole_image",
                "registration_channel": str(
                    (dataset_root / f"round_{round_number}_488.tif").resolve()
                ),
                "registration_manifest_channel": f"round_{round_number}_488.tif",
                "chunk_size_zyx": None,
                "overlap_fraction": None,
                "channels": [
                    {
                        "file": str(
                            (dataset_root / f"round_{round_number}_488.tif").resolve()
                        ),
                        "manifest_file": f"round_{round_number}_488.tif",
                        "output": (
                            f"rounds/round-{round_number:03d}/channels/"
                            f"round_{round_number}_488.tif"
                        ),
                        "interpolation": "linear",
                    }
                ],
            }
            for round_number in sorted(rounds)
        ],
        "policies": {
            "axis_convention": "external-zyx-ants-xyz-v1",
            "output_format": "ome-compatible-tiff-uint16-v1",
            "postprocessing": "clip-uint16-no-normalization-v1",
        },
    }


def _write_plan(output_root: Path, rounds: tuple[int, ...] = (2,)) -> str:
    scientific_json = canonical_json(_scientific_mapping(output_root, rounds))
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


def _geometry() -> ImageGeometry:
    return ImageGeometry(
        shape=(4, 8, 8),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )


def _native_threads() -> dict[str, int]:
    return {name: 1 for name in NATIVE_THREAD_ENV_VARS}


def _request_for_artifact(path: Path, output_root: Path) -> StageRequest:
    round_number = int(
        next(part for part in path.parts if part.startswith("round-"))[6:]
    )
    dataset_root = output_root.parent / "dataset"
    fixed = dataset_root / "fixed.tif"
    moving = dataset_root / f"round_{round_number}_488.tif"
    inputs = (FileIdentity.from_path(fixed), FileIdentity.from_path(moving))
    common = {
        "axis_convention_version": "external-zyx-ants-xyz-v1",
        "crop": False,
        "imaging_round": round_number,
        "pixel_size_xyz": [1.0, 1.0, 1.0],
    }
    registration = (
        WorkflowLayout.for_dataset(output_root, "dataset-1")
        .round(round_number)
        .registration
    )
    affine = registration / f"linear_transform_{round_number}.mat"
    warp = registration / f"nonlinear_warp_{round_number}.nii.gz"
    if path.name.startswith("linear_transform_"):
        return StageRequest.for_identities(
            stage="affine",
            inputs=inputs,
            geometry=_geometry(),
            method="TRSAA",
            accuracy="high",
            native_threads=_native_threads(),
            upstream=None,
            parameters={
                **common,
                "image_interpolation": "linear",
                "mask_interpolation": "nearestNeighbor",
                "moving_geometry": _geometry().to_dict(),
                "registration_parameters": {
                    "aff_metric": "mattes",
                    "aff_random_sampling_rate": 0.5,
                    "aff_sampling": 32,
                    "mask_all_stages": True,
                    "verbose": True,
                    "reg_iterations": [1000, 500, 250, 100],
                },
            },
        )
    if path.name.startswith("nonlinear_warp_"):
        return StageRequest.for_identities(
            stage="nonlinear",
            inputs=inputs,
            geometry=_geometry(),
            method="SyNOnly",
            accuracy="high",
            native_threads=_native_threads(),
            upstream=FileIdentity.from_path(affine),
            parameters={
                **common,
                "image_interpolation": "linear",
                "mask_interpolation": "nearestNeighbor",
                "moving_geometry": _geometry().to_dict(),
                "registration_mode": "whole-image",
                "registration_parameters": {
                    "aff_random_sampling_rate": 1.0,
                    "flow_sigma": 3.0,
                    "initial_transform": "Identity",
                    "mask_all_stages": True,
                    "reg_iterations": [100, 70, 50],
                    "singleprecision": False,
                    "syn_metric": "mattes",
                    "syn_sampling": 32,
                    "total_sigma": 0.5,
                    "verbose": True,
                },
                "scalar_postprocessing_policy": "whole-global-histogram-match-v1",
            },
        )
    if path.name.startswith("nonlinear_registered_"):
        return StageRequest.for_identities(
            stage="registered-image",
            inputs=inputs,
            geometry=_geometry(),
            method="SyNOnly",
            accuracy="high",
            native_threads=_native_threads(),
            upstream=FileIdentity.from_path(warp),
            parameters={
                **common,
                "format": "uint16-tiff",
                "minimum": 0.0,
                "maximum": 1.0,
                "scalar_postprocessing_policy": "whole-global-histogram-match-v1",
            },
        )
    source = dataset_root / path.name
    return StageRequest.for_identities(
        stage="channel-transform-v1",
        inputs=(
            FileIdentity.from_path(fixed),
            FileIdentity.from_path(source),
            FileIdentity.from_path(affine),
            FileIdentity.from_path(warp),
        ),
        geometry=_geometry(),
        method="warp-affine",
        accuracy="not-applicable",
        native_threads=_native_threads(),
        upstream=FileIdentity.from_path(warp),
        parameters={
            "pixel_size_xyz_um": [1.0, 1.0, 1.0],
            "interpolation": "linear",
            "transform_order": ["warp", "affine"],
            "whichtoinvert": [False, False],
            "axis_convention": "external-zyx-ants-xyz-v1",
            "output_format": "ome-compatible-tiff-uint16-v1",
            "postprocessing": "clip-uint16-no-normalization-v1",
        },
    )


def _commit_artifact(
    path: Path,
    output_root: Path,
    *,
    stage_override: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"artifact")
    request = _request_for_artifact(path, output_root)
    if stage_override is not None:
        request = replace(request, stage=stage_override)
    record = CompletionRecord(
        schema_version=1,
        request=request,
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
    scientific = _scientific_mapping(output_root)
    scientific["schema_version"] = True
    plan_path = WorkflowLayout.for_dataset(output_root, "dataset-1").plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(canonical_json(scientific), encoding="utf-8")

    with pytest.raises(WorkflowPlanError, match="schema_version"):
        inspect_workflow_status(output_root, "dataset-1")


def test_status_accepts_embedded_spacing_within_planner_tolerance(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    scientific = _scientific_mapping(output_root)
    embedded_spacing = [1.0 + TIFF_GEOMETRY_ATOL / 2.0, 1.0, 1.0]
    for source in scientific["sources"]:  # type: ignore[union-attr]
        source["geometry"] = {  # type: ignore[index]
            "shape": source["shape_zyx"],  # type: ignore[index]
            "origin": [0.0, 0.0, 0.0],
            "spacing": embedded_spacing,
            "direction": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    plan_path = WorkflowLayout.for_dataset(output_root, "dataset-1").plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(canonical_json(scientific), encoding="utf-8")

    status = inspect_workflow_status(output_root, "dataset-1")

    assert status.rounds[0].status == "pending"


def test_status_rejects_duplicate_channel_source_files(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    scientific = _scientific_mapping(output_root)
    channel = scientific["rounds"][0]["channels"][0]  # type: ignore[index]
    scientific["rounds"][0]["channels"].append(channel.copy())  # type: ignore[index]
    plan_path = WorkflowLayout.for_dataset(output_root, "dataset-1").plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(canonical_json(scientific), encoding="utf-8")

    with pytest.raises(WorkflowPlanError, match=r"channel.*files.*unique"):
        inspect_workflow_status(output_root, "dataset-1")


def test_status_rejects_duplicate_channel_output_paths(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    scientific = _scientific_mapping(output_root)
    fixed_path = scientific["reference"]["channel"]  # type: ignore[index]
    scientific["rounds"][0]["channels"].append(  # type: ignore[index]
        {
            "file": fixed_path,
            "manifest_file": "nested/round_2_488.tif",
            "output": "rounds/round-002/channels/round_2_488.tif",
            "interpolation": "linear",
        }
    )
    plan_path = WorkflowLayout.for_dataset(output_root, "dataset-1").plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(canonical_json(scientific), encoding="utf-8")

    with pytest.raises(WorkflowPlanError, match=r"channel.*outputs.*unique"):
        inspect_workflow_status(output_root, "dataset-1")


def test_status_rejects_registration_of_the_reference_round(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    scientific = _scientific_mapping(output_root, rounds=(1,))
    plan_path = WorkflowLayout.for_dataset(output_root, "dataset-1").plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(canonical_json(scientific), encoding="utf-8")

    with pytest.raises(WorkflowPlanError, match="reference round"):
        inspect_workflow_status(output_root, "dataset-1")


def test_status_rejects_unused_declared_sources(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    scientific = _scientific_mapping(output_root)
    unused = output_root.parent / "dataset" / "unused.tif"
    unused.write_bytes(b"unused")
    stat = unused.stat()
    scientific["sources"].append(  # type: ignore[union-attr]
        {
            "path": str(unused.resolve()),
            "shape_zyx": [4, 8, 8],
            "dtype": "uint16",
            "axes": "ZYX",
            "geometry": None,
            "st_size": stat.st_size,
            "st_mtime_ns": stat.st_mtime_ns,
        }
    )
    scientific["sources"].sort(key=lambda item: item["path"])  # type: ignore[union-attr]
    plan_path = WorkflowLayout.for_dataset(output_root, "dataset-1").plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(canonical_json(scientific), encoding="utf-8")

    with pytest.raises(WorkflowPlanError, match="unused"):
        inspect_workflow_status(output_root, "dataset-1")


def test_status_rejects_source_shape_incompatible_with_reference(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    scientific = _scientific_mapping(output_root)
    moving_path = str((output_root.parent / "dataset" / "round_2_488.tif").resolve())
    moving = next(  # type: ignore[union-attr]
        source for source in scientific["sources"] if source["path"] == moving_path
    )
    moving["shape_zyx"] = [5, 8, 8]
    plan_path = WorkflowLayout.for_dataset(output_root, "dataset-1").plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(canonical_json(scientific), encoding="utf-8")

    with pytest.raises(WorkflowPlanError, match=r"shape.*reference"):
        inspect_workflow_status(output_root, "dataset-1")


def test_status_rejects_dataset_output_symlink_escape(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output_root / "dataset-1").symlink_to(outside, target_is_directory=True)
    marker = outside / "scientific-plan.json"
    marker.write_text(
        canonical_json(_scientific_mapping(output_root)), encoding="utf-8"
    )

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


def test_status_rejects_scientific_plan_symlink_escape(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    scientific_json = canonical_json(_scientific_mapping(output_root))
    layout = WorkflowLayout.for_dataset(output_root, "dataset-1")
    layout.root.mkdir(parents=True)
    outside = tmp_path / "outside-plan.json"
    outside.write_text(scientific_json, encoding="utf-8")
    layout.plan_path.symlink_to(outside)

    with pytest.raises(WorkflowPlanError, match="outside workflow output"):
        inspect_workflow_status(output_root, "dataset-1")


def test_status_rejects_completion_record_symlink_escape(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifact = _expected_artifacts(output_root, 2)[0]
    _commit_artifact(artifact, output_root)
    record = ArtifactStore().record_path(artifact)
    outside = tmp_path / "outside-completion.json"
    record.replace(outside)
    record.symlink_to(outside)

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
    _commit_artifact(artifacts[0], output_root)

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
        _commit_artifact(artifact, output_root)

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
    _commit_artifact(artifact, output_root)
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
        _commit_artifact(artifact, output_root)
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
    _commit_artifact(artifact, output_root)
    artifact.write_bytes(b"changed after commit")

    status = inspect_workflow_status(output_root, "dataset-1")

    assert status.rounds[0].status == "invalid"
    assert status.rounds[0].invalid_artifacts == (str(artifact),)


def test_status_rejects_identity_matching_artifact_with_unexpected_request(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifact = _expected_artifacts(output_root, 2)[0]
    _commit_artifact(artifact, output_root, stage_override="test")

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[0]

    assert round_status.status == "invalid"
    assert round_status.invalid_artifacts == (str(artifact),)


def test_status_rejects_affine_request_with_wrong_scientific_method(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifact = _expected_artifacts(output_root, 2)[0]
    _commit_artifact(artifact, output_root)
    record_path = ArtifactStore().record_path(artifact)
    record = CompletionRecord.read(record_path)
    record_path.write_text(
        replace(record, request=replace(record.request, method="Affine")).to_json(),
        encoding="utf-8",
    )

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[0]

    assert round_status.status == "invalid"
    assert round_status.invalid_artifacts == (str(artifact),)


def test_status_rejects_descendants_of_invalid_affine_provenance(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    affine, warp, *_ = _expected_artifacts(output_root, 2)
    _commit_artifact(affine, output_root)
    _commit_artifact(warp, output_root)
    record_path = ArtifactStore().record_path(affine)
    record = CompletionRecord.read(record_path)
    record_path.write_text(
        replace(record, request=replace(record.request, method="Affine")).to_json(),
        encoding="utf-8",
    )

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[0]

    assert round_status.status == "invalid"
    assert round_status.invalid_artifacts == (str(affine), str(warp))


def test_status_rejects_noncanonical_duplicate_completion_record_keys(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifact = _expected_artifacts(output_root, 2)[0]
    _commit_artifact(artifact, output_root)
    record_path = ArtifactStore().record_path(artifact)
    text = record_path.read_text(encoding="utf-8")
    record_path.write_text(
        text.replace(
            '"schema_version": 1',
            '"schema_version": 1,\n  "schema_version": 1',
            1,
        ),
        encoding="utf-8",
    )

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[0]

    assert round_status.status == "invalid"
    assert round_status.invalid_artifacts == (str(artifact),)


def test_status_conservatively_rejects_crop_geometry_without_full_provenance(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    scientific = _scientific_mapping(output_root)
    scientific["registration"]["crop"] = True  # type: ignore[index]
    plan_path = WorkflowLayout.for_dataset(output_root, "dataset-1").plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(canonical_json(scientific), encoding="utf-8")
    artifact = _expected_artifacts(output_root, 2)[0]
    _commit_artifact(artifact, output_root)
    record_path = ArtifactStore().record_path(artifact)
    record = CompletionRecord.read(record_path)
    cropped = ImageGeometry(
        shape=(2, 4, 4),
        origin=(100.0, 100.0, 100.0),
        spacing=(1.0, 1.0, 1.0),
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    record_path.write_text(
        replace(
            record,
            request=replace(
                record.request,
                geometry=cropped,
                parameters={
                    **record.request.parameters,
                    "crop": True,
                    "moving_geometry": _geometry().to_dict(),
                },
            ),
        ).to_json(),
        encoding="utf-8",
    )

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[0]

    assert round_status.status == "invalid"
    assert round_status.invalid_artifacts == (str(artifact),)


def test_status_rejects_warp_without_affine_provenance(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    affine, warp, *_ = _expected_artifacts(output_root, 2)
    _commit_artifact(affine, output_root)
    _commit_artifact(warp, output_root)
    record_path = ArtifactStore().record_path(warp)
    record = CompletionRecord.read(record_path)
    record_path.write_text(
        replace(record, request=replace(record.request, upstream=None)).to_json(),
        encoding="utf-8",
    )

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[0]

    assert round_status.status == "invalid"
    assert round_status.invalid_artifacts == (str(warp),)


def test_status_rejects_records_after_a_scientific_source_changes(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifacts = _expected_artifacts(output_root, 2)
    for artifact in artifacts:
        _commit_artifact(artifact, output_root)
    (output_root.parent / "dataset" / "round_2_488.tif").write_bytes(b"changed")

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[0]

    assert round_status.status == "invalid"
    assert round_status.invalid_artifacts == tuple(map(str, artifacts))


def test_status_serializes_as_json_compatible_values(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root, rounds=(3, 2))

    payload = inspect_workflow_status(output_root, "dataset-1").to_dict()

    assert [item["round"] for item in payload["rounds"]] == [2, 3]  # type: ignore[index]
    json.dumps(payload, allow_nan=False)


def test_status_invocation_keeps_heavy_runtimes_unloaded() -> None:
    code = """
import sys
import tempfile
from pathlib import Path
from cyexm.workflow import inspect_workflow_status

with tempfile.TemporaryDirectory() as directory:
    inspect_workflow_status(Path(directory) / "output", "dataset-1")

for root in (
    "ants", "numpy", "scipy", "tifffile", "dask", "distributed", "dask_jobqueue"
):
    assert not any(
        name == root or name.startswith(root + ".") for name in sys.modules
    ), root
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
