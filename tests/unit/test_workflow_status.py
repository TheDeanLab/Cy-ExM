from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from tiff_bytes import write_cyexm_uint16_tiff

import cyexm.workflow.status as workflow_status
from cyexm.errors import WorkflowPlanError
from cyexm.registration._request_policy import (
    CHUNK_SCALAR_POLICY,
    WHOLE_SCALAR_POLICY,
    affine_parameters,
    canonical_compact_json,
    channel_transform_parameters,
    chunk_grid,
    chunk_manifest_parameters,
    chunk_manifest_payload,
    common_parameters,
    crop_geometry,
    crop_parameters,
    local_chunk_parameters,
    nonlinear_stage_parameters,
    reference_channel_parameters,
    reference_crop_parameters,
    registered_image_parameters,
)
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
            "registration_channel": str((dataset_root / "fixed.tif").resolve()),
            "registration_manifest_file": "fixed.tif",
            "registration_output": "rounds/round-001/channels/fixed.tif",
            "crop_output": None,
            "channels": [
                {
                    "file": str((dataset_root / "fixed.tif").resolve()),
                    "manifest_file": "fixed.tif",
                    "output": "rounds/round-001/channels/fixed.tif",
                }
            ],
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
    _commit_reference(output_root)
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


def _commit_reference(output_root: Path) -> Path:
    dataset_root = output_root.parent / "dataset"
    source = dataset_root / "fixed.tif"
    output = (
        WorkflowLayout.for_dataset(output_root, "dataset-1")
        .round(1)
        .channel(Path("fixed.tif"))
    )
    request = StageRequest.for_identities(
        stage="reference-channel",
        inputs=(FileIdentity.from_path(source),),
        geometry=_geometry(),
        method="uint16-tiff",
        accuracy="exact",
        native_threads=_native_threads(),
        upstream=None,
        parameters=reference_channel_parameters(index=0, cropped=False),
    )
    ArtifactStore().commit(
        output,
        request,
        lambda path: write_cyexm_uint16_tiff(path, _geometry()),
        lambda _path: True,
    )
    return output


def _native_threads() -> dict[str, int]:
    return {name: 1 for name in NATIVE_THREAD_ENV_VARS}


def _request_for_artifact(path: Path, output_root: Path) -> StageRequest:
    round_number = int(
        next(part for part in path.parts if part.startswith("round-"))[6:]
    )
    dataset_root = output_root.parent / "dataset"
    fixed = (
        WorkflowLayout.for_dataset(output_root, "dataset-1")
        .round(1)
        .channel(Path("fixed.tif"))
    )
    moving = dataset_root / f"round_{round_number}_488.tif"
    inputs = (FileIdentity.from_path(fixed), FileIdentity.from_path(moving))
    common = {
        "axis_convention_version": "external-zyx-ants-xyz-v1",
        "crop": False,
        "imaging_round": round_number,
        "pixel_size_xyz": [1.0, 1.0, 1.0],
        "prepared_fixed": True,
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


def _write_request_artifact(
    path: Path,
    request: StageRequest,
    *,
    contents: bytes = b"artifact",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name == "fixed_reference.tif" and contents == b"artifact":
        write_cyexm_uint16_tiff(path, request.geometry)
    else:
        path.write_bytes(contents)
    record = CompletionRecord(
        schema_version=1,
        request=request,
        artifact=FileIdentity.from_path(path),
    )
    ArtifactStore().record_path(path).write_text(record.to_json(), encoding="utf-8")


def _corrupt_tiff_while_preserving_record_identity(path: Path) -> None:
    record = CompletionRecord.read(ArtifactStore().record_path(path))
    original = bytearray(path.read_bytes())
    assert len(original) >= 8
    original[:4] = b"NOPE"
    path.write_bytes(original)
    accessed = path.stat().st_atime_ns
    os.utime(path, ns=(accessed, record.artifact.st_mtime_ns))
    assert FileIdentity.from_path(path) == record.artifact


def _replace_fixed_reference_geometry(path: Path, geometry: ImageGeometry) -> None:
    record_path = ArtifactStore().record_path(path)
    record = CompletionRecord.read(record_path)
    write_cyexm_uint16_tiff(path, geometry)
    record_path.write_text(
        replace(record, artifact=FileIdentity.from_path(path)).to_json(),
        encoding="utf-8",
    )


def _commit_serial_round(output_root: Path) -> tuple[Path, ...]:
    round_number = 2
    dataset_root = output_root.parent / "dataset"
    fixed = _commit_reference(output_root)
    moving = dataset_root / "round_2_488.tif"
    inputs = (FileIdentity.from_path(fixed), FileIdentity.from_path(moving))
    threads = _native_threads()
    geometry = _geometry()
    common = common_parameters(
        crop=False,
        imaging_round=round_number,
        pixel_size_xyz=(1.0, 1.0, 1.0),
    )
    common["prepared_fixed"] = True
    round_layout = WorkflowLayout.for_dataset(output_root, "dataset-1").round(
        round_number
    )
    artifacts = _expected_artifacts(output_root, round_number)
    affine, warp, registered, channel = artifacts
    _write_request_artifact(
        affine,
        StageRequest.for_identities(
            stage="affine",
            inputs=inputs,
            geometry=geometry,
            method="TRSAA",
            accuracy="high",
            native_threads=threads,
            upstream=None,
            parameters={
                **common,
                **affine_parameters(
                    moving_geometry=geometry,
                    linear_type="TRSAA",
                    accuracy="high",
                ),
            },
        ),
    )
    affine_identity = FileIdentity.from_path(affine)
    chunk_size = (4, 8, 8)
    overlap = 0.0
    chunks = chunk_grid(geometry.shape, chunk_size, overlap)
    local_warps: list[FileIdentity] = []
    chunk_directory = round_layout.registration / "round_2_chunks"
    for chunk in chunks:
        local_warp = chunk_directory / f"chunk_{chunk['chunk_id']:04d}_warp.nii.gz"
        local_geometry = crop_geometry(
            geometry,
            (
                chunk["z_start_ext"],
                chunk["z_end_ext"],
                chunk["y_start_ext"],
                chunk["y_end_ext"],
                chunk["x_start_ext"],
                chunk["x_end_ext"],
            ),
        )
        _write_request_artifact(
            local_warp,
            StageRequest.for_identities(
                stage=f"nonlinear-chunk:{chunk['chunk_id']}",
                inputs=inputs,
                geometry=local_geometry,
                method="SyNOnly",
                accuracy="high",
                native_threads=threads,
                upstream=affine_identity,
                parameters=local_chunk_parameters(
                    chunk=chunk,
                    chunks=chunks,
                    chunk_size_zyx=chunk_size,
                    fixed_mask_geometry=local_geometry,
                    image_shape_zyx=geometry.shape,
                    moving_geometry=local_geometry,
                    moving_mask_geometry=local_geometry,
                    overlap_fraction=overlap,
                    nonlinear_accuracy="high",
                    scalar_postprocessing_policy=CHUNK_SCALAR_POLICY,
                ),
            ),
        )
        local_warps.append(FileIdentity.from_path(local_warp))
    local_dicts = [identity.to_dict() for identity in local_warps]
    manifest = chunk_directory / "manifest.json"
    manifest_payload = chunk_manifest_payload(
        round_number=round_number,
        image_shape_zyx=geometry.shape,
        chunk_size_zyx=chunk_size,
        overlap_fraction=overlap,
        chunks=chunks,
        local_warps=local_dicts,
    )
    _write_request_artifact(
        manifest,
        StageRequest.for_identities(
            stage="nonlinear-chunk-manifest",
            inputs=inputs,
            geometry=geometry,
            method="serial-chunk-grid",
            accuracy="exact",
            native_threads=threads,
            upstream=affine_identity,
            parameters=chunk_manifest_parameters(
                chunk_size_zyx=chunk_size,
                image_shape_zyx=geometry.shape,
                local_warps=local_dicts,
                nonlinear_accuracy="high",
                nonlinear_type="SyNOnly",
                overlap_fraction=overlap,
            ),
        ),
        contents=canonical_compact_json(manifest_payload).encode(),
    )
    nonlinear_evidence = {
        "chunk_grid": [dict(chunk) for chunk in chunks],
        "chunk_manifest": FileIdentity.from_path(manifest).to_dict(),
        "chunk_size_zyx": list(chunk_size),
        "local_warps": local_dicts,
        "overlap_fraction": overlap,
    }
    _write_request_artifact(
        warp,
        StageRequest.for_identities(
            stage="nonlinear",
            inputs=inputs,
            geometry=geometry,
            method="SyNOnly",
            accuracy="high",
            native_threads=threads,
            upstream=affine_identity,
            parameters={
                **common,
                **nonlinear_stage_parameters(
                    moving_geometry=geometry,
                    registration_mode="serial-chunked",
                    nonlinear_accuracy="high",
                    scalar_postprocessing_policy=CHUNK_SCALAR_POLICY,
                    evidence=nonlinear_evidence,
                ),
            },
        ),
    )
    warp_identity = FileIdentity.from_path(warp)
    _write_request_artifact(
        registered,
        StageRequest.for_identities(
            stage="registered-image",
            inputs=inputs,
            geometry=geometry,
            method="SyNOnly",
            accuracy="high",
            native_threads=threads,
            upstream=warp_identity,
            parameters={
                **common,
                **registered_image_parameters(
                    minimum=0.0,
                    maximum=1.0,
                    scalar_postprocessing_policy=CHUNK_SCALAR_POLICY,
                ),
            },
        ),
    )
    _write_request_artifact(
        channel,
        StageRequest.for_identities(
            stage="channel-transform-v1",
            inputs=(inputs[0], inputs[1], affine_identity, warp_identity),
            geometry=geometry,
            method="warp-affine",
            accuracy="not-applicable",
            native_threads=threads,
            upstream=warp_identity,
            parameters=channel_transform_parameters(
                pixel_size_xyz=(1.0, 1.0, 1.0),
                interpolation="linear",
            ),
        ),
    )
    return artifacts


def _commit_cropped_round(output_root: Path) -> tuple[Path, ...]:
    round_number = 2
    dataset_root = output_root.parent / "dataset"
    source_fixed = dataset_root / "fixed.tif"
    moving = dataset_root / "round_2_488.tif"
    threads = _native_threads()
    geometry = _geometry()
    bounds = (0, 4, 0, 8, 0, 8)
    reference_layout = WorkflowLayout.for_dataset(output_root, "dataset-1").round(1)
    reference_crop = reference_layout.reference_crop
    _write_request_artifact(
        reference_crop,
        StageRequest.for_identities(
            stage="reference-crop",
            inputs=(FileIdentity.from_path(source_fixed),),
            geometry=geometry,
            method="robust-bounding-box",
            accuracy="legacy-robust",
            native_threads=threads,
            upstream=None,
            parameters=reference_crop_parameters(),
        ),
        contents=canonical_compact_json(list(bounds)).encode(),
    )
    fixed = reference_layout.channel(Path("fixed.tif"))
    _write_request_artifact(
        fixed,
        StageRequest.for_identities(
            stage="reference-channel",
            inputs=(FileIdentity.from_path(source_fixed),),
            geometry=geometry,
            method="uint16-tiff",
            accuracy="exact",
            native_threads=threads,
            upstream=FileIdentity.from_path(reference_crop),
            parameters=reference_channel_parameters(index=0, cropped=True),
        ),
    )
    inputs = (FileIdentity.from_path(fixed), FileIdentity.from_path(moving))
    common = common_parameters(
        crop=True,
        imaging_round=round_number,
        pixel_size_xyz=(1.0, 1.0, 1.0),
    )
    common["prepared_fixed"] = True
    round_layout = WorkflowLayout.for_dataset(output_root, "dataset-1").round(
        round_number
    )
    artifacts = _expected_artifacts(output_root, round_number)
    affine, warp, registered, channel = artifacts
    _write_request_artifact(
        round_layout.registration / "moving_crop_indices_2.json",
        StageRequest.for_identities(
            stage="crop:moving",
            inputs=inputs,
            geometry=geometry,
            method="robust-bounding-box",
            accuracy="legacy-robust",
            native_threads=threads,
            upstream=None,
            parameters={**common, **crop_parameters("moving")},
        ),
        contents=canonical_compact_json(list(bounds)).encode(),
    )
    _write_request_artifact(
        affine,
        StageRequest.for_identities(
            stage="affine",
            inputs=inputs,
            geometry=geometry,
            method="TRSAA",
            accuracy="high",
            native_threads=threads,
            upstream=None,
            parameters={
                **common,
                **affine_parameters(
                    moving_geometry=geometry,
                    linear_type="TRSAA",
                    accuracy="high",
                ),
            },
        ),
    )
    affine_identity = FileIdentity.from_path(affine)
    _write_request_artifact(
        warp,
        StageRequest.for_identities(
            stage="nonlinear",
            inputs=inputs,
            geometry=geometry,
            method="SyNOnly",
            accuracy="high",
            native_threads=threads,
            upstream=affine_identity,
            parameters={
                **common,
                **nonlinear_stage_parameters(
                    moving_geometry=geometry,
                    registration_mode="whole-image",
                    nonlinear_accuracy="high",
                    scalar_postprocessing_policy=WHOLE_SCALAR_POLICY,
                ),
            },
        ),
    )
    warp_identity = FileIdentity.from_path(warp)
    _write_request_artifact(
        registered,
        StageRequest.for_identities(
            stage="registered-image",
            inputs=inputs,
            geometry=geometry,
            method="SyNOnly",
            accuracy="high",
            native_threads=threads,
            upstream=warp_identity,
            parameters={
                **common,
                **registered_image_parameters(
                    minimum=0.0,
                    maximum=1.0,
                    scalar_postprocessing_policy=WHOLE_SCALAR_POLICY,
                ),
            },
        ),
    )
    _write_request_artifact(
        channel,
        StageRequest.for_identities(
            stage="channel-transform-v1",
            inputs=(inputs[0], inputs[1], affine_identity, warp_identity),
            geometry=geometry,
            method="warp-affine",
            accuracy="not-applicable",
            native_threads=threads,
            upstream=warp_identity,
            parameters=channel_transform_parameters(
                pixel_size_xyz=(1.0, 1.0, 1.0),
                interpolation="linear",
            ),
        ),
    )
    return artifacts


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

    assert status.rounds[1].status == "pending"


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
    fixed_path = scientific["reference"]["registration_channel"]  # type: ignore[index]
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


def test_status_accepts_independently_valid_source_shapes(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    scientific = _scientific_mapping(output_root)
    fixed_path = str((output_root.parent / "dataset" / "fixed.tif").resolve())
    moving_path = str((output_root.parent / "dataset" / "round_2_488.tif").resolve())
    sources = cast(list[dict[str, object]], scientific["sources"])
    fixed = next(source for source in sources if source["path"] == fixed_path)
    moving = next(source for source in sources if source["path"] == moving_path)
    fixed["shape_zyx"] = [4, 8, 7]
    moving["shape_zyx"] = [5, 9, 8]
    plan_path = WorkflowLayout.for_dataset(output_root, "dataset-1").plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(canonical_json(scientific), encoding="utf-8")

    status = inspect_workflow_status(output_root, "dataset-1")

    assert status.rounds[1].status == "pending"


@pytest.mark.parametrize(
    "invalid_shape",
    [
        pytest.param([0, 8, 8], id="zero-dimension"),
        pytest.param([True, 8, 8], id="boolean-dimension"),
        pytest.param([4, 8], id="not-three-dimensional"),
    ],
)
def test_status_rejects_malformed_source_shapes(
    tmp_path: Path,
    invalid_shape: list[object],
) -> None:
    output_root = tmp_path / "output"
    scientific = _scientific_mapping(output_root)
    sources = cast(list[dict[str, object]], scientific["sources"])
    sources[0]["shape_zyx"] = invalid_shape
    plan_path = WorkflowLayout.for_dataset(output_root, "dataset-1").plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(canonical_json(scientific), encoding="utf-8")

    with pytest.raises(WorkflowPlanError, match="shape_zyx"):
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
    assert status.rounds[1].status == "pending"
    assert status.rounds[1].complete_artifacts == ()
    assert status.rounds[1].incomplete_artifacts == tuple(
        str(path) for path in _expected_artifacts(output_root, 2)
    )
    assert status.rounds[1].invalid_artifacts == ()


def test_status_classifies_mixed_complete_and_missing_artifacts_as_partial(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifacts = _expected_artifacts(output_root, 2)
    _commit_artifact(artifacts[0], output_root)

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[1]

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

    assert status.rounds[1].status == "complete"
    assert status.rounds[1].complete_artifacts == tuple(map(str, artifacts))
    assert status.rounds[1].incomplete_artifacts == ()
    assert status.rounds[1].invalid_artifacts == ()
    assert status.complete


def test_status_rejects_reference_channel_with_wrong_scientific_request(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    reference = (
        WorkflowLayout.for_dataset(output_root, "dataset-1")
        .round(1)
        .channel(Path("fixed.tif"))
    )
    record_path = ArtifactStore().record_path(reference)
    record = CompletionRecord.read(record_path)
    record_path.write_text(
        replace(record, request=replace(record.request, method="copy")).to_json(),
        encoding="utf-8",
    )

    status = inspect_workflow_status(output_root, "dataset-1")

    assert status.rounds[0].status == "invalid"
    assert status.rounds[0].invalid_artifacts == (str(reference),)
    assert not status.complete


def test_status_rejects_a_corrupt_fixed_reference_with_restored_identity(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifacts = _expected_artifacts(output_root, 2)
    for artifact in artifacts:
        _commit_artifact(artifact, output_root)
    fixed_reference = (
        WorkflowLayout.for_dataset(output_root, "dataset-1")
        .round(1)
        .channel(Path("fixed.tif"))
    )
    _corrupt_tiff_while_preserving_record_identity(fixed_reference)

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[0]

    assert round_status.status == "invalid"
    assert round_status.complete_artifacts == ()
    assert round_status.invalid_artifacts == (str(fixed_reference),)


@pytest.mark.parametrize(
    ("origin_delta", "expected_status"),
    [
        (TIFF_GEOMETRY_ATOL / 2.0, "complete"),
        (TIFF_GEOMETRY_ATOL * 2.0, "invalid"),
    ],
)
def test_status_compares_fixed_reference_tiff_geometry_with_tolerance(
    tmp_path: Path,
    origin_delta: float,
    expected_status: str,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifacts = _expected_artifacts(output_root, 2)
    for artifact in artifacts:
        _commit_artifact(artifact, output_root)
    fixed_reference = (
        WorkflowLayout.for_dataset(output_root, "dataset-1")
        .round(1)
        .channel(Path("fixed.tif"))
    )
    original = _geometry()
    _replace_fixed_reference_geometry(
        fixed_reference,
        replace(original, origin=(origin_delta, *original.origin[1:])),
    )

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[0]

    assert round_status.status == expected_status


def test_status_marks_an_artifact_directory_without_a_record_invalid(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifacts = _expected_artifacts(output_root, 2)
    artifacts[0].mkdir(parents=True)

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[1]

    assert round_status.status == "invalid"
    assert round_status.invalid_artifacts == (str(artifacts[0]),)
    assert round_status.incomplete_artifacts == tuple(map(str, artifacts[1:]))


def test_status_marks_a_completion_record_directory_invalid(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifact = _expected_artifacts(output_root, 2)[0]
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"artifact")
    ArtifactStore().record_path(artifact).mkdir()

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[1]

    assert round_status.status == "invalid"
    assert round_status.invalid_artifacts == (str(artifact),)


def test_status_uses_the_shared_authoritative_completion_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifacts = _expected_artifacts(output_root, 2)
    for artifact in artifacts:
        _commit_artifact(artifact, output_root)
    calls: list[int] = []

    def completion_evidence(
        task: Any,
        paths: Any,
        geometries: Any,
        **_: object,
    ) -> SimpleNamespace:
        calls.append(task.round)
        assert paths.affine == artifacts[0]
        assert geometries.fixed == _geometry()
        return SimpleNamespace(artifacts=artifacts[:-1], affine=None, warp=None)

    monkeypatch.setattr(
        workflow_status,
        "completion_evidence",
        completion_evidence,
        raising=False,
    )

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[1]

    assert calls == [2]
    assert round_status.status == "invalid"
    assert round_status.complete_artifacts == tuple(map(str, artifacts[:-1]))
    assert round_status.invalid_artifacts == (str(artifacts[-1]),)


def test_status_accepts_only_a_complete_serial_chunk_provenance_chain(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    scientific = _scientific_mapping(output_root)
    scientific_round = scientific["rounds"][0]  # type: ignore[index]
    scientific_round["strategy"] = "serial_chunked"
    scientific_round["chunk_size_zyx"] = [4, 8, 8]
    scientific_round["overlap_fraction"] = 0.0
    plan_path = WorkflowLayout.for_dataset(output_root, "dataset-1").plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(canonical_json(scientific), encoding="utf-8")
    artifacts = _commit_serial_round(output_root)

    complete = inspect_workflow_status(output_root, "dataset-1").rounds[1]

    assert complete.status == "complete"
    assert complete.complete_artifacts == tuple(map(str, artifacts))

    local_warp = (
        WorkflowLayout.for_dataset(output_root, "dataset-1").round(2).registration
        / "round_2_chunks"
        / "chunk_0000_warp.nii.gz"
    )
    ArtifactStore().record_path(local_warp).unlink()

    stale = inspect_workflow_status(output_root, "dataset-1").rounds[1]

    assert stale.status == "invalid"
    assert stale.complete_artifacts == (str(artifacts[0]),)
    assert stale.invalid_artifacts == tuple(map(str, artifacts[1:]))


def test_status_requires_the_complete_crop_provenance_chain(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    scientific = _scientific_mapping(output_root)
    scientific["registration"]["crop"] = True  # type: ignore[index]
    scientific["reference"]["crop_output"] = (  # type: ignore[index]
        "rounds/round-001/registration/reference_crop_indices.json"
    )
    plan_path = WorkflowLayout.for_dataset(output_root, "dataset-1").plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(canonical_json(scientific), encoding="utf-8")
    artifacts = _commit_cropped_round(output_root)

    complete = inspect_workflow_status(output_root, "dataset-1").rounds[1]

    assert complete.status == "complete"
    assert complete.complete_artifacts == tuple(map(str, artifacts))

    moving_crop = (
        WorkflowLayout.for_dataset(output_root, "dataset-1").round(2).registration
        / "moving_crop_indices_2.json"
    )
    moving_crop.write_text("[0,4,0,8,0,7]\n", encoding="utf-8")

    stale = inspect_workflow_status(output_root, "dataset-1").rounds[1]

    assert stale.status == "invalid"
    assert stale.complete_artifacts == ()
    assert stale.invalid_artifacts == tuple(map(str, artifacts))


def test_status_marks_a_corrupt_completion_record_invalid(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifact = _expected_artifacts(output_root, 2)[0]
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"artifact")
    record = ArtifactStore().record_path(artifact)
    record.write_text("{not-json", encoding="utf-8")

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[1]

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

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[1]

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

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[1]

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
    assert status.rounds[1].status == "complete"


def test_status_does_not_treat_metrics_or_run_summaries_as_completion(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    affine, warp, registered, channel = _expected_artifacts(output_root, 2)
    for artifact in (affine, warp, channel):
        _commit_artifact(artifact, output_root)
    layout = WorkflowLayout.for_dataset(output_root, "dataset-1")
    registration = layout.round(2).registration
    for metric_name in (
        "image_metrics_pre-registration.txt",
        "image_metrics_linearly-registered.txt",
        "image_metrics_nonlinear-registered.txt",
    ):
        (registration / metric_name).write_text("metric", encoding="utf-8")
    layout.workflow_runs.mkdir()
    (layout.workflow_runs / "claimed-complete.json").write_text(
        '{"status":"completed"}', encoding="utf-8"
    )

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[1]

    assert round_status.status == "partial"
    assert round_status.complete_artifacts == tuple(map(str, (affine, warp, channel)))
    assert round_status.incomplete_artifacts == (str(registered),)


def test_status_marks_identity_mismatch_invalid(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifact = _expected_artifacts(output_root, 2)[0]
    _commit_artifact(artifact, output_root)
    artifact.write_bytes(b"changed after commit")

    status = inspect_workflow_status(output_root, "dataset-1")

    assert status.rounds[1].status == "invalid"
    assert status.rounds[1].invalid_artifacts == (str(artifact),)


def test_status_rejects_identity_matching_artifact_with_unexpected_request(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    artifact = _expected_artifacts(output_root, 2)[0]
    _commit_artifact(artifact, output_root, stage_override="test")

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[1]

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

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[1]

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

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[1]

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

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[1]

    assert round_status.status == "invalid"
    assert round_status.invalid_artifacts == (str(artifact),)


def test_status_conservatively_rejects_crop_geometry_without_full_provenance(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    scientific = _scientific_mapping(output_root)
    scientific["registration"]["crop"] = True  # type: ignore[index]
    scientific["reference"]["crop_output"] = (  # type: ignore[index]
        "rounds/round-001/registration/reference_crop_indices.json"
    )
    plan_path = WorkflowLayout.for_dataset(output_root, "dataset-1").plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(canonical_json(scientific), encoding="utf-8")
    _commit_reference(output_root)
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

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[1]

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

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[1]

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

    round_status = inspect_workflow_status(output_root, "dataset-1").rounds[1]

    assert round_status.status == "invalid"
    assert round_status.invalid_artifacts == tuple(map(str, artifacts))


def test_status_serializes_as_json_compatible_values(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root, rounds=(3, 2))

    payload = inspect_workflow_status(output_root, "dataset-1").to_dict()

    assert [item["round"] for item in payload["rounds"]] == [1, 2, 3]  # type: ignore[index]
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


def test_complete_status_invocation_keeps_heavy_runtimes_unloaded(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_plan(output_root)
    for artifact in _expected_artifacts(output_root, 2):
        _commit_artifact(artifact, output_root)
    code = f"""
import sys
from pathlib import Path
from cyexm.workflow import inspect_workflow_status

status = inspect_workflow_status(Path({str(output_root)!r}), "dataset-1")
assert status.complete
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
