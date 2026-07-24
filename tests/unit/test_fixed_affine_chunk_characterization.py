from __future__ import annotations

import json
from pathlib import Path

import pytest
from hpc.fixed_affine_chunk_characterization import (
    APPROVED_CHUNK_SIZES,
    FixedAffineCharacterizationError,
    ReadOnlyReferenceRunner,
    RoiYX,
    build_condition,
    clone_accepted_artifact,
    load_characterization_spec,
    validate_fixed_affine_source,
)

from cyexm.config.models import (
    ChannelSpec,
    DatasetManifest,
    ExecutionPaths,
    ExecutionProfile,
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    ParallelismSpec,
    ReferenceChannelSpec,
    ReferenceSpec,
    RegistrationSpec,
    RegistrationStageSpec,
    RoundSpec,
    SlurmSpec,
)
from cyexm.registration.artifacts import (
    ArtifactStore,
    FileIdentity,
    StageRequest,
)
from cyexm.spatial import ImageGeometry
from cyexm.workflow.models import ReferenceChannelTask, ReferenceTask

pytestmark = pytest.mark.unit


def _write_spec(tmp_path: Path, *, replacement: str = "") -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "scientific-plan.json").write_text("{}\n", encoding="utf-8")
    spec = tmp_path / "characterization.yaml"
    spec.write_text(
        f"""\
schema_version: 1
source_workflow_root: {source}
condition_order: ["1125", "512", "256", "128"]
chunk_sizes_zyx:
  "1125": [512, 1125, 1125]
  "512": [512, 512, 512]
  "256": [512, 256, 256]
  "128": [512, 128, 128]
overlap_fraction: 0.15
rounds: [2, 3, 4]
dim_roi_yx: [1600, 1800, 300, 400]
bright_roi_yx: [1227, 1427, 840, 1040]
search_radius_px: 30
{replacement}""",
        encoding="utf-8",
    )
    return spec


def test_characterization_spec_normalizes_the_approved_experiment(
    tmp_path: Path,
) -> None:
    spec = load_characterization_spec(_write_spec(tmp_path))

    assert spec.schema_version == 1
    assert spec.source_workflow_root == (tmp_path / "source").resolve()
    assert spec.condition_order == ("1125", "512", "256", "128")
    assert spec.chunk_sizes_zyx == APPROVED_CHUNK_SIZES
    assert spec.overlap_fraction == 0.15
    assert spec.rounds == (2, 3, 4)
    assert spec.dim_roi_yx == RoiYX(1600, 1800, 300, 400)
    assert spec.bright_roi_yx == RoiYX(1227, 1427, 840, 1040)
    assert spec.search_radius_px == 30


def test_characterization_spec_rejects_unknown_keys(tmp_path: Path) -> None:
    path = _write_spec(tmp_path, replacement="unexpected: true\n")

    with pytest.raises(FixedAffineCharacterizationError, match="keys must be exactly"):
        load_characterization_spec(path)


def test_characterization_spec_rejects_unapproved_chunk_size(
    tmp_path: Path,
) -> None:
    path = _write_spec(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"128": [512, 128, 128]',
            '"128": [512, 64, 64]',
        ),
        encoding="utf-8",
    )

    with pytest.raises(FixedAffineCharacterizationError, match="approved chunk sizes"):
        load_characterization_spec(path)


def test_characterization_spec_rejects_a_symlinked_source_root(
    tmp_path: Path,
) -> None:
    path = _write_spec(tmp_path)
    source_link = tmp_path / "source-link"
    source_link.symlink_to(tmp_path / "source", target_is_directory=True)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            str(tmp_path / "source"),
            str(source_link),
        ),
        encoding="utf-8",
    )

    with pytest.raises(FixedAffineCharacterizationError, match="must not be a symlink"):
        load_characterization_spec(path)


_THREADS = {
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": 30,
    "MKL_NUM_THREADS": 30,
    "NUMEXPR_NUM_THREADS": 30,
    "OMP_NUM_THREADS": 30,
    "OPENBLAS_NUM_THREADS": 30,
    "VECLIB_MAXIMUM_THREADS": 30,
}
_GEOMETRY = ImageGeometry(
    shape=(5, 20, 21),
    origin=(0.0, 0.0, 0.0),
    spacing=(0.324, 0.324, 0.9),
    direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
)


def _commit_bytes(path: Path, request: StageRequest, value: bytes) -> None:
    ArtifactStore().commit(
        path,
        request,
        writer=lambda candidate: candidate.write_bytes(value),
        validator=lambda candidate: candidate.read_bytes() == value,
    )


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "source-workflow"
    dataset = tmp_path / "dataset"
    root.mkdir()
    dataset.mkdir()
    fixed = root / "rounds/round-001/channels/fixed.tif"
    fixed.parent.mkdir(parents=True)
    moving_paths: dict[int, Path] = {}
    fixed_request = StageRequest.for_files(
        stage="reference-channel",
        inputs=(),
        geometry=_GEOMETRY,
        method="copy",
        accuracy="exact",
        native_threads=_THREADS,
        upstream=None,
        parameters={},
    )
    _commit_bytes(fixed, fixed_request, b"fixed")
    rounds: list[dict[str, object]] = []
    for round_number in (2, 3, 4):
        moving = dataset / f"moving-{round_number}.tif"
        moving.write_bytes(f"moving-{round_number}".encode())
        moving_paths[round_number] = moving
        registration = root / f"rounds/round-{round_number:03d}/registration"
        registration.mkdir(parents=True)
        inputs = (FileIdentity.from_path(fixed), FileIdentity.from_path(moving))
        common = {
            "crop": True,
            "imaging_round": round_number,
            "pixel_size_xyz": [0.324, 0.324, 0.9],
            "prepared_fixed": True,
        }
        crop_request = StageRequest.for_identities(
            stage="crop:moving",
            inputs=inputs,
            geometry=_GEOMETRY,
            method="robust-bounding-box",
            accuracy="legacy-robust",
            native_threads=_THREADS,
            upstream=None,
            parameters=common,
        )
        crop = registration / f"moving_crop_indices_{round_number}.json"
        _commit_bytes(crop, crop_request, b"[[0, 5], [0, 20], [0, 21]]\n")
        pre_request = StageRequest.for_identities(
            stage="metrics:pre-registration",
            inputs=inputs,
            geometry=_GEOMETRY,
            method="Correlation+MattesMutualInformation",
            accuracy="full",
            native_threads=_THREADS,
            upstream=None,
            parameters=common,
        )
        pre = registration / "image_metrics_pre-registration.txt"
        _commit_bytes(pre, pre_request, b"Correlation: 0.1\n")
        affine_request = StageRequest.for_identities(
            stage="affine",
            inputs=inputs,
            geometry=_GEOMETRY,
            method="TRSAA",
            accuracy="high",
            native_threads=_THREADS,
            upstream=None,
            parameters={
                **common,
                "registration_parameters": {"aff_metric": "mattes"},
            },
        )
        affine = registration / f"linear_transform_{round_number}.mat"
        _commit_bytes(affine, affine_request, b"affine")
        linear_request = StageRequest.for_identities(
            stage="metrics:linearly-registered",
            inputs=inputs,
            geometry=_GEOMETRY,
            method="Correlation+MattesMutualInformation",
            accuracy="full",
            native_threads=_THREADS,
            upstream=FileIdentity.from_path(affine),
            parameters=common,
        )
        linear = registration / "image_metrics_linearly-registered.txt"
        _commit_bytes(linear, linear_request, b"Correlation: 0.2\n")
        rounds.append(
            {
                "round": round_number,
                "registration_channel": str(moving.resolve()),
                "chunk_size_zyx": [512, 1125, 1125],
                "overlap_fraction": 0.15,
                "strategy": "serial_chunked",
                "channels": [],
            }
        )
    plan = {
        "schema_version": 1,
        "dataset_id": "fixed-affine-source",
        "voxel_spacing_um": {"x": 0.324, "y": 0.324, "z": 0.9},
        "reference": {
            "round": 1,
            "registration_channel": str(fixed.resolve()),
            "registration_output": "rounds/round-001/channels/fixed.tif",
            "channels": [],
        },
        "registration": {
            "crop": True,
            "linear": {"type": "TRSAA", "accuracy": "high"},
            "nonlinear": {
                "type": "SyNOnly",
                "accuracy": "high",
                "metric": "CC",
                "radius": 4,
            },
        },
        "rounds": rounds,
    }
    (root / "scientific-plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return root, fixed


def test_source_validation_accepts_exact_radius_four_affine_state(
    tmp_path: Path,
) -> None:
    root, fixed = _source_fixture(tmp_path)

    source = validate_fixed_affine_source(root)

    assert source.fixed_reference == fixed.resolve()
    assert tuple(item.round for item in source.rounds) == (2, 3, 4)
    assert all(item.fixed_geometry == _GEOMETRY for item in source.rounds)
    assert all(item.affine_record.request.method == "TRSAA" for item in source.rounds)
    assert all(item.affine_record.request.accuracy == "high" for item in source.rounds)


def test_source_validation_rejects_changed_affine_identity(tmp_path: Path) -> None:
    root, _fixed = _source_fixture(tmp_path)
    affine = root / "rounds/round-002/registration/linear_transform_2.mat"
    affine.write_bytes(b"changed")

    with pytest.raises(FixedAffineCharacterizationError, match="identity mismatch"):
        validate_fixed_affine_source(root)


def test_source_validation_rejects_non_radius_four_plan(tmp_path: Path) -> None:
    root, _fixed = _source_fixture(tmp_path)
    plan_path = root / "scientific-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["registration"]["nonlinear"]["radius"] = 6
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(FixedAffineCharacterizationError, match="CC radius 4"):
        validate_fixed_affine_source(root)


def test_accepted_artifact_clone_preserves_bytes_and_rebinds_upstream(
    tmp_path: Path,
) -> None:
    root, _fixed = _source_fixture(tmp_path)
    source = validate_fixed_affine_source(root).rounds[0]
    destination_root = tmp_path / "destination"
    destination_root.mkdir()
    destination_affine = destination_root / source.affine.name
    cloned_affine = clone_accepted_artifact(
        source.affine,
        source.affine_record,
        destination_affine,
    )
    destination_linear = destination_root / source.linear_metrics.name
    cloned_linear = clone_accepted_artifact(
        source.linear_metrics,
        source.linear_metrics_record,
        destination_linear,
        upstream=cloned_affine.artifact,
    )

    assert destination_affine.read_bytes() == source.affine.read_bytes()
    assert destination_linear.read_bytes() == source.linear_metrics.read_bytes()
    assert cloned_affine.request == source.affine_record.request
    assert cloned_linear.request.upstream == cloned_affine.artifact
    assert cloned_linear.request.to_dict() | {"upstream": None} == (
        source.linear_metrics_record.request.to_dict() | {"upstream": None}
    )


def test_accepted_artifact_clone_rejects_changed_source(tmp_path: Path) -> None:
    root, _fixed = _source_fixture(tmp_path)
    source = validate_fixed_affine_source(root).rounds[0]
    source.affine.write_bytes(b"changed")

    with pytest.raises(FixedAffineCharacterizationError, match="identity mismatch"):
        clone_accepted_artifact(
            source.affine,
            source.affine_record,
            tmp_path / "destination.mat",
        )


def test_condition_builder_changes_only_chunk_geometry_and_owned_paths(
    tmp_path: Path,
) -> None:
    root, _fixed = _source_fixture(tmp_path)
    source = validate_fixed_affine_source(root)
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    scratch_root = tmp_path / "scratch"
    output_root.mkdir()
    scratch_root.mkdir()
    manifest = DatasetManifest(
        schema_version=1,
        dataset_id="source",
        voxel_spacing_um=(0.324, 0.324, 0.9),
        reference=ReferenceSpec(
            round=1,
            registration_channel=Path("fixed.tif"),  # type: ignore[arg-type]
            channels=(
                ReferenceChannelSpec(file=Path("fixed.tif")),  # type: ignore[arg-type]
            ),
        ),
        registration=RegistrationSpec(
            crop=True,
            linear=RegistrationStageSpec(type="TRSAA", accuracy="high"),
            nonlinear=NonlinearRegistrationStageSpec(
                type="SyNOnly",
                accuracy="high",
                metric="CC",
                radius=4,
            ),
        ),
        rounds=tuple(
            RoundSpec(
                round=round_number,
                strategy="serial_chunked",
                registration_channel=Path(  # type: ignore[arg-type]
                    f"moving-{round_number}.tif"
                ),
                channels=(
                    ChannelSpec(  # type: ignore[arg-type]
                        file=Path(f"moving-{round_number}.tif"),
                        interpolation="linear",
                    ),
                ),
                chunk_size_zyx=(512, 1125, 1125),
                overlap_fraction=0.15,
            )
            for round_number in (2, 3, 4)
        ),
    )
    profile = ExecutionProfile(
        schema_version=1,
        executor="slurm",
        paths=ExecutionPaths(
            dataset_root=dataset_root,
            output_root=output_root,
            worker_local_directory=scratch_root,
        ),
        native_threads=NativeThreadSpec(itk=30, blas=30),
        local_dask=None,
        slurm=SlurmSpec(
            queue="512GB",
            workers=4,
            cores_per_worker=30,
            processes_per_worker=1,
            memory="500GB",
            walltime="24:00:00",
            interface="ib0",
            scheduler_interface="ib0",
            startup_timeout="15m",
            task_retries=1,
            job_name="fixed-affine",
            job_extra_directives=("--exclusive",),
        ),
        source_path=tmp_path / "execution.yaml",
        parallelism=ParallelismSpec(mode="fine_grained"),
    )

    condition = build_condition(
        label="256",
        chunk_size_zyx=(512, 256, 256),
        manifest=manifest,
        profile=profile,
        source=source,
    )

    assert condition.profile.paths.output_root == output_root / "chunk-256"
    assert condition.profile.paths.worker_local_directory == scratch_root / "chunk-256"
    assert condition.profile.slurm is not None
    assert condition.profile.slurm.job_name == "fixed-affine-c256"
    assert (
        condition.reference_task.registration_channel_output == source.fixed_reference
    )
    assert tuple(task.round for task in condition.round_tasks) == (2, 3, 4)
    assert all(
        task.fixed_image == source.fixed_reference for task in condition.round_tasks
    )
    assert all(task.chunk_size_zyx == (512, 256, 256) for task in condition.round_tasks)
    assert all(task.overlap_fraction == 0.15 for task in condition.round_tasks)
    assert all(len(task.channels) == 1 for task in condition.round_tasks)


def test_read_only_reference_runner_reports_reuse_without_touching_source(
    tmp_path: Path,
) -> None:
    root, fixed = _source_fixture(tmp_path)
    source = validate_fixed_affine_source(root)
    record_path = ArtifactStore().record_path(fixed)
    before = (FileIdentity.from_path(fixed), record_path.read_bytes())
    task = ReferenceTask(
        dataset_id="condition",
        round=1,
        registration_channel_source=fixed,
        registration_channel_output=fixed,
        registration_directory=tmp_path / "condition/reference-registration",
        crop_bounds_path=None,
        log_path=tmp_path / "condition/reference.log",
        voxel_spacing_xyz=(0.324, 0.324, 0.9),
        crop=False,
        native_threads=NativeThreadSpec(itk=30, blas=30),
        channels=(ReferenceChannelTask(source=fixed, output=fixed),),
    )

    result = ReadOnlyReferenceRunner(source)(task)

    assert result.status == "reused"
    assert result.artifact_paths == (str(fixed),)
    assert result.reused_artifacts == (str(fixed),)
    assert (FileIdentity.from_path(fixed), record_path.read_bytes()) == before
