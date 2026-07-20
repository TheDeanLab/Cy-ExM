from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from cyexm.config.models import (
    ChannelSpec,
    DatasetManifest,
    ExecutionPaths,
    ExecutionProfile,
    NativeThreadSpec,
    ReferenceSpec,
    RegistrationSpec,
    RegistrationStageSpec,
    RoundSpec,
)
from cyexm.errors import WorkflowPlanError
from cyexm.spatial import ImageGeometry
from cyexm.workflow.layout import WorkflowLayout
from cyexm.workflow.models import ImageHeader
from cyexm.workflow.planning import _inspect_tiff_header, build_workflow_plan

pytestmark = pytest.mark.unit


def _registration() -> RegistrationSpec:
    return RegistrationSpec(
        crop=False,
        linear=RegistrationStageSpec(type="TRSAA", accuracy="high"),
        nonlinear=RegistrationStageSpec(type="SyNOnly", accuracy="high"),
    )


def _round(round_number: int) -> RoundSpec:
    if round_number == 2:
        return RoundSpec(
            round=2,
            strategy="whole_image",
            registration_channel=PurePosixPath("round_2_488.tif"),
            channels=(
                ChannelSpec(PurePosixPath("round_2_488.tif"), "linear"),
                ChannelSpec(PurePosixPath("round_2_mask.tif"), "nearestNeighbor"),
            ),
        )
    return RoundSpec(
        round=3,
        strategy="serial_chunked",
        registration_channel=PurePosixPath("round_3_488.tif"),
        channels=(
            ChannelSpec(PurePosixPath("round_3_mask.tif"), "nearestNeighbor"),
            ChannelSpec(PurePosixPath("round_3_488.tif"), "linear"),
        ),
        chunk_size_zyx=(2, 4, 4),
        overlap_fraction=0.25,
    )


def _manifest(round_order: tuple[int, ...] = (3, 2)) -> DatasetManifest:
    return DatasetManifest(
        schema_version=1,
        dataset_id="dataset-1",
        voxel_spacing_um=(0.324, 0.324, 0.9),
        reference=ReferenceSpec(
            round=1,
            channel=PurePosixPath("round_1_488.tif"),
        ),
        registration=_registration(),
        rounds=tuple(_round(round_number) for round_number in round_order),
    )


def _execution(
    tmp_path: Path,
    *,
    dataset_root: Path | None = None,
    output_root: Path | None = None,
    executor: str = "sequential",
    native_threads: NativeThreadSpec | None = None,
) -> ExecutionProfile:
    return ExecutionProfile(
        schema_version=1,
        executor=executor,  # type: ignore[arg-type]
        paths=ExecutionPaths(
            dataset_root=dataset_root or tmp_path / "dataset",
            output_root=output_root or tmp_path / "output",
            worker_local_directory=None,
        ),
        native_threads=native_threads or NativeThreadSpec(itk=1, blas=1),
        local_dask=None,
        slurm=None,
        source_path=tmp_path / "execution.yaml",
    )


def _create_manifest_inputs(root: Path, manifest: DatasetManifest) -> None:
    relative_paths = {manifest.reference.channel}
    relative_paths.update(
        channel.file
        for round_spec in manifest.rounds
        for channel in round_spec.channels
    )
    for relative_path in relative_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _header(
    path: Path,
    *,
    shape_zyx: tuple[int, ...] = (4, 8, 8),
    dtype: str = "uint16",
    axes: str = "ZYX",
    geometry: ImageGeometry | None = None,
) -> ImageHeader:
    return ImageHeader(
        path=path,
        shape_zyx=shape_zyx,  # type: ignore[arg-type]
        dtype=dtype,
        axes=axes,
        geometry=geometry,
        st_size=1024,
        st_mtime_ns=1,
    )


def test_plan_resolves_inputs_orders_rounds_and_creates_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(round_order=(3, 2))
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )

    plan = build_workflow_plan(manifest, execution)

    assert tuple(task.round for task in plan.tasks) == (2, 3)
    assert (
        plan.fixed_image == (execution.paths.dataset_root / "round_1_488.tif").resolve()
    )
    assert plan.tasks[0].fixed_image == plan.fixed_image
    assert (
        plan.tasks[0].moving_image
        == (execution.paths.dataset_root / "round_2_488.tif").resolve()
    )
    assert tuple(channel.source.name for channel in plan.tasks[1].channels) == (
        "round_3_mask.tif",
        "round_3_488.tif",
    )
    assert tuple(channel.output.name for channel in plan.tasks[1].channels) == (
        "round_3_mask.tif",
        "round_3_488.tif",
    )
    assert not execution.paths.output_root.exists()


def test_plan_revalidates_directly_constructed_manifest_before_inspection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(round_order=(2,))
    forged = replace(
        manifest,
        reference=replace(manifest.reference, round=2),
    )
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, forged)
    calls: list[Path] = []
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: calls.append(path) or _header(path),
    )

    with pytest.raises(WorkflowPlanError, match="reference round"):
        build_workflow_plan(forged, execution)

    assert calls == []
    assert not execution.paths.output_root.exists()


def test_plan_revalidates_directly_constructed_execution_before_inspection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(round_order=(2,))
    execution = _execution(
        tmp_path,
        native_threads=NativeThreadSpec(itk=True, blas=1),  # type: ignore[arg-type]
    )
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    calls: list[Path] = []
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: calls.append(path) or _header(path),
    )

    with pytest.raises(WorkflowPlanError, match="Execution profile is invalid"):
        build_workflow_plan(manifest, execution)

    assert calls == []
    assert not execution.paths.output_root.exists()


def test_plan_aggregates_missing_inputs_in_manifest_order_without_inspection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(round_order=(2,))
    execution = _execution(tmp_path)
    execution.paths.dataset_root.mkdir()
    existing = execution.paths.dataset_root / "round_2_mask.tif"
    existing.touch()
    calls: list[Path] = []
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: calls.append(path) or _header(path),
    )

    with pytest.raises(WorkflowPlanError) as captured:
        build_workflow_plan(manifest, execution)

    message = str(captured.value)
    assert "reference.channel" in message
    assert "rounds[0].channels[0].file" in message
    assert str(execution.paths.dataset_root / "round_1_488.tif") in message
    assert str(execution.paths.dataset_root / "round_2_488.tif") in message
    assert message.index("reference.channel") < message.index("rounds[0].channels[0]")
    assert calls == []
    assert not execution.paths.output_root.exists()


def test_plan_rejects_a_symlink_that_escapes_the_dataset_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(round_order=(2,))
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    outside = tmp_path / "outside.tif"
    outside.touch()
    reference = execution.paths.dataset_root / manifest.reference.channel
    reference.unlink()
    reference.symlink_to(outside)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )

    with pytest.raises(WorkflowPlanError, match="outside the dataset root") as captured:
        build_workflow_plan(manifest, execution)

    assert "reference.channel" in str(captured.value)
    assert str(outside) in str(captured.value)


def test_plan_inspects_each_unique_resolved_file_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shared = PurePosixPath("shared.tif")
    manifest = DatasetManifest(
        schema_version=1,
        dataset_id="dataset-1",
        voxel_spacing_um=(1.0, 1.0, 1.0),
        reference=ReferenceSpec(1, PurePosixPath("fixed.tif")),
        registration=_registration(),
        rounds=(
            RoundSpec(2, "whole_image", shared, (ChannelSpec(shared, "linear"),)),
            RoundSpec(3, "whole_image", shared, (ChannelSpec(shared, "linear"),)),
        ),
    )
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    calls: list[Path] = []

    def inspect_header(path: Path) -> ImageHeader:
        calls.append(path)
        return _header(path)

    monkeypatch.setattr("cyexm.workflow.planning._inspect_tiff_header", inspect_header)

    build_workflow_plan(manifest, execution)

    assert calls == [
        (execution.paths.dataset_root / "fixed.tif").resolve(),
        (execution.paths.dataset_root / "shared.tif").resolve(),
    ]


def test_plan_rejects_channel_output_name_collisions_after_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = PurePosixPath("nested-a/same.tif")
    second = PurePosixPath("nested-b/same.tif")
    manifest = DatasetManifest(
        schema_version=1,
        dataset_id="dataset-1",
        voxel_spacing_um=(1.0, 1.0, 1.0),
        reference=ReferenceSpec(1, PurePosixPath("fixed.tif")),
        registration=_registration(),
        rounds=(
            RoundSpec(
                2,
                "whole_image",
                first,
                (ChannelSpec(first, "linear"), ChannelSpec(second, "linear")),
            ),
        ),
    )
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )

    with pytest.raises(WorkflowPlanError, match="channel output collision") as captured:
        build_workflow_plan(manifest, execution)

    assert "rounds[0].channels[0].file" in str(captured.value)
    assert "rounds[0].channels[1].file" in str(captured.value)
    assert "same.tif" in str(captured.value)


def test_plan_rejects_roots_that_contain_one_another_after_strict_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    dataset_root = output_root / "dataset"
    manifest = _manifest(round_order=(2,))
    execution = _execution(
        tmp_path,
        dataset_root=dataset_root,
        output_root=output_root,
    )
    _create_manifest_inputs(dataset_root, manifest)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )

    with pytest.raises(
        WorkflowPlanError, match="dataset and output roots must be disjoint"
    ):
        build_workflow_plan(manifest, execution)


def test_plan_rejects_an_existing_output_root_that_is_not_a_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(round_order=(2,))
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    execution.paths.output_root.write_text("do not replace", encoding="utf-8")
    before = execution.paths.output_root.read_bytes()
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )

    with pytest.raises(
        WorkflowPlanError,
        match=r"paths\.output_root must resolve to a directory",
    ):
        build_workflow_plan(manifest, execution)

    assert execution.paths.output_root.read_bytes() == before


def test_plan_rejects_existing_output_symlinks_that_escape_the_output_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(round_order=(2,))
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    execution.paths.output_root.mkdir()
    outside = tmp_path / "outside-output"
    outside.mkdir()
    (execution.paths.output_root / manifest.dataset_id).symlink_to(outside)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )

    with pytest.raises(WorkflowPlanError, match="outside the output root") as captured:
        build_workflow_plan(manifest, execution)

    assert str(outside) in str(captured.value)


@pytest.mark.parametrize("strategy", ["whole_image", "serial_chunked"])
@pytest.mark.parametrize(
    ("registration_filename", "role_fragment"),
    [
        ("linear_transform_2.mat", "registration affine"),
        ("nonlinear_warp_2.nii.gz", "registration warp"),
        ("nonlinear_registered_2.tif", "registration diagnostic"),
    ],
)
def test_plan_rejects_channel_aliases_to_registration_final_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    strategy: str,
    registration_filename: str,
    role_fragment: str,
) -> None:
    channel_file = PurePosixPath(f"nested/{registration_filename}")
    manifest = DatasetManifest(
        schema_version=1,
        dataset_id="dataset-1",
        voxel_spacing_um=(1.0, 1.0, 1.0),
        reference=ReferenceSpec(1, PurePosixPath("fixed.tif")),
        registration=_registration(),
        rounds=(
            RoundSpec(
                round=2,
                strategy=strategy,  # type: ignore[arg-type]
                registration_channel=channel_file,
                channels=(ChannelSpec(channel_file, "linear"),),
                chunk_size_zyx=(2, 4, 4) if strategy == "serial_chunked" else None,
                overlap_fraction=0.25 if strategy == "serial_chunked" else None,
            ),
        ),
    )
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    round_root = (
        execution.paths.output_root / manifest.dataset_id / "rounds" / "round-002"
    )
    registration = round_root / "registration"
    registration.mkdir(parents=True)
    (round_root / "channels").symlink_to(registration, target_is_directory=True)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )

    with pytest.raises(WorkflowPlanError, match="final artifact collision") as captured:
        build_workflow_plan(manifest, execution)

    message = str(captured.value)
    assert role_fragment in message
    assert "channels[0]" in message
    assert str(registration / registration_filename) in message


def test_plan_aggregates_header_failures_and_shape_incompatibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(round_order=(2,))
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)

    def inspect_header(path: Path) -> ImageHeader:
        if path.name == "round_2_488.tif":
            return _header(path, shape_zyx=(4, 8, 9))
        if path.name == "round_2_mask.tif":
            return _header(path, axes="ZCYX")
        return _header(path)

    monkeypatch.setattr("cyexm.workflow.planning._inspect_tiff_header", inspect_header)

    with pytest.raises(WorkflowPlanError) as captured:
        build_workflow_plan(manifest, execution)

    message = str(captured.value)
    assert "round_2_488.tif" in message
    assert "shape" in message
    assert "round_2_mask.tif" in message
    assert "scalar three-dimensional" in message


def test_plan_rejects_header_geometry_spacing_incompatible_with_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(round_order=(2,))
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    incompatible = ImageGeometry(
        shape=(4, 8, 8),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path, geometry=incompatible),
    )

    with pytest.raises(WorkflowPlanError, match="embedded spacing"):
        build_workflow_plan(manifest, execution)


def test_scientific_fingerprint_contains_complete_policy_but_not_execution_choices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    execution = _execution(
        tmp_path,
        executor="sequential",
        native_threads=NativeThreadSpec(itk=7, blas=9),
    )
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )

    plan = build_workflow_plan(manifest, execution)
    scientific = json.loads(plan.scientific_json)

    assert scientific["reference"] == {
        "round": 1,
        "channel": str(plan.fixed_image),
    }
    assert [item["round"] for item in scientific["rounds"]] == [2, 3]
    assert scientific["rounds"][0]["registration_channel"].endswith("round_2_488.tif")
    assert [channel["file"] for channel in scientific["rounds"][1]["channels"]] == [
        str((execution.paths.dataset_root / "round_3_mask.tif").resolve()),
        str((execution.paths.dataset_root / "round_3_488.tif").resolve()),
    ]
    assert scientific["policies"] == {
        "axis_convention": "external-zyx-ants-xyz-v1",
        "output_format": "ome-compatible-tiff-uint16-v1",
        "postprocessing": "clip-uint16-no-normalization-v1",
    }
    assert "executor" not in plan.scientific_json
    assert "native_threads" not in plan.scientific_json
    assert "output_root" not in plan.scientific_json
    assert plan.expected_artifacts == tuple(
        artifact
        for task in plan.tasks
        for artifact in (
            task.registration_directory / f"linear_transform_{task.round}.mat",
            task.registration_directory / f"nonlinear_warp_{task.round}.nii.gz",
            task.registration_directory / f"nonlinear_registered_{task.round}.tif",
            *(channel.output for channel in task.channels),
        )
    )


def test_manifest_alias_and_relative_channel_output_are_scientific_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    fixed = dataset_root / "fixed.tif"
    shared = dataset_root / "shared.tif"
    fixed.touch()
    shared.touch()
    (dataset_root / "alias-a.tif").symlink_to(shared)
    (dataset_root / "alias-b.tif").symlink_to(shared)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )

    def manifest(alias: str) -> DatasetManifest:
        relative = PurePosixPath(alias)
        return DatasetManifest(
            schema_version=1,
            dataset_id="dataset-1",
            voxel_spacing_um=(1.0, 1.0, 1.0),
            reference=ReferenceSpec(1, PurePosixPath("fixed.tif")),
            registration=_registration(),
            rounds=(
                RoundSpec(
                    2,
                    "whole_image",
                    relative,
                    (ChannelSpec(relative, "linear"),),
                ),
            ),
        )

    plan_a = build_workflow_plan(
        manifest("alias-a.tif"),
        _execution(
            tmp_path,
            dataset_root=dataset_root,
            output_root=tmp_path / "output-a",
        ),
    )
    plan_b = build_workflow_plan(
        manifest("alias-b.tif"),
        _execution(
            tmp_path,
            dataset_root=dataset_root,
            output_root=tmp_path / "output-b",
        ),
    )

    channel_a = json.loads(plan_a.scientific_json)["rounds"][0]["channels"][0]
    channel_b = json.loads(plan_b.scientific_json)["rounds"][0]["channels"][0]
    assert channel_a["file"] == channel_b["file"] == str(shared)
    assert channel_a["manifest_file"] == "alias-a.tif"
    assert channel_b["manifest_file"] == "alias-b.tif"
    assert channel_a["output"] == "rounds/round-002/channels/alias-a.tif"
    assert channel_b["output"] == "rounds/round-002/channels/alias-b.tif"
    assert plan_a.tasks[0].channels[0].output.name == "alias-a.tif"
    assert plan_b.tasks[0].channels[0].output.name == "alias-b.tif"
    assert plan_a.scientific_fingerprint != plan_b.scientific_fingerprint


def test_existing_exact_plan_is_accepted_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )
    first = build_workflow_plan(manifest, execution)
    plan_path = WorkflowLayout.for_dataset(
        execution.paths.output_root, manifest.dataset_id
    ).plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("\n  " + first.scientific_json + "\n", encoding="utf-8")
    before = (plan_path.read_bytes(), plan_path.stat().st_mtime_ns)

    second = build_workflow_plan(manifest, execution)

    assert second.scientific_fingerprint == first.scientific_fingerprint
    assert (plan_path.read_bytes(), plan_path.stat().st_mtime_ns) == before


def test_existing_plan_mismatch_reports_deterministic_dotted_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )
    plan = build_workflow_plan(manifest, execution)
    existing = json.loads(plan.scientific_json)
    existing["registration"]["linear"]["accuracy"] = "low"
    existing["rounds"][0]["strategy"] = "serial_chunked"
    plan_path = WorkflowLayout.for_dataset(
        execution.paths.output_root, manifest.dataset_id
    ).plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(existing), encoding="utf-8")

    with pytest.raises(WorkflowPlanError) as captured:
        build_workflow_plan(manifest, execution)

    message = str(captured.value)
    assert "registration.linear.accuracy" in message
    assert "rounds[0].strategy" in message
    assert message.index("registration.linear.accuracy") < message.index(
        "rounds[0].strategy"
    )
    assert "select a new output root" in message


def test_existing_plan_requires_the_exact_snapshot_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(round_order=(2,))
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )
    plan = build_workflow_plan(manifest, execution)
    existing = json.loads(plan.scientific_json)
    existing["unexpected"] = True
    plan_path = WorkflowLayout.for_dataset(
        execution.paths.output_root, manifest.dataset_id
    ).plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(existing), encoding="utf-8")

    with pytest.raises(WorkflowPlanError, match="exact schema") as captured:
        build_workflow_plan(manifest, execution)

    assert "unexpected" in str(captured.value)
    assert "select a new output root" in str(captured.value)


def test_existing_plan_rejects_overflowing_json_floats_as_workflow_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(round_order=(2,))
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )
    plan = build_workflow_plan(manifest, execution)
    plan_path = WorkflowLayout.for_dataset(
        execution.paths.output_root, manifest.dataset_id
    ).plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        plan.scientific_json.replace("0.324", "1e400", 1),
        encoding="utf-8",
    )

    with pytest.raises(WorkflowPlanError, match="finite JSON number"):
        build_workflow_plan(manifest, execution)


def test_existing_plan_signed_zero_mismatch_has_a_dotted_value_difference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = _round(3)
    manifest = DatasetManifest(
        schema_version=1,
        dataset_id="dataset-1",
        voxel_spacing_um=(0.324, 0.324, 0.9),
        reference=ReferenceSpec(1, PurePosixPath("round_1_488.tif")),
        registration=_registration(),
        rounds=(
            RoundSpec(
                round=base.round,
                strategy=base.strategy,
                registration_channel=base.registration_channel,
                channels=base.channels,
                chunk_size_zyx=base.chunk_size_zyx,
                overlap_fraction=0.0,
            ),
        ),
    )
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )
    plan = build_workflow_plan(manifest, execution)
    existing = json.loads(plan.scientific_json)
    existing["rounds"][0]["overlap_fraction"] = -0.0
    plan_path = WorkflowLayout.for_dataset(
        execution.paths.output_root, manifest.dataset_id
    ).plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(existing), encoding="utf-8")

    with pytest.raises(WorkflowPlanError) as captured:
        build_workflow_plan(manifest, execution)

    message = str(captured.value)
    assert "rounds[0].overlap_fraction" in message
    assert "existing=-0.0" in message
    assert "planned=0.0" in message


def test_existing_plan_list_cardinality_is_a_dotted_scientific_difference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(round_order=(2,))
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )
    plan = build_workflow_plan(manifest, execution)
    existing = json.loads(plan.scientific_json)
    existing["rounds"][0]["channels"].pop()
    plan_path = WorkflowLayout.for_dataset(
        execution.paths.output_root, manifest.dataset_id
    ).plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(existing), encoding="utf-8")

    with pytest.raises(WorkflowPlanError) as captured:
        build_workflow_plan(manifest, execution)

    message = str(captured.value)
    assert "fingerprint mismatch" in message
    assert "rounds[0].channels[1]" in message
    assert "planned=" in message


def test_existing_plan_validates_schema_of_added_list_elements(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(round_order=(2,))
    execution = _execution(tmp_path)
    _create_manifest_inputs(execution.paths.dataset_root, manifest)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: _header(path),
    )
    plan = build_workflow_plan(manifest, execution)
    existing = json.loads(plan.scientific_json)
    existing["rounds"][0]["channels"].append({"file": "incomplete"})
    plan_path = WorkflowLayout.for_dataset(
        execution.paths.output_root, manifest.dataset_id
    ).plan_path
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(existing), encoding="utf-8")

    with pytest.raises(WorkflowPlanError, match="exact schema") as captured:
        build_workflow_plan(manifest, execution)

    message = str(captured.value)
    assert "rounds[0].channels[2].interpolation" in message
    assert "rounds[0].channels[2].manifest_file" in message
    assert "rounds[0].channels[2].output" in message


def test_planner_public_signature_has_no_force_escape_hatch() -> None:
    assert tuple(inspect.signature(build_workflow_plan).parameters) == (
        "manifest",
        "execution",
    )
    with pytest.raises(TypeError, match="force"):
        build_workflow_plan(None, None, force=True)  # type: ignore[call-arg]


def test_private_header_inspection_reads_strict_scalar_tiff_metadata(
    tmp_path: Path,
) -> None:
    import numpy as np
    import tifffile

    path = tmp_path / "image.tif"
    tifffile.imwrite(
        path,
        np.zeros((2, 3, 5), dtype=np.uint16),
        metadata={"axes": "ZYX"},
    )

    header = _inspect_tiff_header(path)

    assert header.path == path
    assert header.shape_zyx == (2, 3, 5)
    assert header.dtype == "uint16"
    assert header.axes == "ZYX"
    assert header.geometry is None
    assert header.st_size == path.stat().st_size
    assert header.st_mtime_ns == path.stat().st_mtime_ns


@pytest.mark.parametrize(
    ("axes", "shape", "expected_axes"),
    [
        ("QYX", (2, 3, 4), "QYX"),
        ("ZYX", (1, 3, 4), "YX"),
    ],
)
def test_private_header_inspection_preserves_phase_one_scalar_layouts(
    tmp_path: Path,
    axes: str,
    shape: tuple[int, int, int],
    expected_axes: str,
) -> None:
    import numpy as np
    import tifffile

    path = tmp_path / f"{axes.lower()}-{shape[0]}.tif"
    tifffile.imwrite(
        path,
        np.zeros(shape, dtype=np.uint16),
        metadata={"axes": axes},
        photometric="minisblack",
    )

    header = _inspect_tiff_header(path)

    assert header.shape_zyx == shape
    assert header.axes == expected_axes


def test_private_header_inspection_rejects_a_file_changed_between_metadata_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import numpy as np
    import tifffile

    from cyexm.io import tiff as tiff_module

    path = tmp_path / "changing.tif"
    tifffile.imwrite(
        path,
        np.zeros((2, 3, 5), dtype=np.uint16),
        metadata={"axes": "ZYX"},
        photometric="minisblack",
    )

    def mutate_during_strict_validation(
        source: str | Path,
        _shape: tuple[int, ...],
    ) -> None:
        with Path(source).open("ab") as stream:
            stream.write(b"changed")

    monkeypatch.setattr(
        tiff_module,
        "read_tiff_geometry",
        mutate_during_strict_validation,
    )

    with pytest.raises(
        WorkflowPlanError, match="changed during TIFF header inspection"
    ):
        _inspect_tiff_header(path)


@pytest.mark.parametrize("layout", ["two_dimensional", "rgb"])
def test_private_header_inspection_rejects_non_scalar_or_non_3d_tiffs(
    tmp_path: Path,
    layout: str,
) -> None:
    import numpy as np
    import tifffile

    path = tmp_path / f"{layout}.tif"
    if layout == "two_dimensional":
        tifffile.imwrite(path, np.zeros((3, 4), dtype=np.uint16))
    else:
        tifffile.imwrite(
            path,
            np.zeros((2, 3, 4, 3), dtype=np.uint8),
            photometric="rgb",
            metadata={"axes": "ZYXS"},
        )

    with pytest.raises(WorkflowPlanError, match="scalar three-dimensional"):
        _inspect_tiff_header(path)
