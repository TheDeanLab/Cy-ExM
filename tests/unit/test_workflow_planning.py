from __future__ import annotations

import inspect
import json
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
            *(channel.output for channel in task.channels),
        )
    )


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
