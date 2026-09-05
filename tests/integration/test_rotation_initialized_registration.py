from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
import tifffile
from scipy import ndimage

from cyexm.config.models import (
    ChannelSpec,
    DatasetManifest,
    EstimatedXYRotationSpec,
    EstimatedXYSimilaritySpec,
    ExecutionPaths,
    ExecutionProfile,
    ExplicitXYRotationSpec,
    InitialAlignmentSpec,
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    ParallelismSpec,
    ReferenceChannelSpec,
    ReferenceSpec,
    RegistrationSpec,
    RegistrationStageSpec,
    RoundSpec,
)
from cyexm.registration._request_policy import canonical_compact_json
from cyexm.registration.artifacts import ArtifactStore, CompletionRecord
from cyexm.registration.prepared import PreparedRoundPaths
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.workflow import (
    build_workflow_plan,
    inspect_workflow_status,
    run_workflow,
)

pytestmark = pytest.mark.integration


def _images(root: Path) -> None:
    root.mkdir()
    z, y, x = np.mgrid[:8, :48, :48]
    fixed = (
        50_000 * np.exp(-((z - 3) ** 2 / 3 + (y - 14) ** 2 / 30 + (x - 32) ** 2 / 12))
        + 30_000 * np.exp(-((z - 5) ** 2 / 2 + (y - 35) ** 2 / 15 + (x - 11) ** 2 / 25))
    ).astype(np.uint16)
    moving = ndimage.rotate(
        fixed,
        15.0,
        axes=(1, 2),
        reshape=False,
        order=1,
    ).astype(np.uint16)
    for name, array in (("r2.tif", fixed), ("r1.tif", moving), ("target.tif", moving)):
        tifffile.imwrite(root / name, array, metadata={"axes": "ZYX"})


@pytest.mark.parametrize(
    ("mode", "strategy", "parallelism"),
    [
        ("explicit", "whole_image", "coarse"),
        ("estimate", "whole_image", "coarse"),
        ("explicit", "serial_chunked", "fine_grained"),
        ("similarity", "serial_chunked", "fine_grained"),
    ],
)
def test_rotation_initialized_workflow_completes_with_authoritative_status(
    tmp_path: Path,
    mode: str,
    strategy: str,
    parallelism: str,
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "output"
    _images(data)
    if mode == "explicit":
        rotation = ExplicitXYRotationSpec(mode="explicit", angle_deg=15.0)
    elif mode == "estimate":
        rotation = EstimatedXYRotationSpec(
            mode="estimate",
            search_range_deg=(-25.0, 25.0),
            coarse_step_deg=5.0,
            refinement_step_deg=1.0,
            minimum_score_gain=0.001,
            minimum_peak_margin=0.001,
        )
    else:
        similarity = EstimatedXYSimilaritySpec(
            mode="estimate",
            search_range_deg=(-25.0, 25.0),
            coarse_step_deg=5.0,
            refinement_step_deg=1.0,
            scale_range=(0.9, 1.1),
            coarse_scale_step=0.05,
            refinement_scale_step=0.01,
            minimum_score_gain=0.001,
            minimum_peak_margin=0.0001,
        )
    manifest = DatasetManifest(
        schema_version=1,
        dataset_id=f"rotation-{mode}-{strategy}-{parallelism}",
        voxel_spacing_um=(0.3, 0.3, 0.8),
        reference=ReferenceSpec(
            round=2,
            registration_channel=PurePosixPath("r2.tif"),
            channels=(ReferenceChannelSpec(PurePosixPath("r2.tif")),),
        ),
        registration=RegistrationSpec(
            crop=False,
            linear=RegistrationStageSpec(
                type="Similarity" if mode == "similarity" else "TRSAA",
                accuracy="dry run",
            ),
            nonlinear=NonlinearRegistrationStageSpec(
                type="SyNOnly",
                accuracy="dry run",
                metric="mattes",
                radius=None,
            ),
        ),
        rounds=(
            RoundSpec(
                round=1,
                strategy=strategy,  # type: ignore[arg-type]
                registration_channel=PurePosixPath("r1.tif"),
                channels=(
                    ChannelSpec(PurePosixPath("r1.tif"), "linear"),
                    ChannelSpec(PurePosixPath("target.tif"), "linear"),
                ),
                initial_alignment=(
                    InitialAlignmentSpec(xy_similarity=similarity)
                    if mode == "similarity"
                    else InitialAlignmentSpec(rotation)
                ),
                chunk_size_zyx=(8, 48, 48) if strategy == "serial_chunked" else None,
                overlap_fraction=0.25 if strategy == "serial_chunked" else None,
            ),
        ),
    )
    execution = ExecutionProfile(
        schema_version=1,
        executor="sequential",
        paths=ExecutionPaths(data, output, None),
        native_threads=NativeThreadSpec(1, 1),
        local_dask=None,
        slurm=None,
        source_path=(tmp_path / "execution.yaml").resolve(),
        parallelism=ParallelismSpec(parallelism),  # type: ignore[arg-type]
    )

    plan = build_workflow_plan(manifest, execution)
    result = run_workflow(plan)
    status = inspect_workflow_status(output, manifest.dataset_id)

    assert result.status == "completed"
    assert status.complete is True
    assert all(item.status == "complete" for item in status.rounds)
    registration = plan.tasks[0].registration_directory
    evidence = json.loads(
        (registration / "initial_alignment_1.json").read_text(encoding="utf-8")
    )
    assert evidence["realized"]["accepted"] is True
    assert evidence["realized"]["source"] == (
        "explicit" if mode == "explicit" else "estimated"
    )
    assert evidence["realized"]["angle_deg"] == pytest.approx(15.0, abs=1.0)
    if mode == "similarity":
        assert evidence["realized"]["scale"] == pytest.approx(1.0, abs=0.02)
    moving_status = next(item for item in status.rounds if item.round == 1)
    assert moving_status.initial_alignment == {
        "configured": evidence["configured"],
        "realized": evidence["realized"],
    }
    run_record = json.loads(Path(result.run_record).read_text(encoding="utf-8"))
    assert run_record["initial_alignments"] == [
        {
            "round": 1,
            "configured": evidence["configured"],
            "realized": evidence["realized"],
            "evidence_path": str(registration / "initial_alignment_1.json"),
        }
    ]
    assert tifffile.imread(plan.tasks[0].channels[1].output).shape == (8, 48, 48)
    channel_record = CompletionRecord.read(
        plan.tasks[0].channels[1].output.with_name("target.tif.complete.json")
    )
    assert channel_record.request.parameters["transform_order"] == ["warp", "affine"]
    if parallelism == "fine_grained":
        from cyexm.workflow.fine_task import _validated_prepared_manifest

        prepared = PreparedRoundPaths.for_round(registration, 1)
        descriptor = json.loads(prepared.manifest.read_text(encoding="utf-8"))
        ArtifactStore().record_path(registration / "initial_alignment_1.mat").unlink()
        with pytest.raises(
            ValueError, match="initial alignment completion evidence is not current"
        ):
            _validated_prepared_manifest(
                plan.tasks[0],
                descriptor,
                {name: 1 for name in NATIVE_THREAD_ENV_VARS},
            )
    evidence_record = ArtifactStore().record_path(
        registration / "initial_alignment_1.json"
    )
    forged = json.loads(evidence_record.read_text(encoding="utf-8"))
    forged["request"]["stage"] = "unrelated-stage"
    evidence_record.write_text(canonical_compact_json(forged), encoding="utf-8")

    stale_status = inspect_workflow_status(output, manifest.dataset_id)

    stale_moving = next(item for item in stale_status.rounds if item.round == 1)
    assert stale_moving.initial_alignment is not None
    assert stale_moving.initial_alignment["realized"] is None
