from __future__ import annotations

import json
from pathlib import Path

import pytest
from hpc.fixed_affine_chunk_characterization import (
    FixedAffineFineRunner,
    SourceRoundState,
    clone_shared_prepared_state,
    materialize_shared_prepared_state,
)

from cyexm.config.models import (
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    RegistrationSpec,
    RegistrationStageSpec,
)
from cyexm.registration.artifacts import ArtifactStore, CompletionRecord, FileIdentity
from cyexm.registration.backend import AntsRegistrationBackend
from cyexm.registration.prepared import (
    PreparedManifest,
    PreparedRoundPaths,
    commit_prepared_round,
)
from cyexm.registration.whole_image import ImageRegistration
from cyexm.runtime import NativeThreadConfig
from cyexm.workflow._round_provenance import RegistrationArtifactPaths
from cyexm.workflow.fine_models import RoundPreparationTask
from cyexm.workflow.models import ChannelTask, RoundTask

pytestmark = pytest.mark.integration


def _record(path: Path) -> CompletionRecord:
    return CompletionRecord.read(ArtifactStore().record_path(path))


def test_shared_preparation_reuses_affine_without_estimating_it_again(
    synthetic_tiffs: tuple[Path, Path],
    tmp_path: Path,
    native_threads: NativeThreadConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed, moving = synthetic_tiffs
    source_root = tmp_path / "source"
    source_registration = ImageRegistration(
        fixed,
        moving,
        source_root,
        imaging_round=2,
        pixel_size=(1.0, 1.0, 1.0),
        crop=True,
        prepared_fixed=True,
        native_threads=native_threads,
        linear_type="TRSAA",
        linear_accuracy="dry run",
        nonlinear_type="SyNOnly",
        nonlinear_accuracy="dry run",
        nonlinear_metric="CC",
        nonlinear_radius=4,
    )
    source_manifest = commit_prepared_round(source_registration)
    source_paths = RegistrationArtifactPaths.for_round(source_root, 2)
    source = SourceRoundState(
        round=2,
        fixed_reference=fixed,
        moving_image=moving,
        moving_crop=source_paths.moving_crop,
        pre_metrics=source_paths.pre_metrics,
        affine=source_paths.affine,
        linear_metrics=source_paths.linear_metrics,
        fixed_geometry=source_manifest.geometry,
        moving_crop_record=_record(source_paths.moving_crop),
        pre_metrics_record=_record(source_paths.pre_metrics),
        affine_record=_record(source_paths.affine),
        linear_metrics_record=_record(source_paths.linear_metrics),
    )
    source_affine_identity = FileIdentity.from_path(source.affine)
    shared_root = tmp_path / "shared"
    task = RoundTask(
        dataset_id="fixed-affine-integration",
        round=2,
        strategy="serial_chunked",
        fixed_image=fixed,
        moving_image=moving,
        registration_directory=shared_root,
        log_path=shared_root / "round.log",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        registration=RegistrationSpec(
            crop=True,
            linear=RegistrationStageSpec(type="TRSAA", accuracy="dry run"),
            nonlinear=NonlinearRegistrationStageSpec(
                type="SyNOnly",
                accuracy="dry run",
                metric="CC",
                radius=4,
            ),
        ),
        native_threads=NativeThreadSpec(itk=1, blas=1),
        channels=(
            ChannelTask(
                source=moving,
                output=shared_root / "channel.tif",
                interpolation="linear",
            ),
        ),
        chunk_size_zyx=(16, 16, 16),
        overlap_fraction=0.15,
        prepared_fixed=True,
    )

    def fail_linear(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("affine estimation invoked")

    monkeypatch.setattr(AntsRegistrationBackend, "linear", fail_linear)
    shared = materialize_shared_prepared_state(task, source)

    assert shared.result.status == "completed"
    assert FileIdentity.from_path(source.affine) == source_affine_identity
    assert shared.manifest.affine == FileIdentity.from_path(
        RegistrationArtifactPaths.for_round(shared_root, 2).affine
    )
    prepared = PreparedRoundPaths.for_round(shared_root, 2)
    assert (
        PreparedManifest.from_dict(
            json.loads(prepared.manifest.read_text(encoding="utf-8"))
        )
        == shared.manifest
    )
    assert all(path.is_file() for path in prepared.array_paths)

    condition_root = tmp_path / "condition"
    condition_task = RoundTask(
        dataset_id="fixed-affine-condition",
        round=2,
        strategy="serial_chunked",
        fixed_image=fixed,
        moving_image=moving,
        registration_directory=condition_root,
        log_path=condition_root / "round.log",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        registration=task.registration,
        native_threads=task.native_threads,
        channels=(
            ChannelTask(
                source=moving,
                output=condition_root / "channel.tif",
                interpolation="linear",
            ),
        ),
        chunk_size_zyx=(16, 8, 8),
        overlap_fraction=0.15,
        prepared_fixed=True,
    )
    condition_result = clone_shared_prepared_state(
        RoundPreparationTask.create(condition_task),
        shared,
    )
    condition_prepared = PreparedRoundPaths.for_round(condition_root, 2)

    assert condition_result.status == "completed"
    assert condition_result.prepared_descriptor is not None
    assert all(path.is_file() for path in condition_result.task.output_paths)
    for shared_path, condition_path in zip(
        prepared.array_paths,
        condition_prepared.array_paths,
        strict=True,
    ):
        assert shared_path.read_bytes() == condition_path.read_bytes()

    second_root = tmp_path / "condition-two"
    second_task = RoundTask(
        dataset_id="fixed-affine-condition-two",
        round=2,
        strategy="serial_chunked",
        fixed_image=fixed,
        moving_image=moving,
        registration_directory=second_root,
        log_path=second_root / "round.log",
        voxel_spacing_xyz=(1.0, 1.0, 1.0),
        registration=task.registration,
        native_threads=task.native_threads,
        channels=(
            ChannelTask(
                source=moving,
                output=second_root / "channel.tif",
                interpolation="linear",
            ),
        ),
        chunk_size_zyx=(16, 8, 8),
        overlap_fraction=0.15,
        prepared_fixed=True,
    )
    runner = FixedAffineFineRunner(shared_rounds=(shared,))
    runner_result = runner(RoundPreparationTask.create(second_task))

    assert runner_result.status == "completed"
    assert runner_result.prepared_descriptor is not None
