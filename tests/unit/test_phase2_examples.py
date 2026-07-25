from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import cast

import pytest
import yaml

from cyexm.config import (
    ExecutionPaths,
    LocalDaskSpec,
    NativeThreadSpec,
    NonlinearRegistrationStageSpec,
    ParallelismSpec,
    RegistrationPreprocessingSpec,
    RegistrationSpec,
    RegistrationStageSpec,
    SlurmMemoryQuantity,
    SlurmSpec,
    load_dataset_manifest,
    load_execution_profile,
    normalize_slurm_memory,
)

pytestmark = pytest.mark.unit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATHS = (
    "examples/spheroid-dataset.yaml",
    "examples/execution-sequential.yaml",
    "examples/execution-local-dask.yaml",
    "examples/execution-slurm-512gb.yaml",
)


@pytest.mark.parametrize("relative_path", EXAMPLE_PATHS)
def test_phase2_example_is_checked_in(relative_path: str) -> None:
    assert (REPOSITORY_ROOT / relative_path).is_file()


def test_spheroid_manifest_records_approved_scientific_inputs() -> None:
    manifest = load_dataset_manifest(REPOSITORY_ROOT / EXAMPLE_PATHS[0])

    assert manifest.schema_version == 1
    assert manifest.dataset_id == "spheroid-registration-example"
    assert manifest.voxel_spacing_um == (0.324, 0.324, 0.900)
    assert manifest.reference.round == 1
    assert manifest.reference.registration_channel == PurePosixPath(
        "round_1_488_tubulin.tif"
    )
    assert tuple(channel.file for channel in manifest.reference.channels) == (
        PurePosixPath("round_1_488_tubulin.tif"),
    )
    assert manifest.registration == RegistrationSpec(
        crop=False,
        linear=RegistrationStageSpec(type="TRSAA", accuracy="high"),
        nonlinear=NonlinearRegistrationStageSpec(
            type="SyNOnly", accuracy="high", metric="CC", radius=4
        ),
    )
    assert tuple(item.round for item in manifest.rounds) == (2, 3, 4)
    assert tuple(item.strategy for item in manifest.rounds) == (
        "whole_image",
        "serial_chunked",
        "whole_image",
    )
    assert tuple(item.registration_channel.name for item in manifest.rounds) == (
        "round_2_488_tubulin.tif",
        "round_3_488_tubulin.tif",
        "round_4_561_tubulin.tif",
    )
    assert tuple(
        tuple(
            (channel.file.as_posix(), channel.interpolation)
            for channel in item.channels
        )
        for item in manifest.rounds
    ) == (
        (
            ("round_2_488_tubulin.tif", "linear"),
            ("round_2_561_bcatenin.tif", "linear"),
            ("round_2_647_hsp60.tif", "linear"),
        ),
        (
            ("round_3_488_tubulin.tif", "linear"),
            ("round_3_647_fbl.tif", "linear"),
        ),
        (
            ("round_4_488_vimentin.tif", "linear"),
            ("round_4_561_tubulin.tif", "linear"),
            ("round_4_647_pex14.tif", "linear"),
        ),
    )
    assert manifest.rounds[0].chunk_size_zyx is None
    assert manifest.rounds[0].overlap_fraction is None
    assert manifest.rounds[1].chunk_size_zyx == (512, 512, 512)
    assert manifest.rounds[1].overlap_fraction == 0.15
    assert manifest.rounds[2].chunk_size_zyx is None
    assert manifest.rounds[2].overlap_fraction is None


def test_preprocessed_spheroid_manifest_is_portable_and_uses_default_sigma() -> None:
    path = REPOSITORY_ROOT / "examples/preprocessed-spheroid-dataset.yaml"

    manifest = load_dataset_manifest(path)

    assert manifest.registration.preprocessing == RegistrationPreprocessingSpec(
        filter_sigma_zyx=(12.0, 12.0, 12.0)
    )
    assert manifest.registration.nonlinear.metric == "CC"
    assert manifest.registration.nonlinear.radius == 4
    raw = path.read_text(encoding="utf-8")
    assert "filter_sigma_zyx: 12" in raw
    assert not any(marker in raw for marker in ("/project", "/archive", "/endosome"))


def _yaml_scalars(value: object) -> Iterator[object]:
    if isinstance(value, Mapping):
        for key, item in cast(Mapping[object, object], value).items():
            yield from _yaml_scalars(key)
            yield from _yaml_scalars(item)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for item in cast(Sequence[object], value):
            yield from _yaml_scalars(item)
    else:
        yield value


def test_spheroid_manifest_contains_no_site_specific_absolute_scalar() -> None:
    raw = cast(
        object,
        yaml.safe_load((REPOSITORY_ROOT / EXAMPLE_PATHS[0]).read_text("utf-8")),
    )

    for scalar in _yaml_scalars(raw):
        if isinstance(scalar, str):
            assert not scalar.startswith(("/project", "/archive", "/endosome"))


def test_sequential_example_records_portable_conservative_resources() -> None:
    profile = load_execution_profile(REPOSITORY_ROOT / EXAMPLE_PATHS[1])

    assert profile.schema_version == 1
    assert profile.executor == "sequential"
    assert profile.paths == ExecutionPaths(
        dataset_root=(REPOSITORY_ROOT / "data").resolve(),
        output_root=(REPOSITORY_ROOT / "results").resolve(),
        worker_local_directory=None,
    )
    assert profile.native_threads == NativeThreadSpec(itk=1, blas=1)
    assert profile.parallelism == ParallelismSpec(mode="coarse")
    assert profile.local_dask is None
    assert profile.slurm is None

    raw = cast(
        Mapping[str, object],
        yaml.safe_load((REPOSITORY_ROOT / EXAMPLE_PATHS[1]).read_text("utf-8")),
    )
    assert raw["paths"] == {
        "dataset_root": "../data",
        "output_root": "../results",
        "worker_local_directory": None,
    }


def test_local_dask_example_records_portable_process_resources() -> None:
    profile = load_execution_profile(REPOSITORY_ROOT / EXAMPLE_PATHS[2])

    assert profile.schema_version == 1
    assert profile.executor == "local_dask"
    assert profile.paths == ExecutionPaths(
        dataset_root=(REPOSITORY_ROOT / "data").resolve(),
        output_root=(REPOSITORY_ROOT / "results").resolve(),
        worker_local_directory=(REPOSITORY_ROOT / "scratch").resolve(),
    )
    assert profile.native_threads == NativeThreadSpec(itk=1, blas=1)
    assert profile.parallelism == ParallelismSpec(mode="fine_grained")
    assert profile.local_dask == LocalDaskSpec(
        workers=3,
        threads_per_worker=1,
        processes=True,
        memory_limit="24GB",
        startup_timeout="30s",
        task_retries=0,
    )
    assert profile.slurm is None

    raw = cast(
        Mapping[str, object],
        yaml.safe_load((REPOSITORY_ROOT / EXAMPLE_PATHS[2]).read_text("utf-8")),
    )
    assert raw["paths"] == {
        "dataset_root": "../data",
        "output_root": "../results",
        "worker_local_directory": "../scratch",
    }


def test_slurm_example_records_editable_512gb_resources() -> None:
    profile = load_execution_profile(REPOSITORY_ROOT / EXAMPLE_PATHS[3])

    assert profile.schema_version == 1
    assert profile.executor == "slurm"
    assert profile.paths == ExecutionPaths(
        dataset_root=Path("/archive/path/to/dataset").resolve(),
        output_root=Path("/archive/path/to/results").resolve(),
        worker_local_directory=Path("/project/path/to/cyexm-worker").resolve(),
    )
    assert profile.native_threads == NativeThreadSpec(itk=1, blas=1)
    assert profile.parallelism == ParallelismSpec(mode="fine_grained")
    assert profile.local_dask is None
    assert profile.slurm == SlurmSpec(
        queue="512GB",
        workers=3,
        cores_per_worker=72,
        processes_per_worker=1,
        memory="500GB",
        walltime="72:00:00",
        interface="ib0",
        scheduler_interface="ib0",
        startup_timeout="15m",
        task_retries=0,
        job_name="cyexm-registration",
        job_extra_directives=(),
    )
    assert normalize_slurm_memory(profile.slurm.memory) == SlurmMemoryQuantity(
        profile="500GB",
        semantic_bytes=500_000_000_000,
        scheduler="466G",
        scheduler_bytes=500_363_689_984,
        dask="466GiB",
    )

    text = (REPOSITORY_ROOT / EXAMPLE_PATHS[3]).read_text("utf-8")
    for guidance in (
        "Site-specific editable shared dataset path",
        "Site-specific editable shared results path",
        "Site-specific editable worker scratch path",
        "Every resource below is site-specific and editable",
    ):
        assert guidance in text
