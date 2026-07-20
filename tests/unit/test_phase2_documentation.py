from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATHS = (
    "examples/spheroid-dataset.yaml",
    "examples/execution-sequential.yaml",
    "examples/execution-local-dask.yaml",
    "examples/execution-slurm-512gb.yaml",
)


def _readme() -> str:
    return (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")


def _normalized_readme() -> str:
    return " ".join(_readme().split()).lower()


def test_readme_documents_every_workflow_command_option_and_example() -> None:
    readme = _readme()
    for command in (
        "cyexm workflow validate",
        "cyexm workflow plan",
        "cyexm workflow run",
        "cyexm workflow status",
        "--force-round",
        "--force-all",
        "--json",
        "--log-level",
        "--version",
    ):
        assert command in readme
    for path in EXAMPLE_PATHS:
        assert f"]({path})" in readme


def test_readme_documents_installation_and_scheduler_neutral_architecture() -> None:
    readme = _normalized_readme()
    for command in (
        "uv sync --locked",
        "uv sync --locked --extra workflow",
        "uv sync --locked --extra distributed",
        "uv sync --locked --extra hpc",
    ):
        assert command in readme
    for required in (
        "scientific dataset manifest",
        "execution profile",
        "one immutable `roundtask` per moving round",
        "serial inside each round task",
        "phase 1 completion records are the only restart authority",
        "no nested chunk multiprocessing",
        "planning is mutation-free",
    ):
        assert required in readme


def test_readme_documents_every_scientific_manifest_field_and_rule() -> None:
    readme = _readme()
    for field in (
        "schema_version",
        "dataset_id",
        "voxel_spacing_um",
        "x",
        "y",
        "z",
        "reference",
        "round",
        "channel",
        "registration",
        "crop",
        "linear",
        "type",
        "accuracy",
        "nonlinear",
        "rounds",
        "strategy",
        "registration_channel",
        "channels",
        "file",
        "interpolation",
        "chunk_size_zyx",
        "overlap_fraction",
    ):
        assert f"`{field}`" in readme

    normalized = _normalized_readme()
    for rule in (
        "micrometers in xyz order",
        "`whole_image` or `serial_chunked`",
        "required only for `serial_chunked`",
        "numpy zyx order",
        "`0 <= value < 1`",
        "`linear` or `nearestneighbor`",
        "safe relative posix path",
        "registration channel must be listed",
        "moving rounds are unique",
        "unknown keys",
        "duplicate mapping keys",
        "yaml aliases",
        "custom tags",
        "non-scalar keys",
        "additional yaml documents",
    ):
        assert rule in normalized


def test_readme_documents_every_execution_profile_field_and_rule() -> None:
    readme = _readme()
    for field in (
        "executor",
        "paths",
        "dataset_root",
        "output_root",
        "worker_local_directory",
        "native_threads",
        "itk",
        "blas",
        "local_dask",
        "workers",
        "threads_per_worker",
        "processes",
        "memory_limit",
        "startup_timeout",
        "task_retries",
        "slurm",
        "queue",
        "cores_per_worker",
        "processes_per_worker",
        "memory",
        "walltime",
        "interface",
        "scheduler_interface",
        "job_name",
        "job_extra_directives",
    ):
        assert f"`{field}`" in readme

    normalized = _normalized_readme()
    for rule in (
        "`sequential`, `local_dask`, or `slurm`",
        "relative paths resolve from the execution-profile file",
        "dataset and output roots must be disjoint",
        "worker-backed executors require",
        "must not be `/archive`",
        "positive built-in integers",
        "inherited environment",
        "one-thread default",
        "`processes` must be `true`",
        "`processes_per_worker` must be exactly `1`",
        "nonnegative built-in integer",
        "non-empty single-line string",
        "conflict with managed slurm options",
        "exactly one executor-specific section",
    ):
        assert rule in normalized


def test_readme_documents_output_lifecycle_and_status_semantics() -> None:
    readme = _readme()
    for output in (
        "scientific-plan.json",
        "workflow-runs/",
        "<run-id>.json",
        "rounds/round-002/registration/",
        "linear_transform_2.mat",
        "nonlinear_warp_2.nii.gz",
        "nonlinear_registered_2.tif",
        "rounds/round-002/channels/",
        "rounds/round-002/logs/round-002.log",
        ".complete.json",
    ):
        assert output in readme

    normalized = _normalized_readme()
    for required in (
        "normalized phase 1 registration diagnostic",
        "original channel resampled through the persisted transform chain",
        "clipped to uint16 without intensity normalization",
        "resume is the default",
        "mutation-free plan",
        "read-only status",
        "`pending`, `partial`, `complete`, or `invalid`",
        "scientific fingerprint",
        "force cannot bypass",
        "requires a new output root",
        "operational summaries never authorize artifact reuse",
    ):
        assert required in normalized


def test_readme_documents_executor_retry_cancellation_and_exit_contracts() -> None:
    normalized = _normalized_readme()
    for required in (
        "sequential executor is the numerical and restart reference",
        "process-based `localcluster`",
        "min(pending rounds, configured workers)",
        "one `round_slot`",
        "one process per job",
        "every connected worker must pass preflight",
        "independent round failures",
        "automatic retries default to zero",
        "executor-managed resubmission",
        "cancel unfinished futures",
        "close the owned client and cluster",
        "preserve committed artifacts",
        "coordinator runs in the process that invokes the cli",
        "512gb values are site-specific and editable",
    ):
        assert required in normalized

    for code in ("`0`", "`2`", "`3`", "`4`", "`130`"):
        assert code in normalized


def test_readme_states_hpc_acceptance_scope_and_remaining_non_goals() -> None:
    normalized = _normalized_readme()
    for required in (
        "portable fake-based slurm construction",
        "real slurm submission and hpc acceptance remain pending",
        "`-m hpc`",
        "explicit allocation",
        "parallelchunkedimageregistration",
        "per-chunk dask scheduling",
        "parallel channel transformation",
        "job arrays",
        "cloud or kubernetes executors",
        "automatic resource optimization",
        "no silent fallback",
    ):
        assert required in normalized
