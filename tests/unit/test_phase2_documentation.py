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
        "uv run cyexm workflow validate",
        "uv run cyexm workflow plan",
        "uv run cyexm workflow run",
        "uv run cyexm workflow status",
        "--force-round",
        "--force-all",
        "--json",
        "--log-level",
        "--version",
    ):
        assert command in readme
    for path in EXAMPLE_PATHS:
        assert f"]({path})" in readme

    assert not any(line.startswith("cyexm ") for line in readme.splitlines())


def test_readme_workflow_command_blocks_use_the_uv_managed_environment() -> None:
    readme = _readme()
    for command_block in (
        "uv run cyexm workflow validate examples/spheroid-dataset.yaml \\\n"
        "  --execution examples/execution-sequential.yaml",
        "uv run cyexm workflow plan examples/spheroid-dataset.yaml \\\n"
        "  --execution examples/execution-local-dask.yaml",
        "uv run cyexm workflow run examples/spheroid-dataset.yaml \\\n"
        "  --execution examples/execution-local-dask.yaml",
        "uv run cyexm workflow status results spheroid-registration-example",
        "uv run cyexm --version",
    ):
        assert command_block in readme

    normalized = _normalized_readme()
    assert "`uv sync` does not activate `.venv`" in normalized
    assert "all workflow commands below use `uv run`" in normalized


def test_readme_documents_installation_and_scheduler_neutral_architecture() -> None:
    readme = _normalized_readme()
    for command in (
        "uv sync --locked",
        "uv sync --locked --extra workflow",
        "uv sync --locked --extra distributed",
        "uv sync --locked --extra hpc",
        "uv sync --no-dev --extra examples",
        "uv sync --group dev --extra examples",
    ):
        assert command in readme
    for required in (
        "scientific dataset manifest",
        "execution profile",
        "one immutable `referencetask`",
        "one immutable `roundtask` per moving round",
        "hard executor barrier",
        "serial inside each moving-round task",
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
        "retired singular `reference.channel` form is rejected",
        "scientifically identical bounds",
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

    for exact_row in (
        "| `local_dask` / `workers` | positive built-in integer | Maximum local "
        "worker processes. |",
        "| `local_dask` / `startup_timeout` | non-empty single-line string | "
        "Bounds only Dask `Client` connection and `wait_for_workers`. |",
        "| `local_dask` / `task_retries` | nonnegative built-in integer | "
        "Executor-managed retry count after the first attempt. |",
        "| `slurm` / `workers` | positive built-in integer | Maximum number of "
        "worker jobs. |",
        "| `slurm` / `memory` | positive whole decimal `KB`, `MB`, `GB`, or `TB` "
        "string | Semantic memory request; the executor rounds up to whole binary "
        "GiB for SLURM and Dask. |",
        "| `slurm` / `startup_timeout` | non-empty single-line string | Bounds "
        "only Dask `Client` connection and `wait_for_workers`. |",
        "| `slurm` / `task_retries` | nonnegative built-in integer | "
        "Executor-managed retry count after the first attempt. |",
    ):
        assert exact_row in readme

    normalized = _normalized_readme()
    for rule in (
        "`sequential`, `local_dask`, or `slurm`",
        "relative paths resolve from the execution-profile file",
        "dataset and output roots must be disjoint",
        "worker-backed executors require",
        "must not be `/archive`",
        "inherited environment",
        "one-thread default",
        "`processes` must be `true`",
        "`processes_per_worker` must be exactly `1`",
        "conflict with managed slurm options",
        "exactly one executor-specific section",
        "`500gb` becomes `466g`",
        "500,363,689,984 bytes",
        "never passes a `gb` suffix to `#sbatch --mem`",
        "production-owned callable",
        "workers never need `tests` on `pythonpath`",
        "inside ordinary task execution",
        "`minmemorynode`",
        "`cfgtres` and `alloctres`",
        "job `reqtres` and `alloctres` are optional",
        "`allocmem=0` is unavailable",
    ):
        assert rule in normalized


def test_readme_documents_the_exact_startup_timeout_scope() -> None:
    normalized = _normalized_readme()
    for required in (
        "`startup_timeout` is not an end-to-end startup deadline",
        "does not bound `localcluster` or `slurmcluster` construction",
        "does not bound `scheduler_info()` calls",
        "does not bound the `client.run()` worker-preflight rpc",
    ):
        assert required in normalized


def test_readme_documents_planning_and_execution_thread_boundaries() -> None:
    normalized = _normalized_readme()
    for required in (
        "tiff header inspection runs in an isolated read-only probe process",
        "does not import numpy or tifffile in the coordinator",
        "sequential run applies the resolved native-thread environment",
        "before workflow status inspection or registration begins",
        "dask and slurm coordinators do not apply task native-thread settings",
        "source and completion-evidence tiff headers are inspected in-process",
        "workers do not launch nested header-probe child processes",
    ):
        assert required in normalized


def test_readme_documents_observed_native_thread_scaling() -> None:
    normalized = _normalized_readme()
    for required in (
        "observed threading benchmark",
        "| 1 / 1 | 310.5 s | 1.00x |",
        "| 8 / 8 | 88.3 s | 3.52x |",
        "| 30 / 30 | 69.5 s | 4.47x |",
        "scientifically consistent but not bit-for-bit identical",
        "largest nonlinear-correlation change was 0.003416",
        "one thread remains the conservative default",
        "full-volume stability evidence",
    ):
        assert required in normalized


def test_readme_documents_output_lifecycle_and_status_semantics() -> None:
    readme = _readme()
    for output in (
        "scientific-plan.json",
        "workflow-runs/",
        "<run-id>.json",
        "rounds/round-002/registration/",
        "round-001/",
        "reference_crop_indices.json",
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
        "do not create a per-round `fixed_reference.tif`",
        "forcing the reference round invalidates all moving descendants",
        "reference preparation as its declared round",
    ):
        assert required in normalized


def test_readme_documents_success_only_json_output() -> None:
    normalized = _normalized_readme()
    for required in (
        "`--json` is success-only",
        "failures return before result emission",
        "stdout is empty",
        "plain-text diagnostics go to stderr",
        "no json error object",
        "`--version` prints version text rather than a json result",
    ):
        assert required in normalized


def test_readme_documents_executor_retry_cancellation_and_exit_contracts() -> None:
    readme = _readme()
    normalized = _normalized_readme()
    for required in (
        "sequential executor is the numerical and restart reference",
        "process-based `localcluster`",
        "min(pending tasks, configured workers)",
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

    for exact_row in (
        "| `0` | Every round completed or was validly reused. |",
        "| `2` | Argument, YAML, configuration, input, plan, output, or "
        "fingerprint validation failed. |",
        "| `3` | Executor startup, worker connection, or preflight failure; also "
        "invalid SLURM operational evidence, including validation after round "
        "execution. Valid committed artifacts remain restartable. |",
        "| `4` | At least one round failed after execution began. |",
        "| `130` | Operator interruption or termination through the CLI run handler. |",
    ):
        assert exact_row in readme

    for required in (
        "post-execution slurm evidence validation",
        "preserves the round results in the operational run record",
        "the cli exits `3`",
        "a later ordinary run can reuse those valid artifacts",
    ):
        assert required in normalized


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
