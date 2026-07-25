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
        "`parallelism.mode`",
        "`coarse`",
        "`fine_grained`",
        "omitting `parallelism` normalizes to `coarse`",
        "scientific fingerprint is unchanged",
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
        "round-002-<stage>-<item>.log",
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
        "coarse moving rounds use the shared `round-nnn.log`",
        "fine moving tasks never append to that shared round log",
        "reference task retains its ordinary `round-nnn.log`",
    ):
        assert required in normalized
    assert "stderr and per-round shared logs" not in normalized


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
        "coarse mode, every connected worker must pass preflight before any "
        "registration submission",
        "every later newcomer must pass preflight and activation before that worker "
        "is eligible for scientific submission",
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
        "invalid fine execution evidence for sequential, Local Dask, or SLURM, "
        "including validation after task execution. Unauthenticated current-run "
        "summaries are omitted, while valid committed artifacts remain "
        "restartable. |",
        "| `4` | At least one round failed after execution began. |",
        "| `130` | Operator interruption or termination through the CLI run handler. |",
    ):
        assert exact_row in readme

    for required in (
        "invalid fine execution evidence",
        "unauthenticated current-run summaries are omitted from durable `rounds`",
        "pre-run authenticated reuse remains",
        "valid phase 1 artifacts committed before rejection remain restartable",
    ):
        assert required in normalized


def test_readme_states_hpc_acceptance_scope_and_remaining_non_goals() -> None:
    normalized = _normalized_readme()
    for required in (
        "portable fake-based slurm construction",
        "real slurm submission and hpc acceptance completed on the audited lab "
        "partition",
        "phase 2 acceptance audit",
        "`-m hpc`",
        "explicit allocation",
        "parallelchunkedimageregistration",
        "fine-grained real-slurm production acceptance remains pending",
        "portable real-ants/local-dask evidence is complete",
        "whole-image affine remains",
        "multiple workers per node",
        "job arrays",
        "cloud or kubernetes executors",
        "automatic resource optimization",
        "no silent fallback",
    ):
        assert required in normalized


def test_fine_grained_guide_documents_the_exact_graph_and_resource_contract() -> None:
    guide_path = REPOSITORY_ROOT / "docs/fine-grained-execution.md"
    assert guide_path.is_file()
    guide = " ".join(guide_path.read_text(encoding="utf-8").split()).lower()

    for required in (
        "reference barrier",
        "whole-image affine",
        "prepared fixed-grid bundle",
        "nonlinear chunk",
        "deterministic assembly",
        "every declared moving channel",
        "round validation",
        "`serial_chunked`",
        "ascending chunk id",
        "later ids overwrite earlier values",
        "one immutable scientific task at a time",
        "worker may execute multiple tasks sequentially",
        "one candidate worker",
        "preflight",
        "`round_slot`",
        "demand-driven",
        "round-robin",
        "configured worker cap",
        "one worker/job per dedicated node",
        "workers never create schedulers",
        "multiple workers per node",
        "future optimization",
    ):
        assert required in guide


def test_fine_grained_guide_documents_artifact_authority_and_recovery() -> None:
    guide = " ".join(
        (REPOSITORY_ROOT / "docs/fine-grained-execution.md")
        .read_text(encoding="utf-8")
        .split()
    ).lower()

    for required in (
        "`fixed_scalar.npy`",
        "`moving_linear_scalar.npy`",
        "`fixed_mask.npy`",
        "`moving_linear_mask.npy`",
        "memory-mapped",
        "fixed-reference grid",
        "`chunk_0000_registered_core.npy`",
        "internal progress",
        "public completion",
        "prepared arrays and scalar cores never make a round complete",
        "completion records",
        "coarse and fine-grained modes share",
        "unique per-task logs",
        "submitted siblings settle",
        "cancellation",
        "ordinary later run",
        "unequal",
        "z, y, and x",
        "legacy clearex/ants behavior",
        "portable real-ants/local-dask evidence is complete",
        "real slurm production acceptance remains pending",
        "`format_version` is exactly `2`",
        "`pixeltypes`",
        "`fixed_mask`",
        "`fixed_scalar`",
        "`moving_linear_mask`",
        "`moving_linear_scalar`",
        "`unsigned char`",
        "`unsigned int`",
        "`float`",
        "`double`",
        "not a user-selected execution option",
        "fails closed",
    ):
        assert required in guide


def test_maintained_docs_distinguish_shared_cores_and_rejected_evidence() -> None:
    readme = " ".join(_readme().split()).lower()
    artifacts = " ".join(
        (REPOSITORY_ROOT / "docs/registration-artifacts.md")
        .read_text(encoding="utf-8")
        .split()
    ).lower()
    guide = " ".join(
        (REPOSITORY_ROOT / "docs/fine-grained-execution.md")
        .read_text(encoding="utf-8")
        .split()
    ).lower()

    for text in (readme, artifacts, guide):
        assert "scalar cores are shared coarse/fine serial-chunk artifacts" in text
        assert "prepared arrays are fine-only" in text
    for required in (
        "unauthenticated current-run summaries are omitted from durable `rounds`",
        "pre-run authenticated reuse remains",
        "invalid fine execution evidence",
        "sequential, local dask, and slurm",
    ):
        assert required in readme
    for stale in (
        "coarse serial execution, registered scalar chunk data is reconstructed",
        "fine-only prepared arrays and scalar cores",
        "it preserves the round results in the operational run record",
    ):
        assert stale not in f"{readme} {artifacts} {guide}"


def test_phase_two_acceptance_is_coarse_history_not_fine_slurm_acceptance() -> None:
    acceptance = " ".join(
        (REPOSITORY_ROOT / "docs/phase-2-acceptance.md")
        .read_text(encoding="utf-8")
        .split()
    ).lower()
    for required in (
        "historical coarse round-level phase 2e baseline",
        "[fine-grained registration](fine-grained-execution.md)",
        "fine chunk and declared-channel execution is implemented",
        "fine-grained real slurm production acceptance remains pending tasks 12-13",
    ):
        assert required in acceptance
    assert (
        "per-chunk dask parallelism, parallel channel transformation, and resource "
        "auto-optimization remain future work"
    ) not in acceptance


def test_fixed_affine_chunk_characterization_is_complete_and_navigable() -> None:
    report_path = REPOSITORY_ROOT / "docs/fixed-affine-chunk-size-characterization.md"
    assert report_path.is_file()
    report = report_path.read_text(encoding="utf-8")
    normalized = " ".join(report.split()).lower()
    navigation = (REPOSITORY_ROOT / "docs/myst.yml").read_text(encoding="utf-8")
    index = (REPOSITORY_ROOT / "docs/index.md").read_text(encoding="utf-8")

    assert "<!-- RESULTS_TABLES -->" not in report
    assert "<!-- INTERPRETATION -->" not in report
    assert "fixed-affine-chunk-size-characterization.md" in navigation
    assert "fixed-affine-chunk-size-characterization.md" in index
    for required in (
        "accepted radius-4 affine state held exactly fixed",
        "trsaa estimator was never called",
        "`[512,1125,1125]`",
        "`[512,512,512]`",
        "`[512,256,256]`",
        "`[512,128,128]`",
        "overlap fraction",
        "0.15",
        "y = 1600:1800",
        "x = 300:400",
        "y = 1227:1427",
        "x = 840:1040",
        "boundary optimum",
        "seam-adjacent",
        "geometric jacobian",
        "dataset-specific",
        "does not establish a universal chunk size",
    ):
        assert required in normalized
    for name in (
        "regional-residuals.png",
        "regional-overlays.png",
        "runtime-throughput.png",
        "seam-interior.png",
        "jacobian-validity.png",
    ):
        assert f"images/fixed-affine-chunk-size/{name}" in report
        assert (
            REPOSITORY_ROOT / "docs/images/fixed-affine-chunk-size" / name
        ).is_file()
