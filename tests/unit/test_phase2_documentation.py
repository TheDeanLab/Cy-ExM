import re
from fnmatch import fnmatchcase
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

pytestmark = pytest.mark.unit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
EXAMPLE_PATHS = (
    "examples/spheroid-dataset.yaml",
    "examples/explicit-rotation-dataset.yaml",
    "examples/estimated-rotation-dataset.yaml",
    "examples/execution-sequential.yaml",
    "examples/execution-local-dask.yaml",
    "examples/execution-slurm-512gb.yaml",
    "examples/execution-slurm-512gb-multi-process.yaml",
)


def test_initial_alignment_guide_and_examples_cover_the_public_contract() -> None:
    guide = (DOCS_ROOT / "initial-alignment.md").read_text(encoding="utf-8")
    manifest = (DOCS_ROOT / "dataset-manifest.md").read_text(encoding="utf-8")
    for phrase in (
        "target-blind",
        "physical +Z",
        "appears clockwise",
        "initial_alignment_{round}.json",
        "Automatic rejection",
        "one resampling pass",
    ):
        assert phrase in guide
    for key in (
        "initial_alignment",
        "xy_rotation",
        "angle_deg",
        "search_range_deg",
        "coarse_step_deg",
        "refinement_step_deg",
        "minimum_score_gain",
        "minimum_peak_margin",
    ):
        assert key in manifest

    from cyexm.config import load_dataset_manifest

    for name in (
        "explicit-rotation-dataset.yaml",
        "estimated-rotation-dataset.yaml",
    ):
        parsed = load_dataset_manifest(REPOSITORY_ROOT / "examples" / name)
        assert parsed.reference.round == 2
        assert parsed.rounds[0].initial_alignment is not None


def _myst_toc_documents() -> tuple[str, ...]:
    navigation = (DOCS_ROOT / "myst.yml").read_text(encoding="utf-8")
    return tuple(
        match.group(1)
        for line in navigation.splitlines()
        if (match := re.fullmatch(r"\s*- file:\s*(\S+)\s*", line))
    )


def _myst_exclude_patterns() -> tuple[str, ...]:
    navigation = (DOCS_ROOT / "myst.yml").read_text(encoding="utf-8")
    patterns: list[str] = []
    in_excludes = False
    for line in navigation.splitlines():
        if line.strip() == "exclude:":
            in_excludes = True
            continue
        if not in_excludes:
            continue
        match = re.fullmatch(r"\s*-\s+(\S+)\s*", line)
        if match is not None:
            patterns.append(match.group(1))
            continue
        if line and not line.startswith("    "):
            break
    return tuple(patterns)


PUBLIC_BOOK_DOCUMENTS = _myst_toc_documents()


def _markdown_link_targets(text: str) -> tuple[str, ...]:
    targets: list[str] = []
    for match in re.finditer(r"(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)", text):
        raw = match.group(1).strip()
        if raw.startswith("<") and ">" in raw:
            targets.append(raw[1 : raw.index(">")])
        else:
            targets.append(raw.split(maxsplit=1)[0])
    return tuple(targets)


def _markdown_fragment_ids(path: Path) -> frozenset[str]:
    identifiers: set[str] = set()
    slug_counts: dict[str, int] = {}
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if match := re.fullmatch(r"\(([^)]+)\)=\s*", line):
            identifiers.add(match.group(1))
            continue
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match is None:
            continue
        heading = match.group(1)
        if explicit := re.search(r"\{#([^}]+)\}\s*$", heading):
            identifiers.add(explicit.group(1))
            heading = heading[: explicit.start()]
        heading = re.sub(r"<[^>]+>", "", heading)
        heading = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", heading)
        heading = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", heading)
        heading = heading.replace("`", "").lower().strip()
        slug = re.sub(r"[^\w\- ]", "", heading)
        slug = re.sub(r"\s", "-", slug)
        duplicate_index = slug_counts.get(slug, 0)
        slug_counts[slug] = duplicate_index + 1
        identifiers.add(slug if duplicate_index == 0 else f"{slug}-{duplicate_index}")
    return frozenset(identifiers)


def _readme() -> str:
    return (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")


def _document(name: str) -> str:
    return (DOCS_ROOT / name).read_text(encoding="utf-8")


def _normalized_document(name: str) -> str:
    return " ".join(_document(name).split()).lower()


def _normalized_readme() -> str:
    return " ".join(_readme().split()).lower()


def test_readme_is_a_concise_first_use_guide() -> None:
    readme = _readme()
    headings = [line for line in readme.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Installation",
        "## Register two images",
        "## Register multiple imaging rounds",
        "## Documentation",
        "## License",
    ]
    assert 100 <= len(readme.splitlines()) <= 150
    for required in (
        "git clone https://github.com/TheDeanLab/Cy-ExM.git",
        "uv sync --locked --no-dev",
        "from cyexm.registration import register_round",
        'fixed_image_path=Path("data/reference.tif")',
        'moving_image_path=Path("data/moving.tif")',
        "https://thedeanlab.github.io/Cy-ExM/",
        "examples/spheroid-dataset.yaml",
        "examples/execution-sequential.yaml",
        "LICENSE.md",
    ):
        assert required in readme


def test_readme_omits_advanced_and_historical_material() -> None:
    readme = _normalized_readme()
    for forbidden in (
        "phase 1",
        "phase 2",
        "observed threading benchmark",
        "dask-jobqueue",
        "slurmcluster",
        "development checks",
        "acceptance audit",
        "`worker_local_directory`",
        "`job_extra_directives`",
        "`startup_timeout`",
    ):
        assert forbidden not in readme


def test_public_documentation_home_routes_readers_by_task() -> None:
    index = _normalized_document("index.md")
    for required in (
        "register two images",
        "choose a registration strategy",
        "run multiple imaging rounds",
        "configure advanced or hpc execution",
        "contribute to cy-exm",
    ):
        assert required in index
    assert "**[register two images](getting-started.md):** use the python api" in index
    assert (
        "**[run multiple imaging rounds](workflow-guide.md):** use yaml manifests"
        in index
    )
    assert "phase 1" not in index
    assert "phase 2" not in index


def test_developer_guide_preserves_maintenance_contract() -> None:
    guide = _document("development.md")
    for required in (
        "uv sync --locked --group dev --extra examples --extra hpc",
        'uv run pytest -m "unit and not hpc" tests/unit -v',
        'uv run pytest -m "integration and not hpc" tests/integration -v',
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run basedpyright",
        "uv run pre-commit run --all-files",
        "jupyter book build --html",
        "Dask/distributed 2025.1.0",
        "dask-jobqueue 0.9.0",
        "fresh compatibility evidence",
        "agentic-api-documentation.md",
    ):
        assert required in guide


def test_public_toc_and_local_links_are_resolvable() -> None:
    toc_documents = _myst_toc_documents()
    assert toc_documents
    assert len(toc_documents) == len(set(toc_documents)), toc_documents

    public_paths = [REPOSITORY_ROOT / "README.md"]
    for document in toc_documents:
        path = (DOCS_ROOT / document).resolve()
        assert path.is_relative_to(DOCS_ROOT.resolve()), document
        assert path.is_file(), document
        public_paths.append(path)

    exclude_patterns = _myst_exclude_patterns()
    assert set(exclude_patterns) == {
        "AGENTS.md",
        "legacy-*-report.md",
        "migration-from-clearex.md",
        "superpowers/**",
    }
    excluded_paths: set[Path] = set()
    for pattern in exclude_patterns:
        matches = {
            path.resolve()
            for path in DOCS_ROOT.rglob("*")
            if path.is_file()
            and fnmatchcase(path.relative_to(DOCS_ROOT).as_posix(), pattern)
        }
        assert matches, f"docs/myst.yml exclude pattern matches nothing: {pattern!r}"
        excluded_paths.update(matches)
    assert not excluded_paths.intersection(public_paths)

    for source in public_paths:
        for target in _markdown_link_targets(source.read_text(encoding="utf-8")):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("/"):
                continue
            destination = (
                source
                if not parsed.path
                else (source.parent / unquote(parsed.path)).resolve()
            )
            assert destination.is_relative_to(REPOSITORY_ROOT.resolve()), (
                f"{source}: local target escapes the repository: {target!r}"
            )
            assert destination.exists(), f"{source}: missing local target {target!r}"
            assert destination not in excluded_paths, (
                f"{source}: public link targets excluded page {target!r}"
            )
            if parsed.fragment and destination.suffix.lower() == ".md":
                fragment = unquote(parsed.fragment)
                assert fragment in _markdown_fragment_ids(destination), (
                    f"{source}: missing fragment {fragment!r} in {destination}"
                )


def test_architecture_records_supported_non_goals() -> None:
    guide = _normalized_document("architecture.md")
    for required in (
        "parallelchunkedimageregistration",
        "job arrays",
        "cloud or kubernetes executors",
        "automatic resource optimization",
        "no silent fallback",
    ):
        assert required in guide


def test_architecture_and_workflow_preserve_scheduler_neutral_contract() -> None:
    text = _normalized_document("architecture.md") + _normalized_document(
        "workflow-guide.md"
    )
    for required in (
        "scientific dataset manifest",
        "execution profile",
        "one immutable `referencetask`",
        "one immutable `roundtask` per moving round",
        "hard executor barrier",
        "serial inside each moving-round task",
        "completion records are the only restart authority",
        "no nested chunk multiprocessing",
        "planning is mutation-free",
        "`parallelism.mode`",
        "`coarse`",
        "`fine_grained`",
        "scientific fingerprint is unchanged",
    ):
        assert required in text


def test_package_metadata_advertises_the_documentation_site() -> None:
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'Documentation = "https://thedeanlab.github.io/Cy-ExM/"' in pyproject


def test_public_book_omits_historical_source_repository_references() -> None:
    for page in ("README.md", *PUBLIC_BOOK_DOCUMENTS):
        text = (_readme() if page == "README.md" else _document(page)).lower()
        assert "clearex" not in text, page
        assert "/project/bioinformatics" not in text, page


def test_registration_foundations_separates_science_from_execution() -> None:
    guide = _normalized_document("registration-foundations.md")
    for required in (
        "whole-image registration",
        "serial-chunked registration",
        "scientific registration strategy",
        "execution environment",
        "task granularity",
        "affine",
        "nonlinear",
        "memory",
        "restart",
    ):
        assert required in guide
    assert guide.index("scientific registration strategy") < guide.index(
        "execution environment"
    )
    assert "phase 1" not in guide
    assert "phase 2" not in guide


def test_getting_started_gives_a_complete_source_checkout() -> None:
    guide = _document("getting-started.md")
    for required in (
        "git clone https://github.com/TheDeanLab/Cy-ExM.git",
        "cd Cy-ExM",
        "uv sync --locked --no-dev",
        "uv run",
        "Python 3.12",
        "caller-provided",
    ):
        assert required in guide


def test_registration_foundations_owns_direct_python_scientific_defaults() -> None:
    guide = _normalized_document("registration-foundations.md")
    for required in (
        "direct-python scientific defaults",
        "`trsaa` with the `low` linear preset",
        "`synonly` with the `high` nonlinear preset",
        "omitted `nonlinear_metric` uses `mattes` with 32 bins",
        "selects `cc` explicitly",
        "omitted `nonlinear_radius` uses the provisional scalar voxel radius 4",
        "manifest requires an explicit nonlinear `metric`",
    ):
        assert required in guide
    assert (
        "(registration-foundations.md#direct-python-scientific-defaults)"
        in _document("getting-started.md")
    )


def test_workflow_and_cli_docs_cover_every_command_option_and_example() -> None:
    text = _document("workflow-guide.md") + _document("cli-reference.md")
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
        assert command in text
    for path in EXAMPLE_PATHS:
        assert f"](../{path})" in text

    assert not any(line.startswith("cyexm ") for line in text.splitlines())


def test_workflow_guide_uses_the_uv_managed_environment() -> None:
    guide = _document("workflow-guide.md")
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
        assert command_block in guide

    normalized = _normalized_document("workflow-guide.md")
    assert "`uv sync` does not activate `.venv`" in normalized
    assert "all workflow commands below use `uv run`" in normalized


def test_getting_started_documents_installation_choices() -> None:
    guide = _normalized_document("getting-started.md")
    for command in (
        "uv sync --locked",
        "uv sync --locked --extra workflow",
        "uv sync --locked --extra distributed",
        "uv sync --locked --extra hpc",
        "uv sync --no-dev --extra examples",
        "uv sync --group dev --extra examples",
    ):
        assert command in guide


def test_dataset_manifest_reference_documents_every_field_and_rule() -> None:
    reference = _document("dataset-manifest.md")
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
        assert f"`{field}`" in reference

    normalized = _normalized_document("dataset-manifest.md")
    for rule in (
        "micrometers in xyz order",
        "`whole_image` or `serial_chunked`",
        "required only for `serial_chunked`",
        "numpy zyx order",
        "`0 <= value < 1`",
        "`linear` or `nearestneighbor`",
        "safe relative posix path",
        "registration channel must be listed",
        "moving-channel output basenames must be unique within each round",
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


def test_execution_profile_reference_documents_every_field_and_rule() -> None:
    reference = _document("execution-profile.md")
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
        assert f"`{field}`" in reference

    for exact_row in (
        "| `local_dask` / `workers` | positive built-in integer | Maximum local "
        "worker processes. |",
        "| `local_dask` / `startup_timeout` | non-empty single-line string | "
        "Bounds only Dask `Client` connection and `wait_for_workers`. |",
        "| `local_dask` / `task_retries` | nonnegative built-in integer | "
        "Executor-managed retry count after the first attempt. |",
        "| `slurm` / `workers` | positive built-in integer | Maximum number of "
        "worker jobs/nodes, not Dask process slots. |",
        "| `slurm` / `processes_per_worker` | positive built-in integer | "
        "Processes per job: coarse requires `1`; fine divides job cores and "
        "memory across the processes. |",
        "| `slurm` / `memory` | positive whole decimal `KB`, `MB`, `GB`, or `TB` "
        "string | Semantic memory request; the executor rounds up to whole binary "
        "GiB for SLURM and Dask. |",
        "| `slurm` / `startup_timeout` | non-empty single-line string | Bounds "
        "initial Dask connection/worker waiting and the absolute active fine-SLURM "
        "replacement/reconciliation deadline. |",
        "| `slurm` / `task_retries` | nonnegative built-in integer | "
        "Executor-managed retry count after the first attempt. |",
    ):
        assert exact_row in reference

    normalized = _normalized_document("execution-profile.md")
    for rule in (
        "`sequential`, `local_dask`, or `slurm`",
        "relative paths resolve from the execution-profile file",
        "dataset and output roots must be disjoint",
        "worker-backed executors require",
        "must not be `/archive`",
        "inherited environment",
        "one-thread default",
        "`processes` must be `true`",
        "coarse mode requires `processes_per_worker: 1`",
        "fine mode requires exact core divisibility",
        "native thread count must not exceed the cores per process",
        "job cap and total process-slot capacity",
        "per-process dask memory",
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


def test_execution_profile_documents_the_exact_startup_timeout_scope() -> None:
    normalized = _normalized_document("execution-profile.md")
    for required in (
        "`startup_timeout` is not an end-to-end startup deadline",
        "initial dask `client` connection and `wait_for_workers()`",
        "initial `localcluster` or `slurmcluster` construction",
        "initial worker snapshots through `scheduler_info()`",
        "initial `client.run()` worker-preflight rpc",
        "no cy-exm wall-clock timeout",
        "active fine-slurm replacement and reconciliation",
        "one absolute deadline",
        "worker health and replacement snapshots",
        "replacement worker arrival",
        "replacement worker probe broadcast",
        "resource validation",
        "whole-group activation and final receipt verification",
        "does not reset between steps",
    ):
        assert required in normalized

    hpc = _normalized_document("parallel-hpc.md")
    for required in (
        "`slurm.startup_timeout`",
        "one absolute replacement/reconciliation deadline",
        "initial cluster construction, scheduler snapshots, and worker-preflight rpc",
        "no cy-exm wall-clock timeout",
    ):
        assert required in hpc


def test_native_threading_documents_planning_and_worker_boundaries() -> None:
    normalized = _normalized_document("native-threading.md")
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


def test_native_threading_documents_observed_scaling() -> None:
    text = _normalized_document("native-threading.md")
    for required in (
        "| 1 / 1 | 310.5 s | 1.00x |",
        "| 8 / 8 | 88.3 s | 3.52x |",
        "| 30 / 30 | 69.5 s | 4.47x |",
        "scientifically consistent but not bit-for-bit identical",
        "largest nonlinear-correlation change was 0.003416",
        "one thread remains the conservative default",
        "full-volume stability evidence",
    ):
        assert required in text


@pytest.mark.parametrize(
    ("name", "required"),
    (
        (
            "image-geometry.md",
            (
                "micrometers",
                "(x, y, z)",
                "(z, y, x)",
                "fixed-reference grid",
                "scalar 3-d tiff",
                "origin",
                "spacing",
                "direction",
            ),
        ),
        (
            "preprocessing.md",
            (
                "raw -> crop -> feature estimate -> transform raw",
                "filter_sigma_zyx",
                "voxel",
                "feature-derived registration mask",
                "raw-intensity",
                "dataset-specific",
            ),
        ),
        (
            "outputs-and-recovery.md",
            (
                "resume",
                "force",
                "pending",
                "partial",
                "complete",
                "invalid",
                "diagnostic image",
                "transformed channel",
                ".complete.json",
            ),
        ),
    ),
)
def test_scientific_user_guides_preserve_public_contracts(
    name: str,
    required: tuple[str, ...],
) -> None:
    text = _normalized_document(name)
    for fragment in required:
        assert fragment in text
    assert "phase 1" not in text
    assert "phase 2" not in text


def test_output_and_artifact_guides_document_lifecycle_and_status() -> None:
    text = _document("outputs-and-recovery.md") + _document("registration-artifacts.md")
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
        assert output in text

    normalized = " ".join(text.split()).lower()
    for required in (
        "resume is the default",
        "read-only status",
        "`pending`, `partial`, `complete`, or `invalid`",
        "scientific fingerprint",
        "force cannot bypass",
        "requires a new output root",
        "operational summaries never authorize artifact reuse",
        "forcing the reference round invalidates all moving descendants",
        "normalized registration diagnostic",
        "original channel resampled through the persisted transform chain",
        "clipped to uint16 without intensity normalization",
    ):
        assert required in normalized


def test_cli_reference_documents_success_only_json_output() -> None:
    normalized = _normalized_document("cli-reference.md")
    for required in (
        "`--json` is success-only",
        "failures return before result emission",
        "stdout is empty",
        "plain-text diagnostics go to stderr",
        "no json error object",
        "`--version` prints version text rather than a json result",
    ):
        assert required in normalized


def test_cli_reference_links_to_an_existing_operational_guide() -> None:
    reference = _document("cli-reference.md")
    target = "parallel-hpc.md"

    assert f"]({target})" in reference
    assert (DOCS_ROOT / target).is_file()
    assert "general executor startup, preflight, cancellation, cleanup" in reference
    assert "](fine-grained-execution.md" in reference


def test_workflow_and_cli_document_reference_round_forcing() -> None:
    for name in ("workflow-guide.md", "cli-reference.md"):
        guide = _normalized_document(name)
        for required in (
            "selected rounds, including the reference round",
            "forcing the reference round propagates to every moving descendant",
        ):
            assert required in guide, name


def test_parallel_hpc_guide_documents_executor_lifecycle() -> None:
    text = _document("parallel-hpc.md") + _document("cli-reference.md")
    normalized = " ".join(text.split()).lower()
    for required in (
        "sequential executor is the numerical and restart reference",
        "process-based `localcluster`",
        "min(pending tasks, configured workers)",
        "one `round_slot`",
        "coarse mode uses one process per job",
        "fine mode can run multiple process-isolated workers per job",
        "scales in whole jobs",
        "complete process group",
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
        assert exact_row in text

    for required in (
        "invalid fine execution evidence",
        "unauthenticated current-run summaries are omitted from durable `rounds`",
        "pre-run authenticated reuse remains",
        "valid committed artifacts before rejection remain restartable",
    ):
        assert required in normalized


def test_parallel_hpc_guide_states_acceptance_boundaries() -> None:
    text = _normalized_document("parallel-hpc.md")
    for required in (
        "portable fake-based slurm construction",
        "real slurm submission and hpc acceptance completed on the audited lab "
        "partition",
        "`-m hpc`",
        "explicit allocation",
        "fine-grained real-slurm production acceptance remains pending",
        "portable real-ants/local-dask evidence is complete",
        "focused multi-process real-slurm acceptance remains pending",
        "no silent fallback",
    ):
        assert required in text


def test_fine_grained_guide_documents_the_exact_graph_and_resource_contract() -> None:
    guide_path = REPOSITORY_ROOT / "docs/fine-grained-execution.md"
    assert guide_path.is_file()
    guide = " ".join(guide_path.read_text(encoding="utf-8").split()).lower()

    for required in (
        "fine-grained task admission",
        "choosing a registration strategy",
        "local parallel and hpc execution",
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
        "one candidate job",
        "preflight",
        "`round_slot`",
        "demand-driven",
        "round-robin",
        "configured worker cap",
        "`slurm.workers` is a job cap",
        "`processes_per_worker` is a process count per job",
        "six task slots",
        "466 gib",
        "233-gib",
        "scale-up proceeds in whole jobs",
        "shared slurm job id",
        "distinct worker addresses and pids",
        "workers never create schedulers",
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
        "accepted ants behavior",
        "portable real-ants/local-dask evidence is complete",
        "real slurm production acceptance remains pending",
        "schema version 3",
        "lost process",
        "lost whole job",
        "partial process group",
        "sibling process",
        "replacement timeout",
        "artifactstore remains the only restart authority",
        "performance is characterization",
        "`format_version: 2` when preprocessing is disabled",
        "`format_version: 3` for unmasked asinh features",
        "`format_version: 4` for feature-derived masks",
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
    assert "clearex" not in guide
    assert "phase 1" not in guide
    assert "phase 2" not in guide


def test_registration_artifact_inventories_name_their_execution_surface() -> None:
    guide = _normalized_document("registration-artifacts.md")
    for required in (
        "standalone direct-python",
        "`imageregistration`",
        "`chunkedimageregistration`",
        "yaml workflow layout",
        "`rounds/round-002/registration/`",
        "`reference_crop_indices.json`",
        "[outputs, restart, and recovery](outputs-and-recovery.md)",
    ):
        assert required in guide


def test_maintained_docs_distinguish_shared_cores_and_rejected_evidence() -> None:
    artifacts = _normalized_document("registration-artifacts.md")
    guide = _normalized_document("fine-grained-execution.md")

    for text in (artifacts, guide):
        assert "scalar cores are shared coarse/fine serial-chunk artifacts" in text
        assert "prepared arrays are fine-only" in text
    for required in (
        "unauthenticated current-run summaries are omitted from durable `rounds`",
        "pre-run authenticated reuse remains",
        "invalid fine execution evidence",
        "sequential, local dask, and slurm",
    ):
        assert required in artifacts
    for stale in (
        "coarse serial execution, registered scalar chunk data is reconstructed",
        "fine-only prepared arrays and scalar cores",
        "it preserves the round results in the operational run record",
    ):
        assert stale not in f"{artifacts} {guide}"


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
