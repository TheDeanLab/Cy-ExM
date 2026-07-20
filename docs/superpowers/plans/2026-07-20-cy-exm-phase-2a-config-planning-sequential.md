# Cy-ExM Phase 2A Configuration and Sequential Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strict versioned YAML, immutable workflow planning, read-only status, a scheduler-neutral sequential executor, public workflow APIs, and the `cyexm workflow` CLI without importing scientific or scheduler runtimes at module scope.

**Architecture:** `cyexm.config` decodes two independent YAML documents into frozen values. `cyexm.workflow` resolves safe paths, inspects input headers lazily, fingerprints canonical scientific choices, and creates immutable tasks. `cyexm.execution` owns an injected task-runner protocol and sequential adapter; the CLI delegates to the public workflow API.

**Tech Stack:** Python 3.12, PyYAML 6, frozen/slotted dataclasses, hashlib/json, pathlib, argparse, tifffile header inspection, pytest, uv 0.7.13

## Global Constraints

- Follow the roadmap constraints in `docs/superpowers/plans/2026-07-20-cy-exm-phase-2-roadmap.md`.
- Add optional extras exactly as cumulative groups: `workflow` has `PyYAML>=6,<7`; `distributed` additionally pins `dask==2025.1.0` and `distributed==2025.1.0`; `hpc` additionally pins `dask-jobqueue==0.9.0`.
- Do not add any Phase 2 dependency to base `project.dependencies`.
- The strict loader rejects duplicate keys, aliases, custom tags, non-scalar keys, and additional YAML documents.
- Planning creates no files, directories, clients, clusters, or workers.
- Scientific and execution roots are disjoint; neither may contain the other.
- Force never bypasses a scientific-plan fingerprint mismatch.
- Do not add a real registration task runner in this milestone; the complete runner is Phase 2B. Test the executor and API by injected callable.

---

## File and responsibility map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | Optional dependency groups and `cyexm` console entry point |
| `uv.lock` | Intentionally regenerated lock for approved optional dependencies |
| `src/cyexm/errors.py` | Configuration, plan, executor, round, and aggregate workflow exceptions |
| `src/cyexm/config/AGENTS.md` | Strict data-only configuration rules |
| `src/cyexm/config/_yaml.py` | Safe one-document YAML mapping loader |
| `src/cyexm/config/_validation.py` | Exact-key, built-in-type, path, number, and duration validation helpers |
| `src/cyexm/config/models.py` | Frozen scientific and execution values |
| `src/cyexm/config/dataset.py` | Scientific manifest conversion |
| `src/cyexm/config/execution.py` | Execution profile conversion and path binding |
| `src/cyexm/config/__init__.py` | Lightweight public configuration exports |
| `src/cyexm/workflow/AGENTS.md` | Planning, fingerprint, output, and restart ownership rules |
| `src/cyexm/workflow/models.py` | Immutable plan/task/result/status values |
| `src/cyexm/workflow/layout.py` | Collision-free output paths and canonical JSON |
| `src/cyexm/workflow/planning.py` | Input/header validation and deterministic plan construction |
| `src/cyexm/workflow/status.py` | Read-only completion-record inspection |
| `src/cyexm/workflow/api.py` | Public load/build/run/status orchestration |
| `src/cyexm/workflow/__init__.py` | Public workflow exports |
| `src/cyexm/execution/AGENTS.md` | Scheduler lifecycle and task-resource boundaries |
| `src/cyexm/execution/base.py` | `WorkflowExecutor` and task-runner protocols |
| `src/cyexm/execution/sequential.py` | Canonical in-process execution |
| `src/cyexm/execution/__init__.py` | Lightweight executor exports |
| `src/cyexm/cli/AGENTS.md` | CLI output, exit-code, and side-effect rules |
| `src/cyexm/cli/_parser.py` | Argument parser construction |
| `src/cyexm/cli/_commands.py` | API delegation and human/JSON formatting |
| `src/cyexm/cli/__init__.py` | `main(argv) -> int` and exception-to-exit mapping |
| `src/cyexm/cli/__main__.py` | `python -m cyexm.cli` entry point |
| `tests/unit/test_config_yaml.py` | Adversarial loader coverage |
| `tests/unit/test_config_dataset.py` | Scientific schema coverage |
| `tests/unit/test_config_execution.py` | Machine schema coverage |
| `tests/unit/test_workflow_models.py` | Immutability and serialization coverage |
| `tests/unit/test_workflow_planning.py` | Planning, path, header, order, and fingerprint coverage |
| `tests/unit/test_workflow_status.py` | Read-only status coverage |
| `tests/unit/test_execution_sequential.py` | Executor ordering and independent failure coverage |
| `tests/unit/test_workflow_api.py` | Public orchestration and force coverage |
| `tests/unit/test_cli.py` | Commands, JSON, stderr, and exit codes |
| `tests/unit/test_import_contract.py` | Lightweight Phase 2 imports |
| `tests/unit/test_repository_policy.py` | New AGENTS inventory and dependency boundaries |

---

### Task 1: Declare optional dependencies and guided package boundaries

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/cyexm/config/AGENTS.md`
- Create: `src/cyexm/workflow/AGENTS.md`
- Create: `src/cyexm/execution/AGENTS.md`
- Create: `src/cyexm/cli/AGENTS.md`
- Modify: `tests/unit/test_repository_policy.py`

**Interfaces:**
- Consumes: existing distribution metadata and `AGENT_GUIDANCE_PATHS`
- Produces: install extras `workflow`, `distributed`, and `hpc`; console command `cyexm`; enforced guidance for every new source directory

- [ ] **Step 1: Add failing dependency and guidance inventory tests**

Add assertions that parse `pyproject.toml` and require these exact optional lists:

```python
def test_phase_two_optional_dependencies_are_cumulative_and_pinned() -> None:
    configuration = tomllib.loads(_read_repository_file("pyproject.toml"))
    extras = configuration["project"]["optional-dependencies"]
    assert extras["workflow"] == ["PyYAML>=6,<7"]
    assert extras["distributed"] == [
        "PyYAML>=6,<7",
        "dask==2025.1.0",
        "distributed==2025.1.0",
    ]
    assert extras["hpc"] == [
        "PyYAML>=6,<7",
        "dask==2025.1.0",
        "distributed==2025.1.0",
        "dask-jobqueue==0.9.0",
    ]
    assert configuration["project"]["scripts"] == {"cyexm": "cyexm.cli:main"}

    direct = "\n".join(configuration["project"]["dependencies"]).lower()
    for forbidden in ("pyyaml", "dask", "distributed", "dask-jobqueue"):
        assert forbidden not in direct
```

Extend `AGENT_GUIDANCE_PATHS` with the four new `AGENTS.md` paths and assert their normalized text contains the package responsibility and forbidden imports from the file map.

- [ ] **Step 2: Run the focused tests and observe missing extras and guidance**

```bash
uv run pytest \
  tests/unit/test_repository_policy.py::test_phase_two_optional_dependencies_are_cumulative_and_pinned \
  tests/unit/test_repository_policy.py::test_every_maintained_directory_has_local_agent_guidance -q
```

Expected: both tests fail because the extras, console entry, and new instruction files do not exist.

- [ ] **Step 3: Add the dependency metadata and regenerate the lock deliberately**

Add this metadata:

```toml
[project.scripts]
cyexm = "cyexm.cli:main"

[project.optional-dependencies]
examples = ["jupyterlab>=4", "matplotlib>=3.8", "nbconvert>=7"]
workflow = ["PyYAML>=6,<7"]
distributed = [
  "PyYAML>=6,<7",
  "dask==2025.1.0",
  "distributed==2025.1.0",
]
hpc = [
  "PyYAML>=6,<7",
  "dask==2025.1.0",
  "distributed==2025.1.0",
  "dask-jobqueue==0.9.0",
]
```

Create the four guidance files with the exact package boundaries in the file map, then run:

```bash
uv lock
uv lock --check
uv sync --locked --group dev --extra examples --extra hpc
```

Inspect `git diff -- pyproject.toml uv.lock` and verify the only new package families are PyYAML, Dask/distributed, dask-jobqueue, and their transitive dependencies.

- [ ] **Step 4: Run the focused tests and dependency audit**

```bash
uv run pytest \
  tests/unit/test_repository_policy.py::test_phase_two_optional_dependencies_are_cumulative_and_pinned \
  tests/unit/test_repository_policy.py::test_every_maintained_directory_has_local_agent_guidance -q
uv tree --package cy-exm
```

Expected: `2 passed`; the base dependency list remains unchanged and all three optional groups resolve.

- [ ] **Step 5: Commit the approved dependency surface**

```bash
git add pyproject.toml uv.lock src/cyexm/config/AGENTS.md \
  src/cyexm/workflow/AGENTS.md src/cyexm/execution/AGENTS.md \
  src/cyexm/cli/AGENTS.md tests/unit/test_repository_policy.py
git commit -m "build: declare Phase 2 workflow extras"
```

---

### Task 2: Build the strict one-document YAML boundary

**Files:**
- Create: `src/cyexm/config/_yaml.py`
- Create: `src/cyexm/config/_validation.py`
- Create: `tests/unit/test_config_yaml.py`

**Interfaces:**
- Consumes: `yaml.SafeLoader`
- Produces: `load_yaml_mapping(path: str | Path) -> Mapping[str, object]`; reusable exact-key and built-in-type validators

- [ ] **Step 1: Write adversarial loader tests**

```python
@pytest.mark.parametrize(
    "content, message",
    [
        ("schema_version: 1\nschema_version: 1\n", "duplicate key"),
        ("value: &shared 1\ncopy: *shared\n", "aliases are not allowed"),
        ("value: !python/object:thing {}\n", "custom YAML tags"),
        ("? [one, two]\n: value\n", "mapping keys must be scalar strings"),
        ("schema_version: 1\n---\nschema_version: 1\n", "one YAML document"),
    ],
)
def test_strict_yaml_rejects_ambiguous_documents(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ManifestDecodeError, match=message):
        load_yaml_mapping(path)
```

Also test a missing file, an empty document, a top-level sequence, and a valid nested mapping of scalar/list values.

- [ ] **Step 2: Run the loader tests and observe the missing module**

```bash
uv run pytest tests/unit/test_config_yaml.py -q
```

Expected: collection fails because `cyexm.config` is absent.

- [ ] **Step 3: Add focused configuration exceptions**

Append this hierarchy to `src/cyexm/errors.py`:

```python
class ManifestDecodeError(CyExMError):
    """A YAML document cannot be decoded safely."""


class ConfigurationValidationError(CyExMError):
    """A scientific manifest or execution profile violates its schema."""


class WorkflowPlanError(CyExMError):
    """Resolved workflow inputs or outputs cannot form a safe plan."""


class ExecutorStartupError(CyExMError):
    """The requested execution backend did not become usable."""


class WorkerPreflightError(ExecutorStartupError):
    """One or more workers failed environment preflight."""


class RoundExecutionError(CyExMError):
    """A round task failed after execution began."""


class WorkflowExecutionError(CyExMError):
    """One or more submitted round tasks failed."""

    def __init__(self, failed_rounds: tuple[int, ...], run_record: str) -> None:
        self.failed_rounds = failed_rounds
        self.run_record = run_record
        rounds = ", ".join(str(value) for value in failed_rounds)
        super().__init__(f"Workflow rounds failed: {rounds}; run record: {run_record}")
```

- [ ] **Step 4: Implement the strict loader and shared validators**

Use a private `SafeLoader` subclass. Override `compose_node` to reject `AliasEvent`, override the default mapping constructor to reject duplicate or non-string scalar keys, call `get_single_data()` so a second document fails, and dispose the loader in `finally`. Wrap all `OSError`, `yaml.YAMLError`, and contract failures in `ManifestDecodeError` containing the source path.

The validation module must expose these exact helpers:

```python
def require_exact_keys(
    value: object, expected: frozenset[str], yaml_path: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise ConfigurationValidationError(f"{yaml_path} must be a mapping")
    actual = frozenset(value)
    if actual != expected:
        raise ConfigurationValidationError(
            f"{yaml_path} keys must be exactly {sorted(expected)!r}; got {sorted(actual)!r}"
        )
    return value

def built_in_int(value: object, yaml_path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationValidationError(
            f"{yaml_path} must be a built-in integer >= {minimum}; got {value!r}"
        )
    return value

def finite_positive_float(value: object, yaml_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationValidationError(
            f"{yaml_path} must be a positive finite number; got {value!r}"
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConfigurationValidationError(
            f"{yaml_path} must be a positive finite number; got {value!r}"
        )
    return result

def nonempty_string(value: object, yaml_path: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise ConfigurationValidationError(
            f"{yaml_path} must be a nonempty single-line string; got {value!r}"
        )
    return value

def relative_posix_path(value: object, yaml_path: str) -> PurePosixPath:
    text = nonempty_string(value, yaml_path)
    path = PurePosixPath(text)
    if path.is_absolute() or path == PurePosixPath(".") or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ConfigurationValidationError(
            f"{yaml_path} must be a safe relative POSIX path; got {value!r}"
        )
    return path
```

Each helper raises `ConfigurationValidationError` with the YAML path, offending representation, and expected contract. `built_in_int` uses `type(value) is int`; `finite_positive_float` rejects booleans and non-finite values; `relative_posix_path` rejects absolute paths, `..`, NUL, and empty/dot components.

- [ ] **Step 5: Run the strict-loader tests**

```bash
uv run pytest tests/unit/test_config_yaml.py -q
```

Expected: all loader and primitive-validation tests pass.

- [ ] **Step 6: Commit the safe decoding boundary**

```bash
git add src/cyexm/errors.py src/cyexm/config/_yaml.py \
  src/cyexm/config/_validation.py tests/unit/test_config_yaml.py
git commit -m "feat: add strict workflow YAML decoding"
```

---

### Task 3: Parse the immutable scientific dataset manifest

**Files:**
- Create: `src/cyexm/config/models.py`
- Create: `src/cyexm/config/dataset.py`
- Create: `src/cyexm/config/__init__.py`
- Create: `tests/unit/test_config_dataset.py`

**Interfaces:**
- Consumes: `load_yaml_mapping` and shared validators
- Produces: `ChannelSpec`, `RoundSpec`, `ReferenceSpec`, `RegistrationStageSpec`, `RegistrationSpec`, `DatasetManifest`, and `load_dataset_manifest(path) -> DatasetManifest`

- [ ] **Step 1: Write a valid scientific-manifest test**

```python
def test_load_dataset_manifest_normalizes_xyz_and_zyx_fields(tmp_path: Path) -> None:
    path = tmp_path / "dataset.yaml"
    path.write_text(VALID_DATASET_YAML, encoding="utf-8")
    manifest = load_dataset_manifest(path)
    assert manifest.schema_version == 1
    assert manifest.dataset_id == "dataset-1"
    assert manifest.voxel_spacing_um == (0.324, 0.324, 0.9)
    assert manifest.reference == ReferenceSpec(
        round=1, channel=PurePosixPath("round_1_488.tif")
    )
    assert tuple(item.round for item in manifest.rounds) == (2, 3)
    assert manifest.rounds[1].chunk_size_zyx == (8, 16, 16)
    with pytest.raises(FrozenInstanceError):
        manifest.dataset_id = "changed"  # type: ignore[misc]
```

Define `VALID_DATASET_YAML` in the test with one `whole_image` round and one `serial_chunked` round, explicit channel interpolation, TRSAA/high linear settings, and SyNOnly/high nonlinear settings.

- [ ] **Step 2: Add schema-matrix tests and run red**

Parametrize focused cases for schema version/type, dataset ID regex, XYZ finite positive spacing, reference/moving built-in non-negative rounds, duplicate rounds, reference-round reuse, registration-channel membership, duplicate channel files, supported strategy/interpolation, chunk fields forbidden for whole-image, chunk fields required for serial-chunked, positive ZYX sizes, `0 <= overlap_fraction < 1`, boolean `crop`, and exact Phase 1 method/accuracy choices.

```bash
uv run pytest tests/unit/test_config_dataset.py -q
```

Expected: collection fails because the scientific models do not exist.

- [ ] **Step 3: Add the frozen scientific model declarations**

```python
type PixelSpacingXYZ = tuple[float, float, float]
type ChunkSizeZYX = tuple[int, int, int]
type RegistrationStrategy = Literal["whole_image", "serial_chunked"]
type ChannelInterpolation = Literal["linear", "nearestNeighbor"]


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    file: PurePosixPath
    interpolation: ChannelInterpolation


@dataclass(frozen=True, slots=True)
class RoundSpec:
    round: int
    strategy: RegistrationStrategy
    registration_channel: PurePosixPath
    channels: tuple[ChannelSpec, ...]
    chunk_size_zyx: ChunkSizeZYX | None = None
    overlap_fraction: float | None = None


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
    round: int
    channel: PurePosixPath


@dataclass(frozen=True, slots=True)
class RegistrationStageSpec:
    type: str
    accuracy: str


@dataclass(frozen=True, slots=True)
class RegistrationSpec:
    crop: bool
    linear: RegistrationStageSpec
    nonlinear: RegistrationStageSpec


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: int
    dataset_id: str
    voxel_spacing_um: PixelSpacingXYZ
    reference: ReferenceSpec
    registration: RegistrationSpec
    rounds: tuple[RoundSpec, ...]
```

- [ ] **Step 4: Implement exact scientific conversion**

In `dataset.py`, require exact keys at every mapping layer. Convert YAML list order to tuples without sorting; validate cross-round uniqueness after individual round parsing. Export only the six public models and `load_dataset_manifest` from `cyexm.config`.

Use these fixed accepted values:

```python
_LINEAR_TYPES = frozenset({"Translation", "Rigid", "Similarity", "Affine", "TRSAA"})
_NONLINEAR_TYPES = frozenset({"Elastic", "SyNOnly", "SyN"})
_ACCURACIES = frozenset({"high", "medium", "low", "dry run"})
_STRATEGIES = frozenset({"whole_image", "serial_chunked"})
_INTERPOLATIONS = frozenset({"linear", "nearestNeighbor"})
```

- [ ] **Step 5: Run scientific schema tests**

```bash
uv run pytest tests/unit/test_config_dataset.py -q
```

Expected: all valid, immutable, exact-key, type, value, and cross-field cases pass.

- [ ] **Step 6: Commit the scientific manifest contract**

```bash
git add src/cyexm/config/models.py src/cyexm/config/dataset.py \
  src/cyexm/config/__init__.py tests/unit/test_config_dataset.py
git commit -m "feat: parse scientific dataset manifests"
```

---

### Task 4: Parse and bind immutable execution profiles

**Files:**
- Modify: `src/cyexm/config/models.py`
- Create: `src/cyexm/config/execution.py`
- Modify: `src/cyexm/config/__init__.py`
- Create: `tests/unit/test_config_execution.py`

**Interfaces:**
- Consumes: strict YAML and shared validators
- Produces: `ExecutionPaths`, `NativeThreadSpec`, `LocalDaskSpec`, `SlurmSpec`, `ExecutionProfile`, and `load_execution_profile(path) -> ExecutionProfile`

- [ ] **Step 1: Write valid profile tests for all three executors**

```python
def test_execution_paths_resolve_relative_to_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "profiles" / "sequential.yaml"
    profile_path.parent.mkdir()
    profile_path.write_text(SEQUENTIAL_YAML, encoding="utf-8")
    profile = load_execution_profile(profile_path)
    assert profile.executor == "sequential"
    assert profile.paths.dataset_root == (profile_path.parent / "../data").resolve()
    assert profile.paths.output_root == (profile_path.parent / "../results").resolve()
    assert profile.native_threads == NativeThreadSpec(itk=1, blas=1)
    assert profile.local_dask is None
    assert profile.slurm is None
```

Add equivalent exact-value assertions for `LocalDaskSpec` and the approved 512-GB `SlurmSpec`.

- [ ] **Step 2: Add profile schema failures and run red**

Cover mismatched executor sections, unknown keys, equal/containing roots, `/archive` worker scratch, built-in integer enforcement, process count exactly one for SLURM, nonempty single-line strings, timeout strings, bounded retry values, conflicting/newline/NUL job directives, omitted thread defaults, equal inherited BLAS values, and rejection of unequal inherited BLAS values that cannot be represented by the profile's one uniform `blas` count.

```bash
uv run pytest tests/unit/test_config_execution.py -q
```

Expected: collection fails because execution models are missing.

- [ ] **Step 3: Add frozen execution models**

```python
type ExecutorName = Literal["sequential", "local_dask", "slurm"]


@dataclass(frozen=True, slots=True)
class ExecutionPaths:
    dataset_root: Path
    output_root: Path
    worker_local_directory: Path | None


@dataclass(frozen=True, slots=True)
class NativeThreadSpec:
    itk: int
    blas: int


@dataclass(frozen=True, slots=True)
class LocalDaskSpec:
    workers: int
    threads_per_worker: int
    processes: bool
    memory_limit: str
    startup_timeout: str
    task_retries: int


@dataclass(frozen=True, slots=True)
class SlurmSpec:
    queue: str
    workers: int
    cores_per_worker: int
    processes_per_worker: int
    memory: str
    walltime: str
    interface: str
    scheduler_interface: str
    startup_timeout: str
    task_retries: int
    job_name: str
    job_extra_directives: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    schema_version: int
    executor: ExecutorName
    paths: ExecutionPaths
    native_threads: NativeThreadSpec
    local_dask: LocalDaskSpec | None
    slurm: SlurmSpec | None
    source_path: Path
```

- [ ] **Step 4: Implement exact execution conversion and path containment checks**

Resolve each path with `Path.resolve(strict=False)` relative to `source_path.parent`. Use `Path.is_relative_to` in both directions to reject root containment. Accept a missing `native_threads` mapping by resolving inherited/default values through a pure helper that reads a supplied mapping in tests and `os.environ` only inside `load_execution_profile`; do not mutate the environment. The ITK value follows its one environment variable. Resolve all five BLAS-family variables independently, require their effective inherited/default values to agree, and store that uniform value. If they differ, raise `ConfigurationValidationError` instructing the operator to set explicit `native_threads.blas`; this preserves the two-field YAML and the existing `NativeThreadConfig` contract without silently changing per-library counts.

Reject extra directives whose normalized option begins with any of:

```python
_MANAGED_SLURM_OPTIONS = (
    "--partition",
    "-p",
    "--mem",
    "--mem-per-cpu",
    "--cpus-per-task",
    "-c",
    "--ntasks",
    "-n",
    "--output",
    "-o",
    "--error",
    "-e",
)
```

- [ ] **Step 5: Run profile tests**

```bash
uv run pytest tests/unit/test_config_execution.py -q
```

Expected: all three valid profiles and all exact schema/path/resource failures pass.

- [ ] **Step 6: Commit the execution-profile contract**

```bash
git add src/cyexm/config/models.py src/cyexm/config/execution.py \
  src/cyexm/config/__init__.py tests/unit/test_config_execution.py
git commit -m "feat: parse workflow execution profiles"
```

---

### Task 5: Define workflow values, output layout, and canonical fingerprints

**Files:**
- Create: `src/cyexm/workflow/models.py`
- Create: `src/cyexm/workflow/layout.py`
- Create: `tests/unit/test_workflow_models.py`

**Interfaces:**
- Consumes: configuration models and Phase 1 `ImageGeometry`
- Produces: immutable `ImageHeader`, `ChannelTask`, `RoundTask`, `RoundFailure`, `RoundResult`, `WorkflowPlan`, `WorkflowResult`, `RoundStatus`, `WorkflowStatus`; `WorkflowLayout`; canonical scientific JSON and SHA-256 fingerprint

- [ ] **Step 1: Write model and layout tests**

```python
def test_layout_isolated_each_round_and_channel_output(tmp_path: Path) -> None:
    layout = WorkflowLayout.for_dataset(tmp_path, "dataset-1")
    round_layout = layout.round(2)
    assert layout.plan_path == tmp_path / "dataset-1" / "scientific-plan.json"
    assert round_layout.registration == tmp_path / "dataset-1/rounds/round-002/registration"
    assert round_layout.channels == tmp_path / "dataset-1/rounds/round-002/channels"
    assert round_layout.log == tmp_path / "dataset-1/rounds/round-002/logs/round-002.log"
```

Test frozen mutation failure, Dask-pickle round trip, JSON serialization of every result/status, stable round sorting, manifest channel order, and fingerprint equality across sequential/local/SLURM profiles.

- [ ] **Step 2: Run model tests and observe missing workflow package**

```bash
uv run pytest tests/unit/test_workflow_models.py -q
```

Expected: collection fails because `cyexm.workflow.models` is absent.

- [ ] **Step 3: Add exact workflow models**

Define all models as frozen/slotted dataclasses. `RoundTask` contains only primitives, tuples, `Path`, and immutable config models:

```python
@dataclass(frozen=True, slots=True)
class ChannelTask:
    source: Path
    output: Path
    interpolation: ChannelInterpolation


@dataclass(frozen=True, slots=True)
class RoundTask:
    dataset_id: str
    round: int
    strategy: RegistrationStrategy
    fixed_image: Path
    moving_image: Path
    registration_directory: Path
    log_path: Path
    voxel_spacing_xyz: PixelSpacingXYZ
    registration: RegistrationSpec
    native_threads: NativeThreadSpec
    channels: tuple[ChannelTask, ...]
    chunk_size_zyx: ChunkSizeZYX | None
    overlap_fraction: float | None
    force: bool = False


@dataclass(frozen=True, slots=True)
class ImageHeader:
    path: Path
    shape_zyx: tuple[int, int, int]
    dtype: str
    axes: str
    geometry: ImageGeometry | None
    st_size: int
    st_mtime_ns: int


@dataclass(frozen=True, slots=True)
class RoundFailure:
    stage: str
    exception_type: str
    message: str


@dataclass(frozen=True, slots=True)
class RoundResult:
    dataset_id: str
    round: int
    status: Literal["reused", "completed", "failed", "cancelled"]
    started_at: str
    ended_at: str
    duration_seconds: float
    hostname: str
    pid: int
    dask_task_key: str | None
    slurm_job_id: str | None
    strategy: RegistrationStrategy
    native_threads: tuple[tuple[str, int], ...]
    artifact_paths: tuple[str, ...]
    reused_artifacts: tuple[str, ...]
    log_path: str
    failure: RoundFailure | None


@dataclass(frozen=True, slots=True)
class WorkflowPlan:
    schema_version: int
    dataset_id: str
    manifest: DatasetManifest
    execution: ExecutionProfile
    fixed_image: Path
    tasks: tuple[RoundTask, ...]
    expected_artifacts: tuple[Path, ...]
    scientific_json: str
    scientific_fingerprint: str


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    dataset_id: str
    scientific_fingerprint: str
    status: Literal["completed", "failed", "cancelled"]
    started_at: str
    ended_at: str
    rounds: tuple[RoundResult, ...]
    run_record: str
    execution_evidence_json: str = "{}"

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"


@dataclass(frozen=True, slots=True)
class RoundStatus:
    round: int
    status: Literal["pending", "partial", "complete", "invalid"]
    complete_artifacts: tuple[str, ...]
    incomplete_artifacts: tuple[str, ...]
    invalid_artifacts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkflowStatus:
    dataset_id: str
    scientific_fingerprint: str | None
    rounds: tuple[RoundStatus, ...]
    complete: bool
```

Every result/status implements `to_dict()` using JSON-compatible scalar/list/object values. `scientific_json` and `execution_evidence_json` are already canonical JSON strings, which keeps the dataclasses deeply immutable while allowing API/CLI serialization. Do not include clients, clusters, loaded images, callables, loggers, mutable mappings, or handles.

- [ ] **Step 4: Implement layout and canonical JSON helpers**

Use `json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))` and `hashlib.sha256(encoded.encode("utf-8")).hexdigest()`. The scientific mapping includes schema, dataset ID, resolved source file identities/header evidence, XYZ spacing, registration policy, strategies, channel interpolation, and output-format/postprocessing policy versions. It excludes executor/resources/thread counts/logging/scheduler values.

- [ ] **Step 5: Run model and fingerprint tests**

```bash
uv run pytest tests/unit/test_workflow_models.py -q
```

Expected: all immutable, layout, JSON, pickling, ordering, and fingerprint tests pass.

- [ ] **Step 6: Commit workflow values and layout**

```bash
git add src/cyexm/workflow/models.py src/cyexm/workflow/layout.py \
  tests/unit/test_workflow_models.py
git commit -m "feat: define immutable workflow plans"
```

---

### Task 6: Build the deterministic mutation-free planner

**Files:**
- Create: `src/cyexm/workflow/planning.py`
- Create: `tests/unit/test_workflow_planning.py`

**Interfaces:**
- Consumes: `DatasetManifest`, `ExecutionProfile`, `WorkflowLayout`, Phase 1 strict TIFF geometry/header helpers
- Produces: `build_workflow_plan(manifest, execution) -> WorkflowPlan`

- [ ] **Step 1: Write a valid planner test with a patched header boundary**

```python
def test_plan_resolves_inputs_orders_rounds_and_creates_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = dataset_manifest(round_order=(3, 2))
    execution = sequential_profile(tmp_path)
    monkeypatch.setattr(
        "cyexm.workflow.planning._inspect_tiff_header",
        lambda path: ImageHeader(
            path=path,
            shape_zyx=(4, 8, 8),
            dtype="uint16",
            axes="ZYX",
            geometry=None,
            st_size=1024,
            st_mtime_ns=1,
        ),
    )
    plan = build_workflow_plan(manifest, execution)
    assert tuple(task.round for task in plan.tasks) == (2, 3)
    assert plan.tasks[0].fixed_image == (execution.paths.dataset_root / "round_1_488.tif")
    assert not execution.paths.output_root.exists()
```

Patch only the private header boundary so the public two-argument signature remains exact and unit tests stay scientific-runtime neutral.

- [ ] **Step 2: Add aggregated validation, collision, and fingerprint-mismatch tests**

Test multiple missing inputs in one error, symlink escape, channel/output collision, mismatched shape, non-3D/non-scalar header, one inspection per unique file, output roots containing inputs, prior plan exact match, prior plan field-level mismatch, and force inability to bypass a mismatch.

```bash
uv run pytest tests/unit/test_workflow_planning.py -q
```

Expected: collection fails because the planner is absent.

- [ ] **Step 3: Implement safe input resolution and header inspection**

Resolve every source using `resolve(strict=True)`, require `is_file()`, then require `resolved.is_relative_to(dataset_root.resolve(strict=True))`. Cache headers by resolved path. The production `_inspect_tiff_header` opens `tifffile.TiffFile` inside the function, accepts only strict scalar 3D layouts already permitted by Phase 1, and records shape, dtype, axes, and strict Cy-ExM geometry evidence without loading voxel data.

- [ ] **Step 4: Implement plan construction and prior-plan comparison**

Build tasks in ascending round order and channels in manifest order. Read an existing `scientific-plan.json` without mutation; require exact schema and compare fingerprints. On mismatch, compute a deterministic dotted-field diff and raise `WorkflowPlanError` instructing the caller to select a new output root. The function never creates the output root and has no force parameter.

- [ ] **Step 5: Run planner tests and prove mutation freedom**

```bash
uv run pytest tests/unit/test_workflow_planning.py -q
```

Expected: all resolution, aggregation, header, ordering, fingerprint, and no-write tests pass.

- [ ] **Step 6: Commit deterministic planning**

```bash
git add src/cyexm/workflow/planning.py tests/unit/test_workflow_planning.py
git commit -m "feat: build deterministic workflow plans"
```

---

### Task 7: Add read-only status, sequential execution, and public API orchestration

**Files:**
- Create: `src/cyexm/workflow/status.py`
- Create: `src/cyexm/execution/base.py`
- Create: `src/cyexm/execution/sequential.py`
- Create: `src/cyexm/execution/__init__.py`
- Create: `src/cyexm/workflow/api.py`
- Create: `src/cyexm/workflow/__init__.py`
- Create: `tests/unit/test_workflow_status.py`
- Create: `tests/unit/test_execution_sequential.py`
- Create: `tests/unit/test_workflow_api.py`

**Interfaces:**
- Consumes: immutable workflow values and Phase 1 completion-record readers
- Produces: `WorkflowExecutor.execute`, `SequentialExecutor`, `load_dataset_manifest`, `load_execution_profile`, `build_workflow_plan`, `run_workflow`, and `inspect_workflow_status`

- [ ] **Step 1: Write sequential executor ordering and failure tests**

```python
def test_sequential_executor_runs_every_task_in_order() -> None:
    calls: list[int] = []

    def runner(task: RoundTask) -> RoundResult:
        calls.append(task.round)
        return successful_result(task.round)

    results = SequentialExecutor(runner=runner).execute((task(2), task(3)))
    assert calls == [2, 3]
    assert tuple(result.round for result in results) == (2, 3)
```

Add a runner failure test proving the executor records a structured failed result and continues with the next independent round.

- [ ] **Step 2: Write status and API force tests**

Test a missing plan, pending round, partial channel records, fully complete round, corrupt completion record, `force_rounds`, `force_all`, both-force rejection, unknown forced round rejection, scientific snapshot atomic write before execution, run-summary atomic write, and `WorkflowExecutionError.failed_rounds`/`run_record` after any round fails. Inject the executor factory and runner in unit tests.

```bash
uv run pytest tests/unit/test_workflow_status.py \
  tests/unit/test_execution_sequential.py tests/unit/test_workflow_api.py -q
```

Expected: collection fails because the status/API/executor modules are absent.

- [ ] **Step 3: Define the executor protocol and sequential adapter**

```python
class WorkflowExecutor(Protocol):
    def execute(self, tasks: tuple[RoundTask, ...]) -> tuple[RoundResult, ...]:
        raise NotImplementedError


type RoundTaskRunner = Callable[[RoundTask], RoundResult]
```

`SequentialExecutor` receives a `RoundTaskRunner`, catches `Exception` per task, creates a `RoundFailure` with stage `"round"`, and continues. It re-raises `KeyboardInterrupt` so the API returns cancellation status rather than a failed round.

- [ ] **Step 4: Implement read-only status from artifact records**

`inspect_workflow_status(output_root, dataset_id)` reads the scientific plan and expected artifact list, validates each file with its `.complete.json`, and returns pending/partial/complete/invalid status without importing ANTs, NumPy, SciPy, Dask, or dask-jobqueue. Missing operational run summaries do not affect scientific status.

- [ ] **Step 5: Implement API orchestration and atomic provenance**

`run_workflow` validates force arguments, atomically writes the canonical scientific plan when absent, replaces selected immutable tasks with `force=True`, excludes fully complete unforced tasks, calls the selected executor factory, and atomically writes a UUID-based workflow-run JSON record. It returns `WorkflowResult` only when all rounds completed or were reused. After writing a failed summary, it raises `WorkflowExecutionError` containing the sorted failed rounds and run-record path. An incompatible existing plan fails before the output is changed. A `KeyboardInterrupt` writes `status: cancelled` and re-raises for CLI mapping.

Expose exactly:

```python
__all__ = [
    "build_workflow_plan",
    "inspect_workflow_status",
    "load_dataset_manifest",
    "load_execution_profile",
    "run_workflow",
]
```

- [ ] **Step 6: Run status, executor, and API tests**

```bash
uv run pytest tests/unit/test_workflow_status.py \
  tests/unit/test_execution_sequential.py tests/unit/test_workflow_api.py -q
```

Expected: all status, ordering, continuation, force, fingerprint, provenance, and cancellation tests pass.

- [ ] **Step 7: Commit the sequential public workflow surface**

```bash
git add src/cyexm/workflow/status.py src/cyexm/workflow/api.py \
  src/cyexm/workflow/__init__.py src/cyexm/execution/base.py \
  src/cyexm/execution/sequential.py src/cyexm/execution/__init__.py \
  tests/unit/test_workflow_status.py tests/unit/test_execution_sequential.py \
  tests/unit/test_workflow_api.py
git commit -m "feat: orchestrate sequential workflow plans"
```

---

### Task 8: Add the `cyexm workflow` CLI and exit contracts

**Files:**
- Create: `src/cyexm/cli/_parser.py`
- Create: `src/cyexm/cli/_commands.py`
- Create: `src/cyexm/cli/__init__.py`
- Create: `src/cyexm/cli/__main__.py`
- Create: `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: public workflow API
- Produces: `main(argv: Sequence[str] | None = None) -> int`; validate/plan/run/status commands; exit codes 0, 2, 3, 4, 130

- [ ] **Step 1: Write parser and command-delegation tests**

```python
def test_run_parses_repeatable_force_rounds() -> None:
    namespace = build_parser().parse_args(
        [
            "workflow", "run", "dataset.yaml", "--execution", "local.yaml",
            "--force-round", "2", "--force-round", "4",
        ]
    )
    assert namespace.command == "workflow"
    assert namespace.workflow_command == "run"
    assert namespace.force_round == [2, 4]
    assert namespace.force_all is False
```

Test each exact command, common `--json`, `--log-level`, and `--version`, mutual force exclusion, stdout/stderr separation, API delegation with fakes, and restoration of the prior SIGTERM handler after `run` completes.

- [ ] **Step 2: Write exception-to-exit-code tests and run red**

Parametrize `ConfigurationValidationError`/`WorkflowPlanError` to 2, `ExecutorStartupError`/`WorkerPreflightError` to 3, `WorkflowExecutionError` to 4, and `KeyboardInterrupt` to 130. Invoke the installed SIGTERM handler in a focused test and require the same cancellation path and exit code 130.

```bash
uv run pytest tests/unit/test_cli.py -q
```

Expected: collection fails because `cyexm.cli` is absent.

- [ ] **Step 3: Implement the exact parser surface**

Use `argparse` subparsers for these commands:

```text
cyexm workflow validate DATASET_YAML --execution EXECUTION_YAML
cyexm workflow plan DATASET_YAML --execution EXECUTION_YAML
cyexm workflow run DATASET_YAML --execution EXECUTION_YAML [--force-round N] [--force-round N]
cyexm workflow run DATASET_YAML --execution EXECUTION_YAML --force-all
cyexm workflow status OUTPUT_ROOT DATASET_ID
```

Use a mutually exclusive group for force flags, `type=int`, and no abbreviation. `validate` and `plan` build the plan; only `run` invokes execution; `status` uses the read-only API. Around only the `run` dispatch, install a SIGTERM handler that raises `KeyboardInterrupt`, restore the prior handler in `finally`, and let the existing cancellation/exit-130 path own cleanup and reporting.

- [ ] **Step 4: Implement human and canonical JSON output**

All structured objects provide `to_dict()`. JSON mode writes exactly one `json.dumps(payload, allow_nan=False, sort_keys=True)` object plus newline to stdout. Human mode writes concise results to stdout. Exceptions produce one-line diagnostics on stderr; tracebacks remain available only through logs.

- [ ] **Step 5: Run CLI tests and installed-entry smoke tests**

```bash
uv run pytest tests/unit/test_cli.py -q
uv run cyexm --version
uv run python -m cyexm.cli --version
```

Expected: CLI unit tests pass and both smoke commands print the package version with exit code 0.

- [ ] **Step 6: Commit the CLI**

```bash
git add src/cyexm/cli tests/unit/test_cli.py
git commit -m "feat: add workflow command line interface"
```

---

### Task 9: Enforce lightweight imports and close the Phase 2A gate

**Files:**
- Modify: `tests/unit/test_import_contract.py`
- Modify: `tests/unit/test_repository_policy.py`

**Interfaces:**
- Consumes: all Phase 2A modules
- Produces: import-side-effect regression coverage and a reviewed Phase 2A commit series

- [ ] **Step 1: Add fresh-process import assertions**

```python
def test_phase_two_configuration_and_planning_imports_are_lightweight() -> None:
    code = """
import os
import sys
before = dict(os.environ)
import cyexm.config
import cyexm.workflow
import cyexm.execution
import cyexm.cli
assert before == dict(os.environ)
for root in ("ants", "numpy", "scipy", "tifffile", "dask", "distributed", "dask_jobqueue"):
    assert not any(name == root or name.startswith(root + ".") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
```

- [ ] **Step 2: Run the import and repository-policy tests**

```bash
uv run pytest tests/unit/test_import_contract.py tests/unit/test_repository_policy.py -q
```

Expected: all import and policy tests pass with no scientific/scheduler module loaded.

- [ ] **Step 3: Run the complete portable gate**

```bash
git diff --check
uv lock --check
uv run pytest -m "unit and not hpc" tests/unit -v
uv run pytest -m "integration and not hpc" tests/integration -v
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pre-commit run --all-files
```

Expected: every command exits 0. Record exact test counts and elapsed times in the review handoff.

- [ ] **Step 4: Commit the Phase 2A import contract**

```bash
git add tests/unit/test_import_contract.py tests/unit/test_repository_policy.py
git commit -m "test: enforce lightweight workflow imports"
```

- [ ] **Step 5: Review the milestone diff**

```bash
git log --oneline --decorate main..HEAD
git diff --stat main...HEAD
git status --short
```

Expected: the task commits are present, only approved Phase 2A files changed, and the worktree is clean.
