# Fine-Grained Parallel Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in, restart-safe parallel nonlinear chunk estimation and moving-channel transformation while preserving coarse round execution, serial scientific identity, and exact fixed-grid assembly.

**Architecture:** Extend the existing execution profile with a normalized coarse/fine mode, refactor the accepted registration stages into shared internal primitives, and drive immutable fine-grained tasks from one coordinator-owned fair dispatcher. Sequential, local-Dask, and SLURM executors reuse that dispatcher; workers never submit child work, and scientific reuse remains governed only by ordinary artifact completion evidence.

**Tech Stack:** Python 3.12, ANTSpyX, NumPy `.npy` memory maps, tifffile, Dask/distributed 2025.1.0, dask-jobqueue 0.9.0, SLURMCluster, PyYAML, pytest, Ruff, basedpyright, pre-commit, uv

## Global Constraints

- Work in an isolated worktree created with `superpowers:using-git-worktrees`; do not implement directly in the current main worktree.
- Before every `uv` command, export:

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  ```

- Preserve the read-only ClearEx source at commit `2311a873b783477add7391ff00245f76490147f0`; do not edit it or import it at runtime.
- Preserve the accepted coarse round mode as the default and `ChunkedImageRegistration` as the serial numerical reference.
- Keep `strategy: serial_chunked`; scheduling mode is execution identity, never scientific identity.
- Preserve fixed/moving roles, public XYZ micrometer spacing, NumPy ZYX arrays, unequal native source shapes, and affine application as the first fixed-grid boundary.
- Preserve TRSAA affine policy, SyNOnly nonlinear policy, CC scalar radius or Mattes bins, masks, histogram matching, chunk grid, overlap, canonical core assembly, transform order, and interpolation.
- Keep affine estimation whole-image and one task per moving round. Do not add chunked affine behavior.
- Keep one Dask worker process, one `round_slot`, and one scientific task per dedicated 512-GB SLURM job.
- Keep worker count bounded by the existing executor `workers` value. Do not add per-round or per-stage limits.
- Let submitted sibling chunks and channels settle after an independent failure.
- Persist prepared inputs and chunk scalar cores atomically; do not add automatic cleanup that weakens restart verification.
- Defer scientific and scheduler imports until the selected operation after native-thread configuration.
- Portable tests must never contact SLURM or use active scientific outputs.
- Preserve the 15 unrelated untracked `slurm-*.out` files in the main worktree; never stage, delete, or move them as part of this feature.
- Update user documentation and the most specific inherited `AGENTS.md` files as behavior is introduced.

---

## File and responsibility map

| Path | Responsibility |
| --- | --- |
| `src/cyexm/config/models.py` | Normalized `ParallelismSpec` and execution-profile field |
| `src/cyexm/config/execution.py` | Strict optional `parallelism.mode` YAML decoding and revalidation |
| `src/cyexm/chunking.py` | Standard-library-only canonical chunk geometry shared by registration and workflow |
| `src/cyexm/registration/whole_image.py` | Shared accepted preparation/affine state used by coarse and fine paths |
| `src/cyexm/registration/prepared.py` | Atomic `.npy` fixed-grid prepared bundle and regional reads |
| `src/cyexm/registration/chunk_stages.py` | One local nonlinear chunk, scalar-core persistence, and canonical assembly |
| `src/cyexm/registration/chunked.py` | Serial reference composed from the shared chunk primitives |
| `src/cyexm/workflow/fine_models.py` | Immutable granular tasks, keys, and task results |
| `src/cyexm/workflow/fine_planning.py` | Pure initial graph and prepared-shape chunk expansion |
| `src/cyexm/workflow/fine_task.py` | Lazy worker dispatch to focused registration/channel operations |
| `src/cyexm/execution/dispatcher.py` | Dependency state, fair ready queue, retries, and terminal round collection |
| `src/cyexm/execution/sequential.py` | One-at-a-time execution of the same fine graph |
| `src/cyexm/execution/_dask_common.py` | Dynamic granular submission and independent result settlement |
| `src/cyexm/execution/dask.py` | Bounded local-Dask fine graph |
| `src/cyexm/execution/slurm.py` | Demand-driven scale, preflight-gated slots, evidence, and cleanup |
| `src/cyexm/workflow/api.py` | Mode selection, force/reuse, run records, and public documentation |
| `src/cyexm/workflow/status.py` | Status compatibility with fine-grained partial artifacts |
| `tests/unit/` | Strict schema, artifact, graph, dispatcher, executor, API, and policy regressions |
| `tests/integration/` | Real-ANTs coarse/fine parity, regional reads, shape, failure, and resume |
| `tests/hpc/` | Explicit real-SLURM two-tier acceptance and performance characterization |
| `README.md`, `docs/`, `examples/` | User explanation, operations, artifacts, and checked-in profiles |

## Reuse analysis

- Closest existing symbols: `ExecutionProfile`, `load_execution_profile`,
  `build_workflow_plan`, `run_workflow`, `WorkflowExecutor.execute`,
  `SequentialExecutor`, `LocalDaskExecutor`, `SlurmExecutor`,
  `run_workflow_task`, `ImageRegistration`, `ChunkedImageRegistration`,
  `compute_chunk_grid`, `ArtifactStore`, and `transform_channel`.
- Decision: extend these existing configuration, orchestration, executor, artifact,
  and registration lifecycles. Keep `WorkflowExecutor.execute(reference_task,
  round_tasks)` as the executor entry point; each executor selects the coarse or fine
  implementation from its normalized profile.
- Decision: extract shared internal affine preparation and chunk operations from the
  existing classes so coarse and fine modes call one scientific implementation and
  generate the same authoritative request identities.
- Decision: move pure chunk geometry into `cyexm.chunking`, which both registration
  and workflow may import without scientific or scheduler side effects.
- Rejected alternative: a public `ParallelChunkedImageRegistration` would duplicate
  registration lifecycle, artifact validity, force, and restart behavior.
- Rejected alternative: separate parallel executor classes would duplicate client,
  cluster, preflight, cancellation, and cleanup ownership.
- Rejected alternative: nested Dask clients in round workers would violate the
  accepted scheduler-owner and resource-slot contracts.
- Documentation synchronization: update `run_workflow` and executor Numpydoc,
  module API maps, `See Also`, README canonical examples, workflow/artifact guides,
  execution YAML examples, and layered agent guidance. No new package-root export is
  added.
- API-changing commits must include a concise `Reuse analysis:` paragraph in their
  commit bodies, referring back to this plan.

---

### Task 1: Add the opt-in execution-profile contract

**Files:**
- Modify: `src/cyexm/config/models.py`
- Modify: `src/cyexm/config/execution.py`
- Modify: `src/cyexm/config/__init__.py`
- Modify: `tests/unit/test_config_execution.py`
- Modify: `tests/unit/test_config_yaml.py`
- Modify: `tests/unit/test_workflow_models.py`
- Modify: `src/cyexm/config/AGENTS.md`

**Interfaces:**
- Consumes: existing strict execution YAML parsing and `ExecutionProfile`.
- Produces: `ParallelismMode`, `ParallelismSpec`, and
  `ExecutionProfile.parallelism: ParallelismSpec`, normalized to coarse when omitted.

- [ ] **Step 1: Write strict parser and model tests**

  Add tests equivalent to:

  ```python
  def test_execution_parallelism_defaults_to_coarse(tmp_path: Path) -> None:
      path = write_execution_yaml(tmp_path, executor="sequential")
      profile = load_execution_profile(path)
      assert profile.parallelism == ParallelismSpec(mode="coarse")


  @pytest.mark.parametrize("mode", ["coarse", "fine_grained"])
  def test_execution_accepts_exact_parallelism_modes(
      tmp_path: Path, mode: str
  ) -> None:
      path = write_execution_yaml(
          tmp_path,
          executor="sequential",
          parallelism={"mode": mode},
      )
      assert load_execution_profile(path).parallelism.mode == mode


  @pytest.mark.parametrize(
      "value",
      [
          {},
          {"mode": "fine"},
          {"mode": "fine_grained", "workers": 2},
          "fine_grained",
      ],
  )
  def test_execution_rejects_invalid_parallelism(
      tmp_path: Path, value: object
  ) -> None:
      path = write_execution_yaml(
          tmp_path,
          executor="sequential",
          parallelism=value,
      )
      with pytest.raises(ConfigurationValidationError, match="parallelism"):
          load_execution_profile(path)
  ```

- [ ] **Step 2: Run the focused tests and confirm red**

  ```bash
  uv run pytest tests/unit/test_config_execution.py \
    tests/unit/test_config_yaml.py tests/unit/test_workflow_models.py \
    -k parallelism -q
  ```

  Expected: collection or assertions fail because `ParallelismSpec` and the parsed
  field do not exist.

- [ ] **Step 3: Implement the normalized immutable model**

  Add:

  ```python
  type ParallelismMode = Literal["coarse", "fine_grained"]


  @dataclass(frozen=True, slots=True)
  class ParallelismSpec:
      mode: ParallelismMode = "coarse"
  ```

  Add `parallelism: ParallelismSpec = ParallelismSpec()` as the final
  `ExecutionProfile` field. Export the two supported configuration names from
  `cyexm.config`, and do not export them from `cyexm`.

- [ ] **Step 4: Implement exact optional YAML normalization**

  Add `_PARALLELISM_KEYS = frozenset({"mode"})`,
  `_PARALLELISM_MODES = frozenset({"coarse", "fine_grained"})`, and:

  ```python
  def _parallelism(value: object) -> ParallelismSpec:
      mapping = require_exact_keys(value, _PARALLELISM_KEYS, "parallelism")
      return ParallelismSpec(
          mode=cast(
              ParallelismMode,
              _choice(
                  mapping["mode"],
                  _PARALLELISM_MODES,
                  "parallelism.mode",
              ),
          )
      )
  ```

  Accept the section only when present, otherwise construct `ParallelismSpec()`.
  Revalidate directly constructed profiles through the same helper. Include
  `{"parallelism": {"mode": profile.parallelism.mode}}` in canonical execution
  mappings without changing scientific mappings.

- [ ] **Step 5: Update configuration guidance and run green**

  State in `src/cyexm/config/AGENTS.md` that parallelism is operational, omission
  normalizes to coarse, and no scientific manifest or fingerprint may include it.

  ```bash
  uv run pytest tests/unit/test_config_execution.py \
    tests/unit/test_config_yaml.py tests/unit/test_workflow_models.py -q
  uv run ruff check src/cyexm/config tests/unit/test_config_execution.py \
    tests/unit/test_config_yaml.py tests/unit/test_workflow_models.py
  ```

  Expected: all selected tests and Ruff checks pass.

- [ ] **Step 6: Commit the configuration slice**

  ```bash
  git add src/cyexm/config tests/unit/test_config_execution.py \
    tests/unit/test_config_yaml.py tests/unit/test_workflow_models.py
  git commit -m "feat: configure fine-grained execution"
  ```

  Commit body includes the reuse analysis decision to extend `ExecutionProfile`
  rather than add a second execution document.

---

### Task 2: Extract the accepted whole-image affine preparation boundary

**Files:**
- Modify: `src/cyexm/registration/whole_image.py`
- Modify: `tests/unit/test_whole_image.py`
- Modify: `tests/integration/test_whole_image.py`
- Modify: `src/cyexm/registration/AGENTS.md`

**Interfaces:**
- Consumes: the current first half of `ImageRegistration.register`.
- Produces: private `_LinearPreparation` and
  `ImageRegistration._prepare_linear_state() -> _LinearPreparation`; existing
  `register() -> None` composes it without changing public outputs.

- [ ] **Step 1: Add call-order and artifact characterization tests**

  Add a fake-backend regression that records:

  ```python
  assert events == [
      "load-fixed",
      "load-moving",
      "crop",
      "fixed-mask",
      "moving-mask",
      "pre-metrics",
      "affine",
      "linear-metrics",
      "nonlinear",
      "nonlinear-metrics",
      "registered-image",
  ]
  ```

  Add a real dry-run integration test that snapshots every public output and
  completion record before the refactor and asserts an immediate forced rerun has
  the same artifact set, geometry, request stages, and cache relationships.

- [ ] **Step 2: Run the characterization tests**

  ```bash
  uv run pytest tests/unit/test_whole_image.py \
    tests/integration/test_whole_image.py -k 'stage_order or preparation_boundary' -q
  ```

  Expected: the new test fails because `_prepare_linear_state` is absent.

- [ ] **Step 3: Add the private state object**

  Define:

  ```python
  @dataclass(frozen=True, slots=True)
  class _LinearPreparation:
      backend: RegistrationBackend
      paths: RegistrationPaths
      resolved: ResolvedNativeThreads
      fixed: object
      moving_linear: object
      fixed_mask: object
      moving_linear_mask: object
      affine: object
  ```

  Move source snapshot, image loading, crop, mask, fixed-reference, pre-metrics,
  affine, and linear-metrics work into `_prepare_linear_state`. Keep all existing
  validation, exception translation, request identity, force invalidation, and lazy
  runtime behavior intact.

- [ ] **Step 4: Compose `register` from the shared boundary**

  Replace the duplicated first-half locals with:

  ```python
  state = self._prepare_linear_state()
  warp_path, moving_nonlinear, moving_nonlinear_mask = (
      self._load_or_estimate_nonlinear(
          state.backend,
          state.fixed,
          state.moving_linear,
          state.fixed_mask,
          state.moving_linear_mask,
          state.affine,
          state.paths,
          state.resolved,
      )
  )
  ```

  Keep nonlinear metrics and registered-image commits in their current order.

- [ ] **Step 5: Run the full Phase 1 registration slice**

  ```bash
  uv run pytest tests/unit/test_whole_image.py \
    tests/unit/test_chunked_registration.py \
    tests/integration/test_whole_image.py tests/integration/test_chunked.py -q
  uv run ruff check src/cyexm/registration/whole_image.py \
    tests/unit/test_whole_image.py tests/integration/test_whole_image.py
  ```

  Expected: all tests pass with no public artifact or request change.

- [ ] **Step 6: Commit the refactor**

  ```bash
  git add src/cyexm/registration/whole_image.py \
    src/cyexm/registration/AGENTS.md \
    tests/unit/test_whole_image.py tests/integration/test_whole_image.py
  git commit -m "refactor: expose internal affine preparation state"
  ```

---

### Task 3: Persist and regionally read fixed-grid prepared arrays

**Files:**
- Create: `src/cyexm/registration/prepared.py`
- Create: `tests/unit/test_prepared_registration.py`
- Create: `tests/integration/test_prepared_registration.py`
- Modify: `src/cyexm/registration/_request_policy.py`
- Modify: `src/cyexm/registration/AGENTS.md`
- Modify: `tests/integration/AGENTS.md`

**Interfaces:**
- Produces:
  `PreparedRoundPaths.for_round(output: Path, round_number: int)`,
  `PreparedManifest`,
  `commit_prepared_round(registration: ImageRegistration) -> PreparedManifest`, and
  `read_prepared_region(manifest: PreparedManifest, chunk: ChunkInfo) -> PreparedRegion`.
- `PreparedManifest.to_dict() -> dict[str, object]` and
  `PreparedManifest.from_dict(payload: Mapping[str, object]) -> PreparedManifest`
  provide strict worker/coordinator transport.
- `PreparedRegion` contains
  `fixed_scalar`, `moving_linear_scalar`, `fixed_mask`, and
  `moving_linear_mask` as four owned float32 ZYX arrays plus
  `geometry: ImageGeometry`.

- [ ] **Step 1: Write atomic format and regional-read tests**

  Cover exact paths, `allow_pickle=False`, C-contiguous float32 arrays, manifest
  exact keys, completion records, input mutation detection, force, stale partial
  commits, non-finite values, dtype/shape mismatch, and direction-aware cropped
  origin.

  The core region test uses:

  ```python
  region = read_prepared_region(manifest, chunk)
  np.testing.assert_array_equal(
      region.moving_linear_scalar,
      moving_linear[chunk.extended_slices],
  )
  assert region.geometry.shape == chunk.extended_shape
  assert region.geometry.spacing == fixed_geometry.spacing
  ```

- [ ] **Step 2: Run the tests and confirm red**

  ```bash
  uv run pytest tests/unit/test_prepared_registration.py \
    tests/integration/test_prepared_registration.py -q
  ```

  Expected: collection fails because `cyexm.registration.prepared` does not exist.

- [ ] **Step 3: Implement paths and strict manifest**

  Use the directory `round_{round}_prepared` beneath the registration output and
  exact filenames from the design. The manifest records format version `1`, four
  `FileIdentity` objects, native fixed/moving identities, crop identity or `null`,
  affine identity, fixed geometry, mask/scalar policies, and native-thread values.
  Reject unknown keys, non-finite geometry, object arrays, non-float32 arrays, and
  arrays not exactly ZYX.

- [ ] **Step 4: Implement atomic uncompressed NPY commits**

  Use `numpy.lib.format.open_memmap` only inside an `ArtifactStore.commit` writer.
  Write a C-contiguous float32 copy, flush it, reopen with
  `np.load(path, mmap_mode="r", allow_pickle=False)`, and validate exact shape,
  dtype, finiteness, and content identity before promotion. Commit the manifest last.

- [ ] **Step 5: Implement preparation from the shared affine state**

  `commit_prepared_round` configures native threads, calls
  `_prepare_linear_state`, converts the fixed scalar, affine moving scalar, fixed
  mask, and affine moving mask through the backend, validates exact fixed geometry,
  and commits the bundle. It must not estimate nonlinear transforms.

- [ ] **Step 6: Implement safe regional reconstruction**

  Open each `.npy` read-only with `mmap_mode="r"`, validate its `FileIdentity`, copy
  only `chunk.extended_slices`, and derive the cropped origin using fixed direction,
  spacing, and the XYZ form of the extended ZYX start. Recheck file identities after
  copying.

- [ ] **Step 7: Run and commit the prepared bundle**

  ```bash
  uv run pytest tests/unit/test_prepared_registration.py \
    tests/integration/test_prepared_registration.py -q
  uv run ruff check src/cyexm/registration/prepared.py \
    src/cyexm/registration/_request_policy.py \
    tests/unit/test_prepared_registration.py \
    tests/integration/test_prepared_registration.py
  ```

  Expected: atomicity, geometry, regional reads, and real-ANTs reconstruction pass.

  ```bash
  git add src/cyexm/registration/prepared.py \
    src/cyexm/registration/_request_policy.py \
    src/cyexm/registration/AGENTS.md tests/unit/test_prepared_registration.py \
    tests/integration/test_prepared_registration.py tests/integration/AGENTS.md
  git commit -m "feat: persist fixed-grid registration inputs"
  ```

---

### Task 4: Share one nonlinear chunk implementation between coarse and fine modes

**Files:**
- Create: `src/cyexm/chunking.py`
- Create: `src/cyexm/registration/chunk_stages.py`
- Modify: `src/cyexm/registration/chunked.py`
- Modify: `src/cyexm/registration/_request_policy.py`
- Modify: `tests/unit/test_chunked_geometry.py`
- Modify: `tests/unit/test_chunked_registration.py`
- Create: `tests/unit/test_chunk_stages.py`
- Modify: `tests/integration/test_chunked.py`

**Interfaces:**
- Produces:
  `ChunkInfo`, `compute_chunk_grid`, `extract_chunk`, and `stitch_core` from
  `cyexm.chunking`;
  `ChunkStageContext`;
  `run_nonlinear_chunk(context, region, chunk) -> ChunkStageArtifacts`; and
  `assemble_nonlinear_chunks(context, manifest) -> AssembledNonlinear`.
- `ChunkStageArtifacts` contains `warp_path`, `scalar_core_path`,
  `warp_identity`, `scalar_core_identity`, and `reused`.
- `AssembledNonlinear` contains `warp_path`, `registered_path`,
  `nonlinear_metrics_path`, `chunk_manifest_path`, and `reused`.
- `cyexm.registration.chunked` re-exports the existing chunk geometry names for
  compatibility.

- [ ] **Step 1: Move pure chunk geometry under characterization**

  Add import-identity and behavior tests proving:

  ```python
  assert registration_chunked.compute_chunk_grid is chunking.compute_chunk_grid
  assert registration_chunked.ChunkInfo is chunking.ChunkInfo
  assert compute_chunk_grid((9, 17, 19), (4, 8, 8), 0.25) == expected_grid
  ```

  Run:

  ```bash
  uv run pytest tests/unit/test_chunked_geometry.py -q
  ```

  Expected: import fails because `cyexm.chunking` is absent.

- [ ] **Step 2: Extract the standard-library-only geometry module**

  Move `ChunkInfo`, validation, grid calculation, extraction, and stitching without
  changing field names, ordering, bounds, or error contracts. Keep compatibility
  imports and `__all__` in `registration.chunked`. `cyexm.chunking` must not import
  registration, NumPy, ANTs, Dask, or workflow.

- [ ] **Step 3: Add local-warp and scalar-core TDD coverage**

  Tests require one invocation of the accepted nonlinear backend, unchanged local
  `StageRequest`, atomic warp and `chunk_{id}_registered_core.npy`, correct histogram
  matching, force behavior, low-variance identity warp, input-change detection, and
  reconstruction of a missing scalar core from a valid warp without another
  nonlinear call.

  ```bash
  uv run pytest tests/unit/test_chunk_stages.py -q
  ```

  Expected: collection fails because the stage module is absent.

- [ ] **Step 4: Implement the shared stage context**

  Define a frozen context containing exact source identities, affine identity, fixed
  geometry, registration method/accuracy/metric/radius, chunk grid, overlap, native
  threads, force policy, output paths, backend, and artifact store. Keep backend and
  store injection private to tests; do not expose the context from
  `cyexm.registration`.

- [ ] **Step 5: Extract one-chunk execution**

  Move the current `_local_request`, local-warp reuse/commit, local reconstruction,
  core extraction, validation, and stale behavior into `run_nonlinear_chunk`. The
  local-warp request must serialize identically before and after extraction.
  Scalar-core request identity adds only the accepted local warp and existing scalar
  postprocessing policy.

- [ ] **Step 6: Extract canonical assembly**

  **Correction approved 2026-07-23:** The original exactly-once/non-overlap
  requirement contradicted the canonical ClearEx grid and is replaced by the
  compatibility-preserving rule below.

  Read valid chunks in ascending ID order regardless of completion order, preserve
  the canonical legacy grid including overlapping nominal core/output slices, and
  let later chunk IDs deterministically overwrite earlier scalar and displacement
  values in overlap regions. Prove every fixed-grid voxel is written at least once,
  assemble scalar and vector float32 arrays, commit the existing chunk manifest and
  full-warp request, reconstruct the nonlinear mask, metrics, and registered TIFF
  through the accepted helpers, and reject any gap, malformed/missing/duplicate
  artifact, mismatched ID or request, non-finite value, wrong shape, or wrong
  component count. Expected canonical overlap is valid and must not be rejected.

- [ ] **Step 7: Refactor the serial class to compose the primitives**

  Keep the public constructor, `register`, artifact names, completion records, and
  exception types. Its loop calls `run_nonlinear_chunk` one chunk at a time and then
  `assemble_nonlinear_chunks`. It also writes scalar cores so a later fine run can
  reuse them.

- [ ] **Step 8: Prove coarse compatibility and commit**

  ```bash
  uv run pytest tests/unit/test_chunked_geometry.py \
    tests/unit/test_chunked_registration.py tests/unit/test_chunk_stages.py -q
  uv run pytest tests/integration/test_chunked.py -q
  uv run ruff check src/cyexm/chunking.py \
    src/cyexm/registration/chunked.py \
    src/cyexm/registration/chunk_stages.py tests/unit/test_chunk_stages.py
  ```

  Expected: the complete serial suite passes and characterized public artifacts are
  numerically unchanged.

  ```bash
  git add src/cyexm/chunking.py src/cyexm/registration/chunked.py \
    src/cyexm/registration/chunk_stages.py \
    src/cyexm/registration/_request_policy.py \
    tests/unit/test_chunked_geometry.py \
    tests/unit/test_chunked_registration.py tests/unit/test_chunk_stages.py \
    tests/integration/test_chunked.py
  git commit -m "refactor: share nonlinear chunk stages"
  ```

---

### Task 5: Model and run immutable fine-grained workflow tasks

**Files:**
- Create: `src/cyexm/workflow/fine_models.py`
- Create: `src/cyexm/workflow/fine_planning.py`
- Create: `src/cyexm/workflow/fine_task.py`
- Modify: `src/cyexm/workflow/task.py`
- Modify: `src/cyexm/workflow/models.py`
- Create: `tests/unit/test_fine_workflow_models.py`
- Create: `tests/unit/test_fine_workflow_planning.py`
- Create: `tests/unit/test_fine_workflow_task.py`
- Modify: `src/cyexm/workflow/AGENTS.md`

**Interfaces:**
- Produces immutable exact task types:
  `RoundPreparationTask`, `NonlinearChunkTask`, `ChunkAssemblyTask`,
  `WholeImageRegistrationTask`, `ChannelTransformTask`, and
  `RoundValidationTask`.
- Produces `FineTaskKey`, `FineTaskResult`,
  `WorkflowTaskResult = RoundResult | FineTaskResult`,
  `initial_fine_tasks(round_tasks)`, `chunk_tasks(prepared_result)`, and
  `run_fine_task(task) -> FineTaskResult`.
- Extends `WorkflowTask` and `run_workflow_task` to exact-type dispatch without
  changing coarse task behavior. `WorkflowTaskRunner` returns
  `WorkflowTaskResult`; coarse task inputs still return exact `RoundResult`.

- [ ] **Step 1: Write exact model and serialization tests**

  Require stable keys such as:

  ```python
  assert FineTaskKey(round=2, stage="chunk", item="0007").encoded == (
      "round-002:chunk:0007"
  )
  assert task.log_path.name == "round-002-chunk-0007.log"
  ```

  Reject booleans as IDs, unsupported stages, duplicate outputs, unsafe logs,
  mismatched embedded rounds, malformed channel indices, and a chunk task for a
  whole-image round.

- [ ] **Step 2: Write pure graph-template tests**

  Require reference descendants to begin with one preparation or whole-image task
  per moving round. Given an in-memory prepared descriptor with shape `(9, 17, 19)`,
  require exact canonical chunk tasks and one assembly dependency over every chunk
  key. Given accepted transforms, require one channel task per manifest channel and
  final validation depending on every channel.

  ```bash
  uv run pytest tests/unit/test_fine_workflow_models.py \
    tests/unit/test_fine_workflow_planning.py -q
  ```

  Expected: collection fails because the fine modules do not exist.

- [ ] **Step 3: Implement focused task models**

  Embed the authoritative `RoundTask` by composition rather than copying scientific
  fields. Each task adds only its stage-specific index, chunk, paths, dependencies,
  and unique log. `FineTaskResult` records key, status, timestamps, host, process,
  Dask key, SLURM job, native threads, artifacts, reused artifacts, and optional
  failure. Prepared results also carry a strict JSON-compatible prepared descriptor.

- [ ] **Step 4: Implement pure incremental graph expansion**

  `initial_fine_tasks` creates only tasks knowable before execution. A successful
  preparation descriptor supplies the fixed cropped shape used by
  `compute_chunk_grid`; graph expansion is deterministic and mutation-free. The
  coordinator, not the worker, instantiates child tasks. Whole-image rounds skip
  chunk expansion.

- [ ] **Step 5: Implement lazy exact-type worker dispatch**

  `run_fine_task` configures task native threads before importing registration
  operations. It delegates preparation to `commit_prepared_round`, chunks and
  assembly to `chunk_stages`, whole-image registration to `ImageRegistration`,
  channels to `transform_channel`, and final validation to the existing completion
  provenance helpers. Workers do not import execution adapters or submit work.

- [ ] **Step 6: Test task failure and log isolation**

  Inject each stage failure and assert one `FineTaskResult(status="failed")` with the
  exact stage and unique task log. Assert concurrent tasks never target
  `round-002.log`.

- [ ] **Step 7: Run and commit workflow task modeling**

  ```bash
  uv run pytest tests/unit/test_fine_workflow_models.py \
    tests/unit/test_fine_workflow_planning.py \
    tests/unit/test_fine_workflow_task.py tests/unit/test_workflow_models.py -q
  uv run ruff check src/cyexm/workflow/fine_models.py \
    src/cyexm/workflow/fine_planning.py src/cyexm/workflow/fine_task.py \
    src/cyexm/workflow/task.py tests/unit/test_fine_workflow_models.py \
    tests/unit/test_fine_workflow_planning.py tests/unit/test_fine_workflow_task.py
  ```

  Expected: all exact-type, graph, runner, and import-boundary tests pass.

  ```bash
  git add src/cyexm/workflow tests/unit/test_fine_workflow_models.py \
    tests/unit/test_fine_workflow_planning.py \
    tests/unit/test_fine_workflow_task.py tests/unit/test_workflow_models.py
  git commit -m "feat: model fine-grained workflow tasks"
  ```

---

### Task 6: Implement the fair scheduler-neutral dispatcher and sequential reference

**Files:**
- Create: `src/cyexm/execution/dispatcher.py`
- Modify: `src/cyexm/execution/base.py`
- Modify: `src/cyexm/execution/sequential.py`
- Create: `tests/unit/test_execution_dispatcher.py`
- Modify: `tests/unit/test_execution_sequential.py`
- Modify: `src/cyexm/execution/AGENTS.md`

**Interfaces:**
- Produces `FineGraphController`, with:
  `FineGraphController(round_tasks: tuple[RoundTask, ...])`,
  `pop_ready(limit: int) -> tuple[FineWorkflowTask, ...]`,
  `record(task, result) -> None`,
  `terminal: bool`, and
  `round_results() -> tuple[RoundResult, ...]`.
- `SequentialExecutor.execute` keeps its existing signature and chooses coarse or
  fine behavior from `ExecutionProfile.parallelism`.

- [ ] **Step 1: Write dependency, fairness, and settlement tests**

  Use two chunked rounds with unequal chunk counts. Assert admissions alternate by
  round while both are ready, one round consumes all slots when alone, no task is
  admitted before dependencies succeed, and a failed chunk blocks only its
  assembly. Record sibling successes after failure and require them in the retained
  evidence.

  ```python
  assert [task.round for task in controller.pop_ready(limit=4)] == [2, 3, 2, 3]
  ```

- [ ] **Step 2: Run dispatcher tests and confirm red**

  ```bash
  uv run pytest tests/unit/test_execution_dispatcher.py \
    tests/unit/test_execution_sequential.py -k 'fine or fair or dependency' -q
  ```

  Expected: collection fails because the controller does not exist.

- [ ] **Step 3: Implement deterministic fair admission**

  Store one FIFO deque per round and a rotating sorted round deque. Admit at most
  `limit` exact tasks, skip blocked rounds, and rotate after each admission. Record
  explicit terminal states for succeeded, failed, blocked, cancelled, and
  never-submitted nodes. Reject duplicate results or a result for a non-running key.

- [ ] **Step 4: Implement graph transitions**

  A successful preparation expands canonical chunks; all successful chunks release
  assembly; successful assembly or whole-image registration releases channels; all
  successful channels release final validation. A failure blocks descendants but
  leaves independent ready/running tasks eligible. Final round results include every
  stage failure without fabricating worker results for blocked nodes.

- [ ] **Step 5: Extend sequential execution**

  Give `SequentialExecutor` the normalized profile while retaining its injectable
  coarse runner. In fine mode, repeatedly pop one task, call `run_fine_task`, and
  record it. On `KeyboardInterrupt`, attach completed round and task evidence before
  re-raising through `_SequentialExecutionInterrupted`.

- [ ] **Step 6: Run and commit the dispatcher**

  ```bash
  uv run pytest tests/unit/test_execution_dispatcher.py \
    tests/unit/test_execution_sequential.py -q
  uv run ruff check src/cyexm/execution/dispatcher.py \
    src/cyexm/execution/base.py src/cyexm/execution/sequential.py \
    tests/unit/test_execution_dispatcher.py \
    tests/unit/test_execution_sequential.py
  ```

  Expected: fairness, failure settlement, canonical sequential execution, and coarse
  compatibility pass.

  ```bash
  git add src/cyexm/execution/dispatcher.py src/cyexm/execution/base.py \
    src/cyexm/execution/sequential.py src/cyexm/execution/AGENTS.md \
    tests/unit/test_execution_dispatcher.py \
    tests/unit/test_execution_sequential.py
  git commit -m "feat: dispatch fine-grained workflow tasks"
  ```

---

### Task 7: Execute the fine graph on bounded local Dask

**Files:**
- Modify: `src/cyexm/execution/_dask_common.py`
- Modify: `src/cyexm/execution/dask.py`
- Modify: `tests/unit/test_execution_dask.py`
- Modify: `tests/integration/test_workflow_local_dask.py`
- Modify: `tests/integration/local_dask_helpers.py`

**Interfaces:**
- Consumes: `FineGraphController`, `run_fine_task`, existing
  `ROUND_RESOURCE = {"round_slot": 1}`, retries, and owned lifecycle.
- Produces fine-mode dynamic submission with no nested tasks and the same terminal
  round result contract.

- [ ] **Step 1: Add fake-module fine scheduling tests**

  Require no more futures than connected workers, every submit to use
  `resources={"round_slot": 1}`, `pure=False`, and `retries=0`, dynamic child
  submission only after successful dependencies, sibling settlement, bounded
  executor retries, interruption cancellation, and client-before-cluster cleanup.

  ```bash
  uv run pytest tests/unit/test_execution_dask.py -k fine -q
  ```

  Expected: tests fail because Local Dask always submits coarse rounds.

- [ ] **Step 2: Extract reusable dynamic distributed collection**

  Add a helper that fills available capacity from `controller.pop_ready`, consumes
  `as_completed(with_results=True, raise_errors=False)`, converts malformed or
  raised outcomes to exact task failures, applies configured retries through
  ordinary artifact validation, records the result, and continues until terminal.
  Keep the existing coarse collector unchanged for coarse mode.

- [ ] **Step 3: Add fine-mode LocalCluster behavior**

  Fine mode starts exactly `spec.workers` process workers because future chunk width
  is crop-dependent and cannot be read during mutation-free planning. Every worker
  advertises one `round_slot`. Run the reference prerequisite first in the same
  owned cluster, then the controller. Coarse mode retains its existing
  `min(pending rounds, configured workers)` bound.

- [ ] **Step 4: Add real process-isolation integration**

  Use small dry-run images with at least two chunks and two channels. Assert distinct
  worker process IDs can execute sibling tasks, no worker creates a client, outputs
  match fine sequential execution, and a second run reuses all accepted outputs.

- [ ] **Step 5: Run and commit Local Dask**

  ```bash
  uv run pytest tests/unit/test_execution_dask.py -q
  uv run pytest tests/integration/test_workflow_local_dask.py -q
  uv run ruff check src/cyexm/execution/_dask_common.py \
    src/cyexm/execution/dask.py tests/unit/test_execution_dask.py \
    tests/integration/test_workflow_local_dask.py
  ```

  Expected: coarse and fine local-Dask lifecycle, numerical, retry, and cancellation
  cases pass.

  ```bash
  git add src/cyexm/execution/_dask_common.py src/cyexm/execution/dask.py \
    tests/unit/test_execution_dask.py \
    tests/integration/test_workflow_local_dask.py \
    tests/integration/local_dask_helpers.py
  git commit -m "feat: run fine tasks on local Dask"
  ```

---

### Task 8: Add demand-driven, preflight-gated SLURM task slots

**Files:**
- Modify: `src/cyexm/execution/slurm.py`
- Modify: `src/cyexm/execution/preflight.py`
- Modify: `tests/unit/test_execution_slurm.py`
- Modify: `tests/unit/test_worker_preflight.py`
- Modify: `tests/hpc/test_slurm_environment.py`
- Modify: `src/cyexm/execution/AGENTS.md`

**Interfaces:**
- Consumes: fine distributed collector, `SLURMCluster.scale`, worker probes, and
  Dask Scheduler `add_resources`.
- Produces demand-driven scale events and makes `round_slot=1` eligible only after a
  newly connected worker passes complete preflight.

- [ ] **Step 1: Add exact fake scale and gating tests**

  Cover initial one-worker reference scale, later scale-up as ready width grows,
  configured cap, no scale-down requirement, probe only new addresses, slot
  activation only after successful probes, no scientific submission to an
  unvalidated worker, worker-set instability, activation failure, cancellation
  during scale, and cleanup of every owned job.

  Assert fine-mode worker command starts without `round_slot`, followed by:

  ```python
  assert modules.scheduler_resource_calls == [
      ("tcp://worker-1:1234", {"round_slot": 1}),
      ("tcp://worker-2:1234", {"round_slot": 1}),
  ]
  ```

- [ ] **Step 2: Run the fine SLURM tests and confirm red**

  ```bash
  uv run pytest tests/unit/test_execution_slurm.py \
    tests/unit/test_worker_preflight.py -k 'fine or scale_event or resource_gate' -q
  ```

  Expected: tests fail because all workers advertise `round_slot` before preflight
  and scale occurs only once.

- [ ] **Step 3: Make job-script resource policy mode-specific**

  Coarse mode retains `--resources round_slot=1`. Fine mode starts candidate workers
  without a scientific slot. Update rendered-script validation to require the exact
  mode-specific worker command and continue rejecting user-provided resource
  overrides.

- [ ] **Step 4: Implement preflight-gated activation**

  After a new address passes the existing package, path, thread, memory, network,
  scratch, and job-identity probe, call a coordinator-side scheduler function that
  invokes:

  ```python
  dask_scheduler.add_resources(
      worker_address,
      resources={"round_slot": 1},
  )
  ```

  Snapshot scheduler resources afterward and require exactly one slot on every
  validated worker and none on candidates. Never submit unrestricted tasks; every
  scientific task continues to request `round_slot=1`.

- [ ] **Step 5: Implement demand-driven scale transitions**

  Start one job for the reference barrier. After each controller transition, compute
  `desired = min(spec.workers, running_count + ready_count)` with a minimum of one
  while execution remains active. Scale upward when desired exceeds validated
  capacity, wait for the exact new connected set, probe and activate only newcomers,
  then refill the dispatcher. Record requested, connected, probed, activated, failed,
  and timestamp facts for each scale event.

- [ ] **Step 6: Extend strict evidence validation**

  Version the SLURM execution-evidence schema for fine mode. Validate graph mode,
  task keys and dependencies, scale events, candidate/validated addresses, resource
  activation, per-task attempts and placement, blocked descendants, cancellation,
  and cleanup. Preserve coarse evidence decoding and fail closed on mixed or
  duplicate identities.

- [ ] **Step 7: Run fake and portable HPC-environment tests**

  ```bash
  uv run pytest tests/unit/test_execution_slurm.py \
    tests/unit/test_worker_preflight.py \
    tests/unit/test_hpc_environment.py -q
  uv run pytest tests/hpc/test_slurm_environment.py -m hpc --collect-only -q
  uv run ruff check src/cyexm/execution/slurm.py \
    src/cyexm/execution/preflight.py tests/unit/test_execution_slurm.py \
    tests/unit/test_worker_preflight.py
  ```

  Expected: fake construction, dynamic gating, evidence, failure, and cleanup tests
  pass; collection submits no job.

- [ ] **Step 8: Commit SLURM fine execution**

  ```bash
  git add src/cyexm/execution/slurm.py src/cyexm/execution/preflight.py \
    src/cyexm/execution/AGENTS.md tests/unit/test_execution_slurm.py \
    tests/unit/test_worker_preflight.py tests/unit/test_hpc_environment.py \
    tests/hpc/test_slurm_environment.py
  git commit -m "feat: scale preflighted fine-grained SLURM tasks"
  ```

  Commit body records extension of `SlurmExecutor` rather than a second cluster
  owner.

---

### Task 9: Integrate fine execution with workflow API, status, force, and evidence

**Files:**
- Modify: `src/cyexm/workflow/api.py`
- Modify: `src/cyexm/workflow/status.py`
- Modify: `src/cyexm/workflow/_execution_validation.py`
- Modify: `src/cyexm/workflow/_round_provenance.py`
- Modify: `src/cyexm/workflow/models.py`
- Modify: `tests/unit/test_workflow_api.py`
- Modify: `tests/unit/test_workflow_status.py`
- Modify: `tests/unit/test_workflow_thread_boundaries.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `src/cyexm/workflow/AGENTS.md`

**Interfaces:**
- Keeps `run_workflow(...) -> WorkflowResult` and
  `inspect_workflow_status(...) -> WorkflowStatus` as canonical user entry points.
- Persists normalized mode and granular operational evidence without changing
  scientific plan JSON or fingerprint.

- [ ] **Step 1: Add API identity and reuse tests**

  Build coarse and fine plans from identical scientific YAML and assert:

  ```python
  assert coarse.scientific_json == fine.scientific_json
  assert coarse.scientific_fingerprint == fine.scientific_fingerprint
  assert coarse.execution.parallelism.mode == "coarse"
  assert fine.execution.parallelism.mode == "fine_grained"
  ```

  Run coarse to completion, switch to fine, and require no SyN resubmission. Run a
  partial fine workflow, switch to coarse, and require valid local warps to remain
  reusable.

- [ ] **Step 2: Add failure, force, cancellation, and run-record tests**

  Cover preparation failure, one failed chunk with sibling settlement, blocked
  assembly, assembly failure, one failed channel with sibling settlement, all-round
  ordering, force reference propagation, force one moving round, partial sequential
  interruption, distributed cancellation, exact exit codes, and operational evidence
  validation after scientific work.

- [ ] **Step 3: Run focused API tests and confirm red**

  ```bash
  uv run pytest tests/unit/test_workflow_api.py \
    tests/unit/test_workflow_status.py tests/unit/test_cli.py \
    -k 'fine or granular or parallelism' -q
  ```

  Expected: fine plans still use only coarse executors and run records omit granular
  evidence.

- [ ] **Step 4: Extend the existing executor factory and API**

  Pass the normalized profile into `SequentialExecutor`; keep the existing
  `LocalDaskExecutor` and `SlurmExecutor` classes. Do not add another public
  orchestration function. Preserve complete-round short-circuiting before executor
  construction.

- [ ] **Step 5: Preserve public status authority**

  Continue classifying a round from its accepted public affine, full warp,
  diagnostic, metrics, and channels. Prepared arrays and scalar cores may explain
  partial progress but cannot make a round complete. An invalid prepared dependency
  causes recomputation when fine mode needs it; it does not invalidate an otherwise
  independently verifiable final output.

- [ ] **Step 6: Persist strict granular evidence**

  Include normalized mode, task graph, task results, blocked descendants, scale and
  preflight facts, attempts, placement, timestamps, cancellation, and cleanup in
  `execution_evidence_json`. Keep all of it outside the scientific snapshot.

- [ ] **Step 7: Synchronize Numpydoc and API maps**

  Update `run_workflow`, executor class docstrings, and related `See Also` sections to
  explain coarse/fine mode, resource ownership, side effects, restart authority,
  failure settlement, and scheduler non-ownership by workers. Do not expose internal
  fine task models as supported extension interfaces.

- [ ] **Step 8: Run and commit API integration**

  ```bash
  uv run pytest tests/unit/test_workflow_api.py \
    tests/unit/test_workflow_status.py \
    tests/unit/test_workflow_thread_boundaries.py tests/unit/test_cli.py -q
  uv run ruff check src/cyexm/workflow tests/unit/test_workflow_api.py \
    tests/unit/test_workflow_status.py \
    tests/unit/test_workflow_thread_boundaries.py tests/unit/test_cli.py
  ```

  Expected: identity, reuse, force, failure, cancellation, evidence, CLI, and
  docstring tests pass.

  ```bash
  git add src/cyexm/workflow tests/unit/test_workflow_api.py \
    tests/unit/test_workflow_status.py \
    tests/unit/test_workflow_thread_boundaries.py tests/unit/test_cli.py
  git commit -m "feat: orchestrate fine-grained registration"
  ```

  Commit body includes the final concise reuse analysis for `run_workflow` and the
  executor protocol.

---

### Task 10: Add real-ANTs scientific regression coverage

**Files:**
- Create: `tests/integration/test_fine_grained_workflow.py`
- Modify: `tests/integration/test_chunked.py`
- Modify: `tests/integration/test_channel_transform.py`
- Modify: `tests/integration/test_workflow_local_dask.py`
- Modify: `tests/integration/AGENTS.md`

**Interfaces:**
- Consumes: complete fine sequential and local-Dask workflows.
- Produces portable scientific evidence for exact geometry, deterministic one-thread
  parity, completion-order-independent assembly, and restart.

- [ ] **Step 1: Add seeded one-thread coarse/fine parity**

  Create small deterministic volumes with at least four nontrivial chunks and three
  moving channels. Use one native thread and the accepted test seed. Compare affine
  parameters, each local warp, full warp, diagnostic TIFF, metrics, and transformed
  channels using the repository's strict deterministic comparison helpers.

- [ ] **Step 2: Add unequal-shape and geometry coverage**

  Make fixed, registration, and dependent-channel inputs differ along Z, Y, and X.
  Assert prepared arrays, every local chunk, full warp, diagnostic, and every channel
  land on exact fixed shape, origin, spacing, and direction.

- [ ] **Step 3: Add completion-order and resume cases**

  Feed assembly the same valid chunks in forward, reverse, and permuted completion
  order and require identical results. Inject failure after two chunk commits and
  after one channel commit; rerun and assert valid artifacts retain their file
  identities while only missing work executes.

- [ ] **Step 4: Add local-Dask parity**

  Compare fine sequential and process-based local Dask under the same one-thread
  seed. Assert at least two process IDs, one task per worker slot, and no nested
  client.

- [ ] **Step 5: Run and commit scientific integration**

  ```bash
  uv run pytest tests/integration/test_fine_grained_workflow.py \
    tests/integration/test_chunked.py \
    tests/integration/test_channel_transform.py \
    tests/integration/test_workflow_local_dask.py -q
  uv run ruff check tests/integration/test_fine_grained_workflow.py \
    tests/integration/test_chunked.py \
    tests/integration/test_channel_transform.py \
    tests/integration/test_workflow_local_dask.py
  ```

  Expected: all real-ANTs scientific, geometry, ordering, and restart regressions
  pass.

  ```bash
  git add tests/integration/test_fine_grained_workflow.py \
    tests/integration/test_chunked.py \
    tests/integration/test_channel_transform.py \
    tests/integration/test_workflow_local_dask.py tests/integration/AGENTS.md
  git commit -m "test: verify fine-grained scientific parity"
  ```

---

### Task 11: Document fine-grained execution and update examples and agent guidance

**Files:**
- Modify: `README.md`
- Create: `docs/fine-grained-execution.md`
- Modify: `docs/workflow-guide.md`
- Modify: `docs/registration-artifacts.md`
- Modify: `docs/index.md`
- Modify: `docs/myst.yml`
- Modify: `examples/execution-sequential.yaml`
- Modify: `examples/execution-local-dask.yaml`
- Modify: `examples/execution-slurm-512gb.yaml`
- Modify: `AGENTS.md`
- Modify: `src/cyexm/AGENTS.md`
- Modify: `src/cyexm/runtime/AGENTS.md`
- Modify: `tests/AGENTS.md`
- Modify: `tests/unit/AGENTS.md`
- Modify: `tests/hpc/AGENTS.md`
- Modify: `docs/AGENTS.md`
- Modify: `tests/unit/test_phase2_documentation.py`
- Modify: `tests/unit/test_phase2_examples.py`
- Modify: `tests/unit/test_repository_policy.py`

**Interfaces:**
- Produces complete user-facing explanation and checked-in strict profile examples.

- [ ] **Step 1: Write failing documentation policy tests**

  Require the README and dedicated page to contain exact coarse/fine YAML, task graph
  stages, `serial_chunked` explanation, `.npy` prepared artifacts, scalar cores,
  TIFF export distinction, one-task-per-node contract, worker cap, fairness, cache
  compatibility, disk implications, unique logs, failure/resume, two-tier acceptance,
  and explicit affine/multiple-worker non-goals.

  Require each example to parse and explicitly declare its intended mode. Keep the
  sequential example coarse, local-Dask example fine, and 512-GB SLURM example fine.

- [ ] **Step 2: Run documentation tests and confirm red**

  ```bash
  uv run pytest tests/unit/test_phase2_documentation.py \
    tests/unit/test_phase2_examples.py tests/unit/test_repository_policy.py -q
  ```

  Expected: new policy assertions fail because the page and example fields are
  absent.

- [ ] **Step 3: Write user documentation**

  Document planning, execution, output tree, restart examples, resource behavior,
  interpretation of prepared arrays, internal versus public formats, and the
  production acceptance boundary. Do not imply that fine mode is production-ready
  until Task 13 passes.

- [ ] **Step 4: Update every relevant `AGENTS.md`**

  Root guidance records the opt-in boundary and scientific invariants. Registration
  guidance owns prepared arrays and chunk assembly. Workflow guidance owns immutable
  fine tasks and dynamic coordinator expansion. Execution guidance owns fairness,
  preflight-gated slots, and cleanup. Test/HPC guidance requires two-tier parity,
  placement, failure, cancellation, and performance evidence. Documentation guidance
  prevents premature production claims.

- [ ] **Step 5: Run and commit documentation**

  ```bash
  uv run pytest tests/unit/test_phase2_documentation.py \
    tests/unit/test_phase2_examples.py tests/unit/test_repository_policy.py -q
  uv run pre-commit run --files README.md docs/fine-grained-execution.md \
    docs/workflow-guide.md docs/registration-artifacts.md docs/index.md \
    docs/myst.yml examples/execution-sequential.yaml \
    examples/execution-local-dask.yaml examples/execution-slurm-512gb.yaml \
    AGENTS.md src/cyexm/AGENTS.md src/cyexm/runtime/AGENTS.md \
    tests/AGENTS.md tests/unit/AGENTS.md tests/hpc/AGENTS.md docs/AGENTS.md
  ```

  Expected: documentation, example, policy, and pre-commit checks pass.

  ```bash
  git add README.md docs examples AGENTS.md src/cyexm/AGENTS.md \
    src/cyexm/runtime/AGENTS.md tests/AGENTS.md tests/unit/AGENTS.md \
    tests/hpc/AGENTS.md tests/unit/test_phase2_documentation.py \
    tests/unit/test_phase2_examples.py tests/unit/test_repository_policy.py
  git commit -m "docs: explain fine-grained registration"
  ```

---

### Task 12: Run portable gates and request code review

**Files:**
- Verify: complete repository
- Modify only when a focused failing gate proves an implementation defect

**Interfaces:**
- Produces a clean reviewable implementation commit range before real scheduler use.

- [ ] **Step 1: Run lock and environment checks**

  ```bash
  uv lock --check
  uv sync --locked --group dev --extra examples --extra hpc
  ```

  Expected: lock is unchanged and the project-backed environment synchronizes.

- [ ] **Step 2: Run all portable unit and integration tests**

  ```bash
  uv run pytest -m "unit and not hpc" tests/unit -v
  uv run pytest -m "integration and not hpc" tests/integration -v
  ```

  Expected: all selected tests pass with no scheduler contact.

- [ ] **Step 3: Run formatting, lint, typing, and policy**

  ```bash
  uv run ruff check .
  uv run ruff format --check .
  uv run basedpyright
  uv run pre-commit run --all-files
  ```

  Expected: every gate exits zero.

- [ ] **Step 4: Verify repository scope**

  ```bash
  git status --short
  git diff --check
  git log --oneline --decorate main..HEAD
  ```

  Expected: implementation worktree is clean, no SLURM output is tracked, and commit
  history matches Tasks 1-11.

- [ ] **Step 5: Request two-stage review**

  Use `superpowers:requesting-code-review` against the approved design and this plan.
  First review spec compliance and scientific invariants; second review code quality,
  API reuse, scheduler ownership, and test evidence. Address findings with
  `superpowers:receiving-code-review` and rerun affected plus broad gates.

---

### Task 13: Run exact-commit HPC acceptance and publish the audit

**Files:**
- Create: `tests/hpc/test_fine_grained_workflow.py`
- Modify: `tests/hpc/workflow_acceptance.py`
- Modify: `tests/hpc/deterministic_parity.py`
- Modify: `tests/unit/test_hpc_workflow_acceptance.py`
- Create after successful external run: `docs/fine-grained-acceptance.md`
- Modify after successful external run: `README.md`
- Modify after successful external run: `docs/fine-grained-execution.md`
- Modify after successful external run: `docs/index.md`
- Modify after successful external run: `docs/myst.yml`
- Modify after successful external run: `tests/unit/test_phase2_documentation.py`

**Interfaces:**
- Produces exact-commit representative multithreaded scientific equivalence, seeded
  one-thread strict parity, placement, durability, cleanup, and performance evidence.

- [ ] **Step 1: Add portable harness tests first**

  Unit tests require:

  - identical scientific fingerprints and coarse/fine artifact structure;
  - the existing Tier 1 metric absolute tolerance of `0.01`;
  - exact Tier 2 affine, warp, diagnostic, and channel comparisons;
  - at least two concurrent chunk job/host identities;
  - at least two concurrent channel job/host identities;
  - worker-cap and round-robin evidence;
  - injected chunk and channel failure/resume;
  - cancellation/resume and exact owned-job cleanup;
  - fixed-grid prepared geometry and owned-path confinement;
  - nonlinear-stage fine wall time no greater than `0.8 * coarse wall time`; and
  - wall time, task time, node-hours, peak memory, startup overhead, utilization, and
    chunk-duration summaries.

- [ ] **Step 2: Run harness unit tests and confirm red**

  ```bash
  uv run pytest tests/unit/test_hpc_workflow_acceptance.py \
    -k fine_grained -q
  ```

  Expected: failures identify absent fine-grained evidence builders and validators.

- [ ] **Step 3: Implement the explicit HPC harness**

  Reuse the existing caller-owned-root, scheduler firewall, dataset-copy, comparison,
  deterministic runner, job cleanup, and evidence-validation helpers. Do not duplicate
  the Phase 2E acceptance lifecycle. Add only fine-mode graph, task placement,
  failure injection, and timing records.

- [ ] **Step 4: Run portable harness and collection gates**

  ```bash
  uv run pytest tests/unit/test_hpc_workflow_acceptance.py -q
  uv run pytest tests/hpc/test_fine_grained_workflow.py \
    -m hpc --collect-only -q
  ```

  Expected: unit tests pass and collection submits no job.

- [ ] **Step 5: Commit the final pre-acceptance harness**

  ```bash
  git add tests/hpc/test_fine_grained_workflow.py \
    tests/hpc/workflow_acceptance.py tests/hpc/deterministic_parity.py \
    tests/unit/test_hpc_workflow_acceptance.py
  git commit -m "test: accept fine-grained SLURM execution"
  ```

- [ ] **Step 6: Obtain operator-approved inputs and run the exact commit**

  The operator supplies a copied, inactive multi-chunk manifest, coarse and fine
  execution profiles, and fresh evidence root through environment variables. Verify
  every path is outside the repository and active scientific outputs before running:

  ```bash
  : "${CYEXM_HPC_FINE_MANIFEST:?set approved copied manifest}"
  : "${CYEXM_HPC_COARSE_PROFILE:?set approved coarse profile}"
  : "${CYEXM_HPC_FINE_PROFILE:?set approved fine profile}"
  : "${CYEXM_HPC_FINE_EVIDENCE_ROOT:?set fresh evidence root}"
  test ! -e "$CYEXM_HPC_FINE_EVIDENCE_ROOT"
  uv run pytest tests/hpc/test_fine_grained_workflow.py \
    -m hpc -v --run-hpc
  ```

  Expected: both tiers, placement, fairness, durability, cleanup, and the 20% nonlinear
  wall-time reduction pass on one exact commit. Monitor only at approximately
  five-minute intervals for long-running jobs.

- [ ] **Step 7: Audit evidence before making a production claim**

  Verify exact commit, package versions, scientific and execution hashes, manifest
  identities, task/job/host placement, resources, native threads, comparison
  summaries, timing, node-hours, peak memory, scheduler overhead, failures, resume,
  cancellation, output ownership, queue cleanup, and evidence checksums. Preserve raw
  logs and arrays outside the repository.

- [ ] **Step 8: Publish the acceptance boundary**

  Write `docs/fine-grained-acceptance.md` with summarized, non-sensitive evidence and
  external checksums. Update README and the fine-grained guide from experimental to
  accepted only if every gate passed. State that acceptance is exact-commit,
  dataset-, profile-, cluster-, and thread-policy-specific and does not accept
  multiple workers per node, heterogeneous workers, or chunked affine registration.

- [ ] **Step 9: Run final documentation and portable verification**

  ```bash
  uv run pytest tests/unit/test_phase2_documentation.py \
    tests/unit/test_repository_policy.py -q
  uv run pytest -m "unit and not hpc" tests/unit -q
  uv run pytest -m "integration and not hpc" tests/integration -q
  uv run ruff check .
  uv run ruff format --check .
  uv run basedpyright
  uv run pre-commit run --all-files
  ```

  Expected: all gates pass after the acceptance audit.

- [ ] **Step 10: Commit the acceptance audit**

  ```bash
  git add docs/fine-grained-acceptance.md README.md \
    docs/fine-grained-execution.md docs/index.md docs/myst.yml \
    tests/unit/test_phase2_documentation.py
  git commit -m "docs: record fine-grained runtime acceptance"
  ```

  The commit body records exact commit-under-test, the two-tier outcomes, performance
  result, scheduler cleanup, and external evidence location/checksums without private
  credentials or generated scientific arrays.

---

## Final completion checklist

- [ ] Coarse mode remains the omitted/default execution behavior.
- [ ] Fine mode changes execution identity but not scientific fingerprints.
- [ ] `serial_chunked` remains the manifest strategy and serial numerical reference.
- [ ] Affine estimation remains whole-image and one task per round.
- [ ] Prepared float32 `.npy` arrays are atomic, exact, and regionally read.
- [ ] Coarse and fine local-warp requests are cache compatible.
- [ ] Every nonlinear chunk and channel is independently restartable.
- [ ] Assembly is canonical and completion-order independent.
- [ ] Submitted independent work settles after sibling failure.
- [ ] Sequential, local-Dask, and SLURM use the same fine graph.
- [ ] Workers never create clients or submit child tasks.
- [ ] SLURM candidates cannot run scientific work before preflight and slot activation.
- [ ] The global worker cap and fair cross-round admission are enforced.
- [ ] Operational evidence never authorizes scientific reuse.
- [ ] All relevant API documentation, user guides, examples, and `AGENTS.md` files are synchronized.
- [ ] Portable tests and repository gates pass.
- [ ] Exact-commit two-tier HPC acceptance and the 20% nonlinear wall-time gate pass.
- [ ] Production documentation states the tested boundary and future non-goals.
