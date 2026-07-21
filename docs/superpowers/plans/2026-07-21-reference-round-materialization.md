# Reference-Round Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize every declared reference-round channel exactly once under `rounds/round-001/channels`, use that prepared registration channel as the fixed image for all moving rounds, and remove workflow-managed per-round fixed-reference duplication.

**Architecture:** Manifest schema version 1 gains explicit reference registration/channel declarations. One immutable worker-executed `ReferenceTask` validates geometry, computes or reuses one shared crop, and atomically writes standardized TIFFs before the executor releases moving `RoundTask` objects; workflow registration then runs in explicit prepared-fixed mode.

**Tech Stack:** Python 3.12, PyYAML, pytest, ANTSpyX, NumPy, tifffile, Dask Distributed, dask-jobqueue, SLURMCluster, Ruff, BasedPyright, uv

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-07-21-reference-round-materialization-design.md` and is authoritative.
- Keep manifest `schema_version: 1`; reject the superseded `reference.channel` mapping and do not maintain dual parsing.
- Never move, rename, modify, or auto-discover source TIFFs.
- Reference channels must have exact equal shape and established geometry compatibility with the reference registration channel; never resample incompatible reference channels.
- Compute one exclusive ZYX crop from the reference registration channel and apply those exact bounds to every declared reference channel.
- TIFF is the only implemented output: clip to uint16 without normalization and embed the expected geometry through the existing I/O/backend boundary.
- Preserve source identities, atomic artifacts, completion records, scientific fingerprints, restart behavior, transform order `[warp, affine]`, inversion flags `[False, False]`, interpolation, masks, schedules, and native-thread policy.
- Native moving images may differ along every spatial axis; affine output must still match prepared fixed geometry before nonlinear work.
- Keep one executor/client/cluster lifecycle across the reference prerequisite and moving rounds; reference failure submits no moving task.
- Preserve direct Phase 1 defaults. Prepared-fixed behavior is explicit and workflow-owned.
- Do not implement a speculative format plugin or public `format:` field.
- Do not edit `/archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/spheroid-dataset.yaml` while jobs or a coordinator from its current workflow remain active. Never cancel or clean that run without new user authorization.
- Before every `uv` command, export:

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  ```
- Before Task 1, create the isolated worktree environment from the committed lockfile:

  ```bash
  uv sync --locked --group dev --extra examples --extra hpc
  ```

  Expected: `.venv` is created inside this worktree, dependencies resolve without
  modifying `uv.lock`, and all later `uv run` commands use that environment.

---

### Task 1: Replace the strict schema-1 reference declaration

**Files:**
- Modify: `src/cyexm/config/models.py`
- Modify: `src/cyexm/config/dataset.py`
- Modify: `src/cyexm/config/__init__.py`
- Modify: `tests/unit/test_config_dataset.py`
- Modify: `tests/unit/test_config_yaml.py`
- Modify: all test fixture constructors containing `ReferenceSpec(...)`
- Modify: `src/cyexm/config/AGENTS.md`

**Interfaces:**
- Consumes: `relative_posix_path`, `require_exact_keys`, and the existing strict safe-YAML loader.
- Produces: `ReferenceChannelSpec(file: PurePosixPath)` and `ReferenceSpec(round: int, registration_channel: PurePosixPath, channels: tuple[ReferenceChannelSpec, ...])`.

- [ ] **Step 1: Write failing schema tests**

  Replace the reference portion of the valid fixture in `tests/unit/test_config_dataset.py` with:

  ```yaml
  reference:
    round: 1
    registration_channel: fixed-488.tif
    channels:
      - file: fixed-488.tif
      - file: fixed-647.tif
  ```

  Assert the exact model:

  ```python
  assert manifest.reference == ReferenceSpec(
      round=1,
      registration_channel=PurePosixPath("fixed-488.tif"),
      channels=(
          ReferenceChannelSpec(PurePosixPath("fixed-488.tif")),
          ReferenceChannelSpec(PurePosixPath("fixed-647.tif")),
      ),
  )
  ```

  Add parametrized failures for legacy `channel`, empty channels, duplicate files,
  registration channel absent from the list, duplicate output basenames such as
  `a/fixed.tif` and `b/fixed.tif`, an unknown interpolation key, and unsafe paths.
  The legacy assertion must match both `reference.registration_channel` and
  `reference.channels` in the migration error.

- [ ] **Step 2: Run the schema tests and verify RED**

  Run:

  ```bash
  uv run pytest tests/unit/test_config_dataset.py tests/unit/test_config_yaml.py -q
  ```

  Expected: valid new YAML fails because `_REFERENCE_KEYS` still expects `channel`,
  and the new model symbols are absent.

- [ ] **Step 3: Implement the exact frozen models and parser**

  Add to `src/cyexm/config/models.py`:

  ```python
  @dataclass(frozen=True, slots=True)
  class ReferenceChannelSpec:
      file: PurePosixPath


  @dataclass(frozen=True, slots=True)
  class ReferenceSpec:
      round: int
      registration_channel: PurePosixPath
      channels: tuple[ReferenceChannelSpec, ...]
  ```

  In `dataset.py`, set exact keys to
  `{"round", "registration_channel", "channels"}`, parse each channel with the exact
  key `{"file"}`, require nonempty/unique files and basenames, and require exactly
  one registration-channel match. Extend direct-model revalidation by reconstructing
  this mapping rather than trusting dataclass instances. Export `ReferenceChannelSpec`
  from `cyexm.config`.

- [ ] **Step 4: Mechanically migrate in-repository model fixtures**

  Replace every old constructor with the explicit pattern:

  ```python
  ReferenceSpec(
      round=1,
      registration_channel=PurePosixPath("fixed.tif"),
      channels=(ReferenceChannelSpec(PurePosixPath("fixed.tif")),),
  )
  ```

  Add the new import in each affected test. Do not change scientific behavior in
  these fixtures beyond declaring the already-used fixed source.

- [ ] **Step 5: Document the strict schema rule and verify GREEN**

  Update `src/cyexm/config/AGENTS.md` to require explicit same-round reference
  declarations and forbid legacy dual parsing. Run:

  ```bash
  uv run pytest tests/unit/test_config_dataset.py tests/unit/test_config_yaml.py tests/unit/test_workflow_models.py -q
  uv run ruff check src/cyexm/config tests/unit/test_config_dataset.py tests/unit/test_config_yaml.py
  uv run basedpyright src/cyexm/config
  ```

  Expected: all selected tests, lint, and type checks pass.

- [ ] **Step 6: Commit the schema slice**

  ```bash
  git add src/cyexm/config src/cyexm/config/AGENTS.md tests
  git commit -m "feat: declare all reference-round channels"
  ```

### Task 2: Plan the reference task and shared output layout

**Files:**
- Modify: `src/cyexm/workflow/models.py`
- Modify: `src/cyexm/workflow/layout.py`
- Modify: `src/cyexm/workflow/planning.py`
- Modify: `src/cyexm/workflow/api.py`
- Modify: `tests/unit/test_workflow_models.py`
- Modify: `tests/unit/test_workflow_planning.py`
- Modify: `tests/unit/test_workflow_api.py`
- Modify: `src/cyexm/workflow/AGENTS.md`

**Interfaces:**
- Consumes: Task 1's `ReferenceSpec` and `ReferenceChannelSpec`, inspected `ImageHeader` records, `WorkflowLayout.round(...)`, and `OUTPUT_FORMAT_POLICY`.
- Produces: `ReferenceChannelTask`, `ReferenceTask`, `WorkflowPlan.reference_task`, prepared reference paths, and moving tasks whose `fixed_image` is the prepared registration output.

- [ ] **Step 1: Write failing model and layout tests**

  In `test_workflow_models.py`, construct and round-trip these exact frozen values:

  ```python
  @dataclass(frozen=True, slots=True)
  class ReferenceChannelTask:
      source: Path
      output: Path


  @dataclass(frozen=True, slots=True)
  class ReferenceTask:
      dataset_id: str
      round: int
      registration_channel_source: Path
      registration_channel_output: Path
      registration_directory: Path
      crop_bounds_path: Path | None
      log_path: Path
      voxel_spacing_xyz: PixelSpacingXYZ
      crop: bool
      native_threads: NativeThreadSpec
      channels: tuple[ReferenceChannelTask, ...]
      force: bool = False
  ```

  Add layout assertions for `round-001/channels`,
  `round-001/registration/reference_crop_indices.json`, and `round-001/logs/round-001.log`.

- [ ] **Step 2: Write failing planner tests for geometry and scientific identity**

  Add tests proving that planning:

  - records ordered reference channel declarations and prepared relative outputs;
  - fingerprints order, source identity, and crop policy;
  - sets every moving `RoundTask.fixed_image` to the prepared registration output;
  - expects every reference TIFF and the crop JSON only when crop is enabled;
  - creates no filesystem entries;
  - rejects a reference channel differing in each of shape, origin, spacing, and
    direction while continuing to accept arbitrary moving-source shapes.

  Geometry assertions must use exact shape plus the established TIFF compatibility
  tolerance for serialized origin/spacing/direction.

- [ ] **Step 3: Run the planner/model tests and verify RED**

  ```bash
  uv run pytest tests/unit/test_workflow_models.py tests/unit/test_workflow_planning.py -k 'reference or layout' -q
  ```

  Expected: missing `ReferenceTask`, missing layout paths, and the planner still
  resolves the original source TIFF directly as each moving fixed image.

- [ ] **Step 4: Implement workflow task models and layout**

  Add the dataclasses above, define:

  ```python
  type WorkflowTask = ReferenceTask | RoundTask
  type TaskStrategy = Literal[
      "reference-preparation", "whole_image", "serial_chunked"
  ]
  ```

  Change `RoundResult.strategy` to `TaskStrategy`. Add
  `reference_task: ReferenceTask` to `WorkflowPlan`; keep moving `tasks` sorted.
  Add `RoundLayout.reference_crop` returning
  `registration / "reference_crop_indices.json"`.

- [ ] **Step 5: Build and validate the immutable reference task**

  In `planning.py`, resolve every declared reference source, normalize missing
  embedded geometry to origin zero, manifest spacing, and identity direction, then
  validate each against the registration source. Construct ordered
  `ReferenceChannelTask` values with `RoundLayout.channel(source)`. Set
  `crop_bounds_path` to the reference crop path only when `registration.crop` is
  true. Point moving tasks to `registration_channel_output`.

  Extend output collision/containment checks and scientific mapping. Record
  `registration_channel`, `manifest_file`, `file`, `output`, ordered channels, and
  optional crop output under `reference`. Do not add executor values.

- [ ] **Step 6: Extend direct-plan validation**

  In `workflow/api.py`, validate exact `ReferenceTask` types, safe containment,
  registration membership, unique sources/outputs, matching scientific-plan paths,
  and `RoundTask.fixed_image == plan.reference_task.registration_channel_output`.
  Reject forged prepared paths before executor construction.

- [ ] **Step 7: Update guidance, verify, and commit**

  ```bash
  uv run pytest tests/unit/test_workflow_models.py tests/unit/test_workflow_planning.py tests/unit/test_workflow_api.py -q
  uv run ruff check src/cyexm/workflow tests/unit/test_workflow_models.py tests/unit/test_workflow_planning.py tests/unit/test_workflow_api.py
  uv run basedpyright src/cyexm/workflow
  git add src/cyexm/workflow tests/unit/test_workflow_models.py tests/unit/test_workflow_planning.py tests/unit/test_workflow_api.py
  git commit -m "feat: plan shared reference-round outputs"
  ```

### Task 3: Implement restart-safe reference preparation

**Files:**
- Create: `src/cyexm/registration/crop.py`
- Create: `src/cyexm/workflow/_reference_provenance.py`
- Create: `src/cyexm/workflow/reference_task.py`
- Create: `tests/unit/test_registration_crop.py`
- Create: `tests/unit/test_reference_task.py`
- Modify: `src/cyexm/registration/whole_image.py`
- Modify: `src/cyexm/registration/_request_policy.py`
- Modify: `tests/unit/fakes.py`
- Modify: `src/cyexm/io/AGENTS.md`
- Modify: `tests/unit/AGENTS.md`

**Interfaces:**
- Consumes: Task 2's `ReferenceTask`, `ArtifactStore`, `StageRequest`, `AntsRegistrationBackend`, `geometry_compatible`, `robust_bounding_box`, and `backend.write_clipped_tiff`.
- Produces: shared crop helpers, `reference_completion_evidence(...)`, and `run_reference_task(task: ReferenceTask) -> RoundResult`.

- [ ] **Step 1: Extract shared crop-policy tests before refactoring**

  Write tests for these exact functions in a new `registration/crop.py`:

  - `validate_bounds(value: object, shape: tuple[int, int, int]) -> Bounds3D`
  - `bounds_text(bounds: Bounds3D) -> str`
  - `read_bounds(path: Path, shape: tuple[int, int, int]) -> Bounds3D`
  - `estimate_bounds(backend: RegistrationBackend, image: object) -> Bounds3D`

  Cover canonical compact JSON, boolean/non-integer/out-of-range/empty rejection,
  robust parameters `(down_sampling=4, lower_pct=0.1, upper_pct=99.9)`, and unchanged
  whole-image crop behavior.

- [ ] **Step 2: Run crop tests and verify RED, then extract without behavior change**

  ```bash
  uv run pytest tests/unit/test_registration_crop.py tests/unit/test_whole_image.py -k crop -q
  ```

  Expected: the new module is absent. Move the existing private logic from
  `whole_image.py` into the focused module, import it back, and rerun until all crop
  tests pass before adding reference behavior.

- [ ] **Step 3: Write failing reference-task happy-path tests**

  Use three same-geometry `FakeImage` sources with distinct values. Assert one crop
  estimation, identical backend crop bounds for all channels, outputs in manifest
  order, clip-only uint16 values, derived cropped geometry, one crop completion
  record, one completion record per TIFF, reference strategy, native-thread evidence,
  and no undeclared output.

  Add a no-crop test asserting full images are still rewritten and
  `crop_bounds_path is None`.

- [ ] **Step 4: Write failing geometry and lifecycle tests**

  Add separate tests for mismatched shape, origin, spacing, and direction before the
  first commit; incomplete/corrupt cache entries; exact reuse with no writer calls;
  source mutation before commit; writer failure; completion-record failure; force
  staleness; output/source aliasing; and safe-path validation before native-thread
  configuration.

- [ ] **Step 5: Run reference tests and verify RED**

  ```bash
  uv run pytest tests/unit/test_reference_task.py -q
  ```

  Expected: import failure because `reference_task.py` and provenance helpers do not
  exist.

- [ ] **Step 6: Implement provenance requests and validation**

  In `_request_policy.py`, add exact versioned parameter builders for
  `reference-crop-v1` and `reference-channel-v1`. In
  `_reference_provenance.py`, expose:

  ```python
  @dataclass(frozen=True, slots=True)
  class ReferenceCompletionEvidence:
      artifacts: tuple[Path, ...]
      crop: ArtifactSnapshot | None
      channels: tuple[ArtifactSnapshot | None, ...]


  def reference_completion_evidence(
      task: ReferenceTask,
      source_geometries: tuple[ImageGeometry, ...],
      *,
      snapshot_reader: SnapshotReader,
      geometry_reader: GeometryReader,
  ) -> ReferenceCompletionEvidence:
      raise NotImplementedError
  ```

  Validate exact requests, artifact identities, crop upstream identity, expected
  output geometry, output policy, and source order.

- [ ] **Step 7: Implement `run_reference_task` minimally**

  Mirror `run_round_task` lifecycle without importing ANTs/NumPy/tifffile before
  `configure_native_threads`. Validate all identities and geometries first, reuse or
  atomically commit the crop bounds, crop each loaded image through the backend,
  commit each output with `ArtifactStore`, reopen through the backend, and return a
  round-1 `RoundResult(strategy="reference-preparation")`. On failure, return a
  structured failed result and preserve only independently valid commits.

- [ ] **Step 8: Verify and commit the reference component**

  ```bash
  uv run pytest tests/unit/test_registration_crop.py tests/unit/test_reference_task.py tests/unit/test_whole_image.py -q
  uv run ruff check src/cyexm/registration/crop.py src/cyexm/workflow/reference_task.py src/cyexm/workflow/_reference_provenance.py tests/unit/test_registration_crop.py tests/unit/test_reference_task.py
  uv run basedpyright src/cyexm/registration/crop.py src/cyexm/workflow/reference_task.py src/cyexm/workflow/_reference_provenance.py
  git add src/cyexm/registration/crop.py src/cyexm/registration/whole_image.py src/cyexm/registration/_request_policy.py src/cyexm/workflow/reference_task.py src/cyexm/workflow/_reference_provenance.py src/cyexm/io/AGENTS.md tests/unit
  git commit -m "feat: prepare reference channels once"
  ```

### Task 4: Add explicit prepared-fixed registration mode

**Files:**
- Modify: `src/cyexm/registration/whole_image.py`
- Modify: `src/cyexm/registration/chunked.py`
- Modify: `src/cyexm/workflow/_round_provenance.py`
- Modify: `src/cyexm/workflow/round_task.py`
- Modify: `src/cyexm/workflow/models.py`
- Modify: `tests/unit/test_whole_image.py`
- Modify: `tests/unit/test_chunked_registration.py`
- Modify: `tests/unit/test_round_task.py`
- Modify: `src/cyexm/registration/AGENTS.md`

**Interfaces:**
- Consumes: prepared reference TIFFs from Task 3 and existing Phase 1 registration/channel APIs.
- Produces: `RoundTask.prepared_fixed: bool`, `ImageRegistration(..., prepared_fixed=False)`, and workflow rounds that omit fixed-reference/fixed-crop artifacts while retaining moving crop and all transform evidence.

- [ ] **Step 1: Write failing direct-registration compatibility tests**

  Assert the default remains unchanged: `prepared_fixed` defaults false, fixed and
  moving crops both occur under `crop=True`, and `fixed_reference.tif` remains a
  valid standalone Phase 1 artifact.

  Add prepared-mode tests asserting:

  ```python
  registration = ImageRegistration(
      fixed,
      moving,
      output,
      imaging_round=2,
      crop=True,
      prepared_fixed=True,
      _backend=backend,
  )
  ```

  crops only moving, never creates/writes/reads `fixed_reference.tif` or
  `fixed_crop_indices_2.json`, and still enforces post-affine fixed geometry.

- [ ] **Step 2: Run registration tests and verify RED**

  ```bash
  uv run pytest tests/unit/test_whole_image.py tests/unit/test_chunked_registration.py -k prepared_fixed -q
  ```

  Expected: constructor rejects the unknown keyword.

- [ ] **Step 3: Implement prepared-fixed behavior**

  Add a validated boolean `prepared_fixed=False` to `ImageRegistration`; inherited
  chunked construction passes it through. In `register`, skip
  `_commit_fixed_reference` when true. In `_crop_if_requested`, return fixed unchanged
  and run only the moving crop when `crop and prepared_fixed`; retain both crops for
  the default mode. Include `prepared_fixed` in request parameters so caches cannot
  cross modes.

- [ ] **Step 4: Make workflow provenance conditional and update the round runner**

  Add `prepared_fixed: bool = True` to `RoundTask`, require true in workflow plan
  validation, pass it into registration, and change `_round_provenance` so workflow
  completion neither expects nor validates fixed-reference/fixed-crop artifacts.
  Continue expecting moving crop only when crop is enabled. Remove
  `fixed_reference` and `fixed_crop` from the workflow runtime path protocol without
  deleting them from standalone `RegistrationPaths`.

- [ ] **Step 5: Write and pass round-task scientific regressions**

  Prove that a round uses its prepared fixed path for registration and every
  dependent channel, produces affine/warp/registered/moving-crop/channel evidence,
  emits no duplicate fixed artifacts, and rejects a forged task with
  `prepared_fixed=False` before scientific work.

  ```bash
  uv run pytest tests/unit/test_whole_image.py tests/unit/test_chunked_registration.py tests/unit/test_round_task.py -q
  uv run ruff check src/cyexm/registration src/cyexm/workflow/round_task.py src/cyexm/workflow/_round_provenance.py tests/unit/test_whole_image.py tests/unit/test_chunked_registration.py tests/unit/test_round_task.py
  uv run basedpyright src/cyexm/registration src/cyexm/workflow/round_task.py
  ```

- [ ] **Step 6: Commit prepared-fixed registration**

  ```bash
  git add src/cyexm/registration src/cyexm/workflow/models.py src/cyexm/workflow/round_task.py src/cyexm/workflow/_round_provenance.py tests/unit/test_whole_image.py tests/unit/test_chunked_registration.py tests/unit/test_round_task.py
  git commit -m "feat: consume prepared workflow fixed images"
  ```

### Task 5: Stage reference preparation in sequential workflow, force, and status

**Files:**
- Create: `src/cyexm/workflow/task.py`
- Modify: `src/cyexm/execution/base.py`
- Modify: `src/cyexm/execution/sequential.py`
- Modify: `src/cyexm/workflow/api.py`
- Modify: `src/cyexm/workflow/status.py`
- Modify: `src/cyexm/workflow/__init__.py`
- Modify: `tests/unit/test_execution_sequential.py`
- Modify: `tests/unit/test_workflow_api.py`
- Modify: `tests/unit/test_workflow_status.py`
- Modify: `tests/unit/test_import_contract.py`
- Modify: `src/cyexm/workflow/AGENTS.md`

**Interfaces:**
- Consumes: `ReferenceTask`, `RoundTask`, `run_reference_task`, and `run_round_task`.
- Produces: `run_workflow_task(task: WorkflowTask) -> RoundResult` and `WorkflowExecutor.execute(reference_task, round_tasks)` with a hard prerequisite barrier.

- [ ] **Step 1: Write failing dispatcher and sequential barrier tests**

  Assert `run_workflow_task` dispatches only exact `ReferenceTask`/`RoundTask` types.
  For `SequentialExecutor.execute(reference, rounds)`, record calls and prove reference
  is first, successful reference releases sorted rounds, failed reference returns
  only its result, and cancellation submits nothing afterward.

- [ ] **Step 2: Run sequential tests and verify RED**

  ```bash
  uv run pytest tests/unit/test_execution_sequential.py tests/unit/test_workflow_api.py -k 'reference or prerequisite or force' -q
  ```

  Expected: current executor accepts only a tuple of moving rounds.

- [ ] **Step 3: Implement the scheduler-neutral staged protocol**

  In `base.py` define:

  ```python
  type WorkflowTaskRunner = Callable[[WorkflowTask], RoundResult]


  class WorkflowExecutor(Protocol):
      def execute(
          self,
          reference_task: ReferenceTask,
          round_tasks: tuple[RoundTask, ...],
      ) -> tuple[RoundResult, ...]:
          raise NotImplementedError
  ```

  Add the lazy dispatcher in `workflow/task.py`. Update sequential execution to run
  the reference once, stop on non-success, then preserve existing independent round
  collection.

- [ ] **Step 4: Implement force propagation in `run_workflow`**

  Treat known rounds as `{reference.round, *moving rounds}`. Construct forced tasks
  as follows:

  ```python
  force_reference = force_all or reference.round in force_rounds
  force_moving = force_all or force_reference
  reference_task = replace(plan.reference_task, force=force_reference)
  round_tasks = tuple(
      replace(task, force=force_moving or task.round in force_rounds)
      for task in pending_or_forced_tasks
  )
  ```

  When reference is forced, mark every planned moving artifact stale before executor
  construction. Moving-only force leaves reference records untouched. Always execute
  the reference task so it can validate/reuse before pending rounds.

- [ ] **Step 5: Add reference status parsing and validation**

  Extend the persisted scientific snapshot with a reference contract and use
  `reference_completion_evidence` to produce one `RoundStatus` before moving rounds.
  Preserve persistent states `pending`, `partial`, `complete`, and `invalid`. Remove
  workflow status expectations for moving-round fixed-reference/fixed-crop artifacts.

- [ ] **Step 6: Verify API/status/import behavior and commit**

  ```bash
  uv run pytest tests/unit/test_execution_sequential.py tests/unit/test_workflow_api.py tests/unit/test_workflow_status.py tests/unit/test_import_contract.py -q
  uv run ruff check src/cyexm/execution/base.py src/cyexm/execution/sequential.py src/cyexm/workflow tests/unit/test_execution_sequential.py tests/unit/test_workflow_api.py tests/unit/test_workflow_status.py
  uv run basedpyright src/cyexm/execution/base.py src/cyexm/execution/sequential.py src/cyexm/workflow
  git add src/cyexm/execution/base.py src/cyexm/execution/sequential.py src/cyexm/workflow tests/unit/test_execution_sequential.py tests/unit/test_workflow_api.py tests/unit/test_workflow_status.py tests/unit/test_import_contract.py
  git commit -m "feat: stage reference preparation before registration"
  ```

### Task 6: Preserve the prerequisite barrier in Local Dask

**Files:**
- Modify: `src/cyexm/execution/dask.py`
- Modify: `src/cyexm/execution/_dask_common.py`
- Modify: `tests/unit/test_execution_dask.py`
- Modify: `tests/integration/test_local_dask.py`
- Modify: `tests/integration/test_workflow_local_dask.py`
- Modify: `src/cyexm/execution/AGENTS.md`

**Interfaces:**
- Consumes: Task 5's staged executor protocol and `WorkflowTaskRunner`.
- Produces: one LocalCluster/client lifecycle, one reference future awaited before moving futures, and ordered round results/evidence.

- [ ] **Step 1: Write failing Local Dask staging tests**

  Use bounded synthetic runners to prove exactly one reference task is submitted
  first with `resources={"round_slot": 1}`, no moving key appears before it resolves,
  failure/cancellation prevents moving submission, success releases up to
  `min(moving rounds, configured workers)`, and client/cluster cleanup occurs once.

- [ ] **Step 2: Run focused Dask tests and verify RED**

  ```bash
  uv run pytest tests/unit/test_execution_dask.py -k 'reference or prerequisite' -q
  ```

  Expected: `LocalDaskExecutor.execute` accepts only moving tasks and submits them
  immediately.

- [ ] **Step 3: Implement one-lifecycle staged Dask execution**

  Scale to at least one worker and at most the configured/moving bound. Complete
  existing worker wait and preflight once. Submit and gather the reference future;
  on success, submit moving futures through the unchanged settling/retry helper. Keep
  one attempt namespace per round and return `(reference_result, *moving_results)`.
  Close client then cluster on every path.

- [ ] **Step 4: Add portable process integration coverage**

  Extend `test_workflow_local_dask.py` with a small real reference preparation plus
  moving workflow. Assert reference completion timestamps precede moving starts,
  sequential/local outputs and completion records match, and ordinary rerun preserves
  identities.

- [ ] **Step 5: Verify and commit Local Dask**

  ```bash
  uv run pytest tests/unit/test_execution_dask.py -q
  uv run pytest tests/integration/test_local_dask.py tests/integration/test_workflow_local_dask.py -q
  uv run ruff check src/cyexm/execution/dask.py src/cyexm/execution/_dask_common.py tests/unit/test_execution_dask.py tests/integration/test_local_dask.py tests/integration/test_workflow_local_dask.py
  uv run basedpyright src/cyexm/execution/dask.py src/cyexm/execution/_dask_common.py
  git add src/cyexm/execution src/cyexm/execution/AGENTS.md tests/unit/test_execution_dask.py tests/integration/test_local_dask.py tests/integration/test_workflow_local_dask.py
  git commit -m "feat: gate local rounds on reference preparation"
  ```

### Task 7: Preserve the prerequisite barrier and evidence in SLURM

**Files:**
- Modify: `src/cyexm/execution/slurm.py`
- Modify: `tests/unit/test_execution_slurm.py`
- Modify: `tests/unit/test_hpc_environment.py`
- Modify: `tests/unit/test_hpc_workflow_acceptance.py`
- Modify: `tests/hpc/workflow_acceptance.py`
- Modify: `tests/hpc/test_slurm_workflow.py`
- Modify: `tests/hpc/AGENTS.md`

**Interfaces:**
- Consumes: staged executor protocol, serialized `WorkflowTaskRunner`, existing SLURM worker/preflight/resource/evidence lifecycle.
- Produces: one SLURMCluster/client lifecycle with a recorded reference barrier and no moving submission before reference success.

- [ ] **Step 1: Write failing portable SLURM construction tests**

  Assert lazy construction, runner serialization before scheduler module load, scale
  to the moving concurrency bound with minimum one, all-worker preflight before the
  reference, one `round_slot` reference submission, barrier timestamps/evidence,
  no moving submission after reference failure, and unchanged client-before-cluster
  cleanup. Prove no portable test invokes a real scheduler client.

- [ ] **Step 2: Run focused SLURM tests and verify RED**

  ```bash
  uv run pytest tests/unit/test_execution_slurm.py -k 'reference or prerequisite or evidence' -q
  ```

  Expected: no reference task/evidence exists and moving futures submit immediately.

- [ ] **Step 3: Generalize serialized runner and stage submission**

  Rename `_SerializedRoundTaskRunner` to `_SerializedWorkflowTaskRunner`, accept the
  `WorkflowTask` union, and retain direct cloudpickle serialization before scheduler
  imports. After preflight, submit only reference, validate its result/evidence, then
  call the existing moving-settlement path. Preserve retry count zero inside Dask and
  executor-managed resubmission.

- [ ] **Step 4: Extend operational evidence and validation**

  Record reference requested/started/ended outcome, attempt count, task key,
  hostname/job ID, barrier outcome, and whether moving submission was released.
  Validate that every moving start is at or after successful reference completion.
  Reference failure evidence must show zero moving submissions. Keep scheduler
  allocation validation and exit-code behavior unchanged.

- [ ] **Step 5: Update opt-in HPC harness without running it**

  Extend the harness to expect the reference result and staged timestamps. Do not run
  any `-m hpc` command or contact SLURM during portable implementation. Add the exact
  future approved command to `tests/hpc/AGENTS.md`.

- [ ] **Step 6: Verify portable SLURM behavior and commit**

  ```bash
  uv run pytest tests/unit/test_execution_slurm.py tests/unit/test_hpc_environment.py tests/unit/test_hpc_workflow_acceptance.py -q
  uv run ruff check src/cyexm/execution/slurm.py tests/unit/test_execution_slurm.py tests/unit/test_hpc_environment.py tests/unit/test_hpc_workflow_acceptance.py tests/hpc
  uv run basedpyright src/cyexm/execution/slurm.py tests/hpc
  git add src/cyexm/execution/slurm.py tests/unit/test_execution_slurm.py tests/unit/test_hpc_environment.py tests/unit/test_hpc_workflow_acceptance.py tests/hpc
  git commit -m "feat: gate SLURM rounds on reference preparation"
  ```

### Task 8: Complete real-library, examples, documentation, and dataset acceptance

**Files:**
- Create: `tests/integration/test_reference_task.py`
- Modify: `tests/integration/test_workflow_round_task.py`
- Modify: `tests/integration/test_chunked.py`
- Modify: `tests/integration/AGENTS.md`
- Modify: `examples/spheroid-dataset.yaml`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: relevant child `AGENTS.md` files named in the design
- Modify: `tests/unit/test_phase2_examples.py`
- Modify: `tests/unit/test_phase2_documentation.py`
- Modify: `tests/unit/test_repository_policy.py`
- Conditionally modify after active-run check: `/archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/spheroid-dataset.yaml`

**Interfaces:**
- Consumes: all Tasks 1-7.
- Produces: real TIFF/ANTs evidence, synchronized examples/guidance, safe spheroid migration, and full portable acceptance.

- [ ] **Step 1: Add real reference TIFF integration tests**

  Create three deterministic same-geometry channels no larger than `16 x 32 x 32`,
  with distinct values and non-default origin/direction. Run real reference
  preparation with crop false and true. Assert exact output arrays after clipping,
  identical crop bounds, exact cropped scalar geometry, uint16 dtype, axes,
  completion records, and identity-preserving reuse. Add one geometry-mismatch hard
  failure.

- [ ] **Step 2: Extend real prepared-fixed registration coverage**

  Feed the prepared registration channel into a moving image differing along Z, Y,
  and X. Use bounded dry-run affine and instrument only the nonlinear boundary.
  Assert affine scalar/mask reach prepared fixed geometry, final scalar/warp/channel
  outputs match fixed geometry, and no moving registration directory contains
  `fixed_reference.tif` or fixed-crop JSON.

- [ ] **Step 3: Update checked-in schema/examples and documentation tests first**

  Change example YAML to:

  ```yaml
  reference:
    round: 1
    registration_channel: round_1_488_tubulin.tif
    channels:
      - {file: round_1_488_tubulin.tif}
  ```

  Add failing documentation/policy assertions for the reference task, output tree,
  crop semantics, force propagation, TIFF-only current policy/future-format boundary,
  migration error, and absence of workflow fixed-reference duplicates.

- [ ] **Step 4: Update README and layered guidance**

  Document only implemented behavior. Update root plus config, workflow, execution,
  registration, I/O, unit, integration, and HPC guidance exactly as routed by the
  design. Retain explicit statements that sources are copied, never moved, and that
  reference channels must share native geometry.

- [ ] **Step 5: Safely migrate the external spheroid manifest or defer it**

  Read-only check:

  ```bash
  squeue -u kdean -o '%.18i %.30j %.8T %.10M %R'
  ```

  If any `cyexm-20260721` job or its coordinator remains active, do not edit the
  external manifest; validate a temporary copied manifest under pytest `tmp_path`
  and report the deferred operator edit. If inactive, list actual `round_1_*.tif`
  files under the configured dataset root, explicitly add the scientifically intended
  channels to `reference.channels`, and preserve the selected tubulin registration
  channel. Never infer inclusion solely from a glob without reviewing filenames.

- [ ] **Step 6: Validate and plan the spheroid configuration without submission**

  After a safe external update, run:

  ```bash
  uv run cyexm workflow validate --json \
    --execution /archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/execution-slurm-512gb.yaml \
    /archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/spheroid-dataset.yaml
  uv run cyexm workflow plan --json \
    --execution /archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/execution-slurm-512gb.yaml \
    /archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/spheroid-dataset.yaml
  ```

  Expected: round 1 plus rounds 2-4, prepared reference outputs, exact source shapes,
  no executor construction, and no scheduler submission.

- [ ] **Step 7: Run full portable gates**

  ```bash
  uv lock --check
  uv run pytest -m "unit and not hpc" tests/unit -q
  uv run pytest -m "integration and not hpc" tests/integration -q
  uv run ruff check .
  uv run ruff format --check .
  uv run basedpyright
  uv run pre-commit run --all-files
  ```

  Expected: every command exits zero. Dask dashboard fallback-port warnings are
  acceptable; scheduler contact, silent skips, and new untracked SLURM logs are not.

- [ ] **Step 8: Review, commit, and record exact evidence**

  ```bash
  git diff --check
  git status --short --branch
  git add README.md AGENTS.md examples src tests docs
  git commit -m "docs: document shared reference-round workflow"
  ```

  Re-read every acceptance criterion in the design. Report exact test counts,
  spheroid migration/defer status, branch/commit, and explicit confirmation that no
  scheduler job was submitted or altered.
