# Different Source Shapes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support independently sized scalar 3-D fixed, registration-channel, and dependent-channel TIFFs while preserving ClearEx/ANTs affine resampling onto the fixed reference before nonlinear chunking.

**Architecture:** Planning and persisted-plan validation will validate each source shape independently and continue fingerprinting the exact per-source header. Registration will accept native fixed and moving grids, enforce exact fixed geometry immediately after affine resampling, retain the serial-chunked fixed-grid invariant, and require every transformed dependent channel to land on the fixed grid.

**Tech Stack:** Python 3.11+, pytest, ANTSpyX, NumPy, tifffile, Ruff, basedpyright, uv

## Global Constraints

- ClearEx commit `2311a873b783477add7391ff00245f76490147f0` is read-only and remains the scientific reference.
- Permit arbitrary positive source shapes along Z, Y, and X; do not add padding, cropping, truncation, or pre-registration resampling to equalize shapes.
- Preserve transform order `[warp, affine]`, inversion flags `[False, False]`, interpolation policy, voxel spacing, origin, direction, masks, schedules, and intensity policy.
- Apply affine transforms and dependent-channel transform chains with the fixed image as the reference grid.
- Require post-affine scalars and masks, nonlinear inputs and outputs, warps, and transformed channels to match exact fixed-reference geometry.
- Preserve source identities, exact source shapes, canonical scientific fingerprints, restart semantics, atomic writes, and native-thread policy.
- Do not modify or submit work to the active SLURM acceptance jobs or add their `slurm-10765567.out` and `slurm-10765568.out` files to version control.

---

### Task 1: Accept and fingerprint independent source shapes

**Files:**
- Modify: `tests/unit/test_workflow_planning.py`
- Modify: `tests/unit/test_workflow_status.py`
- Modify: `src/cyexm/workflow/planning.py`
- Modify: `src/cyexm/workflow/status.py`
- Modify: `src/cyexm/workflow/AGENTS.md`

**Interfaces:**
- Consumes: `ImageHeader.shape_zyx: tuple[int, int, int]`, `_header_contract_error(...)`, and the existing canonical scientific-plan source records.
- Produces: `_inspect_headers(...) -> dict[Path, ImageHeader]` with per-source validity only, and `_scientific_snapshot(...) -> _ScientificSnapshot` that accepts independently valid recorded shapes.

- [ ] **Step 1: Replace the planner shape-rejection regression with an unequal-shape acceptance and provenance regression**

  In `tests/unit/test_workflow_planning.py`, replace `test_plan_aggregates_header_failures_and_shape_incompatibility` with a test whose mocked headers assign distinct shapes to the fixed source, registration source, and dependent source, including differences along every spatial axis. Build the plan and assert the source mappings retain those exact `shape_zyx` values. Keep a separate malformed-axes test proving `ZCYX` still fails as a non-scalar header.

- [ ] **Step 2: Run the focused planner tests and verify the compatibility test fails**

  Run:

  ```bash
  uv run pytest tests/unit/test_workflow_planning.py -k 'unequal or header' -q
  ```

  Expected: the new unequal-shape test fails with `WorkflowPlanError` containing `must match reference`; malformed header tests continue to pass.

- [ ] **Step 3: Remove only the cross-source comparison in planning**

  Delete the `reference_path` / `reference_header` comparison loop from `_inspect_headers`. Retain `_header_contract_error`, aggregation of unreadable or malformed TIFF errors, embedded geometry-shape consistency, and manifest-spacing compatibility unchanged.

- [ ] **Step 4: Replace the persisted-plan rejection with acceptance and malformed-shape coverage**

  In `tests/unit/test_workflow_status.py`, replace `test_status_rejects_source_shape_incompatible_with_reference` with `test_status_accepts_independently_valid_source_shapes`. Assign distinct positive three-element shapes to the fixed and moving source records, write the canonical plan, call `inspect_workflow_status`, and assert the round remains `pending`. Add a parametrized malformed-shape test covering a zero dimension, a boolean dimension, and a non-three-element shape, each still raising `WorkflowPlanError` mentioning `shape_zyx`.

- [ ] **Step 5: Run the focused status tests and verify the acceptance test fails**

  Run:

  ```bash
  uv run pytest tests/unit/test_workflow_status.py -k 'source_shape or independently_valid' -q
  ```

  Expected: unequal valid shapes fail with the current reference-shape error; malformed shapes fail under the existing per-source parser.

- [ ] **Step 6: Remove only persisted cross-source equality validation**

  Delete `incompatible_shapes` and its error branch from `src/cyexm/workflow/status.py`. Do not change `_parse_source`, source use/uniqueness validation, canonical serialization, or fingerprint computation.

- [ ] **Step 7: Document the planning contract**

  Add guidance to `src/cyexm/workflow/AGENTS.md` stating that every source shape is independently positive scalar ZYX evidence, native source shapes may differ along every axis, and fixed-grid equality is an output/stage-boundary invariant rather than a planner input invariant.

- [ ] **Step 8: Run and commit the workflow slice**

  Run:

  ```bash
  uv run pytest tests/unit/test_workflow_planning.py tests/unit/test_workflow_status.py -q
  uv run ruff check src/cyexm/workflow/planning.py src/cyexm/workflow/status.py tests/unit/test_workflow_planning.py tests/unit/test_workflow_status.py
  ```

  Expected: all tests and Ruff checks pass.

  Commit:

  ```bash
  git add src/cyexm/workflow/planning.py src/cyexm/workflow/status.py src/cyexm/workflow/AGENTS.md tests/unit/test_workflow_planning.py tests/unit/test_workflow_status.py
  git commit -m "feat: accept independent workflow source shapes"
  ```

### Task 2: Transform independently sized dependent channels onto the fixed grid

**Files:**
- Modify: `tests/unit/test_channel_transform.py`
- Modify: `tests/integration/test_channel_transform.py`
- Modify: `src/cyexm/registration/channels.py`
- Modify: `src/cyexm/registration/AGENTS.md`

**Interfaces:**
- Consumes: `transform_channel(...) -> Path`, `RegistrationBackend.apply_transform_chain(fixed=..., moving=..., affine_path=..., warp_path=..., interpolation=...)`, and `geometry_compatible`.
- Produces: channel transformation from any valid native moving shape to exact fixed `ImageGeometry`, with unchanged completion-record and atomic-commit behavior.

- [ ] **Step 1: Convert the unit rejection test into a fixed-reference invocation test**

  Replace `test_transform_channel_rejects_moving_shape_mismatch_before_transform` with a test using fixed `(8, 8, 8)` and moving `(7, 9, 6)`. Assert one transform-chain call occurs, the fixed image supplied to the backend is the fixed object, the moving image is the unequal source object, and the committed result geometry equals the fixed geometry. Retain `test_transform_channel_rejects_result_that_loses_fixed_geometry` unchanged.

- [ ] **Step 2: Run the two geometry unit tests and verify the acceptance test fails**

  Run:

  ```bash
  uv run pytest tests/unit/test_channel_transform.py -k 'moving_shape or result_that_loses' -q
  ```

  Expected: the unequal-shape invocation fails at the current pre-transform shape check; wrong result geometry still fails.

- [ ] **Step 3: Remove the native shape equality check**

  In `transform_channel`, continue loading and fingerprinting fixed and moving inputs, but remove `moving_geometry` and the `fixed_geometry.shape != moving_geometry.shape` rejection. Keep the fixed geometry in the `StageRequest`, pass `fixed=fixed` to the backend, and retain the exact output geometry guard before commit and during cache reuse.

- [ ] **Step 4: Add a real-ANTs unequal-shape integration regression**

  Add a test in `tests/integration/test_channel_transform.py` with small deterministic fixed and moving arrays differing along Z, Y, and X. Write fixed TIFF metadata with a non-default origin and direction, create an identity affine and a zero warp on the fixed grid, call `transform_channel`, and assert the output array shape and loaded `ImageGeometry` exactly equal the fixed reference. Limit numerical comparison to the physically overlapping identity-transformed region using the documented `rtol=1e-4`, `atol=1e-5`; do not make a registration-quality claim.

- [ ] **Step 5: Document dependent-channel behavior**

  Update `src/cyexm/registration/AGENTS.md` to state that dependent channels may differ from both the reference and registration channel, transform application starts from the native channel grid with fixed as the reference, and promotion requires exact fixed geometry.

- [ ] **Step 6: Run and commit the channel slice**

  Run:

  ```bash
  uv run pytest tests/unit/test_channel_transform.py -q
  uv run pytest tests/integration/test_channel_transform.py -q
  uv run ruff check src/cyexm/registration/channels.py tests/unit/test_channel_transform.py tests/integration/test_channel_transform.py
  ```

  Expected: unit and real-library integration tests pass; equal-shape reuse and failure semantics remain green.

  Commit:

  ```bash
  git add src/cyexm/registration/channels.py src/cyexm/registration/AGENTS.md tests/unit/test_channel_transform.py tests/integration/test_channel_transform.py
  git commit -m "feat: resample unequal channels onto the fixed grid"
  ```

### Task 3: Enforce fixed geometry immediately after affine resampling

**Files:**
- Modify: `tests/unit/test_whole_image.py`
- Modify: `tests/integration/test_chunked.py`
- Modify: `src/cyexm/registration/whole_image.py`
- Modify: `tests/integration/AGENTS.md`

**Interfaces:**
- Consumes: `ImageRegistration._load_or_estimate_affine(...) -> tuple[object, object, object]`, `backend.geometry(...) -> ImageGeometry`, and the existing `ChunkedImageRegistration._estimate_nonlinear` fixed-grid guard.
- Produces: an affine-stage contract that returns scalar and mask on exact fixed geometry or raises `RegistrationStageError` before metrics, nonlinear estimation, or chunk creation.

- [ ] **Step 1: Add unit failures for estimated and cached wrong-grid affine results**

  In `tests/unit/test_whole_image.py`, add focused fake-backend tests for both branches of `_load_or_estimate_affine`: one backend returns an estimated `moving_linear` with a shape or origin differing from fixed; another reconstructs a cached affine onto the wrong grid. Assert `RegistrationStageError` mentions affine result geometry and that no nonlinear warp or registered artifact is committed.

- [ ] **Step 2: Run the new unit tests and verify they fail**

  Run:

  ```bash
  uv run pytest tests/unit/test_whole_image.py -k 'affine_result_geometry' -q
  ```

  Expected: both tests currently proceed beyond the affine boundary instead of raising the specific geometry error.

- [ ] **Step 3: Add one shared affine-result validator**

  Add a private `ImageRegistration` helper that compares `backend.geometry(moving_linear)` and `backend.geometry(moving_linear_mask)` to `backend.geometry(fixed)` with `geometry_compatible`. Call it before returning from both the cached and newly estimated affine branches. Raise `RegistrationStageError` identifying whether the scalar or mask violates fixed-reference geometry. Do not weaken the existing chunked check.

- [ ] **Step 4: Add the real-ANTs serial-chunked regression**

  In `tests/integration/test_chunked.py`, create deterministic native fixed and moving TIFFs no larger than `16 x 32 x 32`, with dimensions different along all three axes. Run `ChunkedImageRegistration` using `dry run`, a bounded chunk grid, and one native thread. Instrument only the nonlinear stage boundary so the test records the fixed and post-affine moving geometries while leaving real ANTs affine application, artifact persistence, reconstruction, and final transform application intact. Assert every nonlinear call receives equal fixed-reference geometry, the final registered TIFF has exact fixed geometry, and the full warp geometry equals fixed.

- [ ] **Step 5: Document the integration invariant**

  Update `tests/integration/AGENTS.md` to require an unequal-native-shape regression whenever affine, nonlinear, chunking, or channel-grid behavior changes, and to assert exact fixed geometry after the first resampling boundary.

- [ ] **Step 6: Run and commit the affine/chunk slice**

  Run:

  ```bash
  uv run pytest tests/unit/test_whole_image.py tests/unit/test_chunked_registration.py -q
  uv run pytest tests/integration/test_chunked.py -q
  uv run ruff check src/cyexm/registration/whole_image.py tests/unit/test_whole_image.py tests/integration/test_chunked.py
  ```

  Expected: affine guards, existing serial fixed-grid rejection, and real-ANTs unequal-shape flow pass.

  Commit:

  ```bash
  git add src/cyexm/registration/whole_image.py tests/unit/test_whole_image.py tests/integration/test_chunked.py tests/integration/AGENTS.md
  git commit -m "fix: enforce fixed geometry after affine resampling"
  ```

### Task 4: Update repository guidance and run scientific acceptance

**Files:**
- Modify: `AGENTS.md`
- Modify: `tests/AGENTS.md`
- Verify: `/archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/spheroid-dataset.yaml`
- Verify: `/archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/execution-slurm-512gb.yaml`

**Interfaces:**
- Consumes: the planner, status, registration, and channel contracts completed in Tasks 1-3.
- Produces: repository-wide scientific guidance and no-submit dataset validation evidence for rounds 2, 3, and 4.

- [ ] **Step 1: Update root and test guidance**

  Add to root `AGENTS.md` that native source shapes may differ independently, the affine stage is the first fixed-grid boundary, and all later scalar/mask/warp/channel products must use exact fixed-reference geometry. Add to `tests/AGENTS.md` that grid-contract changes require fixed and moving shapes differing along all three axes plus exact geometry assertions.

- [ ] **Step 2: Run focused and broad portable verification**

  Export project-backed uv paths, then run:

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  uv run pytest tests/unit -q
  uv run pytest tests/integration -m 'integration and not hpc' -q
  uv run ruff check .
  uv run ruff format --check .
  uv run basedpyright
  ```

  Expected: all portable tests, lint, format, and type checks pass. If the repository’s configured suite excludes a known external HPC acceptance artifact, report that separately rather than altering the active jobs or logs.

- [ ] **Step 3: Validate and plan the approved spheroid configuration without scheduler access**

  Run:

  ```bash
  uv run cyexm workflow validate --json --execution /archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/execution-slurm-512gb.yaml /archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/spheroid-dataset.yaml
  uv run cyexm workflow plan --json --execution /archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/execution-slurm-512gb.yaml /archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/spheroid-dataset.yaml
  ```

  Expected: both commands exit zero, report rounds 2, 3, and 4, preserve each source’s actual shape, and submit no scheduler jobs.

- [ ] **Step 4: Confirm repository scope and commit guidance**

  Run:

  ```bash
  git status --short
  git diff --check
  ```

  Expected: only intended tracked changes plus the pre-existing active `slurm-10765567.out` and `slurm-10765568.out` files are present; no SLURM log is staged.

  Commit:

  ```bash
  git add AGENTS.md tests/AGENTS.md
  git commit -m "docs: define unequal source grid invariants"
  ```

### Task 5: Final review and evidence capture

**Files:**
- Verify: `docs/superpowers/specs/2026-07-21-different-source-shapes-design.md`
- Verify: all files changed in Tasks 1-4

**Interfaces:**
- Consumes: all prior task deliverables.
- Produces: reviewed implementation that satisfies every acceptance criterion without modifying active scheduler state.

- [ ] **Step 1: Review the implementation against every design acceptance criterion**

  Confirm arbitrary positive ZYX source shapes, fixed-reference affine resampling, dependent-channel independence, exact persisted output geometry, deterministic per-source evidence, focused real-ANTs coverage, portable gates, spheroid no-submit planning, and updated guidance.

- [ ] **Step 2: Inspect the final diff and commit history**

  Run:

  ```bash
  git diff main...HEAD --check
  git log --oneline --decorate -8
  git status --short --branch
  ```

  Expected: coherent focused commits, no whitespace errors, and only the two known untracked active SLURM logs outside version control.

- [ ] **Step 3: Record exact verification outcomes in the handoff**

  Report test counts and commands, spheroid round/shape evidence, any environmental skips or failures, current branch/commit, and explicit confirmation that no scheduler job was submitted or altered.
