# MIP Similarity Initial Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` and `superpowers:test-driven-development` in the
> isolated feature worktree. Checkboxes record fresh verification, not intent.

**Goal:** Correct seeded linear registration, add a strict constrained MIP
similarity initializer with complete provenance/QC, document it, configure the
spheroid data, and submit a fresh production workflow.

**Spec:**
`docs/superpowers/specs/2026-09-05-mip-similarity-initial-alignment-design.md`

## Reuse analysis

- Closest existing symbols: `InitialAlignmentSpec`, `InitialAlignmentPolicy`,
  `estimate_xy_rotation`, `create_xy_rotation_transform`,
  `ImageRegistration._load_or_resolve_initial_alignment`, `PreparedManifest`, and
  the shared initial-alignment provenance validators.
- Decision: extend those existing configuration, estimator, transform, artifact,
  prepared/fine, and status boundaries with a sibling `xy_similarity` policy.
- Rejected alternative: a separate preprocessing workflow would duplicate crop,
  source identity, artifact restart, transform application, fine-graph admission,
  and status lifecycles and could interpolate raw channels twice.
- Rejected alternative: adding scale fields to `xy_rotation` would make the public
  name scientifically false and complicate compatibility.
- Documentation updates: dataset schema, initial-alignment guide, preprocessing and
  artifact cross-references, checked-in example, and public model/runtime docstrings.

## Task 1: Correct complete seeded-linear transform handling

- [x] Add a unit regression in `tests/unit/test_linear.py` asserting that a fake
  ANTSpyX complete forward transform is read once and the supplied seed is not
  appended or composed.
- [x] Add a bounded real-ANTs regression in
  `tests/integration/test_registration_kernels.py` using a nonidentity seed and
  unequal ZYX shapes. Assert fixed output geometry and compare the returned scalar
  with applying the returned complete affine once.
- [x] Run the two focused tests and retain the expected RED failure caused by double
  seed composition.
- [x] Remove seed appending/composition from `linear.register_image`; require the
  single complete affine returned by linear ANTSpyX registration.
- [x] Run `tests/unit/test_linear.py` and the focused real-ANTs module GREEN.
- [x] Commit only this defect and its regressions.

## Task 2: Add strict similarity manifest/runtime policy

- [x] Add failing loader/model tests for similarity defaults, all overrides,
  canonical reconstruction, mutual exclusion, unknown keys, scale bounds, scale
  steps, and unchanged rotation omission/evidence.
- [x] Run `uv run pytest tests/unit/test_config_dataset.py -k
  'initial_alignment or similarity' -vv` and retain RED.
- [x] Add frozen `EstimatedXYSimilaritySpec` and runtime
  `EstimatedXYSimilarityPolicy`; extend `InitialAlignmentSpec` and
  `InitialAlignmentPolicy` with an exactly-one rotation/similarity invariant.
- [x] Extend strict YAML normalization and workflow translation without importing a
  scientific runtime into config/planning.
- [x] Run focused config, planning, import-contract, and fingerprint tests GREEN.
- [x] Commit the strict public configuration extension with the reuse analysis in
  the commit body.

## Task 3: Implement deterministic MIP similarity estimation and transform

- [x] Add failing synthetic estimator tests in
  `tests/unit/test_initial_alignment.py` for known uniform scale plus angle and
  translation, deterministic tie breaking, boundary rejection, confidence
  rejection, canonical round trip, and unchanged rotation evidence.
- [x] Add failing physical landmark tests in
  `tests/integration/test_initial_alignment_transforms.py` for uniform XY scale,
  anisotropic spacing, shifted origins, and matching nonidentity directions.
- [x] Run those focused tests and retain RED.
- [x] Refactor shared MIP candidate scoring without changing rotation output, add
  versioned coordinate-refinement similarity estimation, and implement one physical
  3-D XY similarity transform.
- [x] Run both focused modules GREEN and compare the synthetic similarity score
  against rotation-only initialization.
- [x] Commit estimator and transform behavior with policy-version evidence.

## Task 4: Integrate restart-safe artifacts, QC, fine admission, and provenance

- [x] Add failing whole-image tests for similarity JSON/MAT/PNG requests, realized
  scale, geometry and source/crop context, force/restart, rejected evidence, and
  downstream invalidation.
- [x] Add failing prepared/fine/status/run-record tests proving similarity evidence
  is parsed canonically and every artifact completion request is authenticated.
- [x] Run focused tests and retain RED.
- [x] Route both policy kinds through the existing
  `_load_or_resolve_initial_alignment` lifecycle, render QC with exact physical seed
  semantics, and update shared provenance validators/readers.
- [x] Run `test_whole_image.py`, `test_prepared.py`, fine-task, status, API, and
  rotation integration regressions GREEN.
- [x] Commit lifecycle and provenance changes.

## Task 5: Synchronize documentation/examples and run portable gates

- [x] Add `examples/estimated-similarity-dataset.yaml` with portable paths and
  constrained settings; add it to exact-loader documentation tests.
- [x] Update `docs/initial-alignment.md`, `docs/dataset-manifest.md`,
  `docs/registration-artifacts.md`, `docs/preprocessing.md`, applicable README links,
  and API docstrings. State MIP-only estimation, uniform XY scale, coordinate
  convention, confidence limits, restart authority, and the residual 3-D method.
- [x] Run example/documentation tests, formatting, and targeted lint/type checks.
- [x] Export the exact required uv cache and Python locations, then run `uv lock
  --check`, unit tests, portable integration tests, Ruff check/format,
  BasedPyright, and pre-commit.
- [x] Commit documentation/example synchronization and any gate-only corrections.

## Task 6: Configure and launch a fresh spheroid production run

- [x] Reinspect TIFF inventory/geometry and the successful smoke evidence. Preserve
  round 2 as fixed, retain existing per-round constrained angle ranges, use
  `scale_range: [0.90, 1.10]`, and change the 3-D linear method to `Similarity` high.
- [x] Create new dated dataset/execution YAMLs and a launch script in the spheroid
  directory. Do not mutate the separate historical run or delete its output.
- [x] Point the execution profile to a new output root or use a new dataset ID so no
  scientifically invalid cache can be reused.
- [x] Validate and plan with the exact feature-worktree executable; inspect the plan,
  scientific fingerprint, output targets, fine slot capacity, and empty user queue.
- [x] Submit the full-resolution workflow under a durable controller, capture PID,
  stdout/stderr path, Slurm job IDs, exact Git commit, manifests, and validation
  output, and confirm workers enter the queue.
- [x] Report submission state as running, not complete; identify commands and
  artifact paths for later terminal verification.
