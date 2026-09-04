# XY Rotation Initial Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for execution tracking.

**Goal:** Add optional explicit and target-blind automatic XY rotation
initialization to each moving round while preserving the established raw-data,
affine, nonlinear, dependent-channel, restart, and executor contracts.

**Architecture:** A strict per-round manifest value is converted to a low-level
initial-alignment policy. The shared linear-preparation lifecycle resolves and
persists a physical-space seed, supplies it to ANTs linear registration, and writes
the composed complete affine to the existing `linear_transform_{round}.mat` path.
Automatic estimation uses only fixed and moving registration channels. Final raw
registration images and dependent channels continue to be interpolated once through
the existing `[warp, affine]` consumer chain.

**Tech stack:** Python 3.12, NumPy, SciPy, scikit-image, ANTSpyX 0.6.x, tifffile,
PyYAML, pytest, Ruff, basedpyright, pre-commit, Jupyter Book/MyST.

**Approved design:**
`docs/superpowers/specs/2026-09-04-xy-rotation-initial-alignment-design.md`

## Global constraints

- Fixed and moving roles never change.
- Public spacing is micrometers in XYZ; NumPy arrays are ZYX.
- The angle is a correction applied to moving about physical positive Z through the
  cropped moving-image center, using the right-hand rule.
- Automatic estimation consumes only the fixed and moving registration channels.
- Do not materialize pre-rotated source or dependent-channel TIFFs.
- Preserve affine-before-nonlinear estimation and `[warp, affine]` application order.
- `linear_transform_{round}.mat` remains the complete linear moving-to-fixed
  transform, including the seed.
- Preserve raw greater-than-zero diagnostic masks and the existing intensity-feature
  preprocessing boundary.
- Planning remains read-only; the artifact store remains restart authority.
- Keep scientific settings out of execution profiles.
- Keep NumPy, SciPy, scikit-image, and ANTs imports out of module scope in
  registration policy modules.
- Do not modify ClearEx or introduce a ClearEx runtime dependency.
- Preserve the unrelated untracked `build/` directory.
- Do not launch archived-data or HPC work without a separate operator request.
- Before every `uv` command, export exactly:

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  ```

---

## Reuse analysis

- Extend `RoundSpec` and `_round` in `cyexm.config` because they already own strict,
  immutable, per-moving-round scientific intent. Do not add execution-profile fields
  or a second YAML loader.
- Extend `RoundTask`, `_round_tasks`, and `_scientific_mapping` because they are the
  existing immutable planning and fingerprint surfaces. Do not estimate angles while
  planning.
- Extend `ImageRegistration._prepare_linear_state` because it is the single shared
  preparation path used by `ImageRegistration`, `ChunkedImageRegistration`, coarse
  workflow tasks, and fine prepared tasks. Do not create a rotation-aware registration
  class or strategy.
- Extend `RegistrationBackend.linear` and the existing transform helpers because
  they own configured ANTs invocation, physical geometry, and transform persistence.
  Keep scientific imports deferred.
- Extend `RegistrationPaths`, `ArtifactStore`, `StageRequest`, and request-policy
  helpers for strict initialization artifacts. Do not add a second cache or sidecar
  lifecycle.
- Persist a composed affine at the existing public affine path so
  `apply_transform_chain`, `transform_channel`, notebooks, and external consumers
  keep their exact two-transform interface.
- Extend `PreparedManifest` and the existing fine-task producer/reader paths so
  rotation initialization is performed once before task fan-out. Do not add a
  worker-owned estimation graph.
- Extend `_round_provenance`, workflow status, and execution validation together;
  they are the authoritative completion readers. Do not infer seed validity from the
  final affine alone.
- Extend `ImageRegistration` and `register_round` with the existing low-level policy
  value. Do not add a wrapper that duplicates validation, registration, or artifact
  ownership.
- Reuse SciPy rotation/filtering and scikit-image phase cross-correlation already
  available through declared base dependencies. Render the fixed-size QC raster with
  NumPy/scikit-image so no new base or optional dependency is required.

## File and responsibility map

- Modify `src/cyexm/config/models.py`, `dataset.py`, and `__init__.py`: strict
  discriminated config values, per-round field, and public config exports.
- Create `src/cyexm/registration/initial_alignment.py`: runtime policies, estimator,
  canonical evidence, confidence decisions, and dependency-light QC raster.
- Modify `src/cyexm/registration/backend.py`, `linear.py`, and `transforms.py`: seed
  construction, initial-transform ANTs call, affine composition, and physical tests.
- Modify `src/cyexm/registration/_request_policy.py`, `whole_image.py`, and
  `__init__.py`: artifact naming, requests, lifecycle integration, and public API.
- Modify `src/cyexm/registration/prepared.py`: strict prepared formats 5 through 7.
- Modify `src/cyexm/workflow/models.py`, `planning.py`, `round_task.py`,
  `fine_models.py`, `fine_task.py`, `_execution_validation.py`,
  `_round_provenance.py`, `status.py`, and `api.py`: immutable propagation,
  canonical plans, planned artifacts, fine/coarse parity, and authoritative status.
- Modify focused tests named in each task before implementation.
- Create `docs/initial-alignment.md`; modify `docs/dataset-manifest.md`,
  `preprocessing.md`, `registration-foundations.md`, `registration-artifacts.md`,
  `outputs-and-recovery.md`, `architecture.md`, `getting-started.md`, and `myst.yml`.
- Create `examples/explicit-rotation-dataset.yaml` and
  `examples/estimated-rotation-dataset.yaml`.

---

### Task 1: Strict manifest and immutable config models

**Files:**
- Modify: `tests/unit/test_config_dataset.py`
- Modify: `src/cyexm/config/models.py`
- Modify: `src/cyexm/config/dataset.py`
- Modify: `src/cyexm/config/__init__.py`

**Interfaces:**
- `ExplicitXYRotationSpec(mode="explicit", angle_deg=float)`
- `EstimatedXYRotationSpec(mode="estimate", search_range_deg, coarse_step_deg,
  refinement_step_deg, minimum_score_gain, minimum_peak_margin)`
- `InitialAlignmentSpec.xy_rotation: XYRotationSpec`
- `RoundSpec.initial_alignment: InitialAlignmentSpec | None = None`

- [ ] **Step 1: Write strict parser and direct-model tests**

Add complete fixtures for omission, explicit normalization, estimated defaults, all
estimated overrides, and direct `_validate_dataset_manifest` reconstruction. Assert:

```python
assert manifest.rounds[0].initial_alignment is None

rotation = explicit.rounds[0].initial_alignment.xy_rotation
assert rotation == ExplicitXYRotationSpec(mode="explicit", angle_deg=27.7)

rotation = estimated.rounds[0].initial_alignment.xy_rotation
assert rotation == EstimatedXYRotationSpec(
    mode="estimate",
    search_range_deg=(-45.0, 45.0),
    coarse_step_deg=1.0,
    refinement_step_deg=0.1,
    minimum_score_gain=0.01,
    minimum_peak_margin=0.01,
)
```

Parameterize rejection of booleans, strings, NaN/infinity, angles outside
`[-180, 180]`, reversed/equal ranges, nonpositive steps/thresholds, refinement not
smaller than coarse, mixed-mode fields, missing required explicit angle, unknown
keys at both new mapping levels, and initialization on `reference`.

- [ ] **Step 2: Verify RED**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run python -m pytest tests/unit/test_config_dataset.py -k initial_alignment -vv
```

Expected: imports or assertions fail because the models and round key do not exist.

- [ ] **Step 3: Implement the minimal strict models and parser**

Add frozen, slotted config values and a union alias. Parse by exact mode-specific
keys. Keep parsing helpers in `dataset.py`; do not import scientific libraries. The
normalized model definitions are:

```python
@dataclass(frozen=True, slots=True)
class ExplicitXYRotationSpec:
    mode: Literal["explicit"]
    angle_deg: float


@dataclass(frozen=True, slots=True)
class EstimatedXYRotationSpec:
    mode: Literal["estimate"]
    search_range_deg: tuple[float, float] = (-45.0, 45.0)
    coarse_step_deg: float = 1.0
    refinement_step_deg: float = 0.1
    minimum_score_gain: float = 0.01
    minimum_peak_margin: float = 0.01


type XYRotationSpec = ExplicitXYRotationSpec | EstimatedXYRotationSpec


@dataclass(frozen=True, slots=True)
class InitialAlignmentSpec:
    xy_rotation: XYRotationSpec
```

Extend both whole-image and serial-chunked round key sets with optional
`initial_alignment`, preserving all existing exact-key behavior. Reconstruct the
normalized YAML mapping inside `_validate_dataset_manifest` so direct models cannot
bypass any cross-field rule. Export all four public names from `cyexm.config`.

- [ ] **Step 4: Verify GREEN and unchanged omission behavior**

Run the Step 2 command, then:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run python -m pytest tests/unit/test_config_dataset.py -vv
```

Expected: all config tests pass, including existing manifests without the field.

- [ ] **Step 5: Commit Task 1**

Stage only the four Task 1 files. Commit with:

```bash
git commit -m "feat: define rotation initialization manifest policy" \
  -m "Reuse analysis: extends RoundSpec and the strict dataset loader for per-round scientific intent; adds no execution-profile or parallel YAML lifecycle."
```

---

### Task 2: Deterministic target-blind estimator and QC evidence

**Files:**
- Create: `tests/unit/test_initial_alignment.py`
- Create: `src/cyexm/registration/initial_alignment.py`

**Interfaces:**
- `ExplicitXYRotationPolicy`
- `EstimatedXYRotationPolicy`
- `XYRotationEstimate`
- `estimate_xy_rotation(fixed_zyx, moving_zyx, policy) -> XYRotationEstimate`
- `initial_alignment_parameters(policy, result) -> dict[str, object]`
- `render_initial_alignment_qc` returns a `uint8` YX array.

- [ ] **Step 1: Write estimator success tests**

Use an asymmetric synthetic volume containing three unequal Gaussian objects. Create
moving arrays with known `-30.0`, `+7.5`, and `0.0` degree corrections plus known XY
translations. Assert determinism and:

```python
estimate = estimate_xy_rotation(fixed, moving, policy)
assert estimate.accepted is True
assert estimate.angle_deg == pytest.approx(expected, abs=policy.refinement_step_deg)
assert estimate.translation_yx_px == pytest.approx(expected_shift, abs=1.0)
assert estimate.policy == "mip-gradient-phase-correlation-xy-rotation-v1"
assert estimate.coarse_scores
assert estimate.refined_scores
```

Add tie-breaking tests and prove changing a dependent-channel-looking array has no
API route into the estimator.

- [ ] **Step 2: Write estimator failure and evidence tests**

Assert constant/nonfinite input, no overlap, boundary winner, insufficient gain, and
insufficient independent-peak margin each return or raise a typed rejected result
with one stable reason. Assert canonical evidence contains only built-in JSON types,
complete curves, zero-angle score, winning shift, overlap, thresholds, policy, and
`accepted`.

Assert QC is deterministic `uint8`, has an exact documented canvas shape, and shows
nonzero fixed, before, after, overlay, and score-curve regions. It must not require
Matplotlib.

- [ ] **Step 3: Verify RED**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run python -m pytest tests/unit/test_initial_alignment.py -vv
```

Expected: module import fails.

- [ ] **Step 4: Implement the estimator as pure scientific policy**

Keep heavy imports inside call boundaries:

```python
def _scientific_runtime() -> tuple[object, object, object]:
    import numpy as np
    from scipy import ndimage
    from skimage.registration import phase_cross_correlation

    return np, ndimage, phase_cross_correlation
```

Implement the exact approved pipeline: Z MIP, finite 1st/99th percentile scaling,
Gaussian sigma 2, gradient magnitude, expanded zero canvas, phase correlation,
valid-overlap Pearson correlation, inclusive coarse/refined candidate generation,
and deterministic tie breaks. Compare independent peaks only when separated by at
least one coarse step. Normalize a winner within one refinement step of zero as a
valid zero-rotation seed without requiring positive gain over itself, so
translation-only initialization remains valid. Reject in the exact design-specified
order so one input always has one reproducible failure reason.

Represent score rows as immutable built-in values, for example:

```python
@dataclass(frozen=True, slots=True)
class RotationScore:
    angle_deg: float
    translation_yx_px: tuple[float, float]
    correlation: float
    valid_overlap_fraction: float
```

Render a fixed-layout QC raster using NumPy and scikit-image operations only. The
JSON remains semantic evidence; the raster is visual review evidence.

- [ ] **Step 5: Verify GREEN and focused quality**

Run Step 3, then:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run ruff check src/cyexm/registration/initial_alignment.py tests/unit/test_initial_alignment.py
uv run ruff format --check src/cyexm/registration/initial_alignment.py tests/unit/test_initial_alignment.py
```

- [ ] **Step 6: Commit Task 2**

```bash
git commit -m "feat: estimate target-blind xy rotation seeds" \
  -m "Reuse analysis: uses declared SciPy and scikit-image dependencies and returns policy values for the existing registration lifecycle; it does not create a runner or artifact store."
```

---

### Task 3: Physical seed construction and complete affine composition

**Files:**
- Modify: `tests/unit/test_linear.py`
- Modify: `tests/unit/test_backend.py`
- Modify: `tests/unit/test_transforms.py`
- Add focused test: `tests/integration/test_initial_alignment_transforms.py`
- Modify: `src/cyexm/registration/linear.py`
- Modify: `src/cyexm/registration/backend.py`
- Modify: `src/cyexm/registration/transforms.py`

**Interfaces:**
- `RegistrationBackend.create_xy_rotation_seed` returns an in-memory transform.
- `RegistrationBackend.compose_affines` returns one complete affine.
- `register_image` accepts `initial_transform=None` and returns a complete affine.

- [ ] **Step 1: Write mock-boundary tests**

Prove `linear.register_image` passes the supplied seed path through
`ants.registration` through its `initial_transform` parameter, consumes every
returned forward affine,
composes seed and residual in the verified order, and applies only the complete
affine to scalar and mask. Preserve byte-for-byte old kwargs when the seed is `None`.

Add backend protocol tests requiring seed construction and composition without
leaking ANTs types into callers.

- [ ] **Step 2: Write physical landmark tests**

Create tiny ANTs images with anisotropic spacing, nonzero origin, and a nonidentity
direction matrix. Place asymmetric physical landmarks, construct `+30` and `-30`
degree seeds plus translations, and assert transformed landmarks land at analytically
computed positions. Compose a known residual translation and assert applying the
composed affine equals sequential seed/residual application to tolerance.

These tests are the authority for sign and order; do not assert only raw parameter
arrays.

- [ ] **Step 3: Verify RED**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run python -m pytest tests/unit/test_linear.py tests/unit/test_backend.py \
  tests/unit/test_transforms.py -k 'initial or compose or rotation' -vv
```

- [ ] **Step 4: Implement seed and composition helpers**

Use physical center from `ImageGeometry`, the full direction matrix, and homogeneous
affine matrices. Create ANTs transforms only inside the configured backend. Add
`initial_transform` to the existing registration call rather than implementing a
second linear registrar. Convert all returned linear transforms to homogeneous form,
compose in tested moving-to-fixed order, and construct one `AffineTransform` suitable
for the existing `.mat` writer.

When `initial_transform is None`, retain the current first-forward-transform behavior
and request shape. When present, reject displacement transforms in the linear result;
this feature composes affine transforms only.

- [ ] **Step 5: Verify GREEN, then run the real-ANTs micro-test**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run python -m pytest tests/unit/test_linear.py tests/unit/test_backend.py \
  tests/unit/test_transforms.py -vv
```

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python -m pytest tests/integration/test_initial_alignment_transforms.py -vv
```

Expected: the bounded real-ANTs landmark test passes on CPU with one native thread.

- [ ] **Step 6: Commit Task 3**

```bash
git commit -m "feat: compose rotation seeds into linear transforms" \
  -m "Reuse analysis: extends register_image and RegistrationBackend, and preserves the existing complete affine artifact and transform_image consumer contract."
```

---

### Task 4: Restart-safe initialization artifacts in shared preparation

**Files:**
- Modify: `tests/unit/test_whole_image.py`
- Modify: `tests/unit/test_round_task.py`
- Modify: `src/cyexm/registration/_request_policy.py`
- Modify: `src/cyexm/registration/whole_image.py`
- Modify: `src/cyexm/registration/__init__.py`

**Interfaces:**
- `ImageRegistration` gains
  `initial_alignment: InitialAlignmentPolicy | None = None`.
- `register_round` gains and forwards the same `initial_alignment` keyword.
- `RegistrationPaths.initial_alignment_transform/json/qc`
- private `_load_or_resolve_initial_alignment` shared-stage helper

- [ ] **Step 1: Write explicit lifecycle tests**

Assert explicit mode commits all three initialization artifacts atomically, records
the supplied angle and zero translation, passes the seed to linear registration, and
persists the complete affine. Assert exact rerun reuses them and changing the angle
stales the seed and all affine descendants. Omission must retain historical request
parameters and planned paths.

- [ ] **Step 2: Write estimated lifecycle and failure tests**

Inject the estimator at the module boundary. Assert only fixed and moving registration
arrays are supplied; the accepted result commits JSON/transform/QC before affine.
Assert a confidence rejection commits JSON and QC with `accepted: false`, commits no
transform, calls no linear registration, and raises `RegistrationStageError` naming
`initial-alignment` and the diagnostic path.

Simulate interrupted JSON, transform, and QC writes independently. None may be reused,
and a previously valid sibling must not authorize an incomplete set.

- [ ] **Step 3: Verify RED**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run python -m pytest tests/unit/test_whole_image.py tests/unit/test_round_task.py \
  -k initial_alignment -vv
```

- [ ] **Step 4: Implement canonical requests and the shared stage**

Add `initial_alignment_config_parameters(policy) -> dict[str, object]` and
`initial_alignment_result_parameters(result) -> dict[str, object]` request
constructors with fixed policy names and exact JSON-compatible values.

Add the three paths to `RegistrationPaths`. In `_prepare_linear_state`, resolve the
seed after raw cropping and before feature construction. Explicit and estimated modes
share artifact commit/validation. Bind requests to source identities, cropped
geometries and crop bounds, normalized policy, native thread evidence where relevant,
and the estimator policy version.

Add the seed evidence to `affine_parameters` only when enabled; omission must not
alter historical affine requests. Pass the seed into `backend.linear`, then use the
complete returned affine for raw reconstruction and every later stage.

Extend the public class and function Numpydoc with units, coordinate convention,
state transition, outputs, failure modes, and `See Also`. Export the policy values
through the lazy `cyexm.registration` surface without eager scientific imports.

- [ ] **Step 5: Verify GREEN and API compatibility**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run python -m pytest tests/unit/test_whole_image.py tests/unit/test_round_task.py -vv
```

Also run the existing no-initialization golden request tests unchanged.

- [ ] **Step 6: Commit Task 4**

```bash
git commit -m "feat: persist rotation initialization artifacts" \
  -m "Reuse analysis: integrates the seed into ImageRegistration._prepare_linear_state, RegistrationPaths, StageRequest, and ArtifactStore; register_round forwards the same policy and no duplicate lifecycle is added."
```

---

### Task 5: Prepared bundles, serial chunking, and fine-task parity

**Files:**
- Modify: `tests/unit/test_prepared_registration.py`
- Modify: `tests/unit/test_fine_workflow_models.py`
- Modify: `tests/unit/test_fine_workflow_task.py`
- Modify: `src/cyexm/registration/prepared.py`
- Modify: `src/cyexm/workflow/fine_models.py`
- Modify: `src/cyexm/workflow/fine_task.py`

- [ ] **Step 1: Write prepared-format tests**

Preserve formats 2, 3, and 4 for initialization-disabled raw, feature, and
feature-mask bundles. Add strict formats:

```text
5 = rotation + raw registration scalars
6 = rotation + intensity features
7 = rotation + intensity features + feature masks
```

Each new manifest must include exact identities for initialization JSON and transform,
canonical realized evidence, and the complete affine identity. Test round-trip,
unknown/missing keys, cross-version field leakage, source mismatch, malformed
evidence, and stale initialization artifacts.

- [ ] **Step 2: Write coarse/fine equivalence tests**

For explicit and accepted estimated policies, prove `commit_prepared_round` resolves
the seed once and fine tasks consume the resulting complete affine without invoking
the estimator again. Compare scientific request parameters and prepared descriptors
against the coarse path. Test both whole-image and serial-chunked fine task families.

- [ ] **Step 3: Verify RED**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run python -m pytest tests/unit/test_prepared_registration.py \
  tests/unit/test_fine_workflow_models.py tests/unit/test_fine_workflow_task.py \
  -k 'initial_alignment or format_version_5 or format_version_6 or format_version_7' -vv
```

- [ ] **Step 4: Implement strict formats and shared consumption**

Add version-specific exact key sets and validators. Select format from the Cartesian
combination of rotation enabled and current feature/mask state; never reinterpret a
historical version. Embed identities created by the shared preparation stage. Extend
fine task descriptors and validation to compare normalized policy plus realized
artifact evidence before work is admitted.

- [ ] **Step 5: Verify GREEN and full prepared/fine tests**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run python -m pytest tests/unit/test_prepared_registration.py \
  tests/unit/test_fine_workflow_models.py tests/unit/test_fine_workflow_task.py -vv
```

- [ ] **Step 6: Commit Task 5**

```bash
git commit -m "feat: carry initial alignment through prepared work" \
  -m "Reuse analysis: extends PreparedManifest and existing fine preparation descriptors; estimation stays in the shared preparation owner and no worker-owned graph is introduced."
```

---

### Task 6: Workflow planning, fingerprints, status, and executor parity

**Files:**
- Modify: `tests/unit/test_workflow_planning.py`
- Modify: `tests/unit/test_workflow_models.py`
- Modify: `tests/unit/test_workflow_status.py`
- Modify: `tests/unit/test_workflow_api.py`
- Modify: `tests/unit/test_execution_sequential.py`
- Modify: `tests/unit/test_execution_dask.py`
- Modify: `tests/unit/test_execution_slurm.py`
- Modify: `tests/unit/test_fine_workflow_planning.py`
- Modify: `src/cyexm/workflow/models.py`
- Modify: `src/cyexm/workflow/planning.py`
- Modify: `src/cyexm/workflow/round_task.py`
- Modify: `src/cyexm/workflow/fine_planning.py`
- Modify: `src/cyexm/workflow/_execution_validation.py`
- Modify: `src/cyexm/workflow/_round_provenance.py`
- Modify: `src/cyexm/workflow/status.py`
- Modify: `src/cyexm/workflow/api.py`

- [ ] **Step 1: Write planning and fingerprint tests**

Assert `RoundTask` carries the normalized per-round value and scientific JSON emits:

```json
"initial_alignment": {
  "xy_rotation": {
    "mode": "explicit",
    "angle_deg": 27.7
  }
}
```

or the complete normalized estimate mapping. Omission emits no key and preserves the
existing fixture fingerprint. Changing any initialization value changes the
fingerprint. Execution paths, executor, worker count, and coarse/fine mode do not.

Assert enabled plans include the three initialization artifacts for only those
rounds; planning reads headers but performs no estimation and writes nothing.

- [ ] **Step 2: Write status and run-record tests**

Build completed fixtures for explicit and estimated rounds. Status is complete only
when the JSON, transform, QC, affine, warp, registered scalar, and channels all have
matching evidence. Parameterize malformed policy/result fields, mismatched sources,
angle changes, missing QC, rejected automatic results, and transform identity
changes as `partial` or `invalid` according to the existing status contract.

Assert run records expose both configured and realized initialization evidence and
that `--force-round` invalidates seed plus descendants. Preserve existing
initialization-disabled status fixtures unchanged.

- [ ] **Step 3: Write executor and fine-plan parity tests**

Assert sequential, local-Dask, coarse SLURM, and fine-grained plans carry identical
scientific initialization. Verify one seed stage occurs inside each owned round
preparation lifecycle and no executor adds resource fields to scientific JSON.

- [ ] **Step 4: Verify RED**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run python -m pytest tests/unit/test_workflow_planning.py \
  tests/unit/test_workflow_models.py tests/unit/test_workflow_status.py \
  tests/unit/test_workflow_api.py tests/unit/test_fine_workflow_planning.py \
  -k initial_alignment -vv
```

- [ ] **Step 5: Implement immutable propagation and strict readers**

Add `initial_alignment` to `RoundTask`, translate config policy to runtime policy only
at the round runner boundary, and emit normalized config intent in each round's
canonical mapping. Add planned artifact paths conditionally.

Update all producer/reader request constructors together. `_round_provenance` must
validate exact static policy and exact realized evidence from completion snapshots;
it must not recompute estimates during status. Extend execution validation, force
propagation, expected-artifact accounting, and run-record serialization. Preserve old
schemas when the field is absent.

- [ ] **Step 6: Verify GREEN and executor tests**

Run Step 4, then:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run python -m pytest tests/unit/test_execution_sequential.py \
  tests/unit/test_execution_dask.py tests/unit/test_execution_slurm.py \
  tests/unit/test_workflow_planning.py tests/unit/test_workflow_models.py \
  tests/unit/test_workflow_status.py tests/unit/test_workflow_api.py \
  tests/unit/test_fine_workflow_planning.py -vv
```

- [ ] **Step 7: Commit Task 6**

```bash
git commit -m "feat: plan and validate rotation-initialized workflows" \
  -m "Reuse analysis: extends RoundTask, canonical scientific planning, expected artifacts, and authoritative status readers; executor profiles and task families remain unchanged."
```

---

### Task 7: Documentation, schemas, and examples

**Files:**
- Create: `docs/initial-alignment.md`
- Create: `examples/explicit-rotation-dataset.yaml`
- Create: `examples/estimated-rotation-dataset.yaml`
- Modify: `docs/dataset-manifest.md`
- Modify: `docs/preprocessing.md`
- Modify: `docs/registration-foundations.md`
- Modify: `docs/registration-artifacts.md`
- Modify: `docs/outputs-and-recovery.md`
- Modify: `docs/architecture.md`
- Modify: `docs/getting-started.md`
- Modify: `docs/myst.yml`
- Modify: `tests/unit/test_phase2_documentation.py`
- Modify: `tests/unit/test_repository_policy.py`

- [ ] **Step 1: Write failing documentation contract tests**

Require the field table to contain every exact key, type, default, bound, omission
rule, and angle convention. Require the focused guide to contain target blindness,
raw/crop/seed/affine/nonlinear data flow, confidence failures, QC interpretation,
single-pass dependent channels, force/restart behavior, and explicit versus estimated
examples. Require both example manifests to parse and plan against temporary TIFFs.

- [ ] **Step 2: Verify RED**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run python -m pytest tests/unit/test_phase2_documentation.py \
  tests/unit/test_repository_policy.py -k 'initial_alignment or dataset_manifest' -vv
```

- [ ] **Step 3: Write the maintained documentation**

Document these two distinct scientific paths without conflating them:

```text
intensity preprocessing: raw -> crop -> feature estimate -> transforms -> raw output
geometric initialization: raw -> crop -> seed -> residual affine -> nonlinear -> raw output
```

Include a physical-coordinate diagram, a conventional XY-display sign example, an
automatic QC interpretation table, recommended search ranges, failure recovery via
explicit angles, and exact artifact names. Explain that a successful estimator is an
initialization result, not proof of accurate biological registration.

The examples use placeholder relative TIFF paths and retain machine paths/resources
only in execution-profile examples. Add the new guide to `myst.yml` next to
registration preprocessing.

- [ ] **Step 4: Verify GREEN and build documentation**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run python -m pytest tests/unit/test_phase2_documentation.py \
  tests/unit/test_repository_policy.py -vv
```

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
cd docs && uv run --group docs jupyter book build --html
```

Expected: tests pass and the Jupyter Book build has no broken references.

- [ ] **Step 5: Commit Task 7**

```bash
git commit -m "docs: explain xy rotation initial alignment" \
  -m "Reuse analysis: extends the existing manifest, registration, artifact, recovery, architecture, and API documentation; examples use the established dataset/execution separation."
```

---

### Task 8: End-to-end scientific acceptance and regression gates

**Files:**
- Create: `tests/integration/test_rotation_initialized_registration.py`
- Modify only if required by evidence: `docs/phase-2-acceptance.md`

- [ ] **Step 1: Add a bounded real-ANTs workflow integration test**

Generate asymmetric `uint16` ZYX fixed, moving registration, and dependent-channel
TIFFs no larger than `16 x 64 x 64`. Apply a known 20-degree rotation and translation
to moving data. Run one explicit and one automatic whole-image workflow using dry-run
linear/nonlinear schedules and one native thread. Assert:

- both workflows complete with authoritative status;
- estimated angle is within one refinement step;
- transformed registration and dependent-channel shapes equal fixed geometry;
- all three initialization artifacts and completion records validate;
- final dependent-channel evidence still names only `[warp, affine]`; and
- omission produces the historical artifact set.

Repeat the explicit case through a minimal serial-chunked path to establish shared
preparation and fixed-grid behavior.

- [ ] **Step 2: Run focused integration and registration regressions**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python -m pytest tests/integration/test_initial_alignment_transforms.py \
  tests/integration/test_rotation_initialized_registration.py \
  tests/integration/test_registration_preprocessing.py -vv
```

- [ ] **Step 3: Run all portable repository gates**

Run each command separately and retain its fresh output:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv lock --check
```

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv sync --locked --group dev --extra examples --extra hpc
```

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest -m "unit and not hpc" tests/unit -v
```

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest -m "integration and not hpc" tests/integration -v
```

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run ruff check .
```

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run ruff format --check .
```

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run basedpyright
```

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pre-commit run --all-files
```

- [ ] **Step 4: Review artifacts and request code review**

Inspect `git diff --check`, `git status --short`, every new public Numpydoc surface,
the rendered guide, explicit and estimated QC images, and the no-initialization
golden requests. Use `superpowers:requesting-code-review` and resolve findings before
claiming completion.

- [ ] **Step 5: Commit acceptance evidence**

If `docs/phase-2-acceptance.md` needs a maintained entry, record only the portable
evidence actually obtained. Do not claim real SLURM acceptance from mocked tests.

```bash
git commit -m "test: accept rotation initialized registration" \
  -m "Reuse analysis: acceptance exercises the existing workflow CLI, artifact store, registration owners, and executor-neutral plans; no parallel production surface is added."
```

---

### Task 9: Optional operator-approved MDA-MB-231 characterization

This task is deliberately separate from software acceptance and requires explicit
operator authorization because it reads publication data and may submit HPC work.

**External scope:**
`/archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/mda_mb_321`

- [ ] Validate new explicit and estimated manifests for Cell1 and Cell2 without
  changing the completed recovery outputs.
- [ ] Run a truly small end-to-end automatic-estimation smoke test before any full
  cohort launch.
- [ ] Compare estimated angles against the retained target-blind recovery values and
  inspect every automatic QC image.
- [ ] Run explicit and automatic configurations with the same TRSAA/SyN settings in
  separate output roots only after smoke-test acceptance.
- [ ] Establish completion from `cyexm workflow status --json`, terminal records,
  completion metadata, and required TIFFs—not from an empty SLURM queue.
- [ ] Compare registered tubulin overlays, affine geometry, nonlinear deformation
  validity, and dependent-channel consistency for every cell and round.
- [ ] Record limitations: agreement with the external recovery is characterization
  evidence, not ground-truth registration accuracy.

## Final completion criteria

- Every design acceptance criterion has a named passing test or reviewed artifact.
- Existing manifests and artifacts without initialization remain compatible.
- Both modes validate, plan, execute, restart, force, and report status correctly.
- Automatic estimation is target-blind and fails closed under every specified
  ambiguity condition.
- The complete affine preserves the established downstream two-transform chain and
  single final resampling of original channels.
- Whole-image, serial-chunked, coarse, and fine paths share one preparation owner.
- Public API documentation, YAML references, examples, artifact docs, and recovery
  guidance agree with implementation.
- All fresh portable gates pass; any HPC or archived-data claims are separately
  authorized and artifact-backed.
