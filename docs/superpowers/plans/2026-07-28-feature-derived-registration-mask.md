# Feature-derived Registration Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional, deterministic feature-derived mask that restricts every TRSAA and SyN metric stage to morphologically cleaned positive asinh support while preserving raw diagnostics and raw registered outputs.

**Architecture:** Extend the existing preprocessing value object with one optional physical-unit mask policy. Construct masks in `registration.preprocessing`, keep raw diagnostic masks separate from registration metric masks in linear preparation, persist the latter in prepared fine-grained bundles, and bind the complete policy and realized evidence through planning, artifact requests, provenance, validation, and status.

**Tech Stack:** Python 3.12, NumPy, SciPy `ndimage`, ANTSpyX, frozen dataclasses, strict YAML schema version 1, pytest, Ruff, BasedPyright, pre-commit, MyST/Jupyter Book.

## Global Constraints

- Omission of `registration.preprocessing.feature_mask` must preserve current preprocessing behavior and existing prepared format 3/cache identity exactly.
- Feature masks affect transform estimation only; raw greater-than-zero masks continue to own diagnostic metrics.
- Registered registration-channel TIFFs and dependent-channel TIFFs remain transformations of raw inputs.
- Initial support is exactly finite `feature > 0`; connected components use full 3D connectivity.
- Component volume and dilation radius are positive finite scalar micrometer values, provisionally defaulting to `1.0`.
- Morphology runs once on each full cropped image before nonlinear chunk extraction.
- Fixed and moving masks are derived independently.
- Masks are finite float32 arrays containing only `0.0` and `1.0`; empty masks fail before ANTs.
- Mask interpolation remains `nearestNeighbor`, with a `0.5` binary threshold after affine application.
- TRSAA and SyN retain `mask_all_stages=True`; no schedules, metrics, CC radius, transform order, or scalar interpolation change.
- Schema version 1 is extended in place; no compatibility schema or migration layer is added.
- NumPy arrays use ZYX order; manifest voxel spacing and ANTs geometry use XYZ order.
- NumPy and SciPy imports remain lazy inside registration modules.
- No new public Phase 1 class or package-root export is introduced.
- Existing unrelated worktree changes and untracked SLURM logs remain untouched.

## Reuse analysis

- Extend `RegistrationPreprocessingSpec` and the existing strict `_preprocessing`
  parser; do not introduce a second preprocessing configuration hierarchy.
- Extend `background_subtract_asinh`'s focused module with mask construction because
  it already owns the only supported feature algorithm and lazy scientific imports.
- Reuse `ImageRegistration._prepare_linear_state`, backend `apply_affine`,
  nearest-neighbor mask thresholding, and existing chunk extraction. Do not add a
  parallel registration lifecycle.
- Extend `preprocessing_config_parameters` and
  `registration_feature_parameters`; do not create a second request-evidence system.
- Advance the existing strict `PreparedManifest` only for enabled feature masking;
  formats 2 and 3 remain exact for disabled preprocessing and preprocessing without
  feature masking.
- Reuse workflow planning, status, and provenance reconstruction paths and make their
  producer/reader changes together.

## File structure

- `src/cyexm/config/models.py`: normalized immutable feature-mask policy.
- `src/cyexm/config/dataset.py`: strict YAML parsing and canonical serialization.
- `src/cyexm/registration/preprocessing.py`: deterministic mask array algorithm and
  evidence.
- `src/cyexm/registration/_request_policy.py`: canonical static and realized policy
  dictionaries.
- `src/cyexm/registration/whole_image.py`: raw/registration mask separation and
  coarse registration orchestration.
- `src/cyexm/registration/chunked.py`: constructor propagation and serial chunk use.
- `src/cyexm/registration/prepared.py`: strict prepared format 4 and mask evidence.
- `src/cyexm/registration/chunk_stages.py`: unchanged scientific kernel, verified to
  consume prepared registration masks.
- `src/cyexm/workflow/{planning,round_task,fine_task,_round_provenance,status,api}.py`:
  plan propagation, coarse/fine construction, evidence, validation, and status.
- `tests/unit/`: focused policy, algorithm, orchestration, prepared, and workflow
  regression tests.
- `tests/integration/test_registration_preprocessing.py`: real-ANTs raw-output and
  mask-routing regression.
- `README.md`, `docs/`, `examples/preprocessed-spheroid-dataset.yaml`, and layered
  `AGENTS.md`: user and maintainer contracts.

---

### Task 1: Strict optional configuration

**Files:**
- Modify: `src/cyexm/config/models.py`
- Modify: `src/cyexm/config/dataset.py`
- Modify: `src/cyexm/config/__init__.py`
- Test: `tests/unit/test_config_dataset.py`
- Test: `tests/unit/test_phase2_examples.py`

**Interfaces:**
- Produces: `FeatureMaskSpec(min_component_volume_um3: float = 1.0, dilation_radius_um: float = 1.0)`.
- Produces: `RegistrationPreprocessingSpec.feature_mask: FeatureMaskSpec | None`.
- Consumes later: every workflow and registration constructor receives this exact
  normalized object or `None`.

- [ ] **Step 1: Write failing parser and model tests**

```python
def test_feature_mask_omission_and_empty_block_are_distinct(dataset_mapping):
    preprocessing = dataset_mapping["registration"]["preprocessing"] = {}
    omitted = load_dataset_mapping(dataset_mapping)
    assert omitted.registration.preprocessing == RegistrationPreprocessingSpec()
    assert omitted.registration.preprocessing.feature_mask is None

    preprocessing["feature_mask"] = {}
    enabled = load_dataset_mapping(dataset_mapping)
    assert enabled.registration.preprocessing.feature_mask == FeatureMaskSpec(
        min_component_volume_um3=1.0,
        dilation_radius_um=1.0,
    )


@pytest.mark.parametrize(
    "value",
    [True, 0, -1, float("inf"), [1.0]],
)
def test_feature_mask_rejects_invalid_physical_values(dataset_mapping, value):
    dataset_mapping["registration"]["preprocessing"] = {
        "feature_mask": {"dilation_radius_um": value}
    }
    with pytest.raises(
        ConfigurationValidationError,
        match=r"registration\.preprocessing\.feature_mask\.dilation_radius_um",
    ):
        load_dataset_mapping(dataset_mapping)
```

Also assert that canonical serialization writes both explicit `1.0` values and that
unknown keys fail.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest tests/unit/test_config_dataset.py tests/unit/test_phase2_examples.py -q
```

Expected: failure because `FeatureMaskSpec` and `feature_mask` do not exist.

- [ ] **Step 3: Add the immutable model and strict parser**

Add:

```python
@dataclass(frozen=True, slots=True)
class FeatureMaskSpec:
    min_component_volume_um3: float = 1.0
    dilation_radius_um: float = 1.0


@dataclass(frozen=True, slots=True)
class RegistrationPreprocessingSpec:
    filter_sigma_zyx: tuple[float, float, float] = (12.0, 12.0, 12.0)
    feature_mask: FeatureMaskSpec | None = None
```

Parse `feature_mask` only when present. An empty mapping returns `FeatureMaskSpec()`;
an explicit mapping must contain exactly both supported keys or one supported key
whose omitted peer receives its default. Normalize with `finite_positive_float`.
Canonical serialization always emits both feature-mask values when enabled and emits
no `feature_mask` key when disabled.

- [ ] **Step 4: Run focused tests and formatting**

Run:

```bash
uv run pytest tests/unit/test_config_dataset.py tests/unit/test_phase2_examples.py -q
uv run ruff check src/cyexm/config tests/unit/test_config_dataset.py tests/unit/test_phase2_examples.py
uv run ruff format --check src/cyexm/config tests/unit/test_config_dataset.py tests/unit/test_phase2_examples.py
```

Expected: all pass.

- [ ] **Step 5: Commit configuration support**

```bash
git add src/cyexm/config/models.py src/cyexm/config/dataset.py \
  src/cyexm/config/__init__.py tests/unit/test_config_dataset.py \
  tests/unit/test_phase2_examples.py
git commit -m "feat: add optional feature mask configuration"
```

### Task 2: Deterministic physical mask algorithm

**Files:**
- Modify: `src/cyexm/registration/preprocessing.py`
- Modify: `src/cyexm/registration/_request_policy.py`
- Test: `tests/unit/test_registration_preprocessing.py`

**Interfaces:**
- Consumes: `FeatureMaskSpec`.
- Produces: `FeatureMask(array: object, foreground_voxels: int, total_voxels: int)`.
- Produces:
  `feature_mask(feature, spacing_zyx, *, min_component_volume_um3, dilation_radius_um) -> FeatureMask`.
- Produces: `feature_mask_config_parameters(...)` and
  `feature_mask_parameters(..., fixed_mask, moving_mask)`.

- [ ] **Step 1: Write failing numerical tests**

```python
def test_feature_mask_filters_components_and_dilates_in_physical_space():
    feature = np.zeros((5, 9, 9), dtype=np.float32)
    feature[2, 4, 4] = 1.0       # 0.5 um3: removed
    feature[2, 1:3, 1:3] = 1.0  # 2.0 um3: retained
    result = feature_mask(
        feature,
        spacing_zyx=(2.0, 0.5, 0.5),
        min_component_volume_um3=1.0,
        dilation_radius_um=1.0,
    )
    assert result.array.dtype == np.float32
    assert set(np.unique(result.array)) == {0.0, 1.0}
    assert result.array[2, 4, 4] == 0.0
    assert result.array[2, 0, 1] == 1.0
    assert result.array[1, 1, 1] == 0.0  # Z offset is 2 um


def test_feature_mask_fails_when_filtering_removes_all_support():
    feature = np.zeros((3, 3, 3), dtype=np.float32)
    feature[1, 1, 1] = 1.0
    with pytest.raises(ValueError, match="no foreground"):
        feature_mask(
            feature,
            spacing_zyx=(1.0, 1.0, 1.0),
            min_component_volume_um3=2.0,
            dilation_radius_um=1.0,
        )
```

Add tests for strict `> 0`, non-finite feature rejection, invalid spacing/policy
values, full 26-connectivity, deterministic repeats, and exact evidence counts.

- [ ] **Step 2: Run tests and verify failure**

```bash
uv run pytest tests/unit/test_registration_preprocessing.py -q
```

Expected: import failure for `FeatureMask`/`feature_mask`.

- [ ] **Step 3: Implement the focused helper**

Use lazy NumPy/SciPy imports. The central implementation is:

```python
support = np.asarray(feature, dtype=np.float32) > 0.0
labels, component_count = ndi.label(
    support,
    structure=np.ones((3, 3, 3), dtype=bool),
)
minimum_voxels = max(1, math.ceil(min_component_volume_um3 / math.prod(spacing_zyx)))
counts = np.bincount(labels.ravel())
keep = counts >= minimum_voxels
keep[0] = False
cleaned = keep[labels]

radii = tuple(math.ceil(dilation_radius_um / spacing) for spacing in spacing_zyx)
grid = np.ogrid[tuple(slice(-radius, radius + 1) for radius in radii)]
distance_squared = sum(
    (axis * spacing) ** 2 for axis, spacing in zip(grid, spacing_zyx, strict=True)
)
footprint = distance_squared <= dilation_radius_um**2
mask = ndi.binary_dilation(cleaned, structure=footprint).astype(np.float32)
```

Validate the 3D finite feature and exact positive finite ZYX spacing first. Fail if
`cleaned` or the final mask is empty. Add
`FEATURE_MASK_POLICY = "positive-support-components-physical-dilation-v1"` and
canonical static/realized evidence helpers in `_request_policy.py`.

- [ ] **Step 4: Run algorithm tests and static checks**

```bash
uv run pytest tests/unit/test_registration_preprocessing.py -q
uv run ruff check src/cyexm/registration/preprocessing.py \
  src/cyexm/registration/_request_policy.py \
  tests/unit/test_registration_preprocessing.py
uv run basedpyright src/cyexm/registration/preprocessing.py
```

Expected: all pass.

- [ ] **Step 5: Commit the algorithm**

```bash
git add src/cyexm/registration/preprocessing.py \
  src/cyexm/registration/_request_policy.py \
  tests/unit/test_registration_preprocessing.py
git commit -m "feat: derive physical masks from registration features"
```

### Task 3: Separate raw diagnostic masks from registration masks

**Files:**
- Modify: `src/cyexm/registration/whole_image.py`
- Modify: `src/cyexm/registration/chunked.py`
- Modify: `src/cyexm/registration/backend.py`
- Test: `tests/unit/test_whole_image.py`
- Test: `tests/unit/test_chunked_registration.py`

**Interfaces:**
- Consumes: normalized `FeatureMaskSpec | None` and Task 2's helper.
- Produces: `_LinearPreparation.raw_fixed_mask`,
  `_LinearPreparation.raw_moving_linear_mask`,
  `_LinearPreparation.registration_fixed_mask`, and
  `_LinearPreparation.registration_moving_linear_mask`.
- Produces: `ImageRegistration(..., preprocess_feature_mask: FeatureMaskSpec | None = None)`.

- [ ] **Step 1: Write failing orchestration tests**

Build distinct sentinel raw and feature masks in the fake backend, then assert:

```python
assert backend.linear_calls[0].fixed_mask is fixed_feature_mask
assert backend.linear_calls[0].moving_mask is moving_feature_mask
assert backend.metric_calls["pre-registration"].fixed_mask is fixed_raw_mask
assert backend.metric_calls["linearly-registered"].moving_mask is transformed_raw_mask
assert backend.nonlinear_calls[0].fixed_mask is fixed_feature_mask
assert backend.nonlinear_calls[0].moving_mask is transformed_feature_mask
```

Add a disabled regression asserting the existing raw masks remain the registration
masks and request dictionaries are unchanged.

- [ ] **Step 2: Run focused tests and verify failure**

```bash
uv run pytest tests/unit/test_whole_image.py \
  tests/unit/test_chunked_registration.py -q
```

Expected: failure because one mask pair is still shared.

- [ ] **Step 3: Implement mask-role separation**

Keep raw masks immediately after crop. After feature construction, create independent
feature masks using `backend.spacing_xyz(feature)` converted to ZYX, then adapt each
mask array back to the corresponding ANTs geometry. Pass registration masks to
`_load_or_estimate_affine`; independently affine-transform both moving raw and moving
registration masks when they differ.

Use raw masks in all `_commit_metrics` calls. Use registration masks in all
registration calls and returned nonlinear state. Keep the existing disabled branch
object flow unchanged.

Store realized fixed/moving feature-mask counts in
`self._registration_feature_evidence`. Log count and fraction once per preparation.
Propagate `preprocess_feature_mask` through `ChunkedImageRegistration`.

- [ ] **Step 4: Run focused tests and static checks**

```bash
uv run pytest tests/unit/test_whole_image.py \
  tests/unit/test_chunked_registration.py -q
uv run ruff check src/cyexm/registration/whole_image.py \
  src/cyexm/registration/chunked.py src/cyexm/registration/backend.py \
  tests/unit/test_whole_image.py tests/unit/test_chunked_registration.py
uv run basedpyright src/cyexm/registration
```

Expected: all pass.

- [ ] **Step 5: Commit mask routing**

```bash
git add src/cyexm/registration/whole_image.py \
  src/cyexm/registration/chunked.py src/cyexm/registration/backend.py \
  tests/unit/test_whole_image.py tests/unit/test_chunked_registration.py
git commit -m "feat: route feature masks through registration stages"
```

### Task 4: Prepared format 4 and fine-grained mask parity

**Files:**
- Modify: `src/cyexm/registration/prepared.py`
- Modify: `src/cyexm/registration/chunk_stages.py`
- Test: `tests/unit/test_prepared_registration.py`
- Test: `tests/unit/test_chunk_stages.py`
- Test: `tests/unit/test_fine_task.py`

**Interfaces:**
- Consumes: Task 3 registration fixed-grid masks and realized evidence.
- Produces: prepared manifest format 4 only when feature masking is enabled.
- Preserves: exact format 2 without preprocessing and exact format 3 with unmasked
  preprocessing.

- [ ] **Step 1: Write failing strict-manifest tests**

```python
def test_prepared_format_four_requires_canonical_feature_mask_evidence():
    manifest = prepared_manifest(
        format_version=4,
        preprocessing=masked_preprocessing_evidence(),
    )
    assert PreparedManifest.from_dict(manifest.to_dict()) == manifest


@pytest.mark.parametrize("format_version", [2, 3])
def test_legacy_prepared_formats_reject_feature_mask(format_version):
    payload = prepared_manifest(format_version=format_version).to_dict()
    payload["preprocessing"]["feature_mask"] = masked_feature_evidence()
    with pytest.raises(ValueError, match="feature mask"):
        PreparedManifest.from_dict(payload)
```

Assert format 4 rejects policy mismatch, invalid counts/fractions, nonbinary arrays,
and geometry mismatch. Add a fine task test showing the reconstructed regional masks
equal slices from the prepared full-grid registration masks.

- [ ] **Step 2: Run focused tests and verify failure**

```bash
uv run pytest tests/unit/test_prepared_registration.py \
  tests/unit/test_chunk_stages.py tests/unit/test_fine_task.py -q
```

Expected: format 4 is unsupported.

- [ ] **Step 3: Implement strict format 4**

Define `_FORMAT_VERSION_FEATURE_MASK = 4`. Select it only when
`preprocess_feature_mask is not None`. Validate the complete canonical preprocessing
dictionary, fixed/moving foreground counts, equal positive total counts, fractions
equal to `foreground / total`, and array identities.

Before manifest promotion, validate each persisted mask array is finite float32,
binary, nonempty, and exact fixed-grid shape. Continue saving roles as
`fixed_mask.npy` and `moving_linear_mask.npy`; their meaning is already bound by the
format-specific mask policy.

Verify `chunk_stages.py` requires no numerical change: it must reconstruct and slice
the two prepared masks without recomputing morphology.

- [ ] **Step 4: Run prepared/fine tests and checks**

```bash
uv run pytest tests/unit/test_prepared_registration.py \
  tests/unit/test_chunk_stages.py tests/unit/test_fine_task.py -q
uv run ruff check src/cyexm/registration/prepared.py \
  src/cyexm/registration/chunk_stages.py \
  tests/unit/test_prepared_registration.py tests/unit/test_chunk_stages.py \
  tests/unit/test_fine_task.py
uv run basedpyright src/cyexm/registration/prepared.py \
  src/cyexm/registration/chunk_stages.py
```

Expected: all pass.

- [ ] **Step 5: Commit prepared parity**

```bash
git add src/cyexm/registration/prepared.py \
  src/cyexm/registration/chunk_stages.py \
  tests/unit/test_prepared_registration.py tests/unit/test_chunk_stages.py \
  tests/unit/test_fine_task.py
git commit -m "feat: persist feature masks for fine registration"
```

### Task 5: Workflow plan, cache identity, provenance, validation, and status

**Files:**
- Modify: `src/cyexm/workflow/planning.py`
- Modify: `src/cyexm/workflow/round_task.py`
- Modify: `src/cyexm/workflow/fine_task.py`
- Modify: `src/cyexm/workflow/_round_provenance.py`
- Modify: `src/cyexm/workflow/status.py`
- Modify: `src/cyexm/workflow/api.py`
- Test: `tests/unit/test_workflow_planning.py`
- Test: `tests/unit/test_round_task.py`
- Test: `tests/unit/test_fine_task.py`
- Test: `tests/unit/test_workflow_status.py`
- Test: `tests/unit/test_workflow_api.py`

**Interfaces:**
- Consumes: `RegistrationPreprocessingSpec.feature_mask`.
- Produces: identical canonical feature-mask policy at every producer/reader
  boundary.
- Preserves: omission adds no plan or completion-record keys.

- [ ] **Step 1: Write failing propagation and invalidation tests**

Assert the plan contains:

```python
assert plan["registration"]["preprocessing"]["feature_mask"] == {
    "algorithm": "positive-support-components-physical-dilation-v1",
    "min_component_volume_um3": 1.0,
    "dilation_radius_um": 1.0,
}
```

Assert coarse and fine constructors receive the same `FeatureMaskSpec`. Assert a
mask-policy change invalidates affine, warp, registered scalar, and channels but not
reference preparation. Assert status accepts a fully matching completed round and
rejects a missing or altered mask policy.

- [ ] **Step 2: Run workflow tests and verify failure**

```bash
uv run pytest tests/unit/test_workflow_planning.py \
  tests/unit/test_round_task.py tests/unit/test_fine_task.py \
  tests/unit/test_workflow_status.py tests/unit/test_workflow_api.py -q
```

Expected: feature-mask keys are absent or rejected.

- [ ] **Step 3: Update all workflow boundaries together**

Add canonical mask policy to `planning.py`. Pass `FeatureMaskSpec | None` through
coarse and fine registration constructors. Extend `_feature_evidence` to require
exact static policy and realized counts when enabled. Extend strict status key sets
and `_RegistrationContract` with normalized mask values.

Ensure API/status reconstruction distinguishes:

- preprocessing omitted;
- preprocessing enabled with feature masking omitted; and
- preprocessing enabled with canonical feature masking.

Use the same request helper from Task 2 everywhere. Do not hand-build a second
dictionary. Confirm channels inherit invalidation through affine/warp identities
rather than receiving feature preprocessing directly.

- [ ] **Step 4: Run workflow tests and checks**

```bash
uv run pytest tests/unit/test_workflow_planning.py \
  tests/unit/test_round_task.py tests/unit/test_fine_task.py \
  tests/unit/test_workflow_status.py tests/unit/test_workflow_api.py -q
uv run ruff check src/cyexm/workflow tests/unit/test_workflow_planning.py \
  tests/unit/test_round_task.py tests/unit/test_fine_task.py \
  tests/unit/test_workflow_status.py tests/unit/test_workflow_api.py
uv run basedpyright src/cyexm/workflow
```

Expected: all pass.

- [ ] **Step 5: Commit workflow identity**

```bash
git add src/cyexm/workflow/planning.py src/cyexm/workflow/round_task.py \
  src/cyexm/workflow/fine_task.py src/cyexm/workflow/_round_provenance.py \
  src/cyexm/workflow/status.py src/cyexm/workflow/api.py \
  tests/unit/test_workflow_planning.py tests/unit/test_round_task.py \
  tests/unit/test_fine_task.py tests/unit/test_workflow_status.py \
  tests/unit/test_workflow_api.py
git commit -m "feat: bind feature masks to workflow provenance"
```

### Task 6: Real-ANTs scientific regression

**Files:**
- Modify: `tests/integration/test_registration_preprocessing.py`
- Modify: `tests/integration/test_chunked.py`

**Interfaces:**
- Verifies: accepted raw → feature/mask → transform → raw boundary.
- Verifies: unequal native shapes, fixed-grid affine boundary, and chunk masks.

- [ ] **Step 1: Add a failing synthetic integration test**

Create fixed and moving raw 3D arrays with positive acquisition background, two
translated Gaussian/filament structures, unequal shapes along Z, Y, and X, and one
dependent-channel sentinel. Enable asinh and feature masking.

Assertions:

```python
assert output.shape == fixed_raw.shape
assert not np.array_equal(output, fixed_feature)
assert set(np.unique(recorded_fixed_mask)) <= {0.0, 1.0}
assert recorded_fixed_mask.sum() < np.prod(fixed_raw.shape)
assert recorded_moving_linear_mask.shape == fixed_raw.shape
assert dependent_output.shape == fixed_raw.shape
assert completed_provenance["preprocessing"]["feature_mask"]["fixed_foreground_voxels"] > 0
```

Spy on `ants.registration` or the backend boundary to assert both masks and
`mask_all_stages=True` reach TRSAA and SyN.

- [ ] **Step 2: Run the selected integration test and verify failure**

```bash
uv run pytest tests/integration/test_registration_preprocessing.py -q
```

Expected: failure until all enabled-mask routing is complete.

- [ ] **Step 3: Fix only integration defects**

Correct geometry adaptation, evidence reconstruction, or test fixture behavior
revealed by the real backend. Do not change schedules, metrics, or the approved mask
algorithm to make the test pass.

- [ ] **Step 4: Run preprocessing and unequal-shape integration tests**

```bash
uv run pytest tests/integration/test_registration_preprocessing.py \
  tests/integration/test_chunked.py::test_unequal_native_shapes_are_affine_resampled_before_chunking -q
```

Expected: all pass.

- [ ] **Step 5: Commit scientific regression coverage**

```bash
git add tests/integration/test_registration_preprocessing.py \
  tests/integration/test_chunked.py
git commit -m "test: verify feature-masked raw registration"
```

Before committing, inspect `git diff --cached --name-only` and unstage any integration
file not intentionally changed.

### Task 7: Documentation, examples, and maintainer guidance

**Files:**
- Modify: `README.md`
- Modify: `examples/preprocessed-spheroid-dataset.yaml`
- Modify: `docs/workflow-guide.md`
- Modify: `docs/registration-artifacts.md`
- Modify: `docs/index.md`
- Modify: `docs/myst.yml`
- Modify: `docs/asinh-preprocessing-registration-investigation.md`
- Modify: `AGENTS.md`
- Modify: `src/cyexm/registration/AGENTS.md`
- Modify: `src/cyexm/workflow/AGENTS.md`
- Modify: `tests/AGENTS.md`
- Modify: `tests/unit/AGENTS.md`
- Modify: `tests/integration/AGENTS.md`
- Modify: `examples/AGENTS.md`
- Modify: `docs/AGENTS.md`
- Test: `tests/unit/test_phase2_examples.py`

**Interfaces:**
- Documents: exact schema, defaults, physical units, stage behavior, artifacts,
  limitations, and opt-in status.

- [ ] **Step 1: Add the example assertion before editing YAML**

```python
assert manifest.registration.preprocessing.feature_mask == FeatureMaskSpec(
    min_component_volume_um3=1.0,
    dilation_radius_um=1.0,
)
assert "feature_mask:" in raw
assert "min_component_volume_um3: 1.0" in raw
assert "dilation_radius_um: 1.0" in raw
```

- [ ] **Step 2: Run the example test and verify failure**

```bash
uv run pytest tests/unit/test_phase2_examples.py -q
```

Expected: example does not yet enable the mask.

- [ ] **Step 3: Update user and maintainer documentation**

Add this canonical YAML:

```yaml
registration:
  preprocessing:
    filter_sigma_zyx: [12, 12, 12]
    feature_mask:
      min_component_volume_um3: 1.0
      dilation_radius_um: 1.0
```

Explain:

- omission disables feature masking;
- `{}` enables both provisional `1.0` micrometer defaults;
- thresholding is strict positive asinh support;
- component filtering and dilation use physical units;
- fixed and moving masks are independent and applied at all ANTs stages;
- raw metrics and outputs remain raw-derived;
- masking may improve focus but does not guarantee speed or accuracy; and
- scientific characterization remains required before recommending it by default.

Update artifact docs for prepared format 4 and realized mask counts. Update the asinh
investigation page to identify this as a proposed follow-up, not completed evidence.
Update layered `AGENTS.md` rules so future code cannot silently substitute feature
masks for raw diagnostic masks or recompute morphology per chunk.

- [ ] **Step 4: Validate examples and strict documentation build**

```bash
uv run pytest tests/unit/test_phase2_examples.py -q
uv run pre-commit run --files README.md \
  examples/preprocessed-spheroid-dataset.yaml docs/workflow-guide.md \
  docs/registration-artifacts.md docs/index.md docs/myst.yml \
  docs/asinh-preprocessing-registration-investigation.md AGENTS.md \
  src/cyexm/registration/AGENTS.md src/cyexm/workflow/AGENTS.md \
  tests/AGENTS.md tests/unit/AGENTS.md tests/integration/AGENTS.md \
  examples/AGENTS.md docs/AGENTS.md
docs_copy=$(mktemp -d /tmp/cyexm-feature-mask-docs.XXXXXX)
cp -a docs/. \"$docs_copy/\"
(cd \"$docs_copy\" && uv run --project \
  /project/bioinformatics/Danuser_lab/Dean/dean/git/Cy-ExM \
  --group docs jupyter-book build --html --strict)
rm -r \"$docs_copy\"
```

Expected: tests, hooks, and 12-or-more-page strict HTML build pass.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md examples/preprocessed-spheroid-dataset.yaml \
  docs/workflow-guide.md docs/registration-artifacts.md docs/index.md \
  docs/myst.yml docs/asinh-preprocessing-registration-investigation.md \
  AGENTS.md src/cyexm/registration/AGENTS.md \
  src/cyexm/workflow/AGENTS.md tests/AGENTS.md tests/unit/AGENTS.md \
  tests/integration/AGENTS.md examples/AGENTS.md docs/AGENTS.md \
  tests/unit/test_phase2_examples.py
git commit -m "docs: explain optional feature-derived masks"
```

### Task 8: Portable verification and scientific characterization handoff

**Files:**
- Modify only if a verification defect is found: files already listed in Tasks 1–7.
- Create after approved external runs:
  `docs/feature-mask-scientific-characterization.md`

**Interfaces:**
- Verifies: complete portable implementation at one exact commit.
- Defers: production scientific recommendation until external evidence completes.

- [ ] **Step 1: Run targeted feature-mask suite**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest \
  tests/unit/test_registration_preprocessing.py \
  tests/unit/test_config_dataset.py \
  tests/unit/test_whole_image.py \
  tests/unit/test_chunked_registration.py \
  tests/unit/test_prepared_registration.py \
  tests/unit/test_chunk_stages.py \
  tests/unit/test_workflow_planning.py \
  tests/unit/test_round_task.py \
  tests/unit/test_fine_task.py \
  tests/unit/test_workflow_status.py \
  tests/unit/test_workflow_api.py \
  tests/integration/test_registration_preprocessing.py -q
```

Expected: all pass.

- [ ] **Step 2: Run portable repository gates**

```bash
uv lock --check
uv run pytest -m "unit and not hpc" tests/unit -q
uv run pytest -m "integration and not hpc" tests/integration -q
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pre-commit run --all-files
```

Expected: all pass. Do not submit HPC jobs as part of the portable gate.

- [ ] **Step 3: Inspect compatibility and worktree scope**

```bash
git diff origin/main...HEAD --check
git diff origin/main...HEAD --stat
git status --short --branch
```

Confirm preprocessing-disabled plans and prepared format 2 remain exact, unmasked
preprocessing remains format 3, masked preprocessing uses format 4, and unrelated
`.idea/` or SLURM outputs are absent from commits.

- [ ] **Step 4: Record the exact implementation state**

```bash
git rev-parse HEAD
git log --oneline --decorate origin/main..HEAD
```

Use that exact commit in the later external characterization manifest and report.

- [ ] **Step 5: Prepare but do not claim the external comparison**

Create three separate output roots for:

1. preprocessing disabled;
2. asinh preprocessing without `feature_mask`; and
3. asinh preprocessing with `feature_mask: {}`.

Hold dataset, crop, TRSAA/high, SyNOnly/high, nonlinear metric, CC radius 4 where
selected, chunk geometry, overlap, native threads, executor, and source files
constant. After operator-approved runs complete, analyze the previously defined
bright and dim regions, deformation validity, seams, mask foreground fractions, and
stage/total runtime. Only then create
`docs/feature-mask-scientific-characterization.md`; label the option experimental
unless evidence supports a stronger statement.

- [ ] **Step 6: Request code review before integration**

Invoke `superpowers:requesting-code-review`, address confirmed findings using
`superpowers:receiving-code-review`, rerun the affected tests, and use
`superpowers:verification-before-completion` before any merge, push, or completion
claim.
