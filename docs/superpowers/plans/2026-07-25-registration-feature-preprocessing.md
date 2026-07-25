# Optional Registration-Feature Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional Gaussian-background-subtracted, robustly scaled asinh
registration features for affine and nonlinear estimation while applying accepted
transforms, masks, metrics, and public outputs to original raw intensities.

**Architecture:** Extend schema version 1 with an optional normalized preprocessing
model and implement the approved feature calculation in a focused scientific-runtime
module. Preserve the shared `_prepare_linear_state` boundary by carrying both feature
and raw fixed-grid images, then teach whole-image, serial-chunked, and fine-grained
assembly to estimate transforms from features and reconstruct accepted outputs from
raw images. Bind the feature policy to producer and reader cache identities without
changing preprocessing-disabled identities.

**Tech Stack:** Python 3.12, frozen dataclasses, PyYAML, NumPy, SciPy
`ndimage.gaussian_filter`, ANTsPyX, Dask, pytest, Ruff, BasedPyright, Jupyter Book.

## Global Constraints

- Schema version remains exactly `1`; omitting `registration.preprocessing`
  disables the feature and preserves existing disabled plans.
- `registration.preprocessing: {}` enables the feature with canonical
  `filter_sigma_zyx == (12.0, 12.0, 12.0)`.
- A positive finite scalar sigma expands to three axes; a list must contain exactly
  three positive finite built-in numbers in NumPy `(z, y, x)` order.
- Public voxel spacing remains micrometers in `(x, y, z)` order.
- Crop raw fixed and moving images before preprocessing.
- Derive masks from cropped raw values greater than zero.
- Use registration features only for affine and nonlinear estimation.
- Calculate existing metrics from raw intensities and reconstruct opt-in registered
  scalar output from raw intensities.
- Apply the accepted warp-plus-affine chain to original dependent-channel sources;
  never preprocess dependent channels.
- Preserve current preprocessing-disabled whole-image and serial-chunk scalar
  policies.
- Do not implement LCN, `log1p`, winsorization, Anscombe, Frangi, steerable-ridge, or
  automatic sigma selection.
- Do not write normalized public TIFFs.
- Defer NumPy and SciPy imports until scientific execution; config, planning, CLI
  help, and import-contract tests must remain free of scientific-runtime imports.
- Preserve source mutation checks, atomic completion records, stale authorization,
  fixed-grid geometry, transform order, and interpolation policies.
- Keep unrelated `.idea/` files untracked and untouched.

## File structure

- Create `src/cyexm/registration/preprocessing.py`: exact feature algorithm,
  algorithm identity, and derived robust-scale result.
- Create `tests/unit/test_registration_preprocessing.py`: independent numerical and
  validation regression tests for that module.
- Modify `src/cyexm/config/models.py`: immutable `RegistrationPreprocessingSpec` and
  optional `RegistrationSpec.preprocessing`.
- Modify `src/cyexm/config/dataset.py`: strict YAML parsing, canonical validation,
  and normalized model round-trip validation.
- Modify `src/cyexm/config/__init__.py`: export the new public configuration value
  object.
- Modify `src/cyexm/workflow/planning.py`: canonical optional preprocessing mapping
  in `scientific-plan.json`.
- Modify `src/cyexm/registration/_request_policy.py`: lightweight versioned feature
  and raw-reconstruction request parameters.
- Modify `src/cyexm/registration/whole_image.py`: public API option, feature/raw
  linear preparation, raw metrics, whole-image reconstruction, logging, and failure
  context.
- Modify `src/cyexm/registration/chunked.py`: pass feature arrays to chunks and raw
  fixed-grid images to shared assembly.
- Modify `src/cyexm/registration/chunk_stages.py`: distinguish feature chunk
  estimation from raw registered-scalar assembly.
- Modify `src/cyexm/registration/prepared.py`: preserve format 2 for disabled
  bundles and add strict format 3 feature evidence for enabled bundles.
- Modify `src/cyexm/workflow/fine_task.py`: validate format 3 evidence and load/crop/
  affine-resample raw images only in assembly workers.
- Modify `src/cyexm/workflow/round_task.py`: forward canonical preprocessing to
  coarse registration.
- Modify `src/cyexm/workflow/_round_provenance.py`: reconstruct exact expected
  feature-aware requests.
- Modify `src/cyexm/workflow/api.py`: validate task registration settings against
  the scientific snapshot.
- Modify unit and integration tests named in each task.
- Modify `README.md`, `examples/spheroid-dataset.yaml`,
  `docs/registration-foundations.md`, `docs/registration-artifacts.md`, and relevant
  layered `AGENTS.md` files.

## Reuse analysis

- Closest existing symbols: `RegistrationSpec`, `ImageRegistration`,
  `ChunkedImageRegistration`, `register_round`, `commit_prepared_round`,
  `ChunkStageContext`, `assemble_nonlinear_chunks`, `transform_channel`,
  `build_workflow_plan`, and `run_workflow`.
- Decision: extend these existing models and lifecycle owners. Add only one focused
  numerical module because no current symbol owns registration-feature construction.
- Rejected alternative: a preprocessing CLI and normalized TIFF workflow would
  duplicate large images and split raw/feature provenance across user-managed files.
- Rejected alternative: preprocessing inside `AntsRegistrationBackend` would hide a
  scientific policy and duplicate raw-versus-feature decisions across linear,
  nonlinear, metrics, and prepared-state callers.
- Documentation updates: public config export and docstring, `ImageRegistration` and
  `register_round` Numpydoc parameters, README canonical workflow, example manifests,
  artifact/restart semantics, and layered agent guidance.

---

### Task 1: Strict configuration and scientific-plan identity

**Files:**

- Modify: `src/cyexm/config/models.py`
- Modify: `src/cyexm/config/dataset.py`
- Modify: `src/cyexm/config/__init__.py`
- Modify: `src/cyexm/workflow/planning.py`
- Modify: `src/cyexm/workflow/api.py`
- Test: `tests/unit/test_config_dataset.py`
- Test: `tests/unit/test_workflow_planning.py`
- Test: `tests/unit/test_workflow_api.py`
- Test: `tests/unit/test_import_contract.py`

**Interfaces:**

- Produces:
  `RegistrationPreprocessingSpec(filter_sigma_zyx: tuple[float, float, float])`.
- Produces:
  `RegistrationSpec.preprocessing: RegistrationPreprocessingSpec | None = None`.
- Produces:
  `preprocessing_mapping(value: RegistrationPreprocessingSpec | None) -> dict[str, object] | None`
  as a private lightweight canonical mapper in `workflow/planning.py`.
- Consumes no scientific runtime; these modules must not import NumPy or SciPy.

- [ ] **Step 1: Write failing strict-schema and normalization tests**

Add tests that exercise omission, empty mapping, scalar, anisotropic list, equivalent
canonical values, strict rejection, frozen models, and exports:

```python
def test_preprocessing_omission_disables_and_empty_mapping_uses_default(
    tmp_path: Path,
) -> None:
    disabled = _load_data(tmp_path, _valid_data())
    assert disabled.registration.preprocessing is None

    data = _valid_data()
    registration = _mapping_at(data, ("registration",))
    registration["preprocessing"] = {}
    enabled = _load_data(tmp_path, data)
    assert enabled.registration.preprocessing == RegistrationPreprocessingSpec(
        filter_sigma_zyx=(12.0, 12.0, 12.0)
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (6, (6.0, 6.0, 6.0)),
        ([4, 12.5, 20], (4.0, 12.5, 20.0)),
    ],
)
def test_preprocessing_filter_normalizes_scalar_or_zyx_list(
    tmp_path: Path,
    value: object,
    expected: tuple[float, float, float],
) -> None:
    data = _valid_data()
    _mapping_at(data, ("registration",))["preprocessing"] = {
        "filter_sigma_zyx": value
    }
    manifest = _load_data(tmp_path, data)
    assert manifest.registration.preprocessing is not None
    assert manifest.registration.preprocessing.filter_sigma_zyx == expected
```

Parameterize failures for `True`, `0`, `-1`, `float("inf")`, `[1, 2]`,
`[1, 2, 3, 4]`, `[1, False, 3]`, and `{"unknown": 12}`. Update
`test_config_namespace_exports_the_scientific_and_execution_api` with
`RegistrationPreprocessingSpec`.

- [ ] **Step 2: Run configuration tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_config_dataset.py tests/unit/test_import_contract.py -q
```

Expected: FAIL because `RegistrationPreprocessingSpec` and the optional YAML block do
not exist.

- [ ] **Step 3: Implement the immutable model and strict parser**

Add the model after `RegistrationStageSpec`:

```python
@dataclass(frozen=True, slots=True)
class RegistrationPreprocessingSpec:
    filter_sigma_zyx: tuple[float, float, float] = (12.0, 12.0, 12.0)
```

Add `preprocessing` as the final defaulted `RegistrationSpec` field so current direct
constructors remain source-compatible:

```python
@dataclass(frozen=True, slots=True)
class RegistrationSpec:
    crop: bool
    linear: RegistrationStageSpec
    nonlinear: NonlinearRegistrationStageSpec
    preprocessing: RegistrationPreprocessingSpec | None = None
```

In `dataset.py`, allow `_REGISTRATION_KEYS` and
`_REGISTRATION_KEYS | {"preprocessing"}` as the only registration shapes. Implement
private `_preprocessing` and `_filter_sigma_zyx` helpers using `math.isfinite`,
explicit boolean rejection, and exact list length. Make `{}` select the default.
Update `_validate_dataset_manifest` to reconstruct preprocessing through
`_registration`, ensuring manually constructed invalid models fail identically.
Export the new model from `cyexm.config`.

- [ ] **Step 4: Run configuration and import tests**

Run:

```bash
uv run pytest tests/unit/test_config_dataset.py tests/unit/test_import_contract.py -q
```

Expected: PASS, including proof that importing config does not import `numpy` or
`scipy`.

- [ ] **Step 5: Write failing scientific-plan and task-consistency tests**

Add tests that build plans from scalar and list manifests and inspect canonical JSON:

```python
def test_equivalent_preprocessing_forms_have_one_scientific_identity(
    workflow_files: WorkflowFiles,
) -> None:
    scalar = build_plan_with_preprocessing(workflow_files, 12)
    isotropic = build_plan_with_preprocessing(workflow_files, [12, 12, 12])
    assert scalar.scientific_fingerprint == isotropic.scientific_fingerprint
    scientific = json.loads(scalar.scientific_json)
    assert scientific["registration"]["preprocessing"] == {
        "algorithm": "gaussian-background-mad-asinh-float32-v1",
        "filter_sigma_zyx": [12.0, 12.0, 12.0],
    }
```

Also assert disabled `scientific_json` has no `registration.preprocessing`, changing
one axis changes the fingerprint, and `workflow/api.py` rejects a `RoundTask` whose
preprocessing differs from the scientific snapshot.

- [ ] **Step 6: Run planning tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_workflow_planning.py tests/unit/test_workflow_api.py -q
```

Expected: FAIL because the scientific mapping and API consistency checks omit
preprocessing.

- [ ] **Step 7: Implement canonical planning and API validation**

Define the algorithm string in lightweight request policy and import it into planning.
Add `preprocessing` to the registration scientific mapping only when enabled:

```python
if manifest.registration.preprocessing is not None:
    registration_mapping["preprocessing"] = {
        "algorithm": REGISTRATION_FEATURE_POLICY,
        "filter_sigma_zyx": list(
            manifest.registration.preprocessing.filter_sigma_zyx
        ),
    }
```

Teach the strict scientific snapshot reader and task/model comparisons in
`workflow/api.py` to accept exactly the disabled shape or enabled shape and require
the same canonical tuple and algorithm. Do not add a null key for disabled manifests.

- [ ] **Step 8: Run focused planning tests**

Run:

```bash
uv run pytest tests/unit/test_workflow_planning.py tests/unit/test_workflow_api.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/cyexm/config src/cyexm/workflow/planning.py \
  src/cyexm/workflow/api.py tests/unit/test_config_dataset.py \
  tests/unit/test_workflow_planning.py tests/unit/test_workflow_api.py \
  tests/unit/test_import_contract.py
git commit -m "feat(config): add optional registration preprocessing schema"
```

Commit body must include:

```text
Reuse analysis:
- Extended RegistrationSpec and the existing workflow planner.
- Did not add a second manifest loader or preprocessing planner.
- Preserved disabled schema-1 scientific-plan identity.
```

### Task 2: Exact background-subtraction and asinh feature kernel

**Files:**

- Create: `src/cyexm/registration/preprocessing.py`
- Create: `tests/unit/test_registration_preprocessing.py`
- Modify: `src/cyexm/registration/AGENTS.md`

**Interfaces:**

- Produces constant:
  `REGISTRATION_FEATURE_POLICY = "gaussian-background-mad-asinh-float32-v1"`.
- Produces frozen internal value:
  `RegistrationFeature(array: object, robust_scale: float)`.
- Produces:
  `background_subtract_asinh(image: object, filter_sigma_zyx: tuple[float, float, float]) -> RegistrationFeature`.
- NumPy and SciPy are imported only inside a private `_scientific_runtime`.

- [ ] **Step 1: Write the independent numerical regression tests**

Use a local reference expression that does not call production helpers:

```python
def _reference_feature(image: np.ndarray, sigma: tuple[float, float, float]):
    array = np.asarray(image, dtype=np.float32)
    residual = array - ndi.gaussian_filter(array, sigma=sigma, mode="reflect")
    stride = max(1, array.size // 2_000_000)
    sample = residual.ravel()[::stride]
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    scale = max(1.4826 * mad, float(np.finfo(np.float32).eps))
    return np.arcsinh(np.maximum(residual, 0.0) / scale).astype(np.float32), scale


def test_feature_matches_approved_reference_formula() -> None:
    rng = np.random.default_rng(142)
    image = rng.normal(100, 20, size=(9, 17, 23)).astype(np.float32)
    expected, expected_scale = _reference_feature(image, (2.0, 3.0, 4.0))
    result = background_subtract_asinh(image, (2.0, 3.0, 4.0))
    np.testing.assert_array_equal(result.array, expected)
    assert result.robust_scale == expected_scale
```

Add tests for constant input, `float32` output, shape preservation, deterministic
signed-residual-before-clipping scale, anisotropic axis behavior, non-3D rejection,
nonfinite rejection, and no input mutation.

- [ ] **Step 2: Run the kernel tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_registration_preprocessing.py -q
```

Expected: FAIL because `cyexm.registration.preprocessing` does not exist.

- [ ] **Step 3: Implement the minimal focused kernel**

Implement:

```python
@dataclass(frozen=True, slots=True)
class RegistrationFeature:
    array: object
    robust_scale: float


def background_subtract_asinh(
    image: object,
    filter_sigma_zyx: tuple[float, float, float],
) -> RegistrationFeature:
    np, ndi = _scientific_runtime()
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("registration preprocessing requires a 3D image")
    if not bool(np.isfinite(array).all()):
        raise ValueError("registration preprocessing requires finite image values")
    background = ndi.gaussian_filter(
        array,
        sigma=filter_sigma_zyx,
        mode="reflect",
    )
    residual = array - background
    stride = max(1, array.size // 2_000_000)
    sample = residual.ravel()[::stride]
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    scale = max(1.4826 * mad, float(np.finfo(np.float32).eps))
    feature = np.arcsinh(np.maximum(residual, 0.0) / scale).astype(
        np.float32,
        copy=False,
    )
    return RegistrationFeature(feature, scale)
```

Validate the already-canonical sigma defensively without importing configuration.
Give the module a one-responsibility API map and deliberate `__all__`.

- [ ] **Step 4: Run kernel and import-contract tests**

Run:

```bash
uv run pytest tests/unit/test_registration_preprocessing.py \
  tests/unit/test_import_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Update registration agent guidance**

Add exact guidance that feature construction is background subtraction, signed
residual MAD scaling, negative clipping, and asinh; it is used only for transform
estimation; raw data own masks, metrics, and output reconstruction; and scientific
imports remain deferred.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/cyexm/registration/preprocessing.py \
  src/cyexm/registration/AGENTS.md \
  tests/unit/test_registration_preprocessing.py
git commit -m "feat(registration): add exact asinh feature kernel"
```

### Task 3: Feature-aware request policy and whole-image registration

**Files:**

- Modify: `src/cyexm/registration/_request_policy.py`
- Modify: `src/cyexm/registration/whole_image.py`
- Modify: `tests/unit/fakes.py`
- Modify: `tests/unit/test_whole_image.py`
- Modify: `tests/unit/test_repository_policy.py`

**Interfaces:**

- Consumes `background_subtract_asinh` and
  `RegistrationPreprocessingSpec.filter_sigma_zyx`.
- Adds public keyword:
  `preprocess_filter_sigma_zyx: tuple[float, float, float] | None = None` to
  `ImageRegistration` and `register_round`.
- Produces lightweight:
  `preprocessing_config_parameters(filter_sigma_zyx)` and
  `registration_feature_parameters(filter_sigma_zyx, fixed_scale, moving_scale)`.
- Produces:
  `RAW_RECONSTRUCTED_SCALAR_POLICY = "feature-estimation-raw-warp-affine-clip-v1"`.
- Extends `_LinearPreparation` with `raw_fixed`, `raw_moving_linear`,
  `fixed_feature_scale`, and `moving_feature_scale`; existing `fixed` and
  `moving_linear` continue to mean estimator inputs.

- [ ] **Step 1: Write failing public API and estimator-input tests**

Add tests proving the constructor is lazy, validates the optional canonical tuple,
and forwards it through `register_round`. Add a feature/raw split test:

```python
def test_enabled_preprocessing_estimates_from_features_but_metrics_use_raw(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    registration = make_registration(
        tmp_path,
        backend=backend,
        preprocess_filter_sigma_zyx=(1.0, 2.0, 3.0),
    )
    registration.register()

    raw_fixed = backend.loaded_fixed
    assert backend.linear_kwargs[0]["fixed_image"] is not raw_fixed
    assert backend.metric_calls[0][0] is raw_fixed
    assert backend.metric_calls[1][0] is raw_fixed
    assert backend.metric_calls[2][0] is raw_fixed
    assert backend.histogram_match_calls == []
    assert backend.clipped_tiff_images[-1] is backend.raw_nonlinear_result
```

Record `from_array` calls in `FakeBackend` so tests can distinguish reconstructed
feature images from loaded raw images. Assert `zero_mask` receives raw objects before
feature conversion. Add a disabled control that preserves current histogram-match and
writer events exactly.

- [ ] **Step 2: Run focused whole-image tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_whole_image.py \
  -k 'preprocess or feature or defaults_are_legacy' -q
```

Expected: FAIL because the public option and dual image roles do not exist.

- [ ] **Step 3: Add lightweight request-policy helpers**

Add functions that return finite canonical JSON data. The static helper is available
to raw metric requests before features have been calculated:

```python
def preprocessing_config_parameters(
    filter_sigma_zyx: tuple[float, float, float],
) -> dict[str, object]:
    return {
        "algorithm": REGISTRATION_FEATURE_POLICY,
        "filter_sigma_zyx": list(filter_sigma_zyx),
        "dtype": "float32",
        "background_boundary": "reflect",
        "robust_scale": "1.4826-mad-signed-residual-stride-max-2m-v1",
        "negative_residual": "clip-zero-before-asinh-v1",
    }


def registration_feature_parameters(
    *,
    filter_sigma_zyx: tuple[float, float, float],
    fixed_scale: float,
    moving_scale: float,
) -> dict[str, object]:
    return {
        **preprocessing_config_parameters(filter_sigma_zyx),
        "fixed_robust_scale": fixed_scale,
        "moving_robust_scale": moving_scale,
    }
```

Add a helper that conditionally merges the static policy into all three existing
metric requests and the derived-scale policy into feature-dependent requests. Crop,
fixed-reference, and reference-channel request parameters must remain byte-for-byte
unchanged when preprocessing is enabled.

- [ ] **Step 4: Implement feature construction in `_prepare_linear_state`**

Keep raw load, source identity verification, crop, masks, fixed-reference persistence,
and pre-registration metrics in their current order. Add private
`_registration_feature(backend, raw, role)` that calls `backend.to_array`,
`background_subtract_asinh`, and `backend.from_array(reference=raw)`, verifies exact
geometry, and contextualizes failures as:

```text
<fixed|moving> preprocessing stage failed for round <n>
```

When disabled, estimator images alias raw images and scales are `None`. When enabled,
pass features plus raw-derived masks to `_load_or_estimate_affine`. Reconstruct
`raw_moving_linear` with the accepted affine and linear interpolation, while reusing
the returned transformed raw-derived mask. Calculate linear metrics from raw fixed and
raw moving linear.

- [ ] **Step 5: Implement raw nonlinear reconstruction and persistence**

Pass feature fixed/moving images to nonlinear estimation. On an enabled fresh or
cached warp, reconstruct:

```python
raw_nonlinear = backend.apply_warp(
    fixed=state.raw_fixed,
    moving=state.raw_moving_linear,
    warp_path=warp_path,
    interpolation="linear",
)
```

Calculate nonlinear metrics with raw fixed, raw nonlinear, and raw-derived masks.
Persist opt-in `nonlinear_registered_<round>.tif` with
`backend.write_clipped_tiff` and `RAW_RECONSTRUCTED_SCALAR_POLICY`. Retain the
existing whole-image histogram-match and min/max TIFF path when preprocessing is
disabled.

Include the static feature policy in pre-registration metrics. Include normalized
feature parameters and actual scales in affine, linear/nonlinear metrics, nonlinear,
and opt-in registered-image requests and logs. Never place feature parameters in
crop/fixed-reference requests.

- [ ] **Step 6: Run whole-image and repository-policy tests**

Run:

```bash
uv run pytest tests/unit/test_whole_image.py \
  tests/unit/test_repository_policy.py tests/unit/test_import_contract.py -q
```

Expected: PASS, including existing disabled cache and source-mutation tests.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/cyexm/registration/_request_policy.py \
  src/cyexm/registration/whole_image.py \
  tests/unit/fakes.py tests/unit/test_whole_image.py \
  tests/unit/test_repository_policy.py
git commit -m "feat(registration): estimate whole-image transforms from features"
```

Commit body must include:

```text
Reuse analysis:
- Extended ImageRegistration and register_round with one keyword-only option.
- Reused the existing linear preparation and artifact lifecycle.
- Kept preprocessing out of the ANTs backend and dependent-channel transformer.
```

### Task 4: Raw reconstruction for serial nonlinear assembly

**Files:**

- Modify: `src/cyexm/registration/chunk_stages.py`
- Modify: `src/cyexm/registration/chunked.py`
- Modify: `tests/unit/test_chunk_stages.py`
- Modify: `tests/unit/test_chunked_registration.py`

**Interfaces:**

- Consumes `_LinearPreparation.fixed` and `.moving_linear` as feature images.
- Consumes `_LinearPreparation.raw_fixed` and `.raw_moving_linear` as raw metric/output
  images.
- Extends `ChunkStageContext` with:
  `raw_fixed: object`, `raw_moving_linear: object`,
  `preprocessing: dict[str, object] | None`, and
  `registered_scalar_policy: str`.
- Chunk estimation and scalar-core artifacts remain feature-based; final metrics and
  opt-in registered TIFF are raw-based.

- [ ] **Step 1: Write failing serial feature/raw boundary tests**

Add a context test where feature arrays contain compressed values and raw arrays
contain sentinel values:

```python
def test_preprocessed_chunk_assembly_commits_warp_then_reconstructs_raw(
    chunk_context: ChunkStageContext,
    complete_chunk_artifacts: tuple[ChunkStageArtifacts, ...],
) -> None:
    chunk_context.preprocessing = FEATURE_EVIDENCE
    chunk_context.raw_fixed = raw_fixed_with_value(1000)
    chunk_context.raw_moving_linear = raw_moving_with_value(2000)
    assembled = assemble_nonlinear_chunks(chunk_context, complete_chunk_artifacts)
    assert assembled.warp_path.is_file()
    assert chunk_context.backend.metric_calls[-1][0] is chunk_context.raw_fixed
    assert chunk_context.backend.clipped_tiff_values[-1].max() == 2000
```

Add a disabled control proving scalar-core stitch, histogram policy, min/max writer,
request parameters, and artifact names remain unchanged. Assert nonlinear chunk
`fixed_image` and `moving_image` are feature regions while masks are raw-derived.

- [ ] **Step 2: Run serial tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_chunk_stages.py \
  tests/unit/test_chunked_registration.py -k 'preprocess or feature or assembly' -q
```

Expected: FAIL because the context cannot distinguish feature and raw image roles.

- [ ] **Step 3: Extend coarse serial context without duplicating orchestration**

In `ChunkedImageRegistration.register`, pass the complete `_LinearPreparation` into
`_run_nonlinear_chunks`. Construct `ChunkStageContext` with feature images for
`fixed`/`moving_linear`, raw images for `raw_fixed`/`raw_moving_linear`, and the
canonical preprocessing evidence from the parent registration. Local
`run_nonlinear_chunk` continues to operate only on feature regions and raw-derived
masks.

- [ ] **Step 4: Reconstruct raw only after the warp is accepted**

In assembly, continue authenticating local warp and scalar-core evidence and building
the canonical full displacement warp. For enabled preprocessing, treat the stitched
feature scalar as internal evidence only. After committing or reusing the full warp,
apply it to `context.raw_moving_linear`, calculate metrics against
`context.raw_fixed`, and write the raw result with `write_clipped_tiff`.

For disabled preprocessing, execute the existing scalar-core stitch, metric, and TIFF
branches without numerical changes. Parameterize `_finalize_assembled_nonlinear`
through the context instead of adding a second finalizer.

- [ ] **Step 5: Bind chunk requests and invalidation to preprocessing**

Add canonical preprocessing evidence to local warp, scalar core, chunk manifest,
assembled warp, nonlinear metric, and opt-in registered-image requests. Ensure a sigma
change makes completion readers reject every feature-derived chunk artifact, while
crop and raw reference artifacts remain reusable.

- [ ] **Step 6: Run all serial unit tests**

Run:

```bash
uv run pytest tests/unit/test_chunk_stages.py \
  tests/unit/test_chunked_registration.py tests/unit/test_chunked_geometry.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/cyexm/registration/chunk_stages.py \
  src/cyexm/registration/chunked.py tests/unit/test_chunk_stages.py \
  tests/unit/test_chunked_registration.py
git commit -m "feat(registration): reconstruct chunked outputs from raw data"
```

### Task 5: Strict prepared feature bundles and fine-grained assembly

**Files:**

- Modify: `src/cyexm/registration/prepared.py`
- Modify: `src/cyexm/workflow/fine_task.py`
- Modify: `src/cyexm/workflow/fine_models.py`
- Modify: `tests/unit/test_prepared_registration.py`
- Modify: `tests/unit/test_fine_workflow_task.py`
- Modify: `tests/unit/test_fine_workflow_models.py`
- Modify: `tests/unit/test_fine_workflow_planning.py`

**Interfaces:**

- Disabled prepared bundles retain exact format version `2`.
- Enabled bundles use format version `3`.
- Format 3 `PreparedManifest` adds:
  `preprocessing: dict[str, object]`,
  `fixed_feature_scale: float`, and `moving_feature_scale: float`.
- Format 3 scalar arrays are features; mask arrays remain raw-derived.
- Fine chunk tasks need only prepared features and masks. Assembly additionally loads
  raw fixed/moving sources, the accepted affine, crop bounds when present, and warp.

- [ ] **Step 1: Write failing strict format-2/format-3 manifest tests**

Add tests that commit disabled and enabled preparations:

```python
def test_enabled_preparation_uses_strict_feature_manifest_v3(tmp_path: Path) -> None:
    registration = _registration(
        tmp_path,
        preprocess_filter_sigma_zyx=(2.0, 3.0, 4.0),
    )
    manifest = commit_prepared_round(registration)
    assert manifest.format_version == 3
    assert manifest.preprocessing["filter_sigma_zyx"] == [2.0, 3.0, 4.0]
    assert manifest.fixed_feature_scale > 0
    assert manifest.moving_feature_scale > 0
    assert manifest.mask_policy == PREPARED_MASK_POLICY


def test_disabled_preparation_retains_exact_manifest_v2(tmp_path: Path) -> None:
    manifest = commit_prepared_round(_registration(tmp_path))
    assert manifest.format_version == 2
    assert set(manifest.to_dict()) == LEGACY_FORMAT_2_KEYS
```

Test strict rejection of missing/unknown format-3 keys, nonfinite/nonpositive scales,
policy mismatch, array identity change, sigma mismatch, and a format-3 descriptor used
with a disabled task.

- [ ] **Step 2: Run prepared and fine validation tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_prepared_registration.py \
  tests/unit/test_fine_workflow_task.py -k 'manifest or preprocess or prepared' -q
```

Expected: FAIL because only format 2 exists.

- [ ] **Step 3: Implement strict dual manifest formats**

Keep the exact format-2 key set and values. Add an exact format-3 key set containing
the three feature fields. `PreparedManifest.from_dict` selects the key set from
`format_version` and rejects every other version. `to_dict` emits only keys belonging
to that version.

For enabled preparation, persist `state.fixed` and `state.moving_linear` feature
arrays with raw-derived masks. Include preprocessing evidence in every array request,
the manifest request, and the manifest payload. Verify scale values are built-in,
positive, finite floats. Preserve format-2 completion requests exactly when disabled.

- [ ] **Step 4: Update fine descriptor and completion validation**

Teach `_validated_prepared_manifest` to:

- require format 2 exactly for disabled tasks;
- require format 3 exactly for enabled tasks;
- reconstruct canonical preprocessing evidence from the embedded `RoundTask`;
- compare actual robust scales and algorithm policy;
- reconstruct exact array and manifest requests; and
- reject descriptor, completion-record, or on-disk manifest disagreement.

Keep chunk task models embedding the complete `RoundTask`; do not duplicate sigma
fields in `NonlinearChunkTask` or `ChunkAssemblyTask`.

- [ ] **Step 5: Load raw images only for fine assembly**

In `_chunk_context(..., assembly=True)`, load the prepared fixed raw TIFF and moving
raw source. If crop is enabled, read the authenticated moving crop bounds and crop the
moving raw image. Apply the authenticated affine onto the fixed raw grid with linear
interpolation. Set `context.raw_fixed` and `context.raw_moving_linear`.

In `_chunk_context(..., assembly=False)`, do not load raw source arrays; use a private
sentinel or optional fields that `ChunkStageContext.__post_init__` permits only when
`require_existing` is false for regional estimation. Assembly validation must require
both raw images and exact fixed geometry.

- [ ] **Step 6: Run all prepared and fine unit tests**

Run:

```bash
uv run pytest tests/unit/test_prepared_registration.py \
  tests/unit/test_fine_workflow_models.py \
  tests/unit/test_fine_workflow_planning.py \
  tests/unit/test_fine_workflow_task.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

```bash
git add src/cyexm/registration/prepared.py src/cyexm/workflow/fine_task.py \
  src/cyexm/workflow/fine_models.py tests/unit/test_prepared_registration.py \
  tests/unit/test_fine_workflow_task.py \
  tests/unit/test_fine_workflow_models.py \
  tests/unit/test_fine_workflow_planning.py
git commit -m "feat(workflow): persist authenticated registration features"
```

### Task 6: Workflow wiring and authoritative completion validation

**Files:**

- Modify: `src/cyexm/workflow/round_task.py`
- Modify: `src/cyexm/workflow/fine_task.py`
- Modify: `src/cyexm/workflow/_round_provenance.py`
- Modify: `tests/unit/test_round_task.py`
- Modify: `tests/unit/test_workflow_status.py`
- Modify: `tests/unit/test_fine_workflow_task.py`
- Modify: `tests/unit/test_workflow_thread_boundaries.py`

**Interfaces:**

- Consumes `RoundTask.registration.preprocessing`.
- Coarse and fine constructors forward exactly
  `preprocess_filter_sigma_zyx=None` or the canonical tuple.
- Completion readers use the same request-policy helpers as producers.
- Channel requests remain bound transitively through affine and warp identities; they
  do not receive preprocessing as an independent channel parameter.

- [ ] **Step 1: Write failing forwarding and completion tests**

Patch registration constructors and assert the exact keyword:

```python
def test_round_task_forwards_canonical_preprocessing(
    preprocessed_round_task: RoundTask,
) -> None:
    run_round_task(preprocessed_round_task)
    assert captured_registration_kwargs["preprocess_filter_sigma_zyx"] == (
        2.0,
        12.0,
        20.0,
    )
```

Add fine preparation forwarding coverage. Create authenticated artifacts under sigma
`(12, 12, 12)`, change one task axis to `13`, and assert workflow status reports
affine, chunks, warp, registered image, metrics, and channels incomplete/invalid while
reference materialization and crop bounds remain accepted.

- [ ] **Step 2: Run workflow tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_round_task.py \
  tests/unit/test_workflow_status.py tests/unit/test_fine_workflow_task.py \
  tests/unit/test_workflow_thread_boundaries.py -q
```

Expected: FAIL because workflow constructors and readers omit preprocessing.

- [ ] **Step 3: Forward the canonical option through every execution path**

Add one private helper:

```python
def _preprocess_sigma(task: RoundTask) -> tuple[float, float, float] | None:
    spec = task.registration.preprocessing
    return None if spec is None else spec.filter_sigma_zyx
```

Use it in coarse `registration_arguments` and fine `_new_image_registration`.
Constructor setup remains free of NumPy/SciPy imports and scientific filesystem work.

- [ ] **Step 4: Pair every producer request with its completion reader**

Update `_round_provenance.py` expected affine, metrics, nonlinear, chunk, manifest,
and registered-image requests using shared lightweight parameter helpers. Apply
preprocessing evidence only to affected stages. Preserve disabled expected requests
exactly. Keep channel request parameters unchanged and rely on authenticated upstream
transform identities.

- [ ] **Step 5: Run authoritative workflow unit tests**

Run:

```bash
uv run pytest tests/unit/test_round_task.py \
  tests/unit/test_workflow_status.py tests/unit/test_fine_workflow_task.py \
  tests/unit/test_workflow_thread_boundaries.py \
  tests/unit/test_workflow_api.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

```bash
git add src/cyexm/workflow/round_task.py src/cyexm/workflow/fine_task.py \
  src/cyexm/workflow/_round_provenance.py \
  tests/unit/test_round_task.py tests/unit/test_workflow_status.py \
  tests/unit/test_fine_workflow_task.py \
  tests/unit/test_workflow_thread_boundaries.py
git commit -m "feat(workflow): validate feature-aware registration provenance"
```

### Task 7: Scientific and execution-mode integration regressions

**Files:**

- Create: `tests/integration/test_registration_preprocessing.py`
- Modify: `tests/integration/test_whole_image.py`
- Modify: `tests/integration/test_chunked.py`
- Modify: `tests/integration/test_prepared_registration.py`
- Modify: `tests/integration/test_fine_grained_workflow.py`
- Modify: `tests/integration/test_workflow_round_task.py`
- Modify: `tests/integration/test_workflow_local_dask.py`

**Interfaces:**

- Exercises the public configuration/workflow path, not private test-only injection.
- Uses bounded synthetic images with bright and dim structures and known translations.
- Compares output geometry and raw intensity provenance across whole, coarse serial,
  sequential fine-grained, and Local Dask fine-grained modes.

- [ ] **Step 1: Add a bounded scientific fixture**

Create a deterministic fixture with a sloped background, bright and dim Gaussian
filaments, and a known integer translation:

```python
def translated_bright_dim_pair() -> tuple[np.ndarray, np.ndarray]:
    z, y, x = np.mgrid[:12, :48, :48]
    fixed = (
        80.0
        + 0.6 * x
        + 0.2 * y
        + 900.0 * np.exp(-((y - 16) ** 2 + (z - 6) ** 2) / 3.0)
        + 90.0 * np.exp(-((x - 34) ** 2 + (z - 6) ** 2) / 3.0)
    )
    moving = np.roll(fixed, shift=(0, 2, -3), axis=(0, 1, 2))
    return fixed.astype(np.uint16), moving.astype(np.uint16)
```

Write raw TIFFs with exact ZYX axes and spacing. Configure preprocessing with
`filter_sigma_zyx: [2, 6, 6]`, TRSAA dry-run, and SyNOnly dry-run with CC radius 4.

- [ ] **Step 2: Add raw-versus-feature whole-image assertions**

Assert:

- the scientific plan records canonical feature parameters;
- the affine and warp completion records record the algorithm and actual scales;
- output geometry equals fixed geometry;
- output intensities are on the raw uint16 scale, not the small asinh feature scale;
- pre/linear/nonlinear metric files are readable;
- the raw dependent channel is transformed without feature preprocessing; and
- rerun reuses artifacts without invoking registration kernels.

- [ ] **Step 3: Run the whole integration regression**

Run:

```bash
uv run pytest tests/integration/test_registration_preprocessing.py \
  tests/integration/test_whole_image.py -q
```

Expected: PASS within the existing bounded integration runtime.

- [ ] **Step 4: Add serial and fine-grained parity assertions**

Run the same synthetic round through coarse serial-chunked and fine-grained
sequential execution with one canonical chunk grid. Assert exact fixed geometry,
finite raw-scale output, accepted completion records, matching scientific
fingerprints, and deformation fields within the repository's existing deterministic
single-process tolerance. Run Local Dask with one worker and assert it accepts the
same prepared format-3 bundle and completion chain.

- [ ] **Step 5: Run serial, fine, and Local Dask integration tests**

Run:

```bash
uv run pytest tests/integration/test_registration_preprocessing.py \
  tests/integration/test_chunked.py \
  tests/integration/test_prepared_registration.py \
  tests/integration/test_fine_grained_workflow.py \
  tests/integration/test_workflow_round_task.py \
  tests/integration/test_workflow_local_dask.py -q
```

Expected: PASS. This task does not submit SLURM jobs.

- [ ] **Step 6: Add cache-change and source-mutation integration coverage**

Run once with sigma `[2, 6, 6]`, then plan into a new output root with `[2, 7, 6]`.
Assert transform identities differ. Mutate a raw moving input during an injected
feature commit boundary and assert no output receives valid completion evidence.

- [ ] **Step 7: Run the complete new integration module**

Run:

```bash
uv run pytest tests/integration/test_registration_preprocessing.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 7**

```bash
git add tests/integration/test_registration_preprocessing.py \
  tests/integration/test_whole_image.py tests/integration/test_chunked.py \
  tests/integration/test_prepared_registration.py \
  tests/integration/test_fine_grained_workflow.py \
  tests/integration/test_workflow_round_task.py \
  tests/integration/test_workflow_local_dask.py
git commit -m "test(science): verify raw output from feature registration"
```

### Task 8: README, examples, technical docs, and layered guidance

**Files:**

- Modify: `README.md`
- Modify: `examples/spheroid-dataset.yaml`
- Create: `examples/preprocessed-spheroid-dataset.yaml`
- Modify: `examples/AGENTS.md`
- Modify: `src/cyexm/config/models.py`
- Modify: `src/cyexm/registration/whole_image.py`
- Modify: `src/cyexm/registration/chunked.py`
- Modify: `docs/registration-artifacts.md`
- Modify: `docs/registration-foundations.md`
- Modify: `docs/AGENTS.md`
- Modify: `src/AGENTS.md`
- Modify: `src/cyexm/config/AGENTS.md`
- Modify: `src/cyexm/registration/AGENTS.md`
- Modify: `src/cyexm/workflow/AGENTS.md`
- Modify: `tests/AGENTS.md`
- Modify: `tests/unit/AGENTS.md`
- Modify: `tests/unit/test_phase2_examples.py`
- Modify: `tests/unit/test_phase2_documentation.py`
- Modify: `tests/unit/test_repository_policy.py`

**Interfaces:**

- Documents only implemented behavior.
- Uses `filter_sigma_zyx` in NumPy ZYX voxel units, distinct from public XYZ
  micrometer spacing.
- Identifies existing metric files and public channels as raw-intensity products.

- [ ] **Step 1: Write failing example and documentation contract tests**

Require:

- the existing spheroid example to remain preprocessing-disabled by omission;
- a new example with `preprocessing: {}` and a commented anisotropic form;
- README snippets for omission, scalar, isotropic list, and anisotropic list;
- explicit `raw → crop → feature estimate → transform raw` wording;
- no supported-method references to rejected preliminary transforms;
- exact artifact and restart invalidation claims; and
- updated relevant `AGENTS.md` scientific invariants.

- [ ] **Step 2: Run documentation tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_phase2_examples.py \
  tests/unit/test_phase2_documentation.py \
  tests/unit/test_repository_policy.py -q
```

Expected: FAIL because the new example and maintained text do not exist.

- [ ] **Step 3: Update public API documentation**

Add concise Numpydoc to `RegistrationPreprocessingSpec`. Update
`ImageRegistration`, `ChunkedImageRegistration`, and `register_round` parameter
documentation with:

- omission semantics;
- voxel-domain NumPy ZYX axis order;
- scalar expansion performed by YAML rather than the Python API;
- raw masks/metrics/output boundary;
- crop-before-feature order; and
- cache invalidation behavior.

Update the applicable module API map and `See Also` graph without exporting private
kernel helpers as a new high-level workflow.

- [ ] **Step 4: Update README and maintained technical documentation**

Document:

```yaml
# Disabled: omit registration.preprocessing
preprocessing:
  filter_sigma_zyx: [12, 12, 12]
```

Explain the exact signed-residual MAD formula, default 12-voxel-per-axis filter,
anisotropic form, reflective boundary, and that sigma is not in micrometers. State
that TRSAA and nonlinear registration use features, while masks, metrics,
`nonlinear_registered_<round>.tif`, and all declared channel outputs use raw
intensities. Explain which artifacts are invalidated by a sigma change and which
reference/crop artifacts remain reusable.

- [ ] **Step 5: Add reusable example YAML**

Create a complete strict-schema example based on portable filenames. Use:

```yaml
registration:
  crop: false
  preprocessing:
    filter_sigma_zyx: [12, 12, 12]
  linear:
    type: TRSAA
    accuracy: high
  nonlinear:
    type: SyNOnly
    accuracy: high
    metric: CC
```

Include a comment showing `filter_sigma_zyx: 12` and state that scalar/list forms are
scientifically identical. Do not add lab archive paths.

- [ ] **Step 6: Update layered agent guidance**

Record:

- config strict optional-block semantics and canonical tuple;
- deferred NumPy/SciPy imports;
- crop-before-feature and raw-derived mask rules;
- features-only estimator boundary;
- raw metric/output/channel reconstruction;
- format-2 disabled and format-3 enabled prepared manifests;
- paired producer/reader request changes; and
- algorithm changes require a new policy identifier and scientific regression.

- [ ] **Step 7: Run documentation and example tests**

Run:

```bash
uv run pytest tests/unit/test_phase2_examples.py \
  tests/unit/test_phase2_documentation.py \
  tests/unit/test_repository_policy.py -q
```

Expected: PASS.

- [ ] **Step 8: Build the published HTML documentation**

Run the exact checked-in documentation workflow command:

```bash
(cd docs && uv run --group docs jupyter book build --html)
```

Expected: exit 0 with no broken internal links or schema-invalid YAML snippets.

- [ ] **Step 9: Commit Task 8**

```bash
git add README.md examples/preprocessed-spheroid-dataset.yaml \
  examples/spheroid-dataset.yaml examples/AGENTS.md \
  docs/registration-artifacts.md docs/registration-foundations.md docs/AGENTS.md \
  src/AGENTS.md src/cyexm/config/models.py src/cyexm/config/AGENTS.md \
  src/cyexm/registration/whole_image.py src/cyexm/registration/chunked.py \
  src/cyexm/registration/AGENTS.md src/cyexm/workflow/AGENTS.md \
  tests/AGENTS.md tests/unit/AGENTS.md tests/unit/test_phase2_examples.py \
  tests/unit/test_phase2_documentation.py \
  tests/unit/test_repository_policy.py
git commit -m "docs: explain registration feature preprocessing"
```

Before committing, inspect `git diff --cached --name-only` and unstage generated
Jupyter Book output or unrelated files. Commit only maintained source.

### Task 9: Full verification and implementation record

**Files:**

- Modify only files required to correct failures attributable to Tasks 1–8.
- Do not add benchmark outputs, archive data, generated HTML, `.idea/`, or unrelated
  cleanup.

**Interfaces:**

- Verifies the complete public configuration-to-output contract.
- Produces no new behavior beyond the approved design.

- [ ] **Step 1: Run formatting**

Run:

```bash
uv run ruff format src tests
```

Expected: exit 0. Inspect `git diff --stat` and ensure formatting did not touch
unrelated files.

- [ ] **Step 2: Run Ruff**

Run:

```bash
uv run ruff check src tests
```

Expected: PASS.

- [ ] **Step 3: Run BasedPyright**

Run:

```bash
uv run basedpyright
```

Expected: PASS with zero errors.

- [ ] **Step 4: Run the complete unit suite**

Run:

```bash
uv run pytest tests/unit -q
```

Expected: PASS.

- [ ] **Step 5: Run the bounded integration suite**

Run:

```bash
uv run pytest tests/integration -q
```

Expected: PASS. Do not run `tests/hpc` or submit SLURM jobs in this task.

- [ ] **Step 6: Run repository hooks**

Run:

```bash
uv run pre-commit run --all-files
```

Expected: PASS. If a hook formats maintained files, rerun Steps 2–5 after inspecting
the changes.

- [ ] **Step 7: Verify clean scope and diff**

Run:

```bash
git status --short
git diff --check
git diff --stat origin/main...HEAD
git log --oneline origin/main..HEAD
```

Expected: only intentional Cy-ExM source, tests, docs, examples, design, and plan
changes are tracked; `.idea/` remains untracked and untouched.

- [ ] **Step 8: Commit verification-only corrections if present**

If Steps 1–6 required corrections, inspect the scoped tracked changes and commit them:

```bash
git add -u -- src tests docs README.md examples
git commit -m "test: complete preprocessing verification"
```

If no tracked corrections remain, do not create an empty commit.

- [ ] **Step 9: Record final evidence for review**

In the handoff, report:

- exact commit range;
- unit and integration pass counts;
- Ruff, BasedPyright, pre-commit, and docs-build outcomes;
- that no HPC job was submitted;
- any implementation deviation from
  `docs/superpowers/specs/2026-07-25-registration-feature-preprocessing-design.md`;
  and
- the remaining untracked `.idea/` directory as unrelated user state.
