# Configurable CC SyN Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit workflow selection of CC or Mattes nonlinear similarity, normalize an omitted CC radius to the provisional default of 4, and carry the effective policy through ANTs execution, scientific plans, fingerprints, provenance, restart decisions, validation, tests, and documentation.

**Architecture:** A new lightweight package-internal module will be the single pure authority for nonlinear metric normalization and will import no scientific runtime. Strict configuration will normalize YAML into a dedicated nonlinear-stage model; workflow and Phase 1 orchestration will pass that model to the existing registration boundary, while request-policy builders derive cache evidence from the same resolver used by the ANTs call.

**Tech Stack:** Python `>=3.12,<3.13`, frozen dataclasses, strict safe YAML, ANTSpyX 0.6.3, pytest, NumPy, tifffile, Ruff, basedpyright, pre-commit, uv

## Global Constraints

- Preserve affine TRSAA Mattes/32 behavior, affine-before-nonlinear order, SyN schedules, masks, interpolation, histogram postprocessing, precision, geometry, transform order, and native-thread policy.
- Workflow schema version 1 requires an explicit nonlinear `metric`; do not add schema version 2 or a dual parser.
- Support exactly case-sensitive `CC` and `mattes` nonlinear metric names.
- CC accepts one scalar positive built-in integer radius; omission normalizes to 4. Do not accept or flatten anisotropic radii.
- Mattes retains 32 bins and rejects a radius field.
- Omitted CC radius and explicit radius 4 must produce equal normalized models, byte-identical canonical plans, equal fingerprints, and equal cache identities.
- Preserve the standalone Phase 1 omitted-argument Mattes/32 behavior.
- Changing radius invalidates nonlinear warps and descendants while preserving compatible affine evidence.
- Configure native threads before importing ANTSpyX, NumPy, SciPy, or tifffile at scientific operation boundaries.
- Use project storage for uv state by exporting `UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache` and `UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python` before every `uv` command.
- Do not access lab datasets, submit or alter scheduler jobs, or stage the pre-existing `slurm-10766186.out`, `slurm-10766187.out`, and `slurm-10766188.out` files.

---

## File Structure

- Create `src/cyexm/_nonlinear_metric.py`: pure metric names, constants, resolved value object, and strict resolver shared by configuration and registration.
- Create `tests/unit/test_nonlinear_metric.py`: isolated resolver contract.
- Modify `src/cyexm/config/models.py`, `src/cyexm/config/dataset.py`, and `src/cyexm/config/__init__.py`: dedicated normalized nonlinear-stage model and strict YAML/direct-model validation.
- Modify `src/cyexm/workflow/planning.py`, `src/cyexm/workflow/status.py`, `src/cyexm/workflow/api.py`, and `src/cyexm/workflow/_execution_validation.py`: canonical plan, snapshot reconstruction, immutable task validation, and fingerprint comparisons.
- Modify `src/cyexm/registration/nonlinear.py`, `src/cyexm/registration/_request_policy.py`, `src/cyexm/registration/whole_image.py`, and `src/cyexm/registration/chunked.py`: actual ANTs parameters, Phase 1 compatibility, and metric-aware artifact requests.
- Modify `src/cyexm/workflow/round_task.py` and `src/cyexm/workflow/_round_provenance.py`: workflow propagation and status/runtime cache agreement.
- Modify focused unit and integration test modules named in each task below.
- Modify `README.md`, `examples/spheroid-dataset.yaml`, `docs/migration-from-clearex.md`, `docs/brightness-dependence-investigation.md`, and applicable `AGENTS.md` files: current public behavior, migration, provisional default, and future optimization boundary.

---

### Task 1: Create the single pure nonlinear metric policy and wire the low-level kernel

**Files:**
- Create: `src/cyexm/_nonlinear_metric.py`
- Create: `tests/unit/test_nonlinear_metric.py`
- Modify: `tests/unit/test_nonlinear.py`
- Modify: `src/cyexm/registration/nonlinear.py`

**Interfaces:**
- Produces: `NonlinearMetricName`, `DEFAULT_CC_RADIUS`, `MATTES_BINS`, `ResolvedNonlinearMetric`, and `resolve_nonlinear_metric(metric: object = "mattes", radius: object = None) -> ResolvedNonlinearMetric`.
- Produces: `nonlinear.register_image(..., *, metric: object = "mattes", radius: object = None, native_threads: NativeThreadConfig | None = None) -> tuple[object, Path]`.
- Invariant: resolver validation completes before `_configured_runtime` imports scientific libraries.

- [ ] **Step 1: Add failing pure-policy tests**

  Create `tests/unit/test_nonlinear_metric.py` with these contract tests:

  ```python
  import pytest

  from cyexm._nonlinear_metric import resolve_nonlinear_metric

  pytestmark = pytest.mark.unit


  @pytest.mark.parametrize("radius", [None, 4])
  def test_cc_omitted_and_explicit_default_resolve_identically(radius: object) -> None:
      resolved = resolve_nonlinear_metric("CC", radius)
      assert resolved.metric == "CC"
      assert resolved.radius == 4
      assert resolved.bins is None
      assert resolved.syn_sampling == 4


  def test_explicit_cc_radius_is_preserved() -> None:
      resolved = resolve_nonlinear_metric("CC", 2)
      assert resolved.radius == 2
      assert resolved.syn_sampling == 2


  def test_mattes_resolves_to_fixed_bins() -> None:
      resolved = resolve_nonlinear_metric("mattes")
      assert resolved.metric == "mattes"
      assert resolved.radius is None
      assert resolved.bins == 32
      assert resolved.syn_sampling == 32


  @pytest.mark.parametrize("metric", ["cc", "Mattes", "", 1, None])
  def test_unknown_or_nonstring_metric_is_rejected(metric: object) -> None:
      with pytest.raises(ValueError, match="metric"):
          resolve_nonlinear_metric(metric)


  @pytest.mark.parametrize("radius", [True, 0, -1, 1.0, "4", (1, 2, 4)])
  def test_invalid_cc_radius_is_rejected(radius: object) -> None:
      with pytest.raises((TypeError, ValueError), match="radius"):
          resolve_nonlinear_metric("CC", radius)


  def test_mattes_radius_is_rejected() -> None:
      with pytest.raises(ValueError, match="radius"):
          resolve_nonlinear_metric("mattes", 4)
  ```

- [ ] **Step 2: Run the resolver tests and verify RED**

  Run:

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  uv run pytest tests/unit/test_nonlinear_metric.py -q
  ```

  Expected: collection fails because `cyexm._nonlinear_metric` does not exist.

- [ ] **Step 3: Implement the minimal pure policy module**

  Create `src/cyexm/_nonlinear_metric.py` without importing configuration,
  registration, ANTs, NumPy, SciPy, or tifffile:

  ```python
  from dataclasses import dataclass
  from typing import Literal, cast

  type NonlinearMetricName = Literal["CC", "mattes"]

  DEFAULT_CC_RADIUS = 4
  MATTES_BINS = 32
  _METRICS = frozenset({"CC", "mattes"})


  @dataclass(frozen=True, slots=True)
  class ResolvedNonlinearMetric:
      metric: NonlinearMetricName
      radius: int | None
      bins: int | None

      @property
      def syn_sampling(self) -> int:
          value = self.radius if self.metric == "CC" else self.bins
          if value is None:
              raise RuntimeError("resolved nonlinear metric lacks sampling evidence")
          return value


  def resolve_nonlinear_metric(
      metric: object = "mattes", radius: object = None
  ) -> ResolvedNonlinearMetric:
      if type(metric) is not str or metric not in _METRICS:
          raise ValueError(
              "nonlinear metric must be exactly 'CC' or 'mattes'; "
              f"got {metric!r}"
          )
      name = cast(NonlinearMetricName, metric)
      if name == "mattes":
          if radius is not None:
              raise ValueError("nonlinear radius is valid only for metric 'CC'")
          return ResolvedNonlinearMetric(
              metric=name, radius=None, bins=MATTES_BINS
          )
      if radius is None:
          radius = DEFAULT_CC_RADIUS
      if type(radius) is not int:
          raise TypeError(
              "nonlinear CC radius must be a positive built-in integer; "
              f"got {radius!r}"
          )
      if radius < 1:
          raise ValueError(
              "nonlinear CC radius must be at least 1; " f"got {radius!r}"
          )
      return ResolvedNonlinearMetric(metric=name, radius=radius, bins=None)
  ```

- [ ] **Step 4: Run resolver tests and verify GREEN**

  Run `uv run pytest tests/unit/test_nonlinear_metric.py -q` with the required uv
  exports. Expected: all resolver tests pass.

- [ ] **Step 5: Add failing low-level kernel tests**

  In `tests/unit/test_nonlinear.py`, add tests using `install_fake_ants`:

  ```python
  @pytest.mark.parametrize(("radius", "expected"), [(None, 4), (2, 2)])
  def test_nonlinear_cc_forwards_effective_radius(
      monkeypatch: pytest.MonkeyPatch,
      tmp_path: Path,
      radius: int | None,
      expected: int,
  ) -> None:
      calls = install_fake_ants(monkeypatch, "cyexm.registration.nonlinear", tmp_path)
      warp_path: Path | None = None
      try:
          _, warp_path = register_image(
              moving_image=fake_image(),
              fixed_image=fake_image(),
              metric="CC",
              radius=radius,
          )
          assert calls.registration["syn_metric"] == "CC"
          assert calls.registration["syn_sampling"] == expected
      finally:
          discard_path(warp_path)


  def test_invalid_metric_policy_fails_before_scientific_runtime(
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
      monkeypatch.setattr(
          nonlinear,
          "_scientific_runtime",
          lambda: (_ for _ in ()).throw(AssertionError("runtime loaded")),
      )
      with pytest.raises(ValueError, match="radius"):
          register_image(
              moving_image=fake_image(),
              fixed_image=fake_image(),
              metric="mattes",
              radius=4,
          )
  ```

  Retain the existing exact Mattes/32 parameter assertion unchanged as the Phase 1
  compatibility regression.

- [ ] **Step 6: Run kernel tests and verify RED**

  Run `uv run pytest tests/unit/test_nonlinear.py -k 'metric or radius or established_registration_parameters' -q`.
  Expected: the new tests fail because `register_image` does not accept the keywords.

- [ ] **Step 7: Resolve policy before runtime and forward it to ANTSpyX**

  In `src/cyexm/registration/nonlinear.py`, import the pure resolver and add the
  keyword-only parameters. Resolve them immediately after registration type/accuracy
  validation and before `_configured_runtime`:

  ```python
  resolved_metric = resolve_nonlinear_metric(metric, radius)
  ants, np = _configured_runtime(native_threads)
  ```

  Replace fixed values in `kwargs` with:

  ```python
  "syn_metric": resolved_metric.metric,
  "syn_sampling": resolved_metric.syn_sampling,
  ```

- [ ] **Step 8: Run and commit Task 1**

  Run:

  ```bash
  uv run pytest tests/unit/test_nonlinear_metric.py tests/unit/test_nonlinear.py -q
  uv run ruff check src/cyexm/_nonlinear_metric.py src/cyexm/registration/nonlinear.py tests/unit/test_nonlinear_metric.py tests/unit/test_nonlinear.py
  uv run ruff format --check src/cyexm/_nonlinear_metric.py src/cyexm/registration/nonlinear.py tests/unit/test_nonlinear_metric.py tests/unit/test_nonlinear.py
  ```

  Expected: all tests and checks pass.

  Commit:

  ```bash
  git add src/cyexm/_nonlinear_metric.py src/cyexm/registration/nonlinear.py tests/unit/test_nonlinear_metric.py tests/unit/test_nonlinear.py
  git commit -m "feat: add nonlinear metric policy"
  ```

### Task 2: Normalize the strict schema-v1 nonlinear model

**Files:**
- Modify: `src/cyexm/config/models.py`
- Modify: `src/cyexm/config/dataset.py`
- Modify: `src/cyexm/config/__init__.py`
- Modify: `tests/unit/test_config_dataset.py`

**Interfaces:**
- Consumes: `resolve_nonlinear_metric` from Task 1.
- Produces: `NonlinearRegistrationStageSpec(type: str, accuracy: str, metric: NonlinearMetricName, radius: int | None)` and `RegistrationSpec.nonlinear: NonlinearRegistrationStageSpec`.
- Produces: strict YAML where `metric` is required, CC radius is optional/default 4, and Mattes rejects radius.

- [ ] **Step 1: Update test fixtures to declare Mattes explicitly, then add RED schema tests**

  Add `metric: "mattes"` to the nonlinear mapping returned by `_valid_data()` so
  unrelated validation tests retain existing intent. Add focused tests:

  ```python
  def test_cc_omitted_radius_normalizes_to_four(tmp_path: Path) -> None:
      data = _valid_data()
      data["registration"]["nonlinear"]["metric"] = "CC"
      manifest = _load_data(tmp_path, data)
      assert manifest.registration.nonlinear.metric == "CC"
      assert manifest.registration.nonlinear.radius == 4


  def test_cc_explicit_radius_is_preserved(tmp_path: Path) -> None:
      data = _valid_data()
      nonlinear = data["registration"]["nonlinear"]
      nonlinear.update(metric="CC", radius=2)
      assert _load_data(tmp_path, data).registration.nonlinear.radius == 2


  def test_mattes_has_no_radius(tmp_path: Path) -> None:
      manifest = _load_data(tmp_path, _valid_data())
      assert manifest.registration.nonlinear.metric == "mattes"
      assert manifest.registration.nonlinear.radius is None


  @pytest.mark.parametrize("radius", [None, True, 0, -1, 1.0, "4", [1, 2, 4]])
  def test_invalid_explicit_cc_radius_is_rejected(
      tmp_path: Path, radius: object
  ) -> None:
      data = _valid_data()
      data["registration"]["nonlinear"].update(metric="CC", radius=radius)
      with pytest.raises(ConfigurationValidationError, match="radius"):
          _load_data(tmp_path, data)


  def test_mattes_rejects_radius_key(tmp_path: Path) -> None:
      data = _valid_data()
      data["registration"]["nonlinear"]["radius"] = 4
      with pytest.raises(ConfigurationValidationError, match="radius"):
          _load_data(tmp_path, data)
  ```

  Add a missing-metric test by removing `metric` and asserting a precise
  `registration.nonlinear` missing-key error. Extend direct-model forgery tests with
  invalid nonlinear model, metric, and radius cases.

- [ ] **Step 2: Run config tests and verify RED**

  Run `uv run pytest tests/unit/test_config_dataset.py -k 'metric or radius or nonlinear' -q`.
  Expected: new model/import assertions and normalized-radius tests fail.

- [ ] **Step 3: Add the dedicated model and specialized parser**

  In `models.py`, add:

  ```python
  @dataclass(frozen=True, slots=True)
  class NonlinearRegistrationStageSpec:
      type: str
      accuracy: str
      metric: NonlinearMetricName
      radius: int | None
  ```

  Change `RegistrationSpec.nonlinear` to this type and export it from
  `cyexm.config`.

  In `dataset.py`, split stage key policy:

  ```python
  _LINEAR_STAGE_KEYS = frozenset({"type", "accuracy"})
  _NONLINEAR_STAGE_KEYS = frozenset({"type", "accuracy", "metric"})
  _NONLINEAR_CC_STAGE_KEYS = _NONLINEAR_STAGE_KEYS | frozenset({"radius"})
  ```

  Implement `_nonlinear_stage` by selecting the exact key set based only on whether
  the `radius` key is present and requiring `metric`. When the key is present, parse
  it with `built_in_int(..., minimum=1)` before calling the resolver; this preserves
  the distinction between omission and explicit YAML `null` and rejects booleans,
  floats, strings, subclasses, zero, and negative values. Reject any radius key for
  Mattes. Call `resolve_nonlinear_metric` and store the normalized `metric` and
  `radius`. Catch resolver `TypeError`/`ValueError` and raise
  `ConfigurationValidationError` with `registration.nonlinear.metric` or `.radius`
  in the message.

- [ ] **Step 4: Revalidate directly constructed models through normalized mappings**

  Replace the shared linear/nonlinear model loop with type-specific checks. Rebuild
  the nonlinear mapping as:

  ```python
  nonlinear_mapping: dict[str, object] = {
      "type": manifest.registration.nonlinear.type,
      "accuracy": manifest.registration.nonlinear.accuracy,
      "metric": manifest.registration.nonlinear.metric,
  }
  if manifest.registration.nonlinear.metric == "CC":
      nonlinear_mapping["radius"] = manifest.registration.nonlinear.radius
  ```

  Parse it through `_nonlinear_stage` and retain the final exact dataclass-equality
  check.

- [ ] **Step 5: Run and commit Task 2**

  Run:

  ```bash
  uv run pytest tests/unit/test_config_dataset.py tests/unit/test_workflow_models.py -q
  uv run ruff check src/cyexm/config tests/unit/test_config_dataset.py tests/unit/test_workflow_models.py
  uv run basedpyright src/cyexm/config
  ```

  Expected: all schema/model tests, lint, and typing pass.

  Commit:

  ```bash
  git add src/cyexm/config tests/unit/test_config_dataset.py tests/unit/test_workflow_models.py
  git commit -m "feat: add CC metric to workflow schema"
  ```

### Task 3: Carry effective metric identity through plans, snapshots, and task validation

**Files:**
- Modify: `src/cyexm/workflow/planning.py`
- Modify: `src/cyexm/workflow/status.py`
- Modify: `src/cyexm/workflow/api.py`
- Modify: `src/cyexm/workflow/_execution_validation.py`
- Modify: `tests/unit/test_workflow_planning.py`
- Modify: `tests/unit/test_workflow_status.py`
- Modify: `tests/unit/test_workflow_api.py`
- Modify: `tests/unit/test_round_task.py`
- Modify: other unit fixtures constructing `RegistrationSpec`, as identified by `rg -n 'RegistrationStageSpec\(type="SyN' tests`

**Interfaces:**
- Consumes: normalized `NonlinearRegistrationStageSpec` from Task 2.
- Produces: CC plan mapping `{type, accuracy, metric, radius}` and Mattes plan mapping `{type, accuracy, metric, bins}`.
- Produces: status `_RegistrationContract` fields `nonlinear_metric`, `nonlinear_radius`, and exact reconstruction of `RoundTask.registration`.

- [ ] **Step 1: Update direct task fixtures and add RED canonical-plan tests**

  Replace nonlinear `RegistrationStageSpec(...)` constructions with
  `NonlinearRegistrationStageSpec(..., metric="mattes", radius=None)` unless a test
  intentionally exercises CC. Add planner tests that build manifests differing only
  by omitted versus explicit radius 4 and assert:

  ```python
  assert omitted.scientific_json == explicit.scientific_json
  assert omitted.scientific_fingerprint == explicit.scientific_fingerprint
  assert json.loads(omitted.scientific_json)["registration"]["nonlinear"] == {
      "type": "SyNOnly",
      "accuracy": "high",
      "metric": "CC",
      "radius": 4,
  }
  ```

  Add a radius-2 comparison asserting different JSON and fingerprint. Add a Mattes
  plan assertion containing `metric: mattes` and `bins: 32`.

- [ ] **Step 2: Run workflow plan/status tests and verify RED**

  Run:

  ```bash
  uv run pytest tests/unit/test_workflow_planning.py tests/unit/test_workflow_status.py -k 'metric or radius or registration' -q
  ```

  Expected: canonical mapping/status parsing lack the new fields or reject the new
  exact schema.

- [ ] **Step 3: Serialize metric-specific canonical plan evidence**

  Add a pure planning helper:

  ```python
  def _nonlinear_mapping(stage: NonlinearRegistrationStageSpec) -> dict[str, object]:
      if stage.metric == "CC":
          return {
              "type": stage.type,
              "accuracy": stage.accuracy,
              "metric": "CC",
              "radius": stage.radius,
          }
      return {
          "type": stage.type,
          "accuracy": stage.accuracy,
          "metric": "mattes",
          "bins": MATTES_BINS,
      }
  ```

  Use it in `_scientific_mapping`. The normalized model guarantees CC radius is an
  integer and Mattes radius is `None`.

- [ ] **Step 4: Parse exact plan variants and reconstruct the normalized task**

  In `status.py`, separate linear keys from CC and Mattes nonlinear keys. Parse
  `metric` first from a mapping, select the exact expected set, validate radius or
  fixed bins 32, and store normalized fields in `_RegistrationContract`. Construct
  `NonlinearRegistrationStageSpec` in `_task_from_snapshot`.

  Extend `_registration_matches_snapshot`, workflow API validation, and
  `_execution_validation` to compare and strictly validate metric/radius. Mattes must
  have `radius is None`; CC must have a positive built-in integer.

- [ ] **Step 5: Add persisted-plan failure and immutable-task tests**

  Add tests that reject:

  - the old `{type, accuracy}` nonlinear plan;
  - CC without effective radius in the persisted plan;
  - CC with boolean/zero radius;
  - Mattes with radius or a bin value other than 32;
  - a directly forged RoundTask with an invalid nonlinear model.

  Assert failures occur before registration or scheduler construction.

- [ ] **Step 6: Run and commit Task 3**

  Run:

  ```bash
  uv run pytest tests/unit/test_workflow_planning.py tests/unit/test_workflow_status.py tests/unit/test_workflow_api.py tests/unit/test_round_task.py -q
  uv run ruff check src/cyexm/workflow tests/unit/test_workflow_planning.py tests/unit/test_workflow_status.py tests/unit/test_workflow_api.py tests/unit/test_round_task.py
  uv run basedpyright src/cyexm/workflow
  ```

  Expected: canonical identity, strict plan parsing, task reconstruction, and workflow
  API tests pass.

  Commit all fixture/model updates needed for this coherent workflow slice:

  ```bash
  git add src/cyexm/workflow tests/unit
  git commit -m "feat: fingerprint nonlinear metric policy"
  ```

### Task 4: Propagate metric policy through Phase 1 orchestration and cache provenance

**Files:**
- Modify: `src/cyexm/registration/_request_policy.py`
- Modify: `src/cyexm/registration/whole_image.py`
- Modify: `src/cyexm/registration/chunked.py`
- Modify: `src/cyexm/workflow/round_task.py`
- Modify: `src/cyexm/workflow/_round_provenance.py`
- Modify: `tests/unit/test_whole_image.py`
- Modify: `tests/unit/test_chunked_registration.py`
- Modify: `tests/unit/test_round_task.py`
- Modify: `tests/integration/test_workflow_round_task.py`

**Interfaces:**
- Consumes: `resolve_nonlinear_metric` and the normalized workflow model.
- Produces: Phase 1 keyword-only `nonlinear_metric`/`nonlinear_radius`, metric-aware `nonlinear_parameters(accuracy, metric, radius)`, and identical runtime/status request evidence.
- Preserves: default Phase 1 Mattes/32 request dictionaries byte-for-byte where no new arguments are supplied.

- [ ] **Step 1: Add RED Phase 1 forwarding and request-policy tests**

  Add tests proving:

  ```python
  registration = ImageRegistration(
      fixed_path,
      moving_path,
      output,
      nonlinear_metric="CC",
  )
  assert registration.nonlinear_metric == "CC"
  assert registration.nonlinear_radius == 4
  ```

  For a fake backend registration, assert every whole-image or local-chunk nonlinear
  call receives `metric="CC"` and the selected radius. Add direct tests:

  ```python
  assert nonlinear_parameters("high", "CC", None)["syn_sampling"] == 4
  assert nonlinear_parameters("high", "CC", 2)["syn_sampling"] == 2
  assert nonlinear_parameters("high", "mattes", None)["syn_sampling"] == 32
  ```

  Add a restart test: first commit affine and CC-radius-4 nonlinear evidence, rerun
  with radius 2, assert the affine is reused while nonlinear backend calls and
  descendants are recomputed.

- [ ] **Step 2: Run focused orchestration tests and verify RED**

  Run:

  ```bash
  uv run pytest tests/unit/test_whole_image.py tests/unit/test_chunked_registration.py -k 'metric or radius or reuse' -q
  ```

  Expected: constructors reject new keywords or request evidence remains Mattes/32.

- [ ] **Step 3: Make request-policy builders use the shared resolver**

  Change the signature to:

  ```python
  def nonlinear_parameters(
      accuracy: str,
      metric: object = "mattes",
      radius: object = None,
  ) -> dict[str, object]:
      resolved = resolve_nonlinear_metric(metric, radius)
      return {
          "aff_random_sampling_rate": 1.0,
          "flow_sigma": 3.0,
          "initial_transform": "Identity",
          "mask_all_stages": True,
          "reg_iterations": _NONLINEAR_ITERATIONS[accuracy],
          "singleprecision": False,
          "syn_metric": resolved.metric,
          "syn_sampling": resolved.syn_sampling,
          "total_sigma": 0.5,
          "verbose": True,
      }
  ```

  Add `nonlinear_metric` and `nonlinear_radius` arguments to every nonlinear stage,
  local-chunk, and chunk-manifest parameter builder. Do not add them to affine
  parameters.

- [ ] **Step 4: Add normalized Phase 1 constructor state and forwarding**

  Add keyword-only `nonlinear_metric: object = "mattes"` and
  `nonlinear_radius: object = None` to `ImageRegistration`,
  `ChunkedImageRegistration`, and `register_round`. Resolve once in
  `ImageRegistration.__init__` and store:

  ```python
  resolved = resolve_nonlinear_metric(nonlinear_metric, nonlinear_radius)
  self.nonlinear_metric = resolved.metric
  self.nonlinear_radius = resolved.radius
  ```

  Pass both to `backend.nonlinear`, every request-policy call, and chunk-manifest
  evidence. Preserve existing default request equality for Mattes/32.

- [ ] **Step 5: Forward workflow task policy and reconstruct identical status evidence**

  In `round_task.py`, add:

  ```python
  "nonlinear_metric": task.registration.nonlinear.metric,
  "nonlinear_radius": task.registration.nonlinear.radius,
  ```

  Update `_round_provenance.py` expected whole-image, local-chunk, manifest, and full
  warp requests to pass the same values. Tests must compare complete request
  dictionaries rather than checking only that fields exist.

- [ ] **Step 6: Prove downstream invalidation and affine reuse**

  Extend unit and workflow-round integration fakes to run once with radius 4 and
  again with radius 2 against the same affine inputs. Assert:

  - affine backend call count does not increase;
  - every nonlinear chunk or whole-image warp is recomputed;
  - registered scalar and dependent channels become stale through upstream warp
    identity;
  - runtime `status` and `_round_provenance` report the same incomplete/complete
    state.

- [ ] **Step 7: Run and commit Task 4**

  Run:

  ```bash
  uv run pytest tests/unit/test_whole_image.py tests/unit/test_chunked_registration.py tests/unit/test_round_task.py -q
  uv run pytest tests/integration/test_workflow_round_task.py -q
  uv run ruff check src/cyexm/registration src/cyexm/workflow/round_task.py src/cyexm/workflow/_round_provenance.py tests/unit/test_whole_image.py tests/unit/test_chunked_registration.py tests/unit/test_round_task.py tests/integration/test_workflow_round_task.py
  uv run basedpyright src/cyexm/registration src/cyexm/workflow
  ```

  Expected: Phase 1 compatibility, CC forwarding, request identity, invalidation, and
  status/runtime agreement all pass.

  Commit:

  ```bash
  git add src/cyexm/registration src/cyexm/workflow/round_task.py src/cyexm/workflow/_round_provenance.py tests/unit/test_whole_image.py tests/unit/test_chunked_registration.py tests/unit/test_round_task.py tests/integration/test_workflow_round_task.py
  git commit -m "feat: propagate CC policy through registration"
  ```

### Task 5: Add a bounded real-ANTs CC integration characterization

**Files:**
- Modify: `tests/integration/test_registration_kernels.py`
- Modify: `tests/integration/AGENTS.md`

**Interfaces:**
- Consumes: low-level `nonlinear.register_image` from Task 1.
- Produces: real ANTSpyX evidence that scalar CC radius is accepted, a warp is materialized, output remains finite, and exact fixed geometry is preserved.
- Does not produce: a performance claim or an optimized radius.

- [ ] **Step 1: Add the bounded integration test**

  Add a small deterministic volume test using existing integration helpers and one
  native thread. Construct fixed and moving float images with identical geometry and
  several smooth asymmetric structures; scale and offset moving intensity so raw
  brightness differs while correspondence remains. Call:

  ```python
  registered, warp_path = nonlinear.register_image(
      moving_image=moving,
      fixed_image=fixed,
      registration_type="SyNOnly",
      accuracy="dry run",
      metric="CC",
      radius=2,
      native_threads=NativeThreadConfig(itk=1, blas=1),
  )
  ```

  Assert the warp exists and is readable, registered data are finite, and registered
  shape/origin/spacing/direction exactly match fixed. Do not assert convergence or
  that radius 2/4 is superior.

- [ ] **Step 2: Run the integration test and inspect native evidence**

  Run:

  ```bash
  uv run pytest tests/integration/test_registration_kernels.py -k 'cc and radius' -vv
  ```

  Expected: PASS with a real materialized displacement field and no geometry change.
  If ANTSpyX rejects CC/radius, preserve the failure output and stop rather than
  weakening the scientific contract.

- [ ] **Step 3: Document the characterization boundary and commit**

  Add to `tests/integration/AGENTS.md` that metric changes require a bounded real-ANTs
  acceptance test, while dataset-specific optimization remains explicitly selected
  HPC/scientific acceptance.

  Run the complete kernel integration module and Ruff, then commit:

  ```bash
  uv run pytest tests/integration/test_registration_kernels.py -q
  uv run ruff check tests/integration/test_registration_kernels.py
  git add tests/integration/test_registration_kernels.py tests/integration/AGENTS.md
  git commit -m "test: characterize scalar CC registration"
  ```

### Task 6: Synchronize public documentation, examples, migration, and guidance

**Files:**
- Modify: `README.md`
- Modify: `examples/spheroid-dataset.yaml`
- Modify: `docs/migration-from-clearex.md`
- Modify: `docs/brightness-dependence-investigation.md`
- Modify: `AGENTS.md`
- Modify: `src/cyexm/config/AGENTS.md`
- Modify: `src/cyexm/registration/AGENTS.md`
- Modify: `src/cyexm/workflow/AGENTS.md`
- Modify: `tests/AGENTS.md`
- Modify: `tests/unit/AGENTS.md`
- Modify: `docs/AGENTS.md`
- Modify: `examples/AGENTS.md`
- Modify: `tests/unit/test_phase2_examples.py`

**Interfaces:**
- Consumes: implemented public YAML and Python signatures.
- Produces: strict-loader-compatible examples and maintained descriptions of current behavior.

- [ ] **Step 1: Update the example and its exact-model regression**

  Change `examples/spheroid-dataset.yaml` to:

  ```yaml
  nonlinear:
    type: SyNOnly
    accuracy: high
    metric: CC
    # Optional scalar voxel radius. Omission uses the provisional default of 4.
  ```

  Update `test_phase2_examples.py` to expect a normalized CC radius of 4. Run the
  example test and `cyexm workflow validate` against temporary fixture inputs only.

- [ ] **Step 2: Update README and migration documentation**

  Document:

  - required `metric` under schema version 1;
  - optional scalar `radius` for CC and default 4;
  - explicit `metric: mattes` for legacy workflow behavior;
  - Mattes/32 Phase 1 Python defaults;
  - old scientific plans failing the revised exact schema;
  - one scalar voxel radius and physically anisotropic neighborhoods for anisotropic
    voxel spacing;
  - radius 4 as provisional, with no optimality claim.

  Keep code snippets synchronized with actual signatures.

- [ ] **Step 3: Correct the NCC evidence table**

  In `docs/brightness-dependence-investigation.md`, replace delta-only columns with:

  | Round | Raw NCC current -> corrected | Gradient NCC current -> corrected |
  | --- | ---: | ---: |
  | 2 | 0.807 -> 0.907 | 0.557 -> 0.797 |
  | 3 | 0.656 -> 0.883 | 0.433 -> 0.747 |
  | 4 | 0.675 -> 0.848 | 0.420 -> 0.695 |

  Explain that NCC spans -1 to +1, the values are absolute correlations, and the
  diagnostic is a local 2-D MIP screen rather than the exact 3-D ANTs CC objective.

- [ ] **Step 4: Update only applicable scope-specific AGENTS guidance**

  Record the new contract without duplicating parent text:

  - root: workflow metric is explicit; Phase 1 default remains legacy Mattes;
  - config: exact metric/radius rules;
  - registration: shared resolver, scalar radius, and request/runtime agreement;
  - workflow: effective policy is scientific identity;
  - tests: metric changes require pure, provenance, invalidation, and real-ANTs tests;
  - docs/examples: show current CC/default-4 behavior and provisional status.

- [ ] **Step 5: Scan for stale schema examples and run documentation checks**

  Run:

  ```bash
  rg -n -U 'nonlinear:\n(?:.*\n){0,5}?\s+accuracy:' README.md docs examples tests -g '*.md' -g '*.yaml' -g '*.py'
  uv run pytest tests/unit/test_phase2_examples.py tests/unit/test_config_dataset.py -q
  uv run ruff check tests/unit/test_phase2_examples.py
  git diff --check
  ```

  Inspect every public YAML block found by `rg`; each schema-v1 nonlinear mapping must
  declare `metric`. Historical design/plan evidence may remain unchanged when clearly
  labeled historical.

- [ ] **Step 6: Commit Task 6**

  ```bash
  git add README.md examples/spheroid-dataset.yaml docs/migration-from-clearex.md docs/brightness-dependence-investigation.md AGENTS.md src/cyexm/config/AGENTS.md src/cyexm/registration/AGENTS.md src/cyexm/workflow/AGENTS.md tests/AGENTS.md tests/unit/AGENTS.md docs/AGENTS.md examples/AGENTS.md tests/unit/test_phase2_examples.py
  git commit -m "docs: explain configurable CC registration"
  ```

### Task 7: Run full portable verification and review every acceptance criterion

**Files:**
- Verify: all files changed in Tasks 1-6
- Verify: `docs/superpowers/specs/2026-07-21-cc-syn-metric-design.md`
- Verify: `docs/superpowers/plans/2026-07-21-cc-syn-metric.md`

**Interfaces:**
- Consumes: all prior task deliverables.
- Produces: verified implementation evidence without scheduler or production-data mutation.

- [ ] **Step 1: Run locked-environment and portable repository gates**

  Run with the required uv exports:

  ```bash
  uv lock --check
  uv sync --locked --group dev --extra examples --extra hpc
  uv run pytest -m "unit and not hpc" tests/unit -v
  uv run pytest -m "integration and not hpc" tests/integration -v
  uv run ruff check .
  uv run ruff format --check .
  uv run basedpyright
  uv run pre-commit run --all-files
  ```

  Expected: every command exits zero. Do not run HPC-marked tests.

- [ ] **Step 2: Review design acceptance criteria line by line**

  Confirm with code/test evidence:

  1. schema v1 requires `metric` and supports exactly CC/Mattes;
  2. omitted/explicit radius 4 identity is equal;
  3. only positive built-in scalar CC radii are accepted;
  4. Phase 1 defaults remain Mattes/32;
  5. runtime and request evidence share the resolver;
  6. plans/tasks/status/whole/chunk provenance include effective policy;
  7. radius changes preserve affine reuse and invalidate nonlinear descendants;
  8. real ANTs preserves geometry/artifact contracts;
  9. docs/examples/AGENTS are synchronized;
  10. no scheduler or production dataset was touched.

- [ ] **Step 3: Inspect repository scope and history**

  Run:

  ```bash
  git diff --check
  git log --oneline --decorate -12
  git status --short --branch
  ```

  Expected: coherent focused commits; no staged or tracked SLURM logs; only the three
  known untracked SLURM logs may remain outside version control.

- [ ] **Step 4: Report exact evidence**

  Report command outcomes and test counts, the current HEAD, the normalized schema
  example, explicit confirmation that radius 4 remains provisional, and any genuine
  environmental limitation. Do not claim radius optimization or improved spheroid
  registration without a later approved-data study.
