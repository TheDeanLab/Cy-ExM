# Phase 2E Stochastic Registration Parity Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the invalid multithreaded voxel-repeatability gate with representative stochastic scientific equivalence and a separate bounded, seeded, one-thread sequential-versus-SLURM numerical-parity gate.

**Architecture:** The approved-dataset 8/8 run keeps exact structural, provenance, durability, placement, and cleanup assertions while gating finite stage metrics at an absolute difference of `0.01`; transform and voxel differences become recorded characterization. A fresh one-thread coordinator subprocess creates bounded synthetic TIFFs, runs the same scientific plan sequentially and through the production SLURM executor with a fixed ANTs seed, and retains strict numerical comparison. Both tiers must pass before the existing failure, restart, cancellation, and cleanup evidence can close Phase 2E.

**Tech Stack:** Python 3.12, pytest 9, NumPy, tifffile, ANTSpyX 0.6.3, Dask/distributed 2025.1.0, dask-jobqueue 0.9.0, SLURM

## Global Constraints

- Work only in the isolated `agent/phase2e-acceptance-close` worktree until the documentation agent finishes moving `main`.
- Do not modify production registration policy, schema version 1, scientific fingerprints, cache identity, CC-radius behavior, or default native threads.
- Do not access, modify, cancel, or reuse the active spheroid CC-radius runs.
- Keep external inputs, results, evidence, and SLURM logs on `/archive`; keep environments, caches, and worker scratch on `/project`.
- Before every `uv` command, export `UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache` and `UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python`.
- Portable tests must remain scheduler-free. Only commands containing exactly `-m hpc` may submit SLURM jobs.
- The Tier 1 metric tolerance is an absolute `0.01` for both `Correlation` and `MattesMutualInformation` at every persisted stage; it is dataset-specific and not a universal quality threshold.
- Tier 2 uses `ANTS_RANDOM_SEED=20260722` and exactly one ITK and one BLAS-family thread. The seed is acceptance-only instrumentation and never enters public YAML.
- Every owned client and cluster closes on success and failure; final cleanup queries use current user plus exact job name and never mutate unrelated jobs.

---

## File and responsibility map

| Path | Responsibility |
| --- | --- |
| `tests/hpc/workflow_acceptance.py` | Pure comparison evidence, deterministic profiles/manifest/input builders, seeded runner, and owned SLURM helpers |
| `tests/hpc/deterministic_parity.py` | Fresh one-thread coordinator entry point for Tier 2 |
| `tests/hpc/test_slurm_workflow.py` | Tier 1 orchestration, Tier 2 subprocess invocation, durability scenarios, and final combined evidence |
| `tests/unit/test_hpc_workflow_acceptance.py` | Scheduler-free TDD coverage for comparison policy, bounded synthetic inputs, seed restoration, child invocation, and ownership |
| `tests/hpc/AGENTS.md` | Exact two-tier acceptance and seed/thread boundary |
| `docs/phase2e-multithreaded-registration-reproducibility.md` | Scientific failure/root-cause record |
| `docs/phase-2-acceptance.md` | Final audit created only after both tiers and all release gates pass |
| `README.md` | Final Phase 2 acceptance status after external evidence succeeds |
| `tests/unit/test_repository_policy.py` | Final audit and current-status policy |

---

### Task 1: Separate structural equivalence, metric equivalence, and numeric characterization

**Files:**
- Modify: `tests/hpc/workflow_acceptance.py:305-524`
- Modify: `tests/unit/test_hpc_workflow_acceptance.py`

**Interfaces:**
- Produces: `numeric_comparison_evidence(expected, observed, *, location, require_close, rtol=1e-4, atol=1e-5) -> dict[str, object]`
- Produces: `metric_comparison_evidence(reference, candidate, *, location, absolute_tolerance=0.01) -> dict[str, object]`
- Produces: `scientific_equivalence_evidence(reference: WorkflowPlan, candidate: WorkflowPlan) -> dict[str, object]`
- Preserves: `scientific_parity_evidence(reference, candidate) -> dict[str, object]` as the strict Tier 2 comparator

- [ ] **Step 1: Add failing pure numeric-policy tests**

Import `metric_comparison_evidence` and `numeric_comparison_evidence` in `tests/unit/test_hpc_workflow_acceptance.py` and add:

```python
def test_stochastic_numeric_evidence_records_difference_without_requiring_repeatability() -> None:
    import numpy as np

    evidence = numeric_comparison_evidence(
        np.array([0, 100, 200], dtype=np.uint16),
        np.array([0, 102, 197], dtype=np.uint16),
        location="registered image",
        require_close=False,
    )

    assert evidence["array_equal"] is False
    assert evidence["maximum_absolute_difference"] == 3.0
    assert evidence["mean_absolute_difference"] == pytest.approx(5.0 / 3.0)
    assert evidence["close_required"] is False


def test_strict_numeric_evidence_rejects_the_same_difference() -> None:
    import numpy as np

    with pytest.raises(AssertionError):
        numeric_comparison_evidence(
            np.array([0, 100, 200], dtype=np.uint16),
            np.array([0, 102, 197], dtype=np.uint16),
            location="registered image",
            require_close=True,
        )


def test_metric_equivalence_accepts_observed_repeat_variation_and_rejects_excess() -> None:
    accepted = metric_comparison_evidence(
        {"Correlation": 0.8622, "MattesMutualInformation": 0.3864},
        {"Correlation": 0.8661, "MattesMutualInformation": 0.3882},
        location="round 2 nonlinear metrics",
    )
    assert accepted["absolute_tolerance"] == 0.01
    assert accepted["absolute_differences"] == pytest.approx(
        {"Correlation": 0.0039, "MattesMutualInformation": 0.0018}
    )

    with pytest.raises(HpcWorkflowAcceptanceError, match="Correlation"):
        metric_comparison_evidence(
            {"Correlation": 0.86, "MattesMutualInformation": 0.38},
            {"Correlation": 0.84, "MattesMutualInformation": 0.38},
            location="round 2 nonlinear metrics",
        )
```

- [ ] **Step 2: Run the three tests and verify red**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest tests/unit/test_hpc_workflow_acceptance.py \
  -k 'stochastic_numeric or strict_numeric or metric_equivalence' -q
```

Expected: collection fails because the two helper functions do not exist.

- [ ] **Step 3: Implement the two pure comparison helpers**

Move the nested numeric logic out of `scientific_parity_evidence` and implement:

```python
def numeric_comparison_evidence(
    expected: object,
    observed: object,
    *,
    location: str,
    require_close: bool,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict[str, object]:
    import numpy as np

    expected_array = np.asarray(expected)
    observed_array = np.asarray(observed)
    if expected_array.shape != observed_array.shape:
        raise HpcWorkflowAcceptanceError(
            f"{location} shape changed: {expected_array.shape!r} != {observed_array.shape!r}"
        )
    if expected_array.dtype != observed_array.dtype:
        raise HpcWorkflowAcceptanceError(
            f"{location} dtype changed: {expected_array.dtype!r} != {observed_array.dtype!r}"
        )
    if not np.all(np.isfinite(expected_array)) or not np.all(
        np.isfinite(observed_array)
    ):
        raise HpcWorkflowAcceptanceError(f"{location} contains non-finite values")
    if require_close:
        np.testing.assert_allclose(
            observed_array, expected_array, rtol=rtol, atol=atol
        )
    difference = np.abs(
        observed_array.astype(np.float64) - expected_array.astype(np.float64)
    )
    return {
        "shape": list(expected_array.shape),
        "reference_dtype": str(expected_array.dtype),
        "candidate_dtype": str(observed_array.dtype),
        "array_equal": bool(np.array_equal(observed_array, expected_array)),
        "close_required": require_close,
        "maximum_absolute_difference": float(difference.max(initial=0.0)),
        "mean_absolute_difference": float(difference.mean()),
        "p99_9_absolute_difference": float(np.percentile(difference, 99.9)),
        "rtol": rtol,
        "atol": atol,
    }


def metric_comparison_evidence(
    reference: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    location: str,
    absolute_tolerance: float = 0.01,
) -> dict[str, object]:
    import math

    expected_names = {"Correlation", "MattesMutualInformation"}
    if set(reference) != expected_names or set(candidate) != expected_names:
        raise HpcWorkflowAcceptanceError(f"{location} metric schema changed")
    differences: dict[str, float] = {}
    for name in sorted(expected_names):
        left = reference[name]
        right = candidate[name]
        if not math.isfinite(left) or not math.isfinite(right):
            raise HpcWorkflowAcceptanceError(f"{location} {name} is non-finite")
        difference = abs(right - left)
        if difference > absolute_tolerance:
            raise HpcWorkflowAcceptanceError(
                f"{location} {name} difference {difference} exceeds "
                f"{absolute_tolerance}"
            )
        differences[name] = difference
    return {
        "reference": dict(reference),
        "candidate": dict(candidate),
        "absolute_differences": differences,
        "absolute_tolerance": absolute_tolerance,
    }
```

- [ ] **Step 4: Run the focused tests and verify green**

Run the Step 2 command. Expected: all three selected cases pass.

- [ ] **Step 5: Add a failing composed-policy regression**

Extract the current artifact traversal into an internal `_scientific_comparison_evidence(..., require_close: bool, metric_absolute_tolerance: float | None)` and test through injected array/metric readers that the Tier 1 path records non-identical arrays, enforces exact geometry, and rejects a nonlinear metric difference of `0.0100001`. Keep the fixture data under `tmp_path` and use actual canonical metric files written with `write_metrics`.

- [ ] **Step 6: Implement the two public composed policies**

Define:

```python
def scientific_equivalence_evidence(
    reference: WorkflowPlan,
    candidate: WorkflowPlan,
) -> dict[str, object]:
    return _scientific_comparison_evidence(
        reference,
        candidate,
        require_close=False,
        metric_absolute_tolerance=0.01,
    )


def scientific_parity_evidence(
    reference: WorkflowPlan,
    candidate: WorkflowPlan,
) -> dict[str, object]:
    return _scientific_comparison_evidence(
        reference,
        candidate,
        require_close=True,
        metric_absolute_tolerance=0.0,
    )
```

Inside the traversal, read each stage with `cyexm.metrics.read_metrics`, require the nonlinear value to be no smaller than the corresponding pre-registration value for both metrics, call `metric_comparison_evidence`, and call `numeric_comparison_evidence` for affine parameters, warp arrays, registered TIFFs, and channels. Preserve exact transform type/dimension/fixed parameters and exact TIFF/warp geometry.

- [ ] **Step 7: Run all acceptance-helper unit tests**

```bash
uv run pytest tests/unit/test_hpc_workflow_acceptance.py -q
```

Expected: all cases pass without scheduler access.

- [ ] **Step 8: Commit Tier 1 comparison policy**

```bash
git add tests/hpc/workflow_acceptance.py tests/unit/test_hpc_workflow_acceptance.py
git commit -m "test: accept stochastic scientific equivalence"
```

---

### Task 2: Build bounded deterministic acceptance inputs and profiles

**Files:**
- Modify: `tests/hpc/workflow_acceptance.py`
- Modify: `tests/unit/test_hpc_workflow_acceptance.py`

**Interfaces:**
- Extends: `Task5OutputRoots` with `deterministic_inputs`, `deterministic_parallel`, and `deterministic_sequential`
- Produces: `deterministic_acceptance_manifest(input_root: Path) -> DatasetManifest`
- Produces: `deterministic_slurm_profile(profile, output_root) -> ExecutionProfile`
- Produces: `deterministic_sequential_profile(profile, output_root) -> ExecutionProfile`
- Produces: `write_deterministic_inputs(input_root: Path) -> dict[str, object]`
- Produces: `run_seeded_workflow_task(task: WorkflowTask, *, seed: str = "20260722") -> RoundResult`

- [ ] **Step 1: Add failing root-ownership and deterministic-profile tests**

Extend `test_task5_profiles_are_bounded_and_confined_to_named_owned_roots` to require all seven owned children. Add assertions that both deterministic profiles use `NativeThreadSpec(itk=1, blas=1)`, the SLURM profile uses two workers and job name `<approved>-det`, and the sequential profile has no worker scratch or SLURM block.

- [ ] **Step 2: Add failing manifest/input tests**

Add a test that calls `write_deterministic_inputs(tmp_path / "inputs")` and `deterministic_acceptance_manifest(...)`, then asserts:

```python
assert manifest.dataset_id == "phase2e-deterministic-parity"
assert manifest.registration.linear == RegistrationStageSpec(
    type="TRSAA", accuracy="dry run"
)
assert manifest.registration.nonlinear == NonlinearRegistrationStageSpec(
    type="SyNOnly", accuracy="dry run", metric="mattes", radius=None
)
assert tuple(item.strategy for item in manifest.rounds) == (
    "whole_image", "serial_chunked"
)
assert manifest.rounds[1].chunk_size_zyx == (8, 16, 16)
assert manifest.rounds[1].overlap_fraction == 0.25
assert set(input_root.glob("*.tif")) == {
    input_root / "fixed.tif",
    input_root / "moving-whole.tif",
    input_root / "moving-chunked.tif",
}
```

Read every TIFF and require shape `(16, 32, 32)`, dtype `float32`, axes `ZYX`, finite values, and non-identical moving arrays.

- [ ] **Step 3: Run the new tests and verify red**

```bash
uv run pytest tests/unit/test_hpc_workflow_acceptance.py \
  -k 'deterministic or task5_profiles' -q
```

Expected: failures because the roots, builders, and profiles are absent.

- [ ] **Step 4: Implement owned roots and pure synthetic builders**

Create the inputs with `numpy.mgrid`, two offset anisotropic Gaussian structures, and deterministic low-amplitude texture from `numpy.random.default_rng(20260722)`. Write with `tifffile.imwrite(..., metadata={"axes": "ZYX"}, photometric="minisblack")`. The function must require the already-owned input root to be an existing empty real directory, write each TIFF atomically, and return file identities and array facts for evidence.

Build the manifest directly from `DatasetManifest`, `ReferenceSpec`, `RoundSpec`, and registration-stage dataclasses. Do not add YAML or production configuration fields.

- [ ] **Step 5: Implement deterministic profiles**

Use `dataclasses.replace` and the exact one-thread specification:

```python
one_thread = NativeThreadSpec(itk=1, blas=1)
```

The SLURM derivative retains approved queue, cores, memory, interfaces, walltime, and exclusivity; it sets workers to two, retries to zero, and appends `-det` to a job name only if the result remains within the model's length and single-line validation rules.

- [ ] **Step 6: Add the failing seeded-runner restoration test**

Pass a fake `WorkflowTask` and injected production dispatcher. Assert the dispatcher observes `ANTS_RANDOM_SEED=20260722`, then assert the runner restores an absent value and separately restores a pre-existing value after both success and an injected exception.

- [ ] **Step 7: Implement the acceptance-only seeded runner**

```python
def run_seeded_workflow_task(
    task: WorkflowTask,
    *,
    seed: str = "20260722",
    _run_workflow_task: Callable[[WorkflowTask], RoundResult] = _production_workflow_task,
    _environ: MutableMapping[str, str] = os.environ,
) -> RoundResult:
    previous = _environ.get("ANTS_RANDOM_SEED")
    _environ["ANTS_RANDOM_SEED"] = seed
    try:
        return _run_workflow_task(task)
    finally:
        if previous is None:
            _environ.pop("ANTS_RANDOM_SEED", None)
        else:
            _environ["ANTS_RANDOM_SEED"] = previous
```

- [ ] **Step 8: Verify and commit deterministic builders**

```bash
uv run pytest tests/unit/test_hpc_workflow_acceptance.py -q
git add tests/hpc/workflow_acceptance.py tests/unit/test_hpc_workflow_acceptance.py
git commit -m "test: build deterministic parity inputs"
```

Expected: all helper tests pass without scheduler access.

---

### Task 3: Run Tier 2 in a fresh one-thread coordinator subprocess

**Files:**
- Create: `tests/hpc/deterministic_parity.py`
- Modify: `tests/hpc/test_slurm_workflow.py`
- Modify: `tests/hpc/workflow_acceptance.py`
- Modify: `tests/unit/test_hpc_workflow_acceptance.py`

**Interfaces:**
- Produces CLI: `python -m hpc.deterministic_parity --execution PATH --owned-root PATH --evidence PATH`
- Produces evidence: `deterministic-executor-parity.json`
- Produces parent helper: `execute_deterministic_parity_subprocess(...) -> dict[str, object]`

- [ ] **Step 1: Add failing command-construction and environment tests**

Test `execute_deterministic_parity_subprocess` with an injected `subprocess.run`. Require:

```python
assert command[:3] == (sys.executable, "-m", "hpc.deterministic_parity")
assert environment["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] == "1"
assert all(environment[name] == "1" for name in BLAS_THREAD_ENV_VARS)
assert environment["ANTS_RANDOM_SEED"] == "20260722"
assert Path(environment["PYTHONPATH"].split(os.pathsep)[0]).name == "tests"
```

Require a nonzero child exit or missing/malformed evidence to raise `HpcWorkflowAcceptanceError` with captured stdout/stderr paths written under the caller evidence directory.

- [ ] **Step 2: Run the parent-helper tests and verify red**

```bash
uv run pytest tests/unit/test_hpc_workflow_acceptance.py \
  -k deterministic_parity_subprocess -q
```

Expected: failure because the helper does not exist.

- [ ] **Step 3: Implement the fail-closed parent subprocess helper**

Build an explicit argument tuple, copy the environment, overwrite all six native-thread variables and `ANTS_RANDOM_SEED`, prepend the repository `tests` directory to `PYTHONPATH`, and invoke with `check=False`, captured text, the repository root as `cwd`, and a timeout derived from the approved SLURM walltime plus startup allowance. Atomically write child stdout/stderr even on failure. Read and validate the exact evidence schema before returning.

- [ ] **Step 4: Write the deterministic coordinator entry point**

`tests/hpc/deterministic_parity.py` must:

1. parse exactly `--execution`, `--owned-root`, and `--evidence`;
2. require all six native-thread variables to equal `1` and the seed to equal `20260722` before importing NumPy, tifffile, or ANTs;
3. load the approved execution YAML through `load_execution_profile` and derive the two one-thread profiles;
4. call `write_deterministic_inputs` and build both plans;
5. execute the SLURM plan with `execute_owned_slurm_workflow(..., runner=run_seeded_workflow_task, operation="deterministic-parallel")`;
6. execute the sequential plan under the same seed;
7. call strict `scientific_parity_evidence`;
8. require every reported native-thread value to equal one;
9. write `deterministic-executor-parity.json` containing seed, input evidence, both workflow results, strict parity, SLURM execution evidence, and final exact-name queue cleanup; and
10. return nonzero after writing structured failure evidence when any invariant fails.

Imports of scientific libraries must remain inside `main()` after environment validation.

- [ ] **Step 5: Add portable parser/failure tests for the child module**

Invoke `main()` with injected loaders, executors, and comparison functions so no scheduler is contacted. Cover missing arguments, thread mismatch, seed mismatch, malformed owned roots, workflow failure, parity failure, and cleanup failure.

- [ ] **Step 6: Wire Tier 2 into Task 5**

In `_run_task5_acceptance`, call Tier 1 as:

```python
equivalence = scientific_equivalence_evidence(sequential_plan, parallel_plan)
write_evidence_json(context.evidence_dir / "scientific-equivalence.json", equivalence)
```

Then invoke the child after Tier 1 and before injected durability scenarios. Include both records in the final result:

```python
"scientific_equivalence": equivalence,
"deterministic_executor_parity": deterministic_parity,
```

Do not call the strict comparator on the 8/8 approved-dataset pair.

- [ ] **Step 7: Run targeted portable tests**

```bash
uv run pytest tests/unit/test_hpc_workflow_acceptance.py \
  tests/unit/test_hpc_environment.py -q
```

Expected: all tests pass and the scheduler firewall records no external scheduler call.

- [ ] **Step 8: Commit Tier 2 orchestration**

```bash
git add tests/hpc/deterministic_parity.py tests/hpc/test_slurm_workflow.py \
  tests/hpc/workflow_acceptance.py tests/unit/test_hpc_workflow_acceptance.py
git commit -m "test: require deterministic SLURM parity"
```

---

### Task 4: Align guidance and portable policy with the two-tier gate

**Files:**
- Modify: `tests/hpc/AGENTS.md`
- Modify: `docs/AGENTS.md`
- Modify: `tests/unit/test_repository_policy.py`
- Modify: `docs/phase2e-multithreaded-registration-reproducibility.md` only if implementation names differ from the approved design

**Interfaces:**
- Produces: repository policy that forbids a claim of deterministic 8/8 registration while requiring both evidence documents

- [ ] **Step 1: Add failing repository-policy assertions**

Require `tests/hpc/AGENTS.md` to contain `stochastic scientific equivalence`, `metric absolute difference of 0.01`, `ANTS_RANDOM_SEED=20260722`, and `one native thread`. Require the scientific note and approved design to exist. Forbid language asserting that a fixed seed makes multithreaded ANTs registration exact.

- [ ] **Step 2: Run policy tests and verify red**

```bash
uv run pytest tests/unit/test_repository_policy.py -q
```

Expected: the new two-tier guidance phrases are absent.

- [ ] **Step 3: Update only affected guidance**

Add to `tests/hpc/AGENTS.md`:

- approved-dataset 8/8 runs are stochastic scientific-equivalence evidence;
- exact structure remains mandatory and metric absolute differences are bounded at `0.01`;
- voxel/transform deltas are characterization only in Tier 1;
- strict numerical parity requires the separate seeded one-thread subprocess; and
- neither tier authorizes production seed/schema changes.

Add the corresponding documentation boundary to `docs/AGENTS.md`: the root-cause note is scientific characterization and the final acceptance page may claim both tiers only after external evidence passes.

- [ ] **Step 4: Run focused policy and documentation checks**

```bash
uv run pytest tests/unit/test_repository_policy.py \
  tests/unit/test_phase2_documentation.py -q
git diff --check
```

Expected: all tests pass and no whitespace errors are reported.

- [ ] **Step 5: Commit guidance**

```bash
git add tests/hpc/AGENTS.md docs/AGENTS.md tests/unit/test_repository_policy.py \
  docs/phase2e-multithreaded-registration-reproducibility.md
git commit -m "docs: define two-tier Phase 2E parity"
```

---

### Task 5: Run portable gates and integrate the stable documentation branch

**Files:**
- Verify all changed files
- Integrate latest `main` only after the documentation agent reports completion

**Interfaces:**
- Produces: one reviewed candidate commit whose source tree and docs are stable for exact-SHA HPC acceptance

- [ ] **Step 1: Run targeted gates**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv lock --check
uv run pytest tests/unit/test_hpc_workflow_acceptance.py \
  tests/unit/test_hpc_environment.py tests/unit/test_repository_policy.py -q
uv run ruff check tests/hpc tests/unit/test_hpc_workflow_acceptance.py \
  tests/unit/test_repository_policy.py
uv run ruff format --check tests/hpc tests/unit/test_hpc_workflow_acceptance.py \
  tests/unit/test_repository_policy.py
uv run basedpyright tests/hpc tests/unit/test_hpc_workflow_acceptance.py
```

Expected: every command exits zero.

- [ ] **Step 2: Run all portable gates**

```bash
uv run pytest -m "unit and not hpc" tests/unit -q
uv run pytest -m "integration and not hpc" tests/integration -q
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
```

Expected: all tests and static checks pass. Record exact counts and elapsed times.

- [ ] **Step 3: Review isolated branch scope**

```bash
git diff --check main...HEAD
git diff --stat main...HEAD
git status --short
```

Expected: only the approved design, plan, scientific note, HPC harness, unit tests, and affected guidance differ; no SLURM logs or generated evidence are tracked.

- [ ] **Step 4: Wait for a stable documentation `main` and integrate it**

Confirm the documentation agent has stopped committing. Fetch the current `main` SHA, merge it into `agent/phase2e-acceptance-close`, resolve only genuine documentation overlap, and rerun Steps 1-3. Do not rewrite or drop the documentation agent's commits.

- [ ] **Step 5: Request code review and address only actionable findings**

Review against every acceptance criterion in the approved design. Any correction uses a focused commit and reruns the affected targeted tests.

---

### Task 6: Run fresh exact-SHA real-SLURM acceptance

**Files:**
- External: modern approved Zenodo manifest
- External: approved 8/8 SLURM execution profile
- External: new empty evidence and result roots

**Interfaces:**
- Produces: complete Tier 1, Tier 2, placement, failure/resume, cancellation/resume, ownership, and cleanup evidence for one exact commit

- [ ] **Step 1: Create fresh caller-owned roots**

Create these exact new caller-owned paths, each outside the repository and every live spheroid root:

- execution copy: `/archive/bioinformatics/Danuser_lab/Dean/dean/2026-07-21-test-cyexm/configs/execution-slurm-512gb-two-tier-final.yaml`;
- results root: `/archive/bioinformatics/Danuser_lab/Dean/dean/2026-07-21-test-cyexm/results-phase2e-two-tier-final`;
- evidence directory: `/archive/bioinformatics/Danuser_lab/Dean/dean/2026-07-21-test-cyexm/evidence/workflow-two-tier-final`; and
- worker scratch: `/project/bioinformatics/Danuser_lab/Dean/dean/cyexm-hpc-acceptance/2026-07-22-two-tier-final`.

Require all four paths to be absent before setup and the created evidence directory to be empty. Copy the existing approved execution YAML to the new configuration path, change only output/scratch paths and the exact owned job name `cyexm-p2e-two-tier`, and retain 8/8 threads, approved queue, memory, exclusivity, interfaces, and walltime.

- [ ] **Step 2: Export the representative coordinator environment**

```bash
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
```

Do not export `ANTS_RANDOM_SEED` in the parent representative run; Tier 2 owns its fixed child value.

- [ ] **Step 3: Run explicit acceptance**

```bash
uv run pytest -m hpc tests/hpc/test_slurm_workflow.py -v \
  --cyexm-dataset /archive/bioinformatics/Danuser_lab/Dean/dean/2026-07-21-test-cyexm/configs/zenodo-ci-dataset.yaml \
  --cyexm-execution /archive/bioinformatics/Danuser_lab/Dean/dean/2026-07-21-test-cyexm/configs/execution-slurm-512gb-two-tier-final.yaml \
  --cyexm-evidence-dir /archive/bioinformatics/Danuser_lab/Dean/dean/2026-07-21-test-cyexm/evidence/workflow-two-tier-final
```

Monitor owned jobs at approximately five-minute intervals without inspecting or changing unrelated jobs.

- [ ] **Step 4: Validate all machine-readable evidence**

Require:

- `workflow-acceptance.json.status == "completed"`;
- Tier 1 exact structure and every metric difference `<= 0.01`;
- Tier 2 exact arrays/transforms and seed/thread evidence;
- unique overlapping moving-round jobs and hosts;
- completed failure/resume and cancellation/resume evidence;
- confined Task 5 outputs; and
- empty final exact-name queue snapshots for both representative and deterministic job names.

- [ ] **Step 5: Preserve failures without weakening the gate**

If the run fails, preserve its result/evidence roots, identify the first failed invariant, and return to systematic debugging. Never edit passed artifacts, reuse a partial evidence directory, raise a tolerance, or rerun into the same root.

---

### Task 7: Record Phase 2 acceptance and finish the branch

**Files:**
- Create: `docs/phase-2-acceptance.md`
- Modify: `README.md`
- Modify: `tests/unit/test_repository_policy.py`

**Interfaces:**
- Consumes: exact-SHA portable and external evidence
- Produces: final Phase 2E Task 6 audit and accepted README status

- [ ] **Step 1: Add the failing final-audit policy test**

Require the audit to include exact commit, Python/Cy-ExM/ANTSpyX/Dask/distributed/dask-jobqueue/SLURM versions, portable counts/times, profile fingerprint, partition facts, Tier 1 metric bound/results, Tier 2 seed/thread/exact-parity result, job IDs/hosts, failure/restart/cancellation outcomes, cleanup, and external evidence checksums. Forbid large arrays, logs, credentials, and live spheroid paths.

- [ ] **Step 2: Run the policy test and verify red**

```bash
uv run pytest tests/unit/test_repository_policy.py -q
```

Expected: failure because `docs/phase-2-acceptance.md` is absent or incomplete.

- [ ] **Step 3: Write the concise audit and update README status**

Summarize the exact successful evidence. Link the stochastic reproducibility note and correction design. State precisely that the 8/8 gate establishes stochastic scientific equivalence and the seeded one-thread gate establishes strict executor parity; do not claim deterministic multithreaded registration.

- [ ] **Step 4: Run final documentation and release gates**

```bash
uv run pytest tests/unit/test_repository_policy.py \
  tests/unit/test_phase2_documentation.py tests/unit/test_phase2_examples.py -q
uv run pytest -m "unit and not hpc" tests/unit -q
uv run pytest -m "integration and not hpc" tests/integration -q
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pre-commit run --all-files
```

Expected: every command exits zero. Record exact results in the audit.

- [ ] **Step 5: Commit the final acceptance audit**

```bash
git add docs/phase-2-acceptance.md README.md tests/unit/test_repository_policy.py
git commit -m "docs: record Phase 2 runtime acceptance"
```

- [ ] **Step 6: Complete branch review and integrate**

Use `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch`. Merge the reviewed branch into local `main` only after the documentation agent is finished, rerun the final focused checks on merged `main`, and push only with explicit user authorization.
