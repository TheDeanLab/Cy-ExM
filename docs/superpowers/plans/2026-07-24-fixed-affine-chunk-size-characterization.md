# Fixed-Affine Chunk-Size Characterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a reproducible four-condition, nonlinear-only chunk-size experiment
from one validated CC-radius-4 affine state, quantify bright/dim registration
quality and deformation validity, and publish the results with figures in the
Cy-ExM HTML documentation.

**Architecture:** Add a generic explicitly selected HPC characterization harness
beside the existing fine-SLURM acceptance helpers. A custom reference runner validates
the operator-owned completed reference without writing it, while a custom fine
preparation runner clones only the validated moving-crop and affine artifacts into
caller-owned condition roots and then delegates all ordinary preparation, chunk,
assembly, channel, and validation work to Cy-ExM. A separate deterministic analysis
module reads validated outputs and emits compact JSON/CSV evidence and documentation
figures; source images, warps, scheduler logs, and site-specific configuration remain
on `/archive`.

**Tech Stack:** Python 3.12, Cy-ExM internal fine-task APIs, ANTSpyX, NumPy, SciPy,
scikit-image, tifffile, Dask/dask-jobqueue, SLURMCluster, pytest, Matplotlib,
Jupyter Book 2.

## Global Constraints

- The completed radius-4 source tree is read-only.
- All four conditions use CC SyN radius 4, `accuracy: high`, and
  `overlap_fraction: 0.15`.
- Chunk sizes are ZYX `(512, 128, 128)`, `(512, 256, 256)`,
  `(512, 512, 512)`, and `(512, 1125, 1125)`.
- Every condition uses four exclusive 512-GB nodes, one process and one
  `round_slot` per node, 30 cores, 500 GB requested memory, 30 ITK threads, and 30
  threads for each BLAS-family variable.
- TRSAA estimation must never execute; applying the accepted affine is allowed.
- Analyze rounds 2--4 at dim ROI `y=1600:1800, x=300:400` and bright ROI
  `y=1227:1427, x=840:1040`.
- Keep source paths, output paths, large arrays, warps, TIFFs, SLURM logs, and raw
  evidence outside git.
- Do not change the public schema, public workflow CLI, cache-import policy, or
  production registration defaults.
- Use `/project` for environments, caches, and worker scratch; use `/archive` for
  data and experiment outputs.

## Reuse analysis

The closest existing components are `SlurmExecutor`, `FineGraphController`,
`run_fine_task`, `commit_prepared_round`, `ArtifactStore`, and the real-SLURM
acceptance lifecycle in `tests/hpc`. The experiment will reuse them directly:

- `SlurmExecutor` continues to own cluster scaling, worker validation, placement,
  retries, cancellation, and cleanup.
- The ordinary `FineGraphController` continues to materialize the canonical
  preparation/chunk/assembly/channel/validation graph.
- The custom fine runner changes only `RoundPreparationTask`: before delegating to
  `run_fine_task`, it atomically clones the already accepted moving-crop and affine
  artifacts with their exact `StageRequest`.
- Every nonlinear task, assembly, transform, validation, and completion record uses
  the production implementation unchanged.
- The reference runner validates existing completion evidence and returns a truthful
  reused `RoundResult` without calling `run_reference_task`, preventing accidental
  writes into the source tree.

A new public API is unnecessary. The harness remains explicitly selected HPC
scientific characterization code under `tests/hpc`, and its pure analysis helpers
are tested portably.

---

### Task 1: Strict characterization configuration and source-state validation

**Files:**
- Modify: `tests/hpc/AGENTS.md`
- Modify: `tests/hpc/conftest.py`
- Create: `tests/hpc/fixed_affine_chunk_characterization.py`
- Create: `tests/unit/test_fixed_affine_chunk_characterization.py`

**Interfaces:**
- Produces: `FixedAffineCharacterizationSpec`, `SourceRoundState`,
  `load_characterization_spec(path)`, and
  `validate_fixed_affine_source(spec, context)`.
- Consumes: existing `HpcAcceptanceContext`, strict dataset/execution loaders,
  `CompletionRecord`, `FileIdentity`, and canonical archive/project path checks.

- [ ] **Step 1: Add failing strict-loader tests**

Add unit cases that load a temporary YAML mapping with exact keys:

```yaml
schema_version: 1
source_workflow_root: /archive/operator/source
condition_order: [1125, 512, 256, 128]
chunk_sizes_zyx:
  "1125": [512, 1125, 1125]
  "512": [512, 512, 512]
  "256": [512, 256, 256]
  "128": [512, 128, 128]
overlap_fraction: 0.15
rounds: [2, 3, 4]
dim_roi_yx: [1600, 1800, 300, 400]
bright_roi_yx: [1227, 1427, 840, 1040]
search_radius_px: 30
```

Assert exact normalization, rejection of unknown/missing keys, duplicate condition
labels, a 64-pixel condition, any radius other than 30, any overlap other than 0.15,
rounds other than `(2, 3, 4)`, malformed ROIs, symlinked source roots, source roots
inside the new output root, and a writable request that targets the source tree.

- [ ] **Step 2: Run the focused unit test and confirm RED**

Run:

```bash
uv run pytest -m unit tests/unit/test_fixed_affine_chunk_characterization.py -v
```

Expected: collection or import failure because the characterization module and
fixture option do not exist.

- [ ] **Step 3: Implement exact immutable models and loader**

Implement frozen dataclasses with strict built-in types:

```python
@dataclass(frozen=True, slots=True)
class RoiYX:
    y_start: int
    y_end: int
    x_start: int
    x_end: int

@dataclass(frozen=True, slots=True)
class FixedAffineCharacterizationSpec:
    schema_version: int
    source_workflow_root: Path
    condition_order: tuple[str, ...]
    chunk_sizes_zyx: dict[str, tuple[int, int, int]]
    overlap_fraction: float
    rounds: tuple[int, ...]
    dim_roi_yx: RoiYX
    bright_roi_yx: RoiYX
    search_radius_px: int
```

Load YAML with the repository's no-alias/no-tag/no-duplicate policy. Resolve the
source root strictly, require its `scientific-plan.json`, and require condition
values to equal the four approved sizes exactly.

- [ ] **Step 4: Validate every accepted source artifact**

For each round, read the source scientific plan and validate:

```python
@dataclass(frozen=True, slots=True)
class SourceRoundState:
    round: int
    fixed_reference: Path
    moving_image: Path
    moving_crop: Path
    affine: Path
    linear_metrics: Path
    fixed_geometry: ImageGeometry
    source_records: tuple[CompletionRecord, ...]
```

Require current file identities to equal completion records; affine requests to be
TRSAA/high with `prepared_fixed: true`; fixed/moving inputs, round, spacing,
geometry, and all six 30-thread variables to match; and nonlinear source policy to
be SyNOnly/high/CC/radius 4. Write only a detached SHA-256 source manifest beneath
the caller-owned evidence directory.

- [ ] **Step 5: Add the explicit pytest option and fixture**

Add `--cyexm-characterization PATH`. Resolve it only for the new test fixture so
existing HPC tests retain their current command contracts. The fixture must require
the normal exact `-m hpc`, `--run-hpc`, dataset, execution, and evidence options,
then validate the new file before scheduler access.

- [ ] **Step 6: Run unit tests and commit**

Run:

```bash
uv run pytest -m unit tests/unit/test_fixed_affine_chunk_characterization.py -v
uv run ruff check tests/hpc/fixed_affine_chunk_characterization.py \
  tests/hpc/conftest.py tests/unit/test_fixed_affine_chunk_characterization.py
```

Commit:

```bash
git add tests/hpc/AGENTS.md tests/hpc/conftest.py \
  tests/hpc/fixed_affine_chunk_characterization.py \
  tests/unit/test_fixed_affine_chunk_characterization.py
git commit -m "test(hpc): validate fixed-affine chunk characterization"
```

### Task 2: Read-only reference runner and atomic accepted-affine cloning

**Files:**
- Modify: `tests/hpc/fixed_affine_chunk_characterization.py`
- Modify: `tests/unit/test_fixed_affine_chunk_characterization.py`
- Create: `tests/integration/test_fixed_affine_chunk_characterization.py`

**Interfaces:**
- Produces: `run_read_only_reference(task, source_manifest)`,
  `materialize_shared_prepared_state(task, source_round)`,
  `clone_shared_prepared_state(task, shared_round)`, and
  `run_fixed_affine_fine_task(task, shared_rounds)`.
- Consumes: `ArtifactStore.commit`, `run_fine_task`,
  `RegistrationArtifactPaths`, `PreparedRoundPaths`, and source completion records.

- [ ] **Step 1: Write failing no-affine-estimation tests**

Unit tests must inject a sentinel affine estimator that raises
`AssertionError("affine estimation invoked")`. Integration tests use small unequal
ZYX TIFFs, create a real accepted affine and moving-crop record once, then run the
  one shared prepared seed and clone it into two distinct condition roots. Assert:

- the sentinel estimator is never called;
- source artifact and record identities remain unchanged;
- destination affine bytes equal the source bytes;
- affine, crop, and pre-metric requests equal their accepted source requests;
- linear-metric and prepared requests differ only in the exact destination upstream
  identities required by their canonical lineage;
- prepared arrays have exact fixed geometry and byte-identical SHA-256 values across
  conditions;
- destination completion records name destination paths;
- the custom runner delegates non-preparation tasks unchanged.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
uv run pytest -m unit \
  tests/unit/test_fixed_affine_chunk_characterization.py -v
uv run pytest -m integration \
  tests/integration/test_fixed_affine_chunk_characterization.py -v
```

Expected: missing runner and clone functions.

- [ ] **Step 3: Implement atomic cloning through `ArtifactStore`**

For moving crop, pre-metrics, and affine, require the source completion record and
current source identity before and after copying. Commit the destination with the
exact source request:

```python
store.commit(
    destination,
    source_record.request,
    writer=lambda temporary: shutil.copy2(source, temporary),
    validator=lambda candidate: (
        _sha256(candidate) == source_sha256
        and candidate.stat().st_size == source.stat().st_size
    ),
)
```

Reject pre-existing incompatible destinations. Record source and destination hashes
in experiment evidence. Never create or alter a completion record under the source
root.

- [ ] **Step 4: Materialize one shared prepared seed per round**

Create one caller-owned shared registration directory per round. Clone the accepted
crop and affine, then call ordinary `run_fine_task(RoundPreparationTask)` exactly
once. This computes pre/linear metrics, applies the accepted affine once, and commits
the four prepared arrays and manifest without estimating TRSAA. Record every shared
artifact and completion-record hash.

- [ ] **Step 5: Clone shared prepared state into each condition**

Copy the shared affine, crop, pre/linear metrics, and four NPY arrays atomically.
Rebuild only path-dependent upstream identities with `StageRequest.for_identities`.
Create a condition-local `PreparedManifest` containing the destination affine,
crop, and array identities, then commit its exact expected manifest request. Return
a successful preparation `FineTaskResult` containing that descriptor and every
canonical preparation output. The four NPY files must remain byte-identical to the
single shared seed.

- [ ] **Step 6: Implement a truthful read-only reference result**

Validate every declared reference output and completion record, then return an exact
`RoundResult(status="reused")` whose artifacts equal the task outputs and whose
thread evidence equals the configured six variables. Do not call reference
materialization.

- [ ] **Step 7: Implement the custom fine runner**

Use a frozen callable carrying the three validated `SourceRoundState` values:

```python
@dataclass(frozen=True, slots=True)
class FixedAffineFineRunner:
    shared_rounds: tuple[SharedPreparedRound, ...]

    def __call__(self, task: FineWorkflowTask) -> FineTaskResult:
        if type(task) is RoundPreparationTask:
            return clone_shared_prepared_state(
                task, self.by_round[task.round]
            )
        return run_fine_task(task)
```

Only shared-seed construction runs ordinary preparation. Every condition begins
from a cloned successful preparation result; nonlinear chunks and every later stage
execute through ordinary `run_fine_task`. Capture evidence proving no condition
estimated or reapplied affine state.

- [ ] **Step 8: Run focused tests and commit**

Run:

```bash
uv run pytest -m unit \
  tests/unit/test_fixed_affine_chunk_characterization.py -v
uv run pytest -m integration \
  tests/integration/test_fixed_affine_chunk_characterization.py -v
```

Commit:

```bash
git add tests/hpc/fixed_affine_chunk_characterization.py \
  tests/unit/test_fixed_affine_chunk_characterization.py \
  tests/integration/test_fixed_affine_chunk_characterization.py
git commit -m "test(hpc): reuse one accepted affine across chunk conditions"
```

### Task 3: Four-condition fine-SLURM experiment lifecycle

**Files:**
- Modify: `tests/hpc/fixed_affine_chunk_characterization.py`
- Create: `tests/hpc/test_fixed_affine_chunk_characterization.py`
- Modify: `tests/unit/test_fixed_affine_chunk_characterization.py`

**Interfaces:**
- Produces: `build_condition_tasks`, `execute_chunk_condition`, and one explicitly
  selected real-SLURM pytest test.
- Consumes: `SlurmExecutor`, strict `ExecutionProfile`, custom runners from Task 2,
  and existing owned-job evidence/cleanup helpers.

- [ ] **Step 1: Write failing pure lifecycle tests**

With fake executor modules, assert each condition:

- uses a unique dataset ID, output root, scratch root, and job name;
- has only the registration channel for each moving round;
- retains CC radius 4, high accuracy, 0.15 overlap, fixed source affine, and the
  condition's exact chunk size;
- uses four workers, one process, 30 cores, 500 GB, exclusive nodes, and 30/30
  native threads;
- materializes exactly 27, 108, 363, or 1,323 chunk tasks across rounds;
- runs in order 1125, 512, 256, 128;
- writes atomic `condition-summary.json` even on failure;
- closes client before cluster and proves the exact owned queue is empty.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
uv run pytest -m unit \
  tests/unit/test_fixed_affine_chunk_characterization.py -v
```

Expected: missing condition builders and lifecycle.

- [ ] **Step 3: Implement condition planning and execution**

Construct exact `ReferenceTask` and `RoundTask` values from the validated source
plan. Execute:

```python
executor = SlurmExecutor(
    condition_profile,
    _runner=ReadOnlyReferenceRunner(source_reference),
    _fine_runner=FixedAffineFineRunner(source_rounds),
)
results = executor.execute(reference_task, round_tasks)
```

Require successful reference, preparation, every chunk, assembly, one
registration-channel transform, and validation for rounds 2--4. Record exact fine
tasks/results, attempt counts, worker placements, stage intervals, package versions,
git commit, configs, artifact hashes, geometry, and cleanup.

- [ ] **Step 4: Add the real explicitly selected HPC test**

The test calls each condition sequentially and resumes a condition only through
ordinary artifact validation. Before starting the next condition, require successful
cleanup and revalidate all source identities. Fail the test if any condition invokes
an affine estimator, produces a folded/non-finite output, or loses fixed geometry.

- [ ] **Step 5: Run portable policy tests and commit**

Run:

```bash
uv run pytest -m unit \
  tests/unit/test_fixed_affine_chunk_characterization.py -v
uv run ruff check tests/hpc/fixed_affine_chunk_characterization.py \
  tests/hpc/test_fixed_affine_chunk_characterization.py
```

Commit:

```bash
git add tests/hpc/fixed_affine_chunk_characterization.py \
  tests/hpc/test_fixed_affine_chunk_characterization.py \
  tests/unit/test_fixed_affine_chunk_characterization.py
git commit -m "test(hpc): execute nonlinear-only chunk-size sweep"
```

### Task 4: Spatial registration metrics and deformation validity

**Files:**
- Create: `tests/hpc/fixed_affine_chunk_analysis.py`
- Create: `tests/unit/test_fixed_affine_chunk_analysis.py`
- Create: `tests/integration/test_fixed_affine_chunk_analysis.py`
- Modify: `tests/hpc/AGENTS.md`

**Interfaces:**
- Produces: `gradient_ncc_search`, `blockwise_residuals`,
  `classify_seam_proximity`, `jacobian_summary`, and
  `analyze_characterization`.
- Consumes: validated registered TIFFs, fixed TIFF, full warps, canonical chunk
  grids, and condition summaries.

- [ ] **Step 1: Write failing synthetic metric tests**

Create synthetic fixed images and shifted/brightness-scaled moving images. Assert
`gradient_ncc_search` returns exact `(dy, dx)`, Euclidean magnitude, current NCC,
best NCC, runner-up NCC outside Chebyshev distance 2, prominence, and boundary flag.
Cover zero shift, known positive/negative shifts, uniform/undefined regions,
search-boundary optima, and invalid ROI bounds.

- [ ] **Step 2: Define exact algorithms**

Use Z maximum-intensity projections, float64 Sobel gradient magnitude, and normalized
cross-correlation:

```python
def ncc(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    ac = a - a.mean()
    bc = b - b.mean()
    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)
    return float(np.dot(ac.ravel(), bc.ravel()) / denominator)
```

Evaluate every integer shift in `[-30, 30]²`. The reported correction is the shift
applied to the moving ROI to match the fixed ROI. Flag an optimum on any search
boundary as a lower bound. Define prominence as best NCC minus the best score outside
a Chebyshev radius of 2 around the optimum.

- [ ] **Step 3: Implement blockwise and seam analysis**

Use 128×128 QC windows on a 64-pixel stride where a full ±30 search context exists.
For each condition, classify a block center as seam-adjacent when it lies within 16
pixels of an internal Y or X core boundary from `compute_chunk_grid`; otherwise mark
it interior. Record fixed MIP intensity quartile, gradient energy, residual,
confidence, boundary status, and seam class.

- [ ] **Step 4: Implement Jacobian validation**

Use ANTs' geometric Jacobian determinant on the exact fixed image and full warp.
Record minimum, 1st/5th/50th/95th/99th percentiles, maximum, non-finite count, and
count/fraction `<= 0`. Tests create identity, uniform expansion, and a synthetic
folding field and assert the expected sign behavior.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
uv run pytest -m unit tests/unit/test_fixed_affine_chunk_analysis.py -v
uv run pytest -m integration \
  tests/integration/test_fixed_affine_chunk_analysis.py -v
```

Commit:

```bash
git add tests/hpc/AGENTS.md tests/hpc/fixed_affine_chunk_analysis.py \
  tests/unit/test_fixed_affine_chunk_analysis.py \
  tests/integration/test_fixed_affine_chunk_analysis.py
git commit -m "test(hpc): quantify regional and deformation error"
```

### Task 5: Deterministic result tables and publication figures

**Files:**
- Create: `tests/hpc/fixed_affine_chunk_figures.py`
- Create: `tests/unit/test_fixed_affine_chunk_figures.py`
- Create: `docs/images/fixed-affine-chunk-size/.gitkeep`

**Interfaces:**
- Produces: `render_characterization_figures(analysis_json, output_dir)` and compact
  CSV files consumed by the documentation.
- Consumes: Task 4 machine-readable results only.

- [ ] **Step 1: Write failing deterministic-render tests**

Build a tiny synthetic analysis payload and assert exact figure names, dimensions,
axis labels, condition ordering, round ordering, colorblind-safe palette, and stable
CSV bytes. Render twice and compare normalized SVG plus CSV bytes.

- [ ] **Step 2: Implement the required figures**

Generate:

- `regional-residuals.svg` and `.png`: bright/dim residual versus XY chunk size,
  faceted by round;
- `regional-overlays.png`: fixed green versus registered magenta gradient overlays
  for both ROIs, every condition, and every round;
- `runtime-throughput.svg` and `.png`: nonlinear wall time, task time, and chunks per
  hour;
- `seam-interior.svg` and `.png`: residual/confidence by seam class and signal
  quartile;
- `jacobian-validity.svg` and `.png`: determinant distributions and folding counts.

Use fixed condition ordering `1125, 512, 256, 128`, explicit pixel/time units, and
no titles that imply universal optimality.

- [ ] **Step 3: Run tests and commit**

Run:

```bash
uv run pytest -m unit tests/unit/test_fixed_affine_chunk_figures.py -v
```

Commit:

```bash
git add tests/hpc/fixed_affine_chunk_figures.py \
  tests/unit/test_fixed_affine_chunk_figures.py \
  docs/images/fixed-affine-chunk-size/.gitkeep
git commit -m "docs: add deterministic chunk characterization figures"
```

### Task 6: Operator-owned configuration, dry validation, and SLURM launch

**Files:**
- External only: caller-owned characterization YAML, fine execution YAML, evidence
  directory, condition roots, and coordinator log.

**Interfaces:**
- Consumes: Tasks 1--5 and the operator-selected radius-4 source.
- Produces: four complete condition trees and machine-readable evidence on
  `/archive`.

- [ ] **Step 1: Create external configuration**

Use the approved source path from the user request only in the external YAML. Create
a fresh experiment root on `/archive`, a fresh scratch root on `/project`, and a
fine-grained four-worker profile. Do not place either file in git.

- [ ] **Step 2: Validate without scheduler mutation**

Run the strict loader and source validator in a no-submit mode. Confirm exact source
hashes, geometry `(51, 2240, 2223)`, rounds 2--4, CC radius 4, 30/30 threads, four
conditions, expected task counts, and disjoint roots.

- [ ] **Step 3: Launch the explicitly selected HPC test**

Run with the repository's exact explicit HPC options:

```bash
uv run pytest -m hpc --run-hpc \
  tests/hpc/test_fixed_affine_chunk_characterization.py -v -s \
  --cyexm-dataset "$DATASET_YAML" \
  --cyexm-execution "$FINE_EXECUTION_YAML" \
  --cyexm-evidence-dir "$EVIDENCE_DIR" \
  --cyexm-characterization "$CHARACTERIZATION_YAML"
```

Redirect the coordinator transcript to the external experiment root. Record the
coordinator PID and exact command without committing either.

- [ ] **Step 4: Monitor at low frequency**

Inspect `squeue` and condition summaries approximately every five minutes. Do not
poll active scientific futures aggressively. Preserve final SLURM logs and verify
the current-user exact owned job names are empty after every condition.

### Task 7: Analyze completed outputs and write the scientific report

**Files:**
- Create: `docs/fixed-affine-chunk-size-characterization.md`
- Modify: `docs/brightness-dependence-investigation.md`
- Modify: `docs/cc-syn-radius-characterization.md`
- Modify: `docs/fine-grained-execution.md`
- Modify: `docs/index.md`
- Modify: `docs/myst.yml`
- Modify: `docs/AGENTS.md`
- Populate: `docs/images/fixed-affine-chunk-size/`

**Interfaces:**
- Consumes: completed Task 6 evidence and Task 4/5 analysis code.
- Produces: final compact tables, figures, interpretation, and published page.

- [ ] **Step 1: Run analysis and independently validate results**

Generate external full analysis JSON/CSV, then verify source hashes, output geometry,
finite metrics, no affine-estimation events, expected task counts, condition cleanup,
and all ROI/search flags. Keep incomplete conditions out of comparative claims.

- [ ] **Step 2: Generate and inspect figures**

Render all required PNG/SVG assets. Inspect PNGs visually for cropping, legibility,
color behavior, overlay orientation, and correct ROI/condition labels. Cross-check
each plotted point against compact CSV.

- [ ] **Step 3: Write the report**

The Markdown page must contain:

- hypothesis and fixed-affine design;
- exact chunk sizes, overlap, radius, resources, rounds, and ROIs;
- evidence that TRSAA did not run;
- regional residual/current/best NCC/confidence tables;
- runtime/task/retry table;
- seam/interior and Jacobian results;
- figures with descriptive captions;
- limitations, including MIP-based local measurements and one dataset;
- a dataset-specific recommendation with no production-default change.

- [ ] **Step 4: Update technical documentation**

Link the report from the brightness investigation, radius characterization,
fine-grained execution page, documentation landing page, and `myst.yml`. Update
documentation guidance to require fixed-affine characterization claims to identify
the source state, chunk geometry, overlap, metric/radius, regional confidence, seam
behavior, and deformation validity.

- [ ] **Step 5: Add documentation policy tests**

Extend `tests/unit/test_phase2_documentation.py` to require the new page, navigation
entry, figure references, fixed-affine statement, four chunk sizes, 0.15 overlap,
radius 4, both ROIs, and the dataset-specific limitation.

- [ ] **Step 6: Run documentation tests and build HTML**

Run:

```bash
uv run pytest -m unit tests/unit/test_phase2_documentation.py -v
cd docs && BASE_URL=/Cy-ExM uv run --group docs jupyter book build --html
```

Inspect the generated page and verify every image and internal link resolves.

- [ ] **Step 7: Commit report and assets**

Commit:

```bash
git add docs/AGENTS.md docs/myst.yml docs/index.md \
  docs/brightness-dependence-investigation.md \
  docs/cc-syn-radius-characterization.md docs/fine-grained-execution.md \
  docs/fixed-affine-chunk-size-characterization.md \
  docs/images/fixed-affine-chunk-size \
  tests/unit/test_phase2_documentation.py
git commit -m "docs: publish fixed-affine chunk characterization"
```

### Task 8: Repository-wide verification and integration

**Files:**
- Modify only files required by verification findings.

**Interfaces:**
- Consumes: all implementation, scientific evidence, report, and figures.
- Produces: a clean verified feature branch ready for local main integration.

- [ ] **Step 1: Run focused suites**

```bash
uv run pytest -m unit \
  tests/unit/test_fixed_affine_chunk_characterization.py \
  tests/unit/test_fixed_affine_chunk_analysis.py \
  tests/unit/test_fixed_affine_chunk_figures.py \
  tests/unit/test_phase2_documentation.py -v
uv run pytest -m integration \
  tests/integration/test_fixed_affine_chunk_characterization.py \
  tests/integration/test_fixed_affine_chunk_analysis.py -v
```

- [ ] **Step 2: Run portable repository gates**

```bash
uv lock --check
uv run pytest -m "unit and not hpc" tests/unit -v
uv run pytest -m "integration and not hpc" tests/integration -v
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pre-commit run --all-files
```

- [ ] **Step 3: Verify scientific and repository hygiene**

Require:

- no source or generated large data in git;
- no site-specific external YAML or paths in maintained docs;
- no changed production default or public schema;
- no unowned SLURM jobs;
- a clean diff except intentional files;
- every committed figure backed by compact exact results;
- source affine and reference identities unchanged.

- [ ] **Step 4: Record final status**

Summarize commits, external evidence root, condition outcomes, main scientific
result, HTML build result, and any qualified/boundary-limited measurements. Merge
and push only when explicitly authorized by the user.
