# Fixed-Affine Chunk-Size Characterization Design

## Status

Approved experimental design. This specification defines a dataset-specific
scientific characterization; it does not change Cy-ExM production defaults.

## Context

The spheroid CC-radius sweep found that scalar CC SyN radius 4 gave the best
bright-versus-dim tradeoff among the tested radii. The comparison was confounded by
independent stochastic TRSAA estimates, however, and radius 4 retained an
approximately 8--14-pixel corrective displacement in the previously identified dim
region.

The next experiment tests whether smaller nonlinear chunks reduce the influence of
bright, information-rich structures on dim regions. The operator-selected completed
radius-4 workflow is the source of the fixed affine state. Its canonical location
belongs only in the external experiment configuration; its source data and
completed artifacts are read-only inputs to this experiment.

The fixed cropped reference geometry is NumPy ZYX `(51, 2240, 2223)`. Cy-ExM public
spacing remains XYZ micrometers `(0.324, 0.324, 0.900)`.

## Goals

1. Hold each moving round's accepted radius-4 affine transform exactly fixed.
2. Compare CC SyN radius 4 at four ZYX chunk sizes:
   `(512, 128, 128)`, `(512, 256, 256)`, `(512, 512, 512)`, and
   `(512, 1125, 1125)`.
3. Retain `overlap_fraction: 0.15` and every other nonlinear scientific parameter.
4. Avoid TRSAA estimation while allowing inexpensive application of the accepted
   affine transform to the moving registration channel.
5. Quantify residual registration in the previously evaluated bright and dim
   regions for rounds 2--4.
6. Measure nonlinear runtime, task behavior, seam behavior, and deformation
   validity.
7. Publish a reproducible Markdown report with publication-quality figures in the
   Cy-ExM technical HTML documentation.

## Non-goals

- Do not change the schema, public workflow API, cache-import policy, or production
  chunk-size default.
- Do not claim that radius 4 or the winning chunk size is universally optimal.
- Do not rerun affine estimation, alter the accepted affine files, or write into the
  completed radius-4 output tree.
- Do not transform dependent biological channels; the experiment needs only the
  registration channels and deformation fields.
- Do not include the rejected `(512, 64, 64)` configuration, which would create
  5,043 nonlinear tasks across the three moving rounds.
- Do not infer scientific success from a global similarity metric alone.

## Alternatives considered

### Selected: controlled nonlinear-only characterization harness

Build a bounded experiment harness around existing Cy-ExM registration primitives.
It validates and reads the accepted affine state, applies that transform to produce
one fixed-grid moving registration image per round, and submits only nonlinear
chunk estimation, assembly, and validation. Every chunk-size condition receives the
same immutable prepared inputs.

This approach isolates chunk geometry without adding a production cache-import
surface. Rerunning the original chunk size through the same harness makes runtime
comparisons internally consistent.

### Rejected for this experiment: production affine-state import

A general import feature would require schema, planner, cache identity, provenance,
validation, migration, and API work. That may be valuable later, but it expands the
scientific characterization into a production feature and increases the risk of
incorrect artifact authorization.

### Rejected: retain the existing large-chunk result without rerunning it

The existing registered images remain useful as an independent cross-check, but
their coarse three-worker execution is not directly comparable with new
fine-grained runs. It therefore cannot serve as the primary runtime baseline.

## Scientific invariants

- Fixed and moving roles never change.
- All four conditions use the exact accepted affine transform content for the
  corresponding moving round.
- The fixed reference, affine-aligned moving scalar image, masks, local nonlinear
  inputs, full warp, and registered scalar output share the fixed-reference
  geometry.
- Affine transform application precedes nonlinear registration.
- Scalar interpolation is linear and mask interpolation is nearest-neighbor.
- Nonlinear registration is `SyNOnly`, accuracy `high`, metric `CC`, scalar voxel
  radius 4, with the existing greater-than-zero masks.
- Chunk size is ZYX and spacing is public XYZ micrometers.
- `overlap_fraction` remains exactly `0.15`.
- Every worker uses exactly 30 ITK threads and 30 threads for each BLAS-family
  variable, matching the accepted source run.
- Each condition is derived from immutable source identities and has a unique output
  root, scratch root, scheduler job name, and run identity.
- No result is accepted when the source affine identity changes during the
  experiment.

## Experiment architecture

### Immutable source-state validator

The harness reads the radius-4 scientific plan and completion records for:

- the cropped reference registration TIFF and its completion record;
- each moving crop record;
- each accepted affine transform and its completion record;
- each linearly registered metric record used as a provenance cross-check.

It validates artifact identity, request identity, source image identity, round
number, fixed geometry, native threads, and affine lineage. The validator records a
machine-readable source manifest containing canonical paths and SHA-256 hashes.
Validation occurs before submission and again before accepting results.

### Prepared-input stage

For each moving round, the harness reconstructs the same fixed and moving states
used immediately before nonlinear registration:

1. read the cropped fixed reference;
2. crop the moving input with the accepted moving bounds;
3. apply the accepted affine transform onto the fixed grid;
4. construct the unchanged greater-than-zero fixed and moving masks;
5. persist float32 prepared arrays and exact geometry metadata in the experiment
   evidence root.

This stage applies the existing affine but never estimates a new affine. Prepared
inputs are generated once per round, hashed, and reused byte-for-byte by every
chunk-size condition.

### Nonlinear condition runner

Each condition uses the current fine-grained SLURM execution boundary with four
exclusive 512-GB workers, one process and one task slot per node, 30 cores per
worker, 500 GB requested memory per worker, and 30 ITK and BLAS-family threads.
Partition, interface, walltime, and cluster-capacity assumptions remain external
configuration facts and must be validated before submission. The coordinator
creates the canonical chunk grid, submits independent local CC SyN tasks,
deterministically stitches scalar cores and warp cores, writes the full warp and
registered scalar TIFF, and records stage timings and task attempts.

Expected chunk counts per round at fixed geometry are:

| Chunk size ZYX | Chunks per round | Chunks across rounds 2--4 |
| --- | ---: | ---: |
| `(512, 128, 128)` | 441 | 1,323 |
| `(512, 256, 256)` | 121 | 363 |
| `(512, 512, 512)` | 36 | 108 |
| `(512, 1125, 1125)` | 9 | 27 |

Conditions run with equal resources and independently named outputs. They may be
submitted sequentially to limit cluster occupancy; scheduler waiting time is
reported separately from computation and excluded from nonlinear wall time.

### Analysis stage

The analysis consumes only completed, validated condition outputs. For each round
and condition it computes:

- gradient-NCC corrective displacement within a plus-or-minus 30-pixel XY search on
  the Z maximum-intensity projection;
- gradient-NCC at the current position and at the best corrective shift;
- peak prominence or an equivalent best-versus-runner-up confidence statistic;
- the known dim ROI `y=1600:1800`, `x=300:400`;
- the known bright ROI `y=1227:1427`, `x=840:1040`;
- whole-volume Correlation and Mattes mutual information as secondary metrics;
- seam-adjacent versus chunk-interior residual summaries;
- finite-displacement checks, Jacobian determinant distribution, and non-positive
  Jacobian count;
- nonlinear wall time, worker task time, task count, retry/failure count, and
  throughput.

The coordinate convention for reported corrective shifts is `(dy, dx)` in
projection pixels. Residual magnitude is Euclidean pixels. A boundary optimum is
reported as a lower bound and never as a resolved displacement.

## Figures and report

The report will include, at minimum:

1. a line or point plot of dim and bright residual displacement versus XY chunk
   size, separated by moving round;
2. fixed-versus-registered gradient overlay panels for both known ROIs across all
   four chunk sizes;
3. runtime and task-throughput plots;
4. seam-versus-interior residual or confidence summaries;
5. Jacobian distributions and folding counts;
6. a compact table containing the exact numeric measurements behind every figure.

Figures are generated deterministically from a committed analysis script, use
colorblind-safe colors, label pixels and wall-clock units explicitly, and are saved
as publication-quality PNG plus a vector format when supported. Generated figures
that are required by the published page are committed under a focused documentation
asset directory; large arrays, logs, and machine evidence remain on `/archive`.

The new Markdown page is added to `docs/myst.yml`, linked from `docs/index.md`,
`docs/brightness-dependence-investigation.md`, and the relevant fine-grained
technical page. The built HTML must resolve every figure and internal link.

## Persistent-state ownership

The completed radius-4 tree is read-only. New configs, prepared inputs, outputs,
logs, timing evidence, metrics, and analysis tables live beneath one new
caller-owned `/archive` experiment root. Worker scratch remains on `/project`.

The repository contains only:

- the design and implementation plan;
- reusable experiment and analysis code;
- the final Markdown interpretation;
- compact tabular results and documentation figures;
- navigation and technical-documentation updates.

It never contains source images, registered volumes, warps, SLURM logs, private
site paths in portable examples, or scheduler credentials.

## Failure handling

- Fail before scheduler submission if any required source artifact, completion
  record, identity, geometry, thread policy, or affine lineage is invalid.
- Fail if a prepared input differs across conditions.
- Preserve successful chunk outputs for restart; retry only according to the
  explicitly recorded execution profile.
- Treat missing, boundary-limited, non-finite, folded, or low-confidence local
  measurements as failures or qualified observations rather than favorable values.
- Keep incomplete conditions out of comparative plots and mark their report status
  explicitly.
- Close the Dask client before its cluster, verify owned-job cleanup, and never
  cancel unrelated jobs.

## Testing and validation

Portable tests use small synthetic volumes and fake scheduling to prove:

- accepted affine state is reused without invoking affine estimation;
- every condition receives byte-identical prepared inputs;
- chunk size is the only scientific condition difference;
- ROI coordinates, corrective-shift signs, boundary detection, and confidence
  calculations are correct;
- seam classification is correct for overlapping grids;
- Jacobian and folding calculations detect known synthetic fields;
- plots and tables are deterministically regenerated;
- documentation navigation and the HTML build include the new page and figures.

The real experiment additionally records exact git commit, package versions,
configs, source identities, scheduler resources, task attempts, wall times, and
cleanup evidence.

## Acceptance criteria

The characterization is complete only when:

1. all four nonlinear-only conditions complete for rounds 2--4 from one validated
   fixed affine state;
2. no condition invokes affine estimation;
3. prepared inputs are identical across conditions;
4. every output preserves exact fixed-reference geometry and finite values;
5. bright and dim residuals, confidence, runtime, seams, and Jacobians are reported
   for every completed condition;
6. any search-boundary optimum or folding is clearly flagged;
7. source and result evidence are sufficient to reproduce the analysis;
8. the Markdown report distinguishes dataset-specific evidence from production
   policy;
9. the published HTML documentation builds successfully with all figures and links;
10. no production default is changed without a separate scientific decision.
