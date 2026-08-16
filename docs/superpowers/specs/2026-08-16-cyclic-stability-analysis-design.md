# Cyclic Nuclear-Envelope and Organelle Stability Analysis Design

## Status

Approved design for a publication-specific analysis. This specification is the only
artifact from this work that belongs in the Cy-ExM repository. Analysis code,
configuration, intermediate data, results, and manuscript figures remain in the
caller-owned external resubmission workspace.

## Context

The manuscript revision requires quantitative evidence about structural stability
and repeated-labeling performance across five Cy-ExM imaging rounds. Registration
accuracy alone cannot answer whether repeated expansion, shrinking, staining,
stripping, and re-imaging introduce cumulative deformation. Conversely, morphology
measured only after nonlinear registration cannot establish an absence of physical
deformation because the warp may correct the change being measured.

The available 60x U2OS experiment contains five Golgi cells and five Lamin cells,
each imaged for five rounds with repeated DAPI, tubulin, and target labeling. Five
rounds of LAMP1 source ND2 data are also available, but their cell correspondence,
channel export, and registration are not yet complete. Tubulin supplies the
registration channel. Acquisition round 2 is the fixed reference because it has the
strongest tubulin signal. Analyses and figures must use acquisition-round numbering,
not assume that a workflow round label denotes the acquisition round.

An external provenance audit links every existing Golgi and Lamin TIFF triplet to a
unique ND2 source by exact pixel equality in the target, tubulin, and DAPI channels at
the first, middle, and last exported Z planes. The audit also records which exports
omit the first ND2 plane. Numeric ND2 suffixes do not identify a biological cell
across rounds.

All audited acquisitions used a Plan Apo lambda 60x Oil objective with NA 1.4 and
XYZ sampling of approximately `(0.108333, 0.108333, 0.300)` micrometers in the
expanded sample. DAPI and tubulin settings are consistent except for one LAMP1
round-2 tubulin acquisition at 300 ms rather than 200 ms. Golgi and Lamin target
exposures changed from 150 ms in acquisition round 1 to 200 ms in rounds 2--5.
Golgi used 50 percent 640-nm laser power, and Lamin used 40 percent.

Whole-cell morphology is outside this analysis. It will be addressed by a separate
experiment with an appropriate cell-wide marker.

## Goals

1. Quantify global and local deformation required to align each acquisition round
   to acquisition round 2.
2. Quantify residual nuclear-envelope, Golgi, and LAMP1 morphology after alignment
   without relying on one independently chosen segmentation per round.
3. Quantify repeated-labeling signal reproducibility while separating matched
   acquisition settings from the round-1 exposure confounder.
4. Report all cell-level trajectories and avoid voxel-level or object-level
   pseudoreplication.
5. Compare distance-based changes with clearly labeled theoretical lateral and
   axial diffraction scales in expanded-image and nominal native-space units.
6. Produce restart-safe measurements, quality-control evidence, publication-quality
   editable figures, and a Markdown methods/results report.
7. Keep scientific configuration separate from execution resources and support
   sequential, bounded local-process, and optional SLURM-array execution.

## Non-goals

- Do not change Cy-ExM registration behavior, schemas, production defaults, or
  public APIs.
- Do not add the publication-specific analysis runtime to the Cy-ExM package.
- Do not infer whole-cell volume or geometry from DAPI, tubulin, Golgi, Lamin, or
  LAMP1.
- Do not call a nonsignificant round effect proof of stability.
- Do not interpret relabeling intensity as a pure biochemical measurement of
  epitope survival; it also reflects staining and imaging reproducibility.
- Do not interpret morphology after nonlinear registration as proof that no
  deformation occurred before registration.
- Do not infer LAMP1 cell identity from numeric file suffixes.
- Do not select segmentation settings after inspecting which choice produces the
  most favorable biological conclusion.

## Alternatives considered

### Selected: two-layer deformation and residual-morphology analysis

The selected design first quantifies the affine and nonlinear deformation needed to
align tubulin and then measures target morphology on the common registered grid.
Continuous-image measurements are primary, while masks are evaluated across
prespecified threshold and scale ensembles. This design separates deformation,
computational recoverability, residual morphology, and signal reproducibility.

### Rejected as the sole analysis: independent segmentation per round

Independent masks followed by Jaccard, volume, compactness, and component counts are
easy to explain and match language in the proposed revision. Thin Lamin shells and
fragmented Golgi structures make these measurements highly sensitive to threshold,
signal strength, and interpolation, however. They also cannot reveal deformation
that nonlinear registration corrected. Mask metrics remain secondary evidence.

### Rejected as the sole analysis: segmentation-free image similarity

Gradient correlation, structural similarity, and local residual alignment avoid
hard segmentation failures. Used alone, they provide limited biological
interpretability and can resemble another registration-quality analysis. They
remain primary or supporting measurements only when paired with target-specific
geometry.

### Rejected: immediate integration into Cy-ExM

The marker models, nominal expansion factor, exposure correction, reviewer-facing
outputs, and dataset correspondence are publication-specific. Integrating them now
would expand the registration package into a target-morphology framework before the
methods have been validated across markers and datasets. A generic component may be
considered later only after independent reuse is demonstrated.

## Scientific evidence model

The analysis answers three separate questions.

### Deformation required for alignment

Affine scale, anisotropy, and shear describe global geometric change. Rotation and
translation are reported but treated primarily as sample repositioning. The
nonlinear displacement field, spatial displacement gradients, and geometric
Jacobian describe local correction required to align tubulin. Chunk-seam and
chunk-interior measurements remain separate because the registered volumes were
produced by serial chunked nonlinear registration.

These transform-derived measurements do not prove a biological mechanism. They
quantify the deformation required by the selected registration model and include
both specimen change and any model or registration error.

### Residual morphology after alignment

Registered DAPI, Lamin, Golgi, and LAMP1 images share the acquisition-round-2 fixed
grid. Measurements on these images test whether target morphology remains spatially
recoverable after tubulin-derived alignment and whether residual target differences
increase with round.

### Repeated-labeling signal reproducibility

Primary signal measurements use unregistered exported TIFFs to avoid interpolation
effects. Rounds 2--5 form the matched-acquisition comparison. Round 1 appears as a
separate exposure-normalized sensitivity analysis. Results are described as
combined epitope accessibility, staining, and imaging reproducibility.

## Coordinate and resolution policy

Public physical spacing is XYZ micrometers. NumPy volumes are ZYX. Every output must
state its axis convention. Image-space measurements use the expanded-sample
spacing. Nominal native-space values divide image-space distances by `4.5` and are
always labeled estimates because no per-gel expansion measurement is available.

No empirical point-spread-function or bead calibration accompanies these data. The
analysis therefore uses theoretical diffraction scales only as declared reference
bounds, not as measured system resolution. For excitation wavelength `lambda`,
NA `1.4`, and immersion refractive index `1.515`, the reference formulas are:

- lateral Rayleigh scale: `0.61 * lambda / NA`;
- axial scale: `2 * n * lambda / NA^2`.

For the 640-nm target channel, these yield approximately 279 nm lateral and 989 nm
axial in the expanded sample, or 62 nm lateral and 220 nm axial in nominal native
space. DAPI and tubulin quality-control measurements use their documented channel
wavelengths rather than the 640-nm bound. Lateral and axial components remain
separate; a single isotropic Euclidean tolerance is insufficient.

## Provenance manifest

The external analysis owns a strict machine-readable manifest. Each cell-round row
contains:

- cohort, target, and assigned biological-cell identifier;
- acquisition round derived from source location and internal ND2 date;
- exact ND2 source and exported target, tubulin, and DAPI TIFFs;
- exported Z start offset;
- channel exposure, laser power, objective, and voxel spacing;
- nominal expansion factor and resolution policy;
- Cy-ExM dataset ID, workflow round, artifact root, and fixed-reference status; and
- current identities of required source and registration artifacts.

The human-readable correspondence record remains beside the publication data at
`nd2-tiff-file-correspondence.md`. The machine-readable manifest must agree with
it. Acquisition-round numbering is the only numbering used in analysis tables and
figures.

LAMP1 remains unavailable to production analysis until image content establishes
the five cross-round cell identities and the mapping is recorded before export and
registration.

## Quality gates

Quality gates run before target outcomes are inspected.

### Identity and acquisition gate

- Recheck exact source-to-export correspondence.
- Confirm cross-round biological identity from repeated DAPI and tubulin morphology.
- Reject unexpected axes, dtype, channel order, geometry, or acquisition settings.
- Keep the round-1 target exposure change explicit.

### Artifact and geometry gate

- Validate every required Cy-ExM completion record and provenance chain.
- Require fixed-reference geometry for registered targets, registration scalar
  outputs, and full warps.
- Require finite transforms, metrics, and images.
- Do not reuse an artifact whose completion record is missing or stale.

### Scientific registration gate

- Inspect pre-, affine-, and nonlinear tubulin metrics.
- Decompose affine matrices and check determinant, singular values, scale,
  anisotropy, shear, and translation.
- Measure displacement validity, Jacobians, non-positive Jacobian counts, and
  chunk-seam behavior.
- Generate blinded DAPI/tubulin overlays without consulting target-channel outcomes.

No universal correlation cutoff is introduced without characterization. Conflicting
evidence produces a flagged cell-round. Prespecified primary analyses exclude
flagged registrations, list the reasons, and retain all-data sensitivity analyses.
The already observed poor Lamin round-1 registrations must not silently influence a
biological conclusion.

## Target-specific measurements

All morphology endpoints include acquisition-round-2-to-round comparisons and
adjacent-round comparisons. Continuous-image measurements are primary. Mask-derived
conclusions must remain stable across a prespecified physical-scale and
signal-threshold ensemble.

### Nuclear envelope

A consensus DAPI-derived nuclear surface defines a common analysis region. Lamin
intensity is sampled along surface normals in each round to measure:

- local envelope displacement;
- Lamin ridge width;
- surface coverage and continuity; and
- background-corrected ridge intensity.

The normal search window is defined in physical units and is wide enough to detect
loss or displacement without forcing a peak. A missing supported peak remains
missing rather than receiving a favorable zero displacement.

Secondary independently generated nuclear masks provide volume, surface
area-to-volume ratio, sphericity, curvature distributions, resolution-tolerant
surface Dice, boundary-distance distributions, and Jaccard.

### Golgi

Primary Golgi measurements are:

- multiscale gradient correlation within a fixed perinuclear region;
- resolution-tolerant surface Dice and boundary-distance distributions across the
  segmentation ensemble; and
- the distribution of Golgi signal distance from the nuclear surface.

Secondary measurements include target-positive volume, connected-component count,
fragmentation, compactness, and surface area-to-volume ratio. The report includes
metric-versus-threshold curves and a prespecified robust summary rather than one
selected threshold.

### LAMP1

LAMP1 objects are detected with scale-normalized 3D spot detection across a
physical-size and signal-threshold ensemble. Measurements include:

- object count, volume, equivalent diameter, and sphericity;
- nearest-neighbor and nucleus-relative spatial distributions;
- cross-round matched-object fraction;
- lateral and axial displacement of matched objects in theoretical-resolution
  units; and
- fractions of lost and newly detected objects.

Object matching uses a resolution-aware maximum distance and permits unmatched
objects rather than forcing assignments.

### Signal retention

The unregistered exported target TIFFs supply:

- background-corrected integrated intensity per unit exposure;
- signal-to-background ratio;
- signal-positive volume or surface coverage; and
- within-structure intensity percentiles.

Rounds 2--5 are primary. Round 1 remains a separate exposure-normalized sensitivity
analysis. Target-to-DAPI ratios may appear as secondary acquisition controls but do
not replace absolute measurements. No per-round normalization may erase a genuine
round-dependent signal change.

Jaccard is included because it is promised in the revision plan, but it remains
secondary to continuous-image measurements, boundary distances, and
resolution-tolerant Dice.

## Statistical design

The biological replicate is the cell. Voxels, surface vertices, and detected
objects are summarized within a cell before group analysis. Every endpoint reports:

- the complete five-round trajectory for each cell;
- change relative to acquisition round 2;
- adjacent-round change;
- a cell-level median and range or interval; and
- a round-dependent cell-level slope as a descriptive effect size.

Five cells per marker do not support a strong claim from a complex mixed-effects
model or an unqualified formal equivalence test. The report prioritizes individual
trajectories, effect sizes, uncertainty, and sensitivity analyses. A failure to
reject a round effect is not stability evidence.

Primary endpoints are limited to:

- deformation: 95th-percentile local displacement and fraction exceeding one
  theoretical resolution unit;
- Lamin: 95th-percentile envelope displacement and loss of surface continuity;
- Golgi: multiscale gradient similarity and resolution-tolerant surface Dice;
- LAMP1: matched-object displacement and relative object-count change; and
- signal retention: background-corrected signal-to-background trajectory for
  rounds 2--5.

Volume, curvature, compactness, Jaccard, fragmentation, and spatial-distribution
features are secondary. Sensitivity analyses include threshold and scale ensembles,
all registrations versus QC-accepted registrations, and round-1 intensity with and
without exposure normalization.

## External analysis architecture

The implementation root is:

`/archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/cyclic-stability-analysis/`

It contains:

- `README.md` with scientific boundaries and reproducible commands;
- `config/` with the scientific manifest and separate local and SLURM execution
  profiles;
- `scripts/` with PEP 723-compliant validation, analysis, and figure entry points;
- `tests/` with synthetic and cropped-data regression tests; and
- `results/<run-id>/` with an immutable configuration snapshot, task records,
  measurement tables, QC, figures, and report.

Source ND2 files, exported TIFFs, and Cy-ExM artifacts are read-only. Temporary or
high-I/O intermediates may use a dedicated `/project` scratch directory. Python
environments and dependency caches remain on `/project`.

### Data flow

1. Validate the manifest, source correspondence, acquisition metadata, and Cy-ExM
   artifacts.
2. Run registration and identity QC without target-outcome access.
3. Construct one analysis task per marker-cell across all five rounds.
4. Compute deformation, continuous-image, segmentation-ensemble, object, and signal
   summaries.
5. Atomically commit compact task results and provenance.
6. Aggregate only completed tasks into tidy tables.
7. Render figures and the Markdown report from compact tables and selected QC
   images, without reprocessing full volumes.

## Parallel execution

One marker-cell across five rounds is the independent process-queue unit. The
initial queue has ten tasks, with five LAMP1 tasks added after their input gate
passes. This boundary keeps the consensus surface, threshold ensemble, pairwise
comparisons, and cached features in one process.

Supported modes are:

- sequential execution for validation and debugging;
- a bounded local process queue for normal production; and
- an optional SLURM array when a representative pilot demonstrates a memory or
  runtime need.

Each process uses one native thread unless the execution profile explicitly states
otherwise. TIFFs are memory-mapped or processed in spatial blocks. Worker count is
selected only after recording peak memory and I/O for one representative task.
Concurrent `/archive` reads remain bounded. There is no nested multiprocessing.

Each task writes compact measurements, QC summaries, diagnostic thumbnails, input
and configuration identities, software versions, status, and an atomic completion
marker. Reuse requires matching inputs and configuration. A failed task does not
invalidate completed cells, and a restart schedules only missing or stale tasks.

## Figures and report

The minimum figure set contains:

1. representative 60x views and enlarged nuclear-envelope, Golgi, and LAMP1
   regions;
2. affine, nonlinear-displacement, Jacobian, seam, and registration-QC summaries;
3. all-cell morphology trajectories with theoretical resolution bounds where
   applicable;
4. signal-retention trajectories separating rounds 2--5 from the round-1
   sensitivity analysis;
5. threshold and scale robustness plots; and
6. a transparent accounting of flagged or excluded cell-rounds.

Scale bars and distances show expanded-image units and nominal native-space units.
The latter explicitly state the 4.5x assumption. Statistical plots display all five
cells and use consistent cell identities across panels. Bar-only summaries are not
sufficient.

The centralized Matplotlib style includes:

```python
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
```

The figure runtime uses an explicitly available TrueType font such as DejaVu Sans,
a colorblind-safe palette, consistent physical dimensions at final publication
size, and vector PDF/SVG output. Microscopy panels remain raster data with vector
text, scale bars, and annotations. High-resolution PNG previews and editable source
figures accompany the vector output.

The Markdown methods/results report separates direct measurements from
interpretation and supplies wording suitable for the rebuttal without overstating
epitope chemistry, physical stability, or measured resolution.

## Failure handling

- Fail validation before expensive work when source identity, axes, geometry,
  acquisition metadata, completion records, or round mapping are invalid.
- Never infer a missing LAMP1 mapping or silently substitute a numeric suffix.
- Preserve successful marker-cell outputs and retry only failed or stale tasks.
- Treat non-finite transforms, folding, seam-dominated results, boundary-limited
  searches, missing target support, and unresolved object matches as qualified or
  failed observations rather than favorable measurements.
- Keep incomplete marker-cells out of primary aggregates while retaining their
  status and diagnostics.
- Record every exclusion before aggregation and render an all-data sensitivity
  analysis when values exist.
- Never overwrite source ND2, exported TIFF, or Cy-ExM output trees.

## Testing and validation

Synthetic 3D tests cover:

- known affine scale, anisotropy, shear, and nonlinear displacement;
- identity, smooth, folded, and seam-discontinuous warp fields;
- known nuclear-shell displacement, width, continuity loss, and missing signal;
- Golgi-like structures under intensity and segmentation-threshold perturbations;
- LAMP1-like object detection, disappearance, appearance, and matching;
- known intensity loss and background changes;
- XYZ/ZYX handling, physical spacing, 4.5x conversion, and anisotropic theoretical
  resolution bounds;
- equality of sequential and process-queue results;
- atomic completion, interrupted-task recovery, and stale-input rejection; and
- deterministic table and figure generation, including editable font settings.

A small cropped real-data smoke test validates TIFF/ND2 reading, artifact discovery,
memory behavior, and final figure layout before full-resolution production. The
production pilot records one marker-cell's peak RSS, elapsed time, bytes read, and
scratch use before concurrency is selected.

## Reuse analysis

Cy-ExM already owns registration manifests, completion records, workflow status,
transforms, and Correlation/Mattes metrics. The external analysis must consume those
artifacts rather than recreate registration lifecycle, cache authority, or workflow
status. Existing repository characterization code demonstrates Jacobian, seam, and
regional-residual calculations, but test and HPC scripts are not a supported public
analysis API. The implementation plan must inspect the closest maintained symbols
before deciding whether to import a public helper or implement a publication-local
calculation.

No Cy-ExM user-callable API changes are proposed. A future reusable morphology or
deformation-QC API requires a separate design after validation across multiple
markers and an independent dataset.

## Acceptance criteria

The analysis implementation is complete only when:

1. the machine-readable manifest agrees with the audited Golgi and Lamin
   correspondence and acquisition metadata;
2. LAMP1 is either independently mapped, exported, registered, and accepted or is
   explicitly absent from completed results;
3. every included cell-round passes artifact, geometry, identity, and scientific
   registration gates, with exclusions and sensitivity analyses recorded;
4. transform-derived deformation and post-registration target morphology remain
   separate in tables, figures, and prose;
5. primary endpoints are computed in correct physical coordinates and report both
   expanded and nominal native-space units;
6. rounds 2--5 and the round-1 signal sensitivity analysis remain distinct;
7. synthetic, cropped-data, sequential/process-equivalence, restart, and figure
   tests pass;
8. the full analysis completes restart-safely with a recorded resource pilot and
   bounded process queue;
9. every plotted value is recoverable from committed compact tables and immutable
   run metadata;
10. PDF and PS text uses TrueType font type 42, SVG text remains editable, and
    publication figures pass visual review at final size;
11. the Markdown report states the theoretical-resolution and nominal-expansion
    assumptions and avoids unsupported claims; and
12. the Cy-ExM repository contains no publication data, generated analysis output,
    or analysis runtime changes beyond this approved specification and its later
    implementation plan.
