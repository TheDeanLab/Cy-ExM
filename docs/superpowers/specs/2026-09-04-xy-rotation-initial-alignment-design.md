# XY Rotation Initial Alignment Design

## Context and current behavior

Cy-ExM currently begins every moving-round linear registration without a
dataset-declared geometric seed. TRSAA can recover modest pose differences, but a
roughly 30-degree inter-round XY rotation caused a scientifically failed MDA-MB-231
registration. A target-blind recovery estimated an XY correction from the fixed and
moving tubulin channels, applied the same correction to every channel in that round,
and then used TRSAA for residual linear registration.

That recovery proved the scientific value of a rotation seed, but it materialized a
second raw-data tree on expanded canvases. Productizing that exact operational form
would duplicate large inputs and interpolate every dependent channel before its final
fixed-grid resampling.

The existing software already has the correct ownership boundaries:

- dataset manifests own scientific choices;
- execution profiles own paths, resources, and scheduler settings;
- `ImageRegistration._prepare_linear_state` is the shared affine preparation path
  used by whole-image, serial-chunked, coarse, and fine-grained workflows;
- `linear_transform_{round}.mat` is the accepted complete affine artifact; and
- dependent channels are transformed once through the established forward
  `[warp, affine]` chain.

## Goals

- Add optional per-moving-round XY rotation initialization to scientific manifests.
- Support both an explicit correction angle and automatic target-blind estimation.
- Use only the declared fixed and moving registration channels for estimation.
- Keep TRSAA and every other configured linear method responsible for residual
  linear alignment.
- Preserve one final interpolation of each raw registration and dependent channel.
- Persist strict, restart-safe initial-transform and QC evidence.
- Include all normalized settings and derived results in provenance and cache
  identity.
- Preserve current behavior and fingerprints when initialization is omitted.
- Support whole-image and serial-chunked registration under every executor without
  separate scientific implementations.

## Non-goals

- Do not infer rotations from dependent biological target channels.
- Do not modify, replace, or generate source TIFFs as part of normal workflow use.
- Do not add arbitrary 3-D rigid initialization, scale initialization, reflection,
  or landmark registration in this change.
- Do not change the configured linear or nonlinear methods, metrics, schedules,
  masks, crop policy, or interpolation policy.
- Do not silently accept an unidentifiable automatic estimate.
- Do not add rotation settings to execution profiles.
- Do not introduce a new registration strategy or executor task family.

## Public configuration contract

Initialization is per moving round because its correction differs by round. The
optional field is named `initial_alignment`, rather than extending the existing
global `registration.preprocessing` mapping: intensity-feature preprocessing and
geometric transform initialization have different scope and invalidation behavior.

Omission preserves the current workflow exactly.

### Explicit mode

```yaml
rounds:
  - round: 1
    strategy: serial_chunked
    registration_channel: round_1_555_tubulin.tif
    initial_alignment:
      xy_rotation:
        mode: explicit
        angle_deg: 27.7
    chunk_size_zyx: [32, 512, 512]
    overlap_fraction: 0.25
    channels:
      - file: round_1_555_tubulin.tif
        interpolation: linear
```

The explicit mapping has exactly `mode` and `angle_deg`. `angle_deg` is a finite
number in `[-180, 180]` and is normalized to a built-in float.

### Automatic mode

```yaml
rounds:
  - round: 1
    strategy: serial_chunked
    registration_channel: round_1_555_tubulin.tif
    initial_alignment:
      xy_rotation:
        mode: estimate
        search_range_deg: [-45, 45]
        coarse_step_deg: 1
        refinement_step_deg: 0.1
        minimum_score_gain: 0.01
        minimum_peak_margin: 0.01
    chunk_size_zyx: [32, 512, 512]
    overlap_fraction: 0.25
    channels:
      - file: round_1_555_tubulin.tif
        interpolation: linear
```

For `mode: estimate`, all fields after `mode` are optional and normalize to the
values shown above. The search interval contains exactly two finite values in
strictly increasing order within `[-180, 180]`. Step sizes and confidence thresholds
are finite and positive; the refinement step must be smaller than the coarse step.
Unknown keys and fields from the other mode are rejected.

Schema version 1 gains this optional round field. Existing manifests remain valid,
and absence of the field emits no initialization mapping in canonical scientific
JSON. This follows the existing schema-1 compatible-extension policy while retaining
strict key validation.

## Coordinate and transform convention

`angle_deg` is the correction applied to the moving image about the physical center
of its cropped registration-channel geometry. Positive angles follow the right-hand
rule about the moving image's physical positive Z axis. Documentation must include
an identity-direction example and explain the apparent sign in a conventional
row-down image display: positive physical rotation appears clockwise and therefore
has the opposite sign from `scipy.ndimage.rotate`.

The implementation converts array offsets and angles through the full ANTs origin,
spacing, and direction geometry. It must not assume identity direction matrices or
isotropic XY pixels. The seed and the final affine use the same moving-to-fixed
resampling semantics already accepted by Cy-ExM. Focused landmark tests, rather than
parameter inspection alone, establish the sign, center, and composition order.

## Automatic estimator policy

The versioned policy identifier is
`mip-gradient-phase-correlation-xy-rotation-v1`. It operates after the existing crop
step and before affine estimation:

1. Convert each cropped registration image to one ZYX floating array and reject
   nonfinite or constant input.
2. Compute a Z maximum-intensity projection.
3. Robustly scale each projection from its finite 1st and 99th percentiles, clipping
   to `[0, 1]`.
4. Compute XY gradient magnitude after a Gaussian smoothing sigma of 2 pixels.
5. For every inclusive coarse candidate, rotate only the moving 2-D feature about
   its center on a zero-filled expanded canvas with linear interpolation.
6. Pad the fixed feature onto the same canvas, estimate integer translation by phase
   cross-correlation, and score the translated valid overlap with Pearson
   correlation.
7. Search the inclusive refinement interval of plus or minus one coarse step around
   the winning coarse angle, clipped to the configured search range.
8. Select the highest score, breaking exact ties by smaller absolute angle and then
   smaller numerical angle.
9. Convert the winning XY translation to physical XYZ and construct the seed
   transform.

The implementation records the complete coarse and refined score curves, winning
translation, valid-overlap fraction, zero-angle score, score gain, and independent
peak margin. The competing peak for the margin is the best candidate separated from
the winner by at least one coarse step.

An automatic estimate fails closed when either input is invalid, no candidate has
positive valid overlap, the winner lies on either configured search boundary, a
nonzero winner's score gain over zero is below `minimum_score_gain`, or the
independent peak margin is below `minimum_peak_margin`. A winner within one
refinement step of zero is normalized as a valid zero-rotation seed and is not
required to improve upon itself; this permits translation-only initialization. The
error identifies the failed criterion and points to the persisted diagnostic JSON.
No affine or nonlinear work begins after a failed automatic estimate.

## Selected transform architecture

The initial alignment is a transform seed, not a public preprocessed image.

1. Load and optionally crop raw fixed and moving registration images.
2. Construct the explicit seed or estimate it from the raw cropped registration
   images.
3. Construct any existing intensity features and feature-derived masks without
   changing their raw-output boundary.
4. Supply the persisted seed to the configured ANTs linear registration as its
   initial transform.
5. Compose the returned residual affine and initial transform into one affine with
   the exact mapping order verified by physical-landmark tests.
6. Persist that complete transform at the existing
   `linear_transform_{round}.mat` path.
7. Apply the complete affine once to raw registration data before nonlinear
   registration, then apply the nonlinear warp.
8. Apply the established `[warp, affine]` chain once to each original dependent
   channel.

The public affine artifact therefore retains its existing meaning: it is the complete
linear moving-to-fixed transform. No downstream caller needs a three-transform chain,
and historical consumers of the affine-plus-warp pair remain compatible.

## Persistent artifacts and restart authority

Enabled rounds add three planned public artifacts in their registration directory:

- `initial_alignment_{round}.mat`: the exact seed supplied to linear registration;
- `initial_alignment_{round}.json`: canonical settings, result, scores, geometry,
  algorithm version, source identities, and acceptance decision; and
- `initial_alignment_qc_{round}.png`: fixed, unaligned moving, initialized moving,
  and overlay panels plus the angle-score curve.

Each artifact is atomically written and paired with existing completion metadata.
The JSON is the semantic authority; the transform and PNG must match its request and
source identities. Explicit mode records the supplied angle, zero translation, and
`source: explicit`. Estimate mode records `source: estimated` and all realized
evidence.

The initial alignment is included in the affine request. A changed angle, mode,
estimator setting, policy version, source identity, crop result, or realized estimate
invalidates the affine and every descendant: linear metrics, prepared arrays, local
warps, assembled warp, nonlinear metrics, registered scalar, and transformed
channels. Reference materialization remains independently reusable.

Prepared bundle formats 2 through 4 remain valid only for initialization-disabled
historical work. New formats 5, 6, and 7 represent rotation with raw scalars,
rotation plus intensity features, and rotation plus intensity features and feature
masks, respectively. They embed the initial-alignment JSON and transform identities
alongside the complete affine identity.

## Workflow and dependency boundaries

- `cyexm.config` owns strict YAML models and normalization without filesystem or
  scientific-runtime imports.
- A new low-level `cyexm.registration.initial_alignment` module owns the validated
  runtime policy, target-blind estimator, evidence values, and QC rendering. NumPy,
  SciPy, scikit-image, Matplotlib, and ANTs imports remain deferred.
- The registration backend owns ANTs transform construction and affine composition.
- `ImageRegistration._prepare_linear_state` owns seed resolution and passes the
  complete affine into existing downstream stages.
- Workflow planning maps config values into immutable round tasks and fingerprints
  normalized scientific intent. It performs no estimation and writes no artifacts.
- Coarse and fine execution reuse the same preparation implementation. No worker
  creates a nested graph or privately re-estimates a seed.
- The Phase 1 artifact store remains the only restart authority.

`ImageRegistration` is extended with one keyword-only `initial_alignment` value;
`register_round` forwards it. The low-level policy type is exported from
`cyexm.registration`. This extends existing resource owners instead of adding a
parallel high-level runner.

## Failure and recovery behavior

- Invalid YAML fails during `workflow validate` with the exact YAML path.
- Explicit-transform construction errors fail the `initial-alignment` stage.
- Automatic confidence failures persist diagnostic JSON and PNG with
  `accepted: false`, then fail the round before affine registration.
- A cached seed is reused only when its semantic JSON, transform, completion records,
  crop evidence, geometry, and sources all validate.
- Missing, malformed, or mismatched initialization artifacts make workflow status
  `partial` or `invalid`; they are never inferred from an affine transform.
- `--force-round` stales and rebuilds the seed and all moving descendants for that
  round. Forcing the reference continues to propagate to every moving descendant.
- Initial-alignment omission never causes historical affine records to expect new
  fields.

## Documentation and examples

User documentation will distinguish intensity-feature preprocessing from geometric
initial alignment and update:

- the complete dataset-manifest field table and examples;
- a new focused `initial-alignment.md` guide;
- registration foundations and architecture/data-flow diagrams;
- registration artifact, output, fingerprint, force, and recovery behavior;
- API documentation for the new policy types and extended registration entry points;
- one explicit-angle example manifest and one automatic-estimation example manifest;
  and
- the MyST table of contents and documentation contract tests.

The guide will explain angle sign, search-range selection, confidence failures,
target blindness, QC interpretation, and why dependent channels are not used to
choose the seed.

## Alternatives considered

### Materialized rotated TIFF inputs

Rejected as the software architecture. It duplicates large data, requires expanded
canvas bookkeeping, and interpolates dependent channels before their final
registration. It remains a useful external recovery technique and validation oracle.

### A new rotation-aware registration strategy

Rejected. Rotation is initialization shared by both existing strategies, not a third
orchestration lifecycle.

### Apply a separate seed during every channel transform

Rejected. It would change the established downstream transform list and create more
ways for producers and readers to disagree. Persisting the composed affine preserves
the existing two-transform consumer contract.

### Automatic estimation without explicit mode

Rejected. Explicit mode is the deterministic baseline, supports externally reviewed
angles, and provides a direct recovery path when automatic confidence checks reject
an otherwise interpretable dataset.

## Testing

- Strict config tests for omission, both modes, normalization, bounds, cross-field
  rules, unknown keys, and direct model revalidation.
- Pure estimator tests with synthetic asymmetric volumes, known rotations and
  translations, deterministic ties, degenerate inputs, boundary winners, and low
  confidence.
- Physical-geometry tests covering anisotropic spacing, nonzero origin, nonidentity
  direction, transform center, sign, and composition order.
- Backend tests proving the seed reaches ANTs and the persisted affine represents the
  complete transform.
- Whole-image and serial-chunk tests proving raw outputs are transformed once and
  dependent channels retain the `[warp, affine]` chain.
- Artifact tests for atomic commit, exact reuse, partial writes, force behavior,
  source changes, and descendant invalidation.
- Prepared-format and fine-graph tests for enabled/disabled and all feature-mask
  combinations.
- Workflow planning, canonical scientific JSON, status, executor parity, and
  documentation contract tests.
- Focused real-ANTs integration against a small synthetic volume before any archived
  cohort rerun.
- An optional operator-approved MDA-MB-231 characterization comparing software
  estimates and registered overlays with the external recovery evidence.

## Acceptance criteria

- Existing initialization-disabled manifests, fingerprints, transform paths, and
  outputs remain compatible.
- Explicit mode recovers a known synthetic XY rotation with the documented sign and
  produces the same complete affine under whole-image and chunked paths.
- Automatic mode recovers known synthetic rotations and translations within one
  refinement step and rejects every specified low-confidence condition.
- All declared channels use the same accepted round transform without contributing
  to its estimate or receiving a preliminary materialized rotation.
- The final channel-transform contract remains exactly `[warp, affine]` with one
  interpolation onto the fixed grid.
- Scientific plans, completion records, run records, status output, JSON evidence,
  and QC images expose the initialization policy and realized result.
- Sequential, local-Dask, coarse-SLURM planning, and fine-grained planning have the
  same scientific fingerprint for the same manifest.
- Targeted tests, portable unit and integration suites, formatting, linting, typing,
  documentation build, and pre-commit gates pass before release.
