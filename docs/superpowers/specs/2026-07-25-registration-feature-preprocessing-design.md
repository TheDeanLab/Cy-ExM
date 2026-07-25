# Optional Registration-Feature Preprocessing Design

**Status:** Approved design; implementation not yet started
**Date:** 2026-07-25

## Context

Cy-ExM currently estimates linear and nonlinear transforms directly from the raw
registration-channel intensities. Preliminary analysis in
`/archive/bioinformatics/Danuser_lab/Dean/dean/2026-07-25-normalization`
found that one narrowly defined feature transform improved the balance between
bright and dim microtubule signal: Gaussian background subtraction followed by
robust scaling and an inverse hyperbolic sine transform.

The accepted preliminary implementation:

1. converts a finite three-dimensional image to `float32`;
2. estimates a Gaussian background;
3. subtracts that background to form a signed residual;
4. estimates a robust scale from the signed residual;
5. clips negative residuals to zero; and
6. computes `asinh(clipped_residual / robust_scale)`.

The other evaluated transforms—local contrast normalization, `log1p`,
winsorization, Anscombe, Frangi, and steerable-ridge variants—are explicitly
outside this design.

The scientifically important boundary is that preprocessing changes only the
images used to estimate transforms. Cy-ExM must apply the resulting affine and
nonlinear transforms to the original raw registration channel and all original
dependent channels.

## Goals

- Add one optional, strict-schema registration preprocessing policy.
- Reproduce the accepted background-subtraction-plus-asinh algorithm.
- Support one Gaussian sigma for all axes or independent ZYX sigma values.
- Use the feature images for both TRSAA/linear and nonlinear transform
  estimation.
- Use raw images for masks, quality metrics, registered outputs, and dependent
  channel outputs.
- Support whole-image, serial-chunked, and fine-grained execution.
- Bind preprocessing to scientific planning, cache identity, completion
  provenance, and restart validation.
- Preserve the current numerical and artifact path when preprocessing is
  omitted.
- Document the scientific behavior and provide reusable YAML examples.

## Non-goals

- Implement any normalization method other than the accepted Gaussian
  background subtraction, robust scaling, and `asinh`.
- Select or optimize sigma values automatically.
- Interpret Gaussian sigma in physical units.
- Preprocess dependent channels.
- Write normalized TIFFs into the public dataset output.
- Change interpolation, transform order, cropping, mask thresholding, or the
  accepted nonlinear metric configuration.
- Change the behavior of preprocessing-disabled registrations.

## Selected architecture

Preprocessing is an explicit registration-feature layer between raw-image
preparation and transform estimation. It is not a separate user-run conversion
workflow and is not hidden inside the ANTs backend.

The registration engine owns two image roles:

- **Registration features:** preprocessed fixed and moving registration channels
  used only by linear and nonlinear estimators.
- **Scientific images:** cropped raw fixed and moving images used to derive masks,
  calculate metrics, and reconstruct outputs.

This boundary keeps one authoritative raw dataset, makes the scientific operation
visible in configuration and provenance, and works with the existing prepared
fixed-grid bundle used by fine-grained nonlinear workers.

## Configuration schema

Schema version 1 gains an optional `registration.preprocessing` object.
Omitting the object disables preprocessing.

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

The filter may instead be given as one scalar:

```yaml
registration:
  preprocessing:
    filter_sigma_zyx: 12
```

An empty object enables preprocessing with the default:

```yaml
registration:
  preprocessing: {}
```

The block accepts exactly the optional key `filter_sigma_zyx`. The value must be
either one positive finite built-in number or a list of exactly three positive
finite built-in numbers in NumPy ZYX order. Booleans are not numbers for this
schema. Unknown keys, zero, negative values, nonfinite values, and incorrectly
sized lists fail strict validation.

The normalized model always stores a three-float ZYX tuple. The scalar `12`, the
list `[12, 12, 12]`, and an empty preprocessing object therefore normalize to the
same scientific value `(12.0, 12.0, 12.0)` and have the same fingerprint.

When a normalized manifest is serialized, enabled preprocessing is written with
an explicit three-value ZYX list. Disabled preprocessing is omitted.

## Approved feature algorithm

The production function accepts one three-dimensional image and canonical ZYX
sigma values. NumPy and SciPy imports remain deferred until scientific execution.

Given a `float32` input array \(I\):

1. Validate that `I` is three-dimensional and contains only finite values.
2. Compute
   `background = scipy.ndimage.gaussian_filter(I, sigma=sigma_zyx)`.
   The implementation explicitly uses SciPy's reflective boundary behavior so
   the accepted edge policy is not implicit.
3. Compute the signed residual `R = I - background`.
4. Set `stride = max(1, R.size // 2_000_000)`.
5. Take the deterministic sample `S = R.ravel()[::stride]`.
6. Compute `median = median(S)` and
   `mad = median(abs(S - median))`.
7. Compute
   `scale = max(1.4826 * mad, finfo(float32).eps)`.
8. Compute `C = maximum(R, 0)`.
9. Return `asinh(C / scale)` as a finite, shape-preserving `float32` array.

The robust scale deliberately comes from the signed residual before negative
values are clipped. Fixed and moving images are scaled independently. The actual
finite scales are deterministic derived evidence, not user parameters.

The algorithm receives a versioned policy identifier. Any future numerical change
requires a new identifier and corresponding cache/provenance update.

## Scientific data flow

For each moving round:

1. Load the fixed and moving raw registration channels with their declared XYZ
   voxel spacing.
2. Apply the existing robust crop to raw images when cropping is enabled.
3. Derive fixed and moving masks from the cropped raw values using the existing
   greater-than-zero policy.
4. Calculate pre-registration metrics from raw fixed and raw moving images.
5. Independently derive fixed and moving registration features.
6. Estimate the affine transform from the feature images using the raw-derived
   masks.
7. Apply the affine transform to both the moving feature image and the raw moving
   image on the exact fixed grid.
8. Calculate linear metrics from raw fixed and affine-transformed raw moving
   images.
9. Estimate nonlinear deformation from the fixed feature and affine-transformed
   moving feature, again using raw-derived masks.
10. Apply the accepted warp-plus-affine chain to the original raw moving
    registration channel.
11. Calculate nonlinear metrics and reconstruct the registered scalar from raw
    intensities.
12. Apply the identical accepted chain to each declared original dependent
    channel through the existing channel-transform workflow.

Cropping precedes preprocessing. Consequently, the Gaussian background represents
the exact registration field of view and does not incorporate signal outside an
accepted crop.

Preprocessing does not alter the fixed grid, image origins, image directions,
voxel spacing, transform order, or interpolation policies. Images may continue to
have different native sizes along any spatial axis.

## Raw output and metric semantics

The existing pre-, linear-, and nonlinear-registration metric filenames continue
to mean metrics calculated from raw intensities. The preprocessing feature is not
substituted into those files.

When preprocessing is enabled, the registration scalar is reconstructed from the
raw moving registration channel with the accepted transforms. Background
subtraction, robust scaling, and `asinh` are not applied to public outputs.
Dependent channel outputs continue to use the existing uint16 clipping,
no-normalization policy.

The raw registration-channel transform and each raw dependent-channel transform
use the same fixed grid, affine, warp, transform order, and declared
interpolation. Feature data never become a dependent-channel source.

Preprocessing-disabled execution retains the existing registered-scalar
postprocessing behavior, including the current whole-image and serial-chunk
policies. This avoids changing established numerical behavior outside the opt-in
feature workflow.

## Execution-mode integration

### Whole-image registration

The registration engine derives feature images in memory. Linear and nonlinear
estimators receive features, while raw counterparts remain available for metrics
and output reconstruction. Restart requests identify the feature algorithm and
parameters through the raw source identities and deterministic preprocessing
policy.

### Serial-chunked registration

Affine preparation follows the same feature/raw split. Nonlinear chunks receive
fixed and affine-transformed moving features. After the accepted warp is
assembled, the public scalar is reconstructed from the raw registration channel.

### Fine-grained registration

The preparation task persists the fixed feature and affine-transformed moving
feature in the existing internal prepared-round scalar array roles. Its mask
arrays remain raw-derived. Nonlinear chunk workers therefore need no raw
intensities and continue to consume immutable fixed-grid prepared regions.

Assembly authenticates and commits the nonlinear warp, then reconstructs raw
registration output. Channel tasks continue to load original channel sources.
Prepared manifests must record the feature policy and bind their scalar arrays to
the raw sources, affine, crop evidence when present, canonical sigma, algorithm
version, and actual robust scales.

No mode writes a normalized public TIFF.

## Planning, cache identity, and provenance

Enabled preprocessing appears canonically in the scientific plan with:

- the versioned algorithm identifier;
- the canonical three-value ZYX sigma;
- the `float32` computation policy;
- the deterministic signed-residual MAD sampling policy;
- the negative-clipping policy; and
- the `asinh` policy.

Omitted preprocessing does not add a scientific-plan key. Existing disabled
schema-1 plans and fingerprints therefore remain compatible.

Transform-estimation completion requests include the normalized preprocessing
policy. Actual fixed and moving robust scales are recorded as finite derived
evidence in completion provenance and logs. Fine prepared-array and manifest
requests additionally bind the materialized feature-array identities.

Changing from disabled to enabled preprocessing, disabling it, or changing any
sigma axis invalidates the affine transform and every transform-derived
descendant:

- pre-registration, affine, and nonlinear metrics;
- affine transform;
- prepared feature arrays and manifests;
- nonlinear chunks, manifests, and warp;
- registered scalar output; and
- transformed registration and dependent channels through changed transform
  identities.

Unchanged raw reference-channel materialization and crop-bound artifacts remain
reusable because their operations do not depend on preprocessing.

Completion readers reconstruct the same expected requests as producers.
Scientific-plan, task, request-policy, prepared-manifest, and round-completion
validation must agree on the canonical preprocessing value.

## Failure handling

- Configuration errors identify the exact YAML path.
- Runtime validation rejects non-3D or nonfinite feature inputs before invoking
  ANTs.
- Gaussian filtering, robust-scale calculation, and feature construction failures
  are reported as an explicit preprocessing stage for the affected round.
- Atomic artifact rules remain authoritative: a data file without matching
  completion evidence is not reusable.
- A failed or interrupted prepared feature write cannot be accepted as a complete
  array or manifest.
- Source identities are verified across feature computation and artifact commit,
  preserving the current source-mutation protections.
- Cache validation failures mark affected outputs stale according to existing
  restart policy rather than silently reusing them.

## Alternatives considered

### Pre-generate normalized TIFFs

A separate command could write feature TIFFs and require YAML to declare both
feature and raw sources. This was rejected because it duplicates large data,
creates an avoidable pairing error between feature and raw images, and weakens
end-to-end provenance.

### Preprocess inside the ANTs backend

The backend could silently alter images before individual registration calls.
This was rejected because it hides a scientific policy inside a low-level adapter
and makes raw metrics, prepared arrays, cache identity, and backend-independent
testing harder to reason about.

## Test strategy

### Algorithm regression tests

- Compare production output with an independent reference expression of the
  accepted preliminary formula.
- Prove scalar and equivalent isotropic ZYX sigma inputs are identical after
  configuration normalization.
- Demonstrate that anisotropic ZYX sigma acts along the declared axes.
- Prove robust scale is calculated from the signed residual before clipping.
- Verify constant input produces finite zeros.
- Verify shape preservation and `float32` output.
- Reject non-3D and nonfinite arrays.

### Configuration and planning tests

- Omission disables preprocessing.
- An empty object enables the `[12.0, 12.0, 12.0]` default.
- Scalars expand to three axes.
- Strict invalid values and unknown keys fail.
- Canonical serialization uses an explicit ZYX list.
- Equivalent scalar and list forms produce identical scientific plans and
  fingerprints.
- Changing any axis changes the fingerprint.
- Existing preprocessing-disabled manifest and plan fixtures remain valid.

### Registration and restart tests

- Assert linear and nonlinear estimators receive feature images.
- Assert masks originate from raw images.
- Assert all existing metrics receive raw intensity images.
- Assert the registered scalar and dependent channels originate from raw sources.
- Cover whole-image, serial-chunked, and fine-grained paths.
- Verify unchanged enabled preprocessing reuses artifacts.
- Verify a policy or sigma change invalidates transforms, prepared state, and
  descendants while retaining unrelated reference/crop reuse.
- Verify source mutation and interrupted feature artifacts are rejected.
- Verify preprocessing-disabled behavior follows the legacy path.

### Scientific integration tests

- Use synthetic bright and dim structures with a known transformation to show that
  feature preprocessing can affect transform estimation without replacing output
  intensities.
- Compare serial and fine-grained results under one identical preprocessing
  policy.
- Add a targeted real-ANTs integration test where its runtime remains appropriate;
  routine unit tests remain deterministic and lightweight.

## Documentation

Implementation updates:

- the README configuration, execution, artifact, restart, and scientific-semantics
  sections;
- example YAML files for omitted, default-enabled, scalar, and anisotropic
  preprocessing;
- maintained technical documentation describing the feature/raw boundary;
- cache and provenance documentation;
- API documentation for every changed public callable; and
- the most relevant layered `AGENTS.md` files.

Documentation will explain the Gaussian background, signed-residual MAD scale, and
`asinh` transform without presenting the rejected preliminary methods as supported
features.

## Acceptance criteria

The feature is accepted when:

1. the exact strict YAML semantics above are implemented;
2. the production feature output matches the approved formula;
3. TRSAA/linear and nonlinear estimation consume feature images in every execution
   mode;
4. masks, existing metrics, registered output, and channel outputs consume raw
   images;
5. preprocessing provenance and restart identities are complete and producer/reader
   validation agrees;
6. omitted preprocessing preserves current behavior and plan compatibility;
7. targeted, broader, formatting, lint, typing, and documentation checks pass; and
8. README, examples, technical documentation, and layered agent guidance are
   synchronized with the implementation.
