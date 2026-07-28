# Feature-derived registration mask design

**Status:** Approved design; implementation and scientific acceptance remain pending.

## Purpose

Cy-ExM can estimate TRSAA and nonlinear transforms from background-subtracted,
asinh-compressed registration features while applying the accepted transforms to the
original raw images. On the spheroid characterization dataset, approximately 63.5%
of the preliminary asinh feature voxels were zero. The current raw greater-than-zero
masks include the entire positive-valued acquisition and therefore do not exclude
those feature-zero locations from transform estimation.

Add an optional feature-derived mask that restricts registration metrics to
informative positive asinh support. The mask is intended primarily to reduce
uninformative metric participation and improve registration balance between bright
and dim structures. Runtime reduction is a secondary hypothesis, not a promised
outcome, because ANTs still constructs dense image pyramids and deformation fields.

## Scope

This change will:

- add one optional mask policy beneath the existing optional preprocessing policy;
- derive independent fixed and moving registration masks from their corresponding
  asinh feature images;
- remove only physically small disconnected support and dilate retained support by a
  small physical radius;
- use feature masks in every TRSAA and SyN metric stage;
- preserve raw-derived diagnostic masks, metrics, registered outputs, and dependent
  channel outputs;
- preserve identical behavior when the feature-mask block is omitted;
- bind the policy and realized mask evidence to plans, cache identity, provenance,
  validation, status, examples, and documentation; and
- add synthetic and scientific regression tests.

This change will not:

- threshold, floor, or otherwise modify raw public outputs;
- preprocess dependent channels;
- change feature intensities;
- change TRSAA or SyN schedules, metrics, interpolation, transform order, chunk
  geometry, or nonlinear CC radius;
- make feature-derived masking the default; or
- claim a production accuracy or performance improvement before scientific
  characterization.

## Configuration

Schema version 1 is extended in place. Omitting `feature_mask` disables the new
behavior and preserves the exact current preprocessing path:

```yaml
registration:
  preprocessing:
    filter_sigma_zyx: [12, 12, 12]
```

An empty mapping enables the canonical policy:

```yaml
registration:
  preprocessing:
    filter_sigma_zyx: [12, 12, 12]
    feature_mask: {}
```

The explicit equivalent is:

```yaml
registration:
  preprocessing:
    filter_sigma_zyx: [12, 12, 12]
    feature_mask:
      min_component_volume_um3: 1.0
      dilation_radius_um: 1.0
```

Both values are positive finite scalars in physical micrometer units. Serialization
always writes enabled values explicitly. Unknown keys, booleans, zero, negative
values, non-finite values, and non-scalar sequences fail strict validation.

The mask block is invalid when preprocessing is absent because no registration
feature exists from which to derive it. There is no separate `enabled` boolean:
omission disables the policy and presence enables it, matching the existing optional
preprocessing-block semantics.

The two provisional defaults are deliberately exposed for characterization and
cross-dataset use. They are not claimed to be optimized biological constants.

## Mask algorithm

For each cropped fixed or moving registration image:

1. Construct the existing float32 registration feature without numerical changes:
   reflective Gaussian background subtraction, signed-residual MAD scaling,
   negative clipping, and `asinh`.
2. Define initial support as finite `feature > 0`.
3. Label support using full 3D connectivity.
4. Compute voxel volume from the image spacing as
   `spacing_z * spacing_y * spacing_x` after converting ANTs XYZ spacing to NumPy ZYX
   order.
5. Convert `min_component_volume_um3` to a minimum voxel count with
   `ceil(volume_um3 / voxel_volume_um3)`, with a lower bound of one voxel.
6. Remove connected components whose voxel count is less than that threshold.
7. Construct an ellipsoidal binary footprint containing every integer ZYX offset
   whose physical Euclidean distance from the center is less than or equal to
   `dilation_radius_um`. The center voxel is always included.
8. Dilate the retained support once with that footprint.
9. Emit a finite float32 binary mask with exact values `{0.0, 1.0}` and the feature
   image's shape, origin, spacing, and direction.

Mask construction is deterministic and occurs on the full cropped image before
nonlinear chunks are extracted. This prevents connected-component or dilation
decisions from changing at chunk boundaries.

If no foreground remains, Cy-ExM fails before invoking ANTs with an error identifying
the round, image role, and mask policy. It never silently falls back to the raw mask.

## Scientific data flow

The preprocessing-enabled flow becomes:

```text
cropped raw fixed/moving
    ├──> raw > 0 masks ----------------> raw diagnostics and quality metrics
    └──> asinh registration features
              └──> optional feature masks
                         ├──> TRSAA metric masks
                         └──> fixed-grid/chunk SyN metric masks

accepted affine + warp ----------------> cropped raw registration channel
                                     └─> raw dependent channels
```

Fixed and moving feature masks are derived independently. TRSAA receives both masks
with `mask_all_stages=True`. The moving feature mask follows the accepted affine
using nearest-neighbor interpolation and is thresholded at 0.5 on the fixed grid.
Nonlinear registration receives the fixed mask and affine-resampled moving mask.

Serial chunking extracts both masks from full fixed-grid arrays. Fine-grained
execution persists and reads the same full-grid arrays before extracting extended
regions, so coarse and fine execution use scientifically identical mask values.

Scalar interpolation remains linear. Mask interpolation remains nearest-neighbor.
Transforms remain warp-then-affine when applied to raw outputs.

## Internal separation

The existing mask concept is currently used across registration and diagnostic
boundaries. Implementation must distinguish:

- **raw diagnostic masks**, derived from cropped raw values greater than zero and
  retained for pre/post metrics and existing raw scientific evidence; and
- **registration metric masks**, equal to the raw masks when feature masking is
  disabled and equal to feature-derived masks when it is enabled.

The feature-mask helper belongs in `cyexm.registration.preprocessing`, beside the
only supported feature transform. It accepts a feature array, image spacing, and the
canonical policy and returns a binary array plus deterministic evidence. ANTs image
construction and geometry ownership remain in the backend/orchestration layers.

No new public Phase 1 class is introduced. Existing registration constructors gain
only the minimum internal keyword-only policy needed to carry the normalized mask
configuration through coarse and fine execution.

## Artifacts and prepared data

When feature masking is disabled, all current prepared formats and artifact
identities remain byte-for-byte compatible.

When enabled:

- prepared fixed-grid mask roles contain registration metric masks, not raw
  diagnostic masks;
- raw masks remain available in the in-process preparation state for metrics;
- the prepared manifest advances to a new strict format and records the complete
  feature-mask policy and realized fixed/moving mask evidence;
- realized evidence includes mask voxel count, total voxel count, foreground
  fraction, and exact binary value policy; and
- readers reject missing, malformed, nonbinary, non-finite, geometry-mismatched, or
  policy-mismatched masks.

The prepared manifest remains an internal progress artifact. Ordinary Phase 1
completion records remain restart authority.

## Planning, identity, and provenance

The normalized scientific plan records, when enabled:

```yaml
preprocessing:
  algorithm: gaussian-background-subtraction-asinh
  filter_sigma_zyx: [12.0, 12.0, 12.0]
  feature_mask:
    algorithm: positive-support-components-physical-dilation-v1
    min_component_volume_um3: 1.0
    dilation_radius_um: 1.0
```

The algorithm identifier and both values participate in:

- the scientific fingerprint;
- affine completion requests;
- prepared-manifest identity;
- nonlinear chunk and full-warp requests;
- validation reconstruction;
- status reconstruction; and
- provenance exported for completed moving rounds.

Changing the feature-mask policy invalidates affine and every transform-dependent
descendant, including nonlinear registration and transformed channels. It does not
invalidate reference preparation, crop artifacts, or source identities that do not
depend on registration metric masks.

Omitted feature masking contributes no new key to existing scientific plans or
completion requests. This preserves restart compatibility for current workflows.

## Validation and status

Configuration validation fails early for invalid nesting, keys, types, units, or
values. Runtime validation additionally requires:

- fixed and moving masks with the exact corresponding image geometry;
- finite float32 binary mask values;
- at least one foreground voxel;
- realized evidence consistent with the arrays; and
- exact agreement between configured, planned, prepared, and completed policy.

Workflow status parses and reconstructs the optional mask policy. A completed round
with feature masking is accepted only when the full affine-to-channel provenance
chain carries matching policy and mask evidence. Status must not report a false
configuration mismatch for valid completed feature-masked rounds.

Logs report foreground count and fraction for fixed and moving masks at preparation
time. They do not emit feature images or masks as public TIFF products.

## Testing

### Unit tests

- Parse omission, an empty enabling block, and explicit canonical values.
- Reject invalid nesting, unknown keys, booleans, sequences, non-finite values, and
  nonpositive values.
- Round-trip enabled and disabled manifests exactly.
- Verify initial support uses strict `feature > 0`.
- Verify full 3D connected-component filtering at anisotropic spacing.
- Verify ellipsoidal physical dilation and exact `{0.0, 1.0}` float32 output.
- Verify deterministic results and empty-mask failure.
- Verify fixed and moving masks are independently derived.
- Verify raw diagnostic masks remain raw-derived.
- Verify disabled requests and prepared manifests remain unchanged.
- Verify policy changes invalidate affine and descendants but not independent
  reference preparation.
- Verify status accepts matching evidence and rejects malformed or mismatched
  evidence.

### Integration tests

- Run a synthetic unequal-shape registration through TRSAA, affine mask resampling,
  nonlinear chunking, and raw output reconstruction.
- Confirm ANTs receives feature masks with `mask_all_stages=True`.
- Confirm moving-mask resampling is nearest-neighbor and binary on the fixed grid.
- Confirm coarse and fine paths consume identical fixed-grid mask arrays and preserve
  chunk-boundary behavior.
- Confirm registered and dependent-channel outputs originate from raw arrays.
- Confirm restart reuse succeeds only for the exact mask policy.

### Scientific characterization

Use the existing spheroid bright/dim regions and compare, under otherwise identical
settings:

1. preprocessing disabled;
2. asinh preprocessing without feature masking; and
3. asinh preprocessing with the canonical feature mask.

Run both the accepted Mattes configuration and CC SyN radius 4 where practical.
Record regional raw-image registration metrics, deformation validity, seams,
foreground fractions, TRSAA and nonlinear runtimes, and total runtime. Because
TRSAA uses stochastic Mattes sampling, compare replicated or controlled-thread runs
rather than treating one voxelwise result as deterministic.

Feature masking remains opt-in after implementation. Promotion to a recommended
default requires evidence of improved dim-region alignment without unacceptable
bright-region degradation, instability, invalid deformation, or seam artifacts on
an approved dataset.

## Documentation

Update:

- the README schema table and preprocessing section;
- the task-oriented workflow documentation;
- registration-artifact and provenance documentation;
- the preprocessing example YAML;
- published technical documentation navigation; and
- the relevant layered `AGENTS.md` files.

Documentation must state that feature masks affect transform estimation only, raw
outputs are unchanged, physical morphology parameters use micrometers, and runtime
improvement is not guaranteed.

## Acceptance criteria

The implementation is complete when:

1. omission preserves current preprocessing behavior and restart identity;
2. an optional strict feature-mask block normalizes to explicit physical-unit
   values;
3. full-volume deterministic masks follow the approved threshold, component, and
   dilation algorithm;
4. TRSAA and every nonlinear path consume the feature masks at all stages;
5. raw diagnostics, registered images, and dependent channels remain raw-derived;
6. coarse and fine execution share identical fixed-grid mask artifacts;
7. plans, cache identity, provenance, validation, status, examples, and
   documentation agree exactly;
8. portable unit and integration gates pass; and
9. scientific characterization is clearly reported without prematurely changing
   the default.
