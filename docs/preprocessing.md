# Registration preprocessing

## When preprocessing is disabled

Omit `registration.preprocessing` to retain the raw-intensity registration workflow.
Cy-ExM then uses the existing raw greater-than-zero diagnostic masks. Omitting this
mapping is different from supplying an empty mapping.

## Enable background-subtracted asinh features

An empty mapping enables the canonical 12-voxel-per-axis filter. A scalar and an
isotropic three-value list describe the same filter; an anisotropic list is in NumPy
ZYX order:

```yaml
# Disabled: omit registration.preprocessing entirely.

# Enabled with the canonical [12, 12, 12] voxel default:
preprocessing: {}

# These are scientifically identical:
preprocessing: {filter_sigma_zyx: 12}
preprocessing: {filter_sigma_zyx: [12, 12, 12]}

# Anisotropic NumPy ZYX voxel sigmas:
preprocessing: {filter_sigma_zyx: [4, 12, 12]}
```

Cy-ExM estimates a reflective-boundary Gaussian background, subtracts it from the
raw image, and deterministically samples the signed residual to calculate a
MAD-based scale. It clips negative residuals to zero, then computes float32
background-subtracted asinh features. Registration estimates transforms from those
features.

## Units and anisotropic data

`filter_sigma_zyx` is a Gaussian sigma in NumPy ZYX voxel counts, not micrometers.
A scalar expands to the same count on every axis. Therefore an equal voxel sigma is
not necessarily physically isotropic when voxel spacing is anisotropic.

| Setting | Unit and order | Purpose |
| --- | --- | --- |
| `filter_sigma_zyx` | NumPy ZYX voxel counts | Gaussian background-filter sigma |
| `min_component_volume_um3` | Cubic micrometers | Minimum retained feature-mask component |
| `dilation_radius_um` | Micrometers | Physical feature-mask dilation radius |

The optional mask morphology uses physical micrometers and cubic micrometers instead;
see [Optional feature-derived masks](#optional-feature-derived-masks).

## What preprocessing changes

The processing boundary is `raw -> crop -> feature estimate -> transform raw`.
TRSAA and nonlinear registration use the features to estimate transforms. Changing a
sigma invalidates feature-dependent affine work, registration metrics, local chunks,
prepared arrays, the assembled warp, and registered output.

## What preprocessing does not change

Preprocessing does not replace the raw diagnostic masks by default. It does not save
a normalized feature TIFF. The accepted transforms are applied to the raw
registration image and dependent-channel data. Transformed channel TIFFs remain
raw-intensity products; the separately named registration diagnostic retains its
established diagnostic postprocessing. Feature images are internal evidence, not
public image exports.

Large geometric offsets are handled separately by per-round
[`initial_alignment.xy_rotation` or `initial_alignment.xy_similarity`](initial-alignment.md).
Geometric initialization is resolved from raw cropped registration-channel MIPs
before feature construction; it is not an intensity-preprocessing option.

## Optional feature-derived masks

To use a feature-derived registration mask, nest `feature_mask` under enabled
preprocessing:

```yaml
registration:
  preprocessing:
    filter_sigma_zyx: [12, 12, 12]
    feature_mask:
      min_component_volume_um3: 1.0
      dilation_radius_um: 1.0
```

When present, this option uses strict positive asinh support for registration
metrics. It does not replace raw diagnostic masks or raw-intensity outputs. The
fixed and moving masks are derived independently on complete cropped images; the
moving mask is affine-resampled with nearest-neighbor interpolation and thresholded
on the fixed grid. `feature_mask: {}` enables the provisional physical morphology
defaults. The focused [feature-derived registration mask guide](feature-derived-registration-masks.md)
explains the algorithm and interpretation.

## Cache and restart effects

Preprocessing-disabled prepared bundles retain strict format version 2. Enabled
features without masking use format version 3; feature-derived registration masks use
format version 4, which also records the mask algorithm, physical morphology, and
realized foreground evidence. Those records let Cy-ExM invalidate only work whose
request depends on the changed policy. Source identities, authenticated crop bounds,
fixed-reference TIFFs, and reference-channel materialization remain independently
reusable when their own requests still match.

Rotation-enabled equivalents use strict formats 5, 6, and 7 respectively and bind
the initialization JSON, transform, QC, and complete affine identities.

## Evaluate a dataset-specific policy

Feature preprocessing and feature-derived masks are scientific choices, not a
guarantee of faster or more accurate registration. Compare preprocessing disabled,
asinh features without masking, and masking while keeping the schedule, metric,
radius, chunk geometry, overlap, threads, and inputs constant. Evaluate bright and
dim regions, deformation validity, seams, and stage runtime before adopting a
dataset-specific policy. The [asinh preprocessing investigation](asinh-preprocessing-registration-investigation.md)
and [feature-derived registration mask guide](feature-derived-registration-masks.md)
provide the supporting detail.
