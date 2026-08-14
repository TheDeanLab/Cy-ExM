# Feature-derived registration masks

Background-subtracted asinh features can contain a large zero-valued population.
Those locations contain no positive residual structure, but the ordinary raw
greater-than-zero mask may still admit them to the ANTs similarity metric. Cy-ExM
therefore provides an optional feature-derived registration mask.

This option changes only where transforms are estimated. It never thresholds the raw
registration channel, changes a dependent channel, or replaces public output with a
feature image.

For the preprocessing choices, raw-output boundary, cache effects, and dataset-level
evaluation, start with the [Preprocessing overview](preprocessing.md).

## Enabling the mask

Feature masking is nested under enabled preprocessing:

```yaml
registration:
  preprocessing:
    filter_sigma_zyx: [12, 12, 12]
    feature_mask:
      min_component_volume_um3: 1.0
      dilation_radius_um: 1.0
```

`feature_mask: {}` is equivalent to the two explicit `1.0` values. Omitting
`feature_mask` preserves the existing preprocessing behavior and uses raw
greater-than-zero masks for registration. Feature masking cannot be enabled without
preprocessing.

The units deliberately differ:

- `filter_sigma_zyx` is a NumPy ZYX voxel count.
- `min_component_volume_um3` is cubic micrometers.
- `dilation_radius_um` is micrometers.

The physical morphology values make one policy interpretable across datasets with
different voxel spacing. The `1.0` defaults are provisional characterization
starting points, not optimized biological constants.

## Algorithm

Cy-ExM constructs each fixed or moving mask independently:

1. Compute the existing background-subtracted, MAD-scaled asinh feature.
2. Select voxels satisfying strict finite `feature > 0`.
3. Label components using full 3D connectivity.
4. Convert the minimum physical volume to a voxel count using the image spacing and
   remove smaller components.
5. Build an ellipsoidal footprint from the physical dilation radius and anisotropic
   spacing.
6. Dilate once and emit an exact float32 binary mask.

Morphology runs on each complete cropped image before nonlinear chunks are
extracted. Chunk boundaries therefore cannot change component labels or dilation.
An empty result fails before ANTs rather than silently falling back to a different
mask.

## Registration and output flow

```text
cropped raw fixed/moving
    ├──> raw > 0 masks ----------------> diagnostic metrics
    └──> asinh features
              └──> optional feature masks
                         ├──> every TRSAA metric stage
                         └──> every SyN metric stage

accepted affine and warp -------------> raw registration channel
                                    └--> raw dependent channels
```

The moving feature mask follows the affine with nearest-neighbor interpolation and
is thresholded at 0.5 on the fixed grid. Serial and fine-grained nonlinear tasks
extract their masks from the same full fixed-grid arrays. Scalar interpolation and
the warp-then-affine output transform order do not change.

Prepared fine-grained bundles use:

- format 2 when preprocessing is disabled;
- format 3 for asinh preprocessing without feature masking; and
- format 4 when feature masking is enabled.

Format 4 binds the physical policy, fixed and moving foreground counts and
fractions, exact binary-mask policy, and array identities. A policy change
invalidates affine and transform-dependent descendants, while independent reference
preparation and crop evidence remain reusable.

## Scientific interpretation

Masking may reduce uninformative metric participation, but it does not make ANTs
sparse. Image pyramids and deformation fields remain dense, so a substantial runtime
reduction is not guaranteed.

Treat this option as an experimental scientific choice. Compare preprocessing
disabled, asinh without feature masking, and asinh with feature masking while holding
the registration schedule, metric, CC radius, chunk geometry, overlap, threads, and
inputs constant. Evaluate bright and dim regions separately, along with deformation
validity, seams, and stage runtime, before adopting a dataset-specific policy.
