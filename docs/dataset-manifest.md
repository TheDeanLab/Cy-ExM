# Dataset manifest reference

## Purpose and scientific identity

A scientific dataset manifest records the input rounds and channels, physical voxel
spacing, registration choices, and registration strategy. It is deliberately
separate from the machine-specific execution profile, which supplies dataset and
output roots, threads, executor, and resources. The manifest image paths below are
relative to the execution profile's `paths.dataset_root`; callers provide the image
files and edit the paths before validation.

Every mapping has exact keys. Required scientific selections are explicit; the only
nonlinear-metric normalization is CC's documented optional radius default.

## Complete schema

| Location | Type and accepted values | Meaning |
| --- | --- | --- |
| `schema_version` | built-in integer `1` | Selects the strict manifest contract. |
| `dataset_id` | string matching `[A-Za-z0-9][A-Za-z0-9._-]*` | Becomes the dataset output-directory name. |
| `voxel_spacing_um` / `x`, `y`, `z` | three finite positive numbers | Physical voxel spacing in micrometers in XYZ order. |
| `reference` / `round` | nonnegative built-in integer | Fixed-reference imaging round. |
| `reference` / `registration_channel` | safe relative POSIX path | Fixed image used for every moving-round registration; it must occur exactly once in `reference.channels`. |
| `reference` / `channels` | nonempty ordered list with unique `file` values and basenames | Every reference-round image copied to the shared reference output in declaration order. |
| reference channel `file` | safe relative POSIX path | Source below `paths.dataset_root`; its basename becomes the reference channel output name. |
| `registration` / `crop` | built-in boolean | Enables or disables fixed/moving foreground cropping. |
| `registration` / `preprocessing` | optional mapping | Omission disables registration-feature preprocessing; `{}` enables the canonical 12-voxel-per-axis default. |
| `registration` / `preprocessing` / `filter_sigma_zyx` | positive number or three positive numbers | Gaussian background-filter sigma in NumPy ZYX voxel units. A scalar expands to all axes; this is not micrometers. |
| `registration` / `preprocessing` / `feature_mask` | optional mapping | Omission retains raw greater-than-zero registration masks; `{}` enables positive-feature support with the provisional physical morphology defaults. |
| `feature_mask` / `min_component_volume_um3` | positive number | Minimum connected-component volume in cubic micrometers; default `1.0`. |
| `feature_mask` / `dilation_radius_um` | positive number | Physical Euclidean dilation radius in micrometers; default `1.0`. |
| `registration` / `linear` / `type` | `Translation`, `Rigid`, `Similarity`, `Affine`, or `TRSAA` | Registration-library linear method. |
| `registration` / `linear` / `accuracy` | `high`, `medium`, `low`, or `dry run` | Registration-library linear schedule. |
| `registration` / `nonlinear` / `type` | `Elastic`, `SyNOnly`, or `SyN` | Registration-library nonlinear method. |
| `registration` / `nonlinear` / `accuracy` | `high`, `medium`, `low`, or `dry run` | Registration-library nonlinear schedule. |
| `registration` / `nonlinear` / `metric` | exact `CC` or `mattes` | Required nonlinear similarity metric. Affine registration remains Mattes/32 independently. |
| `registration` / `nonlinear` / `radius` | positive built-in integer | Optional only for `CC`; omission normalizes to the provisional default 4. Rejected for `mattes`, which uses 32 bins. |
| `rounds` | non-empty list | Moving-round objects; moving rounds are unique and cannot equal the reference round. |
| moving `round` | nonnegative built-in integer | Round identifier used in task and output names. |
| moving `initial_alignment` | optional exact mapping | Per-round geometric seed; omission preserves the historical workflow. |
| `initial_alignment` / `xy_rotation` / `mode` | `explicit` or `estimate` | Selects a reviewed angle or target-blind registration-channel estimate. |
| explicit `angle_deg` | finite number in `[-180, 180]` | Physical +Z right-hand correction; positive appears clockwise in an identity-direction row-down display. |
| estimate `search_range_deg` | two increasing finite numbers within `[-180, 180]`; default `[-45, 45]` | Inclusive physical-angle search interval. |
| estimate `coarse_step_deg` | positive finite number; default `1.0` | Coarse search spacing in degrees. |
| estimate `refinement_step_deg` | positive finite number smaller than coarse step; default `0.1` | Local refinement spacing in degrees. |
| estimate `minimum_score_gain` | positive finite number; default `0.01` | Required improvement over zero for a nonzero estimate. |
| estimate `minimum_peak_margin` | positive finite number; default `0.01` | Required separation from competing angles. |
| `strategy` | `whole_image` or `serial_chunked` | Selects the registration implementation. |
| `registration_channel` | safe relative POSIX path | The moving image used to estimate transforms. The registration channel must be listed in that round's `channels`. |
| `channels` | list with unique `file` values | Original moving channels resampled through the round transform, in manifest order. |
| channel `file` | safe relative POSIX path | Source below `paths.dataset_root`; its basename becomes the channel output name. |
| channel `interpolation` | `linear` or `nearestNeighbor` | Explicit scalar interpolation for that channel. |
| `chunk_size_zyx` | three positive built-in integers | Required only for `serial_chunked`; chunk size in NumPy ZYX order. Forbidden for `whole_image`. |
| `overlap_fraction` | finite number satisfying `0 <= value < 1` | Required only for `serial_chunked`; forbidden for `whole_image`. |

## Paths and channel rules

Scientific image paths must be safe relative POSIX paths. Absolute paths, empty or
`.` components, `..`, NULs, newlines, and paths that resolve outside the execution
profile's dataset root are rejected. Round numbers are built-in integers rather than
booleans, channel files are unique within a round, and the registration channel must
match a channel entry exactly.

The reference registration channel must occur exactly once in `reference.channels`.
Reference-channel `file` values and their output basenames must both be unique. The
reference task copies each declared source into its shared channels directory; it
never moves or modifies the source file.

Moving-channel output basenames must be unique within each round as well. Distinct
relative paths such as `nested-a/same.tif` and `nested-b/same.tif` have different
`file` values but both map to `channels/same.tif`; planning rejects that output
collision before execution.

## Reference and moving geometry

All declared reference channels must have identical native shape, origin, spacing,
and direction because they were acquired together. If cropping is enabled, Cy-ExM
estimates one crop from the reference registration channel and applies those
scientifically identical bounds to every reference channel.

Moving registration and dependent-channel images may differ from the reference along
any spatial axis. The registration library resamples them onto the fixed reference
grid before nonlinear chunking and before final channel export. Planning requires
every input to be a scalar 3-D TIFF with manifest-compatible physical spacing. See
[Image geometry, voxel spacing, and axes](image-geometry.md) for the physical and
array-coordinate interpretation.

Schema version 1 intentionally uses only `reference.registration_channel` and
`reference.channels`. The retired singular `reference.channel` form is rejected;
there is no dual parser or compatibility schema to maintain.

## Registration strategies

`strategy` selects `whole_image` or `serial_chunked`. The latter requires both
`chunk_size_zyx` and `overlap_fraction`; the former forbids them. `chunk_size_zyx`
uses NumPy ZYX order, while `voxel_spacing_um` remains physical XYZ spacing in
micrometers. See [Choosing a registration strategy](registration-foundations.md)
before choosing either value.

## Preprocessing fields

Omit `registration.preprocessing` to preserve the intensity workflow. An empty
mapping enables the canonical `[12, 12, 12]` voxel default. `filter_sigma_zyx` may
be a positive scalar or a three-value positive NumPy ZYX list; it is in voxel units,
not micrometers. An optional `feature_mask` mapping accepts
`min_component_volume_um3` and `dilation_radius_um`, both positive physical values
whose defaults are `1.0`.

```yaml
# Disabled: omit registration.preprocessing entirely.

# Enabled with the canonical [12, 12, 12] voxel default:
preprocessing: {}

# These are scientifically identical:
preprocessing: {filter_sigma_zyx: 12}
preprocessing: {filter_sigma_zyx: [12, 12, 12]}

# Anisotropic NumPy ZYX voxel sigmas:
preprocessing: {filter_sigma_zyx: [4, 12, 12]}

# Optional feature-derived registration masks:
preprocessing:
  filter_sigma_zyx: [12, 12, 12]
  feature_mask: {}  # provisional 1.0 um3 minimum and 1.0 um radius
```

For the preprocessing boundary, mask behavior, and dataset-specific interpretation,
see [Registration preprocessing](preprocessing.md) and
[Feature-derived registration masks](feature-derived-registration-masks.md).
For geometric initialization, its physical angle convention, confidence failures,
and restart artifacts, see [XY rotation initial alignment](initial-alignment.md).

## Nonlinear metrics and CC radius

`registration.nonlinear.metric` is required and accepts exact `CC` or `mattes`.
Affine registration remains Mattes/32 independently. `radius` is optional only for
`CC`; it must be one positive built-in integer. Omission normalizes to the
provisional value 4. `mattes` rejects `radius` and uses 32 bins.

The registration library exposes CC radius as one scalar voxel count, not
independent per-axis radii. Radius 4 therefore means a 9-by-9-by-9 voxel
neighborhood. With anisotropic voxel spacing that neighborhood is physically
anisotropic even though its voxel radius is equal along X, Y, and Z. The default of 4
is provisional pending dataset-specific characterization; it is not a claim of
optimality.

## Strict YAML validation

The loader uses safe YAML decoding and rejects unknown keys, missing keys, duplicate
mapping keys, YAML aliases, custom tags, non-scalar keys, specialized collection
tags, a non-mapping document, invalid UTF-8, and additional YAML documents. These
rules apply to both YAML files.

## Complete example

The checked-in [spheroid scientific dataset manifest](../examples/spheroid-dataset.yaml)
is a complete version-1 example. Its TIFF filenames are placeholders for
caller-supplied images under the dataset root chosen in an execution profile.

```yaml
schema_version: 1
dataset_id: spheroid-registration-example

# Public spacing is in micrometers and XYZ order. NumPy volume axes are ZYX.
voxel_spacing_um:
  x: 0.324
  y: 0.324
  z: 0.900

reference:
  round: 1
  registration_channel: round_1_488_tubulin.tif
  channels:
    - file: round_1_488_tubulin.tif

registration:
  crop: false
  linear:
    type: TRSAA
    accuracy: high
  nonlinear:
    type: SyNOnly
    accuracy: high
    metric: CC
    # Optional scalar voxel radius. Omission uses the provisional default of 4.

rounds:
  - round: 2
    strategy: whole_image
    registration_channel: round_2_488_tubulin.tif
    channels:
      - {file: round_2_488_tubulin.tif, interpolation: linear}
      - {file: round_2_561_bcatenin.tif, interpolation: linear}
      - {file: round_2_647_hsp60.tif, interpolation: linear}

  - round: 3
    strategy: serial_chunked
    registration_channel: round_3_488_tubulin.tif
    # Chunk sizes are in NumPy ZYX order.
    chunk_size_zyx: [512, 512, 512]
    overlap_fraction: 0.15
    channels:
      - {file: round_3_488_tubulin.tif, interpolation: linear}
      - {file: round_3_647_fbl.tif, interpolation: linear}

  - round: 4
    strategy: whole_image
    registration_channel: round_4_561_tubulin.tif
    channels:
      - {file: round_4_488_vimentin.tif, interpolation: linear}
      - {file: round_4_561_tubulin.tif, interpolation: linear}
      - {file: round_4_647_pex14.tif, interpolation: linear}
```
