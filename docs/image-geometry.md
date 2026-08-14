# Image geometry, voxel spacing, and axes

## Fixed and moving images

The fixed image is the reference that defines the physical destination of the
registration. A moving image is registered to that reference; these roles do not
change. Native fixed, moving registration-channel, and dependent-channel images may
have different shapes along Z, Y, and X.

Every declared reference-round channel, however, must have the same native geometry:
shape, origin, spacing, and direction. Cy-ExM copies each source into the shared
reference `channels` directory without moving the source. When cropping is enabled,
it estimates one crop from the reference registration channel and applies the same
bounds to every declared reference channel.

## Physical XYZ and array ZYX

Public voxel spacing is in micrometers and is written `(x, y, z)`. NumPy arrays and
TIFF volumes use `(z, y, x)`. For example, a public spacing of `(0.15, 0.15, 0.40)`
describes X, Y, then Z, while the first array dimension is Z.

| Quantity | Order | Meaning |
| --- | --- | --- |
| Public voxel spacing | `(x, y, z)` | Physical micrometers along X, Y, and Z |
| NumPy and TIFF volume | `(z, y, x)` | Array dimensions, with Z first |
| Origin and direction | `(x, y, z)` | Physical XYZ geometry |

ANTs indexes image coordinates in physical XYZ order. Cy-ExM changes between that
representation and external arrays at its orchestration boundary. The first three
spatial axes are transposed; a vector image keeps its final vector-component axis,
and those components remain `(x, y, z)` physical components. `ImageGeometry.shape`
is external ZYX, while origin, spacing, and direction are physical XYZ.

## The fixed-reference grid

The affine application is the first fixed-reference grid boundary. After it, the
affine scalar and mask outputs, nonlinear inputs and outputs, full warp, and every
transformed dependent channel match the fixed-reference geometry. This is why
independent native input shapes are permitted without redefining the fixed image.

Cy-ExM applies the affine before the nonlinear warp. Scalar images use linear
interpolation; masks use nearest-neighbor interpolation. These choices preserve
intensity and mask meaning while placing results on the fixed-reference grid.

## Accepted TIFF inputs

Provide a scalar 3-D TIFF and caller-supplied XYZ voxel spacing. A plain TIFF is
accepted only when it is an explicit ZYX volume, a grayscale QYX stack, or
`tifffile`'s reconstructed YX view of a declared leading singleton Z axis. Sample or
channel layouts such as YXS, SYX, and CYX are rejected rather than guessed.

Plain TIFF inputs use zero origin and identity direction. Their spacing comes from
the caller, not from an inferred image layout. Planning also requires manifest-
compatible physical spacing for each scalar 3-D TIFF.

## Cy-ExM geometry metadata

Cy-ExM-generated TIFFs contain strict, versioned shape, origin, spacing, and
direction metadata. The writer verifies that serialized metadata matches the source
geometry before promoting an output. If a TIFF has no Cy-ExM marker, the explicit
plain-TIFF rules above apply. A present but malformed, including truncated, Cy-ExM
payload fails closed rather than falling back to guessed geometry.

The exact metadata payload, completion records, and artifact-validation rules are in
the [registration artifact reference](registration-artifacts.md).

## Common mistakes

- Do not provide spacing in array order; spacing is always `(x, y, z)`.
- Do not treat a native moving shape as the output shape after affine application.
- Do not use a multi-sample TIFF layout as though it were a scalar volume.
- Do not exchange masks and scalar images: their interpolation rules differ.
- Do not estimate a separate reference crop for each declared reference channel.

## Related reference material

For choosing whole-image or serial-chunked registration, read
[Choosing a registration strategy](registration-foundations.md). For exact TIFF
metadata and restart validation, read the
[registration artifact reference](registration-artifacts.md).
