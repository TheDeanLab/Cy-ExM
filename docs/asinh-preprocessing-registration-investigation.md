# Asinh preprocessing registration investigation

:::{warning}
This is a dataset-specific investigation of an observed registration failure on
Cy-ExM commit `8c7c127b5d3ffa985d6e5db1028735fa490f25b9`. It does not establish
a new preprocessing, masking, seeding, or registration default. Registration
preprocessing should remain experimental until the affine-stage instability
described below is resolved and scientifically characterized.
:::

## Conclusion

The intended feature/raw separation is implemented correctly. Cy-ExM estimates
the affine and nonlinear transforms from the background-subtracted asinh feature,
then applies those transforms to the cropped raw registration channel and the
original raw dependent channels. The investigation found no evidence that feature
intensities leak into public TIFF outputs, that the affine is applied twice, or
that the configured ZYX sigma is forwarded in the wrong axis order.

The observed loss of accuracy begins during feature-based TRSAA affine
registration, before nonlinear registration or raw-image reconstruction. The
strongest current explanation is an unstable Mattes affine objective caused by
the interaction of:

- a zero-inflated feature representation;
- masks derived from raw images that are positive across the entire field;
- crop-before-filter boundary effects, especially along the shallow Z axis;
- a 12-voxel Z sigma that is much broader physically than the XY sigmas; and
- stochastic 50% affine sampling without a production-bound random seed.

The completed asinh run also used low affine and nonlinear schedules, whereas
the principal unprocessed comparison used high schedules. That is an important
experimental confound, but partial high-accuracy asinh results show that schedule
selection alone does not explain the failure.

## Verified execution sequence

The implementation and its completion evidence establish the following sequence:

1. Load and crop the raw fixed and moving registration channels.
2. Construct fixed and moving masks from cropped raw values greater than zero.
3. Estimate a reflective Gaussian background independently for each cropped
   image.
4. Calculate a deterministic `1.4826 * MAD` scale from the signed residual,
   clip negative residuals to zero, and evaluate `asinh(residual / scale)` as
   float32.
5. Estimate the TRSAA affine from the fixed and moving features.
6. Apply the affine once to the cropped raw moving registration channel.
7. Estimate each nonlinear chunk warp from the affinely registered features.
8. Assemble the full displacement field and apply it once to the affinely
   registered raw moving channel.
9. Apply the accepted forward `[warp, affine]` chain to each original raw
   dependent channel.

Raw images also own the pre-registration, affine, and nonlinear diagnostic
metrics. The public registered TIFFs retain raw-scale intensities: observed
registration-channel outputs reached thousands of intensity units, whereas the
asinh feature was approximately bounded between zero and six.

Two raw registration-channel outputs are produced through related but distinct
resampling paths. The diagnostic `nonlinear_registered_*.tif` is reconstructed
sequentially from the cropped raw registration channel, while the declared channel
output applies the complete transform chain to the original native channel.
They need not be pixel-identical because of cropping and one-pass versus
two-pass interpolation. The same distinction exists without preprocessing and
does not explain the asinh regression.

## Stage at which accuracy deteriorates

The table reports the workflow's raw-image Correlation metric; higher is better.
The unprocessed comparison uses high TRSAA and high CC SyN with radius 4. The
completed asinh run uses low TRSAA and low CC SyN with radius 4.

| Round | Unprocessed high, affine | Asinh low, affine | Asinh low, nonlinear |
| --- | ---: | ---: | ---: |
| 2 | 0.5122 | 0.3708 | 0.3988 |
| 3 | 0.3887 | 0.2648 | 0.2804 |
| 4 | 0.4045 | 0.00036 | 0.00033 |

The degradation is already substantial after TRSAA. Nonlinear registration makes
only a small improvement in rounds 2 and 3 and cannot recover round 4. The failed
round-4 affine has a determinant of approximately 0.69 and markedly contracts Z;
the successful unprocessed comparison has a determinant of approximately 1.24.

### Partial high-accuracy evidence

Two high-accuracy asinh workflows used the same TRSAA configuration, byte-identical
fixed-reference data, the same moving inputs, and one ITK and one BLAS-family
thread. Their only planned scientific difference occurs later, in the nonlinear
metric. Their affine completion requests are otherwise identical after accounting
for the separate fixed-reference output paths.

| Round | Asinh high, Mattes-workflow affine | Asinh high, CC-workflow affine |
| --- | ---: | ---: |
| 2 | 0.3416 | 0.00359 |
| 3 | 0.2666 | 0.2638 |
| 4 | 0.00113 | 0.4111 |

These are partial results rather than completed nonlinear comparisons. They show
that scientifically equivalent affine requests can enter very different basins on
the asinh representation. The nonlinear metric cannot cause this difference
because it is not used by TRSAA.

## Factors that plausibly destabilize TRSAA

### Zero-inflated features under full-field masks

All four raw registration inputs contained no zero-valued voxels:

| Registration input | Minimum raw value | Zero-valued voxels |
| --- | ---: | ---: |
| Round 1, 488-nm tubulin | 49 | 0 |
| Round 2, 488-nm tubulin | 42 | 0 |
| Round 3, 488-nm tubulin | 52 | 0 |
| Round 4, 561-nm tubulin | 79 | 0 |

The raw-derived greater-than-zero masks therefore cover the complete cropped
volumes. In contrast, 63.51% of the voxels in the preliminary full-volume round-2
asinh feature are exactly zero. Mattes affine registration consequently samples
many zero feature values even though all of those locations are admitted by the
mask. The objective can reward alignment of a large, weakly informative
zero-valued population rather than alignment of tubulin structure.

A feature-derived mask is not automatically the correct remedy: a strict
greater-than-zero feature mask would itself be sparse and could differ between
rounds. Mask design requires separate scientific characterization.

### Crop-before-filter boundary effects

The preliminary round-2 feature was generated from the full 36-slice image.
Production first crops that image to Z indices `[6, 33)`, leaving 27 slices, and
then applies the Gaussian. The other cropped depths are 30 and 37 slices; the
fixed reference has 51 slices.

SciPy's default Gaussian support is approximately four sigmas. At sigma Z=12, the
effective support radius is therefore approximately 48 voxels, which is as large
as or larger than every cropped Z extent. The reflective crop boundary influences
the background estimate throughout each stack, and the effect differs because the
fixed and moving crops have different depths and origins.

For a representative interior round-2 XY patch, comparing the preliminary
full-volume-first feature with the production crop-first calculation gave:

| Measurement | Full-volume first | Crop first |
| --- | ---: | ---: |
| Zero-valued feature fraction | 57.24% | 66.09% |
| Robust scale | 15.46 | 20.33 |

The two features had an all-voxel correlation of 0.962, but 8.85% of voxels
changed zero/nonzero state. Per-plane correlation fell to approximately 0.75 and
0.82 at the two crop boundaries. The implemented formula matches the preliminary
formula; the difference arises from when spatial support is removed.

### Voxel-isotropic but physically anisotropic filtering

The configured sigma `[12, 12, 12]` is in NumPy ZYX voxel units. With voxel
spacing `(x, y, z) = (0.324, 0.324, 0.900)` micrometers, its physical Gaussian
width is:

| Axis | Voxel sigma | Physical sigma |
| --- | ---: | ---: |
| Z | 12 | 10.800 micrometers |
| Y | 12 | 3.888 micrometers |
| X | 12 | 3.888 micrometers |

The Z filter is therefore approximately 2.8 times broader physically than either
XY filter. A Z sigma near 4.3 voxels would have approximately the same physical
width as a 12-voxel XY sigma, but that value is only a diagnostic candidate and
has not been accepted.

### Stochastic affine sampling

TRSAA uses Mattes mutual information with 32 bins and a 50% regular sampling rate.
The production schema and completion provenance do not bind an ANTs random seed.
Separate runs can therefore produce different affine estimates under the same
scientific request. The high-accuracy results above demonstrate materially
different outcomes under otherwise equivalent affine conditions.

Stochasticity is not unique to preprocessing: earlier unprocessed radius sweeps
also showed some round-4 affine variation. The asinh feature appears to make the
objective substantially more fragile, because catastrophic variation also occurs
in rounds 2 and 3.

### Brightness balance is not registration information

The preliminary feature review ranked representations from one moving image using
maximum-intensity-projection bright/dim balance and edge retention. It did not test:

- fixed-versus-moving feature consistency;
- three-dimensional affine convergence;
- crop-before-filter boundary effects;
- Z-axis structure;
- interaction with Mattes sampling and raw-derived masks; or
- replicate stability.

Asinh compresses intensity globally but does not equalize local signal-to-noise or
the spatial density of informative structures. Local CC already normalizes mean
and variance within each neighborhood, so global robust scaling contributes
little to CC. Negative-residual clipping can also discard positional information
and introduce a large zero-valued population.

## Separate workflow-status defect

The investigation identified an operational defect unrelated to registration
accuracy. `cyexm workflow status` parses preprocessing from the persisted scientific
plan but omits it when reconstructing each `RegistrationSpec`. It therefore compares
preprocessing-enabled completion records against preprocessing-disabled requests.

On the completed asinh workflow, status correctly recognized the reference round
but falsely reported moving rounds 2--4 and all their accepted artifacts as
`invalid`. Actual execution uses the planner's preprocessing-enabled round tasks,
so this defect does not alter transform estimation or exported data. It can,
however, mislead restart and operator inspection and should be corrected separately.

## Recommended characterization sequence

The next scientific experiment should stop after affine registration so that
nonlinear runtime and variation do not obscure the failure:

1. Compare raw and asinh TRSAA at the same high schedule.
2. Use one explicit seed for paired comparisons, then repeat across several seeds
   to measure basin stability.
3. Compare crop-first preprocessing with native-volume preprocessing or a
   sufficiently padded crop.
4. Characterize physically narrower Z sigmas, including values near 4--5 voxels.
5. Evaluate candidate masks from three-dimensional evidence rather than adopting
   a strict feature-positive mask by assumption.
6. Measure global and bright/dim regional affine residuals before allowing SyN to
   run.

Only after affine registration is consistently correct should CC-radius-4 SyN be
used to evaluate the nonlinear effect of preprocessing. Any promoted configuration
must preserve raw-output reconstruction, dependent-channel transformation, exact
fixed-grid geometry, and the existing transform-order and interpolation invariants.

## Verification scope

The investigation ran 23 targeted tests covering:

- YAML normalization and scientific-plan identity;
- the exact Gaussian-background/MAD/asinh formula;
- ZYX sigma forwarding;
- feature-based transform estimation with raw-derived masks and metrics;
- raw affine and nonlinear reconstruction;
- serial chunk feature estimation and raw output;
- real-ANTs raw TIFF persistence and restart reuse.

All 23 tests passed. These tests verify implementation plumbing and numerical
formulae; they do not establish scientific registration quality. No source,
configuration, dataset, or output artifacts were changed during the investigation.
