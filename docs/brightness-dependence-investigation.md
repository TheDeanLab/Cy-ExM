# Reducing brightness dependence in nonlinear registration

## Conclusion

The observed brightness dependence can likely be reduced substantially in software,
although it cannot be completely eliminated where the registration channel contains
no corresponding structure. The strongest immediate change is to retain Mattes
mutual information for affine registration and use local cross-correlation for
nonlinear SyN.

The underlying problem is better described as **local information dependence** than
raw brightness dependence. In low-signal areas, Mattes has a relatively weak or
noisy optimization gradient, so SyN's regularization lets brighter structures
elsewhere in the chunk dominate the deformation.

## Evidence from the spheroid dataset

All tested representations recover essentially the same residual displacement in
the dim region of interest. This confirms that the offset is real and that usable
structural information remains there.

| Round | Approximate correction `(dy, dx)` | Raw NCC current -> corrected | Gradient NCC current -> corrected |
| --- | ---: | ---: | ---: |
| 2 | `(+7, -8)` px | 0.807 -> 0.907 | 0.557 -> 0.797 |
| 3 | `(+16, -9)` px | 0.656 -> 0.883 | 0.433 -> 0.747 |

The gradient representation made the correct match approximately 1.4--2.4 times
more distinguishable than raw intensity. Log compression, contrast-limited adaptive
histogram equalization (CLAHE), and local normalization found the same displacement
but generally offered much smaller improvements.

The existing greater-than-zero mask is also ineffective at balancing this dataset:
every voxel in the fixed reference is positive, so its metric mask covers the entire
volume.

## Recommended approach

### 1. Mattes affine followed by CC SyN

Keep the current TRSAA/Mattes affine stage and use the now-configurable nonlinear
metric to test neighborhood cross-correlation (CC) across several radii.

CC computes correlation over local neighborhoods and normalizes local mean and
contrast. That directly reduces sensitivity to local brightness scaling. ANTs
specifically recommends CC for deformable registration of correlated, same-modality
images and notes that larger neighborhoods may help low-information regions. The
established ANTs pattern is also mutual-information affine registration followed by
CC deformable registration. See the
[ANTs guidance](https://github.com/ANTsX/ANTs/wiki/Tips-for-improving-registration-results),
[ANTs SyN metric explanation](https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call),
and [Avants et al. evaluation](https://pubmed.ncbi.nlm.nih.gov/20851191/).

This dataset is well suited to CC because tubulin is the common marker in every
round. Radius selection needs testing because Z is anisotropic: a voxel-radius
neighborhood covers a larger physical distance in Z than in XY.

Cy-ExM now carries `metric: CC` and its effective scalar radius through schema-v1
validation, canonical planning, cache identity, provenance, status reconstruction,
and ANTs execution. Omission uses the provisional radius 4. ANTs does not expose an
anisotropic per-axis CC radius: a scalar radius 4 spans 9 voxels per axis, so an
anisotropic voxel grid produces a physically anisotropic neighborhood. See the
[ANTsPy registration API](https://antspy.readthedocs.io/en/stable/registration.html).

### 2. Add a structure-aware secondary metric

If CC alone does not sufficiently correct the periphery, combine:

- Raw tubulin with local CC.
- A gradient or Laplacian-of-Gaussian (LoG) representation of tubulin with a second
  CC metric.

ANTSpyX 0.6.3 already supports `multivariate_extras` for `SyNOnly`, so both metrics
can estimate one diffeomorphic transform. The derived representation should only
guide transform estimation; the resulting transform would still be applied to the
original intensities and dependent channels.

This is especially promising because gradient NCC provided the strongest local
discrimination in the observed region. ASHLAR similarly uses Laplacian or LoG
filtering before phase correlation to improve alignment confidence in multiplexed
fluorescence images. See the
[ASHLAR paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9525007/).

Estimated implementation difficulty is moderate to high, approximately three to
five focused development days, because derived-image geometry, preprocessing
parameters, memory use, provenance, and scientific regression tests must be
carefully controlled.

### 3. Add spatially resolved registration quality control

Global correlation should not be the primary acceptance criterion. Add a
three-dimensional blockwise residual map, stratified by local signal or structural
confidence, that reports:

- Median and 95th-percentile residual by signal quartile.
- Explicit values for known dim and bright regions.
- Seam-adjacent versus chunk-interior results.
- Warp Jacobian and folding checks.
- Confidence or peak prominence for each block.
- Runtime and global similarity as secondary measurements.

A practical initial target would be low-signal residuals below two pixels while
keeping the bright-center residual below one pixel and introducing no folding or
seam degradation. These thresholds remain proposed acceptance criteria and require
validation on the approved cropped dataset before adoption.

### 4. Use a dense registration label in future acquisitions

The most reliable way to eliminate the dependency is a persistent, densely
distributed reference label acquired in every round, ideally using the same
fluorophore and acquisition settings. NHS-ester pan-protein staining or another
dense biological structure such as actin has been shown to outperform sparse labels
specifically in regions lacking reference signal in cyclic expansion imaging. See
the [dense-label cyclic ExM study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12225881/).

Round 4 also changes tubulin from 488 to 561. That is not the root cause because
rounds 2 and 3 show the same effect, but using the same wavelength across rounds
would reduce chromatic and photometric differences.

## Lower-priority options

- **Flat-field or N4 correction:** useful only if a smooth illumination or bias field
  is first demonstrated. It cannot recover absent structure. See the
  [ANTs N4 documentation](https://github.com/ANTsX/ANTs/wiki/N4BiasFieldCorrection).
- **Smaller chunks:** could give dim structures more relative influence, but may
  remove needed context and increase instability. Test this only after changing the
  metric.
- **Greater overlap:** unlikely to solve the observed region because it lies inside
  a chunk core, away from a seam.
- **More iterations or reduced smoothing:** unlikely to help when the metric lacks a
  strong local gradient and could instead fit noise.
- **MIND or normalized-gradient descriptors:** promising if cross-round intensity
  relationships become unreliable, but they are not native ANTs metrics and would
  require substantially more engineering. See the
  [MIND paper](https://ora.ox.ac.uk/objects/uuid%3A90c69bfd-a3b1-4868-93b4-ccc4375c468d).
- **Arbitrary dependent-channel fusion:** scientifically unsafe because those
  channels label different biology in different rounds.

## Proposed characterization experiment

Before changing chunk geometry or deformation regularization, compare the following
on the approved cropped data:

1. The current Mattes SyN baseline.
2. CC SyN with radius 2.
3. CC SyN with radius 4.
4. CC SyN with radius 6 or 8 to bound the useful neighborhood scale.
5. The best CC configuration plus a low-weight gradient or LoG metric.

Only promote a configuration that improves spatially resolved low-signal residuals,
preserves bright-region alignment and global similarity, produces a physically
valid warp without folding, preserves exact Cy-ExM geometry and transform
invariants, and has acceptable runtime.
