# Fixed-affine nonlinear chunk-size characterization

:::{important}
This is a dataset-specific characterization of one spheroid workflow. It tests a
mechanistic hypothesis about spatial support in low-signal regions; it does not
establish a universal chunk size or change a Cy-ExM production default.
:::

## Question and controlled design

The preceding [CC SyN radius characterization](cc-syn-radius-characterization.md)
found that radius 4 improved the known dim-region residual, but its independent
affine estimates confounded comparisons among radii. This experiment asks a
narrower question: with the accepted radius-4 affine state held exactly fixed, does
reducing the nonlinear XY chunk size reduce the remaining dim-region displacement?

The experiment reused the completed round-2, round-3, and round-4 affine artifacts
from the radius-4 workflow. Their completion records, input identities, fixed
geometry, TRSAA/high policy, native-thread policy, and bytes were validated before
and after every condition. Cy-ExM cloned those artifacts into a caller-owned shared
preparation, applied each accepted affine once to resample the moving data onto the
fixed grid, and cloned that prepared state into every condition. The TRSAA estimator
was never called during the sweep.

All other scientific and operational choices were held constant:

| Variable | Controlled value |
| --- | --- |
| Moving rounds | 2, 3, and 4 |
| Nonlinear transform | high-accuracy SyNOnly |
| Nonlinear metric | CC, scalar voxel radius 4 |
| Chunk size ZYX | `[512,1125,1125]`, `[512,512,512]`, `[512,256,256]`, or `[512,128,128]` |
| Overlap fraction | 0.15 |
| Fixed geometry ZYX | 51 × 2240 × 2223 voxels |
| Native threads | 30 ITK and 30 BLAS-family threads |
| SLURM resources | four exclusive 512-GB nodes, one task slot per node |

The four conditions contain 27, 108, 363, and 1,323 nonlinear chunk tasks across
the three moving rounds, respectively. The 1125-pixel baseline ran first. The
remaining three conditions later ran concurrently with four dedicated nodes each,
so their wall times include possible shared-filesystem contention. Fine-grained
execution changed task placement only; the production preparation, nonlinear
chunk, assembly, transform, validation, artifact, and cache implementations
remained unchanged.

## Spatial quality measurements

Registration quality was measured on Z maximum-intensity projections using the same
fixed-grid observables as the preceding radius study:

- dim ROI: `y = 1600:1800`, `x = 300:400`;
- bright ROI: `y = 1227:1427`, `x = 840:1040`.

For each ROI, float64 Sobel gradient magnitude was calculated for the fixed and
exported registered channel. Every integer correction in ±30 pixels along Y and X
was evaluated by normalized cross-correlation. The reported residual is the
Euclidean magnitude of the correction applied to the moving image. `current NCC`
is the score at zero correction, `best NCC` is the score at the optimum, and
prominence is the best score minus the best score outside a Chebyshev radius of two
pixels around that optimum. A boundary optimum is a lower bound, not a precise
residual.

As a regression check, this implementation reproduced the previously reported
radius-4 dim residuals exactly: 7.81, 14.32, and 13.45 pixels for rounds 2, 3, and
4. The assembled registration scalar and exported registration channel selected
the same correction in every round. The exported channel was retained as the
primary downstream observable.

The whole field was also sampled with 128 × 128-pixel windows on a 64-pixel stride.
Each window retained its fixed-image signal quartile, fixed-gradient energy,
residual, NCC confidence, and search-boundary status. A window was classified as
seam-adjacent when its center lay within 16 pixels of a visible assembled core
transition. Because later overlapping cores overwrite earlier cores in canonical
chunk order, these transitions occur at later core starts, not at both starts and
ends.

Finally, ANTs computed the geometric Jacobian determinant of each assembled
three-dimensional displacement field on the exact fixed domain. Non-finite values
and determinants less than or equal to zero were counted explicitly; the latter
indicate local folding or singular deformation.

## Results

### The named bright and dim regions

Reducing nonlinear spatial support did not change the selected correction in
either named ROI. Every chunk size produced exactly the same residual for a given
round:

| Round | Bright residual (px) | Dim correction `(dy, dx)` (px) | Dim residual (px) |
| --- | ---: | ---: | ---: |
| 2 | 0.00 | `(6, -5)` | 7.81 |
| 3 | 0.00 | `(14, -3)` | 14.32 |
| 4 | 0.00 | `(9, -10)` | 13.45 |

None of these optima reached the ±30-pixel search boundary. Bright-region
prominence remained high, from 0.340 to 0.378. Dim-region prominence remained much
lower, from 0.013 to 0.030, so the dim correction is less uniquely determined even
though its selected integer displacement is stable across conditions.

The current dim-region NCC changed only slightly. Relative to 1125 pixels, the
128-pixel condition changed current NCC by +0.0071, +0.0045, and +0.0064 for rounds
2, 3, and 4. These small score changes did not reduce the residual displacement.
Whole-image correlation also remained close across conditions:

| XY chunk (px) | Round 2 | Round 3 | Round 4 |
| ---: | ---: | ---: | ---: |
| 1125 | 0.6851 | 0.5420 | 0.5566 |
| 512 | 0.6800 | 0.5497 | 0.5496 |
| 256 | 0.6824 | 0.5472 | 0.5490 |
| 128 | 0.6784 | 0.5421 | 0.5480 |

```{figure} images/fixed-affine-chunk-size/regional-residuals.png
:alt: Bright and dim ROI residuals plotted against nonlinear XY chunk size.

The bright ROI remains at zero residual and the dim ROI retains the same
round-specific displacement at every chunk size.
```

```{figure} images/fixed-affine-chunk-size/regional-overlays.png
:alt: Fixed-green and registered-magenta gradient overlays for every condition, round, and ROI.

Fixed-image gradients are green and registered moving-image gradients are magenta.
The overlays visually reproduce the quantitative result: chunk size does not
remove the dim-region displacement.
```

### Whole-field signal and seam behavior

Across 12,672 block measurements, the overall median residual remained essentially
constant: 11.02, 11.02, 11.18, and 11.05 pixels for 1125-, 512-, 256-, and
128-pixel chunks. Signal strength, rather than chunk size, dominated the result:

| XY chunk (px) | Quartile 1, dimmest | Quartile 2 | Quartile 3 | Quartile 4, brightest |
| ---: | ---: | ---: | ---: | ---: |
| 1125 | 26.40 | 17.80 | 1.00 | 0.00 |
| 512 | 26.17 | 17.80 | 1.00 | 0.00 |
| 256 | 26.41 | 17.80 | 1.00 | 0.00 |
| 128 | 25.50 | 17.69 | 0.00 | 0.00 |

Values are median residuals in pixels. Approximately 9–10% of defined windows had
a search-boundary optimum in every condition, concentrated in the lower-signal
quartiles. Those observations are lower bounds and should not be interpreted as
precise displacement estimates.

Seam-adjacent and interior windows cannot be compared without accounting for their
signal distribution. The raw medians were 2.24 versus 13.04 pixels at 512 pixels,
11.68 versus 10.82 pixels at 256 pixels, and 13.10 versus 8.37 pixels at 128
pixels. The apparently favorable 512-pixel seam value reflects where those seams
cross this specimen, not evidence that seams improve registration. At 128 pixels,
the direction reverses and the seam median is 4.73 pixels worse than the interior
median. Within signal quartiles the differences are smaller, but the 128-pixel
condition still shows no consistent benefit and has a seam penalty in quartiles 2
and 4.

```{figure} images/fixed-affine-chunk-size/seam-interior.png
:alt: Mean block residual near assembled chunk seams and in chunk interiors.

Raw seam/interior means are descriptive and remain confounded by specimen signal
distribution. The smallest chunks do not provide a consistent seam benefit.
```

### Deformation validity

All twelve assembled warp fields contained only finite Jacobian determinants and
zero nonpositive determinants; no folding was detected. Smaller chunks nevertheless
produced progressively more extreme local volume changes:

| XY chunk (px) | Minimum | Lowest round-level 1st percentile | Highest round-level 99th percentile | Maximum | Nonpositive voxels |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1125 | 0.000419 | 0.813 | 1.201 | 1.770 | 0 |
| 512 | 0.000001 | 0.809 | 1.206 | 2.269 | 0 |
| 256 | 0.000003 | 0.795 | 1.221 | 4.982 | 0 |
| 128 | 0.000002 | 0.767 | 1.252 | 6.594 | 0 |

The median determinant remained approximately 1.0 in every round. Thus the
smaller-chunk fields are topologically valid by this gate, but their tails are less
conservative.

```{figure} images/fixed-affine-chunk-size/jacobian-validity.png
:alt: Jacobian determinant percentiles and folding counts by chunk size.

No condition folded, while the deformation-distribution tails widened as chunks
became smaller.
```

### Runtime

Wall time increased as chunks became smaller:

| XY chunk (px) | Chunks | Four-node wall time (h) | Four-node allocation (node-h) | Increase from 1125 |
| ---: | ---: | ---: | ---: | ---: |
| 1125 | 27 | 3.53 | 14.11 | — |
| 512 | 108 | 3.83 | 15.29 | 8.4% |
| 256 | 363 | 4.69 | 18.73 | 32.7% |
| 128 | 1,323 | 5.08 | 20.31 | 43.9% |

These are SLURM allocation envelopes rather than summed ANTs task timings. The
1125-pixel baseline ran separately, whereas the other three conditions ran
concurrently and may have contended for shared storage.

```{figure} images/fixed-affine-chunk-size/runtime-throughput.png
:alt: Four-node wall time and completed chunks per hour by XY chunk size.

Task throughput rises as individual chunks become smaller, but total wall time and
node consumption also rise.
```

## Interpretation and limitations

This experiment does not support the hypothesis that reducing the nonlinear chunk
size removes brightness-dependent registration error. The named dim-region
correction is exactly unchanged, whole-field medians are unchanged to within 0.16
pixels, and the smallest chunks add runtime, seam exposure, and more extreme local
Jacobian values. For this dataset, retaining the 1125-pixel XY chunks is the
best-supported choice among those tested. There is no evidence here to change a
Cy-ExM production default.

The result also strengthens the original scientific observation: high-signal
windows generally register well, while dim windows have large, low-prominence,
and sometimes boundary-limited displacement estimates. Altering nonlinear spatial
support alone does not resolve that dependence. Future experiments should instead
test signal-aware preprocessing, masks or weights, multiscale metric choices, or
regularization while preserving the same fixed-affine design.

This characterization has important limits:

- it uses one specimen, one registration channel, CC radius 4, and three moving
  rounds;
- the local metric operates on Z maximum-intensity projections and integer XY
  shifts, so it does not measure Z residuals or subpixel displacement;
- seam classes are observational and remain partially confounded by signal and
  anatomy despite quartile stratification;
- the three smaller-chunk conditions ran concurrently, so their runtime comparison
  includes shared-filesystem effects;
- coordinator evidence serialization failed after each scientific graph completed
  because the characterization harness attempted to call `to_dict()` on a
  `RoundPreparationTask`. The analysis therefore recovered provenance from
  validated completion records and SLURM accounting. All 60 final artifacts used
  here matched their completion records, all expected chunk products were present,
  and the accepted source state revalidated unchanged.

Machine-readable tables corresponding to these figures are stored beside the image
assets as `regional-residuals.csv`, `seam-interior.csv`,
`jacobian-validity.csv`, and `runtime-throughput.csv`.
