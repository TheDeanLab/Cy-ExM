# CC SyN radius characterization

:::{important}
This is a dataset-specific, single-sweep characterization. It supports retaining
the provisional scalar voxel radius of 4, but it does not establish a universally
optimized Cy-ExM default.
:::

Cy-ExM supports the `CC` nonlinear SyN metric with one scalar voxel radius.
This study compared the Mattes SyN baseline with CC radii 2, 4, 6, and 8 on the
cropped spheroid data. A radius spans the same number of voxels along each axis,
so anisotropic voxel spacing makes its physical neighborhood anisotropic.

## Execution and evaluation

Each CC workflow used three exclusive 512-GB SLURM nodes, one moving round per
node, and 30 ITK and BLAS-family threads per worker. The four configurations were
identical apart from dataset identity, output and worker-scratch paths, SLURM job
name, and CC radius. All four workflows and all moving-round tasks completed, and
each workflow reported successful client, cluster, and execution-configuration
cleanup.

Registration quality was evaluated in three complementary ways:

- the workflow's whole-volume Correlation and Mattes mutual-information metrics;
- gradient normalized cross-correlation (NCC) on Z-maximum-intensity projections
  in the previously identified dim region, `y = 1600:1800` and `x = 300:400`;
- the same gradient-NCC displacement search in an objectively selected bright
  central region, `y = 1227:1427` and `x = 840:1040`.

The local result is reported as the magnitude of the corrective two-dimensional
displacement that maximized gradient NCC within a plus-or-minus 30-pixel search.
Zero pixels means that the current registered position was already optimal under
this measurement. This projection-based analysis matches the view in which the
peripheral offset was originally observed; it is not a substitute for a future
three-dimensional residual field.

## Computational performance

| Nonlinear metric | Workflow wall time | Approximate node-hours | Time relative to Mattes |
| --- | ---: | ---: | ---: |
| Mattes baseline | 3:49:57 | 11.5 | 1.00x |
| CC radius 2 | 3:52:54 | 11.6 | 1.01x |
| CC radius 4 | 5:50:49 | 17.5 | 1.53x |
| CC radius 6 | 9:08:44 | 27.4 | 2.39x |
| CC radius 8 | 13:46:08 | 41.3 | 3.59x |

Approximate node-hours are three times workflow wall time because each workflow
held three workers. Dependent-channel transformation took approximately 25 seconds
per channel; nonlinear registration dominated the runtime. Increasing the radius
therefore imposed a large computational cost rather than an I/O or channel-export
cost.

## Spatially resolved registration

Lower residual displacement is better.

| Nonlinear metric | Round 2 dim residual | Round 3 dim residual | Round 4 dim residual | Bright-region residual |
| --- | ---: | ---: | ---: | ---: |
| Mattes baseline | 10.63 px | 18.36 px | 15.00 px | 0 px in every round |
| CC radius 2 | 8.49 px | 17.09 px | at least 31.95 px | Round 4 failed locally |
| CC radius 4 | **7.81 px** | **14.32 px** | **13.45 px** | 0 px in every round |
| CC radius 6 | 7.81 px | 15.81 px | at least 30.15 px | Round 4 failed locally |
| CC radius 8 | 9.22 px | 17.89 px | 16.28 px | 0 px in every round |

The radius-2 and radius-6 round-4 optima reached the search boundary, so their
reported dim-region displacements are lower bounds rather than well-resolved
estimates. Their corresponding bright-region measurements also showed large
misalignment. As discussed below, these failures began in the independently
estimated affine stage and cannot be assigned to the nonlinear radius.

Relative to the Mattes baseline, radius 4 reduced the measured dim-region residual
by approximately 26% in round 2, 22% in round 3, and 10% in round 4. It preserved
zero measured displacement in the bright region. The remaining 8--14-pixel
residual is nevertheless scientifically meaningful: CC radius 4 reduced the
brightness-associated error but did not eliminate it.

## Global similarity

The workflow's final whole-volume Correlation metric favored Mattes even though
radius 4 improved the known dim region.

| Nonlinear metric | Round 2 | Round 3 | Round 4 |
| --- | ---: | ---: | ---: |
| Mattes baseline | **0.726** | **0.616** | **0.608** |
| CC radius 2 | 0.694 | 0.568 | 0.276 |
| CC radius 4 | 0.685 | 0.542 | 0.557 |
| CC radius 6 | 0.672 | 0.540 | 0.313 |
| CC radius 8 | 0.663 | 0.546 | 0.542 |

This is not a contradiction. A global similarity score is dominated by the larger,
brighter, information-rich parts of the image and can obscure a spatially localized
residual. The result reinforces the need to retain spatially resolved, signal-aware
quality control rather than optimizing the CC radius from one global number.

## Interpretation and limitations

Radius 4 provided the best low-signal tradeoff in this sweep. Radius 2 required
approximately the same workflow time as Mattes and retained slightly stronger
global similarity than the other CC radii in rounds 2 and 3, but it left more of
the peripheral residual. Radii 6 and 8 imposed much larger runtimes without a
corresponding registration improvement.

The sweep has an important experimental confound: each configuration independently
reran the stochastic Mattes affine stage. Changing the CC radius changes only the
nonlinear stage, yet the round-4 radius-2 and radius-6 runs already had substantially
worse linearly registered metrics. Those round-4 failures are therefore affine-run
variation, not evidence that the corresponding CC radii intrinsically fail. Smaller
differences among the other affine estimates may also contribute to the measured
radius differences.

This evidence supports retaining radius 4 as the provisional default; it does not
justify labeling radius 4 optimized. Formal promotion would require a fixed-affine
or replicated-seed comparison, a three-dimensional blockwise residual map,
warp-Jacobian and folding checks, and chunk-seam analysis. If the remaining
low-signal residual is unacceptable, the next scientifically motivated experiment
is a low-weight gradient or Laplacian-of-Gaussian secondary CC metric rather than a
larger scalar CC radius.

The subsequent
[fixed-affine nonlinear chunk-size characterization](fixed-affine-chunk-size-characterization.md)
held this radius-4 affine state constant while varying only nonlinear chunk
support. It reproduced these three dim residuals exactly at every tested chunk
size, confirming that smaller chunks alone do not remove the remaining
brightness-associated displacement.
