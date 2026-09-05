# XY rotation and similarity initial alignment

## When to use it

Use one initial-alignment mode on an individual moving round when its registration
channel has a gross in-plane pose difference from the fixed round. Both modes are
target-blind: they use only the fixed and moving registration channels. They supply
one seed to the configured 3-D linear stage and do not introduce a registration
strategy. Omit `initial_alignment` to preserve the historical workflow exactly.

Choose `xy_rotation` when rotation and lateral translation are sufficient. Choose
`xy_similarity` when acquisition or sample expansion can also introduce a modest
uniform XY scale difference. The modes are mutually exclusive.

Explicit rotation is appropriate when inspection supports a correction angle:

```yaml
initial_alignment:
  xy_rotation:
    mode: explicit
    angle_deg: 27.7
```

Automatic rotation searches only the registration channels:

```yaml
initial_alignment:
  xy_rotation:
    mode: estimate
    search_range_deg: [-45, 45]
    coarse_step_deg: 1
    refinement_step_deg: 0.1
    minimum_score_gain: 0.01
    minimum_peak_margin: 0.01
```

Similarity mode estimates angle, lateral translation, and one uniform visible XY
scale from the same MIP features:

```yaml
initial_alignment:
  xy_similarity:
    mode: estimate
    search_range_deg: [-18, 18]
    coarse_step_deg: 1
    refinement_step_deg: 0.1
    scale_range: [0.90, 1.10]
    coarse_scale_step: 0.01
    refinement_scale_step: 0.002
    minimum_score_gain: 0.01
    minimum_peak_margin: 0.01
```

`scale_range` must be positive and strictly enclose `1.0`; the refinement scale
step must be smaller than the coarse step. Scale is visible magnification applied
to the moving XY plane about its physical center. Z scale remains one. Constrain
both angle and scale to ranges justified by acquisition and expansion behavior.

No target or dependent channel selects the transform. This target-blindness avoids
optimizing initialization for the signal later measured.

## Angle and coordinate convention

`angle_deg` is a correction applied about the physical center of the cropped moving
registration image. Positive angles follow the right-hand rule around physical +Z.
For identity direction cosines, physical +X points right and physical +Y points down
in a conventional row-down XY display, so a positive physical angle appears clockwise.
Its sign is therefore opposite to `scipy.ndimage.rotate`.

```text
raw registration channels -> optional crop -> XY rotation/similarity MIP seed
                                                    |
                                                    v
                                      configured 3-D linear residual
                                                    |
                                                    v
                                         complete affine -> nonlinear stage
                                                                  |
                                                                  v
                                             raw channels, one resampling pass
```

Intensity preprocessing is separate: `raw -> crop -> feature estimate -> transforms
-> raw output`. Geometric initialization is `raw -> crop -> MIP seed -> 3-D linear
residual -> nonlinear -> raw output`. The initializer sees raw cropped registration
channels, not feature-normalized or dependent-channel images.

## Automatic confidence and QC

The estimators compare maximum-intensity-projection gradient features on a
physical-isotropic XY grid over a coarse angle grid, refine near the best candidate,
and estimate translation by phase correlation. Similarity then searches the bounded
scale grid at the winning angle and re-refines angle at the winning scale. Every
candidate re-estimates translation.

The implementation uses complete image spacing and direction metadata rather than
assuming square pixels or identity direction. Fixed and moving direction matrices
must match. Automatic mode records an incompatible-direction rejection; explicit
mode stops before creating a seed. Matching arbitrary nonidentity directions are
supported.

Automatic rejection occurs when directions do not match, an optimum is on an angle
boundary, improvement over identity is insufficient, or the peak is not separated
from alternatives. Similarity additionally rejects an optimum on either scale
boundary. JSON records the angle and scale candidate curves, identity score, score
gain, independent peak margin, translation, and valid overlap.

| Evidence | Interpretation |
| --- | --- |
| Fixed, moving, initialized panels | Visual check that gross pose and optional scale improve. |
| Before/after difference panels | A smaller structured residual is encouraging, not biological accuracy. |
| Score plot and JSON curves | The selected optimum should be interior and separated from competitors. |
| Rejection in JSON | Expand only scientifically justified bounds; do not lower thresholds blindly. |

A successful estimate is only an initialization result. Inspect the QC montage and
normal affine/nonlinear outputs before accepting biological measurements. For data
with plausible scale variation, pair the MIP similarity seed with a 3-D `Similarity`
linear stage so the volume can refine the same degrees of freedom. TRSAA remains
available when its larger affine family is scientifically intended.

## Artifacts, restart, and force

An enabled round adds `initial_alignment_{round}.json`,
`initial_alignment_{round}.mat`, and `initial_alignment_qc_{round}.png`, each with a
completion record. JSON records normalized configuration, realized angle, optional
scale, translation, source (`explicit` or `estimated`), score/confidence evidence,
fixed and moving physical geometry, and source/crop identities. Status and run
records repeat configured and available realized evidence.

The `.mat` is the exact seed supplied to the configured 3-D linear stage. ANTSpyX's
returned affine already includes that seed and is persisted once as
`linear_transform_{round}.mat`; downstream channels retain the established
`[warp, affine]` chain and one resampling pass.

An exact rerun reuses the authenticated initialization set. Changing a setting
changes scientific and request identity. `--force-round` recomputes the seed and
descendants. A partial or mismatched set is not authorized by sibling filenames.
Automatic rejection retains JSON and QC evidence, creates no seed, and does not
start linear registration.

See the [dataset manifest reference](dataset-manifest.md), [registration
preprocessing](preprocessing.md), and [registration artifact
reference](registration-artifacts.md).
