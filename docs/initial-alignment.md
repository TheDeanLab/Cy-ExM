# XY rotation initial alignment

## When to use it

Use `initial_alignment.xy_rotation` on an individual moving round when its
registration channel has a large in-plane rotational offset from the fixed round.
The option supplies a seed to the existing TRSAA linear stage; it does not introduce
a new registration strategy. Omit it to preserve the historical workflow exactly.

Explicit mode is appropriate when inspection already supports a correction angle:

```yaml
initial_alignment:
  xy_rotation:
    mode: explicit
    angle_deg: 27.7
```

Automatic mode searches only the fixed and moving registration channels:

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

No target or dependent channel is used to select the angle. This target-blindness
prevents the initialization from optimizing for the signal later measured.

## Angle and coordinate convention

`angle_deg` is a correction applied about the physical center of the cropped moving
registration image. Positive angles follow the right-hand rule around physical +Z.
For identity direction cosines, physical +X points right and physical +Y points down
in a conventional row-down XY display, so a positive physical angle appears clockwise.
Its sign is therefore opposite to `scipy.ndimage.rotate`.

```text
raw registration channels -> optional crop -> XY rotation seed
                                             |
                                             v
                                  TRSAA residual affine
                                             |
                                             v
                                     complete affine -> SyNOnly
                                                           |
                                                           v
                                      raw dependent channels, one resampling pass
```

Intensity preprocessing is separate: `raw -> crop -> feature estimate -> transforms
-> raw output`. Geometric initialization is `raw -> crop -> seed -> residual affine
-> nonlinear -> raw output`. The estimator sees raw cropped registration channels,
not feature-normalized or dependent-channel images.

## Automatic confidence and QC

The estimator compares maximum-intensity-projection gradient features on a
physical-isotropic XY grid over a coarse angle grid, refines near the best candidate,
and estimates translation by phase correlation. It uses the complete image spacing
and direction metadata rather than assuming square pixels or identity direction.
Fixed and moving direction matrices must match for this two-dimensional operation.
Automatic mode records an incompatible-direction rejection with its diagnostics;
explicit mode stops before creating an alignment seed. Arbitrary non-identity
direction matrices are supported when they match.
It fails closed when they do not match, when the best angle lies at the search
boundary, does not improve sufficiently over zero, or is not separated from
alternatives.

| Evidence | Interpretation |
| --- | --- |
| Fixed, moving, initialized panels | Visual check that rotation and translation improve gross alignment. |
| Before/after difference panels | A smaller structured residual is encouraging but is not a biological accuracy measurement. |
| Score plot | The selected peak should be interior and separated from competitors. |
| Rejection in JSON | Widen a scientifically justified range or use a reviewed explicit angle; do not lower thresholds blindly. |

A successful estimate is only an initialization result. Inspect the QC montage and
the normal affine/nonlinear registration outputs before accepting biological
measurements. A practical initial search range is often `[-45, 45]`; widen it only
when acquisition geometry supports larger rotations.

## Artifacts, restart, and force

An enabled round adds `initial_alignment_{round}.json`,
`initial_alignment_{round}.mat`, and `initial_alignment_qc_{round}.png`, each with a
completion record. The JSON records normalized configuration, realized angle and
translation, source (`explicit` or `estimated`), all scores, overlap, and confidence
decisions. Status JSON and run records repeat the configured and available realized
evidence so machine-readable reports do not require inference from filenames. The
`.mat` is the exact seed supplied to TRSAA. The final `linear_transform_{round}.mat` is still one
complete affine, so downstream channels retain the established `[warp, affine]`
chain and are resampled once.

An exact rerun reuses the complete initialization set. Changing any setting changes
the scientific fingerprint and request identity. `--force-round` recomputes the seed
and descendants. An interrupted or mismatched member is not silently authorized by
its siblings. Automatic rejection retains JSON and QC evidence, creates no seed, and
does not start linear registration.

See the [dataset manifest reference](dataset-manifest.md), [registration
preprocessing](preprocessing.md), and [registration artifact
reference](registration-artifacts.md).
