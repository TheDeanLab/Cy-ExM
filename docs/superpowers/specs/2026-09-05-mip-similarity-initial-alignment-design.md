# MIP Similarity Initial Alignment Design

## Status

Approved for implementation on 2026-09-05 after the spheroid reduced-volume
comparison. This specification extends, rather than replaces, the accepted XY
rotation initializer.

## Context

The spheroid rounds differ through in-plane rotation, lateral translation, and a
small uniform expansion change. A reduced-volume comparison showed that a
maximum-intensity-projection (MIP) similarity seed followed by 3-D Similarity
registration materially improved 3-D correlation for every moving round. During
that comparison, a separate defect was identified: ANTSpyX returns a complete
linear transform when `initial_transform` is supplied, but Cy-ExM appended the seed
again and therefore applied it twice.

Cy-ExM already owns strict per-round initialization configuration, MIP feature
construction, physical geometry conversion, restart-safe initialization artifacts,
fine/coarse parity, and status provenance. The new behavior must remain inside that
existing lifecycle.

## Goals

- Correct seeded linear registration so the complete transform returned by ANTSpyX
  is applied exactly once.
- Add an optional, constrained XY similarity initializer that estimates physical
  +Z rotation, XY translation, and one uniform XY scale from registration-channel
  MIPs.
- Preserve `initial_alignment.xy_rotation` byte-for-byte in normalized
  configuration and evidence when the new mode is not selected.
- Persist the realized scale, score curves, confidence decision, physical transform
  geometry, source identities, crop identities, and QC provenance.
- Use the same initialization artifacts and restart authority in whole-image,
  serial-chunked, coarse, and fine-grained execution.
- Document the mode and update the spheroid scientific manifest to use it with 3-D
  `Similarity` residual registration.

## Non-goals

- No anisotropic scale, shear, reflection, or 3-D MIP initializer.
- No dependent channel may influence initialization.
- No change to nonlinear registration, interpolation, fixed/moving roles, or the
  raw-output boundary.
- No automatic choice between rotation and similarity modes.
- No deletion or overwrite of the earlier scientifically invalid spheroid output.

## Public configuration

`initial_alignment` accepts exactly one of the existing `xy_rotation` mapping or a
new `xy_similarity` mapping. Similarity is estimate-only in version 1:

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

The angle fields retain the existing physical convention. `scale_range` is a
strictly increasing positive finite pair containing `1.0`. Both scale steps are
positive and the refinement step is smaller than the coarse step. The coarse step
must span at least one interval within the configured range. Unknown, mixed, or
rotation-only keys fail strict loading. Omission continues to mean no initializer.

## Estimator and confidence policy

The versioned policy identifier is
`mip-gradient-phase-correlation-xy-similarity-v1`. It reuses the current cropped raw
registration-channel MIP, robust percentile scaling, smoothing, gradient magnitude,
physical-isotropic XY conversion, common center-padded canvas, phase-correlation
translation, overlap masking, Pearson score, and deterministic tie breaking.

To bound full-resolution work, optimization is deterministic coordinate refinement:

1. Evaluate the configured coarse and refined angle grids at scale `1.0`.
2. Evaluate the configured coarse and refined scale grids at the winning angle.
3. Re-evaluate a local refined angle grid at the winning scale.
4. Select the best realized candidate across the final scale and angle refinements,
   breaking ties by score, distance of scale from `1.0`, absolute angle, scale, and
   angle.

Every scale candidate is applied about the same moving physical center as rotation;
translation is re-estimated for every candidate. Evidence records stage-labelled
candidate curves. The result is rejected when angle or scale lands on a configured
boundary, the total score gain over `(angle=0, scale=1)` is below
`minimum_score_gain` for a non-identity result, or the independent peak margin is
below `minimum_peak_margin`. Invalid images, incompatible direction matrices, and
no positive overlap retain fail-closed behavior.

## Physical transform and QC

The persisted seed is one 3-D affine whose local matrix contains uniform XY scale
and physical +Z rotation while Z scale remains one. It maps the moving physical
center to the fixed physical center plus the estimated visible XY shift, using the
complete origin, spacing, and direction metadata. The seed is supplied to ANTSpyX;
the single returned affine is the complete residual-plus-seed transform.

QC resamples the initialized moving MIP onto the fixed physical canvas using the
same persisted transform semantics. Its score panel represents the recorded
similarity candidates. The JSON is the semantic authority and includes normalized
configuration, source and crop identities, both geometries, realized angle,
translation, scale, candidate scores, confidence values, and versioned algorithm.
The MAT and PNG remain separately authenticated artifacts whose requests bind the
JSON identity.

## Ownership, restart, and compatibility

The strict dataset loader owns normalization. `initial_alignment.py` owns estimator
policy and canonical evidence. `transforms.py` owns physical affine construction.
`whole_image.py` owns artifact lifecycle. Existing prepared manifests, fine worker
admission, workflow provenance, and status continue to consume the one canonical
initial-alignment evidence parser.

Rotation evidence and manifests remain unchanged. Similarity changes the scientific
fingerprint and all initialization descendants. A stale or partial similarity
artifact fails admission or is recomputed through the Phase 1 artifact store. Force
propagation remains unchanged.

## Testing and acceptance

- A real-ANTs integration regression proves a nonidentity seed is not applied twice
  and that the returned scalar matches ANTSpyX's complete transform semantics.
- Strict configuration tests cover defaults, overrides, mutual exclusion, numeric
  bounds, and canonical reconstruction.
- Deterministic synthetic tests recover known angle, translation, and uniform scale,
  reject boundary/low-confidence results, and validate canonical evidence.
- Physical landmark tests cover center, origin, anisotropic spacing, and matching
  nonidentity direction matrices.
- Whole-image, prepared/fine, status, and documentation tests cover restart and
  provenance parity.
- All portable repository gates pass before any scheduler submission.
- The spheroid manifest validates with constrained per-round angle ranges,
  conservative scale bounds, and 3-D `Similarity`; a fresh full-resolution run is
  submitted without deleting earlier output.

