# Different Source Shapes Scientific Compatibility Design

**Status:** Approved design pending written-spec review

**Date:** 2026-07-21

## Context

The spheroid dataset at
`/archive/bioinformatics/Danuser_lab/Dean/Seweryn/resubmission/TC71_spheroid/SPH_1_20x`
contains a 54-slice round-1 reference and moving rounds with 36, 39, and 46
slices. The moving rounds have matching lateral dimensions in this dataset, but
the scientific requirement is broader: a fixed image, registration channel, and
dependent moving channels may differ along any of the Z, Y, or X axes.

Current Cy-ExM planning rejects every source whose array shape differs from the
reference. The channel-transform primitive repeats that restriction before
applying an accepted transform chain. These checks prevent behavior that worked
in the legacy ClearEx registration workflow and that ANTs natively supports.

ClearEx commit `2311a873b783477add7391ff00245f76490147f0` is the read-only
legacy reference. Its `ChunkedImageRegistration` performs linear registration
between the native fixed and moving grids, applies the affine transform with the
fixed image as the reference, and only then divides the fixed-grid affine result
into chunks for nonlinear registration. Transform application likewise uses the
fixed image as the reference grid. Cy-ExM must preserve that data flow.

## Goals

1. Allow every scalar 3-D source image to have its own positive ZYX shape.
2. Preserve the legacy affine-before-nonlinear workflow.
3. Resample the registration channel onto the fixed-reference grid during the
   affine stage before serial nonlinear chunking.
4. Apply the accepted warp-then-affine chain directly to every dependent channel
   from its native grid.
5. Require every registered scalar, nonlinear warp, and transformed-channel
   output to use the fixed-reference geometry.
6. Preserve source-specific shape evidence in the scientific plan, fingerprint,
   completion records, and status validation.
7. Retain existing restart, identity, atomic-write, interpolation, axis, and
   native-thread guarantees.

## Non-goals

- Do not pad, crop, truncate, or pre-resample source TIFFs merely to equalize
  their array dimensions.
- Do not change voxel-spacing, origin, direction, transform order, inversion,
  interpolation, mask, registration schedule, or intensity policies.
- Do not add a new manifest field or require operators to describe source shapes.
- Do not infer physical offsets from unequal array extents. Cy-ExM continues to
  use embedded geometry when present and its established TIFF geometry defaults
  otherwise.
- Do not introduce chunk-level parallelism or change the round-level scheduler
  boundary.
- Do not edit or depend on the ClearEx repository.

## Alternatives Considered

### Pre-resample every moving input

Pre-resampling could make all sources match the fixed grid before registration,
but it adds an interpolation before transform estimation, changes the legacy
scientific path, increases storage and runtime, and introduces a new artifact
lifecycle. This approach is rejected.

### Remove all shape checks

Removing both input and output checks would accept differing sources, but it
would also allow a backend to return affine, nonlinear, or final channel results
on the wrong grid. This approach is rejected because it weakens scientific
validation.

### Fixed-target stage boundaries

The selected approach permits arbitrary source shapes while retaining strict
fixed-grid checks after every resampling boundary. It matches ClearEx and ANTs
without adding an extra interpolation.

## Selected Architecture

### Planning and scientific-plan validation

`cyexm.workflow.planning` will continue to validate every declared source as a
readable scalar 3-D TIFF with a positive shape and manifest-compatible spacing.
It will no longer compare source shapes to the reference shape. Each inspected
source retains its own `shape_zyx` in the canonical scientific plan, so source
shape differences remain fingerprinted evidence.

`cyexm.workflow.status` will apply the same contract when reading a persisted
scientific plan: every source shape must be independently valid, but source
shapes need not equal the reference. All other canonical ordering, source-use,
policy, artifact, and fingerprint checks remain unchanged.

No same-shape requirement applies between a registration channel and its
dependent channels. This matches direct ANTs transform application: co-acquired
channels may be supplied on different native grids, and each is independently
resampled onto the fixed reference.

### Registration data flow

The round registration path remains:

1. Load the fixed reference and moving registration channel on their native
   grids.
2. Optionally crop each image through the established crop policy.
3. Estimate the affine transform using the native fixed and moving images.
4. Apply that affine with the fixed image as the reference, producing a
   `moving_linear` scalar and mask on the fixed grid.
5. Assert that `moving_linear` and its mask use fixed-reference geometry.
6. Estimate whole-image or serial-chunked nonlinear registration on that common
   fixed grid.
7. Persist a full warp whose geometry matches the fixed reference.

`ChunkedImageRegistration` keeps its strict equality check between the fixed
image and the post-affine `moving_linear` image. That is a downstream invariant,
not a source-input restriction. Chunk bounds continue to be computed from the
fixed-reference shape.

Pre-registration metrics remain defined over the native fixed and moving images,
as in ClearEx and ANTs. Linear and nonlinear metrics use fixed-grid resampled
images.

### Dependent-channel transformation

`cyexm.registration.channels.transform_channel` will load the fixed reference
and one dependent moving source without requiring equal shapes. It will apply
the persisted transform list in the existing order `[warp, affine]`, with both
inversion flags false and the manifest-selected interpolation.

The fixed image remains the ANTs reference for transform application. The result
must exactly match the fixed-reference `ImageGeometry`, including shape, origin,
spacing, and direction, before it can be atomically committed. A backend result
on the moving grid or any other grid remains a hard failure.

The existing input identities, fixed geometry, transform identities,
interpolation, output policy, and source TIFF identity continue to participate in
artifact reuse decisions. Differing source dimensions do not weaken cache
invalidation.

## Failure Handling

- Malformed, non-scalar, non-3-D, empty-axis, unreadable, or spacing-incompatible
  sources fail during planning exactly as before.
- A failed affine resampling or an affine result that does not match fixed
  geometry fails before nonlinear registration or chunk creation.
- A serial-chunked nonlinear input that is not on the fixed grid remains a hard
  registration error.
- A transformed dependent channel that does not match the fixed geometry fails
  before artifact promotion; existing data and completion-record cleanup rules
  remain authoritative.
- Existing scientific plans remain readable. Their equal source shapes are a
  valid special case of the new contract.

## Scientific Regression Tests

### Portable unit tests

1. Planning accepts a fixed reference, registration source, and dependent source
   whose Z, Y, and X dimensions all differ.
2. Planning still records the exact shape of every source and changes the
   scientific fingerprint when a source identity or header changes.
3. Scientific-plan/status validation accepts independently valid unequal shapes
   while still rejecting malformed source shapes.
4. Channel transformation accepts unequal fixed and moving shapes and calls the
   backend once with the fixed image as the transform reference.
5. Channel transformation rejects a backend result that does not exactly match
   fixed geometry.
6. Serial chunk registration continues to reject a post-affine moving image that
   does not match fixed geometry.

### Concrete scientific integration tests

1. A small deterministic real-ANTs transform test uses fixed and moving images
   that differ along all three axes, applies an identity affine and zero warp,
   and proves that the output shape, origin, spacing, and direction exactly match
   the fixed reference.
2. A small deterministic real-ANTs serial-chunked registration test starts with
   unequal native fixed and moving shapes, uses bounded dry-run schedules, and
   proves that affine resampling produces the fixed grid before nonlinear chunk
   processing and that the final scalar and warp retain fixed geometry.
3. Numerical assertions use exact geometry equality. Any scalar tolerance must
   state the specific ANTs floating-point reason and remain limited to overlapping
   transformed values rather than registration-quality claims.

### Dataset acceptance

The edited spheroid manifest and execution profile must pass both:

```bash
uv run cyexm workflow validate --json \
  --execution /archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/execution-slurm-512gb.yaml \
  /archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/spheroid-dataset.yaml

uv run cyexm workflow plan --json \
  --execution /archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/execution-slurm-512gb.yaml \
  /archive/bioinformatics/Danuser_lab/Dean/dean/20267-07-21/spheroid-dataset.yaml
```

Validation must report rounds 2, 3, and 4 without modifying the source dataset or
submitting scheduler jobs.

## Guidance Updates

- Root `AGENTS.md` will state that native source shapes may differ and that fixed
  geometry is the post-resampling contract.
- `src/cyexm/workflow/AGENTS.md` will distinguish per-source header validity from
  fixed-grid output validation.
- `src/cyexm/registration/AGENTS.md` will require affine resampling before chunked
  nonlinear work and fixed-reference geometry for transformed channels.
- `tests/AGENTS.md`, or the narrower applicable child guidance, will require
  unequal-shape scientific regressions when registration-grid behavior changes.

## Acceptance Criteria

The change is accepted only when:

1. arbitrary positive source shapes are supported along Z, Y, and X;
2. legacy affine-to-fixed resampling occurs before nonlinear chunking;
3. dependent channels may differ in shape from both the fixed reference and the
   registration channel;
4. all persisted registered outputs use exact fixed-reference geometry;
5. the scientific plan preserves each source shape and remains deterministic;
6. focused unit and real-ANTs integration regressions pass;
7. all relevant portable project gates pass;
8. the spheroid configuration validates and plans without scheduler access; and
9. the relevant `AGENTS.md` files document the scientific contract.
