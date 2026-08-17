# Insufficient Chunk Support Identity Fallback Design

## Context and current behavior

Serial chunked nonlinear registration already writes an identity displacement when
either scalar chunk has standard deviation below `1e-6`. This handles completely
empty edge chunks, but a nearly empty chunk can have nonzero variance and reach
ANTs. In the U2OS LAMP1 run, Cell 1 round 5 chunk 17 and Cell 3 round 3 chunk 32
then failed twice with ANTSpyX's generic
`RuntimeError: Registration failed with error code 1`.

The prepared fixed-grid masks show the actual failure boundary:

| Chunk | Fine common support | Coarsest support | Coarsest Z/Y/X span |
| --- | ---: | ---: | --- |
| Cell 1, round 5, chunk 17 | 2,736 | 137 | 15 / 12 / 1 |
| Cell 3, round 3, chunk 32 | 133 | 5 | 2 / 1 / 3 |

Here common support is the intersection of the fixed registration mask and the
affine-resampled moving registration mask. The current three-level SyN schedule
uses shrink factors `4x2x1`; both failed masks collapse to a one-voxel-thick support
on at least one coarsest axis. Neighboring successful edge chunks have at least
three to six coarsest voxels on their narrowest axis.

Separately, the existing Cell 3 round-1 LAMP1 TIFF triplet is an exact export of
`DAPI_tubulin_golgi_008.nd2`. The user has biologically confirmed that the
misnamed `DAPI_tubulin_lamp_006.nd2` is the correct Cell 3 round-1 LAMP1
acquisition. That ND2 has 61 Z planes, Cy5/GFP/DAPI channels, `uint16` pixels, and
voxel size `(0.108333333333333, 0.108333333333333, 0.3)` micrometers in XYZ.

## Goals

- Make nonlinear chunk admission deterministic from the exact masks supplied to
  registration.
- Use a zero nonlinear displacement for chunks whose common support cannot sustain
  the coarsest three-dimensional registration level.
- Preserve the accepted affine transform; identity applies only to the nonlinear
  displacement.
- Keep generic ANTs failures visible unless the independent support rule authorizes
  identity.
- Record canonical support evidence for every support-triggered identity warp.
- Preserve reuse of already completed, sufficiently supported local warps.
- Export and register the corrected Cell 3 round-1 acquisition without rewriting
  the original dataset's immutable scientific-plan snapshot.
- Resume the two genuinely incomplete original rounds after the code gate passes.

## Non-goals

- Do not catch every ANTs error-code-1 failure and call it an edge chunk.
- Do not change affine estimation, chunk geometry, overlap, interpolation, nonlinear
  schedules, metrics, or scalar reconstruction.
- Do not mark unsupported pixels as valid.
- Do not delete or relabel the incorrect Golgi-derived export in place.
- Do not bypass Cy-ExM's scientific-plan fingerprint mismatch protection.

## Selected support rule

The producer thresholds both regional registration masks at `> 0.5`, requires
finite values and exact fixed-chunk shape, and computes their logical intersection.
For a nonlinear schedule with `L` levels, the coarsest shrink factor is
`2 ** (L - 1)`. The current three-level schedules therefore use factor 4.

The common mask is sampled on the local fixed-grid pyramid lattice with
`common[::factor, ::factor, ::factor]`. Its canonical evidence contains:

- fine common-support voxel count;
- fine total voxel count;
- coarsest common-support voxel count;
- coarsest support span in ZYX, with `(0, 0, 0)` for empty support;
- coarsest shrink factor; and
- the versioned policy identifier.

Support is sufficient only when the coarsest mask contains at least eight voxels
and spans at least two occupied lattice coordinates along each of Z, Y, and X.
Eight voxels and two samples per axis are the minimum nondegenerate `2x2x2`
three-dimensional lattice. This is a geometric viability test tied to the actual
multiresolution schedule, not a foreground-fraction threshold.

If either mask is absent, support is not assessed and the existing low-variance
policy remains the only identity rule. Malformed, nonfinite, or shape-mismatched
masks fail closed; they never authorize identity.

## Registration and exception flow

For each chunk:

1. Validate scalar arrays and calculate support evidence before cache lookup.
2. Build the local request. Sufficient chunks retain the prior canonical request so
   their existing local warps remain reusable. Insufficient chunks add the exact
   versioned support evidence and `identity-insufficient-support` decision.
3. Reuse only an artifact matching that exact request.
4. If support is insufficient, write an identity local warp without calling ANTs.
5. Otherwise preserve the existing low-variance identity behavior.
6. Otherwise call ANTs normally.
7. Catch only `RuntimeError` whose exact message is
   `Registration failed with error code 1`. Re-evaluate the same immutable support
   rule. It may authorize identity only if the canonical result is insufficient;
   otherwise re-raise the original exception. Other exception types and messages
   always propagate.

Step 7 is defense-in-depth, not an alternate scientific decision rule. Under normal
immutable inputs the preflight already handles every support-authorized identity.
The exception path must never introduce a looser threshold or depend only on ANTs
text.

Temporary warp candidates retain the existing best-effort cleanup. Identity and
estimated warps use the same atomic artifact commit and validation boundary.

## Reader and provenance compatibility

The support policy is conditional rather than a global request-version change.
This preserves exact reuse for sufficiently supported historical local warps.
Support-triggered records carry a strict additional mapping. Authoritative workflow
status accepts either:

- the historical/base local request; or
- the base request plus a canonical insufficient-support mapping whose decision is
  mathematically implied by its typed counts and spans.

Unknown fields, unsupported policy identifiers, noncanonical values, or evidence
that does not imply insufficiency fail closed. The manifest and full assembled warp
remain bound to the identities of every local warp, so changing an edge chunk
invalidates and rebuilds all downstream nonlinear, metric, registered-image, and
channel artifacts for that round.

The final transformed moving mask remains the downstream validity authority.
Identity fallback merely makes the displacement field defined; it does not turn
zero-support pixels into valid pixels. Supported pixels inside an identity chunk
receive affine-only alignment and remain visible as such through the local
completion evidence.

## Corrected Cell 3 round-1 data flow

The original Cell 3 raw directory and scientific plan remain unchanged for
provenance and for resuming its incomplete round 3. A separate correction dataset
contains:

- copied Cell 3 round-2 reference TIFFs;
- an atomic export of ND2 planes 1 through 60 from
  `DAPI_tubulin_lamp_006.nd2`;
- Cy5 as `round_1_647_LAMP1.tif`;
- GFP as `round_1_488_tubulin.tif`; and
- DAPI as `round_1_405_DAPI.tif`.

Every exported pixel is compared exactly against its ND2 channel and plane before
publication. The focused dataset declares acquisition round 2 as reference and only
round 1 as a moving round. It uses a new dataset ID and output root, avoiding any
scientific-plan fingerprint bypass and avoiding reruns of already completed Cell 3
rounds 4 and 5.

The correspondence Markdown records the student naming error, the user's visual
identity confirmation, exact export verification, and both the retained historical
and corrected output locations. The analysis correspondence can then map Cell 3
round 1 to the correction dataset while mapping its other rounds to the original
dataset.

## Alternatives considered

### Foreground fraction threshold

Rejected. A percentage is sensitive to chunk dimensions and can reject a thin but
spatially valid biological region. The selected rule tests the geometry ANTs sees at
its coarsest level.

### Catch every ANTs error code 1

Rejected. ANTSpyX uses the same generic exception for unrelated geometry, numerical,
I/O, mask, and transform failures. An unconditional fallback would silently publish
failed registrations as completed identity results.

### Replace the original Cell 3 TIFFs in place

Rejected. Their identities are embedded in an immutable persisted scientific plan.
Replacement would correctly cause a fingerprint mismatch, and bypassing it would
destroy provenance.

### New full five-round Cell 3 dataset

Scientifically valid but unnecessarily expensive. It would repeat completed rounds
whose inputs and artifacts have not changed. The focused correction dataset provides
the corrected round while retaining exact provenance for the original series.

## Testing

- Unit-test exact support counts, coarsest spans, sufficient and insufficient
  decisions, missing masks, malformed masks, and nonfinite masks.
- Prove a thin nonzero chunk writes identity and never calls nonlinear registration.
- Prove sufficient support still calls nonlinear registration.
- Prove generic code 1 with sufficient support propagates unchanged.
- Prove unrelated runtime messages and exception types propagate.
- Prove support-triggered completion records contain canonical evidence and are
  reused exactly.
- Prove strict workflow status accepts canonical fallback evidence and rejects
  malformed, unknown, or logically sufficient fallback claims.
- Exercise both serial Phase 1 and fine-grained workflow boundaries through their
  shared chunk stage.
- Run focused unit and integration tests, complete non-HPC portable tests, Ruff,
  formatting, basedpyright, and pre-commit before operational use.
- Validate all original and correction dataset/execution YAML pairs before launch.
- Verify corrected TIFF/ND2 equality for every pixel and channel.
- After execution, use `cyexm workflow status` and completion artifacts—not SLURM
  process state—to establish completion.

## Acceptance criteria

- Both observed failed chunks are classified as insufficient from their prepared
  masks and produce recorded identity nonlinear warps.
- Sufficient chunks preserve historical request identity and artifact reuse.
- ANTs code 1 remains fatal for sufficient-support chunks.
- Cell 1 round 5 and original Cell 3 round 3 complete after resumption.
- The corrected Cell 3 round-1 export is exact, its focused workflow completes, and
  the incorrect historical export remains intact and clearly documented.
- All five LAMP1 cells have authoritative registered artifacts for all intended
  acquisition rounds, with Cell 3 round 1 explicitly sourced from the correction
  dataset.
