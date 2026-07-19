# ClearEx migration-parity report

## Scope and result

Cy-ExM was compared with the read-only ClearEx extraction source at commit
`2311a873b783477add7391ff00245f76490147f0` on 2026-07-19. The comparison used
one native thread for ITK and every configured BLAS-family runtime, the deterministic
ANTs seed `20260719`, and an identical `16 x 32 x 32` synthetic volume for fixed and
moving scalar input. Mask behavior uses independently constructed deterministic,
asymmetric fixed and moving masks; the moving mask is shifted by one voxel in Y and
X, and every input and transformed mask has the exact value set `[0.0, 1.0]`.

The numerical migration gate is explicitly **TRSAA low followed by SyNOnly dry
run**. The matched calls use the shared, consumed moving-mask parameter and omit the
fixed mask in both environments. ClearEx's fixed-mask keyword defect is characterized
separately below. All eight migration categories passed. Registration arrays use
`rtol=1e-4` and `atol=1e-5`; chunk geometry and data are exact. Spatial metadata uses
an absolute `1e-7` tolerance for float32 NIfTI spacing serialization in this
historical gate. Current orchestration uses the same scale only for serialized TIFF
geometry comparisons.

The captured low-level parity arrays are direct native ANTs XYZ arrays and remain
unchanged. A later orchestration correction established external NumPy/TIFF/chunk
ZYX versus native ANTs XYZ adapters; it does not alter native affine or displacement
component algebra and does not revise the results or totals in this report.

| Gate | Result | Evidence |
| --- | --- | --- |
| Geometry | PASS | Shape exact; origin, spacing, and direction within `1e-7` |
| Affine scalar | PASS | Maximum absolute difference `0.0` |
| Nonlinear scalar | PASS | Maximum absolute difference `0.0` |
| Mask interpolation | PASS | Input, affine, and nonlinear masks exactly equal with values `[0.0, 1.0]` |
| Transform order | PASS | Warp then affine, both uninverted; maximum difference `5.960464477539063e-8` |
| Chunk grid | PASS | Eight chunk records exactly equal |
| Chunk extraction | PASS | Every fixed and moving extended chunk exactly equal |
| Overlap stitching | PASS | Scalar and three-component vector assemblies exactly equal |

The generated `summary.json` records the baseline policy, versions, geometry,
finite-value checks, similarity measurements, transform types, mask values, chunk
coordinates, extended/core shapes, stitching order, clean pinned checkout state,
external output directory, import provenance, and native-artifact cleanup. The
generated `comparison.json` records every tolerance, maximum difference, result,
the effective Cy-ExM thread configuration, and both required diagnostics. Both files
use evidence schema version 3. The baseline records matching source snapshots before
and after capture, confirms that bytecode writes were disabled before ClearEx was
imported, and requires the comparator to reproduce that exact audit.

The one-chunk integration check estimates one real Rigid dry-run affine followed by
one real SyNOnly dry-run warp. It then supplies that same estimated pair at the two
workflows' expensive estimation boundaries. Whole-image and one-chunk orchestration
still perform real ANTs scalar/mask application, affine and warp persistence,
committed-artifact reads, and TIFF output. This is an application-and-persistence
equivalence check for one genuinely estimated pair; it does not claim that two
independent optimizer runs converge to identical transforms.

## Literal dry-run characterization

This report does **not** relabel the matched-low gate as dry-run parity. The baseline
and comparison scripts also execute and store the literal `accuracy="dry run"`
linear and nonlinear calls in both environments. Those outputs remain separate from
the eight matched-policy categories because the pinned ClearEx linear wrapper
changes its transform to Rigid but supplies `reg_iterations`, `smoothing_sigmas`, and
`shrink_factors`. ANTSpyX's Rigid path consumes the affine-prefixed parameter names,
so the legacy call actually retains its default affine schedule. Cy-ExM corrects the
call to `aff_iterations=(1,)`, `aff_smoothing_sigmas=(0,)`, and
`aff_shrink_factors=(8,)`.

ClearEx also forwards the fixed registration mask as `fixed_mask=`. ANTSpyX 0.6.2's
registration signature consumes `mask=`, not `fixed_mask=`. Cy-ExM corrects that
keyword. The baseline records the ANTSpyX signature and all four received call-key
sets: the two literal calls received a nontrivial `fixed_mask` and no `mask`, while
the matched calls received no fixed-mask value. Fixed masks are therefore not claimed
as equivalently consumed; matched mask parity measures the shared moving-mask
nearest-neighbor transformations.

The independently generated literal-dry comparison therefore records an expected
difference, not a parity pass:

| Literal dry-run diagnostic | Within migration tolerance? | Maximum absolute difference |
| --- | --- | --- |
| Affine scalar | Yes | `0.0` |
| Affine mask | Yes, exactly | `0.0` |
| Nonlinear scalar after that affine | No | `0.028033554553985596` |

The parity command requires the `expected_divergence` classification, all three
literal result records, and at least one result outside the approved tolerance. It
exits nonzero if all literal outputs unexpectedly become equivalent or if the
classification evidence is absent. Every required record must also have the complete
comparison schema, finite inputs and maximum difference, and equal actual/baseline
shapes matching the corresponding shape in `summary.json`; malformed diagnostic
records fail rather than count as divergence. This makes the corrected behavior
visible without weakening `rtol=1e-4`, `atol=1e-5` or fabricating literal dry-run
parity. The separate shifted-volume integration test exercises the corrected bounded
kernels, similarity improvement, and a mask containing both zero and one.

## Environments

| Package | ClearEx environment | Cy-ExM environment |
| --- | --- | --- |
| Python | 3.12.11 | 3.12.11 |
| ANTSpyX | 0.6.2 | 0.6.3 |
| NumPy | 1.26.4 | 2.3.5 |
| SciPy | 1.12.0 | 1.15.3 |
| tifffile | 2025.1.10 | 2026.7.14 |

The six recorded native variables were all `1`:
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`, `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and
`VECLIB_MAXIMUM_THREADS`.

## Reproduction

The baseline command rejects a checkout that is not the exact pin, is dirty, imports
ClearEx from elsewhere, or places output inside that checkout. All native ClearEx
registration artifacts are confined to an exception-safe temporary directory under
the explicit output and removed after durable evidence is copied. Temporary-directory
cleanup errors cannot replace a primary registration exception, while a successful
capture still rejects a residual native directory. The script disables bytecode
writes before importing ClearEx and repeats the exact-pin/clean/outside-output audit
after capture; the comparator requires the matching machine record.

From any durable Cy-ExM checkout, use this checkout-relative command:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export ANTS_RANDOM_SEED=20260719

CYEXM_ROOT=$PWD
CLEAR_EX_ROOT=/project/bioinformatics/Danuser_lab/Dean/dean/git/clearex
BASELINE_DIR=/tmp/cyexm-legacy-baseline

git -C "$CLEAR_EX_ROOT" rev-parse HEAD
git -C "$CLEAR_EX_ROOT" status --short

uv run --project "$CLEAR_EX_ROOT" --no-sync python \
  "$CYEXM_ROOT/tests/integration/legacy_baseline.py" \
  --source-root "$CLEAR_EX_ROOT" \
  --output "$BASELINE_DIR"

uv run --project "$CYEXM_ROOT" --no-sync python \
  "$CYEXM_ROOT/tests/integration/compare_legacy.py" \
  --baseline "$BASELINE_DIR" \
  --report "$BASELINE_DIR/comparison.json"
```

The historical Task 8 evidence used the unmerged Phase 1 worktree explicitly:

```bash
cd /project/bioinformatics/Danuser_lab/Dean/dean/git/clearex
uv run --no-sync python \
  /project/bioinformatics/Danuser_lab/Dean/dean/git/Cy-ExM/.worktrees/phase-1-registration/tests/integration/legacy_baseline.py \
  --source-root /project/bioinformatics/Danuser_lab/Dean/dean/git/clearex \
  --output /tmp/cyexm-legacy-baseline-task8-fix2-20260719

cd /project/bioinformatics/Danuser_lab/Dean/dean/git/Cy-ExM/.worktrees/phase-1-registration
uv run --no-sync python tests/integration/compare_legacy.py \
  --baseline /tmp/cyexm-legacy-baseline-task8-fix2-20260719 \
  --report /tmp/cyexm-legacy-baseline-task8-fix2-20260719/comparison.json
```

Observed source and script results were:

```text
2311a873b783477add7391ff00245f76490147f0
PASS legacy baseline commit=2311a873b783477add7391ff00245f76490147f0 output=/tmp/cyexm-legacy-baseline-task8-fix2-20260719
PASS geometry
PASS affine result
PASS nonlinear result
PASS mask interpolation
PASS transform order
PASS chunk grid
PASS chunk extraction
PASS overlap stitching
PASS diagnostic literal dry expected divergence
PASS diagnostic fixed mask corrected defect
```

The baseline took 5.14 seconds and the comparison took 2.95 seconds on the recorded
one-thread environment.

## Intentionally corrected legacy defects

Cy-ExM preserves the established scientific policy but does not reproduce defects
that can return incomplete, stale, or incorrectly transformed results:

1. Linear registration always defines its optional transformed-mask return instead
   of referencing an unbound local when no moving mask is supplied.
2. `ImageRegistration` consumes the nonlinear kernel's actual scalar-and-warp return
   contract rather than treating it as an ANTs registration dictionary.
3. Reuse requires readable artifacts with matching completion metadata instead of
   trusting filename existence.
4. The fixed reference has a dedicated filename instead of colliding with a moving
   round's `nonlinear_registered_1.tif` output.
5. Masks are transformed as scalar label images with nearest-neighbor interpolation
   and thresholding, rather than applying an image transform API to a vector warp.
6. Persisted nonlinear displacement fields are reused through
   `ants.apply_transforms`, not through the affine-transform reader/application API.
7. Stage metrics receive the actual linearly and nonlinearly registered scalar images
   instead of repeatedly measuring the original moving image.
8. The unverified parallel chunk class is excluded; its suppressed worker failures
   cannot be reported as successful Cy-ExM work.
9. Literal affine dry-run parameters use the ANTSpyX names consumed by Rigid, as
   characterized above.
10. Fixed registration masks use ANTSpyX's consumed `mask=` keyword; the pinned
    ClearEx wrappers instead forward ignored `fixed_mask=` kwargs.
11. Linear and nonlinear ANTSpyX calls run inside owned, exception-safe workspaces,
    so partial native output is isolated even when registration or warp composition
    raises. Linear transforms are consumed before its workspace closes. Nonlinear
    registration copies exactly one readable forward warp to a unique disposable
    candidate outside its workspace and transfers ownership to the immediate caller;
    parity and orchestration callers dispose of that candidate after durable
    persistence. Cleanup is best effort, never deletes a returned external native
    source, and cannot replace registration success or the primary exception.
12. Atomic TIFF validation checks declared `ZYX` metadata and effective `ZYX` series
    axes, with only tifffile's leading singleton-Z squeeze accepted explicitly.
13. At external orchestration boundaries, Cy-ExM transposes the first three spatial
    axes between ZYX arrays and native XYZ ANTs images. Vector components remain the
    final axis in physical XYZ order; native-only zero-mask and identity-warp paths
    are deliberately unchanged.
14. ANTs TIFF outputs persist strict versioned shape, origin, spacing, and direction,
    and whole-image cache readers validate all four fields. Cache requests and the
    schema-version-2 chunk manifest record `external-zyx-ants-xyz-v1`.
