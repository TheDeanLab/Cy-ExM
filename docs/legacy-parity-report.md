# ClearEx migration-parity report

## Scope and result

Cy-ExM was compared with the read-only ClearEx extraction source at commit
`2311a873b783477add7391ff00245f76490147f0` on 2026-07-19. The comparison used
one native thread for ITK and every configured BLAS-family runtime, the deterministic
ANTs seed `20260719`, and an identical `16 x 32 x 32` synthetic volume for fixed and
moving input.

The numerical migration gate is explicitly **TRSAA low followed by SyNOnly dry
run**. Those are shared parameters consumed equivalently by both implementations.
All eight migration categories passed. Registration arrays use `rtol=1e-4` and
`atol=1e-5`; chunk geometry and data are exact. Spatial metadata uses an absolute
`1e-7` tolerance only for float32 NIfTI spacing serialization.

| Gate | Result | Evidence |
| --- | --- | --- |
| Geometry | PASS | Shape exact; origin, spacing, and direction within `1e-7` |
| Affine scalar | PASS | Maximum absolute difference `0.0` |
| Nonlinear scalar | PASS | Maximum absolute difference `0.0` |
| Mask interpolation | PASS | Affine and nonlinear masks exactly equal and binary |
| Transform order | PASS | Warp then affine, both uninverted; maximum difference `5.960464477539063e-8` |
| Chunk grid | PASS | Eight chunk records exactly equal |
| Chunk extraction | PASS | Every fixed and moving extended chunk exactly equal |
| Overlap stitching | PASS | Scalar and three-component vector assemblies exactly equal |

The generated `summary.json` records the baseline policy, versions, geometry,
finite-value checks, similarity measurements, transform types, mask values, chunk
coordinates, extended/core shapes, and stitching order. The generated
`comparison.json` records every tolerance, maximum difference, result, and the
effective Cy-ExM thread configuration.

## Literal dry-run characterization

This report does **not** relabel the matched-low gate as dry-run parity. The baseline
and comparison scripts also execute and store the literal `accuracy="dry run"`
linear and nonlinear calls in both environments. Those outputs are intentionally
excluded from the overall parity verdict because the pinned ClearEx linear wrapper
changes its transform to Rigid but supplies `reg_iterations`, `smoothing_sigmas`, and
`shrink_factors`. ANTSpyX's Rigid path consumes the affine-prefixed parameter names,
so the legacy call actually retains its default affine schedule. Cy-ExM corrects the
call to `aff_iterations=(1,)`, `aff_smoothing_sigmas=(0,)`, and
`aff_shrink_factors=(8,)`.

The independently generated literal-dry comparison therefore records an expected
difference, not a parity pass:

| Literal dry-run diagnostic | Within migration tolerance? | Maximum absolute difference |
| --- | --- | --- |
| Affine scalar | No | `0.21112346649169922` |
| Affine mask | No | `1.0` |
| Nonlinear scalar after that affine | No | `0.21028149127960205` |

This characterization makes the corrected API behavior visible without weakening
the approved `rtol=1e-4`, `atol=1e-5` numerical gate or fabricating dry-run parity.
The separate shifted-volume integration test exercises the corrected bounded kernels,
similarity improvement, and a mask containing both zero and one.

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

The implementation was unmerged when this evidence was generated, so the commands
use the explicit Phase 1 worktree path rather than the main Cy-ExM checkout. The
baseline command writes only to its explicit `/tmp` output and did not change the
ClearEx worktree.

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

git -C /project/bioinformatics/Danuser_lab/Dean/dean/git/clearex rev-parse HEAD

cd /project/bioinformatics/Danuser_lab/Dean/dean/git/clearex
uv run --no-sync python \
  /project/bioinformatics/Danuser_lab/Dean/dean/git/Cy-ExM/.worktrees/phase-1-registration/tests/integration/legacy_baseline.py \
  --output /tmp/cyexm-legacy-baseline-task8-final-20260719

cd /project/bioinformatics/Danuser_lab/Dean/dean/git/Cy-ExM/.worktrees/phase-1-registration
uv run --no-sync python tests/integration/compare_legacy.py \
  --baseline /tmp/cyexm-legacy-baseline-task8-final-20260719 \
  --report /tmp/cyexm-legacy-baseline-task8-final-20260719/comparison.json
```

Observed source and script results:

```text
2311a873b783477add7391ff00245f76490147f0
PASS legacy baseline commit=2311a873b783477add7391ff00245f76490147f0 output=/tmp/cyexm-legacy-baseline-task8-final-20260719
PASS geometry
PASS affine result
PASS nonlinear result
PASS mask interpolation
PASS transform order
PASS chunk grid
PASS chunk extraction
PASS overlap stitching
```

The baseline took 5.29 seconds and the comparison took 3.71 seconds on the recorded
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
