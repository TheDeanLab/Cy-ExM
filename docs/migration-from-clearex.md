# Migrating registration work from ClearEx

Cy-ExM is an independent extraction of the registration workflow from the read-only
ClearEx source at commit `2311a873b783477add7391ff00245f76490147f0`. It shares
scientific dependencies with ClearEx but has no runtime dependency on that package.
The extraction was compared against the pin before the examples were rewritten; see
[the parity report](legacy-parity-report.md) for the numerical evidence.

## Import map

| ClearEx source | Cy-ExM destination | Migration status |
| --- | --- | --- |
| `clearex.registration.ImageRegistration` | `cyexm.registration.ImageRegistration` | Extracted and characterized |
| `clearex.registration.ChunkedImageRegistration` | `cyexm.registration.ChunkedImageRegistration` | Extracted as the serial reference |
| `clearex.registration.register_round` | `cyexm.registration.register_round` | Extracted and characterized |
| `clearex.registration.ParallelChunkedImageRegistration` | None | Excluded because the implementation was unverified |
| `clearex.registration.linear` | `cyexm.registration.linear` | Extracted with corrected contracts |
| `clearex.registration.nonlinear` | `cyexm.registration.nonlinear` | Extracted with explicit warp ownership |
| helpers in `clearex.registration.common` | `cyexm.registration.transforms` and focused I/O modules | Only required helpers extracted |
| `clearex.registration.pipeline` | None | Remains outside Cy-ExM |
| `clearex.registration.tre` | None | Remains outside Cy-ExM |

Replace imports directly. There is no forwarding compatibility package:

```python
from cyexm.registration import (
    ChunkedImageRegistration,
    ImageRegistration,
    register_round,
)
from cyexm.registration.linear import register_image as register_linear_image
from cyexm.registration.nonlinear import register_image as register_nonlinear_image
```

`ImageRegistration` and `ChunkedImageRegistration` now receive a complete request in
their constructors; call `register()` without per-run arguments. Public `pixel_size`
is `(x, y, z)` in micrometers. New operational settings are keyword-only, including
`native_threads`, accuracy presets, and force behavior.

Known imports in `data_analysis/dean_lab/2025-galecki` remain downstream migration
work. Do not delete the frozen ClearEx API until those consumers have been updated and
verified independently.

## Moved examples

The example migration used the old files as narrative and API inventory, not as code
to copy. Generated output and `downloaded_data.zip` were deliberately excluded.

Scripts:

- `image_registration_class.py` and `register_round_function.py` were consolidated
  into `whole_image_registration.py`.
- `linear_registration.py` became the parameterized `linear_registration.py`.
- `nonlinear_registration.py` became the parameterized
  `nonlinear_registration.py` and now persists its owned warp candidate safely.
- `transform_image.py` became `apply_transforms.py` and applies the full forward
  chain in warp-then-affine order.
- The useful serial workflow from the chunk notebook became
  `chunked_registration.py`.

Notebooks retain their recognizable subjects while using current APIs:

- `apply_warp_transform.ipynb`
- `chunked_nonlinear_registration.ipynb`
- `evaluate_histogram_matching.ipynb`
- `image_registration_class.ipynb`
- `nonlinear_registration.ipynb`
- `transform_directory.ipynb`

Every migrated example takes caller-owned paths, constructs `NativeThreadConfig`,
and avoids ClearEx plotting, statistics, file-operation, and test-download helpers.

## Intentional corrections

The extraction preserves fixed/moving roles, TRSAA and SyNOnly policy, masks,
interpolation, and transform order. It intentionally does not reproduce defects that
could create incomplete or scientifically misleading results:

1. Linear registration defines the optional transformed-mask return even when no
   moving mask is supplied.
2. Whole-image orchestration consumes the nonlinear scalar-and-warp tuple rather
   than treating it as an ANTs result dictionary.
3. Cache reuse requires a readable artifact, an exact request match, and a matching
   artifact identity in a completion record.
4. `fixed_reference.tif` no longer collides with a moving round's registered TIFF.
5. Masks use nearest-neighbor interpolation followed by thresholding at 0.5 rather
   than treating a displacement field as an affine image transform.
6. Persisted nonlinear fields are applied through the displacement-field API.
7. Metrics receive the actual pre-, affine-, and nonlinear-stage scalar images.
8. The unverified process-parallel chunk implementation is not exposed.
9. Literal affine dry-run calls use the Rigid parameter names ANTs actually consumes:
   `aff_iterations`, `aff_smoothing_sigmas`, and `aff_shrink_factors`.
10. Fixed registration masks use ANTs' consumed `mask=` keyword; the pinned wrapper
    forwarded the ignored `fixed_mask=` spelling.
11. Native registration output is isolated in exception-safe workspaces. The
    nonlinear kernel returns one unique disposable warp, transfers it to its immediate
    caller, and never returns a shared native source or canonical cache path. Cleanup
    is best effort and cannot replace the primary error.
12. Atomic TIFF validation requires declared `ZYX` metadata and effective `ZYX`
    series axes, allowing only tifffile's documented leading singleton-Z squeeze.
13. External TIFF, NumPy, chunk, and `ImageGeometry.shape` values use ZYX, while
    native ANTs arrays and indexes use XYZ. Cy-ExM now transposes the first three
    spatial axes explicitly; trailing displacement components remain physical XYZ.
14. Cy-ExM-written ANTs TIFFs persist strict versioned shape, origin, spacing, and
    direction metadata. Plain historical TIFF fallback is limited to explicit ZYX,
    grayscale QYX, and a declared singleton-Z stack reconstructed as YX; sample or
    channel layouts are rejected. Accepted fallback TIFFs use caller XYZ spacing with
    zero origin and identity direction, while a present or truncated raw Cy-ExM
    payload fails closed.
15. Every whole-image and chunk cache request, plus the schema-version-2 canonical
    chunk manifest, records `external-zyx-ants-xyz-v1`, so artifacts created under
    the former ambiguous layout are not reused.
16. Affine inspection reports native transform entries as X, Y, and Z and describes
    translation offsets in physical units rather than mislabeled voxels.
17. `transform_directory.ipynb` requires the matching readable `.complete.json`
    record for every supplied transform and crop artifact, and validates exact file
    identity, stage, and axis-convention version before reuse.
18. Serial chunk scalar output preserves the pinned ClearEx fresh-estimation
    ordering: histogram-match each warped chunk against its local affine-moving
    distribution, convert and stitch the validated cores, and do not apply a second
    full-volume histogram match. Cy-ExM intentionally applies that same local policy
    on restart, correcting ClearEx's inconsistent transform-only reuse behavior.
    Whole-image registration retains its canonical single global match. Versioned
    `scalar_postprocessing_policy` evidence binds the full warp, nonlinear metrics,
    final TIFF, and local warps to that scope. Earlier records lacking this evidence
    fail closed and require an intentional forced recompute.

The direct low-level ANTs parity fixtures intentionally remain native-only and are
not transposed. These corrections apply where orchestration crosses external
NumPy/TIFF/chunk boundaries; native affine algebra, warp component algebra,
zero-mask construction, and identity-warp construction are unchanged.

The literal dry-run mismatch is recorded rather than hidden by weaker tolerances.
The migration gate instead compares the shared, consumed TRSAA-low and SyNOnly-dry
policy, while separate integration tests cover the corrected dry-run behavior.
