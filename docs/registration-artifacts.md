# Registration artifacts and restart behavior

Cy-ExM treats a registration output as a committed cache entry only when the artifact
and its own completion JSON are both present, readable, and consistent with the
current request. Filename existence by itself never proves completion.

Use a dedicated output directory for each fixed/moving registration request. The
transform and registered-image names include `{round}`, while the fixed reference and
metric filenames do not; mixing different round requests in one directory therefore
produces a deliberate cache-request mismatch instead of silent reuse.

## Whole-image artifact inventory

For imaging round `{round}`, the workflow may create:

| Artifact | Purpose | Completion record |
| --- | --- | --- |
| `fixed_reference.tif` | uint16 fixed reference used for inspection | `fixed_reference.tif.complete.json` |
| `image_metrics_pre-registration.txt` | similarity before registration | `image_metrics_pre-registration.txt.complete.json` |
| `linear_transform_{round}.mat` | forward affine transform | `linear_transform_{round}.mat.complete.json` |
| `image_metrics_linearly-registered.txt` | similarity after affine application | `image_metrics_linearly-registered.txt.complete.json` |
| `nonlinear_warp_{round}.nii.gz` | full-round forward displacement field | `nonlinear_warp_{round}.nii.gz.complete.json` |
| `image_metrics_nonlinear-registered.txt` | similarity after nonlinear application | `image_metrics_nonlinear-registered.txt.complete.json` |
| `nonlinear_registered_{round}.tif` | final uint16 registered intensity image | `nonlinear_registered_{round}.tif.complete.json` |

When `crop=True`, two additional conditional artifacts are committed before
registration:

| Conditional artifact | Purpose | Completion record |
| --- | --- | --- |
| `fixed_crop_indices_{round}.json` | exclusive fixed ZYX crop bounds | `fixed_crop_indices_{round}.json.complete.json` |
| `moving_crop_indices_{round}.json` | exclusive moving ZYX crop bounds | `moving_crop_indices_{round}.json.complete.json` |

## Serial chunk artifact inventory

Serial chunking creates the whole-image artifacts above plus local nonlinear evidence
under `round_{round}_chunks/`:

| Artifact | Purpose | Completion record |
| --- | --- | --- |
| `chunk_{chunk:04d}_warp.nii.gz` | one readable local forward displacement field | `chunk_{chunk:04d}_warp.nii.gz.complete.json` |
| `manifest.json` | canonical ordered grid and identities of every local warp | `manifest.json.complete.json` |

The round-level `nonlinear_warp_{round}.nii.gz` remains a separate artifact. It is the
full displacement field stitched from all local chunk cores; it is not an alias for a
local warp. In coarse serial execution, registered scalar chunk data is reconstructed
from each committed local warp and stitched in memory. Fine-grained execution
persists an additional internal
`chunk_{chunk:04d}_registered_core.npy` plus its completion record so assembly can
resume without repeating successful worker-local reconstruction.

## Fine-grained internal artifacts

Fine preparation commits four internal float32 arrays and a strict format-version-2
manifest under `round_{round}_prepared/`:

| Artifact | Internal role |
| --- | --- |
| `fixed_scalar.npy` | fixed scalar on the exact reference grid |
| `moving_linear_scalar.npy` | affine-resampled moving scalar on that grid |
| `fixed_mask.npy` | fixed greater-than-zero float mask |
| `moving_linear_mask.npy` | nearest-neighbor affine mask, thresholded on that grid |
| `manifest.json` | identities, geometry, policies, native threads, and exact ANTs pixel types |

These arrays are uncompressed C-contiguous float32 files. Chunk workers validate the
manifest and completion records, memory-map the arrays read-only, and copy only one
extended region. Unequal native source shapes along Z, Y, and X are supported because
affine application is the first exact fixed-grid boundary, matching legacy
ClearEx/ANTs behavior.

Each fine chunk commits both its local warp and scalar core. Assembly ignores worker
completion order: it processes canonical chunks in ascending numeric ID and, in
expected overlapping legacy output slices, later IDs overwrite earlier values.
Every fixed-grid voxel must be covered at least once.

Prepared arrays, scalar cores, and local warps are internal progress. They do not
make a workflow round complete. Public completion still requires the affine, full
warp, diagnostic TIFF, metrics, and every declared transformed channel to pass
ordinary artifact/completion-record validation. The internal `.npy` files remain
retained for restart and can add four full float32 volumes plus scalar cores to disk
usage; they do not change the public TIFF export policy.

Scalar reconstruction is an explicit, versioned part of the request:

- whole-image registration uses
  `whole-global-histogram-match-v1`: apply the committed full warp to the affine
  scalar, then perform one histogram match against the full affine moving image;
- serial chunk registration uses
  `serial-chunk-local-histogram-match-core-stitch-v1`: apply each committed local
  warp, histogram-match against that chunk's affine moving distribution, convert the
  validated core to float32, and stitch the cores without a second global match.

Histogram matching depends on the reference intensity distribution, so matching
each local distribution and then stitching is not generally equivalent to stitching
the warped volume and matching once globally. Preserving this ordering is therefore
a scientific requirement, not merely an output-format choice. In both modes, the
nonlinear mask is independently reconstructed from the committed full warp with
nearest-neighbor interpolation and thresholding at 0.5.

## Completion record contents

Every `<name>.complete.json` uses `schema_version: 1` and contains:

- `request.stage`, `request.method`, and `request.accuracy`;
- `request.inputs`, each with a normalized path, `st_size`, and `st_mtime_ns`;
- `request.geometry`, including shape, origin, spacing, and direction;
- resolved `request.native_threads` for ITK and every BLAS-family variable;
- `request.upstream`, either `null` or the path/size/mtime identity of the immediately
  upstream affine or warp; transitive local-warp and manifest identities are recorded
  in `parameters`;
- stage-specific `request.parameters`, such as round, XYZ pixel size, crop policy,
  interpolation, method schedules, moving geometry, chunk grid, overlap, and local
  warp identities. Every request also records
  `axis_convention_version: external-zyx-ants-xyz-v1`; and
- `artifact`, the normalized path, size, and modification-time identity committed
  after validation.

Each `register()` invocation snapshots both input identities before either image is
loaded. Every whole-image request, local-chunk request, and manifest request created
by that invocation uses the same immutable pair. Cy-ExM verifies the pair after both
loads and immediately before and after every artifact reuse or commit. It aborts with
a registration-stage error if an input changes or disappears; it never silently
reloads changed pixels. If the post-commit check fails, the new completion record is
marked stale. A post-reuse failure leaves the previously valid cache record intact.

Input identities are deliberately stat-based: resolved path, `st_size`, and
`st_mtime_ns`. Cy-ExM does not content-hash source images, so an in-place content
change that preserves all three values is outside this check. Keep source files
immutable for the duration of registration.

The chunk `manifest.json` has its own payload schema in addition to its completion
record. Payload schema version 2 records
`axis_convention_version: external-zyx-ants-xyz-v1`, `round_number`,
`image_shape_zyx`, `chunk_size_zyx`, `overlap_fraction`, ordered chunk bounds, and
each local warp identity.

The effective `scalar_postprocessing_policy` is recorded in the full nonlinear-warp,
nonlinear-registered metrics, and registered-image requests. Every serial local-warp
request records the same serial policy. This binds a scalar descendant to the exact
distribution scope and operation ordering used to produce it.

`request.geometry.shape`, chunk shapes, TIFF arrays, and crop bounds use external
ZYX order. Origin, spacing, direction, ANTs indexes, and displacement component
values remain physical XYZ. The convention version deliberately prevents completion
records or manifests produced with the former ambiguous layout from being reused.

Cy-ExM-generated fixed and final TIFFs embed a strict versioned geometry payload
containing shape, origin, spacing, and direction. Before atomic promotion, the TIFF
writer requires the serialized payload to equal all four source fields exactly.
Tolerance is reserved for comparisons after ANTs has rehydrated the serialized
geometry. Only TIFFs with no `cyexm` marker use fallback geometry, and only when the
effective layout is explicit ZYX, grayscale QYX, or tifffile's reconstructed YX view
of a declared leading singleton Z axis. YXS, SYX, CYX, and other sample/channel
layouts are rejected. Present malformed metadata, including a truncated raw
ImageDescription marker that tifffile cannot shape, fails closed.

## Validation and reuse

On an identical rerun, Cy-ExM checks the following before reuse:

1. Neither the artifact nor its completion record is missing.
2. The completion JSON is readable and has the supported schema.
3. Every request field exactly matches the current normalized request.
4. The current artifact identity matches the identity committed in the record.
5. The artifact-specific reader can reopen the affine, warp, TIFF, metric text, crop
   bounds, or manifest. Whole-image TIFF readers additionally verify reopened shape,
   origin, spacing, and direction against the image that produced the request.

An artifact without its completion record, a record without its artifact, corrupt
JSON, changed input metadata, changed geometry, changed registration policy, changed
thread settings, or an unreadable artifact raises a focused cache-validation error.
Cy-ExM does not guess whether such an entry is safe.

Completion records made before the versioned scalar policy was introduced fail this
exact-request check. This intentionally invalidates serial outputs that may have been
reconstructed with an extra full-volume histogram match. Rerun with `--force` (or
`force_override=True`) to migrate those artifacts; do not edit completion JSON to
claim the new policy.

The `transform_directory.ipynb` example applies the same minimum evidence rule before
reading crop bounds or applying an existing transform: each supplied affine,
nonlinear warp, and optional crop artifact must have a readable completion record,
an exact current `FileIdentity`, the expected stage, and the current
`axis_convention_version`. It then validates the selected records as one provenance
chain before reading bounds or loading an image: affine and nonlinear requests must
name the same two registration inputs and fixed geometry, the nonlinear upstream
must equal the selected affine artifact, and the affine must have no upstream.
Round number, crop policy, and XYZ pixel spacing must match the notebook settings on
every selected record. Crop use is both fixed and moving records or neither; crop
inputs, stages, and fixed/moving roles must belong to the same chain. Finally, the
current `FIXED_IMAGE` identity must equal the recorded fixed input. The original
moving registration channel need not remain available because other moving channels
are the intended transform targets. Copy each artifact together with its
`.complete.json` record; an artifact file by itself, or individually valid artifacts
from unrelated runs, are intentionally unusable in this notebook.

## Stale records and force

When an already committed artifact must not be reused, Cy-ExM atomically moves its
record from `<name>.complete.json` to `<name>.complete.json.stale`. The artifact may
remain in place until its replacement is validated and atomically promoted. A stale
marker is authorization to recompute; it is never proof that the old artifact is
valid. Successful recommit creates a new completion record and removes the stale
marker.

For example, replacing an affine stage makes its downstream linear metrics, full
warp, nonlinear metrics, and registered TIFF stale. A deterministic local chunk
reconstruction failure makes that local warp, the manifest, the full-round warp, and
downstream outputs stale together so transitive evidence cannot be reused.

The workflow scripts expose `--force`, which maps to `force_override=True`. Use it
only when intentionally recomputing the same request. Force skips normal cache reuse,
commits replacements atomically, and invalidates dependent records. It does not make
an unrelated request compatible with old evidence or weaken artifact validation.

## Recovery examples

Recovery normally means rerunning the exact same command with the same inputs and
options. Temporary sibling files are removed automatically.

### Affine failure

If affine estimation or validation fails before commit, there is no valid
`linear_transform_{round}.mat.complete.json`. Correct the underlying resource or
input problem and rerun. Earlier valid fixed-reference and pre-registration metric
artifacts can be reused; the affine stage runs again.

### Nonlinear failure

If the affine completed and nonlinear estimation fails, the affine artifact and its
completion record remain reusable. Rerun the same command: Cy-ExM reconstructs the
affine-aligned moving image from that transform and resumes at nonlinear work. If a
new nonlinear warp was committed but a required reconstruction failed, its record is
marked stale and that warp is recomputed on the next run.

### Chunk failure

If chunk `{chunk}` fails, earlier `chunk_{chunk:04d}_warp.nii.gz` entries with valid
completion records remain committed. The manifest and full-round warp are not
reported complete. Rerun the same serial chunk command: valid earlier local warps are
reopened and applied, and work resumes at the first missing or stale chunk. A changed
chunk size or overlap is a different request and is rejected unless recomputed
intentionally.

If only nonlinear metrics or the final TIFF are stale, serial restart reopens every
valid local warp, reconstructs the same locally matched float32 core stitch, reuses
the valid full warp, and rebuilds only the stale descendants. It never substitutes a
global histogram match during full-warp reuse.
