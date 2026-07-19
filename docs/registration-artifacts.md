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
local warp. Registered scalar chunk data is reconstructed from each committed local
warp and stitched in memory. There are no standalone scalar chunk cache files.

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

The chunk `manifest.json` has its own payload schema in addition to its completion
record. Payload schema version 2 records
`axis_convention_version: external-zyx-ants-xyz-v1`, `round_number`,
`image_shape_zyx`, `chunk_size_zyx`, `overlap_fraction`, ordered chunk bounds, and
each local warp identity.

`request.geometry.shape`, chunk shapes, TIFF arrays, and crop bounds use external
ZYX order. Origin, spacing, direction, ANTs indexes, and displacement component
values remain physical XYZ. The convention version deliberately prevents completion
records or manifests produced with the former ambiguous layout from being reused.

Cy-ExM-generated fixed and final TIFFs embed a strict versioned geometry payload
containing shape, origin, spacing, and direction. Their commit validators and cache
readers reopen the TIFF and compare all four fields with serialization-only
tolerance. Only TIFFs with no `cyexm` metadata key use legacy fallback geometry;
present malformed metadata fails closed.

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
