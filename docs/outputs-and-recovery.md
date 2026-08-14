# Outputs, restart, and recovery

## Output organization

Each workflow writes a dataset directory below its chosen output root. At the top
level, `scientific-plan.json` identifies the compatible scientific plan and
`workflow-runs/<run-id>.json` records an operational run summary. Round data are
organized under `rounds/`: the reference round has its shared `channels` directory
and any `reference_crop_indices.json`, while a moving round has `registration`,
`channels`, and `logs` directories. For example, the moving registration directory is
`rounds/round-002/registration/`; transformed images are in
`rounds/round-002/channels/`; and a coarse log is
`rounds/round-002/logs/round-002.log`. Fine task logs use
`round-002-<stage>-<item>.log`.

For a reference round, the corresponding directory is `rounds/round-001/`. A moving
registration directory can contain `linear_transform_2.mat`,
`nonlinear_warp_2.nii.gz`, and `nonlinear_registered_2.tif`; each has its own
`.complete.json` record. These are examples of the layout, not a substitute for the
exact inventory below.

The exhaustive output tree, exact filenames, and record fields are in the
[registration artifact reference](registration-artifacts.md).

## Completion records and safe reuse

An artifact is reusable only with its matching `.complete.json` completion record.
Cy-ExM writes expensive stages to temporary sibling files, validates them, promotes
them atomically, and then commits the matching record. A data file alone, a record
alone, corrupt JSON, an unreadable artifact, or a request that no longer matches is
not safe to reuse.

The scientific fingerprint binds scientific choices, input identities and headers,
output mapping, and versioned channel policies. Operational summaries never authorize
artifact reuse. They may describe a run, but completion-record validation is the
restart authority.

## Ordinary resume

Resume is the default: rerun the same compatible request after an interruption or
later-stage failure. Valid earlier affine work and independently committed chunks can
be reused, while missing, stale, or invalid work is recomputed after validation. The
reference preparation remains a hard barrier: a reference failure prevents moving
round submission.

## Force selected work

Use force only to intentionally recompute work within the same compatible scientific
plan. It skips normal cache reuse and invalidates dependent records, but force cannot
bypass a scientific fingerprint mismatch or weaken artifact validation. Forcing the
reference round invalidates all moving descendants because their fixed-input
provenance changes; forcing one moving round can leave valid reference artifacts
reusable.

## Read workflow status

Status is read-only: it validates the scientific plan and authoritative completion
records without importing scientific or scheduler runtimes. It reports the reference
preparation alongside moving rounds. Each known round is `pending`, `partial`,
`complete`, or `invalid`; structured status also identifies complete, incomplete, and
invalid artifact paths.

## Diagnostic images and transformed channels

`registration/nonlinear_registered_<round>.tif` is the normalized registration
diagnostic image. A file in `channels/` is a transformed channel: the original channel
resampled through the persisted transform chain onto fixed-reference geometry and
clipped to uint16 without intensity normalization. Use the transformed channel for
downstream original-intensity analysis rather than substituting the diagnostic image.

## Recover from common failures

- **Affine failure:** correct the resource or input problem and rerun. Earlier valid
  reference and pre-registration metric artifacts remain eligible for validation.
- **Nonlinear failure:** a valid affine transform remains reusable; rerunning the
  compatible request resumes nonlinear work. A stale nonlinear warp is recomputed.
- **Chunk failure:** valid earlier chunk warps remain committed. Rerunning resumes at
  the first missing or stale chunk; a changed chunk size or overlap is a different
  request and must be intentionally recomputed.
- **Incompatible plan:** a different scientific fingerprint requires a new output
  root. Do not edit completion records or use force to make incompatible evidence
  appear valid.
- **Invalid status:** inspect the precise artifact reported by read-only status, fix
  the input, policy, or storage problem, then rerun so completion-record validation
  decides what may be reused.

## Exact artifact reference

For the full output inventory, completion-record fields, stale markers, transform
provenance, and detailed recovery rules, read the
[registration artifact reference](registration-artifacts.md).
