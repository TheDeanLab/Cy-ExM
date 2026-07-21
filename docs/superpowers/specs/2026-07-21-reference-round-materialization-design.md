# Reference-Round Materialization Design

**Status:** Approved conversational design pending written-spec review

**Date:** 2026-07-21

## Context

Cy-ExM currently declares only one fixed-reference TIFF through
`reference.channel`. Moving rounds are planned under `rounds/round-NNN`, while no
`rounds/round-001` output is created for the reference round. Each moving-round
registration instead persists its own `registration/fixed_reference.tif` and, when
cropping is enabled, its own fixed-crop bounds. This duplicates the same scientific
reference across moving rounds and leaves downstream consumers without one complete,
shared set of prepared reference and moving channels.

The legacy ClearEx operational workflow handled this manually. Its registration
class wrote the effective fixed registration image as `nonlinear_registered_1.tif`
in a flat save directory. A separate notebook applied the fixed crop to all
reference-round channels and wrote copies beside every transformed moving channel.
That produced a unified downstream dataset, but it lacked a declarative manifest,
atomic artifact lifecycle, exact provenance, scheduler dependency, and structured
round layout.

Cy-ExM will retain the useful legacy behavior while improving its ownership model:
the workflow will prepare all explicitly declared reference channels exactly once
under `rounds/round-001/channels`, and every moving round will consume that shared
prepared registration channel as its fixed image.

The current long-running workflow was started from the preceding schema and code. It
is not modified, cancelled, or migrated by this design. Its outputs are disposable
acceptance data and may be rerun under the new contract after implementation.

## Goals

1. Require every reference-round channel to be explicitly declared in manifest
   schema version 1.
2. Assert that all declared reference channels have the same native geometry as the
   reference registration channel.
3. Compute the fixed-reference crop once and apply the scientifically identical ZYX
   bounds to every reference channel.
4. Materialize standardized, restart-safe reference TIFFs under
   `rounds/round-001/channels` before moving-round registration begins.
5. Make the prepared reference registration channel the only fixed image used by
   workflow-managed moving rounds.
6. Remove workflow-managed per-moving-round `fixed_reference.tif` and fixed-crop
   artifacts.
7. Preserve the direct Phase 1 registration API's existing default behavior for
   callers that do not supply a prepared fixed image.
8. Preserve round-level concurrency after the one prerequisite reference task.
9. Keep output encoding behind the existing output-policy and I/O boundary so future
   formats do not require registration or scheduler redesign.

## Non-goals

- Do not move, rename, or modify source TIFFs.
- Do not discover reference channels through filename globs. The manifest is the
  complete scientific declaration.
- Do not introduce manifest schema version 2 or dual schema maintenance.
- Do not add a public output-format selector in this change; TIFF remains the only
  supported workflow output.
- Do not transform reference channels. They are copied through the standard writer,
  with the shared crop applied when enabled.
- Do not resample a reference channel to conceal incompatible native geometry.
- Do not change moving-round transform order, interpolation, masks, schedules,
  intensity policy, chunking, or round-level parallelism.
- Do not remove the diagnostic `nonlinear_registered_N.tif` moving-round artifact.
- Do not alter, cancel, clean, or reuse the currently running workflow.

## Scientific Invariants

- Public voxel spacing remains XYZ micrometers; arrays and crop bounds remain ZYX.
- The reference registration channel defines the native reference geometry and the
  single fixed crop.
- Every declared reference channel must match that native shape exactly. Origin,
  spacing, and direction must satisfy the repository's established TIFF geometry
  compatibility tolerance, which exists only for serialized floating-point
  representation.
- Reference incompatibility is a hard validation failure. Cy-ExM does not crop,
  pad, or resample incompatible sources into apparent agreement.
- Crop bounds are exclusive `[z0, z1, y0, y1, x0, x1]` bounds computed once from the
  reference registration channel's established robust foreground policy.
- The same bounds are applied without recalculation to every reference channel.
- A cropped reference output derives its origin from the crop offset while retaining
  source spacing and direction.
- Prepared reference outputs use `clip-uint16-no-normalization-v1` and the established
  OME-compatible TIFF geometry policy.
- Native moving registration channels may differ in shape along every spatial axis.
  Affine application still resamples each moving channel onto the prepared fixed
  grid before nonlinear work.
- Dependent moving channels still apply `[warp, affine]` with inversion flags
  `[False, False]` and the prepared reference registration channel as the fixed grid.

## Manifest Schema Version 1

The existing `reference.channel` mapping is replaced rather than supported in
parallel. The strict schema becomes:

```yaml
schema_version: 1
dataset_id: spheroid_001

reference:
  round: 1
  registration_channel: round_1_488_tubulin.tif
  channels:
    - file: round_1_488_tubulin.tif
    - file: round_1_561_example.tif
    - file: round_1_647_example.tif
```

The configuration model adds a frozen `ReferenceChannelSpec` containing one safe
relative POSIX `file`. `ReferenceSpec` contains:

- `round: int`;
- `registration_channel: PurePosixPath`;
- `channels: tuple[ReferenceChannelSpec, ...]`.

Validation requires:

- a nonempty reference channel tuple;
- unique reference source paths;
- the registration channel to appear exactly once;
- unique output basenames, including case-sensitive path identity;
- all paths to resolve safely below `paths.dataset_root`;
- the reference round not to appear among moving rounds;
- the existing scalar 3-D, positive-shape, spacing, axes, identity, and TIFF-header
  contracts for every source;
- reference geometry compatibility as a distinct validation step after all headers
  have been inspected.

The registration channel does not need an interpolation field because reference
channels are never transformed. Unknown legacy key `reference.channel` fails with a
precise migration message showing `reference.registration_channel` and
`reference.channels`. Checked-in examples, README examples, test fixtures, and the
approved spheroid manifest are updated together. No automatic schema migration is
performed.

## Planned Output Layout

For reference round 1 and moving rounds 2 through 4, the workflow layout is:

```text
<output_root>/<dataset_id>/
├── scientific-plan.json
├── rounds/
│   ├── round-001/
│   │   ├── channels/
│   │   │   ├── round_1_488_tubulin.tif
│   │   │   ├── round_1_561_example.tif
│   │   │   └── round_1_647_example.tif
│   │   ├── registration/
│   │   │   └── reference_crop_indices.json
│   │   └── logs/
│   │       └── round-001.log
│   ├── round-002/
│   │   ├── channels/
│   │   ├── registration/
│   │   │   ├── moving_crop_indices_2.json
│   │   │   ├── linear_transform_2.mat
│   │   │   ├── nonlinear_warp_2.nii.gz
│   │   │   ├── nonlinear_registered_2.tif
│   │   │   └── image_metrics_*.txt
│   │   └── logs/
│   └── ...
├── workflow-runs/
└── logs/
```

`reference_crop_indices.json` and its completion record exist only when
`registration.crop` is true. Every reference channel TIFF has its own adjacent
completion record. When cropping is false, the full reference channels are still
rewritten through the standard TIFF writer so all workflow channel outputs share the
same dtype, geometry metadata, validation, and restart policy.

Moving-round workflow directories no longer contain `fixed_reference.tif`,
`fixed_crop_indices_N.json`, or their completion records. The standalone Phase 1 API
may continue to produce those artifacts under its existing default mode.

## Planning and Scientific Identity

Planning creates a frozen `ReferenceTask` in addition to sorted moving `RoundTask`
objects. The reference task contains only immutable, serializable values:

- dataset ID and reference round;
- registration source and prepared output path;
- ordered reference source/output pairs;
- optional shared crop-bounds path;
- voxel spacing and crop policy;
- resolved native-thread configuration;
- round-001 log path;
- force flag.

Each workflow-managed `RoundTask.fixed_image` points to the planned prepared output
for `reference.registration_channel`, even though that artifact does not exist at
planning time. Planning verifies output containment and collisions without creating
it.

The canonical scientific plan records:

- every reference source header and identity;
- the ordered reference-channel declarations;
- the registration-channel role;
- every prepared reference output path relative to the dataset layout;
- the optional shared crop artifact;
- output format and postprocessing policies;
- all existing registration and moving-round evidence.

Reference channel order is scientific identity. Executor choice, resource settings,
and runtime logging remain excluded. A reference source, declaration, crop policy,
geometry, or output-policy change changes the scientific fingerprint.

Plans created by the previous schema do not match the revised schema. Cy-ExM fails
closed with the existing exact-schema/fingerprint behavior and requires a fresh
output root or an operator-archived prior output. It never deletes or rewrites a
prior scientific plan automatically.

## Reference Preparation Component

A focused workflow component owns `prepare_reference_round(task) -> RoundResult`.
It follows the same worker boundary as `run_round_task`:

1. Validate the immutable task and all source/output containment before configuring
   scientific runtimes.
2. Configure and log native ITK and BLAS-family thread settings.
3. Snapshot every source identity before reading data.
4. Load all source TIFFs and verify their geometry against the registration source.
5. When cropping is enabled, reuse or atomically compute the one crop-bounds
   artifact from the registration channel.
6. Apply the exact shared bounds to each reference channel.
7. Derive and verify expected output geometry.
8. Write each output through the central clipped-uint16 TIFF boundary.
9. Reopen and validate shape, origin, spacing, direction, axes, dtype, and identity
   before atomic promotion.
10. Recheck all source identities before reporting reuse or completion.

The crop completion request includes all policy inputs required to prove the bounds:
registration source identity and geometry, robust crop policy, ZYX convention, voxel
spacing, native threads, and bounds encoding. Each channel completion request
includes its own source identity and geometry, the registration source identity,
the crop artifact identity when present, expected output geometry, output format,
and postprocessing policy.

The task validates all native reference geometries before the first channel is
promoted. A later writer or commit failure may leave already committed channels
validly reusable; it must not leave an unrecorded output or claim the reference round
complete.

## Output-Format Boundary

TIFF remains the only public format. This change does not add a speculative plugin
system or unused YAML option. It does avoid embedding TIFF-specific decisions in
reference-task orchestration:

- layout asks the current output policy for the output basename/suffix;
- the reference component delegates encoding and readable-artifact validation to the
  existing TIFF I/O boundary;
- completion requests record the exact output format and postprocessing policy;
- task execution reasons about expected geometry and artifact paths, not TIFF library
  calls.

A future format feature can add a strict manifest field, format-specific path policy,
writer, and reader while retaining the reference task, prerequisite scheduling,
crop provenance, and registration data flow. Until then, `.tif` is required and no
additional format code is implemented.

## Executor and Dependency Model

The executor protocol becomes explicitly staged rather than relying on filesystem
races. It receives one `ReferenceTask` and a tuple of moving `RoundTask` objects. A
single serializable workflow-task runner dispatches the reference task or moving
round task by its frozen task type and returns a `RoundResult` whose round number is
the scientific round.

All executors implement this sequence:

1. Start and validate the selected execution environment.
2. For worker-backed executors, connect and complete all-worker preflight.
3. Submit only the reference task with one `round_slot` resource.
4. Await its terminal result.
5. If it failed or was cancelled, submit no moving tasks and begin ordinary cleanup.
6. If it completed or was reused, submit all pending moving rounds with existing
   independent-failure and bounded-retry behavior.
7. Return results in scientific round order, beginning with the reference round.

Sequential execution performs the same stages in-process. Local Dask and SLURM keep
one client/cluster lifecycle across both stages. They scale to the bounded moving
round concurrency, with a minimum of one worker for reference-only work. The
reference task uses one existing worker; it does not request an additional node
beyond the configured worker pool. Idle workers wait at the prerequisite barrier and
become eligible for moving rounds immediately afterward.

The reference task uses the same serialization, task-key, retry, cancellation,
native-thread, worker identity, and operational-evidence rules as a moving round.
Operational evidence distinguishes `reference-preparation` from
`whole_image`/`serial_chunked` strategy while preserving round number 1. A reference
retry re-enters ordinary artifact validation; Dask opaque retries remain disabled.

## Moving-Round Registration Adaptation

Workflow-managed registration receives an explicit prepared-fixed mode. In this
mode:

- the fixed input is the prepared reference registration channel;
- fixed cropping is skipped because the shared crop is already applied;
- fixed-reference TIFF persistence is skipped;
- fixed-crop artifact creation and reuse are skipped;
- moving cropping remains controlled by `registration.crop` and uses its established
  independent moving foreground bounds;
- pre-registration metrics compare the prepared fixed image with the independently
  cropped native moving registration channel;
- affine estimation accepts the differing native grids and resamples onto the
  prepared fixed geometry;
- existing post-affine, nonlinear, warp, registered-image, and dependent-channel
  fixed-grid checks remain unchanged.

This behavior is explicit rather than inferred from a filename or directory. Direct
Phase 1 construction retains its current default, including current fixed preparation
and persistence, unless the caller deliberately selects prepared-fixed mode.

## Status, Results, and Force Semantics

Artifact-oriented `workflow status` includes the reference round and retains the
existing persistent states `pending`, `partial`, `complete`, and `invalid`. The
reference is complete only when its optional crop artifact and every declared
channel output have valid completion records. Operational `RoundResult` states remain
`reused`, `completed`, `failed`, and `cancelled`; the reference result is round 1 with
strategy/evidence labeled `reference-preparation`.

Force semantics are:

- ordinary run: validate or reuse reference artifacts, then validate or run moving
  rounds;
- `--force-round N` for a moving round: reuse the reference and force only N;
- `--force-round <reference round>`: force the crop and all reference channels, mark
  all moving descendants stale, and force every moving round;
- `--force-all`: force the reference and every moving round.

Forcing the reference is an explicit dependency invalidation, even if rewritten bytes
happen to match. This avoids retaining transforms whose upstream preparation was
deliberately recomputed. Force never bypasses a scientific-plan mismatch.

## Failure Handling

- Missing, unreadable, malformed, non-scalar, non-3-D, spacing-incompatible, or
  geometry-incompatible reference sources fail before reference output promotion.
- A shared crop outside any channel, an empty crop, or output geometry mismatch is a
  hard reference-preparation failure.
- A data file without its exact completion record is incomplete, never reusable.
- A completion record without a readable compatible artifact is invalid.
- Source mutation during preparation aborts before success and preserves only
  previously committed, independently valid artifacts.
- Reference failure or cancellation prevents all moving-round submission.
- Moving-round failure behavior remains independent after the prerequisite succeeds.
- Executor startup, preflight, retry, cancellation, cleanup, and exit-code policies
  remain unchanged; evidence adds the reference prerequisite and barrier outcome.
- Existing output containing the superseded scientific-plan schema is rejected, not
  migrated or deleted.

## Testing Strategy

### Unit coverage

1. Strict YAML accepts the new reference mapping and rejects the legacy key, empty or
   duplicate channels, absent registration membership, basename collisions, and
   unsafe paths.
2. Planning records all reference identities, headers, roles, prepared outputs, crop
   expectation, order, and fingerprint changes without filesystem mutation.
3. Planning rejects a reference channel differing in Z, Y, X, origin, spacing, or
   direction while still allowing arbitrary moving-source shapes.
4. Reference preparation applies one exact crop to all channels and verifies derived
   geometry and uint16 clipping without normalization.
5. Crop and channel artifacts prove atomic commit, exact reuse, incomplete/corrupt
   failure, input mutation, and force behavior.
6. Round tasks point to the prepared fixed output and prepared-fixed registration
   emits no fixed-reference or fixed-crop artifacts.
7. Reference force propagates to every moving task; moving-only force does not touch
   reference artifacts.
8. Status includes round 1 and validates every declared reference artifact.
9. Sequential, local-Dask, and SLURM fakes prove the prerequisite barrier, one
   `round_slot`, bounded scaling, retry, cancellation, no moving submission after
   reference failure, result ordering, evidence, and cleanup.
10. Import-boundary and scheduler-firewall tests continue to prevent coordinator
    scientific imports and portable scheduler access.

### Real-library integration coverage

1. Small deterministic same-geometry reference channels are prepared through real
   tifffile/ANTs behavior with and without cropping.
2. Cropped reference outputs share exact shape, origin, spacing, and direction and
   retain distinct channel values under clip-only uint16 conversion.
3. A prepared fixed channel feeds a moving image whose native Z, Y, and X sizes all
   differ; real affine resampling reaches the fixed grid before bounded serial
   nonlinear chunking.
4. Sequential and local-Dask staged workflows produce equivalent reference and
   moving artifacts and reuse them without rewriting.

### Dataset and HPC acceptance

- Update and validate the approved spheroid manifest with every intended round-1
  channel explicitly listed.
- Mutation-free planning must show round 1 plus moving rounds 2, 3, and 4, exact
  source shapes, prepared output paths, and the reference prerequisite.
- Real SLURM acceptance requires operator approval after the currently running job is
  complete or cancelled. It must prove one reference task finishes before any moving
  task starts, moving rounds then occupy distinct nodes up to the configured bound,
  and no per-round fixed-reference duplicates are written.

## Documentation and Guidance Updates

- Root `AGENTS.md`: shared prepared reference is the workflow fixed-grid authority.
- `src/cyexm/config/AGENTS.md`: strict schema-1 reference declaration and no legacy
  dual parsing.
- `src/cyexm/workflow/AGENTS.md`: immutable reference task, planned nonexistent
  outputs, force propagation, and status ownership.
- `src/cyexm/execution/AGENTS.md`: prerequisite barrier within one executor lifecycle.
- `src/cyexm/registration/AGENTS.md`: explicit prepared-fixed mode and no workflow
  duplicate persistence.
- `src/cyexm/io/AGENTS.md`: current TIFF output policy boundary and future-format
  isolation.
- Applicable test guidance: same-geometry reference channels, identical crops,
  prerequisite ordering, and unequal moving-grid regressions.
- README and examples: revised YAML, output tree, restart/force behavior, TIFF-only
  format statement, and migration error.

## Acceptance Criteria

The feature is accepted only when:

1. schema version 1 explicitly declares the reference registration channel and every
   reference channel;
2. all declared reference channels are proven geometrically identical before output;
3. one crop is calculated and applied identically when enabled;
4. every reference channel is atomically materialized and restart-valid under
   `rounds/round-001/channels`;
5. moving rounds consume the prepared registration channel and create no workflow
   `fixed_reference.tif` or fixed-crop duplicates;
6. reference preparation is a worker-executed prerequisite for every executor;
7. reference failure prevents moving submission, while successful moving rounds
   retain independent failure behavior;
8. reference force invalidates all descendants and moving-only force does not;
9. scientific plans, status, results, operational evidence, README, examples, and
   layered guidance reflect the new contract;
10. TIFF remains the only implemented output while format-specific concerns are
    isolated from task orchestration;
11. focused unit and real-library integration regressions pass;
12. all portable project gates pass;
13. the spheroid configuration validates and plans without scheduler access; and
14. the active legacy-layout workflow and source dataset remain untouched.
