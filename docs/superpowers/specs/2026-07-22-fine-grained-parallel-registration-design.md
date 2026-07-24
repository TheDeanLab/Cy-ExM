# Fine-grained parallel nonlinear registration

## Status

This design was approved on 2026-07-22. Implementation has not started.

The design adds an opt-in, coordinator-owned task graph for parallel nonlinear
chunks and moving-channel transforms. It preserves the accepted coarse round
executor as the default and the serial `ChunkedImageRegistration` implementation
as the numerical reference.

## Context

Cy-ExM currently exposes one scheduler task per moving round. A worker runs affine
registration, every nonlinear chunk, full-warp assembly, and every moving-channel
transform serially. Round-level concurrency works well when a dataset has many
moving rounds, but a dataset with only a few rounds cannot use additional available
nodes during the dominant nonlinear stage.

The CC-radius characterization found that nonlinear registration dominated workflow
runtime. Moving-channel transformation took approximately 25 seconds per channel in
that experiment. Parallel nonlinear chunks are therefore the primary performance
opportunity; parallel channels are included because they have a clean independent
artifact boundary and may matter more for other datasets.

The accepted SLURM contract uses one worker process per dedicated 512-GB node. This
design retains that contract. Multiple worker processes per node, smaller
stage-specific allocations, and parallel affine estimation require separate future
characterization.

## Goals

1. Reduce the wall time of one chunked dataset while bounding its total worker count.
2. Run independent nonlinear chunks concurrently without changing their scientific
   definitions or canonical assembly.
3. Run each declared moving-channel transform independently after its round warp is
   accepted.
4. Preserve coarse and fine-grained cache compatibility, restart behavior, transform
   order, image geometry, interpolation, and scientific fingerprints.
5. Let independent submitted work settle after a sibling failure so valid artifacts
   remain reusable.
6. Schedule ready work fairly across moving rounds while remaining work-conserving.
7. Keep the coordinator as the only owner of scheduler submission, cancellation,
   scaling, and cleanup.
8. Preserve a future boundary at which a different implementation could produce the
   same whole-image affine-preparation outputs.
9. Explain the task graph, intermediate artifacts, resource behavior, and restart
   behavior in user documentation.

## Non-goals

- Do not parallelize, tile, approximate, or otherwise change affine estimation.
- Do not add or export `ParallelChunkedImageRegistration`.
- Do not change the `serial_chunked` scientific strategy name or its algorithm.
- Do not divide whole-image nonlinear registration into spatial tasks.
- Do not run multiple worker processes on one node.
- Do not add heterogeneous or stage-specific SLURM profiles.
- Do not add cross-dataset scheduling, job arrays, cloud execution, or automatic
  resource optimization.
- Do not add automatic deletion of intermediates whose absence would weaken cache
  verification or restart.
- Do not make scheduler state, futures, task results, or operational evidence
  authorize artifact reuse.

Chunked affine estimation is a possible distant extension. The prepared-input
contract must not preclude it, but this milestone provides no chunked affine API,
implementation, configuration, or scientific claim.

## Selected architecture

The selected design is a coordinator-owned flat dependency graph. Workers execute
ordinary, immutable tasks and never construct a client, submit a child task, or
manage another worker.

For a `serial_chunked` moving round, the graph is:

```text
reference preparation
        |
        v
whole-image crop, masks, metrics, and affine estimation
        |
        v
atomic fixed-grid prepared-input bundle
        |
        +--------+--------+--------+
        v        v        v        v
      chunk 0  chunk 1   ...     chunk N
        +--------+--------+--------+
                         |
                         v
              canonical warp assembly,
              diagnostic image, and metrics
                         |
        +----------------+----------------+
        v                v                v
     channel 0        channel 1        channel N
        +----------------+----------------+
                         |
                         v
                  round validation
```

For a `whole_image` round, the existing whole-image registration remains one
scientific task. Its accepted transform then fans out into independent channel
tasks before final round validation.

The existing coarse graph remains the default:

```text
reference preparation -> one monolithic task per moving round
```

Coarse mode continues to call the existing round runner. Fine-grained mode uses the
new task graph. Both modes produce and validate the same public scientific outputs.

### Alternatives rejected

**Stage-wide barriers** would finish all affine tasks before any chunks, all chunks
before any assembly, and all assemblies before any channels. It is simpler, but one
slow round would block ready work from faster rounds and leave nodes idle at stage
boundaries.

**Nested Dask inside a round worker** would preserve the appearance of the current
round API, but complicate resource accounting, serialization, cancellation, failure
collection, and deadlock avoidance. It conflicts with the accepted single scheduler
owner and one task slot per worker.

## Configuration contract

Parallelism is an execution choice, not a scientific dataset choice. The dataset
manifest and its schema-version-1 fingerprint remain unchanged.

The strict execution profile accepts an optional exact section:

```yaml
parallelism:
  mode: coarse
```

or:

```yaml
parallelism:
  mode: fine_grained
```

Omitting `parallelism` normalizes to `mode: coarse`. No other keys or mode spellings
are accepted. Checked-in execution examples declare the mode explicitly.

The setting is part of execution identity and workflow-run evidence. It is excluded
from scientific identity and artifact cache identity. Existing
`local_dask.workers` and `slurm.workers` values remain the hard global concurrency
caps. The first version has no per-stage or per-round concurrency settings.

Fine-grained mode is supported by the sequential, local-Dask, and SLURM executors.
Sequential fine-grained execution runs the same graph one task at a time and is the
direct scheduling reference for distributed execution.

## Task and component boundaries

Fine-grained task models are immutable, exactly validated, and JSON serializable.
They identify their dataset, round, task type, scientific inputs, planned outputs,
native threads, force policy, and task-specific identity. A chunk task additionally
contains one canonical chunk definition. A channel task contains one source,
destination, and interpolation mode.

The internal task types are:

1. `ReferenceTask`, unchanged as the hard workflow prerequisite.
2. A round-preparation task that performs the accepted whole-image moving crop,
   masks, pre-registration metrics, affine estimation, affine metrics, and prepared
   fixed-grid persistence.
3. One nonlinear-chunk task per canonical chunk.
4. One chunk-assembly task per chunked round.
5. One whole-image registration task for each `whole_image` round.
6. One channel-transform task per declared moving channel, including the designated
   registration channel.
7. One round-validation task that returns the terminal round result.

These models remain under the workflow namespace. Low-level prepared-array,
single-chunk, and assembly operations remain focused registration internals rather
than package-root or `cyexm.registration` exports. The stable
`ChunkedImageRegistration` class remains serial and supported.

The existing executor protocol must be extended or composed rather than duplicated
with parallel executor classes. Sequential, local-Dask, and SLURM adapters consume
the same ready-task dispatcher. Scheduler-specific code continues to own only
scheduler lifecycle and operational evidence.

## Prepared fixed-grid inputs

The round-preparation task commits one immutable internal bundle after accepting the
affine transform. The bundle contains:

- `fixed_scalar.npy`;
- `moving_linear_scalar.npy`;
- `fixed_mask.npy`;
- `moving_linear_mask.npy`; and
- `manifest.json`.

The bundle lives in the round registration directory under
`round_<round>_prepared/`. Every file is written atomically and has ordinary
completion evidence. The manifest records the exact fixed and native moving input
identities, moving-crop identity when present, affine identity, array identities,
shape, dtype, public XYZ spacing, origin, direction, scalar and mask policies,
native-thread evidence, and format version.

Arrays use fixed-reference ZYX geometry. Scalars preserve the floating values used
by the accepted registration path. Masks preserve the accepted greater-than-zero,
affine nearest-neighbor, float-mask, and threshold policies. No uint16 clipping,
normalization, or TIFF export policy is applied to prepared arrays.

The `.npy` format is an internal acceleration format, not a new scientific export
format. Each file is an uncompressed, C-contiguous, non-object float32 array read
with `allow_pickle=False`. This supports exact dtype and shape validation plus
memory-mapped regional reads. A chunk worker opens the arrays read-only and copies
only its extended chunk into process-owned memory. It does not load all four complete
volumes.

The manifest and completion records provide spatial meaning that `.npy` does not
encode itself. A chunk worker reconstructs ANTs images using the exact cropped
origin derived from the fixed geometry, direction, spacing, and extended ZYX start.

Prepared inputs are restart dependencies and remain retained by default. Deleting
them may require preparation to run again. No cleanup command may claim to preserve
full restartability unless it can still validate every downstream dependency.

## Nonlinear chunk task

The canonical chunk grid continues to come from `compute_chunk_grid` with the
manifest's ZYX chunk size and overlap fraction. Chunk IDs, extended bounds, core
bounds, and ordering remain unchanged.

Each chunk task:

1. validates the prepared manifest, array identities, affine identity, and its exact
   chunk definition;
2. memory-maps and copies the four extended regions;
3. reconstructs fixed-grid ANTs images with exact cropped geometry;
4. runs the unchanged local SyN policy, including metric, scalar CC radius or Mattes
   bins, masks, schedules, histogram matching, and native threads;
5. validates finite scalar and displacement outputs, float32 policy, geometry, and
   component count;
6. atomically commits the existing
   `round_<round>_chunks/chunk_<id>_warp.nii.gz`; and
7. atomically commits
   `round_<round>_chunks/chunk_<id>_registered_core.npy`.

Each artifact receives its own completion record. A task is successful only when
both artifacts and their records are valid. A partial commit is marked stale and
does not authorize assembly.

The local-warp scientific request remains compatible with the current serial
implementation. Its authoritative inputs remain the native fixed and moving source
identities, accepted affine identity, fixed geometry, chunk definition, masks,
metric policy, and registration policy. The prepared bundle is a validated derived
route to those inputs, not a new scientific choice. Scheduler mode, job identity,
completion order, and worker count are excluded.

The scalar-core request additionally identifies its local warp and scalar
postprocessing policy. A missing scalar core may be reconstructed from a valid local
warp and prepared inputs without rerunning nonlinear estimation.

## Canonical assembly

**Erratum approved 2026-07-23:** The canonical legacy chunk grid has overlapping
nominal core/output slices when overlap is nonzero. The earlier exactly-once,
non-overlapping wording was incorrect. Assembly must preserve ClearEx behavior:
process valid chunks in ascending chunk ID and let later IDs deterministically
overwrite earlier values in overlap regions.

Assembly starts only when every expected local warp and registered scalar core is
valid. It allocates full fixed-grid scalar and vector destinations and processes
chunks in ascending chunk-ID order, regardless of worker completion order.

For each chunk, assembly copies the registered scalar core and displacement core into
the canonical legacy core/output slice. Coverage validation proves:

- every fixed-grid voxel is written at least once;
- no gap remains, while expected overlap is accepted;
- each local displacement has exactly three components;
- every scalar and displacement value is finite; and
- all chunk artifacts match the same prepared manifest, affine, grid, and scientific
  policy.

Assembly commits the same full nonlinear warp, chunk manifest, diagnostic registered
TIFF, nonlinear metrics, and completion records as coarse serial execution. It
reconstructs and thresholds the nonlinear mask under the existing policy. The
canonical ascending-ID overwrite rule makes assembly independent of task completion
order while preserving legacy scalar and displacement values at chunk boundaries.

If an already valid full warp and its downstream registration artifacts are reusable,
fine-grained execution does not require chunk resubmission merely because the
scheduler mode changed. If only local warps are valid, fine-grained execution
reconstructs missing scalar cores and assembles without repeating SyN.

## Channel tasks and final validation

After a round's affine and full warp are accepted, every declared moving channel is
ready independently. Each task calls the accepted one-channel transform lifecycle:

- native source grids may differ along Z, Y, and X;
- the fixed image remains the transform reference;
- transforms are applied in warp-then-affine list order with both inversion flags
  false;
- interpolation remains the declared `linear` or `nearestNeighbor`;
- outputs must exactly match fixed-reference shape, origin, spacing, and direction;
  and
- TIFF output and completion-record policy remain unchanged.

Channel tasks write distinct outputs and logs, so no shared artifact requires
concurrent mutation. Final round validation runs only after every channel reaches a
terminal state. It applies the same completion-evidence checks as the coarse round
runner and returns the ordinary `RoundResult`.

## Fair scheduling and worker lifecycle

The coordinator maintains dependency counts and a ready queue grouped by round. It
admits ready tasks round-robin across rounds. Within a round, canonical task order is
used only as a deterministic admission tie-breaker; it does not constrain completion
order.

The dispatcher is work-conserving:

- when multiple rounds have ready work, each receives opportunities in round-robin
  order;
- when only one round is ready, it may occupy every available worker; and
- the executor never exceeds its configured global worker count.

Each submitted scientific task consumes the existing `round_slot` resource token.
The historical token name remains an internal capacity contract; in fine-grained
mode it means one scientific workflow task, not necessarily one complete round.
Fine-grained SLURM retains one Dask worker process, one advertised token, and one
scientific task per dedicated 512-GB job. Worker CPU, memory, native-thread, scratch,
version, and network preflight remain mandatory.

SLURM scaling is demand driven. The reference barrier starts with the minimum worker
set. As additional ready work increases demand, the owned cluster may scale up to
`slurm.workers`. An unvalidated worker cannot offer an eligible `round_slot` to
scientific tasks. Every newly connected worker must pass the complete preflight
before the coordinator makes its slot eligible and submits work that could be
assigned to it. The dispatcher must not leave an unrestricted scheduler backlog that
could escape this gate. Scale-up, resource eligibility, connected and validated
worker identities, task placement, and cleanup are recorded as operational evidence.
Fine-grained execution never silently falls back to coarse or sequential execution.

## Failure, retry, cancellation, and resume

Failure propagation follows graph dependencies:

- reference failure prevents all moving submission;
- round-preparation failure blocks only that round's chunks or whole-image
  descendants;
- a failed chunk prevents assembly for its round but does not cancel already
  submitted sibling chunks;
- assembly failure prevents that round's channel submission;
- a failed channel does not cancel submitted sibling channels; and
- final validation fails a round when any required terminal artifact is absent or
  invalid.

The coordinator lets independent submitted work settle and reports every terminal
failure. It does not create synthetic results for descendants that were never
eligible for submission.

Bounded retries remain executor-managed. A retry re-enters ordinary artifact
validation, so valid preparation, chunk, assembly, or channel work is reused.
Resume derives ready work from completion evidence and schedules only missing or
stale tasks.

On operator cancellation, the coordinator stops new admission, cancels unfinished
futures, lets ordinary atomic/stale rules govern partial files, writes cancellation
evidence, closes the client, and then closes its owned cluster and configuration.
Committed scientific artifacts remain available for resume.

## Logs and operational evidence

Concurrent tasks must not append to one shared round log. Every task writes a unique
log beneath the round log directory using its deterministic task identity. The
coordinator writes the workflow-run record and a concise round summary after
terminal results settle.

Fine-grained operational evidence records:

- normalized parallelism mode and global worker cap;
- deterministic graph and task keys;
- task type, round, chunk ID or channel identity, and dependency keys;
- admission, start, end, duration, attempt, status, host, process, and SLURM job ID;
- requested and observed resources and native-thread values;
- worker preflight and scale events;
- failures, cancellations, and unsubmitted descendants; and
- client, cluster, configuration, scratch, and owned-job cleanup.

Operational evidence does not enter scientific fingerprints and cannot authorize
artifact reuse.

## Scientific invariants

Fine-grained execution preserves:

- fixed and moving roles;
- public micrometer spacing in XYZ order and array data in ZYX order;
- native source shapes that may differ independently along all spatial axes;
- affine application as the first fixed-grid boundary;
- exact fixed-reference geometry for nonlinear work and transformed channels;
- masks based on values greater than zero;
- TRSAA affine registration and the accepted affine metric and schedule;
- SyNOnly nonlinear registration with exact CC/radius or Mattes/bin policy;
- serial chunk grid, overlap, local registration, histogram matching, and core
  stitching semantics;
- scalar linear interpolation and mask/label nearest-neighbor interpolation;
- warp-then-affine channel transform order;
- ANTs origin, spacing, direction, and vector components;
- one-thread defaults unless environment or explicit configuration overrides them;
  and
- atomic artifacts with validated completion evidence as the only restart
  authority.

Any change to these invariants is a separate scientific change, not a scheduler
optimization.

## Testing

### Unit and policy tests

Tests cover:

- strict `parallelism` parsing, omitted-value normalization, canonical serialization,
  and rejection of unknown keys or values;
- exclusion of execution mode from scientific identity and inclusion in execution
  evidence;
- deterministic task graph construction without filesystem or scheduler mutation;
- ready-queue dependency accounting, bounded admission, round-robin fairness, and
  work conservation;
- exact task-model validation and serialization;
- prepared-array atomic commit, identity, dtype, geometry, mask policy, and
  memory-mapped regional reads;
- unchanged local-warp cache identity across coarse and fine-grained modes;
- reconstruction of a missing scalar core without rerunning SyN;
- canonical assembly under permuted completion order;
- gap, unexpected duplicate artifact, shape, component, dtype, finiteness, identity,
  and geometry failures, while canonical grid overlap remains valid;
- independent chunk and channel failure collection;
- retry, cancellation, unsubmitted descendants, resume, and force propagation;
- unique task logs and complete operational evidence;
- scheduler resource caps and all-worker preflight during scale-up; and
- import contracts proving that models and planning do not import scientific or
  scheduler runtimes.

### Integration tests

Small real ANTs cases cover:

- coarse serial versus fine-grained sequential output;
- fine-grained sequential versus process-based local Dask output;
- unequal fixed, registration-channel, and dependent-channel shapes along Z, Y, and
  X;
- multiple nontrivial nonlinear chunks;
- completion-order-independent full warp and diagnostic scalar assembly;
- channel transform order and interpolation;
- reuse when switching from coarse to fine-grained and back;
- failure after some chunks commit, then exact resume;
- failure after some channels commit, then exact resume; and
- cancellation with valid committed work preserved.

### HPC scientific and operational acceptance

Production readiness requires a fresh, exact-commit run against an operator-approved
multi-chunk dataset and execution profile.

The established two-tier gate applies:

1. representative multithreaded coarse-versus-fine execution must satisfy the
   accepted scientific-equivalence metric and structural tolerances; and
2. seeded one-thread coarse-versus-fine execution must satisfy strict numerical and
   artifact parity.

The run must also demonstrate:

- concurrent nonlinear chunk tasks on distinct SLURM jobs and hosts;
- concurrent channel tasks on distinct SLURM jobs and hosts;
- the configured global worker cap;
- round-robin opportunities when at least two rounds have ready chunks;
- injected chunk failure with sibling settlement and successful resume;
- injected channel failure with sibling settlement and successful resume;
- cancellation, successful resume, and exact owned-job cleanup;
- exact prepared-array and fixed-grid geometry;
- no output outside owned roots; and
- at least a 20% reduction in nonlinear-stage wall time relative to otherwise
  identical coarse execution on the approved benchmark.

The acceptance report records end-to-end and per-stage wall time, task durations,
approximate node-hours, peak worker memory, scheduler/startup overhead, worker
utilization, and the distribution of chunk durations. The wall-time criterion does
not replace numerical or scientific acceptance.

## Documentation

Implementation updates:

- `README.md`;
- the workflow guide;
- registration-artifact documentation;
- execution-profile field documentation;
- checked-in sequential, local-Dask, and SLURM YAML examples;
- a dedicated fine-grained execution and operations page; and
- the relevant layered `AGENTS.md` files.

Documentation explains:

- coarse versus fine-grained execution;
- why `serial_chunked` remains the scientific strategy name;
- the coordinator-owned task graph;
- what prepared `.npy` arrays and scalar cores contain;
- why these internal arrays do not change TIFF export policy;
- the one-task-per-512-GB-node initial resource contract;
- fair scheduling and the global worker cap;
- cache compatibility when switching modes;
- intermediate retention, disk-space implications, and restart behavior;
- unique task logs and status/evidence interpretation;
- failure, retry, cancellation, and resume examples;
- the two-tier scientific acceptance boundary; and
- future possibilities without claiming support for multiple workers per node,
  heterogeneous resources, or chunked affine estimation.

## Acceptance criteria

The implementation is complete only when:

1. omitted execution parallelism preserves current coarse behavior;
2. fine-grained sequential, local-Dask, and SLURM execution use the same immutable
   task graph;
3. workers never create schedulers or submit child tasks;
4. nonlinear chunks and moving channels run independently under the global cap;
5. coarse and fine-grained modes share scientific and public artifact identity;
6. prepared arrays support validated regional reads and exact fixed-grid
   reconstruction;
7. assembly is canonical and completion-order independent;
8. failure, retry, cancellation, force, and resume preserve ordinary artifact
   authority;
9. portable unit, integration, formatting, lint, type, and repository-policy gates
   pass;
10. the exact-commit two-tier HPC acceptance and performance gate passes;
11. user documentation and examples describe the implemented behavior accurately;
    and
12. coarse execution remains the documented production default until the
    fine-grained acceptance report is complete.
