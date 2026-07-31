# Fine-grained registration

Fine-grained execution is an opt-in scheduling mode for Cy-ExM's existing
registration science. It changes where accepted work runs, not fixed and moving
roles, transforms, interpolation, chunk geometry, scalar policies, scientific
fingerprints, or public TIFF outputs. Coarse execution remains the production
default while fine-grained real SLURM production acceptance remains pending.

## Select an execution mode

`parallelism` belongs in the execution profile, never in the scientific dataset
manifest. It accepts exactly one of these forms:

```yaml
parallelism:
  mode: coarse
```

```yaml
parallelism:
  mode: fine_grained
```

Omitting `parallelism` normalizes to `coarse`. The mode is operational evidence but
is excluded from scientific and artifact-cache identity. Switching modes therefore
does not make a scientifically compatible artifact stale.

This complete sequential profile makes the default explicit:

```yaml
schema_version: 1
executor: sequential
parallelism:
  mode: coarse
paths:
  dataset_root: ../data
  output_root: ../results
  worker_local_directory: null
native_threads:
  itk: 1
  blas: 1
```

For local process workers, `workers` is the hard global task cap and
`task_retries` is the number of executor-managed retries after the first attempt:

```yaml
schema_version: 1
executor: local_dask
parallelism:
  mode: fine_grained
paths:
  dataset_root: ../data
  output_root: ../results
  worker_local_directory: ../scratch
native_threads:
  itk: 1
  blas: 1
local_dask:
  workers: 3
  threads_per_worker: 1
  processes: true
  memory_limit: 24GB
  startup_timeout: "30s"
  task_retries: 0
```

The editable dedicated 512-GB-node example uses one worker process and one
scientific task per job:

```yaml
schema_version: 1
executor: slurm
parallelism:
  mode: fine_grained
paths:
  dataset_root: /archive/path/to/dataset
  output_root: /archive/path/to/results
  worker_local_directory: /project/path/to/cyexm-worker
native_threads:
  itk: 1
  blas: 1
slurm:
  queue: 512GB
  workers: 3
  cores_per_worker: 72
  processes_per_worker: 1
  memory: 500GB
  walltime: "72:00:00"
  interface: ib0
  scheduler_interface: ib0
  startup_timeout: "15m"
  task_retries: 0
  job_name: cyexm-registration
  job_extra_directives: []
```

All paths and SLURM resources above are placeholders to edit and validate against
the current site. Native ITK and BLAS threads default conservatively to one; larger
counts require dataset-specific scientific and stability validation.

The companion multi-process profile uses the same three-job cap while providing two
isolated process slots in each job:

```yaml
schema_version: 1
executor: slurm
parallelism:
  mode: fine_grained
paths:
  dataset_root: /archive/path/to/dataset
  output_root: /archive/path/to/results
  worker_local_directory: /project/path/to/cyexm-worker
native_threads:
  itk: 15
  blas: 15
slurm:
  queue: 512GB
  workers: 3
  cores_per_worker: 30
  processes_per_worker: 2
  memory: 500GB
  walltime: "72:00:00"
  interface: ib0
  scheduler_interface: ib0
  startup_timeout: "15m"
  task_retries: 0
  job_name: cyexm-registration-multiprocess
  job_extra_directives: [--exclusive]
```

`slurm.workers` is a job cap; `processes_per_worker` is a process count per
job. This example therefore has at most three jobs/nodes and six task slots.
The 30 job cores divide into 15 cores per process. The semantic 500-GB request
normalizes to 466 GiB for the whole job, then to two conservative 233-GiB Dask
limits. Any division remainder remains job headroom. Coarse mode always requires one
process per job. Fine mode additionally requires even core division and ITK/BLAS
thread counts no larger than the cores per process.

## Coordinator-owned graph

The coordinator, and only the coordinator, expands and admits this task graph:

```text
reference barrier
  -> whole-image affine and round preparation
  -> prepared fixed-grid bundle
  -> nonlinear chunk tasks
  -> deterministic assembly
  -> independent task for every declared moving channel
  -> round validation
```

A `whole_image` moving round uses a whole-image registration task instead of chunk
tasks and assembly. The scientific strategy name remains `serial_chunked` because it
describes the accepted ClearEx-compatible chunk grid, local SyN calls, histogram
policy, and scalar stitching semantics. In fine-grained mode the executor schedules
those canonical chunks independently; this is not a new scientific strategy.

Affine estimation is still a whole-image operation inside preparation. Chunked or
parallel affine estimation is out of scope. Each eligible worker slot executes one
immutable scientific task at a time; a worker may execute multiple tasks
sequentially. Workers never create schedulers, submit descendants, or start nested
multiprocessing.

Ready tasks are admitted fairly in round-robin order across rounds. If only one round
has ready work it may use all available capacity. Local Dask starts the configured
number of process workers, and each advertises one `round_slot`; `workers` is the
configured worker cap.

Fine-grained SLURM initially requests one candidate job for the reference barrier.
Demand-driven scale-up proceeds in whole jobs, using the number of running plus ready
tasks to choose no more than the configured job cap. Every expected process in a job must arrive and
pass preflight as one complete process group before any process in that group receives
its `round_slot`. Admission is one scheduler-side transaction: a failure while
adding any slot restores all resource state from that transaction and commits no
activation provenance for the group. Each process still runs one scientific task at
a time.

Processes in one group have a shared SLURM job ID and hostname but distinct worker
addresses and PIDs. This is expected provenance, not a duplicate identity. A split
job ID, split hostname, repeated address or PID, incorrect process memory, or partial
group fails before scientific admission. Activated jobs are not scaled down while
work remains; the client and complete owned cluster close in order at the end.

## Prepared fixed-grid data

Preparation preserves the legacy ClearEx/ANTs behavior for unequal inputs. Native
fixed, moving registration-channel, and dependent-channel images may differ along
any spatial axis—Z, Y, and X. Affine application is the first fixed-grid boundary:
moving scalar and mask data are resampled to the exact fixed-reference grid before
nonlinear chunking. Final dependent channels are also transformed from their native
grids directly onto that reference grid.

For a moving round `{round}`, fine preparation commits this internal bundle beneath
`registration/round_{round}_prepared/`:

- `fixed_scalar.npy`
- `moving_linear_scalar.npy`
- `fixed_mask.npy`
- `moving_linear_mask.npy`
- `manifest.json`

The arrays are uncompressed, C-contiguous, non-object float32 data. Workers open them
read-only with memory-mapped regional access and copy only an extended chunk into
process-owned memory. The strict prepared manifest's `format_version` is exactly `2`.
Its `pixeltypes` mapping has exactly four canonical roles: `fixed_mask`,
`fixed_scalar`, `moving_linear_mask`, and `moving_linear_scalar`. Each value is the
actual accepted ANTs image type for that role and must be exactly `unsigned char`,
`unsigned int`, `float`, or `double`. It is recorded scientific/cache evidence, not a
user-selected execution option. Missing, extra, unsupported, or request-mismatched
pixel-type evidence fails closed so fine reconstruction cannot silently change
legacy scalar resampling.

The manifest also binds each array identity, fixed-reference grid shape, public XYZ
spacing, origin, direction, affine and source identities, scalar and mask policies,
and native threads. Spatial metadata comes from that manifest; `.npy` alone is not a
scientific image format.

Each chunk commits both
`round_{round}_chunks/chunk_0000_warp.nii.gz` and
`round_{round}_chunks/chunk_0000_registered_core.npy`, with the numeric ID changing
for each canonical chunk. The scalar core is float32 internal progress; the warp is
the local displacement field. Both need ordinary completion records before assembly
can use them.

The example scalar-core filename is `chunk_0000_registered_core.npy`.

Assembly always processes valid chunks in ascending chunk ID, independent of worker
completion order. The legacy grid has expected overlap between nominal output/core
slices; later IDs overwrite earlier values in those overlap regions. Coverage must
have no gaps, and all scalar and three-component displacement values must be finite.
The result is the same public full warp, diagnostic TIFF, and metrics produced by
the coarse serial path.

Internal `.npy` files are retained by default so restart can reuse preparation and
chunk work. They can require substantial additional disk space: four full-volume
float32 prepared arrays plus one float32 scalar core per chunk, completion records,
and local warps. They are memory-mappable acceleration artifacts, not alternatives
to public TIFF exports. Deleting them can make an otherwise incomplete round repeat
preparation or scalar-core reconstruction.

## Completion, cache identity, and status

Prepared arrays, local warps, and scalar cores are internal progress; prepared arrays
and scalar cores never make a round complete. Public completion is determined by the
ordinary accepted outputs and their exact completion records: affine and full warp,
diagnostic TIFF, metrics, and every declared transformed channel. The read-only
status command uses that public authority rather than scheduler evidence.

Coarse and fine-grained modes share scientific fingerprints, public filenames,
ordinary `ArtifactStore` validation, and `StageRequest` identities for their common
scientific/public artifacts. Scalar cores are shared coarse/fine serial-chunk
artifacts; prepared arrays are fine-only. Coarse serial execution invokes the same
chunk stage and commits the same scalar-core request that fine workers use, while
only fine preparation creates the four full-volume arrays and strict prepared
manifest. A valid coarse chunk, full warp, and downstream output can be reused in
fine mode, and a fine result can be reused after switching back to coarse.
`--force-round` and `--force-all` suppress reuse only within an otherwise compatible
scientific plan; force never bypasses a fingerprint mismatch.

Every fine task writes a unique deterministic log under its round `logs/` directory.
These are unique per-task logs rather than a concurrently shared round log.
The workflow-run JSON records the immutable graph, attempts, placement, observed
native threads, failures, cancellation, and cleanup. These records explain an
execution but never authorize scientific reuse. Workers never own scheduler
lifecycle or submit child work.

Fine SLURM uses operational evidence schema version 3. It distinguishes the job cap
from process-slot capacity and records job/process resource derivation, whole-job
scale requests, complete activated groups, shared job/host identities, distinct
worker addresses and PIDs, departed processes, replacements, every placement
attempt, and ordered cleanup. The ArtifactStore remains the only restart authority;
changing `processes_per_worker` does not change a scientific fingerprint,
`StageRequest`, or completion record.

## Failure, cancellation, and restart

A failed task blocks only its descendants. Submitted siblings settle: already
submitted sibling chunks or channels and independent rounds are allowed to finish,
so their valid atomic commits remain useful. A configured retry resubmits the same immutable task through ordinary
artifact validation. No synthetic success is created for descendants that were
never eligible.

A lost process does not automatically invalidate its healthy sibling process. The
departed process is removed from admission, its unfinished task follows ordinary
retry policy, and a replacement process must pass the complete identity/resource
gate before receiving a slot. A lost whole job removes both processes and requires a
complete new process group. If only part of that group arrives, the partial process
group remains ineligible; no half-group scientific work is admitted.

One replacement timeout covers worker arrival, probing, group validation, resource
activation, and final verification. On timeout or malformed replacement, the exact
failure stage is preserved in provenance, active work is cancelled best-effort, and
the client closes before the cluster. A sibling process can remain healthy and finish
an already admitted commit while replacement proceeds. Regardless of executor
evidence, the ArtifactStore remains the only restart authority: valid affine, chunk,
assembly, and channel completion records are reused, while missing or stale work is
resubmitted.

Running two processes per node is not guaranteed to be faster. Memory bandwidth,
filesystem traffic, task size, and scheduler overhead can dominate. Performance is
characterization for a specific dataset and cluster; the focused acceptance report
records timing and memory without imposing a universal speedup threshold.
That report derives realized job/process peaks from raw scale events, strictly
decodes full round/channel scientific comparisons, correlates the loss marker with
the exact task and departure, proves replacement recovery plus healthy-sibling
liveness, and confines every lineage identity to operation-owned jobs.

On operator cancellation the coordinator stops new admission, cancels unfinished
futures, records cancellation evidence, closes the client, and then closes its owned
cluster and configuration. Ordinary completion/stale rules govern interrupted
writes. Restart by repeating the exact command:

```bash
uv run cyexm workflow run path/to/dataset.yaml \
  --execution path/to/execution-fine.yaml
```

An ordinary later run validates and reuses committed preparation, chunks, assembly,
and channel artifacts, then schedules only missing or stale work. To intentionally
repeat one round without changing the scientific plan:

```bash
uv run cyexm workflow run path/to/dataset.yaml \
  --execution path/to/execution-fine.yaml \
  --force-round 3
```

An adaptable output tree is:

```text
<output_root>/<dataset_id>/
├── scientific-plan.json
├── workflow-runs/<run-id>.json
└── rounds/
    ├── round-001/channels/...
    └── round-003/
        ├── registration/
        │   ├── linear_transform_3.mat
        │   ├── nonlinear_warp_3.nii.gz
        │   ├── nonlinear_registered_3.tif
        │   ├── round_3_prepared/
        │   │   ├── fixed_scalar.npy
        │   │   ├── moving_linear_scalar.npy
        │   │   ├── fixed_mask.npy
        │   │   ├── moving_linear_mask.npy
        │   │   └── manifest.json
        │   └── round_3_chunks/
        │       ├── chunk_0000_warp.nii.gz
        │       └── chunk_0000_registered_core.npy
        ├── channels/...
        └── logs/<fine-task-key>.log
```

Every scientific artifact shown also has its own `.complete.json` record.

## Scientific configuration and acceptance boundary

Metric selection is independent of parallelism. Dim regions can provide weaker
registration evidence, as discussed in the
[brightness-dependence investigation](brightness-dependence-investigation.md).
Nonlinear CC uses one scalar voxel radius; omission normalizes to the provisional
radius 4. Changing the metric or radius is a scientific change and can invalidate
nonlinear descendants. Changing coarse versus fine-grained execution is not. The
dataset-specific
[fixed-affine chunk-size characterization](fixed-affine-chunk-size-characterization.md)
uses that fine graph to isolate nonlinear spatial support, including regional
residuals, seam behavior, deformation validity, and four-node runtime. It is
characterization evidence rather than a new production chunk-size default.

Portable real-ANTs/Local-Dask evidence is complete through implementation Task 10,
including scientific equivalence to the sequential fine graph, multiple chunks and
channels, unequal native shapes, resume, cancellation, and cleanup. That evidence is
not a production SLURM claim. Fine-grained real SLURM production acceptance remains
pending the exact-commit two-tier gate: representative multithreaded scientific
equivalence and seeded one-thread strict parity, plus placement, fairness,
failure/recovery, cancellation, cleanup, and performance evidence. The earlier
[Phase 2 acceptance audit](phase-2-acceptance.md) remains the accepted coarse
round-level SLURM baseline.

The multi-process implementation also has a focused, explicitly selected HPC gate
that compares one process with two processes per job, proves same-job overlap,
injects one process loss, and records cleanup, timing, and peak RSS. That real-SLURM
gate remains pending until an operator supplies fresh approved profiles and evidence
storage; portable evidence is not a substitute.
