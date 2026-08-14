# Local parallel and HPC execution

## Choose an execution environment

Choose where workflow tasks run after choosing the scientific registration strategy
in the dataset manifest. This is a machine decision, not a third registration
strategy.

| Environment | Start here when | Operational meaning |
| --- | --- | --- |
| Sequential | Establishing a dataset or debugging a scientific configuration | One CLI process runs the reference barrier and moving work in canonical order. |
| Local process-based | A workstation has capacity for isolated worker processes | A local Dask cluster owns bounded worker processes on the same machine. |
| SLURM | A site administrator has confirmed allocation, storage, and scheduler details | An owned SLURM-backed cluster requests worker jobs under site-specific policy. |

Begin sequentially unless the data and resources call for another environment. The
same manifest can be paired with different compatible execution profiles. Read the
[execution-profile reference](execution-profile.md) for strict YAML and resource
validation, and [Choosing a registration strategy](registration-foundations.md) for
the distinct scientific choice.

## Sequential execution

The sequential executor is the numerical and restart reference. It prepares or
reuses the reference round first, then runs pending moving rounds in canonical order
in the CLI process. A reference failure stops the dependency chain; a moving-round
failure does not prevent later independent moving rounds. It opens no Dask scheduler.
There is no silent fallback: a selected executor either completes its own lifecycle
or reports failure.

## Local process-based execution

`local_dask` lazily creates one owned, process-based `LocalCluster`. It starts
`min(pending tasks, configured workers)` workers in coarse mode and exactly the
configured worker cap in fine-grained mode, where preparation determines later graph
width. Every worker receives one `round_slot`. Even when all moving rounds are
reusable, the cluster runs the reference revalidation barrier and submits no moving
work. The reference prerequisite and all subsequent work use that same cluster.

Each coarse round or fine scientific task requires one slot, so only one runs in a
worker even if its Dask `threads_per_worker` is larger. Fine ready tasks are admitted
round-robin across rounds and remain work-conserving when only one round is ready.
The client and cluster are closed for success, failure, startup error, and
interruption. Dask and distributed are locked at version 2025.1.0 for this optional
capability.

## SLURM execution

`slurm` lazily constructs one owned dask-jobqueue `SLURMCluster`; dask-jobqueue is
locked at version 0.9.0. Coarse mode uses one process per job and scales to
`min(pending tasks, configured workers)`. Fine mode can run multiple
process-isolated workers per job. It starts one complete candidate job for the
reference barrier, then scales in whole jobs on demand without exceeding the
configured job cap. Fine candidates advertise no eligible slot until preflight
accepts the complete process group; the coordinator then activates exactly one
`round_slot` per process. Reference preparation and every submitted moving round or
fine scientific task consume one slot. The reference barrier does not create a second
cluster, and active jobs are not scaled down before ordered cleanup.

Every complete process group shares one SLURM job ID and hostname while each process
has a distinct worker address and PID. CPU and Dask-memory evidence is validated per
process as well as per job. Dedicated-node truth depends on site allocation and
exclusivity policy, so it must be confirmed by site-specific real-SLURM acceptance.
The coordinator runs in the process that invokes the CLI; it owns the client,
cluster, task admission, cancellation, and cleanup lifecycle.

## Coarse and fine-grained work

Coarse is the production default: one moving round is one task. Fine-grained is an
opt-in admission mode for the same `whole_image` or `serial_chunked` scientific
strategies. Its coordinator owns the immutable graph; workers do not create
schedulers, submit descendants, or start nested multiprocessing. The graph retains a
hard reference barrier, whole-image affine preparation, fixed-grid constraints, and
ordinary artifact authority. See [Fine-grained execution](fine-grained-execution.md)
for graph, artifact, recovery, and worker-group detail.

## Processes, task slots, and jobs

One worker-backed process exposes one `round_slot`; one coarse round or fine
scientific task consumes one slot. In coarse SLURM, every job therefore has one
process. In fine SLURM, a job can contain a complete group of process-isolated
workers; each process contributes a slot. `slurm.workers` is the job/node cap, not
the slot count, and scale-up and cleanup own whole jobs.

For example, three jobs with two processes each supply at most six task slots. The
30 job cores in the editable multi-process profile divide into 15 cores per process.
The semantic 500-GB request normalizes to 466 GiB for the whole job and then to two
conservative 233-GiB Dask limits; any remainder is job headroom. These example
values are site-specific and editable, not a statement about a current cluster.
The checked-in 512GB values are site-specific and editable.

## Prepare uv on a quota-limited host

Before any `uv` command, replace `/active/project/path` with a directory on the
active project filesystem approved at your site. Do not place these directories in a
quota-limited home directory or archival storage. Make the exports in both the
interactive shell and the job setup that will invoke `uv`, so cache and managed
Python installation use the same active location.

```bash
export UV_CACHE_DIR=/active/project/path/uv/cache
export UV_PYTHON_INSTALL_DIR=/active/project/path/uv/python

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

uv cache dir
uv python dir
uv sync --locked --extra hpc
```

The final two inspection commands should report the substituted active project
locations. This setup concerns package caches and managed Python installations; it
does not replace profile `worker_local_directory` scratch or dataset/output storage.

## Configure site-specific resources

Edit the SLURM profile before submission: queue/partition, account or QoS directives,
walltime, worker and scheduler interfaces, cores, memory, job cap, process count,
shared dataset and output paths, worker scratch, and coordinator location. Confirm
the site accepts each requested setting. The checked-in 512-GB profiles are editable
starting points and require caller-provided images and paths.

Do not assume a storage mount is visible on compute nodes because it is visible in a
login shell. Confirm shared path access and scratch creation in preflight. Choose a
coordinator location whose policy and stability support the intended run.

## Startup and preflight

Configuration and plan errors occur before executor construction. Startup and
preflight errors abort before expensive tasks. In coarse mode, every connected worker
must pass preflight before any registration submission. In fine mode, the initial
candidate and every later newcomer must pass preflight and activation before that
worker is eligible for scientific submission; already activated workers may continue
their admitted tasks during scale-up.

Preflight checks exact Cy-ExM and dependency versions, hostname and SLURM job
identity, dataset/output visibility, scratch creation and cleanup, all six
native-thread values, memory evidence, and scheduler/worker network identity. A
timeout or mismatch closes owned resources and fails as executor startup rather than
falling back to a different executor. During initial startup,
`slurm.startup_timeout` bounds Dask client connection and worker arrival. Initial
cluster construction, scheduler snapshots, and worker-preflight RPC have no Cy-ExM
wall-clock timeout; demand-driven fine scale-up has the same unbounded follow-up
operations after its bounded worker wait. The exact scope is in the
[execution-profile reference](execution-profile.md#startup-timeout-boundary).

## Retry, cancellation, and cleanup

Automatic retries default to zero. A positive `task_retries` value is an
executor-managed resubmission; Dask's opaque future retry count remains zero, and
each attempt re-enters ordinary artifact validation. Use a small bounded value only
for plausible transient failures. Diagnose deterministic registration errors and
out-of-memory failures instead of repeatedly retrying them.

Independent round failures do not cancel other submitted rounds; submitted work
settles and successful artifacts remain committed. A lost fine-process can be
replaced while a healthy sibling continues; a lost whole job replaces and
revalidates the whole process group. A partially arrived replacement group remains
ineligible. During active fine-SLURM recovery, `slurm.startup_timeout` supplies one
absolute replacement/reconciliation deadline for worker health and replacement
snapshots, replacement arrival, the replacement probe broadcast, resource
validation, and whole-group activation with final receipt verification. The budget
does not reset between these steps. A failed transaction records its exact stage,
cancels active work best-effort, and performs ordered cleanup.

On `KeyboardInterrupt` or CLI termination, distributed executors cancel unfinished
futures and close the owned client and cluster. The SLURM adapter also leaves its
temporary Dask configuration. These steps preserve committed artifacts. Committed
chunks and channels survive interruption;
an ordinary later run validates completion records and schedules only missing or
stale work. For structured failure output and exit codes, read the [command-line
reference](cli-reference.md#failures-and-exit-codes). Invalid fine execution evidence
is an exit-3 path; unauthenticated current-run summaries are omitted from durable
`rounds`, pre-run authenticated reuse remains, and valid committed artifacts before
rejection remain restartable.

## Acceptance boundaries

Portable fake-based SLURM construction, job-template rendering, preflight policy,
failure mapping, evidence validation, cancellation, and cleanup are tested. Real
SLURM submission and HPC acceptance completed on the audited lab partition for the
coarse round-level executor. Only tests explicitly selected with `-m hpc` may contact
a scheduler or use representative data; they require an explicit allocation and
caller-approved dataset, execution profile, output root, and external evidence
directory. Portable CI selects `not hpc` and must never invoke scheduler clients.

That coarse acceptance is specific to its tested commit, approved manifest, and
audited profile. It is not an unconditional guarantee for other clusters, resource
profiles, datasets, or scientific policies. Portable real-ANTs/Local-Dask evidence
is complete for fine-grained execution. Fine-grained real-SLURM production acceptance
remains pending its own exact-commit scientific and operational gate; coarse
acceptance must not be read as acceptance of fine chunk/channel placement or
performance. Focused multi-process real-SLURM acceptance remains pending fresh
operator-approved one-process and two-process profiles. Portable checks prove policy
and evidence decoding, not scheduler execution; timing is characterization without a
required speedup threshold.
