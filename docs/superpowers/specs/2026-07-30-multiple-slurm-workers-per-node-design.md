# Multiple SLURM workers per node design

**Status:** Approved design; implementation and real-HPC acceptance are pending.

**Date:** 2026-07-30

## Context

Cy-ExM currently maps one `dask-jobqueue` job to one Dask worker process and one
`round_slot`. That mapping is deliberately enforced by configuration, rendered-job
validation, worker preflight, identity checks, scaling, replacement handling,
execution evidence, documentation, and HPC acceptance. It is safe but leaves CPU and
memory capacity unused when a fine-grained nonlinear chunk or channel task cannot use
an entire 512-GB node efficiently.

The approved extension permits several process-isolated Dask workers in one SLURM
job and therefore on one allocated node. It applies only to coordinator-owned
fine-grained execution. Coarse tasks retain full-volume round state and remain one
process per SLURM job because admitting several coarse rounds to one node would create
uncharacterized memory pressure.

This is an operational execution change. It must not alter registration algorithms,
the scientific manifest, scientific fingerprints, task definitions, transform
ordering, interpolation, masks, metrics, artifact requests, or cache identity.

## Goals

- Allow fine-grained SLURM profiles to launch a configurable number of Dask worker
  processes in each job.
- Keep `slurm.workers` as the maximum number of SLURM jobs or nodes.
- Use the existing `slurm.processes_per_worker` field as the number of Dask worker
  processes launched by each job.
- Divide CPU and Dask-managed memory predictably among those processes without native
  thread oversubscription.
- Scale in whole SLURM jobs while scheduling one immutable fine task per validated
  Dask process.
- Preflight, activate, replace, and record several process identities that
  legitimately share one SLURM job ID and hostname.
- Preserve exact restart behavior through the existing artifact store.
- Provide portable validation and an explicitly selected real-HPC gate for the
  initial two-process-per-node configuration.
- Document resource derivation, failure handling, provenance, and appropriate use for
  an educated scientific audience.

## Non-goals

- Multiple worker processes for coarse SLURM execution.
- Nested Dask graphs, worker-owned submission, or multiprocessing inside registration
  kernels.
- Parallel affine estimation.
- A new executor, scheduler lifecycle, scientific mode, execution-profile schema, or
  scientific-manifest schema.
- Automatic selection of process count from image size or available memory.
- A universal performance-speedup requirement.
- Changes to chunk geometry, overlap, ANTs settings, native thread defaults, or
  scientific acceptance tolerances.

## Selected approach

Cy-ExM will extend its existing `SLURMCluster` adapter. The adapter will pass the
configured process count through `dask-jobqueue`'s `processes=` parameter.
`dask-jobqueue` will continue to own construction of the managed worker command,
including `--nworkers`, `--nthreads`, and `--memory-limit`.

Cy-ExM will validate the rendered command and make its executor, preflight, scaling,
replacement, and evidence models aware of job groups containing several worker
processes. This reuses the existing cluster ownership, configuration overlay,
reference barrier, fine graph, cleanup order, and task runner.

### Rejected alternatives

Submitting one SLURM job per Dask process and relying on the scheduler to colocate
jobs would not reliably provide multiple workers on one node. It would also make
memory and exclusivity site-specific.

Generating a custom job script with several independent Dask worker commands would
duplicate `dask-jobqueue` worker construction and create another startup,
replacement, and cleanup lifecycle to maintain.

## Configuration contract

Execution-profile schema version 1 remains current and retains its existing keys.

- `slurm.workers` is the positive maximum number of owned SLURM jobs.
- `slurm.processes_per_worker` is the positive number of Dask worker processes in
  each job.
- `parallelism.mode: coarse` requires `processes_per_worker == 1`. Any larger value is
  rejected during strict execution-profile validation and during defensive executor
  revalidation.
- `parallelism.mode: fine_grained` accepts a process count greater than one only when
  `cores_per_worker` is evenly divisible by it.
- `cores_per_process` is exactly
  `cores_per_worker // processes_per_worker`.
- Both `native_threads.itk` and `native_threads.blas` must be less than or equal to
  `cores_per_process`. They may be lower and need not equal it.
- `slurm.memory` remains the total semantic memory request for one SLURM job. The
  scheduler-normalized job allocation is a whole number of binary GiB. The
  per-process Dask limit in GiB is
  `floor(normalized_job_GiB / processes_per_worker)`. It must be at least one GiB.
  Every process receives that exact equal limit, while any remainder of less than
  `processes_per_worker` GiB remains unassigned job headroom. The sum of process
  limits must never exceed the SLURM job request.
- Existing profiles with `processes_per_worker: 1` retain their behavior.

The normalized plan and CLI resource summary will expose:

- maximum SLURM jobs;
- processes and task slots per job;
- maximum total task slots;
- cores per process;
- scheduler-normalized memory per job; and
- rendered Dask memory limit per process.

These are operational values. None enters the scientific plan fingerprint or Phase 1
artifact request parameters.

The initial documented and real-HPC-characterized profile uses:

```yaml
parallelism:
  mode: fine_grained

native_threads:
  itk: 15
  blas: 15

slurm:
  workers: 3
  cores_per_worker: 30
  processes_per_worker: 2
  memory: 500GB
```

This profile has a maximum of three jobs and six fine-task slots. Each process
receives 15 Dask threads. The semantic 500-GB profile normalizes to a 466-GiB
scheduler request, so each of two processes receives an exact 233-GiB Dask limit.

## Scheduler architecture

### Reference barrier

Fine execution initially calls `cluster.scale(jobs=1)`. One job launches exactly
`processes_per_worker` Dask worker processes. The coordinator waits for that exact
process count, probes the complete job group, validates it, and activates one
`round_slot` on each process.

Reference preparation consumes one slot. Other processes in the initial job remain
idle until reference preparation succeeds. This preserves the hard reference barrier:
no moving fine task is submitted before the reference result is successful.

### Fine-graph capacity and scaling

One activated Dask process provides one scientific task slot. The maximum eligible
capacity is:

```text
slurm.workers * slurm.processes_per_worker
```

After the reference barrier, the coordinator derives desired jobs from current
demand:

```text
ceil((running tasks + ready tasks) / processes_per_worker)
```

The result is bounded to at least one and at most `slurm.workers`. Scaling uses
`cluster.scale(jobs=desired_jobs)` and waits for
`desired_jobs * processes_per_worker` connected Dask processes. Scaling therefore
never requests a partial node-level process group. As in current fine execution, this
is a scale-up target: jobs are not scaled down during active scientific work and
remain owned until executor cleanup.

The `FineGraphController` continues to own readiness, dependencies, round-robin
fairness, and canonical result ordering. Its capacity input is the count of activated
process slots, not the number of SLURM jobs. Workers continue to run one immutable
task and never submit descendants.

### Scientific data flow

Fine task definitions are unchanged:

1. prepare fixed-grid state;
2. execute canonical nonlinear chunks;
3. assemble deterministic nonlinear results;
4. transform declared dependent channels; and
5. validate public outputs.

Several tasks may run concurrently in separate processes on one node, but they use
the same shared-filesystem artifacts, atomic commits, completion-record validation,
and fixed-grid scientific invariants as single-process execution.

## Job-group identity and preflight

A worker process remains uniquely identified by its scheduler address and
`(hostname, PID)`. Multiple probes may now share `SLURM_JOB_ID` when they form one
valid job group.

A valid activated job group has:

- one nonempty SLURM job ID;
- one identical nonempty hostname for all processes in the group;
- one identical nonempty network host for all processes in the group;
- exactly `processes_per_worker` connected workers;
- distinct worker addresses;
- distinct positive PIDs;
- the expected package versions and native-thread environment;
- shared dataset/output visibility and usable project scratch;
- the exact derived per-process Dask memory limit; and
- no more than one `round_slot` per process.

The coordinator validates all processes in a newly arrived job before activating any
process from that job. Existing activated jobs may continue scientific work while a
new complete group is probed. A partial, oversized, cross-host, or internally
contradictory group is never eligible.

Rendered-job validation will require:

- the managed SLURM CPU and memory request for the complete job;
- `--nworkers` exactly equal to `processes_per_worker` when greater than one and absent
  when equal to one;
- `--nthreads` exactly equal to `cores_per_process`;
- one exact per-process `--memory-limit`;
- nanny, interface, scratch, death-timeout, and managed resource arguments; and
- no user directive that overrides managed node, task, CPU, memory, process, worker,
  or resource settings.

In coarse mode the sole process continues to advertise one `round_slot` directly in
the rendered command. Fine candidates continue to advertise none until complete
preflight and coordinator activation.

## Failure and replacement behavior

The current artifact store remains the only restart authority. A task interrupted by
worker or job loss may reuse only artifacts with valid matching completion records.
Worker grouping and execution evidence never authorize scientific reuse.

When one Dask process disappears:

1. its scheduler address is removed from eligible capacity;
2. its active task follows existing Dask settlement and configured retry behavior;
3. healthy sibling processes and independent tasks remain eligible;
4. the coordinator waits up to `startup_timeout` for the job group to become complete;
5. a replacement PID/address is fully probed; and
6. the replacement receives a `round_slot` only after the complete group is healthy
   and consistent.

When an entire job disappears, the owned cluster may replace it. The replacement job
must arrive with exactly the configured process count and pass complete job-group
preflight before activation.

Reconciliation fails closed when:

- the timeout expires;
- a group remains partial or exceeds its process count;
- one job ID appears on multiple hosts;
- an address or `(hostname, PID)` is duplicated;
- CPU, memory, package, filesystem, network, scratch, or native-thread evidence is
  invalid;
- scheduler identity changes; or
- the connected job/process population exceeds configured limits.

Failure retains already committed scientific artifacts, settles independent work
according to existing fine-graph rules, and closes the client, cluster, and temporary
configuration in the established order. Cleanup remains limited to resources owned
by the executor.

## Operational provenance

All fine-SLURM runs will use execution-evidence `schema_version: 3` rather than
branching between one-process and multi-process formats. One-process fine profiles
also emit version 3. Historical version-2 workflow-run records remain immutable; the
runtime does not reinterpret them as cache authority.

The updated evidence records:

- configured and currently requested SLURM jobs;
- processes and task slots per job;
- maximum process-slot capacity;
- total job CPU/memory and derived per-process CPU/memory;
- rendered managed job template;
- job ID to hostname to worker-address/PID membership;
- candidate, probe, validation, activation, departure, and replacement events;
- whole-job scale targets and observed process counts;
- exact per-task worker, PID, hostname, and SLURM job placement;
- shared-node concurrent task intervals;
- attempts, cancellation, failure, and reconciliation outcomes; and
- ordered cleanup plus owned-job evidence.

Evidence validation cross-checks task placement against an activated process identity
and its containing job group. Several task placements may share a job ID and hostname,
but simultaneous tasks must have distinct process identities. One-process fine
profiles use the same evidence model with group size one.

## Testing

### Portable unit and integration tests

Configuration tests will prove:

- coarse rejection of process counts greater than one;
- positive process counts;
- exact core divisibility;
- ITK and BLAS bounds against per-process cores;
- unchanged normalization for one process; and
- direct immutable-model revalidation at the executor boundary.

Rendered-template tests will prove exact one- and two-process commands, including
`--nworkers`, `--nthreads`, divided `--memory-limit`, job CPU/memory, resource
advertisement, and conflicting-directive rejection.

Preflight and provenance tests will cover:

- valid shared-job groups;
- duplicate addresses and host/PID identities;
- shared job IDs across inconsistent hosts;
- incomplete and oversized groups;
- incorrect per-process memory or native threads;
- exact group serialization and validation; and
- placement cross-checks for concurrent same-job tasks.

Executor tests with bounded synthetic work will prove:

- one initial job and a hard reference barrier;
- exact process-count waits;
- demand-to-job ceiling arithmetic;
- whole-job scale-up to the configured job cap;
- capacity based on activated processes;
- process-level task admission and fairness;
- single-process departure and replacement;
- whole-job departure and replacement;
- failure on partial-group timeout;
- preservation of healthy siblings and independent tasks where safe; and
- ordered cleanup in all exit paths.

Scientific regression tests will confirm that changing only the execution profile
from one to two processes leaves the scientific plan and cache identities unchanged.

### Explicit real-HPC acceptance

Real-HPC acceptance is opt-in and requires operator-provided dataset, one-process and
two-process execution profiles, disjoint output/evidence roots, approved allocation,
and exact named-job cleanup.

The initial two-process-per-node gate requires:

- scientific artifact/metric parity with the one-process fine workflow under the
  existing accepted tolerance policy;
- at least two fine tasks with overlapping intervals in distinct PIDs sharing one
  SLURM job ID and hostname;
- exact 30-core, two-process, 15-thread-per-process, 500-GB-job resource evidence;
- correct process and job replacement/resume behavior;
- no task placement on an unvalidated process;
- complete owned-job and cleanup evidence; and
- recorded runtime, task-stage duration, worker/process peak RSS, and node aggregate
  memory comparisons.

Performance is characterization, not a hard speedup threshold. If two processes are
slower or create undesirable memory pressure for a workload, the result is documented
as an operational selection boundary rather than hidden or reported as an acceptance
failure.

## Documentation and repository guidance

Implementation must synchronize:

- the `README.md` execution-profile table and SLURM examples;
- workflow and fine-grained execution guides;
- failure/restart and operational provenance documentation;
- a dedicated editable fine-SLURM example using two processes, 30 total cores, 15
  ITK/BLAS threads per process, and 500 GB per job;
- `SlurmExecutor` and related user-callable configuration/API documentation; and
- applicable `AGENTS.md` files at repository, configuration, execution, runtime,
  examples, documentation, unit/integration test, and HPC layers.

Documentation must distinguish:

- jobs/nodes from Dask worker processes and task slots;
- total job resources from per-process limits;
- coarse one-process restrictions from fine multi-process support;
- implemented portable behavior from completed real-HPC acceptance; and
- correctness gates from performance characterization.

Failure documentation must explain partial process loss, full job loss, group
revalidation, timeout, cleanup, and artifact-safe resume. Provenance documentation
must explain shared job IDs, distinct process identities, derived resources, scale
events, and why operational evidence never authorizes scientific cache reuse.

## Acceptance criteria

The implementation is ready for portable integration when:

1. strict schema-version-1 profiles accept valid fine multi-process settings and
   reject every invalid or coarse multi-process combination;
2. rendered worker commands and derived resources are exact and fail closed;
3. scaling, preflight, activation, placement, replacement, provenance, and cleanup
   are process-group aware;
4. scientific plans, fingerprints, and artifact identities are unchanged;
5. all targeted and repository portable gates pass;
6. user documentation, examples, API documentation, and applicable `AGENTS.md` files
   describe the implemented contract consistently; and
7. no real-HPC production claim is made before the explicit two-process acceptance
   gate completes.
