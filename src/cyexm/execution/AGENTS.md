# Execution package guidance

- Execution owns the executor protocol and sequential, Dask, and SLURM adapter
  lifecycle, never registration algorithms or artifact-validity decisions.
- Sequential, bounded local-Dask, and SLURM execution support coarse and opt-in
  fine-grained admission. The coarse SLURM adapter has Phase 2E acceptance; portable
  real-ANTs/Local-Dask fine evidence and multi-process SLURM policy are implemented,
  while fine-grained and focused multi-process real-SLURM production acceptance
  remain pending explicit external gates.
- Scheduler imports occur only inside the selected operation.
- Local Dask uses process-isolated workers. Coarse mode bounds their count to
  `min(pending tasks, configured workers)`; fine mode starts exactly the configured
  worker count because preparation determines future graph width only at runtime.
  Schedule one reference prerequisite in the same owned cluster before moving work.
  Coarse execution continues to schedule one coarse task per moving round.
  The coordinator dynamically admits at most connected capacity from the shared fine
  graph; workers never submit children. Each worker advertises one `round_slot`, and
  every submitted workflow task requests and consumes one. In fine mode that token
  means one scientific workflow task, not one complete round.
- Every executor runs the reference-preparation task as a hard prerequisite in the
  same owned worker pool. Moving-round futures may be submitted only after the
  reference result is successful; a failed reference result ends the execution
  without submitting descendants.
- Coarse SLURM scales to `min(pending tasks, configured workers)` with one worker
  process and one advertised `round_slot` per job; every submitted coarse task
  requests that slot. Fine SLURM instead uses one complete reference job group and
  demand-driven whole-job preflight/activation below. `slurm.workers` remains the job
  cap, while total task capacity is jobs times processes per job. The reference
  barrier must not construct or scale a second cluster.
  Round semantic profile memory upward to whole binary GiB, pass the `G` spelling to
  SLURM and the matching `GiB` spelling to Dask, and record both semantic and
  scheduler-normalized facts. Never copy a decimal `GB` suffix into `#SBATCH`.
  Revalidate directly constructed SLURM profiles before scheduler module loading,
  and validate the production-rendered header and worker command before the first
  scale call. Reject unexpected or duplicate resource and worker options.
  Serialize the configured runner directly before scheduler module loading and carry
  its bytes in a production-owned callable. Reconstruct it inside ordinary task
  execution so nested non-production runners do not become worker imports.
  Coarse mode requires all connected workers to pass preflight before any
  registration task is submitted. In fine mode, every initial or newly connected
  job group must pass complete preflight and atomic group activation before its
  processes are eligible for scientific submission; activated groups may already be
  executing work while a newcomer group is validated. Construction, worker probes,
  and job-script templates are portable evidence only, not proof that a scheduler
  accepted or ran a job.
- Perform fine group activation as one scheduler-side transaction with full resource
  rollback on any member failure. Commit `validated_workers`, activated probes, and
  scale-event activation only after the complete transaction and post-activation
  scheduler snapshot succeed; failed groups must expose no partial `round_slot`.
- Default retries are zero. Any configured retry is an executor-managed resubmission
  through ordinary artifact validation. A post-preflight round failure does not
  cancel independent round tasks; settle them before reporting aggregate failure.
- Fine-grained execution uses the scheduler-neutral `FineGraphController`. Admit
  ready work round-robin across rounds, remain work-conserving when only one round is
  ready, and never release a task before every exact dependency succeeds. Failed
  tasks block only their descendants; already admitted siblings and independent
  rounds must settle.
- Preserve every actual fine-task result in canonical task order. Never synthesize a
  result for blocked or never-submitted descendants. The ordinary `RoundResult` is a
  deterministic public summary: validation supplies successful scalar evidence,
  while the earliest canonical failed stage supplies failure evidence. Final-output
  reuse comes from producing tasks, not validation's read-only artifact scan.
- Every adapter, including sequential, validates exact fine task/result identity,
  configured expanded native-thread evidence, timezone-aware intervals,
  artifact/log/failure structure, and current attempt identity before
  `FineGraphController.record`. Convert invalid or raised worker evidence through the
  shared scheduler-neutral failed-result constructor before applying any configured
  executor retries; malformed evidence must never release descendants. Sequential
  has no retry setting and records one terminal attempt.
- Fine round wall duration is the interval from the earliest actual task start to
  the latest actual task end. All actual results for one round must agree on native
  thread evidence. Full multi-host, process, scheduler, and log provenance remains
  per-task evidence; never concatenate it into singular `RoundResult` fields.
- Sequential and local Dask fine executors retain the actual controller graph,
  task results, attempt counts, blocked descendants, cancellation, and owned-resource
  cleanup in one detached scheduler-neutral evidence snapshot. Public workflow
  validation cross-checks that snapshot after scientific work; never reconstruct it
  from lossy round summaries or let it authorize artifact reuse.
- Fine-grained SLURM starts one complete candidate job group for the reference
  barrier and uses demand-driven scale-up in whole jobs only. The target is the
  ceiling of `(running + ready) / processes_per_job`, bounded by configured jobs.
  Candidate processes advertise no `round_slot`; the coordinator must run complete
  worker preflight on each exact new address, validate exactly the configured number
  of processes sharing one job ID and hostname with distinct addresses/PIDs, add
  exactly one `round_slot` per process, and verify scheduler identity, worker set,
  per-process memory, and resources in the post-activation snapshot before any
  process in the group becomes scientifically eligible. Before recording any fine result or
  releasing descendants, match its SLURM job, worker address, hostname, and PID to
  one activated probe. Record rejected/no-placement outcomes explicitly as failed,
  never as accepted placement evidence. Retain one exact ordered evidence record
  for every scheduler-settled attempt, including attempts later retried; counts,
  attempt keys, claimed placement, acceptance, and rejection reason must agree.
- Serialize both coarse and fine task runners into the importable byte-carrying
  submission wrapper before Dask receives them. Compute nodes must not need the
  coordinator's test modules or a test-only `PYTHONPATH`; this also applies to
  acceptance-only instrumentation and injected-failure runners.
- Keep fine SLURM lifecycle evidence versioned, strict, and operational. The
  scheduler-neutral controller owns immutable canonical task/dependency evidence;
  `_slurm_evidence` owns typed scale-event construction and decoding; `slurm`
  owns cluster scale, preflight, activation, submission, cancellation, and cleanup.
  Persist every settled full `FineTaskResult`, including a preparation descriptor,
  in canonical order and decode it through the authoritative result model and common
  scheduler-neutral validator. Bind those results to the controller's exact immutable
  tasks and use the shared `chunk_tasks`/`channel_tasks` expansion validation; a
  structurally plausible but different descriptor must not authenticate descendants.
  Preserve rejected raw placement claims separately from the synthesized terminal
  failed result; require identity equality only for accepted placements.
  Successful producers require the exact successors materialized by the controller.
  Downstream omission is truthful only after a failed/cancelled/blocking task or an
  observed cancellation/scale failure before that producer settled. Preserve
  partial connected/probed/activated/validated scale state on every failure edge.
  Treat connected/probed evidence as raw candidate observations, not validated
  identity. A raw candidate with malformed or duplicate SLURM identity may remain
  only when that exact scale event records rejection; it must never appear in the
  activated/validated set. Enforce unique address/PID identity on every active
  process. Job ID and hostname must be identical within one configured process group
  and distinct across active job groups. Every accepted placement must bind one
  exact activated process.
  Do not let scheduler evidence authorize scientific artifact reuse.
- Allow no nested tasks. Permit no nested chunk scheduling or silent fallback.
- On operator cancellation, stop submission and cancel unfinished futures. Always
  close the client before its owned cluster on success, failure, and cancellation,
  including partial startup.
- While fine SLURM futures are active, periodically check completion before
  reconciling the exact scheduler/worker identities. Healthy long-running tasks have
  no execution deadline. Treat the Dask wait timeout at the health interval as a
  polling tick that triggers identity/resource reconciliation, not as task failure.
  For an unchanged exact scheduler address and worker-address set, validate scheduler
  resources without rebroadcasting the blocking filesystem/import probe to workers
  running scientific kernels. Re-probe every changed or replacement address before
  admitting it.
  If an activated process departs, bound only the replacement transaction by the
  configured startup timeout; preflight and resource-gate the replacement before
  retry or new admission, record the departed identity, and preserve a healthy
  sibling when the job remains intact. If the whole job departs, replace and
  activate only a complete new process group. A partial group is never eligible.
  Reconciliation at the current job cap must not issue another scale request. Local
  Dask retains its scheduler-native blocking completion behavior.
- A nanny restart can replace a worker address and PID inside the same SLURM
  allocation, so historical activated probes may legitimately share one
  `SLURM_JOB_ID`. Resolve a task result against the unique activated
  hostname/PID/job tuple, derive the worker address from that probe, and retain the
  active-interval check. Never reject a valid replacement merely because an older
  departed probe used the same allocation ID.
- Fine operational evidence is schema version 3 for both one-process and
  multi-process jobs. Record job/process resource derivation, expected process count
  for each scale event, complete group probes, departed processes, replacement
  activation, and exact placement. This evidence is diagnostic; Phase 1 completion
  records remain the only cache/restart authority.
- Use one absolute coordinator-owned startup-timeout deadline for the complete
  replacement transaction: health/arrival snapshots, worker arrival, probe
  broadcast, candidate resource checks, every scheduler activation, and the final
  identity/resource snapshot. Run the remote probe through a bounded daemon worker
  thread so blocking filesystem or import checks do not block the Dask worker event
  loop. The remote timeout wrapper must declare Dask's exact `dask_worker` injection
  keyword and forward that object into the threaded probe so replacement address and
  memory identity remain available. Guard synchronous client/scheduler calls with
  daemon coordinator threads.
  On deadline failure, record the exact stage, best-effort force-cancel active work,
  and close client before cluster. Never apply this control-plane deadline to the
  scientific future itself.
- Scheduler-neutral cancellation evidence must bind the exact caught interruption
  type/message and the actual cancellation outcome. Sequential cancellation is
  explicitly `not_applicable`; distributed cancellation distinguishes no unfinished
  futures, successful forced cancellation, and cancellation failure with its exact
  error. Never infer this evidence after cleanup.
- Treat rendered job templates, requested resources, scheduler/worker identities,
  worker probes, attempts, timestamps, cancellation, and cleanup as operational
  evidence. Operational evidence never authorizes artifact reuse; Phase 1 completion
  records remain authoritative.
