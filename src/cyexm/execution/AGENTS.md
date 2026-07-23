# Execution package guidance

- Execution owns the executor protocol and sequential, Dask, and SLURM adapter
  lifecycle, never registration algorithms or artifact-validity decisions.
- Phase 2D implements sequential, bounded local-Dask, and SLURM execution. The SLURM
  adapter lazily constructs and owns its client and cluster; real SLURM submission and
  HPC acceptance remain unaccepted until Phase 2E.
- Scheduler imports occur only inside the selected operation.
- Local Dask uses process-isolated workers. Coarse mode bounds their count to
  `min(pending tasks, configured workers)`; fine mode starts exactly the configured
  worker count because preparation determines future graph width only at runtime.
  Schedule one reference prerequisite in the same owned cluster before moving work.
  Coarse execution continues to schedule one coarse task per moving round.
  The coordinator dynamically admits at most connected capacity from the shared fine
  graph; workers never submit children. Each worker advertises one `round_slot`, and
  every submitted workflow task requests and consumes one.
- Every executor runs the reference-preparation task as a hard prerequisite in the
  same owned worker pool. Moving-round futures may be submitted only after the
  reference result is successful; a failed reference result ends the execution
  without submitting descendants.
- SLURM scales to `min(pending tasks, configured workers)` with one worker process
  and one advertised `round_slot` per job; every submitted workflow task requests that
  slot. The reference barrier must not construct or scale a second cluster.
  Round semantic profile memory upward to whole binary GiB, pass the `G` spelling to
  SLURM and the matching `GiB` spelling to Dask, and record both semantic and
  scheduler-normalized facts. Never copy a decimal `GB` suffix into `#SBATCH`.
  Revalidate directly constructed SLURM profiles before scheduler module loading,
  and validate the production-rendered header and worker command before the first
  scale call. Reject unexpected or duplicate resource and worker options.
  Serialize the configured runner directly before scheduler module loading and carry
  its bytes in a production-owned callable. Reconstruct it inside ordinary task
  execution so nested non-production runners do not become worker imports.
  Require that all connected workers pass preflight before any registration task is
  submitted. Construction, worker probes, and job-script templates are portable
  evidence only, not proof that a scheduler accepted or ran a job.
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
- Fine round wall duration is the interval from the earliest actual task start to
  the latest actual task end. All actual results for one round must agree on native
  thread evidence. Full multi-host, process, scheduler, and log provenance remains
  per-task evidence; never concatenate it into singular `RoundResult` fields.
- Allow no nested tasks. Permit no nested chunk scheduling or silent fallback.
- On operator cancellation, stop submission and cancel unfinished futures. Always
  close the client before its owned cluster on success, failure, and cancellation,
  including partial startup.
- Treat rendered job templates, requested resources, scheduler/worker identities,
  worker probes, attempts, timestamps, cancellation, and cleanup as operational
  evidence. Operational evidence never authorizes artifact reuse; Phase 1 completion
  records remain authoritative.
