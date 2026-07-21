# Execution package guidance

- Execution owns the executor protocol and sequential, Dask, and SLURM adapter
  lifecycle, never registration algorithms or artifact-validity decisions.
- Phase 2D implements sequential, bounded local-Dask, and SLURM execution. The SLURM
  adapter lazily constructs and owns its client and cluster; real SLURM submission and
  HPC acceptance remain unaccepted until Phase 2E.
- Scheduler imports occur only inside the selected operation.
- Local Dask uses process-isolated workers and bounds their count to
  `min(pending rounds, configured workers)`. Schedule one coarse task per round. Each
  worker advertises one `round_slot`, and every submitted round task requests and
  consumes one.
- Every executor runs the reference-preparation task as a hard prerequisite in the
  same owned worker pool. Moving-round futures may be submitted only after the
  reference result is successful; a failed reference result ends the execution
  without submitting descendants.
- SLURM scales to `min(pending rounds, configured workers)` with one worker process
  and one advertised `round_slot` per job; every submitted round requests that slot.
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
- Allow no nested tasks. Permit no nested chunk scheduling or silent fallback.
- On operator cancellation, stop submission and cancel unfinished futures. Always
  close the client before its owned cluster on success, failure, and cancellation,
  including partial startup.
- Treat rendered job templates, requested resources, scheduler/worker identities,
  worker probes, attempts, timestamps, cancellation, and cleanup as operational
  evidence. Operational evidence never authorizes artifact reuse; Phase 1 completion
  records remain authoritative.
