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
- SLURM scales to `min(pending rounds, configured workers)` with one worker process
  and one advertised `round_slot` per job; every submitted round requests that slot.
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
