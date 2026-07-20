# Execution package guidance

- Execution owns the executor protocol and sequential, Dask, and SLURM adapter
  lifecycle, never registration algorithms or artifact-validity decisions.
- Phase 2C implements sequential and bounded local-Dask execution. SLURM execution
  remains proposed for Phase 2D and must follow its approved plan before landing here.
- Scheduler imports occur only inside the selected operation.
- Local Dask uses process-isolated workers and bounds their count to
  `min(pending rounds, configured workers)`. Schedule one coarse task per round. Each
  worker advertises one `round_slot`, and every submitted round task requests and
  consumes one.
- Default retries are zero. Any configured retry is an executor-managed resubmission
  through ordinary artifact validation; a failed round does not cancel independent
  round tasks.
- Allow no nested tasks. Permit no nested chunk scheduling or silent fallback.
- On operator cancellation, cancel unfinished futures. Always close the client before
  its owned cluster on success, failure, and cancellation, including partial startup.
