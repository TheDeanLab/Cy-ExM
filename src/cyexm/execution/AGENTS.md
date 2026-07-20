# Execution package guidance

- Execution owns the executor protocol and sequential, Dask, and SLURM adapter
  lifecycle, never registration algorithms or artifact-validity decisions.
- Scheduler imports occur only inside the selected operation.
- Schedule one coarse task per round. Each worker advertises one `round_slot`, and
  every submitted round task requests and consumes one.
- Allow no nested chunk scheduling or silent fallback.
- Owned clients, clusters, and jobs are cleaned up on success, failure, and
  cancellation.
