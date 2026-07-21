# Explicit HPC acceptance guidance

- Every test in this directory is marked `hpc`; explicitly selected `hpc` execution
  means a `pytest -m hpc` command with all three caller-supplied Cy-ExM options.
  Fail closed when an option, approved path, scheduler fact, allocation fact, or
  cleanup fact is missing. Never skip by hostname or infer approval from being on a
  login node.
- The dataset manifest, SLURM execution profile, and caller-owned evidence directory
  must be supplied on the command line. Load YAML only through the public strict
  configuration APIs. Require a real SLURM profile and a real allocation; portable
  fake construction is not acceptance evidence.
- Require `--cyexm-dataset PATH`, `--cyexm-execution PATH`, and
  `--cyexm-evidence-dir PATH`. Atomically write `acceptance-context.json` before any
  scheduler inspection, including UTC time, git commit, exact package versions,
  resolved paths, and the complete argument vector.
- Keep datasets, generated registration outputs, submitted scripts, scheduler
  snapshots, and other large evidence on `/archive`. Keep the environment, caches,
  and worker scratch on `/project`; each smoke worker must prove scratch
  create/write/delete behavior.
- Never access ClearEx, the live spheroid registration directory, or its output.
  Reject evidence roots inside the repository or live spheroid path, require an
  existing empty evidence directory, and write only caller-owned evidence there.
- Scheduler inspection is read-only: use `sinfo` and `scontrol` for partition,
  allocation, node, and interface evidence, and a current-user `squeue` query for
  owned cleanup. Bound smoke allocation to `min(2, configured workers)`,
  submit no registration in the environment smoke test, require one process and one
  `round_slot` per job, and preserve exact job IDs, hostnames, partition facts,
  worker probes, and rendered job-script evidence.
- Own every job created by the harness. Close the Dask client before its cluster on
  success or failure, query the recorded job name and IDs after cleanup, and fail if
  any owned job remains. Do not cancel, inspect in depth, or otherwise mutate jobs
  outside the harness's recorded ownership.
- Make no acceptance claim without a real allocation and complete machine-readable
  evidence. Never commit caller data, external evidence, generated arrays, job logs,
  credentials, or site-specific approved paths.
