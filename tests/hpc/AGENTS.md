# Explicit HPC acceptance guidance

- Every test in this directory is marked `hpc`; explicitly selected `hpc` execution
  means the actual pytest invocation contains exactly `-m hpc`, independent of
  marker defaults, and supplies all three caller-owned Cy-ExM options.
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
  Canonically validate configuration files, every manifest input and symlink target,
  and dataset, output, scratch, and evidence roots before writing anything. Require
  them to be canonically disjoint where required from each other, this repository,
  ClearEx, and each live spheroid path. Require an existing empty evidence directory
  and write only caller-owned evidence there.
- Scheduler inspection is read-only: use `sinfo` and `scontrol` for partition,
  allocation, node, and interface evidence, and a current-user `squeue` query for
  owned cleanup. Bound smoke allocation to `min(2, configured workers)`,
  submit no registration in the environment smoke test, require one process and one
  `round_slot` per job, and preserve exact job IDs, hostnames, partition facts,
  worker probes, and rendered job-script evidence.
- Own every job created by the harness. Close the Dask client before its cluster on
  success or failure, query the current user, exact name, and recorded job IDs after
  cleanup, and fail if any owned job remains. Persist every bounded poll attempt and
  its query error or timeout, plus executor evidence, before re-raising. Do not
  cancel, inspect in depth, or otherwise mutate jobs outside the harness's recorded
  ownership.
- Fail closed unless the rendered job script exactly records the approved partition,
  exclusivity, single task/process, cores, memory, worker and scheduler interfaces,
  job name, and `round_slot`. Verify the allocated job partition, scheduler address,
  and coordinator/worker shared filesystem device and mount identities.
- Make no acceptance claim without a real allocation and complete machine-readable
  evidence. Never commit caller data, external evidence, generated arrays, job logs,
  credentials, or site-specific approved paths.
