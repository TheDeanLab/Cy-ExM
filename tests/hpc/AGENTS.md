# Explicit HPC acceptance guidance

- Every test in this directory is marked `hpc`; explicitly selected `hpc` execution
  means the actual pytest invocation contains exactly `-m hpc`, independent of
  marker defaults, includes exact `--run-hpc`, and supplies all three caller-owned
  Cy-ExM options.
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
- Before pytest starts, export all six coordinator native-thread variables to the
  exact profile values. Fail before scheduler access when any value is missing or
  mismatched so later sequential parity never attempts a post-import thread change.
- Treat the approved-dataset 8/8 comparison as stochastic scientific equivalence,
  not deterministic equality. Exact artifact structure, geometry, dtype, transform
  metadata, finite metrics, and nonlinear non-degradation remain mandatory; every
  persisted stage metric must have a metric absolute difference of 0.01 or less.
  Transform and voxel deltas are characterization only in this tier and must be
  recorded rather than promoted to universal scientific tolerances.
- Require strict numerical executor parity in a separate fresh coordinator with
  `ANTS_RANDOM_SEED=20260722`, one native thread for ITK and every BLAS variable,
  bounded synthetic whole-image and serial-chunked inputs, and exact named-job
  cleanup. A fixed seed does not make multithreaded ANTs registration exact. Neither
  acceptance tier authorizes a production seed, schema, or registration-policy
  change.
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
- Dataset validation includes every declared `reference.channels[].file`, not only the
  reference registration channel. Future full workflow acceptance must prove the
  reference task and moving tasks share one owned cluster and that no moving task is
  submitted before successful reference preparation.
- Scheduler inspection is read-only: use `sinfo` and `scontrol` for partition,
  allocation, node, and interface evidence, and a current-user `squeue` query for
  owned cleanup. Bound smoke allocation to `min(2, configured workers)`,
  submit no registration in the environment smoke test, require one process and one
  `round_slot` per job, and preserve exact job IDs, hostnames, partition facts,
  worker probes, and rendered job-script evidence.
- Register only the acceptance helper module for exact, temporary cloudpickle-by-
  value serialization and restore its prior registry state in every exit path. The
  production executor must carry those bytes in its importable runner wrapper;
  never add `tests` to worker `PYTHONPATH` or submit a direct test-module callable.
- Every Task 5 runner receives the complete workflow-task union. Dispatch the
  `ReferenceTask` through ordinary reference preparation before applying injected
  failure or cancellation behavior only to moving `RoundTask` values; never bypass
  or reinterpret the hard reference barrier in acceptance-only code. Workflow-result
  evidence includes that reference result; placement comparisons must select the two
  moving strategies rather than assume they are the only returned results.
- Own every job created by the harness. Close the Dask client before its cluster on
  success or failure, and query only the current user plus exact name after cleanup;
  never pass recorded job IDs to squeue. Sort and deduplicate the recorded IDs,
  classify recorded job IDs locally, preserve unrelated same-name rows, and fail if
  any owned job remains. Persist every bounded poll attempt and its query error or
  timeout, plus executor evidence, before re-raising. Do not cancel, inspect in
  depth, or otherwise mutate jobs outside the harness's recorded ownership.
- Fail closed unless the rendered job script exactly records the approved partition,
  exclusivity, single task/process, cores, memory, worker and scheduler interfaces,
  job name, and `round_slot`. Verify the allocated job partition, scheduler address,
  and coordinator/worker shared filesystem device and mount identities. Record
  semantic and scheduler-normalized memory, derive the exact expected worker bytes
  from the rendered worker limit, and require worker probes, requested/allocated
  TRES, partition capacity, and node capacity to cover the corresponding facts.
  Validate version-compatible SLURM job and node fields: require job CPU/node `TRES`,
  validate exact `MinMemoryNode` and any optional job `ReqTRES`/`AllocTRES`, and
  validate reported node `RealMemory`, `CfgTRES`, `AllocTRES`, or `AllocMem`. Never
  index an optional controller-version field before checking its presence, and fail
  if neither job-request nor node-allocation memory can be proved.
  Treat `AllocMem=0` as unavailable legacy evidence, not as a zero allocation, only
  when another authoritative job request or node allocation proves memory. Require
  positive `AllocMem` to cover normalized bytes and reject malformed or negative
  values without weakening `RealMemory`, `CfgTRES`, or `AllocTRES` validation.
- Make no acceptance claim without a real allocation and complete machine-readable
  evidence. Never commit caller data, external evidence, generated arrays, job logs,
  credentials, or site-specific approved paths.
- Fine-grained production acceptance additionally requires exact coarse-versus-fine
  two-tier scientific parity, nonlinear chunks and channels placed concurrently on
  distinct SLURM jobs/hosts, the configured cap and fair round opportunities,
  injected chunk/channel failure with resume, cancellation with resume, exact owned
  cleanup, and at least a 20% nonlinear-stage wall-time reduction. These real SLURM
  requirements remain pending until the exact-commit external gate is complete.
- The fine harness reuses the Phase 2E caller-owned context and owned-job lifecycle.
  Its test alone additionally requires exact `--cyexm-coarse-execution` and
  `--cyexm-fine-execution` profiles; the former must identify the same file as the
  legacy `--cyexm-execution` option. Never derive either operator approval silently.
- Apply the complete canonical archive/project/protected-root/input validation to
  both approved profiles before creating any scenario root or contacting SLURM.
  The profiles may differ only in coarse/fine mode, source path, and deliberately
  disjoint output root, project scratch root, and job name; worker count, processes,
  cores, memory, threads, queue, network, retries, directives, and controls must be
  exact for a fair comparison. Protected roots include ClearEx and both declared
  live spheroid paths above.
- Preflight two serial-chunked rounds with real dependent channels before any
  acceptance write. Failure, cancellation, and worker-loss controls bind exact
  cross-round task keys. Every summarized claim must cross-check checksum-bound raw
  attempt, placement, probe, scale, marker, deterministic, comparison, and queue
  evidence; a summary boolean is never acceptance authority.
- Cross-worker placement ordering and fairness use only checksum-bound primary-graph
  coordinator submission/readiness timestamps, including preparation readiness;
  never compare worker wall clocks. Cross-derive exact task worker/job/host fields,
  scale sequence, activated-worker cap, concurrency, and fair dispatch from that raw
  graph.
- The performance gate compares a conservative fine coordinator upper bound through
  assembly against the maximum complete per-round coarse nonlinear lower bound.
  Aggregate every nonlinear call in each stable `round-XXX-coarse-main` record using
  one worker-monotonic first-start/last-end span, or explicitly fall back to summed
  call durations when monotonic bounds are unavailable; never take one maximum call.
- Treat each owned-workflow `*-outcome.json` as the complete job-ID and cleanup
  outcome authority, and its checksum-bound `*-queue-after.json[jobs]` as the
  post-cleanup queue authority. Executor `cleanup` proves only ordered
  client/cluster/configuration closure; never infer SLURM ownership from it or from
  task placements. Lineage and placement job IDs must be subsets of the complete
  outcome-owned set so additional initial workers remain visible.
- Keep fine-only report schemas and audit decoders in
  `fine_workflow_acceptance.py`. Keep shared submission, cleanup, fresh-coordinator,
  and acceptance-only worker-runner machinery in `workflow_acceptance.py`.
  Acceptance-only profiling may time the real ANTs nonlinear kernel and record
  worker peak RSS externally, but it must restore patched runtime state in `finally`.
- A worker-loss trial may terminate only the exact dedicated worker process after
  durably writing its one-shot marker. It must never terminate the coordinator, and
  acceptance requires production departure, replacement, recovery, liveness, and
  exact named-job cleanup evidence before making a claim. Dask may expose recovery
  either as a failed coordinator attempt followed by a retry or as a transparent
  scheduler reschedule of the original future. For the latter, require the recovered
  task's coordinator interval to span the checksum-bound loss-marker timestamp and
  require the admitted replacement identity to complete that task or a later
  accepted scientific task.
- `test_fine_worker_replacement_only.py` is a focused production diagnostic for
  rerunning only worker-loss/replacement acceptance after scheduler-lifecycle
  changes. Its report may supplement prior exact-commit gates, but it does not by
  itself replace the complete fine-grained scientific, durability, cancellation,
  deterministic-parity, and performance acceptance matrix.
