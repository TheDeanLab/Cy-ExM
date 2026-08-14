# Execution profile reference

## Purpose

An execution profile is the machine and resource companion to a [scientific dataset
manifest](dataset-manifest.md). The manifest records rounds, channels, voxel spacing,
registration settings, and other scientific choices. The profile records where the
caller owns inputs and outputs, how native libraries are threaded, and where tasks
run. Changing a compatible profile does not change scientific identity.

The profile is strict YAML with `schema_version: 1`. Start from an editable
[checked-in profile](workflow-guide.md#start-from-the-checked-in-examples), replace
paths and resources for the current machine, then validate before running.

## Complete schema

The `executor` controls which one matching resource section is present:

| Location | Type and accepted values | Meaning |
| --- | --- | --- |
| `schema_version` | built-in integer `1` | Selects the strict execution-profile contract. |
| `executor` | `sequential`, `local_dask`, or `slurm` | Selects exactly one executor-specific section. |
| `parallelism` / `mode` | exact `coarse` or `fine_grained` | Operational task granularity. Omission normalizes to `coarse`; checked-in profiles declare it explicitly. |
| `paths` / `dataset_root` | non-empty single-line path string | Root to which manifest image paths are bound. |
| `paths` / `output_root` | non-empty single-line path string | Parent of `<dataset_id>` workflow output. |
| `paths` / `worker_local_directory` | path string or `null` | Dask worker scratch; worker-backed executors require a non-null value. |
| `native_threads` / `itk` | positive built-in integer | ITK thread count carried into every round task. |
| `native_threads` / `blas` | positive built-in integer | Count exported to every supported BLAS-family variable. |
| `local_dask` / `workers` | positive built-in integer | Maximum local worker processes. |
| `local_dask` / `threads_per_worker` | positive built-in integer | Dask threads in each worker; independent of native library counts. |
| `local_dask` / `processes` | built-in boolean `true` | Process isolation is mandatory; `processes` must be `true`. |
| `local_dask` / `memory_limit` | non-empty single-line string | Dask memory limit passed to `LocalCluster`. |
| `local_dask` / `startup_timeout` | non-empty single-line string | Bounds only Dask `Client` connection and `wait_for_workers`. |
| `local_dask` / `task_retries` | nonnegative built-in integer | Executor-managed retry count after the first attempt. |
| `slurm` / `queue` | non-empty single-line string | SLURM queue/partition. |
| `slurm` / `workers` | positive built-in integer | Maximum number of worker jobs/nodes, not Dask process slots. |
| `slurm` / `cores_per_worker` | positive built-in integer | Requested cores and job CPU count. |
| `slurm` / `processes_per_worker` | positive built-in integer | Processes per job: coarse requires `1`; fine divides job cores and memory across the processes. |
| `slurm` / `memory` | positive whole decimal `KB`, `MB`, `GB`, or `TB` string | Semantic memory request; the executor rounds up to whole binary GiB for SLURM and Dask. |
| `slurm` / `walltime` | non-empty single-line string | Job walltime. |
| `slurm` / `interface` | non-empty single-line string | Worker network interface. |
| `slurm` / `scheduler_interface` | non-empty single-line string | Coordinator/scheduler network interface. |
| `slurm` / `startup_timeout` | non-empty single-line string | Bounds only Dask `Client` connection and `wait_for_workers`. |
| `slurm` / `task_retries` | nonnegative built-in integer | Executor-managed retry count after the first attempt. |
| `slurm` / `job_name` | non-empty single-line string | Owned worker-job name. |
| `slurm` / `job_extra_directives` | list of non-empty single-line strings | Site additions that do not conflict with managed SLURM options. |

## Paths and filesystem safety

Absolute profile paths remain absolute; relative paths resolve from the
execution-profile file, not the current working directory. Dataset and output roots
must be disjoint after resolution: neither may contain the other. Existing symlinks
and planned outputs are checked for root escapes and collisions.

Distributed worker-backed executors require `worker_local_directory`. Worker scratch
must not be `/archive` or a descendant of `/archive`; choose active project storage
or approved node-local scratch instead. `dataset_root` and `output_root` identify
caller-owned data, while `worker_local_directory` is disposable worker scratch.

## Sequential profiles

Use `executor: sequential` to establish a dataset or when a single CLI process is
the appropriate environment. Sequential profiles omit both `local_dask` and `slurm`.
They may set `worker_local_directory: null`. The sequential executor is the numerical
and restart reference; it does not open a scheduler.

## Local process-based profiles

Use `executor: local_dask` when the workstation should run isolated local worker
processes. A local profile contains only `local_dask`, requires a non-null worker
scratch directory, and requires `processes: true`. `workers` is the maximum local
worker-process count. `threads_per_worker` controls Dask worker threads and is not a
native ITK or BLAS thread count; see [Native threading and
performance](native-threading.md).

## SLURM profiles

Use `executor: slurm` only after confirming the scheduler, storage, allocation, and
resource settings with the local HPC service. A SLURM profile contains only `slurm`
and requires worker scratch. `queue`, `walltime`, interfaces, account or QoS-like
site directives, shared paths, and coordinator placement are site-specific facts to
validate rather than portable defaults.

`job_extra_directives` may add safe site policy such as an account, QoS, constraint,
mail setting, reservation, or `--exclusive`. It cannot conflict with managed SLURM
options for partition (`--partition`, `-p`), memory (`--mem`, `--mem-per-cpu`),
CPU/task counts (`--cpus-per-task`, `-c`, `--ntasks`, `-n`), or stdout/stderr
(`--output`, `-o`, `--error`, `-e`), including unambiguous long-option abbreviations
and grouped or attached short forms.

## Coarse and fine-grained modes

`parallelism` is operational execution policy, never a scientific-manifest choice.
Omission normalizes to `coarse`. Coarse schedules one complete moving round at a
time. Opt-in `fine_grained` changes task admission: the coordinator may admit
preparation, nonlinear chunk, assembly, and channel tasks independently without
creating a third registration strategy. Read [Fine-grained
execution](fine-grained-execution.md) for the exact graph.

Coarse mode requires `processes_per_worker: 1`. Fine mode requires exact core
divisibility: `cores_per_worker` must divide evenly across the process count.

## CPU, process, and memory rules

All worker/core fields are positive built-in integers and retry fields are a
nonnegative built-in integer; booleans are never accepted as counts. The native
thread count must not exceed the cores per process for either ITK or BLAS.
Explicit `native_threads` values override the inherited environment; an inherited
value overrides the one-thread default. All five inherited BLAS values must agree,
or loading fails and asks for an explicit `blas` value.

`slurm.workers` is the job cap and `workers * processes_per_worker` is the total
process-slot capacity. These values are reported separately so the job cap and total
process-slot capacity cannot be confused. A coarse SLURM job has one process and one
task slot; a fine job can contain a complete group of isolated processes, each with
one slot.

SLURM memory uses positive whole decimal memory with exact uppercase `KB`, `MB`,
`GB`, or `TB` suffixes. These profile units are decimal semantic bytes. The executor
rounds upward to whole binary GiB, passes the matching `GiB` quantity to Dask, and
never passes a `GB` suffix to `#SBATCH --mem`. For example, `500GB` becomes `466G`,
which is 500,363,689,984 bytes, and Dask receives `466GiB`. Invalid, fractional,
zero, unsupported, or overflowing values fail configuration. Operational evidence
records both semantic and scheduler-normalized memory.

Scheduler-normalized memory remains the total job request. Fine mode divides its
whole binary GiB conservatively among processes; any remainder is job headroom, and
the per-process Dask memory limits never sum above the job request.

## Startup-timeout boundary

`startup_timeout` is not an end-to-end startup deadline. It bounds only Dask
`Client` connection and `wait_for_workers`; it does not bound `LocalCluster` or
`SLURMCluster` construction, does not bound `scheduler_info()` calls, and does not
bound the `Client.run()` worker-preflight RPC. Those operations currently have no
Cy-ExM wall-clock timeout, so a stalled constructor or preflight RPC can outlive the
configured value. Treat this as an operational limitation when choosing where to run
the coordinator.

## Strict YAML validation

Exactly one executor-specific section must match `executor`: sequential profiles
contain neither section, local profiles contain only `local_dask`, and SLURM profiles
contain only `slurm`. Unknown or mismatched sections are rejected. The loader also
rejects duplicate mapping keys, YAML aliases, custom tags, non-scalar keys,
additional YAML documents, invalid UTF-8, and non-mapping input.

Native thread counts and all count fields require built-in integers, so booleans,
floats, and strings are not coerced. Every other text value shown above must be a
non-empty single-line string. Validate both profile and manifest before execution:

```bash
uv run cyexm workflow validate path/to/dataset.yaml \
  --execution path/to/execution.yaml
```

The production-owned callable, rather than an import from a test module, is
serialized for worker execution. Workers never need `tests` on `PYTHONPATH`; the
embedded runner is rebuilt inside ordinary task execution, where reconstruction
failure is a normal task failure.

For real-SLURM acceptance, resource records are version-compatible rather than
assumed to expose identical keys. Job `TRES` must prove requested CPU and one node.
When present, `MinMemoryNode` must equal the normalized scheduler request; job
`ReqTRES` and `AllocTRES` are optional but must prove adequate CPU, node, and memory
when reported. Node `RealMemory` must cover the request, and `CfgTRES` and `AllocTRES`
must cover its CPU and memory facts. `AllocMem=0` is unavailable as allocation
evidence rather than proof of zero allocated memory. A positive `AllocMem` must cover
the normalized request; malformed or negative values fail validation. The acceptance
harness fails closed if neither an authoritative job-memory request nor a node
memory-allocation fact is available.

## Starting profiles

Use the [sequential](../examples/execution-sequential.yaml), [local
process-based](../examples/execution-local-dask.yaml), or [SLURM
single-process](../examples/execution-slurm-512gb.yaml) profile as an editable
starting point. The SLURM examples are placeholders, not a claim about any current
cluster; replace paths and validate the local site before submitting work.
