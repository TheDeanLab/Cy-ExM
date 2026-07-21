# Cy-ExM Phase 2 Workflow Runtime Design

**Date:** 2026-07-20

**Status:** Approved in design discussion; implementation has not started

## Executive decision

Cy-ExM Phase 2 will add a scheduler-neutral workflow layer around the stable Phase 1
registration APIs. A strict scientific dataset manifest and a separate execution
profile will produce deterministic round tasks. The same tasks will run through a
sequential executor, a bounded local Dask executor, or a
`dask_jobqueue.SLURMCluster` executor.

Each moving round is one coarse task. On SLURM, each task occupies one dedicated
512 GB node. Whole-image and serial-chunked registration are both supported, selected
per round. Registration within a round remains serial: Phase 2 will not introduce
nested chunk multiprocessing, a Dask chunk graph, or
`ParallelChunkedImageRegistration`.

Phase 1 artifacts and their completion records remain the only restart authority.
Workflow plans, Dask futures, SLURM job identifiers, and summaries are provenance and
operational evidence, not a competing checkpoint system.

## Current behavior

Phase 1 on `main` currently provides:

- restart-safe whole-image registration through `ImageRegistration` and
  `register_round`;
- restart-safe serial chunked nonlinear registration through
  `ChunkedImageRegistration`;
- affine and warp artifacts with validated `.complete.json` records;
- strict image geometry and TIFF boundaries;
- configurable native ITK and BLAS-family thread counts; and
- functions for applying a persisted affine-plus-warp transform chain.

Phase 1 does not provide YAML workflow configuration, dataset planning, multi-channel
round orchestration, Dask execution, SLURM provisioning, or a workflow CLI. Its
registered TIFF is the diagnostic output for the designated registration channel;
there is no artifact-managed operation that resamples every channel in a round.

The motivating spheroid workflow has three independent moving rounds but currently
runs them serially in one local process. Registration stages can run for more than a
day, making round-level parallelism the useful boundary. The cluster currently
advertises an active `512GB` partition, but partition properties and networking are
implementation-time evidence and must be rechecked before HPC acceptance.

## Goals

1. Describe reusable cyclic-registration datasets with strict YAML.
2. Keep scientific choices independent from machine-specific execution settings.
3. Plan and validate a complete workflow before creating outputs or schedulers.
4. Run the same round-task contract sequentially, on local Dask, or through SLURM.
5. Assign at most one round task to each dedicated 512 GB SLURM node.
6. Support whole-image and serial-chunked registration per round without nested
   parallelism.
7. Resample every listed channel through the accepted round transform chain and give
   every output an atomic, validated completion record.
8. Preserve Phase 1 restart semantics and numerical behavior.
9. Provide deterministic status, failure, cancellation, and provenance records.
10. Document the complete runtime in `README.md` and include validated YAML examples.

## Non-goals

- Do not add per-chunk Dask scheduling, multiprocessing inside a round task, or a
  parallel replacement for `ChunkedImageRegistration`.
- Do not recreate `clearex.registration.pipeline`, `clearex.registration.tre`, the
  ClearEx GUI, or ClearEx's general analysis workflow framework.
- Do not add job arrays, cloud or Kubernetes executors, cross-dataset scheduling, or
  automatic resource optimization in Phase 2.
- Do not silently fall back from Dask or SLURM to sequential execution.
- Do not parallelize channel transformation within a round in Phase 2.
- Do not copy, transform, or duplicate reference-round channels unless a later design
  explicitly adds that behavior.
- Do not make Dask, YAML, or SLURM dependencies mandatory for Phase 1 registration.
- Do not make operational workflow state authorize artifact reuse.
- Do not modify or terminate the separate spheroid registration process or its
  outputs during implementation or acceptance.

## Approaches considered

### Purpose-built planner with executor adapters

This is the selected approach. Configuration, planning, round execution, and
scheduler lifecycle have separate interfaces. The sequential executor remains the
debugging and numerical reference. Dask and SLURM are optional adapters that consume
the same immutable tasks.

This approach requires several small interfaces, but it preserves lightweight Phase
1 imports, makes failure behavior testable without a cluster, and prevents scheduler
concerns from leaking into registration.

### Dask for every execution mode

Using Dask even for sequential work would reduce the number of executor paths. It
would also make ordinary planning and local debugging depend on Dask, obscure the
serial numerical reference, and make import and startup behavior harder to reason
about. It is rejected.

### Extract the ClearEx workflow framework

ClearEx has a working Dask/SLURM implementation and useful operational evidence, but
its workflow layer also contains GUI, persistence, ingestion, GPU, and analysis
assumptions that Cy-ExM does not need. Extracting that framework would create a much
larger surface and weaken Cy-ExM's repository independence. Phase 2 will reuse its
characterized dependency versions and focused SLURM patterns, not its entire runtime.

## Package architecture

Phase 2 introduces these boundaries:

| Package | Responsibility | Forbidden ownership |
| --- | --- | --- |
| `cyexm.config` | Strict YAML decoding and immutable manifest/profile models | ANTs imports, Dask imports, filesystem mutation |
| `cyexm.workflow` | Plan validation, task/result models, output layout, public workflow API | Cluster creation, scientific runtime imports at module scope |
| `cyexm.workflow.round_task` | Execute one round and return a serializable result | Dask client creation, cross-round scheduling |
| `cyexm.execution` | Executor protocol and sequential adapter | Registration algorithms |
| `cyexm.execution.dask` | Local Dask client/worker lifecycle | SLURM configuration |
| `cyexm.execution.slurm` | `SLURMCluster` construction, worker preflight, cleanup | Registration algorithms, artifact validity decisions |
| `cyexm.registration.channels` | Restart-safe transformation of one source channel | Workflow scheduling |
| `cyexm.cli` | CLI parsing, logging, lifecycle, exit codes | Scientific algorithms |

Configuration and plan modules must be importable with plain Python data and the YAML
dependency only. Scheduler imports occur inside the operation that opens the selected
executor. The top-level round worker configures native threads before importing ANTs,
NumPy, SciPy, or registration modules.

Dependency direction is:

```text
config
  ↓
workflow models and planner
  ↓
round task ──→ Phase 1 registration and channel transformation
  ↑
executor adapters
  ↑
CLI
```

Registration never imports workflow or executor modules. Executor adapters depend on
workflow models, never the reverse.

## Dependency groups

The base installation remains unchanged. Phase 2 capabilities use cumulative
optional extras:

- `workflow`: `PyYAML>=6,<7` for strict YAML loading and sequential execution;
- `distributed`: workflow dependencies plus `dask==2025.1.0` and
  `distributed==2025.1.0`; and
- `hpc`: distributed dependencies plus `dask-jobqueue==0.9.0`.

These Dask versions match the characterized ClearEx environment. The implementation
must generate and inspect the `uv.lock` change, run local Dask integration tests, and
construct a real `SLURMCluster` during HPC acceptance before the versions are treated
as accepted for Cy-ExM. A version change requires a deliberate dependency update and
fresh compatibility evidence; it must not happen implicitly during `uv sync`.

## Scientific dataset manifest

The manifest contains portable scientific decisions only. Channel paths are relative
POSIX-style paths bound to a dataset root by the execution profile. The manifest has
this version-1 shape:

```yaml
schema_version: 1
dataset_id: tc71-spheroid-1

voxel_spacing_um:
  x: 0.324
  y: 0.324
  z: 0.900

reference:
  round: 1
  channel: round_1_488_tubulin.tif

registration:
  crop: false
  linear:
    type: TRSAA
    accuracy: high
  nonlinear:
    type: SyNOnly
    accuracy: high

rounds:
  - round: 2
    strategy: whole_image
    registration_channel: round_2_488_tubulin.tif
    channels:
      - file: round_2_488_tubulin.tif
        interpolation: linear
      - file: round_2_561_bcatenin.tif
        interpolation: linear
      - file: round_2_647_hsp60.tif
        interpolation: linear

  - round: 3
    strategy: serial_chunked
    registration_channel: round_3_488_tubulin.tif
    chunk_size_zyx: [512, 512, 512]
    overlap_fraction: 0.15
    channels:
      - file: round_3_488_tubulin.tif
        interpolation: linear
      - file: round_3_647_fbl.tif
        interpolation: linear
```

### Scientific field rules

- `schema_version` is the built-in integer `1`.
- `dataset_id` matches `[A-Za-z0-9][A-Za-z0-9._-]*` and becomes one output-directory
  component.
- XYZ voxel-spacing values are finite positive numbers in micrometers. They remain in
  public `(x, y, z)` order.
- Reference and moving round numbers are non-negative built-in integers. Moving round
  numbers are unique and cannot equal the reference round.
- `channel` and `file` values are non-empty relative paths. Absolute paths, `..`, NUL,
  and paths that resolve outside the bound dataset root are rejected.
- A round's `registration_channel` must exactly match one of its channel `file`
  entries.
- A channel filename is unique within its round.
- `strategy` is `whole_image` or `serial_chunked`.
- `chunk_size_zyx` is required only for `serial_chunked` and contains three positive
  built-in integers in NumPy ZYX order.
- `overlap_fraction` is required only for `serial_chunked` and follows Phase 1's
  finite `0 <= value < 1` contract.
- `crop` is a boolean. Linear/nonlinear type and accuracy values use the exact Phase 1
  accepted choices.
- Channel `interpolation` is explicit and is `linear` or `nearestNeighbor`.
- Unknown keys, duplicate YAML mapping keys, YAML aliases, custom tags, non-scalar
  keys, and additional documents are rejected.

The loader uses `yaml.SafeLoader` with explicit duplicate-key and alias rejection,
then validates plain Python mappings into frozen, slotted dataclasses. YAML model
objects do not retain loader-specific node types.

## Execution profile

The execution profile contains machine-specific paths, executor selection, native
threads, and resources. A SLURM profile has this version-1 shape:

```yaml
schema_version: 1
executor: slurm

paths:
  dataset_root: /archive/path/to/source
  output_root: /archive/path/to/results
  worker_local_directory: /project/path/to/scratch

native_threads:
  itk: 1
  blas: 1

slurm:
  queue: 512GB
  workers: 3
  cores_per_worker: 72
  processes_per_worker: 1
  memory: 500GB
  walltime: "72:00:00"
  interface: ib0
  scheduler_interface: ib0
  startup_timeout: "15m"
  task_retries: 0
  job_name: cyexm-registration
  job_extra_directives: []
```

Sequential profiles contain `executor: sequential` and omit local-Dask and SLURM
sections. Local-Dask profiles contain `executor: local_dask` plus `workers`,
`threads_per_worker`, `processes`, `memory_limit`, `startup_timeout`, and
`task_retries`.

### Execution field rules

- `schema_version` is the built-in integer `1`.
- Exactly one executor-specific section matches `executor`.
- Paths may be absolute or relative to the execution-profile file. Planning resolves
  them to absolute paths and records the resolved values.
- Dataset and output roots must be disjoint: neither root may contain the other.
  Planning also rejects any channel/output collision after path resolution.
- `worker_local_directory` must not be under `/archive` in the documented HPC
  configuration. It is created by workers only after preflight and belongs on active
  project or node-local storage.
- Native-thread values are explicit built-in integers of at least one. Omitting the
  section resolves Phase 1's inherited/default policy before task submission; every
  task carries the resolved immutable values.
- Worker, core, process, and retry counts are bounded non-negative or positive
  built-in integers as appropriate. Booleans are rejected.
- SLURM memory, walltime, interface, job name, queue, and timeout values are
  non-empty single-line strings.
- `processes_per_worker` is exactly `1` in Phase 2.
- `job_extra_directives` rejects newlines, NUL, and directives that conflict with
  managed queue, memory, core, process, output, or error settings.
- Unknown keys, duplicate YAML mapping keys, aliases, custom tags, and additional
  documents are rejected exactly as in the scientific manifest.

The checked-in 512 GB example uses current observed cluster values, but the README
will identify them as a site-specific starting point. Implementation and HPC
acceptance must recheck partition availability, effective memory, network interface,
shared mounts, and permitted directives.

## Immutable models

The public configuration and workflow model types are:

- `DatasetManifest`
- `ReferenceSpec`
- `RegistrationSpec`
- `RoundSpec`
- `ChannelSpec`
- `ExecutionProfile`
- `NativeThreadSpec`
- `LocalDaskSpec`
- `SlurmSpec`
- `WorkflowPlan`
- `RoundTask`
- `ChannelTask`
- `RoundResult`
- `WorkflowResult`
- `WorkflowStatus`

All are frozen, slotted dataclasses containing only primitives, tuples, `Path`
objects, and other immutable models. `RoundTask` is directly serializable by Dask and
contains no loaded image, logger, backend, Dask client, or open file handle.

`WorkflowPlan` stores:

- schema version and dataset ID;
- resolved fixed image, dataset root, and output root;
- canonical scientific settings;
- resolved native-thread policy;
- canonically ordered round tasks;
- expected output paths; and
- a deterministic scientific-plan fingerprint.

The fingerprint excludes executor mode, worker counts, queue, walltime, scheduler
addresses, and logging settings. Changing execution machinery must not by itself
invalidate scientific outputs.

## Public Python API

Phase 2 exposes these functions from a workflow namespace:

```python
load_dataset_manifest(path: str | Path) -> DatasetManifest
load_execution_profile(path: str | Path) -> ExecutionProfile
build_workflow_plan(
    manifest: DatasetManifest,
    execution: ExecutionProfile,
) -> WorkflowPlan
run_workflow(
    plan: WorkflowPlan,
    *,
    force_rounds: frozenset[int] = frozenset(),
    force_all: bool = False,
) -> WorkflowResult
inspect_workflow_status(output_root: str | Path, dataset_id: str) -> WorkflowStatus
```

`build_workflow_plan` performs filesystem/header validation but creates no directory,
file, client, cluster, or worker. `inspect_workflow_status` is read-only. Calling
`run_workflow` with both `force_all=True` and non-empty `force_rounds` is invalid.

Executor classes are implementation APIs rather than top-level `cyexm` exports. The
stable Phase 1 `cyexm.registration` exports remain unchanged until the new
artifact-managed channel transform API has its own acceptance evidence.

## Command-line interface

The package adds the `cyexm` console command with these operations:

```text
cyexm workflow validate DATASET_YAML --execution EXECUTION_YAML
cyexm workflow plan DATASET_YAML --execution EXECUTION_YAML
cyexm workflow run DATASET_YAML --execution EXECUTION_YAML
cyexm workflow status OUTPUT_ROOT DATASET_ID
```

`validate` loads both documents and validates input headers and paths. `plan` prints
the canonical plan and resource request without writing outputs or starting an
executor. `run` resumes by default and accepts repeatable `--force-round ROUND` or
mutually exclusive `--force-all`. `status` validates plan, round, and channel records
without importing scientific or scheduler runtimes.

Force options recompute artifacts only within a scientifically compatible plan.
They never accept, replace, or bypass a mismatched scientific-plan fingerprint.

Common flags include `--log-level`, `--json`, and `--version`. `run` returns these
process exit codes:

| Code | Meaning |
| --- | --- |
| `0` | All rounds completed or were validly reused |
| `2` | YAML, configuration, plan, input, or output validation failed |
| `3` | Executor startup or worker preflight failed before round execution |
| `4` | One or more round tasks failed after execution began |
| `130` | Operator interruption |

The CLI writes diagnostics to stderr and structured results to stdout when `--json`
is selected. No selected executor silently falls back to a different executor.

## Deterministic planning

Planning performs all cheap validation before creating output or scheduler state:

1. Decode both YAML documents under strict version-1 rules.
2. Resolve execution roots and every scientific input path.
3. Aggregate missing inputs and unsafe paths into one validation error.
4. Inspect each unique TIFF header once and require a compatible three-dimensional
   scalar image. Record shape, dtype, axes, and resolved geometry evidence.
5. Verify that all channels share the reference shape expected by the transform
   application boundary.
6. Validate registration choices and strategy-specific chunk settings.
7. Derive isolated round output paths and reject collisions.
8. Build `RoundTask` objects in ascending round order and channels in manifest order.
9. Compare the canonical scientific plan with any prior plan snapshot.
10. Classify each round as complete, resumable, or pending from artifact records.
11. Print the complete plan and resource request.

Validation errors identify the YAML path, offending value, expected contract, and
all affected filesystem paths. Planning never returns a partially valid plan.

## Output layout and persistent ownership

Every round gets an isolated Phase 1 output directory because fixed-reference and
metric filenames are not all round-scoped:

```text
<output-root>/<dataset-id>/
  scientific-plan.json
  workflow-runs/
    <run-id>.json
  logs/
    <run-id>.log
  rounds/
    round-002/
      registration/
        linear_transform_2.mat
        linear_transform_2.mat.complete.json
        nonlinear_warp_2.nii.gz
        nonlinear_warp_2.nii.gz.complete.json
        ...Phase 1 diagnostic artifacts...
      channels/
        round_2_488_tubulin.tif
        round_2_488_tubulin.tif.complete.json
        round_2_561_bcatenin.tif
        round_2_561_bcatenin.tif.complete.json
      logs/
        round-002.log
```

`scientific-plan.json` is an atomically written canonical provenance snapshot. An
existing snapshot with a different scientific fingerprint causes planning to fail
with a field-level difference. Version 1 requires a new output root for the changed
scientific plan; force options cannot override this failure. Execution-profile
snapshots and SLURM details live in per-invocation workflow-run records and do not
change the scientific fingerprint.

Workflow-run records are atomically updated operational summaries. They never make an
artifact reusable. Deleting them does not invalidate scientific artifacts. A data
file without its valid artifact completion record is incomplete.

## Restart-safe channel transformation

Phase 2 adds one scheduler-neutral registration primitive for a single source
channel. Its public operation accepts fixed and moving paths, accepted affine and
warp paths, output path, XYZ spacing, interpolation, native threads, and force
policy. It:

1. configures or verifies the resolved native-thread policy;
2. validates source, fixed, affine, and warp identities;
3. loads fixed and moving images using the Phase 1 geometry boundary;
4. applies `[warp, affine]` in the established order without inversion;
5. uses the manifest's explicit scalar interpolation;
6. preserves fixed-reference geometry;
7. clips to the uint16 range without intensity normalization;
8. writes to an atomic temporary destination;
9. validates the written TIFF; and
10. commits an artifact completion record.

The request identity contains fixed/source/affine/warp identities, XYZ spacing,
interpolation, output format, geometry convention, postprocessing policy, and
relevant version fields. A changed transform, source, geometry, interpolation, or
policy invalidates reuse.

Each round transforms channels serially after registration. The designated
registration channel is included. Its channel output is the original intensity image
resampled through the persisted transform chain without normalization. Phase 1's
`nonlinear_registered_<round>.tif` remains a separate registration diagnostic with
its established scalar postprocessing. The README must explain this distinction.

## Round-task contract

`run_round_task(task: RoundTask) -> RoundResult` is a top-level serializable function.
It performs these steps:

1. Apply the task's resolved native-thread environment before scientific imports.
2. Import the Phase 1 registration and channel-transform operations lazily.
3. Run `ImageRegistration` or `ChunkedImageRegistration` in the round's isolated
   registration directory.
4. Preserve Phase 1 force, restart, and artifact validation behavior.
5. Transform every manifest channel serially into the round's channel directory.
6. Validate final completion records and return a JSON-serializable result.

The task does not create a Dask client, submit child tasks, open a cluster, or use
multiprocessing. It writes a per-round log to shared output storage and returns:

- dataset and round identifiers;
- reused/completed/failed status;
- start/end time and duration;
- hostname, process ID, Dask task key when available, and `SLURM_JOB_ID` when present;
- registration strategy and resolved native threads;
- artifact paths and reuse decisions; and
- structured failure information.

## Executor contract

Executors consume a complete `WorkflowPlan` and return one `RoundResult` per round.
They never decide scientific validity.

### Sequential executor

Runs pending tasks in canonical round order in the client process. It is the
debugging, restart, and numerical reference. Complete rounds are reported as reused
without invoking the round worker.

### Local Dask executor

Creates a process-based `LocalCluster` lazily using the execution profile. Each
worker advertises `round_slot=1`, and each submitted round task requires one
`round_slot`. Worker count is bounded by pending rounds and configured workers. Local
Dask tests use deterministic synthetic tasks and temporary paths.

### SLURM executor

Creates `SLURMCluster` lazily after planning succeeds. It uses one worker process per
job and scales to `min(pending_rounds, configured_workers)` jobs. Each worker
advertises `round_slot=1`; every round task consumes it. This resource contract, not
the Dask thread count, guarantees at most one round per allocated node.

The initial characterized profile targets:

- partition `512GB`;
- one process per job;
- 72 requested cores;
- 500 GB Dask memory limit, leaving headroom below advertised node memory;
- 72-hour walltime;
- zero automatic task retries; and
- an operator-configurable scheduler/worker interface.

These are example values, not unconditional cluster facts. HPC acceptance must
record the exact submitted script and observed allocation.

After workers connect, the client probes every worker for:

- Cy-ExM and dependency versions;
- hostname and SLURM job identifier;
- dataset and output visibility;
- worker scratch creation and deletion;
- resolved native-thread environment;
- usable memory evidence; and
- scheduler/worker network identity.

Any failed probe closes the cluster and exits with executor-startup status before an
expensive round begins. Phase 2 uses the coordinator process where the CLI runs; a
batch-launched coordinator is outside this design.

## Failure, retry, and cancellation policy

### Validation and startup

Configuration and plan errors are aggregated and reported before output or scheduler
creation. Executor startup, worker timeout, or preflight failure aborts the run. The
runtime does not continue sequentially after a requested Dask/SLURM failure.

### Round failures

Rounds are independent. A failed round does not cancel other submitted rounds by
default. Successful work remains committed and reusable. After all submitted futures
settle, the workflow writes a summary and exits nonzero if any round failed.

Automatic task retries default to zero. A nonzero bounded value applies at the
executor boundary, but every retry re-enters the normal artifact-validation path.
The README warns that deterministic registration errors and out-of-memory failures
should be diagnosed rather than repeatedly retried.

### Operator cancellation

On `KeyboardInterrupt` or termination:

1. stop new submission;
2. cancel unfinished futures;
3. close the Dask client and owned cluster;
4. allow SLURM worker jobs to terminate;
5. preserve committed artifacts and stale/incomplete evidence; and
6. atomically write a cancelled operational summary.

The command returns exit code `130`. A later ordinary run resumes through artifact
validation.

### Error types

Phase 2 adds focused exceptions for manifest decoding, configuration validation,
plan validation, executor startup, worker preflight, round execution, and aggregate
workflow failure. A round failure records the round number, stage, worker hostname,
Dask task key, SLURM job ID when present, exception type/message, and log path.
Tracebacks remain in logs; concise structured errors remain serializable.

## README contract

The top-level `README.md` is required to document the complete accepted Phase 2
surface, not merely link to internal design artifacts. It will contain:

- the Phase 1 versus Phase 2 boundary;
- the scheduler-neutral round-task architecture;
- installation commands for base, workflow, distributed, and HPC extras;
- the scientific manifest and execution-profile separation;
- every version-1 YAML field and validation rule;
- whole-image and serial-chunked examples;
- validate, plan, run, status, resume, and force commands;
- sequential, local-Dask, and SLURM behavior;
- native-thread and one-round-per-node resource behavior;
- the 512 GB example with site-specific warnings;
- output layout and diagnostic-versus-channel output distinctions;
- artifact reuse, incompatible-plan, failure, retry, and cancellation semantics;
- logs, JSON output, exit codes, and status inspection;
- HPC acceptance scope and remaining limitations; and
- links to all checked-in YAML examples.

README prose and commands land only after the corresponding runtime exists on the
branch. Documentation tests assert exact command names, example paths, and current
behavior so the README cannot lead production behavior.

## Checked-in YAML examples

The repository root `examples/AGENTS.md` governs these files directly; no new example
subdirectory is required:

- `examples/spheroid-dataset.yaml`
- `examples/execution-sequential.yaml`
- `examples/execution-local-dask.yaml`
- `examples/execution-slurm-512gb.yaml`

The spheroid manifest uses the approved round/channel filenames, registration
channels, strategies, and `(0.324, 0.324, 0.900)` micrometer spacing without an
absolute lab path. Execution examples use portable relative roots and clearly label
the SLURM resource values as site-specific. Tests load every example through the
public strict parser. README commands reference the exact filenames.

## Testing strategy

### Unit tests

Unit tests use plain Python data and temporary files. They cover:

- safe YAML behavior, schema versions, types, unknown/duplicate keys, aliases, tags,
  and additional documents;
- path traversal and output-collision rejection;
- immutable model normalization;
- strategy and interpolation validation;
- plan ordering, fingerprinting, prior-plan comparison, and status classification;
- executor selection and CLI exit codes;
- channel artifact request identities and stale/reuse behavior with fakes;
- Dask/SLURM argument construction with fake modules; and
- import contracts proving config/planning do not import scientific or scheduler
  runtimes.

### Registration integration

Small synthetic TIFFs exercise real Phase 1 whole-image and serial-chunked round
tasks. Tests verify transform order, masks, geometry, interpolation, uint16 policy,
atomic output, completion records, force behavior, and restart after injected stage
failure.

### Local Dask integration

A bounded process-based `LocalCluster` executes multiple deterministic synthetic
round tasks. Tests verify one `round_slot` per worker, concurrent placement, result
collection, independent failure, bounded retry, cancellation, client/cluster cleanup,
and unchanged numerical output relative to sequential execution.

### SLURM construction tests

Portable tests replace Dask/SLURM modules with fakes and assert exact queue, memory,
cores, process count, job directives, scheduler options, resource tokens, startup
timeout, worker count, and cleanup behavior. They submit no jobs.

### Documentation and examples

Tests parse all four example YAML files, assert that README commands match the CLI,
and require documentation for every field, exit code, output purpose, executor,
restart rule, and HPC limitation.

### HPC acceptance

Tests marked `hpc` require an explicit allocation and are never part of portable CI.
On the `512GB` partition they must demonstrate:

1. one round task per node with distinct job IDs and hostnames;
2. shared dataset/output visibility on every worker;
3. scratch creation, cleanup, memory evidence, and recorded submitted job scripts;
4. whole-image and serial-chunked execution;
5. numerical agreement with sequential Phase 1 output under existing characterized
   tolerances;
6. valid restart after cancellation and after one failed worker;
7. isolation of a failed round while other rounds finish;
8. Dask client, cluster, and SLURM job cleanup; and
9. complete per-round and machine-readable workflow evidence.

HPC evidence uses bounded synthetic or approved representative inputs and never
touches the ongoing spheroid process or its outputs.

## Delivery sequence

The implementation plan will divide Phase 2 into five reviewable milestones:

1. Strict YAML models, deterministic planner, sequential executor, public API, and
   CLI.
2. Restart-safe transformed-channel artifacts and round-task integration for both
   Phase 1 strategies.
3. Local Dask executor and bounded distributed integration tests.
4. SLURM executor, worker preflight, lifecycle, and portable construction tests.
5. HPC acceptance, complete README documentation, and all four YAML examples.

Each milestone uses TDD, targeted review, and the portable quality gates. A milestone
may be committed and reviewed independently, but Phase 2 is not a released feature
until the complete design acceptance criteria pass.

## Environment assumptions to verify

Implementation and HPC acceptance must record fresh evidence for:

- existence and availability of the `512GB` partition;
- actual CPUs and usable memory per selected node;
- account, QoS, and directive requirements;
- scheduler and worker network interfaces;
- login/coordinator process policy and stability for long runs;
- `/project` and `/archive` visibility from workers;
- worker scratch capacity and cleanup behavior;
- Dask worker environment activation and Cy-ExM version identity;
- scheduler startup and teardown time; and
- resource-token placement across multiple round tasks.

No unchecked environment assumption becomes a hard-coded default. The checked-in
SLURM YAML is an editable example profile.

## Acceptance criteria

Phase 2 is complete when:

1. strict manifest and execution-profile parsers enforce the version-1 contracts;
2. plan/dry-run performs no filesystem or scheduler mutation;
3. the sequential executor is the numerical and restart reference;
4. whole-image and serial-chunked round tasks reuse Phase 1 artifacts;
5. every listed channel has an independently valid atomic output and completion
   record;
6. local Dask executes multiple rounds through the same task contract;
7. SLURM submits at most one round to each dedicated 512 GB node;
8. failed, cancelled, resumed, forced, and fully reused workflows behave as specified;
9. configuration/planning imports remain free of scientific and scheduler runtimes;
10. base Phase 1 installation and APIs remain usable without Phase 2 extras;
11. all portable unit, integration, formatting, lint, type, and pre-commit gates pass;
12. explicit HPC acceptance passes with recorded job and numerical evidence;
13. `README.md` documents the complete accepted runtime; and
14. all four checked-in YAML examples parse and match README commands.
