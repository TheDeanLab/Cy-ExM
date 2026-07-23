# Cy-ExM

Cy-ExM provides reusable cyclic expansion microscopy registration without a runtime
dependency on ClearEx. Its Phase 1 API supports restartable whole-image registration
and a restartable **serial** chunked nonlinear workflow. Its Phase 2 workflow adds
strict YAML configuration, immutable round planning, sequential and local-Dask
execution, and a coarse round-level SLURM adapter accepted on the audited lab
`512GB` profile. Fine-grained local execution is available, while its real-SLURM
production gate remains pending. Both
registration strategies use affine registration before nonlinear registration and
preserve physical image geometry. Nonlinear SyN supports either legacy Mattes mutual
information or local cross-correlation (CC) with a configurable scalar voxel radius.

Cy-ExM requires Python 3.12 and retains the academic-research-use terms in
[LICENSE.md](LICENSE.md).

## Installation

Create the locked base environment for the Phase 1 Python API:

```bash
uv sync --locked
```

Install only the optional capability needed by a workflow. The extras are
cumulative: `distributed` includes `workflow`, and `hpc` includes `distributed`.

```bash
# YAML loading and sequential workflow execution
uv sync --locked --extra workflow

# Process-based local Dask execution
uv sync --locked --extra distributed

# SLURMCluster support through dask-jobqueue
uv sync --locked --extra hpc

# Notebooks and parameterized example scripts
uv sync --locked --extra examples
```

For a lean non-development environment, add `--no-dev`. The optional dependency
versions, including Dask/distributed 2025.1.0 and dask-jobqueue 0.9.0, are locked;
change them only with fresh compatibility evidence.

`uv sync` does not activate `.venv`, so all workflow commands below use `uv run` to
execute in the managed project environment.

The established Phase 1 notebook/example convenience commands remain available:

```bash
# Lean Phase 1 runtime plus examples
uv sync --no-dev --extra examples

# Phase 1 examples plus the development group
uv sync --group dev --extra examples
```

On a quota-limited HPC host, point uv's cache and managed Python installation at the
active project filesystem before either command.

## Native threads

Prior registration runs showed instability when ITK and BLAS-family libraries used
uncontrolled thread counts. Cy-ExM therefore defaults each native runtime to one
thread. Pass an explicit immutable configuration when a dataset and environment have
been validated at higher counts:

```python
from cyexm.runtime import NativeThreadConfig

threads = NativeThreadConfig(itk_threads=1, blas_threads=1)
```

Explicit values override inherited environment values; inherited values override the
one-thread default. Explicit counts must be built-in integers of at least one, so
booleans, floats, and strings are not coerced. Inherited counts remain base-10
strings; surrounding whitespace and a leading `+` remain accepted, while fractional,
non-numeric, zero, and negative values are rejected. Cy-ExM resolves every value
before changing the environment. Configure threads before ANTs, NumPy, or SciPy
initializes; a focused error is raised instead of pretending a late change took
effect.

### Observed threading benchmark

A controlled cropped-spheroid SLURM benchmark compared otherwise identical
high-accuracy Rigid and SyNOnly workflows on two dedicated 512-GB nodes. The dataset,
scientific fingerprint, registration schedules, and output checks were identical;
only the ITK and BLAS-family thread counts changed.

| ITK / BLAS threads | Total runtime | Speedup versus 1/1 |
| --- | ---: | ---: |
| 1 / 1 | 310.5 s | 1.00x |
| 8 / 8 | 88.3 s | 3.52x |
| 30 / 30 | 69.5 s | 4.47x |

The correlation results were scientifically consistent but not bit-for-bit
identical. Pre-registration metrics were exact across runs; the largest
nonlinear-correlation change was 0.003416 (0.411%), final transformed-channel
correlations changed by at most 0.000835, and multithreaded final images retained
voxel-level Pearson correlations of 0.9979 to 0.9991 against the 1/1 outputs.

These measurements characterize one small cropped dataset, not universal scaling or
stability. One thread remains the conservative default. The 8/8 setting captured
most of the measured acceleration and is the validated moderate choice for the next
larger trial. Although 30/30 was fastest here, it should require full-volume stability
evidence before becoming a routine setting.

## Whole-image registration

Public pixel size is expressed in micrometers in `(x, y, z)` order. NumPy and TIFF
volume axes remain `(z, y, x)`.

ANTs stores image arrays and indexes in native `(x, y, z)` order. Cy-ExM transposes
the first three spatial axes at its orchestration boundary; vector components remain
the final axis and keep physical `(x, y, z)` component order. Use
`ants_to_external_array()` and `external_to_ants_array()` when crossing that boundary
directly. `ImageGeometry.shape` is external ZYX, while origin, spacing, and direction
are physical XYZ.

Cy-ExM-written ANTs TIFFs carry strict, versioned shape/origin/spacing/direction
metadata and round-trip that complete geometry. Plain TIFF fallback accepts only an
explicit ZYX volume, a grayscale QYX stack, or tifffile's reconstructed YX view of a
declared leading singleton Z axis. Sample/channel layouts such as YXS, SYX, and CYX
are rejected. Accepted plain TIFFs use caller-supplied XYZ spacing, zero origin, and
identity direction; a present or raw-but-truncated Cy-ExM payload is rejected instead
of ignored.

```python
from pathlib import Path

from cyexm.registration import register_round
from cyexm.runtime import NativeThreadConfig

register_round(
    fixed_image_path=Path("data/fixed.tif"),
    moving_image_path=Path("data/round2_registration_channel.tif"),
    save_directory=Path("results/round2"),
    imaging_round=2,
    pixel_size=(0.15, 0.15, 0.40),
    native_threads=NativeThreadConfig(itk_threads=1, blas_threads=1),
    linear_accuracy="low",
    nonlinear_accuracy="high",
    nonlinear_metric="CC",
    # nonlinear_radius=4,  # Optional for CC; omission normalizes to 4.
)
```

The standalone Phase 1 Python API retains its legacy omitted-argument behavior:
`nonlinear_metric="mattes"` with 32 bins. CC accepts one positive built-in integer
`nonlinear_radius`; an omitted CC radius normalizes to the provisional value 4.

The equivalent command-line example is:

```bash
uv run python examples/scripts/whole_image_registration.py \
  data/fixed.tif data/round2_registration_channel.tif results/round2 \
  --round 2 --pixel-size-x 0.15 --pixel-size-y 0.15 --pixel-size-z 0.40
```

## Serial chunked registration

`ChunkedImageRegistration` estimates one whole-image affine transform, then processes
the nonlinear field serially in overlapping chunks. `chunk_size` follows array order
`(z, y, x)`; pixel size remains public `(x, y, z)` spacing.

```python
from pathlib import Path

from cyexm.registration import ChunkedImageRegistration
from cyexm.runtime import NativeThreadConfig

registration = ChunkedImageRegistration(
    fixed_image_path=Path("data/fixed.tif"),
    moving_image_path=Path("data/round2_registration_channel.tif"),
    save_directory=Path("results/round2-chunked"),
    imaging_round=2,
    pixel_size=(0.15, 0.15, 0.40),
    chunk_size=(512, 512, 512),
    overlap_fraction=0.15,
    native_threads=NativeThreadConfig(itk_threads=1, blas_threads=1),
    nonlinear_metric="CC",
)
registration.register()
```

Chunk execution is deliberately serial in Phase 1. It is the trusted numerical and
restart reference for later orchestration work.

## Outputs and restart

Expensive stages are written to temporary sibling files, validated, promoted
atomically, and then committed with a matching `.complete.json` record. A nonlinear
failure leaves a reusable affine transform; a chunk failure leaves earlier valid
local warp records. Rerun the same request to resume. Use `--force` (or
`force_override=True`) only for intentional recomputation.

Use a dedicated output directory for each fixed/moving request because fixed-reference
and metric filenames are not round-scoped. See
[registration artifacts and recovery](docs/registration-artifacts.md) for every
filename, cache-validation rule, stale marker, and recovery case.

## Examples and migration

Five parameterized scripts live in `examples/scripts`; six cleared notebooks live in
`examples/notebooks`. They require caller-provided paths and do not download data.

Existing users should consult [migration from ClearEx](docs/migration-from-clearex.md)
for the import map, downstream consumer note, moved examples, and intentionally
corrected legacy behavior. The numerical extraction evidence is recorded in the
[legacy parity report](docs/legacy-parity-report.md).
The complete portable gate and measured Phase 1 evidence are recorded in the
[Phase 1 acceptance audit](docs/phase-1-acceptance.md).
The real-cluster orchestration and scientific evidence are recorded in the
[Phase 2 acceptance audit](docs/phase-2-acceptance.md).

## Phase 2 workflow

Phase 2 orchestrates the accepted Phase 1 registration implementations; it does not
replace their scientific or restart behavior. A **scientific dataset manifest**
contains portable inputs and scientific choices. A separate **execution profile**
contains paths, native threads, executor selection, and machine resources. This
separation lets the same manifest run sequentially, on a local workstation, or on an
HPC system without changing its scientific identity.

Planning validates inputs and creates one immutable `ReferenceTask` plus one immutable
`RoundTask` per moving round. The reference task first materializes every declared
reference channel on one shared reference grid. The selected executor then schedules
moving work only after that hard executor barrier succeeds. In `coarse` mode,
registration and channel transformation remain serial inside each moving-round task.
In opt-in `fine_grained` mode, the coordinator expands each round into immutable
preparation, nonlinear-chunk, assembly, channel, and validation tasks. Workers never
submit children or create a nested scheduler. Phase 1 completion records are the only
restart authority, and no scheduler setting enters the artifact cache.
There is no nested chunk multiprocessing in either mode.

`parallelism.mode` belongs only in the execution profile and accepts exact `coarse`
or `fine_grained`. Omitting `parallelism` normalizes to `coarse`. Execution mode is
excluded from scientific identity, so the scientific fingerprint is unchanged when
switching between compatible coarse and fine execution. See
[Fine-grained registration](docs/fine-grained-execution.md) for the graph, internal
artifacts, resource lifecycle, restart behavior, and current acceptance boundary.

The public workflow API is available from `cyexm.workflow`:

```python
from cyexm.workflow import (
    build_workflow_plan,
    inspect_workflow_status,
    load_dataset_manifest,
    load_execution_profile,
    run_workflow,
)

manifest = load_dataset_manifest("examples/spheroid-dataset.yaml")
execution = load_execution_profile("examples/execution-sequential.yaml")
plan = build_workflow_plan(manifest, execution)
result = run_workflow(plan)  # resume is the default
status = inspect_workflow_status(
    execution.paths.output_root,
    manifest.dataset_id,
)
```

`build_workflow_plan` performs path and TIFF-header validation but creates no files,
directories, scheduler clients, clusters, or workers: planning is mutation-free.
TIFF header inspection runs in an isolated read-only probe process, so planning does
not import NumPy or tifffile in the coordinator or change its thread environment.
`inspect_workflow_status` is also read-only. Scheduler libraries and scientific
runtimes are imported lazily only at the selected execution boundary.

### Checked-in YAML examples

The repository provides four version-1 starting points:

- [Spheroid scientific dataset manifest](examples/spheroid-dataset.yaml)
- [Sequential execution profile](examples/execution-sequential.yaml)
- [Local-Dask execution profile](examples/execution-local-dask.yaml)
- [Editable 512-GB SLURM profile](examples/execution-slurm-512gb.yaml)

The manifest contains no lab-specific root. Edit each execution profile's paths and
resources before use. In particular, do not run the SLURM placeholders unchanged.

### Scientific dataset manifest

Every mapping has exact keys. Required scientific selections are explicit; the only
nonlinear metric normalization is CC's documented optional radius default. The fields
are:

| Location | Type and accepted values | Meaning |
| --- | --- | --- |
| `schema_version` | built-in integer `1` | Selects the strict manifest contract. |
| `dataset_id` | string matching `[A-Za-z0-9][A-Za-z0-9._-]*` | Becomes the dataset output-directory name. |
| `voxel_spacing_um` / `x`, `y`, `z` | three finite positive numbers | Physical voxel spacing in micrometers in XYZ order. |
| `reference` / `round` | nonnegative built-in integer | Fixed-reference imaging round. |
| `reference` / `registration_channel` | safe relative POSIX path | Fixed image used for every moving-round registration; it must occur exactly once in `reference.channels`. |
| `reference` / `channels` | nonempty ordered list with unique `file` values and basenames | Every reference-round image copied to the shared reference output in declaration order. |
| reference channel `file` | safe relative POSIX path | Source below `paths.dataset_root`; its basename becomes the reference channel output name. |
| `registration` / `crop` | built-in boolean | Enables or disables Phase 1 fixed/moving foreground cropping. |
| `registration` / `linear` / `type` | `Translation`, `Rigid`, `Similarity`, `Affine`, or `TRSAA` | Phase 1 linear method. |
| `registration` / `linear` / `accuracy` | `high`, `medium`, `low`, or `dry run` | Phase 1 linear schedule. |
| `registration` / `nonlinear` / `type` | `Elastic`, `SyNOnly`, or `SyN` | Phase 1 nonlinear method. |
| `registration` / `nonlinear` / `accuracy` | `high`, `medium`, `low`, or `dry run` | Phase 1 nonlinear schedule. |
| `registration` / `nonlinear` / `metric` | exact `CC` or `mattes` | Required nonlinear similarity metric. Affine registration remains Mattes/32 independently. |
| `registration` / `nonlinear` / `radius` | positive built-in integer | Optional only for `CC`; omission normalizes to the provisional default 4. Rejected for `mattes`, which uses 32 bins. |
| `rounds` | non-empty list | Moving-round objects; moving rounds are unique and cannot equal the reference round. |
| moving `round` | nonnegative built-in integer | Round identifier used in task and output names. |
| `strategy` | `whole_image` or `serial_chunked` | Selects the accepted Phase 1 registration implementation. |
| `registration_channel` | safe relative POSIX path | The moving image used to estimate transforms. The registration channel must be listed in that round's `channels`. |
| `channels` | list with unique `file` values | Original moving channels resampled through the round transform, in manifest order. |
| channel `file` | safe relative POSIX path | Source below `paths.dataset_root`; its basename becomes the channel output name. |
| channel `interpolation` | `linear` or `nearestNeighbor` | Explicit scalar interpolation for that channel. |
| `chunk_size_zyx` | three positive built-in integers | Required only for `serial_chunked`; chunk size in NumPy ZYX order. Forbidden for `whole_image`. |
| `overlap_fraction` | finite number satisfying `0 <= value < 1` | Required only for `serial_chunked`; forbidden for `whole_image`. |

Scientific image paths must be safe relative POSIX paths. Absolute paths, empty or
`.` components, `..`, NULs, newlines, and paths that resolve outside the execution
profile's dataset root are rejected. Round numbers are built-in integers rather than
booleans, channel files are unique within a round, and the registration channel must
match a channel entry exactly. All declared reference channels must have identical
native shape, origin, spacing, and direction because they were acquired together; if
cropping is enabled, Cy-ExM estimates one crop from the reference registration channel
and applies those scientifically identical bounds to every reference channel. Moving
registration and dependent-channel images may differ from the reference along any
spatial axis: ANTs resamples them onto the fixed reference grid before nonlinear
chunking and before final channel export. Planning requires every input to be a scalar
3-D TIFF with manifest-compatible physical spacing.

Schema version 1 intentionally uses only `reference.registration_channel` and
`reference.channels`. The retired singular `reference.channel` form is rejected;
there is no dual parser or compatibility schema to maintain.

ANTs exposes CC radius as one scalar voxel count, not independent per-axis radii.
Radius 4 therefore means a 9-by-9-by-9 voxel neighborhood. With anisotropic voxel
spacing that neighborhood is physically anisotropic even though its voxel radius is
equal along X, Y, and Z. The default of 4 is provisional pending dataset-specific
characterization; it is not a claim of optimality.

The loader uses safe YAML decoding and rejects unknown keys, missing keys, duplicate
mapping keys, YAML aliases, custom tags, non-scalar keys, specialized collection
tags, a non-mapping document, invalid UTF-8, and additional YAML documents. These
rules apply to both YAML files.

### Execution profile

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
| `slurm` / `workers` | positive built-in integer | Maximum number of worker jobs. |
| `slurm` / `cores_per_worker` | positive built-in integer | Requested cores and job CPU count. |
| `slurm` / `processes_per_worker` | built-in integer `1` | One Dask worker process per SLURM job; `processes_per_worker` must be exactly `1`. |
| `slurm` / `memory` | positive whole decimal `KB`, `MB`, `GB`, or `TB` string | Semantic memory request; the executor rounds up to whole binary GiB for SLURM and Dask. |
| `slurm` / `walltime` | non-empty single-line string | Job walltime. |
| `slurm` / `interface` | non-empty single-line string | Worker network interface. |
| `slurm` / `scheduler_interface` | non-empty single-line string | Coordinator/scheduler network interface. |
| `slurm` / `startup_timeout` | non-empty single-line string | Bounds only Dask `Client` connection and `wait_for_workers`. |
| `slurm` / `task_retries` | nonnegative built-in integer | Executor-managed retry count after the first attempt. |
| `slurm` / `job_name` | non-empty single-line string | Owned worker-job name. |
| `slurm` / `job_extra_directives` | list of non-empty single-line strings | Site additions that do not conflict with managed SLURM options. |

Sequential profiles omit both `local_dask` and `slurm`. Local-Dask profiles contain
only `local_dask`; SLURM profiles contain only `slurm`. Exactly one
executor-specific section must match `executor`, and unknown or mismatched sections
are rejected.

Absolute profile paths remain absolute; relative paths resolve from the
execution-profile file, not the current working directory. Dataset and output roots
must be disjoint after resolution: neither may contain the other. Existing symlinks
and planned outputs are checked for root escapes and collisions. Distributed
worker-backed executors require `worker_local_directory`, and worker scratch must not
be `/archive` or a descendant of `/archive`; use active `/project` or approved
node-local storage instead.

All worker/core fields are positive built-in integers and retry fields are a
nonnegative built-in integer; booleans are never accepted as counts. SLURM memory
uses positive whole decimal memory with exact uppercase `KB`, `MB`, `GB`, or `TB`
suffixes. These profile units are decimal semantic bytes. The executor rounds upward
to whole binary GiB, passes the matching `GiB` quantity to Dask, and never passes a
`GB` suffix to `#SBATCH --mem`. For example, `500GB` becomes `466G`, which is
500,363,689,984 bytes, and Dask receives `466GiB`. Invalid, fractional, zero,
unsupported, or overflowing values fail configuration. Operational evidence records
both semantic and scheduler-normalized memory.

Every other SLURM text value is a non-empty single-line string.
`job_extra_directives` cannot conflict with
managed SLURM options for partition (`--partition`, `-p`), memory (`--mem`,
`--mem-per-cpu`), CPU/task counts (`--cpus-per-task`, `-c`, `--ntasks`, `-n`), or
stdout/stderr (`--output`, `-o`, `--error`, `-e`), including unambiguous long-option
abbreviations and grouped or attached short forms.

`startup_timeout` is not an end-to-end startup deadline. It does not bound
`LocalCluster` or `SLURMCluster` construction, does not bound `scheduler_info()`
calls, and does not bound the `Client.run()` worker-preflight RPC. Those operations
currently have no Cy-ExM wall-clock timeout, so a stalled constructor or preflight RPC
can outlive the configured value. Treat this as an operational limitation when
choosing where to run the coordinator.

#### Native-thread resolution

An explicit `native_threads` section overrides the inherited environment. If it is
omitted, the execution profile resolves the inherited environment before building
tasks. Missing variables use the conservative one-thread default. The five inherited
BLAS values must agree, or loading fails and asks for an explicit `blas` value.

| YAML value | Worker environment |
| --- | --- |
| `itk` | `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` |
| `blas` | `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` |

The resolved values are applied before ANTs, NumPy, or SciPy is imported in each
worker. Dask worker threads, requested SLURM cores, and ITK/BLAS native threads are
different controls. Keep ITK and BLAS at one until a particular dataset and runtime
have been characterized safely.

A sequential run applies the resolved native-thread environment in the coordinator
before workflow status inspection or registration begins; the round boundary then
reapplies the same values idempotently. Dask and SLURM coordinators do not apply task
native-thread settings. Their separate worker processes receive and apply the values
before scientific imports. Inside an already configured round worker, source and
completion-evidence TIFF headers are inspected in-process; workers do not launch
nested header-probe child processes. If a caller has already imported ANTs, NumPy,
or SciPy, `run_workflow` still rejects a conflicting late change before writing
workflow outputs; only an exactly idempotent reapplication is safe.

### Validate, plan, run, and inspect

These examples use the checked-in spheroid manifest. First edit or copy an execution
profile so its roots point to caller-owned inputs and outputs.

```bash
# Decode both files, resolve inputs, inspect headers, and construct a plan.
uv run cyexm workflow validate examples/spheroid-dataset.yaml \
  --execution examples/execution-sequential.yaml

# Print the canonical scientific plan and resource request without mutation.
uv run cyexm workflow plan examples/spheroid-dataset.yaml \
  --execution examples/execution-local-dask.yaml

# Execute pending work; an ordinary run resumes valid artifacts by default.
uv run cyexm workflow run examples/spheroid-dataset.yaml \
  --execution examples/execution-local-dask.yaml

# Recompute selected rounds. --force-round is repeatable.
uv run cyexm workflow run examples/spheroid-dataset.yaml \
  --execution examples/execution-local-dask.yaml \
  --force-round 2 --force-round 4

# Recompute every round in the compatible plan.
uv run cyexm workflow run examples/spheroid-dataset.yaml \
  --execution examples/execution-local-dask.yaml --force-all

# Inspect artifacts using only an output root and dataset ID.
uv run cyexm workflow status results spheroid-registration-example
```

`--force-round` and `--force-all` are mutually exclusive. An unknown forced round is
an error. Validate and plan both inspect real input TIFF headers; neither is a
registration dry run. The mutation-free plan prints its canonical scientific JSON
and resolved resource request but never opens the selected executor.

Common options are accepted at the top-level, `workflow`, or action boundary:

```bash
uv run cyexm workflow status results spheroid-registration-example --json
uv run cyexm workflow plan examples/spheroid-dataset.yaml \
  --execution examples/execution-sequential.yaml --log-level DEBUG --json
uv run cyexm --version
```

Human and structured success results go to stdout. `--json` is success-only:
successful workflow actions that reach result emission emit one JSON object.
Failures return before result emission, so stdout is empty, plain-text diagnostics
go to stderr, and there is no JSON error object. `--version` prints version text
rather than a JSON result. Argparse failures can include usage text, and
`--log-level DEBUG` can add a traceback to the stderr diagnostic. Option abbreviation
is disabled.

### Planning, outputs, status, and restart

The scientific fingerprint covers the manifest's scientific choices, resolved input
identities and headers, output mapping, and versioned channel policies. It excludes
executor, resource, native-thread, and logging settings. A successful `run` writes an
atomic canonical `scientific-plan.json`. A later plan with a different scientific
fingerprint fails with field-level differences: force cannot bypass a scientific-plan
fingerprint mismatch, and a scientifically incompatible plan requires a new output
root. `--force-round` and `--force-all` only request recomputation within a compatible
plan.

Schema version 1 now requires nonlinear `metric` evidence, and persisted plans record
either effective CC `radius` or fixed Mattes `bins: 32`. Plans written before this
contract fail exact validation and should be rerun under a new output root. Omitting a
CC radius and explicitly selecting radius 4 produce the same canonical plan,
fingerprint, and cache identity; changing the radius changes all three.

For reference round 1 and moving round 2, the workflow uses the following layout.
Reference sources are copied, never moved or modified. The current export boundary is
TIFF; the shared channel directory and task model keep output-format selection separate
so a future schema can add formats without changing registration semantics.
The moving-round directories are `rounds/round-002/registration/`,
`rounds/round-002/channels/`, and `rounds/round-002/logs/round-002.log`.

```text
<output-root>/<dataset-id>/
  scientific-plan.json
  workflow-runs/
    <run-id>.json
  rounds/
    round-001/
      registration/
        reference_crop_indices.json                 # only when crop: true
        reference_crop_indices.json.complete.json   # only when crop: true
      channels/
        round_1_488_tubulin.tif
        round_1_488_tubulin.tif.complete.json
        <each-other-declared-reference-basename>
        <each-other-declared-reference-basename>.complete.json
      logs/
        round-001.log
    round-002/
      registration/
        image_metrics_pre-registration.txt
        image_metrics_pre-registration.txt.complete.json
        linear_transform_2.mat
        linear_transform_2.mat.complete.json
        image_metrics_linearly-registered.txt
        image_metrics_linearly-registered.txt.complete.json
        nonlinear_warp_2.nii.gz
        nonlinear_warp_2.nii.gz.complete.json
        image_metrics_nonlinear-registered.txt
        image_metrics_nonlinear-registered.txt.complete.json
        nonlinear_registered_2.tif
        nonlinear_registered_2.tif.complete.json
        moving_crop_indices_2.json                # only when crop: true
        moving_crop_indices_2.json.complete.json  # only when crop: true
        round_2_chunks/                            # serial_chunked only
          chunk_0000_warp.nii.gz
          chunk_0000_warp.nii.gz.complete.json
          ...
          manifest.json
          manifest.json.complete.json
      channels/
        round_2_488_tubulin.tif
        round_2_488_tubulin.tif.complete.json
        <each-other-source-basename>
        <each-other-source-basename>.complete.json
      logs/
        round-002.log
```

Every reusable artifact has its own matching `.complete.json`. A file alone, a
completion record alone, an invalid identity, or incompatible request evidence is not
reusable. A normal rerun scans those Phase 1 records and submits only incomplete moving
rounds. Reference preparation still crosses the selected executor barrier so its
sources and shared-grid evidence are revalidated; its task reports `completed` with
reused artifacts when nothing is rewritten. This ordinary resume is the default.
Valid earlier affine or chunk work remains reusable after a later stage fails.

Moving rounds read the prepared registration channel directly from
`rounds/round-001/channels/`; they do not create a per-round `fixed_reference.tif`.
The reference task is a hard executor barrier for sequential, local-Dask, and SLURM
execution. A reference failure prevents moving-round submission. Forcing the reference
round invalidates all moving descendants because their fixed-input provenance changed;
forcing only a moving round leaves valid reference artifacts reusable.

`workflow-runs/<run-id>.json` is an atomic operational summary containing the
execution profile, force selection, round results, timestamps, and available SLURM
evidence. Operational summaries never authorize artifact reuse; they can be deleted
without invalidating scientific artifacts. The current CLI writes diagnostics to
stderr and per-round shared logs; it does not currently create the reserved root
`logs/<run-id>.log` path.

The read-only status command validates the scientific plan and authoritative
completion records without importing scientific or scheduler runtimes. It reports the
reference preparation as its declared round alongside moving rounds. Each known round
is classified as `pending`, `partial`, `complete`, or `invalid`, and JSON output lists
complete, incomplete, and invalid artifact paths.

#### Diagnostic image versus transformed channel

These similarly named outputs have different purposes:

- `registration/nonlinear_registered_<round>.tif` is the normalized Phase 1
  registration diagnostic produced with Phase 1's established histogram/scalar
  postprocessing.
- `channels/<source>.tif` is the original channel resampled through the persisted
  transform chain, placed in fixed-reference geometry, and clipped to uint16 without
  intensity normalization. This includes the designated registration channel.

Use the `channels/` copy for downstream original-intensity analysis. Do not treat the
Phase 1 diagnostic as a substitute for that channel.

### Executor behavior

All executors consume the same immutable tasks and return results in ascending round
order. They do not decide whether an artifact is scientifically reusable and there is
no silent fallback when a selected executor fails.

#### Sequential

The sequential executor is the numerical and restart reference. It prepares or reuses
the reference round first, then runs pending moving rounds in canonical order in the
CLI process. A reference failure stops the dependency chain; a moving-round failure
does not prevent later independent moving rounds. It opens no Dask scheduler.

#### Local Dask

`local_dask` lazily creates one owned, process-based `LocalCluster`. It starts
`min(pending tasks, configured workers)` workers in coarse mode and exactly the
configured worker cap in fine-grained mode, where preparation determines later graph
width. Every worker receives one `round_slot`. Even when all moving rounds are
reusable, the cluster runs the reference revalidation barrier; it submits no moving
work. The reference prerequisite and all subsequent work use that same cluster.
Every coarse round or fine scientific task requires one slot, so only one runs in a
worker even if its Dask `threads_per_worker` is larger. Fine ready tasks are admitted
round-robin across rounds and remain work-conserving when only one round is ready.
The client and cluster are closed for success, failure, startup error, and
interruption.

#### SLURM

`slurm` lazily constructs one owned dask-jobqueue `SLURMCluster`. Coarse mode scales
to `min(pending tasks, configured workers)`. Fine mode starts one candidate worker
for the reference barrier, then scales on demand to no more than the configured
worker cap. It requests one process per job. Fine candidates advertise no eligible
slot until the complete preflight succeeds; the coordinator then activates exactly
one `round_slot`. Reference preparation and every submitted moving round or fine
scientific task consume one slot. The reference barrier does not create a second
cluster. Today this is one worker/job per dedicated node and one task per worker;
multiple workers per node remain a future optimization. Dedicated-node truth still
depends on site allocation/exclusivity policy and must be confirmed by exact-commit
real-SLURM acceptance.

Before any registration submission, every connected worker must pass preflight.
Preflight checks exact Cy-ExM/dependency versions, hostname and SLURM job identity,
dataset/output visibility, scratch creation and cleanup, all six native-thread
values, memory evidence, and scheduler/worker network identity. A timeout or mismatch
closes owned resources and fails as executor startup rather than falling back.

The worker job prologue exports all resolved native-thread values before the worker
command. The coordinator runs in the process that invokes the CLI; a batch-launched
coordinator is not implemented. Confirm site policy and the stability of the chosen
login or interactive coordinator location before a long run.

Before SLURM submission, Cy-ExM directly serializes the configured round runner and
carries those bytes in a production-owned callable. This prevents Distributed's
outer task serialization from reducing an acceptance-only runner to a test-module
import. Workers never need `tests` on `PYTHONPATH`; the embedded runner is rebuilt
inside ordinary task execution, where a reconstruction error becomes a normal task
failure instead of a worker-side task-deserialization failure.

Real acceptance validates version-compatible SLURM resource records rather than
assuming every controller exposes the same keys. Job `TRES` must prove the requested
CPU and one node. When present, `MinMemoryNode` must equal the exact normalized
scheduler request; job `ReqTRES` and `AllocTRES` are optional but must prove adequate
CPU, node, and memory resources when reported. Node `RealMemory` must cover the
request, and reported `CfgTRES` and `AllocTRES` must cover its CPU and memory facts.
Slurm 22.05 may also report the legacy sentinel `AllocMem=0`; `AllocMem=0` is
unavailable as allocation evidence rather than proof of zero allocated memory. A
positive `AllocMem` must cover the normalized request, while malformed or negative
values fail validation.
The harness fails closed if neither an authoritative job memory request nor a node
memory-allocation fact is available.

Each SLURM workflow-run record carries versioned operational evidence: requested
workers/retries/timeouts, the rendered job-script template, connected worker and
preflight observations, task attempt counts, startup phase/outcome, timestamps,
cancellation evidence when applicable, and ordered client/cluster/configuration
cleanup results. This evidence is diagnostic only and never replaces artifact
completion records.

### Failures, retries, cancellation, and exit codes

Configuration and plan errors occur before executor construction. Startup and
preflight errors abort before expensive tasks. Independent round failures do not
cancel other submitted rounds; all submitted work settles, successful artifacts stay
committed, a failed operational summary is written, and the workflow reports the
failed round numbers.

Post-execution SLURM evidence validation is a separate exit-3 path. After submitted
rounds return, Cy-ExM validates the executor's operational evidence before reporting
success. If that validation fails, it preserves the round results in the operational
run record, the CLI exits `3`, and a later ordinary run can reuse those valid
artifacts through normal completion-record validation.

Automatic retries default to zero. A positive `task_retries` value is an
executor-managed resubmission; Dask's opaque future retry count remains zero, and
each attempt re-enters ordinary Phase 1 artifact validation. Use a small bounded
value only for plausible transient failures. Diagnose deterministic registration
errors and out-of-memory failures instead of repeatedly retrying them.

On `KeyboardInterrupt` or CLI termination, distributed executors cancel unfinished
futures and close the owned client and cluster. The SLURM adapter also leaves its
temporary Dask configuration. These steps preserve committed artifacts, write a
cancelled workflow-run summary when possible, and an ordinary later run resumes via
artifact validation.

| Exit code | Meaning |
| --- | --- |
| `0` | Every round completed or was validly reused. |
| `2` | Argument, YAML, configuration, input, plan, output, or fingerprint validation failed. |
| `3` | Executor startup, worker connection, or preflight failure; also invalid SLURM operational evidence, including validation after round execution. Valid committed artifacts remain restartable. |
| `4` | At least one round failed after execution began. |
| `130` | Operator interruption or termination through the CLI run handler. |

### 512-GB SLURM example and acceptance boundary

The checked-in profile currently proposes queue `512GB`, three workers, 72 cores per
worker, one process per worker, a semantic `500GB` memory profile (normalized to
`466G` for SLURM and `466GiB` for Dask), a 72-hour walltime, `ib0`
worker/scheduler interfaces, a 15-minute startup timeout, and zero retries. These
512GB values are site-specific and editable. They are not unconditional facts about
the current cluster; verify partition availability, exclusive-node policy, usable
memory, interfaces, account/QoS directives, shared mounts, scratch, and coordinator
policy before submission.

Portable fake-based SLURM construction, job-template rendering, preflight policy,
failure mapping, evidence validation, cancellation, and cleanup are tested. Real
SLURM submission and HPC acceptance completed on the audited lab partition for the
coarse round-level executor. Only
tests explicitly selected with `-m hpc` may contact a scheduler or use representative
lab data; they require an explicit allocation and caller-approved dataset, execution
profile, output root, and external evidence directory. Portable CI always selects
`not hpc` and must never invoke scheduler clients. Do not use an active scientific
registration or its output as acceptance input.

Real acceptance must demonstrate distinct jobs/hostnames for concurrent whole-image
and serial-chunked rounds, dedicated allocation facts, shared-path and scratch
behavior, package/thread identity, numerical parity with sequential execution,
independent failure, bounded retry, cancellation, ordinary resume, and final job
cleanup. The [Phase 2 acceptance audit](docs/phase-2-acceptance.md) records those
results using a representative multithreaded scientific-equivalence gate and a
seeded one-thread strict executor-parity gate. Acceptance is specific to the tested
commit, approved manifest, and audited execution profile; it is not an unconditional
guarantee for other clusters, resource profiles, datasets, or scientific policies.

Portable real-ANTs/Local-Dask evidence is complete for fine-grained execution through
implementation Task 10. Fine-grained real-SLURM production acceptance remains
pending its own exact-commit two-tier scientific and operational gate; the coarse
Phase 2 audit must not be read as acceptance of fine chunk/channel placement or
performance.

### Remaining non-goals

Phase 2 intentionally does not provide `ParallelChunkedImageRegistration`, chunked
or parallel whole-image affine estimation, nested multiprocessing, multiple workers
per node, job arrays, cross-dataset scheduling, cloud or Kubernetes executors, or
automatic resource optimization. Whole-image affine remains the preparation boundary
even when nonlinear chunks and dependent channels run as fine tasks. It does not
recreate ClearEx's GUI, pipeline, TRE, or general analysis framework, and it never
silently changes executor: no silent fallback is permitted.
Reference-round channels are materialized as shared TIFF outputs, with one identical
crop applied to all of them when requested. Export to other file formats remains a
future extension.

## Development checks

```bash
uv run pytest -m "unit and not hpc" tests/unit -v
uv run pytest -m "integration and not hpc" tests/integration -v
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pre-commit run --all-files
```

`import cyexm` remains lightweight: it exposes version metadata without importing
scientific runtimes, changing thread environment variables, or configuring logging.
