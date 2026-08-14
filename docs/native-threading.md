# Native threading and performance

## Why Cy-ExM defaults to one thread

ITK and BLAS-family libraries can create their own native threads beneath a Python
process. Uncontrolled counts have produced instability in registration runs, so
Cy-ExM defaults each native runtime to one thread. This is separate from Python,
Dask, SLURM cores, and task-slot counts. Increase a count only after a particular
dataset and environment have been characterized for scientific stability and
available memory.

## Configure threads in Python

For direct Python registration, pass an explicit immutable configuration before
ANTs, NumPy, or SciPy initializes:

```python
from cyexm.runtime import NativeThreadConfig

threads = NativeThreadConfig(itk_threads=1, blas_threads=1)
```

Explicit values override inherited environment values; inherited values override the
one-thread default. Explicit counts must be built-in integers of at least one, so
booleans, floats, and strings are not coerced. Inherited counts remain base-10
strings; surrounding whitespace and a leading `+` remain accepted, while fractional,
non-numeric, zero, and negative values are rejected. Cy-ExM resolves every value
before changing the environment.

## Configure threads in an execution profile

Set both resolved library counts in the execution profile:

```yaml
native_threads:
  itk: 1
  blas: 1
```

The scientific dataset manifest intentionally has no thread setting: manifests
describe scientific data and choices, while profiles describe machine resources. The
profile values are positive built-in integers and are carried into every round task.
For SLURM, each value must be no larger than the derived cores per process. The
[execution-profile reference](execution-profile.md) defines profile validation and
resource division.

## Resolution precedence

For each native runtime, explicit configuration wins over an inherited environment
value, which wins over the default of one. When `native_threads` is omitted from an
execution profile, Cy-ExM resolves the inherited environment before building tasks.
All five inherited BLAS variables must agree; otherwise loading fails and asks for an
explicit `blas` value. A late change after a scientific runtime has loaded is rejected
unless every effective environment value is exactly idempotent.

## Environment variables

| Configuration value | Environment variable or variables |
| --- | --- |
| `itk` | `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` |
| `blas` | `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` |

Cy-ExM resolves and exports all six values together. Do not set only one BLAS-family
variable and assume the others follow it.

## Import-time boundary

Configure native threads before ANTs, NumPy, or SciPy imports. A sequential run
applies the resolved native-thread environment in the coordinator before workflow
status inspection or registration begins; the round boundary then reapplies the same
values idempotently. Dask and SLURM coordinators do not apply task native-thread
settings. Their separate worker processes receive and apply the values before
scientific imports.

Planning stays lightweight: TIFF header inspection runs in an isolated read-only
probe process and does not import NumPy or tifffile in the coordinator. Inside an
already configured round worker, source and completion-evidence TIFF headers are
inspected in-process; workers do not launch nested header-probe child processes.

## Thread controls that are not interchangeable

Dask `threads_per_worker`, requested SLURM cores, native ITK/BLAS threads, and a
`round_slot` answer different questions. A worker process may have Dask threads and
allocated cores, but each coarse round or fine scientific task consumes one slot at a
time. Use the [parallel and HPC guide](parallel-hpc.md) to choose task placement;
native counts control libraries inside the process that received the task.

## Observed benchmark

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

## Reproducibility and tuning guidance

These measurements characterize one small cropped dataset, not universal scaling or
stability. One thread remains the conservative default. The 8/8 setting captured
most of the measured acceleration and is the validated moderate choice for the next
larger trial. Although 30/30 was fastest here, it should require full-volume stability
evidence before becoming a routine setting. See the [multithreaded reproducibility
evidence](phase2e-multithreaded-registration-reproducibility.md) for the measured
scientific-equivalence context; it is characterization, not a general tolerance or
seed policy.
