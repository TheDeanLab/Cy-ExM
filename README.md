# Cy-ExM

Cy-ExM provides reusable cyclic expansion microscopy registration without a runtime
dependency on ClearEx. Phase 1 supports restartable whole-image registration and a
restartable **serial** chunked nonlinear workflow. Both use affine registration
before nonlinear registration and preserve physical image geometry.

Cy-ExM requires Python 3.12 and retains the academic-research-use terms in
[LICENSE.md](LICENSE.md).

## Installation

Create a lean locked runtime plus notebook/example dependencies with uv:

```bash
uv sync --no-dev --extra examples
```

For development, include the development group:

```bash
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
)
```

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

## Phase 2 boundary

Phase 1 does not implement a YAML scientific dataset manifest, round-level Dask task
submission, or SLURM cluster provisioning. Those capabilities belong to a separately
designed Phase 2. That design will keep scientific dataset description separate from
execution profiles and will build on Phase 1 completion records rather than creating
a second checkpoint mechanism.

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
