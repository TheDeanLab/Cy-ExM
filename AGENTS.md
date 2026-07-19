# Cy-ExM repository guidance

## Project contract

- The distribution is `cy-exm`, the import package is `cyexm`, and supported Python
  versions are `>=3.12,<3.13`.
- Cy-ExM is an independent extraction from the read-only ClearEx repository at
  `/project/bioinformatics/Danuser_lab/Dean/dean/git/clearex`, pinned to commit
  `2311a873b783477add7391ff00245f76490147f0`. Never edit that source repository or
  add a runtime dependency on ClearEx, and retain its academic-use `LICENSE.md`
  exactly.
- Phase 1 contains whole-image and serial chunked registration only. Do not add Dask,
  distributed execution, SLURM integration, YAML workflow configuration, chunk
  multiprocessing, `ParallelChunkedImageRegistration`,
  `clearex.registration.pipeline`, or `clearex.registration.tre`.
- Child `AGENTS.md` files add scope-specific guidance and inherit, rather than repeat,
  rules from their ancestors.

## Architecture

- Separate registration policy and orchestration from I/O, runtime configuration, and
  metrics. The source-tree guidance defines the permitted dependency direction and
  import contract.
- Artifacts from expensive stages must be written atomically and validated with
  matching completion metadata before reuse.

## Scientific invariants

- Fixed and moving roles do not change. Public pixel sizes are micrometers in
  `(x, y, z)` order; NumPy volumes use `(z, y, x)` order.
- Preserve masks based on values greater than zero, affine-before-nonlinear transform
  order, TRSAA linear registration, SyNOnly nonlinear registration, scalar linear
  interpolation, mask nearest-neighbor interpolation, and ANTs origin, spacing, and
  direction.
- Preserve the legacy class defaults: the `low` linear preset and the `high`
  nonlinear preset.
- Default each ITK and BLAS-family native runtime to one thread unless an inherited
  environment value or explicit configuration overrides that variable.
- Treat transform, axis, interpolation, schedule, and tolerance changes as scientific
  changes that require explicit rationale and focused characterization tests.

## HPC storage and source safety

Before every `uv` command, export these exact locations:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
```

Keep active environments and caches on `/project`. Use `/archive` only for datasets
and large registration outputs. Do not modify or terminate the separate spheroid
registration run or its output directory.

## Project commands

Run targeted tests first, then the relevant portable gates:

```bash
uv lock
uv sync --group dev --extra examples
uv run pytest tests/unit -v
uv run pytest -m integration tests/integration -v
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pre-commit run --all-files
```

Tests marked `hpc` require an explicit lab-storage or cluster allocation and are not
part of portable CI.
