# Cy-ExM repository guidance

## Project contract

- The distribution is `cy-exm`, the import package is `cyexm`, and supported Python
  versions are `>=3.12,<3.13`.
- Cy-ExM is an independent extraction from the read-only ClearEx repository at
  `/project/bioinformatics/Danuser_lab/Dean/dean/git/clearex`, pinned to commit
  `2311a873b783477add7391ff00245f76490147f0`. Never edit that source repository or
  add a runtime dependency on ClearEx, and retain its academic-use `LICENSE.md`
  exactly.
- Phase 1 is the current stable runtime: whole-image and serial chunked registration,
  restart-safe artifacts, strict geometry/TIFF boundaries, and configurable native
  threads. Preserve that API and its acceptance evidence as Phase 2 is implemented.
- Phase 2B is the current workflow surface: Phase 2A's strict scientific dataset
  manifests and execution profiles, deterministic planning and status inspection,
  sequential execution, and `cyexm workflow` CLI now drive the real scheduler-neutral
  round runner and restart-safe transformed-channel artifacts. Dask and SLURM
  executors remain later milestones.
- All Phase 2 additions follow the approved Phase 2 runtime design and implementation
  plans. Keep optional workflow, distributed, and HPC dependencies out of the base
  installation, and do not advertise a later milestone before its acceptance gate.
- Scientific dataset manifests describe rounds, channels, registration inputs, voxel
  spacing, and scientific choices; machine-specific local or HPC settings belong in
  separate execution profiles. The Phase 1 artifact store remains the only restart
  authority, and the serial chunk workflow remains the numerical reference. Do not
  introduce nested chunk multiprocessing, a Dask chunk graph, or
  `ParallelChunkedImageRegistration` inside round-level tasks by default.
- `clearex.registration.pipeline` and `clearex.registration.tre` remain out of scope;
  do not recreate unrelated ClearEx surfaces in this repository.
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
uv lock --check
uv sync --locked --group dev --extra examples
uv run pytest -m "unit and not hpc" tests/unit -v
uv run pytest -m "integration and not hpc" tests/integration -v
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pre-commit run --all-files
```

Run bare `uv lock` only when an approved dependency change intentionally regenerates
`uv.lock`; inspect and commit that lockfile change with the dependency edit.

Tests marked `hpc` require an explicit lab-storage or cluster allocation and are not
part of portable CI.
