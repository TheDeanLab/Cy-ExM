# Cy-ExM repository guidance

## Project contract

- The distribution is `cy-exm`, the import package is `cyexm`, and supported Python
  versions are `>=3.12,<3.13`.
- Cy-ExM is an independent extraction from the read-only ClearEx repository at
  `/project/bioinformatics/Danuser_lab/Dean/dean/git/clearex`, pinned to commit
  `2311a873b783477add7391ff00245f76490147f0`. Never edit that source repository or
  add a runtime dependency on ClearEx, and retain its academic-use `LICENSE.md`
  exactly.
- The Phase 1 API remains supported: whole-image and serial chunked registration,
  restart-safe artifacts, strict geometry/TIFF boundaries, and configurable native
  threads. Preserve that API and its acceptance evidence.
- Phase 2 is the current workflow runtime. Strict scientific dataset manifests and
  separate strict execution profiles drive deterministic planning, status inspection,
  the real scheduler-neutral round runner, and restart-safe transformed-channel
  artifacts through `cyexm workflow validate`, `cyexm workflow plan`,
  `cyexm workflow run`, and `cyexm workflow status`.
- The current executor boundary includes sequential, bounded process-based local Dask,
  and SLURM executors. `parallelism.mode: coarse` is the production default;
  `fine_grained` is opt-in and changes task admission only. Portable real-ANTs and
  Local-Dask fine-graph evidence is complete through implementation Task 10.
  Fine-grained real SLURM production acceptance remains pending the exact-commit HPC
  gate and explicitly selected `hpc` tests with operator-approved inputs and
  allocation.
  Portable CI installs the HPC Python extra but never contacts an external scheduler.
- All Phase 2 additions follow the approved Phase 2 runtime design and implementation
  plans. Keep optional workflow, distributed, and HPC dependencies out of the base
  installation, and do not advertise a later milestone before its acceptance gate.
- Scientific dataset manifests describe rounds, channels, registration inputs, voxel
  spacing, and scientific choices; machine-specific local or HPC settings belong in
  separate execution profiles. The Phase 1 artifact store remains the only restart
  authority, and the serial chunk workflow remains the numerical reference. Do not
  introduce nested chunk multiprocessing, a worker-owned Dask graph, or
  `ParallelChunkedImageRegistration` inside round-level tasks. The opt-in fine graph
  is coordinator-owned and preserves every immutable scientific invariant.
- Schedule one reference-preparation prerequisite followed by either one coarse task
  per moving round or the coordinator-owned immutable fine graph. All executors use
  a hard reference barrier in the same owned execution lifecycle; reference failure
  prevents moving submission. Forcing the reference round propagates to every moving
  descendant.
  Worker-backed executors expose one `round_slot` per worker; each coarse round or
  fine scientific task consumes one slot. Fine SLURM uses one process and one task
  per dedicated job/node today; multiple workers per node require separate scientific
  and operational acceptance.
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
- Native reference, registration-channel, and dependent-channel shapes may differ
  independently along Z, Y, and X. The affine application is the first fixed-grid
  boundary: its scalar and mask outputs, every nonlinear input and output, the full
  warp, and every transformed dependent channel must match the fixed-reference
  geometry.
- Every declared reference-round channel must share one exact native geometry. Copy
  each source into the reference round's shared `channels` directory, never move it;
  when cropping is enabled, estimate one crop from the registration channel and apply
  the identical bounds to all declared reference channels.
- Preserve raw diagnostic masks based on values greater than zero. Optional
  feature-derived registration masks use strict positive asinh support, physical
  component filtering and dilation, and never replace raw diagnostic masks or raw
  outputs. Preserve affine-before-nonlinear transform order, TRSAA linear
  registration, SyNOnly nonlinear registration, scalar linear interpolation, mask
  nearest-neighbor interpolation, and ANTs origin, spacing, and direction.
- Affine registration remains Mattes/32. Nonlinear registration supports exact `CC`
  and `mattes`; CC uses one positive scalar voxel radius with omitted radius normalized
  to the provisional default 4, while Mattes uses 32 bins and rejects a radius.
- Preserve the legacy class defaults: the `low` linear preset and the `high`
  nonlinear preset.
- Default each ITK and BLAS-family native runtime to one thread unless an inherited
  environment value or explicit configuration overrides that variable.
- Treat transform, axis, interpolation, metric, radius, schedule, and tolerance
  changes as scientific changes that require explicit rationale and focused
  characterization tests.

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
uv sync --locked --group dev --extra examples --extra hpc
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

## API reuse and documentation policy

Before adding or changing a user-callable API, search source, tests, documentation,
examples, exports, the proposed name, and plausible synonyms. Identify the closest
existing symbols; prefer composition, extension, parameterization, or repair over a
parallel abstraction. Do not add a wrapper that merely renames an operation or
duplicates its lifecycle, validation, persistence, state management, or cleanup.

A user-callable API is a relatively high-level entry point, resource owner, data
model, executor, or explicitly supported extension interface. It requires full
Numpydoc. Simple pure value objects and helpers may use concise Numpydoc. Treat
documented units, shapes, coordinate systems, state transitions, side effects,
ownership, and errors as contracts.

For multi-step or user-callable API work, record a `Reuse analysis` in the
implementation plan and in the body of the API-changing commit. A PR description
links to both records and notes deviations; preserve the analysis in squash commits.
Read [the agentic API documentation guide](docs/agentic-api-documentation.md) before
designing or documenting such an API.
