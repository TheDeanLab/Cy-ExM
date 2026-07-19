# Cy-ExM Phase 1 Registration Extraction Design

- **Status:** Approved for implementation planning
- **Date:** 2026-07-19
- **Repository:** `TheDeanLab/Cy-ExM`
- **Distribution name:** `cy-exm`
- **Import package:** `cyexm`
**ClearEx extraction baseline:** `2311a873b783477add7391ff00245f76490147f0`

## Summary

Cy-ExM will become the dedicated home for the legacy whole-image registration
workflow that currently lives under `clearex.registration`. The registration
code and the rest of ClearEx use some of the same scientific dependencies, but
they are otherwise independent. A separate repository gives the workflow a
clear API, focused tests, reusable examples, and room for dataset-driven and
HPC execution without coupling those concerns to ClearEx.

This specification covers Phase 1 only: extract and stabilize the proven
whole-image and serial chunked registration paths. Phase 2 will receive its own
design and implementation cycle for YAML dataset manifests, execution profiles,
Dask, and `dask_jobqueue.SLURMCluster` orchestration.

## Background

The immediate workflow registers multiple imaging rounds serially. The current
run spent more than twelve hours in its first moving round while consuming one
logical CPU, despite exposing many operating-system threads. Inspection found
that importing `clearex.registration` globally sets ITK and common BLAS thread
counts to one. Those conservative settings were intentional responses to past
instability, but a module import should not silently enforce them for every
dataset and execution environment.

The current script also processes independent rounds in a Python loop. That is
a natural Phase 2 parallelism boundary: one complete round can become one Dask
task and one SLURM job, with no shared mutable registration state between
rounds. The design must first establish a dependable standalone registration
library on which that orchestration can rely.

The audit also confirmed that
`clearex.registration.ParallelChunkedImageRegistration` is not a dependable
implementation:

- Every submitted chunk carries both complete image objects, causing repeated
  process serialization and potentially large memory duplication.
- Worker exceptions are logged and suppressed.
- The parent still reports success and writes output arrays whose failed
  regions remain zero-filled.
- The exception suppression was introduced in ClearEx commit `04724ab1` while
  the implementation was still encountering empty chunks.
- No automated test exercises the parallel class.

The known-good `ChunkedImageRegistration` remains in scope. The parallel
subclass does not.

## Goals

Phase 1 will:

1. Create an independent, installable Python package for the stable legacy
   registration workflow.
2. Preserve established registration algorithms, transform ordering, spatial
   conventions, masking, interpolation, and output behavior unless a documented
   defect requires correction.
3. Expose whole-image and serial chunked registration through focused modules
   and a familiar public API.
4. Make native ITK and BLAS thread limits explicit, configurable, observable,
   and safe with respect to import order.
5. Make stage artifacts atomic, validated, and restartable so an expensive
   affine result survives a later nonlinear failure.
6. Move relevant registration examples and tests out of ClearEx after the new
   package has demonstrated parity.
7. Document source provenance, downstream migration paths, architectural
   boundaries, and repository-specific operating instructions.

## Non-goals

Phase 1 will not:

- Implement YAML dataset manifests or execution profiles.
- Add Dask, `distributed`, `dask-jobqueue`, or SLURM integration.
- Implement round-level or chunk-level parallel execution.
- Export `ParallelChunkedImageRegistration`, even as an experimental API.
- Extract ClearEx's newer OME-Zarr registration pipeline.
- Extract the target-registration-error module, which depends on
  `clearex.detect.particles`.
- Make ClearEx depend on Cy-ExM through a compatibility shim.
- Remove legacy ClearEx code before known consumers have migrated.
- Change scientific defaults solely for speed.

## Repository approach

Cy-ExM uses a fresh repository rather than filtering ClearEx history or wrapping
ClearEx as a dependency.

A fresh extraction has the smallest long-term coupling and permits the package
to adopt clear module boundaries. Provenance is retained through this design,
the migration map, and commit messages that identify the ClearEx source commit.
History filtering would preserve more line-level ancestry but would also carry
unrelated repository structure and make the boundary harder to establish. A
thin wrapper would be fast initially but would leave Cy-ExM unable to evolve
independently.

The proposed Phase 1 layout is:

```text
Cy-ExM/
├── AGENTS.md
├── README.md
├── LICENSE.md
├── pyproject.toml
├── uv.lock
├── src/
│   ├── AGENTS.md
│   └── cyexm/
│       ├── AGENTS.md
│       ├── __init__.py
│       ├── metrics.py
│       ├── registration/
│       │   ├── AGENTS.md
│       │   ├── __init__.py
│       │   ├── linear.py
│       │   ├── nonlinear.py
│       │   ├── transforms.py
│       │   ├── whole_image.py
│       │   └── chunked.py
│       ├── io/
│       │   ├── AGENTS.md
│       │   ├── __init__.py
│       │   ├── images.py
│       │   └── tiff.py
│       └── runtime/
│           ├── AGENTS.md
│           ├── __init__.py
│           └── threads.py
├── tests/
│   ├── AGENTS.md
│   ├── unit/
│   │   └── AGENTS.md
│   └── integration/
│       └── AGENTS.md
├── examples/
│   ├── AGENTS.md
│   ├── scripts/
│   │   └── AGENTS.md
│   └── notebooks/
│       └── AGENTS.md
└── docs/
    ├── AGENTS.md
    └── superpowers/
        └── specs/
```

Empty implementation directories will not be added merely to reserve names.
The implementation plan may create them as their contents are introduced.
Because Phase 1 extracts ClearEx source, Cy-ExM retains ClearEx's academic-use
license and copyright notice in `LICENSE.md`.

## Package boundaries

### `cyexm.registration`

The registration package owns scientific registration behavior:

- whole-image orchestration and stage lifecycle;
- serial chunk geometry, registration, and stitching;
- linear registration policy;
- nonlinear registration policy; and
- transform creation, application, validation, and composition.

It may depend on `cyexm.io`, `cyexm.runtime`, and `cyexm.metrics`. Those support
packages must not import `cyexm.registration`.

### `cyexm.io`

The I/O package owns image loading, spatial-metadata preservation, TIFF export,
and atomic file replacement. It replaces the small set of ClearEx I/O helpers
needed by the extracted workflow. It does not become a general-purpose copy of
`clearex.io`.

### `cyexm.runtime`

The runtime package owns native-thread configuration and import-order checks.
It has no dependency on ANTs, NumPy, SciPy, or the registration package.

### `cyexm.metrics`

The metrics module contains only measurements used directly by the extracted
registration workflow. The target-registration-error implementation remains in
ClearEx because of its particle-detection dependency.

### Top-level import behavior

`import cyexm` must remain lightweight. It must not import ANTs, NumPy, SciPy,
or registration implementation modules; mutate environment variables; create
logging handlers; inspect datasets; or initialize a scheduler. Scientific
imports are delayed until a registration operation begins, after native-thread
configuration has been resolved.

## Public API

Phase 1 preserves familiar entry points:

```python
from cyexm.registration import (
    ChunkedImageRegistration,
    ImageRegistration,
    register_round,
)

from cyexm.registration.linear import register_image as register_linear_image
from cyexm.registration.nonlinear import register_image as register_nonlinear_image
```

`ImageRegistration`, `ChunkedImageRegistration`, and `register_round` retain the
legacy conceptual responsibilities and core parameters. New parameters are
keyword-only so existing positional call patterns do not silently change
meaning. For compatibility, the Phase 1 class defaults remain TRSAA with the
legacy `low` linear preset and SyNOnly with the legacy `high` nonlinear preset.
Callers can select other existing presets explicitly. Phase 2's dataset workflow
will separately default to the currently used TRSAA `high` and SyNOnly `high`
combination.

`ParallelChunkedImageRegistration` is intentionally absent. Code that imports
it must fail clearly rather than receiving an implementation that can produce
plausible but incomplete output.

The implementation may introduce private dataclasses for stage state and
artifacts, but it will not expand the public API without a separately reviewed
need.

## Native thread configuration

The runtime API centers on an immutable configuration value:

```python
NativeThreadConfig(itk_threads=1, blas_threads=1)
```

The stable default is one ITK thread and one BLAS-family thread, preserving the
behavior selected after prior instability. Users may explicitly request higher
values for a validated dataset and environment.

Effective values are resolved in this order:

1. Explicit `NativeThreadConfig` values supplied to the workflow.
2. Existing process environment values.
3. Conservative package defaults of one.

The BLAS setting covers `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and
`VECLIB_MAXIMUM_THREADS`. The ITK setting covers
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`.

Configuration is applied before ANTs, NumPy, or SciPy initializes a native
runtime. If a caller attempts to change effective settings after a relevant
runtime has loaded, Cy-ExM raises a focused configuration error rather than
claiming the change took effect. Reapplying the same effective configuration is
idempotent. Every registration run logs the resolved values and whether each
came from explicit configuration, the inherited environment, or a default.

Phase 2 will apply the same configuration in each Dask worker before importing
registration modules. It will not create a second thread-control mechanism.

## Scientific invariants

Extraction must preserve the following behavior:

- Fixed and moving image roles remain unchanged.
- Public `pixel_size` tuples are expressed in micrometers in `(x, y, z)` order,
  matching ANTs physical axes and the legacy API. NumPy image arrays use
  `(z, y, x)` axis order. Native ANTs arrays and indexes use `(x, y, z)`, so the
  first three spatial axes are transposed at explicit orchestration boundaries.
  Trailing vector components are never transposed and retain physical XYZ order.
- `ImageGeometry.shape`, chunk grids, TIFF arrays, and crop bounds are external ZYX;
  origin, spacing, direction, affine parameters, and displacement components are
  physical XYZ.
- Cy-ExM ANTs TIFF outputs use strict versioned geometry metadata. Only absence of
  the custom key permits the plain-TIFF fallback; malformed present metadata fails
  closed. Cache requests and canonical manifests carry the axis-convention version.
- Operations wholly inside ANTs, including zero-mask and identity-warp construction,
  stay native XYZ and do not pass through an external-array adapter.
- The existing zero-valued-data mask semantics remain intact.
- Linear registration precedes nonlinear registration.
- The current TRSAA linear policy and SyNOnly nonlinear policy, including their
  accuracy presets, remain available.
- Transform composition and application order remain unchanged.
- Scalar images and masks retain their established interpolation policies.
- ANTs image origin, spacing, and direction are preserved in written outputs.
- Chunk overlap and stitching behavior match the serial legacy implementation
  within documented numerical tolerances.

Refactoring may divide the legacy monolithic module into focused files, but it
must not opportunistically alter these numerical choices.

## Registration lifecycle and artifacts

The stage sequence is:

```text
resolve native threads
    -> validate inputs and image metadata
    -> load images and construct masks
    -> estimate or reuse affine transform
    -> atomically persist affine transform and stage metadata
    -> estimate or reuse nonlinear warp
    -> atomically persist nonlinear warp and stage metadata
    -> apply transforms and write registered images and metrics atomically
```

Each completed expensive stage is durable before the next stage begins. A
nonlinear failure therefore leaves a reusable affine result. Chunked execution
persists completed chunk artifacts so a later run can resume rather than
discarding all finished chunks.

Files are written to a sibling temporary path and promoted with an atomic rename
only after the writer closes successfully and the artifact can be reopened. A
small completion record containing the stage metadata and artifact identity is
promoted last and acts as the cache commit marker. Interrupted temporary files
or artifacts without a matching completion record are never valid cache
entries.

Stage metadata records the normalized input identity, file size and modification
time, image shape and spatial metadata, registration method and accuracy,
thread configuration, and upstream artifact identity. Full content hashing of
multi-gigabyte images is not required. A cache entry is reusable only when its
artifact is readable and its metadata matches the current request.

If an artifact exists without valid matching metadata, Cy-ExM raises a focused
cache error instead of silently overwriting or reusing it. An explicit force
option permits recomputation. After a replacement affine stage is committed,
its nonlinear and final-image descendants are marked stale and cannot be
reused. Stale files may remain in place until atomically replaced by the new
run; they are not mistaken for current results or deleted before replacement
succeeds.

Failures include the stage or chunk identifier, relevant paths, and the
original exception. A run never reports completion when any required stage or
chunk has failed.

## Serial chunked registration

`ChunkedImageRegistration` is the only chunked implementation in the stable
Phase 1 API. Its extraction will be preceded by characterization of:

- chunk-grid construction at image boundaries;
- overlap extent calculation;
- empty-chunk rejection;
- mapping between extended and core slices;
- warp-field shape and component ordering;
- output stitching; and
- restart behavior for completed chunks.

The initial implementation remains serial. This provides a trusted numerical
reference for any future chunk-parallel design and avoids nested multiprocessing
inside future Dask round workers.

A future chunk-parallel implementation must pass an independent acceptance gate:

- numerical agreement with the serial reference;
- propagation of every worker failure;
- atomic and restartable chunk artifacts;
- bounded memory use;
- no complete-volume payload attached to every chunk submission; and
- explicit coordination of worker processes with native thread counts.

That work is not part of either the stable Phase 1 API or the initial
round-level Dask design.

## Logging and errors

Libraries use named loggers and do not configure the root logger. Each workflow
instance receives or creates instance-scoped logging context so simultaneous
runs do not share mutable handler state. Logs include round, stage, and chunk
identifiers where applicable.

Expected operational failures use focused exception types for input validation,
thread configuration, cache validity, registration stages, and chunk failures.
Exceptions retain their original cause. Library code does not terminate the
interpreter. A future CLI will translate failures into nonzero exit status at
the process boundary.

## Extraction and migration

Migration is deliberately staged:

1. Extract stable code and relevant tests from the recorded ClearEx baseline.
2. Replace imports of unrelated ClearEx utilities with focused Cy-ExM modules.
3. Establish characterization and integration parity.
4. Move registration scripts and notebooks whose primary purpose belongs to
   Cy-ExM, replacing embedded machine-specific paths with parameters.
5. Publish a usable Cy-ExM version.
6. Update known downstream consumers in separate, scoped changes.
7. Remove migrated examples and obsolete legacy code from ClearEx only after
   consumer migration is verified.

The initial import migration map is:

| ClearEx source | Cy-ExM destination | Disposition |
| --- | --- | --- |
| `clearex.registration.ImageRegistration` | `cyexm.registration.ImageRegistration` | Extract and characterize |
| `clearex.registration.ChunkedImageRegistration` | `cyexm.registration.ChunkedImageRegistration` | Extract and characterize |
| `clearex.registration.register_round` | `cyexm.registration.register_round` | Extract and characterize |
| `clearex.registration.ParallelChunkedImageRegistration` | None | Exclude as unverified |
| `clearex.registration.linear` | `cyexm.registration.linear` | Extract |
| `clearex.registration.nonlinear` | `cyexm.registration.nonlinear` | Extract |
| registration helpers in `clearex.registration.common` | `cyexm.registration.transforms` or focused local modules | Extract only required helpers |
| `clearex.registration.pipeline` | None | Keep in ClearEx |
| `clearex.registration.tre` | None | Keep in ClearEx |

Known imports under the lab's `data_analysis/dean_lab/2025-galecki` work must
continue to function during the transition. ClearEx's legacy API remains frozen
until those callers have migrated. ClearEx does not gain a Cy-ExM dependency as
a temporary forwarding layer.

Example migration includes an inventory of
`clearex/examples/scripts/registration` and
`clearex/examples/notebooks/registration`. Generated outputs and large
dataset archives are not copied into Git history. Small, redistribution-safe
fixtures are included only when they provide necessary test coverage.

## Testing strategy

### Unit tests

Unit tests cover:

- thread-setting precedence, idempotence, and late-configuration errors;
- lightweight top-level imports;
- chunk geometry and boundary conditions;
- transform ordering and interpolation selection;
- stage metadata matching and invalidation;
- atomic-write failure behavior;
- corrupt and incomplete artifact detection;
- force-recompute behavior; and
- propagation and contextualization of stage and chunk failures.

Thread and import-order tests run in fresh subprocesses so results do not depend
on the test runner's previously imported native libraries.

### Integration tests

Small synthetic 3D images exercise whole-image and serial chunked ANTs
registration. Tests assert shapes, physical metadata, finite values, transform
readability, expected improvement in similarity, and agreement between ClearEx
and Cy-ExM within test-specific documented tolerances. Floating-point results
are not required to be bitwise identical.

Failure-injection tests interrupt the workflow after affine persistence and
during nonlinear or chunk work. The following run must reuse valid upstream
artifacts, reject corrupt artifacts, and resume completed chunks.

Tests are marked as `unit`, `integration`, or `hpc`. Portable unit and bounded
integration checks run in GitHub Actions. Tests requiring lab storage, SLURM,
or representative large data use the `hpc` marker and a documented command;
they do not silently skip based on hostname.

### Quality gates

The repository uses:

- `pytest` for tests;
- Ruff for linting and formatting;
- BasedPyright for static type checking;
- pre-commit for reproducible local policy checks; and
- `uv` with a committed lockfile for project-aware commands.

Targeted tests run first during development. The full portable suite, formatter
check, lint check, and type check must pass before Phase 1 completion.

## Documentation and `AGENTS.md`

The repository uses nested `AGENTS.md` files, which Codex discovers natively,
instead of `CODEX.md`. Each file adds scope-specific guidance and inherits
nonduplicated rules from its ancestors.

- The root file defines the architecture, supported commands, dependency
  policy, HPC storage policy, and global scientific invariants.
- `src/AGENTS.md` defines package dependency direction and import-side-effect
  rules.
- `src/cyexm/AGENTS.md` defines public API and module-size expectations.
- Registration guidance documents coordinate conventions, transform order,
  interpolation, caching, and numerical-change review requirements.
- I/O guidance documents metadata preservation and atomic-write requirements.
- Runtime guidance documents native-thread and import-order invariants.
- Test guidance documents fixture size, subprocess tests, markers, tolerances,
  and failure-injection practices.
- Example guidance prohibits embedded lab paths and requires reproducible,
  parameterized examples.
- Documentation guidance distinguishes current behavior from future Phase 2
  plans.

The README covers installation, conservative thread defaults, basic whole-image
and serial chunked use, output and restart behavior, migration from ClearEx, and
the status of parallel execution.

## Phase 2 boundary

Phase 2 will be designed only after Phase 1 is reviewed and implemented. Its
expected scope is recorded to keep Phase 1 extensible without adding premature
scaffolding:

- A scientific dataset YAML manifest specifies rounds, channel filenames,
  registration channels, voxel spacing, and optionally registration methods and
  accuracy presets.
- HPC and local execution settings live in a separate execution profile or CLI
  surface rather than in the scientific manifest.
- Independent moving rounds become Dask tasks.
- On the lab cluster, each task requests a dedicated 512 GB SLURM node through
  `SLURMCluster`, with one worker process and one round task occupying that job.
- Native thread counts are supplied explicitly to each worker and begin with
  the stable one-thread defaults until higher values are validated.
- The workflow records task state and reuses Phase 1's stage artifacts for
  restart rather than inventing a second checkpoint system.
- Chunk-level multiprocessing is not nested inside round-level Dask tasks.

The Phase 2 design will verify the current cluster's partitions, resource
syntax, worker startup, shared-filesystem visibility, and memory behavior at
implementation time.

## Acceptance criteria

Phase 1 is complete when all of the following are true:

1. `cy-exm` installs reproducibly and `import cyexm` has no native-runtime,
   environment, logging, filesystem, or scheduler side effects.
2. Cy-ExM has no runtime dependency on ClearEx.
3. Whole-image and serial chunked public APIs pass characterization and
   integration tests.
4. Scientific invariants and spatial metadata are preserved.
5. Conservative native-thread defaults remain available, explicit overrides
   work before native imports, and late changes fail clearly.
6. Affine, nonlinear, chunk, image, and metrics artifacts are written atomically
   and support safe restart.
7. A failed required stage or chunk cannot produce a successful run result.
8. Relevant examples use caller-provided paths and execute against Cy-ExM.
9. The source baseline and import migration map are recorded.
10. Portable tests, Ruff formatting and linting, BasedPyright, and pre-commit
    checks pass.
11. The stable API does not expose `ParallelChunkedImageRegistration`.
12. Phase 1 contains no Dask, SLURM, YAML-workflow, or placeholder parallel
    execution implementation.

## Principal risks and mitigations

### Accidental numerical drift

Splitting the monolithic module may subtly change transform order, axis
handling, interpolation, or metadata. Characterization tests are written before
behavioral refactoring, and scientific changes require an explicit rationale.

### Native runtime initialized too early

An eager import can make thread settings ineffective. The top-level package is
lightweight, scientific imports are deferred, and subprocess tests enforce the
ordering contract.

### Invalid cache reuse

Filename existence alone is insufficient. Readability plus matching stage
metadata is required, and upstream replacement invalidates descendants.

### Premature removal from ClearEx

Examples and active analyses may still import legacy modules. The migration is
staged, known consumers are inventoried, and ClearEx remains frozen until their
imports have moved.

### Parallel complexity obscures a stable baseline

The broken process-pool implementation is excluded. Serial chunking becomes the
reference, while round-level Dask orchestration is deferred to a separate
design.
