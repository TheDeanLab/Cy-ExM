# README and Public Documentation Reorganization Design

**Date:** 2026-08-14

**Status:** Approved design; implementation pending

**Scope:** Public README and Jupyter Book information architecture, content
ownership, navigation, and documentation-contract tests

## Context

The repository README currently serves simultaneously as a project landing page,
installation guide, registration tutorial, workflow tutorial, YAML schema reference,
artifact reference, HPC operations guide, acceptance report, and developer guide. It
is 916 lines and approximately 6,800 words. By comparison, the maintained Getting
started and Workflow guide pages are brief, so much of the public documentation is
available only by reading the README from beginning to end.

The current organization also asks scientific readers to understand internal project
phase names, Dask and SLURM implementation details, strict schemas, and historical
acceptance boundaries before they can identify the simplest supported registration
path. Installation begins with `uv sync` without first explaining how to obtain a
source checkout. The quota-limited HPC note does not name the required environment
variables or give executable setup steps.

This design makes the README a concise first-use page and moves durable technical
content into task-oriented public documentation. It preserves implementation and
scientific contracts while changing how readers encounter them.

## Audience

The primary audience is scientists registering volumetric cyclic expansion
microscopy data. Readers may be comfortable with Python and image analysis without
being experts in computer vision, distributed computing, or HPC administration.

Secondary audiences are:

- experienced users configuring local parallel or SLURM execution;
- contributors maintaining code, dependencies, tests, and documentation; and
- reviewers examining characterization and acceptance evidence.

## Goals

1. Limit the README to what Cy-ExM is, source installation, one whole-image Python
   example, a brief YAML-workflow doorway, documentation links, and license terms.
2. Organize the public book around reader tasks and scientific decisions rather than
   implementation chronology.
3. Explain whole-image and serial-chunked registration before introducing execution
   environments or task granularity.
4. Give manifests, execution profiles, CLI behavior, artifact schemas, threading,
   and HPC operations canonical homes under Advanced usage.
5. Give preprocessing a dedicated user-guide page that preserves the boundary
   `raw -> crop -> feature estimate -> transform raw`.
6. Add a public developer area for environment setup, checks, documentation builds,
   architecture, project history, and dependency compatibility policy.
7. Preserve historical acceptance evidence while confining internal phase labels to
   developer history and evidence records.
8. Move documentation tests with their content so behavioral contracts remain
   checked without requiring the README to remain a monolith.

## Non-goals

- No registration, configuration, execution, artifact, CLI, or public Python API
  behavior changes.
- No scientific default, transform order, interpolation, axis, metric, radius,
  schedule, tolerance, preprocessing, or native-thread default changes.
- No dependency or lockfile changes.
- No creation of downloadable example microscopy data.
- No claim that checked-in YAML files run without caller-provided images and edited
  paths.
- No deletion or rewriting of historical design, migration, parity, or acceptance
  records. Historical material that is currently excluded from the public book
  remains excluded.
- No new publishing system; the existing Jupyter Book and GitHub Pages workflow
  remain authoritative.

## Alternatives considered

### Selected: concise README plus task-oriented public book

This approach separates first use, scientific guidance, operational reference,
developer maintenance, and evidence. It requires the largest initial content move
but gives each contract one durable home and best fits a scientific audience.

### Rejected: concise README plus one expanded workflow guide

Moving most content into one large guide would reduce README length but reproduce the
same navigation problem inside the documentation. Scientific concepts, schemas, and
HPC operations would remain interleaved.

### Rejected: concise README plus reference-first documentation

Organizing mainly around API, CLI, and YAML reference would be precise for experienced
developers but would not help scientists decide which registration strategy or
execution path suits their data.

## Information architecture

The public book will use this conceptual hierarchy:

```text
Cy-ExM documentation
|- Start here
|  |- Getting started
|  |- Choosing a registration strategy
|  `- Running a YAML workflow
|- User guide
|  |- Image geometry, voxel spacing, and axes
|  |- Registration preprocessing
|  |- Feature-derived registration masks
|  `- Outputs, restart, and recovery
|- Advanced usage
|  |- Dataset manifest reference
|  |- Execution profile reference
|  |- Command-line reference
|  |- Native threading and performance
|  |- Local parallel and HPC execution
|  |- Fine-grained execution
|  `- Artifact and schema reference
|- Developer guide
|  |- Development environment and checks
|  `- Architecture and project history
`- Characterization and evidence
   |- Scientific characterization reports
   |- Multithreaded reproducibility evidence
   |- Registration-library acceptance
   `- Workflow-runtime acceptance
```

The exact `docs/myst.yml` navigation labels will use capability names. Historical
phase labels will not appear in the README, documentation home, Getting started,
registration guidance, workflow tutorial, or Advanced usage. Existing acceptance
records may retain their phase names inside the record so old plans and evidence can
be interpreted accurately.

## Conceptual boundaries

The first registration overview must distinguish three independent decisions:

| Decision | Public choices | Meaning |
| --- | --- | --- |
| Scientific registration strategy | `whole_image`, `serial_chunked` | How one moving round is registered |
| Execution environment | sequential, local process-based, SLURM | Where workflow tasks run |
| Task granularity | coarse, fine-grained | How workflow work is admitted to an executor |

Whole-image versus serial-chunked registration is introduced first. Execution
environment and granularity are separate operational choices and must not be
presented as additional registration strategies.

The registration overview will explain in plain language that both strategies use a
whole-image affine stage followed by nonlinear registration. Whole-image registration
estimates the nonlinear transform over the complete fixed grid. Serial-chunked
registration processes overlapping nonlinear regions serially and is the
memory-conscious, restartable alternative. Fine-grained execution may schedule
scientific chunk and channel tasks independently, but does not create a third public
registration strategy.

## README contract

The README target is approximately 100-150 lines. Its sections are:

1. `# Cy-ExM`
2. `## Installation`
3. `## Register two images`
4. `## Register multiple imaging rounds`
5. `## Documentation`
6. `## License`

The opening description will state that Cy-ExM registers volumetric images across
cyclic expansion microscopy rounds, supports whole-image and memory-conscious
chunked registration, applies transforms to additional channels, and resumes
validated intermediate work after interruption. It will name Python 3.12 and link
prominently to the hosted documentation.

The README will not contain:

- internal phase terminology;
- historical migration or parity discussion;
- native-thread resolution or benchmark tables;
- complete manifest, profile, CLI, or artifact schemas;
- preprocessing implementation details;
- executor, Dask, or SLURM internals;
- acceptance narratives or project non-goals; or
- developer checks and dependency-maintenance policy.

### Installation

The repository currently has no release or published-package installation path. The
README therefore gives a complete source-checkout route:

```bash
git clone https://github.com/TheDeanLab/Cy-ExM.git
cd Cy-ExM
uv sync --locked --no-dev
```

Python 3.12 and `uv` are named as prerequisites. Optional workflow, distributed,
HPC, examples, notebooks, and development environments are described in the book.

### Minimal whole-image example

The single README example uses the public `register_round()` API and the conservative
native-thread default:

```python
from pathlib import Path

from cyexm.registration import register_round

register_round(
    fixed_image_path=Path("data/reference.tif"),
    moving_image_path=Path("data/moving.tif"),
    save_directory=Path("results/round-2"),
    imaging_round=2,
    pixel_size=(0.15, 0.15, 0.40),
    linear_accuracy="low",
    nonlinear_accuracy="high",
    nonlinear_metric="CC",
)
```

Surrounding prose states that callers supply scalar 3-D TIFFs and their physical
voxel spacing. Detailed XYZ/ZYX interpretation, CC-radius considerations, TIFF
validation, threads, artifacts, and restart semantics are linked rather than
duplicated.

### YAML doorway

The multi-round section contains one short conceptual paragraph and links to:

- the YAML workflow tutorial;
- the checked-in scientific manifest;
- the sequential execution profile; and
- all checked-in workflow starting points.

It explicitly says that the examples require caller-provided images and edited
paths. It does not present validation or execution commands against absent example
data as an immediately runnable demo.

## Proposed public pages

Existing pages are reused when their purpose remains recognizable. New pages are
added only when the content has a distinct reader task.

### Start here

- `docs/index.md`: compact orientation, task routes, audience routes, and the three
  decision boundaries.
- `docs/getting-started.md`: source checkout, base and optional environments, use of
  `uv run`, input prerequisites, and the first whole-image run.
- `docs/registration-foundations.md`: rewritten as the high-level registration
  strategy guide, including the shared affine-then-nonlinear pipeline and a
  whole-image versus serial-chunked decision table.
- `docs/workflow-guide.md`: rewritten as a tutorial explaining manifest/profile
  separation, checked-in starting points, `validate -> plan -> run -> status`,
  ordinary resume, force selection, and links to exact references.

### User guide

- `docs/image-geometry.md`: new practical guide to physical XYZ spacing, NumPy/TIFF
  ZYX arrays, fixed versus moving geometry, the first fixed-grid boundary, and valid
  scalar 3-D TIFF inputs. Strict TIFF metadata details link to the artifact/schema
  reference.
- `docs/preprocessing.md`: new user guide for enabling or disabling registration
  features, units, algorithm purpose, transform/raw-output boundary, cache effects,
  masks, and characterization links.
- `docs/feature-derived-registration-masks.md`: retained as the focused mask page and
  linked from preprocessing.
- `docs/outputs-and-recovery.md`: new conceptual guide to output locations, resume,
  force, status, diagnostic versus transformed channels, and common recovery cases.

### Advanced usage

- `docs/dataset-manifest.md`: new canonical scientific manifest schema and strict
  validation reference.
- `docs/execution-profile.md`: new canonical operational profile schema, paths,
  executor selection, resources, timeouts, and validation reference.
- `docs/cli-reference.md`: new canonical command, option, structured output, logging,
  failure, and exit-code reference.
- `docs/native-threading.md`: new canonical thread precedence, environment mapping,
  configuration boundary, benchmark, reproducibility caveat, and tuning guidance.
- `docs/parallel-hpc.md`: new progressive guide to sequential, local process-based,
  and SLURM execution; resource concepts; worker isolation; coordinator constraints;
  and site validation.
- `docs/fine-grained-execution.md`: retained as the detailed opt-in graph and
  lifecycle guide, with a shorter non-expert introduction.
- `docs/registration-artifacts.md`: retained as the exact artifact, completion-record,
  cache-validation, prepared-bundle, and schema reference. Conceptual recovery
  material moves to `outputs-and-recovery.md` and links back here.

### Developer guide

- `docs/development.md`: new contributor environment, focused and portable checks,
  documentation build, dependency compatibility policy, and API/documentation
  conventions.
- `docs/architecture.md`: new capability-based architecture, supported boundaries,
  persistent-state ownership, and short project-history note needed to interpret
  older phase-labeled plans and acceptance records.

The optional dependency policy is preserved in the developer guide:

> The optional dependency versions, including Dask/distributed 2025.1.0 and
> dask-jobqueue 0.9.0, are locked; change them only with fresh compatibility
> evidence.

The existing API documentation policy remains authoritative. The developer guide
links to it rather than reproducing the full policy.

### Characterization and evidence

Existing characterization and acceptance pages remain evidence records. Navigation
labels describe capabilities rather than phases where possible. The detailed
multithreaded reproducibility record becomes publicly navigable from Native threading
and the evidence section.

## Content ownership and migration

| Current README content | Canonical destination | Treatment |
| --- | --- | --- |
| Opening capability description | README and `docs/index.md` | Rewrite without phase history |
| Base installation | README and Getting started | Add clone and prerequisites |
| Optional extras | Getting started | Explain by user task |
| Locked optional versions | Developer guide | Preserve compatibility statement |
| Quota-limited HPC note | Parallel/HPC guide | Replace with executable steps |
| Native threads and benchmark | Native threading | Consolidate |
| Whole-image registration | README and registration foundations | Minimal example plus full guide |
| Pixel/axis conventions | Image geometry | Consolidate practical guidance |
| Strict TIFF geometry | Image geometry and artifact reference | Split practical and schema detail |
| Serial chunked registration | Registration foundations | Compare with whole image |
| Outputs and restart | Outputs/recovery and artifact reference | Split conceptual and exact detail |
| Checked-in examples | Relevant tutorials | Link at point of use |
| Historical migration/parity links | Excluded historical records | Remove from public entry points |
| Workflow runtime overview | Workflow guide and architecture | Rewrite by capability |
| Checked-in YAML starting points | Workflow guide | Label as editable, not data-complete |
| Dataset manifest table | Dataset manifest reference | Move intact, improve introductions |
| Preprocessing | Preprocessing and mask guide | Consolidate, link evidence |
| Execution profile table | Execution profile reference | Move intact, improve introductions |
| Workflow commands | Workflow guide and CLI reference | Split tutorial from exact contract |
| Output tree and status internals | Outputs/recovery and artifact reference | Split user actions from schema |
| Executor behavior | Parallel/HPC and fine-grained guides | Explain progressively |
| Failure, retry, cancellation, exit codes | CLI and Parallel/HPC | Organize by user response |
| SLURM acceptance boundary | Evidence pages | Keep limitations linked, not embedded |
| Remaining non-goals | Architecture | State as supported boundaries |
| Development checks | Developer guide | Consolidate contributor workflow |

## Quota-limited HPC setup

The public HPC guide replaces the vague cache-location sentence with a generic,
site-adaptable sequence:

```bash
export UV_CACHE_DIR=/active/project/path/uv/cache
export UV_PYTHON_INSTALL_DIR=/active/project/path/uv/python

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

uv cache dir
uv python dir
uv sync --locked --extra hpc
```

The guide explains that both directories belong on an active project or approved
high-performance filesystem, not a quota-limited home directory or archival storage.
Readers must replace the placeholders with site-approved paths and arrange for the
exports to exist in interactive shells or job setup as appropriate. Cluster
partition, account, QoS, interface, memory, shared-mount, scratch, and coordinator
facts remain site-specific validation requirements.

## Scientific and architectural invariants

Documentation movement must preserve these current contracts:

- Public physical spacing is micrometers in XYZ order; NumPy volumes use ZYX order.
- Fixed and moving roles do not change.
- Affine registration precedes nonlinear registration.
- Affine application is the first fixed-grid boundary.
- Scalar interpolation is linear; mask interpolation is nearest-neighbor.
- Raw greater-than-zero diagnostic masks remain distinct from optional feature masks.
- Preprocessing estimates transforms from features and applies accepted transforms to
  raw registration and dependent-channel data.
- Whole-image and serial-chunked workflows remain restart-safe through matching
  artifact completion records.
- Artifact completion records, not operational executor summaries, remain the only
  reuse authority.
- The sequential serial-chunk workflow remains the numerical and restart reference.
- The conservative native-thread default remains one unless inherited or explicitly
  configured.
- Execution mode and machine resources do not change scientific identity.
- Scheduler selection never silently falls back to another executor.

No documentation simplification may imply a new scientific default or broader HPC
acceptance than the current evidence supports.

## Documentation data flow

Readers enter through the README or hosted documentation home. Both direct a first
user to Getting started and registration strategy selection. A multi-round user then
moves to the YAML tutorial, which links to separate scientific-manifest and
execution-profile references. Operational detail flows from those references into
threading, parallel/HPC, fine-grained, CLI, and artifact pages. Scientific claims link
downward to characterization and acceptance evidence instead of embedding full audit
narratives in task guides.

Each exact contract has one canonical owner. Other pages summarize it in plain
language and link to the owner. This prevents a future schema or behavior change from
requiring the same table to be edited in the README and several guides.

## Failure and edge-case documentation

The reorganization must retain user-visible guidance for:

- invalid input TIFF geometry or spacing;
- invalid manifests and profiles;
- planning errors before executor construction;
- incomplete, stale, or incompatible artifacts;
- reference-round failure and moving-round independence;
- executor startup and preflight failure;
- bounded retry and deterministic failures;
- interruption, cancellation, cleanup, and ordinary resume;
- diagnostic registration images versus original-intensity transformed channels;
- incompatible scientific plans requiring a new output root; and
- site-specific SLURM assumptions that cannot be inferred by Cy-ExM.

The tutorial pages explain what the user should do. Exact output, error, and exit-code
contracts belong in the CLI, execution-profile, and artifact references.

## Testing strategy

Existing documentation tests currently require detailed contracts to appear
specifically in `README.md`. Those assertions move with the canonical content.

### README scope

Tests verify:

- required top-level sections;
- source installation and the public whole-image example;
- the hosted documentation URL;
- links to the YAML workflow and starting points; and
- absence of internal phase terms, historical migration material, full schema
  tables, and detailed executor internals.

### Page contracts

- Workflow tests verify the four-command sequence and checked-in starting points in
  `workflow-guide.md`.
- Manifest tests verify every public scientific field, unit, accepted value, and
  strict validation rule in `dataset-manifest.md`.
- Execution tests verify profile fields, paths, resource units, executor constraints,
  and timeout boundaries in `execution-profile.md`.
- CLI tests verify force options, structured-output behavior, logging, failures, and
  exit codes in `cli-reference.md`.
- Thread tests verify precedence, environment variables, the conservative default,
  benchmark values, and reproducibility limitations in `native-threading.md`.
- Artifact tests verify filenames, completion records, restart authority, prepared
  data, status semantics, and diagnostic/transformed distinctions in
  `registration-artifacts.md` and `outputs-and-recovery.md`.
- Navigation tests verify every public page is in `docs/myst.yml` and intentionally
  excluded historical material remains outside the public book.

Tests continue to assert exact commands, field names, units, schemas, safety
boundaries, and measured results. They should not require incidental prose to use one
exact sentence or live in the README when a canonical reference page owns it.

### Build and visual verification

Implementation validation includes:

```bash
uv run pytest -m "unit and not hpc" tests/unit -v
uv run pytest -m "integration and not hpc" tests/integration -v
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pre-commit run --all-files
cd docs && uv run --group docs jupyter book build --html
```

The rendered book is inspected for navigation clarity, link resolution, table width,
code-block wrapping, and readable desktop and narrow layouts. The README is inspected
in GitHub-flavored Markdown rendering.

## Acceptance criteria

1. The README is approximately 100-150 lines and contains only the approved landing,
   installation, minimal example, YAML doorway, documentation, and license content.
2. A new scientific reader can obtain the source, create the base locked environment,
   and identify how to run the whole-image API without encountering internal phase or
   HPC terminology.
3. The documentation navigation reflects the approved Start here, User guide,
   Advanced usage, Developer guide, and Characterization/evidence hierarchy.
4. Whole-image and serial-chunked strategies are compared before execution
   environments and task granularity are introduced.
5. Every substantive README contract in the migration table exists in its canonical
   destination and is not silently discarded.
6. Dataset-manifest, execution-profile, CLI, artifact, and thread references remain
   exact and synchronized with current implementation behavior.
7. The HPC guide names `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR`, provides setup and
   verification commands, and avoids lab-specific filesystem paths.
8. Developer documentation preserves the locked optional-dependency policy and all
   portable development checks.
9. Internal phase terms are absent from the README and ordinary user/advanced guides;
   they remain only where needed to interpret historical developer or evidence
   records.
10. Historical excluded records remain present and outside public navigation.
11. Documentation tests assert canonical pages rather than forcing advanced content
    back into the README.
12. Portable unit/integration, lint, format, typing, pre-commit, and Jupyter Book
    gates pass, and rendered navigation and responsive readability are visually
    acceptable.

## Implementation boundary

This specification authorizes a documentation-only implementation plan. Any runtime,
schema, CLI, dependency, lockfile, scientific-default, or public Python API change
discovered during implementation requires a separate design decision rather than
being folded into the documentation cleanup.
