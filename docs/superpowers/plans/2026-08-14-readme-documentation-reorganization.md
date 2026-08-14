# README and Public Documentation Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Replace the 916-line README with a concise first-use guide and move its
scientific, workflow, schema, HPC, and developer contracts into a task-oriented
public Jupyter Book without changing runtime or scientific behavior.

**Architecture:** Existing documentation pages remain canonical where their purpose
is already clear; focused pages are added for distinct reader tasks. Documentation
contract tests move from README-specific assertions to the page that owns each
contract, and the README is shortened only after every destination page exists and is
tested.

**Tech Stack:** Markdown, MyST/Jupyter Book 2.1.6, GitHub Pages, Python 3.12,
pytest, Ruff, BasedPyright, pre-commit, uv 0.7.13

## Global constraints

- Preserve all current registration, configuration, execution, artifact, CLI, and
  public Python API behavior.
- Preserve physical spacing in micrometers and XYZ order; NumPy volumes remain ZYX.
- Preserve fixed/moving roles, affine-before-nonlinear order, interpolation policies,
  preprocessing boundaries, native-thread defaults, and artifact restart authority.
- Do not change `pyproject.toml` dependencies or `uv.lock`; adding the Documentation
  project URL is the only planned `pyproject.toml` change.
- Do not change or reformat `LICENSE.md`.
- Do not delete or rewrite historical design, migration, parity, or acceptance
  records.
- Do not mention the historical source repository in the README or public book.
- Internal Phase 1/Phase 2 labels may appear only in developer history and evidence
  records, never in the README or ordinary Start here, User guide, or Advanced usage
  prose.
- Checked-in YAML files are editable configuration starting points, not runnable
  sample datasets; every relevant page must say that callers supply images and edit
  paths.
- Every exact schema, command, filename, unit, measured result, and safety boundary
  moved out of the README must remain documented and tested in its canonical page.
- Before every `uv` command on the approved HPC execution host, export exactly:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
```

- If those required `/project` locations are unavailable on the execution host, do
  not create substitute root paths or claim the full uv gates passed. Report the
  environment limitation and run only commands that do not violate repository
  guidance until execution is moved to an approved host.

## Reuse analysis

This is a documentation-only change and introduces no user-callable API. Reuse the
existing `register_round()`, `ChunkedImageRegistration`, workflow CLI, checked-in YAML
examples, and public documentation policy without wrappers or aliases.

Reuse these existing pages instead of creating parallel replacements:

- Rewrite `docs/registration-foundations.md` as the strategy-selection page.
- Rewrite `docs/workflow-guide.md` as the YAML workflow tutorial.
- Retain `docs/feature-derived-registration-masks.md` as the mask authority.
- Retain `docs/fine-grained-execution.md` as the fine-graph authority.
- Retain `docs/registration-artifacts.md` as the detailed artifact/schema authority.
- Retain all characterization and acceptance pages as evidence records.
- Link `docs/agentic-api-documentation.md` from the developer guide instead of
  duplicating its API policy.

Create a page only where the current README contains a distinct reader task with no
canonical home: image geometry, preprocessing, outputs/recovery, dataset manifest,
execution profile, CLI reference, native threading, parallel/HPC operation,
development, and architecture/history.

## File structure

### Existing files to modify

- `README.md`: final concise landing, installation, whole-image example, YAML
  doorway, documentation links, and license.
- `pyproject.toml`: add the hosted Documentation project URL only.
- `docs/myst.yml`: install the approved Start here, User guide, Advanced usage,
  Developer guide, and Characterization/evidence navigation.
- `docs/index.md`: public documentation orientation and task routes.
- `docs/getting-started.md`: source checkout and task-oriented installation.
- `docs/registration-foundations.md`: strategy selection and shared registration
  pipeline.
- `docs/workflow-guide.md`: YAML workflow tutorial.
- `docs/feature-derived-registration-masks.md`: preprocessing and evidence links.
- `docs/fine-grained-execution.md`: non-expert introduction and advanced boundary.
- `docs/registration-artifacts.md`: exact artifact/schema ownership and links to the
  conceptual recovery guide.
- `docs/phase-1-acceptance.md`: descriptive public heading while retaining the
  historical phase identity inside the record.
- `docs/phase-2-acceptance.md`: descriptive public heading while retaining the
  historical phase identity inside the record.
- `tests/unit/test_phase2_documentation.py`: relocate documentation contracts from
  README-specific checks to canonical pages.
- `tests/unit/test_repository_policy.py`: point acceptance-navigation assertions to
  the public book rather than the README.
- `tests/unit/test_examples.py`: verify the current registration documentation across
  canonical pages rather than requiring all terms in the README.

### New public pages

- `docs/image-geometry.md`: practical coordinate, geometry, and TIFF input guidance.
- `docs/preprocessing.md`: registration-feature preprocessing guide.
- `docs/outputs-and-recovery.md`: user-facing outputs, status, resume, and recovery.
- `docs/dataset-manifest.md`: exact scientific YAML schema.
- `docs/execution-profile.md`: exact machine/resource YAML schema.
- `docs/cli-reference.md`: exact workflow command and exit contract.
- `docs/native-threading.md`: exact thread resolution and benchmark.
- `docs/parallel-hpc.md`: sequential, local process-based, and SLURM operations.
- `docs/development.md`: contributor environment, gates, dependency policy, and docs
  build.
- `docs/architecture.md`: current capability architecture, boundaries, state
  ownership, and short historical phase note.

---

### Task 1: Establish first-use documentation and strategy selection

**Files:**

- Modify: `docs/index.md`
- Modify: `docs/getting-started.md`
- Modify: `docs/registration-foundations.md`
- Modify: `docs/myst.yml`
- Modify: `tests/unit/test_phase2_documentation.py`

**Interfaces:**

- Consumes: current `register_round()` and `ChunkedImageRegistration` public names,
  current source-installation commands, and the approved documentation hierarchy.
- Produces: `_document(name: str) -> str` and `_normalized_document(name: str) -> str`
  test helpers; canonical first-use pages that later tasks link to.

- [ ] **Step 1: Add reusable documentation-test helpers and failing first-use tests**

Add these helpers beside `_readme()` in
`tests/unit/test_phase2_documentation.py`:

```python
DOCS_ROOT = REPOSITORY_ROOT / "docs"


def _document(name: str) -> str:
    return (DOCS_ROOT / name).read_text(encoding="utf-8")


def _normalized_document(name: str) -> str:
    return " ".join(_document(name).split()).lower()
```

Add these complete tests:

```python
def test_public_documentation_home_routes_readers_by_task() -> None:
    index = _normalized_document("index.md")
    for required in (
        "register two images",
        "choose a registration strategy",
        "run multiple imaging rounds",
        "configure advanced or hpc execution",
        "contribute to cy-exm",
    ):
        assert required in index
    assert "phase 1" not in index
    assert "phase 2" not in index


def test_registration_foundations_separates_science_from_execution() -> None:
    guide = _normalized_document("registration-foundations.md")
    for required in (
        "whole-image registration",
        "serial-chunked registration",
        "scientific registration strategy",
        "execution environment",
        "task granularity",
        "affine",
        "nonlinear",
        "memory",
        "restart",
    ):
        assert required in guide
    assert guide.index("scientific registration strategy") < guide.index(
        "execution environment"
    )
    assert "phase 1" not in guide
    assert "phase 2" not in guide


def test_getting_started_gives_a_complete_source_checkout() -> None:
    guide = _document("getting-started.md")
    for required in (
        "git clone https://github.com/TheDeanLab/Cy-ExM.git",
        "cd Cy-ExM",
        "uv sync --locked --no-dev",
        "uv run",
        "Python 3.12",
        "caller-provided",
    ):
        assert required in guide
```

Rename `test_readme_documents_installation_and_scheduler_neutral_architecture` to
`test_getting_started_documents_installation_choices`. Retain its six installation
command assertions, read from `getting-started.md`, and remove the scheduler-neutral
assertions from that test. Task 5 reinstates those assertions against the workflow
and architecture pages before the README is shortened.

- [ ] **Step 2: Run the focused tests and verify RED**

Run on the approved host:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest \
  tests/unit/test_phase2_documentation.py::test_public_documentation_home_routes_readers_by_task \
  tests/unit/test_phase2_documentation.py::test_registration_foundations_separates_science_from_execution \
  tests/unit/test_phase2_documentation.py::test_getting_started_gives_a_complete_source_checkout \
  tests/unit/test_phase2_documentation.py::test_getting_started_documents_installation_choices \
  -v
```

Expected: all four tests fail because the current pages use phase-oriented language,
lack task routes, omit the source checkout, and do not own every installation choice.

- [ ] **Step 3: Rewrite the documentation home**

Replace `docs/index.md` with these sections and responsibilities:

````markdown
# Cy-ExM

Cy-ExM registers volumetric images acquired across cyclic expansion microscopy
rounds. It supports whole-image and memory-conscious serial-chunked registration,
applies accepted transforms to additional channels, and resumes validated
intermediate work after interruption.

Cy-ExM requires Python 3.12 and is distributed under academic-research-use terms.

## What do you want to do?

- **Register two images:** link Getting started.
- **Choose a registration strategy:** link Registration foundations.
- **Run multiple imaging rounds:** link Workflow guide.
- **Configure advanced or HPC execution:** link Fine-grained execution until the
  Parallel/HPC page exists, then update this route in Task 5.
- **Contribute to Cy-ExM:** link the existing API documentation guide until the
  Development page exists, then update this route in Task 5.

## Three separate decisions

| Decision | Choices | What it controls |
| --- | --- | --- |
| Scientific registration strategy | Whole-image or serial-chunked | How one moving round is registered |
| Execution environment | Sequential, local process-based, or SLURM | Where workflow tasks run |
| Task granularity | Coarse or fine-grained | How workflow work is admitted to the executor |

## Scientific evidence

- Registration behavior and parameter choices: link the characterization pages.
- Registration-library validation: link Registration-library acceptance.
- Workflow execution validation: link Workflow-runtime acceptance.
````

Use capability language only. Do not mention internal phases in this page.

- [ ] **Step 4: Expand Getting started**

Rewrite `docs/getting-started.md` with these exact sections:

```markdown
# Getting started

## Requirements
## Obtain the source
## Create the base environment
## Choose an optional capability
## Run project commands
## Prepare your first inputs
## Register two images
## Where to go next
```

Include the complete source checkout, `uv sync --locked --no-dev`, cumulative extras,
and `uv run` behavior. State that the reader supplies scalar 3-D TIFFs and physical
voxel spacing, using the exact phrase “caller-provided inputs.” Use the approved
`register_round()` example from the design. Link
strategy and the workflow tutorial directly. Until Task 2 creates the focused
geometry and recovery pages, link both detailed concepts to the existing
`registration-artifacts.md` page so the intermediate book has no broken links. Do not
duplicate the thread benchmark or schemas.

- [ ] **Step 5: Rewrite Registration foundations as the strategy guide**

Use these exact sections:

```markdown
# Choosing a registration strategy

## The shared registration pipeline
## Whole-image registration
## Serial-chunked registration
## Comparison
## Registration strategy versus execution
## Scientific choices shared by both strategies
## Next steps
```

The comparison table must cover nonlinear domain, memory pressure, restart unit,
chunk-size/overlap requirements, and appropriate starting use. State that the affine
stage remains whole-image in both strategies and that serial chunk execution is the
numerical/restart reference. Introduce sequential/local/SLURM and coarse/fine only as
separate operational axes with links to later pages.

- [ ] **Step 6: Update the initial navigation**

Change the first portion of `docs/myst.yml` to:

```yaml
    - file: index.md
    - title: Start here
      children:
        - file: getting-started.md
        - file: registration-foundations.md
        - file: workflow-guide.md
```

Leave the existing scientific/technical and evidence entries in place until the
later pages exist; do not reference nonexistent pages.

- [ ] **Step 7: Run focused tests and build the book**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest \
  tests/unit/test_phase2_documentation.py::test_public_documentation_home_routes_readers_by_task \
  tests/unit/test_phase2_documentation.py::test_registration_foundations_separates_science_from_execution \
  tests/unit/test_phase2_documentation.py::test_getting_started_gives_a_complete_source_checkout \
  tests/unit/test_phase2_documentation.py::test_getting_started_documents_installation_choices \
  -v
cd docs && uv run --group docs jupyter book build --html
```

Expected: four tests pass; Jupyter Book exits 0 without missing-page or broken-link
warnings.

- [ ] **Step 8: Review and commit**

Inspect `git diff --check` and the rendered Start here navigation. Then commit:

```bash
git add docs/index.md docs/getting-started.md docs/registration-foundations.md \
  docs/myst.yml tests/unit/test_phase2_documentation.py
git commit -m "docs: reorganize first-use registration guidance"
```

### Task 2: Add focused scientific user guides

**Files:**

- Create: `docs/image-geometry.md`
- Create: `docs/preprocessing.md`
- Create: `docs/outputs-and-recovery.md`
- Modify: `docs/getting-started.md`
- Modify: `docs/feature-derived-registration-masks.md`
- Modify: `docs/registration-artifacts.md`
- Modify: `docs/myst.yml`
- Modify: `tests/unit/test_phase2_documentation.py`

**Interfaces:**

- Consumes: the existing Geometry/TIFF, preprocessing, feature-mask, output-tree,
  status, force, and completion-record text in `README.md`; exact current behavior in
  the retained artifact and mask pages.
- Produces: three canonical user guides and stable cross-links to the retained
  advanced references.

- [ ] **Step 1: Add failing user-guide contract tests**

Add:

```python
@pytest.mark.parametrize(
    ("name", "required"),
    (
        (
            "image-geometry.md",
            (
                "micrometers",
                "(x, y, z)",
                "(z, y, x)",
                "fixed-reference grid",
                "scalar 3-d tiff",
                "origin",
                "spacing",
                "direction",
            ),
        ),
        (
            "preprocessing.md",
            (
                "raw -> crop -> feature estimate -> transform raw",
                "filter_sigma_zyx",
                "voxel",
                "feature-derived registration mask",
                "raw-intensity",
                "dataset-specific",
            ),
        ),
        (
            "outputs-and-recovery.md",
            (
                "resume",
                "force",
                "pending",
                "partial",
                "complete",
                "invalid",
                "diagnostic image",
                "transformed channel",
                ".complete.json",
            ),
        ),
    ),
)
def test_scientific_user_guides_preserve_public_contracts(
    name: str,
    required: tuple[str, ...],
) -> None:
    text = _normalized_document(name)
    for fragment in required:
        assert fragment in text
    assert "phase 1" not in text
    assert "phase 2" not in text
```

Rename `test_readme_documents_output_lifecycle_and_status_semantics` to
`test_output_and_artifact_guides_document_lifecycle_and_status`. Read the concatenated
text of `outputs-and-recovery.md` and `registration-artifacts.md`, preserving every
existing filename and lifecycle assertion. Its setup becomes:

```python
def test_output_and_artifact_guides_document_lifecycle_and_status() -> None:
    text = _document("outputs-and-recovery.md") + _document(
        "registration-artifacts.md"
    )
    for output in (
        "scientific-plan.json",
        "workflow-runs/",
        "<run-id>.json",
        "rounds/round-002/registration/",
        "round-001/",
        "reference_crop_indices.json",
        "linear_transform_2.mat",
        "nonlinear_warp_2.nii.gz",
        "nonlinear_registered_2.tif",
        "rounds/round-002/channels/",
        "rounds/round-002/logs/round-002.log",
        "round-002-<stage>-<item>.log",
        ".complete.json",
    ):
        assert output in text

    normalized = " ".join(text.split()).lower()
    for required in (
        "resume is the default",
        "read-only status",
        "`pending`, `partial`, `complete`, or `invalid`",
        "scientific fingerprint",
        "force cannot bypass",
        "requires a new output root",
        "operational summaries never authorize artifact reuse",
        "forcing the reference round invalidates all moving descendants",
        "normalized registration diagnostic",
        "original channel resampled through the persisted transform chain",
        "clipped to uint16 without intensity normalization",
    ):
        assert required in normalized
```

In `test_maintained_docs_distinguish_shared_cores_and_rejected_evidence`, stop
reading `README.md`. Assert the shared-core and rejected-evidence contracts against
`registration-artifacts.md` and `fine-grained-execution.md`, their two canonical
owners.

- [ ] **Step 2: Run the test and verify RED**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest \
  tests/unit/test_phase2_documentation.py::test_scientific_user_guides_preserve_public_contracts \
  tests/unit/test_phase2_documentation.py::test_output_and_artifact_guides_document_lifecycle_and_status \
  -v
```

Expected: collection succeeds; all three parameter cases and the lifecycle test fail
with `FileNotFoundError`.

- [ ] **Step 3: Create the image-geometry guide**

Create `docs/image-geometry.md` with:

```markdown
# Image geometry, voxel spacing, and axes

## Fixed and moving images
## Physical XYZ and array ZYX
## The fixed-reference grid
## Accepted TIFF inputs
## Cy-ExM geometry metadata
## Common mistakes
## Related reference material
```

Move the practical whole-image axis and TIFF guidance from the README. Preserve the
ANTs XYZ boundary, external ZYX shapes, vector-component XYZ order, allowed plain
TIFF layouts, caller-supplied XYZ spacing, zero origin/identity direction fallback,
and fail-closed handling of malformed Cy-ExM metadata. Put exact metadata payload and
artifact-record detail in links to `registration-artifacts.md`.

- [ ] **Step 4: Create the preprocessing guide**

Create `docs/preprocessing.md` with:

```markdown
# Registration preprocessing

## When preprocessing is disabled
## Enable background-subtracted asinh features
## Units and anisotropic data
## What preprocessing changes
## What preprocessing does not change
## Optional feature-derived masks
## Cache and restart effects
## Evaluate a dataset-specific policy
```

Move the README examples for omitted, empty, scalar, isotropic, and anisotropic
`registration.preprocessing`. Preserve reflective Gaussian background estimation,
deterministic sampling, MAD scale, positive residual clipping, float32 asinh features,
raw diagnostic masks by default, raw-intensity outputs, feature-mask semantics, and
format versions 2/3/4. Link the mask algorithm and investigations rather than
repeating them.

- [ ] **Step 5: Create the outputs and recovery guide**

Create `docs/outputs-and-recovery.md` with:

```markdown
# Outputs, restart, and recovery

## Output organization
## Completion records and safe reuse
## Ordinary resume
## Force selected work
## Read workflow status
## Diagnostic images and transformed channels
## Recover from common failures
## Exact artifact reference
```

Explain the dataset/round layout at a conceptual level, the four status states,
reference-barrier consequences, force propagation, new-output-root requirement for
scientifically incompatible plans, affine/nonlinear/chunk recovery, and diagnostic
versus original-intensity channel use. Keep the exhaustive tree and schema in
`registration-artifacts.md`.

- [ ] **Step 6: Establish cross-links and navigation**

Add links from `docs/preprocessing.md` to
`feature-derived-registration-masks.md`, and add a reciprocal “Preprocessing
overview” link near the start of the mask page. Add a “User-facing recovery guide”
link near the start of `registration-artifacts.md`. Replace the artifact page's
historical source comparison with a direct statement of current Cy-ExM/ANTs geometry
behavior; preserve the exact geometry invariant. Replace the temporary geometry and
recovery links in `getting-started.md` with `image-geometry.md` and
`outputs-and-recovery.md`.

Replace the current `Scientific and technical foundations` navigation group with:

```yaml
    - title: User guide
      children:
        - file: image-geometry.md
        - file: preprocessing.md
        - file: feature-derived-registration-masks.md
        - file: outputs-and-recovery.md
```

Temporarily keep `registration-artifacts.md` and `fine-grained-execution.md` in a
small `Advanced usage` group until Task 4 completes that group.

- [ ] **Step 7: Run focused tests and build**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest \
  tests/unit/test_phase2_documentation.py::test_scientific_user_guides_preserve_public_contracts \
  tests/unit/test_phase2_documentation.py::test_output_and_artifact_guides_document_lifecycle_and_status \
  tests/unit/test_phase2_documentation.py::test_maintained_docs_distinguish_shared_cores_and_rejected_evidence \
  -v
cd docs && uv run --group docs jupyter book build --html
```

Expected: focused tests and book build pass.

- [ ] **Step 8: Review and commit**

Inspect the rendered geometry and preprocessing unit tables and the recovery links.
Then:

```bash
git add docs/image-geometry.md docs/preprocessing.md docs/outputs-and-recovery.md \
  docs/getting-started.md \
  docs/feature-derived-registration-masks.md docs/registration-artifacts.md \
  docs/myst.yml tests/unit/test_phase2_documentation.py
git commit -m "docs: add focused scientific user guides"
```

### Task 3: Separate the YAML tutorial, manifest schema, and CLI contract

**Files:**

- Create: `docs/dataset-manifest.md`
- Create: `docs/cli-reference.md`
- Modify: `docs/workflow-guide.md`
- Modify: `docs/myst.yml`
- Modify: `tests/unit/test_phase2_documentation.py`

**Interfaces:**

- Consumes: checked-in manifest/profile paths, the current README manifest table,
  workflow command blocks, force/JSON/log/version behavior, exit codes, and the first
  two tasks' canonical guides.
- Produces: one tutorial plus separate exact scientific-schema and CLI references.

- [ ] **Step 1: Redirect workflow, manifest, and CLI tests to canonical pages**

Rename and retarget the existing README tests as follows while preserving their
existing exact assertion tuples:

| Existing test | New test | Read from |
| --- | --- | --- |
| `test_readme_documents_every_workflow_command_option_and_example` | `test_workflow_and_cli_docs_cover_every_command_option_and_example` | concatenated `workflow-guide.md` + `cli-reference.md` |
| `test_readme_workflow_command_blocks_use_the_uv_managed_environment` | `test_workflow_guide_uses_the_uv_managed_environment` | `workflow-guide.md` |
| `test_readme_documents_every_scientific_manifest_field_and_rule` | `test_dataset_manifest_reference_documents_every_field_and_rule` | `dataset-manifest.md` |
| `test_readme_documents_success_only_json_output` | `test_cli_reference_documents_success_only_json_output` | `cli-reference.md` |

For example, the manifest test begins:

```python
def test_dataset_manifest_reference_documents_every_field_and_rule() -> None:
    reference = _document("dataset-manifest.md")
    for field in (
        "schema_version",
        "dataset_id",
        "voxel_spacing_um",
        "reference",
        "registration",
        "rounds",
        "strategy",
        "registration_channel",
        "channels",
        "interpolation",
        "chunk_size_zyx",
        "overlap_fraction",
    ):
        assert f"`{field}`" in reference
```

Retain the existing accepted-value and strict-YAML assertion lists below that loop.
Do not weaken field, unit, path, duplicate-key, alias, custom-tag, or additional
document coverage.

- [ ] **Step 2: Run the retargeted tests and verify RED**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest \
  tests/unit/test_phase2_documentation.py::test_workflow_and_cli_docs_cover_every_command_option_and_example \
  tests/unit/test_phase2_documentation.py::test_workflow_guide_uses_the_uv_managed_environment \
  tests/unit/test_phase2_documentation.py::test_dataset_manifest_reference_documents_every_field_and_rule \
  tests/unit/test_phase2_documentation.py::test_cli_reference_documents_success_only_json_output \
  -v
```

Expected: failures because the new reference files do not exist and the current
workflow guide lacks the complete command tutorial.

- [ ] **Step 3: Rewrite the workflow guide as a tutorial**

Use:

```markdown
# Running a YAML workflow

## When to use the workflow
## Two files with different responsibilities
## Start from the checked-in examples
## Validate inputs
## Inspect the plan
## Run or resume
## Inspect status
## Recompute selected work
## Choose an execution environment
## Exact references
```

Show the exact `uv run cyexm workflow validate`, `plan`, `run`, and `status` blocks.
Explain mutation-free planning/status, normal resume, `--force-round`, `--force-all`,
and the fact that input paths must be edited and populated. Keep exact schemas and
complete option/exit tables out of the tutorial. Link the manifest and CLI references
created in this task; keep the checked-in execution-profile links until Task 4 creates
the canonical execution-profile reference.

- [ ] **Step 4: Create the dataset-manifest reference**

Move the complete `### Scientific dataset manifest` table and strict validation prose
from the README into `docs/dataset-manifest.md`. Organize it as:

```markdown
# Dataset manifest reference

## Purpose and scientific identity
## Complete schema
## Paths and channel rules
## Reference and moving geometry
## Registration strategies
## Preprocessing fields
## Nonlinear metrics and CC radius
## Strict YAML validation
## Complete example
```

Keep every accepted value, unit, conditional field, path restriction, reference
channel rule, geometry rule, retired singular-form rejection, and provisional CC
radius caveat exact. Replace internal phase labels in moved table cells with
`registration library` or another capability-based description. Link preprocessing
and geometry pages for interpretation.

- [ ] **Step 5: Create the CLI reference**

Create `docs/cli-reference.md` with:

```markdown
# Command-line reference

## Global options
## `workflow validate`
## `workflow plan`
## `workflow run`
## `workflow status`
## Force and resume
## Structured JSON and logging
## Failures and exit codes
```

Move every CLI option and exact block from the README. Preserve success-only JSON,
empty stdout on failures, stderr diagnostics, version behavior, and exit codes
0/2/3/4/130 exactly. Link operational failure interpretation to Parallel/HPC and
artifact recovery to Outputs/recovery.

- [ ] **Step 6: Add the new references to temporary Advanced navigation**

Ensure the Advanced group now includes, in this order:

```yaml
        - file: dataset-manifest.md
        - file: cli-reference.md
        - file: fine-grained-execution.md
        - file: registration-artifacts.md
```

- [ ] **Step 7: Run focused tests and build**

Run the four retargeted tests from Step 2, then:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
cd docs && uv run --group docs jupyter book build --html
```

Expected: all focused tests and the build pass; no page duplicates the complete
manifest or CLI tables.

- [ ] **Step 8: Review and commit**

```bash
git add docs/workflow-guide.md docs/dataset-manifest.md docs/cli-reference.md \
  docs/myst.yml tests/unit/test_phase2_documentation.py
git commit -m "docs: separate workflow schemas and CLI reference"
```

### Task 4: Add execution-profile, native-thread, and HPC guidance

**Files:**

- Create: `docs/execution-profile.md`
- Create: `docs/native-threading.md`
- Create: `docs/parallel-hpc.md`
- Modify: `docs/workflow-guide.md`
- Modify: `docs/fine-grained-execution.md`
- Modify: `docs/myst.yml`
- Modify: `tests/unit/test_phase2_documentation.py`

**Interfaces:**

- Consumes: current README execution-profile table, native-thread sections,
  executor/failure/HPC sections, and exact acceptance caveats in retained evidence.
- Produces: canonical operational schema, thread/performance, and executor guides;
  complete Advanced usage navigation.

- [ ] **Step 1: Redirect the execution and thread contracts**

Retarget these existing tests without weakening their assertion tuples:

| Existing test | New test | Canonical text |
| --- | --- | --- |
| `test_readme_documents_every_execution_profile_field_and_rule` | `test_execution_profile_reference_documents_every_field_and_rule` | `execution-profile.md` |
| `test_readme_documents_the_exact_startup_timeout_scope` | `test_execution_profile_documents_the_exact_startup_timeout_scope` | `execution-profile.md` |
| `test_readme_documents_planning_and_execution_thread_boundaries` | `test_native_threading_documents_planning_and_worker_boundaries` | `native-threading.md` |
| `test_readme_documents_observed_native_thread_scaling` | `test_native_threading_documents_observed_scaling` | `native-threading.md` |
| `test_readme_documents_executor_retry_cancellation_and_exit_contracts` | `test_parallel_hpc_guide_documents_executor_lifecycle` | `parallel-hpc.md` + `cli-reference.md` |
| operational assertions in `test_readme_states_hpc_acceptance_scope_and_remaining_non_goals` | `test_parallel_hpc_guide_states_acceptance_boundaries` | `parallel-hpc.md` |

For the benchmark, preserve this complete assertion:

```python
def test_native_threading_documents_observed_scaling() -> None:
    text = _normalized_document("native-threading.md")
    for required in (
        "| 1 / 1 | 310.5 s | 1.00x |",
        "| 8 / 8 | 88.3 s | 3.52x |",
        "| 30 / 30 | 69.5 s | 4.47x |",
        "scientifically consistent but not bit-for-bit identical",
        "largest nonlinear-correlation change was 0.003416",
        "one thread remains the conservative default",
        "full-volume stability evidence",
    ):
        assert required in text


def test_parallel_hpc_guide_states_acceptance_boundaries() -> None:
    text = _normalized_document("parallel-hpc.md")
    for required in (
        "portable fake-based slurm construction",
        "real slurm submission and hpc acceptance completed",
        "`-m hpc`",
        "explicit allocation",
        "portable real-ants/local-dask evidence is complete",
        "fine-grained real-slurm production acceptance remains pending",
        "focused multi-process real-slurm acceptance remains pending",
        "no silent fallback",
    ):
        assert required in text
```

- [ ] **Step 2: Run the retargeted tests and verify RED**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest \
  tests/unit/test_phase2_documentation.py::test_execution_profile_reference_documents_every_field_and_rule \
  tests/unit/test_phase2_documentation.py::test_execution_profile_documents_the_exact_startup_timeout_scope \
  tests/unit/test_phase2_documentation.py::test_native_threading_documents_planning_and_worker_boundaries \
  tests/unit/test_phase2_documentation.py::test_native_threading_documents_observed_scaling \
  tests/unit/test_phase2_documentation.py::test_parallel_hpc_guide_documents_executor_lifecycle \
  tests/unit/test_phase2_documentation.py::test_parallel_hpc_guide_states_acceptance_boundaries \
  -v
```

Expected: failures because the three canonical pages do not yet exist.

- [ ] **Step 3: Create the execution-profile reference**

Move the exact profile table and validation prose from the README into:

```markdown
# Execution profile reference

## Purpose
## Complete schema
## Paths and filesystem safety
## Sequential profiles
## Local process-based profiles
## SLURM profiles
## Coarse and fine-grained modes
## CPU, process, and memory rules
## Startup-timeout boundary
## Strict YAML validation
## Starting profiles
```

Preserve exact executor values, every field, path-resolution/disjoint-root rules,
scratch restrictions, boolean/count validation, exact `500GB -> 466G -> 466GiB`
normalization, core divisibility, job/process slot distinction, managed-directive
conflicts, startup-timeout exclusions, and production-owned runner boundary.

- [ ] **Step 4: Create the native-threading guide**

Use:

```markdown
# Native threading and performance

## Why Cy-ExM defaults to one thread
## Configure threads in Python
## Configure threads in an execution profile
## Resolution precedence
## Environment variables
## Import-time boundary
## Thread controls that are not interchangeable
## Observed benchmark
## Reproducibility and tuning guidance
```

Move the Python and YAML resolution rules, exact ITK/BLAS environment map, validation
semantics, sequential versus worker application boundaries, and the complete 1/8/30
benchmark. Link the multithreaded reproducibility evidence. Do not imply universal
speedup or routine 30-thread safety.

- [ ] **Step 5: Create the parallel/HPC guide**

Use:

```markdown
# Local parallel and HPC execution

## Choose an execution environment
## Sequential execution
## Local process-based execution
## SLURM execution
## Coarse and fine-grained work
## Processes, task slots, and jobs
## Prepare uv on a quota-limited host
## Configure site-specific resources
## Startup and preflight
## Retry, cancellation, and cleanup
## Acceptance boundaries
```

Begin with the approved three-row intent table. Introduce Dask only after the user
understands the operational choices. Preserve reference barrier, no silent fallback,
one slot per process, coarse one-process jobs, fine whole-job process groups,
preflight, ordered cleanup, retry/cancellation, coordinator constraints, portable
versus real-cluster evidence, and pending fine/multi-process real-SLURM boundaries.

The quota section must include exactly:

```bash
export UV_CACHE_DIR=/active/project/path/uv/cache
export UV_PYTHON_INSTALL_DIR=/active/project/path/uv/python

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

uv cache dir
uv python dir
uv sync --locked --extra hpc
```

Explain placeholder substitution, active project storage, avoiding quota-limited home
and archival storage, and persistence in interactive/job setup. Do not include lab
filesystem paths.

- [ ] **Step 6: Improve the fine-grained introduction and finish Advanced navigation**

Keep all exact content in `docs/fine-grained-execution.md`, but rewrite its opening
to define fine-grained task admission in plain language and link back to the strategy
and parallel/HPC pages. Do not rename `serial_chunked` or imply a third registration
strategy. Replace historical source-repository comparisons and phase-labeled audit
links in ordinary prose with direct descriptions of current fixed-grid behavior and
capability-named evidence links; retain all scientific invariants.

Update the workflow tutorial's exact-reference section to link
`execution-profile.md` now that the page exists.

Set Advanced navigation to:

```yaml
    - title: Advanced usage
      children:
        - file: dataset-manifest.md
        - file: execution-profile.md
        - file: cli-reference.md
        - file: native-threading.md
        - file: parallel-hpc.md
        - file: fine-grained-execution.md
        - file: registration-artifacts.md
```

- [ ] **Step 7: Run focused and existing fine-grained tests**

Run the six renamed tests plus the two retained fine-grained contracts:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest \
  tests/unit/test_phase2_documentation.py::test_execution_profile_reference_documents_every_field_and_rule \
  tests/unit/test_phase2_documentation.py::test_execution_profile_documents_the_exact_startup_timeout_scope \
  tests/unit/test_phase2_documentation.py::test_native_threading_documents_planning_and_worker_boundaries \
  tests/unit/test_phase2_documentation.py::test_native_threading_documents_observed_scaling \
  tests/unit/test_phase2_documentation.py::test_parallel_hpc_guide_documents_executor_lifecycle \
  tests/unit/test_phase2_documentation.py::test_parallel_hpc_guide_states_acceptance_boundaries \
  tests/unit/test_phase2_documentation.py::test_fine_grained_guide_documents_the_exact_graph_and_resource_contract \
  tests/unit/test_phase2_documentation.py::test_fine_grained_guide_documents_artifact_authority_and_recovery \
  -v
cd docs && uv run --group docs jupyter book build --html
```

Expected: all focused contracts and the book build pass.

- [ ] **Step 8: Review and commit**

```bash
git add docs/execution-profile.md docs/native-threading.md docs/parallel-hpc.md \
  docs/workflow-guide.md docs/fine-grained-execution.md docs/myst.yml \
  tests/unit/test_phase2_documentation.py
git commit -m "docs: add execution threading and HPC guides"
```

### Task 5: Add developer guidance, architecture, evidence navigation, and site metadata

**Files:**

- Create: `docs/development.md`
- Create: `docs/architecture.md`
- Modify: `docs/index.md`
- Modify: `docs/myst.yml`
- Modify: `docs/phase-1-acceptance.md`
- Modify: `docs/phase-2-acceptance.md`
- Modify: `pyproject.toml`
- Modify: `tests/unit/test_phase2_documentation.py`
- Modify: `tests/unit/test_repository_policy.py`

**Interfaces:**

- Consumes: repository commands and dependency policy from `AGENTS.md`, current
  architecture/acceptance boundaries, hosted Pages URL, and retained API policy.
- Produces: contributor entry point, capability architecture, capability-named
  evidence navigation, and package metadata linking the public book.

- [ ] **Step 1: Add failing developer/navigation tests**

Add:

```python
def test_developer_guide_preserves_maintenance_contract() -> None:
    guide = _document("development.md")
    for required in (
        "uv sync --locked --group dev --extra examples --extra hpc",
        'uv run pytest -m "unit and not hpc" tests/unit -v',
        'uv run pytest -m "integration and not hpc" tests/integration -v',
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run basedpyright",
        "uv run pre-commit run --all-files",
        "jupyter book build --html",
        "Dask/distributed 2025.1.0",
        "dask-jobqueue 0.9.0",
        "fresh compatibility evidence",
        "agentic-api-documentation.md",
    ):
        assert required in guide


def test_public_navigation_contains_every_approved_page() -> None:
    navigation = _document("myst.yml")
    for page in (
        "getting-started.md",
        "registration-foundations.md",
        "workflow-guide.md",
        "image-geometry.md",
        "preprocessing.md",
        "feature-derived-registration-masks.md",
        "outputs-and-recovery.md",
        "dataset-manifest.md",
        "execution-profile.md",
        "cli-reference.md",
        "native-threading.md",
        "parallel-hpc.md",
        "fine-grained-execution.md",
        "registration-artifacts.md",
        "development.md",
        "architecture.md",
        "phase2e-multithreaded-registration-reproducibility.md",
        "phase-1-acceptance.md",
        "phase-2-acceptance.md",
    ):
        assert f"file: {page}" in navigation


def test_architecture_records_supported_non_goals() -> None:
    guide = _normalized_document("architecture.md")
    for required in (
        "parallelchunkedimageregistration",
        "job arrays",
        "cloud or kubernetes executors",
        "automatic resource optimization",
        "no silent fallback",
    ):
        assert required in guide


def test_architecture_and_workflow_preserve_scheduler_neutral_contract() -> None:
    text = _normalized_document("architecture.md") + _normalized_document(
        "workflow-guide.md"
    )
    for required in (
        "scientific dataset manifest",
        "execution profile",
        "one immutable `referencetask`",
        "one immutable `roundtask` per moving round",
        "hard executor barrier",
        "serial inside each moving-round task",
        "completion records are the only restart authority",
        "no nested chunk multiprocessing",
        "planning is mutation-free",
        "`parallelism.mode`",
        "`coarse`",
        "`fine_grained`",
        "scientific fingerprint is unchanged",
    ):
        assert required in text


def test_package_metadata_advertises_the_documentation_site() -> None:
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        'Documentation = "https://thedeanlab.github.io/Cy-ExM/"' in pyproject
    )
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest \
  tests/unit/test_phase2_documentation.py::test_developer_guide_preserves_maintenance_contract \
  tests/unit/test_phase2_documentation.py::test_public_navigation_contains_every_approved_page \
  tests/unit/test_phase2_documentation.py::test_architecture_records_supported_non_goals \
  tests/unit/test_phase2_documentation.py::test_architecture_and_workflow_preserve_scheduler_neutral_contract \
  tests/unit/test_phase2_documentation.py::test_package_metadata_advertises_the_documentation_site \
  -v
```

Expected: developer and architecture files missing, approved navigation incomplete,
and Documentation URL absent.

- [ ] **Step 3: Create the development guide**

Use:

```markdown
# Development guide

## Create the locked development environment
## Run focused tests first
## Run the portable gates
## Build and inspect the documentation
## Maintain optional dependency compatibility
## Change a public API or schema
## Keep historical evidence intact
```

Include all exact commands from the test, the exact locked-version statement, the
rule that intentional dependency changes use fresh evidence and inspect `uv.lock`,
and a link to
`agentic-api-documentation.md`. Explain that HPC-marked tests require explicit
operator-approved resources and are not portable gates. Link the public Parallel/HPC
guide for site-specific uv cache and managed-Python setup; do not include lab paths.

- [ ] **Step 4: Create the architecture/history guide**

Use:

```markdown
# Architecture and project history

## Current capabilities
## Registration library
## Workflow configuration and planning
## Execution boundaries
## Persistent scientific state
## Supported boundaries and non-goals
## Historical phase labels
```

Describe current capabilities without phase names until the final historical section.
Preserve scheduler-neutral planning, reference barrier, coarse/fine separation,
artifact-store ownership, no nested scheduler/multiprocessing, no silent fallback,
and current non-goals. In the last section only, map historical Phase 1 to the
registration-library baseline and historical Phase 2 to workflow runtime work so old
records remain interpretable.

- [ ] **Step 5: Complete navigation and evidence labels**

Update `docs/index.md` so “Configure advanced or HPC execution” links to
`parallel-hpc.md` and “Contribute to Cy-ExM” links to `development.md`, replacing the
temporary existing-page targets from Task 1.

Add:

```yaml
    - title: Developer guide
      children:
        - file: development.md
        - file: architecture.md
        - file: agentic-api-documentation.md
    - title: Characterization and evidence
      children:
        - file: brightness-dependence-investigation.md
        - file: asinh-preprocessing-registration-investigation.md
        - file: cc-syn-radius-characterization.md
        - file: fixed-affine-chunk-size-characterization.md
        - file: phase2e-multithreaded-registration-reproducibility.md
        - file: phase-1-acceptance.md
        - file: phase-2-acceptance.md
```

Retitle the acceptance-page H1 headings to `Registration-library acceptance` and
`Workflow-runtime acceptance`. Add one sentence immediately below each heading that
states its historical phase label. Preserve every audit identity, hash, environment,
result, and limitation below that sentence.

- [ ] **Step 6: Move repository-policy navigation assertions**

In `test_acceptance_audit_records_the_implementation_baseline_and_scope` and
`test_phase_two_acceptance_audit_records_external_two_tier_evidence`, replace the
README link assertions with:

```python
navigation = _read_repository_file("docs/myst.yml")
assert "file: phase-1-acceptance.md" in navigation
assert "file: phase-2-acceptance.md" in navigation
```

In the registration-library audit test, read `docs/development.md` and assert the
portable unit/integration commands there instead of in `README.md`; retain the same
assertions against root `AGENTS.md`. In the workflow-runtime audit test, apply its
three stale-acceptance-phrase exclusions to the acceptance record rather than the
README. Keep every exact audit-evidence assertion unchanged.

- [ ] **Step 7: Add the package Documentation URL**

Change only `[project.urls]`:

```toml
[project.urls]
Documentation = "https://thedeanlab.github.io/Cy-ExM/"
Source = "https://github.com/TheDeanLab/Cy-ExM"
```

Confirm `git diff -- pyproject.toml uv.lock` shows the URL addition and no lockfile
change.

- [ ] **Step 8: Run focused tests and build**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest \
  tests/unit/test_phase2_documentation.py::test_developer_guide_preserves_maintenance_contract \
  tests/unit/test_phase2_documentation.py::test_public_navigation_contains_every_approved_page \
  tests/unit/test_phase2_documentation.py::test_architecture_records_supported_non_goals \
  tests/unit/test_phase2_documentation.py::test_architecture_and_workflow_preserve_scheduler_neutral_contract \
  tests/unit/test_phase2_documentation.py::test_package_metadata_advertises_the_documentation_site \
  tests/unit/test_repository_policy.py::test_acceptance_audit_records_the_implementation_baseline_and_scope \
  tests/unit/test_repository_policy.py::test_phase_two_acceptance_audit_records_external_two_tier_evidence \
  -v
cd docs && uv run --group docs jupyter book build --html
```

Expected: all tests and the build pass with no missing-page warnings.

- [ ] **Step 9: Review and commit**

```bash
git add docs/development.md docs/architecture.md docs/index.md docs/myst.yml \
  docs/phase-1-acceptance.md docs/phase-2-acceptance.md pyproject.toml \
  tests/unit/test_phase2_documentation.py tests/unit/test_repository_policy.py
git commit -m "docs: add developer guide and capability navigation"
```

### Task 6: Replace the README and complete integrated validation

**Files:**

- Modify: `README.md`
- Modify: `tests/unit/test_phase2_documentation.py`
- Modify: `tests/unit/test_examples.py`
- Modify: `tests/unit/test_repository_policy.py`
- Modify as required by verified build/render defects only: public files from Tasks
  1-5

**Interfaces:**

- Consumes: every canonical destination created in Tasks 1-5 and the approved README
  contract.
- Produces: final 100-150-line README, page-owned documentation tests, passing
  portable gates, and visually verified public documentation.

- [ ] **Step 1: Add the final README scope test and finish test relocation**

Replace the remaining README-wide contract tests with:

```python
def test_readme_is_a_concise_first_use_guide() -> None:
    readme = _readme()
    headings = [line for line in readme.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Installation",
        "## Register two images",
        "## Register multiple imaging rounds",
        "## Documentation",
        "## License",
    ]
    assert 100 <= len(readme.splitlines()) <= 150
    for required in (
        "git clone https://github.com/TheDeanLab/Cy-ExM.git",
        "uv sync --locked --no-dev",
        "from cyexm.registration import register_round",
        "fixed_image_path=Path(\"data/reference.tif\")",
        "moving_image_path=Path(\"data/moving.tif\")",
        "https://thedeanlab.github.io/Cy-ExM/",
        "examples/spheroid-dataset.yaml",
        "examples/execution-sequential.yaml",
        "LICENSE.md",
    ):
        assert required in readme


def test_readme_omits_advanced_and_historical_material() -> None:
    readme = _normalized_readme()
    for forbidden in (
        "phase 1",
        "phase 2",
        "observed threading benchmark",
        "dask-jobqueue",
        "slurmcluster",
        "development checks",
        "acceptance audit",
        "`worker_local_directory`",
        "`job_extra_directives`",
        "`startup_timeout`",
    ):
        assert forbidden not in readme
```

Delete only README-specific tests whose complete contract has already been moved to a
canonical-page test. Keep all fine-grained, artifact, characterization, and audit
tests.

Rename `test_user_documentation_matches_phase_one_contract` in
`tests/unit/test_examples.py` to
`test_user_documentation_preserves_registration_contract`. Read the combined text of
`README.md`, `docs/registration-foundations.md`, `docs/native-threading.md`,
`docs/registration-artifacts.md`, and `docs/architecture.md`; preserve assertions for
`NativeThreadConfig`, XYZ order, `ChunkedImageRegistration`, serial execution,
restart, Dask, and SLURM in that combined text. Require only source installation,
`register_round`, documentation links, and license in the README itself. Remove
`"Phase 2"` and the historical migration-page link from the public README assertion
tuple, but keep the separate exact assertions that validate the excluded migration
record itself.

- [ ] **Step 2: Run the README tests and verify RED**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest \
  tests/unit/test_phase2_documentation.py::test_readme_is_a_concise_first_use_guide \
  tests/unit/test_phase2_documentation.py::test_readme_omits_advanced_and_historical_material \
  -v
```

Expected: the current 916-line README fails the exact heading, length, and forbidden
content assertions.

- [ ] **Step 3: Replace the README**

Write exactly these sections:

````markdown
# Cy-ExM

Cy-ExM registers volumetric images acquired across cyclic expansion microscopy
rounds. It supports whole-image and memory-conscious serial-chunked registration,
applies the resulting transforms to additional imaging channels, and can resume
validated intermediate work after interruption.

Cy-ExM requires Python 3.12 and retains the academic-research-use terms in
[LICENSE.md](LICENSE.md). See the
[complete documentation](https://thedeanlab.github.io/Cy-ExM/) for scientific
guidance, workflow configuration, and advanced execution.

New users should begin with the installation and two-image example below.

## Installation

Install Python 3.12 and `uv`, then clone the repository and create its locked base
environment:

```bash
git clone https://github.com/TheDeanLab/Cy-ExM.git
cd Cy-ExM
uv sync --locked --no-dev
```

Confirm that the package is available in the managed environment:

```bash
uv run python -c "import cyexm; print(cyexm.__version__)"
```

The [installation guide](docs/getting-started.md) explains optional capabilities for
notebooks, YAML workflows, local parallel execution, HPC execution, and development.

## Register two images

Supply a fixed reference TIFF, a moving TIFF, and the physical voxel spacing measured
for your data. Save this example as `register_images.py`:

The quick start assumes that:

- both inputs are scalar 3-D TIFF volumes;
- `reference.tif` defines the fixed coordinate system;
- `moving.tif` is the image transformed onto that system; and
- `pixel_size` contains the measured voxel spacing for the input data.

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

Run it in the managed environment:

```bash
uv run python register_images.py
```

Cy-ExM writes restart-safe registration artifacts below `results/round-2`. Running
the same request again reuses artifacts whose completion evidence is still valid.

Before adapting the example, read the guides to
[registration strategies](docs/registration-foundations.md),
[image geometry](docs/image-geometry.md), and
[outputs and recovery](docs/outputs-and-recovery.md).

## Register multiple imaging rounds

For several imaging rounds and channels, Cy-ExM can read a scientific dataset
manifest together with a separate execution profile. Start with the
[example manifest](examples/spheroid-dataset.yaml) and
[sequential execution profile](examples/execution-sequential.yaml), then follow the
[YAML workflow guide](docs/workflow-guide.md).

The checked-in YAML files are editable starting points. Supply the corresponding
images and replace their paths before validating or running them.

## Documentation

- [Getting started](docs/getting-started.md)
- [Choosing a registration strategy](docs/registration-foundations.md)
- [Registration preprocessing](docs/preprocessing.md)
- [Running a YAML workflow](docs/workflow-guide.md)
- [Local parallel and HPC execution](docs/parallel-hpc.md)
- [Development guide](docs/development.md)

## License

Cy-ExM is available for academic research under the terms in
[LICENSE.md](LICENSE.md).
````

Use the exact approved `register_round()` block from the specification. Do not add
badges, release/PyPI instructions, complete extra-install tables, phase terminology,
historical migration links, schemas, thread details, executor internals, or developer
commands.

- [ ] **Step 4: Run focused documentation tests**

Run the entire documentation-contract module plus the renamed examples documentation
test and both acceptance-policy tests:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest tests/unit/test_phase2_documentation.py -v
uv run pytest \
  tests/unit/test_examples.py::test_user_documentation_preserves_registration_contract \
  tests/unit/test_repository_policy.py::test_acceptance_audit_records_the_implementation_baseline_and_scope \
  tests/unit/test_repository_policy.py::test_phase_two_acceptance_audit_records_external_two_tier_evidence \
  -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Check scope and forbidden public content**

Run:

```bash
wc -l README.md
rg -n -i '\bphase (1|2)\b|clearex' \
  README.md docs/index.md docs/getting-started.md \
  docs/registration-foundations.md docs/workflow-guide.md \
  docs/image-geometry.md docs/preprocessing.md docs/outputs-and-recovery.md \
  docs/dataset-manifest.md docs/execution-profile.md docs/cli-reference.md \
  docs/native-threading.md docs/parallel-hpc.md \
  docs/fine-grained-execution.md docs/registration-artifacts.md
rg -n '/project/bioinformatics|/archive/bioinformatics' README.md \
  docs/index.md docs/getting-started.md docs/registration-foundations.md \
  docs/workflow-guide.md docs/image-geometry.md docs/preprocessing.md \
  docs/outputs-and-recovery.md docs/dataset-manifest.md \
  docs/execution-profile.md docs/cli-reference.md docs/native-threading.md \
  docs/parallel-hpc.md docs/fine-grained-execution.md \
  docs/registration-artifacts.md
git diff --check
git diff -- pyproject.toml uv.lock
```

Expected: README is 100-150 lines; both `rg` commands return no matches; diff check is
clean; `pyproject.toml` shows only Documentation metadata and `uv.lock` is unchanged.

- [ ] **Step 6: Run the complete portable verification gates**

On the approved host:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv lock --check
uv sync --locked --group dev --extra examples --extra hpc
uv run pytest -m "unit and not hpc" tests/unit -v
uv run pytest -m "integration and not hpc" tests/integration -v
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pre-commit run --all-files
cd docs && uv run --group docs jupyter book build --html
```

Expected: every command exits 0. Do not run HPC-marked tests or contact a scheduler.

- [ ] **Step 7: Inspect rendered output**

Serve `docs/_build/html` locally and inspect:

- documentation home at desktop and approximately 390-pixel viewport width;
- all five navigation groups and every child page;
- README rendering in GitHub-flavored Markdown;
- manifest/profile/CLI/thread tables for horizontal overflow;
- Python, YAML, and shell blocks for wrapping/copyability;
- internal links and evidence links;
- absence of orphaned pages, duplicate H1 headings, or phase terminology in ordinary
  user/advanced pages.

If visual or link defects are found, make the smallest documentation correction,
rerun its focused test and Jupyter build, and include the correction in this task.

- [ ] **Step 8: Review the complete diff against the specification**

Use the migration table in
`docs/superpowers/specs/2026-08-14-readme-documentation-reorganization-design.md` as a
line-by-line checklist. Confirm every current README section has a canonical
destination, no runtime/source file changed, `LICENSE.md` is byte-identical, and
historical excluded records remain present.

- [ ] **Step 9: Commit the integrated README and test migration**

```bash
git add README.md tests/unit/test_phase2_documentation.py \
  tests/unit/test_examples.py tests/unit/test_repository_policy.py
git add docs pyproject.toml
git commit -m "docs: streamline README and complete public guide"
```

Before committing, inspect the staged paths and unstage any unrelated or generated
file, especially `docs/_build/`. The commit body must state that this is a
documentation-only reorganization with no API, schema, dependency, lockfile, or
scientific-behavior change.

## Completion handoff

Report:

- README line count and final section list;
- new and rewritten public pages;
- documentation URL metadata;
- exact focused and portable gate results;
- Jupyter Book build and visual-QA result;
- confirmation that `uv.lock`, runtime source, and `LICENSE.md` did not change; and
- any gate not run because the approved `/project` uv environment was unavailable.
