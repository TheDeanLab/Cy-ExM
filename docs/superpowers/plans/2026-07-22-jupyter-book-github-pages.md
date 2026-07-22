# Jupyter Book GitHub Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish Cy-ExM narrative documentation at
`https://TheDeanLab.github.io/Cy-ExM/` from GitHub Actions whenever documentation
changes land on `main`.

**Architecture:** Add Jupyter Book 2 as a documentation-only dependency group and
configure a MyST Book Theme site rooted at `docs/`. Keep maintained Markdown pages
as sources, add focused narrative guide pages, and deploy the rendered static
artifact with GitHub Pages' supported Actions.

**Tech Stack:** Python 3.12; uv 0.7.13; Jupyter Book 2.1.6; MyST Markdown; GitHub
Actions; GitHub Pages.

## Global Constraints

- The distribution remains `cy-exm`, the import package remains `cyexm`, and the
  supported Python range remains `>=3.12,<3.13`.
- Keep Jupyter Book and documentation tooling out of base runtime dependencies;
  use a separate `docs` dependency group.
- Use MyST Markdown; do not introduce Jupyter Book 1, Sphinx Book Theme, or a
  generated API reference in this release.
- Do not publish raw datasets, generated registration artifacts, private paths,
  credentials, or large command transcripts.
- Documentation builds must not execute notebooks, download datasets, access lab
  storage, contact a scheduler, or submit SLURM work.
- Document only current behavior. Clearly identify portable SLURM construction
  evidence and label CC-radius characterization as ongoing until accepted.
- CC radius is one scalar voxel radius; radius 4 is provisional.
- The current untracked `slurm-*.out` files are user-owned and must not be staged,
  changed, or deleted.
- Before every local `uv` command, export
  `UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache` and
  `UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python`.
- Preserve the existing software CI workflow and its `contents: read` permission.
  The new deployment job alone receives `pages: write` and `id-token: write`.

---

## File structure

- Modify: `pyproject.toml` — define the documentation-only dependency group.
- Modify: `uv.lock` — lock the approved Jupyter Book dependency.
- Create: `docs/myst.yml` — project/site metadata, explicit navigation, exclusions,
  and Book Theme configuration.
- Create: `docs/index.md` — project documentation landing page.
- Create: `docs/getting-started.md` — installation and first registration path.
- Create: `docs/registration-foundations.md` — roles, geometry, transform, mask,
  metric, and native-thread invariants.
- Create: `docs/workflow-guide.md` — manifest/profile workflow tutorial and executor
  boundaries.
- Create: `docs/cc-syn-radius-characterization.md` — explicit ongoing-study page.
- Modify: `docs/brightness-dependence-investigation.md` — present the existing
  analysis as evidence with clear status and site-oriented navigation.
- Create: `.github/workflows/docs-pages.yml` — documentation build and Pages
  deployment workflow.
- Modify: `.gitignore` — ignore the Jupyter Book output directory.

## Task 1: Add the reproducible documentation toolchain

**Files:**
- Modify: `pyproject.toml:31-38`
- Modify: `uv.lock`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `uv.lock`, repository Python constraint, and the existing `dev` group.
- Produces: `uv run --group docs jupyter book build docs --html`, which later tasks
  use to render `docs/_build/html`.

- [ ] **Step 1: Add the isolated documentation dependency group**

  In `pyproject.toml`, add this group after `dev` without modifying base
  dependencies or existing extras:

  ```toml
  docs = ["jupyter-book==2.1.6"]
  ```

- [ ] **Step 2: Regenerate the lock file intentionally**

  Run:

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  uv lock
  uv lock --check
  ```

  Expected: the lock update contains Jupyter Book 2.1.6 and its resolved
  dependencies; `uv lock --check` exits 0.

- [ ] **Step 3: Ignore only generated Book output**

  Add this exact line to `.gitignore` in the documentation/build-output section:

  ```gitignore
  docs/_build/
  ```

- [ ] **Step 4: Verify the fresh docs environment and CLI**

  Run:

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  uv sync --locked --group docs
  uv run --group docs jupyter book --version
  ```

  Expected: synchronization succeeds without changing `uv.lock`, and the version
  begins with `2.1.6`.

- [ ] **Step 5: Commit the dependency boundary**

  ```bash
  git add pyproject.toml uv.lock .gitignore
  git commit -m "docs: add Jupyter Book toolchain"
  ```

## Task 2: Create the MyST site configuration and landing page

**Files:**
- Create: `docs/myst.yml`
- Create: `docs/index.md`

**Interfaces:**
- Consumes: the Task 1 `jupyter book` command and all listed Markdown files.
- Produces: explicit Book Theme navigation and a root page for `docs/_build/html`.

- [ ] **Step 1: Create the failing configuration/build check**

  Run before adding the configuration:

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  uv run --group docs jupyter book build docs --html
  ```

  Expected: FAIL because `docs/myst.yml` does not exist and there is no MyST site.

- [ ] **Step 2: Add `docs/myst.yml`**

  Create this complete configuration:

  ```yaml
  version: 1
  project:
    title: Cy-ExM
    description: Reusable cyclic expansion microscopy registration workflows.
    github: TheDeanLab/Cy-ExM
    toc:
      - file: index.md
      - title: Getting started
        children:
          - file: getting-started.md
      - title: Scientific and technical foundations
        children:
          - file: registration-foundations.md
          - file: registration-artifacts.md
          - file: workflow-guide.md
      - title: Characterization and evidence
        children:
          - file: brightness-dependence-investigation.md
          - file: cc-syn-radius-characterization.md
          - file: phase-1-acceptance.md
          - file: legacy-parity-report.md
    exclude:
      - AGENTS.md
      - migration-from-clearex.md
      - superpowers/**
  site:
    template: book-theme
    title: Cy-ExM documentation
    options:
      logo_text: Cy-ExM
  ```

- [ ] **Step 3: Add the landing page**

  Create `docs/index.md` with this content:

  ```markdown
  # Cy-ExM

  Cy-ExM provides reusable cyclic expansion microscopy registration workflows.
  It is an independent implementation with restart-safe whole-image and serial
  chunked registration, plus a deterministic Phase 2 runtime for planning and
  executing round-level work.

  The documentation focuses on using the software responsibly and understanding
  the scientific choices that affect its output. It distinguishes accepted behavior
  from active characterization work.

  ## Start here

  - Follow [Getting started](getting-started.md) to install Cy-ExM and run the
    smallest supported workflow.
  - Read [Registration foundations](registration-foundations.md) before changing
    transforms, metrics, masks, interpolation, or native-thread settings.
  - Use [Artifact recovery and reproducibility](registration-artifacts.md) to
    understand restart-safe output reuse.

  ## Scientific characterization

  The [brightness-dependence investigation](brightness-dependence-investigation.md)
  documents why dim regions can be harder to register and the evidence supporting
  local cross-correlation for nonlinear SyN. The
  [CC SyN radius characterization](cc-syn-radius-characterization.md) is explicitly
  ongoing and is not a statement of an optimized default.
  ```

- [ ] **Step 4: Run the focused build and inspect its root artifact**

  Run:

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  BASE_URL=/Cy-ExM uv run --group docs jupyter book build docs --html
  test -f docs/_build/html/index.html
  ```

  Expected: the build exits 0 and `docs/_build/html/index.html` exists.

- [ ] **Step 5: Commit the navigable site shell**

  ```bash
  git add docs/myst.yml docs/index.md
  git commit -m "docs: configure Cy-ExM Book Theme site"
  ```

## Task 3: Add the narrative tutorials and scientific evidence pages

**Files:**
- Create: `docs/getting-started.md`
- Create: `docs/registration-foundations.md`
- Create: `docs/workflow-guide.md`
- Create: `docs/cc-syn-radius-characterization.md`
- Modify: `docs/brightness-dependence-investigation.md`

**Interfaces:**
- Consumes: public `cyexm` CLI and API behavior documented in `README.md`, current
  registration invariants in `docs/AGENTS.md`, and the `myst.yml` entries from
  Task 2.
- Produces: MyST pages that build without a Python API generator, data access,
  notebook execution, or unsupported SLURM claims.

- [ ] **Step 1: Create focused authoring assertions before prose changes**

  Add the following commands to the implementation notes and run them before adding
  pages; they must fail because the listed files are absent:

  ```bash
  test -f docs/getting-started.md
  test -f docs/registration-foundations.md
  test -f docs/workflow-guide.md
  test -f docs/cc-syn-radius-characterization.md
  ```

  Expected: a non-zero exit status.

- [ ] **Step 2: Write `docs/getting-started.md` and `docs/workflow-guide.md`**

  `getting-started.md` must provide the existing `uv sync --locked` installation
  commands, explain optional extras (`workflow`, `distributed`, `hpc`), and link to
  workflow guidance. `workflow-guide.md` must provide the existing `validate`,
  `plan`, `run`, and `status` commands from `README.md`; state that planning and
  status are read-only; explain the sequential, bounded local-Dask, and SLURM
  executor boundary; and state that portable SLURM checks are construction evidence,
  not real-cluster acceptance.

  Include these exact code blocks:

  ```bash
  uv sync --locked --extra workflow
  uv run cyexm workflow validate examples/spheroid-dataset.yaml \
    --execution examples/execution-sequential.yaml
  uv run cyexm workflow plan examples/spheroid-dataset.yaml \
    --execution examples/execution-local-dask.yaml
  uv run cyexm workflow run examples/spheroid-dataset.yaml \
    --execution examples/execution-local-dask.yaml
  uv run cyexm workflow status results spheroid-registration-example
  ```

- [ ] **Step 3: Write `docs/registration-foundations.md`**

  Include these accepted invariants verbatim in prose: public pixel sizes are
  micrometers in `(x, y, z)` order; NumPy volumes are `(z, y, x)`; affine
  application is the first fixed-grid boundary; masks use values greater than zero;
  affine precedes nonlinear registration; scalar data use linear interpolation and
  masks use nearest-neighbor interpolation; affine is TRSAA/Mattes with 32 bins;
  nonlinear registration is SyNOnly; CC radius is one positive scalar voxel radius;
  omitted CC radius uses provisional value 4; and native ITK/BLAS-family threads
  default to one unless inherited or configured explicitly.

- [ ] **Step 4: Write the ongoing CC-radius page**

  Create `docs/cc-syn-radius-characterization.md` containing:

  ```markdown
  # CC SyN radius characterization

  :::{important}
  This characterization is ongoing. It does not establish an optimized Cy-ExM
  default or supersede the provisional scalar voxel radius of 4.
  :::

  Cy-ExM supports the `CC` nonlinear SyN metric with one scalar voxel radius.
  The study compares the Mattes SyN baseline with CC radii 2, 4, and 6 or 8 on
  approved cropped data. A radius spans the same number of voxels along each axis,
  so anisotropic voxel spacing makes its physical neighborhood anisotropic.

  A candidate is promoted only if it improves spatially resolved low-signal
  residuals, preserves bright-region alignment and global similarity, produces no
  folding or seam degradation, preserves Cy-ExM geometry and transform invariants,
  and has acceptable runtime. Results will be added only after that evidence is
  reviewed.
  ```

- [ ] **Step 5: Adapt the brightness investigation for publication**

  Retain its existing measured table, sources, recommendations, and proposed
  experiment. Add this status callout immediately after the title:

  ```markdown
  :::{note}
  The measured observations below are specific to the investigated spheroid data.
  Proposed acceptance thresholds and the CC-radius comparison remain characterization
  work; they are not production-default claims.
  :::
  ```

  Change the final section heading from `## Proposed characterization experiment` to
  `## Ongoing CC-radius characterization`.

- [ ] **Step 6: Run content assertions and the documentation build**

  Run:

  ```bash
  rg -n "This characterization is ongoing|provisional scalar voxel radius of 4" \
    docs/cc-syn-radius-characterization.md
  rg -n "specific to the investigated spheroid data" \
    docs/brightness-dependence-investigation.md
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  BASE_URL=/Cy-ExM uv run --group docs jupyter book build docs --html
  ```

  Expected: both `rg` commands report one match and the Book build exits 0.

- [ ] **Step 7: Commit the narrative content**

  ```bash
  git add docs/getting-started.md docs/registration-foundations.md \
    docs/workflow-guide.md docs/cc-syn-radius-characterization.md \
    docs/brightness-dependence-investigation.md
  git commit -m "docs: add Cy-ExM narrative guides"
  ```

## Task 4: Build and publish with GitHub Pages Actions

**Files:**
- Create: `.github/workflows/docs-pages.yml`

**Interfaces:**
- Consumes: the Task 1 locked `docs` group and Task 2 build output at
  `docs/_build/html`.
- Produces: a Pages artifact and a deployment from `main` to the configured
  `github-pages` environment.

- [ ] **Step 1: Write the workflow YAML and validate it locally**

  Create `.github/workflows/docs-pages.yml` with this complete content:

  ```yaml
  name: Documentation Pages

  on:
    push:
      branches: [main]
      paths:
        - "docs/**"
        - "pyproject.toml"
        - "uv.lock"
        - ".github/workflows/docs-pages.yml"
    workflow_dispatch:

  permissions:
    contents: read

  concurrency:
    group: "github-pages"
    cancel-in-progress: false

  jobs:
    build:
      runs-on: ubuntu-latest
      timeout-minutes: 15
      steps:
        - uses: actions/checkout@v6.0.2
        - uses: astral-sh/setup-uv@v8.3.2
          with:
            version: "0.7.13"
            python-version: "3.12"
            enable-cache: true
        - name: Set up Pages
          uses: actions/configure-pages@v5
        - name: Sync locked documentation environment
          run: uv sync --locked --group docs
        - name: Build documentation
          env:
            BASE_URL: /Cy-ExM
          run: uv run --group docs jupyter book build docs --html
        - name: Upload Pages artifact
          uses: actions/upload-pages-artifact@v4
          with:
            path: docs/_build/html

    deploy:
      if: github.ref == 'refs/heads/main'
      needs: build
      runs-on: ubuntu-latest
      permissions:
        pages: write
        id-token: write
      environment:
        name: github-pages
        url: ${{ steps.deployment.outputs.page_url }}
      steps:
        - name: Deploy to GitHub Pages
          id: deployment
          uses: actions/deploy-pages@v4
  ```

  Validate parser syntax without contacting GitHub:

  ```bash
  ruby -e 'require "yaml"; YAML.load_file(".github/workflows/docs-pages.yml")'
  ```

  Expected: exit 0.

- [ ] **Step 2: Verify the built artifact is suitable for upload**

  Run:

  ```bash
  test -f docs/_build/html/index.html
  test ! -L docs/_build/html
  ```

  Expected: exit 0; the deployable root is a real directory with an HTML entrypoint.

- [ ] **Step 3: Commit the deployment workflow**

  ```bash
  git add .github/workflows/docs-pages.yml
  git commit -m "ci: publish documentation to GitHub Pages"
  ```

## Task 5: Run the final portable documentation gates

**Files:**
- Verify only: `pyproject.toml`, `uv.lock`, `docs/`, `.github/workflows/docs-pages.yml`

**Interfaces:**
- Consumes: all earlier tasks.
- Produces: evidence that the documentation can render locally and the source tree
  remains formatted and lint-clean.

- [ ] **Step 1: Check the lock and build from the final source tree**

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  uv lock --check
  uv sync --locked --group docs
  BASE_URL=/Cy-ExM uv run --group docs jupyter book build docs --html
  test -f docs/_build/html/index.html
  ```

  Expected: all commands exit 0.

- [ ] **Step 2: Run repository formatting and linting**

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  uv run ruff check .
  uv run ruff format --check .
  ```

  Expected: both commands exit 0. Markdown and workflow files should not cause
  Python lint changes.

- [ ] **Step 3: Inspect final scope and commit any validation-driven correction**

  ```bash
  git status --short
  git diff --check HEAD~4..HEAD
  ```

  Expected: only intended documentation, lockfile, ignore, and workflow changes are
  tracked; existing `slurm-*.out` files remain untracked and untouched. If validation
  requires a source correction, stage only that correction and commit it with a
  focused `docs:` or `ci:` message.
