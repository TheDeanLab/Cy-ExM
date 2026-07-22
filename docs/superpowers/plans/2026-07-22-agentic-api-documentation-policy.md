# Agentic API Documentation Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Cy-ExM's API-reuse and Numpydoc policy concise at the repository
root, detailed in a maintained guide, and auditable through plans, commits, and PRs.

**Architecture:** Keep a universal decision rule in `AGENTS.md`, move detailed
Numpydoc guidance to `docs/agentic-api-documentation.md`, and make source and
documentation child guidance refer to the same semantic API classification. The
policy records detailed reuse analysis in implementation plans, the final decision in
qualifying commit bodies, and links both from PR descriptions.

**Tech Stack:** Markdown, MyST-compatible Markdown, Git commit messages.

## Global Constraints

- Preserve all existing scientific, HPC-storage, dependency, and source-safety rules.
- A user-callable API is a relatively high-level entry point, resource owner, data
  model, executor, or explicitly supported extension interface.
- Full Numpydoc is required for those interfaces; concise Numpydoc is permitted for
  simple pure value objects and helpers.
- Do not add documentation tooling, CI, pre-commit hooks, or dependencies in this
  policy-only change.
- Do not stage or modify existing untracked `slurm-*.out` files.

---

### Task 1: Establish the concise universal policy

**Files:**
- Modify: `AGENTS.md:117-737`

**Interfaces:**
- Consumes: the existing scientific, execution, and validation rules in root
  `AGENTS.md`.
- Produces: a root-level rule that directs agents to
  `docs/agentic-api-documentation.md` before adding or changing a user-callable API.

- [x] **Step 1: Replace the detailed appended policy with the concise root contract**

  Replace lines 117-737 with a section that states all of the following:

  ```markdown
  ## API reuse and documentation policy

  Before adding or changing a user-callable API, search source, tests,
  documentation, examples, exports, the proposed name, and plausible synonyms.
  Identify the closest existing symbols; prefer composition, extension,
  parameterization, or repair over a parallel abstraction. Do not add a wrapper that
  merely renames an operation or duplicates its lifecycle, validation, persistence,
  state management, or cleanup.

  A user-callable API is a relatively high-level entry point, resource owner, data
  model, executor, or explicitly supported extension interface. It requires full
  Numpydoc. Simple pure value objects and helpers may use concise Numpydoc. Treat
  documented units, shapes, coordinate systems, state transitions, side effects,
  ownership, and errors as contracts.

  For multi-step or user-callable API work, record a `Reuse analysis` in the
  implementation plan and in the body of the API-changing commit. A PR description
  links to both records and notes deviations; preserve the analysis in squash commits.
  Read `docs/agentic-api-documentation.md` before designing or documenting such an
  API.
  ```

- [x] **Step 2: Inspect the resulting root section**

  Run:

  ```bash
  sed -n '105,170p' AGENTS.md
  ```

  Expected: the short policy follows the existing test guidance, links to the detailed
  guide, and no generic acquisition examples or Numpydoc validation catalog remain in
  the root file.

- [x] **Step 3: Commit the root-policy change**

  ```bash
  git add AGENTS.md
  git commit -m "docs: condense root API documentation policy"
  ```

### Task 2: Add the maintained API-documentation guide

**Files:**
- Create: `docs/agentic-api-documentation.md`

**Interfaces:**
- Consumes: the detailed policy currently appended to `AGENTS.md`, the accepted
  scope from the design specification, and Cy-ExM's documented spatial conventions.
- Produces: the referenced guide for API discovery, Numpydoc content, module maps,
  change synchronization, validation adoption, and `Reuse analysis` templates.

- [x] **Step 1: Create the guide with the approved section structure**

  Write a MyST-compatible Markdown document containing these headings, in order:

  ```markdown
  # Agentic API documentation guide
  ## Scope and decision order
  ## Discovery commands
  ## Public API design
  ## Numpydoc contract
  ## Numpydoc section guidance
  ## Class and module API maps
  ## Public API change synchronization
  ## Reuse analysis records
  ## Numpydoc validation adoption
  ## Canonical Cy-ExM template
  ```

  Preserve useful detailed guidance from the prior root policy, but replace generic
  acquisition examples with Cy-ExM-oriented examples such as `build_workflow_plan`,
  `run_workflow`, `ImageRegistration`, `ChunkedImageRegistration`, and
  `transform_channel`. State the fixed/moving invariant, public `(x, y, z)` spacing,
  and NumPy `(z, y, x)` array convention whenever an example accepts image geometry.
  State that Numpydoc validation configuration is an adoption recommendation, not a
  configured present-day gate.

- [x] **Step 2: Add the durable reuse-record template**

  Include this implementation-plan template and specify that it is mandatory for
  multi-step or user-callable API work:

  ```markdown
  ## Reuse analysis

  - Closest existing symbols: `build_workflow_plan`, `run_workflow`.
  - Decision: extend `build_workflow_plan` with a keyword-only option.
  - Rejected alternative: a separate planner would duplicate validation, ordering,
    and artifact-lifecycle rules.
  - Documentation updates: API map, `See Also`, and canonical example.
  ```

  Include a concise commit-body version, require PR descriptions to link the plan and
  relevant commit(s), and require squash commits to retain the analysis.

- [x] **Step 3: Inspect Markdown and repository links**

  Run:

  ```bash
  rg -n '^(#|##|###) |\[.*\]\([^)]*\)' docs/agentic-api-documentation.md
  ```

  Expected: all required headings appear once, and any relative links resolve to
  tracked repository files.

- [x] **Step 4: Commit the guide**

  ```bash
  git add docs/agentic-api-documentation.md
  git commit -m "docs: add agentic API documentation guide"
  ```

### Task 3: Add scoped source and documentation rules

**Files:**
- Modify: `src/cyexm/AGENTS.md:1-17`
- Modify: `docs/AGENTS.md:1-20`

**Interfaces:**
- Consumes: `AGENTS.md` concise policy and `docs/agentic-api-documentation.md`
  detailed requirements.
- Produces: source-specific API classification and documentation-specific
  synchronization rules without duplicating the detailed guide.

- [x] **Step 1: Add source-tree API classification guidance**

  Append to `src/cyexm/AGENTS.md`:

  ```markdown
  ## User-callable API documentation

  For a user-callable entry point, resource owner, data model, executor, or explicit
  extension interface, follow `docs/agentic-api-documentation.md` and provide full
  Numpydoc. Simple pure value objects and helpers may use concise Numpydoc.

  An internal symbol is reusable only when it is explicitly designated as an
  extension interface and documented as such; do not turn implementation details into
  public contracts merely to share code. Before adding a user-callable symbol, record
  the closest existing symbols and why they cannot be safely reused in the required
  `Reuse analysis` record.
  ```

- [x] **Step 2: Add documentation synchronization guidance**

  Append to `docs/AGENTS.md`:

  ```markdown
  ## API documentation synchronization

  When a user-callable API changes, follow `agentic-api-documentation.md`. Keep its
  module API map, `See Also` decision boundaries, canonical examples, and relevant
  user-facing pages synchronized with the implemented signature and behavior. Use
  examples that preserve Cy-ExM's fixed/moving roles, public XYZ spacing, and NumPy
  ZYX volume order.
  ```

- [x] **Step 3: Verify the references and duplicate-policy boundary**

  Run:

  ```bash
  rg -n 'agentic-api-documentation|Reuse analysis|user-callable API' \
    AGENTS.md src/cyexm/AGENTS.md docs/AGENTS.md
  ```

  Expected: root, source, and documentation guidance agree on the guide location and
  use `Reuse analysis` only for qualifying work.

- [x] **Step 4: Commit the scoped guidance**

  ```bash
  git add src/cyexm/AGENTS.md docs/AGENTS.md
  git commit -m "docs: scope API documentation guidance"
  ```

### Task 4: Validate documentation-only changes

**Files:**
- Verify: `AGENTS.md`
- Verify: `src/cyexm/AGENTS.md`
- Verify: `docs/AGENTS.md`
- Verify: `docs/agentic-api-documentation.md`

**Interfaces:**
- Consumes: completed policy documents.
- Produces: evidence that Markdown is clean and the configured documentation build
  accepts maintained documentation without contacting datasets, schedulers, or SLURM.

- [x] **Step 1: Check whitespace and Markdown content**

  Run:

  ```bash
  git diff --check HEAD~3..HEAD
  rg -n 'TBD|TODO|Agent Notes|Reuse Guidance' \
    AGENTS.md src/cyexm/AGENTS.md docs/AGENTS.md docs/agentic-api-documentation.md
  ```

  Expected: `git diff --check` is silent; the search returns no prohibited custom
  Numpydoc section names or placeholders.

- [x] **Step 2: Build the maintained documentation**

  Run:

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  uv run jb build docs
  ```

  Expected: the documentation build succeeds without executing notebooks, accessing
  lab storage, or contacting a scheduler. If `jb` is not provided by the locked
  environment, record that it is unavailable rather than installing a dependency.

  Result: `uv run jb build docs` could not spawn `jb`; the locked environment does
  not provide that executable. No dependency was installed.

- [x] **Step 3: Inspect final scope**

  Run:

  ```bash
  git status --short
  git log --format=fuller -3
  ```

  Expected: only intended policy files are staged or committed; pre-existing
  untracked `slurm-*.out` files remain untouched; qualifying policy commits retain a
  concise `Reuse analysis` where the policy requires it.
