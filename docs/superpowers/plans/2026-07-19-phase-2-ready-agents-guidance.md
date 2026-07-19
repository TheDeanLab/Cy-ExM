# Phase 2-Ready AGENTS Guidance Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every maintained Cy-ExM directory layer accurate local agent guidance while preserving Phase 1 as the stable runtime and recording safe boundaries for a separately designed Phase 2.

**Architecture:** The root `AGENTS.md` remains the authoritative project contract, and child files add only directory-specific rules. Five new instruction files complete directory coverage; existing child guidance is corrected in focused groups. One repository-policy module mechanically enforces inventory, reproducible commands, current-versus-proposed language, and the critical cross-layer boundaries.

**Tech Stack:** Markdown, Python 3.12, pytest, uv 0.7.13, Ruff, BasedPyright, pre-commit

## Global Constraints

- Work only in the isolated `docs/phase-2-agents-guidance` branch and its worktree.
- The merged Phase 1 runtime remains the current production surface.
- Do not add YAML parsing, manifests, execution profiles, CLIs, Dask, distributed, dask-jobqueue, SLURM, or scheduler runtime code.
- Do not change production APIs, registration behavior, dependencies, `uv.lock`, examples, CI behavior, or `docs/phase-1-acceptance.md`.
- Preserve the read-only ClearEx pin `2311a873b783477add7391ff00245f76490147f0`, the exact academic-use license, and the separate spheroid registration process and outputs.
- Treat the Phase 1 artifact store and serial chunk workflow as the restart and numerical reference for future execution layers.
- Keep scientific dataset manifests separate from machine-specific execution profiles.
- Do not nest chunk multiprocessing, a Dask chunk graph, or `ParallelChunkedImageRegistration` inside round-level Dask work by default.
- Before every uv command, export:

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  ```

- For scientific tests, set all six native-thread variables to `1` before Python starts.
- Use `apply_patch` for tracked edits. Keep generated caches, package metadata, and review packages out of commits.
- Follow TDD for every policy change: add one focused failing assertion, observe the expected failure, make the minimum guidance edit, and rerun the focused test.

---

## File map

| Path | Responsibility in this refresh |
| --- | --- |
| `AGENTS.md` | Stable Phase 1 contract, Phase 2 runtime design gate, reproducible commands, and cross-phase invariants |
| `.github/AGENTS.md` | General GitHub automation security, permissions, action pinning, and portability |
| `.github/workflows/AGENTS.md` | Workflow-specific uv, Python, native-thread, selector, timeout, and HPC boundaries |
| `docs/AGENTS.md` | Current behavior versus proposed Phase 2 documentation |
| `docs/superpowers/AGENTS.md` | Dated design/plan artifact lifecycle and generated-evidence exclusions |
| `docs/superpowers/plans/AGENTS.md` | Executable plan content, TDD, evidence, and historical-plan policy |
| `docs/superpowers/specs/AGENTS.md` | Design content, alternatives, failure/testing criteria, assumptions, and historical-spec policy |
| `examples/AGENTS.md` | Current Phase 1 examples and the gate for any future Phase 2 examples |
| `src/AGENTS.md` | Future scheduler boundary without module-import side effects |
| `src/cyexm/AGENTS.md` | Stable registration API and future workflow/configuration namespace boundary |
| `src/cyexm/registration/AGENTS.md` | Correct ownership of backend writers versus artifact lifecycle/orchestration |
| `src/cyexm/runtime/AGENTS.md` | Per-worker thread resolution before scientific imports and no scheduler ownership |
| `tests/AGENTS.md` | Marker taxonomy and the narrow concrete-ANTs unit exception |
| `tests/integration/AGENTS.md` | Bounded future local-scheduler checks versus explicit HPC SLURM checks |
| `tests/unit/AGENTS.md` | Dependency-neutral future manifest/schema policy tests |
| `tests/unit/test_repository_policy.py` | Exact inventory and enforceable cross-file guidance assertions |

The existing `examples/notebooks/AGENTS.md`, `examples/scripts/AGENTS.md`,
`src/cyexm/io/AGENTS.md`, and all unchanged ancestor rules remain in force and receive
a final accuracy review rather than gratuitous edits.

---

### Task 1: Make the root contract current and reproducible

**Files:**
- Modify: `tests/unit/test_repository_policy.py:22-30,197-210`
- Modify: `AGENTS.md:4-17,55-70`

**Interfaces:**
- Consumes: `_read_repository_file(path: str) -> str`
- Produces: `_normalized_repository_file(path: str) -> str` for all later policy tasks; a root contract that later child files inherit

- [ ] **Step 1: Add the normalization helper and failing root-guidance test**

Add this helper after `_read_repository_file`:

```python
def _normalized_repository_file(path: str) -> str:
    return " ".join(_read_repository_file(path).split()).lower()
```

Add this test after `test_ci_runs_only_portable_phase_one_gates_with_bounded_native_threads`:

```python
def test_root_guidance_tracks_stable_phase_one_and_gates_phase_two() -> None:
    root = _normalized_repository_file("AGENTS.md")
    for required in (
        "phase 1 is the current stable runtime",
        "dedicated phase 2 runtime design and implementation plan",
        "scientific dataset manifests",
        "execution profiles",
        "serial chunk workflow remains",
        "nested chunk multiprocessing",
        "uv lock --check",
        "uv sync --locked --group dev --extra examples",
    ):
        assert required in root

    root_lines = set(_read_repository_file("AGENTS.md").splitlines())
    assert "uv lock" not in root_lines
    assert "uv sync --group dev --extra examples" not in root_lines
```

- [ ] **Step 2: Run the focused test and observe the stale-root failure**

Run:

```bash
uv run pytest tests/unit/test_repository_policy.py::test_root_guidance_tracks_stable_phase_one_and_gates_phase_two -q
```

Expected: FAIL first on `"phase 1 is the current stable runtime"` because the existing
root file describes Phase 1 only as a permanent prohibition boundary.

- [ ] **Step 3: Replace the Phase 1 scope bullets with the exact current/future contract**

Replace the existing Phase 1 prohibition bullet in the root `Project contract`
section with:

```markdown
- Phase 1 is the current stable runtime: whole-image and serial chunked registration,
  restart-safe artifacts, strict geometry/TIFF boundaries, and configurable native
  threads. Preserve that API and its acceptance evidence while Phase 2 is designed.
- Do not add Phase 2 runtime code, dependencies, or public APIs until a dedicated
  Phase 2 runtime design and implementation plan are approved. This gate covers YAML
  scientific dataset manifests, execution profiles, Dask/distributed execution,
  dask-jobqueue/SLURM integration, and new workflow CLIs.
- Scientific dataset manifests describe rounds, channels, registration inputs, voxel
  spacing, and scientific choices; machine-specific local or HPC settings belong in
  separate execution profiles. The Phase 1 artifact store remains the only restart
  authority, and the serial chunk workflow remains the numerical reference. Do not
  introduce nested chunk multiprocessing, a Dask chunk graph, or
  `ParallelChunkedImageRegistration` inside round-level tasks by default.
- `clearex.registration.pipeline` and `clearex.registration.tre` remain out of scope;
  do not recreate unrelated ClearEx surfaces in this repository.
```

Leave the ClearEx pin, license, inheritance, scientific invariants, storage rules, and
spheroid-process protection unchanged.

- [ ] **Step 4: Replace mutable setup commands with locked validation commands**

Change the root command block to begin with:

```bash
uv lock --check
uv sync --locked --group dev --extra examples
```

Immediately below the block, add:

```markdown
Run bare `uv lock` only when an approved dependency change intentionally regenerates
`uv.lock`; inspect and commit that lockfile change with the dependency edit.
```

- [ ] **Step 5: Run the focused root test and diff check**

Run:

```bash
uv run pytest tests/unit/test_repository_policy.py::test_root_guidance_tracks_stable_phase_one_and_gates_phase_two -q
git diff --check
```

Expected: `1 passed`; `git diff --check` prints nothing.

- [ ] **Step 6: Commit the root contract**

```bash
git add AGENTS.md tests/unit/test_repository_policy.py
git commit -m "docs: gate Phase 2 work in root guidance"
```

---

### Task 2: Complete CI and planning-directory guidance

**Files:**
- Modify: `tests/unit/test_repository_policy.py:11-30`
- Create: `.github/AGENTS.md`
- Create: `.github/workflows/AGENTS.md`
- Create: `docs/superpowers/AGENTS.md`
- Create: `docs/superpowers/plans/AGENTS.md`
- Create: `docs/superpowers/specs/AGENTS.md`

**Interfaces:**
- Consumes: `_normalized_repository_file(path: str) -> str`
- Produces: `AGENT_GUIDANCE_PATHS: frozenset[str]`; exact local guidance at all 18 maintained layers

- [ ] **Step 1: Add the expected inventory and failing coverage test**

Add this constant below `THREAD_ENVIRONMENT`:

```python
AGENT_GUIDANCE_PATHS = frozenset(
    {
        "AGENTS.md",
        ".github/AGENTS.md",
        ".github/workflows/AGENTS.md",
        "docs/AGENTS.md",
        "docs/superpowers/AGENTS.md",
        "docs/superpowers/plans/AGENTS.md",
        "docs/superpowers/specs/AGENTS.md",
        "examples/AGENTS.md",
        "examples/notebooks/AGENTS.md",
        "examples/scripts/AGENTS.md",
        "src/AGENTS.md",
        "src/cyexm/AGENTS.md",
        "src/cyexm/io/AGENTS.md",
        "src/cyexm/registration/AGENTS.md",
        "src/cyexm/runtime/AGENTS.md",
        "tests/AGENTS.md",
        "tests/integration/AGENTS.md",
        "tests/unit/AGENTS.md",
    }
)
```

Add this test after the root-guidance test:

```python
def test_every_maintained_directory_has_local_agent_guidance() -> None:
    actual = {
        str(path.relative_to(REPOSITORY_ROOT))
        for path in REPOSITORY_ROOT.rglob("AGENTS.md")
        if not {".git", ".worktrees", ".venv"}
        & set(path.relative_to(REPOSITORY_ROOT).parts)
    }
    assert actual == AGENT_GUIDANCE_PATHS

    required_by_path = {
        ".github/AGENTS.md": (
            "least-privilege",
            "version-pinned",
            "public github-hosted runners",
            "lab paths",
        ),
        ".github/workflows/AGENTS.md": (
            "uv sync --locked --group dev --extra examples",
            "python 3.12",
            "all six native-thread",
            "unit and not hpc",
            "integration and not hpc",
            "slurm",
        ),
        "docs/superpowers/AGENTS.md": (
            "dated",
            "design",
            "implementation plan",
            "generated review packages",
        ),
        "docs/superpowers/plans/AGENTS.md": (
            "exact files",
            "tdd",
            "evidence",
            "historical plans",
        ),
        "docs/superpowers/specs/AGENTS.md": (
            "current behavior",
            "alternatives",
            "failure handling",
            "acceptance criteria",
            "erratum",
        ),
    }
    for path, required_fragments in required_by_path.items():
        guidance = _normalized_repository_file(path)
        for fragment in required_fragments:
            assert fragment in guidance, (path, fragment)
```

- [ ] **Step 2: Run the coverage test and observe the missing-file failure**

Run:

```bash
uv run pytest tests/unit/test_repository_policy.py::test_every_maintained_directory_has_local_agent_guidance -q
```

Expected: FAIL because the five new guidance paths are absent from `actual`.

- [ ] **Step 3: Create `.github/AGENTS.md`**

```markdown
# GitHub automation guidance

- Keep workflow and repository permissions least-privilege; ordinary CI needs only
  read access to repository contents.
- Use immutable commit pins or explicit version-pinned releases for third-party
  actions. Verify a requested upgrade exists and review its release notes before
  changing the pin.
- Keep portable validation on public GitHub-hosted runners. Never embed credentials,
  private tokens, lab paths, `/project` or `/archive` locations, or cluster account
  details in committed automation.
- Do not make release publication, branch-protection changes, repository visibility
  changes, or destructive maintenance part of an unrelated implementation task.
- Workflow changes must update their exact repository-policy assertions and pass YAML
  parsing plus the affected local commands before review.
```

- [ ] **Step 4: Create `.github/workflows/AGENTS.md`**

```markdown
# GitHub Actions workflow guidance

- Preserve Python 3.12, the repository-pinned uv version, and
  `uv sync --locked --group dev --extra examples`; CI must never regenerate the lock.
- Set all six native-thread variables to `1` at job scope before Python starts:
  `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`, `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and
  `VECLIB_MAXIMUM_THREADS`.
- Keep explicit portable selectors: `unit and not hpc` for `tests/unit` and
  `integration and not hpc` for `tests/integration`. Retain a finite job timeout.
- Do not access lab storage, submit SLURM jobs, initialize Dask clusters, or make a
  portable check conditional on hostnames. Future HPC gates require an approved lab
  runner or an explicit manual cluster command.
- Keep workflow permissions read-only unless a reviewed job has a documented need for
  one additional permission. Never expose secrets to pull requests from untrusted
  code.
```

- [ ] **Step 5: Create `docs/superpowers/AGENTS.md`**

```markdown
# Design and implementation artifact guidance

- Store dated, reviewable design specifications under `specs/` and executable
  implementation plans under `plans/`; use `YYYY-MM-DD-<topic>.md` names.
- A design records approved decisions and boundaries. Its implementation plan is a
  separate artifact written only after the design is reviewed.
- Keep completed Phase 1 artifacts as historical evidence. Record a correction in a
  new dated document or an explicitly labeled erratum instead of silently rewriting
  history.
- Do not commit generated review packages, temporary evidence, agent ledgers, command
  transcripts, or files from `.superpowers/sdd`.
- Link maintained user documentation to current behavior; design and plan documents
  may describe proposed work only when they label it clearly.
```

- [ ] **Step 6: Create `docs/superpowers/plans/AGENTS.md`**

```markdown
# Implementation-plan guidance

- Name exact files, interfaces, dependency order, commands, expected results, review
  gates, and commit boundaries. An implementer must not need unstated repository
  knowledge.
- Break work into independently testable tasks with explicit TDD red and green steps.
  Show complete test and implementation snippets rather than placeholders.
- Use checkboxes as execution state, not prediction. Do not mark a step complete until
  fresh evidence from its exact command exists.
- Include focused tests before broad gates and preserve repository safety constraints
  in every task that can touch external data, native runtimes, or GitHub state.
- Completed historical plans remain evidence of the path actually taken. Use a new
  correction plan when scope or implementation changes materially; do not rewrite
  historical plans to conceal that change.
```

- [ ] **Step 7: Create `docs/superpowers/specs/AGENTS.md`**

```markdown
# Design-specification guidance

- Separate current behavior from proposed behavior. Define context, goals, non-goals,
  alternatives, selected architecture, data flow, failure handling, testing, and
  acceptance criteria.
- State scientific invariants, public coordinate conventions, dependency boundaries,
  and ownership of persistent state wherever the design can affect registration.
- Label environment assumptions and verify them during implementation. Cluster
  partition, account, resource syntax, shared-filesystem, startup, and memory details
  are evidence requirements, not facts to invent.
- Resolve ambiguous public APIs and error behavior in the specification before an
  implementation plan is written. Avoid placeholders and open-ended requirements.
- Preserve completed historical specifications. Put a material correction in a new
  dated specification or add an explicitly labeled erratum that explains the change.
```

- [ ] **Step 8: Run the coverage test and inspect inheritance**

Run:

```bash
uv run pytest tests/unit/test_repository_policy.py::test_every_maintained_directory_has_local_agent_guidance -q
fd --hidden --type f '^AGENTS\.md$' --exclude .git --exclude .worktrees --exclude .venv . | sort
git diff --check
```

Expected: `1 passed`; the file listing contains exactly the 18 paths in
`AGENT_GUIDANCE_PATHS`; `git diff --check` prints nothing.

- [ ] **Step 9: Commit complete directory coverage**

```bash
git add .github/AGENTS.md .github/workflows/AGENTS.md \
  docs/superpowers/AGENTS.md docs/superpowers/plans/AGENTS.md \
  docs/superpowers/specs/AGENTS.md tests/unit/test_repository_policy.py
git commit -m "docs: add CI and planning agent guidance"
```

---

### Task 3: Refresh documentation and example boundaries

**Files:**
- Modify: `tests/unit/test_repository_policy.py`
- Modify: `docs/AGENTS.md:1-18`
- Modify: `examples/AGENTS.md:1-20`

**Interfaces:**
- Consumes: `_normalized_repository_file(path: str) -> str`
- Produces: unambiguous current-versus-proposed language for user documentation and teaching material

- [ ] **Step 1: Add the failing docs/examples boundary test**

```python
def test_documentation_and_examples_cannot_lead_phase_two_runtime() -> None:
    documentation = _normalized_repository_file("docs/AGENTS.md")
    for required in (
        "merged phase 1 behavior",
        "proposed phase 2",
        "not installable behavior",
        "dedicated runtime design",
    ):
        assert required in documentation

    examples = _normalized_repository_file("examples/AGENTS.md")
    for required in (
        "phase 1 examples are the current runnable surface",
        "runtime api is implemented and accepted",
        "examples never lead production behavior",
        "scientific manifest",
        "execution profile",
    ):
        assert required in examples
```

- [ ] **Step 2: Run the test and observe the indefinite-prohibition failure**

Run:

```bash
uv run pytest tests/unit/test_repository_policy.py::test_documentation_and_examples_cannot_lead_phase_two_runtime -q
```

Expected: FAIL first on `"merged phase 1 behavior"`.

- [ ] **Step 3: Replace the first documentation bullet and add the runtime-design rule**

Use these first two bullets in `docs/AGENTS.md`:

```markdown
- Describe merged Phase 1 behavior separately from proposed Phase 2 work. YAML
  manifests, round-level schedulers, and cluster execution are not installable
  behavior until their dedicated runtime design, implementation, and acceptance gates
  are complete.
- Design and plan documents may describe proposed Phase 2 decisions when they label
  status and assumptions explicitly. User guides and API examples describe only the
  runtime available on the referenced branch.
```

Keep the existing API synchronization, artifact/restart, scientific-statement,
migration, and sensitive-data rules unchanged.

- [ ] **Step 4: Replace the Phase 1-only example bullet with the exact transition**

Replace the final `examples/AGENTS.md` bullet with:

```markdown
- Phase 1 examples are the current runnable surface. Add a Phase 2 example only after
  its runtime API is implemented and accepted; examples never lead production
  behavior. Future manifest examples keep the scientific manifest separate from any
  local or HPC execution profile and remain runnable without cluster services where
  the API permits.
```

- [ ] **Step 5: Verify the boundary test and unchanged child accuracy**

Run:

```bash
uv run pytest tests/unit/test_repository_policy.py::test_documentation_and_examples_cannot_lead_phase_two_runtime -q
bat --style=plain --paging=never examples/notebooks/AGENTS.md examples/scripts/AGENTS.md
git diff --check
```

Expected: `1 passed`; the child files remain consistent with the refreshed parent;
`git diff --check` prints nothing.

- [ ] **Step 6: Commit documentation and example boundaries**

```bash
git add docs/AGENTS.md examples/AGENTS.md tests/unit/test_repository_policy.py
git commit -m "docs: distinguish current and proposed workflows"
```

---

### Task 4: Correct source ownership and future execution boundaries

**Files:**
- Modify: `tests/unit/test_repository_policy.py`
- Modify: `src/AGENTS.md:1-14`
- Modify: `src/cyexm/AGENTS.md:1-17`
- Modify: `src/cyexm/registration/AGENTS.md:1-26`
- Modify: `src/cyexm/runtime/AGENTS.md:1-10`

**Interfaces:**
- Consumes: `_normalized_repository_file(path: str) -> str`
- Produces: a future execution namespace boundary; correct backend/artifact ownership; explicit per-worker thread safety

- [ ] **Step 1: Add the failing source-boundary test**

```python
def test_source_guidance_keeps_execution_out_of_registration_and_runtime() -> None:
    source = _normalized_repository_file("src/AGENTS.md")
    for required in (
        "future execution boundary",
        "scheduler imports",
        "module scope",
        "no import-time cluster",
    ):
        assert required in source

    package = _normalized_repository_file("src/cyexm/AGENTS.md")
    assert "workflow or configuration namespace" in package
    assert "scheduler concerns do not belong in `cyexm.registration`" in package

    registration = _normalized_repository_file("src/cyexm/registration/AGENTS.md")
    for required in (
        "backend artifact writer methods",
        "`artifacts.py`, `whole_image.py`, and `chunked.py`",
        "must not own workflow scheduling",
    ):
        assert required in registration

    runtime = _normalized_repository_file("src/cyexm/runtime/AGENTS.md")
    for required in (
        "worker process",
        "before importing ants, numpy, or scipy",
        "must not initialize or own a scheduler",
    ):
        assert required in runtime
```

- [ ] **Step 2: Run the test and observe the missing future-boundary failure**

Run:

```bash
uv run pytest tests/unit/test_repository_policy.py::test_source_guidance_keeps_execution_out_of_registration_and_runtime -q
```

Expected: FAIL first on `"future execution boundary"`.

- [ ] **Step 3: Add the future execution boundary to `src/AGENTS.md`**

Append these bullets:

```markdown
- A future Phase 2 execution boundary may import scheduler libraries only inside the
  operation that creates or connects to a scheduler. Keep scheduler imports out of
  module scope and preserve no import-time cluster, client, worker, filesystem, or
  network activity.
- Scientific manifest models and planning policy must remain usable without importing
  ANTs or scheduler libraries. Machine-specific execution adapters depend on those
  models, never the reverse.
```

- [ ] **Step 4: Add the package namespace boundary to `src/cyexm/AGENTS.md`**

Append:

```markdown
- A future workflow or configuration namespace requires its own approved runtime
  design and plan. Scheduler concerns do not belong in `cyexm.registration`; keep the
  stable Phase 1 registration API usable without Dask, distributed, dask-jobqueue, or
  YAML workflow dependencies.
```

- [ ] **Step 5: Correct the overbroad registration writer rule**

Replace the bullet beginning `Artifact writer methods` with:

```markdown
- Backend artifact writer methods must write exactly to the supplied path so callers
  can provide atomic temporary destinations. `backend.py` must not own artifact
  lifecycle or workflow scheduling; completion records and orchestration remain in
  `artifacts.py`, `whole_image.py`, and `chunked.py`.
```

- [ ] **Step 6: Add the worker-process rule to runtime guidance**

Append to `src/cyexm/runtime/AGENTS.md`:

```markdown
- Every local or distributed worker process must resolve and apply its explicit native
  thread configuration before importing ANTs, NumPy, or SciPy. This package reports
  thread policy; it must not initialize or own a scheduler.
```

- [ ] **Step 7: Run the source test, import-contract test, and diff check**

Run:

```bash
uv run pytest tests/unit/test_repository_policy.py::test_source_guidance_keeps_execution_out_of_registration_and_runtime -q
uv run pytest tests/unit/test_import_contract.py -q
git diff --check
```

Expected: source policy `1 passed`; import contract `4 passed`; no whitespace errors.

- [ ] **Step 8: Commit source boundaries**

```bash
git add src/AGENTS.md src/cyexm/AGENTS.md \
  src/cyexm/registration/AGENTS.md src/cyexm/runtime/AGENTS.md \
  tests/unit/test_repository_policy.py
git commit -m "docs: define future execution ownership"
```

---

### Task 5: Align test guidance and run the complete acceptance gate

**Files:**
- Modify: `tests/unit/test_repository_policy.py`
- Modify: `tests/AGENTS.md:1-12`
- Modify: `tests/integration/AGENTS.md:1-31`
- Modify: `tests/unit/AGENTS.md:1-13`

**Interfaces:**
- Consumes: `_normalized_repository_file(path: str) -> str`; the root Phase 2 gate and future execution boundary
- Produces: non-contradictory unit/integration/HPC classification for current and future tests; final executable acceptance evidence

- [ ] **Step 1: Add the failing test-taxonomy guidance test**

```python
def test_test_guidance_separates_boundary_objects_local_schedulers_and_hpc() -> None:
    tests = _normalized_repository_file("tests/AGENTS.md")
    for required in (
        "narrow concrete-image exception",
        "real registration belongs in integration",
        "real slurm submission",
        "hpc",
    ):
        assert required in tests

    unit = _normalized_repository_file("tests/unit/AGENTS.md")
    for required in (
        "future manifest and planning policy",
        "plain python data",
        "must not import scheduler",
    ):
        assert required in unit

    integration = _normalized_repository_file("tests/integration/AGENTS.md")
    for required in (
        "approved scheduler dependency",
        "bounded local-scheduler",
        "actual slurm submission remains `hpc`",
    ):
        assert required in integration
```

- [ ] **Step 2: Run the test and observe the parent-taxonomy failure**

Run:

```bash
uv run pytest tests/unit/test_repository_policy.py::test_test_guidance_separates_boundary_objects_local_schedulers_and_hpc -q
```

Expected: FAIL first on `"narrow concrete-image exception"`.

- [ ] **Step 3: Clarify the parent test taxonomy**

Replace the first `tests/AGENTS.md` bullet with:

```markdown
- Mark tests as `unit`, `integration`, or `hpc`. Unit tests are isolated or mocked,
  except for the narrow concrete-image exception defined by `tests/unit/AGENTS.md`;
  real registration belongs in integration. Tests that require representative lab
  data, real SLURM submission, or cluster services are explicitly selected as `hpc`.
```

- [ ] **Step 4: Add dependency-neutral future policy to unit guidance**

Append to `tests/unit/AGENTS.md`:

```markdown
- Future manifest and planning policy tests use plain Python data and temporary files;
  they must not import scheduler or scientific runtimes. Scheduler adapter behavior
  starts at an approved integration boundary.
```

- [ ] **Step 5: Add future local-scheduler versus SLURM guidance to integration tests**

Append to `tests/integration/AGENTS.md`:

```markdown
- After a scheduler dependency is approved and declared, bounded local-scheduler
  checks may use the `integration` marker with deterministic synthetic tasks. Actual
  SLURM submission remains `hpc` and requires an explicit cluster allocation, cleanup
  assertions, and machine-readable job evidence.
```

- [ ] **Step 6: Run every focused repository-policy test**

Run:

```bash
uv run pytest tests/unit/test_repository_policy.py -q
```

Expected: `8 passed` (the three existing policy tests plus five new tests).

- [ ] **Step 7: Run the complete portable verification matrix**

Run:

```bash
uv lock --check
uv sync --locked --group dev --extra examples
uv run pytest -m "unit and not hpc" tests/unit -q
uv run pytest -m "integration and not hpc" tests/integration -q
uv run pytest tests/unit tests/integration -q
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
uv run pre-commit validate-config .pre-commit-config.yaml
uv run pre-commit run --all-files
```

Expected:

- lock and sync exit zero without changing `uv.lock`;
- `609` unit tests pass;
- `40` integration tests pass;
- `649` combined portable tests pass;
- Ruff reports `59 files already formatted` and no lint errors;
- BasedPyright reports `0 errors, 0 warnings, 0 notes`;
- all four pre-commit hooks pass.

- [ ] **Step 8: Audit every instruction file and forbidden scope**

Run:

```bash
fd --hidden --type f '^AGENTS\.md$' \
  --exclude .git --exclude .worktrees --exclude .venv . | sort
rg -n "Phase 1|Phase 2|Dask|SLURM|YAML|uv lock|uv sync|artifact writer" \
  --glob 'AGENTS.md' .
git diff --check main...HEAD
git diff --exit-code main...HEAD -- pyproject.toml uv.lock \
  .github/workflows/ci.yml docs/phase-1-acceptance.md \
  examples/notebooks/*.ipynb examples/scripts/*.py \
  src/cyexm/errors.py src/cyexm/io src/cyexm/metrics.py \
  src/cyexm/registration/*.py src/cyexm/runtime/*.py src/cyexm/spatial.py
```

Expected: exactly 18 instruction files; every Phase 2 statement is clearly proposed or
gated; the protected production/dependency/example/workflow diff is empty.

- [ ] **Step 9: Commit test guidance**

```bash
git add tests/AGENTS.md tests/integration/AGENTS.md tests/unit/AGENTS.md \
  tests/unit/test_repository_policy.py
git commit -m "docs: align test guidance for future execution"
```

- [ ] **Step 10: Verify final branch state**

Run:

```bash
git status --short
git log --oneline --decorate main..HEAD
```

Expected: empty status. History contains the approved design and plan, one
implementation commit for each of Tasks 1 through 5, and only documented preflight or
review-fix commits beyond that composition. Do not enforce an exact commit count.
