# SLURM Memory Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Convert strict decimal SLURM profile memory into an explicit ceiling binary
scheduler/Dask quantity and prove the exact request through portable and real-HPC
acceptance evidence.

**Architecture:** A dependency-light configuration helper is the single parser and
normalizer. `SlurmExecutor` uses its scheduler and Dask spellings for construction,
the production rendered-template validator returns the exact worker byte limit, and
the acceptance harness compares worker, TRES, partition, and node facts against the
recorded semantic and normalized values.

**Tech Stack:** Python 3.12 standard library, frozen dataclasses, pytest, existing
dask-jobqueue fake boundary, strict JSON operational evidence.

## Global Constraints

- `500GB` means exactly 500,000,000,000 semantic bytes.
- Its ceiling binary scheduler quantity is `466G`, exactly 500,363,689,984 bytes.
- Dask receives `466GiB`; `#SBATCH` receives `466G` and never a `GB` suffix.
- Reject unsupported syntax and any parse or ceiling result above `2**63 - 1` bytes.
- Do not run `pytest -m hpc`, a scheduler client, an approved dataset, ClearEx, or the
  live spheroid workflow.
- Use the required `/project` uv cache and Python roots and set all six native-thread
  variables to one for broad gates.

---

### Task 1: Central strict memory quantity

**Files:**
- Create: `src/cyexm/config/memory.py`
- Modify: `src/cyexm/config/execution.py`
- Modify: `src/cyexm/config/__init__.py`
- Test: `tests/unit/test_config_memory.py`
- Test: `tests/unit/test_config_execution.py`
- Test: `tests/unit/test_phase2_examples.py`

**Interfaces:**
- Produces: `SlurmMemoryQuantity(profile, semantic_bytes, scheduler,
  scheduler_bytes, dask)` and `normalize_slurm_memory(value, location)`.
- Consumes: `ConfigurationValidationError` and the existing immutable `SlurmSpec`.

- [x] **Step 1: Write failing parser and current-profile tests**

```python
def test_approved_decimal_memory_rounds_up_to_binary_gib() -> None:
    assert normalize_slurm_memory("500GB") == SlurmMemoryQuantity(
        profile="500GB",
        semantic_bytes=500_000_000_000,
        scheduler="466G",
        scheduler_bytes=500_363_689_984,
        dask="466GiB",
    )
```

Add parameterized invalid cases for zero, negative, fractional, lowercase,
whitespace, unsupported `G`/`GiB`, non-strings, semantic overflow, and ceiling
overflow. Require strict YAML loading and direct-profile revalidation to use the same
parser, and require the checked-in `500GB` example to produce the quantity above.

- [x] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest -q tests/unit/test_config_memory.py tests/unit/test_config_execution.py tests/unit/test_phase2_examples.py`

Expected: collection/import failures for the absent memory interface and profile
validation failures showing unsupported quantities still pass.

- [x] **Step 3: Implement the minimal dependency-light normalizer**

Use an exact `([1-9][0-9]*)(KB|MB|GB|TB)` full match, decimal multipliers, integer
ceiling to whole `G` binary units, and explicit signed-64-bit checks before
multiplication and after ceiling. Keep the profile spelling immutable; expose the
pure helper from `cyexm.config`; call it from `_slurm` so direct revalidation shares
the same contract.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass.

### Task 2: Exact construction, template validation, and evidence

**Files:**
- Modify: `src/cyexm/execution/slurm.py`
- Modify: `tests/unit/test_execution_slurm.py`
- Modify: `tests/unit/test_workflow_api.py`

**Interfaces:**
- Consumes: `normalize_slurm_memory(spec.memory)`.
- Produces: cluster `memory="466GiB"`, `job_mem="466G"`; exact requested-memory
  evidence; `validate_slurm_job_script(...) -> int` returning rendered worker bytes.

- [x] **Step 1: Write failing construction, rendering, and evidence tests**

Require the fake cluster kwargs to contain `memory="466GiB"` and `job_mem="466G"`,
the rendered script to contain `#SBATCH --mem=466G` and dask-jobqueue's
`--memory-limit 466.00GiB`, and the validator to return 500,363,689,984. Require
requested evidence to contain this exact object:

```python
{
    "profile": "500GB",
    "semantic_bytes": 500_000_000_000,
    "scheduler": "466G",
    "scheduler_bytes": 500_363_689_984,
    "dask": "466GiB",
}
```

Add fail-closed evidence mismatches and rendered worker values below, above, or
syntactically different from the normalized byte count.

- [x] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest -q tests/unit/test_execution_slurm.py tests/unit/test_workflow_api.py`

Expected: assertions show the current `500GB` construction/header and absent memory
evidence, while the validator returns `None`.

- [x] **Step 3: Implement normalized construction and validation**

Normalize once after strict profile validation. Add the exact memory object to
nonzero requested evidence and validate it by recomputing the quantity. Pass its
`dask` spelling as `memory` and `scheduler` spelling as `job_mem`. Parse the one
rendered `--memory-limit` value with Dask's lazy-loaded `parse_bytes`, require it to
equal `scheduler_bytes`, and return that integer before scaling.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass with zero unsafe scale or
submission calls.

### Task 3: Acceptance comparisons and repository contract

**Files:**
- Modify: `tests/hpc/environment.py`
- Modify: `tests/hpc/test_slurm_environment.py`
- Modify: `tests/unit/test_hpc_environment.py`
- Modify: `README.md`
- Modify: `src/cyexm/config/AGENTS.md`
- Modify: `src/cyexm/execution/AGENTS.md`
- Modify: `tests/hpc/AGENTS.md`
- Modify: `tests/unit/test_phase2_documentation.py`
- Modify: `tests/unit/test_repository_policy.py`
- Modify: `.superpowers/sdd/phase2e-task-4-report.md` (ignored report)

**Interfaces:**
- Consumes: requested memory evidence and the worker byte count returned by
  production template validation.
- Produces: exact worker-probe comparison and scheduler/node capacity comparisons
  against 500,363,689,984 normalized bytes.

- [x] **Step 1: Write failing acceptance and documentation tests**

Update realistic scripts to `--mem=466G` and `--memory-limit 466.00GiB`. Require
worker smoke records and probes to equal the rendered worker bytes, `ReqTRES` and
`AllocTRES` to cover scheduler bytes, and partition/node MiB facts converted to bytes
to cover the request. Require README and directory guidance to distinguish semantic
decimal memory from ceiling binary scheduler/Dask memory.

- [x] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest -q tests/unit/test_hpc_environment.py tests/unit/test_phase2_documentation.py tests/unit/test_repository_policy.py`

Expected: acceptance expectations and documentation fragments fail against the old
500GB header/semantic-only comparisons.

- [x] **Step 3: Implement acceptance and documentation contract**

Return worker bytes from the acceptance validator wrapper; remove the duplicated
acceptance `_memory_bytes`; derive quantities through production normalization and
template validation; use byte-safe ceiling comparisons for partition, TRES, and node
facts. Update README, inherited `AGENTS.md` guidance, and the ignored implementation
report with RED/GREEN and gate evidence.

- [x] **Step 4: Run focused tests and verify GREEN**

Run all Task 1-3 focused modules. Expected: all pass without an HPC selector.

### Task 4: Full verification and implementation commit

**Files:**
- Verify: entire repository

**Interfaces:**
- Consumes: all three task deliverables.
- Produces: one detailed implementation commit and a clean worktree.

- [x] **Step 1: Run every portable gate**

Run lock check and locked sync,  `pytest -m "unit and not hpc" tests/unit -q`,
`pytest -m "integration and not hpc" tests/integration -q`, Ruff check, Ruff format
check, BasedPyright, all-files pre-commit, and `git diff --check` with the required uv
and six thread environment variables.

- [x] **Step 2: Audit the exact staged diff**

Verify no external evidence, dataset content, scheduler output, credentials, lock
changes, or unrelated user edits are staged.

- [x] **Step 3: Commit**

```bash
git commit -m "fix: normalize SLURM memory requests"
```

Report the exact commit SHA, portable gate counts, and explicit confirmation that no
real selector, scheduler, dataset, ClearEx, or live workflow access occurred.
