# Phase 2E Reference Dispatch Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the modern shared-reference barrier behave correctly in workflow failure reporting and in the real-SLURM Phase 2E acceptance harness.

**Architecture:** Every executor may return only the failed `ReferenceTask` result because moving rounds are deliberately not submitted across a failed hard barrier. The workflow API must accept that one-result terminal shape and write the ordinary failed run record. Acceptance runners must dispatch the reference prerequisite through `run_workflow_task` while retaining their injected behavior only for moving `RoundTask` values.

**Tech Stack:** Python 3.12, pytest, Dask/distributed 2025.1.0, dask-jobqueue 0.9.0, SLURM

## Global Constraints

- Preserve the hard reference barrier: no moving task may be submitted after reference failure.
- Preserve exact task/result validation and Phase 1 completion records as the only restart authority.
- Keep portable tests scheduler-free; only the explicit `-m hpc` rerun may contact SLURM.
- Do not access or modify the active spheroid run.
- Use the shared `/project` uv cache and Python installation.

---

### Task 1: Report a failed reference barrier without requiring unsubmitted descendants

**Files:**
- Modify: `tests/unit/test_workflow_api.py`
- Modify: `src/cyexm/workflow/api.py`

**Interfaces:**
- Consumes: `WorkflowExecutor.execute(reference_task, round_tasks) -> tuple[RoundResult, ...]`
- Produces: `run_workflow(...)` raising `WorkflowExecutionError(failed_rounds=(reference_round,))` with a persisted failed run record when the executor returns one failed reference result.

- [x] **Step 1: Write a failing unit regression**

Add a fake executor that returns `(_reference_result(reference_task, failed=True),)` while moving rounds exist. Assert `run_workflow(plan)` raises `WorkflowExecutionError`, `failed_rounds == (1,)`, and the run record contains only the failed `reference-preparation` result.

- [x] **Step 2: Run the regression and observe the structural error**

Run:

```bash
uv run pytest tests/unit/test_workflow_api.py::test_reference_failure_records_only_the_unsubmitted_barrier -q
```

Expected: FAIL because `_ordered_results` currently requires reference and every unsubmitted moving round.

- [x] **Step 3: Implement the minimal terminal-result selection**

In `run_workflow`, validate the executor result tuple as either the full submitted sequence or exactly one failed reference result. Continue through ordinary result/run-record construction for the latter; reject every other partial result shape.

- [x] **Step 4: Run targeted workflow tests**

```bash
uv run pytest tests/unit/test_workflow_api.py -q
```

Expected: all workflow API tests pass.

---

### Task 2: Dispatch reference preparation through every Phase 2E acceptance runner

**Files:**
- Modify: `tests/unit/test_hpc_workflow_acceptance.py`
- Modify: `tests/hpc/workflow_acceptance.py`
- Modify: `tests/hpc/test_slurm_workflow.py`

**Interfaces:**
- Consumes: `WorkflowTask = ReferenceTask | RoundTask` and `run_workflow_task(task) -> RoundResult`
- Produces: real, failure-injection, cancellation, and resume runners that execute the reference prerequisite normally before applying moving-round-specific acceptance behavior.

- [x] **Step 1: Write failing runner-dispatch regressions**

Add unit cases passing a `ReferenceTask` to `run_failure_task` and `run_cancellation_task` with an injected workflow-task dispatcher. Assert each returns the dispatcher result and creates no moving-round coordination marker.

- [x] **Step 2: Run regressions and observe rejection of the reference task**

```bash
uv run pytest tests/unit/test_hpc_workflow_acceptance.py -q
```

Expected: FAIL because both helpers currently classify the reference round as unexpected.

- [x] **Step 3: Implement reference-aware dispatch**

Change helper inputs to `WorkflowTask`. When `type(task) is ReferenceTask`, return `_run_workflow_task(task)` immediately. Retain current injected behavior for exact `RoundTask` values. In `test_slurm_workflow.py`, replace every ordinary `run_round_task` acceptance runner with `run_workflow_task`.

- [x] **Step 4: Run targeted acceptance tests**

```bash
uv run pytest tests/unit/test_hpc_workflow_acceptance.py \
  tests/unit/test_workflow_api.py -q
```

Expected: all targeted tests pass without scheduler access.

- [x] **Step 5: Accept reference-inclusive placement evidence**

After the first corrected real-SLURM run completed all three parallel scientific
tasks, add a red/green regression requiring one completed `reference-preparation`
result plus the two completed moving strategies. Compute overlap and unique
job/hostname evidence only from the moving rounds.

---

### Task 3: Verify, integrate, and rerun exact-main acceptance

**Files:**
- External: approved Zenodo YAML, fresh output root, and fresh evidence directory

**Interfaces:**
- Consumes: corrected exact `main` commit and the modern-schema `8/8` execution profile
- Produces: current-main Phase 2E Task 5 evidence for placement, parity, failure isolation, cancellation, restart, and cleanup.

- [x] **Step 1: Run portable correction gates**

```bash
uv run pytest -m "unit and not hpc" tests/unit -q
uv run pytest -m "integration and not hpc" tests/integration -q
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
```

- [x] **Step 2: Review and merge the correction to main**

Confirm only the plan, focused production correction, harness correction, and regressions changed. Commit and merge locally so acceptance records the corrected exact `main` SHA.

- [ ] **Step 3: Run explicit acceptance with fresh caller-owned roots**

Run `uv run pytest -m hpc tests/hpc/test_slurm_workflow.py -v` with the modern Zenodo manifest, `8/8` SLURM profile, and a new empty evidence directory. Monitor at approximately five-minute intervals.

- [ ] **Step 4: Evaluate evidence**

Require completed workflow acceptance, scientific parity at `rtol=1e-4` and `atol=1e-5`, unique concurrent jobs/hosts, successful failure/cancellation resumes, and an empty exact-name `squeue` result before recommending Phase 2E Task 6.
