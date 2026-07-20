# Cy-ExM Phase 2C Local Dask Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute independent real round tasks concurrently on a bounded process-based local Dask cluster while preserving identical task, artifact, failure, retry, cancellation, and result semantics.

**Architecture:** `LocalDaskExecutor` lazily imports distributed, owns a `LocalCluster` and `Client` for one invocation, gives each worker one `round_slot`, and submits every round with a matching resource request. Result collection preserves plan order while allowing independent futures to settle; cleanup occurs in `finally` for success, task failure, startup failure, and interruption.

**Tech Stack:** Python 3.12, Dask/distributed 2025.1.0, process-based LocalCluster, Phase 2B round tasks, pytest unit fakes and bounded integration tests

## Global Constraints

- Complete Phase 2A and Phase 2B first.
- Import Dask/distributed only inside executor-opening operations; importing `cyexm.execution.dask` must not load scheduler modules.
- Local Dask uses worker processes, not threads in the coordinator process.
- Every worker advertises `round_slot=1`; every submitted task requests `resources={"round_slot": 1}`.
- Bound workers to `min(pending_rounds, configured_workers)` and never create a cluster for zero pending rounds.
- Submit exactly one coarse task per round. Round tasks remain serial internally.
- Default retries are zero. A configured retry is an executor-managed resubmission that re-enters ordinary Phase 1 artifact validation; Dask's opaque future retry counter remains zero.
- A failed round does not cancel other round futures.
- Operator interruption cancels unfinished futures and closes both client and cluster.
- Never silently fall back to sequential execution.

---

## File and responsibility map

| Path | Responsibility |
| --- | --- |
| `src/cyexm/config/execution.py` | Require process-isolated local execution and valid local resource strings |
| `src/cyexm/execution/dask.py` | Lazy LocalCluster lifecycle, resource submission, collection, retry, cancellation |
| `src/cyexm/execution/__init__.py` | Accepted local executor export |
| `src/cyexm/workflow/api.py` | Select local executor from immutable profile |
| `tests/unit/test_execution_dask.py` | Fake module construction, arguments, ordering, and cleanup |
| `tests/unit/test_workflow_api.py` | Local executor selection and no-fallback behavior |
| `tests/integration/local_dask_helpers.py` | Picklable synthetic task runners for bounded integration tests |
| `tests/integration/test_local_dask.py` | Real LocalCluster concurrency, placement, retry, failure, and cleanup |
| `tests/integration/test_workflow_local_dask.py` | Sequential/local numerical and restart agreement |
| `tests/unit/test_import_contract.py` | Lazy scheduler import contract |
| `src/cyexm/execution/AGENTS.md` | Current local-Dask resource and lifecycle rules |

---

### Task 1: Tighten the local execution profile for process isolation

**Files:**
- Modify: `src/cyexm/config/execution.py`
- Modify: `tests/unit/test_config_execution.py`

**Interfaces:**
- Consumes: Phase 2A `LocalDaskSpec`
- Produces: a local-Dask profile safe for native scientific runtimes

- [ ] **Step 1: Add a failing thread-only profile test**

```python
def test_local_dask_requires_worker_processes(tmp_path: Path) -> None:
    path = write_local_profile(tmp_path, processes=False)
    with pytest.raises(
        ConfigurationValidationError,
        match=r"local_dask\.processes.*must be true",
    ):
        load_execution_profile(path)
```

Add focused cases requiring positive built-in `workers` and `threads_per_worker`, nonempty single-line `memory_limit` and `startup_timeout`, and non-negative built-in retries.

- [ ] **Step 2: Run the profile tests and observe acceptance of threads-only mode**

```bash
uv run pytest tests/unit/test_config_execution.py -q
```

Expected: the new `processes: false` case fails because the parser currently accepts it.

- [ ] **Step 3: Enforce process isolation**

After validating a built-in boolean, reject `False` with `ConfigurationValidationError` at `local_dask.processes`. Keep `threads_per_worker` configurable because it describes the Dask worker, while `NativeThreadSpec` independently controls ITK/BLAS.

- [ ] **Step 4: Run and commit the profile correction**

```bash
uv run pytest tests/unit/test_config_execution.py -q
git add src/cyexm/config/execution.py tests/unit/test_config_execution.py
git commit -m "fix: require process-based local Dask workers"
```

Expected: all execution-profile tests pass.

---

### Task 2: Construct and own a bounded LocalCluster lazily

**Files:**
- Create: `src/cyexm/execution/dask.py`
- Modify: `src/cyexm/execution/__init__.py`
- Create: `tests/unit/test_execution_dask.py`

**Interfaces:**
- Consumes: `ExecutionProfile`, `RoundTask`, `RoundResult`, `run_round_task`
- Produces: `LocalDaskExecutor(profile: ExecutionProfile, *, _runner: RoundTaskRunner = run_round_task).execute(tasks) -> tuple[RoundResult, ...]`

- [ ] **Step 1: Write fake-construction and submission tests**

```python
def test_local_executor_uses_one_round_slot_per_worker() -> None:
    modules = FakeDistributedModules()
    executor = LocalDaskExecutor(local_profile(workers=4), _modules=lambda: modules)
    results = executor.execute((task(2), task(3)))
    assert modules.cluster_kwargs == {
        "n_workers": 2,
        "threads_per_worker": 1,
        "processes": True,
        "memory_limit": "8GB",
        "local_directory": str(executor.profile.paths.worker_local_directory),
        "resources": {"round_slot": 1},
    }
    assert [call.resources for call in modules.submit_calls] == [
        {"round_slot": 1},
        {"round_slot": 1},
    ]
    assert tuple(result.round for result in results) == (2, 3)
```

The fake records `Client`, `LocalCluster`, `submit`, `as_completed`, `cancel`, and close calls without importing distributed.

- [ ] **Step 2: Add lifecycle, retry, key, and zero-task tests**

Cover worker bounding, deterministic keys `cyexm-<dataset>-round-<NNN>`, `pure=False`, Dask submission `retries=0`, executor-managed bounded resubmission for both failed results and raised future exceptions, startup timeout, no cluster for an empty task tuple, completion order differing from plan order, one exhausted failed round while others finish, client-close before cluster-close, startup failure cleanup, and cancellation of unfinished futures on `KeyboardInterrupt`.

```bash
uv run pytest tests/unit/test_execution_dask.py -q
```

Expected: collection fails because `LocalDaskExecutor` is absent.

- [ ] **Step 3: Implement lazy module loading and cluster construction**

Define a private `_distributed_modules()` that imports `Client`, `LocalCluster`, `as_completed`, and `parse_timedelta` inside the function. Construct only for a nonempty task tuple:

```python
cluster = LocalCluster(
    n_workers=min(len(tasks), spec.workers),
    threads_per_worker=spec.threads_per_worker,
    processes=True,
    memory_limit=spec.memory_limit,
    local_directory=str(profile.paths.worker_local_directory),
    resources={"round_slot": 1},
)
client = Client(cluster, timeout=spec.startup_timeout)
client.wait_for_workers(
    min(len(tasks), spec.workers),
    timeout=parse_timedelta(spec.startup_timeout),
)
```

Wrap construction errors in `ExecutorStartupError` with requested worker count and timeout.

- [ ] **Step 4: Implement resource submission and independent collection**

Store the top-level `_runner` callable supplied to the constructor so portable integration tests can submit picklable synthetic work while production defaults to `run_round_task`. Submit each task with `pure=False`, its deterministic key, `retries=0`, and `resources={"round_slot": 1}`. Track attempts per round. When a future raises or returns `status == "failed"`, resubmit the same immutable task while attempts remain under `1 + spec.task_retries`; each resubmission therefore performs normal artifact validation. After exhaustion, retain the final structured failure. Iterate `as_completed(futures, with_results=True, raise_errors=False)`, store terminal results by round, and return original task order. Do not cancel other rounds after one terminal failure.

- [ ] **Step 5: Implement interruption and deterministic cleanup**

On `KeyboardInterrupt`, call `client.cancel(unfinished, force=True)` and re-raise. In `finally`, close the client and owned cluster even if cancellation or close raises; preserve the primary exception and log cleanup errors. Never shut down a scheduler not owned by this executor.

- [ ] **Step 6: Run fake executor tests**

```bash
uv run pytest tests/unit/test_execution_dask.py -q
```

Expected: all construction, resource, ordering, retry, failure, cancellation, and cleanup tests pass.

- [ ] **Step 7: Commit the local executor**

```bash
git add src/cyexm/execution/dask.py src/cyexm/execution/__init__.py \
  tests/unit/test_execution_dask.py
git commit -m "feat: execute rounds on bounded local Dask"
```

---

### Task 3: Select local Dask through the public workflow API

**Files:**
- Modify: `src/cyexm/workflow/api.py`
- Modify: `tests/unit/test_workflow_api.py`
- Modify: `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: `LocalDaskExecutor`
- Produces: local-Dask behavior for the unchanged `run_workflow(plan)` and CLI run command

- [ ] **Step 1: Add executor-selection and no-fallback tests**

```python
def test_run_workflow_selects_local_dask_from_profile(
    monkeypatch: pytest.MonkeyPatch, local_plan: WorkflowPlan
) -> None:
    selected: list[str] = []
    monkeypatch.setattr(
        "cyexm.execution.dask.LocalDaskExecutor.execute",
        lambda self, tasks: selected.append(self.profile.executor)
        or tuple(successful_result(task.round) for task in tasks),
    )
    result = run_workflow(local_plan)
    assert selected == ["local_dask"]
    assert result.succeeded
```

Add a constructor/startup failure case that reaches CLI exit 3 and proves `SequentialExecutor.execute` was never called.

- [ ] **Step 2: Run focused API and CLI tests**

```bash
uv run pytest tests/unit/test_workflow_api.py tests/unit/test_cli.py -q
```

Expected: new local selection cases fail because the factory knows only sequential execution.

- [ ] **Step 3: Add lazy executor selection**

In the private executor factory, import `LocalDaskExecutor` only under `profile.executor == "local_dask"`. Keep sequential selection unchanged. Leave `slurm` as an explicit `ExecutorStartupError("SLURM execution is not available in this installed milestone")` until Phase 2D; do not substitute another executor.

- [ ] **Step 4: Run and commit API selection**

```bash
uv run pytest tests/unit/test_workflow_api.py tests/unit/test_cli.py -q
git add src/cyexm/workflow/api.py tests/unit/test_workflow_api.py tests/unit/test_cli.py
git commit -m "feat: select local Dask workflow execution"
```

Expected: local selection and no-fallback tests pass.

---

### Task 4: Prove real LocalCluster placement, failure, retry, and cleanup

**Files:**
- Create: `tests/integration/local_dask_helpers.py`
- Create: `tests/integration/test_local_dask.py`

**Interfaces:**
- Consumes: process-based `LocalDaskExecutor`
- Produces: bounded real-distributed evidence independent of ANTSpyX registration cost

- [ ] **Step 1: Add picklable synthetic helpers**

Define module-level helpers that accept a `RoundTask`, record worker address, hostname, PID, start, and end into JSON under the task output, and return `RoundResult`. A retry helper uses atomic `open(path, "x")`: the first attempt raises and the second returns success. A failure helper raises for one configured round and succeeds for all others.

- [ ] **Step 2: Write the one-slot concurrency test**

Use two workers and four tasks. Assert exactly two distinct worker addresses/PIDs, no worker reports overlapping task intervals, at least one interval overlaps across different workers, and returned results follow round order even when completion order differs.

- [ ] **Step 3: Write bounded retry and independent-failure tests**

With retries zero, assert a failing round is reported once and other rounds finish. With retries one, assert the marker helper has exactly two attempts and succeeds. After each invocation, assert no child process owned by the created cluster remains and worker scratch has no active Dask lock/process directories.

- [ ] **Step 4: Run the real LocalCluster suite**

```bash
uv run pytest tests/integration/test_local_dask.py -v
```

Expected: all process placement, one-slot, result-order, failure, retry, and cleanup cases pass without external scheduler access.

- [ ] **Step 5: Commit bounded LocalCluster evidence**

```bash
git add tests/integration/local_dask_helpers.py tests/integration/test_local_dask.py
git commit -m "test: prove bounded local Dask execution"
```

---

### Task 5: Compare sequential and local-Dask scientific outputs

**Files:**
- Create: `tests/integration/test_workflow_local_dask.py`

**Interfaces:**
- Consumes: real Phase 2B round task, sequential executor, local executor
- Produces: numerical and restart equivalence across executor modes

- [ ] **Step 1: Write a two-round synthetic workflow comparison**

Build one whole-image and one serial-chunked round using `dry run` accuracy and the existing bounded fixtures. Run once to a sequential output root and once to a local-Dask output root with two workers. Compare affine/warp existence and every channel TIFF's dtype, shape, geometry, and values.

- [ ] **Step 2: Add numerical tolerance and restart assertions**

For estimated floating values, use `rtol=1e-4`, `atol=1e-5` with a comment identifying the established ANTSpyX migration tolerance. Rerun the local workflow and assert all results are `reused` and file identities are unchanged.

- [ ] **Step 3: Run the scientific local-Dask comparison**

```bash
uv run pytest tests/integration/test_workflow_local_dask.py -v
```

Expected: sequential/local outputs agree, both rounds occupy distinct worker processes when concurrent, and the second run reuses every authoritative artifact.

- [ ] **Step 4: Commit executor-equivalence evidence**

```bash
git add tests/integration/test_workflow_local_dask.py
git commit -m "test: compare local Dask and sequential workflows"
```

---

### Task 6: Close the Phase 2C import, guidance, and portable gates

**Files:**
- Modify: `tests/unit/test_import_contract.py`
- Modify: `src/cyexm/execution/AGENTS.md`

**Interfaces:**
- Consumes: accepted local-Dask behavior
- Produces: current executor guidance and import-side-effect regression coverage

- [ ] **Step 1: Add lazy scheduler import assertions**

```python
def test_import_local_dask_executor_does_not_load_distributed() -> None:
    code = """
import sys
import cyexm.execution.dask
for root in ("dask", "distributed"):
    assert not any(name == root or name.startswith(root + ".") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
```

- [ ] **Step 2: Update executor guidance to current local behavior**

Record process isolation, one `round_slot`, bounded worker count, no nested tasks, zero default retries, independent failures, and client/cluster cleanup. Keep SLURM statements explicitly proposed until Phase 2D.

- [ ] **Step 3: Run the complete portable gate**

```bash
git diff --check
uv lock --check
uv run pytest -m "unit and not hpc" tests/unit -v
uv run pytest -m "integration and not hpc" tests/integration -v
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pre-commit run --all-files
```

Expected: every command exits 0 and real LocalCluster tests remain within the portable five-minute integration budget.

- [ ] **Step 4: Commit guidance and import checks**

```bash
git add tests/unit/test_import_contract.py src/cyexm/execution/AGENTS.md
git commit -m "docs: define local Dask execution boundaries"
```

- [ ] **Step 5: Review the milestone diff**

```bash
git log --oneline --decorate main..HEAD
git diff --stat HEAD~6..HEAD
git status --short
```

Expected: six focused Phase 2C commits and a clean worktree.
