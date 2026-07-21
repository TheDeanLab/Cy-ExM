# Cy-ExM Phase 2D SLURM Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provision one Dask worker process per SLURM job, verify every allocated worker before registration, execute at most one round per worker/node, capture operational evidence, and clean up all owned scheduler resources.

**Architecture:** `SlurmExecutor` lazily constructs `dask_jobqueue.SLURMCluster`, scales it to the bounded pending-round count, waits for workers, and runs a standard-library-only probe on every worker. Only a fully successful preflight permits round submission. Each worker and task carries one `round_slot`; results reuse the local-Dask collection policy while SLURM-specific job scripts, allocations, probes, and cleanup remain execution evidence rather than restart state.

**Tech Stack:** Python 3.12, Dask/distributed 2025.1.0, dask-jobqueue 0.9.0, SLURMCluster, importlib.metadata, pathlib/tempfile, pytest fakes

## Global Constraints

- Complete Phase 2A through Phase 2C first.
- Import `dask_jobqueue`, Dask, and distributed only inside the selected SLURM execution operation.
- Use one worker process per SLURM job and require one `round_slot` per submitted round.
- Scale to `min(pending_rounds, configured_workers)`; do not request jobs for complete rounds.
- Treat queue, cores, memory, walltime, interface, account/QoS directives, and shared mounts as profile values requiring fresh acceptance evidence.
- Do not hard-code the observed `512GB` cluster facts into Python defaults.
- Run preflight on every connected worker before any registration future is submitted.
- A single failed probe aborts startup, closes client/cluster, and maps to exit code 3.
- A round failure after successful preflight does not cancel other submitted rounds. Configured retries are explicit executor resubmissions with Dask future retries fixed at zero.
- Workflow-run evidence never authorizes artifact reuse.
- Portable tests construct no real cluster and submit no SLURM job.

---

## File and responsibility map

| Path | Responsibility |
| --- | --- |
| `src/cyexm/execution/preflight.py` | Serializable worker probe and immutable evidence |
| `src/cyexm/execution/slurm.py` | Lazy SLURMCluster construction, scaling, preflight, submission, collection, cleanup |
| `src/cyexm/execution/__init__.py` | Accepted SLURM executor exports |
| `src/cyexm/workflow/models.py` | Execution evidence in workflow results/run records |
| `src/cyexm/workflow/api.py` | Select SLURM and persist operational evidence |
| `tests/unit/test_worker_preflight.py` | Version/path/scratch/thread/memory/network probe behavior |
| `tests/unit/test_execution_slurm.py` | Exact fake construction, script, resource, lifecycle, and cleanup |
| `tests/unit/test_workflow_api.py` | SLURM selection and evidence persistence |
| `tests/unit/test_cli.py` | Startup/preflight/round failure exit separation |
| `tests/unit/test_import_contract.py` | Lazy dask-jobqueue import contract |
| `src/cyexm/execution/AGENTS.md` | Current SLURM ownership and resource guidance |

---

### Task 1: Implement a serializable standard-library worker probe

**Files:**
- Create: `src/cyexm/execution/preflight.py`
- Create: `tests/unit/test_worker_preflight.py`

**Interfaces:**
- Consumes: resolved shared roots, scratch path, expected package versions, `NativeThreadSpec`, optional injected worker
- Produces: `WorkerProbe`; `probe_worker` with explicit dataset/output/scratch/version/thread keyword arguments returning `dict[str, object]`; `validate_worker_probes(probes: Mapping[str, Mapping[str, object]]) -> tuple[WorkerProbe, ...]`

- [ ] **Step 1: Write the successful probe test**

```python
def test_probe_records_worker_identity_paths_threads_and_memory(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    scratch = tmp_path / "scratch"
    dataset.mkdir()
    output.mkdir()
    probe = probe_worker(
        dataset_root=str(dataset),
        output_root=str(output),
        worker_local_directory=str(scratch),
        expected_versions={"cy-exm": "0.1.0"},
        expected_threads=dict.fromkeys(NATIVE_THREAD_ENV_VARS, "1"),
        _version=lambda name: "0.1.0",
        _worker=FakeWorker(address="tcp://10.0.0.1:1234", memory_limit=500_000_000_000),
    )
    assert probe["ok"] is True
    assert probe["worker_address"] == "tcp://10.0.0.1:1234"
    assert probe["memory_limit_bytes"] == 500_000_000_000
    assert not scratch.joinpath("cyexm-preflight").exists()
```

- [ ] **Step 2: Add focused probe failure cases**

Cover package mismatch/missing distribution, dataset unreadable, output unwritable, scratch create/write/delete failure, native-thread mismatch/missing variable, missing worker address, unusable memory limit, missing hostname, and absent/present `SLURM_JOB_ID`. Test `validate_worker_probes` aggregates every worker's failures and includes addresses/hostnames/job IDs.

```bash
uv run pytest tests/unit/test_worker_preflight.py -q
```

Expected: collection fails because `preflight.py` is absent.

- [ ] **Step 3: Define immutable probe evidence**

```python
@dataclass(frozen=True, slots=True)
class WorkerProbe:
    ok: bool
    worker_address: str
    hostname: str
    pid: int
    slurm_job_id: str | None
    package_versions: tuple[tuple[str, str], ...]
    native_threads: tuple[tuple[str, str], ...]
    dataset_visible: bool
    output_writable: bool
    scratch_usable: bool
    memory_limit_bytes: int
    network_host: str
    errors: tuple[str, ...]
```

Provide strict `from_dict` and `to_dict` methods using JSON-compatible primitives.

- [ ] **Step 4: Implement the worker probe without scientific imports**

Use `importlib.metadata.version`, `socket.gethostname`, `os.getpid`, `os.environ`, `Path`, and `tempfile.NamedTemporaryFile`. Create and delete a unique scratch directory and a unique output-root probe file in `finally`. Read `dask_worker.address` and `dask_worker.memory_manager.memory_limit` from the injected worker object. Return all failures in `errors` rather than raising inside the remote worker.

- [ ] **Step 5: Run and commit preflight tests**

```bash
uv run pytest tests/unit/test_worker_preflight.py -q
git add src/cyexm/execution/preflight.py tests/unit/test_worker_preflight.py
git commit -m "feat: probe SLURM worker environments"
```

Expected: successful evidence and every aggregated failure case pass.

---

### Task 2: Construct, scale, and close SLURMCluster exactly

**Files:**
- Create: `src/cyexm/execution/slurm.py`
- Modify: `src/cyexm/execution/__init__.py`
- Create: `tests/unit/test_execution_slurm.py`

**Interfaces:**
- Consumes: `SlurmSpec`, Phase 2C result-collection policy, `probe_worker`, `validate_worker_probes`
- Produces: `SlurmExecutor(profile, *, _runner: RoundTaskRunner = run_round_task).execute(tasks) -> tuple[RoundResult, ...]`; `SlurmExecutor.execution_evidence -> Mapping[str, object]`

- [ ] **Step 1: Write exact SLURMCluster construction tests**

```python
def test_slurm_executor_builds_one_process_jobs_with_round_resource() -> None:
    modules = FakeSlurmModules()
    executor = SlurmExecutor(slurm_profile(), _modules=lambda: modules)
    executor.execute((task(2), task(3), task(4)))
    assert modules.cluster_kwargs == {
        "queue": "512GB",
        "cores": 72,
        "processes": 1,
        "memory": "500GB",
        "walltime": "72:00:00",
        "interface": "ib0",
        "scheduler_options": {"interface": "ib0"},
        "local_directory": "/project/cyexm-worker",
        "job_name": "cyexm-registration",
        "job_extra_directives": [],
        "worker_extra_args": ["--resources", "round_slot=1"],
    }
    assert modules.scale_calls == [{"jobs": 3}]
    assert modules.wait_for_workers_calls == [(3, 900.0)]
```

- [ ] **Step 2: Add construction, preflight, submission, and cleanup failures**

Cover worker bounding, zero pending rounds, exact job script capture, coordinator package versions, every worker probed, failed probe before submit, wait timeout, scale failure, client creation failure, round resources/keys, Dask `retries=0`, bounded executor resubmission, independent exhausted round failure, `KeyboardInterrupt` cancellation, client close, cluster close, and proof no sequential executor is invoked.

```bash
uv run pytest tests/unit/test_execution_slurm.py -q
```

Expected: collection fails because `SlurmExecutor` is absent.

- [ ] **Step 3: Implement lazy construction and bounded scaling**

Inside `_slurm_modules()`, import `SLURMCluster`, `Client`, `as_completed`, and `parse_timedelta`. Construct with exact profile values from the test, capture `cluster.job_script()` before scale, call `cluster.scale(jobs=worker_count)`, create the client, and call `client.wait_for_workers(worker_count, timeout=parse_timedelta(spec.startup_timeout))`.

- [ ] **Step 4: Run preflight before registration submission**

Build expected versions for `cy-exm`, `dask`, `distributed`, `dask-jobqueue`, `antspyx`, `numpy`, `scipy`, and `tifffile` from the coordinator. Call `client.run(probe_worker, ...)`, validate all returned mappings, and raise `WorkerPreflightError` containing the aggregate if any probe fails. No call to `client.submit` may precede successful validation.

- [ ] **Step 5: Submit one-slot rounds and collect independent results**

Store the constructor's top-level `_runner` callable, defaulting to `run_round_task`, so explicit HPC durability tests can submit controlled picklable failure/cancellation work. Use the same task key, `pure=False`, resource, ordered-result, and executor-managed bounded-resubmission policy as Local Dask, with every Dask submission using `retries=0`. Store submitted job script, requested workers, scheduler address, worker probes, attempt counts, start/end timestamps, and cleanup status in `execution_evidence`.

- [ ] **Step 6: Implement owned-resource cleanup**

On interruption, cancel unfinished futures with `force=True` and re-raise. In `finally`, close client then cluster. If startup fails after cluster creation, still close the cluster. Record cleanup exceptions without replacing the primary exception.

- [ ] **Step 7: Run and commit fake SLURM construction**

```bash
uv run pytest tests/unit/test_execution_slurm.py -q
git add src/cyexm/execution/slurm.py src/cyexm/execution/__init__.py \
  tests/unit/test_execution_slurm.py
git commit -m "feat: execute rounds through SLURMCluster"
```

Expected: all exact construction, preflight ordering, task resource, and lifecycle tests pass without job submission.

---

### Task 3: Persist SLURM operational evidence through the public API

**Files:**
- Modify: `src/cyexm/workflow/models.py`
- Modify: `src/cyexm/workflow/api.py`
- Modify: `tests/unit/test_workflow_api.py`
- Modify: `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: `SlurmExecutor.execution_evidence`
- Produces: SLURM selection and JSON-compatible workflow-run evidence without changing scientific fingerprint/reuse

- [ ] **Step 1: Add selection, evidence, and exit-code tests**

```python
def test_slurm_execution_evidence_is_operational_not_scientific(
    monkeypatch: pytest.MonkeyPatch, slurm_plan: WorkflowPlan
) -> None:
    fake = FakeExecutor(
        results=successful_results(slurm_plan),
        execution_evidence={"submitted_job_script": "#!/bin/bash\n#SBATCH -p 512GB\n"},
    )
    result = run_workflow(slurm_plan, _executor_factory=lambda profile: fake)
    run_record = json.loads(Path(result.run_record).read_text(encoding="utf-8"))
    assert run_record["execution"]["submitted_job_script"].startswith("#!/bin/bash")
    assert "submitted_job_script" not in slurm_plan.scientific_json
```

Add CLI exit 3 for construction/wait/preflight failures, exit 4 for post-preflight round failures, and exit 130 for interruption.

- [ ] **Step 2: Run focused API/CLI tests**

```bash
uv run pytest tests/unit/test_workflow_api.py tests/unit/test_cli.py -q
```

Expected: SLURM selection/evidence cases fail because API selection is not implemented.

- [ ] **Step 3: Add lazy SLURM selection and evidence serialization**

Import `SlurmExecutor` only under `profile.executor == "slurm"`. After executor completion, copy its JSON-compatible `execution_evidence` into the per-run operational record and encode it canonically into `WorkflowResult.execution_evidence_json`; never include it in `scientific-plan.json`, fingerprint input, or status reuse checks.

- [ ] **Step 4: Run and commit public SLURM behavior**

```bash
uv run pytest tests/unit/test_workflow_api.py tests/unit/test_cli.py -q
git add src/cyexm/workflow/models.py src/cyexm/workflow/api.py \
  tests/unit/test_workflow_api.py tests/unit/test_cli.py
git commit -m "feat: record SLURM workflow evidence"
```

Expected: selection, evidence ownership, and all exit-code distinctions pass.

---

### Task 4: Close the Phase 2D portable construction gate

**Files:**
- Modify: `tests/unit/test_import_contract.py`
- Modify: `src/cyexm/execution/AGENTS.md`

**Interfaces:**
- Consumes: accepted portable SLURM construction behavior
- Produces: import-side-effect regression protection and current SLURM execution guidance

- [ ] **Step 1: Add a fresh-process dask-jobqueue import test**

```python
def test_import_slurm_executor_does_not_load_scheduler_modules() -> None:
    code = """
import sys
import cyexm.execution.preflight
import cyexm.execution.slurm
for root in ("dask", "distributed", "dask_jobqueue"):
    assert not any(name == root or name.startswith(root + ".") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
```

- [ ] **Step 2: Update execution guidance to current SLURM behavior**

Record owned cluster/client lifecycle, one process and one round slot per job, all-worker preflight before registration, independent post-start failures, operator cancellation, and operational-only job/probe evidence. State that real submission remains unaccepted until Phase 2E.

- [ ] **Step 3: Run exact portable construction and full gates**

```bash
uv run pytest tests/unit/test_worker_preflight.py tests/unit/test_execution_slurm.py -v
git diff --check
uv lock --check
uv run pytest -m "unit and not hpc" tests/unit -v
uv run pytest -m "integration and not hpc" tests/integration -v
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pre-commit run --all-files
```

Expected: every command exits 0; no `sbatch`, `squeue`, or real external scheduler call occurs in portable tests.

- [ ] **Step 4: Commit import and guidance checks**

```bash
git add tests/unit/test_import_contract.py src/cyexm/execution/AGENTS.md
git commit -m "docs: define SLURM execution boundaries"
```

- [ ] **Step 5: Review the milestone diff**

```bash
git log --oneline --decorate main..HEAD
git diff --stat HEAD~4..HEAD
git status --short
```

Expected: four focused Phase 2D commits and a clean worktree. Do not label the runtime HPC-accepted yet.
