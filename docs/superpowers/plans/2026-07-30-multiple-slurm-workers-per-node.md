# Multiple SLURM Workers per Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permit several process-isolated Dask fine-task workers in each owned SLURM
job while preserving coarse one-process execution, scientific identity, restart
authority, and exact operational evidence.

**Architecture:** Extend the existing strict execution profile and owned
`dask_jobqueue.SLURMCluster` lifecycle. Derive one immutable job/process resource
model, scale in whole jobs, admit one task slot per validated process, and replace the
one-job/one-process identity rule with complete SLURM job groups. Keep the
coordinator-owned fine graph and Phase 1 artifact store unchanged.

**Tech Stack:** Python 3.12, frozen dataclasses, PyYAML, Dask Distributed 2025.1.0,
dask-jobqueue 0.9.0, SLURM, pytest, Ruff, BasedPyright, pre-commit, Jupyter Book,
ANTSpyX, NumPy, tifffile.

## Global Constraints

- Preserve execution-profile and scientific-manifest schema version 1.
- `slurm.workers` remains the maximum number of owned SLURM jobs/nodes.
- `slurm.processes_per_worker` is the number of Dask worker processes per job.
- Coarse mode must reject `processes_per_worker > 1`.
- Fine mode requires `cores_per_worker % processes_per_worker == 0`.
- ITK and BLAS thread counts must each be no greater than cores per process.
- Divide scheduler-normalized whole-GiB job memory conservatively and equally among
  processes; never let summed Dask process limits exceed the job request.
- One validated process advertises exactly one `round_slot`; one task consumes it.
- Scale only in whole jobs. Do not scale active jobs down before executor cleanup.
- Preserve the hard reference barrier and coordinator-owned fine graph.
- Do not add nested Dask graphs, nested multiprocessing, worker-owned submission,
  parallel affine estimation, or a new registration strategy.
- Do not change scientific fingerprints, task definitions, StageRequest identity,
  transforms, interpolation, masks, metrics, chunk geometry, or artifact filenames.
- The Phase 1 `ArtifactStore` remains the only restart authority. Execution evidence
  never authorizes scientific reuse.
- Keep Dask, distributed, dask-jobqueue, ANTs, NumPy, and SciPy imports deferred to
  their established operation boundaries.
- Do not edit historical completed designs/plans or the pinned ClearEx repository.
- Do not access, modify, terminate, or reuse the live spheroid registration output.
- Add both canonical spellings of
  `/archive/bioinformatics/Danuser_lab/Dean/dean/2026-07-28-feature-mask-cc-radius4`
  to the real-HPC protected-root set before accepting any new profile.
- Keep environments/caches on `/project`; before every `uv` command export:

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  ```

- Preserve unrelated working-tree changes. Stage only files named by the current
  task.
- Real-HPC tests require exact `-m hpc`, `--run-hpc`, caller-owned manifests,
  profiles, evidence roots, approved allocation, and exact named-job cleanup.
- Performance is characterization, not a pass/fail speedup threshold.

## File and responsibility map

- Create `src/cyexm/config/_slurm_resources.py`: one standard-library-only source of
  truth for per-job and per-process resource derivation.
- Modify `src/cyexm/config/execution.py`: strict cross-field validation for mode,
  process count, cores, native threads, and memory.
- Modify `src/cyexm/cli/_commands.py`: read-only derived resource reporting.
- Modify `src/cyexm/execution/preflight.py`: complete job-group identity and
  per-process resource validation.
- Modify `src/cyexm/execution/_slurm_evidence.py`: exact version-3 fine-SLURM group,
  scale, activation, replacement, and placement evidence.
- Modify `src/cyexm/execution/slurm.py`: render, validate, scale, preflight, activate,
  reconcile, and clean up multi-process jobs.
- Modify `tests/unit/test_config_execution.py`,
  `tests/unit/test_config_memory.py`, `tests/unit/test_cli.py`,
  `tests/unit/test_worker_preflight.py`, and
  `tests/unit/test_execution_slurm.py`: portable TDD coverage.
- Modify `tests/unit/test_workflow_api.py`: scientific-plan, fingerprint, run-record,
  and cache-identity invariance.
- Create `tests/hpc/multi_process_slurm_acceptance.py` and
  `tests/hpc/test_multi_process_slurm.py`: explicit one-versus-two-process real-SLURM
  scientific, placement, loss/replacement, cleanup, memory, and timing gate.
- Modify `tests/hpc/conftest.py`, `tests/hpc/environment.py`, and existing reusable
  acceptance helpers only where the new focused gate needs shared validation.
- Create `tests/unit/test_multi_process_slurm_acceptance.py`: fail-closed decoder and
  report tests without scheduler contact.
- Create `examples/execution-slurm-512gb-multi-process.yaml`: editable two-process
  fine profile.
- Modify `README.md`, `docs/workflow-guide.md`,
  `docs/fine-grained-execution.md`, `docs/registration-artifacts.md`,
  `tests/unit/test_phase2_examples.py`,
  `tests/unit/test_phase2_documentation.py`: maintained user documentation and
  executable documentation contracts.
- Modify applicable repository guidance:
  `AGENTS.md`, `src/cyexm/config/AGENTS.md`,
  `src/cyexm/execution/AGENTS.md`, `src/cyexm/runtime/AGENTS.md`,
  `examples/AGENTS.md`, `docs/AGENTS.md`, `tests/unit/AGENTS.md`,
  `tests/integration/AGENTS.md`, and `tests/hpc/AGENTS.md`.

## Reuse analysis

- Closest existing configuration symbols: `SlurmSpec`,
  `_validate_execution_profile`, and `normalize_slurm_memory`.
- Decision: retain `SlurmSpec.processes_per_worker` and add one internal derived
  resource value object. Do not add a second SLURM profile, schema version, or public
  synonym such as `workers_per_node`.
- Closest existing resource owner: `SlurmExecutor`.
- Decision: extend its current
  `SLURMCluster(processes=spec.processes_per_worker)` construction, reference
  barrier, dynamic scale loop, preflight, replacement, and cleanup. Do not create a
  parallel executor or custom job-script launcher.
- Closest existing identity gate: `validate_slurm_worker_identities`.
- Decision: extend it to return strict job groups while retaining one-process behavior
  as group size one. Do not build an acceptance-only identity policy.
- Closest existing task scheduler: `FineGraphController`.
- Decision: continue supplying eligible process-slot capacity to it. Job arithmetic
  stays in `SlurmExecutor`; task readiness/fairness remains in the controller.
- Closest existing evidence model: `FineScaleEvent`, `FineSlurmEvidence`, and
  `validate_fine_slurm_evidence`.
- Decision: advance all fine-SLURM evidence to exact version 3. Do not maintain
  separate single-process and multi-process evidence branches.
- Closest existing HPC machinery: `workflow_acceptance.py`,
  `fine_workflow_acceptance.py`, and `fine_acceptance_manifest`.
- Decision: compose those owned-submission, profiling, scientific comparison,
  worker-loss, and cleanup helpers in a new focused gate. Do not duplicate job
  cancellation or queue-ownership logic.
- Documentation updates: synchronize `SlurmExecutor` Numpydoc, README resource
  tables, maintained workflow/failure/provenance pages, example YAML, and applicable
  `AGENTS.md` contracts.

---

### Task 1: Derive and validate fine per-job/per-process resources

**Files:**
- Create: `src/cyexm/config/_slurm_resources.py`
- Modify: `src/cyexm/config/execution.py`
- Test: `tests/unit/test_config_execution.py`
- Test: `tests/unit/test_config_memory.py`

**Interfaces:**
- Consumes: `SlurmSpec`, `NativeThreadSpec`, `ParallelismSpec`,
  `SlurmMemoryQuantity`, `normalize_slurm_memory`.
- Produces:

  ```python
  @dataclass(frozen=True, slots=True)
  class SlurmJobResources:
      job_cap: int
      processes_per_job: int
      cores_per_job: int
      cores_per_process: int
      task_slots_per_job: int
      max_task_slots: int
      job_memory: SlurmMemoryQuantity
      process_memory_gib: int
      process_memory_bytes: int
      process_memory: str
      dask_managed_memory: str

  ```

- `derive_slurm_job_resources(spec: SlurmSpec, native_threads:
  NativeThreadSpec, parallelism: ParallelismSpec) -> SlurmJobResources` is the exact
  constructor/validator signature.
- Later tasks use this exact value object for CLI output, rendered-template
  validation, runtime evidence, and scheduler counts.

- [ ] **Step 1: Replace the one-only parser test with failing cross-field tests**

  Add tests with exact accepted values:

  ```python
  def test_fine_slurm_accepts_two_processes_with_divided_resources(
      tmp_path: Path,
  ) -> None:
      data = _valid_data("slurm")
      data["parallelism"] = {"mode": "fine_grained"}
      data["native_threads"] = {"itk": 15, "blas": 15}
      slurm = _mapping_at(data, "slurm")
      slurm["cores_per_worker"] = 30
      slurm["processes_per_worker"] = 2

      profile = _load_profile(tmp_path, data)

      assert profile.slurm is not None
      resources = derive_slurm_job_resources(
          profile.slurm, profile.native_threads, profile.parallelism
      )
      assert resources == SlurmJobResources(
          job_cap=3,
          processes_per_job=2,
          cores_per_job=30,
          cores_per_process=15,
          task_slots_per_job=2,
          max_task_slots=6,
          job_memory=normalize_slurm_memory("500GB"),
          process_memory_gib=233,
          process_memory_bytes=233 * 1024**3,
          process_memory="233GiB",
          dask_managed_memory="466GiB",
      )
  ```

  Add parameterized rejection cases:

  ```python
  @pytest.mark.parametrize(
      ("mode", "cores", "processes", "itk", "blas", "memory", "message"),
      [
          ("coarse", 30, 2, 15, 15, "500GB",
           "coarse SLURM execution requires processes_per_worker == 1"),
          ("fine_grained", 31, 2, 15, 15, "500GB",
           "cores_per_worker must be evenly divisible"),
          ("fine_grained", 30, 2, 16, 15, "500GB",
           "native_threads.itk must not exceed 15 cores per process"),
          ("fine_grained", 30, 2, 15, 16, "500GB",
           "native_threads.blas must not exceed 15 cores per process"),
          ("fine_grained", 30, 2, 1, 1, "1KB",
           "per-process Dask memory must be at least 1GiB"),
      ],
  )
  def test_slurm_rejects_invalid_process_resource_combinations(
      tmp_path: Path,
      mode: str,
      cores: int,
      processes: int,
      itk: int,
      blas: int,
      memory: str,
      message: str,
  ) -> None:
      data = _valid_data("slurm")
      data["parallelism"] = {"mode": mode}
      data["native_threads"] = {"itk": itk, "blas": blas}
      slurm = _mapping_at(data, "slurm")
      slurm["cores_per_worker"] = cores
      slurm["processes_per_worker"] = processes
      slurm["memory"] = memory

      with pytest.raises(ConfigurationValidationError, match=message):
          _load_profile(tmp_path, data)
  ```

  Retain built-in-integer rejection for `True`, `0`, `1.0`, and negative values.

- [ ] **Step 2: Run the focused tests and verify red**

  Run:

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  uv run pytest -m "unit and not hpc" \
    tests/unit/test_config_execution.py tests/unit/test_config_memory.py -v
  ```

  Expected: failures because process counts other than one are rejected and the
  resource interface does not exist.

- [ ] **Step 3: Implement the immutable resource derivation**

  Use only standard-library arithmetic:

  ```python
  _BINARY_GIB = 1024**3

  @dataclass(frozen=True, slots=True)
  class SlurmJobResources:
      job_cap: int
      processes_per_job: int
      cores_per_job: int
      cores_per_process: int
      task_slots_per_job: int
      max_task_slots: int
      job_memory: SlurmMemoryQuantity
      process_memory_gib: int
      process_memory_bytes: int
      process_memory: str
      dask_managed_memory: str

  def derive_slurm_job_resources(spec, native_threads, parallelism):
      processes = spec.processes_per_worker
      if parallelism.mode == "coarse" and processes != 1:
          raise ConfigurationValidationError(
              "coarse SLURM execution requires processes_per_worker == 1"
          )
      if spec.cores_per_worker % processes:
          raise ConfigurationValidationError(
              "slurm.cores_per_worker must be evenly divisible by "
              "slurm.processes_per_worker"
          )
      cores_per_process = spec.cores_per_worker // processes
      for name, value in (
          ("native_threads.itk", native_threads.itk),
          ("native_threads.blas", native_threads.blas),
      ):
          if value > cores_per_process:
              raise ConfigurationValidationError(
                  f"{name} must not exceed {cores_per_process} cores per process"
              )
      memory = normalize_slurm_memory(spec.memory)
      normalized_gib = memory.scheduler_bytes // _BINARY_GIB
      process_gib = normalized_gib // processes
      if process_gib < 1:
          raise ConfigurationValidationError(
              "per-process Dask memory must be at least 1GiB"
          )
      return SlurmJobResources(
          job_cap=spec.workers,
          processes_per_job=processes,
          cores_per_job=spec.cores_per_worker,
          cores_per_process=cores_per_process,
          task_slots_per_job=processes,
          max_task_slots=spec.workers * processes,
          job_memory=memory,
          process_memory_gib=process_gib,
          process_memory_bytes=process_gib * _BINARY_GIB,
          process_memory=f"{process_gib}GiB",
          dask_managed_memory=f"{process_gib * processes}GiB",
      )
  ```

  Parse `processes_per_worker` through
  `built_in_int(mapping["processes_per_worker"],
  "slurm.processes_per_worker", minimum=1)`, then call the helper at the end of
  SLURM `_validate_execution_profile`. This ensures YAML-loaded and directly
  constructed immutable profiles use the same gate.

- [ ] **Step 4: Run focused tests and static checks**

  Run the Step 2 command, then:

  ```bash
  uv run ruff check src/cyexm/config tests/unit/test_config_execution.py \
    tests/unit/test_config_memory.py
  uv run basedpyright src/cyexm/config tests/unit/test_config_execution.py \
    tests/unit/test_config_memory.py
  ```

  Expected: all pass.

- [ ] **Step 5: Commit Task 1**

  ```bash
  git add src/cyexm/config/_slurm_resources.py \
    src/cyexm/config/execution.py \
    tests/unit/test_config_execution.py tests/unit/test_config_memory.py
  git commit -m "feat: validate SLURM process resources"
  ```

  Commit body records that the existing `SlurmSpec` was extended and no alternate
  profile/resource abstraction was exposed publicly.

### Task 2: Report derived resources without changing scientific identity

**Files:**
- Modify: `src/cyexm/cli/_commands.py`
- Modify: `src/cyexm/workflow/api.py`
- Test: `tests/unit/test_cli.py`
- Test: `tests/unit/test_workflow_api.py`

**Interfaces:**
- Consumes:
  `derive_slurm_job_resources(spec, native_threads, parallelism)` from Task 1.
- Produces: `resource_request.slurm.derived` with exact keys:

  ```python
  {
      "job_cap": 3,
      "processes_per_job": 2,
      "task_slots_per_job": 2,
      "max_task_slots": 6,
      "cores_per_process": 15,
      "job_memory": "466G",
      "process_memory": "233GiB",
      "process_memory_bytes": 250181844992,
  }
  ```

- The persisted `execution_profile` mapping retains declared schema fields only.

- [ ] **Step 1: Write failing CLI and workflow identity tests**

  Construct two validated fine profiles that differ only in execution source/output
  paths and `processes_per_worker` (one versus two), while both use the same dataset
  manifest. Assert:

  ```python
  assert single_plan.scientific_json == multi_plan.scientific_json
  assert single_plan.scientific_fingerprint == multi_plan.scientific_fingerprint
  assert "derived" not in _execution_profile_mapping(multi_profile)["slurm"]
  ```

  Add a CLI plan test asserting the exact `derived` mapping above for 30 cores,
  two processes, 15/15 native threads, and 500 GB.

- [ ] **Step 2: Run tests and verify red**

  ```bash
  uv run pytest -m "unit and not hpc" \
    tests/unit/test_cli.py tests/unit/test_workflow_api.py -v
  ```

  Expected: the CLI lacks derived job/process facts.

- [ ] **Step 3: Add operational-only CLI mapping**

  Import `derive_slurm_job_resources` inside `_resource_request` only when a SLURM
  profile is present. Add a `_Parallelism` protocol with `mode: str` and add
  `parallelism: _Parallelism` to `_Execution`, so the helper receives the exact
  declared mode without importing scheduler libraries. Add declared fields through
  `_slurm_mapping` and place derived fields under one `derived` object. Do not add
  them to scientific JSON or `_execution_profile_mapping`.

- [ ] **Step 4: Run focused tests**

  Run the Step 2 command and:

  ```bash
  uv run ruff check src/cyexm/cli/_commands.py src/cyexm/workflow/api.py \
    tests/unit/test_cli.py tests/unit/test_workflow_api.py
  uv run basedpyright src/cyexm/cli/_commands.py src/cyexm/workflow/api.py
  ```

  Expected: all pass and both execution profiles have identical science.

- [ ] **Step 5: Commit Task 2**

  ```bash
  git add src/cyexm/cli/_commands.py src/cyexm/workflow/api.py \
    tests/unit/test_cli.py tests/unit/test_workflow_api.py
  git commit -m "feat: report SLURM process capacity"
  ```

### Task 3: Render and strictly validate multi-process SLURM jobs

**Files:**
- Modify: `src/cyexm/execution/slurm.py`
- Modify: `tests/unit/test_execution_slurm.py`

**Interfaces:**
- Consumes: `SlurmJobResources`.
- Produces:

  ```python
  def validate_slurm_job_script(
      script: str,
      profile: ExecutionProfile,
  ) -> int:
      """Return the exact validated per-process Dask memory bytes."""
  ```

- [ ] **Step 1: Write failing rendered-template tests**

  Extend `_profile` and `_FakeSlurmModules.SLURMCluster` to model the real
  dask-jobqueue command:

  ```text
  --nworkers 2 --nthreads 15 --memory-limit 233.00GiB
  ```

  Add tests asserting:

  ```python
  assert modules.cluster_kwargs["cores"] == 30
  assert modules.cluster_kwargs["processes"] == 2
  assert modules.cluster_kwargs["memory"] == "466GiB"
  assert modules.cluster_kwargs["job_cpu"] == 30
  assert modules.cluster_kwargs["job_mem"] == "466G"
  assert validate_slurm_job_script(script, profile) == 233 * 1024**3
  ```

  Add failures for missing/duplicate/wrong `--nworkers`, `--nthreads 30`,
  `--memory-limit 466GiB`, and user-injected managed `--nworkers`.

- [ ] **Step 2: Run exact rendered-script tests and verify red**

  ```bash
  uv run pytest -m "unit and not hpc" tests/unit/test_execution_slurm.py \
    -k "rendered or template or worker_memory or cluster_kwargs" -v
  ```

  Expected: failures at the current one-process assumptions.

- [ ] **Step 3: Use derived resources in construction and validation**

  In `SlurmExecutor.execute`, derive resources before module loading and pass:

  ```python
  cores=resources.cores_per_job
  processes=resources.processes_per_job
  memory=resources.dask_managed_memory
  job_cpu=resources.cores_per_job
  job_mem=resources.job_memory.scheduler
  ```

  In rendered validation require `--nthreads=cores_per_process`; require exactly one
  `--nworkers=P` only when `P > 1`; require it absent for `P == 1`; and require parsed
  `--memory-limit` to equal `process_memory_bytes`.

- [ ] **Step 4: Run rendered-script and regression tests**

  Run Step 2, then:

  ```bash
  uv run pytest -m "unit and not hpc" tests/unit/test_execution_slurm.py -v
  uv run ruff check src/cyexm/execution/slurm.py \
    tests/unit/test_execution_slurm.py
  uv run basedpyright src/cyexm/execution/slurm.py
  ```

  Expected: all SLURM unit tests pass for one-process and rendered two-process paths.

- [ ] **Step 5: Commit Task 3**

  ```bash
  git add src/cyexm/execution/slurm.py tests/unit/test_execution_slurm.py
  git commit -m "feat: render multi-process SLURM jobs"
  ```

### Task 4: Validate complete SLURM job groups

**Files:**
- Modify: `src/cyexm/execution/preflight.py`
- Modify: `tests/unit/test_worker_preflight.py`
- Modify: `tests/unit/test_execution_slurm.py`

**Interfaces:**
- Produces:

  ```python
  @dataclass(frozen=True, slots=True)
  class SlurmWorkerGroup:
      slurm_job_id: str
      hostname: str
      network_host: str
      probes: tuple[WorkerProbe, ...]

      @property
      def addresses(self) -> tuple[str, ...]:
          return tuple(probe.worker_address for probe in self.probes)
  ```

- `validate_slurm_worker_identities(probes: tuple[WorkerProbe, ...], *,
  processes_per_job: int = 1, expected_process_memory_bytes: int | None = None) ->
  tuple[SlurmWorkerGroup, ...]` is the exact validation signature.
- Groups are canonical by exact SLURM job ID string and worker address.

- [ ] **Step 1: Write failing group tests**

  Add a valid group with two probes sharing job ID `123`, hostname `node-a`, network
  host `10.0.0.1`, distinct addresses/PIDs, and 233-GiB limits. Assert one canonical
  group and both addresses.

  Parameterize failures for:

  - one probe when two are required;
  - three probes when two are required;
  - one job ID on `node-a` and `node-b`;
  - inconsistent network hosts;
  - duplicate `(hostname, PID)`;
  - duplicate worker address; and
  - one 466-GiB process in a 233-GiB group.

- [ ] **Step 2: Run preflight tests and verify red**

  ```bash
  uv run pytest -m "unit and not hpc" \
    tests/unit/test_worker_preflight.py -k "slurm_worker" -v
  ```

  Expected: duplicate job IDs are rejected instead of grouped.

- [ ] **Step 3: Implement group validation**

  Validate all probe payloads first. Group by non-null `slurm_job_id`; validate exact
  group size, one hostname/network host, unique address and host/PID globally, and
  exact expected memory when provided. Return immutable groups. Keep defaults at one
  process so coarse and existing callers retain strict behavior.

- [ ] **Step 4: Bind executor probe batches to group validation**

  Update `_validated_probe_batch` to receive `SlurmJobResources` and validate the
  current complete probe population with:

  ```python
  validate_slurm_worker_identities(
      tuple(combined_current_probes),
      processes_per_job=resources.processes_per_job,
      expected_process_memory_bytes=resources.process_memory_bytes,
  )
  ```

  Do not validate historical departed probes as a current group.

- [ ] **Step 5: Run focused and broad preflight tests**

  ```bash
  uv run pytest -m "unit and not hpc" \
    tests/unit/test_worker_preflight.py tests/unit/test_execution_slurm.py -v
  uv run ruff check src/cyexm/execution/preflight.py \
    tests/unit/test_worker_preflight.py tests/unit/test_execution_slurm.py
  uv run basedpyright src/cyexm/execution/preflight.py
  ```

  Expected: all pass.

- [ ] **Step 6: Commit Task 4**

  ```bash
  git add src/cyexm/execution/preflight.py \
    tests/unit/test_worker_preflight.py tests/unit/test_execution_slurm.py
  git commit -m "feat: validate SLURM worker groups"
  ```

### Task 5: Emit exact version-3 job/process provenance

**Files:**
- Modify: `src/cyexm/execution/_slurm_evidence.py`
- Modify: `src/cyexm/execution/slurm.py`
- Modify: `tests/unit/test_execution_slurm.py`

**Interfaces:**
- `FineScaleEvent` changes from ambiguous `requested` workers to:

  ```python
  @dataclass(slots=True)
  class FineScaleEvent:
      requested_jobs: int
      expected_processes: int
      started_at: str
      connected: list[str] = field(default_factory=list)
      departed: list[str] = field(default_factory=list)
      probed: list[str] = field(default_factory=list)
      activated: list[str] = field(default_factory=list)
      activated_probes: list[WorkerProbe] = field(default_factory=list)
      failed: list[dict[str, str]] = field(default_factory=list)
      ended_at: str | None = None
  ```

- `FineSlurmEvidence` constructor becomes:

  ```python
  FineSlurmEvidence(
      job_cap: int,
      processes_per_job: int,
      worker_cap: int,
  )
  ```

- Exact fine execution evidence uses `schema_version: 3`.
- `requested` evidence distinguishes:

  ```python
  {
      "jobs": current_requested_jobs,
      "configured_jobs": resources.job_cap,
      "processes_per_job": resources.processes_per_job,
      "task_slots_per_job": resources.task_slots_per_job,
      "max_task_slots": resources.max_task_slots,
      "cores_per_process": resources.cores_per_process,
      "process_memory_bytes": resources.process_memory_bytes,
      "startup_timeout": spec.startup_timeout,
      "task_retries": spec.task_retries,
      "memory": _memory_evidence(resources.job_memory),
  }
  ```

- [ ] **Step 1: Write failing strict evidence tests**

  Add direct model/validator tests asserting version 2 is rejected, version 3 is
  required, job/process keys are exact, `worker_cap == job_cap * processes_per_job`,
  each scale event satisfies
  `expected_processes == requested_jobs * processes_per_job`, and event job requests
  increase except replacement/health events.

  Add corruption cases for:

  - `configured_jobs=3`, `worker_cap=3`, `processes_per_job=2`;
  - requested jobs exceeding job cap;
  - expected processes exceeding slot cap;
  - activated probes that cannot form complete job groups;
  - same-job concurrent placements using the same PID; and
  - placement outside the authenticated activation interval.

- [ ] **Step 2: Run evidence tests and verify red**

  ```bash
  uv run pytest -m "unit and not hpc" tests/unit/test_execution_slurm.py \
    -k "evidence or scale_event or placement" -v
  ```

  Expected: current evidence is version 2 and assumes one job per process.

- [ ] **Step 3: Implement one unified version-3 evidence model**

  Update exact key sets, serializers, validators, error messages, scale events, and
  `SlurmExecutor._begin_evidence`. Define a fine-only exact requested-key set rather
  than weakening the existing coarse requested schema. Both one- and two-process
  fine profiles emit version 3. Coarse evidence retains its existing non-fine
  format. Historical version-2 JSON remains a historical file and is not accepted as
  current runtime evidence.

  Cross-check activated probe groups with Task 4's validator and derive explicit
  current job membership from authenticated probes. Accept shared job ID/hostname
  placement only when process identities differ and intervals overlap legitimately.

- [ ] **Step 4: Run exact evidence suite**

  ```bash
  uv run pytest -m "unit and not hpc" tests/unit/test_execution_slurm.py -v
  uv run ruff check src/cyexm/execution/_slurm_evidence.py \
    src/cyexm/execution/slurm.py tests/unit/test_execution_slurm.py
  uv run basedpyright src/cyexm/execution/_slurm_evidence.py \
    src/cyexm/execution/slurm.py
  ```

  Expected: all pass.

- [ ] **Step 5: Commit Task 5**

  ```bash
  git add src/cyexm/execution/_slurm_evidence.py \
    src/cyexm/execution/slurm.py tests/unit/test_execution_slurm.py
  git commit -m "feat: record SLURM job process groups"
  ```

### Task 6: Scale fine execution in whole jobs and process slots

**Files:**
- Modify: `src/cyexm/execution/slurm.py`
- Modify: `tests/unit/test_execution_slurm.py`

**Interfaces:**
- Consumes: `SlurmJobResources`, group validation, version-3 evidence.
- Produces `_desired_slurm_jobs(*, running_tasks: int, ready_tasks: int,
  resources: SlurmJobResources) -> int`, returning
  `max(1, min(job_cap, ceil((running + ready) / processes_per_job)))`.

- [ ] **Step 1: Make the fake scheduler model jobs and processes independently**

  Replace `connected_workers[:requested_jobs]` with job-group slicing. Store
  `address -> probe` and return all `requested_jobs * processes_per_job` addresses.
  Render two worker addresses/PIDs per fake job with one job ID/hostname.

- [ ] **Step 2: Write failing reference and scale tests**

  For `job_cap=3`, `processes_per_job=2`, assert:

  ```python
  assert modules.scale_calls[0] == {"jobs": 1}
  assert modules.wait_for_workers_calls[0] == (2, 900.0)
  assert initial_event["requested_jobs"] == 1
  assert initial_event["expected_processes"] == 2
  assert all_initial_processes_are_activated_before_reference_submission
  assert no_moving_submission_precedes_reference_success
  ```

  With five ready/running tasks, assert scale to three jobs, wait for six processes,
  activate four newcomers as two complete groups, and never call scale-down when
  demand later falls.

- [ ] **Step 3: Run scale tests and verify red**

  ```bash
  uv run pytest -m "unit and not hpc" tests/unit/test_execution_slurm.py \
    -k "fine and (scale or reference or activation)" -v
  ```

  Expected: startup waits for one process and demand is interpreted as job count.

- [ ] **Step 4: Implement process-count startup and job-count scaling**

  Start with `cluster.scale(jobs=1)`, wait for `processes_per_job`, preflight the
  complete initial group, activate every process, and submit only the reference.

  After the barrier, compute desired jobs with `_desired_slurm_jobs`; scale only when
  the target exceeds current requested jobs; wait for
  `desired_jobs * processes_per_job`; preflight newly complete groups; then activate
  their processes. Return `len(validated_workers)` as controller capacity.

- [ ] **Step 5: Run the SLURM unit suite**

  ```bash
  uv run pytest -m "unit and not hpc" tests/unit/test_execution_slurm.py -v
  uv run ruff check src/cyexm/execution/slurm.py \
    tests/unit/test_execution_slurm.py
  uv run basedpyright src/cyexm/execution/slurm.py
  ```

  Expected: all pass.

- [ ] **Step 6: Commit Task 6**

  ```bash
  git add src/cyexm/execution/slurm.py tests/unit/test_execution_slurm.py
  git commit -m "feat: scale fine SLURM process slots"
  ```

### Task 7: Reconcile one-process and whole-job loss safely

**Files:**
- Modify: `src/cyexm/execution/slurm.py`
- Modify: `src/cyexm/execution/_slurm_evidence.py`
- Modify: `tests/unit/test_execution_slurm.py`

**Interfaces:**
- Extends existing `wait_and_reconcile_workers`.
- Reconciliation target is
  `requested_jobs * processes_per_job` connected processes.
- Current healthy probes, not full historical probe history, define complete groups.

- [ ] **Step 1: Write failing one-process-loss tests**

  Start with two jobs/two processes each. Remove one address from job `101`, keep its
  sibling and both processes in job `102`, then inject a replacement address with job
  `101`, the same hostname/network host, and a new PID.

  Assert:

  - the lost slot is removed;
  - healthy siblings remain eligible;
  - wait target is four processes;
  - the new process is probed before activation;
  - job `101` is complete before the new slot activates;
  - the replacement event records one departure and one activation;
  - task retry/reschedule uses ordinary artifact validation; and
  - current groups end at exactly two processes each.

- [ ] **Step 2: Write failing whole-job and fail-closed tests**

  Remove both processes of job `101` and replace them with two processes under job
  `103`. Add failures for:

  - only one replacement process arriving before `startup_timeout`;
  - a replacement split across two job IDs;
  - a replacement job ID spanning two hostnames;
  - one wrong memory limit;
  - activation failure on the second process; and
  - connected process count exceeding `requested_jobs * processes_per_job`.

  Assert ordered `client`, `cluster`, `configuration` cleanup and no activation of an
  invalid or incomplete replacement group.

- [ ] **Step 3: Run replacement tests and verify red**

  ```bash
  uv run pytest -m "unit and not hpc" tests/unit/test_execution_slurm.py \
    -k "replacement or departure or worker_loss or reconcile" -v
  ```

  Expected: current one-process identity and wait arithmetic fail.

- [ ] **Step 4: Implement group-aware reconciliation**

  Track current validated probes separately from historical observed/activated
  probes. Remove departed addresses from eligible current state immediately. Wait
  under the existing single monotonic deadline for the exact process target. Probe
  the current population, validate complete groups, preserve unchanged healthy
  identities, activate only new identities, and record every transition before
  returning controller capacity.

  Do not extend the deadline across scheduler snapshot, arrival, probe, group
  validation, activation, or post-activation verification.

- [ ] **Step 5: Run replacement, evidence, and complete SLURM tests**

  ```bash
  uv run pytest -m "unit and not hpc" tests/unit/test_execution_slurm.py -v
  uv run ruff check src/cyexm/execution/slurm.py \
    src/cyexm/execution/_slurm_evidence.py \
    tests/unit/test_execution_slurm.py
  uv run basedpyright src/cyexm/execution/slurm.py \
    src/cyexm/execution/_slurm_evidence.py
  ```

  Expected: all pass.

- [ ] **Step 6: Commit Task 7**

  ```bash
  git add src/cyexm/execution/slurm.py \
    src/cyexm/execution/_slurm_evidence.py \
    tests/unit/test_execution_slurm.py
  git commit -m "feat: reconcile SLURM process groups"
  ```

### Task 8: Prove workflow and cache invariance

**Files:**
- Modify: `tests/unit/test_workflow_api.py`
- Modify: `tests/unit/test_execution_slurm.py`
- Modify: `tests/integration/test_workflow_local_dask.py`

**Interfaces:**
- Consumes the completed executor behavior.
- Produces regression evidence only; no production API.

- [ ] **Step 1: Add a scientific/cache invariance regression**

  Build the same serial-chunked manifest under validated one-process and two-process
  fine profiles. Assert exact equality of:

  - `scientific_json`;
  - `scientific_fingerprint`;
  - planned fine task keys and dependencies;
  - expected public artifact paths;
  - chunk `StageRequest` parameters; and
  - status/cache decisions after one profile commits artifacts and the other resumes.

  Assert the two run records differ only in operational execution profile/evidence
  fields and both preserve the same scientific fingerprint.

- [ ] **Step 2: Run the regression and confirm it passes**

  ```bash
  uv run pytest -m "unit and not hpc" \
    tests/unit/test_workflow_api.py tests/unit/test_execution_slurm.py -v
  uv run pytest -m "integration and not hpc" \
    tests/integration/test_workflow_local_dask.py -v
  ```

  Expected: all pass; no registration algorithm changes are required.

- [ ] **Step 3: Run scheduler-neutral fine regression coverage**

  ```bash
  uv run pytest -m "unit and not hpc" tests/unit -v
  uv run pytest -m "integration and not hpc" tests/integration -v
  ```

  Expected: all portable unit and integration tests pass.

- [ ] **Step 4: Commit Task 8**

  ```bash
  git add tests/unit/test_workflow_api.py \
    tests/unit/test_execution_slurm.py \
    tests/integration/test_workflow_local_dask.py
  git commit -m "test: preserve science across SLURM process counts"
  ```

### Task 9: Add focused real-HPC multi-process acceptance

**Files:**
- Create: `tests/hpc/multi_process_slurm_acceptance.py`
- Create: `tests/hpc/test_multi_process_slurm.py`
- Create: `tests/unit/test_multi_process_slurm_acceptance.py`
- Modify: `tests/hpc/conftest.py`
- Modify: `tests/hpc/environment.py`
- Modify only as reused: `tests/hpc/workflow_acceptance.py`
- Modify only as reused: `tests/hpc/fine_workflow_acceptance.py`

**Interfaces:**
- Adds exact pytest options:

  ```text
  --cyexm-single-process-execution PATH
  --cyexm-multi-process-execution PATH
  ```

- Produces `build_multi_process_acceptance_report(*, context:
  HpcAcceptanceContext, single_profile: ExecutionProfile, multi_profile:
  ExecutionProfile, single_result: WorkflowResult, multi_result: WorkflowResult,
  replacement_result: WorkflowResult, evidence_dir: Path) -> dict[str, object]`.

- Writes `multi-process-slurm-acceptance.json` only after checksum-binding raw
  profiles, execution evidence, placements, process-loss marker, outcome/queue
  records, scientific comparison, and performance/memory measurements.

- [ ] **Step 1: Write portable fail-closed profile tests**

  Extend `LIVE_SPHEROID_ROOTS` with the exact `/archive` and `/endosome/archive`
  spellings of the active 2026-07-28 feature-mask run before validating either
  profile.

  Test exact accepted pair:

  - both `fine_grained`;
  - same dataset root, exactly three `workers`, queue, 30 job cores, 500 GB,
    walltime, interfaces, exact `--exclusive`, at least one retry, and 15/15 native
    threads;
  - single profile has one process; multi profile has two;
  - output roots, scratch roots, source paths, and job names are distinct;
  - all protected/live roots remain disjoint.

  Reject every mismatch, coarse mode, multi count other than two, non-empty scenario
  root, missing option, profile aliasing, or non-archive output.

- [ ] **Step 2: Write portable report-decoder tests**

  Build synthetic raw evidence with two overlapping task placements that share job
  ID/hostname and use distinct worker addresses/PIDs. Assert accepted report fields:

  ```python
  assert report["schema_version"] == 1
  assert report["scientific_parity"]["accepted"] is True
  assert report["placement"]["same_job_distinct_process_overlap"] is True
  assert report["resources"]["processes_per_job"] == 2
  assert report["resources"]["cores_per_process"] == 15
  assert report["resources"]["process_memory_bytes"] == 233 * 1024**3
  assert report["cleanup"]["accepted"] is True
  assert "speedup" in report["performance_characterization"]
  ```

  Corrupt each raw source independently and require nonzero failure: same PID,
  non-overlapping intervals, different jobs, missing checksum, unvalidated placement,
  stale cleanup queue, wrong resource limit, missing replacement marker, or claimed
  speedup without raw timing.

- [ ] **Step 3: Run portable acceptance-helper tests and verify red**

  ```bash
  uv run pytest -m "unit and not hpc" \
    tests/unit/test_multi_process_slurm_acceptance.py -v
  ```

  Expected: modules/options/report builder do not exist.

- [ ] **Step 4: Implement strict profiles and report derivation**

  Reuse `fine_acceptance_manifest`, `execute_owned_slurm_workflow`,
  `run_profiled_fine_task`, `run_fine_worker_loss_task`,
  `scientific_equivalence_evidence`, owned outcome/queue validators, and checksum
  helpers. Keep all scheduler mutation in the existing owned lifecycle.

  Derive overlap using coordinator-bound placement intervals:

  ```python
  overlap = max(a_start, b_start) < min(a_end, b_end)
  same_process = (a_worker, a_pid) == (b_worker, b_pid)
  accepted = (
      overlap
      and a_job == b_job
      and a_host == b_host
      and not same_process
      and a_placement_accepted
      and b_placement_accepted
  )
  ```

  Aggregate per-process peak RSS by job ID for node-level peak characterization.
  Calculate speedup as `single_duration / multi_duration` and record it without a
  minimum threshold.

- [ ] **Step 5: Implement the explicitly selected HPC test**

  The test runs fresh one-process and two-process fine workflows, then a two-process
  single-worker-loss scenario. It requires:

  - accepted scientific artifact/metric comparison;
  - at least one same-job/distinct-process overlap;
  - exact job/process resource facts;
  - committed-artifact preservation and resume after one process loss;
  - healthy sibling liveness;
  - replacement group revalidation;
  - exact named-job cleanup for every scenario; and
  - timing/RSS characterization.

  It writes raw evidence before its summary and never substitutes a summary boolean
  for raw authority.

- [ ] **Step 6: Run portable helper tests and static checks**

  ```bash
  uv run pytest -m "unit and not hpc" \
    tests/unit/test_multi_process_slurm_acceptance.py \
    tests/unit/test_hpc_environment.py -v
  uv run ruff check tests/hpc tests/unit/test_multi_process_slurm_acceptance.py
  uv run basedpyright tests/hpc tests/unit/test_multi_process_slurm_acceptance.py
  ```

  Expected: all pass without contacting SLURM.

- [ ] **Step 7: Commit the focused gate**

  ```bash
  git add tests/hpc/multi_process_slurm_acceptance.py \
    tests/hpc/test_multi_process_slurm.py tests/hpc/conftest.py \
    tests/hpc/environment.py tests/hpc/workflow_acceptance.py \
    tests/hpc/fine_workflow_acceptance.py \
    tests/unit/test_multi_process_slurm_acceptance.py
  git commit -m "test: gate multi-process SLURM execution"
  ```

- [ ] **Step 8: Run real-HPC acceptance only with explicit operator-owned inputs**

  The operator supplies existing canonical files and an existing empty evidence
  directory through these required environment variables. The command fails before
  scheduler access if any variable is unset:

  ```bash
  : "${CYEXM_HPC_DATASET:?set CYEXM_HPC_DATASET to the approved manifest}"
  : "${CYEXM_HPC_SINGLE:?set CYEXM_HPC_SINGLE to the one-process profile}"
  : "${CYEXM_HPC_MULTI:?set CYEXM_HPC_MULTI to the two-process profile}"
  : "${CYEXM_HPC_EVIDENCE:?set CYEXM_HPC_EVIDENCE to an empty evidence directory}"
  ```

  Before scheduler access, resolve each path strictly and let the fixture reject
  missing files, nonempty evidence, protected roots, or profile mismatch.
  Then run exactly:

  ```bash
  uv run pytest -m hpc --run-hpc \
    --cyexm-dataset "$CYEXM_HPC_DATASET" \
    --cyexm-execution "$CYEXM_HPC_SINGLE" \
    --cyexm-single-process-execution "$CYEXM_HPC_SINGLE" \
    --cyexm-multi-process-execution "$CYEXM_HPC_MULTI" \
    --cyexm-evidence-dir "$CYEXM_HPC_EVIDENCE" \
    tests/hpc/test_multi_process_slurm.py -v
  ```

  Expected: PASS with complete machine-readable evidence and no owned jobs remaining.
  If explicit operator inputs are not supplied, record real-HPC acceptance as pending;
  do not skip, fabricate, or generalize portable evidence.

### Task 10: Synchronize examples, user documentation, API docs, and AGENTS contracts

**Files:**
- Create: `examples/execution-slurm-512gb-multi-process.yaml`
- Modify: `README.md`
- Modify: `docs/workflow-guide.md`
- Modify: `docs/fine-grained-execution.md`
- Modify: `docs/registration-artifacts.md`
- Modify: `src/cyexm/execution/slurm.py`
- Modify: `tests/unit/test_phase2_examples.py`
- Modify: `tests/unit/test_phase2_documentation.py`
- Modify: `AGENTS.md`
- Modify: `src/cyexm/config/AGENTS.md`
- Modify: `src/cyexm/execution/AGENTS.md`
- Modify: `src/cyexm/runtime/AGENTS.md`
- Modify: `examples/AGENTS.md`
- Modify: `docs/AGENTS.md`
- Modify: `tests/unit/AGENTS.md`
- Modify: `tests/integration/AGENTS.md`
- Modify: `tests/hpc/AGENTS.md`

**Interfaces:**
- Documents only implemented behavior. Real-HPC status remains pending until Task 9
  Step 8 passes.
- The new example is exact schema version 1 with fine mode, three job cap, 30 job
  cores, two processes, 15/15 native threads, 500 GB, and site-editable paths.

- [ ] **Step 1: Write failing executable documentation tests**

  Extend `EXAMPLE_PATHS` and assert the exact new profile:

  ```python
  assert profile.parallelism.mode == "fine_grained"
  assert profile.native_threads == NativeThreadSpec(itk=15, blas=15)
  assert profile.slurm is not None
  assert profile.slurm.workers == 3
  assert profile.slurm.cores_per_worker == 30
  assert profile.slurm.processes_per_worker == 2
  assert profile.slurm.memory == "500GB"
  ```

  Replace the README phrase contract
  ``processes_per_worker must be exactly 1`` with assertions for:

  - coarse exactly one;
  - fine positive process count;
  - exact core divisibility;
  - native-thread upper bound;
  - total job versus per-process memory;
  - job cap versus total slot capacity;
  - group preflight and replacement timeout;
  - version-3 shared-job provenance;
  - operational evidence is not cache authority; and
  - performance is characterization.

- [ ] **Step 2: Run documentation tests and verify red**

  ```bash
  uv run pytest -m "unit and not hpc" \
    tests/unit/test_phase2_examples.py \
    tests/unit/test_phase2_documentation.py -v
  ```

  Expected: missing example and obsolete one-process-only prose fail.

- [ ] **Step 3: Add the example and maintained user documentation**

  The example contains:

  ```yaml
  schema_version: 1
  executor: slurm
  parallelism:
    mode: fine_grained
  paths:
    dataset_root: /archive/path/to/dataset
    output_root: /archive/path/to/results
    worker_local_directory: /project/path/to/cyexm-worker
  native_threads:
    itk: 15
    blas: 15
  slurm:
    queue: 512GB
    workers: 3
    cores_per_worker: 30
    processes_per_worker: 2
    memory: 500GB
    walltime: "72:00:00"
    interface: ib0
    scheduler_interface: ib0
    startup_timeout: "15m"
    task_retries: 0
    job_name: cyexm-registration-multiprocess
    job_extra_directives:
      - --exclusive
  ```

  Explain that `workers: 3` means at most three jobs/nodes and therefore six task
  slots here. Explain the 466-GiB normalized job request and two 233-GiB limits.

  Add failure/restart prose for a lost process, lost job, partial group timeout,
  sibling liveness, complete revalidation, artifact resume, and ordered cleanup.
  Explain shared job IDs plus distinct addresses/PIDs in version-3 provenance.

  Keep affine whole-image and retain the warning that multi-process use is fine-only.

- [ ] **Step 4: Update `SlurmExecutor` Numpydoc**

  Document job cap, process-slot capacity, reference behavior, group preflight,
  replacement, cleanup, and the boundary with `FineGraphController` and
  `ArtifactStore`. Preserve `See Also` decision boundaries.

- [ ] **Step 5: Update all applicable AGENTS files**

  Replace future-only/one-process fine restrictions with the exact implemented
  fine-only group contract. Preserve historical acceptance claims. Add guidance that:

  - coarse remains one process/job;
  - fine groups share one job/host but use unique address/PID;
  - CPU/memory derive per process;
  - scale and cleanup own whole jobs;
  - every process has one slot;
  - real-HPC production claims require the focused gate; and
  - docs must distinguish implementation from completed external acceptance.

- [ ] **Step 6: Run docs tests and build the published HTML**

  ```bash
  uv run pytest -m "unit and not hpc" \
    tests/unit/test_phase2_examples.py \
    tests/unit/test_phase2_documentation.py -v
  (
    cd docs
    BASE_URL=/Cy-ExM uv run --group docs jupyter book build --html
  )
  ```

  Expected: tests pass and `docs/_build/html` completes without broken-page errors.
  Do not stage generated `_build` output.

- [ ] **Step 7: Commit documentation and guidance**

  ```bash
  git add examples/execution-slurm-512gb-multi-process.yaml README.md \
    docs/workflow-guide.md docs/fine-grained-execution.md \
    docs/registration-artifacts.md src/cyexm/execution/slurm.py \
    tests/unit/test_phase2_examples.py \
    tests/unit/test_phase2_documentation.py \
    AGENTS.md src/cyexm/config/AGENTS.md \
    src/cyexm/execution/AGENTS.md src/cyexm/runtime/AGENTS.md \
    examples/AGENTS.md docs/AGENTS.md tests/unit/AGENTS.md \
    tests/integration/AGENTS.md tests/hpc/AGENTS.md
  git commit -m "docs: explain multi-process SLURM execution"
  ```

  Commit body records that `SlurmExecutor` and existing schema fields were extended;
  no alternate executor or scientific mode was added.

### Task 11: Run final portable gates and review the exact diff

**Files:**
- Modify only if a gate exposes a defect in files already owned by Tasks 1–10.

**Interfaces:**
- Produces final portable verification evidence. It does not replace Task 9 real-HPC
  evidence.

- [ ] **Step 1: Confirm no live-run or unrelated file entered the branch**

  ```bash
  git status --short
  git diff --name-only 827af66..HEAD
  git diff --check 827af66..HEAD
  ```

  Expected: only planned source, test, example, docs, AGENTS, design, and plan files;
  no SLURM logs, `.idea`, `_build`, datasets, generated arrays, or live output.

- [ ] **Step 2: Run lock, unit, integration, formatting, lint, and type gates**

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  uv lock --check
  uv run pytest -m "unit and not hpc" tests/unit -v
  uv run pytest -m "integration and not hpc" tests/integration -v
  uv run ruff check .
  uv run ruff format --check .
  uv run basedpyright
  uv run pre-commit run --all-files
  ```

  Expected: every command exits zero.

- [ ] **Step 3: Validate the example resource derivation**

  ```bash
  uv run python - <<'PY'
  from cyexm.config import load_execution_profile
  from cyexm.config._slurm_resources import derive_slurm_job_resources

  profile = load_execution_profile(
      "examples/execution-slurm-512gb-multi-process.yaml"
  )
  assert profile.slurm is not None
  resources = derive_slurm_job_resources(
      profile.slurm,
      profile.native_threads,
      profile.parallelism,
  )
  assert resources.job_cap == 3
  assert resources.processes_per_job == 2
  assert resources.max_task_slots == 6
  assert resources.cores_per_process == 15
  assert resources.process_memory == "233GiB"
  print(resources)
  PY
  uv run pytest -m "unit and not hpc" tests/unit/test_workflow_api.py \
    -k "process_count and scientific" -v
  ```

  Expected: schema/resource validation succeeds and the focused regression proves
  scientific fingerprints are identical across one- and two-process profiles.

- [ ] **Step 4: Inspect commit and API documentation history**

  ```bash
  git log --oneline 827af66..HEAD
  git log --format=full 827af66..HEAD
  ```

  Expected: focused commits, with reuse analysis in API-changing commit bodies and no
  unrelated history.

- [ ] **Step 5: Request code review**

  Use `superpowers:requesting-code-review` against the design, this plan, and
  `827af66..HEAD`. Resolve every correctness finding with a focused test and commit.

- [ ] **Step 6: Record final status accurately**

  If Task 9 real-HPC execution passed, documentation may state exact-commit
  multi-process acceptance and link its checksum-bound report. If it was not run or
  failed, retain “portable implementation complete; real-HPC acceptance pending” and
  report the exact blocker without weakening gates.
