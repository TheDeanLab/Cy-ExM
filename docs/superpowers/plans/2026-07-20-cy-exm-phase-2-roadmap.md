# Cy-ExM Phase 2 Runtime Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved Phase 2 workflow runtime through five independently reviewable, test-first milestones.

**Architecture:** Strict scientific and execution configuration produces immutable round tasks. Scheduler-neutral orchestration sends those tasks through sequential, local-Dask, or SLURM executors while Phase 1 completion records remain the sole restart authority. Each milestone fixes and tests one boundary before the next scheduler layer is added.

**Tech Stack:** Python 3.12, frozen dataclasses, PyYAML 6, ANTSpyX, tifffile, Dask/distributed 2025.1.0, dask-jobqueue 0.9.0, pytest, uv 0.7.13, Ruff, BasedPyright, pre-commit, SLURM

## Global Constraints

- Execute the milestone plans in the order listed below; later plans consume exact interfaces committed by earlier plans.
- Work in an isolated worktree created with `superpowers:using-git-worktrees`; never implement directly on `main`.
- Preserve Phase 1 APIs and completion-record semantics. A workflow summary, future, or SLURM job identifier never authorizes artifact reuse.
- Keep the scientific manifest separate from the machine-specific execution profile.
- Keep configuration and planning imports free of ANTs, NumPy, SciPy, Dask, distributed, and dask-jobqueue.
- Resolve native threads in every round worker before scientific imports. Initial examples use `itk: 1` and `blas: 1`.
- Submit one coarse task per moving round. Do not add nested multiprocessing, per-chunk Dask scheduling, or `ParallelChunkedImageRegistration`.
- Use one Dask `round_slot` resource per worker and require one slot for every round task.
- Treat the SLURM `512GB`, 72-core, 500-GB-memory, 72-hour profile as an editable observed example until fresh HPC acceptance records the actual allocation.
- Keep environments and uv caches on `/project`; never move active environments or uv caches to `/archive`.
- Do not modify ClearEx or the separate spheroid registration process and its outputs.
- Use `apply_patch` for tracked edits and TDD for every behavior.

Before every uv command:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
```

Before scientific tests, also export all six thread variables as `1`:

```bash
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

---

## Ordered milestone plans

1. [Configuration, planning, sequential execution, API, and CLI](2026-07-20-cy-exm-phase-2a-config-planning-sequential.md)
2. [Restart-safe channel transformation and real round tasks](2026-07-20-cy-exm-phase-2b-channel-round-task.md)
3. [Bounded local-Dask execution](2026-07-20-cy-exm-phase-2c-local-dask.md)
4. [SLURM construction, worker preflight, and lifecycle](2026-07-20-cy-exm-phase-2d-slurm.md)
5. [HPC acceptance, complete README, and example YAML](2026-07-20-cy-exm-phase-2e-hpc-docs-examples.md)

## Cross-milestone interface lock

| Producer | Interface consumed later |
| --- | --- |
| Phase 2A | `DatasetManifest`, `ExecutionProfile`, `WorkflowPlan`, `RoundTask`, `RoundResult`, `WorkflowResult`, `WorkflowStatus` |
| Phase 2A | `load_dataset_manifest`, `load_execution_profile`, `build_workflow_plan`, `run_workflow`, `inspect_workflow_status` |
| Phase 2A | `WorkflowExecutor.execute(tasks) -> tuple[RoundResult, ...]` and `SequentialExecutor` |
| Phase 2B | `transform_channel` and `run_round_task` |
| Phase 2C | `LocalDaskExecutor` with one `round_slot` per worker and bounded retry/cancellation |
| Phase 2D | `SlurmExecutor`, `WorkerProbe`, and preflight evidence included in workflow-run records |
| Phase 2E | Four validated YAML examples, explicit `hpc` tests/evidence, and the complete accepted `README.md` surface |

Changing one of these interfaces during execution requires updating every unexecuted dependent plan before continuing. Do not silently adapt names in downstream tasks.

## Approved-specification coverage

| Design area | Executable plan coverage |
| --- | --- |
| Package/dependency/import boundaries | Phase 2A Tasks 1 and 9; Phase 2B Task 6; Phase 2C Task 6; Phase 2D Task 4 |
| Strict scientific YAML and immutable values | Phase 2A Tasks 2-3 |
| Separate execution YAML and native/resource rules | Phase 2A Task 4; Phase 2C Task 1 |
| Deterministic planning, fingerprints, path/header safety | Phase 2A Tasks 5-6 |
| Public API, CLI, exit codes, output/status ownership | Phase 2A Tasks 7-8; Phase 2B Task 4 |
| Restart-safe multi-channel transformation | Phase 2B Tasks 1-2 and 5 |
| Whole-image and serial-chunked round tasks | Phase 2B Tasks 3 and 5 |
| Sequential/local-Dask/SLURM executors | Phase 2A Task 7; Phase 2C Tasks 2-5; Phase 2D Tasks 1-3 |
| Failure, bounded retry, interruption, and cleanup | Phase 2A Tasks 7-8; Phase 2C Tasks 2 and 4; Phase 2D Tasks 1-3; Phase 2E Task 5 |
| README and four checked-in YAML examples | Phase 2E Tasks 1-3 |
| Portable, integration, and explicit HPC acceptance | Every milestone gate; Phase 2E Tasks 4-6 |
| Fresh 512-GB site assumptions and one node per round | Phase 2D preflight plus Phase 2E Tasks 4-5 |

## Review and integration gates

After each milestone:

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

Record the exact counts and elapsed times in the milestone review. Do not claim HPC acceptance from portable fake-construction or LocalCluster tests. Phase 2E supplies the separate explicitly selected HPC command and machine-readable evidence contract.
