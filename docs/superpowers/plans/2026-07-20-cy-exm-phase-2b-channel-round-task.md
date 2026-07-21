# Cy-ExM Phase 2B Channel Artifacts and Round Tasks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute real whole-image or serial-chunked registration for one immutable round task, then resample every listed channel through the persisted warp-plus-affine chain with independent restart-safe artifacts.

**Architecture:** A new registration primitive transforms one source channel and owns only its artifact lifecycle. The scheduler-neutral top-level round worker configures native threads before lazy scientific imports, invokes the existing Phase 1 registration class selected by the task, transforms channels serially, and returns structured machine-readable evidence. Workflow execution binds this worker without introducing Dask or multiprocessing.

**Tech Stack:** Python 3.12, Phase 1 `ArtifactStore`, ANTSpyX, NumPy, tifffile, frozen workflow tasks, pytest unit fakes, bounded real-registration integration tests

## Global Constraints

- Complete Phase 2A first and use its committed interfaces without renaming them.
- Preserve Phase 1 `ImageRegistration`, `ChunkedImageRegistration`, and `register_round` behavior and defaults.
- Never use or export `ParallelChunkedImageRegistration`.
- Apply transforms in `[warp, affine]` order with `[False, False]` inversion.
- Use explicit `linear` or `nearestNeighbor` channel interpolation.
- Channel TIFF output clips to `[0, 65535]` and casts to `uint16` without histogram matching or intensity normalization.
- Preserve fixed-reference origin, spacing, direction, and shape.
- Give each channel output its own valid `.complete.json`; a TIFF alone is incomplete.
- Transform channels serially after registration; the registration channel is included and is distinct from Phase 1's diagnostic nonlinear output.
- Configure native threads before importing ANTs, NumPy, or SciPy inside the worker process.
- Do not add a scheduler, client, future, child task, or multiprocessing pool to `run_round_task`.

---

## File and responsibility map

| Path | Responsibility |
| --- | --- |
| `src/cyexm/io/tiff.py` | Geometry-preserving, clipping-only uint16 TIFF writer |
| `src/cyexm/registration/backend.py` | Testable transform-chain and clipped-writer backend boundary |
| `src/cyexm/registration/channels.py` | One restart-safe source-channel transformation |
| `src/cyexm/registration/__init__.py` | Accepted `transform_channel` export without widening unrelated Phase 1 exports |
| `src/cyexm/workflow/round_task.py` | Top-level serializable real round worker |
| `src/cyexm/workflow/api.py` | Default runner binding and final result aggregation |
| `src/cyexm/workflow/status.py` | Exact registration/channel completion classification |
| `tests/unit/fakes.py` | Backend calls for transform chain and clipped persistence |
| `tests/unit/test_tiff.py` | Clipping-only encoding and geometry tests |
| `tests/unit/test_channel_transform.py` | Request identity, atomicity, force, and reuse tests |
| `tests/unit/test_round_task.py` | Strategy selection, serial channel order, evidence, and failure tests |
| `tests/unit/test_workflow_api.py` | Default real-runner binding and workflow result tests |
| `tests/integration/test_channel_transform.py` | Real transform-chain channel output |
| `tests/integration/test_workflow_round_task.py` | Real whole-image and serial-chunked round execution/restart |
| `tests/unit/test_import_contract.py` | Worker module remains import-light until invocation |

---

### Task 1: Add a geometry-preserving clipping-only TIFF boundary

**Files:**
- Modify: `src/cyexm/io/tiff.py`
- Modify: `src/cyexm/registration/backend.py`
- Modify: `tests/unit/fakes.py`
- Modify: `tests/unit/test_tiff.py`
- Modify: `tests/unit/test_backend.py`

**Interfaces:**
- Consumes: existing TIFF geometry metadata and atomic-write implementation
- Produces: `write_clipped_uint16_tiff(image, path) -> None`; backend methods `apply_transform_chain` and `write_clipped_tiff`

- [ ] **Step 1: Write clipping-without-normalization tests**

```python
def test_write_clipped_uint16_tiff_does_not_normalize(tmp_path: Path) -> None:
    import numpy as np
    import tifffile

    source = np.asarray([[[-4.0, 0.0, 12.9, 65535.0, 70000.0]]], dtype=np.float32)
    destination = tmp_path / "channel.tif"
    write_clipped_uint16_tiff(source, destination)
    written = tifffile.imread(destination)
    assert written.dtype == np.dtype(np.uint16)
    assert written.tolist() == [[[0, 0, 12, 65535, 65535]]]
```

Add focused cases for NaN/positive infinity/negative infinity mapping to zero, signed and unsigned integer inputs, invalid dimensions/types/extensions, atomic temporary cleanup, and exact ANTs fixed geometry metadata.

- [ ] **Step 2: Run the focused TIFF tests and observe the missing writer**

```bash
uv run pytest tests/unit/test_tiff.py -q
```

Expected: collection fails because `write_clipped_uint16_tiff` is absent.

- [ ] **Step 3: Implement the clipping writer without changing the existing scaling writer**

Reuse the existing image-to-external-ZYX and metadata helpers. For floating input, call `np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)`, then `np.clip(data, 0, 65535).astype(np.uint16)`. Use the same suffix-preserving `atomic_write` and validate axes, dtype, shape, exact array values, and `ImageGeometry`. Do not call `write_uint16_tiff` because its established Phase 1 behavior normalizes to a requested range.

- [ ] **Step 4: Extend the backend protocol and fake**

Add these exact methods to `RegistrationBackend`, `AntsRegistrationBackend`, and `FakeBackend`:

```python
def apply_transform_chain(
    self,
    *,
    fixed: object,
    moving: object,
    affine_path: Path,
    warp_path: Path,
    interpolation: str,
) -> object:
    raise NotImplementedError


def write_clipped_tiff(self, image: object, path: Path) -> None:
    raise NotImplementedError
```

The production adapter delegates to `registration.transforms.apply_transform_chain` and `io.tiff.write_clipped_uint16_tiff` after `_prepare()`. The fake records exact transform order, inversion policy, interpolation, and output values.

- [ ] **Step 5: Run TIFF and backend tests**

```bash
uv run pytest tests/unit/test_tiff.py tests/unit/test_backend.py -q
```

Expected: clipping, geometry, atomicity, and adapter-delegation tests pass; existing normalized Phase 1 TIFF tests remain unchanged.

- [ ] **Step 6: Commit the clipping-only I/O boundary**

```bash
git add src/cyexm/io/tiff.py src/cyexm/registration/backend.py \
  tests/unit/fakes.py tests/unit/test_tiff.py tests/unit/test_backend.py
git commit -m "feat: write clipped channel TIFF outputs"
```

---

### Task 2: Implement one restart-safe transformed-channel artifact

**Files:**
- Create: `src/cyexm/registration/channels.py`
- Modify: `src/cyexm/registration/__init__.py`
- Create: `tests/unit/test_channel_transform.py`

**Interfaces:**
- Consumes: `ArtifactStore`, `StageRequest`, `RegistrationBackend`, `NativeThreadConfig`
- Produces: `transform_channel(*, fixed_image_path, moving_image_path, affine_path, warp_path, output_path, pixel_size, interpolation, native_threads=None, force_override=False, _backend=None, _artifact_store=None) -> Path`; the two underscored injections remain test-only

- [ ] **Step 1: Write the successful transform and request-identity test**

```python
def test_transform_channel_commits_transform_identity(tmp_path: Path) -> None:
    fixed, moving, affine, warp = source_files(tmp_path)
    output = tmp_path / "registered-channel.tif"
    backend = FakeBackend()
    result = transform_channel(
        fixed_image_path=fixed,
        moving_image_path=moving,
        affine_path=affine,
        warp_path=warp,
        output_path=output,
        pixel_size=(0.324, 0.324, 0.9),
        interpolation="linear",
        native_threads=NativeThreadConfig(1, 1),
        _backend=backend,
    )
    record = CompletionRecord.read(output.with_name(output.name + ".complete.json"))
    assert result == output.resolve()
    assert backend.transform_lists == [(warp.resolve(), affine.resolve())]
    assert record.request.parameters["transform_order"] == ["warp", "affine"]
    assert record.request.parameters["whichtoinvert"] == [False, False]
    assert record.request.parameters["postprocessing"] == "clip-uint16-no-normalization-v1"
```

- [ ] **Step 2: Add reuse, stale, force, and failure tests**

Cover exact reuse without backend transform, changed source/fixed/affine/warp identity, changed XYZ spacing, changed interpolation, both `linear` and `nearestNeighbor` forwarding, changed policy version, data without record, record without data, unreadable cached TIFF, source mutation during execution, writer failure cleanup, completion-record failure, and intentional force recomputation.

```bash
uv run pytest tests/unit/test_channel_transform.py -q
```

Expected: collection fails because `cyexm.registration.channels` is absent.

- [ ] **Step 3: Implement validation and request construction**

Validate built-in path-like inputs through `FileIdentity.from_path`; validate pixel size through the established Phase 1 XYZ validator behavior; accept only `linear` and `nearestNeighbor`; call `configure_native_threads` before constructing the production backend. Build a stage request with:

```python
StageRequest.for_files(
    stage="channel-transform-v1",
    inputs=(fixed, moving, affine, warp),
    geometry=backend.geometry(fixed_image),
    method="warp-affine",
    accuracy="not-applicable",
    native_threads=resolved.values,
    upstream=FileIdentity.from_path(warp),
    parameters={
        "pixel_size_xyz_um": list(pixel_size_xyz),
        "interpolation": interpolation,
        "transform_order": ["warp", "affine"],
        "whichtoinvert": [False, False],
        "axis_convention": AXIS_CONVENTION_VERSION,
        "output_format": "ome-compatible-tiff-uint16-v1",
        "postprocessing": "clip-uint16-no-normalization-v1",
    },
)
```

- [ ] **Step 4: Implement reuse, transform, and atomic commit**

If forced, call `ArtifactStore.mark_stale(output)` before reuse. Load fixed and moving through the backend, require compatible shapes and fixed-reference geometry, apply the chain once, and commit through `backend.write_clipped_tiff`. Re-stat all four input identities immediately before successful return and raise `RegistrationStageError` if any changed. Export only `transform_channel` from `cyexm.registration.channels` and add it to `cyexm.registration.__all__` after its focused tests pass.

- [ ] **Step 5: Run all channel artifact tests**

```bash
uv run pytest tests/unit/test_channel_transform.py tests/unit/test_artifacts.py -q
```

Expected: all identity, reuse, force, stale, atomicity, and transform-policy tests pass.

- [ ] **Step 6: Commit transformed-channel artifacts**

```bash
git add src/cyexm/registration/channels.py src/cyexm/registration/__init__.py \
  tests/unit/test_channel_transform.py
git commit -m "feat: persist restart-safe channel transforms"
```

---

### Task 3: Implement the top-level scheduler-neutral round worker

**Files:**
- Create: `src/cyexm/workflow/round_task.py`
- Create: `tests/unit/test_round_task.py`

**Interfaces:**
- Consumes: Phase 2A `RoundTask`/`RoundResult`, Phase 1 registration classes, `transform_channel`
- Produces: `run_round_task(task: RoundTask) -> RoundResult`

- [ ] **Step 1: Write whole-image strategy and serial channel-order tests**

```python
def test_round_task_registers_then_transforms_channels_in_manifest_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    install_fake_round_runtime(monkeypatch, calls)
    task = round_task(tmp_path, strategy="whole_image", channel_names=("b.tif", "a.tif"))
    result = run_round_task(task)
    assert calls == [
        ("register", "whole_image"),
        ("channel", "b.tif"),
        ("channel", "a.tif"),
    ]
    assert result.status == "completed"
    assert result.round == task.round
```

The fake registration writes valid affine and warp completion records in the task's isolated registration directory so the worker follows the real artifact contract.

- [ ] **Step 2: Add strategy, evidence, and structured-failure tests**

Test exact constructor arguments for both `ImageRegistration` and `ChunkedImageRegistration`, ZYX chunk parameters only for chunked, force propagation, thread configuration before fake scientific import, output/log directory creation only on invocation, hostname/PID/SLURM/Dask-key evidence, registration failure, a middle-channel failure that prevents later channels, and preservation of previously committed artifacts.

```bash
uv run pytest tests/unit/test_round_task.py -q
```

Expected: collection fails because `run_round_task` is absent.

- [ ] **Step 3: Implement thread-first lazy runtime loading**

At the start of `run_round_task`, call:

```python
threads = NativeThreadConfig(
    itk_threads=task.native_threads.itk,
    blas_threads=task.native_threads.blas,
)
resolved = configure_native_threads(threads)
```

Only after this call, import `ImageRegistration`, `ChunkedImageRegistration`, `RegistrationPaths`, and `transform_channel` inside a private `_runtime()` function. Do not import Dask; obtain a task key only by checking for an already loaded `distributed` module and handling absence as `None`.

- [ ] **Step 4: Implement registration selection and channel loop**

Create the round log parent and registration directory. Instantiate the selected Phase 1 class with exact fixed/moving paths, round, XYZ spacing, crop, force, thread config, linear type/accuracy, and nonlinear type/accuracy. Pass `chunk_size` and `overlap_fraction` only for `serial_chunked`. After `.register()`, derive affine and warp through `RegistrationPaths.for_round`, validate their completion records, and invoke `transform_channel` for every channel tuple in order.

- [ ] **Step 5: Implement result and failure evidence**

Capture UTC start/end timestamps, monotonic duration, `socket.gethostname()`, `os.getpid()`, optional `SLURM_JOB_ID`, optional Dask key, strategy, resolved thread mapping, registration/channel artifact strings, and reuse decisions. Catch ordinary exceptions into a failed `RoundResult` with exception class/message and log path; re-raise `KeyboardInterrupt` and `SystemExit`.

- [ ] **Step 6: Run round-worker unit tests**

```bash
uv run pytest tests/unit/test_round_task.py -q
```

Expected: all strategy, ordering, thread, force, evidence, and failure-isolation tests pass.

- [ ] **Step 7: Commit the real round worker**

```bash
git add src/cyexm/workflow/round_task.py tests/unit/test_round_task.py
git commit -m "feat: execute scheduler-neutral round tasks"
```

---

### Task 4: Bind the real runner and exact workflow status semantics

**Files:**
- Modify: `src/cyexm/workflow/api.py`
- Modify: `src/cyexm/workflow/status.py`
- Modify: `tests/unit/test_workflow_api.py`
- Modify: `tests/unit/test_workflow_status.py`

**Interfaces:**
- Consumes: `run_round_task`
- Produces: default sequential `run_workflow(plan)` and exact pending/partial/complete/invalid round classification

- [ ] **Step 1: Add default-runner and resume tests**

```python
def test_run_workflow_uses_round_worker_and_skips_complete_rounds(
    monkeypatch: pytest.MonkeyPatch, complete_plan: WorkflowPlan
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        "cyexm.workflow.round_task.run_round_task",
        lambda task: calls.append(task.round) or successful_result(task.round),
    )
    result = run_workflow(complete_plan)
    assert calls == []
    assert tuple(item.status for item in result.rounds) == ("reused",)
```

Add incomplete-channel, stale-registration, forced-round, force-all, and mixed success/failure cases.

- [ ] **Step 2: Run focused API/status tests and observe missing default binding**

```bash
uv run pytest tests/unit/test_workflow_api.py tests/unit/test_workflow_status.py -q
```

Expected: the new default-runner and exact channel-status cases fail.

- [ ] **Step 3: Bind the runner lazily and classify from authoritative records**

The sequential executor's production runner imports `run_round_task` inside the executor-opening operation. Status validates exact expected affine, warp, and channel record paths. A complete workflow requires all expected scientific artifacts; diagnostic metrics and workflow-run summaries do not authorize completion. `run_workflow` returns reused `RoundResult` entries for skipped rounds so the final ordered result always contains every round.

- [ ] **Step 4: Run focused workflow tests**

```bash
uv run pytest tests/unit/test_workflow_api.py tests/unit/test_workflow_status.py \
  tests/unit/test_execution_sequential.py -q
```

Expected: all real-runner binding, resume, force, ordering, and authoritative-status tests pass.

- [ ] **Step 5: Commit real sequential workflow execution**

```bash
git add src/cyexm/workflow/api.py src/cyexm/workflow/status.py \
  tests/unit/test_workflow_api.py tests/unit/test_workflow_status.py
git commit -m "feat: run complete sequential registrations"
```

---

### Task 5: Characterize real channel and round-task behavior

**Files:**
- Create: `tests/integration/test_channel_transform.py`
- Create: `tests/integration/test_workflow_round_task.py`

**Interfaces:**
- Consumes: real Phase 1 ANTSpyX runtime, synthetic TIFF fixtures, `transform_channel`, `run_round_task`
- Produces: numerical, geometry, restart, and strategy acceptance evidence for the scheduler-neutral task

- [ ] **Step 1: Write the real transform-chain integration test**

Inside the test function, import NumPy/tifffile after the autouse one-thread fixture. Create a fixed image, moving channel, identity affine, and deterministic zero warp. Assert the channel output equals clipped source intensities exactly, uses uint16, has fixed geometry, and has a valid completion record. Run once more and assert artifact/record identities are unchanged.

- [ ] **Step 2: Run the channel integration test**

```bash
uv run pytest tests/integration/test_channel_transform.py -v
```

Expected: PASS with no skipped test and no temporary siblings left under `tmp_path`.

- [ ] **Step 3: Write bounded whole-image and serial-chunked round tests**

Use the existing `16 x 32 x 32` maximum fixture, `dry run` accuracies, and chunks no larger than the fixture. For each strategy, assert affine/warp records, registration diagnostic, every channel output/record, manifest channel order in result evidence, and second-run reuse. Inject one nonlinear-stage failure, rerun, and assert the valid affine record was reused.

- [ ] **Step 4: Run round integration tests with all scientific threads bounded**

```bash
uv run pytest tests/integration/test_workflow_round_task.py -v
```

Expected: whole-image, serial-chunked, multi-channel, and restart cases pass within the five-minute portable integration budget.

- [ ] **Step 5: Compare channel output with direct transform application**

Add an assertion using the existing transform helper as the independent reference. Use exact equality for identity transforms; for estimated transforms use the documented `rtol=1e-4`, `atol=1e-5` migration tolerance and explain that it covers established ANTSpyX floating-point variability.

- [ ] **Step 6: Commit real round-task characterization**

```bash
git add tests/integration/test_channel_transform.py \
  tests/integration/test_workflow_round_task.py
git commit -m "test: characterize workflow round execution"
```

---

### Task 6: Close the Phase 2B import and portable gates

**Files:**
- Modify: `tests/unit/test_import_contract.py`
- Modify: `src/cyexm/registration/AGENTS.md`
- Modify: `src/cyexm/workflow/AGENTS.md`

**Interfaces:**
- Consumes: accepted channel and round-task behavior
- Produces: current package guidance and proof that import remains side-effect free

- [ ] **Step 1: Add a fresh-process worker import test**

```python
def test_round_task_module_defers_scientific_imports_until_execution() -> None:
    code = """
import os
import sys
before = dict(os.environ)
import cyexm.registration.channels
import cyexm.workflow.round_task
assert before == dict(os.environ)
for root in ("ants", "numpy", "scipy", "tifffile", "dask", "distributed"):
    assert not any(name == root or name.startswith(root + ".") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
```

- [ ] **Step 2: Update local guidance from future to current behavior**

Document in registration guidance that `channels.py` owns only one channel's artifact lifecycle and cannot schedule. Document in workflow guidance that `round_task.py` is the only real registration execution boundary, configures threads before lazy imports, and cannot create Dask clients or child tasks.

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

Expected: every command exits 0; record exact unit/integration totals and elapsed times.

- [ ] **Step 4: Commit the Phase 2B import and guidance contract**

```bash
git add tests/unit/test_import_contract.py src/cyexm/registration/AGENTS.md \
  src/cyexm/workflow/AGENTS.md
git commit -m "docs: define round task runtime boundaries"
```

- [ ] **Step 5: Review the milestone diff**

```bash
git log --oneline --decorate main..HEAD
git diff --stat HEAD~6..HEAD
git status --short
```

Expected: six focused milestone commits and a clean worktree.
