# Cy-ExM Phase 2E HPC Acceptance and Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the complete runtime on dedicated 512-GB SLURM nodes, document every accepted behavior in `README.md`, and ship four strict reusable YAML examples including the spheroid rounds and channels.

**Architecture:** Checked-in examples exercise the same strict public loaders used in production. Portable documentation tests lock commands, fields, paths, semantics, and warnings to the implementation. Explicitly selected HPC tests write bounded machine-readable evidence outside the repository, compare SLURM outputs with the sequential reference, and support a concise committed acceptance audit only after real allocation, restart, failure, cancellation, and cleanup evidence passes.

**Tech Stack:** Python 3.12, Cy-ExM workflow CLI, PyYAML, Dask/distributed, dask-jobqueue, SLURM `512GB` partition, pytest `hpc` marker, Markdown

## Global Constraints

- Complete Phase 2A through Phase 2D first.
- Do not edit, stop, inspect through scientific libraries, or reuse output from the separate long-running spheroid registration process.
- The spheroid manifest contains filenames and scientific settings only; it contains no absolute lab path.
- Keep execution paths in separate YAML profiles.
- Treat `512GB`, 72 cores, 500 GB Dask memory, 72 hours, and `ib0` as editable site examples until the real acceptance run records current facts.
- HPC tests are never part of portable CI and require an explicit profile, dataset, and evidence directory.
- An explicitly selected HPC test with missing required arguments fails with a clear error; it does not silently skip by host or scheduler availability.
- Write large logs, job scripts, scheduler snapshots, arrays, and outputs to the caller's evidence/output path on `/archive`, never into git.
- Do not claim Phase 2 or HPC acceptance until every explicit command and evidence assertion in Tasks 4 and 5 passes.
- `README.md` documents the complete accepted public runtime, not internal aspirations or unavailable commands.

---

## File and responsibility map

| Path | Responsibility |
| --- | --- |
| `examples/spheroid-dataset.yaml` | Portable scientific manifest for reference round 1 and moving rounds 2-4 |
| `examples/execution-sequential.yaml` | Reusable local serial profile |
| `examples/execution-local-dask.yaml` | Reusable bounded process-based LocalCluster profile |
| `examples/execution-slurm-512gb.yaml` | Editable site-specific one-node-per-round profile |
| `tests/unit/test_phase2_examples.py` | Parse examples and assert exact spheroid scientific values |
| `README.md` | Complete Phase 1/2 user, YAML, execution, restart, failure, output, and HPC guide |
| `tests/unit/test_phase2_documentation.py` | Lock README commands, example links, fields, exits, and semantic warnings |
| `.github/workflows/ci.yml` | Install HPC Python extras for portable import/type/fake-construction tests without scheduler access |
| `AGENTS.md` | Current accepted Phase 2 commands and boundaries |
| `.github/workflows/AGENTS.md` | Current portable extra/marker guidance |
| `examples/AGENTS.md` | Current manifest/profile example rules |
| `tests/hpc/AGENTS.md` | Explicit scheduler/data/evidence safety contract |
| `tests/hpc/conftest.py` | Required HPC CLI options and evidence-root fixtures |
| `tests/hpc/test_slurm_environment.py` | Partition/allocation/mount/network/preflight/scratch/job-script evidence |
| `tests/hpc/test_slurm_workflow.py` | Placement, numerical parity, failure, cancellation, restart, and cleanup evidence |
| `docs/phase-2-acceptance.md` | Concise reproducible acceptance audit after evidence passes |
| `tests/unit/test_repository_policy.py` | Guidance inventory, CI selector, docs/examples, and acceptance-audit policy |

---

### Task 1: Add and strictly validate all four YAML examples

**Files:**
- Create: `examples/spheroid-dataset.yaml`
- Create: `examples/execution-sequential.yaml`
- Create: `examples/execution-local-dask.yaml`
- Create: `examples/execution-slurm-512gb.yaml`
- Create: `tests/unit/test_phase2_examples.py`

**Interfaces:**
- Consumes: public strict configuration loaders
- Produces: four checked-in reusable version-1 documents referenced by README commands

- [ ] **Step 1: Write failing example inventory and exact-value tests**

```python
EXAMPLE_PATHS = (
    "examples/spheroid-dataset.yaml",
    "examples/execution-sequential.yaml",
    "examples/execution-local-dask.yaml",
    "examples/execution-slurm-512gb.yaml",
)


def test_spheroid_manifest_records_approved_scientific_inputs() -> None:
    manifest = load_dataset_manifest(REPOSITORY_ROOT / EXAMPLE_PATHS[0])
    assert manifest.voxel_spacing_um == (0.324, 0.324, 0.900)
    assert manifest.reference.channel == PurePosixPath("round_1_488_tubulin.tif")
    assert tuple(item.round for item in manifest.rounds) == (2, 3, 4)
    assert tuple(item.registration_channel.name for item in manifest.rounds) == (
        "round_2_488_tubulin.tif",
        "round_3_488_tubulin.tif",
        "round_4_561_tubulin.tif",
    )
```

Assert exact channel tuples for rounds 2-4 and exact executor/resource values in all profiles. Assert no manifest scalar begins with `/project`, `/archive`, or `/endosome`.

- [ ] **Step 2: Run the example tests and observe missing files**

```bash
uv run pytest tests/unit/test_phase2_examples.py -q
```

Expected: all four inventory cases fail because the YAML files are absent.

- [ ] **Step 3: Add the spheroid scientific manifest**

Create a schema-version-1 manifest with dataset ID `spheroid-registration-example`, reference `round_1_488_tubulin.tif`, XYZ spacing `(0.324, 0.324, 0.900)`, `crop: false`, TRSAA/high, and SyNOnly/high. Include:

```yaml
rounds:
  - round: 2
    strategy: whole_image
    registration_channel: round_2_488_tubulin.tif
    channels:
      - {file: round_2_488_tubulin.tif, interpolation: linear}
      - {file: round_2_561_bcatenin.tif, interpolation: linear}
      - {file: round_2_647_hsp60.tif, interpolation: linear}
  - round: 3
    strategy: serial_chunked
    registration_channel: round_3_488_tubulin.tif
    chunk_size_zyx: [512, 512, 512]
    overlap_fraction: 0.15
    channels:
      - {file: round_3_488_tubulin.tif, interpolation: linear}
      - {file: round_3_647_fbl.tif, interpolation: linear}
  - round: 4
    strategy: whole_image
    registration_channel: round_4_561_tubulin.tif
    channels:
      - {file: round_4_488_vimentin.tif, interpolation: linear}
      - {file: round_4_561_tubulin.tif, interpolation: linear}
      - {file: round_4_647_pex14.tif, interpolation: linear}
```

- [ ] **Step 4: Add three separate execution profiles**

Use portable editable paths in sequential/local examples. The SLURM profile uses `/archive/path/to/dataset`, `/archive/path/to/results`, `/project/path/to/cyexm-worker`, queue `512GB`, workers 3, cores 72, one process, memory `500GB`, walltime `72:00:00`, interfaces `ib0`, timeout `15m`, retries 0, and no extra directives. Comments identify every absolute SLURM path/resource as site-specific and editable.

- [ ] **Step 5: Parse all examples through public loaders**

```bash
uv run pytest tests/unit/test_phase2_examples.py -q
```

Expected: all exact scientific/profile and portability assertions pass.

- [ ] **Step 6: Commit checked-in configuration examples**

```bash
git add examples/spheroid-dataset.yaml examples/execution-sequential.yaml \
  examples/execution-local-dask.yaml examples/execution-slurm-512gb.yaml \
  tests/unit/test_phase2_examples.py
git commit -m "docs: add reusable workflow YAML examples"
```

---

### Task 2: Rewrite README as the complete accepted runtime guide

**Files:**
- Modify: `README.md`
- Create: `tests/unit/test_phase2_documentation.py`

**Interfaces:**
- Consumes: complete Phase 2 public API/CLI and checked-in examples
- Produces: one authoritative user guide whose commands and semantics are policy-tested

- [ ] **Step 1: Add failing README contract tests**

```python
def test_readme_documents_every_workflow_command_and_example() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    for command in (
        "cyexm workflow validate",
        "cyexm workflow plan",
        "cyexm workflow run",
        "cyexm workflow status",
        "--force-round",
        "--force-all",
    ):
        assert command in readme
    for path in EXAMPLE_PATHS:
        assert f"]({path})" in readme
```

Add required-section/phrase assertions for every schema field, executor, output directory, diagnostic/channel distinction, force/fingerprint rule, retry/cancellation policy, exit code, native-thread setting, `round_slot`, 512-GB warning, HPC selector, and limitation.

- [ ] **Step 2: Run the documentation test and observe the Phase 1-only README**

```bash
uv run pytest tests/unit/test_phase2_documentation.py -q
```

Expected: failures list the missing Phase 2 sections and commands.

- [ ] **Step 3: Document installation and architecture**

Retain the Phase 1 guide, then add exact install commands for base, workflow, distributed, and HPC extras. Explain strict manifest/profile separation, immutable planning, one task per moving round, serial work inside each task, Phase 1 artifacts as sole restart authority, and the absence of nested chunk multiprocessing.

- [ ] **Step 4: Document every version-1 YAML field and example**

Use tables for scientific fields and execution fields. State units/order, types, allowed values, conditional chunk fields, interpolation, relative scientific paths, profile-relative machine paths, exact-key/duplicate/alias/tag/additional-document rejection, disjoint roots, native-thread resolution, local resources, and every SLURM value/directive restriction. Link all four exact example paths.

- [ ] **Step 5: Document commands, outputs, and lifecycle**

Show validate, mutation-free plan, resume-by-default run, repeated force-round, force-all, status, JSON, and log-level examples. Include exit codes 0/2/3/4/130. Show the full output tree. Explain that `nonlinear_registered_<round>.tif` is a normalized Phase 1 diagnostic while `channels/<source>.tif` is the original channel resampled and clipped without normalization.

- [ ] **Step 6: Document executor, restart, failure, and HPC behavior**

Explain sequential reference behavior, process-based LocalCluster, one `round_slot`, SLURMCluster preflight, one process/job/node, independent failures, zero default retry, bounded retry, cancellation/cleanup, compatible force, incompatible-plan new-root requirement, workflow-run operational evidence, coordinator location, editable 512-GB example, acceptance scope, and remaining non-goals.

- [ ] **Step 7: Run README and example contract tests**

```bash
uv run pytest tests/unit/test_phase2_documentation.py \
  tests/unit/test_phase2_examples.py -q
```

Expected: every documented command, path, field, semantic distinction, warning, and example assertion passes.

- [ ] **Step 8: Commit complete user documentation**

```bash
git add README.md tests/unit/test_phase2_documentation.py
git commit -m "docs: document the Phase 2 workflow runtime"
```

---

### Task 3: Update portable CI and repository guidance for current Phase 2

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `AGENTS.md`
- Modify: `.github/workflows/AGENTS.md`
- Modify: `examples/AGENTS.md`
- Modify: `tests/unit/test_repository_policy.py`

**Interfaces:**
- Consumes: accepted portable unit/integration suite
- Produces: locked CI with workflow/distributed extras and current Phase 2 guidance

- [ ] **Step 1: Add failing current-runtime and CI tests**

Require CI sync exactly:

```text
uv sync --locked --group dev --extra examples --extra hpc
```

Require root guidance to describe Phase 2 as current, list strict manifests/profiles, preserve Phase 1 artifact authority, forbid nested chunk scheduling, retain explicit HPC selection, and use the same locked sync command. Require example guidance to identify all four current files rather than calling Phase 2 proposed.

- [ ] **Step 2: Run repository-policy tests and observe stale Phase 2 language**

```bash
uv run pytest tests/unit/test_repository_policy.py -q
```

Expected: CI command and current-versus-proposed guidance assertions fail.

- [ ] **Step 3: Update CI without adding real HPC execution**

Change only the locked sync step and portable job name/scope needed for Phase 2 unit, bounded LocalCluster integration, SLURM import/type checks, and fake construction. Keep all six one-thread environment values. Continue running `unit and not hpc` and `integration and not hpc`. Installing the `hpc` Python extra is permitted; do not invoke `sinfo`, `sbatch`, `squeue`, or any external scheduler in public CI.

- [ ] **Step 4: Update all affected AGENTS guidance**

Replace future/proposed gates with exact current Phase 2 package boundaries, examples, commands, resource contract, and explicit HPC exception. Preserve the ClearEx pin, license, storage policy, Phase 1 API, and live spheroid-run protection.

- [ ] **Step 5: Run and commit policy updates**

```bash
uv run pytest tests/unit/test_repository_policy.py -q
git add .github/workflows/ci.yml AGENTS.md .github/workflows/AGENTS.md \
  examples/AGENTS.md tests/unit/test_repository_policy.py
git commit -m "ci: run portable Phase 2 workflow gates"
```

Expected: policy tests pass and CI contains no lab path, SLURM command, or HPC selector.

---

### Task 4: Add explicit real-SLURM environment and smoke acceptance

**Files:**
- Create: `tests/hpc/AGENTS.md`
- Create: `tests/hpc/__init__.py`
- Create: `tests/hpc/conftest.py`
- Create: `tests/hpc/test_slurm_environment.py`
- Modify: `tests/unit/test_repository_policy.py`

**Interfaces:**
- Consumes: a caller-supplied dataset manifest, SLURM execution profile, and empty evidence directory
- Produces: machine-readable current partition/allocation/mount/network/preflight/scratch/script/cleanup evidence

- [ ] **Step 1: Add the HPC guidance inventory test**

Extend `AGENT_GUIDANCE_PATHS` with `tests/hpc/AGENTS.md`. Require guidance text for explicit selection, no hostname skip, caller-owned evidence, `/archive` large outputs, `/project` scratch/environment, no live spheroid access, job cleanup, and no acceptance claim without real allocation.

- [ ] **Step 2: Add required pytest options and fail-closed fixtures**

Register:

```text
--cyexm-dataset PATH
--cyexm-execution PATH
--cyexm-evidence-dir PATH
```

When an HPC test is selected, require all three options, require the evidence directory to be an existing empty directory outside the repository and live spheroid output, load both YAML files through public APIs, require `executor: slurm`, and write an atomic `acceptance-context.json` with timestamp, git commit, package versions, resolved paths, and CLI arguments.

- [ ] **Step 3: Write current environment and allocation tests**

Use `sinfo`/`scontrol` read-only commands to record partition state, configured CPUs/memory, oversubscription/exclusive-allocation policy, and interface facts. Run a bounded no-registration executor task on `min(2, configured_workers)` jobs. Assert distinct job IDs/hostnames, an exclusive node allocation for each job, shared dataset/output visibility, scratch create/delete, native threads, package versions, memory evidence, one round slot, recorded job script, and no jobs with the recorded name after cluster close. If the partition does not provide exclusive allocations, fail acceptance until the execution profile requests an approved exclusive-node directive and the strict example/profile tests are updated and rerun.

- [ ] **Step 4: Run the explicit smoke acceptance**

```bash
uv run pytest -m hpc tests/hpc/test_slurm_environment.py -v \
  --cyexm-dataset /archive/approved/cyexm/spheroid-dataset.yaml \
  --cyexm-execution /archive/approved/cyexm/execution-slurm-512gb.yaml \
  --cyexm-evidence-dir /archive/approved/cyexm/evidence/environment
```

Expected: every selected test passes on real allocated workers; evidence includes submitted script, partition snapshot, probes, job IDs, hostnames, scheduler address, and cleanup result. Replace `/archive/approved/cyexm` with the operator-approved path before running; never point it at the active spheroid-run directory.

- [ ] **Step 5: Commit the explicit HPC harness after smoke passes**

```bash
git add tests/hpc tests/unit/test_repository_policy.py
git commit -m "test: add explicit SLURM acceptance harness"
```

---

### Task 5: Prove scientific parity, restart, failure isolation, cancellation, and cleanup on SLURM

**Files:**
- Create: `tests/hpc/test_slurm_workflow.py`

**Interfaces:**
- Consumes: approved bounded representative inputs and Phase 2D SLURM executor
- Produces: final one-round-per-node, numerical, durability, and cleanup acceptance evidence

- [ ] **Step 1: Add one-node-per-round and numerical-reference tests**

Run at least two pending rounds, including one `whole_image` and one `serial_chunked`, with worker count at least two. Assert unique SLURM job ID and hostname per concurrently executing round. Run the identical plans sequentially into a separate reference root, then compare transforms, channel dtype/shape/geometry, and numeric arrays using exact comparisons where deterministic and the established `rtol=1e-4`, `atol=1e-5` registration tolerance otherwise.

- [ ] **Step 2: Add independent failure and resume tests**

Use an injected top-level test runner that fails one selected round after another round commits. Assert successful rounds finish, the workflow exits with round-failure status, and a normal rerun reuses committed artifacts while recomputing only incomplete work through ordinary artifact validation.

- [ ] **Step 3: Add cancellation and cleanup tests**

Start bounded long-enough synthetic tasks, trigger coordinator interruption through the test harness, assert exit semantics 130, unfinished future cancellation, cluster closure, no remaining named jobs, committed artifact preservation, cancelled run summary, and successful ordinary resume.

- [ ] **Step 4: Run the complete explicit workflow acceptance**

```bash
uv run pytest -m hpc tests/hpc/test_slurm_workflow.py -v \
  --cyexm-dataset /archive/approved/cyexm/spheroid-dataset.yaml \
  --cyexm-execution /archive/approved/cyexm/execution-slurm-512gb.yaml \
  --cyexm-evidence-dir /archive/approved/cyexm/evidence/workflow
```

Expected: placement, whole-image, serial-chunked, sequential parity, independent failure, cancellation, restart, and final `squeue` cleanup assertions all pass. Evidence records every command/result without committing generated output.

- [ ] **Step 5: Commit the workflow acceptance test after the real run passes**

```bash
git add tests/hpc/test_slurm_workflow.py
git commit -m "test: accept SLURM workflow durability"
```

---

### Task 6: Record final acceptance and run every release gate

**Files:**
- Create: `docs/phase-2-acceptance.md`
- Modify: `README.md`
- Modify: `tests/unit/test_repository_policy.py`

**Interfaces:**
- Consumes: portable and external HPC evidence from Tasks 4-5
- Produces: concise reproducible acceptance audit and final accepted README status

- [ ] **Step 1: Add a failing acceptance-audit policy test**

Require the audit to record git commit under test, Python/Cy-ExM/ANTSpyX/Dask/distributed/dask-jobqueue/SLURM versions, portable test counts/times, profile fingerprint, partition/allocation facts, job IDs/hostnames, submitted-script hash, sequential tolerance, failure/cancellation/restart outcomes, cleanup outcome, and external evidence paths. Forbid embedding large logs, arrays, credentials, or live spheroid output paths.

- [ ] **Step 2: Run every portable gate fresh**

```bash
git diff --check
uv lock --check
uv sync --locked --group dev --extra examples --extra hpc
uv run pytest -m "unit and not hpc" tests/unit -v
uv run pytest -m "integration and not hpc" tests/integration -v
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pre-commit run --all-files
```

Expected: every command exits 0. Capture exact versions, counts, and elapsed times from this run.

- [ ] **Step 3: Revalidate external HPC evidence and scheduler cleanup**

Run the exact Task 4 and Task 5 commands against clean evidence directories, then run a read-only `squeue` query for recorded job names/IDs. Expected: both test modules pass and no owned worker job remains.

- [ ] **Step 4: Write the concise acceptance audit and finalize README status**

Summarize the exact fresh evidence required by the policy test, link the approved design and five plans, identify external evidence paths/checksums, and distinguish portable fake construction from real submission evidence. Update README's acceptance section from pending to the recorded result only after these facts exist.

- [ ] **Step 5: Run policy/docs tests and commit the audit**

```bash
uv run pytest tests/unit/test_repository_policy.py \
  tests/unit/test_phase2_documentation.py tests/unit/test_phase2_examples.py -q
git add docs/phase-2-acceptance.md README.md tests/unit/test_repository_policy.py
git commit -m "docs: record Phase 2 runtime acceptance"
```

Expected: audit, README, example, and repository-policy tests pass.

- [ ] **Step 6: Review the complete Phase 2 implementation**

```bash
git log --oneline --decorate main..HEAD
git diff --stat main...HEAD
git status --short
```

Expected: only approved runtime, tests, documentation, examples, guidance, dependency, and lock changes are present; the worktree is clean. Request code review before integration.
