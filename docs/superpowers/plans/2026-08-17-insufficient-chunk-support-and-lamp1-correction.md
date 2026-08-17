# Insufficient Chunk Support and LAMP1 Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable identity-nonlinear fallback for geometrically
insufficient edge chunks, export the confirmed Cell 3 round-1 LAMP1 acquisition,
and complete every currently incomplete LAMP1 registration round.

**Architecture:** The shared chunk stage computes strict common-mask support on the
coarsest nonlinear pyramid lattice before cache lookup. Only canonical insufficient
support authorizes identity; sufficiently supported chunks retain their historical
request identity and generic ANTs failures remain fatal. Publication operations use
a separate focused correction dataset for Cell 3 round 1 while the original dataset
resumes Cell 3 round 3 unchanged.

**Tech Stack:** Python 3.12, NumPy, ANTSpyX 0.6.3, tifffile, nd2 0.11.3,
PyYAML, pytest, Ruff, basedpyright, pre-commit, Cy-ExM workflow CLI, SLURM/Dask.

**Approved design:**
`docs/superpowers/specs/2026-08-17-insufficient-chunk-support-identity-design.md`

## Global Constraints

- Preserve fixed and moving roles. Identity means zero nonlinear displacement after
  the already accepted affine transform.
- Public spacing remains micrometers in XYZ; NumPy volumes and support spans are ZYX.
- Threshold registration masks at `> 0.5`; malformed or nonfinite masks fail closed.
- Never convert a generic ANTs failure into identity without the canonical support
  decision.
- Preserve exact request identity for sufficiently supported historical chunks.
- Keep all scientific imports deferred behind the configured native-runtime boundary.
- Do not edit ClearEx or introduce a ClearEx runtime dependency.
- Keep the original incorrect Cell 3 round-1 TIFFs and scientific plan unchanged.
- New external data artifacts must be staged and validated before atomic promotion.
- Use artifact-backed workflow status rather than SLURM state to claim completion.
- Preserve the user's unrelated `.gitignore` modification.
- Before every `uv` command export exactly:

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  ```

---

## Reuse analysis

- `run_nonlinear_chunk` in `registration/chunk_stages.py` is the existing shared
  Phase 1 and fine-grained nonlinear task boundary. Extend it rather than adding a
  second chunk executor.
- `local_chunk_parameters` is the canonical producer/reader request constructor.
  Extend its optional evidence surface rather than creating a parallel completion
  schema.
- `_serial_chunk_evidence` is the authoritative workflow-status reader. Teach it to
  validate the one conditional evidence mapping rather than weakening general
  request comparison.
- `RegistrationBackend.write_identity_warp` and the existing atomic artifact store
  already own identity serialization and promotion; reuse them exactly.
- `cyexm workflow validate/run/status` remain the only public planning, execution,
  and completion interfaces. No new user-callable API is needed.

## File and responsibility map

- Modify `src/cyexm/registration/_request_policy.py`: versioned support constants,
  strict evidence constructor/parser, and optional local-request evidence.
- Modify `src/cyexm/registration/chunk_stages.py`: support measurement and the
  identity admission/exception boundary.
- Modify `src/cyexm/workflow/_round_provenance.py`: strict recognition of canonical
  conditional fallback evidence.
- Modify `tests/unit/test_chunk_stages.py`: support and producer regressions.
- Modify `tests/unit/test_workflow_status.py`: authoritative reader regressions.
- Modify `tests/unit/test_workflow_api.py` only if its exact request fixture requires
  a conditional-evidence case.
- Create publication-side export/configuration/provenance artifacts under
  `/archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/u2os_lysosomes`
  after the code gate passes.
- Modify publication-side `nd2-tiff-file-correspondence.md` and analysis
  `config/correspondence.csv` only after corrected outputs validate.

---

### Task 1: Canonical support policy and identity producer

**Files:**
- Modify: `tests/unit/test_chunk_stages.py`
- Modify: `src/cyexm/registration/_request_policy.py`
- Modify: `src/cyexm/registration/chunk_stages.py`

**Interfaces:**
- Produces private `ChunkSupportEvidence` with integer counts/spans, factor, policy,
  and `insufficient: bool`.
- Produces private `_chunk_support_evidence(np, backend, region, accuracy)`.
- Extends `local_chunk_parameters(..., insufficient_support=None)` with a strict
  optional mapping; omission preserves byte-for-byte historical parameters.

- [ ] **Step 1: Write failing support-classification tests**

Add tests that construct `(8, 8, 8)` mask images and assert:

```python
thin = np.zeros((8, 8, 8), dtype=np.float32)
thin[:, :, :2] = 1.0
evidence = _chunk_support_evidence(
    np, backend, _region_with_masks(thin, thin), "high"
)
assert evidence.insufficient is True
assert evidence.coarsest_shrink_factor == 4
assert evidence.coarsest_span_zyx == (2, 2, 1)

full = np.ones((8, 8, 8), dtype=np.float32)
evidence = _chunk_support_evidence(
    np, backend, _region_with_masks(full, full), "high"
)
assert evidence.insufficient is False
assert evidence.coarsest_voxels == 8
assert evidence.coarsest_span_zyx == (2, 2, 2)
```

Also assert missing masks return `None`, while wrong shape and nonfinite masks raise
`ChunkOutputValidationError`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_chunk_stages.py -k support -vv
```

Expected: collection/import failure for `_chunk_support_evidence` or missing support
evidence behavior.

- [ ] **Step 3: Implement the minimal support value and calculation**

Keep NumPy supplied by the caller. Derive the shrink factor from the canonical
nonlinear iteration schedule rather than hard-coding the production accuracy name.
The core calculation is:

```python
fixed = _validated_mask_array(np, backend, region.fixed_mask, expected_shape, "fixed")
moving = _validated_mask_array(np, backend, region.moving_mask, expected_shape, "moving")
common = (fixed > 0.5) & (moving > 0.5)
coarse = common[::factor, ::factor, ::factor]
indices = np.where(coarse)
span = (
    (0, 0, 0)
    if not bool(coarse.any())
    else tuple(int(axis.max() - axis.min() + 1) for axis in indices)
)
insufficient = int(coarse.sum()) < 8 or any(value < 2 for value in span)
```

Use built-in integers and tuples in the immutable evidence value so canonical JSON
does not receive NumPy scalar types.

- [ ] **Step 4: Run the focused support tests and verify GREEN**

Run the command from Step 2. Expected: all `support` tests pass.

- [ ] **Step 5: Write failing producer behavior tests**

Add separate tests proving:

```python
artifact = run_nonlinear_chunk(context, thin_region, chunk)
assert backend.nonlinear_calls == 0
assert backend.identity_chunk_ids == [chunk.chunk_id]
request = CompletionRecord.read(store.record_path(artifact.warp_path)).request
assert request.parameters["insufficient_support"]["decision"] == (
    "identity-insufficient-support"
)
```

For sufficient support, assert nonlinear is called and
`"insufficient_support" not in request.parameters`. Configure a backend to raise
`RuntimeError("Registration failed with error code 1")` for sufficient support and
assert the identical exception propagates. Add distinct-message and `ValueError`
propagation cases.

- [ ] **Step 6: Run producer tests and verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_chunk_stages.py \
  -k 'insufficient or sufficient_support or ants_error' -vv
```

Expected: thin-support case calls nonlinear or lacks evidence.

- [ ] **Step 7: Implement conditional request evidence and identity flow**

Add a strict constructor in `_request_policy.py` that emits:

```python
{
    "policy": "common-mask-coarsest-2x2x2-v1",
    "decision": "identity-insufficient-support",
    "fine_common_voxels": fine_count,
    "fine_total_voxels": total_count,
    "coarsest_common_voxels": coarse_count,
    "coarsest_shrink_factor": factor,
    "coarsest_span_zyx": list(span),
}
```

Only add this mapping to local parameters when `insufficient` is true. In
`run_nonlinear_chunk`, calculate evidence before `_local_request`, prefer the
support identity decision over low variance, and preserve the existing atomic writer.
Wrap only `backend.nonlinear`. On exact code-1 `RuntimeError`, re-evaluate the same
support helper and use identity only if the result is insufficient; otherwise use a
bare `raise` to preserve traceback and exception identity.

- [ ] **Step 8: Run producer tests and verify GREEN**

Run the command from Step 6, then all chunk-stage tests:

```bash
uv run python -m pytest tests/unit/test_chunk_stages.py -vv
```

Expected: all pass.

- [ ] **Step 9: Commit Task 1**

Run scoped Ruff and diff checks, then stage only the three Task 1 files. Commit with
a body containing the Reuse analysis above:

```bash
git commit -m "fix: use identity for unsupported edge chunks" \
  -m "Reuse analysis: extends the shared run_nonlinear_chunk and canonical local request; reuses RegistrationBackend.write_identity_warp and the artifact store; adds no public API."
```

---

### Task 2: Authoritative status compatibility

**Files:**
- Modify: `tests/unit/test_workflow_status.py`
- Modify: `src/cyexm/workflow/_round_provenance.py`
- Modify: `src/cyexm/registration/_request_policy.py` if the parser belongs beside
  the constructor.

**Interfaces:**
- Consumes the exact `insufficient_support` mapping from Task 1.
- Produces strict `validate_insufficient_support_evidence(value) -> dict[str, object]`.

- [ ] **Step 1: Write failing reader tests**

Create a completed serial-chunk fixture whose one local completion request adds the
canonical insufficient-support mapping. Assert workflow status remains completed.
Parameterize mutations that must make the round incomplete:

```python
{"policy": "unknown"}
{"decision": "estimated-nonlinear"}
{"coarsest_common_voxels": 8, "coarsest_span_zyx": [2, 2, 2]}
{"coarsest_span_zyx": [2, 2, 1.0]}
{"fine_common_voxels": -1}
{"unexpected": True}
```

Retain an unchanged historical/base request fixture and assert it is accepted.

- [ ] **Step 2: Run reader tests and verify RED**

Run:

```bash
uv run python -m pytest tests/unit/test_workflow_status.py \
  -k insufficient_support -vv
```

Expected: canonical fallback record is rejected before reader support exists.

- [ ] **Step 3: Implement strict conditional request recognition**

Read the completion snapshot first, calculate the historical base parameters, then:

```python
actual = snapshot[0].request.parameters if snapshot is not None else None
fallback = actual.get("insufficient_support") if isinstance(actual, dict) else None
if fallback is not None:
    parameters["insufficient_support"] = validate_insufficient_support_evidence(
        fallback
    )
```

The validator requires exact keys and built-in types, positive totals/factor,
`0 <= fine <= total`, nonnegative coarse count/spans, and the decision implication:
`coarse_count < 8 or any(span < 2)`. It returns the canonical mapping and rejects
unknown fields. The ordinary exact `_request_matches` call remains authoritative.

- [ ] **Step 4: Run reader tests and verify GREEN**

Run the command from Step 2, then:

```bash
uv run python -m pytest tests/unit/test_workflow_status.py \
  tests/unit/test_workflow_api.py tests/unit/test_round_task.py -vv
```

Expected: all pass.

- [ ] **Step 5: Run shared feature gates and commit Task 2**

Run:

```bash
uv run python -m pytest tests/unit/test_chunk_stages.py \
  tests/unit/test_fine_workflow_task.py \
  tests/unit/test_workflow_status.py \
  tests/integration/test_chunked.py \
  tests/integration/test_fine_grained_workflow.py -vv
uv run ruff check src/cyexm/registration/_request_policy.py \
  src/cyexm/registration/chunk_stages.py \
  src/cyexm/workflow/_round_provenance.py \
  tests/unit/test_chunk_stages.py tests/unit/test_workflow_status.py
uv run ruff format --check src/cyexm/registration/_request_policy.py \
  src/cyexm/registration/chunk_stages.py \
  src/cyexm/workflow/_round_provenance.py \
  tests/unit/test_chunk_stages.py tests/unit/test_workflow_status.py
git diff --check
```

Commit only Task 2 files with message `fix: attest unsupported chunk fallback`.

---

### Task 3: Portable repository verification

**Files:**
- No intended source changes; fix only failures caused by Tasks 1–2.

**Interfaces:**
- Consumes the completed implementation.
- Produces fresh verification evidence before production execution.

- [ ] **Step 1: Run the configured portable gates**

Run sequentially:

```bash
uv lock --check
uv sync --locked --group dev --extra examples --extra hpc
uv run pytest -m "unit and not hpc" tests/unit -v
uv run pytest -m "integration and not hpc" tests/integration -v
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pre-commit run --all-files
```

Expected: every command exits zero. If a gate exposes a feature regression, return
to a new RED test before changing production code.

- [ ] **Step 2: Confirm repository scope**

Run `git status --short`, `git diff --check`, and inspect every feature commit.
Confirm the unrelated `.gitignore` change remains unstaged and unmodified.

---

### Task 4: Corrected Cell 3 round-1 export and focused workflow

**Files:**
- Create: publication-side corrected raw directory with six TIFFs for reference
  round 2 and corrected moving round 1.
- Create: publication-side corrected dataset YAML and execution YAML.
- Create: publication-side export provenance JSON containing ND2 identity, channel
  mapping, skipped plane, TIFF identities, and exact comparison result.

**Interfaces:**
- Consumes `DAPI_tubulin_lamp_006.nd2` and the original Cell 3 round-2 TIFF triplet.
- Produces a new one-moving-round workflow dataset and authoritative output root.

- [ ] **Step 1: Snapshot source identities and available capacity**

Record SHA-256, size, and mtime for the ND2 and copied reference TIFF sources.
Validate free space on `/archive` and calculate launch capacity from the new execution
profile before creating outputs.

- [ ] **Step 2: Export into a unique staging directory**

Use `nd2==0.11.3` and `tifffile` in a PEP 723 compliant publication-side script.
Read each frame inside the open ND2 context, copy channel arrays, omit ND2 plane 0,
and write ImageJ-compatible `uint16` ZYX TIFFs with `photometric="minisblack"`.
Never overwrite the original Cell 3 directory.

- [ ] **Step 3: Verify every exported pixel**

Reopen each TIFF and compare its full array exactly to ND2 planes 1:61 for channel
indices Cy5=0, GFP=1, DAPI=2. Require shapes `(60, 2250, 2250)`, `uint16`, and exact
voxel metadata. Write canonical pretty JSON only after all comparisons pass.

- [ ] **Step 4: Copy and verify the round-2 reference triplet**

Copy, do not hard-link or symlink, the original Cell 3 round-2 TIFFs into the staged
correction root. Verify source and destination SHA-256 equality.

- [ ] **Step 5: Atomically promote the correction root**

Rename the fully validated staging directory to its final unique path on the same
filesystem. Preserve the old incorrect triplet in place.

- [ ] **Step 6: Create and validate YAML pair**

The dataset uses acquisition round 2 as reference and declares only moving round 1
with the existing serial chunk geometry and registration policy. The execution
profile uses a new dataset ID, new worker-local directory, and a bounded SLURM
profile calculated from current resources.

Run `cyexm workflow validate` and `cyexm workflow plan`; expected: exit zero and a
new scientific plan with exactly one moving round.

- [ ] **Step 7: Launch and monitor the corrected workflow**

Run `cyexm workflow run` using the validated pair. Monitor scheduler state for
operations, but wait for the command and artifact records. Require `cyexm workflow
status` to report reference round 2 and moving round 1 completed.

---

### Task 5: Resume incomplete original rounds and update provenance

**Files:**
- Modify: publication-side `nd2-tiff-file-correspondence.md`.
- Modify: analysis `config/correspondence.csv` by adding the validated LAMP1 rows,
  with Cell 3 round 1 pointing to the correction dataset.
- Create: a timestamped bounded resume launcher and status log under
  `u2os_lysosomes`.

**Interfaces:**
- Consumes the tested Cy-ExM commit and three validated YAML pairs: original Cell 1,
  original Cell 3, and corrected Cell 3 round 1.
- Produces authoritative completion evidence for Cell 1 round 5, Cell 3 round 3,
  and corrected Cell 3 round 1.

- [ ] **Step 1: Revalidate original plans and status**

Confirm Cells 2, 4, and 5 remain completed. Confirm only Cell 1 round 5 and original
Cell 3 round 3 are incomplete. Verify original Cell 3 source identities still match
its persisted plan.

- [ ] **Step 2: Launch bounded original resumes**

Run the original Cell 1 and Cell 3 YAML pairs without `--force-round`. Their artifact
stores must reuse completed supported chunks, recompute support-identity edge chunks,
and assemble only the incomplete rounds. Do not launch completed cells.

- [ ] **Step 3: Verify authoritative completion**

After commands exit, require workflow status to report all original intended rounds
complete. Inspect the failed-chunk completion records and assert their decision is
`identity-insufficient-support` with the measured counts and spans. Confirm final
moving validity masks exclude unsupported source regions.

- [ ] **Step 4: Update correspondence and analysis configuration**

Replace the provisional Cell 3 note with the user-confirmed `lamp_006` assignment,
exact export evidence, and correction dataset/output paths while retaining the
historical error. Add all 25 LAMP1 rows to `config/correspondence.csv`; use the
correction manifest/profile/dataset ID only for Cell 3 acquisition round 1.

- [ ] **Step 5: Run publication validation and final inventory**

Run the analysis input validator with ND2 pixel checks enabled. Require 25 exact
LAMP1 channel matches and authoritative Cy-ExM completion for every admitted row.
Save a concise Markdown completion report listing code commit, YAML identities,
workflow-run records, fallback chunks, and the corrected source mapping.

