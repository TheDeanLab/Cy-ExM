# Chunked Cache Identity Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make serial chunked registration validate its transitive local evidence before full-warp reuse, while correcting local failure invalidation, crop geometry, float32 assembly, and request/backend metadata agreement.

**Architecture:** `ImageRegistration` remains the owner of the full nonlinear lifecycle and gains protected mode/evidence fields plus an opt-in prepare-before-reuse path. `ChunkedImageRegistration` still overrides only `_estimate_nonlinear`; that hook validates/reuses local artifacts and the canonical manifest, assembles a unique candidate, and publishes immutable evidence used to build the parent full-warp request. The parent always cleans a prepared candidate and leaves a valid cached full record untouched on transient reconstruction failures.

**Tech Stack:** Python 3.12, NumPy, ANTSpyX, pytest, Ruff, BasedPyright, atomic completion records.

## Global Constraints

- Preserve the stable registration API and the inheritance boundary: `ChunkedImageRegistration` overrides only `_estimate_nonlinear` among lifecycle methods.
- Keep unit tests free of real ANTs; place bounded real-ANTs geometry characterization under `tests/integration` with `pytest.mark.integration`.
- Keep all scientific imports lazy in package code.
- Use serial execution only and retain the parent-owned final warp, scalar, mask, metrics, and registered-image lifecycle.
- Use one final correction commit with subject `fix: make chunked cache identity transitive`; do not amend or push.

---

### Task 1: Pin crop-index semantics

**Files:**
- Modify: `tests/unit/fakes.py`
- Modify: `tests/unit/test_backend.py`
- Create: `tests/integration/test_backend_crop.py`
- Modify: `src/cyexm/registration/backend.py`

**Interfaces:**
- Consumes: public ZYX bounds `(z0, z1, y0, y1, x0, x1)`.
- Produces: `crop_image()` origins computed by `ants.transform_index_to_physical_point(image, (z0, y0, x0))` for scalar, mask, and vector images.

- [ ] Add a fake-image `index_to_physical_point()` method and change the unit crop expectation to the direct `(z, y, x)` ANTs index.
- [ ] Add a bounded integration test creating small scalar and mask ANTs images with nontrivial origin, spacing, and direction; compare cropped origins exactly to ANTs' `transform_index_to_physical_point(image, (z0, y0, x0))` and arrays to NumPy ZYX slices.
- [ ] Run the focused unit test and integration test and record their expected failures against the reversed-index implementation.
- [ ] Replace manual reversed-index origin arithmetic with the direct ANTs call; update `FakeBackend.crop_image()` and chunk-id inference consistently.
- [ ] Re-run backend, whole-image, chunked, and integration crop tests.

### Task 2: Make the full-warp request transitive

**Files:**
- Modify: `tests/unit/test_chunked_registration.py`
- Modify: `src/cyexm/registration/whole_image.py`
- Modify: `src/cyexm/registration/chunked.py`

**Interfaces:**
- Consumes: protected `_nonlinear_mode`, `_nonlinear_request_evidence`, and `_prepare_nonlinear_before_reuse` policy owned by `ImageRegistration`.
- Produces: a nonlinear `StageRequest` whose parameters contain mode plus chunk config/grid, canonical manifest `FileIdentity`, and ordered local-warp `FileIdentity` values.

- [ ] Add failing tests proving a whole-image full warp cannot satisfy a chunked run, chunk config changes reject cache reuse, and missing/corrupt/mismatched local artifacts or manifest are detected even when the full warp exists.
- [ ] Add a failing normal-rerun test proving locals and full warp retain identical identities, no local nonlinear backend calls occur, the prepared candidate is removed, and downstream artifacts remain reusable.
- [ ] Add a failing completion-record test asserting `registration_mode == "serial-chunked"`, exact chunk config/grid, manifest identity, and ordered local identities.
- [ ] Add protected whole-image defaults and a helper that builds the nonlinear request after merging mode/evidence.
- [ ] In the parent lifecycle, optionally prepare first, then build/reuse/commit; own the candidate in a `finally` block so request, reuse, commit, and reconstruction exceptions all remove it.
- [ ] In the chunked hook, require existing local/manifest pairs when any full-warp cache entry exists, validate every local and the manifest before publishing evidence, and preserve first-run/resume creation when no full entry exists.
- [ ] Treat an explicitly stale full entry as recoverable, and stale the transitive manifest when a local warp is freshly replaced so force-failure resume can renew local, manifest, and full records.
- [ ] Re-run the focused cache tests and whole-image regression tests.

### Task 3: Invalidate only failed fresh local commits

**Files:**
- Modify: `tests/unit/test_chunked_registration.py`
- Modify: `src/cyexm/registration/chunked.py`

**Interfaces:**
- Consumes: `_load_or_commit_local_warp()` cache/fresh status.
- Produces: `(warp_array, freshly_committed)` and a stale completion record only when postcommit reconstruction/core validation fails for a fresh artifact.

- [ ] Add a failing test proving an invalid scalar after a fresh commit moves the local completion record to `.stale`, then a later run recomputes and succeeds.
- [ ] Add a failing test proving a transient reconstruction failure of a cached local retains its valid record.
- [ ] Return fresh/cache status from local loading and mark only a fresh local artifact stale when subsequent reconstruction, cast, finiteness, or placement fails.
- [ ] Re-run focused restart tests.

### Task 4: Enforce float32 cores and metadata agreement

**Files:**
- Modify: `tests/unit/fakes.py`
- Modify: `tests/unit/test_chunked_registration.py`
- Modify: `src/cyexm/registration/chunked.py`

**Interfaces:**
- Consumes: scalar and displacement core arrays from local reconstruction/cache.
- Produces: exact `numpy.float32` scalar/warp cores with post-cast finiteness checks and exact float32 full-candidate rereads.

- [ ] Extend the fake backend narrowly to emit float64/overflow scalar and warp arrays and non-float32 full-candidate rereads.
- [ ] Add failing tests for finite-float64 values that overflow during float32 conversion, a non-float32 full candidate, and exact equality between recorded and actual local `verbose` values.
- [ ] Extract each core, cast under an overflow-suppressed NumPy context, reject nonfinite post-cast values, and place only the float32 core.
- [ ] Require full-candidate reread dtype to equal `np.float32` exactly and set the local backend `verbose` argument to the recorded value.
- [ ] Re-run focused dtype and metadata tests.

### Task 5: Verify and commit

**Files:**
- Modify: `.superpowers/sdd/task-7-report.md` if the ignored report remains the Task 7 handoff mechanism.

**Interfaces:**
- Consumes: all corrected implementation and tests.
- Produces: one clean commit on top of `2e2a933620813465baef041f03242df47c7d88b7`.

- [ ] Run targeted backend/chunked/whole-image tests, the bounded integration crop test, the full unit suite, Ruff, BasedPyright, and the fresh import contract with `/project` uv cache/Python locations exported for every `uv` command.
- [ ] Remove only generated project caches, inspect the diff for architecture/API regressions, and update the Task 7 report with exact evidence.
- [ ] Commit tracked source/tests with subject `fix: make chunked cache identity transitive` without amending or pushing.
- [ ] Confirm the worktree is clean and report commit hash, verification counts, and any residual concerns.
