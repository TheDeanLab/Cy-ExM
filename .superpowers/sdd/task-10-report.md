# Task 10 Report: Real-ANTs Fine Scientific Regressions

## Outcome

Task 10 adds bounded portable real-ANTs evidence for the complete fine-grained
scientific path. Seeded one-thread coarse and fine execution now compare the
affine, every local warp and scalar core, assembled warp, diagnostic TIFF, all
three metrics, and three transformed moving channels on a four-chunk legacy grid.
The fixed, registration, and two dependent-channel sources differ along Z, Y, and
X, while every prepared, nonlinear, diagnostic, and channel product is required to
land on the exact fixed-reference grid.

The new regression exposed and fixed one genuine fine-worker defect: prepared
regional reconstruction discarded the legacy fixed image's ANTs pixel type. An
integer reference therefore produced floating scalar resampling in fine mode while
coarse legacy execution retained integer resampling. Transform fields were bitwise
identical, but scalar cores differed by up to 3.25 intensity units and the uint16
diagnostic differed by up to 4.

## Scientific defect and repair

Root-cause evidence established:

- coarse and fine affine parameters were identical;
- all four corresponding local warps were bitwise identical;
- prepared and legacy moving-linear regional arrays were float32 and
  sample-for-sample identical;
- first-run and cached-affine coarse nonlinear reruns produced bitwise-identical
  integer scalar cores and diagnostics; and
- the only material boundary difference was the fixed regional ANTs pixel type:
  legacy coarse used `unsigned int`, while fine reconstruction silently used
  `float`.

The internal prepared manifest is now format version 2 and records the exact
validated ANTs pixel type for fixed scalar, moving-linear scalar, fixed mask, and
moving-linear mask. Per-array and manifest completion requests bind those values.
Fine workers reconstruct each regional image with its recorded type and fail closed
on missing, unsupported, or request-mismatched evidence. The `.npy` storage contract
remains uncompressed C-contiguous float32.

Real-ANTs regressions pin both sides of the boundary:

- a uint16-derived workflow reference reconstructs as ANTs `unsigned int` and
  matches legacy coarse scalar quantization; and
- a floating direct prepared reference remains ANTs `float` with fractional
  samples, preventing a blanket integer cast.

Legacy scientific behavior remains unchanged: affine fixed-grid resampling still
precedes nonlinear chunking, chunks retain the canonical legacy definitions, and
ascending canonical chunk IDs still overwrite earlier IDs in overlaps.

## Portable scientific evidence

- Four nontrivial nonlinear chunks and three moving channels use exact
  `NativeThreadSpec(1, 1)` and seed `20260719`.
- The `dry run` accuracy label selects the bounded iteration schedule but still
  executes real `ants.registration` and transform application. The test requires a
  nonidentity affine and at least one nonzero displacement field.
- Fixed `(16, 32, 32)`, registration `(14, 29, 28)`, signal `(15, 30, 29)`, and
  label `(13, 31, 30)` inputs differ on every spatial axis.
- Numerical comparisons retain the repository tolerance
  `rtol=1e-4`, `atol=1e-5`; no tolerance was weakened.
- Production assembly receives the same four valid artifacts in forward, reverse,
  and deterministic `(2, 0, 3, 1)` completion order. Full warp and diagnostic
  arrays are identical under the approved ascending-ID overwrite rule.
- A staged interruption after two committed chunks preserves all artifact and
  completion-record identities across affine/metrics, the prepared bundle, and both
  chunks. Resume pins preparation and chunks 0/1 as reused, chunks 2/3 and assembly
  as completed, and interruption after completed channel 0 preserves the entire
  accepted chain exactly. Final resume pins preparation, all chunks, assembly, and
  channel 0 as reused; only channels 1/2 and fresh full-chain validation complete.
- Every corresponding coarse/fine scalar-core array is bitwise identical with exact
  float32 dtype and canonical core shape. Its completion request is bound to the
  exact local warp identity, regional fixed geometry, scalar policy, and native
  thread evidence.
- Process Local Dask compares sequential affine, every local warp/core, full warp,
  all prepared arrays, metrics, diagnostic, transformed channels, geometry, and
  authoritative requests. Two chunk workers have distinct PIDs, each uses one slot,
  and the guarded worker forbids nested clients, submissions, and executor imports.
- Coarse and fine plans retain byte-identical scientific JSON and fingerprints;
  ordinary completion records remain the only reuse authority.

## Guidance

Registration, workflow, unit-test, and integration-test `AGENTS.md` files now bind
prepared pixel-type evidence, fail-closed validation, integer/float regressions,
four-chunk/three-channel parity, completion-order permutations, and restart
identity requirements.

## Verification

- Focused Task 10 real-ANTs suite: 9 passed in 39.08 seconds.
- Full portable unit suite: 2,266 passed in 29.62 seconds.
- Full portable integration suite: 61 passed in 48.88 seconds.
- `uv lock --check`: passed.
- Repository Ruff check and format check: passed.
- Repository BasedPyright: 0 errors, warnings, or notes.
- `pre-commit run --all-files`: passed.
- `git diff --check`: passed.

No SLURM jobs were submitted, no archive or active scientific datasets were used,
and the unrelated `src/cy_exm.egg-info/` tree was left untouched.
