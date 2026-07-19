# Phase 1 acceptance audit

## Audit identity

- **Audit date:** 2026-07-19
- **Committed implementation baseline under audit:**
  `3768882870f1b163fbbef8c8447c409d98c44394` (`fix: normalize notebook crop
  origin`), committed 2026-07-19 at 12:25:53 -05:00.
- **Acceptance-gate commit:** this document is committed as part of the acceptance
  gate, and a commit cannot contain its own commit hash. The resulting gate commit is
  identified by repository history and the audit handoff rather than by inventing a
  future SHA.

The audit used the locked project environment on Linux and exercised only portable
unit and bounded integration work. It did not select HPC tests or access lab storage.

## Runtime and tool versions

| Component | Measured version | Provenance |
| --- | --- | --- |
| Python | 3.12.11 | CPython in the uv-managed project environment |
| ANTSpyX | 0.6.3 | Installed `antspyx` distribution and `ants.__version__` |
| ITK | 5.4.5 | `/ITKVersion` in an HDF5 image written by the installed ANTSpyX native binary |
| NumPy | 2.3.5 | Installed `numpy` distribution |
| SciPy | 1.15.3 | Installed `scipy` distribution |
| tifffile | 2026.7.14 | Installed `tifffile` distribution |
| uv | 0.7.13 | `uv --version`; required by project and CI configuration |
| pytest | 9.1.1 | `uv run pytest --version` |
| Ruff | 0.15.22 | `uv run ruff --version` |
| BasedPyright | 1.38.3 (Pyright 1.1.408) | `uv run basedpyright --version` |
| pre-commit | 4.6.0 | `uv run pre-commit --version` |

ITK is not installed as a standalone `itk` or `SimpleITK` Python distribution in
this environment. The recorded version therefore comes from the linked implementation
used by ANTSpyX itself, not from unrelated package metadata.

The required uv version is pinned to exactly 0.7.13 in both
`[tool.uv].required-version` and the setup-uv action input. This makes the measured
resolver and lock behavior reproducible locally and in CI.

## Portable validation results

All commands below ran on 2026-07-19 with uv's cache and managed-Python locations on
`/project` as required by repository policy.

| Command | Fresh result |
| --- | --- |
| `uv lock --check` | Exit 0; 133 packages resolved |
| `uv sync --locked --group dev --extra examples` | Exit 0; 133 packages resolved and 130 packages audited |
| `uv run pytest -m "unit and not hpc" tests/unit -v` | 416 passed, 0 skipped, in 5.61 s |
| `uv run pytest -m "integration and not hpc" tests/integration -v` | 36 passed, 0 skipped, in 4.04 s |
| `uv run ruff format --check .` | Exit 0; 58 files already formatted |
| `uv run ruff check .` | Exit 0; all checks passed |
| `uv run basedpyright` | Exit 0; 0 errors, 0 warnings, 0 notes |
| `uv run pre-commit run --all-files` | 4 of 4 hooks passed |
| `uv run pytest tests/unit tests/integration -q` | 452 passed in 7.41 s |

`uv run pre-commit validate-config .pre-commit-config.yaml` also exited zero. The
pre-commit configuration and GitHub workflow both parsed as YAML using the already
locked environment; no YAML package was added as a direct Cy-ExM dependency. The CI
action refs were confirmed to exist: `actions/checkout@v7` and
`astral-sh/setup-uv@v8.3.2`.

CI fixes all six native-thread variables at the job level before Python starts:
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`, `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and
`VECLIB_MAXIMUM_THREADS`. Its selectors are explicitly `unit and not hpc` and
`integration and not hpc`; the workflow contains no ClearEx, Dask, SLURM, lab path,
or lab-storage access.

Collection-time policy requires every test to have exactly one `unit`, `integration`,
or `hpc` marker matching its first directory. Fresh collection selected all 416 unit
tests with `-m unit`, all 36 bounded integration tests with `integration and not hpc`,
and all 452 portable tests together. The marked module inventory was 16 unit modules
and 6 integration modules.

## Acceptance evidence

- **Legacy parity:** the [legacy parity report](legacy-parity-report.md) records the
  ClearEx comparison, tolerances, source pin, and characterized corrections.
- **Import side effects:** all four tests in `test_import_contract.py` passed. They
  confirmed lightweight package/runtime/registration imports, unchanged environment
  and root logging state, no filesystem work during construction, and no eager ANTs,
  NumPy, SciPy, or tifffile import.
- **Thread defaults and overrides:** all six tests in `test_threads.py` passed. They
  confirmed one thread for every native runtime by default, explicit values over
  inherited values, inherited values over defaults per variable, stable reporting,
  idempotent reapplication, and rejection of a late change.
- **Affine resume:** the focused unit test proved that an affine artifact survives an
  injected nonlinear failure and is reused. The bounded real-library integration test
  `test_resume_reuses_affine_after_injected_nonlinear_failure` also passed.
- **Chunk resume:** the focused unit test
  `test_resume_reuses_completed_chunk` passed, proving a valid completed chunk is
  reused after a later chunk fails.
- **Examples migrated:** the exact inventory test passed for five parameterized
  scripts and six cleared notebooks. The all-files policy checks also confirmed that
  examples use Cy-ExM and contain no lab paths, ClearEx imports, Dask, SLURM, or YAML
  workflow configuration.
- **Forbidden dependencies and exports:** executable audits confirmed that
  `cyexm.registration` does not expose `ParallelChunkedImageRegistration` and that
  Cy-ExM declares no direct ClearEx, Dask, distributed, dask-jobqueue, or PyYAML
  dependency. The prescribed source search returned ten hits; inspection found only
  negative tests and `AGENTS.md` policy text, with no runtime import or dependency.

The focused import, thread, inventory, affine-resume, and chunk-resume audit command
collected and passed all 13 selected tests in 1.03 s.

## Deferred Phase 2 scope

YAML dataset manifests, Dask or distributed execution, SLURM orchestration, and
parallel chunk registration remain deferred to a separately designed Phase 2. No
placeholder implementation or runtime dependency for that scope was added in Phase 1.
