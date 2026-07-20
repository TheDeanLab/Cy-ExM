# GitHub Actions workflow guidance

- Preserve Python 3.12, the repository-pinned uv version, and
  `uv sync --locked --group dev --extra examples`; CI must never regenerate the lock.
- Set all six native-thread variables to `1` at job scope before Python starts:
  `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`, `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and
  `VECLIB_MAXIMUM_THREADS`.
- Keep explicit portable selectors: `unit and not hpc` for `tests/unit` and
  `integration and not hpc` for `tests/integration`. Retain a finite job timeout.
- Do not access lab storage, submit SLURM jobs, connect to external Dask clusters, or
  make a portable check conditional on hostnames. Bounded process-based LocalCluster
  integration tests remain portable. Future HPC gates require an approved lab runner
  or an explicit manual cluster command.
- Keep workflow permissions read-only unless a reviewed job has a documented need for
  one additional permission. Never expose secrets to pull requests from untrusted
  code.
