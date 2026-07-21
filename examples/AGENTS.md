# Example guidance

- Examples must be runnable teaching material built only on Cy-ExM and its declared
  dependencies. Do not import test helpers, ClearEx utilities, schedulers, or lab-only
  packages.
- Keep datasets and destinations caller-owned. Never embed lab paths, download data,
  assume a repository-relative dataset, or commit generated registration artifacts.
- Express public voxel spacing in micrometers as `(x, y, z)` even though NumPy volume
  axes are ordered `(z, y, x)`. Name both conventions where they meet.
- Construct `NativeThreadConfig` explicitly before a scientific runtime loads. Keep
  the conservative one-thread defaults visible and explain that larger values require
  dataset-specific validation.
- Preserve fixed/moving roles, affine-before-nonlinear estimation, warp-then-affine
  application order, scalar linear interpolation, and label nearest-neighbor
  interpolation.
- Notebooks must have no saved outputs or execution counts. Scripts and notebooks
  must remain useful on local workstations without cluster services.
- Phase 2 is the current workflow runtime: strict YAML validation,
  deterministic planning, real sequential round execution, restart-safe transformed
  channels, the workflow CLI, and bounded process-based local Dask execution. The
  SLURMCluster adapter is implemented behind portable construction and preflight gates.
  Examples never lead production behavior. Keep each scientific manifest separate from
  its local or HPC execution profile. Real cluster claims require explicitly selected
  HPC acceptance with operator-approved inputs and allocation.
- The four current workflow examples are `examples/spheroid-dataset.yaml`,
  `examples/execution-sequential.yaml`, `examples/execution-local-dask.yaml`, and
  `examples/execution-slurm-512gb.yaml`. Keep them strict-loader compatible and preserve
  their separation between reusable scientific settings and editable machine resources.
