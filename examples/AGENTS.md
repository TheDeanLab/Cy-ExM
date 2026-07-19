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
- Phase 1 examples cover whole-image and serial chunked registration only. Do not add
  workflow manifests or parallel execution examples before their separately reviewed
  phase.
