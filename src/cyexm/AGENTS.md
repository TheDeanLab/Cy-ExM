# Package guidance

- Keep `cyexm.__init__` limited to lightweight package metadata. Stable registration
  symbols belong in `cyexm.registration`, not at the package root.
- The Phase 1 registration API is intentionally narrow: `ImageRegistration`,
  `ChunkedImageRegistration`, and `register_round`, plus the low-level linear and
  nonlinear registration entry points in their own modules.
- Never export `ParallelChunkedImageRegistration`. Additions to the public API require
  a separately reviewed need, and new parameters on compatibility APIs are
  keyword-only.
- Keep modules focused on one responsibility. Split I/O, runtime, metrics, artifact
  lifecycle, backend adaptation, registration kernels, and orchestration rather than
  rebuilding a monolithic registration module.
- Prefer explicit typed value objects and narrow internal interfaces. Keep the ANTs
  backend adapter internal so unit tests can exercise policy without a native runtime.
- The configuration, workflow, execution, and CLI namespaces implement the Phase 2
  workflow surface and must not become package-root exports. Scheduler concerns do
  not belong in `cyexm.registration`; keep the stable Phase 1 registration API and
  the Phase 2B channel primitive usable without Dask, distributed, dask-jobqueue, or
  YAML workflow dependencies.
