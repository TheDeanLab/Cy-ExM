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
- Fine-grained task models and scheduler-neutral evidence remain internal to the
  workflow/execution namespaces. Do not export prepared-array, chunk-stage,
  dispatcher, worker-placement, or scheduler-lifecycle internals from the package
  root or stable `cyexm.registration` API.

## User-callable API documentation

For a user-callable entry point, resource owner, data model, executor, or explicit
extension interface, follow `docs/agentic-api-documentation.md` and provide full
Numpydoc. Simple pure value objects and helpers may use concise Numpydoc.

An internal symbol is reusable only when it is explicitly designated as an extension
interface and documented as such; do not turn implementation details into public
contracts merely to share code. Before adding a user-callable symbol, record the
closest existing symbols and why they cannot be safely reused in the required `Reuse
analysis` record.
