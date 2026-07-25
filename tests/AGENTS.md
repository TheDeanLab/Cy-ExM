# Test-suite guidance

- Mark tests as `unit`, `integration`, or `hpc`. Unit tests are isolated or mocked,
  except for the narrow concrete-image exception defined by `tests/unit/AGENTS.md`;
  real registration belongs in integration. Tests that require representative lab
  data, real SLURM submission, or access to external scheduler or cluster services
  are explicitly selected as `hpc`. Deterministic bounded in-process or
  local-scheduler checks remain portable integration when permitted by
  `tests/integration/AGENTS.md`.
- Keep synthetic 3D fixtures small and deterministic. Use temporary directories for
  artifacts and inject failures at stage boundaries to prove restart behavior.
- Document a scientific reason for each floating-point tolerance. Assert exact values
  for pure geometry and metadata where numerical variability is not expected.
- Any change to registration-grid behavior must include fixed and moving native
  volumes whose shapes differ along all three spatial axes. Prove the affine result
  enters nonlinear work on the fixed grid and assert fixed-reference geometry for
  persisted scalar, mask, warp, and dependent-channel products.
- Nonlinear metric changes require pure policy tests, request/provenance invalidation
  tests, and a bounded real-ANTs integration test. Dataset-specific radius selection
  remains explicitly selected HPC/scientific acceptance.
- Portable unit and integration tests must not access lab storage or silently skip by
  hostname. HPC tests require a documented, explicitly selected command.
- The portable scheduler firewall covers semantic command positions for subprocess,
  `os.system`, and representative `os.exec*`/`os.spawn*` launch surfaces, propagates
  into ordinary child Python interpreters, and prepends owned SLURM-client shims for
  contained child shells. It is defense in depth for the exercised portable surfaces,
  not a claim that arbitrary process-launch mechanisms are fail-closed. HPC-marked
  tests are explicitly exempt and retain the environment needed for approved real
  scheduler acceptance.
- Fine-grained portable coverage proves the scheduler-neutral graph, real-ANTs
  sequential/Local-Dask parity, unequal spatial shapes, worker isolation, failure,
  cancellation, cleanup, and restart without external scheduler contact. It must not
  be described as real-SLURM placement or production acceptance.
- Registration-feature changes require exact kernel tests, raw-versus-feature
  boundary tests for whole/serial/fine execution, cache invalidation coverage, and a
  bounded real-ANTs regression proving raw-scale fixed-grid output.
