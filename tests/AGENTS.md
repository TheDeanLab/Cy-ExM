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
- Portable unit and integration tests must not access lab storage or silently skip by
  hostname. HPC tests require a documented, explicitly selected command.
