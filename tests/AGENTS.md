# Test-suite guidance

- Mark tests as `unit`, `integration`, or `hpc`: unit tests are isolated or mocked;
  integration tests are bounded checks using real scientific libraries; HPC tests
  require lab storage, representative data, or cluster services.
- Keep synthetic 3D fixtures small and deterministic. Use temporary directories for
  artifacts and inject failures at stage boundaries to prove restart behavior.
- Document a scientific reason for each floating-point tolerance. Assert exact values
  for pure geometry and metadata where numerical variability is not expected.
- Portable unit and integration tests must not access lab storage or silently skip by
  hostname. HPC tests require a documented, explicitly selected command.
