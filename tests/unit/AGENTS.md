# Unit-test guidance

- Exercise pure policy with real standard-library objects or narrow fakes. Small
  in-memory ANTs images are allowed only when a concrete image, geometry, or TIFF
  boundary contract cannot be represented honestly by a fake; never invoke ANTs
  registration or require network, cluster, or lab-data access.
- Give each test one observable behavior and an outcome-oriented name. Keep fixtures
  local unless several unit modules genuinely share them.
- Use `tmp_path` for filesystem behavior and assert both the intended artifact and
  cleanup of temporary or stale state.
- Keep import-contract checks in fresh subprocesses and explicitly assert that ANTs,
  NumPy, and SciPy remain absent when testing lightweight namespaces.
