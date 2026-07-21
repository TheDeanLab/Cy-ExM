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
- Phase 2 manifest, execution-profile, planning, sequential-executor, workflow, and
  CLI policy tests use plain Python data, narrow fakes, and temporary files. Import
  contracts must prove channel and round-task modules defer scheduler and scientific
  runtimes; real registration remains in integration tests. Scheduler adapter
  behavior starts at an approved integration boundary.
- Scheduler-firewall regressions use inert scheduler shims ahead of host executables
  so RED tests never reach host scheduler clients. Exercise direct, wrapped, shell,
  child-Python, and representative standard-library launch surfaces with safe
  sentinels; never test a missed guard by allowing an actual SLURM client call.
- Reference preparation tests must prove one shared crop is applied identically to
  every declared channel, exact-grid mismatches fail before writes, independently
  valid commits survive later failures, and restart/force behavior is evidence-based.
- Cropped status regressions must use a nontrivial prepared-reference crop and fixed
  and moving native shapes that differ along Z, Y, and X so the native source grid
  cannot accidentally satisfy prepared-grid provenance.
- Status tests must cover non-default native thread evidence on the reference round;
  the scientific plan intentionally excludes execution-profile thread settings.
- Executor tests must make the reference barrier observable: reference submission
  precedes moving submissions, failure prevents descendant submission, and one owned
  client/cluster lifecycle spans both stages.
