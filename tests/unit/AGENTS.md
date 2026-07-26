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
- Metric-policy unit tests must compare complete normalized plans and request
  dictionaries, prove strict scalar-radius typing, and prove radius changes preserve
  affine evidence while invalidating nonlinear descendants.
- Executor tests must make the reference barrier observable: reference submission
  precedes moving submissions, failure prevents descendant submission, and one owned
  client/cluster lifecycle spans both stages.
- Fine executor tests must inject malformed task identity, native-thread order and
  values, timestamps, artifacts, logs, and prepared manifests before controller
  settlement. Every adapter must convert rejection into one exact failed attempt,
  release no descendants, and continue independent siblings according to its retry
  policy.
- Prepared-manifest tests must fail closed on missing, extra, unsupported, or
  request-mismatched ANTs pixel types for every fixed scalar, moving scalar, and mask
  role; fake floating images must not imply that integer fixed grids are equivalent.
- Documentation/example policy tests load every checked-in YAML through production
  loaders, enforce explicit coarse/fine modes, and keep layered AGENTS ownership and
  the fine-grained real-SLURM pending boundary synchronized.
- Fine-HPC portable tests validate report builders and exact decoders with synthetic
  external evidence only. Cover forged identities, checksums, paths, geometry,
  placements, fairness, durability, cancellation, replacement, timing, and the 20%
  nonlinear wall-time gate without contacting a scheduler or terminating a process.
- Preserve exact prepared format-2 keys when preprocessing is disabled. Format-3
  tests must reject policy, sigma, scale, descriptor, completion-record, and source
  identity mismatches and prove regional workers do not load raw full volumes.
