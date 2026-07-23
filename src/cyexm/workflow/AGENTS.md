# Workflow package guidance

- Phase 2 workflow owns immutable plans, tasks, and results; deterministic path and
  header validation; output layout, status, fingerprinting, and public
  orchestration. Phase 2B binds sequential execution to the real round worker.
- Planning creates no files, directories, clients, clusters, or workers.
- Planning owns one immutable reference-preparation prerequisite. It must preserve
  declared channel order, require identical native reference grids, and point every
  moving task at the prepared registration-channel TIFF under the reference round's
  shared `channels` directory.
- Workflow must not mutate a plan or create a scheduler.
- Fine-grained workflow models embed the authoritative `RoundTask`; do not copy its
  scientific fields into stage tasks. Initial and incremental graph expansion must
  remain pure and use the prepared manifest's fixed-grid ZYX shape with the shared
  canonical chunk grid.
- Fine task workers configure native threads before lazy scientific imports, execute
  exactly one immutable stage, and never create clients, workers, or descendant
  tasks. Concurrent stages use deterministic per-task logs rather than the coarse
  round log.
- `run_workflow` remains the only public execution entry point for coarse and fine
  modes. Validate the exact mode-appropriate summary log: successful fine rounds use
  their validation-task log, while failed or cancelled summaries use the canonical
  representative terminal-task log.
- Persist a strict scheduler-neutral fine evidence envelope for sequential and local
  Dask execution, and validate the corresponding fine-graph extension for SLURM.
  Cross-check public summaries against actual controller task results after
  scientific work; missing or malformed evidence fails closed and never becomes
  scientific reuse authority.
- Treat results derived during the current execution as provisional until the exact
  execution evidence validates. On evidence rejection or malformed cancellation
  evidence, persist only reuse results authenticated before this run; never write a
  rejected partial summary into the durable workflow record.
- Validate successful preparation evidence with the authoritative
  `PreparedManifest` parser before graph expansion. Malformed worker descriptors
  must become ordinary current-attempt failures so configured retries and
  independent-sibling settlement remain intact.
- Fine failure results may retain only task-planned artifacts explicitly accepted by
  the current delegate through the exact Phase 1 artifact-store request boundary and
  still backed by the identical canonical completion record. Never infer partial
  progress from current file identities or thread counts alone; force suppresses all
  reused-artifact evidence.
- Fine chunk assembly preserves the legacy ascending-ID overwrite rule for expected
  overlapping output slices. Scheduling and completion order must not change the
  canonical assembly order or scientific artifact identity.
- Phase 1 completion records remain the only restart authority.
- Status must reconstruct moving-round provenance from the prepared reference TIFF's
  persisted geometry, not the native reference source grid. An unreadable prepared
  reference cannot authorize any moving-round completion chain.
- Status must infer configured native thread counts from recorded completion evidence
  for both reference and moving rounds; use the one-thread default only when no valid
  evidence exists.
- Scientific fingerprints exclude executor, resource, and logging settings.
- Scientific fingerprints and strict persisted plans include effective nonlinear
  metric evidence: CC radius or fixed Mattes bins. Omitted CC radius and explicit 4
  share identity; a different radius invalidates nonlinear descendants, not affine.
- Validate each source TIFF shape independently as positive scalar ZYX evidence.
  Native reference, registration-channel, and dependent-channel shapes may differ
  along every spatial axis; fixed-grid equality is a post-resampling output contract,
  not a planning input requirement.
- Force never bypasses a scientific-plan fingerprint mismatch.
- Forcing the reference task forces every moving descendant; forcing only a moving
  task must preserve reusable reference evidence. Status includes the reference round.
- A failed reference barrier prevents descendant submission, so an executor may
  return only that failed reference result. Record the real reference failure through
  ordinary workflow failure semantics; do not require results for unsubmitted moving
  descendants or mask the failure with a result-cardinality error.
- Use no module-scope scientific or scheduler imports.
- `round_task.py` is the only real registration execution boundary in workflow. It
  configures the task's native threads before its lazy registration/scientific
  imports, executes channels serially, and must not create a Dask client, submit
  child tasks, schedule other rounds, or start multiprocessing.
