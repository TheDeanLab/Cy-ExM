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
- Phase 1 completion records remain the only restart authority.
- Scientific fingerprints exclude executor, resource, and logging settings.
- Validate each source TIFF shape independently as positive scalar ZYX evidence.
  Native reference, registration-channel, and dependent-channel shapes may differ
  along every spatial axis; fixed-grid equality is a post-resampling output contract,
  not a planning input requirement.
- Force never bypasses a scientific-plan fingerprint mismatch.
- Forcing the reference task forces every moving descendant; forcing only a moving
  task must preserve reusable reference evidence. Status includes the reference round.
- Use no module-scope scientific or scheduler imports.
- `round_task.py` is the only real registration execution boundary in workflow. It
  configures the task's native threads before its lazy registration/scientific
  imports, executes channels serially, and must not create a Dask client, submit
  child tasks, schedule other rounds, or start multiprocessing.
