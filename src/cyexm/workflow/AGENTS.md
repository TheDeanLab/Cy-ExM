# Workflow package guidance

- Phase 2 workflow owns immutable plans, tasks, and results; deterministic path and
  header validation; output layout, status, fingerprinting, and public
  orchestration. Phase 2B binds sequential execution to the real round worker.
- Planning creates no files, directories, clients, clusters, or workers.
- Workflow must not mutate a plan or create a scheduler.
- Phase 1 completion records remain the only restart authority.
- Scientific fingerprints exclude executor, resource, and logging settings.
- Force never bypasses a scientific-plan fingerprint mismatch.
- Use no module-scope scientific or scheduler imports.
- `round_task.py` is the only real registration execution boundary in workflow. It
  configures the task's native threads before its lazy registration/scientific
  imports, executes channels serially, and must not create a Dask client, submit
  child tasks, schedule other rounds, or start multiprocessing.
