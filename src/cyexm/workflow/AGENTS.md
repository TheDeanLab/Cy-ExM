# Workflow package guidance

- Phase 2A workflow owns immutable plans, tasks, and results; deterministic path and
  header validation; output layout, status, fingerprinting, and public
  orchestration.
- Planning creates no files, directories, clients, clusters, or workers.
- Workflow must not mutate a plan or create a scheduler.
- Phase 1 completion records remain the only restart authority.
- Scientific fingerprints exclude executor, resource, and logging settings.
- Force never bypasses a scientific-plan fingerprint mismatch.
- Use no module-scope scientific or scheduler imports.
