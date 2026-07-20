# Workflow package guidance

- Workflow owns immutable plans, tasks, and results; deterministic path and header
  validation; output layout, status, fingerprinting, and public orchestration.
- Planning creates no files, directories, clients, clusters, or workers.
- Phase 1 completion records remain the only restart authority.
- Scientific fingerprints exclude executor, resource, and logging settings.
- Use no module-scope scientific or scheduler imports.
