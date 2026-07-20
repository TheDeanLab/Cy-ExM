# Documentation guidance

- Describe current behavior by separating merged Phase 1 behavior from the Phase 2A
  runtime available on this branch: strict YAML validation, deterministic planning
  and status inspection, injected sequential execution, and the workflow CLI.
  Clearly label the real round runner, transformed channels, Dask, SLURM, and HPC
  acceptance as later Phase 2 milestones until their implementation and acceptance
  gates are complete.
- Design and plan documents may describe later Phase 2 decisions when they label
  status and assumptions explicitly. User guides and API examples describe only the
  runtime available on the referenced branch.
- Keep API examples synchronized with signatures in `src/cyexm`. Use public XYZ
  spacing, explicit native-thread configuration, and exact artifact filenames.
- Artifact and restart claims must follow the implementation: a data file alone is
  not a valid cache entry, each artifact has its own completion record, and stale
  records authorize recomputation rather than reuse.
- Treat transform order, interpolation, axes, masks, schedules, and tolerance values
  as scientific statements. Link evidence or explain the characterization behind
  them instead of simplifying them into ambiguous prose.
- Historical ClearEx names and lab consumer paths belong only in migration or parity
  documents. General guides and examples must remain portable and use Cy-ExM names.
- Do not include credentials, private GitHub URLs with tokens, generated evidence,
  dataset-specific output, or large command transcripts.
