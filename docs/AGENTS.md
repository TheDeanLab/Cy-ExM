# Documentation guidance

- Describe current behavior separately from proposed work. Label the YAML manifest,
  round-level scheduler, and cluster execution as Phase 2; never present them as
  installable Phase 1 features.
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
