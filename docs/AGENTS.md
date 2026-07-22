# Documentation guidance

- Describe current behavior by separating merged Phase 1 behavior from the Phase 2D
  portable runtime available on this branch: strict YAML validation, deterministic
  planning and status inspection, real sequential round execution, restart-safe
  transformed channels, the workflow CLI, bounded process-based local Dask execution,
  SLURMCluster construction, and all-worker preflight. Clearly label fake-based
  portable SLURM evidence as construction evidence only; real SLURM submission and
  HPC acceptance remain Phase 2E work until their explicit acceptance gates complete.
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
- Treat `phase2e-multithreaded-registration-reproducibility.md` as scientific
  characterization of the observed ANTs behavior, not a production tolerance or
  seed policy. A final Phase 2E acceptance page may claim both tiers—the
  multithreaded scientific-equivalence gate and seeded one-thread strict-parity
  gate—only after complete external evidence passes for one exact commit.
- Describe CC radius as one scalar voxel radius and identify radius 4 as provisional.
  Do not imply ANTs accepts per-axis radii or that repository defaults are optimized
  for a representative dataset without explicit scientific acceptance evidence.
- Historical ClearEx names and lab consumer paths belong only in migration or parity
  documents. General guides and examples must remain portable and use Cy-ExM names.
- Do not include credentials, private GitHub URLs with tokens, generated evidence,
  dataset-specific output, or large command transcripts.
