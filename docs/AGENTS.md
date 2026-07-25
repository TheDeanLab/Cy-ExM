# Documentation guidance

- Describe current behavior by separating the accepted coarse Phase 2E runtime from
  opt-in fine-grained execution. Portable real-ANTs/Local-Dask fine evidence is
  complete through Task 10; fine-grained real SLURM production acceptance remains
  pending Tasks 12-13. Never generalize the accepted coarse Phase 2 audit to fine
  placement, failure, cancellation, cleanup, or performance.
- Design and plan documents may describe later Phase 2 decisions when they label
  status and assumptions explicitly. User guides and API examples describe only the
  runtime available on the referenced branch.
- Fine-grained examples must remain exact-schema loader compatible and declare mode
  explicitly: sequential is coarse, while Local Dask and the editable 512-GB SLURM
  example are fine-grained. Describe one task per slot/node today and multiple
  workers per node only as a future optimization.
- Keep API examples synchronized with signatures in `src/cyexm`. Use public XYZ
  spacing, explicit native-thread configuration, and exact artifact filenames.
- Artifact and restart claims must follow the implementation: a data file alone is
  not a valid cache entry, each artifact has its own completion record, and stale
  records authorize recomputation rather than reuse.
- Prepared fixed-grid arrays are fine-only, while registered scalar cores are shared
  coarse/fine serial-chunk artifacts committed by the common chunk stage. Rejected
  fine execution evidence omits unauthenticated current-run summaries from durable
  rounds while retaining pre-run authenticated reuse and restartable Phase 1 commits.
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
- Fixed-affine chunk-size characterization claims must identify the accepted source
  state, ZYX chunk geometry, overlap, nonlinear metric and radius, regional
  confidence or boundary status, seam behavior, and deformation validity. Keep
  dataset-specific recommendations distinct from production defaults.

## API documentation synchronization

When a user-callable API changes, follow `agentic-api-documentation.md`. Keep its
module API map, `See Also` decision boundaries, canonical examples, and relevant
user-facing pages synchronized with the implemented signature and behavior. Use
examples that preserve Cy-ExM's fixed/moving roles, public XYZ spacing, and NumPy ZYX
volume order.

Registration-preprocessing documentation must preserve the
raw → crop → feature estimate → transform raw boundary. State that masks and metrics
are raw-derived, sigmas are NumPy ZYX voxel counts, and feature images are never
public TIFFs.
