# Phase 2E stochastic registration parity correction

## Status and decision

The approved Phase 2E acceptance design required independently estimated sequential
and SLURM registration arrays to satisfy `rtol=1e-4` and `atol=1e-5`. Real-cluster
evidence demonstrated that this requirement tests multithreaded ANTs repeatability,
not the Cy-ExM executor boundary. The replacement is a two-tier gate: representative
multithreaded scientific equivalence plus bounded deterministic executor parity.

This correction changes only the explicit HPC acceptance harness, its portable
unit characterization, and acceptance documentation. It does not change production
registration policy, schema version 1, artifact cache identity, default thread
counts, or the approved CC-radius experiment.

## Evidence and root cause

The failed run used the same inputs, scientific fingerprint, and eight-thread native
policy in both executors. Its first numerical divergence occurred in the TRSAA
affine transform. Independent eight-thread SLURM repetitions also differed. A
bounded experiment produced different affine parameters with eight threads both
without and with a fixed seed, but exactly identical parameters with one thread and
the same seed.

The scientific record is maintained in
[`docs/phase2e-multithreaded-registration-reproducibility.md`](../../phase2e-multithreaded-registration-reproducibility.md).

## Tier 1: representative multithreaded scientific equivalence

Tier 1 retains the approved Zenodo dataset, the modern schema, the approved 8/8
native-thread policy, at least two moving rounds, and distinct one-node-per-round
SLURM placement. Sequential and SLURM executions remain independent estimates.

The gate requires:

1. identical scientific fingerprints, task order, round strategies, channel source
   identities, interpolation choices, and normalized registration policy;
2. exact artifact shape, dtype, origin, spacing, direction, and component structure;
3. valid affine and warp artifacts, finite affine parameters, finite displacement
   values, exact affine transform type/dimension/fixed parameters, and exact warp
   geometry;
4. finite pre-, affine-, and nonlinear-stage correlation and Mattes mutual
   information values for each moving round;
5. an absolute sequential-versus-SLURM difference no greater than `0.01` for each
   corresponding stage metric;
6. no degradation from the pre-registration endpoint to the nonlinear endpoint in
   either execution; and
7. the existing placement, independent-failure, restart, cancellation, artifact
   preservation, output ownership, and exact-name scheduler cleanup requirements.

The harness records, but does not threshold, affine-parameter differences, warp
difference summaries, registered-image difference summaries, and transformed-channel
difference summaries. Each summary includes equality, maximum absolute difference,
mean absolute difference, and a high-percentile absolute difference. These values are
scientific characterization, not deterministic pass criteria.

The `0.01` metric bound is dataset-specific acceptance policy. The available repeated
eight-thread evidence varied by less than `0.004`; the wider bound avoids fitting the
gate to a handful of repeats while remaining small on the metrics' observed scale.

## Tier 2: deterministic executor parity

Tier 2 creates small deterministic scalar TIFF inputs inside the caller-owned Task 5
root. It builds an in-memory schema-version-1 manifest with one whole-image and one
serial-chunked moving round. Inputs and outputs remain disjoint from the approved
dataset, live spheroid data, ClearEx, and the repository.

The harness derives two otherwise identical profiles:

- a sequential profile with one ITK and one BLAS-family thread; and
- a bounded SLURM profile with one process per worker, one `round_slot`, and one ITK
  and one BLAS-family thread.

An acceptance-only top-level runner sets one documented fixed `ANTS_RANDOM_SEED`
before importing or invoking the production workflow task and restores worker-local
state on exit. The coordinator uses the same seed before the sequential run. This
seed is acceptance instrumentation only; it does not enter production schema or
cache identity.

Tier 2 requires exact plan policy and geometry plus the established strict numerical
comparison for affine parameters, warp arrays, registered TIFFs, and transformed
channels. It also records worker identity, job identity, rendered native-thread
environment, and cleanup. A seed or thread mismatch fails before scientific work.

The synthetic volumes and schedules are deliberately bounded so this supplemental
job tests serialization and execution rather than consuming representative-dataset
runtime. Tier 1 remains the scientific and operational acceptance run.

## Evidence schema and failure behavior

The existing `scientific-parity.json` becomes the Tier 1 scientific-equivalence
record. Its top-level policy identifies the comparison as stochastic,
`metric_absolute_tolerance: 0.01`, and `voxel_differences_are_characterization: true`.
Every artifact comparison retains exact structural fields and adds non-gating numeric
summaries.

A new `deterministic-executor-parity.json` records Tier 2's seed, resolved native
threads, synthetic input identities, sequential and SLURM results, exact comparisons,
and cleanup. The final `workflow-acceptance.json` succeeds only when both tiers and
all durability scenarios succeed.

Any non-finite metric or transform value, structural mismatch, metric difference
above the Tier 1 bound, Tier 2 numerical mismatch, owned-job leak, or incomplete
durability scenario fails closed and preserves the ordinary evidence and cleanup
behavior.

## Testing

Portable unit tests will first characterize the current false failure using arrays
that are structurally valid and scientifically equivalent by metrics but not
voxel-identical. Additional tests require rejection of geometry, dtype, finiteness,
metric-bound, seed, and native-thread violations. The deterministic runner and
synthetic-manifest builders receive isolated unit coverage without scheduler access.

After targeted tests and all portable gates pass, the harness runs once against a
fresh caller-owned evidence directory on the exact final `main` commit. Phase 2E
Task 6 may proceed only if both tiers, all durability scenarios, and final scheduler
cleanup pass.

## Non-goals

- Do not make production multithreaded registration deterministic.
- Do not add a random-seed field to schema version 1.
- Do not loosen or remove exact geometry, provenance, artifact, or cleanup checks.
- Do not select a CC radius or modify the active radius characterization.
- Do not introduce per-chunk Dask execution or parallel channel transformation.
