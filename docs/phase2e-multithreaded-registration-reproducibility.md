# Multithreaded ANTs registration reproducibility in Phase 2E

## Status

This note records the root cause of a failed Cy-ExM Phase 2E scientific-parity
check. It is characterization evidence, not a claim that multithreaded ANTs
registration is defective or that a production registration should use one native
thread.

## Observed failure

The Phase 2E acceptance harness ran the same scientific plan through the SLURM and
sequential executors with eight ITK threads and eight BLAS-family threads. Both
workflows completed, used the same inputs and scientific fingerprint, and produced
valid artifacts with matching geometry. The final voxelwise comparison nevertheless
failed at `rtol=1e-4` and `atol=1e-5`:

- 14,214,776 of 46,137,150 voxels differed in the first failing comparison
  (30.8%);
- the largest reported absolute intensity difference was 2,609; and
- the divergence was already present in the affine TRSAA transform, before
  nonlinear registration or dependent-channel export.

This was not isolated to the sequential-versus-SLURM pair. Repeated eight-thread
SLURM runs of the same inputs also produced different affine parameters and output
arrays. Pairwise round-2 registered-image comparisons between the available SLURM
runs differed at approximately 26.8 million to 31.2 million voxels, with mean
absolute intensity differences of approximately 1.6 to 2.3 intensity units.

## Root cause experiment

ANTs' Mattes affine metric uses random spatial sampling when
`aff_random_sampling_rate` is less than one. Cy-ExM's accepted high-accuracy TRSAA
policy uses this sampling behavior. Floating-point reductions performed by multiple
native threads can also depend on execution order.

A bounded synthetic TRSAA experiment separated random seeding from native-thread
count:

| Native ITK/BLAS threads | ANTs seed | Repeated affine parameters |
| --- | --- | --- |
| 8 | unset | different |
| 8 | fixed | different |
| 1 | fixed | exactly identical |

Thus a fixed ANTs seed alone did not make the eight-thread registration exactly
repeatable. A fixed seed combined with one native thread did. This agrees with the
reproducibility constraint exposed by ANTSpyX's deterministic configuration, which
sets the ITK thread count to one.

## Scientific interpretation

The failed check conflated two different questions:

1. **Executor parity:** do sequential and SLURM execution preserve the same plan,
   inputs, geometry, registration policy, artifact semantics, and scientific
   quality?
2. **Optimizer repeatability:** do two independent multithreaded stochastic
   registrations produce numerically identical transforms and voxels?

The second property is not guaranteed by the configured ANTs optimization. It is
therefore not a valid proxy for the first.

The independently estimated results remained close in their scientific endpoint
metrics. Across the available eight-thread and sequential runs, nonlinear
correlation varied by less than 0.004 for each accepted moving round. The failed
voxelwise threshold was consequently much stricter than the natural run-to-run
variation of the optimizer and did not indicate a loss of registration quality.

## Acceptance consequence

Phase 2E will use two complementary gates:

- The representative eight-thread approved-dataset run will require exact plan,
  provenance, geometry, dtype, artifact, placement, restart, cancellation, and
  cleanup invariants. It will compare finite stage-quality metrics within an
  explicitly recorded absolute bound of 0.01. Voxel and transform differences will
  be summarized as characterization evidence rather than required to be identical.
- A bounded synthetic registration with one native thread and a fixed seed will
  require strict numerical parity between sequential and SLURM execution. This
  isolates the executor boundary under conditions where ANTs itself is repeatable.

Neither gate silently relaxes scientific behavior. The first tests the actual
parallel operating configuration without pretending it is deterministic; the
second retains a strict numerical executor check under controlled deterministic
conditions.

## Limitations

The 0.01 metric bound is an acceptance threshold for the approved Phase 2E dataset,
not a universal registration-quality threshold. It is deliberately wider than the
observed less-than-0.004 run-to-run range and must be revisited for materially
different acceptance data or metrics. Spatially resolved residuals, deformation
regularity, and chunk seams remain separate scientific characterization targets.
