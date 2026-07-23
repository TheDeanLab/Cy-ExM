# Phase 2 acceptance audit

## Audit identity and scope

- **Audit date:** 2026-07-22.
- **Committed implementation baseline under audit:**
  `3bfe59b760d20f684446689324dc2074fa3dbcd0`
  (`test: account for reference dispatch on resume`).
- **Acceptance command result:** one explicitly selected HPC test passed in
  18,129.84 seconds (5:02:09).
- **Acceptance-gate commit:** this document is committed after the external run,
  and a commit cannot contain its own commit hash. Repository history identifies
  the later audit and integration commits without inventing a future SHA.

The exact baseline was tested with the approved bounded Zenodo-derived manifest and
an explicit SLURM execution profile. The external run exercised reference
preparation, whole-image and serial-chunked moving rounds, dependent-channel
transformation, sequential comparison, deterministic executor comparison,
independent failure and resume, cancellation and resume, output ownership, and
scheduler cleanup. The registered data, scheduler logs, and full machine-readable
evidence remain external to the repository.

## Runtime and scheduler versions

| Component | Measured version |
| --- | --- |
| Python | 3.12.11 |
| Cy-ExM | 0.1.0 |
| ANTSpyX | 0.6.3 |
| Dask | 2025.1.0 |
| distributed | 2025.1.0 |
| dask-jobqueue | 0.9.0 |
| NumPy | 2.3.5 |
| SciPy | 1.15.3 |
| tifffile | 2026.7.14 |
| SLURM | 22.05.9 |

The `512GB` partition reported `State=UP`, 97 nodes, 6984 CPUs, and aggregate
configured memory of 48,500 GB at audit time. The representative gate requested
two exclusive workers, one process per worker, 72 scheduler CPUs and 466 GB per
worker, a six-hour walltime, the `ib0` scheduler and worker interface, and eight
threads for each ITK and BLAS-family runtime. Each worker reported 500,363,689,984
usable bytes and confirmed dataset visibility, writable output, usable scratch,
package identity, and native-thread identity.

Configuration identity was fail-closed:

- dataset-manifest SHA-256:
  `3142688f7e11df6d58a4464db8adf4b9dc7e707c8dd91278bd345855ada1eefc`;
- execution-profile fingerprint (SHA-256):
  `89e62968574d0644cf6ff7ef62389bfb32f5bfcf8dc359d67233c7dd4b373c0b`;
- scientific fingerprint:
  `240b9a5084c665ba3b962d7a02ce5b3c37302edb293bafcbddc65754f8beaba2`;
- representative rendered-job-script SHA-256:
  `aad6b2e6e7facb62524b78d8ea6f0702ad311e3d9ba99ae8913a0e7446b37fb3`.

## Portable validation baseline

The current branch was measured before materializing this audit page so that the
new audit-policy test could be observed failing first.

| Command | Recorded result |
| --- | --- |
| Unit suite with only the not-yet-satisfiable final-audit test deselected | 2042 passed, 1 deselected in 71.93 s |
| Bounded integration suite | 55 passed in 82.43 s |

The final documentation, complete unit, integration, lint, format, type, and
all-files policy results are recorded below after the page itself satisfied the
audit gate.

| Final command | Fresh result |
| --- | --- |
| Focused Phase 2 documentation and policy tests | 117 passed in 2.02 s |
| Complete portable unit suite | 2043 passed in 26.81 s |
| Complete bounded integration suite | 55 passed in 23.80 s |
| `ruff check` | Exit 0 after wrapping one overlong policy-test string |
| `ruff format --check` | Exit 0; 134 files already formatted |
| `basedpyright` | Exit 0; 0 errors, 0 warnings, 0 notes |
| `pre-commit run --all-files` | All configured hooks passed |

## Two-tier scientific result

### Tier 1: representative multithreaded execution

The representative run used eight native threads for all six recorded ITK and
BLAS-family variables. Moving rounds 2 and 3 ran concurrently as separate SLURM
jobs and on distinct hosts:

| Round strategy | Job ID | Host |
| --- | ---: | --- |
| Whole image | 10767265 | NucleusB079 |
| Serial chunked | 10767266 | NucleusB092 |

The SLURM and independently estimated sequential workflows had exact structure,
geometry, dtype, interpolation, and scientific provenance. Their largest absolute
metric difference was `0.001567840576171875`, below the approved absolute bound of
`0.01`. This establishes **stochastic scientific equivalence** for this approved
dataset and profile. Voxel arrays, affine parameters, and warps were not required to
be identical and are retained as characterization only.

That distinction follows the measured behavior documented in the
[multithreaded registration reproducibility note](phase2e-multithreaded-registration-reproducibility.md)
and the approved
[two-tier correction design](superpowers/specs/2026-07-22-phase2e-stochastic-parity-design.md).
A fixed seed does not make multithreaded ANTs registration numerically exact.

### Tier 2: deterministic executor parity

The deterministic child coordinator used `ANTS_RANDOM_SEED=20260722` and one native
thread for every recorded ITK and BLAS-family variable. SLURM jobs `10767783` and
`10767784` ran on `NucleusB078` and `NucleusB079`. Against an independently executed
sequential workflow, every compared registered image, transformed channel, affine
parameter vector, warp, and metric was exactly equal with zero relative and absolute
tolerance; the maximum numeric difference was 0.0. The deterministic exact-name
queue was empty after cleanup.

This establishes **strict executor parity** under the bounded one-thread seeded
instrumentation. It does not claim deterministic multithreaded registration and
does not introduce a production seed policy.

## Failure, cancellation, and restart

The independent-failure scenario proved ordering: round 2 committed successfully
before an injected failure stopped round 3. On an ordinary resume, the reference
artifacts were revalidated and reused, round 2 was reused in full, and round 3 was
recomputed. The failure and resume jobs were `10767798` on `NucleusB078`, `10767797`
on `NucleusB086`, and `10768100` on `NucleusB078`.

The cancellation scenario terminated the coordinator with exit code 130 after
round 2 committed while round 3 remained unfinished. Its ordinary resume again
proved that round 2 was reused and round 3 was recomputed. The cancellation and
resume jobs were `10768142` on `NucleusB078`, `10768143` on `NucleusB094`, and
`10768506` on `NucleusB078`.

Reference preparation is deliberately redispatched through the executor barrier on
resume, but its committed artifacts were reused. This is expected orchestration
semantics, not reference recomputation.

## Cleanup and ownership

Client, cluster, and configuration cleanup succeeded in every completed, failed,
cancelled, and resumed scenario. Forced cancellation reported one unfinished future
and succeeded. Final exact-name queues were empty for both representative and
deterministic job names, with no unrelated jobs classified as owned. Output
ownership was confined to the caller-owned acceptance root, and no pre-existing
entry appeared in the general result root.

## External evidence checksums

The following SHA-256 values identify the concise machine-readable evidence used by
this audit. Raw images, deformation arrays, and scheduler logs are intentionally not
embedded here.

| Evidence record | SHA-256 |
| --- | --- |
| Consolidated workflow acceptance | `8ffc175f6a15aa8a222f53da2b1bdaad0186dd4fce9336e9df2050b3eb9473a0` |
| Representative scientific equivalence | `defa5a9627a0378e10ef659af46be18089a986de2d153bf21e7ee13772d1860b` |
| Deterministic executor parity | `6599e59a5f1734e53dd81a52eb7548e8cd404fb6883ba26f3805a4c7630033ef` |
| Failure and resume | `25145674fdfbaa162e2a1201fe0fc4204ced0c0bb76b40ab786bc3604823231b` |
| Cancellation and resume | `97ef27e426d30791a6a95e38c10455f0ae6fc091534a30b20436066b2b1c5cb8` |
| Final scheduler queue | `8115ac75d03d8f2008b0414392f2aadb9d9a8ce033f8b61e07d6d1bb96c45106` |
| Final output ownership | `bddb53b2626ed504f40e1e2b1ce0690e0ed0eb72da4ae5b9d2f8c7f75025a695` |

The ten scheduler logs corresponding exactly to the ten recorded job IDs were
preserved beside the external evidence after the run. The evidence root remains
read-only audit material and was not reused for another acceptance attempt.

## Acceptance boundary

Phase 2E is accepted for the tested Cy-ExM baseline, approved manifest, and audited
lab `512GB` profile. Portable CI remains scheduler-free. Other clusters, partitions,
thread counts, memory profiles, datasets, registration policies, or nested/per-chunk
parallelism require their own operational and scientific validation. Per-chunk Dask
parallelism, parallel channel transformation, and resource auto-optimization remain
future work.
