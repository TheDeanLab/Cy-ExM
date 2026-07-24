# Workflow guide

The Phase 2 workflow separates scientific dataset manifests from machine-specific
execution profiles. A manifest records rounds, channels, registration inputs, voxel
spacing, and scientific choices. An execution profile selects paths, native threads,
and one executor.

Start by validating inputs, then inspect the immutable plan before running:

```bash
uv run cyexm workflow validate examples/spheroid-dataset.yaml \
  --execution examples/execution-sequential.yaml
uv run cyexm workflow plan examples/spheroid-dataset.yaml \
  --execution examples/execution-local-dask.yaml
uv run cyexm workflow run examples/spheroid-dataset.yaml \
  --execution examples/execution-local-dask.yaml
uv run cyexm workflow status results spheroid-registration-example
```

Validation decodes the files, resolves inputs, and inspects TIFF headers. Planning
and status are read-only: planning prints the canonical scientific plan and resolved
resource request without opening the selected executor, while status inspects
artifacts using an output root and dataset ID. `run` executes pending work and
normally resumes valid artifacts through their completion records.

Cy-ExM supports three executor boundaries and two task granularities. The sequential
executor is the numerical and restart reference. The bounded local-Dask executor
uses a process-based local cluster, and the SLURM executor lazily constructs one
owned `SLURMCluster`. Every worker has one eligible task slot.

The execution profile selects the granularity:

```yaml
parallelism:
  mode: coarse
```

or:

```yaml
parallelism:
  mode: fine_grained
```

Omission preserves the `coarse` default. Coarse mode schedules one complete moving
round per task. Fine mode uses the same public workflow command and scientific plan,
but the coordinator schedules preparation, canonical nonlinear chunks, deterministic
assembly, declared moving channels, and validation separately. The stable
`serial_chunked` strategy name still describes the scientific chunking behavior;
parallelism is an execution choice. See
[Fine-grained registration](fine-grained-execution.md) for profiles, graph stages,
resource admission, internal arrays, task logs, failure behavior, and restart
examples.

Each executor prepares the reference round before submitting moving work. Native
fixed, registration-channel, and dependent-channel shapes may differ along Z, Y, and
X. Matching legacy ClearEx and ANTs, affine application resamples scalar and mask
data to the exact fixed-reference grid before nonlinear chunking, and dependent
channels are transformed directly from their native grids onto that grid.

The coarse round-level SLURM executor has the exact-commit acceptance recorded in the
[Phase 2 audit](phase-2-acceptance.md). Portable real-ANTs/Local-Dask evidence is
complete for the fine graph through implementation Task 10. Fine-grained real-SLURM
production acceptance remains pending explicitly selected `hpc` tests with an
operator-approved allocation, inputs, external operational evidence, scientific
parity, task placement, cancellation, cleanup, and performance measurements.
