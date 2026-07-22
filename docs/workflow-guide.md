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

Cy-ExM supports three executor boundaries. The sequential executor is the numerical
and restart reference. The bounded local-Dask executor uses a process-based local
cluster with one round slot per worker. The SLURM executor lazily constructs an owned
`SLURMCluster`, with one process per worker job and one round slot per worker. Each
executor prepares the reference round before submitting moving rounds; registration
and channel transformation remain serial inside a round task.

Portable SLURM checks cover construction, job-template rendering, preflight policy,
failure mapping, cancellation, and cleanup. They are construction evidence, not
real-cluster acceptance. Real SLURM submission and HPC acceptance require explicitly
selected `hpc` tests, an operator-approved allocation and inputs, and external
operational evidence.
