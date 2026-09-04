# Running a YAML workflow

## When to use the workflow

Use the YAML workflow when one scientific dataset contains a fixed reference round,
multiple moving rounds, or multiple channels to transform. It makes the scientific
choices explicit, validates the input images before registration, and records
completion evidence so an ordinary later run can resume valid work.

## Two files with different responsibilities

A [scientific dataset manifest](dataset-manifest.md) describes the dataset: image
rounds and channels, physical voxel spacing, registration settings, and the selected
registration strategy. It is portable scientific identity.

An execution profile describes the machine-specific environment: caller-owned input
and output roots, native thread counts, and the selected executor and resources. The
same manifest can be used with a different compatible profile without changing the
scientific identity. Read the [execution-profile reference](execution-profile.md)
for the strict machine and resource schema, then start from one of the checked-in
profiles linked below and edit it for your machine.

## Start from the checked-in examples

Use these version-1 YAML files as editable starting points:

- [Spheroid scientific dataset manifest](../examples/spheroid-dataset.yaml)
- [Preprocessed spheroid scientific manifest](../examples/preprocessed-spheroid-dataset.yaml)
- [Explicit XY-rotation scientific manifest](../examples/explicit-rotation-dataset.yaml)
- [Estimated XY-rotation scientific manifest](../examples/estimated-rotation-dataset.yaml)
- [Sequential execution profile](../examples/execution-sequential.yaml)
- [Local-Dask execution profile](../examples/execution-local-dask.yaml)
- [Editable 512-GB SLURM profile](../examples/execution-slurm-512gb.yaml)
- [Editable multi-process 512-GB SLURM profile](../examples/execution-slurm-512gb-multi-process.yaml)

These files do not include microscopy data. Before any command, supply the image
files, edit every manifest image path as needed, and edit the execution profile so
its paths and resources identify caller-owned inputs and outputs. In particular, do
not submit the SLURM placeholders unchanged.

`uv sync` does not activate `.venv`. All workflow commands below use `uv run`, which
runs Cy-ExM in the managed environment.

## Validate inputs

Validation decodes both YAML files, resolves the caller-supplied image paths, inspects
TIFF headers, and constructs a plan. It does not start registration.

```bash
uv run cyexm workflow validate examples/spheroid-dataset.yaml \
  --execution examples/execution-sequential.yaml
```

## Inspect the plan

Planning validates the same scientific inputs and prints the canonical scientific
plan and resolved resource request. It is mutation-free: it creates no files or
directories and does not open the selected executor.

```bash
uv run cyexm workflow plan examples/spheroid-dataset.yaml \
  --execution examples/execution-local-dask.yaml
```

## Run or resume

Run executes pending work. Ordinary resume is the default: matching completion
records, rather than executor summaries, authorize reuse of valid artifacts.

```bash
uv run cyexm workflow run examples/spheroid-dataset.yaml \
  --execution examples/execution-local-dask.yaml
```

## Inspect status

Status is read-only. It inspects the output root and dataset ID against the
scientific plan and authoritative completion records; it does not start an executor.

```bash
uv run cyexm workflow status results spheroid-registration-example
```

## Recompute selected work

Use `--force-round` to request recomputation of one or more selected rounds,
including the reference round, in a compatible scientific plan. It is repeatable.
Forcing the reference round propagates to every moving descendant because each
depends on its prepared fixed input. Use `--force-all` to request recomputation of
every round in that compatible plan; the two options are mutually exclusive. An
unknown forced round is an error. Force cannot bypass a scientific-plan fingerprint
mismatch, which requires a new output root.

```bash
uv run cyexm workflow run examples/spheroid-dataset.yaml \
  --execution examples/execution-local-dask.yaml \
  --force-round 2 --force-round 4

uv run cyexm workflow run examples/spheroid-dataset.yaml \
  --execution examples/execution-local-dask.yaml --force-all
```

## Choose an execution environment

Begin with the sequential profile when establishing a dataset. A local-Dask profile
uses separate local worker processes. A SLURM profile is for a site where you have
confirmed the scheduler, storage, allocation, and resource settings with your local
HPC support. These choices control where work runs; they do not change the dataset
manifest's scientific choices.

## Exact references

Read the [dataset manifest reference](dataset-manifest.md) for the complete
scientific schema and strict YAML rules. Read the [execution-profile
reference](execution-profile.md) for the complete operational schema. Read the
[command-line reference](cli-reference.md) for every command option, structured
output, logging behavior, and exit code. To print the installed Cy-ExM version, run:

```bash
uv run cyexm --version
```

For artifact meanings, ordinary recovery, and status interpretation, see
[Outputs, restart, and recovery](outputs-and-recovery.md).
