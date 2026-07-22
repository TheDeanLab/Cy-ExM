# Getting started

Cy-ExM requires Python 3.12. Create the locked base environment with:

```bash
uv sync --locked
```

Add only the optional capability required for your work. Extras are cumulative:
`distributed` includes `workflow`, and `hpc` includes `distributed`.

```bash
# YAML loading and sequential workflow execution
uv sync --locked --extra workflow

# Process-based local Dask execution
uv sync --locked --extra distributed

# SLURMCluster support through dask-jobqueue
uv sync --locked --extra hpc
```

For a lean non-development environment, add `--no-dev`. The locked environment
keeps the documented dependencies reproducible; regenerate the lock only when an
intentional dependency change has supporting compatibility evidence.

The [workflow guide](workflow-guide.md) explains the dataset manifest, execution
profiles, commands, and executor boundaries.
