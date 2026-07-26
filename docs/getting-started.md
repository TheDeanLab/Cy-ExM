# Getting started

Cy-ExM requires Python 3.12. Start with the locked base environment when you
want to use the Python registration API:

```bash
uv sync --locked
```

Add the optional capability that matches your task. Extras are cumulative:
`distributed` includes `workflow`, and `hpc` includes `distributed`.

```bash
# YAML manifests and sequential workflow execution
uv sync --locked --extra workflow

# Local process-based parallel execution with Dask
uv sync --locked --extra distributed

# SLURM/HPC execution through dask-jobqueue
uv sync --locked --extra hpc

# Notebooks and parameterized example scripts
uv sync --locked --extra examples
```

Add `--no-dev` for a lean environment. `uv sync` does not activate `.venv`, so
run project commands with `uv run`.

The established example conveniences are:

```bash
# Lean runtime plus examples
uv sync --no-dev --extra examples

# Examples plus development and testing tools
uv sync --group dev --extra examples
```

On a quota-limited HPC system, configure uv's cache and managed Python location
according to your site's storage and module guidance; choose a location with
adequate quota and performance before creating the environment.

The [workflow guide](workflow-guide.md) explains the dataset manifest, execution
profiles, commands, and executor boundaries.
