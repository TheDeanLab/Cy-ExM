# Getting started

## Requirements

Cy-ExM requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

## Obtain the source

Cy-ExM is installed from a source checkout:

```bash
git clone https://github.com/TheDeanLab/Cy-ExM.git
cd Cy-ExM
```

## Create the base environment

Create the locked runtime environment without development tools:

```bash
uv sync --locked --no-dev
```

## Choose an optional capability

Extras are cumulative: `distributed` includes `workflow`, and `hpc` includes
`distributed`. Choose the smallest capability that supports your work:

```bash
# YAML manifests and sequential workflow execution
uv sync --locked --extra workflow

# Local process-based parallel execution with Dask
uv sync --locked --extra distributed

# SLURM/HPC execution through dask-jobqueue
uv sync --locked --extra hpc

# Notebooks and parameterized example scripts
uv sync --locked --extra examples

# Lean runtime plus examples
uv sync --no-dev --extra examples

# Examples plus development and testing tools
uv sync --group dev --extra examples
```

## Run project commands

`uv sync` does not activate `.venv`. Run project commands with `uv run`, which uses
the environment created for this checkout.

## Prepare your first inputs

Your caller-provided inputs are scalar 3-D TIFFs for a fixed reference and a moving
image, plus their physical voxel spacing. The public spacing order is micrometers in
`(x, y, z)`; NumPy volumes use `(z, y, x)`. For the current detailed guidance, read
[image geometry, voxel spacing, and axes](registration-artifacts.md) and
[outputs, restart, and recovery](registration-artifacts.md).

## Register two images

The fixed image remains the reference; the moving image is registered to it. This
whole-image example writes restart-safe artifacts to a dedicated output directory:

```python
from pathlib import Path

from cyexm.registration import register_round

register_round(
    fixed_image_path=Path("data/reference.tif"),
    moving_image_path=Path("data/moving.tif"),
    save_directory=Path("results/round-2"),
    imaging_round=2,
    pixel_size=(0.15, 0.15, 0.40),
    linear_accuracy="low",
    nonlinear_accuracy="high",
    nonlinear_metric="CC",
)
```

## Where to go next

Read [Choosing a registration strategy](registration-foundations.md) to decide
between whole-image and serial-chunked registration. For multiple imaging rounds,
continue to the [workflow tutorial](workflow-guide.md).
