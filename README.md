# Cy-ExM

Cy-ExM registers volumetric images acquired across cyclic expansion microscopy
rounds. It supports whole-image and memory-conscious serial-chunked registration,
applies the resulting transforms to additional imaging channels, and can resume
validated intermediate work after interruption.

Cy-ExM requires Python 3.12 and retains the academic-research-use terms in
[LICENSE.md](LICENSE.md). See the
[complete documentation](https://thedeanlab.github.io/Cy-ExM/) for scientific
guidance, workflow configuration, and advanced execution.

New users should begin with the installation and two-image example below.

## Installation

Install Python 3.12 and `uv`, then clone the repository and create its locked base
environment:

```bash
git clone https://github.com/TheDeanLab/Cy-ExM.git
cd Cy-ExM
uv sync --locked --no-dev
```

Confirm that the package is available in the managed environment:

```bash
uv run python -c "import cyexm; print(cyexm.__version__)"
```

The [installation guide](docs/getting-started.md) explains optional capabilities for
notebooks, YAML workflows, local parallel execution, HPC execution, and development.

## Register two images

Supply a fixed reference TIFF, a moving TIFF, and the physical voxel spacing measured
for your data. Save this example as `register_images.py`:

The quick start assumes that:

- both inputs are scalar 3-D TIFF volumes;
- `reference.tif` defines the fixed coordinate system;
- `moving.tif` is the image transformed onto that system; and
- `pixel_size` contains the measured voxel spacing for the input data.

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

Run it in the managed environment:

```bash
uv run python register_images.py
```

Cy-ExM writes restart-safe registration artifacts below `results/round-2`. Running
the same request again reuses artifacts whose completion evidence is still valid.

Before adapting the example, read the guides to
[registration strategies](docs/registration-foundations.md),
[image geometry](docs/image-geometry.md), and
[outputs and recovery](docs/outputs-and-recovery.md).

## Register multiple imaging rounds

For several imaging rounds and channels, Cy-ExM can read a scientific dataset
manifest together with a separate execution profile. Start with the
[example manifest](examples/spheroid-dataset.yaml) and
[sequential execution profile](examples/execution-sequential.yaml), then follow the
[YAML workflow guide](docs/workflow-guide.md).

The checked-in YAML files are editable starting points. Supply the corresponding
images and replace their paths before validating or running them.

## Documentation

- [Getting started](docs/getting-started.md)
- [Choosing a registration strategy](docs/registration-foundations.md)
- [Registration preprocessing](docs/preprocessing.md)
- [Running a YAML workflow](docs/workflow-guide.md)
- [Local parallel and HPC execution](docs/parallel-hpc.md)
- [Development guide](docs/development.md)

## License

Cy-ExM is available for academic research under the terms in
[LICENSE.md](LICENSE.md).
