# Cy-ExM

Cy-ExM is an independent Python package for reusable cyclic expansion microscopy
registration workflows. Phase 1 is scoped to whole-image and restartable serial
chunk registration behavior from the recorded ClearEx baseline while keeping the
new package free of a runtime dependency on ClearEx.

The project requires Python 3.12. Create the locked development environment with:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv sync --group dev
```

The top-level import is intentionally lightweight and exposes version metadata:

```python
import cyexm

print(cyexm.__version__)
```

Run the current unit tests with `uv run pytest tests/unit -v`.

Cy-ExM retains the ClearEx academic-research-use license. See [LICENSE.md](LICENSE.md)
before using or redistributing the software.
