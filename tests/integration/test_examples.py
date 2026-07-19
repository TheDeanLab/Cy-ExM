from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cyexm.runtime import NativeThreadConfig

pytestmark = pytest.mark.integration


def _code_source(cell: dict[str, object]) -> str:
    source = cell["source"]
    if isinstance(source, str):
        return source
    assert isinstance(source, list)
    assert all(isinstance(line, str) for line in source)
    return "".join(source)


def _load_notebook_crop_helper(
    ants_module: object,
    crop_array: Callable[[Any, object], Any],
) -> Callable[[Any, object], Any]:
    path = Path("examples/notebooks/transform_directory.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        module = ast.parse(_code_source(cell), filename=f"{path}:cell-{cell_index}")
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name == "crop_spatial_image":
                helper_module = ast.Module(body=[node], type_ignores=[])
                ast.fix_missing_locations(helper_module)
                namespace = {
                    "ants": ants_module,
                    "crop_array": crop_array,
                }
                exec(compile(helper_module, str(path), "exec"), namespace)
                helper = namespace["crop_spatial_image"]
                assert callable(helper)
                return helper
    raise AssertionError(f"crop_spatial_image not found in {path}")


def test_transform_directory_crop_helper_preserves_real_ants_geometry(
    native_threads: NativeThreadConfig,
) -> None:
    del native_threads
    import ants
    import numpy as np

    from cyexm.io.images import crop_array

    array = np.arange(6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8)
    direction = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    image = ants.from_numpy(
        array,
        origin=(10.0, 20.0, 30.0),
        spacing=(0.5, 1.0, 2.0),
        direction=direction,
    )
    bounds = (1, 5, 2, 6, 3, 7)
    crop_spatial_image = _load_notebook_crop_helper(ants, crop_array)

    cropped = crop_spatial_image(image, bounds)

    expected_origin = tuple(
        float(value)
        for value in ants.transform_index_to_physical_point(image, (1, 2, 3))
    )
    assert cropped.origin == expected_origin
    assert cropped.spacing == image.spacing
    np.testing.assert_array_equal(cropped.direction, image.direction)
    np.testing.assert_array_equal(cropped.numpy(), array[1:5, 2:6, 3:7])


def test_transform_directory_crop_helper_returns_original_when_crop_is_disabled(
    native_threads: NativeThreadConfig,
) -> None:
    del native_threads
    import ants
    import numpy as np

    from cyexm.io.images import crop_array

    image = ants.from_numpy(np.zeros((2, 3, 4), dtype=np.float32))
    crop_spatial_image = _load_notebook_crop_helper(ants, crop_array)

    assert crop_spatial_image(image, None) is image
