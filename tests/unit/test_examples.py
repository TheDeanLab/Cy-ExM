from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from numbers import Real
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, get_type_hints

import pytest

if TYPE_CHECKING:
    from cyexm.registration.artifacts import CompletionRecord, FileIdentity
    from cyexm.spatial import ImageGeometry

pytestmark = pytest.mark.unit

SCRIPT_NAMES = (
    "apply_transforms.py",
    "chunked_registration.py",
    "linear_registration.py",
    "nonlinear_registration.py",
    "whole_image_registration.py",
)
NOTEBOOK_NAMES = (
    "apply_warp_transform.ipynb",
    "chunked_nonlinear_registration.ipynb",
    "evaluate_histogram_matching.ipynb",
    "image_registration_class.ipynb",
    "nonlinear_registration.ipynb",
    "transform_directory.ipynb",
)
FORBIDDEN_EXAMPLE_TEXT = (
    "/archive/bioinformatics",
    "/endosome/archive",
    "/project/bioinformatics",
    "import clearex",
    "from clearex",
    "ParallelChunkedImageRegistration",
    "downloaded_data.zip",
)


def _script_paths() -> tuple[Path, ...]:
    return tuple(Path("examples/scripts") / name for name in SCRIPT_NAMES)


def _notebook_paths() -> tuple[Path, ...]:
    return tuple(Path("examples/notebooks") / name for name in NOTEBOOK_NAMES)


def _load_script(path: Path) -> ModuleType:
    module_name = f"cyexm_example_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _code_source(cell: dict[str, object]) -> str:
    source = cell["source"]
    if isinstance(source, str):
        return source
    assert isinstance(source, list)
    assert all(isinstance(line, str) for line in source)
    return "".join(source)


def _load_notebook_function(
    name: str,
    namespace: dict[str, object],
):
    path = Path("examples/notebooks/transform_directory.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        module = ast.parse(_code_source(cell), filename=f"{path}:cell-{cell_index}")
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                helper_module = ast.Module(body=[node], type_ignores=[])
                ast.fix_missing_locations(helper_module)
                exec(compile(helper_module, str(path), "exec"), namespace)
                helper = namespace[name]
                assert callable(helper)
                return helper
    raise AssertionError(f"{name} not found in {path}")


def _write_notebook_completion_record(
    artifact: Path,
    *,
    stage: str,
    axis_convention_version: str,
) -> Path:
    from cyexm.registration.artifacts import (
        ArtifactStore,
        CompletionRecord,
        FileIdentity,
        StageRequest,
    )
    from cyexm.spatial import ImageGeometry

    source = artifact.parent / "source.tif"
    source.write_bytes(b"source")
    request = StageRequest.for_files(
        stage=stage,
        inputs=(source,),
        geometry=ImageGeometry(
            shape=(2, 3, 4),
            origin=(0.0, 0.0, 0.0),
            spacing=(0.2, 0.3, 0.5),
            direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        method="test",
        accuracy="exact",
        native_threads={},
        upstream=None,
        parameters={"axis_convention_version": axis_convention_version},
    )
    record = CompletionRecord(
        schema_version=1,
        request=request,
        artifact=FileIdentity.from_path(artifact),
    )
    record_path = ArtifactStore().record_path(artifact)
    record_path.write_text(record.to_json(), encoding="utf-8")
    return record_path


def _write_transform_directory_record(
    artifact: Path,
    *,
    stage: str,
    inputs: tuple[FileIdentity, ...],
    geometry: ImageGeometry,
    upstream: FileIdentity | None,
    parameters: dict[str, object],
) -> CompletionRecord:
    from cyexm.registration.artifacts import (
        ArtifactStore,
        CompletionRecord,
        FileIdentity,
        StageRequest,
    )

    request = StageRequest.for_identities(
        stage=stage,
        inputs=inputs,
        geometry=geometry,
        method="test",
        accuracy="exact",
        native_threads={},
        upstream=upstream,
        parameters=parameters,
    )
    record = CompletionRecord(
        schema_version=1,
        request=request,
        artifact=FileIdentity.from_path(artifact),
    )
    ArtifactStore().record_path(artifact).write_text(record.to_json(), encoding="utf-8")
    return record


def _write_transform_directory_chain(
    tmp_path: Path,
    *,
    crop: bool,
    round_number: int = 2,
) -> SimpleNamespace:
    from cyexm.registration.artifacts import FileIdentity
    from cyexm.spatial import AXIS_CONVENTION_VERSION, ImageGeometry

    fixed_image = tmp_path / "fixed.tif"
    moving_image = tmp_path / "moving.tif"
    fixed_image.write_bytes(b"fixed")
    moving_image.write_bytes(b"moving")
    inputs = (
        FileIdentity.from_path(fixed_image),
        FileIdentity.from_path(moving_image),
    )
    geometry = ImageGeometry(
        shape=(2, 3, 4),
        origin=(0.0, 0.0, 0.0),
        spacing=(0.15, 0.15, 0.15),
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    parameters = {
        "axis_convention_version": AXIS_CONVENTION_VERSION,
        "crop": crop,
        "imaging_round": round_number,
        "pixel_size_xyz": (0.15, 0.15, 0.15),
    }
    affine_path = tmp_path / f"linear_transform_{round_number}.mat"
    affine_path.write_bytes(b"affine")
    affine_record = _write_transform_directory_record(
        affine_path,
        stage="affine",
        inputs=inputs,
        geometry=geometry,
        upstream=None,
        parameters=parameters,
    )
    nonlinear_path = tmp_path / f"nonlinear_warp_{round_number}.nii.gz"
    nonlinear_path.write_bytes(b"warp")
    nonlinear_record = _write_transform_directory_record(
        nonlinear_path,
        stage="nonlinear",
        inputs=inputs,
        geometry=geometry,
        upstream=affine_record.artifact,
        parameters=parameters,
    )
    fixed_crop_path = None
    moving_crop_path = None
    fixed_crop_record = None
    moving_crop_record = None
    if crop:
        fixed_crop_path = tmp_path / f"fixed_crop_indices_{round_number}.json"
        moving_crop_path = tmp_path / f"moving_crop_indices_{round_number}.json"
        fixed_crop_path.write_text("[0,2,0,3,0,4]\n", encoding="utf-8")
        moving_crop_path.write_text("[0,2,0,3,0,4]\n", encoding="utf-8")
        fixed_crop_record = _write_transform_directory_record(
            fixed_crop_path,
            stage="crop:fixed",
            inputs=inputs,
            geometry=geometry,
            upstream=None,
            parameters={**parameters, "image_role": "fixed"},
        )
        moving_crop_record = _write_transform_directory_record(
            moving_crop_path,
            stage="crop:moving",
            inputs=inputs,
            geometry=geometry,
            upstream=None,
            parameters={**parameters, "image_role": "moving"},
        )
    return SimpleNamespace(
        fixed_image=fixed_image,
        moving_image=moving_image,
        affine_path=affine_path,
        nonlinear_path=nonlinear_path,
        fixed_crop_path=fixed_crop_path,
        moving_crop_path=moving_crop_path,
        affine_record=affine_record,
        nonlinear_record=nonlinear_record,
        fixed_crop_record=fixed_crop_record,
        moving_crop_record=moving_crop_record,
        round_number=round_number,
    )


def _rewrite_transform_directory_record(path: Path, record: object) -> None:
    from cyexm.registration.artifacts import ArtifactStore, CompletionRecord

    assert isinstance(record, CompletionRecord)
    ArtifactStore().record_path(path).write_text(record.to_json(), encoding="utf-8")


def _load_transform_directory_artifact_validator():
    from cyexm.registration.artifacts import (
        ArtifactStore,
        CompletionRecord,
        FileIdentity,
    )
    from cyexm.spatial import AXIS_CONVENTION_VERSION

    return _load_notebook_function(
        "validate_cached_artifact",
        {
            "ArtifactStore": ArtifactStore,
            "CompletionRecord": CompletionRecord,
            "FileIdentity": FileIdentity,
            "AXIS_CONVENTION_VERSION": AXIS_CONVENTION_VERSION,
        },
    )


def _load_transform_directory_chain_validator():
    from cyexm.registration.artifacts import FileIdentity

    namespace = {"FileIdentity": FileIdentity, "Real": Real, "math": math}
    normalize = _load_notebook_function("normalize_pixel_size_xyz", namespace)
    namespace["normalize_pixel_size_xyz"] = normalize
    return _load_notebook_function("validate_registration_chain", namespace)


def _validated_transform_directory_records(chain: SimpleNamespace) -> SimpleNamespace:
    validate = _load_transform_directory_artifact_validator()
    return SimpleNamespace(
        affine=validate(chain.affine_path, "affine"),
        nonlinear=validate(chain.nonlinear_path, "nonlinear"),
        fixed_crop=validate(chain.fixed_crop_path, "crop:fixed"),
        moving_crop=validate(chain.moving_crop_path, "crop:moving"),
    )


def test_expected_example_inventory_exists() -> None:
    assert {path.name for path in Path("examples/scripts").glob("*.py")} == set(
        SCRIPT_NAMES
    )
    assert {path.name for path in Path("examples/notebooks").glob("*.ipynb")} == set(
        NOTEBOOK_NAMES
    )


def test_examples_are_portable_and_use_cyexm() -> None:
    paths = (*_script_paths(), *_notebook_paths())
    assert paths
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_EXAMPLE_TEXT:
            assert forbidden not in text, f"{forbidden!r} found in {path}"
        assert "cyexm" in text
        assert "dask" not in text.lower()
        assert "slurm" not in text.lower()
        assert "yaml" not in text.lower()


def test_notebooks_are_valid_clear_portable_documents() -> None:
    for path in _notebook_paths():
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert isinstance(notebook["nbformat_minor"], int)
        assert isinstance(notebook["metadata"], dict)
        cells = notebook["cells"]
        assert isinstance(cells, list)
        assert cells

        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
        assert code_cells
        first_code = _code_source(code_cells[0])
        assert "EDIT THESE PATHS" in first_code
        assert "from pathlib import Path" in first_code
        assert "Path(" in first_code

        for index, cell in enumerate(code_cells):
            assert cell.get("outputs", []) == [], f"saved output in {path}"
            assert cell.get("execution_count") is None, (
                f"saved execution count in {path}"
            )
            compile(_code_source(cell), f"{path}:cell-{index}", "exec")


def test_transform_directory_notebook_preserves_optional_crop_geometry() -> None:
    path = Path("examples/notebooks/transform_directory.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        _code_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "FIXED_CROP_INDICES: Path | None" in source
    assert "MOVING_CROP_INDICES: Path | None" in source
    assert "transform_index_to_physical_point" in source
    assert "(x0, y0, z0)" in source
    assert "ants_to_external_array" in source
    assert "external_to_ants_array" in source
    assert "tuple(float(value) for value in origin)" in source
    assert "origin=origin" in source
    assert "spacing=image.spacing" in source
    assert "direction=image.direction" in source
    assert "if bounds is None" in source


def test_transform_directory_notebook_validates_every_reused_artifact_first() -> None:
    path = Path("examples/notebooks/transform_directory.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        _code_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert "CompletionRecord" in source
    assert "FileIdentity" in source
    assert "AXIS_CONVENTION_VERSION" in source
    calls = (
        'affine_record = validate_cached_artifact(AFFINE_TRANSFORM, "affine")',
        'nonlinear_record = validate_cached_artifact(NONLINEAR_WARP, "nonlinear")',
        "fixed_crop_record = validate_cached_artifact(",
        "moving_crop_record = validate_cached_artifact(",
    )
    for call in calls:
        assert call in source
    chain_validation = source.rindex("validate_registration_chain(")
    bounds_read = source.index("fixed_bounds = read_bounds")
    fixed_load = source.index("fixed_full = load_ants_image")
    transform_application = source.index("registered = transform_image(")
    assert all(source.index(call) < chain_validation for call in calls)
    assert chain_validation < bounds_read
    assert chain_validation < fixed_load
    assert chain_validation < transform_application


def test_transform_directory_artifact_validator_accepts_current_record(
    tmp_path: Path,
) -> None:
    from cyexm.registration.artifacts import (
        ArtifactStore,
        CompletionRecord,
        FileIdentity,
    )
    from cyexm.spatial import AXIS_CONVENTION_VERSION

    artifact = tmp_path / "linear_transform_2.mat"
    artifact.write_bytes(b"affine")
    _write_notebook_completion_record(
        artifact,
        stage="affine",
        axis_convention_version=AXIS_CONVENTION_VERSION,
    )
    validator = _load_notebook_function(
        "validate_cached_artifact",
        {
            "ArtifactStore": ArtifactStore,
            "CompletionRecord": CompletionRecord,
            "FileIdentity": FileIdentity,
            "AXIS_CONVENTION_VERSION": AXIS_CONVENTION_VERSION,
        },
    )

    result = validator(artifact, "affine")

    assert isinstance(result, CompletionRecord)
    assert result == CompletionRecord.read(ArtifactStore().record_path(artifact))


@pytest.mark.parametrize("crop", [False, True], ids=["no-crop", "crop"])
def test_transform_directory_accepts_a_complete_provenance_chain(
    tmp_path: Path,
    crop: bool,
) -> None:
    chain = _write_transform_directory_chain(tmp_path, crop=crop)
    chain.moving_image.unlink()
    records = _validated_transform_directory_records(chain)
    validate_chain = _load_transform_directory_chain_validator()

    validate_chain(
        records.affine,
        records.nonlinear,
        records.fixed_crop,
        records.moving_crop,
        chain.fixed_image,
        chain.round_number,
        (0.15, 0.15, 0.15),
    )


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        pytest.param("nonlinear-upstream", "upstream", id="nonlinear-upstream"),
        pytest.param("affine-upstream", "upstream", id="affine-upstream"),
        pytest.param("input-count", "inputs", id="input-count"),
        pytest.param("common-inputs", "inputs", id="common-inputs"),
        pytest.param("round-value", "round", id="round-value"),
        pytest.param("round-boolean", "round", id="round-boolean"),
        pytest.param("round-non-integer", "round", id="round-non-integer"),
        pytest.param("crop-flag", "crop policy", id="crop-flag"),
        pytest.param("crop-boolean", "crop policy", id="crop-boolean"),
        pytest.param("crop-inputs", "crop inputs", id="crop-inputs"),
        pytest.param("crop-stage", "fixed crop stage", id="crop-stage"),
        pytest.param("crop-role", "fixed crop role", id="crop-role"),
        pytest.param("crop-upstream", "fixed crop upstream", id="crop-upstream"),
        pytest.param("one-sided-crop", "crop policy", id="one-sided-crop"),
        pytest.param("fixed-identity", "fixed image", id="fixed-identity"),
        pytest.param("pixel-size", "pixel spacing", id="pixel-size"),
        pytest.param("pixel-shape", "pixel spacing", id="pixel-shape"),
        pytest.param("pixel-boolean", "pixel spacing", id="pixel-boolean"),
        pytest.param("pixel-nonnumeric", "pixel spacing", id="pixel-nonnumeric"),
        pytest.param("pixel-zero", "pixel spacing", id="pixel-zero"),
        pytest.param("pixel-negative", "pixel spacing", id="pixel-negative"),
        pytest.param("pixel-nan", "pixel spacing", id="pixel-nan"),
        pytest.param("pixel-infinity", "pixel spacing", id="pixel-infinity"),
        pytest.param("pixel-overflow", "pixel spacing", id="pixel-overflow"),
        pytest.param("geometry", "geometry", id="geometry"),
    ],
)
def test_transform_directory_rejects_mismatched_provenance_chain(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    from cyexm.registration.artifacts import FileIdentity
    from cyexm.spatial import ImageGeometry

    uses_crop = failure in {
        "crop-inputs",
        "crop-role",
        "crop-stage",
        "crop-upstream",
        "one-sided-crop",
    }
    round_number = 1 if failure == "round-boolean" else 2
    chain = _write_transform_directory_chain(
        tmp_path,
        crop=uses_crop,
        round_number=round_number,
    )
    fixed_image = chain.fixed_image
    if failure == "nonlinear-upstream":
        chain.nonlinear_record = replace(
            chain.nonlinear_record,
            request=replace(
                chain.nonlinear_record.request,
                upstream=chain.nonlinear_record.artifact,
            ),
        )
        _rewrite_transform_directory_record(
            chain.nonlinear_path, chain.nonlinear_record
        )
    elif failure == "affine-upstream":
        chain.affine_record = replace(
            chain.affine_record,
            request=replace(
                chain.affine_record.request,
                upstream=chain.nonlinear_record.artifact,
            ),
        )
        _rewrite_transform_directory_record(chain.affine_path, chain.affine_record)
    elif failure == "input-count":
        chain.affine_record = replace(
            chain.affine_record,
            request=replace(
                chain.affine_record.request,
                inputs=(chain.affine_record.request.inputs[0],),
            ),
        )
        _rewrite_transform_directory_record(chain.affine_path, chain.affine_record)
    elif failure == "common-inputs":
        unrelated_moving = tmp_path / "unrelated-moving.tif"
        unrelated_moving.write_bytes(b"unrelated")
        chain.nonlinear_record = replace(
            chain.nonlinear_record,
            request=replace(
                chain.nonlinear_record.request,
                inputs=(
                    chain.nonlinear_record.request.inputs[0],
                    FileIdentity.from_path(unrelated_moving),
                ),
            ),
        )
        _rewrite_transform_directory_record(
            chain.nonlinear_path, chain.nonlinear_record
        )
    elif failure in {"round-value", "round-boolean", "round-non-integer"}:
        parameters = dict(chain.nonlinear_record.request.parameters)
        parameters["imaging_round"] = {
            "round-value": 3,
            "round-boolean": True,
            "round-non-integer": float(chain.round_number),
        }[failure]
        chain.nonlinear_record = replace(
            chain.nonlinear_record,
            request=replace(chain.nonlinear_record.request, parameters=parameters),
        )
        _rewrite_transform_directory_record(
            chain.nonlinear_path, chain.nonlinear_record
        )
    elif failure in {"crop-flag", "crop-boolean"}:
        parameters = dict(chain.nonlinear_record.request.parameters)
        parameters["crop"] = True if failure == "crop-flag" else 0
        chain.nonlinear_record = replace(
            chain.nonlinear_record,
            request=replace(chain.nonlinear_record.request, parameters=parameters),
        )
        _rewrite_transform_directory_record(
            chain.nonlinear_path, chain.nonlinear_record
        )
    elif failure == "crop-inputs":
        unrelated_moving = tmp_path / "unrelated-moving.tif"
        unrelated_moving.write_bytes(b"unrelated")
        chain.fixed_crop_record = replace(
            chain.fixed_crop_record,
            request=replace(
                chain.fixed_crop_record.request,
                inputs=(
                    chain.fixed_crop_record.request.inputs[0],
                    FileIdentity.from_path(unrelated_moving),
                ),
            ),
        )
        _rewrite_transform_directory_record(
            chain.fixed_crop_path, chain.fixed_crop_record
        )
    elif failure == "crop-role":
        parameters = dict(chain.fixed_crop_record.request.parameters)
        parameters["image_role"] = "moving"
        chain.fixed_crop_record = replace(
            chain.fixed_crop_record,
            request=replace(chain.fixed_crop_record.request, parameters=parameters),
        )
        _rewrite_transform_directory_record(
            chain.fixed_crop_path, chain.fixed_crop_record
        )
    elif failure == "crop-upstream":
        chain.fixed_crop_record = replace(
            chain.fixed_crop_record,
            request=replace(
                chain.fixed_crop_record.request,
                upstream=chain.affine_record.artifact,
            ),
        )
        _rewrite_transform_directory_record(
            chain.fixed_crop_path, chain.fixed_crop_record
        )
    elif failure == "fixed-identity":
        fixed_image = tmp_path / "unrelated-fixed.tif"
        fixed_image.write_bytes(b"unrelated-fixed")
    elif failure in {
        "pixel-size",
        "pixel-shape",
        "pixel-boolean",
        "pixel-nonnumeric",
        "pixel-zero",
        "pixel-negative",
        "pixel-overflow",
    }:
        parameters = dict(chain.nonlinear_record.request.parameters)
        parameters["pixel_size_xyz"] = {
            "pixel-size": [0.2, 0.15, 0.15],
            "pixel-shape": [0.15, 0.15],
            "pixel-boolean": [True, 0.15, 0.15],
            "pixel-nonnumeric": ["0.15", 0.15, 0.15],
            "pixel-zero": [0.0, 0.15, 0.15],
            "pixel-negative": [-0.15, 0.15, 0.15],
            "pixel-overflow": [10**400, 0.15, 0.15],
        }[failure]
        chain.nonlinear_record = replace(
            chain.nonlinear_record,
            request=replace(chain.nonlinear_record.request, parameters=parameters),
        )
        _rewrite_transform_directory_record(
            chain.nonlinear_path, chain.nonlinear_record
        )
    elif failure == "geometry":
        chain.nonlinear_record = replace(
            chain.nonlinear_record,
            request=replace(
                chain.nonlinear_record.request,
                geometry=ImageGeometry(
                    shape=(3, 3, 4),
                    origin=(0.0, 0.0, 0.0),
                    spacing=(0.15, 0.15, 0.15),
                    direction=(
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0),
                    ),
                ),
            ),
        )
        _rewrite_transform_directory_record(
            chain.nonlinear_path, chain.nonlinear_record
        )

    records = _validated_transform_directory_records(chain)
    if failure == "one-sided-crop":
        records.moving_crop = None
    elif failure == "crop-stage":
        records.fixed_crop = replace(
            records.fixed_crop,
            request=replace(records.fixed_crop.request, stage="crop:moving"),
        )
    elif failure in {"pixel-nan", "pixel-infinity"}:
        parameters = dict(records.nonlinear.request.parameters)
        parameters["pixel_size_xyz"] = [
            math.nan if failure == "pixel-nan" else math.inf,
            0.15,
            0.15,
        ]
        records.nonlinear = replace(
            records.nonlinear,
            request=replace(records.nonlinear.request, parameters=parameters),
        )
    validate_chain = _load_transform_directory_chain_validator()

    with pytest.raises(ValueError, match=message):
        validate_chain(
            records.affine,
            records.nonlinear,
            records.fixed_crop,
            records.moving_crop,
            fixed_image,
            chain.round_number,
            (0.15, 0.15, 0.15),
        )


@pytest.mark.parametrize("target", ["fixed", "moving"])
@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        pytest.param("imaging_round", 3, "round", id="round"),
        pytest.param("crop", 1, "crop policy", id="non-bool-crop"),
        pytest.param(
            "pixel_size_xyz",
            [0.2, 0.15, 0.15],
            "pixel spacing",
            id="pixel-mismatch",
        ),
        pytest.param(
            "pixel_size_xyz",
            [0.15, 0.15],
            "pixel spacing",
            id="pixel-malformed",
        ),
    ],
)
def test_transform_directory_checks_every_selected_crop_record(
    tmp_path: Path,
    target: str,
    parameter: str,
    value: object,
    message: str,
) -> None:
    chain = _write_transform_directory_chain(tmp_path, crop=True)
    record = getattr(chain, f"{target}_crop_record")
    path = getattr(chain, f"{target}_crop_path")
    parameters = dict(record.request.parameters)
    parameters[parameter] = value
    record = replace(record, request=replace(record.request, parameters=parameters))
    _rewrite_transform_directory_record(path, record)
    records = _validated_transform_directory_records(chain)
    validate_chain = _load_transform_directory_chain_validator()

    with pytest.raises(ValueError, match=message):
        validate_chain(
            records.affine,
            records.nonlinear,
            records.fixed_crop,
            records.moving_crop,
            chain.fixed_image,
            chain.round_number,
            (0.15, 0.15, 0.15),
        )


@pytest.mark.parametrize(
    ("configured_round", "recorded_round"),
    [
        pytest.param(True, 1, id="boolean"),
        pytest.param(1.0, 1, id="float"),
        pytest.param("1", 1, id="string"),
        pytest.param(-1, -1, id="negative"),
    ],
)
def test_transform_directory_rejects_invalid_configured_round(
    tmp_path: Path,
    configured_round: object,
    recorded_round: int,
) -> None:
    chain = _write_transform_directory_chain(
        tmp_path,
        crop=False,
        round_number=recorded_round,
    )
    records = _validated_transform_directory_records(chain)
    validate_chain = _load_transform_directory_chain_validator()

    with pytest.raises(ValueError, match="round"):
        validate_chain(
            records.affine,
            records.nonlinear,
            records.fixed_crop,
            records.moving_crop,
            chain.fixed_image,
            configured_round,
            (0.15, 0.15, 0.15),
        )


@pytest.mark.parametrize(
    "pixel_size_xyz",
    [
        pytest.param((True, 0.15, 0.15), id="boolean"),
        pytest.param(("0.15", 0.15, 0.15), id="nonnumeric"),
        pytest.param((0.0, 0.15, 0.15), id="zero"),
        pytest.param((-0.15, 0.15, 0.15), id="negative"),
        pytest.param((math.nan, 0.15, 0.15), id="nan"),
        pytest.param((math.inf, 0.15, 0.15), id="infinity"),
        pytest.param((10**400, 0.15, 0.15), id="overflow"),
        pytest.param((0.15, 0.15), id="wrong-length"),
        pytest.param(None, id="noniterable"),
    ],
)
def test_transform_directory_rejects_invalid_configured_pixel_spacing(
    tmp_path: Path,
    pixel_size_xyz: object,
) -> None:
    chain = _write_transform_directory_chain(tmp_path, crop=False)
    records = _validated_transform_directory_records(chain)
    validate_chain = _load_transform_directory_chain_validator()

    with pytest.raises(ValueError, match="pixel spacing configuration"):
        validate_chain(
            records.affine,
            records.nonlinear,
            records.fixed_crop,
            records.moving_crop,
            chain.fixed_image,
            chain.round_number,
            pixel_size_xyz,
        )


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        pytest.param("missing-record", "completion record", id="missing-record"),
        pytest.param("malformed-record", "completion record", id="malformed-record"),
        pytest.param("identity", "identity", id="artifact-identity"),
        pytest.param("stage", "stage", id="wrong-stage"),
        pytest.param("axis", "axis convention", id="legacy-axis"),
    ],
)
def test_transform_directory_artifact_validator_rejects_unsafe_reuse(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    from cyexm.registration.artifacts import (
        ArtifactStore,
        CompletionRecord,
        FileIdentity,
    )
    from cyexm.spatial import AXIS_CONVENTION_VERSION

    artifact = tmp_path / "linear_transform_2.mat"
    artifact.write_bytes(b"affine")
    if failure != "missing-record":
        record_path = _write_notebook_completion_record(
            artifact,
            stage="nonlinear" if failure == "stage" else "affine",
            axis_convention_version=(
                "legacy-ambiguous-v0" if failure == "axis" else AXIS_CONVENTION_VERSION
            ),
        )
        if failure == "malformed-record":
            record_path.write_text("{not-json", encoding="utf-8")
        elif failure == "identity":
            artifact.write_bytes(b"changed-affine")
    validator = _load_notebook_function(
        "validate_cached_artifact",
        {
            "ArtifactStore": ArtifactStore,
            "CompletionRecord": CompletionRecord,
            "FileIdentity": FileIdentity,
            "AXIS_CONVENTION_VERSION": AXIS_CONVENTION_VERSION,
        },
    )

    with pytest.raises(ValueError, match=message):
        validator(artifact, "affine")


@pytest.mark.parametrize("path", _script_paths(), ids=lambda path: path.stem)
def test_script_exposes_typed_main_and_help(
    path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script(path)
    signature = inspect.signature(module.main)
    assert tuple(signature.parameters) == ("argv",)
    assert signature.parameters["argv"].default is None
    hints = get_type_hints(module.main)
    assert hints["return"] is int
    assert hints["argv"] == Sequence[str] | None

    with pytest.raises(SystemExit) as raised:
        module.main(["--help"])
    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "--pixel-size-x" in help_text
    assert "--pixel-size-y" in help_text
    assert "--pixel-size-z" in help_text
    assert "--itk-threads" in help_text
    assert "--blas-threads" in help_text


@pytest.mark.parametrize("path", _script_paths(), ids=lambda path: path.stem)
def test_script_main_parses_paths_and_dispatches_without_registration(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(path)
    positional_count = {
        "whole_image_registration": 3,
        "chunked_registration": 3,
        "linear_registration": 4,
        "nonlinear_registration": 4,
        "apply_transforms": 5,
    }[path.stem]
    argv = [f"input-{index}.tif" for index in range(positional_count)]
    argv.extend(
        [
            "--pixel-size-x",
            "0.41",
            "--pixel-size-y",
            "0.42",
            "--pixel-size-z",
            "1.25",
            "--itk-threads",
            "2",
            "--blas-threads",
            "3",
        ]
    )
    captured: list[object] = []
    monkeypatch.setattr(module, "_run", captured.append)

    assert module.main(argv) == 0
    assert len(captured) == 1
    args = captured[0]
    assert isinstance(args, object)
    assert args.pixel_size_x == 0.41
    assert args.pixel_size_y == 0.42
    assert args.pixel_size_z == 1.25
    assert args.itk_threads == 2
    assert args.blas_threads == 3
    path_values = [value for value in vars(args).values() if isinstance(value, Path)]
    assert len(path_values) == positional_count


@pytest.mark.parametrize(
    "script_name",
    ("whole_image_registration.py", "chunked_registration.py"),
)
def test_workflow_scripts_expose_round_accuracy_and_force_options(
    script_name: str,
) -> None:
    module = _load_script(Path("examples/scripts") / script_name)
    args = module.build_parser().parse_args(
        [
            "fixed.tif",
            "moving.tif",
            "output",
            "--round",
            "7",
            "--linear-accuracy",
            "medium",
            "--nonlinear-accuracy",
            "low",
            "--force",
        ]
    )
    assert args.round_number == 7
    assert args.linear_accuracy == "medium"
    assert args.nonlinear_accuracy == "low"
    assert args.force is True


def test_importing_scripts_keeps_scientific_runtimes_unloaded() -> None:
    code = """
import importlib.util
from pathlib import Path
import sys

for path in sorted(Path("examples/scripts").glob("*.py")):
    spec = importlib.util.spec_from_file_location("example_" + path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

for root in ("ants", "numpy", "scipy", "tifffile"):
    assert not any(name == root or name.startswith(root + ".") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_nonlinear_script_cleanup_cannot_replace_persistence_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(Path("examples/scripts/nonlinear_registration.py"))

    class FailingCandidate:
        def __fspath__(self) -> str:
            return "candidate-warp.nii.gz"

        def unlink(self, *, missing_ok: bool) -> None:
            assert missing_ok is True
            raise OSError("cleanup failed")

    candidate = FailingCandidate()
    monkeypatch.setattr(module, "configure_native_threads", lambda config: None)
    monkeypatch.setattr(module, "load_ants_image", lambda path, spacing: object())
    monkeypatch.setattr(module, "zero_mask", lambda image: object())
    monkeypatch.setattr(
        module,
        "register_image",
        lambda *args, **kwargs: (object(), candidate),
    )

    def fail_persistence(*args: object, **kwargs: object) -> None:
        raise RuntimeError("persist failed")

    monkeypatch.setattr(module, "atomic_write", fail_persistence)
    args = module.build_parser().parse_args(
        ["fixed.tif", "moving.tif", "warp.nii.gz", "registered.tif"]
    )

    with pytest.raises(RuntimeError, match="persist failed"):
        module._run(args)


def test_nonlinear_script_validates_a_committed_warp_by_reopening_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script(Path("examples/scripts/nonlinear_registration.py"))
    warp_path = tmp_path / "warp.nii.gz"
    warp_path.write_bytes(b"warp")
    monkeypatch.setitem(
        sys.modules,
        "ants",
        SimpleNamespace(
            image_read=lambda path: SimpleNamespace(dimension=3),
        ),
    )
    assert module._warp_is_readable(warp_path) is True

    def unreadable(path: str) -> object:
        raise OSError(path)

    monkeypatch.setitem(
        sys.modules,
        "ants",
        SimpleNamespace(image_read=unreadable),
    )
    assert module._warp_is_readable(warp_path) is False


@pytest.mark.parametrize(
    ("interpolation", "expected_range"),
    (("linear", (10.0, 20.0)), ("nearestNeighbor", (2.0, 7.0))),
)
def test_apply_script_forwards_the_canonical_chain_and_output_range(
    interpolation: str,
    expected_range: tuple[float, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(Path("examples/scripts/apply_transforms.py"))

    class Values:
        def __init__(self, minimum: float, maximum: float) -> None:
            self.minimum = minimum
            self.maximum = maximum

        def min(self) -> float:
            return self.minimum

        def max(self) -> float:
            return self.maximum

    class Image:
        def __init__(self, minimum: float, maximum: float) -> None:
            self.values = Values(minimum, maximum)

        def numpy(self) -> Values:
            return self.values

    fixed = Image(0.0, 1.0)
    moving = Image(10.0, 20.0)
    transformed = Image(2.0, 7.0)
    events: list[object] = []

    monkeypatch.setattr(
        module,
        "configure_native_threads",
        lambda config: events.append(("threads", config)),
    )
    monkeypatch.setattr(
        module,
        "load_ants_image",
        lambda path, spacing: fixed if path.name == "fixed.tif" else moving,
    )

    def transform(**kwargs: object) -> object:
        events.append(("transform", kwargs))
        return transformed

    monkeypatch.setattr(module, "transform_image", transform)
    monkeypatch.setattr(
        module,
        "write_uint16_tiff",
        lambda image, path, minimum, maximum: events.append(
            ("write", image, path, minimum, maximum)
        ),
    )
    args = module.build_parser().parse_args(
        [
            "fixed.tif",
            "moving.tif",
            "affine.mat",
            "warp.nii.gz",
            "registered.tif",
            "--interpolation",
            interpolation,
            "--pixel-size-x",
            "0.4",
            "--pixel-size-y",
            "0.5",
            "--pixel-size-z",
            "1.2",
            "--itk-threads",
            "2",
            "--blas-threads",
            "3",
        ]
    )

    module._run(args)

    assert events[0] == ("threads", module.NativeThreadConfig(2, 3))
    transform_kwargs = events[1][1]
    assert transform_kwargs["fixed"] is fixed
    assert transform_kwargs["moving"] is moving
    assert transform_kwargs["affine_path"] == Path("affine.mat")
    assert transform_kwargs["warp_path"] == Path("warp.nii.gz")
    assert transform_kwargs["interpolation"] == interpolation
    assert events[2] == (
        "write",
        transformed,
        Path("registered.tif"),
        *expected_range,
    )


def test_whole_workflow_script_forwards_xyz_threads_accuracy_and_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(Path("examples/scripts/whole_image_registration.py"))
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(module, "register_round", lambda **kwargs: calls.append(kwargs))
    args = module.build_parser().parse_args(
        [
            "fixed.tif",
            "moving.tif",
            "output",
            "--round",
            "4",
            "--pixel-size-x",
            "0.4",
            "--pixel-size-y",
            "0.5",
            "--pixel-size-z",
            "1.2",
            "--itk-threads",
            "2",
            "--blas-threads",
            "3",
            "--linear-accuracy",
            "medium",
            "--nonlinear-accuracy",
            "low",
            "--force",
        ]
    )

    module._run(args)

    assert calls == [
        {
            "fixed_image_path": Path("fixed.tif"),
            "moving_image_path": Path("moving.tif"),
            "save_directory": Path("output"),
            "imaging_round": 4,
            "crop": False,
            "pixel_size": (0.4, 0.5, 1.2),
            "force_override": True,
            "native_threads": module.NativeThreadConfig(2, 3),
            "linear_accuracy": "medium",
            "nonlinear_accuracy": "low",
        }
    ]


def test_chunked_workflow_script_constructs_one_serial_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(Path("examples/scripts/chunked_registration.py"))
    constructed: list[dict[str, object]] = []
    register_calls: list[None] = []

    class Registration:
        def __init__(self, **kwargs: object) -> None:
            constructed.append(kwargs)

        def register(self) -> None:
            register_calls.append(None)

    monkeypatch.setattr(module, "ChunkedImageRegistration", Registration)
    args = module.build_parser().parse_args(
        [
            "fixed.tif",
            "moving.tif",
            "output",
            "--round",
            "4",
            "--chunk-size-z",
            "64",
            "--chunk-size-y",
            "128",
            "--chunk-size-x",
            "256",
            "--overlap-fraction",
            "0.2",
            "--force",
        ]
    )

    module._run(args)

    assert len(constructed) == 1
    assert constructed[0]["chunk_size"] == (64, 128, 256)
    assert constructed[0]["overlap_fraction"] == 0.2
    assert constructed[0]["imaging_round"] == 4
    assert constructed[0]["force_override"] is True
    assert constructed[0]["native_threads"] == module.NativeThreadConfig(1, 1)
    assert register_calls == [None]


def test_nested_example_and_documentation_guidance_is_specific() -> None:
    required = {
        Path("examples/AGENTS.md"): (
            "lab paths",
            "saved outputs",
            "(x, y, z)",
        ),
        Path("examples/scripts/AGENTS.md"): (
            "argparse",
            "NativeThreadConfig",
            "main(argv",
        ),
        Path("examples/notebooks/AGENTS.md"): (
            "first code cell",
            "Path",
            "execution_count",
        ),
        Path("docs/AGENTS.md"): (
            "current behavior",
            "Phase 2",
            "artifact",
        ),
    }
    for path, phrases in required.items():
        text = path.read_text(encoding="utf-8")
        for phrase in phrases:
            assert phrase in text, f"{phrase!r} missing from {path}"


def test_user_documentation_preserves_registration_contract() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for phrase in (
        "git clone https://github.com/TheDeanLab/Cy-ExM.git",
        "uv sync --locked --no-dev",
        "register_round",
        "docs/getting-started.md",
        "docs/registration-foundations.md",
        "docs/workflow-guide.md",
        "LICENSE.md",
    ):
        assert phrase in readme

    documentation = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "docs/registration-foundations.md",
            "docs/native-threading.md",
            "docs/registration-artifacts.md",
            "docs/architecture.md",
        )
    )
    for phrase in (
        "NativeThreadConfig",
        "(x, y, z)",
        "ChunkedImageRegistration",
        "serial",
        "restart",
        "Dask",
        "SLURM",
    ):
        assert phrase in documentation

    migration = Path("docs/migration-from-clearex.md").read_text(encoding="utf-8")
    for phrase in (
        "2311a873b783477add7391ff00245f76490147f0",
        "data_analysis/dean_lab/2025-galecki",
        "clearex.registration.pipeline",
        "clearex.registration.tre",
        "fixed_mask=",
        "aff_iterations",
        "disposable",
        "ZYX",
        *SCRIPT_NAMES,
        *NOTEBOOK_NAMES,
    ):
        assert phrase in migration

    artifacts = Path("docs/registration-artifacts.md").read_text(encoding="utf-8")
    for phrase in (
        "linear_transform_{round}.mat",
        "nonlinear_warp_{round}.nii.gz",
        "nonlinear_registered_{round}.tif",
        "fixed_reference.tif",
        "image_metrics_pre-registration.txt",
        "image_metrics_linearly-registered.txt",
        "image_metrics_nonlinear-registered.txt",
        "fixed_crop_indices_{round}.json",
        "moving_crop_indices_{round}.json",
        "round_{round}_chunks",
        "chunk_{chunk:04d}_warp.nii.gz",
        "manifest.json",
        ".complete.json",
        ".complete.json.stale",
        "schema_version",
        "native_threads",
        "parameters",
        "--force",
    ):
        assert phrase in artifacts
    for phrase in ("affine failure", "nonlinear failure", "chunk failure"):
        assert phrase in artifacts.lower()
