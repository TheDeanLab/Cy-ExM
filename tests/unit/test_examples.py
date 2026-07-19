from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import get_type_hints

import pytest

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
    assert "origin=origin" in source
    assert "spacing=image.spacing" in source
    assert "direction=image.direction" in source
    assert "if bounds is None" in source


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


def test_user_documentation_matches_phase_one_contract() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for phrase in (
        "uv sync --extra examples",
        "NativeThreadConfig",
        "(x, y, z)",
        "ChunkedImageRegistration",
        "serial",
        "restart",
        "Dask",
        "SLURM",
        "Phase 2",
        "docs/migration-from-clearex.md",
        "docs/registration-artifacts.md",
    ):
        assert phrase in readme

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
