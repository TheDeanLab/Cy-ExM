from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_CLEAR_EX_COMMIT = "2311a873b783477add7391ff00245f76490147f0"
_CLEAR_EX_ROOT = Path("/project/bioinformatics/Danuser_lab/Dean/dean/git/clearex")
_PIXEL_SIZE = (0.2, 0.2, 0.5)
_SHAPE = (16, 32, 32)
_CHUNK_SIZE = (10, 20, 20)
_OVERLAP = 0.2
_ANTS_RANDOM_SEED = "20260719"
_THREAD_VARIABLES = (
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_CHUNK_FIELDS = (
    "chunk_id",
    "z_start",
    "z_end",
    "y_start",
    "y_end",
    "x_start",
    "x_end",
    "z_start_ext",
    "z_end_ext",
    "y_start_ext",
    "y_end_ext",
    "x_start_ext",
    "x_end_ext",
)


def _configure_one_thread() -> None:
    for name in _THREAD_VARIABLES:
        os.environ[name] = "1"
    os.environ["ANTS_RANDOM_SEED"] = _ANTS_RANDOM_SEED


def _git_output(source_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_source_checkout(
    source_root: Path,
    output: Path,
    *,
    expected_commit: str = _CLEAR_EX_COMMIT,
) -> dict[str, object]:
    checkout = source_root.resolve()
    destination = output.resolve()
    if not checkout.is_dir():
        raise FileNotFoundError(f"ClearEx source checkout does not exist: {checkout}")
    if destination.is_relative_to(checkout):
        raise ValueError("baseline output must be outside the ClearEx checkout")

    top_level = Path(_git_output(checkout, "rev-parse", "--show-toplevel")).resolve()
    if top_level != checkout:
        raise RuntimeError(
            f"supplied ClearEx source must be the checkout root: {checkout}"
        )
    commit = _git_output(checkout, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise RuntimeError(
            f"ClearEx source does not match expected pin {expected_commit}: {commit}"
        )
    status = _git_output(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise RuntimeError(
            "ClearEx source checkout must be clean before baseline capture"
        )
    return {
        "checkout_root": str(checkout),
        "source_commit": commit,
        "git_status": "clean",
        "output_directory": str(destination),
        "output_outside_checkout": True,
    }


def _validate_import_provenance(
    modules: Mapping[str, object], source_root: Path
) -> dict[str, str]:
    checkout = source_root.resolve()
    provenance: dict[str, str] = {}
    for name, module in modules.items():
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"imported module {name!r} has no filesystem provenance")
        resolved = Path(module_file).resolve()
        if not resolved.is_relative_to(checkout):
            raise RuntimeError(
                f"imported module {name!r} resolves outside the supplied source "
                f"checkout: {resolved}"
            )
        provenance[name] = str(resolved)
    return provenance


def _synthetic_volume(np: Any) -> Any:
    z, y, x = np.mgrid[:16, :32, :32]
    primary = np.exp(
        -(((z - 5.0) / 2.5) ** 2 + ((y - 11.0) / 5.0) ** 2 + ((x - 20.0) / 4.0) ** 2)
    )
    secondary = np.exp(
        -(((z - 11.0) / 2.0) ** 2 + ((y - 23.0) / 3.5) ** 2 + ((x - 9.0) / 5.5) ** 2)
    )
    ramp = 0.07 * (x / 31.0) + 0.03 * (y / 31.0) + 0.02 * (z / 15.0)
    volume = 0.1 + 0.65 * primary + 0.35 * secondary + ramp
    return volume.astype(np.float32)


def _deterministic_masks(np: Any) -> tuple[Any, Any]:
    """Return asymmetric binary masks independent of the synthetic intensities."""
    z, y, x = np.mgrid[:16, :32, :32]
    ellipsoid = ((z - 7.0) / 5.0) ** 2 + ((y - 13.0) / 8.0) ** 2 + (
        (x - 18.0) / 7.0
    ) ** 2 <= 1.0
    spur = (z >= 9) & (z < 14) & (y >= 20) & (y < 27) & (x >= 5) & (x < 12)
    notch = (z >= 5) & (z < 9) & (y >= 10) & (y < 15) & (x >= 18) & (x < 24)
    fixed = (ellipsoid | spur) & ~notch
    moving = np.roll(fixed, shift=(0, 1, -1), axis=(0, 1, 2))
    return fixed.astype(np.float32), moving.astype(np.float32)


@contextmanager
def _controlled_registration_outputs(
    ants: Any, directory: Path
) -> Iterator[tuple[list[dict[str, object]], list[str]]]:
    """Keep every legacy ANTSpyX native artifact inside one owned directory."""
    original = ants.registration
    signature_parameters = list(inspect.signature(original).parameters)
    calls: list[dict[str, object]] = []

    def registration(**kwargs: object) -> object:
        index = len(calls)
        received_keywords = sorted(kwargs)
        prefix = directory / f"call-{index:02d}-"
        calls.append(
            {
                "index": index,
                "received_keywords": received_keywords,
                "received_fixed_mask": kwargs.get("fixed_mask") is not None,
                "received_mask_keyword": "mask" in kwargs,
                "outprefix": str(prefix),
            }
        )
        controlled = dict(kwargs)
        controlled["outprefix"] = str(prefix)
        return original(**controlled)

    ants.registration = registration
    try:
        yield calls, signature_parameters
    finally:
        ants.registration = original


def _geometry(image: Any) -> dict[str, object]:
    return {
        "shape": [int(value) for value in image.shape],
        "origin": [float(value) for value in image.origin],
        "spacing": [float(value) for value in image.spacing],
        "direction": [
            [float(value) for value in row] for row in image.direction.tolist()
        ],
    }


def _mse(np: Any, fixed: Any, moving: Any) -> float:
    return float(np.mean((np.asarray(fixed) - np.asarray(moving)) ** 2))


def _correlation(np: Any, fixed: Any, moving: Any) -> float:
    correlation = np.corrcoef(np.asarray(fixed).ravel(), np.asarray(moving).ravel())
    return float(correlation[0, 1])


def _chunk_dict(chunk: object) -> dict[str, int]:
    return {name: int(getattr(chunk, name)) for name in _CHUNK_FIELDS}


def _extended_slices(chunk: Any) -> tuple[slice, slice, slice]:
    return (
        slice(int(chunk.z_start_ext), int(chunk.z_end_ext)),
        slice(int(chunk.y_start_ext), int(chunk.y_end_ext)),
        slice(int(chunk.x_start_ext), int(chunk.x_end_ext)),
    )


def _core_slices(chunk: Any) -> tuple[slice, slice, slice]:
    return (
        slice(int(chunk.z_start), int(chunk.z_end)),
        slice(int(chunk.y_start), int(chunk.y_end)),
        slice(int(chunk.x_start), int(chunk.x_end)),
    )


def _core_within_extended(chunk: Any) -> tuple[slice, slice, slice]:
    return (
        slice(
            int(chunk.z_start) - int(chunk.z_start_ext),
            int(chunk.z_end) - int(chunk.z_start_ext),
        ),
        slice(
            int(chunk.y_start) - int(chunk.y_start_ext),
            int(chunk.y_end) - int(chunk.y_start_ext),
        ),
        slice(
            int(chunk.x_start) - int(chunk.x_start_ext),
            int(chunk.x_end) - int(chunk.x_start_ext),
        ),
    )


def _legacy_chunks(
    chunked_type: type,
    fixed: Any,
    moving: Any,
    np: Any,
) -> tuple[list[Any], dict[str, Any], Any, Any]:
    helper = object.__new__(chunked_type)
    helper._log = logging.getLogger("cyexm.legacy-parity.chunk-helper")
    helper.chunk_size = _CHUNK_SIZE
    helper.overlap_fraction = _OVERLAP
    chunks = list(helper._compute_chunk_grid(_SHAPE))

    extractions: dict[str, Any] = {}
    scalar_stitch = np.zeros(_SHAPE, dtype=np.float32)
    vector_stitch = np.zeros((*_SHAPE, 3), dtype=np.float32)
    for chunk in chunks:
        fixed_chunk, moving_chunk = helper.extract_chunk(chunk, fixed, moving)
        chunk_id = int(chunk.chunk_id)
        extractions[f"fixed_{chunk_id:04d}"] = fixed_chunk.numpy()
        extractions[f"moving_{chunk_id:04d}"] = moving_chunk.numpy()

        extended = fixed[_extended_slices(chunk)] + np.float32(chunk_id)
        vector_extended = np.stack(
            (extended, extended + np.float32(1), extended + np.float32(2)),
            axis=-1,
        )
        scalar_stitch[_core_slices(chunk)] = extended[_core_within_extended(chunk)]
        vector_stitch[_core_slices(chunk)] = vector_extended[
            _core_within_extended(chunk)
        ]
    return chunks, extractions, scalar_stitch, vector_stitch


def create_baseline(output: Path, source_root: Path) -> dict[str, object]:
    source_validation = _validate_source_checkout(source_root, output)
    commit = str(source_validation["source_commit"])
    _configure_one_thread()

    import ants
    import clearex
    import clearex.registration as legacy_registration
    import numpy as np
    import scipy
    import tifffile
    from clearex.registration import ChunkedImageRegistration
    from clearex.registration import linear as legacy_linear
    from clearex.registration import nonlinear as legacy_nonlinear

    import_provenance = _validate_import_provenance(
        {
            "clearex": clearex,
            "clearex.registration": legacy_registration,
            "clearex.registration.linear": legacy_linear,
            "clearex.registration.nonlinear": legacy_nonlinear,
        },
        source_root,
    )
    output.mkdir(parents=True, exist_ok=True)
    fixed_data = _synthetic_volume(np)
    moving_data = fixed_data.copy()
    fixed_mask_data, moving_mask_data = _deterministic_masks(np)
    fixed_path = output / "fixed.tif"
    moving_path = output / "moving.tif"
    tifffile.imwrite(fixed_path, fixed_data, metadata={"axes": "ZYX"})
    tifffile.imwrite(moving_path, moving_data, metadata={"axes": "ZYX"})

    fixed = ants.from_numpy(fixed_data, spacing=_PIXEL_SIZE)
    moving = ants.from_numpy(moving_data, spacing=_PIXEL_SIZE)
    fixed_mask = ants.from_numpy(fixed_mask_data, spacing=_PIXEL_SIZE)
    moving_mask = ants.from_numpy(moving_mask_data, spacing=_PIXEL_SIZE)

    with tempfile.TemporaryDirectory(
        dir=output, prefix=".legacy-ants-native-"
    ) as native_directory_text:
        native_directory = Path(native_directory_text)
        with _controlled_registration_outputs(
            ants, native_directory
        ) as registration_characterization:
            registration_calls, antspyx_signature_parameters = (
                registration_characterization
            )
            literal_dry_affine, literal_dry_mask, _literal_dry_transform = (
                legacy_linear.register_image(
                    moving,
                    fixed,
                    fixed_mask=fixed_mask,
                    moving_mask=moving_mask,
                    accuracy="dry run",
                    verbose=False,
                )
            )
            literal_dry_nonlinear, _literal_dry_warp = legacy_nonlinear.register_image(
                literal_dry_affine,
                fixed,
                moving_mask=literal_dry_mask,
                fixed_mask=fixed_mask,
                accuracy="dry run",
                verbose=False,
            )
            literal_dry_arrays = {
                "affine_result": literal_dry_affine.numpy(),
                "affine_mask": literal_dry_mask.numpy(),
                "nonlinear_result": literal_dry_nonlinear.numpy(),
            }

            # The matched numerical gate includes only parameters both wrappers
            # actually deliver to ANTSpyX. ClearEx forwards fixed masks with the
            # unconsumed ``fixed_mask`` keyword, so both matched calls omit it.
            affine_image, affine_mask, affine = legacy_linear.register_image(
                moving,
                fixed,
                fixed_mask=None,
                moving_mask=moving_mask,
                registration_type="TRSAA",
                accuracy="low",
                verbose=False,
            )
            affine_path = output / "affine.mat"
            ants.write_transform(affine, str(affine_path))
            nonlinear_image, temporary_warp = legacy_nonlinear.register_image(
                affine_image,
                fixed,
                moving_mask=affine_mask,
                fixed_mask=None,
                accuracy="dry run",
                verbose=False,
            )
            warp = ants.image_read(str(temporary_warp))
            warp_path = output / "nonlinear_warp.nii.gz"
            ants.image_write(warp, str(warp_path))
            raw_nonlinear_mask = ants.apply_transforms(
                fixed=fixed,
                moving=affine_mask,
                transformlist=[str(warp_path)],
                whichtoinvert=[False],
                interpolator="nearestNeighbor",
            )
            nonlinear_mask = ants.threshold_image(
                raw_nonlinear_mask,
                low_thresh=0.5,
                high_thresh=1.0,
                inval=1,
                outval=0,
            )
            transform_order = ants.apply_transforms(
                fixed=fixed,
                moving=moving,
                transformlist=[str(warp_path), str(affine_path)],
                whichtoinvert=[False, False],
                interpolator="linear",
            )
    if native_directory.exists():
        raise RuntimeError("legacy ANTSpyX native artifact directory was not removed")
    if len(registration_calls) != 4:
        raise RuntimeError(
            "legacy baseline did not execute the four characterized calls"
        )
    for call, label in zip(
        registration_calls,
        ("literal_linear", "literal_nonlinear", "matched_linear", "matched_nonlinear"),
        strict=True,
    ):
        call["label"] = label

    for name, array in literal_dry_arrays.items():
        np.save(output / f"literal_dry_{name}.npy", array, allow_pickle=False)

    arrays = {
        "fixed": fixed_data,
        "moving": moving_data,
        "fixed_mask_input": fixed_mask_data,
        "moving_mask_input": moving_mask_data,
        "affine_result": affine_image.numpy(),
        "affine_mask": affine_mask.numpy(),
        "nonlinear_result": nonlinear_image.numpy(),
        "nonlinear_mask": nonlinear_mask.numpy(),
        "transform_order": transform_order.numpy(),
        "warp": warp.numpy(),
    }
    for name, array in arrays.items():
        np.save(output / f"{name}.npy", array, allow_pickle=False)

    chunks, extractions, scalar_stitch, vector_stitch = _legacy_chunks(
        ChunkedImageRegistration,
        fixed_data,
        moving_data,
        np,
    )
    np.savez_compressed(output / "chunk_extractions.npz", **extractions)
    np.save(output / "stitch_scalar.npy", scalar_stitch, allow_pickle=False)
    np.save(output / "stitch_vector.npy", vector_stitch, allow_pickle=False)

    summary: dict[str, object] = {
        "schema_version": 2,
        "source_commit": commit,
        "source_validation": source_validation,
        "import_provenance": import_provenance,
        "native_artifact_cleanup": {
            "strategy": "caller-owned temporary directory",
            "directory": str(native_directory),
            "removed": True,
        },
        "versions": {
            "python": os.sys.version.split()[0],
            "antspyx": ants.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "tifffile": tifffile.__version__,
        },
        "threads": {name: os.environ[name] for name in _THREAD_VARIABLES},
        "ants_random_seed": os.environ["ANTS_RANDOM_SEED"],
        "comparison_policy": {
            "linear_registration_type": "TRSAA",
            "linear_accuracy": "low",
            "nonlinear_registration_type": "SyNOnly",
            "nonlinear_accuracy": "dry run",
            "fixed_mask": None,
            "moving_mask": "deterministic asymmetric binary mask",
            "reason": (
                "shared consumed parameters; fixed masks are excluded because "
                "ClearEx forwards an ANTSpyX keyword that is not consumed"
            ),
        },
        "fixed_mask_characterization": {
            "classification": "corrected_legacy_defect",
            "legacy_wrapper_keyword": "fixed_mask",
            "antspyx_consumed_keyword": "mask",
            "antspyx_signature_parameters": antspyx_signature_parameters,
            "legacy_keyword_in_antspyx_signature": (
                "fixed_mask" in antspyx_signature_parameters
            ),
            "corrected_keyword_in_antspyx_signature": (
                "mask" in antspyx_signature_parameters
            ),
            "literal_fixed_mask_provided": True,
            "matched_fixed_mask": None,
            "registration_calls": registration_calls,
        },
        "literal_dry_run": {
            "classification": "expected_divergence",
            "linear_registration_type_argument": "TRSAA",
            "linear_accuracy_argument": "dry run",
            "effective_linear_registration_type": "Rigid",
            "nonlinear_registration_type_argument": "SyNOnly",
            "nonlinear_accuracy_argument": "dry run",
            "shapes": {
                name: list(array.shape) for name, array in literal_dry_arrays.items()
            },
            "finite": {
                name: bool(np.isfinite(array).all())
                for name, array in literal_dry_arrays.items()
            },
        },
        "geometry": {
            "fixed": _geometry(fixed),
            "affine": _geometry(affine_image),
            "nonlinear": _geometry(nonlinear_image),
            "warp": _geometry(warp),
        },
        "shapes": {name: list(array.shape) for name, array in arrays.items()},
        "finite": {
            name: bool(np.isfinite(array).all()) for name, array in arrays.items()
        },
        "similarity": {
            "pre_mse": _mse(np, fixed_data, moving_data),
            "affine_mse": _mse(np, fixed_data, arrays["affine_result"]),
            "nonlinear_mse": _mse(np, fixed_data, arrays["nonlinear_result"]),
            "pre_correlation": _correlation(np, fixed_data, moving_data),
            "affine_correlation": _correlation(np, fixed_data, arrays["affine_result"]),
            "nonlinear_correlation": _correlation(
                np, fixed_data, arrays["nonlinear_result"]
            ),
        },
        "transform_types": {
            "affine": str(affine.transform_type),
            "warp_has_components": bool(warp.has_components),
            "warp_dtype": str(warp.numpy().dtype),
            "forward_order": ["nonlinear_warp", "affine"],
            "inversion_flags": [False, False],
        },
        "mask_values": {
            "fixed_input": [float(value) for value in np.unique(fixed_mask_data)],
            "moving_input": [float(value) for value in np.unique(moving_mask_data)],
            "affine": [float(value) for value in np.unique(affine_mask.numpy())],
            "nonlinear": [float(value) for value in np.unique(nonlinear_mask.numpy())],
        },
        "chunk": {
            "image_shape_zyx": list(_SHAPE),
            "chunk_size_zyx": list(_CHUNK_SIZE),
            "overlap_fraction": _OVERLAP,
            "grid": [_chunk_dict(chunk) for chunk in chunks],
            "extended_shapes": [
                list(fixed_data[_extended_slices(chunk)].shape) for chunk in chunks
            ],
            "core_shapes": [
                [
                    int(chunk.z_end) - int(chunk.z_start),
                    int(chunk.y_end) - int(chunk.y_start),
                    int(chunk.x_end) - int(chunk.x_start),
                ]
                for chunk in chunks
            ],
            "extraction_keys": sorted(extractions),
            "stitch_order": [int(chunk.chunk_id) for chunk in chunks],
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture the pinned ClearEx registration baseline."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=_CLEAR_EX_ROOT)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    summary = create_baseline(
        arguments.output.resolve(), arguments.source_root.resolve()
    )
    print(
        "PASS legacy baseline "
        f"commit={summary['source_commit']} output={arguments.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
