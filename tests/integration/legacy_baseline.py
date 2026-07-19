from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
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


def _source_commit(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


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
    commit = _source_commit(source_root)
    if commit != _CLEAR_EX_COMMIT:
        raise RuntimeError(
            f"ClearEx source mismatch: expected {_CLEAR_EX_COMMIT}, found {commit}"
        )
    _configure_one_thread()

    import ants
    import numpy as np
    import scipy
    import tifffile
    from clearex.registration import ChunkedImageRegistration
    from clearex.registration import linear as legacy_linear
    from clearex.registration import nonlinear as legacy_nonlinear

    output.mkdir(parents=True, exist_ok=True)
    fixed_data = _synthetic_volume(np)
    moving_data = fixed_data.copy()
    fixed_path = output / "fixed.tif"
    moving_path = output / "moving.tif"
    tifffile.imwrite(fixed_path, fixed_data, metadata={"axes": "ZYX"})
    tifffile.imwrite(moving_path, moving_data, metadata={"axes": "ZYX"})

    fixed = ants.from_numpy(fixed_data, spacing=_PIXEL_SIZE)
    moving = ants.from_numpy(moving_data, spacing=_PIXEL_SIZE)
    fixed_mask = ants.from_numpy(
        (fixed_data > 0).astype(np.float32), spacing=_PIXEL_SIZE
    )
    moving_mask = ants.from_numpy(
        (moving_data > 0).astype(np.float32), spacing=_PIXEL_SIZE
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
    for name, array in literal_dry_arrays.items():
        np.save(output / f"literal_dry_{name}.npy", array, allow_pickle=False)

    # Shared TRSAA-low parameters provide a numerical migration comparison. The
    # literal legacy dry-run call above is retained separately because it wires its
    # Rigid schedule through names that ANTSpyX does not consume for affine stages.
    affine_image, affine_mask, affine = legacy_linear.register_image(
        moving,
        fixed,
        fixed_mask=fixed_mask,
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
        fixed_mask=fixed_mask,
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

    arrays = {
        "fixed": fixed_data,
        "moving": moving_data,
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
        "schema_version": 1,
        "source_commit": commit,
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
            "reason": (
                "shared consumed parameters; legacy literal dry-run is captured "
                "separately because its affine schedule uses ineffective names"
            ),
        },
        "literal_dry_run": {
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
