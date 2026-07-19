from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from cyexm.registration._native_paths import (
    cleanup_registration_transforms,
    registration_transform_paths,
)
from cyexm.runtime import NativeThreadConfig, configure_native_threads

logger = logging.getLogger(__name__)

_SUPPORTED_REGISTRATION_TYPES = frozenset({"Elastic", "SyNOnly", "SyN"})
_ITERATION_SCHEDULES = {
    "high": (100, 70, 50),
    "medium": (50, 35, 25),
    "low": (5, 5, 5),
    "dry run": (1, 1, 1),
}


def _scientific_runtime() -> tuple[Any, Any]:
    import ants
    import numpy as np

    return ants, np


def _configured_runtime(
    native_threads: NativeThreadConfig | None,
) -> tuple[Any, Any]:
    resolved = configure_native_threads(native_threads)
    logger.info("Native thread settings: %s", resolved.describe())
    return _scientific_runtime()


def _validate_policy(registration_type: str, accuracy: str) -> None:
    if registration_type not in _SUPPORTED_REGISTRATION_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_REGISTRATION_TYPES))
        raise ValueError(
            f"Unsupported registration type: {registration_type}. "
            f"Supported types are: {supported}."
        )
    if accuracy not in _ITERATION_SCHEDULES:
        supported = ", ".join(sorted(_ITERATION_SCHEDULES))
        raise ValueError(
            f"Unsupported accuracy: {accuracy}. Supported values are: {supported}."
        )


def _is_displacement_field(path: Path) -> bool:
    return path.name.endswith((".nii", ".nii.gz"))


def _materialize_forward_warp(
    ants: Any,
    *,
    fixed_image: Any,
    moving_image: Any,
    forward_transforms: object,
) -> Path:
    if not isinstance(forward_transforms, list | tuple) or not forward_transforms:
        raise FileNotFoundError("ANTs registration did not produce a forward warp")
    transform_paths = [Path(path) for path in forward_transforms]
    for path in transform_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"ANTs forward warp component does not exist: {path}"
            )
    if len(transform_paths) == 1 and _is_displacement_field(transform_paths[0]):
        return transform_paths[0]

    descriptor, prefix_text = tempfile.mkstemp(
        dir=transform_paths[0].parent,
        prefix="cyexm-composite-",
        suffix="-",
    )
    os.close(descriptor)
    prefix = Path(prefix_text)
    prefix.unlink()
    composed = ants.apply_transforms(
        fixed=fixed_image,
        moving=moving_image,
        transformlist=[str(path) for path in transform_paths],
        whichtoinvert=[False] * len(transform_paths),
        interpolator="linear",
        compose=str(prefix),
    )
    if composed is None:
        raise FileNotFoundError("ANTs did not materialize the composite forward warp")
    warp_path = Path(composed)
    if not warp_path.is_file():
        raise FileNotFoundError(
            f"ANTs composite forward warp does not exist: {warp_path}"
        )
    return warp_path


def register_image(
    moving_image: Any,
    fixed_image: Any,
    moving_mask: Any | None = None,
    fixed_mask: Any | None = None,
    registration_type: str = "SyNOnly",
    accuracy: str = "high",
    verbose: bool = False,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> tuple[object, Path]:
    """Return the histogram-matched scalar and one materialized forward warp."""
    _validate_policy(registration_type, accuracy)
    ants, np = _configured_runtime(native_threads)

    if isinstance(fixed_image, np.ndarray):
        fixed_image = ants.from_numpy(fixed_image)
    if isinstance(moving_image, np.ndarray):
        moving_image = ants.from_numpy(moving_image)
    if fixed_image.dimension != moving_image.dimension:
        raise ValueError("Both images must have the same number of dimensions.")

    kwargs: dict[str, object] = {
        "fixed": fixed_image,
        "moving": moving_image,
        "mask": fixed_mask,
        "moving_mask": moving_mask,
        "type_of_transform": registration_type,
        "flow_sigma": 3.0,
        "total_sigma": 0.5,
        "syn_metric": "mattes",
        "syn_sampling": 32,
        "singleprecision": False,
        "aff_random_sampling_rate": 1.0,
        "verbose": verbose,
        "initial_transform": "Identity",
        "mask_all_stages": True,
        "reg_iterations": _ITERATION_SCHEDULES[accuracy],
    }
    logger.info(
        "Using %s registration with %s iterations.",
        registration_type,
        kwargs["reg_iterations"],
    )
    registered = ants.registration(**kwargs)
    native_paths = registration_transform_paths(registered)
    warp_path: Path | None = None
    completed = False
    try:
        warp_path = _materialize_forward_warp(
            ants,
            fixed_image=fixed_image,
            moving_image=moving_image,
            forward_transforms=registered["fwdtransforms"],
        )

        transformed_image = ants.resample_image_to_target(
            image=registered["warpedmovout"],
            target=fixed_image,
            interp_type="linear",
        )
        transformed_image = ants.histogram_match_image(
            source_image=transformed_image,
            reference_image=moving_image,
        )
        completed = True
        return transformed_image, warp_path
    finally:
        returned = () if warp_path is None else (warp_path,)
        cleanup_registration_transforms(
            native_paths | set(returned),
            preserve=returned if completed else (),
        )


def transform_image(
    moving_image: Any,
    fixed_image: Any,
    transformed_image: Any,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Apply an existing forward warp and match the source histogram."""
    ants, np = _configured_runtime(native_threads)
    if isinstance(fixed_image, np.ndarray):
        fixed_image = ants.from_numpy(fixed_image)
    if isinstance(moving_image, np.ndarray):
        moving_image = ants.from_numpy(moving_image)

    transformlist = list(transformed_image["fwdtransforms"])
    warped_image = ants.apply_transforms(
        fixed=fixed_image,
        moving=moving_image,
        transformlist=transformlist,
        whichtoinvert=[False] * len(transformlist),
        interpolator="linear",
    )
    return ants.histogram_match_image(warped_image, moving_image)


def inspect_warp_transform(
    image_path: str | Path,
    transform_path: str | Path,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> dict[str, float | int]:
    """Print displacement-magnitude statistics inside an image-derived mask."""
    ants, np = _configured_runtime(native_threads)
    import tifffile

    warp = ants.image_read(filename=str(transform_path))
    image = ants.from_numpy(data=tifffile.imread(image_path))
    mask = ants.get_mask(image)
    warp_array = warp.numpy()
    dimension = warp.dimension
    if warp_array.ndim == dimension + 1 and warp_array.shape[-1] == dimension:
        displacement = warp_array
    elif warp_array.ndim == dimension + 1 and warp_array.shape[0] == dimension:
        displacement = np.moveaxis(warp_array, 0, -1)
    else:
        raise ValueError(
            f"Unexpected warp array shape {warp_array.shape} for dim={dimension}"
        )

    amplitude = np.linalg.norm(displacement, axis=-1)
    values = amplitude[mask.numpy().astype(bool)]
    values = values[np.isfinite(values)]
    stats: dict[str, float | int] = {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "std": float(values.std(ddof=0)),
        "max": float(values.max()),
        "n_vox": int(values.size),
    }
    print(stats)
    return stats


__all__ = ["inspect_warp_transform", "register_image", "transform_image"]
