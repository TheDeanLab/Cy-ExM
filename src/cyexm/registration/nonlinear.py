from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
    """Return the histogram-matched scalar image and ANTs forward-warp path."""
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
        "fixed_mask": fixed_mask,
        "moving_mask": moving_mask,
        "type_of_transform": registration_type,
        "flow_sigma": 3.0,
        "total_sigma": 0.5,
        "shrink_factors": [4, 2, 1],
        "smoothing_sigmas": [2, 1, 0],
        "syn_metric": "mattes",
        "metric_weight": 1.0,
        "number_of_bins": 32,
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
    warp_path = Path(registered["fwdtransforms"][0])
    if not warp_path.is_file():
        raise FileNotFoundError(f"ANTs forward warp does not exist: {warp_path}")

    transformed_image = ants.resample_image_to_target(
        image=registered["warpedmovout"],
        target=fixed_image,
        interp_type="linear",
    )
    transformed_image = ants.histogram_match_image(
        source_image=transformed_image,
        reference_image=moving_image,
    )
    return transformed_image, warp_path


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
