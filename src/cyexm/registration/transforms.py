from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cyexm.runtime import NativeThreadConfig, configure_native_threads

logger = logging.getLogger(__name__)


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


def apply_affine(
    *,
    fixed: Any,
    moving: Any,
    transform: Any,
    interpolation: str,
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Apply an in-memory affine transform in the fixed image space."""
    _configured_runtime(native_threads)
    return transform.apply_to_image(
        image=moving,
        reference=fixed,
        interpolation=interpolation,
    )


def apply_warp(
    *,
    fixed: Any,
    moving: Any,
    warp_path: Path,
    interpolation: str,
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Apply a forward displacement field without inversion."""
    ants, _ = _configured_runtime(native_threads)
    return ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=[str(warp_path)],
        whichtoinvert=[False],
        interpolator=interpolation,
    )


def apply_transform_chain(
    *,
    fixed: Any,
    moving: Any,
    affine_path: Path,
    warp_path: Path,
    interpolation: str,
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Apply the forward warp and affine in their established list order."""
    ants, _ = _configured_runtime(native_threads)
    return ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=[str(warp_path), str(affine_path)],
        whichtoinvert=[False, False],
        interpolator=interpolation,
    )


def transform_image(
    *,
    fixed: Any,
    moving: Any,
    affine_path: Path,
    warp_path: Path,
    interpolation: str = "linear",
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Apply a persisted warp-plus-affine chain to an image."""
    return apply_transform_chain(
        fixed=fixed,
        moving=moving,
        affine_path=affine_path,
        warp_path=warp_path,
        interpolation=interpolation,
        native_threads=native_threads,
    )


def export_affine_transform(
    affine_transform: object,
    path: str | Path,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> None:
    """Write an affine transform exactly to the supplied artifact path."""
    ants, _ = _configured_runtime(native_threads)
    ants.write_transform(affine_transform, str(path))


def import_affine_transform(
    path: str | Path,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Read an affine transform from the supplied artifact path."""
    ants, _ = _configured_runtime(native_threads)
    return ants.read_transform(str(path))


def set_origin_and_spacing(
    image: Any,
    origin: tuple[float, float, float] | None = None,
    spacing: tuple[float, float, float] | None = None,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Update optional XYZ origin and spacing values in place."""
    resolved = configure_native_threads(native_threads)
    logger.info("Native thread settings: %s", resolved.describe())
    if origin is not None:
        image.set_origin(new_origin=origin)
    if spacing is not None:
        image.set_spacing(new_spacing=spacing)
    logger.info("Image origin set to: %s", image.origin)
    logger.info("Image spacing (pixel size in XYZ) set to: %s", image.spacing)
    return image


__all__ = [
    "apply_affine",
    "apply_transform_chain",
    "apply_warp",
    "export_affine_transform",
    "import_affine_transform",
    "set_origin_and_spacing",
    "transform_image",
]
