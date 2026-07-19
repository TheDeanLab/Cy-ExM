from __future__ import annotations

import logging
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from cyexm.io.images import ants_geometry, load_ants_image, zero_mask
from cyexm.io.tiff import write_uint16_tiff
from cyexm.runtime import NativeThreadConfig, configure_native_threads
from cyexm.spatial import ImageGeometry, PixelSize

from . import linear as linear_kernel
from . import nonlinear as nonlinear_kernel
from . import transforms as transform_helpers

logger = logging.getLogger(__name__)


def _scientific_runtime() -> tuple[Any, Any]:
    import ants
    import numpy as np

    return ants, np


@runtime_checkable
class RegistrationBackend(Protocol):
    def load_image(self, path: Path, pixel_size_xyz: PixelSize) -> object:
        raise NotImplementedError

    def zero_mask(self, image: object) -> object:
        raise NotImplementedError

    def geometry(self, image: object) -> ImageGeometry:
        raise NotImplementedError

    def to_array(self, image: object) -> object:
        raise NotImplementedError

    def from_array(
        self,
        array: object,
        *,
        reference: object,
        has_components: bool = False,
    ) -> object:
        raise NotImplementedError

    def crop_image(
        self,
        image: object,
        bounds_zyx: tuple[int, int, int, int, int, int],
    ) -> object:
        raise NotImplementedError

    def linear(self, **kwargs: object) -> tuple[object, object | None, object]:
        raise NotImplementedError

    def nonlinear(self, **kwargs: object) -> tuple[object, Path]:
        raise NotImplementedError

    def apply_affine(
        self,
        *,
        fixed: object,
        moving: object,
        transform: object,
        interpolation: str,
    ) -> object:
        raise NotImplementedError

    def apply_warp(
        self,
        *,
        fixed: object,
        moving: object,
        warp_path: Path,
        interpolation: str,
    ) -> object:
        raise NotImplementedError

    def write_affine(self, transform: object, path: Path) -> None:
        raise NotImplementedError

    def read_affine(self, path: Path) -> object:
        raise NotImplementedError

    def write_warp(self, warp: object, path: Path) -> None:
        raise NotImplementedError

    def read_warp(self, path: Path) -> object:
        raise NotImplementedError

    def write_identity_warp(self, reference: object, path: Path) -> None:
        raise NotImplementedError

    def write_tiff(
        self,
        image: object,
        path: Path,
        *,
        min_value: float,
        max_value: float,
    ) -> None:
        raise NotImplementedError


class AntsRegistrationBackend:
    def __init__(self, native_threads: NativeThreadConfig | None = None) -> None:
        self.native_threads = native_threads

    def _prepare(self) -> None:
        resolved = configure_native_threads(self.native_threads)
        logger.info("Native thread settings: %s", resolved.describe())

    def load_image(self, path: Path, pixel_size_xyz: PixelSize) -> object:
        self._prepare()
        return load_ants_image(path, pixel_size_xyz)

    def zero_mask(self, image: object) -> object:
        self._prepare()
        return zero_mask(image)

    def geometry(self, image: object) -> ImageGeometry:
        self._prepare()
        return ants_geometry(image)

    def to_array(self, image: object) -> object:
        self._prepare()
        runtime_image: Any = image
        return runtime_image.numpy()

    def from_array(
        self,
        array: object,
        *,
        reference: object,
        has_components: bool = False,
    ) -> object:
        self._prepare()
        ants, np = _scientific_runtime()
        runtime_reference: Any = reference
        return ants.from_numpy(
            np.asarray(array),
            origin=runtime_reference.origin,
            spacing=runtime_reference.spacing,
            direction=runtime_reference.direction,
            has_components=has_components,
        )

    def crop_image(
        self,
        image: object,
        bounds_zyx: tuple[int, int, int, int, int, int],
    ) -> object:
        self._prepare()
        ants, np = _scientific_runtime()
        runtime_image: Any = image
        if len(bounds_zyx) != 6 or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in bounds_zyx
        ):
            raise TypeError("bounds_zyx must contain exactly six integers")
        z0, z1, y0, y1, x0, x1 = (int(value) for value in bounds_zyx)
        data = np.asarray(runtime_image.numpy())
        if data.ndim not in {3, 4}:
            raise ValueError("image data must have three spatial dimensions")
        spatial_shape = data.shape[:3]
        pairs = ((z0, z1), (y0, y1), (x0, x1))
        if any(
            start < 0 or stop > size or start >= stop
            for (start, stop), size in zip(pairs, spatial_shape, strict=True)
        ):
            raise ValueError("bounds_zyx must define a non-empty image region")
        cropped = data[z0:z1, y0:y1, x0:x1, ...]

        start_xyz = (x0, y0, z0)
        scaled_offset = tuple(
            start * spacing
            for start, spacing in zip(start_xyz, runtime_image.spacing, strict=True)
        )
        physical_offset = tuple(
            sum(
                float(runtime_image.direction[row][column]) * scaled_offset[column]
                for column in range(3)
            )
            for row in range(3)
        )
        origin = tuple(
            float(base) + delta
            for base, delta in zip(runtime_image.origin, physical_offset, strict=True)
        )
        return ants.from_numpy(
            cropped,
            origin=origin,
            spacing=runtime_image.spacing,
            direction=runtime_image.direction,
            has_components=bool(getattr(runtime_image, "has_components", False)),
        )

    def linear(self, **kwargs: object) -> tuple[object, object | None, object]:
        register: Any = linear_kernel.register_image
        return register(
            **kwargs,
            native_threads=self.native_threads,
        )

    def nonlinear(self, **kwargs: object) -> tuple[object, Path]:
        register: Any = nonlinear_kernel.register_image
        return register(
            **kwargs,
            native_threads=self.native_threads,
        )

    def apply_affine(
        self,
        *,
        fixed: object,
        moving: object,
        transform: object,
        interpolation: str,
    ) -> object:
        return transform_helpers.apply_affine(
            fixed=fixed,
            moving=moving,
            transform=transform,
            interpolation=interpolation,
            native_threads=self.native_threads,
        )

    def apply_warp(
        self,
        *,
        fixed: object,
        moving: object,
        warp_path: Path,
        interpolation: str,
    ) -> object:
        return transform_helpers.apply_warp(
            fixed=fixed,
            moving=moving,
            warp_path=warp_path,
            interpolation=interpolation,
            native_threads=self.native_threads,
        )

    def write_affine(self, transform: object, path: Path) -> None:
        transform_helpers.export_affine_transform(
            transform,
            path,
            native_threads=self.native_threads,
        )

    def read_affine(self, path: Path) -> object:
        return transform_helpers.import_affine_transform(
            path,
            native_threads=self.native_threads,
        )

    def write_warp(self, warp: object, path: Path) -> None:
        self._prepare()
        ants, _ = _scientific_runtime()
        ants.image_write(warp, str(path))

    def read_warp(self, path: Path) -> object:
        self._prepare()
        ants, _ = _scientific_runtime()
        return ants.image_read(str(path))

    def write_identity_warp(self, reference: object, path: Path) -> None:
        self._prepare()
        ants, np = _scientific_runtime()
        runtime_reference: Any = reference
        shape = (*runtime_reference.shape, runtime_reference.dimension)
        displacement = np.zeros(shape, dtype=np.float32)
        warp = ants.from_numpy(
            displacement,
            origin=runtime_reference.origin,
            spacing=runtime_reference.spacing,
            direction=runtime_reference.direction,
            has_components=True,
        )
        ants.image_write(warp, str(path))

    def write_tiff(
        self,
        image: object,
        path: Path,
        *,
        min_value: float,
        max_value: float,
    ) -> None:
        self._prepare()
        write_uint16_tiff(image, path, min_value, max_value)


__all__ = ["AntsRegistrationBackend", "RegistrationBackend"]
