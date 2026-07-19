from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cyexm.spatial import ImageGeometry, PixelSize

_ANTSPYX_REGISTRATION_PARAMETERS = frozenset(
    {
        "fixed",
        "moving",
        "type_of_transform",
        "initial_transform",
        "outprefix",
        "mask",
        "moving_mask",
        "mask_all_stages",
        "grad_step",
        "flow_sigma",
        "total_sigma",
        "aff_metric",
        "aff_sampling",
        "aff_random_sampling_rate",
        "syn_metric",
        "syn_sampling",
        "reg_iterations",
        "aff_iterations",
        "aff_shrink_factors",
        "aff_smoothing_sigmas",
        "write_composite_transform",
        "verbose",
        "multivariate_extras",
        "restrict_transformation",
        "smoothing_in_mm",
        "singleprecision",
        "use_legacy_histogram_matching",
    }
)


@dataclass
class FakeImage:
    data: np.ndarray = field(
        default_factory=lambda: np.arange(8**3, dtype=np.float32).reshape(8, 8, 8)
    )
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    direction: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    has_components: bool = False
    applied_interpolations: list[str] = field(default_factory=list)
    index_to_physical_calls: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def dimension(self) -> int:
        return 3

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape[:3]

    @property
    def components(self) -> int:
        return self.data.shape[-1] if self.has_components else 1

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def min(self) -> np.generic:
        return self.data.min()

    def max(self) -> np.generic:
        return self.data.max()

    def clone(self) -> FakeImage:
        return FakeImage(
            data=self.data.copy(),
            origin=self.origin,
            spacing=self.spacing,
            direction=self.direction,
            has_components=self.has_components,
        )

    def index_to_physical_point(
        self, index: tuple[int, int, int]
    ) -> tuple[float, float, float]:
        self.index_to_physical_calls.append(index)
        scaled = tuple(
            coordinate * spacing
            for coordinate, spacing in zip(index, self.spacing, strict=True)
        )
        offset = tuple(
            sum(self.direction[row][column] * scaled[column] for column in range(3))
            for row in range(3)
        )
        return tuple(
            base + delta for base, delta in zip(self.origin, offset, strict=True)
        )  # type: ignore[return-value]

    def apply_to_image(
        self,
        image: object,
        *,
        reference: object,
        interpolation: str,
    ) -> FakeImage:
        del image, reference
        self.applied_interpolations.append(interpolation)
        return self.clone()

    def set_origin(self, *, new_origin: tuple[float, float, float]) -> None:
        self.origin = new_origin

    def set_spacing(self, *, new_spacing: tuple[float, float, float]) -> None:
        self.spacing = new_spacing


@dataclass
class FakeTransform(FakeImage):
    parameters: np.ndarray = field(
        default_factory=lambda: np.concatenate((np.eye(3).ravel(), np.zeros(3)))
    )
    fixed_parameters: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class FakeAntsCalls:
    registration: dict[str, object] = field(default_factory=dict)
    registration_call_count: int = 0
    transformlist: list[str] = field(default_factory=list)
    whichtoinvert: list[bool] | None = None
    interpolator: str | None = None
    transform: FakeTransform = field(default_factory=FakeTransform)
    warp_path: Path = Path("fake-warp.nii.gz")
    forward_transforms: list[Path] | None = None
    inverse_transforms: list[Path] = field(default_factory=list)
    compose: str | None = None
    composed_path: Path | None = None
    histogram_matched: FakeImage = field(default_factory=FakeImage)
    resampled: FakeImage = field(default_factory=FakeImage)
    applied: FakeImage = field(default_factory=FakeImage)
    thresholded: FakeImage = field(default_factory=FakeImage)
    registered_warped: FakeImage = field(default_factory=FakeImage)
    read_warp: FakeImage = field(default_factory=FakeImage)
    threshold: dict[str, object] = field(default_factory=dict)
    resample: dict[str, object] = field(default_factory=dict)
    histogram_match: dict[str, object] = field(default_factory=dict)
    apply_transforms: dict[str, object] = field(default_factory=dict)
    read_transform_path: str | None = None
    image_read_path: str | None = None
    image_write_path: str | None = None
    image_write_image: object | None = None
    write_transform_path: str | None = None
    write_transform_value: object | None = None
    from_numpy_calls: list[dict[str, object]] = field(default_factory=list)
    index_to_physical_calls: list[tuple[object, tuple[int, int, int]]] = field(
        default_factory=list
    )
    events: list[str] = field(default_factory=list)


def fake_image(
    *,
    data: np.ndarray | None = None,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    direction: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    has_components: bool = False,
) -> FakeImage:
    return FakeImage(
        data=(
            np.arange(8**3, dtype=np.float32).reshape(8, 8, 8)
            if data is None
            else np.array(data, copy=True)
        ),
        origin=origin,
        spacing=spacing,
        direction=direction,
        has_components=has_components,
    )


@dataclass
class FakeBackend:
    fail_linear: bool = False
    fail_nonlinear: bool = False
    fail_affine_write: bool = False
    fail_affine_interpolation: str | None = None
    fail_warp_interpolation: str | None = None
    fail_histogram_match: bool = False
    fail_threshold_mask: bool = False
    fail_to_array_call: int | None = None
    fail_tiff_role: str | None = None
    fail_crop_image: bool = False
    fail_metrics: bool = False
    fail_scalar_from_array: bool = False
    affine_bytes: bytes = b"affine-fake"
    warp_bytes: bytes = b"warp-fake"
    metric_values: dict[str, float] = field(
        default_factory=lambda: {
            "Correlation": 0.75,
            "MattesMutualInformation": 0.25,
        }
    )
    fixed_image: FakeImage = field(
        default_factory=lambda: fake_image(
            data=np.pad(
                np.ones((4, 4, 4), dtype=np.float32),
                ((1, 3), (2, 2), (1, 3)),
            )
        )
    )
    moving_image: FakeImage = field(
        default_factory=lambda: fake_image(
            data=np.pad(
                np.full((4, 4, 4), 2.0, dtype=np.float32),
                ((2, 2), (1, 3), (2, 2)),
            )
        )
    )
    linear_calls: int = 0
    nonlinear_calls: int = 0
    read_affine_calls: int = 0
    read_warp_calls: int = 0
    read_warp_paths: list[Path] = field(default_factory=list)
    threshold_mask_calls: int = 0
    to_array_calls: int = 0
    events: list[str] = field(default_factory=list)
    load_paths: list[Path] = field(default_factory=list)
    linear_kwargs: list[dict[str, object]] = field(default_factory=list)
    nonlinear_kwargs: list[dict[str, object]] = field(default_factory=list)
    affine_interpolations: list[str] = field(default_factory=list)
    warp_applications: list[tuple[Path, str]] = field(default_factory=list)
    histogram_match_calls: list[tuple[object, object]] = field(default_factory=list)
    metric_calls: list[tuple[object, object, object | None, object | None]] = field(
        default_factory=list
    )
    tiff_writes: list[tuple[Path, float, float]] = field(default_factory=list)
    tiff_images: list[object] = field(default_factory=list)
    crop_bounds: list[tuple[int, int, int, int, int, int]] = field(default_factory=list)
    temporary_warps: list[Path] = field(default_factory=list)
    nonlinear_candidates: list[FakeImage] = field(default_factory=list)
    _test_directory: Path | None = None

    def load_image(self, path: Path, pixel_size_xyz: PixelSize) -> object:
        self.events.append(f"load:{path.name}")
        self.load_paths.append(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        if payload.startswith(b"corrupt"):
            raise ValueError(f"unreadable image: {path}")
        if payload.startswith(b"tiff-fake:"):
            decoded = json.loads(payload.removeprefix(b"tiff-fake:").decode())
            geometry = ImageGeometry.from_dict(decoded["geometry"])
            return fake_image(
                data=np.zeros(geometry.shape, dtype=np.float32),
                origin=geometry.origin,
                spacing=geometry.spacing,
                direction=geometry.direction,
            )
        if path.name == "fixed.tif":
            template = self.fixed_image
            self._test_directory = path.parent
        elif path.name == "moving.tif":
            template = self.moving_image
            self._test_directory = path.parent
        elif "fixed_reference" in path.name:
            template = self.fixed_image
        else:
            template = self.moving_image
        loaded = template.clone()
        loaded.spacing = tuple(float(value) for value in pixel_size_xyz)
        return loaded

    def zero_mask(self, image: object) -> object:
        source = image
        if not isinstance(source, FakeImage):
            raise TypeError("image must be a FakeImage")
        self.events.append("zero-mask")
        return fake_image(
            data=(source.data > 0).astype(np.float32),
            origin=source.origin,
            spacing=source.spacing,
            direction=source.direction,
        )

    def threshold_mask(self, image: object) -> object:
        source = image
        if not isinstance(source, FakeImage):
            raise TypeError("image must be a FakeImage")
        self.events.append("threshold-mask")
        self.threshold_mask_calls += 1
        if self.fail_threshold_mask:
            raise RuntimeError("injected mask threshold failure")
        return fake_image(
            data=((source.data >= 0.5) & (source.data <= 1.0)).astype(np.float32),
            origin=source.origin,
            spacing=source.spacing,
            direction=source.direction,
        )

    def geometry(self, image: object) -> ImageGeometry:
        source = image
        if not isinstance(source, FakeImage):
            raise TypeError("image must be a FakeImage")
        return ImageGeometry(
            shape=source.shape,
            origin=source.origin,
            spacing=source.spacing,
            direction=source.direction,
        )

    def to_array(self, image: object) -> object:
        source = image
        if not isinstance(source, FakeImage):
            raise TypeError("image must be a FakeImage")
        self.to_array_calls += 1
        if self.fail_to_array_call == self.to_array_calls:
            raise RuntimeError("injected image range failure")
        return source.numpy()

    def from_array(
        self,
        array: object,
        *,
        reference: object,
        has_components: bool = False,
    ) -> object:
        source = reference
        if not isinstance(source, FakeImage):
            raise TypeError("reference must be a FakeImage")
        if self.fail_scalar_from_array and not has_components:
            raise RuntimeError("injected scalar array conversion failure")
        return fake_image(
            data=np.asarray(array),
            origin=source.origin,
            spacing=source.spacing,
            direction=source.direction,
            has_components=has_components,
        )

    def crop_image(
        self,
        image: object,
        bounds_zyx: tuple[int, int, int, int, int, int],
    ) -> object:
        source = image
        if not isinstance(source, FakeImage):
            raise TypeError("image must be a FakeImage")
        self.crop_bounds.append(bounds_zyx)
        if self.fail_crop_image:
            raise RuntimeError("injected crop application failure")
        z0, z1, y0, y1, x0, x1 = bounds_zyx
        origin = source.index_to_physical_point((x0, y0, z0))
        return fake_image(
            data=source.data[z0:z1, y0:y1, x0:x1],
            origin=origin,
            spacing=source.spacing,
            direction=source.direction,
        )

    def linear(self, **kwargs: object) -> tuple[object, object | None, object]:
        self.events.append("linear")
        self.linear_calls += 1
        self.linear_kwargs.append(dict(kwargs))
        if self.fail_linear:
            raise RuntimeError("injected linear failure")
        moving = kwargs["moving_image"]
        moving_mask = kwargs.get("moving_mask")
        if not isinstance(moving, FakeImage):
            raise TypeError("moving_image must be a FakeImage")
        transformed_mask = (
            None if moving_mask is None else moving_mask.clone()  # type: ignore[union-attr]
        )
        return moving.clone(), transformed_mask, self.affine_bytes

    def nonlinear(self, **kwargs: object) -> tuple[object, Path]:
        self.events.append("nonlinear")
        self.nonlinear_calls += 1
        self.nonlinear_kwargs.append(dict(kwargs))
        if self.fail_nonlinear:
            raise RuntimeError("injected nonlinear failure")
        moving = kwargs["moving_image"]
        if not isinstance(moving, FakeImage):
            raise TypeError("moving_image must be a FakeImage")
        if self._test_directory is None:
            raise RuntimeError("no test directory was established")
        warp_path = (
            self._test_directory / f"temporary-warp-{self.nonlinear_calls}.nii.gz"
        )
        warp_path.write_bytes(self.warp_bytes)
        self.temporary_warps.append(warp_path)
        candidate = fake_image(
            data=moving.data + 100.0,
            origin=moving.origin,
            spacing=moving.spacing,
            direction=moving.direction,
        )
        self.nonlinear_candidates.append(candidate)
        return candidate, warp_path

    def apply_affine(
        self,
        *,
        fixed: object,
        moving: object,
        transform: object,
        interpolation: str,
    ) -> object:
        del fixed, transform
        if not isinstance(moving, FakeImage):
            raise TypeError("moving must be a FakeImage")
        self.events.append(f"apply-affine:{interpolation}")
        self.affine_interpolations.append(interpolation)
        if self.fail_affine_interpolation == interpolation:
            raise RuntimeError(f"injected {interpolation} affine application failure")
        return moving.clone()

    def apply_warp(
        self,
        *,
        fixed: object,
        moving: object,
        warp_path: Path,
        interpolation: str,
    ) -> object:
        del fixed
        if not isinstance(moving, FakeImage):
            raise TypeError("moving must be a FakeImage")
        if not warp_path.is_file():
            raise FileNotFoundError(warp_path)
        self.events.append(f"apply-warp:{interpolation}")
        self.warp_applications.append((warp_path, interpolation))
        if self.fail_warp_interpolation == interpolation:
            raise RuntimeError(f"injected {interpolation} warp application failure")
        offset = 10.0 if interpolation == "linear" else 0.0
        return fake_image(
            data=moving.data + offset,
            origin=moving.origin,
            spacing=moving.spacing,
            direction=moving.direction,
        )

    def histogram_match(self, *, source: object, reference: object) -> object:
        if not isinstance(source, FakeImage) or not isinstance(reference, FakeImage):
            raise TypeError("source and reference must be FakeImages")
        self.events.append("histogram-match")
        self.histogram_match_calls.append((source, reference))
        if self.fail_histogram_match:
            raise RuntimeError("injected histogram matching failure")
        return fake_image(
            data=source.data + 1_000.0,
            origin=source.origin,
            spacing=source.spacing,
            direction=source.direction,
        )

    def write_affine(self, transform: object, path: Path) -> None:
        self.events.append("write-affine")
        if self.fail_affine_write:
            path.write_bytes(b"partial-affine")
            raise RuntimeError("injected affine write failure")
        if not isinstance(transform, bytes):
            raise TypeError("transform must be bytes")
        path.write_bytes(transform)

    def read_affine(self, path: Path) -> object:
        self.events.append("read-affine")
        self.read_affine_calls += 1
        value = path.read_bytes()
        if not value.startswith(b"affine"):
            raise ValueError(f"unreadable affine: {path}")
        return value

    def write_warp(self, warp: object, path: Path) -> None:
        if isinstance(warp, bytes):
            path.write_bytes(warp)
        else:
            path.write_bytes(self.warp_bytes)

    def read_warp(self, path: Path) -> object:
        self.events.append("read-warp")
        self.read_warp_calls += 1
        self.read_warp_paths.append(path)
        value = path.read_bytes()
        if not value.startswith((b"warp", b"identity")):
            raise ValueError(f"unreadable warp: {path}")
        return fake_image(
            data=np.zeros((*self.fixed_image.shape, 3), dtype=np.float32),
            has_components=True,
        )

    def write_identity_warp(self, reference: object, path: Path) -> None:
        del reference
        path.write_bytes(b"identity-warp")

    def write_tiff(
        self,
        image: object,
        path: Path,
        *,
        min_value: float,
        max_value: float,
    ) -> None:
        if not isinstance(image, FakeImage):
            raise TypeError("image must be a FakeImage")
        self.events.append("write-tiff")
        self.tiff_writes.append((path, min_value, max_value))
        self.tiff_images.append(image)
        if self.fail_tiff_role is not None and self.fail_tiff_role in path.name:
            path.write_bytes(b"partial-tiff")
            raise RuntimeError("injected TIFF writer failure")
        payload = {"geometry": self.geometry(image).to_dict()}
        path.write_bytes(b"tiff-fake:" + json.dumps(payload, sort_keys=True).encode())

    def calculate_metrics(
        self,
        fixed: object,
        moving: object,
        fixed_mask: object | None = None,
        moving_mask: object | None = None,
        sampling: str = "regular",
        sampling_pct: float = 1.0,
    ) -> dict[str, float]:
        del sampling, sampling_pct
        self.events.append("metrics")
        self.metric_calls.append((fixed, moving, fixed_mask, moving_mask))
        if self.fail_metrics:
            raise RuntimeError("injected metrics failure")
        return dict(self.metric_values)


@dataclass
class FakeChunkBackend(FakeBackend):
    fail_chunk: int | None = None
    missing_warp_chunk: int | None = None
    invalid_warp_chunk: int | None = None
    invalid_scalar_chunk: int | None = None
    invalid_mask_chunk: int | None = None
    overflow_scalar_chunk: int | None = None
    overflow_warp_chunk: int | None = None
    backend_value_error_chunk: int | None = None
    full_warp_float64: bool = False
    warp_origin_delta: float = 0.0
    chunk_depth: int = 8
    fixed_image: FakeImage = field(
        default_factory=lambda: fake_image(
            data=np.arange(16 * 8 * 8, dtype=np.float32).reshape(16, 8, 8)
        )
    )
    moving_image: FakeImage = field(
        default_factory=lambda: fake_image(
            data=(np.arange(16 * 8 * 8, dtype=np.float32).reshape(16, 8, 8) + 100.0)
        )
    )
    registered_chunk_ids: list[int] = field(default_factory=list)
    identity_chunk_ids: list[int] = field(default_factory=list)
    written_warp_paths: list[Path] = field(default_factory=list)

    def _chunk_id(self, image: FakeImage) -> int:
        spacing_z = image.spacing[2]
        if spacing_z <= 0.0:
            raise ValueError("spacing must be positive")
        physical_offset = np.asarray(image.origin) - np.asarray(self.fixed_image.origin)
        scaled_index = np.linalg.solve(np.asarray(image.direction), physical_offset)
        z_start = round(float(scaled_index[2]) / spacing_z)
        return int(z_start // self.chunk_depth)

    @staticmethod
    def _warp_payload(
        image: FakeImage,
        *,
        chunk_id: int | None,
        components: int = 3,
    ) -> bytes:
        payload = {
            "chunk_id": chunk_id,
            "components": components,
            "direction": image.direction,
            "origin": image.origin,
            "shape": image.shape,
            "spacing": image.spacing,
        }
        return ("warp-json:" + json.dumps(payload, sort_keys=True)).encode()

    @staticmethod
    def _decoded_warp(payload: bytes) -> dict[str, object] | None:
        if not payload.startswith(b"warp-json:"):
            return None
        decoded = json.loads(payload.removeprefix(b"warp-json:").decode())
        if not isinstance(decoded, dict):
            raise TypeError("warp payload must be an object")
        return decoded

    def nonlinear(self, **kwargs: object) -> tuple[object, Path]:
        self.events.append("nonlinear")
        self.nonlinear_calls += 1
        self.nonlinear_kwargs.append(dict(kwargs))
        fixed = kwargs["fixed_image"]
        moving = kwargs["moving_image"]
        if not isinstance(fixed, FakeImage) or not isinstance(moving, FakeImage):
            raise TypeError("chunk images must be FakeImages")
        chunk_id = self._chunk_id(fixed)
        self.registered_chunk_ids.append(chunk_id)
        if self.fail_chunk == chunk_id:
            raise RuntimeError(f"injected chunk {chunk_id} failure")
        if self._test_directory is None:
            raise RuntimeError("no test directory was established")
        warp_path = self._test_directory / (
            f"temporary-chunk-{chunk_id}-{self.nonlinear_calls}.nii.gz"
        )
        components = 2 if self.invalid_warp_chunk == chunk_id else 3
        if self.missing_warp_chunk == chunk_id:
            warp_path.unlink(missing_ok=True)
        else:
            warp_path.write_bytes(
                self._warp_payload(
                    fixed,
                    chunk_id=chunk_id,
                    components=components,
                )
            )
        self.temporary_warps.append(warp_path)
        candidate = fake_image(
            data=moving.data + 100.0,
            origin=moving.origin,
            spacing=moving.spacing,
            direction=moving.direction,
        )
        self.nonlinear_candidates.append(candidate)
        return candidate, warp_path

    def write_identity_warp(self, reference: object, path: Path) -> None:
        if not isinstance(reference, FakeImage):
            raise TypeError("reference must be a FakeImage")
        chunk_id = self._chunk_id(reference)
        self.identity_chunk_ids.append(chunk_id)
        path.write_bytes(self._warp_payload(reference, chunk_id=chunk_id))

    def write_warp(self, warp: object, path: Path) -> None:
        if not isinstance(warp, FakeImage):
            raise TypeError("warp must be a FakeImage")
        self.written_warp_paths.append(path)
        path.write_bytes(self._warp_payload(warp, chunk_id=None))

    def read_warp(self, path: Path) -> object:
        self.events.append("read-warp")
        self.read_warp_calls += 1
        self.read_warp_paths.append(path)
        decoded = self._decoded_warp(path.read_bytes())
        if decoded is None:
            return super().read_warp(path)
        shape = tuple(int(value) for value in decoded["shape"])  # type: ignore[union-attr]
        origin = tuple(float(value) for value in decoded["origin"])  # type: ignore[union-attr]
        origin = (origin[0] + self.warp_origin_delta, origin[1], origin[2])
        spacing = tuple(float(value) for value in decoded["spacing"])  # type: ignore[union-attr]
        direction = tuple(
            tuple(float(value) for value in row)
            for row in decoded["direction"]  # type: ignore[union-attr]
        )
        components = int(decoded["components"])  # type: ignore[arg-type]
        chunk_id = decoded.get("chunk_id")
        dtype = (
            np.float64 if chunk_id is None and self.full_warp_float64 else np.float32
        )
        value = 0.0
        if (
            self.overflow_warp_chunk is not None
            and chunk_id == self.overflow_warp_chunk
        ):
            dtype = np.float64
            value = np.finfo(np.float64).max
        return fake_image(
            data=np.full((*shape, components), value, dtype=dtype),
            origin=origin,  # type: ignore[arg-type]
            spacing=spacing,  # type: ignore[arg-type]
            direction=direction,  # type: ignore[arg-type]
            has_components=True,
        )

    def apply_warp(
        self,
        *,
        fixed: object,
        moving: object,
        warp_path: Path,
        interpolation: str,
    ) -> object:
        transformed = super().apply_warp(
            fixed=fixed,
            moving=moving,
            warp_path=warp_path,
            interpolation=interpolation,
        )
        if not isinstance(transformed, FakeImage):
            raise TypeError("transformed image must be a FakeImage")
        decoded = self._decoded_warp(warp_path.read_bytes())
        chunk_id = None if decoded is None else decoded.get("chunk_id")
        if (
            interpolation == "linear"
            and self.backend_value_error_chunk is not None
            and chunk_id == self.backend_value_error_chunk
        ):
            raise ValueError("injected backend chunk value error")
        if (
            interpolation == "linear"
            and self.overflow_scalar_chunk is not None
            and chunk_id == self.overflow_scalar_chunk
        ):
            transformed.data = np.full(
                transformed.shape,
                np.finfo(np.float64).max,
                dtype=np.float64,
            )
        if (
            interpolation == "linear"
            and self.invalid_scalar_chunk is not None
            and chunk_id == self.invalid_scalar_chunk
        ):
            transformed.data = transformed.data[:-1]
        if (
            interpolation == "nearestNeighbor"
            and self.invalid_mask_chunk is not None
            and chunk_id == self.invalid_mask_chunk
        ):
            transformed.data = transformed.data[:-1]
        return transformed


def install_fake_ants(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    tmp_path: Path | None = None,
) -> FakeAntsCalls:
    module = importlib.import_module(module_name)
    warp_path = (
        tmp_path / "warp.nii.gz" if tmp_path is not None else Path("fake-warp.nii.gz")
    )
    if tmp_path is not None:
        warp_path.write_bytes(b"warp")
    calls = FakeAntsCalls(warp_path=warp_path)

    def registration(**kwargs: object) -> dict[str, object]:
        unexpected = set(kwargs) - _ANTSPYX_REGISTRATION_PARAMETERS
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise TypeError(f"Unexpected ANTSpyX registration keyword(s): {names}")
        calls.registration_call_count += 1
        calls.registration = dict(kwargs)
        return {
            "fwdtransforms": [
                str(path)
                for path in (
                    [calls.warp_path]
                    if calls.forward_transforms is None
                    else calls.forward_transforms
                )
            ],
            "invtransforms": [str(path) for path in calls.inverse_transforms],
            "warpedmovout": calls.registered_warped,
        }

    def read_transform(filename: str) -> FakeTransform:
        calls.read_transform_path = filename
        return calls.transform

    def threshold_image(
        image: object,
        *,
        low_thresh: float,
        high_thresh: float,
        inval: int,
        outval: int,
    ) -> FakeImage:
        calls.threshold = {
            "image": image,
            "low_thresh": low_thresh,
            "high_thresh": high_thresh,
            "inval": inval,
            "outval": outval,
        }
        return calls.thresholded

    def resample_image_to_target(
        *, image: object, target: object, interp_type: str
    ) -> FakeImage:
        calls.resample = {
            "image": image,
            "target": target,
            "interp_type": interp_type,
        }
        return calls.resampled

    def histogram_match_image(
        source_image: object, reference_image: object
    ) -> FakeImage:
        calls.histogram_match = {
            "source_image": source_image,
            "reference_image": reference_image,
        }
        return calls.histogram_matched

    def apply_transforms(
        *,
        fixed: object,
        moving: object,
        transformlist: list[str],
        interpolator: str,
        whichtoinvert: list[bool] | None = None,
        compose: str | None = None,
    ) -> object:
        calls.transformlist = list(transformlist)
        calls.whichtoinvert = None if whichtoinvert is None else list(whichtoinvert)
        calls.interpolator = interpolator
        calls.compose = compose
        calls.apply_transforms = {
            "fixed": fixed,
            "moving": moving,
            "transformlist": list(transformlist),
            "whichtoinvert": calls.whichtoinvert,
            "interpolator": interpolator,
            "compose": compose,
        }
        if compose is not None:
            calls.composed_path = Path(
                compose if compose.endswith(".h5") else f"{compose}comptx.nii.gz"
            )
            calls.composed_path.write_bytes(b"composite-warp")
            return str(calls.composed_path)
        return calls.applied

    def image_read(filename: str) -> FakeImage:
        calls.image_read_path = filename
        return calls.read_warp

    def image_write(image: object, filename: str) -> None:
        calls.image_write_image = image
        calls.image_write_path = filename
        Path(filename).write_bytes(b"image")

    def write_transform(transform: object, filename: str) -> None:
        calls.write_transform_value = transform
        calls.write_transform_path = filename
        Path(filename).write_bytes(b"transform")

    def from_numpy(
        data: object,
        *,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        direction: object | None = None,
        has_components: bool = False,
    ) -> FakeImage:
        normalized_direction = (
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            if direction is None
            else tuple(tuple(float(value) for value in row) for row in direction)
        )
        call = {
            "data": np.asarray(data),
            "origin": origin,
            "spacing": spacing,
            "direction": normalized_direction,
            "has_components": has_components,
        }
        calls.from_numpy_calls.append(call)
        return fake_image(
            data=np.asarray(data),
            origin=origin,
            spacing=spacing,
            direction=normalized_direction,
            has_components=has_components,
        )

    def get_mask(image: object) -> FakeImage:
        source = image.numpy()  # type: ignore[attr-defined]
        return fake_image(data=np.ones(source.shape[:3], dtype=np.float32))

    def transform_index_to_physical_point(
        image: object, index: tuple[int, int, int]
    ) -> tuple[float, float, float]:
        if not isinstance(image, FakeImage):
            raise TypeError("image must be a FakeImage")
        calls.index_to_physical_calls.append((image, index))
        return image.index_to_physical_point(index)

    fake_ants = SimpleNamespace(
        registration=registration,
        read_transform=read_transform,
        threshold_image=threshold_image,
        resample_image_to_target=resample_image_to_target,
        histogram_match_image=histogram_match_image,
        apply_transforms=apply_transforms,
        image_read=image_read,
        image_write=image_write,
        write_transform=write_transform,
        from_numpy=from_numpy,
        get_mask=get_mask,
        transform_index_to_physical_point=transform_index_to_physical_point,
    )

    def scientific_runtime() -> tuple[object, object]:
        calls.events.append("runtime")
        return fake_ants, np

    monkeypatch.setattr(module, "_scientific_runtime", scientific_runtime)
    return calls


__all__ = [
    "FakeAntsCalls",
    "FakeBackend",
    "FakeChunkBackend",
    "FakeImage",
    "FakeTransform",
    "fake_image",
    "install_fake_ants",
]
