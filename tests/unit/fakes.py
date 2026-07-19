from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


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
        calls.registration_call_count += 1
        calls.registration = dict(kwargs)
        return {
            "fwdtransforms": [str(calls.warp_path)],
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
    ) -> FakeImage:
        calls.transformlist = list(transformlist)
        calls.whichtoinvert = None if whichtoinvert is None else list(whichtoinvert)
        calls.interpolator = interpolator
        calls.apply_transforms = {
            "fixed": fixed,
            "moving": moving,
            "transformlist": list(transformlist),
            "whichtoinvert": calls.whichtoinvert,
            "interpolator": interpolator,
        }
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
    )

    def scientific_runtime() -> tuple[object, object]:
        calls.events.append("runtime")
        return fake_ants, np

    monkeypatch.setattr(module, "_scientific_runtime", scientific_runtime)
    return calls


__all__ = [
    "FakeAntsCalls",
    "FakeImage",
    "FakeTransform",
    "fake_image",
    "install_fake_ants",
]
