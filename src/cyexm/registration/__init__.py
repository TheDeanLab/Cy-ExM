"""Stable whole-image and serial registration entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cyexm.registration.channels import transform_channel
    from cyexm.registration.chunked import ChunkedImageRegistration
    from cyexm.registration.whole_image import ImageRegistration, register_round


def __getattr__(name: str) -> object:
    if name == "transform_channel":
        from cyexm.registration.channels import transform_channel

        value = transform_channel
    elif name == "ChunkedImageRegistration":
        from cyexm.registration.chunked import ChunkedImageRegistration

        value = ChunkedImageRegistration
    elif name in {"ImageRegistration", "register_round"}:
        from cyexm.registration import whole_image

        value = getattr(whole_image, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = [
    "ChunkedImageRegistration",
    "ImageRegistration",
    "register_round",
    "transform_channel",
]
