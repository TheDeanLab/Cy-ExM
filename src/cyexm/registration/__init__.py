"""Stable whole-image and serial registration entry points."""

from cyexm.registration.chunked import ChunkedImageRegistration
from cyexm.registration.whole_image import ImageRegistration, register_round

__all__ = [
    "ChunkedImageRegistration",
    "ImageRegistration",
    "register_round",
]
