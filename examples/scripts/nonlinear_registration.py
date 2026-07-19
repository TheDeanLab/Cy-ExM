"""Estimate a warp and encode the registered scalar over its resulting range."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from cyexm.io.atomic import atomic_write
from cyexm.io.images import load_ants_image, zero_mask
from cyexm.io.tiff import write_uint16_tiff
from cyexm.registration.nonlinear import register_image
from cyexm.runtime import NativeThreadConfig, configure_native_threads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixed_image", type=Path)
    parser.add_argument("moving_image", type=Path, help="affine-aligned moving image")
    parser.add_argument("warp_output", type=Path, help="output .nii.gz warp")
    parser.add_argument("registered_output", type=Path, help="output .tif image")
    parser.add_argument("--pixel-size-x", type=float, default=0.15)
    parser.add_argument("--pixel-size-y", type=float, default=0.15)
    parser.add_argument("--pixel-size-z", type=float, default=0.15)
    parser.add_argument("--itk-threads", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument(
        "--registration-type",
        choices=("Elastic", "SyNOnly", "SyN"),
        default="SyNOnly",
    )
    parser.add_argument(
        "--accuracy",
        choices=("dry run", "low", "medium", "high"),
        default="high",
    )
    return parser


def _warp_is_readable(path: Path) -> bool:
    try:
        import ants

        return bool(ants.image_read(str(path)).dimension == 3)
    except Exception:
        return False


def _run(args: argparse.Namespace) -> None:
    native_threads = NativeThreadConfig(args.itk_threads, args.blas_threads)
    configure_native_threads(native_threads)
    pixel_size = (args.pixel_size_x, args.pixel_size_y, args.pixel_size_z)
    fixed = load_ants_image(args.fixed_image, pixel_size)
    moving = load_ants_image(args.moving_image, pixel_size)
    transformed, candidate_warp = register_image(
        moving,
        fixed,
        moving_mask=zero_mask(moving),
        fixed_mask=zero_mask(fixed),
        registration_type=args.registration_type,
        accuracy=args.accuracy,
        verbose=True,
        native_threads=native_threads,
    )

    def copy_warp(path: Path) -> None:
        shutil.copyfile(candidate_warp, path)

    try:
        atomic_write(
            args.warp_output,
            copy_warp,
            _warp_is_readable,
        )
    finally:
        with suppress(OSError):
            candidate_warp.unlink(missing_ok=True)

    runtime_transformed: Any = transformed
    values = runtime_transformed.numpy()
    write_uint16_tiff(
        transformed,
        args.registered_output,
        float(values.min()),
        float(values.max()),
    )


def main(argv: Sequence[str] | None = None) -> int:
    _run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
