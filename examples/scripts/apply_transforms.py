"""Apply warp then affine, retaining the source intensity or current label range."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cyexm.io.images import load_ants_image
from cyexm.io.tiff import write_uint16_tiff
from cyexm.registration.transforms import transform_image
from cyexm.runtime import NativeThreadConfig, configure_native_threads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixed_image", type=Path, help="fixed reference image")
    parser.add_argument("moving_image", type=Path, help="moving intensity or label")
    parser.add_argument("affine_transform", type=Path, help="persisted .mat file")
    parser.add_argument("nonlinear_warp", type=Path, help="persisted .nii.gz file")
    parser.add_argument("output_image", type=Path, help="output .tif image")
    parser.add_argument("--pixel-size-x", type=float, default=0.15)
    parser.add_argument("--pixel-size-y", type=float, default=0.15)
    parser.add_argument("--pixel-size-z", type=float, default=0.15)
    parser.add_argument("--itk-threads", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument(
        "--interpolation",
        choices=("linear", "nearestNeighbor"),
        default="linear",
    )
    return parser


def _run(args: argparse.Namespace) -> None:
    native_threads = NativeThreadConfig(args.itk_threads, args.blas_threads)
    configure_native_threads(native_threads)
    pixel_size = (args.pixel_size_x, args.pixel_size_y, args.pixel_size_z)
    fixed = load_ants_image(args.fixed_image, pixel_size)
    moving = load_ants_image(args.moving_image, pixel_size)
    transformed = transform_image(
        fixed=fixed,
        moving=moving,
        affine_path=args.affine_transform,
        warp_path=args.nonlinear_warp,
        interpolation=args.interpolation,
        native_threads=native_threads,
    )
    runtime_transformed: Any = transformed
    runtime_moving: Any = moving
    values = (
        runtime_transformed.numpy()
        if args.interpolation == "nearestNeighbor"
        else runtime_moving.numpy()
    )
    write_uint16_tiff(
        transformed,
        args.output_image,
        float(values.min()),
        float(values.max()),
    )


def main(argv: Sequence[str] | None = None) -> int:
    _run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
