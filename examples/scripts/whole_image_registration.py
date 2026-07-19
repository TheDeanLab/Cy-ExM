"""Run restartable whole-image affine and nonlinear registration."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from cyexm.registration import register_round
from cyexm.runtime import NativeThreadConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixed_image", type=Path, help="fixed reference TIFF or NPY")
    parser.add_argument("moving_image", type=Path, help="moving TIFF or NPY")
    parser.add_argument("output_directory", type=Path, help="artifact directory")
    parser.add_argument("--round", dest="round_number", type=int, default=0)
    parser.add_argument("--pixel-size-x", type=float, default=0.15)
    parser.add_argument("--pixel-size-y", type=float, default=0.15)
    parser.add_argument("--pixel-size-z", type=float, default=0.15)
    parser.add_argument("--itk-threads", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument(
        "--linear-accuracy",
        choices=("dry run", "low", "medium", "high"),
        default="low",
    )
    parser.add_argument(
        "--nonlinear-accuracy",
        choices=("dry run", "low", "medium", "high"),
        default="high",
    )
    parser.add_argument("--force", action="store_true", help="recompute cached stages")
    return parser


def _run(args: argparse.Namespace) -> None:
    native_threads = NativeThreadConfig(
        itk_threads=args.itk_threads,
        blas_threads=args.blas_threads,
    )
    register_round(
        fixed_image_path=args.fixed_image,
        moving_image_path=args.moving_image,
        save_directory=args.output_directory,
        imaging_round=args.round_number,
        crop=args.crop,
        pixel_size=(args.pixel_size_x, args.pixel_size_y, args.pixel_size_z),
        force_override=args.force,
        native_threads=native_threads,
        linear_accuracy=args.linear_accuracy,
        nonlinear_accuracy=args.nonlinear_accuracy,
    )


def main(argv: Sequence[str] | None = None) -> int:
    _run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
