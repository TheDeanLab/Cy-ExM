# Script example guidance

- Every script uses `argparse`, accepts all input and output paths from the caller,
  and defines `main(argv: Sequence[str] | None = None) -> int` for importable testing.
- End executable scripts with `raise SystemExit(main())`; never start work during
  import.
- Accept pixel size as separate XYZ flags and pass an explicit `NativeThreadConfig`
  built from the ITK and BLAS flags.
- Whole-image and chunked workflows expose round number, both accuracy presets, and
  force-recompute behavior. The chunked script remains serial.
- Low-level nonlinear examples own the unique disposable warp returned by the kernel:
  promote it to the caller's destination and remove the candidate in `finally`.
- Apply a complete persisted transform chain in `[warp, affine]` order with neither
  transform inverted. Use `linear` for intensity images and `nearestNeighbor` for
  discrete labels.
- Keep module-scope imports lightweight so `--help` never initializes ANTs, NumPy,
  SciPy, or tifffile.
