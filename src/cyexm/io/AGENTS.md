# I/O guidance

- Keep this package independent of `cyexm.registration`; registration may depend on
  I/O, but not the reverse.
- Preserve complete filename suffixes on same-directory temporary files so native
  image writers select the intended format.
- Validate temporary artifacts before promoting them with `os.replace`, and clean up
  every temporary path after success or failure.
- Preserve image geometry at every I/O boundary. Public pixel sizes use micrometers
  in `(x, y, z)` order while NumPy volumes use `(z, y, x)` order.
- Defer ANTs, NumPy, SciPy, and tifffile imports until native-thread settings have
  been resolved at the operation boundary.
- Reference-round preparation must use the existing atomic TIFF writer boundary:
  clip to uint16 without normalization, preserve the cropped or full source geometry,
  reopen for validation, and pair every output with completion metadata.
