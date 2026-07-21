# Registration package guidance

- Keep low-level registration policy in `linear.py`, `nonlinear.py`, and
  `transforms.py`; keep native-runtime and I/O adaptation behind `backend.py`.
- Do not import ANTs, NumPy, SciPy, or tifffile at module scope. Any public
  operation that needs a scientific runtime must first call
  `configure_native_threads`, log the resolved description with the module's
  named logger, and only then import or request that runtime.
- Validate registration method and accuracy before requesting ANTs. Preserve
  the characterized TRSAA and SyNOnly schedules, metrics, sampling settings,
  masks, interpolation modes, histogram matching, and single-precision policy.
- Apply forward transforms in warp-then-affine list order with both inversion
  flags false. Scalars use linear interpolation; labels and masks use
  `nearestNeighbor`, with transformed masks thresholded at 0.5.
- Keep affine decomposition helpers callable without runtime configuration;
  they are characterization utilities over arrays already supplied by callers.
- Preserve image origin, spacing, direction, vector components, and the public
  ZYX crop-bound convention when adapting ANTs images.
- Backend artifact writer methods must write exactly to the supplied path so callers
  can provide atomic temporary destinations. `backend.py` must not own workflow
  scheduling or artifact lifecycle. Completion-record primitives remain in
  `artifacts.py`; `whole_image.py`, `chunked.py`, and `channels.py` each own
  orchestration of their scoped artifacts.
- `channels.py` owns only one source channel's restart-safe transform artifact
  lifecycle. It configures native threads before requesting the lazy scientific
  backend, transforms through the accepted warp-plus-affine chain, and must not
  schedule rounds, create clients or child tasks, or import workflow/executor code.
- Native fixed, registration-channel, and dependent-channel shapes may differ along
  every spatial axis. Apply each dependent channel from its native grid with the
  fixed image as the transform reference, then require exact fixed-reference shape,
  origin, spacing, and direction before artifact promotion.
- Linear registration accepts native fixed and moving grids, but affine resampling
  must produce the exact fixed-reference scalar and mask geometry before whole-image
  or serial-chunked nonlinear work. Keep chunk geometry strict on that fixed grid.
- `RegistrationBackend.nonlinear` and every `_estimate_nonlinear` override transfer
  ownership of one unique disposable warp candidate to their immediate caller. The
  caller must persist or consume it and then use the shared best-effort cleanup helper;
  a backend must never return a canonical cache path or shared native source.
