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
- Resolve nonlinear metric policy before loading scientific runtimes. Preserve
  standalone Mattes/32 defaults; forward exact `CC` plus its positive scalar radius
  through whole-image and every serial-chunk call and authoritative request record.
- `preprocessing.py` owns the only supported registration feature: reflective
  Gaussian background subtraction, a deterministic 1.4826-MAD scale calculated
  from the signed residual before clipping, negative clipping at zero, and asinh.
  Keep its NumPy and SciPy imports deferred and change its versioned policy identity
  whenever numerical behavior changes.
- Registration features are transform-estimation inputs only. Crop raw images before
  deriving features, derive masks from cropped raw values greater than zero, and use
  raw images for all existing metrics, registered scalar reconstruction, and channel
  transformation. Never preprocess dependent channels or write feature TIFFs as
  public outputs.
- Apply forward transforms in warp-then-affine list order with both inversion
  flags false. Scalars use linear interpolation; labels and masks use
  `nearestNeighbor`, with transformed masks cast to float and thresholded at 0.5 so
  ANTSpyX metric masks retain the required floating pixel type.
- Keep affine decomposition helpers callable without runtime configuration;
  they are characterization utilities over arrays already supplied by callers.
- Preserve image origin, spacing, direction, vector components, and the public
  ZYX crop-bound convention when adapting ANTs images.
- Backend artifact writer methods must write exactly to the supplied path so callers
  can provide atomic temporary destinations. `backend.py` must not own workflow
  scheduling or artifact lifecycle. Completion-record primitives remain in
  `artifacts.py`; `whole_image.py`, `chunked.py`, and `channels.py` each own
  orchestration of their scoped artifacts.
- Artifact progress observers may report a commit only after both atomic artifact and
  canonical completion-record promotion, and may report reuse only after exact
  `StageRequest`, identity, and reader validation. Keep observation context-local and
  do not turn it into an alternate cache-validation path.
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
- Keep `_prepare_linear_state` as the shared private boundary for source snapshots,
  image loading, crop, masks, fixed-reference persistence, pre-registration metrics,
  affine preparation, and linear metrics. Consumers must preserve that stage order
  and use its accepted fixed-grid scalar, mask, affine, paths, backend, and resolved
  native-thread state without repeating preparation.
- `prepared.py` owns the internal fixed-grid float32 NPY bundle. Commit each array
  through the ordinary artifact store, promote the strict manifest last, and let
  workers copy only one extended region from validated read-only memory maps. The
  manifest must record and bind the exact supported ANTs pixel type for all four
  prepared roles; reconstruct each regional image with that type so an integer
  legacy fixed grid retains its scalar-resampling behavior while floating inputs
  remain unquantized.
- `chunk_stages.py` owns one prepared-region nonlinear task, float32 scalar-core
  persistence, and full-grid assembly. Preserve the legacy ascending-ID overwrite
  rule for expected overlapping output slices; task completion order must not change
  scalar or displacement assembly.
- Pre-registration metrics resample unequal moving scalar and mask grids to the fixed
  grid for diagnosis only; this must not replace or pre-resample the native moving
  inputs supplied to affine estimation.
- Standalone Phase 1 registration retains `prepared_fixed=False`, including its
  fixed crop and persisted `fixed_reference.tif`. Workflow rounds explicitly use
  `prepared_fixed=True`: never recrop or duplicate the prepared fixed TIFF, but keep
  the moving crop and every downstream fixed-grid assertion.
- `RegistrationBackend.nonlinear` and every `_estimate_nonlinear` override transfer
  ownership of one unique disposable warp candidate to their immediate caller. The
  caller must persist or consume it and then use the shared best-effort cleanup helper;
  a backend must never return a canonical cache path or shared native source.
- Preprocessing-disabled prepared manifests remain exact format 2; enabled feature
  manifests are strict format 3 and bind the algorithm, ZYX voxel sigma, and actual
  fixed/moving robust scales.
- Crop raw images before feature construction, derive masks from raw values greater
  than zero, estimate affine/nonlinear transforms from features only, and reconstruct
  metrics and public scalar output from raw fixed-grid data. Any algorithm change
  requires a new policy identifier and a bounded scientific regression.
