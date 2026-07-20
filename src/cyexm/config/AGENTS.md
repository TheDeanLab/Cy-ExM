# Configuration package guidance

- Configuration owns strict safe YAML decoding plus frozen scientific-manifest and
  execution-profile models. Keep the two documents separate.
- Reject unknown and duplicate keys, aliases, custom tags, additional documents, and
  unsafe paths.
- This package must not import ANTs, NumPy, SciPy, Dask, distributed, or
  dask-jobqueue. It must not mutate the filesystem or environment or initialize a
  scheduler.
