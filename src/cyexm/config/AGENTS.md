# Configuration package guidance

- Phase 2A configuration owns strict safe YAML decoding plus frozen
  scientific-manifest and execution-profile models. Keep the two documents separate.
- Reject unknown and duplicate keys, non-scalar YAML keys, aliases, custom tags,
  additional documents, and unsafe paths.
- Schema version 1 requires `reference.registration_channel` plus a nonempty ordered
  `reference.channels` list. Require the registration channel exactly once, require
  unique source paths and output basenames, and do not maintain legacy
  `reference.channel` dual parsing.
- Schema version 1 also requires nonlinear `metric` exactly `CC` or `mattes`. Normalize
  omitted CC radius to built-in integer 4, reject anisotropic or invalid radii, reject
  radius for Mattes, and keep the normalized nonlinear model free of runtime imports.
- Scientific and execution roots are disjoint, neither containing the other.
- Parallelism is operational execution policy. Omission normalizes to coarse, and no
  scientific manifest or fingerprint may include the parallelism mode.
- Parse `slurm.memory` centrally as positive whole decimal memory with exact `KB`,
  `MB`, `GB`, or `TB` suffixes. Reject invalid, fractional, zero, unsupported, or
  signed-64-bit-overflowing semantic and ceiling-normalized quantities.
- Treat every long, short, attached, abbreviated, or equals-form node, task,
  resource, or worker option in `slurm.job_extra_directives` as managed and reject
  it. Preserve safe site policy such as account, QoS, constraint, mail,
  reservation, and exact `--exclusive`; resource-shaping SLURM aliases may not
  override executor ownership.
- This package must not import ANTs, NumPy, SciPy, Dask, distributed, or
  dask-jobqueue. It must not mutate the filesystem or environment or initialize a
  scheduler.
- Schema version 1 owns the optional `registration.preprocessing` block. Omission
  disables it; `{}` defaults to `(12.0, 12.0, 12.0)`; a positive scalar expands to
  three axes; and a list must be an exact positive finite NumPy ZYX triple.
