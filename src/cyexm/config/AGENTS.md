# Configuration package guidance

- Phase 2A configuration owns strict safe YAML decoding plus frozen
  scientific-manifest and execution-profile models. Keep the two documents separate.
- Reject unknown and duplicate keys, non-scalar YAML keys, aliases, custom tags,
  additional documents, and unsafe paths.
- Scientific and execution roots are disjoint, neither containing the other.
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
