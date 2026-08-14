# Architecture and project history

## Current capabilities

Cy-ExM registers volumetric imaging rounds with whole-image and serial-chunked
scientific registration strategies. A workflow validates a scientific dataset
manifest separately from an execution profile, then provides deterministic planning,
status inspection, and restart-safe transformation of declared dependent channels.
The fixed reference and moving roles remain distinct, public spacing is in XYZ order,
and NumPy volumes are in ZYX order.

## Registration library

The registration library owns policy, image I/O boundaries, and artifact production.
It uses whole-image affine registration before nonlinear registration. Affine
application is the first fixed-grid boundary: scalar and mask data entering nonlinear
work, the full warp, and transformed dependent channels must match the fixed-reference
geometry. Scalar interpolation is linear and mask interpolation is nearest-neighbor.

The serial-chunked workflow is the memory-conscious numerical and restart reference.
It preserves raw diagnostic masks based on values greater than zero; optional
feature-derived masks inform registration metrics but do not replace raw outputs or
diagnostic masks.

## Workflow configuration and planning

The scientific dataset manifest holds rounds, channels, registration choices, and
physical spacing. The execution profile holds machine-specific paths, native threads,
executor selection, and resources. Planning is mutation-free: it validates and
normalizes these inputs before execution and derives the immutable scientific work
without writing artifacts or constructing an executor.

The workflow creates one immutable `ReferenceTask` for preparation of the reference
round, followed by one immutable `RoundTask` per moving round in coarse execution.
Reference failure stops moving-round submission through a hard executor barrier.
Forcing the reference propagates to every descendant, while a failure in one moving
round does not block independent moving rounds.

## Execution boundaries

The sequential, local process-based, and SLURM executors run the same scheduler-neutral
plan. `parallelism.mode` is operational rather than scientific: `coarse` is the
production default and schedules one task per moving round, while `fine_grained` is
an opt-in coordinator-owned graph for preparation, nonlinear chunks, assembly, and
declared channels. The scientific fingerprint is unchanged when the execution mode
changes.

Each worker-backed process exposes one `round_slot`; each coarse round or fine
scientific task consumes one slot. Registration remains serial inside each
moving-round task in coarse mode. Workers do not create schedulers, submit
descendants, or start nested schedulers. The coordinator owns reference admission,
task admission, cancellation, and cleanup. Fine-grained execution does not change the
scientific registration strategy or fixed-grid requirements.

## Persistent scientific state

The artifact store owns restart state. Artifacts are written atomically and reused
only with matching completion records; completion records are the only restart
authority. A data file or operational executor summary alone is not a valid cache
entry. Missing or stale completion evidence causes recomputation.

The common serial chunk stage produces reusable scalar cores in both execution modes;
prepared fixed-grid arrays are an internal fine-grained optimization. Public outputs,
scientific fingerprints, and completion requirements remain compatible across the
modes where their scientific requests match.

## Supported boundaries and non-goals

Cy-ExM supports the registration strategies, executor environments, and task
granularities described above. It does not provide `ParallelChunkedImageRegistration`,
job arrays, cloud or Kubernetes executors, or automatic resource optimization.
Chunked or parallel affine estimation is also out of scope.

There is no nested chunk multiprocessing or worker-owned scheduling graph. A selected
executor owns one lifecycle and reports its result; there is no silent fallback to a
different executor. Real-HPC evidence is an explicitly selected, operator-approved
gate; portable tests do not claim scheduler placement or production acceptance.

## Historical phase labels

Historical Phase 1 identifies the registration-library baseline: whole-image and
serial-chunked registration, strict geometry and TIFF boundaries, restart-safe
artifacts, and native-thread configuration. Historical Phase 2 identifies the
workflow-runtime work: strict manifests and profiles, scheduler-neutral planning,
executors, and round-level artifact orchestration. These labels remain only so older
plans and acceptance records can be interpreted; current guidance is organized by
capability.
