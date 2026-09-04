# Choosing a registration strategy

## The shared registration pipeline

Cy-ExM keeps the fixed reference and moving-image roles distinct: the fixed image
defines the target geometry, and the moving image is transformed onto it. Both
strategies estimate a whole-image affine transform before nonlinear registration.
Applying that affine is the first fixed-grid boundary, so nonlinear inputs and
outputs use the fixed-reference geometry.

Public physical voxel spacing is expressed in micrometers in `(x, y, z)` order,
while NumPy volumes use `(z, y, x)` order. Scalar images use linear interpolation;
masks use nearest-neighbor interpolation. Accepted transforms are applied to the raw
registration image and each dependent channel rather than replacing those outputs
with registration features.

An optional per-moving-round [XY rotation initial alignment](initial-alignment.md)
can seed this same affine stage before TRSAA when acquisition introduces a large
in-plane offset. The complete persisted affine still owns the downstream contract.

## Whole-image registration

Whole-image registration estimates the nonlinear transform over the complete fixed
grid. It is a practical starting point when the volume and its working data fit the
available memory. Validated stage artifacts can be reused after interruption only
when their matching completion records remain valid.

## Serial-chunked registration

Serial-chunked registration uses the same whole-image affine stage, then processes
overlapping nonlinear regions serially. It is the memory-conscious option for larger
volumes. Each successful local nonlinear chunk has validated artifacts, so a rerun
can continue from valid work rather than repeat every chunk. Serial chunk execution
is the numerical and restart reference.

## Comparison

| Strategy | Nonlinear domain | Memory pressure | Restart unit | Chunk-size and overlap requirements | Appropriate starting use |
| --- | --- | --- | --- | --- | --- |
| Whole-image | Complete fixed grid | Higher for larger volumes | Validated whole-image stage | None | Volumes that fit available memory |
| Serial-chunked | Overlapping regions of the fixed grid | Bounded per nonlinear chunk | Validated local nonlinear chunk | Choose chunk size in NumPy `(z, y, x)` order and an overlap fraction | Larger volumes or memory-conscious registration |

## Registration strategy versus execution

Scientific registration strategy selects whole-image or serial-chunked registration
for one moving round. It is separate from the execution environment, which can be
sequential, local process-based, or SLURM. Task granularity is a second operational
axis: coarse or fine-grained controls how workflow work is admitted to an executor,
not the transform strategy. See the [workflow guide](workflow-guide.md) and
[fine-grained execution guide](fine-grained-execution.md) for those operational
choices.

## Scientific choices shared by both strategies

Do not change fixed and moving roles to accommodate an execution environment. In
both strategies, affine registration precedes nonlinear registration, affine
application establishes the fixed grid, scalar data use linear interpolation, and
mask data use nearest-neighbor interpolation. Restart safety depends on matching
artifact completion records, not on a filename or executor summary alone.

## Direct-Python scientific defaults

The standalone direct-Python API uses `TRSAA` with the `low` linear preset, followed
by `SyNOnly` with the `high` nonlinear preset. In standalone direct Python, an
omitted `nonlinear_metric` uses `mattes` with 32 bins. The Getting started example
selects `CC` explicitly. Its omitted `nonlinear_radius` uses the provisional scalar
voxel radius 4. That radius is one positive voxel count, not a per-axis physical
radius or a claim of dataset-specific optimality.

The YAML route is intentionally stricter: the scientific dataset manifest requires
an explicit nonlinear `metric` of `CC` or `mattes`. If that explicit choice is `CC`,
an omitted radius likewise normalizes to the provisional scalar voxel radius 4.

## Next steps

Start with [Getting started](getting-started.md) for the whole-image API example.
For detailed geometry, artifacts, and recovery behavior, see
[Registration artifacts and restart behavior](registration-artifacts.md). Use the
[workflow guide](workflow-guide.md) when you are ready to register multiple rounds.
