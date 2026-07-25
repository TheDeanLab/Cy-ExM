# Cy-ExM

Cy-ExM provides reusable cyclic expansion microscopy registration workflows.
It is an independent implementation with restart-safe whole-image and serial
chunked registration, plus a deterministic Phase 2 runtime for planning and
executing round-level work.

The documentation focuses on using the software responsibly and understanding
the scientific choices that affect its output. It distinguishes accepted behavior
from active characterization work.

## Start here

- Follow [Getting started](getting-started.md) to install Cy-ExM and run the
  smallest supported workflow.
- Read [Registration foundations](registration-foundations.md) before changing
  transforms, metrics, masks, interpolation, or native-thread settings.
- Use [Artifact recovery and reproducibility](registration-artifacts.md) to
  understand restart-safe output reuse.
- Use [Fine-grained registration](fine-grained-execution.md) to configure the
  opt-in chunk/channel task graph and understand its current acceptance boundary.

## Scientific characterization

The [brightness-dependence investigation](brightness-dependence-investigation.md)
documents why dim regions can be harder to register and the evidence supporting
local cross-correlation for nonlinear SyN. The
[CC SyN radius characterization](cc-syn-radius-characterization.md) is explicitly
a dataset-specific completed sweep and is not a statement of a universally
optimized default. The
[fixed-affine nonlinear chunk-size characterization](fixed-affine-chunk-size-characterization.md)
then isolates nonlinear spatial support while holding the accepted affine state
constant.

## Acceptance evidence

The [Phase 1 acceptance audit](phase-1-acceptance.md) records the portable
registration-library baseline. The [Phase 2 acceptance audit](phase-2-acceptance.md)
records the real-cluster orchestration, scientific-equivalence, deterministic-parity,
restart, cancellation, cleanup, and output-ownership gates.
