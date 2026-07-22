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

## Scientific characterization

The [brightness-dependence investigation](brightness-dependence-investigation.md)
documents why dim regions can be harder to register and the evidence supporting
local cross-correlation for nonlinear SyN. The
[CC SyN radius characterization](cc-syn-radius-characterization.md) is explicitly
ongoing and is not a statement of an optimized default.
