# Cy-ExM

Cy-ExM registers volumetric images acquired across cyclic expansion microscopy
rounds. It supports whole-image and memory-conscious serial-chunked registration,
applies accepted transforms to additional channels, and resumes validated
intermediate work after interruption.

Cy-ExM requires Python 3.12 and is distributed under academic-research-use terms.

## What do you want to do?

- **[Register two images](getting-started.md):** install from source and run the
  smallest supported registration.
- **[Choose a registration strategy](registration-foundations.md):** compare
  whole-image and serial-chunked registration.
- **[Run multiple imaging rounds](workflow-guide.md):** use the workflow tutorial.
- **[Configure advanced or HPC execution](fine-grained-execution.md):** understand
  the available execution choices.
- **[Contribute to Cy-ExM](agentic-api-documentation.md):** follow the current API
  and documentation conventions.

## Three separate decisions

| Decision | Choices | What it controls |
| --- | --- | --- |
| Scientific registration strategy | Whole-image or serial-chunked | How one moving round is registered |
| Execution environment | Sequential, local process-based, or SLURM | Where workflow tasks run |
| Task granularity | Coarse or fine-grained | How workflow work is admitted to the executor |

## Scientific evidence

- Registration behavior and parameter choices: [brightness](brightness-dependence-investigation.md),
  [asinh preprocessing](asinh-preprocessing-registration-investigation.md),
  [CC radius](cc-syn-radius-characterization.md), and
  [fixed-affine chunk size](fixed-affine-chunk-size-characterization.md)
  characterizations.
- Registration-library validation: [Registration-library acceptance](phase-1-acceptance.md).
- Workflow execution validation: [Workflow-runtime acceptance](phase-2-acceptance.md).
