# Getting Started Installation Guide Design

**Status:** Approved design; implementation completed in `e8694f4`
**Date:** 2026-07-26

## Context

The Getting Started page currently documents the three workflow-related optional
extras but omits installation choices described in the repository README: the
base Python API, notebook and example support, development/testing tools, and
the `--no-dev` option. Its HPC cache advice must be portable rather than
prescribing a filesystem layout specific to the Danuser laboratory.

## Goals

- Give an educated scientific audience a short, task-oriented installation path.
- Keep every supported installation command consistent with `README.md`.
- Explain what each optional extra enables in plain language.
- Explain that optional extras are cumulative.
- Give portable HPC guidance without assuming a particular cluster filesystem.

## Non-goals

- Explain registration strategies, executor modes, YAML schema, or SLURM
  resource settings.
- Add dependencies, change packaging, or alter any runtime behavior.
- Recommend a specific cache path, module system, scheduler, or administrator
  policy for third-party HPC systems.

## Alternatives considered

1. **One comprehensive install command.** Easy to copy, but imposes notebook,
   distributed, HPC, and development dependencies on users who do not need them.
2. **A dependency matrix.** Complete, but makes the first page feel like package
   administration rather than a scientific on-ramp.
3. **Selected: task-oriented commands.** Start from the smallest applicable
   installation and provide short commands for workflow, local parallel,
   SLURM/HPC, examples/notebooks, and development work. This retains precision
   while allowing a scientist to choose by purpose.

## Content design

The revised page will use the following order:

1. State the Python 3.12 requirement and introduce `uv sync --locked` as the
   reproducible base installation for the Python API.
2. State that extras are cumulative (`distributed` includes `workflow`; `hpc`
   includes `distributed`).
3. Present compact, commented commands for:
   - base API;
   - YAML and sequential workflows (`workflow`);
   - local Dask workflows (`distributed`);
   - SLURM workflows (`hpc`);
   - notebooks and parameterized examples (`examples`);
   - development/testing (`--group dev`, optionally with examples);
   - lean deployment (`--no-dev`).
4. Add a short, portable HPC note: use a cache and managed-Python location with
   adequate quota and performance; consult local cluster guidance for its
   preferred locations and any required modules.
5. Link the workflow guide for the next step, without explaining execution modes
   on this page.

## Data flow and error handling

This is documentation-only. Commands remain the authoritative commands already
published in the README; the page will not invent fallback installers or claim
that every HPC exposes the same paths or scheduler setup. Users whose cluster
does not provide `uv` are directed implicitly to their local support guidance,
rather than given an untested installation procedure.

## Testing and acceptance criteria

- The Getting Started commands and descriptions match `README.md` exactly.
- The page contains no Danuser-lab-specific `/project` or `/archive` advice.
- The page does not introduce workflow/execution-mode explanations.
- The documentation build succeeds.
- Existing documentation links resolve.
