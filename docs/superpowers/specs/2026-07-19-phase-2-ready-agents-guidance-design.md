# Phase 2-Ready AGENTS Guidance Refresh Design

**Date:** 2026-07-19

**Status:** Approved

## Context

Cy-ExM Phase 1 is merged on `main` and provides the stable whole-image and serial
chunked registration baseline. The next product-design cycle will cover reusable YAML
dataset manifests, local workflow planning, round-level Dask execution, and an HPC
execution profile based on `dask_jobqueue.SLURMCluster`. The repository already has
13 layered `AGENTS.md` files, but five maintained directory layers have no local
guidance, some root commands are less reproducible than CI, and a registration rule
is scoped more broadly than the code it governs.

The guidance must accurately describe the merged Phase 1 implementation while making
the transition into Phase 2 safe. It must not present unimplemented scheduler or
manifest features as current behavior or pre-approve dependencies and APIs that still
need a dedicated Phase 2 design.

## Goals

1. Give the repository root and every maintained directory layer an `AGENTS.md` file.
2. Make all inherited and local instructions accurate for the merged Phase 1 tree.
3. Record Phase 2 architectural guardrails without implementing Phase 2 or treating
   its expected components as released features.
4. Make validation commands reproduce the locked CI environment.
5. Add a policy test that detects missing instruction files, stale commands, and key
   contradictions.
6. Record the practical order of work after this guidance refresh.

## Non-goals

- Do not add YAML parsing, manifests, execution profiles, CLIs, Dask, distributed,
  dask-jobqueue, SLURM, or scheduler runtime code.
- Do not change production APIs, registration behavior, dependencies, the lockfile,
  examples, CI behavior, or scientific acceptance evidence.
- Do not rewrite completed Phase 1 specifications or plans as though they were
  authored after implementation.
- Do not design the complete Phase 2 schema or cluster interface in this refresh.

## Approaches considered

### Update only the existing 13 files

This is the smallest diff, but it leaves CI and design/plan directories governed only
by distant ancestor guidance. It also cannot enforce the original goal of detailed
instructions at each maintained directory layer.

### Complete layered coverage and refresh existing guidance

This is the selected approach. Five new local instruction files complete coverage,
the 13 existing files are checked and corrected, and an executable inventory test
prevents later drift. Phase 2 expectations are written as design guardrails rather
than promises of implemented behavior.

### Encode the complete Phase 2 architecture now

This would reduce later documentation work, but it would prematurely freeze the YAML
schema, public API, dependency groups, cluster resource syntax, and failure model
before discovery against real datasets and the current SLURM environment.

## Directory coverage

After the refresh, these 18 maintained layers each have local guidance:

1. `AGENTS.md`
2. `.github/AGENTS.md`
3. `.github/workflows/AGENTS.md`
4. `docs/AGENTS.md`
5. `docs/superpowers/AGENTS.md`
6. `docs/superpowers/plans/AGENTS.md`
7. `docs/superpowers/specs/AGENTS.md`
8. `examples/AGENTS.md`
9. `examples/notebooks/AGENTS.md`
10. `examples/scripts/AGENTS.md`
11. `src/AGENTS.md`
12. `src/cyexm/AGENTS.md`
13. `src/cyexm/io/AGENTS.md`
14. `src/cyexm/registration/AGENTS.md`
15. `src/cyexm/runtime/AGENTS.md`
16. `tests/AGENTS.md`
17. `tests/integration/AGENTS.md`
18. `tests/unit/AGENTS.md`

Generated directories, `.git`, `.worktrees`, virtual environments, caches, and Python
package metadata are not maintained instruction layers.

## Root guidance

The root file remains the authoritative project contract. It will:

- describe Phase 1 as the current stable runtime rather than the repository's
  permanent maximum scope;
- require an approved Phase 2 specification and implementation plan before adding
  manifest, scheduler, cluster, dependency, or public-API changes;
- retain the ClearEx source pin, license, scientific invariants, storage policy,
  source-safety rules, and protection for the separate spheroid registration run;
- identify the Phase 1 artifact store and serial chunk workflow as the restart and
  numerical reference for later execution layers;
- require scientific manifests to remain separate from local or HPC execution
  profiles;
- prohibit nested chunk multiprocessing inside a round-level Dask task unless a later
  design and independent numerical acceptance gate explicitly replace this rule; and
- use `uv lock --check` and
  `uv sync --locked --group dev --extra examples` for ordinary validation. Bare
  `uv lock` is reserved for an intentional dependency update.

## New CI guidance

`.github/AGENTS.md` will govern repository automation generally. It will require
least-privilege permissions, immutable or version-pinned third-party actions, no
credentials or lab paths, and portable checks on public GitHub-hosted runners.

`.github/workflows/AGENTS.md` will govern workflow YAML specifically. It will preserve
locked uv synchronization, Python 3.12, all six one-thread native environment values,
explicit unit and bounded-integration selectors, timeouts, and the prohibition on lab
storage or SLURM access from portable CI. Future HPC validation must use an explicitly
approved lab runner or manual cluster command rather than silently widening the
portable workflow.

## New specification and plan guidance

`docs/superpowers/AGENTS.md` will define dated, reviewable design and plan artifacts
and keep generated review packages out of version control.

`docs/superpowers/specs/AGENTS.md` will require specifications to separate current
behavior, decisions, alternatives, non-goals, failure handling, testing, and
acceptance criteria. Completed historical specifications remain historical records;
later corrections belong in a new dated document or an explicitly labeled erratum.

`docs/superpowers/plans/AGENTS.md` will require implementation plans to name exact
files, tests, commands, dependency order, TDD checkpoints, review gates, and commit
boundaries. A plan must not mark work complete before evidence exists, and completed
historical plans are not rewritten to hide implementation changes.

## Existing guidance corrections

Every existing file will be checked against its directory and ancestors. Expected
material corrections are:

- `docs/AGENTS.md` and example guidance will distinguish the stable Phase 1 surface
  from approved future Phase 2 work instead of describing Phase 2 only as an indefinite
  prohibition.
- `src/AGENTS.md` will reserve scheduler imports and side effects for a future
  execution boundary and continue to require native-thread resolution before
  scientific imports.
- `src/cyexm/AGENTS.md` will keep the Phase 1 API stable while allowing a separately
  designed workflow/configuration namespace rather than placing scheduler concerns in
  the registration package.
- `src/cyexm/registration/AGENTS.md` will scope the artifact-writer restriction to
  backend writer methods. Artifact lifecycle and orchestration correctly remain in
  `artifacts.py`, `whole_image.py`, and `chunked.py`.
- test guidance will preserve the existing narrow exception for small concrete ANTs
  image/geometry/TIFF boundary objects while keeping real registration in integration
  tests.
- example guidance will permit Phase 2 examples only after the corresponding runtime
  surface is approved and implemented; examples cannot lead production behavior.

Most existing scientific and safety rules remain unchanged. The refresh will avoid
duplicating ancestor rules where inheritance is already clear.

## Phase 2 guardrails

The refreshed instructions will record the following expected boundaries for the next
design without freezing its detailed API:

- The scientific dataset manifest describes rounds, channel filenames, registration
  channels, voxel spacing, and scientific registration choices.
- Machine-specific settings, including local versus cluster execution, queue,
  account, memory, worker startup, and scaling limits, live in a separate execution
  profile or CLI surface.
- A local planner/validator is built before distributed execution so schema and task
  semantics can be tested without a scheduler.
- Independent moving rounds are the initial Dask task boundary.
- The initial HPC profile is expected to request one worker process and one round task
  on a dedicated 512 GB SLURM node, subject to verification of the current partition,
  account, resource syntax, shared filesystem, startup, and memory behavior.
- Every worker applies explicit native-thread configuration before importing ANTs,
  NumPy, or SciPy.
- Round tasks reuse Phase 1 completion records and artifact validation rather than
  adding a competing checkpoint system.
- Serial chunk execution remains inside each round task; no nested process queue,
  Dask chunk graph, or `ParallelChunkedImageRegistration` is introduced by default.

## Policy test

`tests/unit/test_repository_policy.py` will be extended rather than creating a new
test module. The test will:

- assert the exact 18-file `AGENTS.md` inventory;
- assert that every maintained directory has its expected local file;
- require locked validation commands at the root;
- require current-versus-future Phase 2 language and the separate manifest/execution
  boundary;
- require the CI, specification, plan, registration, test, and example rules described
  above; and
- reject known stale or contradictory phrases where a positive assertion alone could
  pass misleading prose.

The test validates repository policy text, not English style. It should normalize
whitespace where line wrapping is irrelevant and use exact paths and key clauses for
stable requirements.

## Validation

Implementation validation will run in this order:

1. the focused repository-policy test;
2. `git diff --check` and a manual inheritance/contradiction review;
3. the complete portable unit selector;
4. the bounded integration selector, because root and test instructions affect how
   that suite is invoked;
5. Ruff format and lint checks;
6. BasedPyright; and
7. all pre-commit hooks.

The refresh must leave `pyproject.toml`, `uv.lock`, production modules, examples, and
workflow behavior unchanged.

## Next work sequence

After this guidance refresh is reviewed and merged, the recommended sequence is:

1. Phase 2 discovery against representative dataset layouts and current SLURM
   configuration.
2. A dedicated Phase 2 design for the YAML scientific schema, execution profile,
   planner, failure model, and public CLI/API.
3. YAML validation and a local sequential planner/executor built with TDD.
4. A local Dask execution path using the same round-task contract.
5. A `SLURMCluster` execution profile with one round per dedicated 512 GB node.
6. Bounded HPC acceptance covering submission, restart, cancellation, shared paths,
   memory, logs, and numerical agreement with serial Phase 1 output.
7. Portable examples and operational documentation only after the corresponding
   runtime surfaces pass their acceptance gates.

## Acceptance criteria

The guidance refresh is complete when:

1. all 18 maintained layers contain an accurate `AGENTS.md`;
2. current Phase 1 behavior is never confused with proposed Phase 2 behavior;
3. Phase 2 guardrails preserve manifest/execution separation, serial numerical
   reference behavior, native-thread safety, and Phase 1 artifact reuse;
4. root commands reproduce locked CI resolution;
5. CI, plans, specifications, registration, tests, and examples have precise local
   rules;
6. the repository-policy test detects missing files and the identified stale
   formulations;
7. no production, dependency, lockfile, example, or workflow behavior changes; and
8. all portable validation gates pass from a clean worktree.
