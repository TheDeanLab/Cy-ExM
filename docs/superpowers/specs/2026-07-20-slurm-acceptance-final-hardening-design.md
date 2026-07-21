# SLURM Acceptance Final Hardening Design

Status: approved for implementation on 2026-07-20.

## Context

The Phase 2E environment harness currently scopes cleanup with `squeue --jobs` and
validates only part of the SLURM resource-shaping option surface. A completed job ID
can make a scheduler-side ID query fail instead of proving that the owned job has
departed. Alternate SLURM option spellings can also bypass configuration validation,
and `SlurmExecutor` does not reject an unsafe rendered template before scaling.

This correction affects operational safety only. Registration algorithms, scientific
configuration, coordinate conventions, artifacts, datasets, and round scheduling are
unchanged.

## Goals and non-goals

Cleanup must query only the current user and exact harness job name, then classify
rows locally against sorted recorded job IDs. Unrelated same-name jobs remain visible
in evidence but neither become owned nor block successful owned cleanup.

Resource, node, task, output-path, and managed worker-option collisions must fail
closed at the strict configuration/direct-profile boundary. The actual rendered
dask-jobqueue template must independently satisfy the managed contract before the
first `cluster.scale` call.

Safe site directives remain supported, including account, QoS, constraint, mail, and
the approved `--exclusive` directive. The correction does not submit a job, broaden
the HPC test's authority, add a scheduler mutation command, or implement general
SLURM script parsing.

## Selected architecture

The cleanup collector always invokes `squeue` with `--user`, `--name`, and the exact
output format, never `--jobs`. It persists the sorted owned ID set in the query
record, parses every returned row, and separates owned rows from unrelated rows. The
poll terminates successfully when the owned set has no observed rows, even when
unrelated same-name rows remain. Query errors and timeouts retain the existing durable
attempt evidence.

Strict execution-profile validation recognizes long, abbreviated long, short,
attached, joined, and equals forms for managed SLURM options. The protected surface
includes node/task counts, per-task CPU and memory variants, partition and job name,
output/error/chdir paths, and related resource-shaping aliases. Directly constructed
profiles receive the same validation when `SlurmExecutor` begins.

After cluster construction renders a template, a production-owned validator parses
the header and worker command. It requires exactly the executor-managed job name,
partition, task count, CPU, memory, walltime, exclusivity policy, process count,
interface, local directory, and `round_slot` facts, with no conflicting aliases. The
validator runs before `cluster.scale`; a failure closes the unscaled owned cluster and
configuration context through the existing cleanup path.

## Alternatives

Keeping scheduler-side ID filtering was rejected because departed IDs can turn an
empty cleanup result into an `squeue` error. Configuration-only validation was
rejected because direct models and renderer changes could still bypass it. A complete
extra-directive whitelist was rejected because it would unnecessarily forbid safe
site policy such as account, QoS, constraint, and mail settings.

## Failure handling and evidence

Malformed queue rows and command failures remain explicit query-error outcomes. A
timeout contains every owned observation plus every unrelated row returned by each
snapshot. Unsafe directives raise configuration validation errors; unsafe rendered
templates raise executor startup errors before any scale or job submission attempt.
Existing cleanup evidence records cluster/configuration close attempts.

## Testing and acceptance

Portable tests use fake `squeue` results and fake dask-jobqueue modules only. They
cover one and multiple owned IDs, a departed ID, unrelated same-name rows, durable
query failure and timeout evidence, and deterministic sorted query records. Directive
tests cover every managed long/short/equals/joined/alias family, while good safe
extras and the exact approved profile remain accepted. Executor tests prove unsafe
profiles or rendered templates produce zero `scale` calls and no job, while the good
template scales normally.

Acceptance requires the focused RED/GREEN evidence, all portable unit and integration
tests, lock validation, Ruff, formatting, BasedPyright, pre-commit, and diff checks.
No real `-m hpc` run or approved dataset access is part of this correction.
