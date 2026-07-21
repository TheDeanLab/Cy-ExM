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

## Memory-normalization erratum

The profile's SLURM memory value is a semantic decimal quantity: the approved
`500GB` value means exactly 500,000,000,000 bytes. It must never be copied verbatim
to `#SBATCH --mem`, because SLURM accepts integer `K`, `M`, `G`, or `T` suffixes and
interprets them as binary units. Cy-ExM parses the strict profile value once and
rounds it up to the smallest whole binary unit at the GiB scale for this profile.
Thus `500GB` becomes the scheduler quantity `466G`, exactly 500,363,689,984 bytes.

The centralized quantity carries three spellings of two facts: the profile spelling
and its semantic byte count, the SLURM spelling and its ceiling-normalized byte
count, and the Dask `GiB` spelling of that same normalized count. The executor passes
`466G` as `job_mem` and `466GiB` as Dask worker `memory`. This avoids Dask's decimal
interpretation of bare `466G`, makes the rendered worker limit exactly match the
scheduler-normalized bytes, and keeps both at or above the semantic request. Strict
loading and direct-profile revalidation reject unsupported units, fractions, zero,
negative values, and quantities whose parse or ceiling normalization exceeds the
supported signed 64-bit byte range.

The rendered-template validator requires the exact normalized scheduler header and
worker limit before scaling. Operational evidence records the profile spelling,
semantic bytes, scheduler spelling, normalized bytes, and Dask spelling. Real HPC
acceptance derives the rendered worker byte limit, requires every worker probe to
match it exactly, requires requested and allocated TRES memory to cover the
normalized scheduler bytes, and requires the actual node capacity to cover that
request. The current `500GB` profile is a portable regression.

Passing `500GB` only as Dask memory and allowing its two-decimal worker rendering was
rejected because the reconstructed worker limit can fall slightly below the semantic
request. Passing bare `466G` to Dask was rejected because Dask treats `G` as decimal
and would under-allocate the worker. Depending only on dask-jobqueue's implicit
header conversion was rejected because Cy-ExM needs explicit, durable normalized
facts for fail-closed validation and acceptance evidence.
