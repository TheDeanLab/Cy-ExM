# Task 9 Report: Public Fine Workflow Integration

## Outcome

Task 9 integrates fine-grained execution into the existing `run_workflow` and
`inspect_workflow_status` surfaces without adding another public orchestrator.
Coarse execution remains the default and preserves its existing `{}` execution
payload. Fine sequential and local-Dask runs now persist one strict
scheduler-neutral operational envelope; fine SLURM continues to use its strict
schema-version-2 envelope and validated `fine_graph` extension.

## Reuse analysis

- Extended `run_workflow`, `_validate_round_result`, `SequentialExecutor`, and
  `LocalDaskExecutor`; no parallel workflow API, executor hierarchy, or restart
  authority was added.
- Reused `FineGraphController` graph descriptions, canonical task ordering, and
  `validate_fine_graph_description` rather than reimplementing dependency or
  blocked-descendant semantics.
- Reused the existing SLURM evidence decoder unchanged for scheduler-specific
  scale, preflight, placement, cancellation, and cleanup validation.
- Phase 1 completion records remain the only scientific reuse authority.

## TDD evidence

Focused RED cases were observed before their production fixes:

- fine success summary rejected the authoritative validation-task log;
- scheduler-neutral fine evidence was omitted as `{}`;
- SequentialExecutor and LocalDaskExecutor exposed no granular evidence;
- failed fine summaries rejected the representative stage log;
- forged summaries could disagree with task evidence;
- missing fine evidence did not fail closed;
- interrupted Local Dask work omitted a submitted but unsettled attempt;
- cleanup failures lost exact exception evidence; and
- fine task evidence could embed a forged `RoundTask`.

Each case was made GREEN independently before the focused regression suite.

## Implemented contracts

- Mode-aware round validation keeps coarse `RoundTask.log_path` exact. Successful
  fine summaries require the validation-task log; failure and cancellation require
  an exact canonical representative fine-task log.
- Fine evidence is versioned, canonical, detached, and validated after scientific
  work. It includes normalized mode/executor/cap, canonical task graph and
  dependencies, every actual task result, attempts, blocked descendants,
  cancellation, and exact cleanup outcomes.
- Fine results are bound to the exact submitted authoritative `RoundTask`, exact
  expanded native-thread evidence, task outputs, dependencies, timestamps, logs,
  and failure contract. Public summaries are cross-checked against the canonical
  representative task evidence.
- Sequential interruption retains the actual partial graph, settled results, and
  admissions. Local Dask retains live submissions even when interruption occurs
  before a future settles, cancels unfinished futures, and records owned cleanup.
- Missing or malformed fine evidence fails closed through the existing durable
  execution-evidence failure path.
- Scientific JSON and fingerprints remain byte-identical across coarse and fine
  profiles. Operational evidence remains outside scientific identity.
- Coarse-complete to fine excludes accepted moving work from the graph. Fine-partial
  to coarse preserves local warp evidence for ordinary Phase 1 validation.
- Public status ignores invalid internal prepared manifests and scalar cores when
  all independently verifiable public outputs remain valid.
- Force propagation, deterministic result ordering, CLI exit mapping, native-thread
  boundaries, coarse SLURM evidence, and coordinator-only scheduling retain their
  existing tested behavior.

## Documentation

- Expanded Numpydoc ownership, side effects, failure settlement, restart authority,
  and coordinator-only scheduling for `run_workflow`, `SequentialExecutor`,
  `LocalDaskExecutor`, and `SlurmExecutor`.
- Updated workflow and execution `AGENTS.md` guidance for authoritative logs,
  scheduler-neutral evidence, exact summary cross-checking, and cleanup evidence.

## Verification

- Focused API/status/thread/CLI/sequential/Dask/dispatcher/SLURM regressions:
  525 passed.
- Portable unit suite: all 2,238 collected non-HPC unit tests passed.
- Portable integration suite: all 58 collected non-HPC integration tests passed.
- `uv lock --check`: passed.
- `uv sync --locked --group dev --extra examples --extra hpc`: passed.
- Repository-wide Ruff check and format check: passed.
- Repository-wide BasedPyright: passed with 0 errors, warnings, or notes.
- `pre-commit run --all-files`: passed.
- `git diff --check`: passed.

No SLURM jobs were submitted, no active datasets were accessed, and the unrelated
`src/cy_exm.egg-info/` tree was left untouched.
