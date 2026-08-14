# Command-line reference

Run all commands through `uv run` so they use the managed environment. The commands
below operate on caller-supplied images and edited YAML paths; the checked-in YAML
files are starting points, not bundled data.

## Global options

`--json`, `--log-level`, and `--version` are accepted at the top-level `cyexm`,
`workflow`, or action boundary. Option abbreviation is disabled.

| Option | Accepted values | Effect |
| --- | --- | --- |
| `--json` | flag | Write one structured JSON result to stdout on success. |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` | Set diagnostic log verbosity; the default is `INFO`. Input is normalized to uppercase. |
| `--version` | flag | Print `cyexm <version>` and exit. |
| `--help` | flag | Print argparse help and exit. |

```bash
uv run cyexm --version
uv run cyexm workflow status results spheroid-registration-example --json
uv run cyexm workflow plan examples/spheroid-dataset.yaml \
  --execution examples/execution-sequential.yaml --log-level DEBUG --json
```

## `workflow validate`

```
$ cyexm workflow validate DATASET_YAML --execution EXECUTION_YAML
```

Decode the dataset manifest and execution profile, resolve the inputs, inspect TIFF
headers, and construct a plan. Validation is not a registration dry run. On success,
the result identifies the dataset, moving rounds, and scientific fingerprint.

```bash
uv run cyexm workflow validate examples/spheroid-dataset.yaml \
  --execution examples/execution-sequential.yaml
```

## `workflow plan`

```
$ cyexm workflow plan DATASET_YAML --execution EXECUTION_YAML
```

Validate the same inputs and print the canonical scientific plan and resolved resource
request. Planning is mutation-free: it creates no files or directories and does not
open the selected executor.

```bash
uv run cyexm workflow plan examples/spheroid-dataset.yaml \
  --execution examples/execution-local-dask.yaml
```

## `workflow run`

```
$ cyexm workflow run DATASET_YAML --execution EXECUTION_YAML \
  [--force-round N | --force-all]
```

Execute pending work. Ordinary resume is the default: only matching artifact
completion records authorize reuse. `--force-round N` is repeatable and requests
recomputation of selected moving rounds; `--force-all` requests all rounds. The
options are mutually exclusive, and an unknown forced round is an error.

```bash
uv run cyexm workflow run examples/spheroid-dataset.yaml \
  --execution examples/execution-local-dask.yaml

uv run cyexm workflow run examples/spheroid-dataset.yaml \
  --execution examples/execution-local-dask.yaml \
  --force-round 2 --force-round 4

uv run cyexm workflow run examples/spheroid-dataset.yaml \
  --execution examples/execution-local-dask.yaml --force-all
```

## `workflow status`

```
$ cyexm workflow status OUTPUT_ROOT DATASET_ID
```

Inspect the scientific plan and authoritative completion records under an output
root. Status is read-only and does not import scientific or scheduler runtimes. Its
JSON output lists complete, incomplete, and invalid artifact paths.

```bash
uv run cyexm workflow status results spheroid-registration-example
```

## Force and resume

Force affects only a compatible scientific plan. It cannot bypass a scientific-plan
fingerprint mismatch; a scientifically incompatible plan requires a new output root.
Forcing the reference round invalidates all moving descendants because their
fixed-input provenance changes. See [Outputs, restart, and recovery](outputs-and-recovery.md)
for artifact authority, status states, and safe recovery.

## Structured JSON and logging

Human and structured success results go to stdout. `--json` is success-only:
successful workflow actions that reach result emission emit one JSON object.
Failures return before result emission, so stdout is empty, plain-text diagnostics go
to stderr, and there is no JSON error object. `--version` prints version text rather
than a JSON result. Argparse failures can include usage text, and `--log-level DEBUG`
can add a traceback to the stderr diagnostic.

## Failures and exit codes

Configuration and plan errors occur before executor construction. Startup and
preflight errors abort before expensive tasks. Independent round failures do not
cancel other submitted rounds; all submitted work settles, successful artifacts stay
committed, a failed operational summary is written, and the workflow reports the
failed round numbers. For executor, preflight, cancellation, and site-specific
interpretation, see [Fine-grained execution](fine-grained-execution.md).

| Exit code | Meaning |
| --- | --- |
| `0` | Every round completed or was validly reused. |
| `2` | Argument, YAML, configuration, input, plan, output, or fingerprint validation failed. |
| `3` | Executor startup, worker connection, or preflight failure; also invalid fine execution evidence for sequential, Local Dask, or SLURM, including validation after task execution. Unauthenticated current-run summaries are omitted, while valid committed artifacts remain restartable. |
| `4` | At least one round failed after execution began. |
| `130` | Operator interruption or termination through the CLI run handler. |

Automatic retries default to zero. A positive `task_retries` value is an
executor-managed resubmission; Dask's opaque future retry count remains zero, and
each attempt re-enters ordinary artifact validation. Diagnose deterministic
registration and out-of-memory failures instead of repeatedly retrying them.
