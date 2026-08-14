# Development guide

This guide is the contributor entry point for maintaining Cy-ExM. It keeps the
locked environment, portable checks, and public documentation in step with the
implemented registration and workflow contracts.

## Create the locked development environment

Use Python 3.12 and create the complete locked contributor environment from a source
checkout:

```bash
uv lock --check
uv sync --locked --group dev --extra examples --extra hpc
```

The `hpc` extra installs optional workflow dependencies but does not contact a
scheduler. For site-specific `uv` cache and managed-Python locations on a
quota-limited system, follow [Local parallel and HPC execution](parallel-hpc.md).

## Run focused tests first

Run the smallest affected test before broader validation. Keep portable unit and
integration tests separate from explicitly selected HPC work:

```bash
uv run pytest path/to/affected_test.py -v
```

HPC-marked tests require explicit operator-approved resources, inputs, profiles, and
an allocation. They are not portable gates and must not be selected merely because
the optional dependencies are installed.

## Run the portable gates

Before handing off a change, run the repository's portable checks:

```bash
uv run pytest -m "unit and not hpc" tests/unit -v
uv run pytest -m "integration and not hpc" tests/integration -v
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pre-commit run --all-files
```

These checks do not establish site-specific scheduler behavior. Real-HPC acceptance
remains an operator-approved gate with its own approved inputs and external evidence.

## Build and inspect the documentation

Build the public book after changing a public page or navigation:

```bash
cd docs
uv run --group docs jupyter book build --html
```

Inspect the rendered navigation, links, tables, and code blocks. Do not commit
generated build output unless a separate publication workflow explicitly requires it.

## Maintain optional dependency compatibility

The optional dependency versions, including Dask/distributed 2025.1.0 and
dask-jobqueue 0.9.0, are locked. Change them only with fresh compatibility evidence.
For an intentional dependency change, inspect the resulting `uv.lock` update and
commit it with the dependency edit; do not regenerate the lockfile for documentation
or unrelated maintenance changes.

## Change a public API or schema

Before changing a user-callable API or a schema, search existing source, tests,
documentation, examples, exports, and related names for a reusable contract. Follow
the authoritative [API documentation policy](agentic-api-documentation.md), including
its reuse analysis and required Numpydoc, documentation, and example updates.

Treat transform order, interpolation, axes, masks, schedules, tolerances, metric, and
radius changes as scientific changes. Preserve fixed and moving roles, public XYZ
spacing, and NumPy ZYX volume order in documentation and tests.

## Keep historical evidence intact

Characterization and acceptance records are audit evidence. Preserve their identities,
hashes, environments, results, and limitations. New capability guidance should link
to the relevant evidence instead of revising a historical record to make it describe
a later implementation.
