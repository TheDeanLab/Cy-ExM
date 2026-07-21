# CLI package guidance

- The Phase 2A CLI owns parsing, logging setup, lifecycle, formatting, and exit-code
  mapping while delegating to the public workflow API.
- Diagnostics go to stderr and structured JSON and human results to stdout.
- No scientific algorithms belong in the CLI. Allow no scheduler construction,
  filesystem or network activity, or environment mutation at import time.
- Exit semantics are 0 success, 2 config/plan, 3 startup/preflight, 4 round failure,
  and 130 interruption.
