from __future__ import annotations

import json
import logging
import sys
from collections.abc import Sequence

from cyexm.cli._commands import CommandOutput, dispatch
from cyexm.cli._parser import build_parser
from cyexm.errors import (
    ConfigurationValidationError,
    ExecutorStartupError,
    InputValidationError,
    ManifestDecodeError,
    RoundExecutionError,
    ThreadConfigurationError,
    WorkflowExecutionError,
    WorkflowPlanError,
)

_LOG = logging.getLogger(__name__)


def _configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _one_line(value: BaseException) -> str:
    return " ".join(str(value).splitlines()) or type(value).__name__


def _diagnostic(error: BaseException, exit_code: int) -> int:
    _LOG.debug("CLI command failed", exc_info=error)
    print(f"cyexm: error: {_one_line(error)}", file=sys.stderr)
    return exit_code


def _emit(output: CommandOutput, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(output.payload, allow_nan=False, sort_keys=True))
    else:
        print(output.human)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Cy-ExM command-line interface and return its process exit code."""
    parser = build_parser()
    try:
        namespace = parser.parse_args(argv)
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 1

    _configure_logging(namespace.log_level)
    try:
        output = dispatch(namespace)
    except KeyboardInterrupt:
        print("cyexm: interrupted", file=sys.stderr)
        return 130
    except (
        ConfigurationValidationError,
        InputValidationError,
        ManifestDecodeError,
        ThreadConfigurationError,
        WorkflowPlanError,
    ) as error:
        return _diagnostic(error, 2)
    except ExecutorStartupError as error:
        return _diagnostic(error, 3)
    except (RoundExecutionError, WorkflowExecutionError) as error:
        return _diagnostic(error, 4)

    _emit(output, as_json=namespace.json)
    return 0


__all__ = ["main"]
