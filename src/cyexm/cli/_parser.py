from __future__ import annotations

import argparse

from cyexm import __version__

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _log_level(value: str) -> str:
    return value.upper()


def _add_common_options(
    parser: argparse.ArgumentParser,
    *,
    include_defaults: bool,
) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        default=False if include_defaults else argparse.SUPPRESS,
        help="write one structured JSON result to stdout",
    )
    parser.add_argument(
        "--log-level",
        type=_log_level,
        choices=_LOG_LEVELS,
        default="INFO" if include_defaults else argparse.SUPPRESS,
        help="set diagnostic log verbosity",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"cyexm {__version__}",
    )


def _workflow_action_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    help_text: str,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        name,
        allow_abbrev=False,
        help=help_text,
    )
    _add_common_options(parser, include_defaults=False)
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build the exact Cy-ExM workflow command-line surface."""
    parser = argparse.ArgumentParser(prog="cyexm", allow_abbrev=False)
    _add_common_options(parser, include_defaults=True)
    commands = parser.add_subparsers(dest="command", required=True)

    workflow = commands.add_parser(
        "workflow",
        allow_abbrev=False,
        help="validate, plan, run, or inspect a registration workflow",
    )
    _add_common_options(workflow, include_defaults=False)
    workflow_commands = workflow.add_subparsers(
        dest="workflow_command",
        required=True,
    )

    for command, help_text in (
        ("validate", "validate workflow inputs and construct a plan"),
        ("plan", "print the validated workflow plan and resource request"),
    ):
        action = _workflow_action_parser(
            workflow_commands,
            command,
            help_text=help_text,
        )
        action.add_argument("dataset_yaml", metavar="DATASET_YAML")
        action.add_argument(
            "--execution",
            dest="execution_yaml",
            metavar="EXECUTION_YAML",
            required=True,
        )

    run = _workflow_action_parser(
        workflow_commands,
        "run",
        help_text="execute or resume a validated workflow",
    )
    run.add_argument("dataset_yaml", metavar="DATASET_YAML")
    run.add_argument(
        "--execution",
        dest="execution_yaml",
        metavar="EXECUTION_YAML",
        required=True,
    )
    force = run.add_mutually_exclusive_group()
    force.add_argument(
        "--force-round",
        action="append",
        default=[],
        type=int,
        metavar="N",
    )
    force.add_argument("--force-all", action="store_true")

    status = _workflow_action_parser(
        workflow_commands,
        "status",
        help_text="inspect authoritative workflow artifact status",
    )
    status.add_argument("output_root", metavar="OUTPUT_ROOT")
    status.add_argument("dataset_id", metavar="DATASET_ID")

    return parser


__all__ = ["build_parser"]
