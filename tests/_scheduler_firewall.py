"""Portable-test protection against accidental external SLURM client calls."""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

SCHEDULER_COMMANDS = frozenset(
    {"sacct", "sbatch", "scancel", "scontrol", "sinfo", "squeue", "srun"}
)
SCHEDULER_FIREWALL_ENV = "CYEXM_PORTABLE_SCHEDULER_FIREWALL"
_SHELL_COMMANDS = frozenset({"bash", "dash", "ksh", "sh", "zsh"})
_SHELL_SEPARATORS = frozenset({"&", "&&", "(", ")", ";", ";;", "|", "||"})
_SHELL_PREFIX_KEYWORDS = frozenset(
    {"!", "do", "elif", "else", "if", "then", "until", "while"}
)
_SHELL_END_KEYWORDS = frozenset({"done", "esac", "fi"})
_ENV_NO_VALUE_OPTIONS = frozenset(
    {"-0", "--debug", "-i", "--ignore-environment", "--null", "-v"}
)
_ENV_VALUE_OPTIONS = frozenset(
    {"-C", "--argv0", "--chdir", "-S", "--split-string", "-u", "--unset"}
)
_SIMPLE_WRAPPERS = frozenset({"command", "exec", "nohup", "setsid"})
_WRAPPER_VALUE_OPTIONS = {
    "nice": frozenset({"-n", "--adjustment"}),
    "stdbuf": frozenset({"-e", "--error", "-i", "--input", "-o", "--output"}),
    "sudo": frozenset(
        {
            "-C",
            "--chdir",
            "--chroot",
            "--close-from",
            "--command-timeout",
            "-D",
            "-g",
            "--group",
            "-h",
            "--host",
            "--other-user",
            "-p",
            "--prompt",
            "-R",
            "-r",
            "--role",
            "-T",
            "-t",
            "--type",
            "-U",
            "-u",
            "--user",
        }
    ),
    "time": frozenset({"-f", "--format", "-o", "--output"}),
}
_MAX_PARSE_DEPTH = 16


def _command_arguments(arguments: object) -> tuple[str, ...]:
    if isinstance(arguments, (str, bytes, os.PathLike)):
        return (os.fsdecode(arguments),)
    if isinstance(arguments, Sequence):
        return tuple(
            os.fsdecode(argument)
            if isinstance(argument, (str, bytes, os.PathLike))
            else repr(argument)
            for argument in arguments
        )
    return (repr(arguments),)


def _basename(token: str) -> str:
    return Path(token).name


def _is_assignment(token: str) -> bool:
    name, separator, _value = token.partition("=")
    return bool(separator and name and name.isidentifier())


def _strip_leading_assignments(tokens: tuple[str, ...]) -> tuple[str, ...]:
    index = 0
    while index < len(tokens) and _is_assignment(tokens[index]):
        index += 1
    return tokens[index:]


def _strip_generic_options(
    tokens: tuple[str, ...],
    *,
    value_options: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return tokens[index + 1 :]
        if not token.startswith("-") or token == "-":
            return tokens[index:]
        option = token.split("=", 1)[0]
        if option in value_options and "=" not in token:
            index += 2
        else:
            index += 1
    return ()


def _unwrap_env(tokens: tuple[str, ...]) -> tuple[str, ...]:
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return tokens[index + 1 :]
        if _is_assignment(token) or token in _ENV_NO_VALUE_OPTIONS:
            index += 1
            continue
        option = token.split("=", 1)[0]
        if option in _ENV_VALUE_OPTIONS:
            index += 1 if "=" in token else 2
            continue
        if token.startswith("-u") and token != "-u":
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        return tokens[index:]
    return ()


def _shell_command_text(tokens: tuple[str, ...]) -> str | None:
    index = 1
    while index < len(tokens):
        option = tokens[index]
        if option in {"-O", "+O", "--init-file", "--rcfile"}:
            index += 2
            continue
        if option.startswith("-") and not option.startswith("--"):
            if "c" in option[1:]:
                command_index = index + 1
                return tokens[command_index] if command_index < len(tokens) else None
            index += 1
            continue
        index += 1
    return None


def _unwrap_timeout(tokens: tuple[str, ...]) -> tuple[str, ...]:
    command = _strip_generic_options(
        tokens,
        value_options=frozenset({"-k", "--kill-after", "-s", "--signal"}),
    )
    return command[1:] if command else ()


def _parse_argv(tokens: tuple[str, ...], *, depth: int) -> str | None:
    if depth > _MAX_PARSE_DEPTH:
        return None
    tokens = _strip_leading_assignments(tokens)
    if not tokens:
        return None
    executable = _basename(tokens[0])
    if executable in SCHEDULER_COMMANDS:
        return executable
    arguments = tokens[1:]
    if executable == "env":
        return _parse_argv(_unwrap_env(arguments), depth=depth + 1)
    if executable in _SHELL_COMMANDS:
        command = _shell_command_text(tokens)
        return None if command is None else _parse_shell(command, depth=depth + 1)
    if executable in _SIMPLE_WRAPPERS:
        return _parse_argv(
            _strip_generic_options(arguments),
            depth=depth + 1,
        )
    if executable in _WRAPPER_VALUE_OPTIONS:
        return _parse_argv(
            _strip_generic_options(
                arguments,
                value_options=_WRAPPER_VALUE_OPTIONS[executable],
            ),
            depth=depth + 1,
        )
    if executable == "timeout":
        return _parse_argv(_unwrap_timeout(arguments), depth=depth + 1)
    return None


def _parse_shell_segment(tokens: tuple[str, ...], *, depth: int) -> str | None:
    while tokens and tokens[0] in _SHELL_PREFIX_KEYWORDS:
        tokens = tokens[1:]
    if not tokens or tokens[0] in _SHELL_END_KEYWORDS:
        return None
    return _parse_argv(tokens, depth=depth + 1)


def _parse_shell(command: str, *, depth: int) -> str | None:
    if depth > _MAX_PARSE_DEPTH:
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = tuple(lexer)
    except ValueError:
        return None

    segment: list[str] = []
    for token in (*tokens, ";"):
        if token in _SHELL_SEPARATORS:
            blocked = _parse_shell_segment(tuple(segment), depth=depth + 1)
            if blocked is not None:
                return blocked
            segment.clear()
        else:
            segment.append(token)
    return None


def scheduler_command(arguments: object, *, shell: bool = False) -> str | None:
    """Return a scheduler executable only when it occupies a command position."""
    tokens = _command_arguments(arguments)
    if not tokens:
        return None
    if shell:
        return _parse_shell(tokens[0], depth=0)
    return _parse_argv(tokens, depth=0)


def _raise_blocked(command: str) -> None:
    raise RuntimeError(
        f"portable tests must not invoke external scheduler command {command!r}"
    )


def create_scheduler_shims(root: Path) -> Path:
    """Create inert scheduler-named executables for child shell containment."""
    root.mkdir(parents=True, exist_ok=True)
    for command in sorted(SCHEDULER_COMMANDS):
        executable = root / command
        executable.write_text(
            "#!/bin/sh\n"
            f'printf "portable tests must not invoke external scheduler command '
            f"'{command}'\\n\" >&2\n"
            "exit 126\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
    return root


def _command_from_path_and_argv(path: object, argv: object) -> str | None:
    return scheduler_command((path,)) or scheduler_command(argv)


def install_process_guards(
    patch: Callable[[object, str, object], None],
) -> None:
    """Patch stdlib process-launch surfaces while preserving caller ownership."""
    original_popen = subprocess.Popen

    def guarded_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        arguments = kwargs.get("args", args[0] if args else ())
        shell = kwargs.get("shell") is True
        command = scheduler_command(arguments, shell=shell)
        if command is None and kwargs.get("executable") is not None:
            command = scheduler_command((kwargs["executable"],))
        if command is not None:
            _raise_blocked(command)
        return original_popen(*args, **kwargs)

    patch(subprocess, "Popen", guarded_popen)

    original_system = os.system

    def guarded_system(command_text: str) -> int:
        command = scheduler_command(command_text, shell=True)
        if command is not None:
            _raise_blocked(command)
        return original_system(command_text)

    patch(os, "system", guarded_system)

    for name in ("execv", "execve", "execvp", "execvpe"):
        original = getattr(os, name, None)
        if original is None:
            continue

        def guarded_execv(
            path: object,
            argv: object,
            *args: Any,
            _original: Callable[..., Any] = original,
        ) -> Any:
            command = _command_from_path_and_argv(path, argv)
            if command is not None:
                _raise_blocked(command)
            return _original(path, argv, *args)

        patch(os, name, guarded_execv)

    for name in ("execl", "execle", "execlp", "execlpe"):
        original = getattr(os, name, None)
        if original is None:
            continue
        has_environment = name.endswith("e")

        def guarded_execl(
            path: object,
            *args: Any,
            _original: Callable[..., Any] = original,
            _has_environment: bool = has_environment,
        ) -> Any:
            argv = args[:-1] if _has_environment else args
            command = _command_from_path_and_argv(path, argv)
            if command is not None:
                _raise_blocked(command)
            return _original(path, *args)

        patch(os, name, guarded_execl)

    for name in ("spawnv", "spawnve", "spawnvp", "spawnvpe"):
        original = getattr(os, name, None)
        if original is None:
            continue

        def guarded_spawnv(
            mode: int,
            path: object,
            argv: object,
            *args: Any,
            _original: Callable[..., Any] = original,
        ) -> Any:
            command = _command_from_path_and_argv(path, argv)
            if command is not None:
                _raise_blocked(command)
            return _original(mode, path, argv, *args)

        patch(os, name, guarded_spawnv)

    for name in ("spawnl", "spawnle", "spawnlp", "spawnlpe"):
        original = getattr(os, name, None)
        if original is None:
            continue
        has_environment = name.endswith("e")

        def guarded_spawnl(
            mode: int,
            path: object,
            *args: Any,
            _original: Callable[..., Any] = original,
            _has_environment: bool = has_environment,
        ) -> Any:
            argv = args[:-1] if _has_environment else args
            command = _command_from_path_and_argv(path, argv)
            if command is not None:
                _raise_blocked(command)
            return _original(mode, path, *args)

        patch(os, name, guarded_spawnl)

    for name in ("posix_spawn", "posix_spawnp"):
        original = getattr(os, name, None)
        if original is None:
            continue

        def guarded_posix_spawn(
            path: object,
            argv: object,
            *args: Any,
            _original: Callable[..., Any] = original,
            **kwargs: Any,
        ) -> Any:
            command = _command_from_path_and_argv(path, argv)
            if command is not None:
                _raise_blocked(command)
            return _original(path, argv, *args, **kwargs)

        patch(os, name, guarded_posix_spawn)


__all__ = [
    "SCHEDULER_COMMANDS",
    "SCHEDULER_FIREWALL_ENV",
    "create_scheduler_shims",
    "install_process_guards",
    "scheduler_command",
]
