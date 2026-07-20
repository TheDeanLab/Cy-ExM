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
_ENV_VALUE_OPTIONS = frozenset({"-C", "--argv0", "--chdir", "-u", "--unset"})
_SIMPLE_WRAPPERS = frozenset({"nohup", "setsid"})
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
_UNPARSEABLE_COMMAND = "<unparseable command>"


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


def _split_env_string(value: str) -> tuple[str, ...] | None:
    try:
        return tuple(shlex.split(value, comments=False, posix=True))
    except ValueError:
        return None


def _parse_env(tokens: tuple[str, ...], *, depth: int) -> str | None:
    if depth > _MAX_PARSE_DEPTH:
        return _UNPARSEABLE_COMMAND
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return _parse_argv(tokens[index + 1 :], depth=depth + 1)
        if _is_assignment(token) or token in _ENV_NO_VALUE_OPTIONS:
            index += 1
            continue
        split_value: str | None = None
        next_index = index + 1
        if token in {"-S", "--split-string"}:
            if next_index >= len(tokens):
                return _UNPARSEABLE_COMMAND
            split_value = tokens[next_index]
            next_index += 1
        elif token.startswith("-S") and token != "-S":
            split_value = token[2:]
        elif token.startswith("--split-string="):
            split_value = token.partition("=")[2]
        if split_value is not None:
            split_tokens = _split_env_string(split_value)
            if split_tokens is None:
                return _UNPARSEABLE_COMMAND
            return _parse_env(
                (*split_tokens, *tokens[next_index:]),
                depth=depth + 1,
            )
        option = token.split("=", 1)[0]
        if option in _ENV_VALUE_OPTIONS:
            if "=" in token:
                index += 1
            elif index + 1 < len(tokens):
                index += 2
            else:
                return _UNPARSEABLE_COMMAND
            continue
        if token.startswith("-u") and token != "-u":
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        return _parse_argv(tokens[index:], depth=depth + 1)
    return None


def _unwrap_exec(tokens: tuple[str, ...]) -> tuple[str, ...] | None:
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return tokens[index + 1 :]
        if token == "-a":
            if index + 1 >= len(tokens):
                return None
            index += 2
            continue
        if token.startswith("-") and token != "-":
            options = token[1:]
            if options and set(options) <= {"c", "l"}:
                index += 1
                continue
            if options.endswith("a") and set(options[:-1]) <= {"c", "l"}:
                if index + 1 >= len(tokens):
                    return None
                index += 2
                continue
            return None
        return tokens[index:]
    return ()


def _parse_command_wrapper(tokens: tuple[str, ...], *, depth: int) -> str | None:
    index = 0
    resolution_query = False
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            if resolution_query:
                return None
            return _parse_argv(tokens[index + 1 :], depth=depth + 1)
        if not token.startswith("-") or token == "-":
            break
        options = token[1:]
        if not options or not set(options) <= {"V", "p", "v"}:
            return _UNPARSEABLE_COMMAND
        resolution_query = resolution_query or "v" in options or "V" in options
        index += 1
    if resolution_query:
        return None
    return _parse_argv(tokens[index:], depth=depth + 1)


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
        return _UNPARSEABLE_COMMAND
    tokens = _strip_leading_assignments(tokens)
    if not tokens:
        return None
    executable = _basename(tokens[0])
    if executable in SCHEDULER_COMMANDS:
        return executable
    arguments = tokens[1:]
    if executable == "env":
        return _parse_env(arguments, depth=depth + 1)
    if executable in _SHELL_COMMANDS:
        command = _shell_command_text(tokens)
        return None if command is None else _parse_shell(command, depth=depth + 1)
    if executable == "exec":
        command = _unwrap_exec(arguments)
        if command is None:
            return _UNPARSEABLE_COMMAND
        return _parse_argv(command, depth=depth + 1)
    if executable == "command":
        return _parse_command_wrapper(arguments, depth=depth + 1)
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


def _backtick_substitutions(command: str) -> tuple[tuple[str, ...], str] | None:
    substitutions: list[str] = []
    masked: list[str] = []
    quote: str | None = None
    comment_position = True
    index = 0
    while index < len(command):
        character = command[index]
        if quote == "'":
            masked.append(character)
            if character == "'":
                quote = None
            index += 1
            continue
        if character == "\\":
            masked.append(character)
            if index + 1 < len(command):
                masked.append(command[index + 1])
                index += 2
            else:
                index += 1
            comment_position = False
            continue
        if quote is None and character == "#" and comment_position:
            newline = command.find("\n", index)
            if newline < 0:
                masked.append(command[index:])
                break
            masked.append(command[index : newline + 1])
            index = newline + 1
            comment_position = True
            continue
        if character == "'" and quote is None:
            quote = "'"
            masked.append(character)
            comment_position = False
            index += 1
            continue
        if character == '"':
            quote = None if quote == '"' else '"'
            masked.append(character)
            comment_position = False
            index += 1
            continue
        if character == "`":
            closing = index + 1
            while closing < len(command):
                if command[closing] == "\\" and closing + 1 < len(command):
                    closing += 2
                    continue
                if command[closing] == "`":
                    break
                closing += 1
            if closing >= len(command):
                return None
            substitutions.append(command[index + 1 : closing])
            masked.append("x")
            comment_position = False
            index = closing + 1
            continue
        masked.append(character)
        if quote is None:
            comment_position = character.isspace() or character in ";&|()"
        index += 1
    return tuple(substitutions), "".join(masked)


def _parse_shell(command: str, *, depth: int) -> str | None:
    if depth > _MAX_PARSE_DEPTH:
        return _UNPARSEABLE_COMMAND
    extracted = _backtick_substitutions(command)
    if extracted is None:
        return _UNPARSEABLE_COMMAND
    substitutions, masked_command = extracted
    for substitution in substitutions:
        nested_command = substitution.replace("\\`", "`")
        blocked = _parse_shell(nested_command, depth=depth + 1)
        if blocked is not None:
            return blocked
    try:
        lexer = shlex.shlex(masked_command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = tuple(lexer)
    except ValueError:
        return _UNPARSEABLE_COMMAND

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
    if command == _UNPARSEABLE_COMMAND:
        raise RuntimeError(
            "portable tests must not execute an unparseable or excessively nested "
            "command"
        )
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
