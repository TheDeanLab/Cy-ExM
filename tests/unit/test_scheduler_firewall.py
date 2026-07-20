from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import conftest as suite_config
import pytest

pytestmark = pytest.mark.unit

_SENTINEL = "UNGUARDED-SCHEDULER-SHIM"


@pytest.fixture
def safe_host_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    """Put inert sentinels before host clients so RED cannot reach SLURM."""
    executable_root = tmp_path / "safe-host-clients"
    executable_root.mkdir()
    for name in (*sorted(suite_config.EXTERNAL_SCHEDULER_COMMANDS), "bash", "sh"):
        executable = executable_root / name
        executable.write_text(
            f"#!/bin/sh\nprintf '{_SENTINEL}:{name}\\n' >&2\nexit 91\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
    monkeypatch.setenv(
        "PATH",
        f"{executable_root}{os.pathsep}{os.environ.get('PATH', '')}",
    )
    yield executable_root


@pytest.mark.parametrize(
    ("arguments", "expected"),
    (
        (("env", "-u", "NAME", "sbatch"), "sbatch"),
        (("env", "--unset", "NAME", "sh", "-c", "squeue"), "squeue"),
        (("env", "-i", "CYEXM_TEST=1", "scontrol"), "scontrol"),
        (("bash", "-lc", "squeue"), "squeue"),
        (("bash", "-ec", "/usr/bin/sacct"), "sacct"),
        (("/usr/bin/sbatch", "job.sh"), "sbatch"),
        (("nohup", "srun"), "srun"),
        (("nice", "-n", "5", "sinfo"), "sinfo"),
        (("sudo", "-u", "nobody", "squeue"), "squeue"),
    ),
)
def test_scheduler_parser_recurses_through_execution_wrappers(
    arguments: tuple[str, ...],
    expected: str,
) -> None:
    assert suite_config._external_scheduler_command(arguments) == expected


@pytest.mark.parametrize(
    "command",
    (
        'printf "%s\\n" "squeue" # sbatch is documentation',
        "echo 'scontrol' # srun",
        "# sbatch squeue scontrol",
    ),
)
def test_scheduler_parser_allows_quoted_data_and_comments(command: str) -> None:
    assert suite_config._external_scheduler_command(command, shell=True) is None


@pytest.mark.parametrize(
    ("command", "expected"),
    (
        ("echo ready; squeue", "squeue"),
        ("echo $(scontrol)", "scontrol"),
        ("CYEXM_TEST=1 /usr/bin/sbatch job.sh", "sbatch"),
    ),
)
def test_scheduler_parser_finds_shell_command_positions(
    command: str,
    expected: str,
) -> None:
    assert suite_config._external_scheduler_command(command, shell=True) == expected


@pytest.mark.parametrize(
    ("arguments", "expected"),
    (
        (("env", "-u", "NAME", "sbatch"), "sbatch"),
        (("env", "sh", "-c", "squeue"), "squeue"),
        (("bash", "-lc", "scontrol"), "scontrol"),
    ),
)
def test_popen_blocks_wrapped_scheduler_commands_before_execution(
    safe_host_clients: Path,
    arguments: tuple[str, ...],
    expected: str,
) -> None:
    del safe_host_clients
    with pytest.raises(
        RuntimeError,
        match=(
            f"portable tests must not invoke external scheduler command {expected!r}"
        ),
    ):
        subprocess.run(arguments, check=False)


def test_popen_blocks_an_absolute_scheduler_executable_before_execution(
    safe_host_clients: Path,
) -> None:
    executable = safe_host_clients / "sbatch"
    with pytest.raises(RuntimeError, match="scheduler command 'sbatch'"):
        subprocess.run([executable], check=False)


def test_popen_allows_scheduler_names_as_shell_data(
    safe_host_clients: Path,
) -> None:
    del safe_host_clients
    completed = subprocess.run(
        [
            "/bin/sh",
            "-c",
            'printf "%s\\n" "squeue" # sbatch is documentation',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "squeue\n"


@pytest.mark.parametrize("absolute", [False, True])
def test_child_python_blocks_scheduler_subprocesses(
    safe_host_clients: Path,
    absolute: bool,
) -> None:
    executable = str(safe_host_clients / "squeue") if absolute else "squeue"
    code = f"import subprocess\nsubprocess.run([{executable!r}], check=False)\n"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "portable tests must not invoke external scheduler command 'squeue'" in (
        completed.stderr
    )
    assert _SENTINEL not in completed.stderr


def test_child_python_blocks_scheduler_execvp(safe_host_clients: Path) -> None:
    del safe_host_clients
    code = "import os\nos.execvp('squeue', ['squeue'])\n"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "portable tests must not invoke external scheduler command 'squeue'" in (
        completed.stderr
    )
    assert _SENTINEL not in completed.stderr


def test_os_system_blocks_scheduler_commands(safe_host_clients: Path) -> None:
    del safe_host_clients
    with pytest.raises(RuntimeError, match="scheduler command 'squeue'"):
        os.system("squeue")


def test_os_spawnvp_blocks_scheduler_commands(safe_host_clients: Path) -> None:
    del safe_host_clients
    with pytest.raises(RuntimeError, match="scheduler command 'scontrol'"):
        os.spawnvp(os.P_WAIT, "scontrol", ["scontrol"])


def test_firewall_prepends_owned_shims_to_path() -> None:
    for command in suite_config.EXTERNAL_SCHEDULER_COMMANDS:
        resolved = shutil.which(command)
        assert resolved is not None
        assert "cyexm-scheduler-firewall" in resolved


def test_owned_shim_contains_a_dynamic_shell_command() -> None:
    completed = subprocess.run(
        ["/bin/sh", "-c", 'client=squeue; "$client"'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 126
    assert "portable tests must not invoke external scheduler command 'squeue'" in (
        completed.stderr
    )


def test_firewall_activation_restores_environment_and_process_functions_exactly(
    tmp_path: Path,
) -> None:
    controlled = (
        "PATH",
        "PYTHONPATH",
        suite_config.SCHEDULER_FIREWALL_ENV,
    )
    before_environment = {name: os.environ.get(name) for name in controlled}
    before_functions = (subprocess.Popen, os.system, os.execvp, os.spawnvp)
    patcher = pytest.MonkeyPatch()
    suite_config._activate_scheduler_firewall(patcher, tmp_path / "nested")
    patcher.undo()

    assert {name: os.environ.get(name) for name in controlled} == before_environment
    assert (subprocess.Popen, os.system, os.execvp, os.spawnvp) == before_functions
