from __future__ import annotations

import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

import cyexm.cli._commands as cli_commands
import cyexm.workflow.api as workflow_api
from cyexm import __version__
from cyexm.cli import main
from cyexm.cli._parser import build_parser
from cyexm.errors import (
    ConfigurationValidationError,
    ExecutorStartupError,
    WorkerPreflightError,
    WorkflowExecutionError,
    WorkflowPlanError,
)

pytestmark = pytest.mark.unit


class _StructuredResult:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, object]:
        return dict(self._payload)


class _FakeWorkflowApi:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.manifest = object()
        self.execution = SimpleNamespace(
            executor="sequential",
            paths=SimpleNamespace(
                dataset_root=Path("/dataset"),
                output_root=Path("/output"),
                worker_local_directory=None,
            ),
            native_threads=SimpleNamespace(itk=1, blas=1),
            parallelism=SimpleNamespace(mode="coarse"),
            local_dask=None,
            slurm=None,
        )
        self.plan = SimpleNamespace(
            dataset_id="dataset-1",
            scientific_fingerprint="abc123",
            scientific_json=(
                '{"dataset_id":"dataset-1","rounds":[{"round":2}],"schema_version":1}'
            ),
            tasks=(SimpleNamespace(round=2),),
            execution=self.execution,
        )
        self.result = _StructuredResult(
            {
                "dataset_id": "dataset-1",
                "rounds": [],
                "run_record": "/output/dataset-1/workflow-runs/run.json",
                "status": "completed",
            }
        )
        self.status = _StructuredResult(
            {
                "complete": False,
                "dataset_id": "dataset-1",
                "rounds": [],
                "scientific_fingerprint": "abc123",
            }
        )

    def load_dataset_manifest(self, path: str) -> object:
        self.calls.append(("load_dataset_manifest", path))
        return self.manifest

    def load_execution_profile(self, path: str) -> object:
        self.calls.append(("load_execution_profile", path))
        return self.execution

    def build_workflow_plan(self, manifest: object, execution: object) -> object:
        self.calls.append(("build_workflow_plan", manifest, execution))
        return self.plan

    def run_workflow(
        self,
        plan: object,
        *,
        force_rounds: frozenset[int],
        force_all: bool,
    ) -> _StructuredResult:
        self.calls.append(("run_workflow", plan, force_rounds, force_all))
        return self.result

    def inspect_workflow_status(
        self, output_root: str, dataset_id: str
    ) -> _StructuredResult:
        self.calls.append(("inspect_workflow_status", output_root, dataset_id))
        return self.status


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> _FakeWorkflowApi:
    api = _FakeWorkflowApi()
    monkeypatch.setattr(cli_commands, "_load_workflow_api", lambda: api)
    return api


def test_run_parses_repeatable_force_rounds() -> None:
    namespace = build_parser().parse_args(
        [
            "workflow",
            "run",
            "dataset.yaml",
            "--execution",
            "local.yaml",
            "--force-round",
            "2",
            "--force-round",
            "4",
        ]
    )

    assert namespace.command == "workflow"
    assert namespace.workflow_command == "run"
    assert namespace.force_round == [2, 4]
    assert namespace.force_all is False


@pytest.mark.parametrize("workflow_command", ["validate", "plan"])
def test_validation_commands_parse_the_exact_inputs(workflow_command: str) -> None:
    namespace = build_parser().parse_args(
        [
            "workflow",
            workflow_command,
            "dataset.yaml",
            "--execution",
            "execution.yaml",
        ]
    )

    assert namespace.command == "workflow"
    assert namespace.workflow_command == workflow_command
    assert namespace.dataset_yaml == "dataset.yaml"
    assert namespace.execution_yaml == "execution.yaml"


def test_status_parses_output_root_and_dataset_id() -> None:
    namespace = build_parser().parse_args(
        ["workflow", "status", "/output", "dataset-1"]
    )

    assert namespace.workflow_command == "status"
    assert namespace.output_root == "/output"
    assert namespace.dataset_id == "dataset-1"


def test_common_options_are_accepted_at_the_command_boundary() -> None:
    namespace = build_parser().parse_args(
        [
            "workflow",
            "plan",
            "dataset.yaml",
            "--execution",
            "execution.yaml",
            "--json",
            "--log-level",
            "debug",
        ]
    )

    assert namespace.json is True
    assert namespace.log_level == "DEBUG"


def test_common_options_are_accepted_before_the_command() -> None:
    namespace = build_parser().parse_args(
        [
            "--json",
            "--log-level",
            "warning",
            "workflow",
            "status",
            "/output",
            "dataset-1",
        ]
    )

    assert namespace.json is True
    assert namespace.log_level == "WARNING"


def test_common_options_are_accepted_between_workflow_and_the_action() -> None:
    namespace = build_parser().parse_args(
        [
            "workflow",
            "--json",
            "--log-level",
            "error",
            "status",
            "/output",
            "dataset-1",
        ]
    )

    assert namespace.json is True
    assert namespace.log_level == "ERROR"


def test_force_all_and_force_round_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(
            [
                "workflow",
                "run",
                "dataset.yaml",
                "--execution",
                "execution.yaml",
                "--force-all",
                "--force-round",
                "2",
            ]
        )

    assert caught.value.code == 2


def test_parser_rejects_abbreviated_options() -> None:
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(
            ["workflow", "plan", "dataset.yaml", "--exec", "execution.yaml"]
        )

    assert caught.value.code == 2


def test_validate_builds_a_plan_without_running_it(
    fake_api: _FakeWorkflowApi,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "--json",
            "workflow",
            "validate",
            "dataset.yaml",
            "--execution",
            "execution.yaml",
        ]
    )

    assert exit_code == 0
    assert fake_api.calls == [
        ("load_dataset_manifest", "dataset.yaml"),
        ("load_execution_profile", "execution.yaml"),
        ("build_workflow_plan", fake_api.manifest, fake_api.execution),
    ]
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "dataset_id": "dataset-1",
        "rounds": [2],
        "scientific_fingerprint": "abc123",
        "valid": True,
    }
    assert (
        captured.out
        == json.dumps(json.loads(captured.out), allow_nan=False, sort_keys=True) + "\n"
    )


def test_plan_prints_scientific_plan_and_resource_request(
    fake_api: _FakeWorkflowApi,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "workflow",
            "plan",
            "dataset.yaml",
            "--execution",
            "execution.yaml",
            "--json",
        ]
    )

    assert exit_code == 0
    assert not any(call[0] == "run_workflow" for call in fake_api.calls)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["scientific_plan"] == {
        "dataset_id": "dataset-1",
        "rounds": [{"round": 2}],
        "schema_version": 1,
    }
    assert payload["resource_request"] == {
        "executor": "sequential",
        "native_threads": {"blas": 1, "itk": 1},
        "paths": {
            "dataset_root": "/dataset",
            "output_root": "/output",
            "worker_local_directory": None,
        },
    }
    assert captured.err == ""


def test_plan_reports_exact_derived_slurm_process_resources(
    fake_api: _FakeWorkflowApi,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_api.execution = SimpleNamespace(
        executor="slurm",
        paths=SimpleNamespace(
            dataset_root=Path("/dataset"),
            output_root=Path("/output"),
            worker_local_directory=Path("/scratch"),
        ),
        native_threads=SimpleNamespace(itk=15, blas=15),
        parallelism=SimpleNamespace(mode="fine_grained"),
        local_dask=None,
        slurm=SimpleNamespace(
            queue="512GB",
            workers=3,
            cores_per_worker=30,
            processes_per_worker=2,
            memory="500GB",
            walltime="72:00:00",
            interface="ib0",
            scheduler_interface="ib0",
            startup_timeout="15m",
            task_retries=0,
            job_name="cyexm-registration",
            job_extra_directives=(),
        ),
    )
    fake_api.plan.execution = fake_api.execution

    exit_code = main(
        [
            "workflow",
            "plan",
            "dataset.yaml",
            "--execution",
            "execution.yaml",
            "--json",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["resource_request"]["slurm"]["derived"] == {
        "cores_per_process": 15,
        "job_cap": 3,
        "job_memory": "466G",
        "max_task_slots": 6,
        "process_memory": "233GiB",
        "process_memory_bytes": 250181844992,
        "processes_per_job": 2,
        "task_slots_per_job": 2,
    }
    assert captured.err == ""


def test_human_plan_includes_the_canonical_plan_and_resource_request(
    fake_api: _FakeWorkflowApi,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "workflow",
            "plan",
            "dataset.yaml",
            "--execution",
            "execution.yaml",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "Planned dataset dataset-1: 1 round(s) using sequential; scientific "
        "fingerprint abc123",
        "Scientific plan: " + fake_api.plan.scientific_json,
        "Resource request: "
        + json.dumps(
            {
                "executor": "sequential",
                "native_threads": {"blas": 1, "itk": 1},
                "paths": {
                    "dataset_root": "/dataset",
                    "output_root": "/output",
                    "worker_local_directory": None,
                },
            },
            allow_nan=False,
            sort_keys=True,
        ),
    ]
    assert captured.err == ""


def test_run_delegates_force_arguments_and_prints_one_json_object(
    fake_api: _FakeWorkflowApi,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "workflow",
            "run",
            "dataset.yaml",
            "--execution",
            "execution.yaml",
            "--force-round",
            "4",
            "--force-round",
            "2",
            "--json",
        ]
    )

    assert exit_code == 0
    assert fake_api.calls[-1] == (
        "run_workflow",
        fake_api.plan,
        frozenset({2, 4}),
        False,
    )
    captured = capsys.readouterr()
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out) == fake_api.result.to_dict()
    assert captured.err == ""


def test_run_delegates_force_all_without_forced_rounds(
    fake_api: _FakeWorkflowApi,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "workflow",
            "run",
            "dataset.yaml",
            "--execution",
            "execution.yaml",
            "--force-all",
            "--json",
        ]
    )

    assert exit_code == 0
    assert fake_api.calls[-1] == (
        "run_workflow",
        fake_api.plan,
        frozenset(),
        True,
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out) == fake_api.result.to_dict()
    assert captured.err == ""


def test_status_uses_only_the_read_only_status_api(
    fake_api: _FakeWorkflowApi,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--json", "workflow", "status", "/output", "dataset-1"])

    assert exit_code == 0
    assert fake_api.calls == [("inspect_workflow_status", "/output", "dataset-1")]
    captured = capsys.readouterr()
    assert json.loads(captured.out) == fake_api.status.to_dict()
    assert captured.err == ""


def test_human_results_go_only_to_stdout(
    fake_api: _FakeWorkflowApi,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["workflow", "status", "/output", "dataset-1", "--log-level", "INFO"]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out == "Workflow dataset-1: incomplete; 0 round(s)\n"
    assert captured.err == ""


def test_main_returns_two_for_argparse_errors_without_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["workflow", "run", "dataset.yaml"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_version_uses_package_metadata_and_returns_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--version"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out == f"cyexm {__version__}\n"
    assert captured.err == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["workflow", "--version"],
        ["workflow", "status", "--version"],
    ],
    ids=("workflow", "workflow-action"),
)
def test_nested_version_uses_package_metadata_and_returns_success(
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    exit_code = main(argv)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out == f"cyexm {__version__}\n"
    assert captured.err == ""


@pytest.mark.parametrize(
    ("error", "expected_exit"),
    [
        (ConfigurationValidationError("bad configuration"), 2),
        (WorkflowPlanError("bad\nplan"), 2),
        (ExecutorStartupError("startup failed"), 3),
        (WorkerPreflightError("worker failed"), 3),
        (WorkflowExecutionError((2,), "/run.json"), 4),
        (KeyboardInterrupt(), 130),
    ],
)
def test_expected_exceptions_map_to_exact_exit_codes_and_one_line_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
    expected_exit: int,
) -> None:
    api = _FakeWorkflowApi()

    def fail(
        plan: object,
        *,
        force_rounds: frozenset[int],
        force_all: bool,
    ) -> _StructuredResult:
        del plan, force_rounds, force_all
        raise error

    api.run_workflow = fail  # type: ignore[method-assign]
    monkeypatch.setattr(cli_commands, "_load_workflow_api", lambda: api)

    exit_code = main(
        [
            "workflow",
            "run",
            "dataset.yaml",
            "--execution",
            "execution.yaml",
        ]
    )

    assert exit_code == expected_exit
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "Traceback" not in captured.err


def test_local_dask_constructor_failure_maps_to_exit_three_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cyexm.execution import dask as dask_execution
    from cyexm.execution.sequential import SequentialExecutor

    api = _FakeWorkflowApi()
    api.execution = SimpleNamespace(
        executor="local_dask",
        paths=api.execution.paths,
        native_threads=api.execution.native_threads,
        local_dask=SimpleNamespace(
            workers=2,
            threads_per_worker=1,
            processes=True,
            memory_limit="2GB",
            startup_timeout="30s",
            task_retries=0,
        ),
        slurm=None,
    )
    api.plan.execution = api.execution
    sequential_calls: list[object] = []

    class FailingLocalDaskExecutor:
        def __init__(self, profile: object) -> None:
            del profile
            raise ExecutorStartupError("injected local Dask constructor failure")

    def run_through_production_factory(
        plan: object,
        *,
        force_rounds: frozenset[int],
        force_all: bool,
    ) -> _StructuredResult:
        del force_rounds, force_all
        workflow_api._default_executor_factory(plan.execution)  # type: ignore[attr-defined]
        raise AssertionError("local executor construction unexpectedly returned")

    api.run_workflow = run_through_production_factory  # type: ignore[method-assign]
    monkeypatch.setattr(cli_commands, "_load_workflow_api", lambda: api)
    monkeypatch.setattr(
        dask_execution,
        "LocalDaskExecutor",
        FailingLocalDaskExecutor,
    )
    monkeypatch.setattr(
        SequentialExecutor,
        "execute",
        lambda _self, tasks: sequential_calls.append(tasks) or (),
    )

    exit_code = main(
        [
            "workflow",
            "run",
            "dataset.yaml",
            "--execution",
            "local.yaml",
        ]
    )

    assert exit_code == 3
    assert sequential_calls == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "cyexm: error: injected local Dask constructor failure\n"


def test_slurm_constructor_failure_maps_to_exit_three_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cyexm.execution import slurm as slurm_execution
    from cyexm.execution.sequential import SequentialExecutor

    api = _FakeWorkflowApi()
    api.execution = SimpleNamespace(
        executor="slurm",
        paths=SimpleNamespace(
            dataset_root=Path("/dataset"),
            output_root=Path("/output"),
            worker_local_directory=Path("/project/cyexm-scratch"),
        ),
        native_threads=api.execution.native_threads,
        local_dask=None,
        slurm=SimpleNamespace(
            queue="512GB",
            workers=2,
            cores_per_worker=72,
            processes_per_worker=1,
            memory="500GB",
            walltime="72:00:00",
            interface="ib0",
            scheduler_interface="ib0",
            startup_timeout="15m",
            task_retries=0,
            job_name="cyexm-registration",
            job_extra_directives=(),
        ),
    )
    api.plan.execution = api.execution
    sequential_calls: list[object] = []

    class FailingSlurmExecutor:
        def __init__(self, profile: object) -> None:
            del profile
            raise ExecutorStartupError("injected SLURM constructor failure")

    def run_through_production_factory(
        plan: object,
        *,
        force_rounds: frozenset[int],
        force_all: bool,
    ) -> _StructuredResult:
        del force_rounds, force_all
        workflow_api._default_executor_factory(plan.execution)  # type: ignore[attr-defined]
        raise AssertionError("SLURM executor construction unexpectedly returned")

    api.run_workflow = run_through_production_factory  # type: ignore[method-assign]
    monkeypatch.setattr(cli_commands, "_load_workflow_api", lambda: api)
    monkeypatch.setattr(slurm_execution, "SlurmExecutor", FailingSlurmExecutor)
    monkeypatch.setattr(
        SequentialExecutor,
        "execute",
        lambda _self, tasks: sequential_calls.append(tasks) or (),
    )

    exit_code = main(
        [
            "workflow",
            "run",
            "dataset.yaml",
            "--execution",
            "slurm.yaml",
        ]
    )

    assert exit_code == 3
    assert sequential_calls == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "cyexm: error: injected SLURM constructor failure\n"


@pytest.mark.parametrize(
    ("error", "expected_exit"),
    [
        (ExecutorStartupError("SLURM worker wait failed"), 3),
        (WorkerPreflightError("SLURM worker preflight failed"), 3),
        (WorkflowExecutionError((2,), "/run.json"), 4),
        (KeyboardInterrupt("SLURM cancellation"), 130),
    ],
    ids=("worker-wait", "worker-preflight", "round-failure", "interruption"),
)
def test_slurm_lifecycle_failures_map_to_stable_cli_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
    expected_exit: int,
) -> None:
    api = _FakeWorkflowApi()

    def fail(
        plan: object,
        *,
        force_rounds: frozenset[int],
        force_all: bool,
    ) -> _StructuredResult:
        del plan, force_rounds, force_all
        raise error

    api.run_workflow = fail  # type: ignore[method-assign]
    monkeypatch.setattr(cli_commands, "_load_workflow_api", lambda: api)

    exit_code = main(
        [
            "workflow",
            "run",
            "dataset.yaml",
            "--execution",
            "slurm.yaml",
        ]
    )

    assert exit_code == expected_exit
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1


def test_run_restores_the_prior_sigterm_handler_after_success(
    fake_api: _FakeWorkflowApi,
) -> None:
    previous = signal.getsignal(signal.SIGTERM)
    installed: list[signal.Handlers] = []

    def observe_handler(
        plan: object,
        *,
        force_rounds: frozenset[int],
        force_all: bool,
    ) -> _StructuredResult:
        del plan, force_rounds, force_all
        installed.append(signal.getsignal(signal.SIGTERM))
        return fake_api.result

    fake_api.run_workflow = observe_handler  # type: ignore[method-assign]

    try:
        assert (
            main(
                [
                    "workflow",
                    "run",
                    "dataset.yaml",
                    "--execution",
                    "execution.yaml",
                ]
            )
            == 0
        )
        assert len(installed) == 1
        assert callable(installed[0])
        assert signal.getsignal(signal.SIGTERM) is previous
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_installed_sigterm_handler_uses_the_keyboard_interrupt_exit_path(
    fake_api: _FakeWorkflowApi,
    capsys: pytest.CaptureFixture[str],
) -> None:
    previous = signal.getsignal(signal.SIGTERM)

    def terminate_during_run(
        plan: object,
        *,
        force_rounds: frozenset[int],
        force_all: bool,
    ) -> _StructuredResult:
        del plan, force_rounds, force_all
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        raise AssertionError("SIGTERM handler returned")

    fake_api.run_workflow = terminate_during_run  # type: ignore[method-assign]

    try:
        exit_code = main(
            [
                "workflow",
                "run",
                "dataset.yaml",
                "--execution",
                "execution.yaml",
            ]
        )
        assert exit_code == 130
        assert signal.getsignal(signal.SIGTERM) is previous
    finally:
        signal.signal(signal.SIGTERM, previous)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "cyexm: interrupted\n"
