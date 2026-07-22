from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyexm.runtime import NATIVE_THREAD_ENV_VARS

pytestmark = pytest.mark.unit


def _write_workflow_inputs(tmp_path: Path) -> tuple[Path, Path]:
    import numpy as np
    import tifffile

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for name in ("fixed.tif", "moving.tif"):
        tifffile.imwrite(
            dataset_root / name,
            np.zeros((2, 3, 4), dtype=np.uint16),
            metadata={"axes": "ZYX"},
            photometric="minisblack",
        )

    manifest_path = tmp_path / "dataset.yaml"
    manifest_path.write_text(
        """\
schema_version: 1
dataset_id: thread-boundary
voxel_spacing_um: {x: 1.0, y: 1.0, z: 1.0}
reference:
  round: 1
  registration_channel: fixed.tif
  channels: [{file: fixed.tif}]
registration:
  crop: false
  linear: {type: TRSAA, accuracy: high}
  nonlinear: {type: SyNOnly, accuracy: high, metric: mattes}
rounds:
  - round: 2
    strategy: whole_image
    registration_channel: moving.tif
    channels:
      - {file: moving.tif, interpolation: linear}
""",
        encoding="utf-8",
    )
    execution_path = tmp_path / "execution.yaml"
    execution_path.write_text(
        f"""\
schema_version: 1
executor: sequential
paths:
  dataset_root: {dataset_root}
  output_root: {tmp_path / "output"}
  worker_local_directory: null
native_threads: {{itk: 2, blas: 3}}
""",
        encoding="utf-8",
    )
    return manifest_path, execution_path


def _fresh_environment(native_value: str | None = None) -> dict[str, str]:
    environ = os.environ.copy()
    for name in NATIVE_THREAD_ENV_VARS:
        if native_value is None:
            environ.pop(name, None)
        else:
            environ[name] = native_value
    return environ


def test_public_api_sequential_run_configures_threads_after_mutation_free_planning(
    tmp_path: Path,
) -> None:
    manifest_path, execution_path = _write_workflow_inputs(tmp_path)
    script = r"""
import json
import os
import sys

from cyexm.runtime import (
    NATIVE_THREAD_ENV_VARS,
    NativeThreadConfig,
    configure_native_threads,
)
from cyexm.workflow import (
    build_workflow_plan,
    load_dataset_manifest,
    load_execution_profile,
    run_workflow,
)
from cyexm.workflow.models import RoundResult


class Executor:
    def execute(self, reference_task, tasks):
        config = NativeThreadConfig(itk_threads=2, blas_threads=3)
        configure_native_threads(config)
        import numpy
        configure_native_threads(config)
        expected = {
            NATIVE_THREAD_ENV_VARS[0]: "2",
            **{name: "3" for name in NATIVE_THREAD_ENV_VARS[1:]},
        }
        actual = {name: os.environ.get(name) for name in NATIVE_THREAD_ENV_VARS}
        assert actual == expected
        results = []
        for task in (reference_task, *tasks):
            is_reference = task is reference_task
            artifacts = (
                (
                    *(
                        ()
                        if task.crop_bounds_path is None
                        else (task.crop_bounds_path,)
                    ),
                    *(channel.output for channel in task.channels),
                )
                if is_reference
                else (
                    task.registration_directory / f"linear_transform_{task.round}.mat",
                    task.registration_directory / f"nonlinear_warp_{task.round}.nii.gz",
                    task.registration_directory
                    / f"nonlinear_registered_{task.round}.tif",
                    *(channel.output for channel in task.channels),
                )
            )
            results.append(
                RoundResult(
                    dataset_id=task.dataset_id,
                    round=task.round,
                    status="completed",
                    started_at="2026-07-20T10:00:00Z",
                    ended_at="2026-07-20T10:00:01Z",
                    duration_seconds=1.0,
                    hostname="fresh-process",
                    pid=os.getpid(),
                    dask_task_key=None,
                    slurm_job_id=None,
                    strategy=(
                        "reference-preparation" if is_reference else task.strategy
                    ),
                    native_threads=(("itk", 2), ("blas", 3)),
                    artifact_paths=tuple(str(path) for path in artifacts),
                    reused_artifacts=(),
                    log_path=str(task.log_path),
                    failure=None,
                )
            )
        return tuple(results)


manifest = load_dataset_manifest(sys.argv[1])
execution = load_execution_profile(sys.argv[2])
plan = build_workflow_plan(manifest, execution)
result = run_workflow(plan, _executor_factory=lambda _profile: Executor())
print(json.dumps({"status": result.status, "numpy_loaded": "numpy" in sys.modules}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(manifest_path), str(execution_path)],
        check=False,
        capture_output=True,
        text=True,
        env=_fresh_environment(),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout == '{"status": "completed", "numpy_loaded": true}\n'


@pytest.mark.parametrize(
    "native_value",
    [pytest.param(None, id="absent"), pytest.param("7", id="preexisting")],
)
def test_cli_sequential_run_applies_explicit_threads_after_planning(
    tmp_path: Path,
    native_value: str | None,
) -> None:
    manifest_path, execution_path = _write_workflow_inputs(tmp_path)
    script = r"""
import os
import sys

import cyexm.workflow.reference_task as reference_task
import cyexm.workflow.round_task as round_task
from cyexm.cli import main
from cyexm.runtime import (
    NATIVE_THREAD_ENV_VARS,
    NativeThreadConfig,
    configure_native_threads,
)
from cyexm.workflow.models import RoundResult


def run(task):
    config = NativeThreadConfig(
        itk_threads=task.native_threads.itk,
        blas_threads=task.native_threads.blas,
    )
    configure_native_threads(config)
    import numpy
    configure_native_threads(config)
    expected = {
        NATIVE_THREAD_ENV_VARS[0]: "2",
        **{name: "3" for name in NATIVE_THREAD_ENV_VARS[1:]},
    }
    assert {name: os.environ.get(name) for name in NATIVE_THREAD_ENV_VARS} == expected
    is_reference = not hasattr(task, "strategy")
    artifacts = (
        (
            *(() if task.crop_bounds_path is None else (task.crop_bounds_path,)),
            *(channel.output for channel in task.channels),
        )
        if is_reference
        else (
            task.registration_directory / f"linear_transform_{task.round}.mat",
            task.registration_directory / f"nonlinear_warp_{task.round}.nii.gz",
            task.registration_directory / f"nonlinear_registered_{task.round}.tif",
            *(channel.output for channel in task.channels),
        )
    )
    return RoundResult(
        dataset_id=task.dataset_id,
        round=task.round,
        status="completed",
        started_at="2026-07-20T10:00:00Z",
        ended_at="2026-07-20T10:00:01Z",
        duration_seconds=1.0,
        hostname="fresh-process",
        pid=os.getpid(),
        dask_task_key=None,
        slurm_job_id=None,
        strategy="reference-preparation" if is_reference else task.strategy,
        native_threads=(("itk", 2), ("blas", 3)),
        artifact_paths=tuple(str(path) for path in artifacts),
        reused_artifacts=(),
        log_path=str(task.log_path),
        failure=None,
    )


reference_task.run_reference_task = run
round_task.run_round_task = run
raise SystemExit(
    main(
        [
            "--json",
            "workflow",
            "run",
            sys.argv[1],
            "--execution",
            sys.argv[2],
        ]
    )
)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(manifest_path), str(execution_path)],
        check=False,
        capture_output=True,
        text=True,
        env=_fresh_environment(native_value),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["status"] == "completed"


@pytest.mark.parametrize("command", ["validate", "plan"])
def test_read_only_cli_planning_does_not_load_science_or_mutate_threads(
    tmp_path: Path,
    command: str,
) -> None:
    manifest_path, execution_path = _write_workflow_inputs(tmp_path)
    script = r"""
import os
import sys

from cyexm.cli import main
from cyexm.runtime import NATIVE_THREAD_ENV_VARS

before = {name: os.environ.get(name) for name in NATIVE_THREAD_ENV_VARS}
assert "numpy" not in sys.modules
assert "tifffile" not in sys.modules
exit_code = main(
    [
        "--json",
        "workflow",
        sys.argv[1],
        sys.argv[2],
        "--execution",
        sys.argv[3],
    ]
)
assert exit_code == 0
assert {name: os.environ.get(name) for name in NATIVE_THREAD_ENV_VARS} == before
assert "numpy" not in sys.modules
assert "tifffile" not in sys.modules
"""

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            command,
            str(manifest_path),
            str(execution_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_fresh_environment(),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout)
    assert not (tmp_path / "output").exists()


def test_public_api_rejects_a_late_conflicting_sequential_thread_change(
    tmp_path: Path,
) -> None:
    manifest_path, execution_path = _write_workflow_inputs(tmp_path)
    script = r"""
import os
import sys

import numpy

from cyexm.errors import ThreadConfigurationError
from cyexm.runtime import NATIVE_THREAD_ENV_VARS
from cyexm.workflow import (
    build_workflow_plan,
    load_dataset_manifest,
    load_execution_profile,
    run_workflow,
)

manifest = load_dataset_manifest(sys.argv[1])
execution = load_execution_profile(sys.argv[2])
plan = build_workflow_plan(manifest, execution)
try:
    run_workflow(
        plan,
        _executor_factory=lambda _profile: (_ for _ in ()).throw(
            AssertionError("executor constructed after a late thread conflict")
        ),
    )
except ThreadConfigurationError as error:
    assert "already loaded" in str(error)
else:
    raise AssertionError("late conflicting thread change was accepted")
assert {name: os.environ.get(name) for name in NATIVE_THREAD_ENV_VARS} == {
    name: "1" for name in NATIVE_THREAD_ENV_VARS
}
assert not execution.paths.output_root.exists()
print("rejected")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(manifest_path), str(execution_path)],
        check=False,
        capture_output=True,
        text=True,
        env=_fresh_environment("1"),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout == "rejected\n"


def test_cli_maps_a_late_sequential_thread_conflict_to_exit_two(
    tmp_path: Path,
) -> None:
    manifest_path, execution_path = _write_workflow_inputs(tmp_path)
    script = r"""
import sys

import numpy

from cyexm.cli import main

raise SystemExit(
    main(
        [
            "--json",
            "workflow",
            "run",
            sys.argv[1],
            "--execution",
            sys.argv[2],
        ]
    )
)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(manifest_path), str(execution_path)],
        check=False,
        capture_output=True,
        text=True,
        env=_fresh_environment("1"),
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == (
        "cyexm: error: Cannot change ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS "
        "after scientific runtime already loaded: numpy\n"
    )
    assert "Traceback" not in completed.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("executor", ["local_dask", "slurm"])
def test_distributed_coordinators_do_not_apply_task_native_threads(
    monkeypatch: pytest.MonkeyPatch,
    executor: str,
) -> None:
    import cyexm.runtime as runtime
    import cyexm.workflow.api as workflow_api

    calls: list[object] = []
    monkeypatch.setattr(
        runtime,
        "configure_native_threads",
        lambda config: calls.append(config),
    )
    profile = SimpleNamespace(
        executor=executor,
        native_threads=SimpleNamespace(itk=2, blas=3),
    )

    workflow_api._configure_sequential_native_threads(profile)  # type: ignore[arg-type]

    assert calls == []
