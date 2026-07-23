from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import cast

import pytest

from cyexm.errors import WorkerPreflightError
from cyexm.execution.preflight import (
    WorkerProbe,
    probe_worker,
    validate_slurm_worker_identities,
    validate_worker_probes,
)
from cyexm.runtime import NATIVE_THREAD_ENV_VARS

pytestmark = pytest.mark.unit


class _MemoryManager:
    def __init__(self, memory_limit: object) -> None:
        self.memory_limit = memory_limit


class _FakeWorker:
    def __init__(self, *, address: object, memory_limit: object) -> None:
        self.address = address
        self.memory_manager = _MemoryManager(memory_limit)


def _prepare_threads(monkeypatch: pytest.MonkeyPatch, value: str = "1") -> None:
    for name in NATIVE_THREAD_ENV_VARS:
        monkeypatch.setenv(name, value)


def _probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    worker: object | None = None,
    dask_worker: object | None = None,
    version: Callable[[str], str] | None = None,
    expected_versions: Mapping[str, str] | None = None,
    expected_threads: Mapping[str, str] | None = None,
    prepare_threads: bool = True,
) -> dict[str, object]:
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    dataset.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if prepare_threads:
        _prepare_threads(monkeypatch)
    selected_worker = (
        _FakeWorker(
            address="tcp://10.0.0.1:1234",
            memory_limit=500_000_000_000,
        )
        if worker is None and dask_worker is None
        else worker
    )
    kwargs: dict[str, object] = {
        "dataset_root": str(dataset),
        "output_root": str(output),
        "worker_local_directory": str(tmp_path / "scratch"),
        "expected_versions": (
            {"cy-exm": "0.1.0"} if expected_versions is None else expected_versions
        ),
        "expected_threads": (
            dict.fromkeys(NATIVE_THREAD_ENV_VARS, "1")
            if expected_threads is None
            else expected_threads
        ),
        "_version": version or (lambda _name: "0.1.0"),
    }
    if selected_worker is not None:
        kwargs["_worker"] = selected_worker
    if dask_worker is not None:
        kwargs["dask_worker"] = dask_worker
    return probe_worker(**kwargs)  # type: ignore[arg-type]


def _errors(probe: Mapping[str, object]) -> tuple[str, ...]:
    value = probe["errors"]
    assert type(value) is list
    assert all(type(item) is str for item in value)
    return tuple(value)  # type: ignore[arg-type]


def test_probe_records_worker_identity_paths_threads_and_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    scratch = tmp_path / "scratch"
    dataset.mkdir()
    output.mkdir()
    _prepare_threads(monkeypatch)

    probe = probe_worker(
        dataset_root=str(dataset),
        output_root=str(output),
        worker_local_directory=str(scratch),
        expected_versions={"cy-exm": "0.1.0"},
        expected_threads=dict.fromkeys(NATIVE_THREAD_ENV_VARS, "1"),
        _version=lambda _name: "0.1.0",
        _worker=_FakeWorker(
            address="tcp://10.0.0.1:1234",
            memory_limit=500_000_000_000,
        ),
    )

    assert probe["ok"] is True
    assert probe["worker_address"] == "tcp://10.0.0.1:1234"
    assert probe["network_host"] == "10.0.0.1"
    assert probe["memory_limit_bytes"] == 500_000_000_000
    assert probe["dataset_visible"] is True
    assert probe["output_writable"] is True
    assert probe["scratch_usable"] is True
    assert probe["package_versions"] == [["cy-exm", "0.1.0"]]
    assert dict(cast(list[list[str]], probe["native_threads"])) == dict.fromkeys(
        NATIVE_THREAD_ENV_VARS,
        "1",
    )
    assert probe["errors"] == []
    assert not scratch.joinpath("cyexm-preflight").exists()
    assert not tuple(scratch.glob("cyexm-preflight-*"))
    assert not tuple(output.glob(".cyexm-preflight-*"))


def test_probe_accepts_remote_dask_worker_keyword_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _FakeWorker(
        address="tcp://worker.example.org:8786",
        memory_limit=10_000,
    )

    probe = _probe(tmp_path, monkeypatch, dask_worker=worker)

    assert probe["ok"] is True
    assert probe["worker_address"] == "tcp://worker.example.org:8786"
    assert probe["network_host"] == "worker.example.org"


def test_slurm_identity_validation_rejects_missing_and_duplicate_job_identity() -> None:
    base = WorkerProbe.from_dict(
        {
            "ok": True,
            "worker_address": "tcp://node-a:1234",
            "hostname": "node-a",
            "pid": 12,
            "slurm_job_id": "9001",
            "package_versions": [],
            "native_threads": [],
            "dataset_visible": True,
            "output_writable": True,
            "scratch_usable": True,
            "memory_limit_bytes": 1,
            "network_host": "node-a",
            "errors": [],
        }
    )

    validate_slurm_worker_identities((base,))
    with pytest.raises(WorkerPreflightError, match="SLURM_JOB_ID is required"):
        validate_slurm_worker_identities((replace(base, slurm_job_id=None),))
    with pytest.raises(WorkerPreflightError, match="duplicate SLURM_JOB_ID"):
        validate_slurm_worker_identities(
            (
                base,
                replace(
                    base,
                    worker_address="tcp://node-b:1234",
                    hostname="node-b",
                    network_host="node-b",
                ),
            )
        )


def test_probe_reports_package_mismatches_and_missing_distributions_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def version(name: str) -> str:
        if name == "missing-package":
            raise PackageNotFoundError(name)
        return "2.0.0"

    probe = _probe(
        tmp_path,
        monkeypatch,
        version=version,
        expected_versions={"cy-exm": "1.0.0", "missing-package": "3.0.0"},
    )

    assert probe["ok"] is False
    assert probe["package_versions"] == [
        ["cy-exm", "2.0.0"],
        ["missing-package", "<missing>"],
    ]
    errors = "\n".join(_errors(probe))
    assert "cy-exm" in errors and "1.0.0" in errors and "2.0.0" in errors
    assert "missing-package" in errors and "not installed" in errors


def test_probe_actually_reads_dataset_directory_instead_of_trusting_mode_bits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_scandir = os.scandir

    def scandir(
        path: str | os.PathLike[str],
    ) -> Iterator[os.DirEntry[str]]:
        if Path(path) == tmp_path / "dataset":
            raise PermissionError("injected read denial")
        return cast(Iterator[os.DirEntry[str]], original_scandir(path))

    monkeypatch.setattr(os, "scandir", scandir)

    probe = _probe(tmp_path, monkeypatch)

    assert probe["dataset_visible"] is False
    assert any("dataset" in error and "read" in error for error in _errors(probe))


def test_probe_actually_writes_output_instead_of_trusting_mode_bits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_named_temporary_file = tempfile.NamedTemporaryFile

    def named_temporary_file(*args: object, **kwargs: object) -> object:
        if Path(str(kwargs["dir"])) == tmp_path / "output":
            raise PermissionError("injected output write denial")
        return original_named_temporary_file(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", named_temporary_file)

    probe = _probe(tmp_path, monkeypatch)

    assert probe["output_writable"] is False
    assert any("output" in error and "write" in error for error in _errors(probe))


def test_probe_reports_output_delete_failure_and_preserves_write_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_unlink = Path.unlink

    def unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.parent == tmp_path / "output" and path.name.startswith(
            ".cyexm-preflight-"
        ):
            raise PermissionError("injected output delete denial")
        original_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "unlink", unlink)

    probe = _probe(tmp_path, monkeypatch)

    assert probe["output_writable"] is False
    assert any("output" in error and "delete" in error for error in _errors(probe))


def test_probe_reports_scratch_create_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tempfile,
        "mkdtemp",
        lambda **_kwargs: (_ for _ in ()).throw(
            PermissionError("injected scratch create denial")
        ),
    )

    probe = _probe(tmp_path, monkeypatch)

    assert probe["scratch_usable"] is False
    assert any("scratch" in error and "create" in error for error in _errors(probe))


def test_probe_reports_scratch_write_failure_and_still_removes_unique_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_named_temporary_file = tempfile.NamedTemporaryFile

    def named_temporary_file(*args: object, **kwargs: object) -> object:
        directory = Path(str(kwargs["dir"]))
        if directory.name.startswith("cyexm-preflight-"):
            raise PermissionError("injected scratch write denial")
        return original_named_temporary_file(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", named_temporary_file)

    probe = _probe(tmp_path, monkeypatch)

    assert probe["scratch_usable"] is False
    assert any("scratch" in error and "write" in error for error in _errors(probe))
    assert not tuple((tmp_path / "scratch").glob("cyexm-preflight-*"))


def test_probe_reports_scratch_delete_failure_without_removing_existing_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    existing = scratch / "cyexm-preflight"
    existing.mkdir(parents=True)
    sentinel = existing / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    original_rmtree = shutil.rmtree

    def rmtree(path: str | os.PathLike[str], *args: object, **kwargs: object) -> None:
        candidate = Path(path)
        if candidate.name.startswith("cyexm-preflight-"):
            raise PermissionError("injected scratch delete denial")
        original_rmtree(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(shutil, "rmtree", rmtree)

    probe = _probe(tmp_path, monkeypatch)

    assert probe["scratch_usable"] is False
    assert any("scratch" in error and "delete" in error for error in _errors(probe))
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_probe_preserves_primary_scratch_failure_when_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_named_temporary_file = tempfile.NamedTemporaryFile

    def named_temporary_file(*args: object, **kwargs: object) -> object:
        directory = Path(str(kwargs["dir"]))
        if directory.name.startswith("cyexm-preflight-"):
            raise OSError("primary scratch write failure")
        return original_named_temporary_file(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", named_temporary_file)
    monkeypatch.setattr(
        shutil,
        "rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("secondary cleanup failure")),
    )

    probe = _probe(tmp_path, monkeypatch)

    errors = "\n".join(_errors(probe))
    assert "primary scratch write failure" in errors
    assert "secondary cleanup failure" in errors


def test_probe_reports_every_missing_or_mismatched_native_thread_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_threads(monkeypatch, "1")
    missing = NATIVE_THREAD_ENV_VARS[0]
    mismatched = NATIVE_THREAD_ENV_VARS[1]
    monkeypatch.delenv(missing)
    monkeypatch.setenv(mismatched, "2")

    probe = _probe(tmp_path, monkeypatch, prepare_threads=False)

    threads = dict(cast(list[list[str]], probe["native_threads"]))
    assert threads[missing] == "<missing>"
    assert threads[mismatched] == "2"
    errors = "\n".join(_errors(probe))
    assert missing in errors and "missing" in errors
    assert mismatched in errors and "expected '1'" in errors and "found '2'" in errors


@pytest.mark.parametrize(
    ("worker", "message"),
    [
        (_FakeWorker(address="", memory_limit=100), "worker address"),
        (_FakeWorker(address="tcp://worker:1234", memory_limit=0), "memory limit"),
        (_FakeWorker(address="tcp://worker:1234", memory_limit=True), "memory limit"),
    ],
)
def test_probe_reports_unusable_worker_identity_or_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker: _FakeWorker,
    message: str,
) -> None:
    probe = _probe(tmp_path, monkeypatch, worker=worker)

    assert probe["ok"] is False
    assert any(message in error for error in _errors(probe))


def test_probe_reports_missing_hostname(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "")

    probe = _probe(tmp_path, monkeypatch)

    assert probe["hostname"] == ""
    assert any("hostname" in error for error in _errors(probe))


def test_probe_records_absent_and_present_slurm_job_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    absent = _probe(tmp_path / "absent", monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    present = _probe(tmp_path / "present", monkeypatch)

    assert absent["slurm_job_id"] is None
    assert present["slurm_job_id"] == "123456"


def test_probe_collects_independent_identity_version_thread_and_path_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in NATIVE_THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "")
    monkeypatch.setattr(socket, "gethostname", lambda: "")
    scratch_file = tmp_path / "scratch-file"
    scratch_file.write_text("not a directory", encoding="utf-8")

    probe = probe_worker(
        dataset_root=str(tmp_path / "missing-dataset"),
        output_root=str(tmp_path / "missing-output"),
        worker_local_directory=str(scratch_file),
        expected_versions={"cy-exm": "expected"},
        expected_threads=dict.fromkeys(NATIVE_THREAD_ENV_VARS, "1"),
        _version=lambda _name: "wrong",
        _worker=_FakeWorker(address="", memory_limit=0),
    )

    assert probe["ok"] is False
    errors = "\n".join(_errors(probe))
    for expected in (
        "worker address",
        "memory limit",
        "hostname",
        "SLURM_JOB_ID",
        "version mismatch",
        NATIVE_THREAD_ENV_VARS[0],
        "dataset root",
        "output root",
        "scratch probe",
    ):
        assert expected in errors


def _valid_probe(*, address: str = "tcp://worker-a:1234") -> WorkerProbe:
    return WorkerProbe(
        ok=True,
        worker_address=address,
        hostname="worker-a",
        pid=1234,
        slurm_job_id="98765",
        package_versions=(("cy-exm", "0.1.0"),),
        native_threads=tuple(
            sorted(dict.fromkeys(NATIVE_THREAD_ENV_VARS, "1").items())
        ),
        dataset_visible=True,
        output_writable=True,
        scratch_usable=True,
        memory_limit_bytes=500_000_000_000,
        network_host="worker-a",
        errors=(),
    )


def test_worker_probe_strictly_round_trips_json_primitives() -> None:
    original = _valid_probe()

    payload = original.to_dict()
    decoded = json.loads(json.dumps(payload))

    assert WorkerProbe.from_dict(decoded) == original
    assert type(payload) is dict
    assert type(payload["package_versions"]) is list
    assert type(payload["native_threads"]) is list
    assert type(payload["errors"]) is list


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ok", 1),
        ("pid", True),
        ("memory_limit_bytes", True),
        ("worker_address", 123),
        ("slurm_job_id", 123),
        ("package_versions", (("cy-exm", "0.1.0"),)),
        ("native_threads", (("OMP_NUM_THREADS", "1"),)),
        ("errors", ()),
    ],
)
def test_worker_probe_rejects_nonexact_json_types(field: str, value: object) -> None:
    payload = _valid_probe().to_dict()
    payload[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        WorkerProbe.from_dict(payload)


def test_worker_probe_rejects_unknown_keys_and_duplicate_pair_names() -> None:
    extra = _valid_probe().to_dict()
    extra["unexpected"] = "value"
    duplicate = _valid_probe().to_dict()
    duplicate["package_versions"] = [
        ["cy-exm", "0.1.0"],
        ["cy-exm", "0.1.0"],
    ]

    with pytest.raises(ValueError, match="keys"):
        WorkerProbe.from_dict(extra)
    with pytest.raises(ValueError, match=r"duplicate.*cy-exm"):
        WorkerProbe.from_dict(duplicate)


def test_worker_probe_canonicalizes_pair_order_deterministically() -> None:
    payload = _valid_probe().to_dict()
    payload["package_versions"] = [["z", "1"], ["a", "2"]]

    decoded = WorkerProbe.from_dict(payload)

    assert decoded.package_versions == (("a", "2"), ("z", "1"))
    assert decoded.to_dict()["package_versions"] == [["a", "2"], ["z", "1"]]


def test_worker_probe_requires_ok_to_match_empty_error_state() -> None:
    payload = _valid_probe().to_dict()
    payload["errors"] = ["injected failure"]

    with pytest.raises(ValueError, match="ok"):
        WorkerProbe.from_dict(payload)


def test_validate_worker_probes_returns_deterministic_worker_order() -> None:
    worker_b = _valid_probe(address="tcp://worker-b:1234")
    worker_a = _valid_probe(address="tcp://worker-a:1234")

    decoded = validate_worker_probes(
        {
            "tcp://worker-b:1234": worker_b.to_dict(),
            "tcp://worker-a:1234": worker_a.to_dict(),
        }
    )

    assert tuple(probe.worker_address for probe in decoded) == (
        "tcp://worker-a:1234",
        "tcp://worker-b:1234",
    )


def test_validate_worker_probes_aggregates_all_worker_failures_with_identity() -> None:
    worker_a = WorkerProbe(
        ok=False,
        worker_address="tcp://10.0.0.1:1234",
        hostname="node-a",
        pid=100,
        slurm_job_id="111",
        package_versions=(("cy-exm", "wrong"),),
        native_threads=(),
        dataset_visible=False,
        output_writable=True,
        scratch_usable=True,
        memory_limit_bytes=500,
        network_host="10.0.0.1",
        errors=("package mismatch", "dataset unreadable"),
    )
    worker_b = WorkerProbe(
        ok=False,
        worker_address="tcp://10.0.0.2:1234",
        hostname="node-b",
        pid=200,
        slurm_job_id=None,
        package_versions=(),
        native_threads=(),
        dataset_visible=True,
        output_writable=False,
        scratch_usable=False,
        memory_limit_bytes=0,
        network_host="10.0.0.2",
        errors=("output unwritable", "scratch unusable", "memory limit unusable"),
    )

    with pytest.raises(WorkerPreflightError) as caught:
        validate_worker_probes(
            {
                "scheduler-address-a": worker_a.to_dict(),
                "scheduler-address-b": worker_b.to_dict(),
            }
        )

    message = str(caught.value)
    for expected in (
        "scheduler-address-a",
        "tcp://10.0.0.1:1234",
        "node-a",
        "111",
        "package mismatch",
        "dataset unreadable",
        "scheduler-address-b",
        "tcp://10.0.0.2:1234",
        "node-b",
        "SLURM_JOB_ID=<none>",
        "output unwritable",
        "scratch unusable",
        "memory limit unusable",
    ):
        assert expected in message


def test_validate_worker_probes_aggregates_malformed_and_reported_failures() -> None:
    failed = WorkerProbe(
        ok=False,
        worker_address="tcp://worker-b:1234",
        hostname="node-b",
        pid=200,
        slurm_job_id="222",
        package_versions=(),
        native_threads=(),
        dataset_visible=False,
        output_writable=False,
        scratch_usable=False,
        memory_limit_bytes=0,
        network_host="worker-b",
        errors=("reported failure",),
    )

    with pytest.raises(WorkerPreflightError) as caught:
        validate_worker_probes(
            {
                "scheduler-address-a": {"ok": True},
                "scheduler-address-b": failed.to_dict(),
            }
        )

    message = str(caught.value)
    assert "scheduler-address-a" in message
    assert "invalid probe payload" in message
    assert "scheduler-address-b" in message
    assert "reported failure" in message
