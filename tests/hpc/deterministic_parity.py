"""Fresh-process deterministic SLURM parity coordinator for Phase 2E."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

_SEED = "20260722"
_ITK_THREAD_ENV_VAR = "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"
_BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded seeded one-thread sequential/SLURM parity."
    )
    parser.add_argument("--execution", required=True, type=Path)
    parser.add_argument("--owned-root", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    return parser


def _write_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def _require_pristine_environment(environ: Mapping[str, str]) -> None:
    expected = {
        _ITK_THREAD_ENV_VAR: "1",
        **{name: "1" for name in _BLAS_THREAD_ENV_VARS},
        "ANTS_RANDOM_SEED": _SEED,
    }
    mismatches = [
        f"{name}={environ.get(name)!r}, expected {value!r}"
        for name, value in expected.items()
        if environ.get(name) != value
    ]
    if mismatches:
        _fail(
            "deterministic coordinator environment mismatch: " + "; ".join(mismatches)
        )


def _require_owned_root(root: Path) -> dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        _fail(f"owned root must be an existing real directory: {root}")
    roots = {
        "inputs": root / "deterministic-inputs",
        "parallel": root / "deterministic-parallel",
        "sequential": root / "deterministic-sequential",
    }
    for role, path in roots.items():
        if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            _fail(
                f"deterministic {role} root must be an existing empty real directory: "
                f"{path}"
            )
    return roots


def _require_one_thread_results(result: object, role: str) -> None:
    rounds = getattr(result, "rounds", None)
    if not isinstance(rounds, tuple) or not rounds:
        _fail(f"deterministic {role} result has no round evidence")
    for round_result in rounds:
        values = getattr(round_result, "native_threads", None)
        if not isinstance(values, tuple) or not values:
            _fail(f"deterministic {role} round has no native-thread evidence")
        if any(value != 1 for _name, value in values):
            _fail(f"deterministic {role} round did not use one native thread")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence_path = args.evidence
    evidence_writable = not evidence_path.exists() and evidence_path.parent.is_dir()
    failure: BaseException | None = None
    final_queue: dict[str, object] | None = None
    collect_queue_snapshot: Callable[..., dict[str, object]] | None = None
    deterministic_job_name: str | None = None
    user: str | None = None
    try:
        if not evidence_writable:
            _fail(
                "evidence must be a new file below an existing directory: "
                f"{evidence_path}"
            )
        _require_pristine_environment(os.environ)

        # Scientific and project imports intentionally occur only after the fresh
        # coordinator proves its native-thread environment.
        from cyexm.config import load_execution_profile
        from cyexm.workflow.api import run_workflow
        from cyexm.workflow.planning import build_workflow_plan
        from hpc.environment import collect_queue_snapshot as collect_queue
        from hpc.workflow_acceptance import (
            deterministic_acceptance_manifest,
            deterministic_sequential_profile,
            deterministic_slurm_profile,
            execute_owned_slurm_workflow,
            run_seeded_workflow_task,
            scientific_parity_evidence,
            write_deterministic_inputs,
        )

        roots = _require_owned_root(args.owned_root)
        approved = load_execution_profile(args.execution)
        parallel_profile = deterministic_slurm_profile(approved, roots["parallel"])
        sequential_profile = deterministic_sequential_profile(
            approved, roots["sequential"]
        )
        parallel_profile = replace(
            parallel_profile,
            paths=replace(parallel_profile.paths, dataset_root=roots["inputs"]),
        )
        sequential_profile = replace(
            sequential_profile,
            paths=replace(sequential_profile.paths, dataset_root=roots["inputs"]),
        )
        input_evidence = write_deterministic_inputs(roots["inputs"])
        manifest = deterministic_acceptance_manifest(roots["inputs"])
        parallel_plan = build_workflow_plan(manifest, parallel_profile)
        sequential_plan = build_workflow_plan(manifest, sequential_profile)
        if (
            parallel_plan.scientific_fingerprint
            != sequential_plan.scientific_fingerprint
        ):
            _fail("deterministic plans do not share a scientific fingerprint")
        if parallel_profile.slurm is None:
            _fail("deterministic SLURM profile lost its scheduler specification")

        user = getpass.getuser()
        collect_queue_snapshot = collect_queue
        deterministic_job_name = parallel_profile.slurm.job_name
        parallel = execute_owned_slurm_workflow(
            parallel_plan,
            runner=run_seeded_workflow_task,
            evidence_dir=evidence_path.parent,
            operation="deterministic-parallel",
            user=user,
        )
        sequential = run_workflow(sequential_plan)
        if not parallel.succeeded or not sequential.succeeded:
            _fail("deterministic sequential or SLURM workflow did not succeed")
        _require_one_thread_results(parallel, "SLURM")
        _require_one_thread_results(sequential, "sequential")
        parity = scientific_parity_evidence(sequential_plan, parallel_plan)
        final_queue = collect_queue(deterministic_job_name, user=user)
        if final_queue.get("jobs") != []:
            _fail("deterministic named SLURM jobs remain after strict parity")

        native_threads = {
            _ITK_THREAD_ENV_VAR: 1,
            **{name: 1 for name in _BLAS_THREAD_ENV_VARS},
        }
        _write_json(
            evidence_path,
            {
                "schema_version": 1,
                "status": "completed",
                "seed": _SEED,
                "native_threads": native_threads,
                "inputs": input_evidence,
                "parallel": parallel.to_dict(),
                "sequential": sequential.to_dict(),
                "strict_parity": parity,
                "final_queue": final_queue,
            },
        )
        return 0
    except BaseException as error:
        failure = error
        if (
            final_queue is None
            and collect_queue_snapshot is not None
            and deterministic_job_name is not None
            and user is not None
        ):
            try:
                final_queue = collect_queue_snapshot(
                    deterministic_job_name,
                    user=user,
                )
            except BaseException as queue_error:
                final_queue = {
                    "jobs": None,
                    "cleanup_error": {
                        "exception_type": type(queue_error).__name__,
                        "message": str(queue_error),
                    },
                }
        if evidence_writable:
            try:
                _write_json(
                    evidence_path,
                    {
                        "schema_version": 1,
                        "status": "failed",
                        "seed": _SEED,
                        "error": {
                            "exception_type": type(error).__name__,
                            "message": str(error),
                        },
                        "final_queue": final_queue,
                    },
                )
            except BaseException as evidence_error:
                print(
                    f"failed to write deterministic failure evidence: {evidence_error}",
                    file=sys.stderr,
                )
        print(f"deterministic parity failed: {error}", file=sys.stderr)
    if failure is not None:
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
