from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from hpc import fixed_affine_chunk_characterization as characterization
from hpc.conftest import FixedAffineCharacterizationContext
from hpc.environment import HpcAcceptanceContext, write_evidence_json
from hpc.fixed_affine_chunk_characterization import (
    build_condition,
    build_shared_preparation_tasks,
    execute_chunk_condition,
    materialize_shared_prepared_state,
    validate_fixed_affine_source,
)

pytestmark = pytest.mark.hpc


@contextmanager
def _characterization_module_by_value() -> Iterator[None]:
    import cloudpickle

    module = sys.modules[characterization.__name__]
    registry = getattr(cloudpickle.cloudpickle, "_PICKLE_BY_VALUE_MODULES", None)
    if not isinstance(registry, set):
        raise RuntimeError("cloudpickle registration state is unavailable")
    was_registered = module.__name__ in registry
    if not was_registered:
        cloudpickle.register_pickle_by_value(module)
    try:
        yield
    finally:
        if not was_registered and module.__name__ in registry:
            cloudpickle.unregister_pickle_by_value(module)


def test_fixed_affine_chunk_size_characterization(
    hpc_context: HpcAcceptanceContext,
    fixed_affine_characterization_context: FixedAffineCharacterizationContext,
) -> None:
    """Run four sequential nonlinear-only conditions from one accepted affine."""
    characterization_context = fixed_affine_characterization_context
    source = characterization_context.source
    spec = characterization_context.spec
    profile = hpc_context.profile
    evidence_dir = hpc_context.evidence_dir

    conditions = tuple(
        build_condition(
            label=label,
            chunk_size_zyx=spec.chunk_sizes_zyx[label],
            manifest=hpc_context.manifest,
            profile=profile,
            source=source,
        )
        for label in spec.condition_order
    )
    assert tuple(item.label for item in conditions) == ("1125", "512", "256", "128")
    assert all(not item.condition_root.exists() for item in conditions)
    assert not conditions[0].shared_root.exists()

    shared_tasks = build_shared_preparation_tasks(conditions[0])
    source_by_round = {item.round: item for item in source.rounds}
    shared_rounds = tuple(
        materialize_shared_prepared_state(task, source_by_round[task.round])
        for task in shared_tasks
    )
    assert validate_fixed_affine_source(source.workflow_root) == source
    write_evidence_json(
        evidence_dir / "shared-prepared-summary.json",
        {
            "schema_version": 1,
            "source_workflow_root": str(source.workflow_root),
            "shared_root": str(conditions[0].shared_root),
            "rounds": [
                {
                    "round": item.task.round,
                    "result": item.result.to_dict(),
                }
                for item in shared_rounds
            ],
        },
    )

    summaries: list[dict[str, object]] = []
    with _characterization_module_by_value():
        for condition in conditions:
            summaries.append(
                execute_chunk_condition(
                    condition=condition,
                    source=source,
                    shared_rounds=shared_rounds,
                    evidence_dir=evidence_dir,
                    git_commit=hpc_context.git_commit,
                    package_versions=hpc_context.package_versions,
                )
            )
            write_evidence_json(
                evidence_dir / "experiment-progress.json",
                {
                    "schema_version": 1,
                    "condition_order": list(spec.condition_order),
                    "completed_conditions": [
                        str(summary["condition"]) for summary in summaries
                    ],
                    "summaries": summaries,
                },
            )

    write_evidence_json(
        evidence_dir / "experiment-summary.json",
        {
            "schema_version": 1,
            "characterization_config": str(characterization_context.path),
            "dataset_config": str(hpc_context.dataset_path),
            "execution_config": str(hpc_context.execution_path),
            "condition_order": list(spec.condition_order),
            "summaries": summaries,
        },
    )
