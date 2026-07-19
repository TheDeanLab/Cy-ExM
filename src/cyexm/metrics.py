from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cyexm.errors import RegistrationStageError
from cyexm.io.atomic import atomic_write
from cyexm.io.images import external_to_ants_array

_METRIC_TYPES = ("Correlation", "MattesMutualInformation")


def _from_external_array(ants: Any, array: Any, reference: Any | None = None) -> Any:
    kwargs: dict[str, object] = {}
    if reference is not None:
        kwargs = {
            "origin": reference.origin,
            "spacing": reference.spacing,
            "direction": reference.direction,
        }
    return ants.from_numpy(external_to_ants_array(array), **kwargs)


def _metrics_text(metrics: Mapping[str, object]) -> str:
    return "".join(f"{name}: {metrics[name]}\n" for name in sorted(metrics))


def calculate_metrics(
    fixed: Any,
    moving: Any,
    fixed_mask: Any | None = None,
    moving_mask: Any | None = None,
    sampling: str = "regular",
    sampling_pct: float = 1.0,
) -> dict[str, float]:
    """Calculate the two legacy ClearEx similarity metrics all-or-nothing."""
    import ants
    import numpy as np

    if isinstance(fixed, np.ndarray):
        fixed = _from_external_array(ants, fixed)
    if isinstance(moving, np.ndarray):
        moving = _from_external_array(ants, moving)
    if isinstance(fixed_mask, np.ndarray):
        fixed_mask = _from_external_array(ants, fixed_mask, fixed)
    if isinstance(moving_mask, np.ndarray):
        moving_mask = _from_external_array(ants, moving_mask, moving)

    results: dict[str, float] = {}
    for metric_type in _METRIC_TYPES:
        try:
            value = ants.image_similarity(
                fixed,
                moving,
                metric_type=metric_type,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
                sampling_strategy=sampling,
                sampling_percentage=sampling_pct,
            )
            results[metric_type] = -float(value)
        except Exception as error:
            raise RegistrationStageError(
                f"ANTs image similarity failed for metric {metric_type}"
            ) from error
    return results


def write_metrics(metrics: Mapping[str, object], path: str | Path) -> None:
    """Write alphabetically sorted metric text through an atomic promotion."""
    destination = Path(path)
    text = _metrics_text(metrics)

    def writer(temporary: Path) -> None:
        temporary.write_text(text, encoding="utf-8")

    atomic_write(
        destination,
        writer,
        lambda temporary: temporary.read_text(encoding="utf-8") == text,
    )


def read_metrics(path: str | Path) -> dict[str, float]:
    """Read a canonical metric artifact and reject partial or malformed data."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    metrics: dict[str, float] = {}
    if not text or not text.endswith("\n"):
        raise ValueError("Metric artifact must be non-empty and newline terminated")
    for line in text.splitlines():
        name, separator, raw_value = line.partition(": ")
        if not separator or not name or name in metrics:
            raise ValueError("Metric artifact contains a malformed or duplicate entry")
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError("Metric artifact contains a non-numeric value") from error
        if not math.isfinite(value):
            raise ValueError("Metric artifact values must be finite")
        metrics[name] = value
    if _metrics_text(metrics) != text:
        raise ValueError("Metric artifact is not canonical")
    return metrics


__all__ = ["calculate_metrics", "read_metrics", "write_metrics"]
