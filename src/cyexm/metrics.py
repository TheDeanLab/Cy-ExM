from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cyexm.errors import RegistrationStageError
from cyexm.io.atomic import atomic_write

_METRIC_TYPES = ("Correlation", "MattesMutualInformation")


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
        fixed = ants.from_numpy(fixed)
    if isinstance(moving, np.ndarray):
        moving = ants.from_numpy(moving)

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
    text = "".join(f"{name}: {metrics[name]}\n" for name in sorted(metrics))

    def writer(temporary: Path) -> None:
        temporary.write_text(text, encoding="utf-8")

    atomic_write(
        destination,
        writer,
        lambda temporary: temporary.read_text(encoding="utf-8") == text,
    )


__all__ = ["calculate_metrics", "write_metrics"]
