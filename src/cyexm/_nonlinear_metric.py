from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

type NonlinearMetricName = Literal["CC", "mattes"]

DEFAULT_CC_RADIUS = 4
MATTES_BINS = 32
_METRICS = frozenset({"CC", "mattes"})


@dataclass(frozen=True, slots=True)
class ResolvedNonlinearMetric:
    metric: NonlinearMetricName
    radius: int | None
    bins: int | None

    @property
    def syn_sampling(self) -> int:
        value = self.radius if self.metric == "CC" else self.bins
        if value is None:
            raise RuntimeError("resolved nonlinear metric lacks sampling evidence")
        return value


def resolve_nonlinear_metric(
    metric: object = "mattes", radius: object = None
) -> ResolvedNonlinearMetric:
    if type(metric) is not str or metric not in _METRICS:
        raise ValueError(
            f"nonlinear metric must be exactly 'CC' or 'mattes'; got {metric!r}"
        )
    name = cast(NonlinearMetricName, metric)
    if name == "mattes":
        if radius is not None:
            raise ValueError("nonlinear radius is valid only for metric 'CC'")
        return ResolvedNonlinearMetric(metric=name, radius=None, bins=MATTES_BINS)
    if radius is None:
        radius = DEFAULT_CC_RADIUS
    if type(radius) is not int:
        raise TypeError(
            f"nonlinear CC radius must be a positive built-in integer; got {radius!r}"
        )
    if radius < 1:
        raise ValueError(f"nonlinear CC radius must be at least 1; got {radius!r}")
    return ResolvedNonlinearMetric(metric=name, radius=radius, bins=None)


__all__ = [
    "DEFAULT_CC_RADIUS",
    "MATTES_BINS",
    "NonlinearMetricName",
    "ResolvedNonlinearMetric",
    "resolve_nonlinear_metric",
]
