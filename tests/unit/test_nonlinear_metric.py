from __future__ import annotations

import pytest

from cyexm._nonlinear_metric import resolve_nonlinear_metric

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("radius", [None, 4])
def test_cc_omitted_and_explicit_default_resolve_identically(radius: object) -> None:
    resolved = resolve_nonlinear_metric("CC", radius)

    assert resolved.metric == "CC"
    assert resolved.radius == 4
    assert resolved.bins is None
    assert resolved.syn_sampling == 4


def test_explicit_cc_radius_is_preserved() -> None:
    resolved = resolve_nonlinear_metric("CC", 2)

    assert resolved.radius == 2
    assert resolved.syn_sampling == 2


def test_mattes_resolves_to_fixed_bins() -> None:
    resolved = resolve_nonlinear_metric("mattes")

    assert resolved.metric == "mattes"
    assert resolved.radius is None
    assert resolved.bins == 32
    assert resolved.syn_sampling == 32


@pytest.mark.parametrize("metric", ["cc", "Mattes", "", 1, None])
def test_unknown_or_nonstring_metric_is_rejected(metric: object) -> None:
    with pytest.raises(ValueError, match="metric"):
        resolve_nonlinear_metric(metric)


@pytest.mark.parametrize("radius", [True, 0, -1, 1.0, "4", (1, 2, 4)])
def test_invalid_cc_radius_is_rejected(radius: object) -> None:
    with pytest.raises((TypeError, ValueError), match="radius"):
        resolve_nonlinear_metric("CC", radius)


def test_integer_subclass_cc_radius_is_rejected() -> None:
    class Radius(int):
        pass

    with pytest.raises(TypeError, match="radius"):
        resolve_nonlinear_metric("CC", Radius(4))


def test_mattes_radius_is_rejected() -> None:
    with pytest.raises(ValueError, match="radius"):
        resolve_nonlinear_metric("mattes", 4)
