from pathlib import Path

import pytest

from cyexm.registration.crop import (
    bounds_text,
    estimate_bounds,
    read_bounds,
    validate_bounds,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "value",
    ["bounds", b"bounds", (), (0, 1, 0, 1, 0), (False, 1, 0, 1, 0, 1)],
)
def test_validate_bounds_rejects_non_integer_six_tuples(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="six integers"):
        validate_bounds(value, (4, 8, 8))


@pytest.mark.parametrize(
    "value",
    [(-1, 1, 0, 1, 0, 1), (0, 5, 0, 1, 0, 1), (1, 1, 0, 1, 0, 1)],
)
def test_validate_bounds_rejects_empty_or_out_of_range_regions(value: object) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_bounds(value, (4, 8, 8))


def test_bounds_text_and_read_bounds_require_canonical_compact_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bounds.json"
    path.write_text("[0,4,1,7,2,8]\n", encoding="utf-8")

    bounds = read_bounds(path, (4, 8, 8))

    assert bounds == (0, 4, 1, 7, 2, 8)
    assert bounds_text(bounds) == "[0,4,1,7,2,8]\n"
    path.write_text("[0, 4, 1, 7, 2, 8]", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        read_bounds(path, (4, 8, 8))


def test_estimate_bounds_preserves_legacy_robust_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, int, float, float]] = []
    geometry = type("Geometry", (), {"shape": (4, 8, 8)})()
    backend = type(
        "Backend",
        (),
        {
            "to_array": lambda self, image: image,
            "geometry": lambda self, image: geometry,
        },
    )()
    monkeypatch.setattr(
        "cyexm.registration.crop.robust_bounding_box",
        lambda array, *, down_sampling, lower_pct, upper_pct: (
            calls.append((array, down_sampling, lower_pct, upper_pct))
            or (0, 4, 1, 7, 2, 8)
        ),
    )

    assert estimate_bounds(backend, "image") == (0, 4, 1, 7, 2, 8)  # type: ignore[arg-type]
    assert calls == [("image", 4, 0.1, 99.9)]
