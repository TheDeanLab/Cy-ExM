from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cyexm.errors import RegistrationStageError
from cyexm.metrics import calculate_metrics, read_metrics, write_metrics

pytestmark = pytest.mark.unit


class FakeImage:
    def __init__(
        self,
        name: str,
        *,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        spacing: tuple[float, float, float] = (0.1, 0.2, 0.3),
        direction: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ) -> None:
        self.name = name
        self.shape = (2, 3, 4)
        self.origin = origin
        self.spacing = spacing
        self.direction = direction


def test_calculate_metrics_ports_clear_ex_metric_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converted: list[np.ndarray] = []
    calls: list[dict[str, Any]] = []

    def from_numpy(array: np.ndarray) -> FakeImage:
        converted.append(array)
        return FakeImage(f"converted-{len(converted)}")

    def image_similarity(fixed: FakeImage, moving: FakeImage, **kwargs: Any) -> float:
        calls.append({"fixed": fixed, "moving": moving, **kwargs})
        return {
            "Correlation": -0.75,
            "MattesMutualInformation": -0.25,
        }[kwargs["metric_type"]]

    fake_ants = SimpleNamespace(
        from_numpy=from_numpy,
        image_similarity=image_similarity,
    )
    monkeypatch.setitem(sys.modules, "ants", fake_ants)
    fixed = np.zeros((2, 3, 4), dtype=np.float32)
    moving = np.ones((2, 3, 4), dtype=np.float32)
    fixed_mask = FakeImage("fixed-mask")
    moving_mask = FakeImage("moving-mask")

    metrics = calculate_metrics(
        fixed,
        moving,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        sampling="random",
        sampling_pct=0.4,
    )

    assert metrics == {"Correlation": 0.75, "MattesMutualInformation": 0.25}
    np.testing.assert_array_equal(converted[0], np.transpose(fixed, (2, 1, 0)))
    np.testing.assert_array_equal(converted[1], np.transpose(moving, (2, 1, 0)))
    assert [call["metric_type"] for call in calls] == [
        "Correlation",
        "MattesMutualInformation",
    ]
    assert all(call["fixed_mask"] is fixed_mask for call in calls)
    assert all(call["moving_mask"] is moving_mask for call in calls)
    assert all(call["sampling_strategy"] == "random" for call in calls)
    assert all(call["sampling_percentage"] == 0.4 for call in calls)


def test_calculate_metrics_ndarray_masks_use_matching_reference_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converted: list[dict[str, object]] = []

    def from_numpy(array: np.ndarray, **kwargs: object) -> FakeImage:
        converted.append({"array": array, **kwargs})
        return FakeImage(f"converted-{len(converted)}", **kwargs)  # type: ignore[arg-type]

    fake_ants = SimpleNamespace(
        from_numpy=from_numpy,
        image_similarity=lambda *args, **kwargs: -0.5,
    )
    monkeypatch.setitem(sys.modules, "ants", fake_ants)
    fixed_direction = (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    moving_direction = (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
    fixed = FakeImage(
        "fixed",
        origin=(10.0, 20.0, 30.0),
        spacing=(0.2, 0.3, 0.5),
        direction=fixed_direction,
    )
    moving = FakeImage(
        "moving",
        origin=(-1.0, -2.0, -3.0),
        spacing=(0.7, 0.8, 0.9),
        direction=moving_direction,
    )
    fixed_mask = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    moving_mask = fixed_mask + 1

    calculate_metrics(
        fixed,
        moving,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
    )

    assert len(converted) == 2
    for call, mask, reference in (
        (converted[0], fixed_mask, fixed),
        (converted[1], moving_mask, moving),
    ):
        np.testing.assert_array_equal(
            call["array"],
            np.transpose(mask, (2, 1, 0)),
        )
        assert call["origin"] == reference.origin
        assert call["spacing"] == reference.spacing
        assert call["direction"] == reference.direction


@pytest.mark.parametrize(
    ("failed_metric", "expected_calls"),
    [
        pytest.param("Correlation", ["Correlation"], id="first-metric"),
        pytest.param(
            "MattesMutualInformation",
            ["Correlation", "MattesMutualInformation"],
            id="second-metric",
        ),
    ],
)
def test_calculate_metrics_raises_instead_of_returning_partial_results(
    monkeypatch: pytest.MonkeyPatch,
    failed_metric: str,
    expected_calls: list[str],
) -> None:
    calls: list[str] = []

    def image_similarity(fixed: FakeImage, moving: FakeImage, **kwargs: Any) -> float:
        del fixed, moving
        metric_type = kwargs["metric_type"]
        calls.append(metric_type)
        if metric_type == failed_metric:
            raise RuntimeError("injected ANTs metric failure")
        return -0.75

    fake_ants = SimpleNamespace(
        from_numpy=lambda array: FakeImage(str(array.shape)),
        image_similarity=image_similarity,
    )
    monkeypatch.setitem(sys.modules, "ants", fake_ants)

    with pytest.raises(RegistrationStageError, match=failed_metric) as error:
        calculate_metrics(FakeImage("fixed"), FakeImage("moving"))

    assert isinstance(error.value.__cause__, RuntimeError)
    assert calls == expected_calls


def test_metric_file_is_sorted_and_atomic(tmp_path: Path) -> None:
    destination = tmp_path / "metrics.txt"

    write_metrics(
        {"MattesMutualInformation": 0.25, "Correlation": 0.75},
        destination,
    )

    assert destination.read_text() == (
        "Correlation: 0.75\nMattesMutualInformation: 0.25\n"
    )
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_metric_file_round_trips_through_strict_reader(tmp_path: Path) -> None:
    destination = tmp_path / "metrics.txt"
    expected = {"MattesMutualInformation": 0.25, "Correlation": 0.75}

    write_metrics(expected, destination)

    assert read_metrics(destination) == expected


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param("Correlation = 0.75\n", id="missing-colon"),
        pytest.param("Correlation: nan\n", id="non-finite"),
        pytest.param("Correlation: 0.75\nCorrelation: 0.5\n", id="duplicate"),
        pytest.param("Correlation: 0.750\n", id="non-canonical"),
    ],
)
def test_metric_reader_rejects_malformed_or_noncanonical_files(
    tmp_path: Path,
    contents: str,
) -> None:
    destination = tmp_path / "metrics.txt"
    destination.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        read_metrics(destination)


def test_metric_write_failure_keeps_the_previous_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "metrics.txt"
    destination.write_text("previous\n", encoding="utf-8")
    real_write_text = Path.write_text

    def failing_write_text(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        real_write_text(
            path,
            "partial\n",
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
        raise RuntimeError("injected metric failure")

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    with pytest.raises(RuntimeError, match="injected metric failure"):
        write_metrics({"Correlation": 0.75}, destination)

    assert destination.read_text(encoding="utf-8") == "previous\n"
    assert list(tmp_path.glob(".*.tmp*")) == []
