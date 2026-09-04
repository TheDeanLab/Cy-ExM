from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy import ndimage

from cyexm.registration.initial_alignment import (
    ROTATION_ESTIMATOR_POLICY,
    EstimatedXYRotationPolicy,
    estimate_xy_rotation,
    initial_alignment_parameters,
    render_initial_alignment_qc,
)

pytestmark = pytest.mark.unit


def _asymmetric_volume() -> np.ndarray:
    yy, xx = np.mgrid[:72, :80]
    image = (
        1.0 * np.exp(-((yy - 17) ** 2 + (xx - 21) ** 2) / 30.0)
        + 0.75 * np.exp(-((yy - 51) ** 2 + (xx - 58) ** 2) / 55.0)
        + 0.45 * np.exp(-((yy - 26) ** 2 + (xx - 63) ** 2) / 18.0)
    )
    image[38:43, 14:50] += 0.5
    return np.stack((image * 0.7, image, image * 0.85)).astype(np.float32)


def _policy(**overrides: float | tuple[float, float]) -> EstimatedXYRotationPolicy:
    values: dict[str, object] = {
        "search_range_deg": (-35.0, 35.0),
        "coarse_step_deg": 2.0,
        "refinement_step_deg": 0.25,
        "minimum_score_gain": 0.005,
        "minimum_peak_margin": 0.005,
    }
    values.update(overrides)
    return EstimatedXYRotationPolicy(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("applied_angle", [-24.0, 18.0])
def test_estimator_recovers_xy_rotation_correction(applied_angle: float) -> None:
    fixed = _asymmetric_volume()
    moving = ndimage.rotate(
        fixed,
        angle=applied_angle,
        axes=(1, 2),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    estimate = estimate_xy_rotation(fixed, moving, _policy())

    assert estimate.accepted is True
    assert estimate.rejection_reason is None
    # SciPy uses row-down display coordinates, so its positive display angle is
    # the negative of the manifest's physical +Z right-hand rotation. The
    # correction returned by the estimator is therefore the same signed value
    # as the angle used here to construct the misalignment.
    assert estimate.angle_deg == pytest.approx(applied_angle, abs=0.25)
    assert estimate.policy == ROTATION_ESTIMATOR_POLICY
    assert estimate.coarse_scores
    assert estimate.refined_scores


def test_estimator_recovers_translation_at_zero_rotation() -> None:
    fixed = _asymmetric_volume()
    applied_shift = (0.0, 5.0, -7.0)
    moving = ndimage.shift(
        fixed,
        shift=applied_shift,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    estimate = estimate_xy_rotation(fixed, moving, _policy())

    assert estimate.accepted is True
    assert estimate.angle_deg == pytest.approx(0.0, abs=0.25)
    assert estimate.translation_yx_px == pytest.approx((-5.0, 7.0), abs=1.0)


def test_estimator_normalizes_substep_rotation_to_zero() -> None:
    fixed = _asymmetric_volume()
    moving = ndimage.rotate(fixed, 0.2, axes=(1, 2), reshape=False, order=1)

    estimate = estimate_xy_rotation(fixed, moving, _policy())

    assert estimate.accepted is True
    assert estimate.angle_deg == 0.0


def test_estimator_is_deterministic_and_evidence_is_json_compatible() -> None:
    fixed = _asymmetric_volume()
    moving = ndimage.rotate(fixed, 12.0, axes=(1, 2), reshape=False, order=1)
    policy = _policy()

    first = estimate_xy_rotation(fixed, moving, policy)
    second = estimate_xy_rotation(fixed, moving, policy)

    assert first == second
    parameters = initial_alignment_parameters(policy, first)
    assert parameters["algorithm"] == ROTATION_ESTIMATOR_POLICY
    assert parameters["mode"] == "estimate"
    assert parameters["accepted"] is True
    assert isinstance(parameters["coarse_scores"], list)
    assert isinstance(parameters["refined_scores"], list)


@pytest.mark.parametrize(
    ("moving", "reason"),
    [
        (np.zeros((3, 72, 80), dtype=np.float32), "constant-moving-input"),
        (
            np.full((3, 72, 80), np.nan, dtype=np.float32),
            "nonfinite-moving-input",
        ),
    ],
)
def test_estimator_rejects_invalid_inputs(moving: np.ndarray, reason: str) -> None:
    estimate = estimate_xy_rotation(_asymmetric_volume(), moving, _policy())

    assert estimate.accepted is False
    assert estimate.rejection_reason == reason
    assert estimate.angle_deg is None
    assert estimate.translation_yx_px is None


def test_estimator_rejects_boundary_winner() -> None:
    fixed = _asymmetric_volume()
    moving = ndimage.rotate(fixed, 25.0, axes=(1, 2), reshape=False, order=1)

    estimate = estimate_xy_rotation(
        fixed,
        moving,
        _policy(search_range_deg=(-10.0, 10.0)),
    )

    assert estimate.accepted is False
    assert estimate.rejection_reason == "winner-at-search-boundary"


def test_estimator_rejects_configured_score_gain() -> None:
    fixed = _asymmetric_volume()
    moving = ndimage.rotate(fixed, 10.0, axes=(1, 2), reshape=False, order=1)

    estimate = estimate_xy_rotation(
        fixed,
        moving,
        _policy(minimum_score_gain=2.0),
    )

    assert estimate.accepted is False
    assert estimate.rejection_reason == "insufficient-score-gain"


def test_estimator_rejects_configured_peak_margin() -> None:
    fixed = _asymmetric_volume()
    moving = ndimage.rotate(fixed, 10.0, axes=(1, 2), reshape=False, order=1)

    estimate = estimate_xy_rotation(
        fixed,
        moving,
        _policy(minimum_peak_margin=2.0),
    )

    assert estimate.accepted is False
    assert estimate.rejection_reason == "insufficient-peak-margin"


def test_qc_raster_is_deterministic_uint8_evidence() -> None:
    fixed = _asymmetric_volume()
    moving = ndimage.rotate(fixed, 18.0, axes=(1, 2), reshape=False, order=1)
    estimate = estimate_xy_rotation(fixed, moving, _policy())

    first = render_initial_alignment_qc(fixed, moving, estimate)
    second = render_initial_alignment_qc(fixed, moving, estimate)

    assert first.dtype == np.uint8
    assert first.shape == (512, 768)
    assert np.array_equal(first, second)
    assert np.count_nonzero(first) > 1000
    assert np.count_nonzero(first[:, 512:]) > 100


def test_policy_and_estimate_are_immutable() -> None:
    policy = _policy()
    estimate = estimate_xy_rotation(_asymmetric_volume(), _asymmetric_volume(), policy)

    with pytest.raises(FrozenInstanceError):
        policy.coarse_step_deg = 3.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        estimate.accepted = False  # type: ignore[misc]
