from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy import ndimage

from cyexm.registration.initial_alignment import (
    ROTATION_ESTIMATOR_POLICY,
    SIMILARITY_ESTIMATOR_POLICY,
    EstimatedXYRotationPolicy,
    EstimatedXYSimilarityPolicy,
    ExplicitXYRotationPolicy,
    InitialAlignmentPolicy,
    XYRotationEstimate,
    estimate_xy_rotation,
    estimate_xy_similarity,
    initial_alignment_parameters,
    initial_alignment_result_parameters,
    parse_initial_alignment_result,
    render_initial_alignment_qc,
)
from cyexm.spatial import ImageGeometry

pytestmark = pytest.mark.unit


def test_similarity_policy_normalizes_bounded_search_settings() -> None:
    similarity = EstimatedXYSimilarityPolicy(
        search_range_deg=(-18, 18),
        coarse_step_deg=2,
        refinement_step_deg=0.25,
        scale_range=(0.92, 1.08),
        coarse_scale_step=0.02,
        refinement_scale_step=0.005,
        minimum_score_gain=0.02,
        minimum_peak_margin=0.004,
    )

    assert similarity.search_range_deg == (-18.0, 18.0)
    assert similarity.scale_range == (0.92, 1.08)
    assert similarity.coarse_scale_step == 0.02
    assert similarity.refinement_scale_step == 0.005
    assert InitialAlignmentPolicy(xy_similarity=similarity).xy_rotation is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"scale_range": (1.0, 1.1)},
        {"scale_range": (0.9, 1.0)},
        {"scale_range": (1.1, 0.9)},
        {"coarse_scale_step": 0.0},
        {"coarse_scale_step": 0.01, "refinement_scale_step": 0.01},
        {"scale_range": (0.99, 1.01), "coarse_scale_step": 0.03},
    ],
)
def test_similarity_policy_rejects_invalid_scale_search(
    overrides: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        EstimatedXYSimilarityPolicy(**overrides)  # type: ignore[arg-type]


def test_initial_alignment_policy_requires_exactly_one_mode() -> None:
    similarity = EstimatedXYSimilarityPolicy()
    rotation = EstimatedXYRotationPolicy()

    with pytest.raises(ValueError, match="exactly one"):
        InitialAlignmentPolicy()
    with pytest.raises(ValueError, match="exactly one"):
        InitialAlignmentPolicy(xy_rotation=rotation, xy_similarity=similarity)


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


def _similarity_policy(
    **overrides: float | tuple[float, float],
) -> EstimatedXYSimilarityPolicy:
    values: dict[str, object] = {
        "search_range_deg": (-20.0, 20.0),
        "coarse_step_deg": 2.0,
        "refinement_step_deg": 0.5,
        "scale_range": (0.9, 1.1),
        "coarse_scale_step": 0.02,
        "refinement_scale_step": 0.005,
        "minimum_score_gain": 0.002,
        "minimum_peak_margin": 0.0001,
    }
    values.update(overrides)
    return EstimatedXYSimilarityPolicy(**values)  # type: ignore[arg-type]


def _center_scale(volume: np.ndarray, scale: float) -> np.ndarray:
    center = (np.asarray(volume.shape[1:], dtype=np.float64) - 1.0) / 2.0
    matrix = np.eye(3, dtype=np.float64)
    matrix[1:, 1:] /= scale
    offset = np.zeros(3, dtype=np.float64)
    offset[1:] = center - matrix[1:, 1:] @ center
    return ndimage.affine_transform(
        volume,
        matrix=matrix,
        offset=offset,
        output_shape=volume.shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def test_similarity_estimator_recovers_rotation_translation_and_uniform_scale() -> None:
    fixed = _asymmetric_volume()
    expected_scale = 1.06
    moving = _center_scale(fixed, 1.0 / expected_scale)
    moving = ndimage.rotate(
        moving,
        angle=12.0,
        axes=(1, 2),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    moving = ndimage.shift(
        moving,
        shift=(0.0, 4.0, -6.0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    estimate = estimate_xy_similarity(fixed, moving, _similarity_policy())
    rotation_only = estimate_xy_rotation(fixed, moving, _policy())

    assert estimate.accepted is True
    assert estimate.policy == SIMILARITY_ESTIMATOR_POLICY
    assert estimate.angle_deg == pytest.approx(12.0, abs=0.5)
    assert estimate.scale == pytest.approx(expected_scale, abs=0.01)
    assert estimate.translation_yx_px == pytest.approx((-4.0, 6.0), abs=2.0)
    assert estimate.score_gain is not None and estimate.score_gain > 0.02
    assert estimate.coarse_angle_scores
    assert estimate.refined_scale_scores
    assert estimate.joint_scores
    assert (
        max(item.correlation for item in estimate.joint_scores)
        > max(
            item.correlation
            for item in (*rotation_only.coarse_scores, *rotation_only.refined_scores)
        )
        + 0.01
    )


def test_similarity_estimator_rejects_scale_at_search_boundary() -> None:
    fixed = _asymmetric_volume()
    moving = _center_scale(fixed, 1.0 / 1.12)

    estimate = estimate_xy_similarity(
        fixed,
        moving,
        _similarity_policy(scale_range=(0.96, 1.08)),
    )

    assert estimate.accepted is False
    assert estimate.rejection_reason == "winner-at-scale-boundary"


def test_similarity_result_round_trips_canonical_evidence() -> None:
    policy = InitialAlignmentPolicy(xy_similarity=_similarity_policy())
    similarity = policy.xy_similarity
    assert similarity is not None
    result = estimate_xy_similarity(
        _asymmetric_volume(),
        _center_scale(_asymmetric_volume(), 1.0 / 1.04),
        similarity,
    )

    payload = initial_alignment_result_parameters(result)

    assert payload["scale"] == pytest.approx(1.04, abs=0.01)
    assert parse_initial_alignment_result(payload, policy) == result


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


def test_estimator_scores_rotation_in_physical_xy_space() -> None:
    isotropic = _asymmetric_volume()
    fixed = isotropic[:, :, ::2]
    moving_isotropic = ndimage.rotate(
        isotropic,
        angle=18.0,
        axes=(1, 2),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    moving = moving_isotropic[:, :, ::2]
    geometry = ImageGeometry(
        shape=fixed.shape,
        origin=(0.0, 0.0, 0.0),
        spacing=(2.0, 1.0, 1.0),
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )

    estimate = estimate_xy_rotation(
        fixed,
        moving,
        _policy(),
        fixed_geometry=geometry,
        moving_geometry=geometry,
    )

    assert estimate.accepted is True
    assert estimate.angle_deg == pytest.approx(18.0, abs=0.5)


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


def test_realized_evidence_parser_rejects_explicit_angle_drift() -> None:
    policy = InitialAlignmentPolicy(ExplicitXYRotationPolicy(angle_deg=27.7))
    result = XYRotationEstimate(
        policy="explicit-xy-rotation-v1",
        accepted=True,
        rejection_reason=None,
        angle_deg=27.7,
        translation_yx_px=(0.0, 0.0),
        zero_angle_score=None,
        score_gain=None,
        peak_margin=None,
        valid_overlap_fraction=None,
        coarse_scores=(),
        refined_scores=(),
    )
    payload = initial_alignment_result_parameters(result)

    assert parse_initial_alignment_result(payload, policy) == result
    payload["angle_deg"] = 12.0
    with pytest.raises(ValueError, match=r"explicit.*inconsistent"):
        parse_initial_alignment_result(payload, policy)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"accepted": True, "rejection_reason": None}, "confidence policy"),
        ({"valid_overlap_fraction": 1.1}, r"overlap.*\[0, 1\]"),
        ({"angle_deg": 7.0}, "winner disagrees"),
        ({"score_gain": 0.5}, "score gain disagrees"),
        ({"peak_margin": 0.5}, "peak margin disagrees"),
    ],
)
def test_realized_evidence_parser_rejects_forged_estimate_decisions(
    mutation: dict[str, object], message: str
) -> None:
    fixed = _asymmetric_volume()
    moving = ndimage.rotate(fixed, 10.0, axes=(1, 2), reshape=False, order=1)
    policy = InitialAlignmentPolicy(_policy(minimum_score_gain=2.0))
    result = estimate_xy_rotation(fixed, moving, policy.xy_rotation)
    assert result.accepted is False
    payload = initial_alignment_result_parameters(result)
    payload.update(mutation)

    with pytest.raises(ValueError, match=message):
        parse_initial_alignment_result(payload, policy)


def test_realized_evidence_parser_accepts_estimator_output() -> None:
    fixed = _asymmetric_volume()
    moving = ndimage.rotate(fixed, 10.0, axes=(1, 2), reshape=False, order=1)
    policy = InitialAlignmentPolicy(_policy())
    result = estimate_xy_rotation(fixed, moving, policy.xy_rotation)

    assert (
        parse_initial_alignment_result(
            initial_alignment_result_parameters(result), policy
        )
        == result
    )


def test_policy_and_estimate_are_immutable() -> None:
    policy = _policy()
    estimate = estimate_xy_rotation(_asymmetric_volume(), _asymmetric_volume(), policy)

    with pytest.raises(FrozenInstanceError):
        policy.coarse_step_deg = 3.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        estimate.accepted = False  # type: ignore[misc]
