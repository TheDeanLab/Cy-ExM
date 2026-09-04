"""Target-blind initial alignment policies for registration channels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

ROTATION_ESTIMATOR_POLICY = "mip-gradient-phase-correlation-xy-rotation-v1"


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True, slots=True)
class ExplicitXYRotationPolicy:
    """Explicit physical rotation correction for one moving image."""

    angle_deg: float
    mode: Literal["explicit"] = "explicit"

    def __post_init__(self) -> None:
        angle = _finite_float(self.angle_deg, "angle_deg")
        if not -180.0 <= angle <= 180.0:
            raise ValueError("angle_deg must be within [-180, 180]")
        object.__setattr__(self, "angle_deg", angle)


@dataclass(frozen=True, slots=True)
class EstimatedXYRotationPolicy:
    """Settings for target-blind XY rotation estimation."""

    search_range_deg: tuple[float, float] = (-45.0, 45.0)
    coarse_step_deg: float = 1.0
    refinement_step_deg: float = 0.1
    minimum_score_gain: float = 0.01
    minimum_peak_margin: float = 0.01
    mode: Literal["estimate"] = "estimate"

    def __post_init__(self) -> None:
        if type(self.search_range_deg) is not tuple or len(self.search_range_deg) != 2:
            raise TypeError("search_range_deg must be a two-value tuple")
        lower = _finite_float(self.search_range_deg[0], "search_range_deg[0]")
        upper = _finite_float(self.search_range_deg[1], "search_range_deg[1]")
        if lower < -180.0 or upper > 180.0 or lower >= upper:
            raise ValueError(
                "search_range_deg must be strictly increasing within [-180, 180]"
            )
        coarse = _finite_float(self.coarse_step_deg, "coarse_step_deg")
        refinement = _finite_float(self.refinement_step_deg, "refinement_step_deg")
        gain = _finite_float(self.minimum_score_gain, "minimum_score_gain")
        margin = _finite_float(self.minimum_peak_margin, "minimum_peak_margin")
        if coarse <= 0.0 or refinement <= 0.0:
            raise ValueError("rotation search steps must be positive")
        if refinement >= coarse:
            raise ValueError("refinement_step_deg must be smaller than coarse_step_deg")
        if gain <= 0.0 or margin <= 0.0:
            raise ValueError("rotation confidence thresholds must be positive")
        object.__setattr__(self, "search_range_deg", (lower, upper))
        object.__setattr__(self, "coarse_step_deg", coarse)
        object.__setattr__(self, "refinement_step_deg", refinement)
        object.__setattr__(self, "minimum_score_gain", gain)
        object.__setattr__(self, "minimum_peak_margin", margin)


type XYRotationPolicy = ExplicitXYRotationPolicy | EstimatedXYRotationPolicy


@dataclass(frozen=True, slots=True)
class InitialAlignmentPolicy:
    """Optional geometric initialization accepted by registration entry points."""

    xy_rotation: XYRotationPolicy


@dataclass(frozen=True, slots=True)
class RotationScore:
    """Score and translation for one candidate correction angle."""

    angle_deg: float
    translation_yx_px: tuple[float, float]
    correlation: float
    valid_overlap_fraction: float


@dataclass(frozen=True, slots=True)
class XYRotationEstimate:
    """Deterministic result and diagnostic evidence from rotation estimation."""

    policy: str
    accepted: bool
    rejection_reason: str | None
    angle_deg: float | None
    translation_yx_px: tuple[float, float] | None
    zero_angle_score: float | None
    score_gain: float | None
    peak_margin: float | None
    valid_overlap_fraction: float | None
    coarse_scores: tuple[RotationScore, ...]
    refined_scores: tuple[RotationScore, ...]


def _scientific_runtime() -> tuple[Any, Any, Any]:
    import numpy as np
    from scipy import ndimage
    from skimage.registration import phase_cross_correlation

    return np, ndimage, phase_cross_correlation


def _rejected(reason: str) -> XYRotationEstimate:
    return XYRotationEstimate(
        policy=ROTATION_ESTIMATOR_POLICY,
        accepted=False,
        rejection_reason=reason,
        angle_deg=None,
        translation_yx_px=None,
        zero_angle_score=None,
        score_gain=None,
        peak_margin=None,
        valid_overlap_fraction=None,
        coarse_scores=(),
        refined_scores=(),
    )


def _projection_feature(np: Any, ndimage: Any, volume: Any, role: str) -> Any:
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"{role}-input-must-be-zyx")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"nonfinite-{role}-input")
    projection = np.max(array.astype(np.float32, copy=False), axis=0)
    lower, upper = np.percentile(projection, (1.0, 99.0))
    if not math.isfinite(float(lower)) or not math.isfinite(float(upper)):
        raise ValueError(f"nonfinite-{role}-input")
    if float(upper) <= float(lower):
        raise ValueError(f"constant-{role}-input")
    normalized = np.clip((projection - lower) / (upper - lower), 0.0, 1.0)
    smoothed = ndimage.gaussian_filter(normalized, sigma=2.0, mode="reflect")
    gradient_y, gradient_x = np.gradient(smoothed)
    return np.hypot(gradient_y, gradient_x).astype(np.float32, copy=False)


def _canvas_size(*images: Any) -> int:
    return math.ceil(
        max(math.hypot(int(image.shape[0]), int(image.shape[1])) for image in images)
    )


def _center_pad(np: Any, image: Any, size: int, *, value: float = 0.0) -> Any:
    output = np.full((size, size), value, dtype=image.dtype)
    y0 = (size - int(image.shape[0])) // 2
    x0 = (size - int(image.shape[1])) // 2
    output[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]] = image
    return output


def _candidate_angles(lower: float, upper: float, step: float) -> tuple[float, ...]:
    count = math.floor((upper - lower) / step)
    values = [round(lower + index * step, 12) for index in range(count + 1)]
    if not math.isclose(values[-1], upper, abs_tol=1e-12):
        values.append(round(upper, 12))
    return tuple(values)


def _score_angle(
    np: Any,
    ndimage: Any,
    phase_cross_correlation: Any,
    fixed: Any,
    moving: Any,
    fixed_footprint: Any,
    moving_footprint: Any,
    angle: float,
) -> RotationScore | None:
    # Candidate angles use the public physical +Z right-hand convention.
    # scipy.ndimage.rotate uses row-down display coordinates, whose sign is
    # opposite for an identity physical direction matrix.
    display_angle = -angle
    rotated = ndimage.rotate(
        moving,
        angle=display_angle,
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    rotated_footprint = ndimage.rotate(
        moving_footprint,
        angle=display_angle,
        reshape=False,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    shift, _, _ = phase_cross_correlation(
        fixed,
        rotated,
        upsample_factor=1,
        normalization=None,
    )
    shift_yx = float(shift[0]), float(shift[1])
    aligned = ndimage.shift(
        rotated,
        shift=shift_yx,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    aligned_footprint = ndimage.shift(
        rotated_footprint,
        shift=shift_yx,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    overlap = fixed_footprint & (aligned_footprint > 0)
    overlap_count = int(np.count_nonzero(overlap))
    if overlap_count < 4:
        return None
    fixed_values = fixed[overlap].astype(np.float64, copy=False)
    moving_values = aligned[overlap].astype(np.float64, copy=False)
    fixed_values = fixed_values - fixed_values.mean()
    moving_values = moving_values - moving_values.mean()
    denominator = float(
        np.sqrt(
            np.dot(fixed_values, fixed_values) * np.dot(moving_values, moving_values)
        )
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        correlation = -1.0
    else:
        correlation = float(np.dot(fixed_values, moving_values) / denominator)
    return RotationScore(
        angle_deg=float(angle),
        translation_yx_px=shift_yx,
        correlation=correlation,
        valid_overlap_fraction=float(overlap_count / overlap.size),
    )


def _best(scores: tuple[RotationScore, ...]) -> RotationScore:
    return min(
        scores,
        key=lambda item: (-item.correlation, abs(item.angle_deg), item.angle_deg),
    )


def estimate_xy_rotation(
    fixed_zyx: Any,
    moving_zyx: Any,
    policy: EstimatedXYRotationPolicy,
) -> XYRotationEstimate:
    """Estimate an XY correction from fixed and moving registration channels."""
    if type(policy) is not EstimatedXYRotationPolicy:
        raise TypeError("policy must be an EstimatedXYRotationPolicy")
    np, ndimage, phase_cross_correlation = _scientific_runtime()
    try:
        fixed_feature = _projection_feature(np, ndimage, fixed_zyx, "fixed")
        moving_feature = _projection_feature(np, ndimage, moving_zyx, "moving")
    except ValueError as error:
        return _rejected(str(error))

    size = _canvas_size(fixed_feature, moving_feature)
    fixed = _center_pad(np, fixed_feature, size)
    moving = _center_pad(np, moving_feature, size)
    fixed_footprint = _center_pad(
        np, np.ones_like(fixed_feature, dtype=bool), size, value=False
    )
    moving_footprint = _center_pad(
        np, np.ones_like(moving_feature, dtype=np.uint8), size
    )

    def score(angle: float) -> RotationScore | None:
        return _score_angle(
            np,
            ndimage,
            phase_cross_correlation,
            fixed,
            moving,
            fixed_footprint,
            moving_footprint,
            angle,
        )

    lower, upper = policy.search_range_deg
    coarse_items: list[RotationScore] = []
    for angle in _candidate_angles(lower, upper, policy.coarse_step_deg):
        item = score(angle)
        if item is not None:
            coarse_items.append(item)
    coarse = tuple(coarse_items)
    if not coarse:
        return _rejected("no-positive-valid-overlap")
    coarse_best = _best(coarse)
    refine_lower = max(lower, coarse_best.angle_deg - policy.coarse_step_deg)
    refine_upper = min(upper, coarse_best.angle_deg + policy.coarse_step_deg)
    refined_items: list[RotationScore] = []
    for angle in _candidate_angles(
        refine_lower,
        refine_upper,
        policy.refinement_step_deg,
    ):
        item = score(angle)
        if item is not None:
            refined_items.append(item)
    refined = tuple(refined_items)
    winner = _best(refined or coarse)
    zero = score(0.0)
    if abs(winner.angle_deg) <= policy.refinement_step_deg and zero is not None:
        winner = zero
    zero_score = winner.correlation if zero is None else zero.correlation
    score_gain = winner.correlation - zero_score
    competitors = tuple(
        item
        for item in (*coarse, *refined)
        if abs(item.angle_deg - winner.angle_deg) >= policy.coarse_step_deg - 1e-12
    )
    peak_margin = (
        winner.correlation - _best(competitors).correlation if competitors else 1.0
    )
    reason: str | None = None
    if math.isclose(winner.angle_deg, lower, abs_tol=1e-12) or math.isclose(
        winner.angle_deg, upper, abs_tol=1e-12
    ):
        reason = "winner-at-search-boundary"
    elif (
        abs(winner.angle_deg) > policy.refinement_step_deg
        and score_gain < policy.minimum_score_gain
    ):
        reason = "insufficient-score-gain"
    elif peak_margin < policy.minimum_peak_margin:
        reason = "insufficient-peak-margin"
    return XYRotationEstimate(
        policy=ROTATION_ESTIMATOR_POLICY,
        accepted=reason is None,
        rejection_reason=reason,
        angle_deg=winner.angle_deg,
        translation_yx_px=winner.translation_yx_px,
        zero_angle_score=zero_score,
        score_gain=score_gain,
        peak_margin=peak_margin,
        valid_overlap_fraction=winner.valid_overlap_fraction,
        coarse_scores=coarse,
        refined_scores=refined,
    )


def _score_mapping(score: RotationScore) -> dict[str, object]:
    return {
        "angle_deg": score.angle_deg,
        "translation_yx_px": list(score.translation_yx_px),
        "correlation": score.correlation,
        "valid_overlap_fraction": score.valid_overlap_fraction,
    }


def initial_alignment_parameters(
    policy: EstimatedXYRotationPolicy,
    result: XYRotationEstimate,
) -> dict[str, object]:
    """Return canonical JSON-compatible automatic initialization evidence."""
    if type(policy) is not EstimatedXYRotationPolicy:
        raise TypeError("policy must be an EstimatedXYRotationPolicy")
    if type(result) is not XYRotationEstimate:
        raise TypeError("result must be an XYRotationEstimate")
    return {
        "algorithm": result.policy,
        "mode": policy.mode,
        "search_range_deg": list(policy.search_range_deg),
        "coarse_step_deg": policy.coarse_step_deg,
        "refinement_step_deg": policy.refinement_step_deg,
        "minimum_score_gain": policy.minimum_score_gain,
        "minimum_peak_margin": policy.minimum_peak_margin,
        "accepted": result.accepted,
        "rejection_reason": result.rejection_reason,
        "angle_deg": result.angle_deg,
        "translation_yx_px": (
            None if result.translation_yx_px is None else list(result.translation_yx_px)
        ),
        "zero_angle_score": result.zero_angle_score,
        "score_gain": result.score_gain,
        "peak_margin": result.peak_margin,
        "valid_overlap_fraction": result.valid_overlap_fraction,
        "coarse_scores": [_score_mapping(item) for item in result.coarse_scores],
        "refined_scores": [_score_mapping(item) for item in result.refined_scores],
    }


def _display_panel(np: Any, image: Any) -> Any:
    from skimage.transform import resize

    array = np.asarray(image, dtype=np.float32)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum > minimum:
        array = (array - minimum) / (maximum - minimum)
    else:
        array = np.zeros_like(array)
    resized = resize(
        array,
        (256, 256),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    return np.clip(np.rint(resized * 255.0), 0, 255).astype(np.uint8)


def _score_panel(np: Any, estimate: XYRotationEstimate) -> Any:
    panel = np.zeros((256, 256), dtype=np.uint8)
    scores = (*estimate.coarse_scores, *estimate.refined_scores)
    if not scores:
        return panel
    angles = np.asarray([item.angle_deg for item in scores], dtype=np.float64)
    values = np.asarray([item.correlation for item in scores], dtype=np.float64)
    angle_span = float(angles.max() - angles.min())
    value_span = float(values.max() - values.min())
    xs = (
        np.full(angles.shape, 128, dtype=int)
        if angle_span == 0.0
        else np.rint((angles - angles.min()) / angle_span * 255.0).astype(int)
    )
    ys = (
        np.full(values.shape, 128, dtype=int)
        if value_span == 0.0
        else 255 - np.rint((values - values.min()) / value_span * 255.0).astype(int)
    )
    panel[np.clip(ys, 0, 255), np.clip(xs, 0, 255)] = 255
    return panel


def render_initial_alignment_qc(
    fixed_zyx: Any,
    moving_zyx: Any,
    estimate: XYRotationEstimate,
) -> Any:
    """Render a deterministic grayscale QC montage without Matplotlib."""
    np, ndimage, _ = _scientific_runtime()
    fixed = np.max(np.asarray(fixed_zyx, dtype=np.float32), axis=0)
    moving = np.max(np.asarray(moving_zyx, dtype=np.float32), axis=0)
    aligned = moving
    if estimate.angle_deg is not None and estimate.translation_yx_px is not None:
        aligned = ndimage.rotate(
            moving,
            angle=-estimate.angle_deg,
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        aligned = ndimage.shift(
            aligned,
            shift=estimate.translation_yx_px,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    fixed_panel = _display_panel(np, fixed)
    moving_panel = _display_panel(np, moving)
    aligned_panel = _display_panel(np, aligned)
    before_difference = np.abs(
        fixed_panel.astype(np.int16) - moving_panel.astype(np.int16)
    ).astype(np.uint8)
    after_difference = np.abs(
        fixed_panel.astype(np.int16) - aligned_panel.astype(np.int16)
    ).astype(np.uint8)
    return np.block(
        [
            [fixed_panel, moving_panel, before_difference],
            [aligned_panel, after_difference, _score_panel(np, estimate)],
        ]
    )


__all__ = [
    "ROTATION_ESTIMATOR_POLICY",
    "EstimatedXYRotationPolicy",
    "ExplicitXYRotationPolicy",
    "InitialAlignmentPolicy",
    "RotationScore",
    "XYRotationEstimate",
    "XYRotationPolicy",
    "estimate_xy_rotation",
    "initial_alignment_parameters",
    "render_initial_alignment_qc",
]
