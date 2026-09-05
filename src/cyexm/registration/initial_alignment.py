"""Target-blind initial alignment policies for registration channels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

from cyexm.spatial import ImageGeometry

ROTATION_ESTIMATOR_POLICY = "mip-gradient-phase-correlation-xy-rotation-v1"
SIMILARITY_ESTIMATOR_POLICY = "mip-gradient-phase-correlation-xy-similarity-v1"
EXPLICIT_ROTATION_POLICY = "explicit-xy-rotation-v1"


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
class EstimatedXYSimilarityPolicy:
    """Settings for target-blind XY similarity estimation."""

    search_range_deg: tuple[float, float] = (-45.0, 45.0)
    coarse_step_deg: float = 1.0
    refinement_step_deg: float = 0.1
    scale_range: tuple[float, float] = (0.9, 1.1)
    coarse_scale_step: float = 0.01
    refinement_scale_step: float = 0.002
    minimum_score_gain: float = 0.01
    minimum_peak_margin: float = 0.01
    mode: Literal["estimate"] = "estimate"

    def __post_init__(self) -> None:
        angle = EstimatedXYRotationPolicy(
            search_range_deg=self.search_range_deg,
            coarse_step_deg=self.coarse_step_deg,
            refinement_step_deg=self.refinement_step_deg,
            minimum_score_gain=self.minimum_score_gain,
            minimum_peak_margin=self.minimum_peak_margin,
        )
        if type(self.scale_range) is not tuple or len(self.scale_range) != 2:
            raise TypeError("scale_range must be a two-value tuple")
        lower = _finite_float(self.scale_range[0], "scale_range[0]")
        upper = _finite_float(self.scale_range[1], "scale_range[1]")
        coarse = _finite_float(self.coarse_scale_step, "coarse_scale_step")
        refinement = _finite_float(self.refinement_scale_step, "refinement_scale_step")
        if not lower < 1.0 < upper:
            raise ValueError("scale_range must be strictly increasing around 1.0")
        if coarse <= 0.0 or refinement <= 0.0:
            raise ValueError("scale search steps must be positive")
        if refinement >= coarse:
            raise ValueError(
                "refinement_scale_step must be smaller than coarse_scale_step"
            )
        if coarse > upper - lower:
            raise ValueError("coarse_scale_step must not exceed the scale range span")
        object.__setattr__(self, "search_range_deg", angle.search_range_deg)
        object.__setattr__(self, "coarse_step_deg", angle.coarse_step_deg)
        object.__setattr__(self, "refinement_step_deg", angle.refinement_step_deg)
        object.__setattr__(self, "scale_range", (lower, upper))
        object.__setattr__(self, "coarse_scale_step", coarse)
        object.__setattr__(self, "refinement_scale_step", refinement)
        object.__setattr__(self, "minimum_score_gain", angle.minimum_score_gain)
        object.__setattr__(self, "minimum_peak_margin", angle.minimum_peak_margin)


@dataclass(frozen=True, slots=True)
class InitialAlignmentPolicy:
    """Optional geometric initialization accepted by registration entry points."""

    xy_rotation: XYRotationPolicy | None = None
    xy_similarity: EstimatedXYSimilarityPolicy | None = None

    def __post_init__(self) -> None:
        rotation_valid = type(self.xy_rotation) in {
            ExplicitXYRotationPolicy,
            EstimatedXYRotationPolicy,
        }
        similarity_valid = type(self.xy_similarity) is EstimatedXYSimilarityPolicy
        if rotation_valid == similarity_valid:
            raise ValueError(
                "initial alignment requires exactly one of xy_rotation or xy_similarity"
            )


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


@dataclass(frozen=True, slots=True)
class SimilarityScore:
    """Score and translation for one XY similarity candidate."""

    angle_deg: float
    scale: float
    translation_yx_px: tuple[float, float]
    correlation: float
    valid_overlap_fraction: float


@dataclass(frozen=True, slots=True)
class XYSimilarityEstimate:
    """Deterministic result and evidence from XY similarity estimation."""

    policy: str
    accepted: bool
    rejection_reason: str | None
    angle_deg: float | None
    scale: float | None
    translation_yx_px: tuple[float, float] | None
    identity_score: float | None
    score_gain: float | None
    peak_margin: float | None
    valid_overlap_fraction: float | None
    coarse_angle_scores: tuple[SimilarityScore, ...]
    refined_angle_scores: tuple[SimilarityScore, ...]
    coarse_scale_scores: tuple[SimilarityScore, ...]
    refined_scale_scores: tuple[SimilarityScore, ...]
    joint_scores: tuple[SimilarityScore, ...]


type InitialAlignmentResult = XYRotationEstimate | XYSimilarityEstimate


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


def _physical_isotropic_image(
    np: Any,
    ndimage: Any,
    feature: Any,
    geometry: ImageGeometry,
    target_spacing: float,
) -> Any:
    spacing_yx = geometry.spacing[1], geometry.spacing[0]
    output_shape = tuple(
        max(1, round((size - 1) * spacing / target_spacing) + 1)
        for size, spacing in zip(feature.shape, spacing_yx, strict=True)
    )
    zoom = tuple(
        output / source
        for output, source in zip(output_shape, feature.shape, strict=True)
    )
    return ndimage.zoom(
        feature,
        zoom=zoom,
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32, copy=False)


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
    translation_scale_yx: tuple[float, float],
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
    display_shift_yx = float(shift[0]), float(shift[1])
    aligned = ndimage.shift(
        rotated,
        shift=display_shift_yx,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    aligned_footprint = ndimage.shift(
        rotated_footprint,
        shift=display_shift_yx,
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
        translation_yx_px=(
            display_shift_yx[0] * translation_scale_yx[0],
            display_shift_yx[1] * translation_scale_yx[1],
        ),
        correlation=correlation,
        valid_overlap_fraction=float(overlap_count / overlap.size),
    )


def _best(scores: tuple[RotationScore, ...]) -> RotationScore:
    return min(
        scores,
        key=lambda item: (-item.correlation, abs(item.angle_deg), item.angle_deg),
    )


def _score_similarity_candidate(
    np: Any,
    ndimage: Any,
    phase_cross_correlation: Any,
    fixed: Any,
    moving: Any,
    fixed_footprint: Any,
    moving_footprint: Any,
    angle: float,
    scale: float,
    translation_scale_yx: tuple[float, float],
) -> SimilarityScore | None:
    center = (np.asarray(moving.shape, dtype=np.float64) - 1.0) / 2.0
    scale_matrix = np.eye(2, dtype=np.float64) / scale
    offset = center - scale_matrix @ center
    scaled = ndimage.affine_transform(
        moving,
        matrix=scale_matrix,
        offset=offset,
        output_shape=moving.shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    scaled_footprint = ndimage.affine_transform(
        moving_footprint,
        matrix=scale_matrix,
        offset=offset,
        output_shape=moving_footprint.shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    display_angle = -angle
    transformed = ndimage.rotate(
        scaled,
        angle=display_angle,
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    transformed_footprint = ndimage.rotate(
        scaled_footprint,
        angle=display_angle,
        reshape=False,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    shift, _, _ = phase_cross_correlation(
        fixed,
        transformed,
        upsample_factor=1,
        normalization=None,
    )
    display_shift_yx = float(shift[0]), float(shift[1])
    aligned = ndimage.shift(
        transformed,
        shift=display_shift_yx,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    aligned_footprint = ndimage.shift(
        transformed_footprint,
        shift=display_shift_yx,
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
    correlation = (
        -1.0
        if denominator <= 0.0 or not math.isfinite(denominator)
        else float(np.dot(fixed_values, moving_values) / denominator)
    )
    return SimilarityScore(
        angle_deg=float(angle),
        scale=float(scale),
        translation_yx_px=(
            display_shift_yx[0] * translation_scale_yx[0],
            display_shift_yx[1] * translation_scale_yx[1],
        ),
        correlation=correlation,
        valid_overlap_fraction=float(overlap_count / overlap.size),
    )


def _best_similarity(scores: tuple[SimilarityScore, ...]) -> SimilarityScore:
    return min(
        scores,
        key=lambda item: (
            -item.correlation,
            abs(item.scale - 1.0),
            abs(item.angle_deg),
            item.scale,
            item.angle_deg,
        ),
    )


def _rejected_similarity(reason: str) -> XYSimilarityEstimate:
    return XYSimilarityEstimate(
        policy=SIMILARITY_ESTIMATOR_POLICY,
        accepted=False,
        rejection_reason=reason,
        angle_deg=None,
        scale=None,
        translation_yx_px=None,
        identity_score=None,
        score_gain=None,
        peak_margin=None,
        valid_overlap_fraction=None,
        coarse_angle_scores=(),
        refined_angle_scores=(),
        coarse_scale_scores=(),
        refined_scale_scores=(),
        joint_scores=(),
    )


def estimate_xy_similarity(
    fixed_zyx: Any,
    moving_zyx: Any,
    policy: EstimatedXYSimilarityPolicy,
    *,
    fixed_geometry: ImageGeometry | None = None,
    moving_geometry: ImageGeometry | None = None,
) -> XYSimilarityEstimate:
    """Estimate a constrained XY similarity correction from registration MIPs."""
    if type(policy) is not EstimatedXYSimilarityPolicy:
        raise TypeError("policy must be an EstimatedXYSimilarityPolicy")
    np, ndimage, phase_cross_correlation = _scientific_runtime()
    try:
        fixed_feature = _projection_feature(np, ndimage, fixed_zyx, "fixed")
        moving_feature = _projection_feature(np, ndimage, moving_zyx, "moving")
    except ValueError as error:
        return _rejected_similarity(str(error))
    translation_scale_yx = (1.0, 1.0)
    if (fixed_geometry is None) != (moving_geometry is None):
        raise ValueError("fixed_geometry and moving_geometry must be supplied together")
    if fixed_geometry is not None and moving_geometry is not None:
        if (
            type(fixed_geometry) is not ImageGeometry
            or type(moving_geometry) is not ImageGeometry
        ):
            raise TypeError("geometries must be exact ImageGeometry values")
        if fixed_geometry.shape != tuple(fixed_zyx.shape):
            raise ValueError("fixed_geometry shape does not match fixed_zyx")
        if moving_geometry.shape != tuple(moving_zyx.shape):
            raise ValueError("moving_geometry shape does not match moving_zyx")
        if not bool(
            np.allclose(
                np.asarray(fixed_geometry.direction),
                np.asarray(moving_geometry.direction),
                rtol=0.0,
                atol=1e-8,
            )
        ):
            return _rejected_similarity("incompatible-fixed-moving-direction")
        target_spacing = min(
            fixed_geometry.spacing[0],
            fixed_geometry.spacing[1],
            moving_geometry.spacing[0],
            moving_geometry.spacing[1],
        )
        fixed_feature = _physical_isotropic_image(
            np, ndimage, fixed_feature, fixed_geometry, target_spacing
        )
        moving_feature = _physical_isotropic_image(
            np, ndimage, moving_feature, moving_geometry, target_spacing
        )
        translation_scale_yx = (
            target_spacing / moving_geometry.spacing[1],
            target_spacing / moving_geometry.spacing[0],
        )
    size = math.ceil(
        max(
            math.hypot(*fixed_feature.shape),
            math.hypot(*moving_feature.shape) * policy.scale_range[1],
        )
    )
    fixed = _center_pad(np, fixed_feature, size)
    moving = _center_pad(np, moving_feature, size)
    fixed_footprint = _center_pad(
        np, np.ones_like(fixed_feature, dtype=bool), size, value=False
    )
    moving_footprint = _center_pad(
        np, np.ones_like(moving_feature, dtype=np.uint8), size
    )

    def score(angle: float, scale: float) -> SimilarityScore | None:
        return _score_similarity_candidate(
            np,
            ndimage,
            phase_cross_correlation,
            fixed,
            moving,
            fixed_footprint,
            moving_footprint,
            angle,
            scale,
            translation_scale_yx,
        )

    def evaluate_angles(
        values: tuple[float, ...], scale: float
    ) -> tuple[SimilarityScore, ...]:
        items: list[SimilarityScore] = []
        for angle in values:
            item = score(angle, scale)
            if item is not None:
                items.append(item)
        return tuple(items)

    def evaluate_scales(
        angle: float, values: tuple[float, ...]
    ) -> tuple[SimilarityScore, ...]:
        items: list[SimilarityScore] = []
        for scale in values:
            item = score(angle, scale)
            if item is not None:
                items.append(item)
        return tuple(items)

    lower_angle, upper_angle = policy.search_range_deg
    coarse_angle = evaluate_angles(
        _candidate_angles(lower_angle, upper_angle, policy.coarse_step_deg), 1.0
    )
    if not coarse_angle:
        return _rejected_similarity("no-positive-valid-overlap")
    angle_winner = _best_similarity(coarse_angle)
    refined_angle = evaluate_angles(
        _candidate_angles(
            max(lower_angle, angle_winner.angle_deg - policy.coarse_step_deg),
            min(upper_angle, angle_winner.angle_deg + policy.coarse_step_deg),
            policy.refinement_step_deg,
        ),
        1.0,
    )
    angle_winner = _best_similarity(refined_angle or coarse_angle)
    lower_scale, upper_scale = policy.scale_range
    coarse_scale = evaluate_scales(
        angle_winner.angle_deg,
        _candidate_angles(lower_scale, upper_scale, policy.coarse_scale_step),
    )
    if not coarse_scale:
        return _rejected_similarity("no-positive-valid-overlap")
    scale_winner = _best_similarity(coarse_scale)
    refined_scale = evaluate_scales(
        angle_winner.angle_deg,
        _candidate_angles(
            max(lower_scale, scale_winner.scale - policy.coarse_scale_step),
            min(upper_scale, scale_winner.scale + policy.coarse_scale_step),
            policy.refinement_scale_step,
        ),
    )
    scale_winner = _best_similarity(refined_scale or coarse_scale)
    joint_scores = evaluate_angles(
        _candidate_angles(
            max(lower_angle, angle_winner.angle_deg - policy.coarse_step_deg),
            min(upper_angle, angle_winner.angle_deg + policy.coarse_step_deg),
            policy.refinement_step_deg,
        ),
        scale_winner.scale,
    )
    winner = _best_similarity(joint_scores or refined_scale or coarse_scale)
    identity = score(0.0, 1.0)
    identity_score = winner.correlation if identity is None else identity.correlation
    score_gain = winner.correlation - identity_score
    all_scores = (
        *coarse_angle,
        *refined_angle,
        *coarse_scale,
        *refined_scale,
        *joint_scores,
    )
    competitors = tuple(
        item
        for item in all_scores
        if abs(item.angle_deg - winner.angle_deg) >= policy.coarse_step_deg - 1e-12
        or abs(item.scale - winner.scale) >= policy.coarse_scale_step - 1e-12
    )
    peak_margin = (
        winner.correlation - _best_similarity(competitors).correlation
        if competitors
        else 1.0
    )
    reason: str | None = None
    if math.isclose(winner.angle_deg, lower_angle, abs_tol=1e-12) or math.isclose(
        winner.angle_deg, upper_angle, abs_tol=1e-12
    ):
        reason = "winner-at-search-boundary"
    elif math.isclose(winner.scale, lower_scale, abs_tol=1e-12) or math.isclose(
        winner.scale, upper_scale, abs_tol=1e-12
    ):
        reason = "winner-at-scale-boundary"
    elif (
        abs(winner.angle_deg) > policy.refinement_step_deg
        or abs(winner.scale - 1.0) > policy.refinement_scale_step
    ) and score_gain < policy.minimum_score_gain:
        reason = "insufficient-score-gain"
    elif peak_margin < policy.minimum_peak_margin:
        reason = "insufficient-peak-margin"
    return XYSimilarityEstimate(
        policy=SIMILARITY_ESTIMATOR_POLICY,
        accepted=reason is None,
        rejection_reason=reason,
        angle_deg=winner.angle_deg,
        scale=winner.scale,
        translation_yx_px=winner.translation_yx_px,
        identity_score=identity_score,
        score_gain=score_gain,
        peak_margin=peak_margin,
        valid_overlap_fraction=winner.valid_overlap_fraction,
        coarse_angle_scores=coarse_angle,
        refined_angle_scores=refined_angle,
        coarse_scale_scores=coarse_scale,
        refined_scale_scores=refined_scale,
        joint_scores=joint_scores,
    )


def estimate_xy_rotation(
    fixed_zyx: Any,
    moving_zyx: Any,
    policy: EstimatedXYRotationPolicy,
    *,
    fixed_geometry: ImageGeometry | None = None,
    moving_geometry: ImageGeometry | None = None,
) -> XYRotationEstimate:
    """Estimate an XY correction from fixed and moving registration channels.

    Geometry-aware calls resample projection features onto a physical-isotropic
    grid before scoring. Omitting both geometries retains pixel-space behavior for
    direct array use; supplying only one geometry is invalid.
    """
    if type(policy) is not EstimatedXYRotationPolicy:
        raise TypeError("policy must be an EstimatedXYRotationPolicy")
    np, ndimage, phase_cross_correlation = _scientific_runtime()
    try:
        fixed_feature = _projection_feature(np, ndimage, fixed_zyx, "fixed")
        moving_feature = _projection_feature(np, ndimage, moving_zyx, "moving")
    except ValueError as error:
        return _rejected(str(error))

    translation_scale_yx = (1.0, 1.0)
    if (fixed_geometry is None) != (moving_geometry is None):
        raise ValueError("fixed_geometry and moving_geometry must be supplied together")
    if fixed_geometry is not None and moving_geometry is not None:
        if (
            type(fixed_geometry) is not ImageGeometry
            or type(moving_geometry) is not ImageGeometry
        ):
            raise TypeError("geometries must be exact ImageGeometry values")
        if fixed_geometry.shape != tuple(fixed_zyx.shape):
            raise ValueError("fixed_geometry shape does not match fixed_zyx")
        if moving_geometry.shape != tuple(moving_zyx.shape):
            raise ValueError("moving_geometry shape does not match moving_zyx")
        if not bool(
            np.allclose(
                np.asarray(fixed_geometry.direction),
                np.asarray(moving_geometry.direction),
                rtol=0.0,
                atol=1e-8,
            )
        ):
            return _rejected("incompatible-fixed-moving-direction")
        target_spacing = min(
            fixed_geometry.spacing[0],
            fixed_geometry.spacing[1],
            moving_geometry.spacing[0],
            moving_geometry.spacing[1],
        )
        fixed_feature = _physical_isotropic_image(
            np, ndimage, fixed_feature, fixed_geometry, target_spacing
        )
        moving_feature = _physical_isotropic_image(
            np, ndimage, moving_feature, moving_geometry, target_spacing
        )
        translation_scale_yx = (
            target_spacing / moving_geometry.spacing[1],
            target_spacing / moving_geometry.spacing[0],
        )

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
            translation_scale_yx,
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


def _similarity_score_mapping(score: SimilarityScore) -> dict[str, object]:
    return {
        "angle_deg": score.angle_deg,
        "scale": score.scale,
        "translation_yx_px": list(score.translation_yx_px),
        "correlation": score.correlation,
        "valid_overlap_fraction": score.valid_overlap_fraction,
    }


def _result_source(result: InitialAlignmentResult) -> str:
    return "explicit" if result.policy == EXPLICIT_ROTATION_POLICY else "estimated"


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
        "source": _result_source(result),
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


def initial_alignment_config_parameters(
    policy: InitialAlignmentPolicy,
) -> dict[str, object]:
    """Return the canonical JSON-compatible initialization configuration."""
    if type(policy) is not InitialAlignmentPolicy:
        raise TypeError("policy must be an InitialAlignmentPolicy")
    rotation = policy.xy_rotation
    if type(rotation) is ExplicitXYRotationPolicy:
        value: dict[str, object] = {
            "mode": "explicit",
            "angle_deg": rotation.angle_deg,
        }
        return {"xy_rotation": value}
    elif type(rotation) is EstimatedXYRotationPolicy:
        value = {
            "mode": "estimate",
            "search_range_deg": list(rotation.search_range_deg),
            "coarse_step_deg": rotation.coarse_step_deg,
            "refinement_step_deg": rotation.refinement_step_deg,
            "minimum_score_gain": rotation.minimum_score_gain,
            "minimum_peak_margin": rotation.minimum_peak_margin,
        }
        return {"xy_rotation": value}
    similarity = policy.xy_similarity
    if type(similarity) is EstimatedXYSimilarityPolicy:
        return {
            "xy_similarity": {
                "mode": "estimate",
                "search_range_deg": list(similarity.search_range_deg),
                "coarse_step_deg": similarity.coarse_step_deg,
                "refinement_step_deg": similarity.refinement_step_deg,
                "scale_range": list(similarity.scale_range),
                "coarse_scale_step": similarity.coarse_scale_step,
                "refinement_scale_step": similarity.refinement_scale_step,
                "minimum_score_gain": similarity.minimum_score_gain,
                "minimum_peak_margin": similarity.minimum_peak_margin,
            }
        }
    raise TypeError("unsupported initial alignment policy")


def initial_alignment_result_parameters(
    result: InitialAlignmentResult,
) -> dict[str, object]:
    """Return canonical JSON-compatible realized initialization evidence."""
    if type(result) is XYRotationEstimate:
        return {
            "algorithm": result.policy,
            "source": _result_source(result),
            "accepted": result.accepted,
            "rejection_reason": result.rejection_reason,
            "angle_deg": result.angle_deg,
            "translation_yx_px": (
                None
                if result.translation_yx_px is None
                else list(result.translation_yx_px)
            ),
            "zero_angle_score": result.zero_angle_score,
            "score_gain": result.score_gain,
            "peak_margin": result.peak_margin,
            "valid_overlap_fraction": result.valid_overlap_fraction,
            "coarse_scores": [_score_mapping(item) for item in result.coarse_scores],
            "refined_scores": [_score_mapping(item) for item in result.refined_scores],
        }
    if type(result) is XYSimilarityEstimate:
        return {
            "algorithm": result.policy,
            "source": "estimated",
            "accepted": result.accepted,
            "rejection_reason": result.rejection_reason,
            "angle_deg": result.angle_deg,
            "scale": result.scale,
            "translation_yx_px": (
                None
                if result.translation_yx_px is None
                else list(result.translation_yx_px)
            ),
            "identity_score": result.identity_score,
            "score_gain": result.score_gain,
            "peak_margin": result.peak_margin,
            "valid_overlap_fraction": result.valid_overlap_fraction,
            "coarse_angle_scores": [
                _similarity_score_mapping(item) for item in result.coarse_angle_scores
            ],
            "refined_angle_scores": [
                _similarity_score_mapping(item) for item in result.refined_angle_scores
            ],
            "coarse_scale_scores": [
                _similarity_score_mapping(item) for item in result.coarse_scale_scores
            ],
            "refined_scale_scores": [
                _similarity_score_mapping(item) for item in result.refined_scale_scores
            ],
            "joint_scores": [
                _similarity_score_mapping(item) for item in result.joint_scores
            ],
        }
    raise TypeError("result must be an XY rotation or similarity estimate")


def _parse_similarity_result(
    value: object,
    policy: EstimatedXYSimilarityPolicy,
) -> XYSimilarityEstimate:
    expected = initial_alignment_result_parameters(_rejected_similarity("x"))
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("initial alignment result keys are not canonical")

    def finite(item: object, name: str) -> float:
        if type(item) is not float or not math.isfinite(item):
            raise ValueError(f"initial alignment {name} must be a finite float")
        return item

    def optional_finite(name: str) -> float | None:
        item = value[name]
        return None if item is None else finite(item, name)

    def scores(name: str) -> tuple[SimilarityScore, ...]:
        raw = value[name]
        if not isinstance(raw, list):
            raise TypeError(f"initial alignment {name} must be a JSON array")
        parsed: list[SimilarityScore] = []
        keys = {
            "angle_deg",
            "scale",
            "translation_yx_px",
            "correlation",
            "valid_overlap_fraction",
        }
        for item in raw:
            if not isinstance(item, dict) or set(item) != keys:
                raise ValueError(f"initial alignment {name} entry is not canonical")
            translation = item["translation_yx_px"]
            if not isinstance(translation, list) or len(translation) != 2:
                raise ValueError("initial alignment score translation is invalid")
            overlap = finite(item["valid_overlap_fraction"], "valid_overlap_fraction")
            if not 0.0 <= overlap <= 1.0:
                raise ValueError("initial alignment overlap must be within [0, 1]")
            scale = finite(item["scale"], "scale")
            if scale <= 0.0:
                raise ValueError("initial alignment scale must be positive")
            parsed.append(
                SimilarityScore(
                    angle_deg=finite(item["angle_deg"], "angle_deg"),
                    scale=scale,
                    translation_yx_px=(
                        finite(translation[0], "translation_yx_px[0]"),
                        finite(translation[1], "translation_yx_px[1]"),
                    ),
                    correlation=finite(item["correlation"], "correlation"),
                    valid_overlap_fraction=overlap,
                )
            )
        return tuple(parsed)

    if (
        value["algorithm"] != SIMILARITY_ESTIMATOR_POLICY
        or value["source"] != "estimated"
    ):
        raise ValueError("estimated initial alignment policy is inconsistent")
    if type(value["accepted"]) is not bool:
        raise TypeError("initial alignment accepted must be a built-in boolean")
    rejection = value["rejection_reason"]
    if rejection is not None and (type(rejection) is not str or not rejection):
        raise TypeError("initial alignment rejection_reason must be text or null")
    raw_translation = value["translation_yx_px"]
    translation = (
        None
        if raw_translation is None
        else (
            finite(raw_translation[0], "translation_yx_px[0]"),
            finite(raw_translation[1], "translation_yx_px[1]"),
        )
        if isinstance(raw_translation, list) and len(raw_translation) == 2
        else None
    )
    if raw_translation is not None and translation is None:
        raise ValueError("initial alignment translation_yx_px is invalid")
    result = XYSimilarityEstimate(
        policy=SIMILARITY_ESTIMATOR_POLICY,
        accepted=value["accepted"],
        rejection_reason=rejection,
        angle_deg=optional_finite("angle_deg"),
        scale=optional_finite("scale"),
        translation_yx_px=translation,
        identity_score=optional_finite("identity_score"),
        score_gain=optional_finite("score_gain"),
        peak_margin=optional_finite("peak_margin"),
        valid_overlap_fraction=optional_finite("valid_overlap_fraction"),
        coarse_angle_scores=scores("coarse_angle_scores"),
        refined_angle_scores=scores("refined_angle_scores"),
        coarse_scale_scores=scores("coarse_scale_scores"),
        refined_scale_scores=scores("refined_scale_scores"),
        joint_scores=scores("joint_scores"),
    )
    all_scores = (
        *result.coarse_angle_scores,
        *result.refined_angle_scores,
        *result.coarse_scale_scores,
        *result.refined_scale_scores,
        *result.joint_scores,
    )
    realized = (
        result.angle_deg,
        result.scale,
        result.translation_yx_px,
        result.identity_score,
        result.score_gain,
        result.peak_margin,
        result.valid_overlap_fraction,
    )
    if not all_scores:
        if result.accepted or any(item is not None for item in realized):
            raise ValueError("scoreless estimate has realized evidence")
        if result.rejection_reason not in {
            "fixed-input-must-be-zyx",
            "moving-input-must-be-zyx",
            "nonfinite-fixed-input",
            "nonfinite-moving-input",
            "constant-fixed-input",
            "constant-moving-input",
            "incompatible-fixed-moving-direction",
            "no-positive-valid-overlap",
        }:
            raise ValueError("scoreless estimate has an unknown rejection reason")
    else:
        if any(item is None for item in realized):
            raise ValueError("scored estimate is missing realized evidence")
        assert result.angle_deg is not None
        assert result.scale is not None
        assert result.translation_yx_px is not None
        assert result.identity_score is not None
        assert result.score_gain is not None
        assert result.peak_margin is not None
        assert result.valid_overlap_fraction is not None
        if not 0.0 <= result.valid_overlap_fraction <= 1.0:
            raise ValueError("initial alignment overlap must be within [0, 1]")
        winner = _best_similarity(result.joint_scores or result.refined_scale_scores)
        if (
            winner.angle_deg != result.angle_deg
            or winner.scale != result.scale
            or winner.translation_yx_px != result.translation_yx_px
            or winner.valid_overlap_fraction != result.valid_overlap_fraction
        ):
            raise ValueError("estimated winner disagrees with score curves")
        if not math.isclose(
            result.score_gain,
            winner.correlation - result.identity_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("estimated score gain disagrees with score curves")
        competitors = tuple(
            item
            for item in all_scores
            if abs(item.angle_deg - winner.angle_deg) >= policy.coarse_step_deg - 1e-12
            or abs(item.scale - winner.scale) >= policy.coarse_scale_step - 1e-12
        )
        expected_margin = (
            winner.correlation - _best_similarity(competitors).correlation
            if competitors
            else 1.0
        )
        if not math.isclose(
            result.peak_margin, expected_margin, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("estimated peak margin disagrees with score curves")
        lower_angle, upper_angle = policy.search_range_deg
        lower_scale, upper_scale = policy.scale_range
        reason: str | None = None
        if math.isclose(result.angle_deg, lower_angle, abs_tol=1e-12) or math.isclose(
            result.angle_deg, upper_angle, abs_tol=1e-12
        ):
            reason = "winner-at-search-boundary"
        elif math.isclose(result.scale, lower_scale, abs_tol=1e-12) or math.isclose(
            result.scale, upper_scale, abs_tol=1e-12
        ):
            reason = "winner-at-scale-boundary"
        elif (
            abs(result.angle_deg) > policy.refinement_step_deg
            or abs(result.scale - 1.0) > policy.refinement_scale_step
        ) and result.score_gain < policy.minimum_score_gain:
            reason = "insufficient-score-gain"
        elif result.peak_margin < policy.minimum_peak_margin:
            reason = "insufficient-peak-margin"
        if result.rejection_reason != reason or result.accepted != (reason is None):
            raise ValueError("estimated acceptance disagrees with confidence policy")
    if initial_alignment_result_parameters(result) != value:
        raise ValueError("initial alignment result is not canonical")
    return result


def parse_initial_alignment_result(
    value: object,
    policy: InitialAlignmentPolicy,
) -> InitialAlignmentResult:
    """Parse strict mode-aware realized initial-alignment evidence."""
    if type(policy.xy_similarity) is EstimatedXYSimilarityPolicy:
        return _parse_similarity_result(value, policy.xy_similarity)
    expected_keys = frozenset(initial_alignment_result_parameters(_rejected("x")))
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("initial alignment result keys are not canonical")

    def finite(item: object, name: str) -> float:
        if type(item) is not float or not math.isfinite(item):
            raise ValueError(f"initial alignment {name} must be a finite float")
        return item

    def optional_finite(name: str) -> float | None:
        item = value[name]
        return None if item is None else finite(item, name)

    def parse_scores(name: str) -> tuple[RotationScore, ...]:
        raw = value[name]
        if not isinstance(raw, list):
            raise TypeError(f"initial alignment {name} must be a JSON array")
        parsed: list[RotationScore] = []
        score_keys = {
            "angle_deg",
            "translation_yx_px",
            "correlation",
            "valid_overlap_fraction",
        }
        for item in raw:
            if not isinstance(item, dict) or set(item) != score_keys:
                raise ValueError(f"initial alignment {name} entry is not canonical")
            translation = item["translation_yx_px"]
            if not isinstance(translation, list) or len(translation) != 2:
                raise ValueError("initial alignment score translation is invalid")
            overlap = finite(item["valid_overlap_fraction"], "valid_overlap_fraction")
            if not 0.0 <= overlap <= 1.0:
                raise ValueError("initial alignment overlap must be within [0, 1]")
            parsed.append(
                RotationScore(
                    angle_deg=finite(item["angle_deg"], "angle_deg"),
                    translation_yx_px=(
                        finite(translation[0], "translation_yx_px[0]"),
                        finite(translation[1], "translation_yx_px[1]"),
                    ),
                    correlation=finite(item["correlation"], "correlation"),
                    valid_overlap_fraction=overlap,
                )
            )
        return tuple(parsed)

    if type(value["accepted"]) is not bool:
        raise TypeError("initial alignment accepted must be a built-in boolean")
    rejection = value["rejection_reason"]
    if rejection is not None and (type(rejection) is not str or not rejection):
        raise TypeError("initial alignment rejection_reason must be text or null")
    raw_translation = value["translation_yx_px"]
    if raw_translation is None:
        translation = None
    elif isinstance(raw_translation, list) and len(raw_translation) == 2:
        translation = (
            finite(raw_translation[0], "translation_yx_px[0]"),
            finite(raw_translation[1], "translation_yx_px[1]"),
        )
    else:
        raise ValueError("initial alignment translation_yx_px is invalid")
    result = XYRotationEstimate(
        policy=str(value["algorithm"]),
        accepted=value["accepted"],
        rejection_reason=rejection,
        angle_deg=optional_finite("angle_deg"),
        translation_yx_px=translation,
        zero_angle_score=optional_finite("zero_angle_score"),
        score_gain=optional_finite("score_gain"),
        peak_margin=optional_finite("peak_margin"),
        valid_overlap_fraction=optional_finite("valid_overlap_fraction"),
        coarse_scores=parse_scores("coarse_scores"),
        refined_scores=parse_scores("refined_scores"),
    )
    rotation = policy.xy_rotation
    if type(rotation) is ExplicitXYRotationPolicy:
        if (
            result.policy != EXPLICIT_ROTATION_POLICY
            or value["source"] != "explicit"
            or not result.accepted
            or result.rejection_reason is not None
            or result.angle_deg != rotation.angle_deg
            or result.translation_yx_px != (0.0, 0.0)
            or any(
                item is not None
                for item in (
                    result.zero_angle_score,
                    result.score_gain,
                    result.peak_margin,
                    result.valid_overlap_fraction,
                )
            )
            or result.coarse_scores
            or result.refined_scores
        ):
            raise ValueError("explicit initial alignment result is inconsistent")
    elif type(rotation) is EstimatedXYRotationPolicy:
        if result.policy != ROTATION_ESTIMATOR_POLICY or value["source"] != "estimated":
            raise ValueError("estimated initial alignment policy is inconsistent")
        if result.accepted != (result.rejection_reason is None):
            raise ValueError("estimated acceptance and rejection reason disagree")
        realized_values = (
            result.angle_deg,
            result.translation_yx_px,
            result.zero_angle_score,
            result.score_gain,
            result.peak_margin,
            result.valid_overlap_fraction,
        )
        scores = (*result.coarse_scores, *result.refined_scores)
        if not scores:
            if result.accepted or any(item is not None for item in realized_values):
                raise ValueError("scoreless estimate has realized evidence")
            allowed_rejections = {
                "fixed-input-must-be-zyx",
                "moving-input-must-be-zyx",
                "nonfinite-fixed-input",
                "nonfinite-moving-input",
                "constant-fixed-input",
                "constant-moving-input",
                "incompatible-fixed-moving-direction",
                "no-positive-valid-overlap",
            }
            if result.rejection_reason not in allowed_rejections:
                raise ValueError("scoreless estimate has an unknown rejection reason")
        else:
            if any(item is None for item in realized_values):
                raise ValueError("scored estimate is missing realized evidence")
            assert result.angle_deg is not None
            assert result.translation_yx_px is not None
            assert result.zero_angle_score is not None
            assert result.score_gain is not None
            assert result.peak_margin is not None
            assert result.valid_overlap_fraction is not None
            if not 0.0 <= result.valid_overlap_fraction <= 1.0:
                raise ValueError("initial alignment overlap must be within [0, 1]")
            lower, upper = rotation.search_range_deg
            if any(
                item.angle_deg < lower - 1e-12 or item.angle_deg > upper + 1e-12
                for item in scores
            ):
                raise ValueError("estimated score angle is outside the search range")
            raw_winner = _best(result.refined_scores or result.coarse_scores)
            normalized_to_zero = (
                abs(raw_winner.angle_deg) <= rotation.refinement_step_deg
                and result.angle_deg == 0.0
            )
            if not normalized_to_zero and result.angle_deg != raw_winner.angle_deg:
                raise ValueError("estimated winner disagrees with score curves")
            matching_winners = tuple(
                item
                for item in scores
                if item.angle_deg == result.angle_deg
                and item.translation_yx_px == result.translation_yx_px
                and item.valid_overlap_fraction == result.valid_overlap_fraction
            )
            if matching_winners:
                winner_score = matching_winners[0].correlation
            elif normalized_to_zero and result.score_gain == 0.0:
                winner_score = result.zero_angle_score
            else:
                raise ValueError("estimated winner is absent from score curves")
            if not math.isclose(
                result.score_gain,
                winner_score - result.zero_angle_score,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("estimated score gain disagrees with score curves")
            competitors = tuple(
                item
                for item in scores
                if abs(item.angle_deg - result.angle_deg)
                >= rotation.coarse_step_deg - 1e-12
            )
            expected_margin = (
                winner_score - _best(competitors).correlation if competitors else 1.0
            )
            if not math.isclose(
                result.peak_margin,
                expected_margin,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("estimated peak margin disagrees with score curves")
            reason: str | None = None
            if math.isclose(result.angle_deg, lower, abs_tol=1e-12) or math.isclose(
                result.angle_deg, upper, abs_tol=1e-12
            ):
                reason = "winner-at-search-boundary"
            elif (
                abs(result.angle_deg) > rotation.refinement_step_deg
                and result.score_gain < rotation.minimum_score_gain
            ):
                reason = "insufficient-score-gain"
            elif result.peak_margin < rotation.minimum_peak_margin:
                reason = "insufficient-peak-margin"
            if result.rejection_reason != reason:
                raise ValueError(
                    "estimated acceptance disagrees with confidence policy"
                )
    else:  # pragma: no cover - protected by InitialAlignmentPolicy
        raise TypeError("unsupported xy_rotation policy")
    if initial_alignment_result_parameters(result) != value:
        raise ValueError("initial alignment result is not canonical")
    return result


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


def _score_panel(np: Any, estimate: InitialAlignmentResult) -> Any:
    panel = np.zeros((256, 256), dtype=np.uint8)
    if isinstance(estimate, XYRotationEstimate):
        scores = (*estimate.coarse_scores, *estimate.refined_scores)
    else:
        scores = (
            *estimate.coarse_angle_scores,
            *estimate.refined_angle_scores,
            *estimate.joint_scores,
        )
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
    estimate: InitialAlignmentResult,
    *,
    fixed_geometry: ImageGeometry | None = None,
    moving_geometry: ImageGeometry | None = None,
) -> Any:
    """Render a deterministic grayscale QC montage without Matplotlib."""
    np, ndimage, _ = _scientific_runtime()
    fixed = np.max(np.asarray(fixed_zyx, dtype=np.float32), axis=0)
    moving = np.max(np.asarray(moving_zyx, dtype=np.float32), axis=0)
    display_translation_yx = estimate.translation_yx_px
    if fixed_geometry is not None and moving_geometry is not None:
        target_spacing = min(
            fixed_geometry.spacing[0],
            fixed_geometry.spacing[1],
            moving_geometry.spacing[0],
            moving_geometry.spacing[1],
        )
        fixed = _physical_isotropic_image(
            np, ndimage, fixed, fixed_geometry, target_spacing
        )
        moving = _physical_isotropic_image(
            np, ndimage, moving, moving_geometry, target_spacing
        )
        if display_translation_yx is not None:
            display_translation_yx = (
                display_translation_yx[0] * moving_geometry.spacing[1] / target_spacing,
                display_translation_yx[1] * moving_geometry.spacing[0] / target_spacing,
            )
    visible_scale = (
        estimate.scale
        if isinstance(estimate, XYSimilarityEstimate) and estimate.scale is not None
        else 1.0
    )
    size = math.ceil(
        max(math.hypot(*fixed.shape), math.hypot(*moving.shape) * visible_scale)
    )
    fixed = _center_pad(np, fixed, size)
    moving = _center_pad(np, moving, size)
    aligned = moving
    if estimate.angle_deg is not None and display_translation_yx is not None:
        if isinstance(estimate, XYSimilarityEstimate) and estimate.scale is not None:
            center = (np.asarray(moving.shape, dtype=np.float64) - 1.0) / 2.0
            matrix = np.eye(2, dtype=np.float64) / estimate.scale
            aligned = ndimage.affine_transform(
                aligned,
                matrix=matrix,
                offset=center - matrix @ center,
                output_shape=aligned.shape,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
        aligned = ndimage.rotate(
            aligned,
            angle=-estimate.angle_deg,
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        aligned = ndimage.shift(
            aligned,
            shift=display_translation_yx,
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
    "EXPLICIT_ROTATION_POLICY",
    "ROTATION_ESTIMATOR_POLICY",
    "SIMILARITY_ESTIMATOR_POLICY",
    "EstimatedXYRotationPolicy",
    "EstimatedXYSimilarityPolicy",
    "ExplicitXYRotationPolicy",
    "InitialAlignmentPolicy",
    "RotationScore",
    "SimilarityScore",
    "XYRotationEstimate",
    "XYRotationPolicy",
    "XYSimilarityEstimate",
    "estimate_xy_rotation",
    "estimate_xy_similarity",
    "initial_alignment_config_parameters",
    "initial_alignment_parameters",
    "initial_alignment_result_parameters",
    "parse_initial_alignment_result",
    "render_initial_alignment_qc",
]
