from __future__ import annotations

import logging
import os
from typing import Any

from cyexm.runtime import NativeThreadConfig, configure_native_threads

logger = logging.getLogger(__name__)

_SUPPORTED_REGISTRATION_TYPES = frozenset(
    {"Translation", "Rigid", "Similarity", "Affine", "TRSAA"}
)
_SUPPORTED_ACCURACIES = frozenset({"high", "medium", "low", "dry run"})
_TRSAA_ITERATIONS = {
    "high": (1000, 500, 250, 100),
    "medium": (500, 250, 100, 50),
    "low": (100, 50, 25, 10),
}


def _scientific_runtime() -> tuple[Any, Any]:
    import ants
    import numpy as np

    return ants, np


def _configured_runtime(
    native_threads: NativeThreadConfig | None,
) -> tuple[Any, Any]:
    resolved = configure_native_threads(native_threads)
    logger.info("Native thread settings: %s", resolved.describe())
    return _scientific_runtime()


def _validate_policy(registration_type: str, accuracy: str) -> None:
    if registration_type not in _SUPPORTED_REGISTRATION_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_REGISTRATION_TYPES))
        raise ValueError(
            f"Unsupported registration type: {registration_type}. "
            f"Supported types are: {supported}."
        )
    if accuracy not in _SUPPORTED_ACCURACIES:
        supported = ", ".join(sorted(_SUPPORTED_ACCURACIES))
        raise ValueError(
            f"Unsupported accuracy: {accuracy}. Supported values are: {supported}."
        )


def register_image(
    moving_image: Any,
    fixed_image: Any,
    fixed_mask: Any | None = None,
    moving_mask: Any | None = None,
    registration_type: str = "TRSAA",
    accuracy: str = "high",
    verbose: bool = False,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> tuple[object, object | None, object]:
    """Return transformed scalar image, optional transformed mask, and affine."""
    _validate_policy(registration_type, accuracy)
    ants, np = _configured_runtime(native_threads)

    if isinstance(fixed_image, np.ndarray):
        fixed_image = ants.from_numpy(fixed_image)
    if isinstance(moving_image, np.ndarray):
        moving_image = ants.from_numpy(moving_image)
    if fixed_image.dimension != moving_image.dimension:
        raise ValueError("Both images must have the same number of dimensions.")

    kwargs: dict[str, object] = {
        "fixed": fixed_image,
        "moving": moving_image,
        "fixed_mask": fixed_mask,
        "moving_mask": moving_mask,
        "type_of_transform": registration_type,
        "aff_metric": "mattes",
        "aff_sampling": 32,
        "aff_random_sampling_rate": 0.50,
        "verbose": verbose,
        "mask_all_stages": True,
    }
    if registration_type == "TRSAA" and accuracy != "dry run":
        kwargs["reg_iterations"] = _TRSAA_ITERATIONS[accuracy]
        kwargs["smoothing_sigmas"] = (3, 2, 1, 0)
        kwargs["shrink_factors"] = (8, 4, 2, 1)
    if accuracy == "dry run":
        kwargs["type_of_transform"] = "Rigid"
        kwargs["reg_iterations"] = (1,)
        kwargs["smoothing_sigmas"] = (0,)
        kwargs["shrink_factors"] = (8,)
        kwargs["aff_random_sampling_rate"] = 0.1

    iterations = kwargs.get("reg_iterations")
    if iterations is None:
        logger.info("Using %s registration.", kwargs["type_of_transform"])
    else:
        logger.info(
            "Using %s registration with %s iterations.",
            kwargs["type_of_transform"],
            iterations,
        )

    registered = ants.registration(**kwargs)
    transform = ants.read_transform(filename=registered["fwdtransforms"][0])
    transformed_image = transform.apply_to_image(
        image=moving_image,
        reference=fixed_image,
        interpolation="linear",
    )

    transformed_mask: object | None = None
    if moving_mask is not None:
        transformed_mask = transform.apply_to_image(
            image=moving_mask,
            reference=fixed_image,
            interpolation="nearestNeighbor",
        )
        transformed_mask = ants.threshold_image(
            transformed_mask,
            low_thresh=0.5,
            high_thresh=1.0,
            inval=1,
            outval=0,
        )

    return transformed_image, transformed_mask, transform


def inspect_affine_transform(
    affine_transform: Any,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> None:
    """Print the translation, rotation, shear, scale, and full affine matrix."""
    ants, np = _configured_runtime(native_threads)
    if isinstance(affine_transform, (str, os.PathLike)):
        affine_transform = ants.read_transform(str(affine_transform))

    transform_parameters = affine_transform.parameters
    center_of_rotation = affine_transform.fixed_parameters
    affine_matrix = transform_parameters[:9].reshape(3, 3)
    translation_vector = transform_parameters[9:]
    offset = (
        translation_vector + center_of_rotation - affine_matrix @ center_of_rotation
    )

    final_matrix = np.eye(4)
    final_matrix[:3, :3] = affine_matrix
    final_matrix[:3, 3] = offset
    logger.info("Affine Transform Matrix:\n%s", final_matrix)
    print("Complete Affine 4x4 matrix:\n", final_matrix)

    _extract_translation(final_matrix)
    _extract_rotation(final_matrix)
    _extract_shear(final_matrix)
    _extract_scale(final_matrix)


def _extract_scale(affine_matrix: Any) -> None:
    """Print scale factors from a supplied 4-by-4 affine matrix."""
    import numpy as np
    from scipy.linalg import polar, rq

    polar_decomposed = polar(a=affine_matrix[:3, :3])
    scaling_shear = polar_decomposed[1]
    scale_matrix, _ = rq(a=scaling_shear)
    scale_vector = np.abs(np.diagonal(a=scale_matrix))
    scale_labels = ["Z", "Y", "X"]
    for label, scale in zip(scale_labels, scale_vector, strict=True):
        logger.info("%s Scale: %.2f-fold", label, scale)
        print(f"{label} Scale: {scale:.2f}-fold")


def _extract_shear(affine_matrix: Any) -> None:
    """Print shear angles from a supplied 4-by-4 affine matrix."""
    import numpy as np
    from scipy.linalg import rq

    shear, _ = rq(a=affine_matrix[:3, :3])
    _, sy, sz = np.diag(v=shear)
    shear_xy = shear[0, 1] / sy if sy != 0 else 0
    shear_xz = shear[0, 2] / sz if sz != 0 else 0
    shear_yz = shear[1, 2] / sz if sz != 0 else 0

    angles = [
        np.degrees(np.arctan(shear_xy)),
        np.degrees(np.arctan(shear_xz)),
        np.degrees(np.arctan(shear_yz)),
    ]
    for label, angle in zip(["XY", "XZ", "YZ"], angles, strict=True):
        logger.info("Shear Angle %s: %.2f degrees", label, angle)
        print(f"Shear angle {label}: {angle:.2f} degrees")


def _extract_translation(affine_matrix: Any) -> None:
    """Print translations from a supplied 4-by-4 affine matrix."""
    translation = affine_matrix[:3, 3]
    for label, distance in zip(["Z", "Y", "X"], translation, strict=True):
        logger.info("Translation %s: %.2f voxels", label, distance)
        print(f"Translation distance in {label}: {distance:.2f} voxels")


def _extract_rotation(affine_matrix: Any) -> None:
    """Print Euler rotations from a supplied 4-by-4 affine matrix."""
    import numpy as np
    from scipy.linalg import polar
    from scipy.spatial.transform import Rotation

    rotation, _ = polar(a=affine_matrix[:3, :3])
    if np.linalg.det(rotation) < 0:
        logger.warning(
            "Transform contains reflection (negative determinant). "
            "Reporting zero rotation."
        )
        print("Warning: Transform contains reflection (negative determinant).")
        euler_angles_deg = np.array([0.0, 0.0, 0.0])
    else:
        euler_angles_deg = Rotation.from_matrix(matrix=rotation).as_euler(
            seq="xyz", degrees=True
        )

    for axis, angle in zip(
        ["X (roll)", "Y (pitch)", "Z (yaw)"], euler_angles_deg, strict=True
    ):
        logger.info("Rotation %s: %.2f degrees", axis, angle)
        print(f"Rotation around {axis} axis: {angle:.2f} degrees")


def transform_image(
    moving_image: Any,
    fixed_image: Any,
    affine_transform: Any,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Apply an affine to a scalar image and match its source histogram."""
    ants, np = _configured_runtime(native_threads)
    if isinstance(fixed_image, np.ndarray):
        fixed_image = ants.from_numpy(fixed_image)
    if isinstance(moving_image, np.ndarray):
        moving_image = ants.from_numpy(moving_image)

    warped_image = affine_transform.apply_to_image(
        moving_image,
        reference=fixed_image,
        interpolation="linear",
    )
    return ants.histogram_match_image(warped_image, moving_image)


__all__ = ["inspect_affine_transform", "register_image", "transform_image"]
