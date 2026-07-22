from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from cyexm._nonlinear_metric import resolve_nonlinear_metric
from cyexm.io.images import external_to_ants_array
from cyexm.registration._native_paths import discard_path
from cyexm.runtime import NativeThreadConfig, configure_native_threads

logger = logging.getLogger(__name__)

_SUPPORTED_REGISTRATION_TYPES = frozenset({"Elastic", "SyNOnly", "SyN"})
_ITERATION_SCHEDULES = {
    "high": (100, 70, 50),
    "medium": (50, 35, 25),
    "low": (5, 5, 5),
    "dry run": (1, 1, 1),
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
    if accuracy not in _ITERATION_SCHEDULES:
        supported = ", ".join(sorted(_ITERATION_SCHEDULES))
        raise ValueError(
            f"Unsupported accuracy: {accuracy}. Supported values are: {supported}."
        )


def _is_displacement_field(path: Path) -> bool:
    return path.name.endswith((".nii", ".nii.gz"))


def _from_external_array(ants: Any, array: Any, reference: Any | None = None) -> Any:
    kwargs: dict[str, object] = {}
    if reference is not None:
        kwargs = {
            "origin": reference.origin,
            "spacing": reference.spacing,
            "direction": reference.direction,
        }
    return ants.from_numpy(external_to_ants_array(array), **kwargs)


def _materialize_forward_warp(
    ants: Any,
    *,
    fixed_image: Any,
    moving_image: Any,
    forward_transforms: object,
    workspace: Path,
) -> Path:
    if not isinstance(forward_transforms, list | tuple) or not forward_transforms:
        raise FileNotFoundError("ANTs registration did not produce a forward warp")
    transform_paths = [Path(path) for path in forward_transforms]
    for path in transform_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"ANTs forward warp component does not exist: {path}"
            )
    if len(transform_paths) == 1 and _is_displacement_field(transform_paths[0]):
        return transform_paths[0]

    descriptor, prefix_text = tempfile.mkstemp(
        dir=workspace,
        prefix="cyexm-composite-",
        suffix="-",
    )
    os.close(descriptor)
    prefix = Path(prefix_text)
    discard_path(prefix)
    composed = ants.apply_transforms(
        fixed=fixed_image,
        moving=moving_image,
        transformlist=[str(path) for path in transform_paths],
        whichtoinvert=[False] * len(transform_paths),
        interpolator="linear",
        compose=str(prefix),
    )
    if composed is None:
        raise FileNotFoundError("ANTs did not materialize the composite forward warp")
    warp_path = Path(composed)
    if not warp_path.is_file():
        raise FileNotFoundError(
            f"ANTs composite forward warp does not exist: {warp_path}"
        )
    return warp_path


def _copy_disposable_forward_warp(
    ants: Any,
    source: Path,
    *,
    workspace: Path,
) -> Path:
    descriptor, candidate_text = tempfile.mkstemp(
        dir=workspace.parent,
        prefix="cyexm-forward-warp-",
        suffix=".nii.gz",
    )
    os.close(descriptor)
    candidate = Path(candidate_text)
    try:
        shutil.copyfile(source, candidate)
        ants.image_read(str(candidate))
    except BaseException:
        discard_path(candidate)
        raise
    return candidate


def register_image(
    moving_image: Any,
    fixed_image: Any,
    moving_mask: Any | None = None,
    fixed_mask: Any | None = None,
    registration_type: str = "SyNOnly",
    accuracy: str = "high",
    verbose: bool = False,
    *,
    metric: object = "mattes",
    radius: object = None,
    native_threads: NativeThreadConfig | None = None,
) -> tuple[object, Path]:
    """Return the histogram-matched scalar and one materialized forward warp."""
    _validate_policy(registration_type, accuracy)
    resolved_metric = resolve_nonlinear_metric(metric, radius)
    ants, np = _configured_runtime(native_threads)

    if isinstance(fixed_image, np.ndarray):
        fixed_image = _from_external_array(ants, fixed_image)
    if isinstance(moving_image, np.ndarray):
        moving_image = _from_external_array(ants, moving_image)
    if isinstance(fixed_mask, np.ndarray):
        fixed_mask = _from_external_array(ants, fixed_mask, fixed_image)
    if isinstance(moving_mask, np.ndarray):
        moving_mask = _from_external_array(ants, moving_mask, moving_image)
    if fixed_image.dimension != moving_image.dimension:
        raise ValueError("Both images must have the same number of dimensions.")

    kwargs: dict[str, object] = {
        "fixed": fixed_image,
        "moving": moving_image,
        "mask": fixed_mask,
        "moving_mask": moving_mask,
        "type_of_transform": registration_type,
        "flow_sigma": 3.0,
        "total_sigma": 0.5,
        "syn_metric": resolved_metric.metric,
        "syn_sampling": resolved_metric.syn_sampling,
        "singleprecision": False,
        "aff_random_sampling_rate": 1.0,
        "verbose": verbose,
        "initial_transform": "Identity",
        "mask_all_stages": True,
        "reg_iterations": _ITERATION_SCHEDULES[accuracy],
    }
    logger.info(
        "Using %s registration with %s iterations.",
        registration_type,
        kwargs["reg_iterations"],
    )
    candidate: Path | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix="cyexm-nonlinear-native-",
            ignore_cleanup_errors=True,
        ) as workspace_text:
            workspace = Path(workspace_text)
            kwargs["outprefix"] = str(workspace / "ants-")
            registered = ants.registration(**kwargs)
            native_warp = _materialize_forward_warp(
                ants,
                fixed_image=fixed_image,
                moving_image=moving_image,
                forward_transforms=registered["fwdtransforms"],
                workspace=workspace,
            )

            transformed_image = ants.resample_image_to_target(
                image=registered["warpedmovout"],
                target=fixed_image,
                interp_type="linear",
            )
            transformed_image = ants.histogram_match_image(
                source_image=transformed_image,
                reference_image=moving_image,
            )
            candidate = _copy_disposable_forward_warp(
                ants,
                native_warp,
                workspace=workspace,
            )
        return transformed_image, candidate
    except BaseException:
        discard_path(candidate)
        raise


def transform_image(
    moving_image: Any,
    fixed_image: Any,
    transformed_image: Any,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> object:
    """Apply an existing forward warp and match the source histogram."""
    ants, np = _configured_runtime(native_threads)
    if isinstance(fixed_image, np.ndarray):
        fixed_image = _from_external_array(ants, fixed_image)
    if isinstance(moving_image, np.ndarray):
        moving_image = _from_external_array(ants, moving_image)

    transformlist = list(transformed_image["fwdtransforms"])
    warped_image = ants.apply_transforms(
        fixed=fixed_image,
        moving=moving_image,
        transformlist=transformlist,
        whichtoinvert=[False] * len(transformlist),
        interpolator="linear",
    )
    return ants.histogram_match_image(warped_image, moving_image)


def inspect_warp_transform(
    image_path: str | Path,
    transform_path: str | Path,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> dict[str, float | int]:
    """Print displacement-magnitude statistics inside an image-derived mask."""
    ants, np = _configured_runtime(native_threads)
    import tifffile

    warp = ants.image_read(filename=str(transform_path))
    image = ants.from_numpy(data=external_to_ants_array(tifffile.imread(image_path)))
    mask = ants.get_mask(image)
    warp_array = warp.numpy()
    dimension = warp.dimension
    if warp_array.ndim == dimension + 1 and warp_array.shape[-1] == dimension:
        displacement = warp_array
    elif warp_array.ndim == dimension + 1 and warp_array.shape[0] == dimension:
        displacement = np.moveaxis(warp_array, 0, -1)
    else:
        raise ValueError(
            f"Unexpected warp array shape {warp_array.shape} for dim={dimension}"
        )

    amplitude = np.linalg.norm(displacement, axis=-1)
    values = amplitude[mask.numpy().astype(bool)]
    values = values[np.isfinite(values)]
    stats: dict[str, float | int] = {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "std": float(values.std(ddof=0)),
        "max": float(values.max()),
        "n_vox": int(values.size),
    }
    print(stats)
    return stats


__all__ = ["inspect_warp_transform", "register_image", "transform_image"]
