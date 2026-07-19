from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_CLEAR_EX_COMMIT = "2311a873b783477add7391ff00245f76490147f0"
_REGISTRATION_RTOL = 1e-4
_REGISTRATION_ATOL = 1e-5
_GEOMETRY_ATOL = 1e-7
_CHUNK_FIELDS = (
    "chunk_id",
    "z_start",
    "z_end",
    "y_start",
    "y_end",
    "x_start",
    "x_end",
    "z_start_ext",
    "z_end_ext",
    "y_start_ext",
    "y_end_ext",
    "x_start_ext",
    "x_end_ext",
)


def _geometry(image: Any) -> dict[str, object]:
    return {
        "shape": [int(value) for value in image.shape],
        "origin": [float(value) for value in image.origin],
        "spacing": [float(value) for value in image.spacing],
        "direction": [
            [float(value) for value in row] for row in image.direction.tolist()
        ],
    }


def _chunk_dict(chunk: object) -> dict[str, int]:
    return {name: int(getattr(chunk, name)) for name in _CHUNK_FIELDS}


def _array_result(
    np: Any,
    actual: Any,
    expected: Any,
    *,
    exact: bool,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> dict[str, object]:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        return {
            "passed": False,
            "actual_shape": list(actual_array.shape),
            "expected_shape": list(expected_array.shape),
        }
    passed = bool(
        np.array_equal(actual_array, expected_array)
        if exact
        else np.allclose(
            actual_array,
            expected_array,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
    )
    difference = np.abs(
        actual_array.astype(np.float64) - expected_array.astype(np.float64)
    )
    return {
        "passed": passed,
        "shape": list(actual_array.shape),
        "max_abs_difference": float(difference.max(initial=0.0)),
        "rtol": rtol,
        "atol": atol,
        "exact": exact,
    }


def _geometry_result(
    np: Any,
    actual: dict[str, dict[str, object]],
    expected: dict[str, object],
) -> dict[str, object]:
    details: dict[str, object] = {}
    passed = True
    for role in ("fixed", "affine", "nonlinear", "warp"):
        expected_geometry = expected[role]
        if not isinstance(expected_geometry, dict):
            raise TypeError(f"baseline geometry {role!r} must be an object")
        actual_geometry = actual[role]
        role_checks = {
            "shape": actual_geometry["shape"] == expected_geometry["shape"],
            "origin": bool(
                np.allclose(
                    actual_geometry["origin"],
                    expected_geometry["origin"],
                    rtol=0.0,
                    atol=_GEOMETRY_ATOL,
                )
            ),
            "spacing": bool(
                np.allclose(
                    actual_geometry["spacing"],
                    expected_geometry["spacing"],
                    rtol=0.0,
                    atol=_GEOMETRY_ATOL,
                )
            ),
            "direction": bool(
                np.allclose(
                    actual_geometry["direction"],
                    expected_geometry["direction"],
                    rtol=0.0,
                    atol=_GEOMETRY_ATOL,
                )
            ),
        }
        details[role] = role_checks
        passed = passed and all(role_checks.values())
    return {"passed": passed, "atol": _GEOMETRY_ATOL, "details": details}


def _chunk_results(
    np: Any,
    baseline: Path,
    fixed_data: Any,
    moving_data: Any,
    chunk_summary: dict[str, object],
) -> dict[str, dict[str, object]]:
    from cyexm.registration.chunked import (
        compute_chunk_grid,
        extract_chunk,
        stitch_core,
    )

    shape = tuple(int(value) for value in chunk_summary["image_shape_zyx"])
    chunk_size = tuple(int(value) for value in chunk_summary["chunk_size_zyx"])
    overlap = float(chunk_summary["overlap_fraction"])
    chunks = compute_chunk_grid(shape, chunk_size, overlap)
    grid = [_chunk_dict(chunk) for chunk in chunks]
    expected_grid = chunk_summary["grid"]
    expected_extended_shapes = chunk_summary["extended_shapes"]
    expected_core_shapes = chunk_summary["core_shapes"]
    grid_passed = (
        grid == expected_grid
        and [list(chunk.extended_shape) for chunk in chunks] == expected_extended_shapes
        and [list(chunk.core_shape) for chunk in chunks] == expected_core_shapes
    )

    extraction_passed = True
    extraction_maximum = 0.0
    with np.load(baseline / "chunk_extractions.npz", allow_pickle=False) as expected:
        for chunk in chunks:
            for role, array in (("fixed", fixed_data), ("moving", moving_data)):
                key = f"{role}_{chunk.chunk_id:04d}"
                actual = extract_chunk(array, chunk)
                expected_array = expected[key]
                extraction_passed = extraction_passed and bool(
                    np.array_equal(actual, expected_array)
                )
                extraction_maximum = max(
                    extraction_maximum,
                    float(
                        np.abs(
                            np.asarray(actual, dtype=np.float64)
                            - np.asarray(expected_array, dtype=np.float64)
                        ).max(initial=0.0)
                    ),
                )

    scalar_stitch = np.zeros(shape, dtype=np.float32)
    vector_stitch = np.zeros((*shape, 3), dtype=np.float32)
    for chunk in chunks:
        extended = extract_chunk(fixed_data, chunk) + np.float32(chunk.chunk_id)
        vector_extended = np.stack(
            (extended, extended + np.float32(1), extended + np.float32(2)),
            axis=-1,
        )
        stitch_core(scalar_stitch, extended, chunk)
        stitch_core(vector_stitch, vector_extended, chunk)
    expected_scalar = np.load(baseline / "stitch_scalar.npy", allow_pickle=False)
    expected_vector = np.load(baseline / "stitch_vector.npy", allow_pickle=False)
    scalar_result = _array_result(np, scalar_stitch, expected_scalar, exact=True)
    vector_result = _array_result(np, vector_stitch, expected_vector, exact=True)
    return {
        "chunk_grid": {
            "passed": grid_passed,
            "chunk_count": len(chunks),
            "grid": grid,
        },
        "chunk_extraction": {
            "passed": extraction_passed,
            "max_abs_difference": extraction_maximum,
        },
        "overlap_stitching": {
            "passed": bool(scalar_result["passed"]) and bool(vector_result["passed"]),
            "scalar": scalar_result,
            "vector": vector_result,
        },
    }


def compare_baseline(baseline: Path) -> dict[str, object]:
    summary_path = baseline / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != 1:
        raise RuntimeError("unsupported legacy baseline schema")
    if summary.get("source_commit") != _CLEAR_EX_COMMIT:
        raise RuntimeError("legacy baseline source commit does not match the pin")
    if not all(summary.get("finite", {}).values()):
        raise RuntimeError("legacy baseline contains nonfinite arrays")
    expected_policy = {
        "linear_registration_type": "TRSAA",
        "linear_accuracy": "low",
        "nonlinear_registration_type": "SyNOnly",
        "nonlinear_accuracy": "dry run",
    }
    policy = summary.get("comparison_policy")
    if not isinstance(policy, dict) or any(
        policy.get(name) != value for name, value in expected_policy.items()
    ):
        raise RuntimeError("legacy baseline comparison policy is not TRSAA-low/SyN dry")
    literal_policy = summary.get("literal_dry_run")
    if not isinstance(literal_policy, dict) or (
        literal_policy.get("linear_accuracy_argument") != "dry run"
        or literal_policy.get("effective_linear_registration_type") != "Rigid"
        or literal_policy.get("nonlinear_accuracy_argument") != "dry run"
        or not all(literal_policy.get("finite", {}).values())
    ):
        raise RuntimeError("legacy literal dry-run characterization is incomplete")

    os.environ["ANTS_RANDOM_SEED"] = str(summary["ants_random_seed"])
    from cyexm.runtime import NativeThreadConfig, configure_native_threads

    native_threads = NativeThreadConfig(itk_threads=1, blas_threads=1)
    resolved = configure_native_threads(native_threads)

    import ants
    import numpy as np
    import scipy
    import tifffile

    from cyexm.io.images import zero_mask
    from cyexm.registration import linear, nonlinear
    from cyexm.registration.backend import AntsRegistrationBackend
    from cyexm.registration.transforms import apply_transform_chain

    fixed_data = np.load(baseline / "fixed.npy", allow_pickle=False)
    moving_data = np.load(baseline / "moving.npy", allow_pickle=False)
    expected_geometry = summary["geometry"]
    fixed_geometry = expected_geometry["fixed"]
    fixed = ants.from_numpy(
        fixed_data,
        origin=tuple(fixed_geometry["origin"]),
        spacing=tuple(fixed_geometry["spacing"]),
        direction=np.asarray(fixed_geometry["direction"], dtype=np.float64),
    )
    moving = ants.from_numpy(
        moving_data,
        origin=tuple(fixed_geometry["origin"]),
        spacing=tuple(fixed_geometry["spacing"]),
        direction=np.asarray(fixed_geometry["direction"], dtype=np.float64),
    )
    fixed_mask = zero_mask(fixed)
    moving_mask = zero_mask(moving)
    literal_dry_affine, literal_dry_mask, _literal_dry_transform = (
        linear.register_image(
            moving,
            fixed,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            accuracy="dry run",
            native_threads=native_threads,
        )
    )
    if literal_dry_mask is None:
        raise RuntimeError("Cy-ExM dry-run registration did not return the input mask")
    literal_dry_nonlinear, _literal_dry_warp = nonlinear.register_image(
        literal_dry_affine,
        fixed,
        moving_mask=literal_dry_mask,
        fixed_mask=fixed_mask,
        accuracy="dry run",
        native_threads=native_threads,
    )
    literal_dry_comparison = {
        "affine_result": _array_result(
            np,
            literal_dry_affine.numpy(),
            np.load(baseline / "literal_dry_affine_result.npy", allow_pickle=False),
            exact=False,
            rtol=_REGISTRATION_RTOL,
            atol=_REGISTRATION_ATOL,
        ),
        "affine_mask": _array_result(
            np,
            literal_dry_mask.numpy(),
            np.load(baseline / "literal_dry_affine_mask.npy", allow_pickle=False),
            exact=True,
        ),
        "nonlinear_result": _array_result(
            np,
            literal_dry_nonlinear.numpy(),
            np.load(baseline / "literal_dry_nonlinear_result.npy", allow_pickle=False),
            exact=False,
            rtol=_REGISTRATION_RTOL,
            atol=_REGISTRATION_ATOL,
        ),
        "interpretation": (
            "informational expected difference: Cy-ExM uses ANTSpyX affine schedule "
            "names consumed by Rigid while the pinned legacy wrapper does not"
        ),
    }
    affine_image, affine_mask, affine = linear.register_image(
        moving,
        fixed,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        registration_type="TRSAA",
        accuracy="low",
        native_threads=native_threads,
    )
    if affine_mask is None:
        raise RuntimeError("Cy-ExM linear registration did not return the input mask")
    nonlinear_image, warp_path = nonlinear.register_image(
        affine_image,
        fixed,
        moving_mask=affine_mask,
        fixed_mask=fixed_mask,
        accuracy="dry run",
        native_threads=native_threads,
    )
    backend = AntsRegistrationBackend(native_threads)
    raw_nonlinear_mask = backend.apply_warp(
        fixed=fixed,
        moving=affine_mask,
        warp_path=warp_path,
        interpolation="nearestNeighbor",
    )
    nonlinear_mask = backend.threshold_mask(raw_nonlinear_mask)
    with tempfile.TemporaryDirectory(prefix="cyexm-legacy-parity-") as directory:
        affine_path = Path(directory) / "affine.mat"
        backend.write_affine(affine, affine_path)
        transform_order = apply_transform_chain(
            fixed=fixed,
            moving=moving,
            affine_path=affine_path,
            warp_path=warp_path,
            interpolation="linear",
            native_threads=native_threads,
        )

    warp = ants.image_read(str(warp_path))
    actual_geometry = {
        "fixed": _geometry(fixed),
        "affine": _geometry(affine_image),
        "nonlinear": _geometry(nonlinear_image),
        "warp": _geometry(warp),
    }
    results: dict[str, dict[str, object]] = {
        "geometry": _geometry_result(np, actual_geometry, expected_geometry),
        "affine_result": _array_result(
            np,
            affine_image.numpy(),
            np.load(baseline / "affine_result.npy", allow_pickle=False),
            exact=False,
            rtol=_REGISTRATION_RTOL,
            atol=_REGISTRATION_ATOL,
        ),
        "nonlinear_result": _array_result(
            np,
            nonlinear_image.numpy(),
            np.load(baseline / "nonlinear_result.npy", allow_pickle=False),
            exact=False,
            rtol=_REGISTRATION_RTOL,
            atol=_REGISTRATION_ATOL,
        ),
    }
    affine_mask_result = _array_result(
        np,
        affine_mask.numpy(),
        np.load(baseline / "affine_mask.npy", allow_pickle=False),
        exact=True,
    )
    nonlinear_mask_result = _array_result(
        np,
        nonlinear_mask.numpy(),
        np.load(baseline / "nonlinear_mask.npy", allow_pickle=False),
        exact=True,
    )
    affine_mask_values = sorted(
        float(value) for value in np.unique(affine_mask.numpy())
    )
    nonlinear_mask_values = sorted(
        float(value) for value in np.unique(nonlinear_mask.numpy())
    )
    expected_mask_values = summary["mask_values"]
    results["mask_interpolation"] = {
        "passed": bool(affine_mask_result["passed"])
        and bool(nonlinear_mask_result["passed"])
        and bool(affine_mask_values)
        and bool(nonlinear_mask_values)
        and set(affine_mask_values).issubset({0.0, 1.0})
        and set(nonlinear_mask_values).issubset({0.0, 1.0})
        and affine_mask_values == expected_mask_values["affine"]
        and nonlinear_mask_values == expected_mask_values["nonlinear"],
        "affine": affine_mask_result,
        "nonlinear": nonlinear_mask_result,
        "values": {
            "affine": affine_mask_values,
            "nonlinear": nonlinear_mask_values,
        },
    }
    transform_order_result = _array_result(
        np,
        transform_order.numpy(),
        np.load(baseline / "transform_order.npy", allow_pickle=False),
        exact=False,
        rtol=_REGISTRATION_RTOL,
        atol=_REGISTRATION_ATOL,
    )
    expected_transform = summary["transform_types"]
    results["transform_order"] = {
        **transform_order_result,
        "passed": bool(transform_order_result["passed"])
        and expected_transform["forward_order"] == ["nonlinear_warp", "affine"]
        and expected_transform["inversion_flags"] == [False, False]
        and affine.transform_type == expected_transform["affine"],
        "forward_order": ["nonlinear_warp", "affine"],
        "inversion_flags": [False, False],
        "affine_type": affine.transform_type,
    }
    results.update(
        _chunk_results(
            np,
            baseline,
            fixed_data,
            moving_data,
            summary["chunk"],
        )
    )

    import sys

    report: dict[str, object] = {
        "schema_version": 1,
        "source_commit": summary["source_commit"],
        "registration_tolerance": {
            "rtol": _REGISTRATION_RTOL,
            "atol": _REGISTRATION_ATOL,
        },
        "legacy_versions": summary["versions"],
        "cyexm_versions": {
            "python": sys.version.split()[0],
            "antspyx": ants.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "tifffile": tifffile.__version__,
        },
        "native_threads": dict(resolved.values),
        "ants_random_seed": os.environ["ANTS_RANDOM_SEED"],
        "comparison_policy": summary["comparison_policy"],
        "literal_dry_run_comparison": literal_dry_comparison,
        "results": results,
        "passed": all(bool(result["passed"]) for result in results.values()),
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Cy-ExM with a pinned ClearEx baseline."
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = compare_baseline(arguments.baseline.resolve())
    if arguments.report is not None:
        report_path = arguments.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for name, result in report["results"].items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} {name.replace('_', ' ')}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
