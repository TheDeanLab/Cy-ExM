"""Lightweight canonical registration request policy shared by workflow readers."""

from __future__ import annotations

from typing import Literal

WHOLE_SCALAR_POLICY = "whole-global-histogram-match-v1"
CHUNK_SCALAR_POLICY = "serial-chunk-local-histogram-match-core-stitch-v1"
OUTPUT_FORMAT_POLICY = "ome-compatible-tiff-uint16-v1"
CHANNEL_POSTPROCESSING_POLICY = "clip-uint16-no-normalization-v1"

_LINEAR_ITERATIONS = {
    "high": [1000, 500, 250, 100],
    "medium": [500, 250, 100, 50],
    "low": [100, 50, 25, 10],
}
_NONLINEAR_ITERATIONS = {
    "high": [100, 70, 50],
    "medium": [50, 35, 25],
    "low": [5, 5, 5],
    "dry run": [1, 1, 1],
}


def linear_parameters(linear_type: str, accuracy: str) -> dict[str, object]:
    """Return the canonical Phase 1 affine registration parameter evidence."""
    parameters: dict[str, object] = {
        "aff_metric": "mattes",
        "aff_random_sampling_rate": 0.5,
        "aff_sampling": 32,
        "mask_all_stages": True,
        "verbose": True,
    }
    if linear_type == "TRSAA" and accuracy != "dry run":
        parameters["reg_iterations"] = _LINEAR_ITERATIONS[accuracy]
    if accuracy == "dry run":
        parameters.update(
            effective_registration_type="Rigid",
            aff_iterations=[1],
            aff_smoothing_sigmas=[0],
            aff_shrink_factors=[8],
            aff_random_sampling_rate=0.1,
        )
    return parameters


def nonlinear_parameters(accuracy: str) -> dict[str, object]:
    """Return the canonical Phase 1 nonlinear registration parameter evidence."""
    return {
        "aff_random_sampling_rate": 1.0,
        "flow_sigma": 3.0,
        "initial_transform": "Identity",
        "mask_all_stages": True,
        "reg_iterations": _NONLINEAR_ITERATIONS[accuracy],
        "singleprecision": False,
        "syn_metric": "mattes",
        "syn_sampling": 32,
        "total_sigma": 0.5,
        "verbose": True,
    }


def scalar_policy(strategy: Literal["whole_image", "serial_chunked"]) -> str:
    """Return the Phase 1 registered-scalar policy for a workflow strategy."""
    return WHOLE_SCALAR_POLICY if strategy == "whole_image" else CHUNK_SCALAR_POLICY


__all__ = [
    "CHANNEL_POSTPROCESSING_POLICY",
    "CHUNK_SCALAR_POLICY",
    "OUTPUT_FORMAT_POLICY",
    "WHOLE_SCALAR_POLICY",
    "linear_parameters",
    "nonlinear_parameters",
    "scalar_policy",
]
