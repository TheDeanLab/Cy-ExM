from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

import numpy as np
import pytest
from hpc.fixed_affine_chunk_figures import render_characterization_figures
from PIL import Image

pytestmark = pytest.mark.unit


def _encoded(value: int) -> dict[str, object]:
    array = np.full((20, 24), value, dtype=np.uint8)
    return {
        "shape_yx": [20, 24],
        "encoding": "zlib+base64+uint8",
        "data": base64.b64encode(zlib.compress(array.tobytes())).decode("ascii"),
    }


def _payload() -> dict[str, object]:
    conditions = ["1125", "512", "256", "128"]
    regional = []
    overlays = []
    blocks = []
    jacobians = []
    for condition_index, condition in enumerate(conditions):
        for round_number in (2, 3, 4):
            for roi_index, roi in enumerate(("dim", "bright")):
                regional.append(
                    {
                        "condition": condition,
                        "round": round_number,
                        "roi": roi,
                        "residual_px": condition_index + roi_index + round_number / 10,
                        "best_ncc": 0.8,
                        "prominence": 0.1,
                        "boundary_optimum": False,
                        "defined": True,
                    }
                )
                overlays.append(
                    {
                        "condition": condition,
                        "round": round_number,
                        "roi": roi,
                        "fixed": _encoded(80),
                        "registered": _encoded(120),
                    }
                )
            for seam in ("seam", "interior"):
                blocks.append(
                    {
                        "condition": condition,
                        "round": round_number,
                        "seam_class": seam,
                        "intensity_quartile": 1,
                        "residual": {"defined": True, "residual_px": 1.5},
                    }
                )
            jacobians.append(
                {
                    "condition": condition,
                    "round": round_number,
                    "percentile_01": 0.8,
                    "percentile_50": 1.0,
                    "percentile_99": 1.2,
                    "nonpositive_count": 0,
                }
            )
    return {
        "schema_version": 1,
        "condition_order": conditions,
        "regional_residuals": regional,
        "overlay_crops": overlays,
        "blockwise_residuals": blocks,
        "jacobians": jacobians,
        "runtimes": [
            {
                "condition": condition,
                "nonlinear_wall_seconds": 1000 + index * 100,
                "chunk_task_seconds": 2000 + index * 200,
                "chunk_count": 10 + index,
                "chunks_per_hour": 30 + index,
            }
            for index, condition in enumerate(conditions)
        ],
    }


def test_characterization_figures_and_csv_are_deterministic(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps(_payload()), encoding="utf-8")
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_paths = render_characterization_figures(analysis, first)
    second_paths = render_characterization_figures(analysis, second)

    expected = {
        "regional-residuals.svg",
        "regional-residuals.png",
        "regional-overlays.png",
        "runtime-throughput.svg",
        "runtime-throughput.png",
        "seam-interior.svg",
        "seam-interior.png",
        "jacobian-validity.svg",
        "jacobian-validity.png",
        "regional-residuals.csv",
        "runtime-throughput.csv",
        "seam-interior.csv",
        "jacobian-validity.csv",
    }
    assert {path.name for path in first_paths} == expected
    assert {path.name for path in second_paths} == expected
    for name in expected:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    svg = (first / "regional-residuals.svg").read_text(encoding="utf-8")
    assert "XY chunk size (pixels)" in svg
    assert "Residual correction (pixels)" in svg
    with Image.open(first / "regional-overlays.png") as image:
        assert image.width >= 1200
        assert image.height >= 1200
