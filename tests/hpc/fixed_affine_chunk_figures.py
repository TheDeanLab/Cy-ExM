"""Deterministic tables and publication figures for chunk characterization."""

from __future__ import annotations

import base64
import csv
import json
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_PALETTE = ("#0072B2", "#D55E00", "#009E73")
_CONDITIONS = ("1125", "512", "256", "128")
_ROUNDS = (2, 3, 4)


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _rows(payload: Mapping[str, object], key: str) -> list[dict[str, object]]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{key} must be a list of JSON objects")
    return cast(list[dict[str, object]], value)


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fields: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _save_pair(figure: plt.Figure, root: Path, stem: str) -> tuple[Path, Path]:
    svg = root / f"{stem}.svg"
    png = root / f"{stem}.png"
    figure.savefig(
        svg,
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "Cy-ExM"},
    )
    figure.savefig(
        png,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "Cy-ExM"},
    )
    plt.close(figure)
    return svg, png


def _decode_image(value: object) -> np.ndarray:
    mapping = _mapping(value, "encoded image")
    if mapping.get("encoding") != "zlib+base64+uint8":
        raise ValueError("unsupported overlay encoding")
    shape = mapping.get("shape_yx")
    data = mapping.get("data")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(item) is not int or item <= 0 for item in shape)
        or not isinstance(data, str)
    ):
        raise ValueError("malformed encoded overlay")
    raw = zlib.decompress(base64.b64decode(data, validate=True))
    return np.frombuffer(raw, dtype=np.uint8).reshape(tuple(shape))


def _regional_figure(rows: list[dict[str, object]]) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, roi in zip(axes, ("bright", "dim"), strict=True):
        for round_number, color in zip(_ROUNDS, _PALETTE, strict=True):
            values = [
                float(
                    next(
                        row["residual_px"]
                        for row in rows
                        if row.get("condition") == condition
                        and row.get("round") == round_number
                        and row.get("roi") == roi
                    )
                )
                for condition in _CONDITIONS
            ]
            axis.plot(
                _CONDITIONS,
                values,
                marker="o",
                color=color,
                label=f"Round {round_number}",
            )
        axis.set_title(f"{roi.capitalize()} ROI")
        axis.set_xlabel("XY chunk size (pixels)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Residual correction (pixels)")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    return figure


def _runtime_figure(rows: list[dict[str, object]]) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    ordered = [
        next(row for row in rows if row.get("condition") == condition)
        for condition in _CONDITIONS
    ]
    axes[0].bar(
        _CONDITIONS,
        [float(row["nonlinear_wall_seconds"]) / 3600.0 for row in ordered],
        color=_PALETTE[0],
    )
    axes[0].set_xlabel("XY chunk size (pixels)")
    axes[0].set_ylabel("Nonlinear wall time (hours)")
    axes[1].bar(
        _CONDITIONS,
        [float(row["chunks_per_hour"]) for row in ordered],
        color=_PALETTE[1],
    )
    axes[1].set_xlabel("XY chunk size (pixels)")
    axes[1].set_ylabel("Completed chunks per hour")
    figure.tight_layout()
    return figure


def _seam_figure(rows: list[dict[str, object]]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 4))
    width = 0.36
    positions = np.arange(len(_CONDITIONS))
    for offset, seam_class, color in (
        (-width / 2, "interior", _PALETTE[0]),
        (width / 2, "seam", _PALETTE[1]),
    ):
        means = []
        for condition in _CONDITIONS:
            values = [
                float(cast(dict[str, object], row["residual"])["residual_px"])
                for row in rows
                if row.get("condition") == condition
                and row.get("seam_class") == seam_class
                and cast(dict[str, object], row["residual"]).get("defined") is True
            ]
            means.append(float(np.mean(values)))
        axis.bar(positions + offset, means, width, color=color, label=seam_class)
    axis.set_xticks(positions, _CONDITIONS)
    axis.set_xlabel("XY chunk size (pixels)")
    axis.set_ylabel("Mean block residual (pixels)")
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def _jacobian_figure(rows: list[dict[str, object]]) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    positions = np.arange(len(_CONDITIONS))
    medians = []
    lows = []
    highs = []
    folds = []
    for condition in _CONDITIONS:
        selected = [row for row in rows if row.get("condition") == condition]
        medians.append(
            float(np.mean([float(row["percentile_50"]) for row in selected]))
        )
        lows.append(float(np.min([float(row["percentile_01"]) for row in selected])))
        highs.append(float(np.max([float(row["percentile_99"]) for row in selected])))
        folds.append(sum(int(row["nonpositive_count"]) for row in selected))
    axes[0].errorbar(
        positions,
        medians,
        yerr=(np.asarray(medians) - lows, np.asarray(highs) - medians),
        fmt="o",
        color=_PALETTE[0],
        capsize=4,
    )
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_xticks(positions, _CONDITIONS)
    axes[0].set_xlabel("XY chunk size (pixels)")
    axes[0].set_ylabel("Geometric Jacobian determinant")
    axes[1].bar(_CONDITIONS, folds, color=_PALETTE[1])
    axes[1].set_xlabel("XY chunk size (pixels)")
    axes[1].set_ylabel("Nonpositive determinant voxels")
    figure.tight_layout()
    return figure


def _overlay_figure(rows: list[dict[str, object]]) -> plt.Figure:
    ordered = [
        next(
            row
            for row in rows
            if row.get("condition") == condition
            and row.get("round") == round_number
            and row.get("roi") == roi
        )
        for condition in _CONDITIONS
        for round_number in _ROUNDS
        for roi in ("bright", "dim")
    ]
    figure, axes = plt.subplots(12, 2, figsize=(12, 24))
    for axis, row in zip(axes.flat, ordered, strict=True):
        fixed = _decode_image(row["fixed"]).astype(np.float32) / 255.0
        registered = _decode_image(row["registered"]).astype(np.float32) / 255.0
        overlay = np.zeros((*fixed.shape, 3), dtype=np.float32)
        overlay[..., 0] = registered
        overlay[..., 1] = fixed
        overlay[..., 2] = registered
        axis.imshow(overlay, interpolation="nearest")
        axis.set_title(
            f"{row['condition']} px · round {row['round']} · {row['roi']}",
            fontsize=8,
        )
        axis.axis("off")
    figure.tight_layout()
    return figure


def render_characterization_figures(
    analysis_json: Path,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Render stable CSV tables and scientific figures from exact analysis JSON."""
    payload = _mapping(
        json.loads(analysis_json.read_text(encoding="utf-8")),
        "analysis",
    )
    if tuple(payload.get("condition_order", ())) != _CONDITIONS:
        raise ValueError("analysis condition order is not the approved sweep")
    output_dir.mkdir(parents=True, exist_ok=False)
    matplotlib.rcParams["svg.hashsalt"] = "cyexm-fixed-affine-chunk-size"
    regional = _rows(payload, "regional_residuals")
    runtime = _rows(payload, "runtimes")
    seam = _rows(payload, "blockwise_residuals")
    jacobian = _rows(payload, "jacobians")
    overlays = _rows(payload, "overlay_crops")

    paths: list[Path] = []
    paths.extend(
        _save_pair(_regional_figure(regional), output_dir, "regional-residuals")
    )
    paths.append(
        _save_pair(_overlay_figure(overlays), output_dir, "regional-overlays")[1]
    )
    # The SVG overlay is intentionally discarded: the raster source is embedded
    # without adding scientific information and creates an unnecessarily large file.
    discarded_overlay_svg = output_dir / "regional-overlays.svg"
    discarded_overlay_svg.unlink()
    paths.extend(_save_pair(_runtime_figure(runtime), output_dir, "runtime-throughput"))
    paths.extend(_save_pair(_seam_figure(seam), output_dir, "seam-interior"))
    paths.extend(
        _save_pair(_jacobian_figure(jacobian), output_dir, "jacobian-validity")
    )

    regional_csv = output_dir / "regional-residuals.csv"
    _write_csv(
        regional_csv,
        regional,
        (
            "condition",
            "round",
            "roi",
            "correction_dy_px",
            "correction_dx_px",
            "residual_px",
            "current_ncc",
            "best_ncc",
            "runner_up_ncc",
            "prominence",
            "boundary_optimum",
            "defined",
        ),
    )
    runtime_csv = output_dir / "runtime-throughput.csv"
    _write_csv(
        runtime_csv,
        runtime,
        (
            "condition",
            "nonlinear_wall_seconds",
            "chunk_task_seconds",
            "chunk_count",
            "chunks_per_hour",
            "condition_elapsed_seconds",
        ),
    )
    seam_csv = output_dir / "seam-interior.csv"
    seam_flat = [
        {
            **row,
            **{
                f"residual_{key}": value
                for key, value in cast(dict[str, object], row["residual"]).items()
            },
        }
        for row in seam
    ]
    _write_csv(
        seam_csv,
        seam_flat,
        (
            "condition",
            "round",
            "center_y",
            "center_x",
            "fixed_mean_intensity",
            "fixed_gradient_energy",
            "intensity_quartile",
            "seam_class",
            "residual_residual_px",
            "residual_best_ncc",
            "residual_prominence",
            "residual_boundary_optimum",
            "residual_defined",
        ),
    )
    jacobian_csv = output_dir / "jacobian-validity.csv"
    _write_csv(
        jacobian_csv,
        jacobian,
        (
            "condition",
            "round",
            "voxel_count",
            "finite_count",
            "nonfinite_count",
            "nonpositive_count",
            "nonpositive_fraction",
            "minimum",
            "percentile_01",
            "percentile_05",
            "percentile_50",
            "percentile_95",
            "percentile_99",
            "maximum",
        ),
    )
    paths.extend((regional_csv, runtime_csv, seam_csv, jacobian_csv))
    return tuple(paths)


__all__ = ["render_characterization_figures"]
