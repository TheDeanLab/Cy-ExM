from __future__ import annotations

import os
from dataclasses import dataclass, fields
from numbers import Integral
from pathlib import Path
from typing import Any, Self

from cyexm.chunking import (
    ChunkInfo,
    _positive_int_triple,
    _validated_overlap,
    compute_chunk_grid,
    extract_chunk,
    stitch_core,
)
from cyexm.errors import (
    CacheValidationError,
    ChunkRegistrationError,
    RegistrationStageError,
)
from cyexm.registration._request_policy import (
    CHUNK_SCALAR_POLICY,
)
from cyexm.registration.artifacts import ArtifactStore, FileIdentity
from cyexm.registration.backend import RegistrationBackend
from cyexm.registration.chunk_stages import (
    AssembledNonlinear,
    ChunkOutputValidationError,
    ChunkRegion,
    ChunkStageContext,
    assemble_nonlinear_chunks,
    run_nonlinear_chunk,
)
from cyexm.registration.whole_image import (
    ImageRegistration,
    RegistrationPaths,
    _LinearPreparation,
)
from cyexm.runtime import NativeThreadConfig, configure_native_threads
from cyexm.spatial import PixelSize


def _scientific_runtime() -> Any:
    import numpy as np

    return np


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


@dataclass(frozen=True, slots=True)
class ChunkPaths:
    directory: Path
    warp: Path
    completion_record: Path

    @classmethod
    def for_chunk(
        cls,
        output: Path,
        round_number: int,
        chunk_id: int,
    ) -> Self:
        normalized_round = _nonnegative_int(round_number, "round_number")
        normalized_chunk = _nonnegative_int(chunk_id, "chunk_id")
        directory = Path(output) / f"round_{normalized_round}_chunks"
        warp = directory / f"chunk_{normalized_chunk:04d}_warp.nii.gz"
        return cls(
            directory=directory,
            warp=warp,
            completion_record=ArtifactStore().record_path(warp),
        )


class ChunkedImageRegistration(ImageRegistration):
    def __init__(
        self,
        fixed_image_path: str | os.PathLike[str],
        moving_image_path: str | os.PathLike[str],
        save_directory: str | os.PathLike[str],
        imaging_round: int = 0,
        pixel_size: PixelSize = (0.15, 0.15, 0.15),
        crop: bool = False,
        enable_logging: bool = True,
        force_override: bool = False,
        *,
        prepared_fixed: bool = False,
        chunk_size: tuple[int, int, int] = (512, 512, 512),
        overlap_fraction: float = 0.15,
        native_threads: NativeThreadConfig | None = None,
        linear_type: str = "TRSAA",
        linear_accuracy: str = "low",
        nonlinear_type: str = "SyNOnly",
        nonlinear_accuracy: str = "high",
        nonlinear_metric: object = "mattes",
        nonlinear_radius: object = None,
        preprocess_filter_sigma_zyx: tuple[float, float, float] | None = None,
        _backend: RegistrationBackend | None = None,
        _artifact_store: ArtifactStore | None = None,
    ) -> None:
        super().__init__(
            fixed_image_path,
            moving_image_path,
            save_directory,
            imaging_round,
            pixel_size,
            crop,
            enable_logging,
            force_override,
            prepared_fixed=prepared_fixed,
            native_threads=native_threads,
            linear_type=linear_type,
            linear_accuracy=linear_accuracy,
            nonlinear_type=nonlinear_type,
            nonlinear_accuracy=nonlinear_accuracy,
            nonlinear_metric=nonlinear_metric,
            nonlinear_radius=nonlinear_radius,
            preprocess_filter_sigma_zyx=preprocess_filter_sigma_zyx,
            _backend=_backend,
            _artifact_store=_artifact_store,
        )
        self.chunk_size = _positive_int_triple(chunk_size, "chunk_size")
        self.overlap_fraction = _validated_overlap(overlap_fraction)
        self._nonlinear_mode = "serial-chunked"
        self._scalar_postprocessing_policy = CHUNK_SCALAR_POLICY
        self._prepare_nonlinear_before_reuse = True

    def chunk_paths(self, chunk_id: int) -> ChunkPaths:
        return ChunkPaths.for_chunk(
            Path(self.save_directory),
            self.imaging_round,
            chunk_id,
        )

    def _manifest_path(self) -> Path:
        return self.chunk_paths(0).directory / "manifest.json"

    @staticmethod
    def _chunk_dict(chunk: ChunkInfo) -> dict[str, int]:
        return {item.name: int(getattr(chunk, item.name)) for item in fields(chunk)}

    @staticmethod
    def _chunk_bounds(chunk: ChunkInfo) -> tuple[int, int, int, int, int, int]:
        return (
            chunk.z_start_ext,
            chunk.z_end_ext,
            chunk.y_start_ext,
            chunk.y_end_ext,
            chunk.x_start_ext,
            chunk.x_end_ext,
        )

    @staticmethod
    def _validated_scalar_array(
        np: Any,
        backend: RegistrationBackend,
        image: object,
        expected_shape: tuple[int, int, int],
        role: str,
        *,
        finite: bool = True,
    ) -> Any:
        array = np.asarray(backend.to_array(image))
        if array.shape != expected_shape:
            raise ChunkOutputValidationError(
                f"{role} scalar shape {array.shape} does not match {expected_shape}"
            )
        if finite and not bool(np.isfinite(array).all()):
            raise ChunkOutputValidationError(f"{role} scalar values must all be finite")
        return array

    def _mark_chunk_output_stale(self, chunk_id: int) -> None:
        paths = RegistrationPaths.for_round(
            Path(self.save_directory), self.imaging_round
        )
        for artifact in (
            self.chunk_paths(chunk_id).warp,
            self._manifest_path(),
            paths.warp,
            paths.nonlinear_metrics,
            paths.registered,
        ):
            self._store().mark_stale(artifact)

    def register(self) -> None:
        state = self._prepare_linear_state()
        try:
            self._run_nonlinear_chunks(state)
        except (CacheValidationError, ChunkRegistrationError, RegistrationStageError):
            raise
        except Exception as error:
            raise RegistrationStageError(
                f"nonlinear registration failed for round {self.imaging_round}"
            ) from error

    def _run_nonlinear_chunks(
        self,
        state: _LinearPreparation,
    ) -> AssembledNonlinear:
        self._nonlinear_request_evidence = {}
        resolved = configure_native_threads(self.native_threads)
        np = _scientific_runtime()
        backend = state.backend
        fixed = state.fixed
        moving_linear = state.moving_linear
        fixed_mask = state.fixed_mask
        moving_linear_mask = state.moving_linear_mask
        fixed_geometry = backend.geometry(fixed)
        image_shape = _positive_int_triple(fixed_geometry.shape, "fixed image shape")
        fixed_array = self._validated_scalar_array(
            np, backend, fixed, image_shape, "fixed image", finite=False
        )
        moving_array = self._validated_scalar_array(
            np,
            backend,
            moving_linear,
            image_shape,
            "moving image",
            finite=False,
        )
        if backend.geometry(moving_linear).shape != image_shape:
            raise ValueError("fixed and moving image shapes must match")
        fixed_mask_array = self._validated_scalar_array(
            np, backend, fixed_mask, image_shape, "fixed mask", finite=False
        )
        moving_mask_array = self._validated_scalar_array(
            np,
            backend,
            moving_linear_mask,
            image_shape,
            "moving mask",
            finite=False,
        )
        chunks = compute_chunk_grid(
            image_shape,
            self.chunk_size,
            self.overlap_fraction,
        )
        affine_identity = FileIdentity.from_path(
            RegistrationPaths.for_round(
                Path(self.save_directory), self.imaging_round
            ).affine
        )
        round_paths = RegistrationPaths.for_round(
            Path(self.save_directory), self.imaging_round
        )
        full_record = self._store().record_path(round_paths.warp)
        full_stale = self._store().stale_path(round_paths.warp)
        require_existing_evidence = not self.force_override and (
            full_record.exists()
            or (round_paths.warp.exists() and not full_stale.exists())
        )
        context = ChunkStageContext(
            source_identities=self._input_identity_snapshot(),
            affine_identity=affine_identity,
            fixed_geometry=fixed_geometry,
            fixed=fixed,
            moving_linear=moving_linear,
            raw_fixed=state.raw_fixed,
            raw_moving_linear=state.raw_moving_linear,
            fixed_mask=fixed_mask,
            moving_linear_mask=moving_linear_mask,
            preprocessing=(
                None
                if self._registration_feature_evidence is None
                else dict(self._registration_feature_evidence)
            ),
            registered_scalar_policy=self._scalar_postprocessing_policy,
            nonlinear_type=self.nonlinear_type,
            nonlinear_accuracy=self.nonlinear_accuracy,
            nonlinear_metric=self.nonlinear_metric,
            nonlinear_radius=self.nonlinear_radius,
            chunks=chunks,
            chunk_size=self.chunk_size,
            overlap_fraction=self.overlap_fraction,
            native_threads=dict(resolved.values),
            force=self.force_override,
            output_directory=Path(self.save_directory),
            round_number=self.imaging_round,
            pixel_size=self.pixel_size,
            crop=self.crop,
            prepared_fixed=self.prepared_fixed,
            backend=backend,
            store=self._store(),
            verify_sources=self._verify_source_identities,
            reuse_artifact=self._reuse_artifact,
            commit_artifact=self._commit_artifact,
            mark_chunk_stale=self._mark_chunk_output_stale,
            require_existing=require_existing_evidence,
        )
        artifacts = []

        for chunk in chunks:
            try:
                fixed_block = extract_chunk(fixed_array, chunk)
                moving_block = extract_chunk(moving_array, chunk)
                extract_chunk(fixed_mask_array, chunk)
                extract_chunk(moving_mask_array, chunk)
                bounds = self._chunk_bounds(chunk)
                fixed_chunk = backend.crop_image(fixed, bounds)
                moving_chunk = backend.crop_image(moving_linear, bounds)
                fixed_mask_chunk = backend.crop_image(fixed_mask, bounds)
                moving_mask_chunk = backend.crop_image(moving_linear_mask, bounds)
                artifacts.append(
                    run_nonlinear_chunk(
                        context,
                        ChunkRegion(
                            fixed=fixed_chunk,
                            moving=moving_chunk,
                            fixed_mask=fixed_mask_chunk,
                            moving_mask=moving_mask_chunk,
                            fixed_array=fixed_block,
                            moving_array=moving_block,
                        ),
                        chunk,
                    )
                )
            except Exception as error:
                if isinstance(error, ChunkOutputValidationError):
                    self._mark_chunk_output_stale(chunk.chunk_id)
                raise ChunkRegistrationError(chunk.chunk_id, str(error)) from error

        return assemble_nonlinear_chunks(context, artifacts)


__all__ = [
    "ChunkInfo",
    "ChunkPaths",
    "ChunkedImageRegistration",
    "compute_chunk_grid",
    "extract_chunk",
    "stitch_core",
]
