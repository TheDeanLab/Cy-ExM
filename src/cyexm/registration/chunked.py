from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, fields
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Self

from cyexm.errors import CacheValidationError, ChunkRegistrationError
from cyexm.registration._native_paths import discard_path
from cyexm.registration._request_policy import (
    CHUNK_SCALAR_POLICY,
    canonical_compact_json,
    chunk_grid,
    chunk_manifest_parameters,
    chunk_manifest_payload,
    local_chunk_parameters,
)
from cyexm.registration.artifacts import ArtifactStore, FileIdentity, StageRequest
from cyexm.registration.backend import RegistrationBackend
from cyexm.registration.whole_image import ImageRegistration, RegistrationPaths
from cyexm.runtime import NativeThreadConfig, configure_native_threads
from cyexm.spatial import ImageGeometry, PixelSize


class _ChunkOutputValidationError(ValueError):
    pass


def _scientific_runtime() -> Any:
    import numpy as np

    return np


def _positive_int_triple(value: object, name: str) -> tuple[int, int, int]:
    if isinstance(value, str | bytes):
        raise TypeError(f"{name} must contain exactly three positive integers")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(
            f"{name} must contain exactly three positive integers"
        ) from error
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three positive integers")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in values):
        raise TypeError(f"{name} must contain exactly three positive integers")
    normalized = tuple(int(item) for item in values)
    if any(item <= 0 for item in normalized):
        raise ValueError(f"{name} must contain exactly three positive integers")
    return normalized  # type: ignore[return-value]


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _validated_overlap(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("overlap_fraction must be a finite number")
    overlap = float(value)
    if not math.isfinite(overlap):
        raise ValueError("overlap_fraction must be finite")
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError("overlap_fraction must satisfy 0 <= value < 1")
    return overlap


@dataclass(frozen=True, slots=True)
class ChunkInfo:
    chunk_id: int
    z_start: int
    z_end: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int
    z_start_ext: int
    z_end_ext: int
    y_start_ext: int
    y_end_ext: int
    x_start_ext: int
    x_end_ext: int

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{item.name} must be an integer")
        if self.chunk_id < 0:
            raise ValueError("chunk_id must be a non-negative integer")
        for axis in "zyx":
            start = getattr(self, f"{axis}_start")
            end = getattr(self, f"{axis}_end")
            start_ext = getattr(self, f"{axis}_start_ext")
            end_ext = getattr(self, f"{axis}_end_ext")
            if start_ext < 0 or start < 0 or start >= end:
                raise ValueError(f"{axis} core bounds must be non-empty")
            if start_ext > start or end > end_ext:
                raise ValueError(f"{axis} extended bounds must contain the core")

    @property
    def core_shape(self) -> tuple[int, int, int]:
        return (
            self.z_end - self.z_start,
            self.y_end - self.y_start,
            self.x_end - self.x_start,
        )

    @property
    def extended_shape(self) -> tuple[int, int, int]:
        return (
            self.z_end_ext - self.z_start_ext,
            self.y_end_ext - self.y_start_ext,
            self.x_end_ext - self.x_start_ext,
        )

    @property
    def core_within_extended(self) -> tuple[slice, slice, slice]:
        z0 = self.z_start - self.z_start_ext
        y0 = self.y_start - self.y_start_ext
        x0 = self.x_start - self.x_start_ext
        return (
            slice(z0, z0 + self.core_shape[0]),
            slice(y0, y0 + self.core_shape[1]),
            slice(x0, x0 + self.core_shape[2]),
        )

    @property
    def extended_slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.z_start_ext, self.z_end_ext),
            slice(self.y_start_ext, self.y_end_ext),
            slice(self.x_start_ext, self.x_end_ext),
        )

    @property
    def core_slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.z_start, self.z_end),
            slice(self.y_start, self.y_end),
            slice(self.x_start, self.x_end),
        )


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


def compute_chunk_grid(
    image_shape: tuple[int, int, int] | list[int],
    chunk_size: tuple[int, int, int] | list[int] = (512, 512, 512),
    overlap_fraction: float = 0.15,
) -> tuple[ChunkInfo, ...]:
    return tuple(
        ChunkInfo(**item)
        for item in chunk_grid(image_shape, chunk_size, overlap_fraction)
    )


def _array_shape(array: object, name: str) -> tuple[int, ...]:
    shape = getattr(array, "shape", None)
    if not isinstance(shape, tuple | list):
        raise TypeError(f"{name} must expose an array shape")
    if len(shape) not in {3, 4}:
        raise ValueError(f"{name} must have three spatial dimensions")
    spatial = _positive_int_triple(shape[:3], f"{name} spatial shape")
    if len(shape) == 4:
        component_count = shape[3]
        if (
            isinstance(component_count, bool)
            or not isinstance(component_count, Integral)
            or int(component_count) != 3
        ):
            raise ValueError(f"{name} warp data must have exactly three components")
        return (*spatial, 3)
    return spatial


def extract_chunk(array: object, chunk: ChunkInfo) -> Any:
    shape = _array_shape(array, "array")
    if any(
        stop > size
        for stop, size in zip(
            (chunk.z_end_ext, chunk.y_end_ext, chunk.x_end_ext),
            shape[:3],
            strict=True,
        )
    ):
        raise ValueError("chunk extended bounds exceed the array shape")
    try:
        extracted: Any = array[chunk.extended_slices]  # type: ignore[index]
    except (IndexError, TypeError) as error:
        raise TypeError("array must support three-dimensional slicing") from error
    extracted_shape = _array_shape(extracted, "extracted chunk")
    expected = (*chunk.extended_shape, *shape[3:])
    if extracted_shape != expected:
        raise ValueError(
            f"extracted chunk shape {extracted_shape} does not match {expected}"
        )
    copy = getattr(extracted, "copy", None)
    return copy() if callable(copy) else extracted


def stitch_core(destination: object, extended: object, chunk: ChunkInfo) -> None:
    destination_shape = _array_shape(destination, "destination")
    extended_shape = _array_shape(extended, "extended chunk")
    expected_extended = (*chunk.extended_shape, *destination_shape[3:])
    if extended_shape != expected_extended:
        raise ValueError(
            f"extended chunk shape {extended_shape} does not match {expected_extended}"
        )
    if any(
        stop > size
        for stop, size in zip(
            (chunk.z_end, chunk.y_end, chunk.x_end),
            destination_shape[:3],
            strict=True,
        )
    ):
        raise ValueError("chunk core bounds exceed the destination shape")
    core: Any = extended[chunk.core_within_extended]  # type: ignore[index]
    core_shape = _array_shape(core, "chunk core")
    expected_core = (*chunk.core_shape, *destination_shape[3:])
    if core_shape != expected_core:
        raise ValueError(
            f"chunk core shape {core_shape} does not match {expected_core}"
        )
    destination[chunk.core_slices] = core  # type: ignore[index]


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
            raise _ChunkOutputValidationError(
                f"{role} scalar shape {array.shape} does not match {expected_shape}"
            )
        if finite and not bool(np.isfinite(array).all()):
            raise _ChunkOutputValidationError(
                f"{role} scalar values must all be finite"
            )
        return array

    @staticmethod
    def _read_local_warp(
        np: Any,
        backend: RegistrationBackend,
        path: Path,
        expected_geometry: ImageGeometry,
    ) -> tuple[object, Any]:
        warp = backend.read_warp(path)
        geometry = backend.geometry(warp)
        if not ChunkedImageRegistration._geometry_compatible(
            np, geometry, expected_geometry
        ):
            raise ValueError(
                "local warp physical geometry does not match the fixed chunk"
            )
        array = np.asarray(backend.to_array(warp))
        expected_shape = (*expected_geometry.shape, 3)
        if array.shape != expected_shape:
            if array.ndim != 4 or array.shape[-1] != 3:
                raise ValueError("local warp must have exactly three components")
            raise ValueError(
                f"local warp shape {array.shape} does not match {expected_shape}"
            )
        if not bool(np.isfinite(array).all()):
            raise ValueError("local warp values must all be finite")
        return warp, array

    @staticmethod
    def _geometry_compatible(
        np: Any,
        actual: ImageGeometry,
        expected: ImageGeometry,
    ) -> bool:
        if actual.shape != expected.shape:
            return False
        # NIfTI header geometry is commonly serialized at float32 precision.
        # The tolerance covers that roundoff while remaining far below a voxel.
        return bool(
            np.allclose(actual.origin, expected.origin, rtol=2e-7, atol=1e-6)
            and np.allclose(actual.spacing, expected.spacing, rtol=2e-7, atol=1e-6)
            and np.allclose(
                actual.direction,
                expected.direction,
                rtol=2e-7,
                atol=1e-6,
            )
        )

    @staticmethod
    def _float32_core(
        np: Any,
        array: Any,
        chunk: ChunkInfo,
        role: str,
    ) -> Any:
        try:
            extended_shape = _array_shape(array, f"{role} extended chunk")
        except (TypeError, ValueError) as error:
            raise _ChunkOutputValidationError(str(error)) from error
        components = extended_shape[3:]
        expected_extended = (*chunk.extended_shape, *components)
        if extended_shape != expected_extended:
            raise _ChunkOutputValidationError(
                f"{role} extended chunk shape {extended_shape} "
                f"does not match {expected_extended}"
            )
        core = array[chunk.core_within_extended]
        expected_core = (*chunk.core_shape, *components)
        try:
            core_shape = _array_shape(core, f"{role} core")
        except (TypeError, ValueError) as error:
            raise _ChunkOutputValidationError(str(error)) from error
        if core_shape != expected_core:
            raise _ChunkOutputValidationError(
                f"{role} core shape does not match {expected_core}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.asarray(core, dtype=np.float32)
        if not bool(np.isfinite(converted).all()):
            raise _ChunkOutputValidationError(
                f"{role} core values must remain finite after float32 conversion"
            )
        return converted

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

    def _local_request(
        self,
        backend: RegistrationBackend,
        fixed_chunk: object,
        moving_chunk: object,
        fixed_mask_chunk: object | None,
        moving_mask_chunk: object | None,
        chunk: ChunkInfo,
        chunks: tuple[ChunkInfo, ...],
        image_shape: tuple[int, int, int],
        native_threads: dict[str, int],
        affine_identity: FileIdentity,
    ) -> StageRequest:
        self._verify_source_identities()
        parameters = local_chunk_parameters(
            chunk=self._chunk_dict(chunk),
            chunks=[self._chunk_dict(item) for item in chunks],
            chunk_size_zyx=self.chunk_size,
            fixed_mask_geometry=(
                None if fixed_mask_chunk is None else backend.geometry(fixed_mask_chunk)
            ),
            image_shape_zyx=image_shape,
            moving_geometry=backend.geometry(moving_chunk),
            moving_mask_geometry=(
                None
                if moving_mask_chunk is None
                else backend.geometry(moving_mask_chunk)
            ),
            overlap_fraction=self.overlap_fraction,
            nonlinear_accuracy=self.nonlinear_accuracy,
            nonlinear_metric=self.nonlinear_metric,
            nonlinear_radius=self.nonlinear_radius,
            scalar_postprocessing_policy=self._scalar_postprocessing_policy,
        )
        return StageRequest.for_identities(
            stage=f"nonlinear-chunk:{chunk.chunk_id}",
            inputs=self._input_identity_snapshot(),
            geometry=backend.geometry(fixed_chunk),
            method=self.nonlinear_type,
            accuracy=self.nonlinear_accuracy,
            native_threads=native_threads,
            upstream=affine_identity,
            parameters=parameters,
        )

    def _load_or_commit_local_warp(
        self,
        np: Any,
        backend: RegistrationBackend,
        fixed_chunk: object,
        moving_chunk: object,
        fixed_mask_chunk: object | None,
        moving_mask_chunk: object | None,
        chunk: ChunkInfo,
        request: StageRequest,
        low_variance: bool,
        *,
        require_existing: bool,
    ) -> tuple[Any, bool]:
        paths = self.chunk_paths(chunk.chunk_id)
        paths.directory.mkdir(parents=True, exist_ok=True)
        expected_geometry = backend.geometry(fixed_chunk)

        def reader(path: Path) -> tuple[object, Any]:
            return self._read_local_warp(np, backend, path, expected_geometry)

        cached = None
        if not self.force_override:
            cached = self._reuse_artifact(paths.warp, request, reader)
        if cached is not None:
            return cached[1], False
        if require_existing:
            raise CacheValidationError(
                f"Required local chunk cache entry is missing: {paths.warp}"
            )

        temporary_warp: Path | None = None
        try:
            if low_variance:

                def writer(path: Path) -> None:
                    backend.write_identity_warp(fixed_chunk, path)

            else:
                _candidate_scalar, temporary_warp = backend.nonlinear(
                    moving_image=moving_chunk,
                    fixed_image=fixed_chunk,
                    fixed_mask=fixed_mask_chunk,
                    moving_mask=moving_mask_chunk,
                    registration_type=self.nonlinear_type,
                    accuracy=self.nonlinear_accuracy,
                    metric=self.nonlinear_metric,
                    radius=self.nonlinear_radius,
                    verbose=True,
                )
                reader(temporary_warp)

                def writer(path: Path) -> None:
                    if temporary_warp is None:
                        raise RuntimeError("local warp candidate was not prepared")
                    shutil.copyfile(temporary_warp, path)

            def validator(path: Path) -> bool:
                try:
                    reader(path)
                except Exception:
                    return False
                return True

            self._commit_artifact(paths.warp, request, writer, validator)
            try:
                committed = reader(paths.warp)[1]
            except Exception:
                self._store().mark_stale(paths.warp)
                raise
            return committed, True
        finally:
            discard_path(temporary_warp)

    def _reconstruct_local_images(
        self,
        np: Any,
        backend: RegistrationBackend,
        fixed_chunk: object,
        moving_chunk: object,
        moving_mask_chunk: object | None,
        chunk: ChunkInfo,
    ) -> Any:
        path = self.chunk_paths(chunk.chunk_id).warp
        raw_scalar = backend.apply_warp(
            fixed=fixed_chunk,
            moving=moving_chunk,
            warp_path=path,
            interpolation="linear",
        )
        scalar = backend.histogram_match(source=raw_scalar, reference=moving_chunk)
        scalar_array = self._validated_scalar_array(
            np,
            backend,
            scalar,
            chunk.extended_shape,
            "registered chunk",
        )
        if moving_mask_chunk is not None:
            raw_mask = backend.apply_warp(
                fixed=fixed_chunk,
                moving=moving_mask_chunk,
                warp_path=path,
                interpolation="nearestNeighbor",
            )
            mask = backend.threshold_mask(raw_mask)
            self._validated_scalar_array(
                np,
                backend,
                mask,
                chunk.extended_shape,
                "registered mask chunk",
            )
        return scalar_array

    def _manifest_payload(
        self,
        chunks: tuple[ChunkInfo, ...],
        image_shape: tuple[int, int, int],
        warp_identities: tuple[FileIdentity, ...],
    ) -> dict[str, object]:
        return chunk_manifest_payload(
            round_number=self.imaging_round,
            image_shape_zyx=image_shape,
            chunk_size_zyx=self.chunk_size,
            overlap_fraction=self.overlap_fraction,
            chunks=[self._chunk_dict(chunk) for chunk in chunks],
            local_warps=[identity.to_dict() for identity in warp_identities],
        )

    @staticmethod
    def _canonical_json(payload: dict[str, object]) -> str:
        return canonical_compact_json(payload)

    def _commit_manifest(
        self,
        backend: RegistrationBackend,
        fixed: object,
        chunks: tuple[ChunkInfo, ...],
        image_shape: tuple[int, int, int],
        native_threads: dict[str, int],
        affine_identity: FileIdentity,
        warp_identities: tuple[FileIdentity, ...],
        *,
        require_existing: bool,
    ) -> FileIdentity:
        self._verify_source_identities()
        path = self._manifest_path()
        payload = self._manifest_payload(chunks, image_shape, warp_identities)
        text = self._canonical_json(payload)
        request = StageRequest.for_identities(
            stage="nonlinear-chunk-manifest",
            inputs=self._input_identity_snapshot(),
            geometry=backend.geometry(fixed),
            method="serial-chunk-grid",
            accuracy="exact",
            native_threads=native_threads,
            upstream=affine_identity,
            parameters=chunk_manifest_parameters(
                chunk_size_zyx=self.chunk_size,
                image_shape_zyx=image_shape,
                local_warps=[identity.to_dict() for identity in warp_identities],
                nonlinear_accuracy=self.nonlinear_accuracy,
                nonlinear_metric=self.nonlinear_metric,
                nonlinear_radius=self.nonlinear_radius,
                nonlinear_type=self.nonlinear_type,
                overlap_fraction=self.overlap_fraction,
            ),
        )

        def reader(source: Path) -> dict[str, object]:
            candidate_text = source.read_text(encoding="utf-8")
            candidate = json.loads(candidate_text)
            if not isinstance(candidate, dict):
                raise TypeError("chunk manifest must be a JSON object")
            if self._canonical_json(candidate) != candidate_text:
                raise ValueError("chunk manifest is not canonical")
            if candidate != payload:
                raise ValueError("chunk manifest contents do not match the grid")
            return candidate

        if not self.force_override:
            cached = self._reuse_artifact(path, request, reader)
            if cached is not None:
                return FileIdentity.from_path(path)
            if require_existing:
                raise CacheValidationError(
                    f"Required chunk manifest cache entry is missing: {path}"
                )

        def writer(destination: Path) -> None:
            destination.write_text(text, encoding="utf-8")

        self._commit_artifact(
            path,
            request,
            writer=writer,
            validator=lambda candidate: reader(candidate) == payload,
        )
        return FileIdentity.from_path(path)

    def _write_full_candidate_warp(
        self,
        np: Any,
        backend: RegistrationBackend,
        fixed: object,
        displacement: Any,
        image_shape: tuple[int, int, int],
    ) -> Path:
        warp = backend.from_array(displacement, reference=fixed, has_components=True)
        descriptor, candidate_name = tempfile.mkstemp(
            dir=Path(self.save_directory),
            prefix=".chunked-full-warp-",
            suffix=".nii.gz",
        )
        os.close(descriptor)
        candidate = Path(candidate_name)
        discard_path(candidate)
        try:
            backend.write_warp(warp, candidate)
            read = backend.read_warp(candidate)
            geometry = backend.geometry(read)
            if not self._geometry_compatible(np, geometry, backend.geometry(fixed)):
                raise ValueError(
                    "assembled warp geometry does not match the fixed image"
                )
            array = np.asarray(backend.to_array(read))
            if array.shape != (*image_shape, 3):
                raise ValueError("assembled warp must have shape (z, y, x, 3)")
            if array.dtype != np.dtype(np.float32):
                raise ValueError("assembled warp dtype must be exactly float32")
            if not bool(np.isfinite(array).all()):
                raise ValueError("assembled warp values must all be finite")
        except BaseException:
            discard_path(candidate)
            raise
        return candidate

    def _estimate_nonlinear(
        self,
        backend: RegistrationBackend,
        fixed: object,
        moving_linear: object,
        fixed_mask: object,
        moving_linear_mask: object,
    ) -> tuple[object, Path]:
        """Return a unique disposable full-warp candidate owned by the caller."""
        self._nonlinear_request_evidence = {}
        resolved = configure_native_threads(self.native_threads)
        np = _scientific_runtime()
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
        registered = np.zeros(image_shape, dtype=np.float32)
        displacement = np.zeros((*image_shape, 3), dtype=np.float32)
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
                if fixed_block.size == 0 or moving_block.size == 0:
                    raise ValueError("extracted scalar chunk must not be empty")
                if not bool(np.isfinite(fixed_block).all()) or not bool(
                    np.isfinite(moving_block).all()
                ):
                    raise ValueError("fixed and moving chunk values must all be finite")
                low_variance = bool(
                    np.std(fixed_block) < 1e-6 or np.std(moving_block) < 1e-6
                )
                request = self._local_request(
                    backend,
                    fixed_chunk,
                    moving_chunk,
                    fixed_mask_chunk,
                    moving_mask_chunk,
                    chunk,
                    chunks,
                    image_shape,
                    dict(resolved.values),
                    affine_identity,
                )
                warp_array, freshly_committed = self._load_or_commit_local_warp(
                    np,
                    backend,
                    fixed_chunk,
                    moving_chunk,
                    fixed_mask_chunk,
                    moving_mask_chunk,
                    chunk,
                    request,
                    low_variance,
                    require_existing=require_existing_evidence,
                )
                if freshly_committed:
                    self._store().mark_stale(self._manifest_path())
                try:
                    scalar_array = self._reconstruct_local_images(
                        np,
                        backend,
                        fixed_chunk,
                        moving_chunk,
                        moving_mask_chunk,
                        chunk,
                    )
                    scalar_core = self._float32_core(
                        np, scalar_array, chunk, "registered scalar"
                    )
                    warp_core = self._float32_core(
                        np, warp_array, chunk, "displacement warp"
                    )
                    registered[chunk.core_slices] = scalar_core
                    displacement[chunk.core_slices] = warp_core
                except _ChunkOutputValidationError:
                    self._mark_chunk_output_stale(chunk.chunk_id)
                    raise
                except Exception:
                    if freshly_committed:
                        self._store().mark_stale(self.chunk_paths(chunk.chunk_id).warp)
                    raise
            except Exception as error:
                raise ChunkRegistrationError(chunk.chunk_id, str(error)) from error

        warp_identities = tuple(
            FileIdentity.from_path(self.chunk_paths(chunk.chunk_id).warp)
            for chunk in chunks
        )
        manifest_identity = self._commit_manifest(
            backend,
            fixed,
            chunks,
            image_shape,
            dict(resolved.values),
            affine_identity,
            warp_identities,
            require_existing=require_existing_evidence,
        )
        self._nonlinear_request_evidence = {
            "chunk_grid": [self._chunk_dict(chunk) for chunk in chunks],
            "chunk_manifest": manifest_identity.to_dict(),
            "chunk_size_zyx": self.chunk_size,
            "local_warps": [identity.to_dict() for identity in warp_identities],
            "overlap_fraction": self.overlap_fraction,
        }
        registered_image = backend.from_array(registered, reference=fixed)
        candidate = self._write_full_candidate_warp(
            np,
            backend,
            fixed,
            displacement,
            image_shape,
        )
        return registered_image, candidate


__all__ = [
    "ChunkInfo",
    "ChunkPaths",
    "ChunkedImageRegistration",
    "compute_chunk_grid",
    "extract_chunk",
    "stitch_core",
]
