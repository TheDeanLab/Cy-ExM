"""Atomic fixed-grid arrays used by internal nonlinear chunk workers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from cyexm.errors import CacheValidationError
from cyexm.registration._request_policy import (
    PREPARED_MASK_POLICY,
    PREPARED_SCALAR_POLICY,
    canonical_compact_json,
    common_parameters,
)
from cyexm.registration.artifacts import FileIdentity, StageRequest
from cyexm.spatial import ImageGeometry

if TYPE_CHECKING:
    from cyexm.registration.backend import RegistrationBackend
    from cyexm.registration.chunked import ChunkInfo
    from cyexm.registration.whole_image import ImageRegistration, _LinearPreparation

_FORMAT_VERSION = 1
_MANIFEST_KEYS = frozenset(
    {
        "format_version",
        "fixed_scalar",
        "moving_linear_scalar",
        "fixed_mask",
        "moving_linear_mask",
        "native_fixed",
        "native_moving",
        "moving_crop",
        "affine",
        "geometry",
        "scalar_policy",
        "mask_policy",
        "native_threads",
    }
)


def _scientific_runtime() -> Any:
    import numpy as np

    return np


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _identity(value: object, name: str) -> FileIdentity:
    return FileIdentity.from_dict(_mapping(value, name), name=name)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


@dataclass(frozen=True, slots=True)
class PreparedRoundPaths:
    """Exact internal artifact paths for one prepared moving round."""

    directory: Path
    fixed_scalar: Path
    moving_linear_scalar: Path
    fixed_mask: Path
    moving_linear_mask: Path
    manifest: Path

    @classmethod
    def for_round(cls, output: Path, round_number: int) -> Self:
        if isinstance(round_number, bool) or not isinstance(round_number, int):
            raise TypeError("round_number must be a non-negative integer")
        if round_number < 0:
            raise ValueError("round_number must be a non-negative integer")
        directory = Path(output) / f"round_{round_number}_prepared"
        return cls(
            directory=directory,
            fixed_scalar=directory / "fixed_scalar.npy",
            moving_linear_scalar=directory / "moving_linear_scalar.npy",
            fixed_mask=directory / "fixed_mask.npy",
            moving_linear_mask=directory / "moving_linear_mask.npy",
            manifest=directory / "manifest.json",
        )

    @property
    def array_paths(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.fixed_scalar,
            self.moving_linear_scalar,
            self.fixed_mask,
            self.moving_linear_mask,
        )


@dataclass(frozen=True, slots=True)
class PreparedManifest:
    """Strict spatial and provenance evidence for one prepared array bundle."""

    format_version: int
    fixed_scalar: FileIdentity
    moving_linear_scalar: FileIdentity
    fixed_mask: FileIdentity
    moving_linear_mask: FileIdentity
    native_fixed: FileIdentity
    native_moving: FileIdentity
    moving_crop: FileIdentity | None
    affine: FileIdentity
    geometry: ImageGeometry
    scalar_policy: str
    mask_policy: str
    native_threads: dict[str, int]

    def __post_init__(self) -> None:
        if self.format_version != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported prepared format version: {self.format_version}"
            )
        for name in (
            "fixed_scalar",
            "moving_linear_scalar",
            "fixed_mask",
            "moving_linear_mask",
            "native_fixed",
            "native_moving",
            "affine",
        ):
            identity = getattr(self, name)
            if FileIdentity.from_dict(identity.to_dict(), name=name) != identity:
                raise ValueError(f"{name} identity is not canonical")
        if self.moving_crop is not None and (
            FileIdentity.from_dict(self.moving_crop.to_dict(), name="moving_crop")
            != self.moving_crop
        ):
            raise ValueError("moving_crop identity is not canonical")
        if ImageGeometry.from_dict(self.geometry.to_dict()) != self.geometry:
            raise ValueError("geometry is not canonical")
        if self.scalar_policy != PREPARED_SCALAR_POLICY:
            raise ValueError("unsupported prepared scalar policy")
        if self.mask_policy != PREPARED_MASK_POLICY:
            raise ValueError("unsupported prepared mask policy")
        if not isinstance(self.native_threads, dict) or not self.native_threads:
            raise ValueError("native_threads must be a non-empty JSON object")
        if any(
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for name, value in self.native_threads.items()
        ):
            raise ValueError("native_threads values must be positive integers")
        if list(self.native_threads) != sorted(self.native_threads):
            raise ValueError("native_threads must use canonical sorted ordering")

    @property
    def array_identities(
        self,
    ) -> tuple[FileIdentity, FileIdentity, FileIdentity, FileIdentity]:
        return (
            self.fixed_scalar,
            self.moving_linear_scalar,
            self.fixed_mask,
            self.moving_linear_mask,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "fixed_scalar": self.fixed_scalar.to_dict(),
            "moving_linear_scalar": self.moving_linear_scalar.to_dict(),
            "fixed_mask": self.fixed_mask.to_dict(),
            "moving_linear_mask": self.moving_linear_mask.to_dict(),
            "native_fixed": self.native_fixed.to_dict(),
            "native_moving": self.native_moving.to_dict(),
            "moving_crop": (
                None if self.moving_crop is None else self.moving_crop.to_dict()
            ),
            "affine": self.affine.to_dict(),
            "geometry": self.geometry.to_dict(),
            "scalar_policy": self.scalar_policy,
            "mask_policy": self.mask_policy,
            "native_threads": dict(self.native_threads),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        if set(payload) != _MANIFEST_KEYS:
            raise ValueError(
                "prepared manifest keys must be exactly: "
                + ", ".join(sorted(_MANIFEST_KEYS))
            )
        format_version = payload["format_version"]
        if isinstance(format_version, bool) or not isinstance(format_version, int):
            raise TypeError("format_version must be an integer")
        moving_crop = payload["moving_crop"]
        native_threads = _mapping(payload["native_threads"], "native_threads")
        parsed_threads: dict[str, int] = {}
        for name, value in native_threads.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"native_threads[{name!r}] must be an integer")
            parsed_threads[name] = value
        return cls(
            format_version=format_version,
            fixed_scalar=_identity(payload["fixed_scalar"], "fixed_scalar"),
            moving_linear_scalar=_identity(
                payload["moving_linear_scalar"], "moving_linear_scalar"
            ),
            fixed_mask=_identity(payload["fixed_mask"], "fixed_mask"),
            moving_linear_mask=_identity(
                payload["moving_linear_mask"], "moving_linear_mask"
            ),
            native_fixed=_identity(payload["native_fixed"], "native_fixed"),
            native_moving=_identity(payload["native_moving"], "native_moving"),
            moving_crop=(
                None if moving_crop is None else _identity(moving_crop, "moving_crop")
            ),
            affine=_identity(payload["affine"], "affine"),
            geometry=ImageGeometry.from_dict(_mapping(payload["geometry"], "geometry")),
            scalar_policy=_string(payload["scalar_policy"], "scalar_policy"),
            mask_policy=_string(payload["mask_policy"], "mask_policy"),
            native_threads=parsed_threads,
        )


@dataclass(frozen=True, slots=True)
class PreparedRegion:
    """Four owned float32 ZYX arrays and their cropped fixed-grid geometry."""

    fixed_scalar: Any
    moving_linear_scalar: Any
    fixed_mask: Any
    moving_linear_mask: Any
    geometry: ImageGeometry


def _prepared_array(
    np: Any,
    backend: RegistrationBackend,
    image: object,
    geometry: ImageGeometry,
) -> Any:
    source = np.asarray(backend.to_array(image))
    if source.dtype.hasobject:
        raise TypeError("prepared arrays must not have object dtype")
    if source.ndim != 3:
        raise ValueError("prepared arrays must be exactly three-dimensional ZYX")
    if tuple(source.shape) != geometry.shape:
        raise ValueError(
            f"prepared array shape {source.shape} does not match {geometry.shape}"
        )
    if not bool(np.isfinite(source).all()):
        raise ValueError("prepared array values must all be finite")
    return np.array(source, dtype=np.float32, order="C", copy=True)


def _validate_array(
    np: Any,
    path: Path,
    geometry: ImageGeometry,
    *,
    expected: Any | None = None,
    finite: bool = True,
) -> bool:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        valid = (
            array.ndim == 3
            and tuple(array.shape) == geometry.shape
            and array.dtype == np.dtype(np.float32)
            and bool(array.flags.c_contiguous)
        )
        if valid and finite:
            valid = bool(np.isfinite(array).all())
        if valid and expected is not None:
            valid = bool(np.array_equal(array, expected))
        return valid
    except (OSError, TypeError, ValueError):
        return False


def _read_validated_array(np: Any, path: Path, geometry: ImageGeometry) -> bool:
    if not _validate_array(np, path, geometry):
        raise ValueError("prepared array does not satisfy its fixed-grid contract")
    return True


def _array_request(
    registration: ImageRegistration,
    state: _LinearPreparation,
    geometry: ImageGeometry,
    role: str,
    affine: FileIdentity,
) -> StageRequest:
    parameters = common_parameters(
        crop=registration.crop,
        imaging_round=registration.imaging_round,
        pixel_size_xyz=registration.pixel_size,
    )
    parameters.update(
        {
            "format_version": _FORMAT_VERSION,
            "prepared_role": role,
            "scalar_policy": PREPARED_SCALAR_POLICY,
            "mask_policy": PREPARED_MASK_POLICY,
        }
    )
    return StageRequest.for_identities(
        stage=f"prepared:{role}",
        inputs=registration._input_identity_snapshot(),
        geometry=geometry,
        method=registration.linear_type,
        accuracy=registration.linear_accuracy,
        native_threads=state.resolved.values,
        upstream=affine,
        parameters=parameters,
    )


def _write_array(np: Any, destination: Path, array: Any) -> None:
    mapped = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float32,
        shape=array.shape,
        fortran_order=False,
    )
    mapped[...] = array
    mapped.flush()
    del mapped


def _read_manifest(path: Path) -> PreparedManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PreparedManifest.from_dict(_mapping(payload, "prepared manifest"))


def _require_identity(identity: FileIdentity, role: str) -> None:
    try:
        current = FileIdentity.from_path(Path(identity.path))
    except OSError as error:
        raise CacheValidationError(f"{role} identity is unreadable") from error
    if current != identity:
        raise CacheValidationError(f"{role} identity changed during preparation")


def commit_prepared_round(registration: ImageRegistration) -> PreparedManifest:
    """Prepare affine state and atomically persist its four fixed-grid arrays."""
    state = registration._prepare_linear_state()
    backend = state.backend
    geometry = backend.geometry(state.fixed)
    for role, image in (
        ("fixed scalar", state.fixed),
        ("moving scalar", state.moving_linear),
        ("fixed mask", state.fixed_mask),
        ("moving mask", state.moving_linear_mask),
    ):
        if backend.geometry(image) != geometry:
            raise ValueError(f"{role} geometry does not match the fixed geometry")

    np = _scientific_runtime()
    arrays = (
        _prepared_array(np, backend, state.fixed, geometry),
        _prepared_array(np, backend, state.moving_linear, geometry),
        _prepared_array(np, backend, state.fixed_mask, geometry),
        _prepared_array(np, backend, state.moving_linear_mask, geometry),
    )
    paths = PreparedRoundPaths.for_round(
        Path(registration.save_directory), registration.imaging_round
    )
    paths.directory.mkdir(parents=True, exist_ok=True)
    affine = FileIdentity.from_path(state.paths.affine)
    moving_crop = (
        FileIdentity.from_path(state.paths.moving_crop) if registration.crop else None
    )
    array_identities: list[FileIdentity] = []
    for role, path, array in zip(
        (
            "fixed_scalar",
            "moving_linear_scalar",
            "fixed_mask",
            "moving_linear_mask",
        ),
        paths.array_paths,
        arrays,
        strict=True,
    ):
        _require_identity(affine, "affine")
        if moving_crop is not None:
            _require_identity(moving_crop, "moving crop")
        request = _array_request(registration, state, geometry, role, affine)
        cached = None
        if not registration.force_override:
            cached = registration._reuse_artifact(
                path,
                request,
                lambda candidate: _read_validated_array(
                    np,
                    candidate,
                    geometry,
                ),
            )
        if cached is None:
            registration._commit_artifact(
                path,
                request,
                writer=lambda candidate, value=array: _write_array(
                    np, candidate, value
                ),
                validator=lambda candidate, value=array: _validate_array(
                    np,
                    candidate,
                    geometry,
                    expected=value,
                ),
            )
        array_identities.append(FileIdentity.from_path(path))
        _require_identity(affine, "affine")
        if moving_crop is not None:
            _require_identity(moving_crop, "moving crop")

    native_fixed, native_moving = registration._input_identity_snapshot()
    for identity, path, role in zip(
        array_identities,
        paths.array_paths,
        (
            "fixed scalar",
            "moving linear scalar",
            "fixed mask",
            "moving linear mask",
        ),
        strict=True,
    ):
        try:
            _require_identity(identity, role)
        except Exception:
            registration._store().mark_stale(path)
            raise
    manifest = PreparedManifest(
        format_version=_FORMAT_VERSION,
        fixed_scalar=array_identities[0],
        moving_linear_scalar=array_identities[1],
        fixed_mask=array_identities[2],
        moving_linear_mask=array_identities[3],
        native_fixed=native_fixed,
        native_moving=native_moving,
        moving_crop=moving_crop,
        affine=affine,
        geometry=geometry,
        scalar_policy=PREPARED_SCALAR_POLICY,
        mask_policy=PREPARED_MASK_POLICY,
        native_threads=dict(sorted(state.resolved.values.items())),
    )
    manifest_request = StageRequest.for_identities(
        stage="prepared:manifest",
        inputs=(
            native_fixed,
            native_moving,
            *(identity for identity in manifest.array_identities),
            *(() if moving_crop is None else (moving_crop,)),
        ),
        geometry=geometry,
        method=registration.linear_type,
        accuracy=registration.linear_accuracy,
        native_threads=manifest.native_threads,
        upstream=manifest.affine,
        parameters={
            "format_version": _FORMAT_VERSION,
            "scalar_policy": PREPARED_SCALAR_POLICY,
            "mask_policy": PREPARED_MASK_POLICY,
        },
    )
    cached_manifest = None
    if not registration.force_override:
        cached_manifest = registration._reuse_artifact(
            paths.manifest,
            manifest_request,
            _read_manifest,
        )
        if cached_manifest is not None and cached_manifest != manifest:
            raise CacheValidationError("cached prepared manifest does not match")
    if cached_manifest is None:

        def write_manifest(path: Path) -> None:
            path.write_text(
                canonical_compact_json(manifest.to_dict()),
                encoding="utf-8",
            )

        registration._commit_artifact(
            paths.manifest,
            manifest_request,
            writer=write_manifest,
            validator=lambda path: _read_manifest(path) == manifest,
        )
    try:
        for identity, role in zip(
            manifest.array_identities,
            (
                "fixed scalar",
                "moving linear scalar",
                "fixed mask",
                "moving linear mask",
            ),
            strict=True,
        ):
            _require_identity(identity, role)
        _require_identity(affine, "affine")
        if moving_crop is not None:
            _require_identity(moving_crop, "moving crop")
    except Exception:
        registration._store().mark_stale(paths.manifest)
        raise
    return manifest


def _verify_identity(identity: FileIdentity, role: str) -> Path:
    path = Path(identity.path)
    try:
        current = FileIdentity.from_path(path)
    except OSError as error:
        raise CacheValidationError(f"{role} identity is unreadable: {path}") from error
    if current != identity:
        raise CacheValidationError(f"{role} identity mismatch: {path}")
    return path


def _copy_region(
    np: Any,
    identity: FileIdentity,
    role: str,
    geometry: ImageGeometry,
    chunk: ChunkInfo,
) -> Any:
    path = _verify_identity(identity, role)
    try:
        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, TypeError, ValueError) as error:
        raise CacheValidationError(f"{role} array is unreadable: {path}") from error
    if mapped.ndim != 3 or tuple(mapped.shape) != geometry.shape:
        raise ValueError(f"{role} array shape does not match the prepared geometry")
    if mapped.dtype != np.dtype(np.float32):
        raise ValueError(f"{role} array dtype must be float32")
    if not bool(mapped.flags.c_contiguous):
        raise ValueError(f"{role} array must be C-contiguous")
    try:
        copied = np.array(
            mapped[chunk.extended_slices],
            dtype=np.float32,
            order="C",
            copy=True,
        )
    except (IndexError, ValueError) as error:
        raise ValueError("chunk extended bounds exceed prepared geometry") from error
    if tuple(copied.shape) != chunk.extended_shape:
        raise ValueError("chunk extended bounds exceed prepared geometry")
    if not bool(np.isfinite(copied).all()):
        raise ValueError(f"{role} regional values must all be finite")
    _verify_identity(identity, role)
    return copied


def read_prepared_region(
    manifest: PreparedManifest,
    chunk: ChunkInfo,
) -> PreparedRegion:
    """Copy one extended chunk from four read-only prepared-array memory maps."""
    if not isinstance(manifest, PreparedManifest):
        raise TypeError("manifest must be a PreparedManifest")
    if any(
        stop > size
        for stop, size in zip(
            (chunk.z_end_ext, chunk.y_end_ext, chunk.x_end_ext),
            manifest.geometry.shape,
            strict=True,
        )
    ):
        raise ValueError("chunk extended bounds exceed prepared geometry")
    provenance = [
        (manifest.native_fixed, "native fixed"),
        (manifest.native_moving, "native moving"),
        (manifest.affine, "affine"),
    ]
    if manifest.moving_crop is not None:
        provenance.append((manifest.moving_crop, "crop"))
    for identity, role in provenance:
        _verify_identity(identity, role)

    np = _scientific_runtime()
    arrays = tuple(
        _copy_region(np, identity, role, manifest.geometry, chunk)
        for identity, role in zip(
            manifest.array_identities,
            (
                "fixed scalar",
                "moving linear scalar",
                "fixed mask",
                "moving linear mask",
            ),
            strict=True,
        )
    )
    for identity, role in zip(
        manifest.array_identities,
        (
            "fixed scalar",
            "moving linear scalar",
            "fixed mask",
            "moving linear mask",
        ),
        strict=True,
    ):
        _verify_identity(identity, role)
    start_xyz = (chunk.x_start_ext, chunk.y_start_ext, chunk.z_start_ext)
    scaled = tuple(
        spacing * index
        for spacing, index in zip(
            manifest.geometry.spacing,
            start_xyz,
            strict=True,
        )
    )
    origin_values = tuple(
        manifest.geometry.origin[row]
        + sum(
            manifest.geometry.direction[row][column] * scaled[column]
            for column in range(3)
        )
        for row in range(3)
    )
    origin = (origin_values[0], origin_values[1], origin_values[2])
    if not all(math.isfinite(value) for value in origin):
        raise ValueError("cropped prepared origin must be finite")
    for identity, role in provenance:
        _verify_identity(identity, role)
    return PreparedRegion(
        fixed_scalar=arrays[0],
        moving_linear_scalar=arrays[1],
        fixed_mask=arrays[2],
        moving_linear_mask=arrays[3],
        geometry=ImageGeometry(
            shape=chunk.extended_shape,
            origin=origin,
            spacing=manifest.geometry.spacing,
            direction=manifest.geometry.direction,
        ),
    )


__all__ = [
    "PreparedManifest",
    "PreparedRegion",
    "PreparedRoundPaths",
    "commit_prepared_round",
    "read_prepared_region",
]
