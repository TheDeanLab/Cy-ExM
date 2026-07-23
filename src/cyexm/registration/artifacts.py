from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from cyexm.errors import CacheValidationError
from cyexm.io.atomic import atomic_write
from cyexm.spatial import ImageGeometry

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

_FILE_IDENTITY_KEYS = frozenset({"path", "st_size", "st_mtime_ns"})
_STAGE_REQUEST_KEYS = frozenset(
    {
        "stage",
        "inputs",
        "geometry",
        "method",
        "accuracy",
        "native_threads",
        "upstream",
        "parameters",
    }
)
_COMPLETION_RECORD_KEYS = frozenset({"schema_version", "request", "artifact"})


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            normalized[key] = _json_value(item)
        return normalized
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    raise TypeError(f"Value is not JSON-compatible: {value!r}")


def _json_mapping(value: Mapping[str, object]) -> dict[str, JsonValue]:
    normalized = _json_value(value)
    if not isinstance(normalized, dict):
        raise TypeError("Expected a JSON object")
    return normalized


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} keys must be exactly: {', '.join(sorted(expected))}")


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class FileIdentity:
    path: str
    st_size: int
    st_mtime_ns: int

    @classmethod
    def from_path(cls, path: Path) -> Self:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
        return cls(
            path=str(resolved),
            st_size=stat.st_size,
            st_mtime_ns=stat.st_mtime_ns,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "path": self.path,
            "st_size": self.st_size,
            "st_mtime_ns": self.st_mtime_ns,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        name: str = "file identity",
    ) -> Self:
        _require_exact_keys(payload, _FILE_IDENTITY_KEYS, name)
        path = payload["path"]
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        return cls(
            path=path,
            st_size=_integer(payload["st_size"], "st_size"),
            st_mtime_ns=_integer(payload["st_mtime_ns"], "st_mtime_ns"),
        )


@dataclass(frozen=True, slots=True)
class StageRequest:
    stage: str
    inputs: tuple[FileIdentity, ...]
    geometry: ImageGeometry
    method: str
    accuracy: str
    native_threads: dict[str, int]
    upstream: FileIdentity | None
    parameters: dict[str, JsonValue]

    @classmethod
    def for_identities(
        cls,
        *,
        stage: str,
        inputs: Iterable[FileIdentity],
        geometry: ImageGeometry,
        method: str,
        accuracy: str,
        native_threads: Mapping[str, int],
        upstream: FileIdentity | None,
        parameters: Mapping[str, object],
    ) -> Self:
        return cls(
            stage=stage,
            inputs=tuple(inputs),
            geometry=geometry,
            method=method,
            accuracy=accuracy,
            native_threads=dict(native_threads),
            upstream=upstream,
            parameters=_json_mapping(parameters),
        )

    @classmethod
    def for_files(
        cls,
        *,
        stage: str,
        inputs: Iterable[Path],
        geometry: ImageGeometry,
        method: str,
        accuracy: str,
        native_threads: Mapping[str, int],
        upstream: FileIdentity | None,
        parameters: Mapping[str, object],
    ) -> Self:
        return cls.for_identities(
            stage=stage,
            inputs=(FileIdentity.from_path(path) for path in inputs),
            geometry=geometry,
            method=method,
            accuracy=accuracy,
            native_threads=native_threads,
            upstream=upstream,
            parameters=parameters,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "stage": self.stage,
            "inputs": [identity.to_dict() for identity in self.inputs],
            "geometry": _json_mapping(self.geometry.to_dict()),
            "method": self.method,
            "accuracy": self.accuracy,
            "native_threads": dict(self.native_threads),
            "upstream": None if self.upstream is None else self.upstream.to_dict(),
            "parameters": _json_mapping(self.parameters),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        _require_exact_keys(payload, _STAGE_REQUEST_KEYS, "request")
        inputs = payload["inputs"]
        if not isinstance(inputs, list):
            raise TypeError("inputs must be a JSON array")
        native_threads = _mapping(payload["native_threads"], "native_threads")
        upstream = payload["upstream"]
        geometry = _mapping(payload["geometry"], "geometry")
        parameters = _mapping(payload["parameters"], "parameters")
        return cls(
            stage=_string(payload["stage"], "stage"),
            inputs=tuple(
                FileIdentity.from_dict(
                    _mapping(item, "input identity"), name="input identity"
                )
                for item in inputs
            ),
            geometry=ImageGeometry.from_dict(geometry),
            method=_string(payload["method"], "method"),
            accuracy=_string(payload["accuracy"], "accuracy"),
            native_threads={
                name: _integer(value, f"native_threads[{name!r}]")
                for name, value in native_threads.items()
            },
            upstream=(
                None
                if upstream is None
                else FileIdentity.from_dict(
                    _mapping(upstream, "upstream"), name="upstream identity"
                )
            ),
            parameters=_json_mapping(parameters),
        )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


@dataclass(frozen=True, slots=True)
class CompletionRecord:
    schema_version: int
    request: StageRequest
    artifact: FileIdentity

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported completion-record schema: {self.schema_version}"
            )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "request": self.request.to_dict(),
            "artifact": self.artifact.to_dict(),
        }

    def to_json(self) -> str:
        return (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def read(cls, path: Path) -> Self:
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = _mapping(payload, "completion record")
        _require_exact_keys(record, _COMPLETION_RECORD_KEYS, "completion record")
        return cls(
            schema_version=_integer(record["schema_version"], "schema_version"),
            request=StageRequest.from_dict(_mapping(record["request"], "request")),
            artifact=FileIdentity.from_dict(
                _mapping(record["artifact"], "artifact"), name="artifact identity"
            ),
        )


@dataclass(frozen=True, slots=True)
class AcceptedArtifactProgress:
    """One exact request accepted by the artifact store in this context."""

    path: Path
    record: CompletionRecord
    reused: bool


_PROGRESS_OBSERVER: ContextVar[Callable[[AcceptedArtifactProgress], None] | None] = (
    ContextVar("cyexm_artifact_progress_observer", default=None)
)


@contextmanager
def observe_artifact_progress(
    observer: Callable[[AcceptedArtifactProgress], None],
) -> Iterator[None]:
    """Observe only artifacts committed or exactly reused inside this context."""
    token = _PROGRESS_OBSERVER.set(observer)
    try:
        yield
    finally:
        _PROGRESS_OBSERVER.reset(token)


def _report_progress(
    path: Path,
    record: CompletionRecord,
    *,
    reused: bool,
) -> None:
    observer = _PROGRESS_OBSERVER.get()
    if observer is not None:
        observer(AcceptedArtifactProgress(path, record, reused))


class ArtifactStore:
    def record_path(self, artifact: Path) -> Path:
        return artifact.with_name(f"{artifact.name}.complete.json")

    def stale_path(self, artifact: Path) -> Path:
        record = self.record_path(artifact)
        return record.with_name(f"{record.name}.stale")

    def reuse[T](
        self,
        artifact: Path,
        request: StageRequest,
        reader: Callable[[Path], T],
    ) -> T | None:
        record_path = self.record_path(artifact)
        stale_path = self.stale_path(artifact)
        if artifact.is_file() and stale_path.is_file() and not record_path.exists():
            return None
        if not artifact.exists() and not record_path.exists():
            return None
        if not artifact.is_file() or not record_path.is_file():
            raise CacheValidationError(f"Incomplete cache entry: {artifact}")
        try:
            record = CompletionRecord.read(record_path)
        except Exception as error:
            raise CacheValidationError(
                f"Cached completion record is unreadable: {record_path}"
            ) from error
        if record.request != request:
            raise CacheValidationError(f"Cached request mismatch: {artifact}")
        try:
            artifact_identity = FileIdentity.from_path(artifact)
        except OSError as error:
            raise CacheValidationError(
                f"Cached artifact identity is unreadable: {artifact}"
            ) from error
        if record.artifact != artifact_identity:
            raise CacheValidationError(f"Cached artifact identity mismatch: {artifact}")
        try:
            value = reader(artifact)
        except Exception as error:
            raise CacheValidationError(
                f"Cached artifact is unreadable: {artifact}"
            ) from error
        _report_progress(artifact, record, reused=True)
        return value

    def commit(
        self,
        artifact: Path,
        request: StageRequest,
        writer: Callable[[Path], None],
        validator: Callable[[Path], bool],
    ) -> None:
        atomic_write(artifact, writer, validator)
        record = CompletionRecord(
            schema_version=1,
            request=request,
            artifact=FileIdentity.from_path(artifact),
        )

        def write_record(path: Path) -> None:
            path.write_text(record.to_json(), encoding="utf-8")

        atomic_write(
            self.record_path(artifact),
            write_record,
            lambda path: CompletionRecord.read(path) == record,
        )
        self.stale_path(artifact).unlink(missing_ok=True)
        _report_progress(artifact, record, reused=False)

    def mark_stale(self, artifact: Path) -> None:
        record = self.record_path(artifact)
        if record.exists():
            os.replace(record, self.stale_path(artifact))


__all__ = [
    "AcceptedArtifactProgress",
    "ArtifactStore",
    "CompletionRecord",
    "FileIdentity",
    "StageRequest",
    "observe_artifact_progress",
]
