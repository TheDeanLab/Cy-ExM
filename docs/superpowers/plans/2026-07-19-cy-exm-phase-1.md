# Cy-ExM Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build an independent, tested Cy-ExM package containing the proven whole-image and serial chunked ClearEx registration workflows, configurable native threading, and restart-safe artifacts.

**Architecture:** A lightweight cyexm package defers all scientific imports until runtime configuration is resolved. Focused registration kernels sit behind an internal ANTs backend; whole-image and serial chunked orchestrators use a shared atomic artifact store whose completion records make every expensive stage restartable. ClearEx source commit 2311a873b783477add7391ff00245f76490147f0 is the behavioral and provenance baseline, but Cy-ExM has no runtime dependency on ClearEx.

**Tech Stack:** Python 3.12, uv, antspyx, NumPy, SciPy, scikit-image, tifffile, pytest, Ruff, BasedPyright, pre-commit, GitHub Actions.

## Global Constraints

- Work in an isolated git worktree created at execution time; do not implement directly on main.
- Keep environments and uv caches on /project. Export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache and UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python before uv commands.
- Use /archive only for datasets and large registration outputs, never for environments or dependency caches.
- Treat /project/bioinformatics/Danuser_lab/Dean/dean/git/clearex at commit 2311a873b783477add7391ff00245f76490147f0 as read-only extraction input during this plan.
- Retain the ClearEx academic-use copyright and license on extracted source.
- The distribution name is cy-exm and the import package is cyexm.
- Public pixel-size tuples are micrometers in (x, y, z) order; NumPy volumes are (z, y, x).
- Stable defaults are one ITK thread and one thread for each BLAS-family variable unless inherited environment values or explicit configuration override them.
- Preserve TRSAA, SyNOnly, masks based on values greater than zero, affine-before-nonlinear ordering, scalar linear interpolation, mask nearest-neighbor interpolation, and ANTs origin/spacing/direction.
- Preserve the legacy class defaults: TRSAA low for linear registration and SyNOnly high for nonlinear registration.
- Do not add Dask, distributed, dask-jobqueue, SLURM, YAML workflow configuration, or chunk multiprocessing.
- Do not export ParallelChunkedImageRegistration.
- Do not extract clearex.registration.pipeline or clearex.registration.tre.
- Do not modify or terminate the separate spheroid registration run or its output directory.
- Use apply_patch for hand-authored file changes. Mechanical formatter and notebook-output cleanup commands are allowed.
- Follow TDD for every behavior: failing targeted test, minimal implementation, passing targeted test, then commit.

---

## File and responsibility map

The final Phase 1 tree introduces the following maintained files:

| Path | Responsibility |
| --- | --- |
| pyproject.toml | Package metadata, dependencies, pytest markers, Ruff, and BasedPyright policy |
| .pre-commit-config.yaml | Reproducible local checks |
| .github/workflows/ci.yml | Portable lint, type, unit, and bounded integration checks |
| AGENTS.md | Repository-wide architecture, scientific invariants, HPC storage rules, and commands |
| src/AGENTS.md | Dependency direction and import-side-effect rules |
| src/cyexm/AGENTS.md | Public API and module-boundary rules |
| src/cyexm/__init__.py | Lightweight version metadata only |
| src/cyexm/errors.py | Focused public exception hierarchy |
| src/cyexm/spatial.py | Dependency-neutral PixelSize and ImageGeometry values |
| src/cyexm/runtime/threads.py | Native-thread resolution, application, and import-order protection |
| src/cyexm/io/atomic.py | Suffix-preserving temporary paths and atomic replacement |
| src/cyexm/io/images.py | Array loading, ANTs conversion, masks, geometry, and cropping |
| src/cyexm/io/tiff.py | Legacy-compatible uint16 TIFF export |
| src/cyexm/metrics.py | ANTs similarity metrics and deterministic metric serialization |
| src/cyexm/registration/artifacts.py | Stage requests, completion records, cache validation, and staleness |
| src/cyexm/registration/backend.py | Internal ANTs backend protocol and production adapter |
| src/cyexm/registration/linear.py | Linear registration and affine inspection/application |
| src/cyexm/registration/nonlinear.py | Nonlinear registration and warp inspection/application |
| src/cyexm/registration/transforms.py | Transform-chain application and transform I/O |
| src/cyexm/registration/whole_image.py | Whole-image lifecycle, paths, and register_round |
| src/cyexm/registration/chunked.py | Pure chunk geometry and serial chunked nonlinear orchestration |
| src/cyexm/registration/__init__.py | Stable registration exports only |
| tests/unit | Pure and mocked tests, including subprocess import/thread tests |
| tests/integration | Small real-ANTs tests and migration-parity utilities |
| examples/scripts | Parameterized library examples with no lab-specific paths |
| examples/notebooks | Cleared, parameterized registration notebooks |
| docs/migration-from-clearex.md | Provenance, import map, intentional corrections, and consumer migration |
| docs/registration-artifacts.md | Output names, completion records, cache validity, and restart semantics |

## Task 1: Establish the package and import contract

**Files:**
- Create: pyproject.toml
- Create: README.md
- Create: LICENSE.md
- Create: AGENTS.md
- Create: src/AGENTS.md
- Create: src/cyexm/AGENTS.md
- Create: src/cyexm/__init__.py
- Create: tests/AGENTS.md
- Create: tests/unit/AGENTS.md
- Create: tests/unit/test_import_contract.py
- Modify: uv.lock

**Interfaces:**
- Consumes: No package code.
- Produces: import cyexm, cyexm.__version__, and project-aware uv commands used by all later tasks.

- [ ] **Step 1: Write the failing lightweight-import test**

~~~python
# tests/unit/test_import_contract.py
from __future__ import annotations

import subprocess
import sys


def test_import_cyexm_is_lightweight_and_side_effect_free() -> None:
    code = """
import os
import sys

before = dict(os.environ)
import cyexm
after = dict(os.environ)

assert before == after
assert "ants" not in sys.modules
assert "numpy" not in sys.modules
assert "scipy" not in sys.modules
assert isinstance(cyexm.__version__, str)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
~~~

- [ ] **Step 2: Run the test and verify the package does not exist**

Run:

~~~bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run pytest tests/unit/test_import_contract.py -v
~~~

Expected: FAIL because cyexm cannot be imported.

- [ ] **Step 3: Add the minimal package and project configuration**

Use this project metadata and tool policy:

~~~toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "cy-exm"
version = "0.1.0"
description = "Reusable cyclic expansion microscopy registration workflows."
readme = "README.md"
license = {file = "LICENSE.md"}
requires-python = ">=3.12,<3.13"
authors = [{name = "The Dean Lab, UT Southwestern Medical Center"}]
dependencies = [
  "antspyx>=0.6,<0.7",
  "numpy>=1.26,<3",
  "scikit-image>=0.22,<1",
  "scipy>=1.12,<2",
  "tifffile>=2025.1.10",
]

[project.optional-dependencies]
examples = ["jupyterlab>=4", "matplotlib>=3.8", "nbconvert>=7"]

[dependency-groups]
dev = [
  "basedpyright==1.38.3",
  "pre-commit>=4",
  "pytest>=8",
  "pytest-cov>=6",
  "ruff>=0.14.8",
]

[project.urls]
Source = "https://github.com/TheDeanLab/Cy-ExM"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-ra --strict-markers"
testpaths = ["tests"]
markers = [
  "unit: isolated or mocked checks",
  "integration: bounded checks using real scientific libraries",
  "hpc: checks requiring lab storage, representative data, or cluster services",
]

[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]

[tool.basedpyright]
include = ["src/cyexm"]
pythonVersion = "3.12"
typeCheckingMode = "basic"
reportMissingTypeStubs = "none"
~~~

Keep src/cyexm/__init__.py free of scientific imports:

~~~python
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cy-exm")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
~~~

Copy LICENSE.md exactly from the recorded ClearEx baseline. Write the root and
nested AGENTS.md files with the constraints in this plan, exact uv commands,
the dependency direction cyexm.registration -> cyexm.io/runtime/metrics, and
the rule that child AGENTS.md files add scope-specific guidance without
repeating parent text.

- [ ] **Step 4: Lock dependencies and run the import test**

Run:

~~~bash
uv lock
uv run pytest tests/unit/test_import_contract.py -v
uv run ruff check src/cyexm tests/unit/test_import_contract.py
uv run ruff format --check src/cyexm tests/unit/test_import_contract.py
~~~

Expected: one passing test and no Ruff findings.

- [ ] **Step 5: Commit the package foundation**

~~~bash
git add pyproject.toml uv.lock README.md LICENSE.md AGENTS.md src tests
git commit -m "build: establish Cy-ExM package foundation"
~~~

## Task 2: Add native-thread configuration

**Files:**
- Create: src/cyexm/runtime/AGENTS.md
- Create: src/cyexm/runtime/__init__.py
- Create: src/cyexm/runtime/threads.py
- Create: src/cyexm/errors.py
- Create: tests/conftest.py
- Create: tests/unit/test_threads.py
- Modify: tests/unit/test_import_contract.py

**Interfaces:**
- Consumes: cyexm.errors.ThreadConfigurationError.
- Produces: NativeThreadConfig, ResolvedNativeThreads, configure_native_threads(config=None), NATIVE_THREAD_ENV_VARS, and a deterministic description of effective values and sources.

- [ ] **Step 1: Write failing resolution and late-import tests**

~~~python
from __future__ import annotations

import os

import pytest

from cyexm.errors import ThreadConfigurationError
from cyexm.runtime import NativeThreadConfig, configure_native_threads


def test_defaults_set_every_native_runtime_to_one() -> None:
    environ: dict[str, str] = {}
    resolved = configure_native_threads(
        environ=environ,
        loaded_modules=set(),
    )
    assert resolved.itk_threads == 1
    assert set(resolved.values.values()) == {1}
    assert set(resolved.sources.values()) == {"default"}


def test_explicit_values_override_inherited_environment() -> None:
    environ = {
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "16",
    }
    resolved = configure_native_threads(
        NativeThreadConfig(itk_threads=2, blas_threads=3),
        environ=environ,
        loaded_modules=set(),
    )
    assert resolved.itk_threads == 2
    assert all(value == 3 for name, value in resolved.values.items() if name != "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")
    assert all(source == "explicit" for source in resolved.sources.values())


def test_inherited_values_beat_defaults_per_variable() -> None:
    environ = {"OPENBLAS_NUM_THREADS": "6"}
    resolved = configure_native_threads(
        environ=environ,
        loaded_modules=set(),
    )
    assert resolved.values["OPENBLAS_NUM_THREADS"] == 6
    assert resolved.sources["OPENBLAS_NUM_THREADS"] == "environment"
    assert resolved.values["MKL_NUM_THREADS"] == 1


def test_late_change_raises_when_scientific_runtime_is_loaded() -> None:
    environ = {"OMP_NUM_THREADS": "1"}
    with pytest.raises(ThreadConfigurationError, match="already loaded"):
        configure_native_threads(
            NativeThreadConfig(blas_threads=4),
            environ=environ,
            loaded_modules={"numpy"},
        )


def test_reapplying_same_values_after_import_is_idempotent() -> None:
    environ = {name: "1" for name in (
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )}
    resolved = configure_native_threads(
        NativeThreadConfig(itk_threads=1, blas_threads=1),
        environ=environ,
        loaded_modules={"ants", "numpy"},
    )
    assert set(resolved.values.values()) == {1}
~~~

- [ ] **Step 2: Run the tests and verify missing runtime symbols**

Run: uv run pytest tests/unit/test_threads.py -v

Expected: FAIL during collection because cyexm.runtime is absent.

- [ ] **Step 3: Implement deterministic per-variable configuration**

~~~python
# src/cyexm/errors.py
class CyExMError(Exception):
    """Base exception for expected Cy-ExM failures."""


class ThreadConfigurationError(CyExMError):
    """Native thread settings cannot be applied safely."""


class InputValidationError(CyExMError):
    """An input path, image, or spatial setting is invalid."""


class CacheValidationError(CyExMError):
    """A cached stage is incomplete, corrupt, or incompatible."""


class RegistrationStageError(CyExMError):
    """A named registration stage failed."""


class ChunkRegistrationError(RegistrationStageError):
    """A serial chunk failed."""

    def __init__(self, chunk_id: int, message: str) -> None:
        self.chunk_id = chunk_id
        super().__init__(f"Chunk {chunk_id}: {message}")
~~~

~~~python
# src/cyexm/runtime/threads.py
from __future__ import annotations

import os
import sys
from collections.abc import AbstractSet, Mapping, MutableMapping
from dataclasses import dataclass

from cyexm.errors import ThreadConfigurationError

ITK_THREAD_ENV_VAR = "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"
BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
NATIVE_THREAD_ENV_VARS = (ITK_THREAD_ENV_VAR, *BLAS_THREAD_ENV_VARS)
SCIENTIFIC_MODULE_ROOTS = frozenset({"ants", "numpy", "scipy"})


@dataclass(frozen=True, slots=True)
class NativeThreadConfig:
    itk_threads: int | None = None
    blas_threads: int | None = None


@dataclass(frozen=True, slots=True)
class ResolvedNativeThreads:
    values: Mapping[str, int]
    sources: Mapping[str, str]

    @property
    def itk_threads(self) -> int:
        return self.values[ITK_THREAD_ENV_VAR]

    def describe(self) -> str:
        return ", ".join(
            f"{name}={self.values[name]} ({self.sources[name]})"
            for name in NATIVE_THREAD_ENV_VARS
        )


def _positive_int(value: int | str, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ThreadConfigurationError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise ThreadConfigurationError(f"{name} must be a positive integer")
    return parsed


def configure_native_threads(
    config: NativeThreadConfig | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
    loaded_modules: AbstractSet[str] | None = None,
) -> ResolvedNativeThreads:
    config = config or NativeThreadConfig()
    target_environ = os.environ if environ is None else environ
    module_names = set(sys.modules) if loaded_modules is None else set(loaded_modules)
    loaded_roots = {name.split(".", 1)[0] for name in module_names}
    runtime_loaded = bool(loaded_roots & SCIENTIFIC_MODULE_ROOTS)

    values: dict[str, int] = {}
    sources: dict[str, str] = {}
    for name in NATIVE_THREAD_ENV_VARS:
        explicit = (
            config.itk_threads
            if name == ITK_THREAD_ENV_VAR
            else config.blas_threads
        )
        if explicit is not None:
            value = _positive_int(explicit, name)
            source = "explicit"
        elif name in target_environ:
            value = _positive_int(target_environ[name], name)
            source = "environment"
        else:
            value = 1
            source = "default"

        current = target_environ.get(name)
        if runtime_loaded and current is not None and _positive_int(current, name) != value:
            loaded = ", ".join(sorted(loaded_roots & SCIENTIFIC_MODULE_ROOTS))
            raise ThreadConfigurationError(
                f"Cannot change {name} after scientific runtime already loaded: {loaded}"
            )
        if runtime_loaded and current is None:
            raise ThreadConfigurationError(
                f"Cannot set {name} after scientific runtime already loaded"
            )
        values[name] = value
        sources[name] = source

    for name, value in values.items():
        target_environ[name] = str(value)
    return ResolvedNativeThreads(values=values, sources=sources)
~~~

Export the three public symbols from runtime/__init__.py. Add a subprocess test
that imports cyexm.runtime and confirms ants, numpy, and scipy remain unloaded.
Add tests/conftest.py with literal names from NATIVE_THREAD_ENV_VARS and
os.environ.setdefault(name, "1"). This is test-process setup rather than
package behavior; it makes later real-NumPy tests an idempotent reapplication
of values that existed before registration. Add a unit assertion that describe
reports every variable, value, and source in stable NATIVE_THREAD_ENV_VARS
order.

- [ ] **Step 4: Run the thread and import tests**

Run:

~~~bash
uv run pytest tests/unit/test_threads.py tests/unit/test_import_contract.py -v
uv run basedpyright
~~~

Expected: all tests pass and BasedPyright reports zero errors.

- [ ] **Step 5: Commit native-thread control**

~~~bash
git add src/cyexm/errors.py src/cyexm/runtime tests/conftest.py tests/unit
git commit -m "feat: add safe native thread configuration"
~~~

## Task 3: Build atomic artifacts and completion records

**Files:**
- Create: src/cyexm/io/AGENTS.md
- Create: src/cyexm/io/__init__.py
- Create: src/cyexm/io/atomic.py
- Create: src/cyexm/spatial.py
- Create: src/cyexm/registration/artifacts.py
- Create: tests/unit/test_atomic.py
- Create: tests/unit/test_artifacts.py

**Interfaces:**
- Consumes: cyexm.errors.CacheValidationError.
- Produces: atomic_write(destination, writer, validator), dependency-neutral PixelSize and ImageGeometry, FileIdentity.from_path, StageRequest, CompletionRecord, ArtifactStore.reuse, ArtifactStore.commit, and ArtifactStore.mark_stale.

- [ ] **Step 1: Write failing atomic-write tests**

~~~python
from pathlib import Path

import pytest

from cyexm.io.atomic import atomic_write


def test_atomic_write_preserves_compound_extension(tmp_path: Path) -> None:
    destination = tmp_path / "warp.nii.gz"
    seen: list[Path] = []

    def writer(path: Path) -> None:
        seen.append(path)
        path.write_bytes(b"warp")

    atomic_write(destination, writer, lambda path: path.read_bytes() == b"warp")
    assert destination.read_bytes() == b"warp"
    assert seen[0].name.endswith(".nii.gz")
    assert seen[0] != destination


def test_failed_write_keeps_previous_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "transform.mat"
    destination.write_bytes(b"old")

    def writer(path: Path) -> None:
        path.write_bytes(b"partial")
        raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        atomic_write(destination, writer, lambda path: True)
    assert destination.read_bytes() == b"old"
    assert list(tmp_path.glob(".*.tmp*")) == []
~~~

- [ ] **Step 2: Write failing cache lifecycle tests**

~~~python
from pathlib import Path

import pytest

from cyexm.errors import CacheValidationError
from cyexm.registration.artifacts import ArtifactStore, StageRequest
from cyexm.spatial import ImageGeometry


def request(source: Path, accuracy: str = "low") -> StageRequest:
    return StageRequest.for_files(
        stage="affine",
        inputs=(source,),
        geometry=ImageGeometry(
            shape=(8, 16, 16),
            origin=(0.0, 0.0, 0.0),
            spacing=(0.15, 0.15, 0.4),
            direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        method="TRSAA",
        accuracy=accuracy,
        native_threads={"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": 1},
        upstream=None,
        parameters={},
    )


def test_commit_then_reuse(tmp_path: Path) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    artifact = tmp_path / "linear_transform_2.mat"
    store = ArtifactStore()
    store.commit(
        artifact,
        request(source),
        writer=lambda path: path.write_bytes(b"affine"),
        validator=lambda path: path.read_bytes() == b"affine",
    )
    value = store.reuse(artifact, request(source), Path.read_bytes)
    assert value == b"affine"


def test_modified_artifact_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    artifact = tmp_path / "linear_transform_2.mat"
    store = ArtifactStore()
    store.commit(
        artifact,
        request(source),
        writer=lambda path: path.write_bytes(b"affine"),
        validator=lambda path: True,
    )
    artifact.write_bytes(b"corrupt-longer")
    with pytest.raises(CacheValidationError, match="identity"):
        store.reuse(artifact, request(source), Path.read_bytes)


def test_mismatched_request_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "moving.tif"
    source.write_bytes(b"source")
    artifact = tmp_path / "linear_transform_2.mat"
    store = ArtifactStore()
    store.commit(
        artifact,
        request(source, "low"),
        writer=lambda path: path.write_bytes(b"affine"),
        validator=lambda path: True,
    )
    with pytest.raises(CacheValidationError, match="request"):
        store.reuse(artifact, request(source, "high"), Path.read_bytes)
~~~

- [ ] **Step 3: Run both test modules and verify missing APIs**

Run: uv run pytest tests/unit/test_atomic.py tests/unit/test_artifacts.py -v

Expected: FAIL during collection because cyexm.io.atomic and
cyexm.registration.artifacts do not exist.

- [ ] **Step 4: Implement suffix-preserving atomic replacement**

~~~python
# src/cyexm/io/atomic.py
from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path

from cyexm.errors import CacheValidationError


def temporary_sibling(destination: Path) -> Path:
    suffix = "".join(destination.suffixes)
    base = destination.name.removesuffix(suffix)
    return destination.with_name(f".{base}.{uuid.uuid4().hex}.tmp{suffix}")


def atomic_write(
    destination: Path,
    writer: Callable[[Path], None],
    validator: Callable[[Path], bool],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_sibling(destination)
    try:
        writer(temporary)
        if not temporary.is_file() or not validator(temporary):
            raise CacheValidationError(
                f"Temporary artifact failed validation: {temporary}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
~~~

- [ ] **Step 5: Implement canonical completion records**

Define PixelSize as tuple[float, float, float] and ImageGeometry as a frozen,
slotted dataclass in cyexm.spatial. This neutral location prevents cyexm.io from
depending on cyexm.registration.

Use frozen dataclasses whose to_dict output contains only JSON primitives.
FileIdentity contains resolve(strict=True), st_size, and st_mtime_ns.
StageRequest.for_files captures input identities at call time. CompletionRecord
stores schema_version=1, the exact request dictionary, and the committed
artifact identity. JSON uses sort_keys=True, indent=2, and a trailing newline.

~~~python
class ArtifactStore:
    def record_path(self, artifact: Path) -> Path:
        return artifact.with_name(f"{artifact.name}.complete.json")

    def stale_path(self, artifact: Path) -> Path:
        record = self.record_path(artifact)
        return record.with_name(f"{record.name}.stale")

    def reuse(self, artifact: Path, request: StageRequest, reader):
        record_path = self.record_path(artifact)
        if artifact.is_file() and self.stale_path(artifact).is_file() and not record_path.exists():
            return None
        if not artifact.exists() and not record_path.exists():
            return None
        if not artifact.is_file() or not record_path.is_file():
            raise CacheValidationError(f"Incomplete cache entry: {artifact}")
        record = CompletionRecord.read(record_path)
        if record.request != request:
            raise CacheValidationError(f"Cached request mismatch: {artifact}")
        if record.artifact != FileIdentity.from_path(artifact):
            raise CacheValidationError(f"Cached artifact identity mismatch: {artifact}")
        try:
            return reader(artifact)
        except Exception as error:
            raise CacheValidationError(f"Cached artifact is unreadable: {artifact}") from error

    def commit(self, artifact: Path, request: StageRequest, writer, validator) -> None:
        atomic_write(artifact, writer, validator)
        record = CompletionRecord(
            schema_version=1,
            request=request,
            artifact=FileIdentity.from_path(artifact),
        )
        atomic_write(
            self.record_path(artifact),
            lambda path: path.write_text(record.to_json(), encoding="utf-8"),
            lambda path: CompletionRecord.read(path) == record,
        )
        self.stale_path(artifact).unlink(missing_ok=True)

    def mark_stale(self, artifact: Path) -> None:
        record = self.record_path(artifact)
        if record.exists():
            os.replace(record, self.stale_path(artifact))
~~~

Add tests proving a failed replacement leaves the previous artifact and record
usable, a missing half of a cache entry raises, changing an input file rejects
reuse, mark_stale returns a cache miss without deleting the artifact, and a
successful replacement removes the stale marker. A bare artifact with neither
a completion record nor a stale marker remains a CacheValidationError.

- [ ] **Step 6: Run artifact tests and static checks**

Run:

~~~bash
uv run pytest tests/unit/test_atomic.py tests/unit/test_artifacts.py -v
uv run ruff check src/cyexm/io src/cyexm/registration/artifacts.py tests/unit
uv run basedpyright
~~~

Expected: all targeted tests pass with zero lint or type errors.

- [ ] **Step 7: Commit the artifact layer**

~~~bash
git add src/cyexm/io src/cyexm/spatial.py src/cyexm/registration/artifacts.py tests/unit
git commit -m "feat: add restart-safe artifact store"
~~~

## Task 4: Add image I/O, spatial metadata, cropping, and metrics

**Files:**
- Create: src/cyexm/io/images.py
- Create: src/cyexm/io/tiff.py
- Create: src/cyexm/metrics.py
- Create: tests/unit/test_images.py
- Create: tests/unit/test_tiff.py
- Create: tests/unit/test_metrics.py

**Interfaces:**
- Consumes: InputValidationError, cyexm.spatial.ImageGeometry, and atomic_write.
- Produces: load_array(path), load_ants_image(path, pixel_size_xyz), ants_geometry(image), zero_mask(image), robust_bounding_box(array), crop_array(array, bounds), write_uint16_tiff(image, path, min_value, max_value), calculate_metrics, and write_metrics.

- [ ] **Step 1: Write failing array, spacing, and mask tests**

~~~python
from pathlib import Path

import numpy as np
import pytest
import tifffile

from cyexm.errors import InputValidationError
from cyexm.io.images import load_ants_image, load_array, zero_mask


def test_load_tiff_preserves_zyx_and_assigns_xyz_spacing(tmp_path: Path) -> None:
    path = tmp_path / "volume.tif"
    data = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)
    tifffile.imwrite(path, data, metadata={"axes": "ZYX"})
    image = load_ants_image(path, (0.15, 0.15, 0.4))
    np.testing.assert_array_equal(image.numpy(), data)
    assert image.spacing == pytest.approx((0.15, 0.15, 0.4))
    assert image.origin == pytest.approx((0.0, 0.0, 0.0))


def test_load_array_rejects_non_3d_data(tmp_path: Path) -> None:
    path = tmp_path / "plane.npy"
    np.save(path, np.zeros((4, 4), dtype=np.uint16))
    with pytest.raises(InputValidationError, match="three-dimensional"):
        load_array(path)


def test_zero_mask_preserves_geometry() -> None:
    import ants

    image = ants.from_numpy(
        np.array([[[0, 2], [3, 0]]], dtype=np.float32),
        origin=(1.0, 2.0, 3.0),
        spacing=(0.2, 0.3, 0.4),
    )
    mask = zero_mask(image)
    np.testing.assert_array_equal(mask.numpy(), image.numpy() > 0)
    assert mask.origin == image.origin
    assert mask.spacing == image.spacing
    np.testing.assert_array_equal(mask.direction, image.direction)
~~~

- [ ] **Step 2: Write failing TIFF and metric tests**

~~~python
def test_uint16_export_matches_legacy_range_mapping(tmp_path: Path) -> None:
    data = np.array([[[0.0, 5.0, 10.0, np.nan]]], dtype=np.float32)
    destination = tmp_path / "registered.tif"
    write_uint16_tiff(data, destination, min_value=100.0, max_value=1100.0)
    written = tifffile.imread(destination)
    assert written.dtype == np.uint16
    np.testing.assert_array_equal(written, [[[100, 600, 1100, 100]]])


def test_metric_file_is_sorted_and_atomic(tmp_path: Path) -> None:
    destination = tmp_path / "metrics.txt"
    write_metrics(
        {"MattesMutualInformation": 0.25, "Correlation": 0.75},
        destination,
    )
    assert destination.read_text() == (
        "Correlation: 0.75\nMattesMutualInformation: 0.25\n"
    )
~~~

- [ ] **Step 3: Run the tests and verify missing modules**

Run: uv run pytest tests/unit/test_images.py tests/unit/test_tiff.py tests/unit/test_metrics.py -v

Expected: FAIL during collection because the I/O functions are absent.

- [ ] **Step 4: Implement local scientific imports and geometry preservation**

load_array supports .npy, .tif, and .tiff, resolves the input path, verifies it
is a file, loads eagerly, and rejects anything except a three-dimensional
numeric array. load_ants_image validates exactly three positive finite
pixel-size values and constructs the ANTs image with origin (0, 0, 0) and the
public (x, y, z) spacing unchanged. It does not configure threads itself:
registration entry points configure threads before calling I/O, while a direct
I/O caller is responsible for configuring before importing scientific
libraries.

~~~python
def zero_mask(image):
    import ants
    import numpy as np

    data = (image.numpy() > 0).astype(np.float32)
    return ants.from_numpy(
        data,
        origin=image.origin,
        spacing=image.spacing,
        direction=image.direction,
    )


def ants_geometry(image) -> ImageGeometry:
    return ImageGeometry(
        shape=tuple(int(value) for value in image.shape),
        origin=tuple(float(value) for value in image.origin),
        spacing=tuple(float(value) for value in image.spacing),
        direction=tuple(
            tuple(float(value) for value in row)
            for row in image.direction
        ),
    )
~~~

Port identify_robust_bounding_box and identify_minimal_bounding_box from
ClearEx source commit 2311a873, but keep only NumPy input support, move
threshold_otsu into this module, validate percentile and downsampling values,
and return Python ints in (z0, z1, y0, y1, x0, x1) order. Add exact tests for an
off-center cuboid, empty foreground, and exclusive upper bounds.

- [ ] **Step 5: Implement legacy-compatible TIFF and metrics behavior**

write_uint16_tiff accepts an ANTs image or NumPy array, replaces nonfinite values
with zero, clips negatives, maps the data range to the caller's min/max range,
converts to uint16, and writes through atomic_write. A constant image maps to
min_value without division by zero.

calculate_metrics ports the two ClearEx metrics, Correlation and
MattesMutualInformation, and raises RegistrationStageError when either ANTs
metric fails; it must not silently return a partial dictionary. write_metrics
sorts keys and commits the text file atomically.

- [ ] **Step 6: Run targeted and combined tests**

Run:

~~~bash
uv run pytest tests/unit/test_images.py tests/unit/test_tiff.py tests/unit/test_metrics.py -v
uv run pytest tests/unit -q
uv run ruff check src/cyexm tests/unit
uv run basedpyright
~~~

Expected: all unit tests pass and quality checks report zero errors.

- [ ] **Step 7: Commit focused I/O support**

~~~bash
git add src/cyexm/io src/cyexm/metrics.py tests/unit
git commit -m "feat: add registration image IO and metrics"
~~~

## Task 5: Extract and correct the low-level registration kernels

**Files:**
- Create: src/cyexm/registration/AGENTS.md
- Create: src/cyexm/registration/__init__.py
- Create: src/cyexm/registration/linear.py
- Create: src/cyexm/registration/nonlinear.py
- Create: src/cyexm/registration/transforms.py
- Create: src/cyexm/registration/backend.py
- Create: tests/unit/test_linear.py
- Create: tests/unit/test_nonlinear.py
- Create: tests/unit/test_transforms.py
- Create: tests/unit/test_backend.py
- Create: tests/unit/fakes.py

**Interfaces:**
- Consumes: configure_native_threads, NativeThreadConfig, I/O helpers, and registration exceptions.
- Produces: linear.register_image returning (transformed_image, transformed_mask_or_none, affine_transform); nonlinear.register_image returning (transformed_image, warp_path); transform application helpers; RegistrationBackend protocol; and AntsRegistrationBackend.

- [ ] **Step 1: Port the affine-decomposition characterization tests**

Read tests/registration/test_linear.py from ClearEx commit 2311a873. Add its
TestExtractTranslation, TestExtractRotation, TestExtractScale,
TestExtractShear, TestInspectAffineTransform, and
TestAffineMatrixDecomposition classes to tests/unit/test_linear.py. Change only
the import root from clearex.registration.linear to
cyexm.registration.linear. These tests preserve established affine inspection
behavior.

Add registration-contract tests:

~~~python
def test_linear_high_trsaa_uses_established_schedule(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(
        monkeypatch,
        "cyexm.registration.linear",
        tmp_path,
    )
    _, transformed_mask, transform = register_image(
        moving_image=fake_image(),
        fixed_image=fake_image(),
        registration_type="TRSAA",
        accuracy="high",
    )
    assert calls.registration["reg_iterations"] == (1000, 500, 250, 100)
    assert calls.registration["smoothing_sigmas"] == (3, 2, 1, 0)
    assert calls.registration["shrink_factors"] == (8, 4, 2, 1)
    assert transformed_mask is None
    assert transform is calls.transform


@pytest.mark.parametrize("accuracy", ["unknown", "", "HIGHER"])
def test_linear_rejects_unknown_accuracy(accuracy: str) -> None:
    with pytest.raises(ValueError, match="accuracy"):
        register_image(
            moving_image=fake_image(),
            fixed_image=fake_image(),
            registration_type="TRSAA",
            accuracy=accuracy,
        )
~~~

The fake ANTs module records registration kwargs, returns a fake forward
transform path, and provides a transform whose apply_to_image records
interpolation.

- [ ] **Step 2: Write nonlinear and transform-order tests**

~~~python
def test_nonlinear_high_synonly_uses_established_schedule(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = install_fake_ants(
        monkeypatch,
        "cyexm.registration.nonlinear",
        tmp_path,
    )
    transformed, warp_path = register_image(
        moving_image=fake_image(),
        fixed_image=fake_image(),
        registration_type="SyNOnly",
        accuracy="high",
    )
    assert calls.registration["reg_iterations"] == (100, 70, 50)
    assert calls.registration["shrink_factors"] == [4, 2, 1]
    assert calls.registration["smoothing_sigmas"] == [2, 1, 0]
    assert calls.registration["initial_transform"] == "Identity"
    assert transformed is calls.histogram_matched
    assert warp_path == calls.warp_path


def test_transform_chain_uses_warp_then_affine(monkeypatch, tmp_path: Path) -> None:
    calls = install_fake_ants(monkeypatch, "cyexm.registration.transforms")
    apply_transform_chain(
        fixed=fake_image(),
        moving=fake_image(),
        affine_path=tmp_path / "linear.mat",
        warp_path=tmp_path / "warp.nii.gz",
        interpolation="linear",
    )
    assert calls.transformlist == [
        str(tmp_path / "warp.nii.gz"),
        str(tmp_path / "linear.mat"),
    ]
    assert calls.whichtoinvert == [False, False]
~~~

Add parameterized tests for all established accuracy schedules:

- TRSAA high (1000, 500, 250, 100), medium (500, 250, 100, 50), low
  (100, 50, 25, 10), and dry run Rigid (1).
- SyNOnly high (100, 70, 50), medium (50, 35, 25), low (5, 5, 5), and
  dry run (1, 1, 1).
- Linear masks use nearestNeighbor and are thresholded at 0.5.
- Scalar transforms use linear; label transforms use nearestNeighbor.
- Unsupported registration methods and accuracy names raise ValueError before
  calling ANTs.

tests/unit/fakes.py defines every helper referenced above. FakeImage has
dimension=3, shape=(8, 8, 8), identity spatial metadata, numpy(), min(), max(),
and a clone-returning apply_to_image that records interpolation. FakeTransform
inherits that apply behavior. FakeAntsCalls stores registration kwargs,
transformlist, whichtoinvert, transform, warp_path, and histogram_matched.
install_fake_ants creates a real small warp_path under tmp_path, builds an
object with registration, read_transform, threshold_image,
resample_image_to_target, histogram_match_image, apply_transforms, image_read,
image_write, write_transform, and from_numpy callables, and monkeypatches the
target module's private _scientific_runtime function to return that object plus
NumPy. fake_image returns a fresh FakeImage. Each callable records its inputs
and returns the object named by the corresponding FakeAntsCalls field.

- [ ] **Step 3: Run tests and confirm missing kernel modules**

Run:

~~~bash
uv run pytest tests/unit/test_linear.py tests/unit/test_nonlinear.py tests/unit/test_transforms.py tests/unit/test_backend.py -v
~~~

Expected: FAIL because the new registration modules are absent.

- [ ] **Step 4: Extract the kernels with lazy imports and corrected contracts**

Use the named functions from the recorded ClearEx files as extraction input:

- linear.py: register_image, inspect_affine_transform, _extract_scale,
  _extract_shear, _extract_translation, _extract_rotation, and transform_image.
- nonlinear.py: register_image, transform_image, and inspect_warp_transform.
- common.py: export_affine_transform, import_affine_transform, transform_image,
  and set_origin_and_spacing, placed by responsibility in transforms.py.

Do not retain module-level imports of ants, NumPy, SciPy, or tifffile. Each
public operation first calls configure_native_threads(native_threads), then
imports its scientific runtime locally and logs resolved.describe() through the
module's named logger.

Make these intentional defect corrections:

~~~python
def register_image(
    moving_image,
    fixed_image,
    fixed_mask=None,
    moving_mask=None,
    registration_type: str = "TRSAA",
    accuracy: str = "high",
    verbose: bool = False,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> tuple[object, object | None, object]:
    """Return transformed scalar image, optional transformed mask, and affine."""
~~~

Initialize transformed_mask = None before the optional mask branch. Validate
accuracy explicitly. Avoid indexing kwargs["reg_iterations"] for transform
types that do not define it. Preserve Mattes, 32 bins, 0.50 random sampling,
mask_all_stages=True, the established schedules, linear scalar interpolation,
and nearest-neighbor mask interpolation.

~~~python
def register_image(
    moving_image,
    fixed_image,
    moving_mask=None,
    fixed_mask=None,
    registration_type: str = "SyNOnly",
    accuracy: str = "high",
    verbose: bool = False,
    *,
    native_threads: NativeThreadConfig | None = None,
) -> tuple[object, Path]:
    """Return the transformed scalar image and ANTs forward-warp path."""
~~~

Preserve flow_sigma=3.0, total_sigma=0.5, Mattes, 32 bins,
singleprecision=False, initial_transform="Identity", mask_all_stages=True, and
histogram matching to the moving image. Validate that the returned forward
transform path exists before returning it.

- [ ] **Step 5: Implement the backend boundary**

~~~python
class RegistrationBackend(Protocol):
    def load_image(self, path: Path, pixel_size_xyz: PixelSize) -> object:
        raise NotImplementedError

    def zero_mask(self, image: object) -> object:
        raise NotImplementedError

    def geometry(self, image: object) -> ImageGeometry:
        raise NotImplementedError

    def to_array(self, image: object) -> object:
        raise NotImplementedError

    def from_array(
        self,
        array: object,
        *,
        reference: object,
        has_components: bool = False,
    ) -> object:
        raise NotImplementedError

    def crop_image(
        self,
        image: object,
        bounds_zyx: tuple[int, int, int, int, int, int],
    ) -> object:
        raise NotImplementedError

    def linear(self, **kwargs: object) -> tuple[object, object | None, object]:
        raise NotImplementedError

    def nonlinear(self, **kwargs: object) -> tuple[object, Path]:
        raise NotImplementedError

    def apply_affine(
        self,
        *,
        fixed: object,
        moving: object,
        transform: object,
        interpolation: str,
    ) -> object:
        raise NotImplementedError

    def apply_warp(
        self,
        *,
        fixed: object,
        moving: object,
        warp_path: Path,
        interpolation: str,
    ) -> object:
        raise NotImplementedError

    def write_affine(self, transform: object, path: Path) -> None:
        raise NotImplementedError

    def read_affine(self, path: Path) -> object:
        raise NotImplementedError

    def write_warp(self, warp: object, path: Path) -> None:
        raise NotImplementedError

    def read_warp(self, path: Path) -> object:
        raise NotImplementedError

    def write_identity_warp(self, reference: object, path: Path) -> None:
        raise NotImplementedError

    def write_tiff(
        self,
        image: object,
        path: Path,
        *,
        min_value: float,
        max_value: float,
    ) -> None:
        raise NotImplementedError
~~~

AntsRegistrationBackend implements the protocol by delegating to the modules
created in this task. Its constructor accepts NativeThreadConfig and configures
threads before every first scientific import. Tests use a fake backend and do
not import ANTs. PixelSize and ImageGeometry come from cyexm.spatial. Extend the
subprocess import-contract test to import cyexm.registration and assert that
ants, numpy, and scipy remain absent.

- [ ] **Step 6: Run kernel tests and the import contract**

Run:

~~~bash
uv run pytest tests/unit/test_linear.py tests/unit/test_nonlinear.py tests/unit/test_transforms.py tests/unit/test_backend.py -v
uv run pytest tests/unit/test_import_contract.py tests/unit/test_threads.py -v
uv run ruff check src/cyexm/registration tests/unit
uv run basedpyright
~~~

Expected: all tests pass; importing cyexm and cyexm.registration leaves
scientific modules unloaded.

- [ ] **Step 7: Commit the low-level kernels**

~~~bash
git add src/cyexm/registration tests/unit
git commit -m "feat: extract registration kernels"
~~~

## Task 6: Implement restartable whole-image registration

**Files:**
- Create: src/cyexm/registration/whole_image.py
- Create: tests/unit/test_whole_image.py
- Modify: src/cyexm/registration/__init__.py
- Modify: src/cyexm/registration/backend.py
- Modify: src/cyexm/metrics.py
- Modify: tests/unit/fakes.py

**Interfaces:**
- Consumes: RegistrationBackend, ArtifactStore, StageRequest, NativeThreadConfig, and metric functions.
- Produces: RegistrationPaths.for_round, ImageRegistration, and register_round with the stable import path.

- [ ] **Step 1: Write constructor and public-API tests**

~~~python
def test_registration_defaults_are_legacy_compatible(tmp_path: Path) -> None:
    fixed = write_stub(tmp_path / "fixed.tif")
    moving = write_stub(tmp_path / "moving.tif")
    registration = ImageRegistration(fixed, moving, tmp_path / "result")
    assert registration.pixel_size == (0.15, 0.15, 0.15)
    assert registration.imaging_round == 0
    assert registration.linear_type == "TRSAA"
    assert registration.linear_accuracy == "low"
    assert registration.nonlinear_type == "SyNOnly"
    assert registration.nonlinear_accuracy == "high"
    assert registration.crop is False
    assert registration.force_override is False


def test_registration_paths_are_round_scoped(tmp_path: Path) -> None:
    paths = RegistrationPaths.for_round(tmp_path, 3)
    assert paths.affine.name == "linear_transform_3.mat"
    assert paths.warp.name == "nonlinear_warp_3.nii.gz"
    assert paths.registered.name == "nonlinear_registered_3.tif"
    assert paths.fixed_reference.name == "fixed_reference.tif"
~~~

The fixed reference name intentionally replaces the legacy constructor's
nonlinear_registered_1.tif side effect, which could collide with round output.
Record this correction in docs/migration-from-clearex.md in Task 9.

- [ ] **Step 2: Write affine checkpoint and resume tests**

~~~python
def test_affine_survives_nonlinear_failure_and_is_reused(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(fail_nonlinear=True)
    registration = make_registration(tmp_path, backend=backend)
    with pytest.raises(RegistrationStageError, match="nonlinear"):
        registration.register()

    paths = RegistrationPaths.for_round(tmp_path / "output", 2)
    assert paths.affine.is_file()
    assert ArtifactStore().record_path(paths.affine).is_file()
    assert not paths.warp.exists()

    resumed = FakeBackend()
    make_registration(tmp_path, backend=resumed).register()
    assert resumed.linear_calls == 0
    assert resumed.read_affine_calls == 1
    assert resumed.nonlinear_calls == 1
~~~

Extend tests/unit/fakes.py with FakeBackend, a complete RegistrationBackend
implementation over FakeImage. Its load_image returns distinct fixed and moving
images; linear and nonlinear increment counters; nonlinear optionally raises
RuntimeError("injected nonlinear failure") and otherwise writes a temporary
warp under the test directory; transform readers and writers use small byte
files; apply methods return clones; identity-warp and TIFF writers create
readable small files; and metric calculation returns fixed values. The
make_registration helper writes fixed.tif and moving.tif stubs, selects output
round 2, and injects FakeBackend plus a real ArtifactStore.

Add tests that:

- Corrupt affine or missing completion JSON raises CacheValidationError.
- force_override recomputes affine, marks the old warp and final image stale,
  and never reuses them.
- Failure while writing a replacement affine leaves the previous committed
  affine reusable.
- A changed input file, spacing, method, accuracy, or upstream affine identity
  rejects stale reuse.
- A nonlinear failure never writes final image or nonlinear completion records.
- Metrics and final TIFF are committed only after both transforms succeed.
- Crop bounds are committed atomically, reused only for matching inputs, and
  use exclusive (z, y, x) upper indices.
- Library logging uses a named logger without adding root handlers.
- The first run log contains every effective native-thread value and its source.

- [ ] **Step 3: Run the tests and verify missing orchestration**

Run: uv run pytest tests/unit/test_whole_image.py -v

Expected: FAIL because whole_image.py does not exist.

- [ ] **Step 4: Implement paths and the lazy constructor**

~~~python
@dataclass(frozen=True, slots=True)
class RegistrationPaths:
    output: Path
    affine: Path
    warp: Path
    registered: Path
    fixed_reference: Path
    pre_metrics: Path
    linear_metrics: Path
    nonlinear_metrics: Path
    fixed_crop: Path
    moving_crop: Path

    @classmethod
    def for_round(cls, output: Path, round_number: int) -> RegistrationPaths:
        return cls(
            output=output,
            affine=output / f"linear_transform_{round_number}.mat",
            warp=output / f"nonlinear_warp_{round_number}.nii.gz",
            registered=output / f"nonlinear_registered_{round_number}.tif",
            fixed_reference=output / "fixed_reference.tif",
            pre_metrics=output / "image_metrics_pre-registration.txt",
            linear_metrics=output / "image_metrics_linearly-registered.txt",
            nonlinear_metrics=output / "image_metrics_nonlinear-registered.txt",
            fixed_crop=output / f"fixed_crop_indices_{round_number}.json",
            moving_crop=output / f"moving_crop_indices_{round_number}.json",
        )
~~~

ImageRegistration.__init__ stores and validates scalar configuration without
loading NumPy or ANTs. It accepts the legacy positional parameters followed by
keyword-only native_threads, linear_type, linear_accuracy, nonlinear_type,
nonlinear_accuracy, _backend, and _artifact_store. Input existence, output
directory creation, thread application, and image loading occur in register.

- [ ] **Step 5: Implement the stage lifecycle**

The register method follows this exact order:

~~~python
def register(self) -> None:
    resolved = configure_native_threads(self.native_threads)
    backend = self._backend or AntsRegistrationBackend(self.native_threads)
    paths = RegistrationPaths.for_round(Path(self.save_directory), self.imaging_round)
    paths.output.mkdir(parents=True, exist_ok=True)
    self._log.info("Native thread settings: %s", resolved.describe())

    fixed = backend.load_image(Path(self.fixed_image_path), self.pixel_size)
    moving = backend.load_image(Path(self.moving_image_path), self.pixel_size)
    fixed, moving = self._crop_if_requested(fixed, moving, paths)
    fixed_mask = backend.zero_mask(fixed)
    moving_mask = backend.zero_mask(moving)

    self._commit_fixed_reference(backend, fixed, paths, resolved)
    self._commit_metrics("pre-registration", fixed, moving, fixed_mask, moving_mask, paths, resolved)

    affine, moving_linear, moving_linear_mask = self._load_or_estimate_affine(
        backend, fixed, moving, fixed_mask, moving_mask, paths, resolved
    )
    self._commit_metrics("linearly-registered", fixed, moving_linear, fixed_mask, moving_linear_mask, paths, resolved)

    warp_path, moving_nonlinear, moving_nonlinear_mask = self._load_or_estimate_nonlinear(
        backend, fixed, moving_linear, fixed_mask, moving_linear_mask, affine, paths, resolved
    )
    self._commit_metrics("nonlinear-registered", fixed, moving_nonlinear, fixed_mask, moving_nonlinear_mask, paths, resolved)
    self._commit_registered_image(backend, moving_nonlinear, paths, resolved, warp_path)
~~~

Each private stage builds StageRequest from both input files, current geometry,
method, accuracy, resolved thread values, parameters, and upstream artifact
identity. Reuse calls ArtifactStore.reuse with the backend's real reader. New
work writes through ArtifactStore.commit and validates by reopening the
artifact. Catch lower-level exceptions only to add stage context with
RegistrationStageError and raise from the original exception.

When nonlinear registration returns a temporary warp path, copy its bytes only
inside the ArtifactStore writer. Apply that committed warp to the linearly
transformed mask with nearestNeighbor. Never call apply_to_image on an ANTs
vector image.

Factor actual nonlinear estimation into the protected hook
_estimate_nonlinear(backend, fixed, moving_linear, fixed_mask,
moving_linear_mask) -> tuple[object, Path]. The default hook delegates to
backend.nonlinear. The parent _load_or_estimate_nonlinear method owns cache
reuse, full-warp commit, mask transformation, metrics, and final-image
lifecycle. Task 7 overrides only this estimation hook.

- [ ] **Step 6: Implement register_round without hidden behavior**

~~~python
def register_round(
    fixed_image_path: str | os.PathLike[str],
    moving_image_path: str | os.PathLike[str],
    save_directory: str | os.PathLike[str],
    imaging_round: int = 0,
    crop: bool = False,
    enable_logging: bool = True,
    *,
    pixel_size: PixelSize = (0.15, 0.15, 0.15),
    force_override: bool = False,
    native_threads: NativeThreadConfig | None = None,
    linear_type: str = "TRSAA",
    linear_accuracy: str = "low",
    nonlinear_type: str = "SyNOnly",
    nonlinear_accuracy: str = "high",
) -> None:
    ImageRegistration(
        fixed_image_path,
        moving_image_path,
        save_directory,
        imaging_round,
        pixel_size,
        crop,
        enable_logging,
        force_override,
        native_threads=native_threads,
        linear_type=linear_type,
        linear_accuracy=linear_accuracy,
        nonlinear_type=nonlinear_type,
        nonlinear_accuracy=nonlinear_accuracy,
    ).register()
~~~

Export ImageRegistration and register_round from cyexm.registration without
eager scientific imports.

- [ ] **Step 7: Run lifecycle tests**

Run:

~~~bash
uv run pytest tests/unit/test_whole_image.py tests/unit/test_artifacts.py -v
uv run pytest tests/unit/test_import_contract.py -v
uv run ruff check src/cyexm tests/unit
uv run basedpyright
~~~

Expected: all tests pass; the subprocess import test confirms no scientific
runtime is loaded before register is called.

- [ ] **Step 8: Commit whole-image registration**

~~~bash
git add src/cyexm tests/unit
git commit -m "feat: add restartable whole-image registration"
~~~

## Task 7: Extract serial chunked registration

**Files:**
- Create: src/cyexm/registration/chunked.py
- Create: tests/unit/test_chunked_geometry.py
- Create: tests/unit/test_chunked_registration.py
- Modify: src/cyexm/registration/__init__.py
- Modify: src/cyexm/registration/backend.py

**Interfaces:**
- Consumes: ImageRegistration, RegistrationBackend, ArtifactStore, and nonlinear.register_image.
- Produces: ChunkInfo, ChunkPaths.for_chunk, compute_chunk_grid, extract_chunk, stitch_core, and ChunkedImageRegistration.

- [ ] **Step 1: Write pure chunk-grid characterization tests**

Use the legacy algorithm as the numerical reference:

~~~python
def test_single_chunk_covers_smaller_volume() -> None:
    chunks = compute_chunk_grid(
        (32, 64, 96),
        chunk_size=(512, 512, 512),
        overlap_fraction=0.15,
    )
    assert chunks == (
        ChunkInfo(
            chunk_id=0,
            z_start=0, z_end=32,
            y_start=0, y_end=64,
            x_start=0, x_end=96,
            z_start_ext=0, z_end_ext=32,
            y_start_ext=0, y_end_ext=64,
            x_start_ext=0, x_end_ext=96,
        ),
    )


def test_legacy_grid_positions_and_overlap_are_preserved() -> None:
    chunks = compute_chunk_grid(
        (20, 20, 1000),
        chunk_size=(20, 20, 512),
        overlap_fraction=0.15,
    )
    assert [(chunk.x_start, chunk.x_end) for chunk in chunks] == [
        (0, 512),
        (436, 948),
        (872, 1000),
    ]
    assert [(chunk.x_start_ext, chunk.x_end_ext) for chunk in chunks] == [
        (0, 550),
        (398, 986),
        (834, 1000),
    ]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [((0, 8, 8), 0.1), ((8, 8, 8), -0.1), ((8, 8, 8), 1.0)],
)
def test_invalid_chunk_configuration_is_rejected(chunk_size, overlap) -> None:
    with pytest.raises(ValueError):
        compute_chunk_grid((16, 16, 16), chunk_size, overlap)
~~~

Add tests for exclusive upper bounds, no empty chunks, extended-slice shapes,
core-relative slices, and warp component dimension 3.

- [ ] **Step 2: Write serial failure and resume tests**

~~~python
def test_chunk_failure_propagates_without_final_success(tmp_path: Path) -> None:
    backend = FakeChunkBackend(fail_chunk=1)
    registration = make_chunked_registration(
        tmp_path,
        backend=backend,
        chunk_size=(8, 8, 8),
        overlap_fraction=0.0,
    )
    with pytest.raises(ChunkRegistrationError, match="Chunk 1"):
        registration.register()
    assert registration.chunk_paths(0).warp.is_file()
    assert not RegistrationPaths.for_round(tmp_path / "output", 2).warp.exists()


def test_resume_reuses_completed_chunk(tmp_path: Path) -> None:
    first = FakeChunkBackend(fail_chunk=1)
    with pytest.raises(ChunkRegistrationError):
        make_chunked_registration(tmp_path, backend=first).register()

    resumed = FakeChunkBackend()
    make_chunked_registration(tmp_path, backend=resumed).register()
    assert 0 not in resumed.registered_chunk_ids
    assert 1 in resumed.registered_chunk_ids
~~~

Add a test in a fresh subprocess:

~~~python
def test_parallel_chunked_class_is_not_exported() -> None:
    import cyexm.registration as registration

    assert not hasattr(registration, "ParallelChunkedImageRegistration")
~~~

- [ ] **Step 3: Run tests and confirm the chunk module is absent**

Run:

~~~bash
uv run pytest tests/unit/test_chunked_geometry.py tests/unit/test_chunked_registration.py -v
~~~

Expected: FAIL during collection because cyexm.registration.chunked does not
exist.

- [ ] **Step 4: Implement immutable chunk geometry**

Port ChunkInfo and _compute_chunk_grid from the ClearEx baseline into pure
module-level code. Remove transform paths from ChunkInfo; artifact paths are
derived from round output and chunk_id. Validate a three-positive-integer image
shape and chunk size plus 0 <= overlap_fraction < 1. Return a tuple ordered by
z, then y, then x.

~~~python
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

    @property
    def core_shape(self) -> tuple[int, int, int]:
        return (
            self.z_end - self.z_start,
            self.y_end - self.y_start,
            self.x_end - self.x_start,
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
~~~

Add a frozen ChunkPaths dataclass. for_chunk(output, round_number, chunk_id)
returns a round_<round_number>_chunks directory, a
chunk_<chunk_id:04d>_warp.nii.gz path, and the corresponding completion-record
path obtained from ArtifactStore. ChunkedImageRegistration.chunk_paths(chunk_id)
delegates to this constructor, which defines the helper used by the resume
tests.

- [ ] **Step 5: Implement serial registration and per-chunk commits**

ChunkedImageRegistration subclasses ImageRegistration, adds keyword-only
chunk_size=(512, 512, 512) and overlap_fraction=0.15, and overrides only the
_estimate_nonlinear hook. It uses the parent affine, cache, full-warp, metrics,
and final-image stages unchanged.

For each chunk in deterministic order:

1. Extract fixed, moving, and optional mask arrays using extended bounds.
2. Reject empty arrays with ChunkRegistrationError.
3. If either scalar block has standard deviation below 1e-6, commit an identity
   displacement field and return the unchanged moving core.
4. Reject NaN or infinity with ChunkRegistrationError instead of writing a
   successful zero-filled region.
5. Reuse a compatible committed local warp or run nonlinear registration.
6. Commit the local warp and completion record before placing the core result.
7. Place scalar and 3-component warp cores only after shape validation.
8. On any exception, raise ChunkRegistrationError(chunk_id, message) from the
   original exception and do not commit full-volume outputs.

Use float32 for the registered volume and assembled displacement field. After
all chunks complete, commit the chunk manifest JSON and write the assembled
displacement field to a temporary warp path returned with the assembled
registered image. The parent nonlinear lifecycle commits the full-volume warp,
applies it to masks with nearestNeighbor, and commits the registered TIFF and
nonlinear metrics exactly once.

Extend FakeBackend as FakeChunkBackend in tests/unit/fakes.py. It records
registered_chunk_ids, raises on the configured fail_chunk, implements array
conversion with NumPy, and writes valid fake scalar and three-component warp
cores. make_chunked_registration creates deterministic 16 by 8 by 8 fake
volumes, output round 2, chunk_size=(8, 8, 8), and zero overlap unless the test
overrides those values.

- [ ] **Step 6: Run chunk and whole-image regression tests**

Run:

~~~bash
uv run pytest tests/unit/test_chunked_geometry.py tests/unit/test_chunked_registration.py -v
uv run pytest tests/unit/test_whole_image.py -v
uv run ruff check src/cyexm/registration tests/unit
uv run basedpyright
~~~

Expected: all tests pass and the stable registration namespace exports only
ImageRegistration, ChunkedImageRegistration, and register_round.

- [ ] **Step 7: Commit serial chunking**

~~~bash
git add src/cyexm/registration tests/unit
git commit -m "feat: add restartable serial chunk registration"
~~~

## Task 8: Prove real-ANTs behavior and migration parity

**Files:**
- Create: tests/integration/AGENTS.md
- Create: tests/integration/conftest.py
- Create: tests/integration/test_registration_kernels.py
- Create: tests/integration/test_whole_image.py
- Create: tests/integration/test_chunked.py
- Create: tests/integration/legacy_baseline.py
- Create: tests/integration/compare_legacy.py
- Create: docs/legacy-parity-report.md

**Interfaces:**
- Consumes: all stable Phase 1 registration APIs.
- Produces: bounded real-ANTs tests and a reproducible legacy comparison report.

- [ ] **Step 1: Add deterministic synthetic-volume fixtures**

~~~python
@pytest.fixture
def synthetic_pair() -> tuple[np.ndarray, np.ndarray]:
    z, y, x = np.mgrid[:16, :32, :32]
    fixed = np.exp(
        -(
            ((z - 8.0) / 3.0) ** 2
            + ((y - 16.0) / 6.0) ** 2
            + ((x - 16.0) / 6.0) ** 2
        )
    ).astype(np.float32)
    moving = np.roll(fixed, shift=(0, 1, -1), axis=(0, 1, 2))
    fixed[fixed < 1e-4] = 0
    moving[moving < 1e-4] = 0
    return fixed, moving
~~~

Place pytestmark = pytest.mark.integration at module scope in each
test_registration_kernels.py, test_whole_image.py, and test_chunked.py.
The integration environment sets explicit one-thread NativeThreadConfig. The
cross-repository parity case uses identical fixed and moving input so the
legacy random affine sampling does not create a shifted optimum; the separate
shifted-volume test asserts invariants and similarity improvement rather than
pixel identity.

- [ ] **Step 2: Write real kernel and metadata tests**

~~~python
@pytest.mark.integration
def test_dry_run_kernels_preserve_geometry(synthetic_pair) -> None:
    import ants

    fixed_data, moving_data = synthetic_pair
    fixed = ants.from_numpy(fixed_data, spacing=(0.2, 0.2, 0.5))
    moving = ants.from_numpy(moving_data, spacing=(0.2, 0.2, 0.5))
    affine_image, affine_mask, affine = linear.register_image(
        moving,
        fixed,
        moving_mask=zero_mask(moving),
        fixed_mask=zero_mask(fixed),
        accuracy="dry run",
        native_threads=NativeThreadConfig(1, 1),
    )
    nonlinear_image, warp = nonlinear.register_image(
        affine_image,
        fixed,
        accuracy="dry run",
        native_threads=NativeThreadConfig(1, 1),
    )
    assert affine_mask is not None
    assert nonlinear_image.shape == fixed.shape
    assert nonlinear_image.spacing == fixed.spacing
    assert Path(warp).is_file()
    assert np.isfinite(nonlinear_image.numpy()).all()
~~~

Add bounded tests for:

- Whole-image dry-run output and completion records.
- Resume after an injected failure between affine and nonlinear stages.
- A one-chunk serial workflow agreeing with whole-image nonlinear output within
  rtol=1e-4 and atol=1e-5 when both use the same committed warp.
- TIFF output axes, dtype, and ANTs spatial metadata.
- No root logger handlers added by library import or execution.

- [ ] **Step 3: Run bounded integration tests**

Run:

~~~bash
uv run pytest -m integration tests/integration -v
~~~

Expected: all integration tests pass in under five minutes on one CPU allocation.

- [ ] **Step 4: Capture the ClearEx baseline in its own environment**

legacy_baseline.py writes deterministic input TIFFs, uses the fixed volume
itself as the moving volume for the parity case, runs the ClearEx low-level
linear and nonlinear dry-run kernels, and writes output arrays plus a JSON
summary containing shapes, spacing, finite-value checks, similarity metrics,
transform types, source commit, serial chunk-grid coordinates, extended/core
slice shapes, and deterministic overlap-stitch results. compare_legacy.py runs
the equivalent Cy-ExM kernels and chunk helpers, compares geometry and chunk
data exactly, and compares registration arrays with rtol=1e-4 and atol=1e-5.

Run:

~~~bash
git -C /project/bioinformatics/Danuser_lab/Dean/dean/git/clearex rev-parse HEAD
cd /project/bioinformatics/Danuser_lab/Dean/dean/git/clearex
uv run python /project/bioinformatics/Danuser_lab/Dean/dean/git/Cy-ExM/tests/integration/legacy_baseline.py --output /tmp/cyexm-legacy-baseline
cd /project/bioinformatics/Danuser_lab/Dean/dean/git/Cy-ExM
uv run python tests/integration/compare_legacy.py --baseline /tmp/cyexm-legacy-baseline
~~~

Expected: the first command prints exactly
2311a873b783477add7391ff00245f76490147f0; the comparison exits zero and prints
PASS for geometry, affine result, nonlinear result, mask interpolation, and
transform order, chunk grid, chunk extraction, and overlap stitching.

- [ ] **Step 5: Record the parity result**

Write docs/legacy-parity-report.md with the source commit, exact commands,
package versions, tolerances, pass/fail table, and the intentionally corrected
defects: unbound linear mask, nonlinear return mismatch in ImageRegistration,
unsafe cache-by-existence, fixed-reference filename collision, vector-image mask
application, nonlinear-warp reuse through the wrong ANTs transform API,
stage-metric calls that reused the original moving scalar image, and suppressed
parallel-worker failures.

- [ ] **Step 6: Commit integration evidence**

~~~bash
git add tests/integration docs/legacy-parity-report.md
git commit -m "test: verify ANTs registration parity"
~~~

## Task 9: Migrate examples and complete repository guidance

**Files:**
- Create: examples/AGENTS.md
- Create: examples/scripts/AGENTS.md
- Create: examples/notebooks/AGENTS.md
- Create: examples/scripts/whole_image_registration.py
- Create: examples/scripts/linear_registration.py
- Create: examples/scripts/nonlinear_registration.py
- Create: examples/scripts/apply_transforms.py
- Create: examples/scripts/chunked_registration.py
- Create: examples/notebooks/apply_warp_transform.ipynb
- Create: examples/notebooks/chunked_nonlinear_registration.ipynb
- Create: examples/notebooks/evaluate_histogram_matching.ipynb
- Create: examples/notebooks/image_registration_class.ipynb
- Create: examples/notebooks/nonlinear_registration.ipynb
- Create: examples/notebooks/transform_directory.ipynb
- Create: tests/unit/test_examples.py
- Create: docs/AGENTS.md
- Create: docs/migration-from-clearex.md
- Create: docs/registration-artifacts.md
- Modify: README.md

**Interfaces:**
- Consumes: stable Cy-ExM APIs and documented artifact names.
- Produces: parameterized examples, complete nested AGENTS.md guidance, and user-facing migration/restart documentation.

- [ ] **Step 1: Write the failing example-policy test**

~~~python
from pathlib import Path


FORBIDDEN = (
    "/archive/bioinformatics",
    "/endosome/archive",
    "/project/bioinformatics",
    "import clearex",
    "from clearex",
    "ParallelChunkedImageRegistration",
)


def test_examples_are_portable_and_use_cyexm() -> None:
    paths = [
        *Path("examples/scripts").glob("*.py"),
        *Path("examples/notebooks").glob("*.ipynb"),
    ]
    assert paths
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN:
            assert forbidden not in text, f"{forbidden!r} found in {path}"
        assert "cyexm" in text


def test_notebooks_have_no_saved_outputs() -> None:
    import json

    for path in Path("examples/notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("outputs", []) == []
                assert cell.get("execution_count") is None
~~~

- [ ] **Step 2: Run the policy test and verify there are no examples**

Run: uv run pytest tests/unit/test_examples.py -v

Expected: FAIL because no example files exist.

- [ ] **Step 3: Rewrite the five script workflows with argparse**

Each script defines main(argv: Sequence[str] | None = None) -> int, accepts
input and output paths as command-line arguments, accepts --pixel-size-x,
--pixel-size-y, and --pixel-size-z, and uses
NativeThreadConfig(--itk-threads, --blas-threads). The whole-image and chunked
scripts also accept --round, --linear-accuracy, --nonlinear-accuracy, and
--force. No script downloads data, assumes a repository-relative dataset, or
configures Dask.

Every script ends with:

~~~python
if __name__ == "__main__":
    raise SystemExit(main())
~~~

Run python -m compileall examples/scripts after creation.

- [ ] **Step 4: Migrate and clear the six notebooks**

Use the corresponding ClearEx notebooks as narrative input, replace imports
with Cy-ExM, replace hard-coded paths with a first cell containing Path
variables that the reader edits, and remove dependencies on ClearEx plotting,
statistics, and file-operation modules. Use NumPy and Matplotlib locally where
the visualization is germane. Do not copy downloaded_data.zip.

Clear outputs mechanically:

~~~bash
uv run jupyter nbconvert --clear-output --inplace examples/notebooks/*.ipynb
~~~

Run the policy test after clearing.

- [ ] **Step 5: Write migration, artifact, README, and AGENTS guidance**

docs/migration-from-clearex.md contains the import map from the design, source
commit, known data_analysis/dean_lab/2025-galecki consumer note, list of moved
examples, excluded pipeline and tre modules, and the intentional corrections
from Task 8.

docs/registration-artifacts.md lists every filename, its completion JSON,
request fields, validation behavior, stale-record behavior, force semantics,
and recovery examples after affine, nonlinear, and chunk failures.

README.md includes installation with uv, the conservative thread rationale,
whole-image and serial chunked examples, (x, y, z) spacing, output/restart
behavior, migration link, and an explicit statement that round-level
Dask/SLURM support belongs to Phase 2.

Nested AGENTS.md files contain the detailed layer-specific constraints listed in
the approved design. In particular, registration/AGENTS.md requires scientific
invariant review for transform, axis, interpolation, and schedule changes;
runtime/AGENTS.md prohibits scientific imports; tests AGENTS.md defines
unit/integration/hpc markers and tolerance rationale; examples AGENTS.md bans
embedded lab paths and committed notebook outputs.

- [ ] **Step 6: Run example and documentation checks**

Run:

~~~bash
uv run pytest tests/unit/test_examples.py -v
uv run python -m compileall -q examples/scripts
rg -n "/archive/bioinformatics|/endosome/archive|/project/bioinformatics|clearex\\.registration|ParallelChunked" README.md docs examples
~~~

Expected: tests pass, compileall exits zero, and rg reports only deliberate
historical references inside docs/migration-from-clearex.md and the parity
report.

- [ ] **Step 7: Commit examples and documentation**

~~~bash
git add README.md AGENTS.md src/AGENTS.md src/cyexm/AGENTS.md src/cyexm/*/AGENTS.md tests examples docs
git commit -m "docs: migrate registration examples and guidance"
~~~

## Task 10: Add CI and run the Phase 1 acceptance audit

**Files:**
- Create: .pre-commit-config.yaml
- Create: .github/workflows/ci.yml
- Create: docs/phase-1-acceptance.md
- Modify: pyproject.toml
- Modify: README.md

**Interfaces:**
- Consumes: the full Phase 1 package, tests, examples, and docs.
- Produces: reproducible portable checks and recorded acceptance evidence.

- [ ] **Step 1: Add pre-commit and CI configuration**

Use local uv-backed hooks so the same locked tools run locally and in CI:

~~~yaml
repos:
  - repo: local
    hooks:
      - id: ruff-check
        name: Ruff check
        entry: uv run ruff check
        language: system
        types_or: [python, pyi]
      - id: ruff-format
        name: Ruff format check
        entry: uv run ruff format --check
        language: system
        types_or: [python, pyi]
      - id: basedpyright
        name: BasedPyright
        entry: uv run basedpyright
        language: system
        pass_filenames: false
      - id: pytest-unit
        name: Unit tests
        entry: uv run pytest tests/unit -q
        language: system
        pass_filenames: false
~~~

GitHub Actions uses Ubuntu, Python 3.12, astral-sh/setup-uv, uv sync
--group dev, Ruff check/format, BasedPyright, pytest tests/unit, and bounded
pytest -m integration tests/integration. It does not run hpc tests or access
lab storage. Set all six native-thread environment variables to 1 at the job
level before Python starts.

- [ ] **Step 2: Run formatting and apply only formatter changes**

Run:

~~~bash
uv run ruff format .
uv run ruff check . --fix
git diff --check
~~~

Expected: commands exit zero and git diff --check prints nothing. Inspect every
non-format Ruff fix before staging.

- [ ] **Step 3: Run the complete portable validation suite**

Run:

~~~bash
uv sync --group dev --extra examples
uv run pytest tests/unit -v
uv run pytest -m integration tests/integration -v
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
uv run pre-commit run --all-files
~~~

Expected: every command exits zero, no tests are silently skipped, and no hpc
test is selected.

- [ ] **Step 4: Audit forbidden dependencies and exports**

Run:

~~~bash
uv run python -c "import cyexm.registration as r; assert not hasattr(r, 'ParallelChunkedImageRegistration')"
uv run python -c "import importlib.metadata as m; deps=m.requires('cy-exm') or []; forbidden=('clearex','dask','distributed','dask-jobqueue','pyyaml'); assert not any(any(word in dep.lower() for word in forbidden) for dep in deps), deps"
rg -n "ParallelChunkedImageRegistration|import clearex|from clearex|dask|SLURMCluster|pyyaml" src tests/unit examples pyproject.toml
~~~

Expected: both Python commands exit zero. rg returns only the negative export
assertion in tests and explanatory text that is explicitly inspected; no
runtime import or dependency is present.

- [ ] **Step 5: Record acceptance evidence**

docs/phase-1-acceptance.md records:

- git commit and date;
- Python, antspyx, ITK, NumPy, SciPy, and tifffile versions;
- exact validation commands and their pass counts;
- legacy parity report link;
- import-side-effect result;
- thread default/override result;
- affine and chunk resume result;
- examples migrated;
- forbidden-dependency/export result; and
- deferred Phase 2 scope.

Do not create a version tag or GitHub release in this task. Release publication
requires a separate explicit decision after code review.

- [ ] **Step 6: Commit the acceptance gate**

~~~bash
git add .github .pre-commit-config.yaml pyproject.toml README.md docs tests src examples uv.lock
git commit -m "ci: enforce Phase 1 acceptance gates"
~~~

- [ ] **Step 7: Verify the final branch state**

Run:

~~~bash
git status --short
git log --oneline --decorate -10
uv run pytest tests/unit tests/integration -q
~~~

Expected: git status prints nothing, the log shows one focused commit per task,
and the complete portable suite passes.

## Execution completion

After Task 10, use superpowers:verification-before-completion before claiming
success. Then use superpowers:requesting-code-review and
superpowers:finishing-a-development-branch to review and choose whether to
merge, open a pull request, or retain the implementation branch. Do not remove
the corresponding legacy ClearEx code or update external analysis repositories
until the Cy-ExM implementation has been reviewed and the migration step is
separately authorized.
