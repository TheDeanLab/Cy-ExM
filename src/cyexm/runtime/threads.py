from __future__ import annotations

import os
import sys
from collections.abc import Mapping, MutableMapping, Set
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


def _positive_explicit_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ThreadConfigurationError(f"{name} must be a positive integer")
    return value


def _positive_environment_int(value: str, name: str) -> int:
    try:
        parsed = int(value, 10)
    except (TypeError, ValueError) as error:
        raise ThreadConfigurationError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise ThreadConfigurationError(f"{name} must be a positive integer")
    return parsed


def configure_native_threads(
    config: NativeThreadConfig | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
    loaded_modules: Set[str] | None = None,
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
            config.itk_threads if name == ITK_THREAD_ENV_VAR else config.blas_threads
        )
        if explicit is not None:
            value = _positive_explicit_int(explicit, name)
            source = "explicit"
        elif name in target_environ:
            value = _positive_environment_int(target_environ[name], name)
            source = "environment"
        else:
            value = 1
            source = "default"

        current = target_environ.get(name)
        if (
            runtime_loaded
            and current is not None
            and _positive_environment_int(current, name) != value
        ):
            loaded = ", ".join(sorted(loaded_roots & SCIENTIFIC_MODULE_ROOTS))
            raise ThreadConfigurationError(
                f"Cannot change {name} after scientific runtime already loaded: "
                f"{loaded}"
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
