from __future__ import annotations

import pytest

from cyexm.errors import ThreadConfigurationError
from cyexm.runtime import (
    NATIVE_THREAD_ENV_VARS,
    NativeThreadConfig,
    configure_native_threads,
)


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
    assert all(
        value == 3
        for name, value in resolved.values.items()
        if name != "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"
    )
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
    environ = {
        name: "1"
        for name in (
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
    resolved = configure_native_threads(
        NativeThreadConfig(itk_threads=1, blas_threads=1),
        environ=environ,
        loaded_modules={"ants", "numpy"},
    )
    assert set(resolved.values.values()) == {1}


def test_describe_reports_values_and_sources_in_stable_order() -> None:
    resolved = configure_native_threads(
        environ={},
        loaded_modules=set(),
    )

    assert resolved.describe() == ", ".join(
        f"{name}=1 (default)" for name in NATIVE_THREAD_ENV_VARS
    )
