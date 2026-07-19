from __future__ import annotations

from typing import cast

import pytest

from cyexm.errors import ThreadConfigurationError
from cyexm.runtime import (
    NATIVE_THREAD_ENV_VARS,
    NativeThreadConfig,
    configure_native_threads,
)

pytestmark = pytest.mark.unit


class _IntSubclass(int):
    pass


class _DeceptiveIntSubclass(int):
    def __getattribute__(self, name: str) -> object:
        if name == "__class__":
            return int
        return super().__getattribute__(name)


def _config_with_explicit_value(field: str, value: object) -> NativeThreadConfig:
    typed_value = cast(int, value)
    if field == "itk_threads":
        return NativeThreadConfig(itk_threads=typed_value)
    assert field == "blas_threads"
    return NativeThreadConfig(blas_threads=typed_value)


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


@pytest.mark.parametrize("field", ["itk_threads", "blas_threads"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="boolean"),
        pytest.param(2.0, id="integer-valued-float"),
        pytest.param(2.5, id="fractional-float"),
        pytest.param("2", id="string"),
        pytest.param(0, id="zero"),
        pytest.param(-2, id="negative"),
        pytest.param(_IntSubclass(2), id="int-subclass"),
        pytest.param(_DeceptiveIntSubclass(2), id="deceptive-int-subclass"),
        pytest.param(object(), id="unrelated-object"),
    ],
)
def test_invalid_explicit_values_are_rejected_without_environment_mutation(
    field: str,
    value: object,
) -> None:
    environ = {
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "7",
        "PRESERVE_ME": "unchanged",
    }
    before = dict(environ)
    config = _config_with_explicit_value(field, value)

    with pytest.raises(ThreadConfigurationError, match="positive integer"):
        configure_native_threads(
            config,
            environ=environ,
            loaded_modules=set(),
        )

    assert environ == before


@pytest.mark.parametrize(
    ("field", "value", "expected_names"),
    [
        pytest.param(
            "itk_threads",
            2,
            (NATIVE_THREAD_ENV_VARS[0],),
            id="itk",
        ),
        pytest.param(
            "blas_threads",
            3,
            NATIVE_THREAD_ENV_VARS[1:],
            id="blas",
        ),
    ],
)
def test_builtin_integer_explicit_values_are_accepted(
    field: str,
    value: int,
    expected_names: tuple[str, ...],
) -> None:
    environ: dict[str, str] = {}

    resolved = configure_native_threads(
        _config_with_explicit_value(field, value),
        environ=environ,
        loaded_modules=set(),
    )

    assert all(resolved.values[name] == value for name in expected_names)
    assert all(resolved.sources[name] == "explicit" for name in expected_names)


@pytest.mark.parametrize(
    "raw_value",
    [
        pytest.param("6", id="digits"),
        pytest.param(" 6 ", id="surrounding-whitespace"),
        pytest.param("+6", id="leading-plus"),
    ],
)
def test_inherited_positive_integer_strings_beat_defaults_per_variable(
    raw_value: str,
) -> None:
    environ = {"OPENBLAS_NUM_THREADS": raw_value}
    resolved = configure_native_threads(
        environ=environ,
        loaded_modules=set(),
    )
    assert resolved.values["OPENBLAS_NUM_THREADS"] == 6
    assert resolved.sources["OPENBLAS_NUM_THREADS"] == "environment"
    assert resolved.values["MKL_NUM_THREADS"] == 1


@pytest.mark.parametrize(
    "name",
    [NATIVE_THREAD_ENV_VARS[0], NATIVE_THREAD_ENV_VARS[-1]],
    ids=["itk-first", "blas-last"],
)
@pytest.mark.parametrize(
    "raw_value",
    [
        pytest.param("2.5", id="fractional"),
        pytest.param("six", id="nonnumeric"),
        pytest.param("0", id="zero"),
        pytest.param("-2", id="negative"),
    ],
)
def test_invalid_environment_strings_reject_without_partial_mutation(
    name: str,
    raw_value: str,
) -> None:
    environ = {name: raw_value, "PRESERVE_ME": "unchanged"}
    before = dict(environ)

    with pytest.raises(ThreadConfigurationError, match="positive integer"):
        configure_native_threads(environ=environ, loaded_modules=set())

    assert environ == before


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


def test_late_idempotent_comparison_uses_environment_string_parser() -> None:
    environ = {name: " +1 " for name in NATIVE_THREAD_ENV_VARS}

    resolved = configure_native_threads(
        NativeThreadConfig(itk_threads=1, blas_threads=1),
        environ=environ,
        loaded_modules={"numpy"},
    )

    assert set(resolved.values.values()) == {1}
    assert set(environ.values()) == {"1"}


def test_invalid_late_environment_value_rejects_without_mutation() -> None:
    environ = {name: "1" for name in NATIVE_THREAD_ENV_VARS}
    environ[NATIVE_THREAD_ENV_VARS[-1]] = "2.5"
    before = dict(environ)

    with pytest.raises(ThreadConfigurationError, match="positive integer"):
        configure_native_threads(
            NativeThreadConfig(itk_threads=1, blas_threads=1),
            environ=environ,
            loaded_modules={"numpy"},
        )

    assert environ == before


def test_describe_reports_values_and_sources_in_stable_order() -> None:
    resolved = configure_native_threads(
        environ={},
        loaded_modules=set(),
    )

    assert resolved.describe() == ", ".join(
        f"{name}=1 (default)" for name in NATIVE_THREAD_ENV_VARS
    )
