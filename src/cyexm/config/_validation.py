import math
from collections.abc import Mapping
from pathlib import PurePosixPath

from cyexm.errors import ConfigurationValidationError


def require_exact_keys(
    value: object, expected: frozenset[str], yaml_path: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationValidationError(
            f"{yaml_path} must be a mapping; got {value!r}"
        )
    if not all(isinstance(key, str) for key in value):
        raise ConfigurationValidationError(
            f"{yaml_path} mapping keys must be strings; got {value!r}"
        )
    actual = frozenset(value)
    if actual != expected:
        raise ConfigurationValidationError(
            f"{yaml_path} keys must be exactly {sorted(expected)!r}; "
            f"got {sorted(actual)!r}"
        )
    return value


def built_in_int(value: object, yaml_path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationValidationError(
            f"{yaml_path} must be a built-in integer >= {minimum}; got {value!r}"
        )
    return value


def finite_positive_float(value: object, yaml_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationValidationError(
            f"{yaml_path} must be a positive finite number; got {value!r}"
        )
    try:
        result = float(value)
    except OverflowError as error:
        raise ConfigurationValidationError(
            f"{yaml_path} must be a positive finite number; got {value!r}"
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise ConfigurationValidationError(
            f"{yaml_path} must be a positive finite number; got {value!r}"
        )
    return result


def nonempty_string(value: object, yaml_path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ConfigurationValidationError(
            f"{yaml_path} must be a nonempty single-line string; got {value!r}"
        )
    return value


def relative_posix_path(value: object, yaml_path: str) -> PurePosixPath:
    text = nonempty_string(value, yaml_path)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path == PurePosixPath(".")
        or any(part in {"", ".", ".."} for part in text.split("/"))
    ):
        raise ConfigurationValidationError(
            f"{yaml_path} must be a safe relative POSIX path; got {value!r}"
        )
    return path
