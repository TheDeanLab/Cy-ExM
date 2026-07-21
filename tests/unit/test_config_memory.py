from __future__ import annotations

import pytest

from cyexm.config import SlurmMemoryQuantity, normalize_slurm_memory
from cyexm.errors import ConfigurationValidationError

pytestmark = pytest.mark.unit


class _StringAlias(str):
    pass


def test_approved_decimal_memory_rounds_up_to_binary_gib() -> None:
    assert normalize_slurm_memory("500GB") == SlurmMemoryQuantity(
        profile="500GB",
        semantic_bytes=500_000_000_000,
        scheduler="466G",
        scheduler_bytes=500_363_689_984,
        dask="466GiB",
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (
            "1KB",
            SlurmMemoryQuantity("1KB", 1_000, "1G", 1_073_741_824, "1GiB"),
        ),
        (
            "1MB",
            SlurmMemoryQuantity(
                "1MB",
                1_000_000,
                "1G",
                1_073_741_824,
                "1GiB",
            ),
        ),
        (
            "1GB",
            SlurmMemoryQuantity(
                "1GB",
                1_000_000_000,
                "1G",
                1_073_741_824,
                "1GiB",
            ),
        ),
        (
            "1TB",
            SlurmMemoryQuantity(
                "1TB",
                1_000_000_000_000,
                "932G",
                1_000_727_379_968,
                "932GiB",
            ),
        ),
    ),
)
def test_decimal_units_always_ceil_to_whole_binary_gib(
    value: str,
    expected: SlurmMemoryQuantity,
) -> None:
    assert normalize_slurm_memory(value) == expected


@pytest.mark.parametrize(
    "value",
    (
        None,
        True,
        500,
        _StringAlias("500GB"),
        "",
        "0GB",
        "-1GB",
        "+1GB",
        "1.5GB",
        "500G",
        "466GiB",
        "500gb",
        " 500GB",
        "500GB ",
        "500 GB",
        "9223372036854776KB",
        "9223372036854775KB",
    ),
)
def test_slurm_memory_rejects_invalid_or_overflowing_quantities(value: object) -> None:
    with pytest.raises(ConfigurationValidationError, match=r"slurm\.memory"):
        normalize_slurm_memory(value)


def test_slurm_memory_error_uses_the_callers_location() -> None:
    with pytest.raises(ConfigurationValidationError, match=r"profile\.slurm\.memory"):
        normalize_slurm_memory("500G", "profile.slurm.memory")
