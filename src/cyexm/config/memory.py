"""Strict semantic-to-scheduler memory normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass

from cyexm.errors import ConfigurationValidationError

_DECIMAL_MEMORY = re.compile(r"([1-9][0-9]*)(KB|MB|GB|TB)")
_DECIMAL_MULTIPLIERS = {
    "KB": 1_000,
    "MB": 1_000_000,
    "GB": 1_000_000_000,
    "TB": 1_000_000_000_000,
}
_MAX_MEMORY_BYTES = (1 << 63) - 1
_BINARY_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class SlurmMemoryQuantity:
    """One semantic profile quantity and its exact scheduler/Dask spellings."""

    profile: str
    semantic_bytes: int
    scheduler: str
    scheduler_bytes: int
    dask: str


def normalize_slurm_memory(
    value: object,
    location: str = "slurm.memory",
) -> SlurmMemoryQuantity:
    """Parse strict decimal memory and ceil it to a SLURM binary quantity."""
    if type(value) is not str or (match := _DECIMAL_MEMORY.fullmatch(value)) is None:
        raise ConfigurationValidationError(
            f"{location} must be a positive whole decimal quantity with an exact "
            f"KB, MB, GB, or TB suffix; got {value!r}"
        )
    amount_text, decimal_unit = match.groups()
    amount = int(amount_text, 10)
    decimal_multiplier = _DECIMAL_MULTIPLIERS[decimal_unit]
    if amount > _MAX_MEMORY_BYTES // decimal_multiplier:
        raise ConfigurationValidationError(
            f"{location} exceeds the supported signed 64-bit byte range; got {value!r}"
        )
    semantic_bytes = amount * decimal_multiplier
    scheduler_amount = (semantic_bytes + _BINARY_GIB - 1) // _BINARY_GIB
    if scheduler_amount > _MAX_MEMORY_BYTES // _BINARY_GIB:
        raise ConfigurationValidationError(
            f"{location} exceeds the supported signed 64-bit byte range after "
            f"SLURM ceiling normalization; got {value!r}"
        )
    scheduler_bytes = scheduler_amount * _BINARY_GIB
    return SlurmMemoryQuantity(
        profile=value,
        semantic_bytes=semantic_bytes,
        scheduler=f"{scheduler_amount}G",
        scheduler_bytes=scheduler_bytes,
        dask=f"{scheduler_amount}GiB",
    )


__all__ = ["SlurmMemoryQuantity", "normalize_slurm_memory"]
