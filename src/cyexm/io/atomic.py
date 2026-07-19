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
