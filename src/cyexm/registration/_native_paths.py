from __future__ import annotations

from contextlib import suppress
from pathlib import Path


def discard_path(path: Path | None) -> None:
    """Best-effort removal for one disposable path owned by the current caller."""
    if path is None:
        return
    with suppress(OSError):
        path.unlink(missing_ok=True)


__all__: list[str] = []
