from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path


def registration_transform_paths(
    registration: Mapping[str, object],
) -> set[Path]:
    """Return the deduplicated native transform paths owned by a registration."""
    paths: set[Path] = set()
    for key in ("fwdtransforms", "invtransforms"):
        values = registration.get(key, ())
        if isinstance(values, (str, os.PathLike)):
            values = (values,)
        if not isinstance(values, Iterable):
            continue
        for value in values:
            if isinstance(value, (str, os.PathLike)):
                paths.add(Path(value))
    return paths


def cleanup_registration_transforms(
    paths: Iterable[Path],
    *,
    preserve: Iterable[Path] = (),
) -> None:
    """Best-effort cleanup of native paths without deleting transferred ownership."""
    preserved = {Path(path) for path in preserve}
    for path in {Path(path) for path in paths} - preserved:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Cleanup must not mask a registration or postprocessing exception.
            continue


__all__: list[str] = []
