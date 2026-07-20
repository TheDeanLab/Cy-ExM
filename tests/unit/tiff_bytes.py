"""Standard-library TIFF fixtures for lightweight workflow-status tests."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from cyexm.spatial import AXIS_CONVENTION_VERSION, ImageGeometry


def _entry(tag: int, field_type: int, count: int, value: int) -> bytes:
    if field_type == 3 and count == 1:
        encoded = struct.pack("<H", value) + b"\0\0"
    else:
        encoded = struct.pack("<I", value)
    return struct.pack("<HHI", tag, field_type, count) + encoded


def write_cyexm_uint16_tiff(path: Path, geometry: ImageGeometry) -> None:
    """Write a small classic, uncompressed, scalar uint16 ZYX TIFF fixture."""
    description = (
        json.dumps(
            {
                "shape": list(geometry.shape),
                "axes": "ZYX",
                "cyexm": {
                    "schema_version": 1,
                    "axis_convention_version": AXIS_CONVENTION_VERSION,
                    "geometry": geometry.to_dict(),
                },
            }
        ).encode("ascii")
        + b"\0"
    )
    depth, height, width = geometry.shape
    byte_count = height * width * 2
    page_layouts: list[tuple[int, int | None, int, int]] = []
    cursor = 8
    for page in range(depth):
        entry_count = 12 if page == 0 else 11
        ifd_size = 2 + entry_count * 12 + 4
        description_offset = cursor + ifd_size if page == 0 else None
        payload_offset = cursor + ifd_size + (len(description) if page == 0 else 0)
        if payload_offset % 2:
            payload_offset += 1
        block_end = payload_offset + byte_count
        next_ifd = block_end if page + 1 < depth else 0
        page_layouts.append((cursor, description_offset, payload_offset, next_ifd))
        cursor = block_end

    encoded = bytearray(cursor)
    encoded[:8] = b"II" + struct.pack("<HI", 42, 8)
    for page, (ifd_offset, description_offset, payload_offset, next_ifd) in enumerate(
        page_layouts
    ):
        entries = [
            _entry(256, 4, 1, width),
            _entry(257, 4, 1, height),
            _entry(258, 3, 1, 16),
            _entry(259, 3, 1, 1),
            _entry(262, 3, 1, 1),
        ]
        if page == 0:
            assert description_offset is not None
            entries.append(_entry(270, 2, len(description), description_offset))
        entries.extend(
            (
                _entry(273, 4, 1, payload_offset),
                _entry(277, 3, 1, 1),
                _entry(278, 4, 1, height),
                _entry(279, 4, 1, byte_count),
                _entry(284, 3, 1, 1),
                _entry(339, 3, 1, 1),
            )
        )
        entries.sort(key=lambda entry: struct.unpack_from("<H", entry)[0])
        ifd = struct.pack("<H", len(entries)) + b"".join(entries)
        ifd += struct.pack("<I", next_ifd)
        encoded[ifd_offset : ifd_offset + len(ifd)] = ifd
        if page == 0:
            assert description_offset is not None
            encoded[description_offset : description_offset + len(description)] = (
                description
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


__all__ = ["write_cyexm_uint16_tiff"]
