from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("expected exactly one TIFF path", file=sys.stderr)
        return 2

    from cyexm.runtime import NativeThreadConfig, configure_native_threads

    configure_native_threads(NativeThreadConfig(itk_threads=1, blas_threads=1))
    from cyexm.workflow.planning import (
        _header_mapping,
        _inspect_tiff_header_in_process,
    )

    try:
        header = _inspect_tiff_header_in_process(Path(arguments[0]))
    except Exception as error:
        print(" ".join(str(error).splitlines()), file=sys.stderr)
        return 1
    print(json.dumps(_header_mapping(header), allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
