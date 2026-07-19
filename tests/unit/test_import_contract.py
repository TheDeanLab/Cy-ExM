from __future__ import annotations

import subprocess
import sys


def test_import_cyexm_is_lightweight_and_side_effect_free() -> None:
    code = """
import os
import sys

before = dict(os.environ)
import cyexm
after = dict(os.environ)

assert before == after
assert "ants" not in sys.modules
assert "numpy" not in sys.modules
assert "scipy" not in sys.modules
assert isinstance(cyexm.__version__, str)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_import_cyexm_runtime_keeps_scientific_libraries_unloaded() -> None:
    code = """
import sys

import cyexm.runtime

assert "ants" not in sys.modules
assert "numpy" not in sys.modules
assert "scipy" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_registration_modules_keep_scientific_libraries_unloaded() -> None:
    code = """
import os
import sys

before = dict(os.environ)
import cyexm.registration
import cyexm.registration.backend
import cyexm.registration.linear
import cyexm.registration.nonlinear
import cyexm.registration.transforms
import cyexm.registration.whole_image
after = dict(os.environ)

assert before == after
for root in ("ants", "numpy", "scipy", "tifffile"):
    assert not any(name == root or name.startswith(root + ".") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_whole_image_constructor_is_import_and_filesystem_neutral() -> None:
    code = """
import logging
import os
from pathlib import Path
import sys
import tempfile

before_environment = dict(os.environ)
before_handlers = tuple(logging.getLogger().handlers)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    output = root / "output"
    from cyexm.registration.whole_image import ImageRegistration
    ImageRegistration(root / "missing-fixed.tif", root / "missing-moving.tif", output)
    assert not output.exists()

assert before_environment == dict(os.environ)
assert before_handlers == tuple(logging.getLogger().handlers)
for root in ("ants", "numpy", "scipy", "tifffile"):
    assert not any(name == root or name.startswith(root + ".") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
