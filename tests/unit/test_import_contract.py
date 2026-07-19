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
