from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


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


def test_phase_two_configuration_and_planning_imports_are_lightweight() -> None:
    code = """
import os
import sys

before = dict(os.environ)
import cyexm.config
import cyexm.workflow
import cyexm.execution
import cyexm.cli

assert before == dict(os.environ)
for root in (
    "ants",
    "numpy",
    "scipy",
    "tifffile",
    "dask",
    "distributed",
    "dask_jobqueue",
):
    assert not any(name == root or name.startswith(root + ".") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_round_worker_import_defers_registration_operations_until_invocation() -> None:
    code = """
import os
import sys

before = dict(os.environ)
import cyexm.workflow.round_task

assert before == dict(os.environ)
for name in (
    "cyexm.registration.channels",
    "cyexm.registration.chunked",
    "cyexm.registration.whole_image",
):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_round_task_module_defers_scientific_imports_until_execution() -> None:
    code = """
import os
import sys

before = dict(os.environ)
import cyexm.registration.channels
import cyexm.workflow.round_task

assert before == dict(os.environ)
for root in (
    "ants",
    "numpy",
    "scipy",
    "tifffile",
    "dask",
    "distributed",
    "dask_jobqueue",
):
    assert not any(name == root or name.startswith(root + ".") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_registration_namespace_resolves_stable_exports_lazily() -> None:
    code = """
import sys
import cyexm.registration as registration

assert registration.__all__ == [
    "ChunkedImageRegistration",
    "ImageRegistration",
    "register_round",
    "transform_channel",
]
for name in (
    "cyexm.registration.channels",
    "cyexm.registration.chunked",
    "cyexm.registration.whole_image",
):
    assert name not in sys.modules, name

assert registration.ImageRegistration.__name__ == "ImageRegistration"
assert "cyexm.registration.whole_image" in sys.modules
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
