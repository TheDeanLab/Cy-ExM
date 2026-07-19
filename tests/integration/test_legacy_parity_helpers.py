from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import compare_legacy, legacy_baseline

pytestmark = pytest.mark.integration


@pytest.fixture
def git_checkout(tmp_path: Path) -> tuple[Path, str]:
    checkout = tmp_path / "source"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    tracked = checkout / "tracked.txt"
    tracked.write_text("pinned\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=CyExM Test",
            "-c",
            "user.email=cyexm@example.invalid",
            "commit",
            "-q",
            "-m",
            "pin",
        ],
        cwd=checkout,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return checkout, commit


def test_parity_masks_are_cross_script_nontrivial_asymmetric_and_image_independent(
    native_threads: object,
) -> None:
    del native_threads
    import numpy as np

    baseline_masks = legacy_baseline._deterministic_masks(np)
    comparison_masks = compare_legacy._deterministic_masks(np)

    for baseline, comparison in zip(baseline_masks, comparison_masks, strict=True):
        np.testing.assert_array_equal(baseline, comparison)
        np.testing.assert_array_equal(np.unique(baseline), (0.0, 1.0))
        assert not np.array_equal(baseline, np.flip(baseline))
        assert not np.array_equal(
            baseline,
            (legacy_baseline._synthetic_volume(np) > 0).astype(np.float32),
        )


def test_source_checkout_requires_the_expected_pin(
    git_checkout: tuple[Path, str],
    tmp_path: Path,
) -> None:
    checkout, commit = git_checkout

    provenance = legacy_baseline._validate_source_checkout(
        checkout,
        tmp_path / "baseline",
        expected_commit=commit,
    )

    assert provenance["checkout_root"] == str(checkout.resolve())
    assert provenance["source_commit"] == commit
    with pytest.raises(RuntimeError, match="expected pin"):
        legacy_baseline._validate_source_checkout(
            checkout,
            tmp_path / "other-baseline",
            expected_commit="0" * 40,
        )


def test_source_checkout_must_be_clean(
    git_checkout: tuple[Path, str],
    tmp_path: Path,
) -> None:
    checkout, commit = git_checkout
    (checkout / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be clean"):
        legacy_baseline._validate_source_checkout(
            checkout,
            tmp_path / "baseline",
            expected_commit=commit,
        )


def test_source_checkout_rejects_an_output_directory_inside_it(
    git_checkout: tuple[Path, str],
) -> None:
    checkout, commit = git_checkout

    with pytest.raises(ValueError, match="outside the ClearEx checkout"):
        legacy_baseline._validate_source_checkout(
            checkout,
            checkout / "baseline",
            expected_commit=commit,
        )


def test_import_provenance_must_resolve_under_the_supplied_checkout(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "source"
    package_file = checkout / "src" / "clearex" / "__init__.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")
    imported = {"clearex": SimpleNamespace(__file__=str(package_file))}

    provenance = legacy_baseline._validate_import_provenance(imported, checkout)

    assert provenance == {"clearex": str(package_file.resolve())}
    outside = tmp_path / "elsewhere" / "clearex.py"
    outside.parent.mkdir()
    outside.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="outside the supplied source checkout"):
        legacy_baseline._validate_import_provenance(
            {"clearex": SimpleNamespace(__file__=str(outside))},
            checkout,
        )


def test_literal_dry_diagnostic_requires_a_present_result_to_diverge() -> None:
    comparisons = {
        "affine_result": {"passed": False},
        "affine_mask": {"passed": False},
        "nonlinear_result": {"passed": False},
    }

    diagnostic = compare_legacy._literal_dry_expected_divergence(comparisons)

    assert diagnostic["classification"] == "expected_divergence"
    assert diagnostic["passed"] is True
    for comparison in comparisons.values():
        comparison["passed"] = True
    assert (
        compare_legacy._literal_dry_expected_divergence(comparisons)["passed"] is False
    )


@pytest.mark.parametrize("invalid_passed", [None, "false", 0])
def test_literal_dry_diagnostic_rejects_non_boolean_result_evidence(
    invalid_passed: object,
) -> None:
    comparisons = {
        "affine_result": {"passed": False},
        "affine_mask": {"passed": invalid_passed},
        "nonlinear_result": {"passed": True},
    }

    diagnostic = compare_legacy._literal_dry_expected_divergence(comparisons)

    assert diagnostic["passed"] is False
    assert diagnostic["all_required_results_boolean"] is False


def test_fixed_mask_diagnostic_requires_captured_legacy_keyword_evidence() -> None:
    characterization = {
        "classification": "corrected_legacy_defect",
        "legacy_wrapper_keyword": "fixed_mask",
        "antspyx_consumed_keyword": "mask",
        "legacy_keyword_in_antspyx_signature": False,
        "corrected_keyword_in_antspyx_signature": True,
        "literal_fixed_mask_provided": True,
        "matched_fixed_mask": None,
        "registration_calls": [
            {
                "label": label,
                "received_fixed_mask": label.startswith("literal_"),
                "received_mask_keyword": False,
            }
            for label in (
                "literal_linear",
                "literal_nonlinear",
                "matched_linear",
                "matched_nonlinear",
            )
        ],
    }

    diagnostic = compare_legacy._fixed_mask_corrected_defect(characterization)

    assert diagnostic["passed"] is True
    characterization["registration_calls"][0]["received_mask_keyword"] = True
    assert (
        compare_legacy._fixed_mask_corrected_defect(characterization)["passed"] is False
    )
