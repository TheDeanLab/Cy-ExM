from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import compare_legacy, legacy_baseline

pytestmark = pytest.mark.integration


def _valid_array_record(*, passed: bool, exact: bool = False) -> dict[str, object]:
    return {
        "schema_version": 1,
        "passed": passed,
        "comparison_valid": True,
        "shape_matches": True,
        "actual_shape": [2, 2],
        "expected_shape": [2, 2],
        "actual_finite": True,
        "expected_finite": True,
        "max_abs_difference": 0.0 if passed else 0.25,
        "rtol": 0.0 if exact else 1e-4,
        "atol": 0.0 if exact else 1e-5,
        "exact": exact,
    }


def _valid_literal_comparisons() -> dict[str, dict[str, object]]:
    return {
        "affine_result": _valid_array_record(passed=True),
        "affine_mask": _valid_array_record(passed=True, exact=True),
        "nonlinear_result": _valid_array_record(passed=False),
    }


def _expected_literal_shapes() -> dict[str, list[int]]:
    return {
        "affine_result": [2, 2],
        "affine_mask": [2, 2],
        "nonlinear_result": [2, 2],
    }


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


def test_capture_audit_records_matching_clean_pin_before_and_after(
    git_checkout: tuple[Path, str],
    tmp_path: Path,
) -> None:
    checkout, commit = git_checkout
    previous = sys.dont_write_bytecode
    try:
        payload, audit = legacy_baseline._run_capture_with_source_audit(
            checkout,
            tmp_path / "baseline",
            lambda before: {"captured_commit": before["source_commit"]},
            expected_commit=commit,
        )
    finally:
        sys.dont_write_bytecode = previous

    assert payload == {"captured_commit": commit}
    assert audit["before"] == audit["after"]
    assert audit["unchanged"] is True
    assert audit["bytecode_writes_disabled"] is True


def test_capture_audit_rejects_a_checkout_dirtied_during_capture(
    git_checkout: tuple[Path, str],
    tmp_path: Path,
) -> None:
    checkout, commit = git_checkout
    previous = sys.dont_write_bytecode

    def dirty_checkout(_before: dict[str, object]) -> dict[str, object]:
        (checkout / "tracked.txt").write_text("changed\n", encoding="utf-8")
        return {}

    try:
        with pytest.raises(RuntimeError, match="must be clean"):
            legacy_baseline._run_capture_with_source_audit(
                checkout,
                tmp_path / "baseline",
                dirty_checkout,
                expected_commit=commit,
            )
    finally:
        sys.dont_write_bytecode = previous


def test_post_capture_audit_error_does_not_mask_primary_capture_error(
    git_checkout: tuple[Path, str],
    tmp_path: Path,
) -> None:
    checkout, commit = git_checkout
    previous = sys.dont_write_bytecode

    def fail_and_dirty(_before: dict[str, object]) -> dict[str, object]:
        (checkout / "tracked.txt").write_text("changed\n", encoding="utf-8")
        raise ValueError("injected primary capture failure")

    try:
        with pytest.raises(
            ValueError,
            match="injected primary capture failure",
        ) as error:
            legacy_baseline._run_capture_with_source_audit(
                checkout,
                tmp_path / "baseline",
                fail_and_dirty,
                expected_commit=commit,
            )
    finally:
        sys.dont_write_bytecode = previous

    assert any(
        "post-capture source audit also failed" in note
        for note in error.value.__notes__
    )


def test_comparator_requires_complete_matching_post_capture_audit(
    git_checkout: tuple[Path, str],
    tmp_path: Path,
) -> None:
    checkout, commit = git_checkout
    validation = legacy_baseline._validate_source_checkout(
        checkout,
        tmp_path / "baseline",
        expected_commit=commit,
    )
    audit = {
        "before": validation,
        "after": dict(validation),
        "unchanged": True,
        "bytecode_writes_disabled": True,
    }

    compare_legacy._require_source_audit(audit, expected_commit=commit)

    audit["after"] = {**validation, "git_status": "dirty"}
    with pytest.raises(RuntimeError, match="post-capture source audit"):
        compare_legacy._require_source_audit(audit, expected_commit=commit)


def test_literal_dry_diagnostic_requires_a_present_result_to_diverge() -> None:
    comparisons = _valid_literal_comparisons()

    expected_shapes = _expected_literal_shapes()
    diagnostic = compare_legacy._literal_dry_expected_divergence(
        comparisons,
        expected_shapes,
    )

    assert diagnostic["classification"] == "expected_divergence"
    assert diagnostic["passed"] is True
    assert diagnostic["all_required_records_valid"] is True
    for comparison in comparisons.values():
        comparison["passed"] = True
    assert (
        compare_legacy._literal_dry_expected_divergence(
            comparisons,
            expected_shapes,
        )["passed"]
        is False
    )


@pytest.mark.parametrize("invalid_passed", [None, "false", 0])
def test_literal_dry_diagnostic_rejects_non_boolean_result_evidence(
    invalid_passed: object,
) -> None:
    comparisons = _valid_literal_comparisons()
    comparisons["affine_mask"]["passed"] = invalid_passed

    diagnostic = compare_legacy._literal_dry_expected_divergence(
        comparisons,
        _expected_literal_shapes(),
    )

    assert diagnostic["passed"] is False
    assert diagnostic["all_required_results_boolean"] is False


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        pytest.param("schema_version", 99, id="schema-version"),
        pytest.param("comparison_valid", False, id="comparison-invalid"),
        pytest.param("shape_matches", False, id="shape-mismatch"),
        pytest.param("actual_finite", False, id="current-nonfinite"),
        pytest.param("expected_finite", False, id="baseline-nonfinite"),
        pytest.param("max_abs_difference", None, id="missing-maximum"),
        pytest.param("max_abs_difference", float("inf"), id="nonfinite-maximum"),
        pytest.param("rtol", "1e-4", id="malformed-tolerance"),
    ],
)
def test_literal_dry_diagnostic_rejects_malformed_comparison_records(
    field: str,
    invalid_value: object,
) -> None:
    comparisons = _valid_literal_comparisons()
    comparisons["nonlinear_result"][field] = invalid_value

    diagnostic = compare_legacy._literal_dry_expected_divergence(
        comparisons,
        _expected_literal_shapes(),
    )

    assert diagnostic["passed"] is False
    assert diagnostic["all_required_records_valid"] is False
    assert diagnostic["invalid_results"] == ["nonlinear_result"]


def test_literal_dry_diagnostic_anchors_shapes_to_baseline_summary() -> None:
    comparisons = _valid_literal_comparisons()
    comparisons["nonlinear_result"]["actual_shape"] = [3, 3]
    comparisons["nonlinear_result"]["expected_shape"] = [3, 3]

    diagnostic = compare_legacy._literal_dry_expected_divergence(
        comparisons,
        _expected_literal_shapes(),
    )

    assert diagnostic["passed"] is False
    assert diagnostic["all_shapes_match_baseline_summary"] is False
    assert diagnostic["invalid_results"] == ["nonlinear_result"]


def test_array_result_records_complete_finite_comparison_schema(
    native_threads: object,
) -> None:
    del native_threads
    import numpy as np

    result = compare_legacy._array_result(
        np,
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 2.0], dtype=np.float32),
        exact=False,
        rtol=1e-4,
        atol=1e-5,
    )

    assert result == {
        "schema_version": 1,
        "passed": False,
        "comparison_valid": True,
        "shape_matches": True,
        "actual_shape": [2],
        "expected_shape": [2],
        "actual_finite": True,
        "expected_finite": True,
        "max_abs_difference": 1.0,
        "rtol": 1e-4,
        "atol": 1e-5,
        "exact": False,
    }


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param([0.0, float("nan")], [0.0, 1.0], id="current-nonfinite"),
        pytest.param([0.0, 1.0], [0.0, float("inf")], id="baseline-nonfinite"),
        pytest.param([0.0], [0.0, 1.0], id="wrong-shape"),
    ],
)
def test_array_result_marks_nonfinite_or_wrong_shape_evidence_invalid(
    native_threads: object,
    actual: list[float],
    expected: list[float],
) -> None:
    del native_threads
    import numpy as np

    result = compare_legacy._array_result(
        np,
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        exact=False,
        rtol=1e-4,
        atol=1e-5,
    )

    assert result["passed"] is False
    assert result["comparison_valid"] is False
    assert result["max_abs_difference"] is None


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
