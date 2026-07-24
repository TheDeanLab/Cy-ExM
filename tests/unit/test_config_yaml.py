from pathlib import Path, PurePosixPath
from re import escape

import pytest

from cyexm.config._validation import (
    built_in_int,
    finite_positive_float,
    nonempty_string,
    relative_posix_path,
    require_exact_keys,
)
from cyexm.config._yaml import load_yaml_mapping
from cyexm.errors import (
    ConfigurationValidationError,
    CyExMError,
    ExecutorStartupError,
    ManifestDecodeError,
    RoundExecutionError,
    WorkerPreflightError,
    WorkflowExecutionError,
    WorkflowPlanError,
)

pytestmark = pytest.mark.unit


def test_parallelism_configuration_is_not_exported_from_package_root() -> None:
    import cyexm

    assert "ParallelismMode" not in cyexm.__all__
    assert "ParallelismSpec" not in cyexm.__all__
    assert not hasattr(cyexm, "ParallelismMode")
    assert not hasattr(cyexm, "ParallelismSpec")


def test_configuration_errors_share_the_package_error_boundary() -> None:
    assert issubclass(ManifestDecodeError, CyExMError)
    assert issubclass(ConfigurationValidationError, CyExMError)
    assert issubclass(WorkflowPlanError, CyExMError)
    assert issubclass(ExecutorStartupError, CyExMError)
    assert issubclass(WorkerPreflightError, ExecutorStartupError)
    assert issubclass(RoundExecutionError, CyExMError)
    assert issubclass(WorkflowExecutionError, CyExMError)


def test_workflow_execution_error_records_failed_rounds_and_run_record() -> None:
    error = WorkflowExecutionError((2, 5), "runs/latest.json")

    assert error.failed_rounds == (2, 5)
    assert error.run_record == "runs/latest.json"
    assert str(error) == ("Workflow rounds failed: 2, 5; run record: runs/latest.json")


@pytest.mark.parametrize(
    "content, message",
    [
        ("schema_version: 1\nschema_version: 1\n", "duplicate key"),
        ("value: &shared 1\ncopy: *shared\n", "aliases are not allowed"),
        ("value: !python/object:thing {}\n", "custom YAML tags"),
        ("? [one, two]\n: value\n", "mapping keys must be scalar strings"),
        ("1: value\n", "mapping keys must be scalar strings"),
        ("schema_version: 1\n---\nschema_version: 1\n", "one YAML document"),
    ],
)
def test_strict_yaml_rejects_ambiguous_documents(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ManifestDecodeError, match=message):
        load_yaml_mapping(path)


@pytest.mark.parametrize(
    "content",
    [
        "value: !!pairs\n  - ? [one, two]\n    : item\n",
        "value: !!pairs\n  - one: first\n  - one: second\n",
        "value: !!omap\n  - ? [one, two]\n    : item\n",
        "value: !!omap\n  - one: first\n  - one: second\n",
    ],
)
def test_strict_yaml_rejects_specialized_collection_key_bypasses(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(
        ManifestDecodeError, match="specialized YAML collection tags"
    ) as error:
        load_yaml_mapping(path)

    assert str(path) in str(error.value)


def test_strict_yaml_wraps_missing_file_error_with_source_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"

    with pytest.raises(ManifestDecodeError, match=escape(str(path))):
        load_yaml_mapping(path)


def test_strict_yaml_wraps_invalid_utf8_with_source_path(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_bytes(b"schema_version: \xff\n")

    with pytest.raises(ManifestDecodeError, match=escape(str(path))):
        load_yaml_mapping(path)


@pytest.mark.parametrize("content", ["", "- one\n- two\n"])
def test_strict_yaml_requires_a_top_level_mapping(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ManifestDecodeError, match="top-level mapping"):
        load_yaml_mapping(path)


def test_strict_yaml_loads_nested_scalar_and_list_values(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "schema_version: 1\nround:\n  name: round-1\n  channels: [DAPI, GFP]\n",
        encoding="utf-8",
    )

    result = load_yaml_mapping(path)

    assert result == {
        "schema_version": 1,
        "round": {"name": "round-1", "channels": ["DAPI", "GFP"]},
    }


def test_require_exact_keys_returns_a_matching_string_key_mapping() -> None:
    value = {"name": "round-1", "index": 1}

    result = require_exact_keys(value, frozenset({"name", "index"}), "round")

    assert result is value


def test_require_exact_keys_reports_a_nonmapping_value() -> None:
    value = [("name", "round-1")]

    with pytest.raises(ConfigurationValidationError) as error:
        require_exact_keys(value, frozenset({"name"}), "round")

    assert str(error.value) == f"round must be a mapping; got {value!r}"


def test_require_exact_keys_reports_non_string_mapping_keys() -> None:
    value = {1: "round-1"}

    with pytest.raises(ConfigurationValidationError) as error:
        require_exact_keys(value, frozenset({"name"}), "round")

    assert str(error.value) == f"round mapping keys must be strings; got {value!r}"


def test_require_exact_keys_reports_expected_and_actual_keys() -> None:
    with pytest.raises(
        ConfigurationValidationError,
        match=(
            r"round keys must be exactly \['index', 'name'\]; "
            r"got \['name', 'unexpected'\]"
        ),
    ):
        require_exact_keys(
            {"name": "round-1", "unexpected": True},
            frozenset({"name", "index"}),
            "round",
        )


def test_built_in_int_accepts_the_configured_minimum() -> None:
    assert built_in_int(1, "round.index", minimum=1) == 1


@pytest.mark.parametrize("value", [True, 1.0, 0])
def test_built_in_int_rejects_non_builtin_or_below_minimum_values(
    value: object,
) -> None:
    with pytest.raises(
        ConfigurationValidationError,
        match=r"round.index must be a built-in integer >= 1; got",
    ):
        built_in_int(value, "round.index", minimum=1)


@pytest.mark.parametrize("value, expected", [(1, 1.0), (1.5, 1.5)])
def test_finite_positive_float_accepts_positive_numbers(
    value: object, expected: float
) -> None:
    assert finite_positive_float(value, "voxel_size.x") == expected


@pytest.mark.parametrize("value", [True, "1", 0, -1, float("inf"), float("nan")])
def test_finite_positive_float_rejects_invalid_numbers(value: object) -> None:
    with pytest.raises(
        ConfigurationValidationError,
        match=escape("voxel_size.x must be a positive finite number; got"),
    ):
        finite_positive_float(value, "voxel_size.x")


def test_finite_positive_float_wraps_integer_conversion_overflow() -> None:
    value = 10**400

    with pytest.raises(ConfigurationValidationError) as error:
        finite_positive_float(value, "voxel_size.x")

    assert str(error.value) == (
        f"voxel_size.x must be a positive finite number; got {value!r}"
    )


def test_nonempty_string_accepts_a_single_line_value() -> None:
    assert nonempty_string("round-1", "round.name") == "round-1"


@pytest.mark.parametrize(
    "value", [1, "", "round\x00name", "round\nname", "round\rname"]
)
def test_nonempty_string_rejects_invalid_values(value: object) -> None:
    with pytest.raises(
        ConfigurationValidationError,
        match=escape("round.name must be a nonempty single-line string; got"),
    ):
        nonempty_string(value, "round.name")


def test_relative_posix_path_returns_a_safe_relative_path() -> None:
    assert relative_posix_path("rounds/r1.tif", "round.path") == PurePosixPath(
        "rounds/r1.tif"
    )


@pytest.mark.parametrize(
    "value",
    [
        "/rounds/r1.tif",
        ".",
        "../r1.tif",
        "rounds/../r1.tif",
        "foo/./bar",
        "foo//bar",
    ],
)
def test_relative_posix_path_rejects_unsafe_paths(value: object) -> None:
    with pytest.raises(
        ConfigurationValidationError,
        match=escape("round.path must be a safe relative POSIX path; got"),
    ):
        relative_posix_path(value, "round.path")
