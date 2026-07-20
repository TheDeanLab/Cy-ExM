from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path, PurePosixPath

import pytest
import yaml

from cyexm.config import (
    ChannelSpec,
    DatasetManifest,
    ReferenceSpec,
    RegistrationSpec,
    RegistrationStageSpec,
    RoundSpec,
    load_dataset_manifest,
)
from cyexm.errors import ConfigurationValidationError

pytestmark = pytest.mark.unit


VALID_DATASET_YAML = """\
schema_version: 1
dataset_id: dataset-1
voxel_spacing_um:
  x: 0.324
  y: 0.324
  z: 0.9
reference:
  round: 1
  channel: round_1_488.tif
registration:
  crop: true
  linear:
    type: TRSAA
    accuracy: high
  nonlinear:
    type: SyNOnly
    accuracy: high
rounds:
  - round: 2
    strategy: whole_image
    registration_channel: round_2_488.tif
    channels:
      - file: round_2_488.tif
        interpolation: linear
      - file: round_2_mask.tif
        interpolation: nearestNeighbor
  - round: 3
    strategy: serial_chunked
    registration_channel: round_3_488.tif
    chunk_size_zyx: [8, 16, 16]
    overlap_fraction: 0.25
    channels:
      - file: round_3_488.tif
        interpolation: linear
"""

type _DataPath = tuple[str | int, ...]


def _valid_data() -> dict[str, object]:
    value = yaml.safe_load(VALID_DATASET_YAML)
    assert isinstance(value, dict)
    return value


def _mapping_at(data: dict[str, object], path: _DataPath) -> dict[str, object]:
    current: object = data
    for part in path:
        assert isinstance(current, dict | list)
        current = current[part]  # type: ignore[index]
    assert isinstance(current, dict)
    return current


def _set_at(data: dict[str, object], path: _DataPath, value: object) -> None:
    parent = _mapping_at(data, path[:-1]) if len(path) == 1 else None
    if parent is not None:
        parent[path[-1]] = value  # type: ignore[index]
        return

    current: object = data
    for part in path[:-1]:
        assert isinstance(current, dict | list)
        current = current[part]  # type: ignore[index]
    assert isinstance(current, dict | list)
    current[path[-1]] = value  # type: ignore[index]


def _write_manifest(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _load_data(tmp_path: Path, data: dict[str, object]) -> DatasetManifest:
    return load_dataset_manifest(_write_manifest(tmp_path, data))


def test_config_namespace_exports_only_the_scientific_manifest_api() -> None:
    import cyexm.config as config

    assert config.__all__ == (
        "ChannelSpec",
        "DatasetManifest",
        "ReferenceSpec",
        "RegistrationSpec",
        "RegistrationStageSpec",
        "RoundSpec",
        "load_dataset_manifest",
    )


def test_load_dataset_manifest_normalizes_xyz_and_zyx_fields(tmp_path: Path) -> None:
    path = tmp_path / "dataset.yaml"
    path.write_text(VALID_DATASET_YAML, encoding="utf-8")

    manifest = load_dataset_manifest(path)

    assert manifest.schema_version == 1
    assert manifest.dataset_id == "dataset-1"
    assert manifest.voxel_spacing_um == (0.324, 0.324, 0.9)
    assert manifest.reference == ReferenceSpec(
        round=1, channel=PurePosixPath("round_1_488.tif")
    )
    assert manifest.registration == RegistrationSpec(
        crop=True,
        linear=RegistrationStageSpec(type="TRSAA", accuracy="high"),
        nonlinear=RegistrationStageSpec(type="SyNOnly", accuracy="high"),
    )
    assert manifest.rounds[0] == RoundSpec(
        round=2,
        strategy="whole_image",
        registration_channel=PurePosixPath("round_2_488.tif"),
        channels=(
            ChannelSpec(PurePosixPath("round_2_488.tif"), "linear"),
            ChannelSpec(PurePosixPath("round_2_mask.tif"), "nearestNeighbor"),
        ),
    )
    assert tuple(item.round for item in manifest.rounds) == (2, 3)
    assert manifest.rounds[1].chunk_size_zyx == (8, 16, 16)
    with pytest.raises(FrozenInstanceError):
        manifest.dataset_id = "changed"  # type: ignore[misc]


def test_load_dataset_manifest_preserves_round_and_channel_order(
    tmp_path: Path,
) -> None:
    data = _valid_data()
    rounds = data["rounds"]
    assert isinstance(rounds, list)
    rounds.reverse()
    first_channels = rounds[1]["channels"]
    assert isinstance(first_channels, list)
    first_channels.reverse()

    manifest = _load_data(tmp_path, data)

    assert tuple(item.round for item in manifest.rounds) == (3, 2)
    assert tuple(channel.file for channel in manifest.rounds[1].channels) == (
        PurePosixPath("round_2_mask.tif"),
        PurePosixPath("round_2_488.tif"),
    )


@pytest.mark.parametrize(
    ("mapping_path", "missing_key"),
    [
        ((), "dataset_id"),
        (("voxel_spacing_um",), "x"),
        (("reference",), "channel"),
        (("registration",), "crop"),
        (("registration", "linear"), "accuracy"),
        (("registration", "nonlinear"), "type"),
        (("rounds", 0), "registration_channel"),
        (("rounds", 1), "chunk_size_zyx"),
        (("rounds", 0, "channels", 0), "interpolation"),
    ],
)
def test_load_dataset_manifest_rejects_missing_keys_at_every_mapping_layer(
    tmp_path: Path, mapping_path: _DataPath, missing_key: str
) -> None:
    data = _valid_data()
    del _mapping_at(data, mapping_path)[missing_key]

    with pytest.raises(ConfigurationValidationError, match="keys must be exactly"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize(
    "mapping_path",
    [
        (),
        ("voxel_spacing_um",),
        ("reference",),
        ("registration",),
        ("registration", "linear"),
        ("registration", "nonlinear"),
        ("rounds", 0),
        ("rounds", 1),
        ("rounds", 0, "channels", 0),
    ],
)
def test_load_dataset_manifest_rejects_unknown_keys_at_every_mapping_layer(
    tmp_path: Path, mapping_path: _DataPath
) -> None:
    data = _valid_data()
    _mapping_at(data, mapping_path)["unexpected"] = "value"

    with pytest.raises(ConfigurationValidationError, match="keys must be exactly"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize("value", [True, 1.0, 2, "1"])
def test_load_dataset_manifest_requires_schema_version_one_as_a_builtin_integer(
    tmp_path: Path, value: object
) -> None:
    data = _valid_data()
    data["schema_version"] = value

    with pytest.raises(ConfigurationValidationError, match="schema_version"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize(
    "value", ["", "-dataset", ".dataset", "dataset/name", "dataset name", 1]
)
def test_load_dataset_manifest_requires_a_safe_dataset_identifier(
    tmp_path: Path, value: object
) -> None:
    data = _valid_data()
    data["dataset_id"] = value

    with pytest.raises(ConfigurationValidationError, match="dataset_id"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize("value", [True, "0.324", 0, -0.1, float("inf"), float("nan")])
def test_load_dataset_manifest_requires_finite_positive_xyz_spacing(
    tmp_path: Path, value: object
) -> None:
    data = _valid_data()
    spacing = _mapping_at(data, ("voxel_spacing_um",))
    spacing["z"] = value

    with pytest.raises(ConfigurationValidationError, match=r"voxel_spacing_um\.z"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("reference", "round"), True),
        (("reference", "round"), 1.0),
        (("reference", "round"), -1),
        (("rounds", 0, "round"), True),
        (("rounds", 0, "round"), 2.0),
        (("rounds", 0, "round"), -1),
    ],
)
def test_load_dataset_manifest_requires_nonnegative_builtin_round_numbers(
    tmp_path: Path, field_path: _DataPath, value: object
) -> None:
    data = _valid_data()
    _set_at(data, field_path, value)

    with pytest.raises(ConfigurationValidationError, match="round"):
        _load_data(tmp_path, data)


def test_load_dataset_manifest_rejects_duplicate_moving_rounds(tmp_path: Path) -> None:
    data = _valid_data()
    _set_at(data, ("rounds", 1, "round"), 2)

    with pytest.raises(ConfigurationValidationError, match="unique"):
        _load_data(tmp_path, data)


def test_load_dataset_manifest_rejects_reference_round_reuse(tmp_path: Path) -> None:
    data = _valid_data()
    _set_at(data, ("rounds", 0, "round"), 1)

    with pytest.raises(ConfigurationValidationError, match="reference"):
        _load_data(tmp_path, data)


def test_load_dataset_manifest_requires_registration_channel_membership(
    tmp_path: Path,
) -> None:
    data = _valid_data()
    _set_at(data, ("rounds", 0, "registration_channel"), "missing.tif")

    with pytest.raises(ConfigurationValidationError, match="registration_channel"):
        _load_data(tmp_path, data)


def test_load_dataset_manifest_rejects_duplicate_channel_files(tmp_path: Path) -> None:
    data = _valid_data()
    channels = _mapping_at(data, ("rounds", 0))["channels"]
    assert isinstance(channels, list)
    channels.append(deepcopy(channels[0]))

    with pytest.raises(ConfigurationValidationError, match="unique"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize("value", ["parallel_chunked", "WholeImage", 1])
def test_load_dataset_manifest_rejects_unsupported_registration_strategy(
    tmp_path: Path, value: object
) -> None:
    data = _valid_data()
    _set_at(data, ("rounds", 0, "strategy"), value)

    with pytest.raises(ConfigurationValidationError, match="strategy"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("parallel_chunked", id="invalid-string"),
        pytest.param(1, id="invalid-type"),
    ],
)
def test_load_dataset_manifest_reports_invalid_serial_shaped_strategy_at_field(
    tmp_path: Path, value: object
) -> None:
    data = _valid_data()
    _set_at(data, ("rounds", 1, "strategy"), value)

    with pytest.raises(
        ConfigurationValidationError,
        match=r"rounds\[1\]\.strategy must be one of",
    ):
        _load_data(tmp_path, data)


@pytest.mark.parametrize("value", ["nearest", "Linear", 1])
def test_load_dataset_manifest_rejects_unsupported_channel_interpolation(
    tmp_path: Path, value: object
) -> None:
    data = _valid_data()
    _set_at(data, ("rounds", 0, "channels", 0, "interpolation"), value)

    with pytest.raises(ConfigurationValidationError, match="interpolation"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize("chunk_key", ["chunk_size_zyx", "overlap_fraction"])
def test_load_dataset_manifest_forbids_chunk_fields_for_whole_image(
    tmp_path: Path, chunk_key: str
) -> None:
    data = _valid_data()
    whole_image = _mapping_at(data, ("rounds", 0))
    whole_image[chunk_key] = [8, 16, 16] if chunk_key == "chunk_size_zyx" else 0.25

    with pytest.raises(ConfigurationValidationError, match="keys must be exactly"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize("chunk_key", ["chunk_size_zyx", "overlap_fraction"])
def test_load_dataset_manifest_requires_chunk_fields_for_serial_chunked(
    tmp_path: Path, chunk_key: str
) -> None:
    data = _valid_data()
    del _mapping_at(data, ("rounds", 1))[chunk_key]

    with pytest.raises(ConfigurationValidationError, match="keys must be exactly"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize(
    "value",
    [None, [8, 16], [8, 16, 16, 16], [0, 16, 16], [True, 16, 16], [8.0, 16, 16]],
)
def test_load_dataset_manifest_requires_three_positive_builtin_zyx_sizes(
    tmp_path: Path, value: object
) -> None:
    data = _valid_data()
    _set_at(data, ("rounds", 1, "chunk_size_zyx"), value)

    with pytest.raises(ConfigurationValidationError, match="chunk_size_zyx"):
        _load_data(tmp_path, data)


def test_load_dataset_manifest_accepts_zero_chunk_overlap(tmp_path: Path) -> None:
    data = _valid_data()
    _set_at(data, ("rounds", 1, "overlap_fraction"), 0)

    manifest = _load_data(tmp_path, data)

    assert manifest.rounds[1].overlap_fraction == 0.0


@pytest.mark.parametrize("value", [True, "0.25", -0.1, 1, float("inf"), float("nan")])
def test_load_dataset_manifest_requires_finite_overlap_below_one(
    tmp_path: Path, value: object
) -> None:
    data = _valid_data()
    _set_at(data, ("rounds", 1, "overlap_fraction"), value)

    with pytest.raises(ConfigurationValidationError, match="overlap_fraction"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_load_dataset_manifest_requires_builtin_boolean_crop(
    tmp_path: Path, value: object
) -> None:
    data = _valid_data()
    _set_at(data, ("registration", "crop"), value)

    with pytest.raises(ConfigurationValidationError, match="crop"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize(
    "linear_type", ["Translation", "Rigid", "Similarity", "Affine", "TRSAA"]
)
def test_load_dataset_manifest_accepts_exact_phase_one_linear_types(
    tmp_path: Path, linear_type: str
) -> None:
    data = _valid_data()
    _set_at(data, ("registration", "linear", "type"), linear_type)

    manifest = _load_data(tmp_path, data)

    assert manifest.registration.linear.type == linear_type


@pytest.mark.parametrize("nonlinear_type", ["Elastic", "SyNOnly", "SyN"])
def test_load_dataset_manifest_accepts_exact_phase_one_nonlinear_types(
    tmp_path: Path, nonlinear_type: str
) -> None:
    data = _valid_data()
    _set_at(data, ("registration", "nonlinear", "type"), nonlinear_type)

    manifest = _load_data(tmp_path, data)

    assert manifest.registration.nonlinear.type == nonlinear_type


@pytest.mark.parametrize("accuracy", ["high", "medium", "low", "dry run"])
@pytest.mark.parametrize("stage", ["linear", "nonlinear"])
def test_load_dataset_manifest_accepts_exact_phase_one_accuracies(
    tmp_path: Path, stage: str, accuracy: str
) -> None:
    data = _valid_data()
    _set_at(data, ("registration", stage, "accuracy"), accuracy)

    manifest = _load_data(tmp_path, data)

    parsed_stage = getattr(manifest.registration, stage)
    assert parsed_stage.accuracy == accuracy


@pytest.mark.parametrize(
    ("stage", "field", "value"),
    [
        ("linear", "type", "SyN"),
        ("linear", "type", "trsaa"),
        ("nonlinear", "type", "Affine"),
        ("nonlinear", "type", "synonly"),
        ("linear", "accuracy", "highest"),
        ("nonlinear", "accuracy", "Dry Run"),
        ("linear", "accuracy", 1),
    ],
)
def test_load_dataset_manifest_rejects_values_outside_phase_one_stage_choices(
    tmp_path: Path, stage: str, field: str, value: object
) -> None:
    data = _valid_data()
    _set_at(data, ("registration", stage, field), value)

    with pytest.raises(ConfigurationValidationError, match=field):
        _load_data(tmp_path, data)


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("reference", "channel"), "/absolute.tif"),
        (("rounds", 0, "registration_channel"), "../outside.tif"),
        (("rounds", 0, "channels", 0, "file"), "bad//path.tif"),
    ],
)
def test_load_dataset_manifest_requires_safe_relative_channel_paths(
    tmp_path: Path, field_path: _DataPath, value: object
) -> None:
    data = _valid_data()
    _set_at(data, field_path, value)

    with pytest.raises(ConfigurationValidationError, match="safe relative POSIX path"):
        _load_data(tmp_path, data)


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("voxel_spacing_um",), [0.324, 0.324, 0.9]),
        (("reference",), "round_1_488.tif"),
        (("registration",), []),
        (("rounds",), {}),
        (("rounds", 0, "channels"), {}),
        (("rounds", 0, "channels", 0), "round_2_488.tif"),
    ],
)
def test_load_dataset_manifest_rejects_wrong_collection_shapes(
    tmp_path: Path, field_path: _DataPath, value: object
) -> None:
    data = _valid_data()
    _set_at(data, field_path, value)

    with pytest.raises(ConfigurationValidationError):
        _load_data(tmp_path, data)
