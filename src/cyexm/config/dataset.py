import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from cyexm.config._validation import (
    built_in_int,
    finite_positive_float,
    nonempty_string,
    relative_posix_path,
    require_exact_keys,
)
from cyexm.config._yaml import load_yaml_mapping
from cyexm.config.models import (
    ChannelInterpolation,
    ChannelSpec,
    ChunkSizeZYX,
    DatasetManifest,
    PixelSpacingXYZ,
    ReferenceSpec,
    RegistrationSpec,
    RegistrationStageSpec,
    RegistrationStrategy,
    RoundSpec,
)
from cyexm.errors import ConfigurationValidationError

_LINEAR_TYPES = frozenset({"Translation", "Rigid", "Similarity", "Affine", "TRSAA"})
_NONLINEAR_TYPES = frozenset({"Elastic", "SyNOnly", "SyN"})
_ACCURACIES = frozenset({"high", "medium", "low", "dry run"})
_STRATEGIES = frozenset({"whole_image", "serial_chunked"})
_INTERPOLATIONS = frozenset({"linear", "nearestNeighbor"})

_DATASET_KEYS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "voxel_spacing_um",
        "reference",
        "registration",
        "rounds",
    }
)
_VOXEL_SPACING_KEYS = frozenset({"x", "y", "z"})
_REFERENCE_KEYS = frozenset({"round", "channel"})
_REGISTRATION_KEYS = frozenset({"crop", "linear", "nonlinear"})
_STAGE_KEYS = frozenset({"type", "accuracy"})
_WHOLE_IMAGE_ROUND_KEYS = frozenset(
    {"round", "strategy", "registration_channel", "channels"}
)
_SERIAL_CHUNKED_ROUND_KEYS = _WHOLE_IMAGE_ROUND_KEYS | frozenset(
    {"chunk_size_zyx", "overlap_fraction"}
)
_CHANNEL_KEYS = frozenset({"file", "interpolation"})
_DATASET_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _list(value: object, yaml_path: str) -> list[object]:
    if not isinstance(value, list):
        raise ConfigurationValidationError(f"{yaml_path} must be a list; got {value!r}")
    return value


def _choice(value: object, accepted: frozenset[str], yaml_path: str) -> str:
    if not isinstance(value, str) or value not in accepted:
        raise ConfigurationValidationError(
            f"{yaml_path} must be one of {sorted(accepted)!r}; got {value!r}"
        )
    return value


def _dataset_id(value: object) -> str:
    dataset_id = nonempty_string(value, "dataset_id")
    if _DATASET_ID_PATTERN.fullmatch(dataset_id) is None:
        raise ConfigurationValidationError(
            f"dataset_id must match [A-Za-z0-9][A-Za-z0-9._-]*; got {value!r}"
        )
    return dataset_id


def _voxel_spacing(value: object) -> PixelSpacingXYZ:
    mapping = require_exact_keys(value, _VOXEL_SPACING_KEYS, "voxel_spacing_um")
    return (
        finite_positive_float(mapping["x"], "voxel_spacing_um.x"),
        finite_positive_float(mapping["y"], "voxel_spacing_um.y"),
        finite_positive_float(mapping["z"], "voxel_spacing_um.z"),
    )


def _reference(value: object) -> ReferenceSpec:
    mapping = require_exact_keys(value, _REFERENCE_KEYS, "reference")
    return ReferenceSpec(
        round=built_in_int(mapping["round"], "reference.round"),
        channel=relative_posix_path(mapping["channel"], "reference.channel"),
    )


def _stage(
    value: object, yaml_path: str, accepted_types: frozenset[str]
) -> RegistrationStageSpec:
    mapping = require_exact_keys(value, _STAGE_KEYS, yaml_path)
    return RegistrationStageSpec(
        type=_choice(mapping["type"], accepted_types, f"{yaml_path}.type"),
        accuracy=_choice(mapping["accuracy"], _ACCURACIES, f"{yaml_path}.accuracy"),
    )


def _registration(value: object) -> RegistrationSpec:
    mapping = require_exact_keys(value, _REGISTRATION_KEYS, "registration")
    crop = mapping["crop"]
    if type(crop) is not bool:
        raise ConfigurationValidationError(
            f"registration.crop must be a built-in boolean; got {crop!r}"
        )
    return RegistrationSpec(
        crop=crop,
        linear=_stage(mapping["linear"], "registration.linear", _LINEAR_TYPES),
        nonlinear=_stage(
            mapping["nonlinear"], "registration.nonlinear", _NONLINEAR_TYPES
        ),
    )


def _channel(value: object, yaml_path: str) -> ChannelSpec:
    mapping = require_exact_keys(value, _CHANNEL_KEYS, yaml_path)
    interpolation = _choice(
        mapping["interpolation"], _INTERPOLATIONS, f"{yaml_path}.interpolation"
    )
    return ChannelSpec(
        file=relative_posix_path(mapping["file"], f"{yaml_path}.file"),
        interpolation=cast(ChannelInterpolation, interpolation),
    )


def _chunk_size(value: object, yaml_path: str) -> ChunkSizeZYX:
    items = _list(value, yaml_path)
    if len(items) != 3:
        raise ConfigurationValidationError(
            f"{yaml_path} must contain exactly three values; got {value!r}"
        )
    return (
        built_in_int(items[0], f"{yaml_path}[0]", minimum=1),
        built_in_int(items[1], f"{yaml_path}[1]", minimum=1),
        built_in_int(items[2], f"{yaml_path}[2]", minimum=1),
    )


def _overlap_fraction(value: object, yaml_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationValidationError(
            f"{yaml_path} must be a finite number from zero up to one; got {value!r}"
        )
    try:
        overlap = float(value)
    except OverflowError as error:
        raise ConfigurationValidationError(
            f"{yaml_path} must be a finite number from zero up to one; got {value!r}"
        ) from error
    if not math.isfinite(overlap) or not 0.0 <= overlap < 1.0:
        raise ConfigurationValidationError(
            f"{yaml_path} must be a finite number from zero up to one; got {value!r}"
        )
    return overlap


def _round(value: object, index: int) -> RoundSpec:
    yaml_path = f"rounds[{index}]"
    raw_strategy = value.get("strategy") if isinstance(value, Mapping) else None
    expected_keys = (
        _SERIAL_CHUNKED_ROUND_KEYS
        if raw_strategy == "serial_chunked"
        else _WHOLE_IMAGE_ROUND_KEYS
    )
    mapping = require_exact_keys(value, expected_keys, yaml_path)
    strategy = cast(
        RegistrationStrategy,
        _choice(mapping["strategy"], _STRATEGIES, f"{yaml_path}.strategy"),
    )
    channels = tuple(
        _channel(item, f"{yaml_path}.channels[{channel_index}]")
        for channel_index, item in enumerate(
            _list(mapping["channels"], f"{yaml_path}.channels")
        )
    )
    channel_files = tuple(channel.file for channel in channels)
    if len(set(channel_files)) != len(channel_files):
        raise ConfigurationValidationError(
            f"{yaml_path}.channels file values must be unique; got {channel_files!r}"
        )
    registration_channel = relative_posix_path(
        mapping["registration_channel"], f"{yaml_path}.registration_channel"
    )
    if registration_channel not in channel_files:
        raise ConfigurationValidationError(
            f"{yaml_path}.registration_channel must match a channel file; "
            f"got {registration_channel!s}"
        )

    chunk_size_zyx: ChunkSizeZYX | None = None
    overlap_fraction: float | None = None
    if strategy == "serial_chunked":
        chunk_size_zyx = _chunk_size(
            mapping["chunk_size_zyx"], f"{yaml_path}.chunk_size_zyx"
        )
        overlap_fraction = _overlap_fraction(
            mapping["overlap_fraction"], f"{yaml_path}.overlap_fraction"
        )

    return RoundSpec(
        round=built_in_int(mapping["round"], f"{yaml_path}.round"),
        strategy=strategy,
        registration_channel=registration_channel,
        channels=channels,
        chunk_size_zyx=chunk_size_zyx,
        overlap_fraction=overlap_fraction,
    )


def _rounds(value: object, reference_round: int) -> tuple[RoundSpec, ...]:
    rounds = tuple(
        _round(item, index) for index, item in enumerate(_list(value, "rounds"))
    )
    round_numbers = tuple(item.round for item in rounds)
    if len(set(round_numbers)) != len(round_numbers):
        raise ConfigurationValidationError(
            f"rounds round values must be unique; got {round_numbers!r}"
        )
    if reference_round in round_numbers:
        raise ConfigurationValidationError(
            f"rounds must not reuse reference round {reference_round}; "
            f"got {round_numbers!r}"
        )
    return rounds


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    mapping = require_exact_keys(load_yaml_mapping(path), _DATASET_KEYS, "dataset")
    schema_version = built_in_int(mapping["schema_version"], "schema_version")
    if schema_version != 1:
        raise ConfigurationValidationError(
            f"schema_version must be the built-in integer 1; got {schema_version!r}"
        )
    reference = _reference(mapping["reference"])
    return DatasetManifest(
        schema_version=schema_version,
        dataset_id=_dataset_id(mapping["dataset_id"]),
        voxel_spacing_um=_voxel_spacing(mapping["voxel_spacing_um"]),
        reference=reference,
        registration=_registration(mapping["registration"]),
        rounds=_rounds(mapping["rounds"], reference.round),
    )
