import math
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import cast

from cyexm._nonlinear_metric import NonlinearMetricName, resolve_nonlinear_metric
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
    EstimatedXYRotationSpec,
    ExplicitXYRotationSpec,
    FeatureMaskSpec,
    InitialAlignmentSpec,
    NonlinearRegistrationStageSpec,
    PixelSpacingXYZ,
    ReferenceChannelSpec,
    ReferenceSpec,
    RegistrationPreprocessingSpec,
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
_REFERENCE_KEYS = frozenset({"round", "registration_channel", "channels"})
_REFERENCE_CHANNEL_KEYS = frozenset({"file"})
_REGISTRATION_KEYS = frozenset({"crop", "linear", "nonlinear"})
_REGISTRATION_PREPROCESSING_KEYS = frozenset({"filter_sigma_zyx", "feature_mask"})
_FEATURE_MASK_KEYS = frozenset({"min_component_volume_um3", "dilation_radius_um"})
_LINEAR_STAGE_KEYS = frozenset({"type", "accuracy"})
_NONLINEAR_STAGE_KEYS = frozenset({"type", "accuracy", "metric"})
_NONLINEAR_CC_STAGE_KEYS = _NONLINEAR_STAGE_KEYS | frozenset({"radius"})
_WHOLE_IMAGE_ROUND_KEYS = frozenset(
    {"round", "strategy", "registration_channel", "channels"}
)
_SERIAL_CHUNKED_ROUND_KEYS = _WHOLE_IMAGE_ROUND_KEYS | frozenset(
    {"chunk_size_zyx", "overlap_fraction"}
)
_CHANNEL_KEYS = frozenset({"file", "interpolation"})
_INITIAL_ALIGNMENT_KEYS = frozenset({"xy_rotation"})
_EXPLICIT_XY_ROTATION_KEYS = frozenset({"mode", "angle_deg"})
_ESTIMATED_XY_ROTATION_KEYS = frozenset(
    {
        "mode",
        "search_range_deg",
        "coarse_step_deg",
        "refinement_step_deg",
        "minimum_score_gain",
        "minimum_peak_margin",
    }
)
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
    if isinstance(value, Mapping) and "channel" in value:
        raise ConfigurationValidationError(
            "schema_version 1 reference.channel is no longer supported; declare "
            "reference.registration_channel and reference.channels"
        )
    mapping = require_exact_keys(value, _REFERENCE_KEYS, "reference")
    channels = tuple(
        ReferenceChannelSpec(
            file=relative_posix_path(
                require_exact_keys(
                    item,
                    _REFERENCE_CHANNEL_KEYS,
                    f"reference.channels[{index}]",
                )["file"],
                f"reference.channels[{index}].file",
            )
        )
        for index, item in enumerate(_list(mapping["channels"], "reference.channels"))
    )
    if not channels:
        raise ConfigurationValidationError("reference.channels must not be empty")
    channel_files = tuple(channel.file for channel in channels)
    if len(set(channel_files)) != len(channel_files):
        raise ConfigurationValidationError(
            f"reference.channels file values must be unique; got {channel_files!r}"
        )
    basenames = tuple(channel.file.name for channel in channels)
    if len(set(basenames)) != len(basenames):
        raise ConfigurationValidationError(
            f"reference.channels output basenames must be unique; got {basenames!r}"
        )
    registration_channel = relative_posix_path(
        mapping["registration_channel"], "reference.registration_channel"
    )
    if channel_files.count(registration_channel) != 1:
        raise ConfigurationValidationError(
            "reference.registration_channel must match exactly one reference.channels "
            f"file; got {registration_channel!s}"
        )
    return ReferenceSpec(
        round=built_in_int(mapping["round"], "reference.round"),
        registration_channel=registration_channel,
        channels=channels,
    )


def _stage(
    value: object, yaml_path: str, accepted_types: frozenset[str]
) -> RegistrationStageSpec:
    mapping = require_exact_keys(value, _LINEAR_STAGE_KEYS, yaml_path)
    return RegistrationStageSpec(
        type=_choice(mapping["type"], accepted_types, f"{yaml_path}.type"),
        accuracy=_choice(mapping["accuracy"], _ACCURACIES, f"{yaml_path}.accuracy"),
    )


def _nonlinear_stage(value: object) -> NonlinearRegistrationStageSpec:
    yaml_path = "registration.nonlinear"
    has_radius = isinstance(value, Mapping) and "radius" in value
    expected_keys = _NONLINEAR_CC_STAGE_KEYS if has_radius else _NONLINEAR_STAGE_KEYS
    mapping = require_exact_keys(value, expected_keys, yaml_path)
    metric = cast(
        NonlinearMetricName,
        _choice(mapping["metric"], frozenset({"CC", "mattes"}), f"{yaml_path}.metric"),
    )
    radius: int | None = None
    if has_radius:
        radius = built_in_int(mapping["radius"], f"{yaml_path}.radius", minimum=1)
    try:
        resolved = resolve_nonlinear_metric(metric, radius)
    except (TypeError, ValueError) as error:
        raise ConfigurationValidationError(f"{yaml_path}.radius: {error}") from error
    return NonlinearRegistrationStageSpec(
        type=_choice(mapping["type"], _NONLINEAR_TYPES, f"{yaml_path}.type"),
        accuracy=_choice(mapping["accuracy"], _ACCURACIES, f"{yaml_path}.accuracy"),
        metric=resolved.metric,
        radius=resolved.radius,
    )


def _filter_sigma_zyx(value: object) -> tuple[float, float, float]:
    yaml_path = "registration.preprocessing.filter_sigma_zyx"
    if isinstance(value, list):
        if len(value) != 3:
            raise ConfigurationValidationError(
                f"{yaml_path} must be a positive finite scalar or contain exactly "
                f"three positive finite ZYX values; got {value!r}"
            )
        return (
            finite_positive_float(value[0], f"{yaml_path}[0]"),
            finite_positive_float(value[1], f"{yaml_path}[1]"),
            finite_positive_float(value[2], f"{yaml_path}[2]"),
        )
    scalar = finite_positive_float(value, yaml_path)
    return scalar, scalar, scalar


def _preprocessing(value: object) -> RegistrationPreprocessingSpec:
    if not isinstance(value, Mapping):
        raise ConfigurationValidationError(
            f"registration.preprocessing must be a mapping; got {value!r}"
        )
    actual = frozenset(value)
    if not all(isinstance(key, str) for key in value) or not actual.issubset(
        _REGISTRATION_PREPROCESSING_KEYS
    ):
        raise ConfigurationValidationError(
            "registration.preprocessing keys must be exactly a subset of "
            f"{sorted(_REGISTRATION_PREPROCESSING_KEYS)!r}; got {sorted(actual)!r}"
        )
    feature_mask: FeatureMaskSpec | None = None
    if "feature_mask" in value:
        feature_mask = _feature_mask(value["feature_mask"])
    return RegistrationPreprocessingSpec(
        filter_sigma_zyx=(
            _filter_sigma_zyx(value["filter_sigma_zyx"])
            if "filter_sigma_zyx" in value
            else (12.0, 12.0, 12.0)
        ),
        feature_mask=feature_mask,
    )


def _feature_mask(value: object) -> FeatureMaskSpec:
    yaml_path = "registration.preprocessing.feature_mask"
    if not isinstance(value, Mapping):
        raise ConfigurationValidationError(
            f"{yaml_path} must be a mapping; got {value!r}"
        )
    actual = frozenset(value)
    if not all(isinstance(key, str) for key in value) or not actual.issubset(
        _FEATURE_MASK_KEYS
    ):
        raise ConfigurationValidationError(
            f"{yaml_path} keys must be exactly a subset of "
            f"{sorted(_FEATURE_MASK_KEYS)!r}; got {sorted(actual)!r}"
        )
    return FeatureMaskSpec(
        min_component_volume_um3=(
            finite_positive_float(
                value["min_component_volume_um3"],
                f"{yaml_path}.min_component_volume_um3",
            )
            if "min_component_volume_um3" in value
            else 1.0
        ),
        dilation_radius_um=(
            finite_positive_float(
                value["dilation_radius_um"],
                f"{yaml_path}.dilation_radius_um",
            )
            if "dilation_radius_um" in value
            else 1.0
        ),
    )


def _registration(value: object) -> RegistrationSpec:
    if not isinstance(value, Mapping):
        raise ConfigurationValidationError(
            f"registration must be a mapping; got {value!r}"
        )
    expected = (
        _REGISTRATION_KEYS | frozenset({"preprocessing"})
        if "preprocessing" in value
        else _REGISTRATION_KEYS
    )
    mapping = require_exact_keys(value, expected, "registration")
    crop = mapping["crop"]
    if type(crop) is not bool:
        raise ConfigurationValidationError(
            f"registration.crop must be a built-in boolean; got {crop!r}"
        )
    return RegistrationSpec(
        crop=crop,
        linear=_stage(mapping["linear"], "registration.linear", _LINEAR_TYPES),
        nonlinear=_nonlinear_stage(mapping["nonlinear"]),
        preprocessing=(
            _preprocessing(mapping["preprocessing"])
            if "preprocessing" in mapping
            else None
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


def _finite_float(value: object, yaml_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationValidationError(
            f"{yaml_path} must be a finite number; got {value!r}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationValidationError(
            f"{yaml_path} must be a finite number; got {value!r}"
        )
    return result


def _xy_rotation(value: object, yaml_path: str) -> object:
    if not isinstance(value, Mapping):
        raise ConfigurationValidationError(
            f"{yaml_path} must be a mapping; got {value!r}"
        )
    mode = value.get("mode")
    if mode == "explicit":
        mapping = require_exact_keys(value, _EXPLICIT_XY_ROTATION_KEYS, yaml_path)
        angle = _finite_float(mapping["angle_deg"], f"{yaml_path}.angle_deg")
        if not -180.0 <= angle <= 180.0:
            raise ConfigurationValidationError(
                f"{yaml_path}.angle_deg must be within [-180, 180]; got {angle!r}"
            )
        return ExplicitXYRotationSpec(mode="explicit", angle_deg=angle)
    if mode != "estimate":
        raise ConfigurationValidationError(
            f"{yaml_path}.mode must be one of ['estimate', 'explicit']; got {mode!r}"
        )
    actual = frozenset(value)
    if not all(isinstance(key, str) for key in value) or not actual.issubset(
        _ESTIMATED_XY_ROTATION_KEYS
    ):
        raise ConfigurationValidationError(
            f"{yaml_path} keys must be exactly a subset of "
            f"{sorted(_ESTIMATED_XY_ROTATION_KEYS)!r}; got {sorted(actual)!r}"
        )
    search_value = value.get("search_range_deg", [-45.0, 45.0])
    search = _list(search_value, f"{yaml_path}.search_range_deg")
    if len(search) != 2:
        raise ConfigurationValidationError(
            f"{yaml_path}.search_range_deg must contain exactly two values; "
            f"got {search_value!r}"
        )
    lower = _finite_float(search[0], f"{yaml_path}.search_range_deg[0]")
    upper = _finite_float(search[1], f"{yaml_path}.search_range_deg[1]")
    if lower < -180.0 or upper > 180.0 or lower >= upper:
        raise ConfigurationValidationError(
            f"{yaml_path}.search_range_deg must be strictly increasing within "
            f"[-180, 180]; got {search_value!r}"
        )
    coarse = finite_positive_float(
        value.get("coarse_step_deg", 1.0), f"{yaml_path}.coarse_step_deg"
    )
    refinement = finite_positive_float(
        value.get("refinement_step_deg", 0.1),
        f"{yaml_path}.refinement_step_deg",
    )
    if refinement >= coarse:
        raise ConfigurationValidationError(
            f"{yaml_path}.refinement_step_deg must be smaller than "
            f"coarse_step_deg; got {refinement!r} >= {coarse!r}"
        )
    return EstimatedXYRotationSpec(
        mode="estimate",
        search_range_deg=(lower, upper),
        coarse_step_deg=coarse,
        refinement_step_deg=refinement,
        minimum_score_gain=finite_positive_float(
            value.get("minimum_score_gain", 0.01),
            f"{yaml_path}.minimum_score_gain",
        ),
        minimum_peak_margin=finite_positive_float(
            value.get("minimum_peak_margin", 0.01),
            f"{yaml_path}.minimum_peak_margin",
        ),
    )


def _initial_alignment(value: object, yaml_path: str) -> InitialAlignmentSpec:
    mapping = require_exact_keys(value, _INITIAL_ALIGNMENT_KEYS, yaml_path)
    return InitialAlignmentSpec(
        xy_rotation=cast(
            ExplicitXYRotationSpec | EstimatedXYRotationSpec,
            _xy_rotation(mapping["xy_rotation"], f"{yaml_path}.xy_rotation"),
        )
    )


def _round(value: object, index: int) -> RoundSpec:
    yaml_path = f"rounds[{index}]"
    if isinstance(value, Mapping) and "strategy" in value:
        raw_strategy = value["strategy"]
    else:
        preliminary_mapping = require_exact_keys(
            value, _WHOLE_IMAGE_ROUND_KEYS, yaml_path
        )
        raw_strategy = preliminary_mapping["strategy"]
    strategy = cast(
        RegistrationStrategy,
        _choice(raw_strategy, _STRATEGIES, f"{yaml_path}.strategy"),
    )
    expected_keys = (
        _SERIAL_CHUNKED_ROUND_KEYS
        if strategy == "serial_chunked"
        else _WHOLE_IMAGE_ROUND_KEYS
    )
    if isinstance(value, Mapping) and "initial_alignment" in value:
        expected_keys |= frozenset({"initial_alignment"})
    mapping = require_exact_keys(value, expected_keys, yaml_path)
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
        initial_alignment=(
            _initial_alignment(
                mapping["initial_alignment"], f"{yaml_path}.initial_alignment"
            )
            if "initial_alignment" in mapping
            else None
        ),
    )


def _rounds(value: object, reference_round: int) -> tuple[RoundSpec, ...]:
    rounds = tuple(
        _round(item, index) for index, item in enumerate(_list(value, "rounds"))
    )
    if not rounds:
        raise ConfigurationValidationError("rounds must not be empty")
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


def _model_relative_path(value: object) -> object:
    return value.as_posix() if isinstance(value, PurePosixPath) else value


def _validate_dataset_manifest(manifest: DatasetManifest) -> None:
    """Revalidate a directly constructed manifest against the YAML contract."""
    if type(manifest) is not DatasetManifest:
        raise ConfigurationValidationError(
            f"manifest must be a DatasetManifest; got {manifest!r}"
        )
    if type(manifest.schema_version) is not int or manifest.schema_version != 1:
        raise ConfigurationValidationError(
            "schema_version must be the built-in integer 1; "
            f"got {manifest.schema_version!r}"
        )
    if type(manifest.dataset_id) is not str:
        raise ConfigurationValidationError(
            f"dataset_id must be a built-in string; got {manifest.dataset_id!r}"
        )
    if type(manifest.rounds) is not tuple:
        raise ConfigurationValidationError("rounds must be an exact immutable tuple")
    if type(manifest.reference) is not ReferenceSpec:
        raise ConfigurationValidationError(
            f"reference must be a ReferenceSpec; got {manifest.reference!r}"
        )
    if type(manifest.reference.round) is not int or manifest.reference.round < 0:
        raise ConfigurationValidationError(
            "reference.round must be a nonnegative built-in integer; "
            f"got {manifest.reference.round!r}"
        )
    if type(manifest.reference.registration_channel) is not PurePosixPath:
        raise ConfigurationValidationError(
            "reference.registration_channel must be a PurePosixPath; "
            f"got {manifest.reference.registration_channel!r}"
        )
    if type(manifest.reference.channels) is not tuple:
        raise ConfigurationValidationError(
            "reference.channels must be an exact immutable tuple"
        )
    reference_channels: list[dict[str, object]] = []
    for index, channel in enumerate(manifest.reference.channels):
        channel_path = f"reference.channels[{index}]"
        if type(channel) is not ReferenceChannelSpec:
            raise ConfigurationValidationError(
                f"{channel_path} must be a ReferenceChannelSpec; got {channel!r}"
            )
        if type(channel.file) is not PurePosixPath:
            raise ConfigurationValidationError(
                f"{channel_path}.file must be a PurePosixPath; got {channel.file!r}"
            )
        reference_channels.append({"file": _model_relative_path(channel.file)})
    if type(manifest.registration) is not RegistrationSpec:
        raise ConfigurationValidationError(
            f"registration must be a RegistrationSpec; got {manifest.registration!r}"
        )
    if type(manifest.registration.crop) is not bool:
        raise ConfigurationValidationError(
            "registration.crop must be a built-in boolean; "
            f"got {manifest.registration.crop!r}"
        )
    linear_stage = manifest.registration.linear
    if type(linear_stage) is not RegistrationStageSpec:
        raise ConfigurationValidationError(
            f"registration.linear must be a RegistrationStageSpec; got {linear_stage!r}"
        )
    nonlinear_stage = manifest.registration.nonlinear
    if type(nonlinear_stage) is not NonlinearRegistrationStageSpec:
        raise ConfigurationValidationError(
            "registration.nonlinear must be a NonlinearRegistrationStageSpec; "
            f"got {nonlinear_stage!r}"
        )
    for stage_name, values in (
        (
            "linear",
            (("type", linear_stage.type), ("accuracy", linear_stage.accuracy)),
        ),
        (
            "nonlinear",
            (
                ("type", nonlinear_stage.type),
                ("accuracy", nonlinear_stage.accuracy),
                ("metric", nonlinear_stage.metric),
            ),
        ),
    ):
        for field_name, value in values:
            if type(value) is not str:
                raise ConfigurationValidationError(
                    f"registration.{stage_name}.{field_name} must be a built-in "
                    f"string; got {value!r}"
                )
    if nonlinear_stage.radius is not None and type(nonlinear_stage.radius) is not int:
        raise ConfigurationValidationError(
            "registration.nonlinear.radius must be a positive built-in integer "
            f"or null; got {nonlinear_stage.radius!r}"
        )
    spacing = manifest.voxel_spacing_um
    if type(spacing) is not tuple or len(spacing) != 3:
        raise ConfigurationValidationError(
            "voxel_spacing_um must be an exact tuple containing three built-in "
            f"floats; got {spacing!r}"
        )
    for index, value in enumerate(spacing):
        if type(value) is not float or not math.isfinite(value) or value <= 0.0:
            raise ConfigurationValidationError(
                f"voxel_spacing_um[{index}] must be a positive finite built-in "
                f"float; got {value!r}"
            )
    parsed_reference = _reference(
        {
            "round": manifest.reference.round,
            "registration_channel": _model_relative_path(
                manifest.reference.registration_channel
            ),
            "channels": reference_channels,
        }
    )
    nonlinear_mapping: dict[str, object] = {
        "type": nonlinear_stage.type,
        "accuracy": nonlinear_stage.accuracy,
        "metric": nonlinear_stage.metric,
    }
    if nonlinear_stage.metric == "CC":
        nonlinear_mapping["radius"] = nonlinear_stage.radius
    registration_mapping: dict[str, object] = {
        "crop": manifest.registration.crop,
        "linear": {
            "type": linear_stage.type,
            "accuracy": linear_stage.accuracy,
        },
        "nonlinear": nonlinear_mapping,
    }
    preprocessing = manifest.registration.preprocessing
    if preprocessing is not None:
        if type(preprocessing) is not RegistrationPreprocessingSpec:
            raise ConfigurationValidationError(
                "registration.preprocessing must be a "
                "RegistrationPreprocessingSpec or null; "
                f"got {preprocessing!r}"
            )
        sigma = preprocessing.filter_sigma_zyx
        if type(sigma) is not tuple or len(sigma) != 3:
            raise ConfigurationValidationError(
                "registration.preprocessing.filter_sigma_zyx must be an exact "
                "tuple containing three built-in floats; "
                f"got {sigma!r}"
            )
        for index, value in enumerate(sigma):
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ConfigurationValidationError(
                    "registration.preprocessing.filter_sigma_zyx"
                    f"[{index}] must be a positive finite built-in float; "
                    f"got {value!r}"
                )
        preprocessing_mapping: dict[str, object] = {"filter_sigma_zyx": list(sigma)}
        feature_mask = preprocessing.feature_mask
        if feature_mask is not None:
            if type(feature_mask) is not FeatureMaskSpec:
                raise ConfigurationValidationError(
                    "registration.preprocessing.feature_mask must be a "
                    f"FeatureMaskSpec or null; got {feature_mask!r}"
                )
            mask_mapping: dict[str, float] = {}
            for name in (
                "min_component_volume_um3",
                "dilation_radius_um",
            ):
                value = getattr(feature_mask, name)
                if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                    raise ConfigurationValidationError(
                        f"registration.preprocessing.feature_mask.{name} must "
                        f"be a positive finite built-in float; got {value!r}"
                    )
                mask_mapping[name] = value
            preprocessing_mapping["feature_mask"] = mask_mapping
        registration_mapping["preprocessing"] = preprocessing_mapping
    parsed_registration = _registration(registration_mapping)
    round_mappings: list[dict[str, object]] = []
    for index, round_spec in enumerate(manifest.rounds):
        round_path = f"rounds[{index}]"
        if type(round_spec) is not RoundSpec:
            raise ConfigurationValidationError(
                f"{round_path} must be a RoundSpec; got {round_spec!r}"
            )
        if type(round_spec.round) is not int or round_spec.round < 0:
            raise ConfigurationValidationError(
                f"{round_path}.round must be a nonnegative built-in integer; "
                f"got {round_spec.round!r}"
            )
        if type(round_spec.strategy) is not str:
            raise ConfigurationValidationError(
                f"{round_path}.strategy must be a built-in string; "
                f"got {round_spec.strategy!r}"
            )
        if type(round_spec.registration_channel) is not PurePosixPath:
            raise ConfigurationValidationError(
                f"{round_path}.registration_channel must be a PurePosixPath; "
                f"got {round_spec.registration_channel!r}"
            )
        if type(round_spec.channels) is not tuple:
            raise ConfigurationValidationError(
                f"{round_path}.channels must be an exact immutable tuple"
            )
        channels: list[dict[str, object]] = []
        for channel_index, channel in enumerate(round_spec.channels):
            channel_path = f"{round_path}.channels[{channel_index}]"
            if type(channel) is not ChannelSpec:
                raise ConfigurationValidationError(
                    f"{channel_path} must be a ChannelSpec; got {channel!r}"
                )
            if type(channel.file) is not PurePosixPath:
                raise ConfigurationValidationError(
                    f"{channel_path}.file must be a PurePosixPath; got {channel.file!r}"
                )
            if type(channel.interpolation) is not str:
                raise ConfigurationValidationError(
                    f"{channel_path}.interpolation must be a built-in string; "
                    f"got {channel.interpolation!r}"
                )
            channels.append(
                {
                    "file": _model_relative_path(channel.file),
                    "interpolation": channel.interpolation,
                }
            )
        round_mapping: dict[str, object] = {
            "round": round_spec.round,
            "strategy": round_spec.strategy,
            "registration_channel": _model_relative_path(
                round_spec.registration_channel
            ),
            "channels": channels,
        }
        initial_alignment = round_spec.initial_alignment
        if initial_alignment is not None:
            if type(initial_alignment) is not InitialAlignmentSpec:
                raise ConfigurationValidationError(
                    f"{round_path}.initial_alignment must be an "
                    f"InitialAlignmentSpec or null; got {initial_alignment!r}"
                )
            rotation = initial_alignment.xy_rotation
            if type(rotation) is ExplicitXYRotationSpec:
                if type(rotation.mode) is not str or rotation.mode != "explicit":
                    raise ConfigurationValidationError(
                        f"{round_path}.initial_alignment.xy_rotation.mode must be "
                        f"the built-in string 'explicit'; got {rotation.mode!r}"
                    )
                if type(rotation.angle_deg) is not float:
                    raise ConfigurationValidationError(
                        f"{round_path}.initial_alignment.xy_rotation.angle_deg must "
                        f"be a built-in float; got {rotation.angle_deg!r}"
                    )
                rotation_mapping: dict[str, object] = {
                    "mode": rotation.mode,
                    "angle_deg": rotation.angle_deg,
                }
            elif type(rotation) is EstimatedXYRotationSpec:
                if type(rotation.mode) is not str or rotation.mode != "estimate":
                    raise ConfigurationValidationError(
                        f"{round_path}.initial_alignment.xy_rotation.mode must be "
                        f"the built-in string 'estimate'; got {rotation.mode!r}"
                    )
                rotation_mapping = {
                    "mode": rotation.mode,
                    "search_range_deg": list(rotation.search_range_deg),
                    "coarse_step_deg": rotation.coarse_step_deg,
                    "refinement_step_deg": rotation.refinement_step_deg,
                    "minimum_score_gain": rotation.minimum_score_gain,
                    "minimum_peak_margin": rotation.minimum_peak_margin,
                }
            else:
                raise ConfigurationValidationError(
                    f"{round_path}.initial_alignment.xy_rotation has an unsupported "
                    f"type; got {rotation!r}"
                )
            round_mapping["initial_alignment"] = {"xy_rotation": rotation_mapping}
        if round_spec.strategy == "serial_chunked":
            chunk_size = round_spec.chunk_size_zyx
            if type(chunk_size) is not tuple or len(chunk_size) != 3:
                raise ConfigurationValidationError(
                    f"{round_path}.chunk_size_zyx must be an exact tuple of three "
                    "positive built-in integers"
                )
            for chunk_index, value in enumerate(chunk_size):
                if type(value) is not int or value < 1:
                    raise ConfigurationValidationError(
                        f"{round_path}.chunk_size_zyx[{chunk_index}] must be a "
                        f"positive built-in integer; got {value!r}"
                    )
            overlap = round_spec.overlap_fraction
            if type(overlap) is not float or not math.isfinite(overlap):
                raise ConfigurationValidationError(
                    f"{round_path}.overlap_fraction must be a finite built-in "
                    f"float; got {overlap!r}"
                )
            if not 0.0 <= overlap < 1.0:
                raise ConfigurationValidationError(
                    f"{round_path}.overlap_fraction must satisfy 0 <= value < 1; "
                    f"got {overlap!r}"
                )
            round_mapping.update(
                chunk_size_zyx=list(chunk_size),
                overlap_fraction=overlap,
            )
        elif round_spec.strategy == "whole_image" and (
            round_spec.chunk_size_zyx is not None
            or round_spec.overlap_fraction is not None
        ):
            raise ConfigurationValidationError(
                f"{round_path}.chunk_size_zyx and overlap_fraction must be null "
                "for whole_image"
            )
        round_mappings.append(round_mapping)
    parsed = DatasetManifest(
        schema_version=built_in_int(manifest.schema_version, "schema_version"),
        dataset_id=_dataset_id(manifest.dataset_id),
        voxel_spacing_um=_voxel_spacing(
            {"x": spacing[0], "y": spacing[1], "z": spacing[2]}
        ),
        reference=parsed_reference,
        registration=parsed_registration,
        rounds=_rounds(round_mappings, parsed_reference.round),
    )
    if parsed != manifest:
        raise ConfigurationValidationError(
            "directly constructed manifest does not match its normalized contract"
        )


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    mapping = require_exact_keys(load_yaml_mapping(path), _DATASET_KEYS, "dataset")
    schema_version = built_in_int(mapping["schema_version"], "schema_version")
    if schema_version != 1:
        raise ConfigurationValidationError(
            f"schema_version must be the built-in integer 1; got {schema_version!r}"
        )
    reference = _reference(mapping["reference"])
    manifest = DatasetManifest(
        schema_version=schema_version,
        dataset_id=_dataset_id(mapping["dataset_id"]),
        voxel_spacing_um=_voxel_spacing(mapping["voxel_spacing_um"]),
        reference=reference,
        registration=_registration(mapping["registration"]),
        rounds=_rounds(mapping["rounds"], reference.round),
    )
    _validate_dataset_manifest(manifest)
    return manifest
