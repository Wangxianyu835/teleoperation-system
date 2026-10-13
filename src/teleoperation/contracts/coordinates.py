from numbers import Real


COORDINATE_FRAME = "l21"
COORDINATE_ALIGNMENT = "source_to_l21_xyz"
LEGACY_PALM_LOCAL_ALIGNMENT = "palm_local_to_l21_v1"
LEGACY_WORLD_PALM_LOCAL_ALIGNMENT = "mediapipe_world_palm_local_to_l21_v1"
PALM_LOCAL_COORDINATE_ALIGNMENT = "palm_local_to_l21_v2"
WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT = "mediapipe_world_wrist_index_middle_canonical_right_v1"
CANONICAL_HAND_FRAME = "hand_canonical_right"
CANONICAL_ANGLE_CONVENTION = "l21_canonical_right_angles_v1"
NORMALIZED_LANDMARK_SPACE = "mediapipe_normalized"
WORLD_LANDMARK_SPACE = "mediapipe_world_meters"
SUPPORTED_COORDINATE_ALIGNMENTS = frozenset({
    COORDINATE_ALIGNMENT, PALM_LOCAL_COORDINATE_ALIGNMENT, WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT,
    LEGACY_PALM_LOCAL_ALIGNMENT, LEGACY_WORLD_PALM_LOCAL_ALIGNMENT,
})


def validate_coordinate_alignment(value: object, context: str) -> str:
    """Decode a declared identifier; never infer a mode from absent metadata."""
    if value is None:
        raise ValueError(
            f"{context} does not declare coordinate_alignment. "
            "Refusing to infer legacy or palm-local coordinates."
        )
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{context} coordinate_alignment must be UTF-8") from error
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} coordinate_alignment must be a nonempty scalar string")
    if value not in SUPPORTED_COORDINATE_ALIGNMENTS:
        raise ValueError(
            f"{context}: Unsupported coordinate_alignment={value!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_COORDINATE_ALIGNMENTS))}"
        )
    return value

PALM_LOCAL_METADATA = {
    "coordinate_frame": COORDINATE_FRAME,
    "coordinate_alignment": PALM_LOCAL_COORDINATE_ALIGNMENT,
    "source_landmark_space": "mediapipe_normalized",
    "palm_basis_version": "wrist_index_middle_svd_v1",
    "palm_reference_indices": "0,5,9",
    "palm_longitudinal": "wrist_to_middle_mcp",
    "palm_lateral": "middle_mcp_to_index_mcp_projected",
    "palm_normal": "three_point_svd_signed_toward_index",
}

WORLD_PALM_LOCAL_METADATA = {
    **PALM_LOCAL_METADATA,
    "coordinate_alignment": WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT,
    "coordinate_frame": CANONICAL_HAND_FRAME,
    "origin": "wrist",
    "canonical_hand": "right",
    "handedness_normalization": "dex_left_then_y_reflection",
    "source_landmark_space": WORLD_LANDMARK_SPACE,
    "units": "m",
    "length_unit": "m",
    "source_scale": 1.0,
}


def is_world_alignment(alignment: str) -> bool:
    return alignment in (WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT, LEGACY_WORLD_PALM_LOCAL_ALIGNMENT)


def coordinate_frame_for_alignment(alignment: str) -> str:
    return CANONICAL_HAND_FRAME if alignment == WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT else COORDINATE_FRAME


def landmark_source(landmark_space: str) -> str:
    if landmark_space not in ("world", "normalized"):
        raise ValueError(f"Unknown landmark space: {landmark_space!r}")
    return WORLD_LANDMARK_SPACE if landmark_space == "world" else NORMALIZED_LANDMARK_SPACE


def palm_local_metadata(landmark_space: str) -> dict:
    landmark_source(landmark_space)
    return dict(WORLD_PALM_LOCAL_METADATA if landmark_space == "world" else PALM_LOCAL_METADATA)


def validate_world_contract(metadata, context: str) -> None:
    """World coordinates retain metres and the unscaled human hand geometry."""
    for name, expected in (("source_landmark_space", WORLD_LANDMARK_SPACE), ("length_unit", "m")):
        value = metadata.get(name)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if value != expected:
            raise ValueError(f"{context} must declare {name}={expected!r}; got {value!r}")
    scale = metadata.get("source_scale")
    if isinstance(scale, bool) or not isinstance(scale, Real) or scale != 1.0:
        raise ValueError(f"{context} world landmarks require source_scale=1.0; got {scale!r}")
    units = metadata.get("units", "m")
    if isinstance(units, bytes):
        units = units.decode("utf-8")
    if units != "m":
        raise ValueError(f"{context} world landmarks require units='m'")


def validate_landmark_contract(alignment, metadata, context):
    if is_world_alignment(alignment):
        validate_world_contract(metadata, context)
    if alignment == WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT:
        for name in ("coordinate_frame", "palm_basis_version", "canonical_hand", "handedness_normalization"):
            value = metadata.get(name)
            if isinstance(value, bytes): value = value.decode("utf-8")
            if value != WORLD_PALM_LOCAL_METADATA[name]:
                raise ValueError(f"{context} must declare {name}={WORLD_PALM_LOCAL_METADATA[name]!r}; got {value!r}")
    elif alignment in (PALM_LOCAL_COORDINATE_ALIGNMENT, LEGACY_PALM_LOCAL_ALIGNMENT):
        # Historical normalized checkpoints may omit provenance.
        space = metadata.get("source_landmark_space", NORMALIZED_LANDMARK_SPACE)
        if isinstance(space, bytes):
            space = space.decode("utf-8")
        if space not in (None, NORMALIZED_LANDMARK_SPACE):
            raise ValueError(f"{context} normalized alignment requires source_landmark_space={NORMALIZED_LANDMARK_SPACE!r}")
