COORDINATE_FRAME = "l21"
COORDINATE_ALIGNMENT = "source_to_l21_xyz"
PALM_LOCAL_COORDINATE_ALIGNMENT = "palm_local_to_l21_v1"
SUPPORTED_COORDINATE_ALIGNMENTS = frozenset({
    COORDINATE_ALIGNMENT, PALM_LOCAL_COORDINATE_ALIGNMENT,
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
    "palm_basis_version": "wrist_four_mcp_pinky_to_index_v1",
    "palm_longitudinal": "wrist_to_four_mcp_mean",
    "palm_lateral": "pinky_mcp_to_index_mcp",
    "palm_normal": "lateral_cross_longitudinal",
}
