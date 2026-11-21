"""FFG channel definitions with a stable order for datasets and plots."""

from __future__ import annotations

from dataclasses import dataclass

EXPECTED_FFG_CHANNELS = 21

CHANNEL_NAMES = (
    "thumb_side_0",
    "thumb_rotation_side_1",
    "thumb_flex_2",
    "thumb_flex_3",
    "thumb_tip_4",
    "index_roll",
    "index_root_flex",
    "index_middle_flex",
    "index_tip_flex",
    "middle_roll",
    "middle_root_flex",
    "middle_middle_flex",
    "middle_tip_flex",
    "ring_roll",
    "ring_root_flex",
    "ring_middle_flex",
    "ring_tip_flex",
    "pinky_roll",
    "pinky_root_flex",
    "pinky_middle_flex",
    "pinky_tip_flex",
)

CHANNEL_GROUPS = {
    "Thumb": tuple(range(0, 5)),
    "Index": tuple(range(5, 9)),
    "Middle": tuple(range(9, 13)),
    "Ring": tuple(range(13, 17)),
    "Little": tuple(range(17, 21)),
}

FLEXION_GROUPS = {
    "Thumb": (2, 3, 4),
    "Index": (6, 7, 8),
    "Middle": (10, 11, 12),
    "Ring": (14, 15, 16),
    "Little": (18, 19, 20),
}

SIDE_CHANNELS = {
    "Thumb": (0, 1),
    "Index": (5,),
    "Middle": (9,),
    "Ring": (13,),
    "Little": (17,),
}

FLEXION_CHANNEL_INDICES = tuple(
    index for indices in FLEXION_GROUPS.values() for index in indices
)
SIDE_CHANNEL_INDICES = tuple(
    index for indices in SIDE_CHANNELS.values() for index in indices
)


@dataclass(frozen=True)
class FFGChannelLayout:
    """Provisional FFG channel metadata.

    The official transport reports 21 ordered positions. The aliases above are
    usable for visualisation; the retargeting owner should confirm any
    anatomical aliases against the physical device and calibration output.
    """

    names: tuple[str, ...] = CHANNEL_NAMES
    unit: str = "rad"
    layout_id: str = "ffg_raw21_5_4_4_4_4_v1"
    anatomical_aliases_confirmed: bool = False

    def __post_init__(self) -> None:
        if len(self.names) != EXPECTED_FFG_CHANNELS:
            raise ValueError(
                f"FFG layout requires {EXPECTED_FFG_CHANNELS} names, got {len(self.names)}"
            )
