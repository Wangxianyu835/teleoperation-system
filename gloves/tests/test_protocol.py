from __future__ import annotations

import array
import struct

import numpy as np

from ffg_capture.transport.vendor.linkerforce import FrameHandler, FrameParser, HandType


def test_float_position_frame_is_parsed_to_21_radians():
    degrees = [float((index * 13) % 360) for index in range(21)]
    payload = struct.pack("<21f", *degrees)
    frame = FrameHandler.pack_data(0x03, payload)
    result = FrameHandler(HandType.right).handle_frame(array.array("B", frame))
    assert result is not None
    assert len(result["poslist"]) == 21
    np.testing.assert_allclose(result["poslist"], np.deg2rad(degrees), atol=1e-6)


def test_compressed_position_frame_is_parsed():
    hundredths = [index * 1000 for index in range(21)]
    payload = b"".join(struct.pack("<H", value) for value in hundredths)
    frame = FrameHandler.pack_data(0x05, payload)
    result = FrameHandler(HandType.right).handle_frame(array.array("B", frame))
    assert result is not None
    np.testing.assert_allclose(
        result["poslist"],
        np.deg2rad([value / 100.0 for value in hundredths]),
        atol=1e-6,
    )


def test_bad_checksum_is_rejected():
    payload = struct.pack("<21f", *([10.0] * 21))
    frame = bytearray(FrameHandler.pack_data(0x03, payload))
    frame[-1] = (frame[-1] + 1) & 0xFF
    parser = FrameParser()
    completed = [parser.process_byte(value) for value in frame]
    assert not any(completed)
