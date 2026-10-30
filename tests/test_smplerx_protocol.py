import json
import unittest

import numpy as np

from teleoperation.contracts.body import BodyFrameId, SmplerXResult
from teleoperation.inputs.smplerx import (
    MAX_PACKET_BYTES,
    PacketError,
    PacketReader,
    SmplerXClient,
    decode_packet,
    encode_packet,
)


class FakeTransport:
    def __init__(self):
        self.sent = []
        self.incoming = []

    def send(self, packet):
        self.sent.append(packet)

    def poll(self):
        return self.incoming.pop(0) if self.incoming else None


def result(session, sequence):
    return SmplerXResult(
        frame_id=BodyFrameId(session, sequence, sequence / 30),
        completed_time_s=sequence / 30 + 0.01,
        model_id="s32",
        model_hash="a" * 64,
        arrays={"left_arm": np.zeros((3, 3), dtype=np.float32)},
        valid={"left": True, "right": False},
    )


class SmplerXProtocolTests(unittest.TestCase):
    def test_packet_reader_accepts_fragmented_json_frame_and_arrays_are_explicit(self):
        packet = encode_packet("result", {"sample": result("s", 1)})
        reader = PacketReader()
        messages = []
        for byte in packet:
            messages.extend(reader.feed(bytes((byte,))))
        self.assertEqual(len(messages), 1)
        decoded = decode_packet(messages[0])
        self.assertEqual(decoded["type"], "result")
        decoded_result = decoded["payload"]["sample"]
        self.assertEqual(decoded_result.arrays["left_arm"].shape, (3, 3))
        self.assertEqual(decoded_result.arrays["left_arm"].dtype, np.dtype("float32"))

    def test_packet_reader_rejects_invalid_length_and_json(self):
        reader = PacketReader()
        with self.assertRaises(PacketError):
            reader.feed((MAX_PACKET_BYTES + 1).to_bytes(4, "big"))
        with self.assertRaises(PacketError):
            reader.feed((3).to_bytes(4, "big") + b"no!")

    def test_latest_pending_frame_replaces_old_one(self):
        transport = FakeTransport()
        client = SmplerXClient(transport, session_id="s")
        client.submit(result("s", 1).frame_id, np.zeros((2, 2, 3), dtype=np.uint8))
        client.submit(result("s", 2).frame_id, np.zeros((2, 2, 3), dtype=np.uint8))
        client.submit(result("s", 3).frame_id, np.zeros((2, 2, 3), dtype=np.uint8))
        self.assertEqual(len(transport.sent), 1)
        transport.incoming.append(result("s", 1))
        self.assertEqual(client.poll().frame_id.sequence, 1)
        self.assertEqual(len(transport.sent), 2)
        submitted = decode_packet(transport.sent[-1])
        self.assertEqual(submitted["payload"]["frame_id"].sequence, 3)

    def test_old_session_result_is_rejected(self):
        transport = FakeTransport()
        client = SmplerXClient(transport, session_id="new")
        client.submit(BodyFrameId("new", 1, 0.0), np.zeros((2, 2, 3), dtype=np.uint8))
        transport.incoming.append(result("old", 1))
        with self.assertRaises(PacketError):
            client.poll()


if __name__ == "__main__":
    unittest.main()
