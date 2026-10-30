import io
import queue
import unittest
from types import SimpleNamespace
from teleoperation.inputs.smplerx import MAX_PACKET_BYTES
from teleoperation.inputs.smplerx_process import SmplerXWorkerTransport


class FragmentedStream(io.BytesIO):
    def read(self, count=-1):
        return super().read(min(count, 2))


class SmplerXTransportTests(unittest.TestCase):
    def transport(self, stream):
        transport = SmplerXWorkerTransport.__new__(SmplerXWorkerTransport)
        transport._process = SimpleNamespace(stdout=stream)
        transport._results = queue.Queue()
        return transport

    def test_stdout_reassembles_short_reads(self):
        packet = (8).to_bytes(4, 'big') + b'12345678'
        transport = self.transport(FragmentedStream(packet))
        transport._read_stdout()
        self.assertEqual(transport._results.get_nowait(), packet)

    def test_oversize_prefix_is_rejected_before_reading_payload(self):
        transport = self.transport(io.BytesIO((MAX_PACKET_BYTES + 1).to_bytes(4, 'big')))
        transport._read_stdout()
        self.assertIsInstance(transport._results.get_nowait(), ValueError)
