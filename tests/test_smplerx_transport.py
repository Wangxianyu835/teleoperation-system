import io
import queue
import unittest
import threading
from types import SimpleNamespace
from teleoperation.inputs.smplerx import MAX_PACKET_BYTES
from teleoperation.inputs.smplerx_process import SmplerXWorkerTransport


class FragmentedStream(io.BytesIO):
    def read(self, count=-1):
        return super().read(min(count, 2))


class SmplerXTransportTests(unittest.TestCase):
    def test_send_returns_while_worker_pipe_is_blocked(self):
        blocked = threading.Event()
        done = threading.Event()
        transport = self.transport(io.BytesIO())
        transport._closed = False
        transport._send_queue = queue.Queue(maxsize=1)
        transport._process.poll = lambda: None
        transport._process.stdin = SimpleNamespace(write=lambda data: blocked.wait(2), flush=lambda: None)
        caller = threading.Thread(target=lambda: (transport.send(b'packet'), done.set()), daemon=True)
        caller.start()
        try:
            self.assertTrue(done.wait(.2), 'submit blocked on the worker pipe')
            self.assertEqual(transport._send_queue.get_nowait(), b'packet')
        finally:
            blocked.set()
            caller.join(3)

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
