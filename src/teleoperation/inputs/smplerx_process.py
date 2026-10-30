"""Subprocess transport for the isolated SMPLer-X worker.

The transport keeps worker stdout reserved for framed protocol packets.  Any
diagnostics are collected from stderr and can be shown by the application when
the worker exits or fails.
"""

from __future__ import annotations

import queue
import subprocess
import threading
from pathlib import Path


class SmplerXWorkerTransport:
    def __init__(self, *, worker_python, smplerx_root, checkpoint, body_model_dir,
                 model="s32", worker_script=None):
        self.worker_python = Path(worker_python).expanduser().resolve()
        self.smplerx_root = Path(smplerx_root).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.body_model_dir = Path(body_model_dir).expanduser().resolve()
        self.model = model
        self.worker_script = Path(worker_script or Path(__file__).resolve().parents[3]
                                  / "external" / "smplerx_worker" / "worker.py").resolve()
        for name, path in (("worker python", self.worker_python), ("SMPLer-X root", self.smplerx_root),
                           ("SMPLer-X checkpoint", self.checkpoint), ("SMPL-X model directory", self.body_model_dir),
                           ("worker script", self.worker_script)):
            if not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")
        self._results: queue.Queue[bytes | BaseException] = queue.Queue()
        self._closed = False
        self.stderr_lines: list[str] = []
        self._process = subprocess.Popen(
            [str(self.worker_python), str(self.worker_script),
             "--smplerx-root", str(self.smplerx_root),
             "--checkpoint", str(self.checkpoint),
             "--body-model-dir", str(self.body_model_dir),
             "--model", self.model],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(self.smplerx_root), bufsize=0,
        )
        self._stdout_thread = threading.Thread(target=self._read_stdout, name="smplerx-worker-stdout", daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, name="smplerx-worker-stderr", daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self):
        try:
            while True:
                prefix = self._process.stdout.read(4)
                if not prefix:
                    break
                length = int.from_bytes(prefix, "big")
                packet = prefix + self._process.stdout.read(length)
                if len(packet) != length + 4:
                    raise RuntimeError("SMPLer-X worker returned a truncated packet")
                self._results.put(packet)
        except BaseException as error:
            self._results.put(error)

    def _read_stderr(self):
        for line in iter(self._process.stderr.readline, b""):
            self.stderr_lines.append(line.decode("utf-8", errors="replace").rstrip())

    def send(self, packet: bytes) -> None:
        if self._closed or self._process.poll() is not None:
            raise RuntimeError("SMPLer-X worker is not running")
        self._process.stdin.write(packet)
        self._process.stdin.flush()

    def poll(self):
        try:
            value = self._results.get_nowait()
        except queue.Empty:
            if self._process.poll() is not None:
                message = "SMPLer-X worker exited"
                if self.stderr_lines:
                    message += ": " + self.stderr_lines[-1]
                raise RuntimeError(message)
            return None
        if isinstance(value, BaseException):
            raise RuntimeError(str(value)) from value
        return value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=2)
