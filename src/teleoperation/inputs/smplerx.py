"""Small, model-independent protocol client for the SMPLer-X worker."""
from __future__ import annotations

import base64
import json
import struct
from typing import Any

import numpy as np

from teleoperation.contracts.body import BodyFrameId, SmplerXResult


PROTOCOL_VERSION = 1
MAX_PACKET_BYTES = 16 * 1024 * 1024


class PacketError(ValueError):
    pass


def _encode(value: Any) -> Any:
    if isinstance(value, BodyFrameId):
        return {"__type__": "frame_id", "session_id": value.session_id,
                "sequence": value.sequence, "capture_time_s": value.capture_time_s}
    if isinstance(value, SmplerXResult):
        return {
            "__type__": "result",
            "frame_id": _encode(value.frame_id),
            "completed_time_s": value.completed_time_s,
            "model_id": value.model_id,
            "model_hash": value.model_hash,
            "arrays": {name: _encode_array(array) for name, array in value.arrays.items()},
            "valid": dict(value.valid),
            "reason": value.reason,
            "metadata": dict(value.metadata or {}),
        }
    if isinstance(value, np.ndarray):
        return _encode_array(value)
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported protocol value: {type(value).__name__}")


def _encode_array(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    if value.dtype.kind not in "biuf":
        raise TypeError("protocol arrays must use numeric dtypes")
    return {"__type__": "array", "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": base64.b64encode(value.tobytes(order="C")).decode("ascii")}


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    marker = value.get("__type__")
    if marker == "frame_id":
        return BodyFrameId(value["session_id"], value["sequence"], value["capture_time_s"])
    if marker == "array":
        try:
            dtype = np.dtype(value["dtype"])
            shape = tuple(int(item) for item in value["shape"])
            raw = base64.b64decode(value["data"], validate=True)
            array = np.frombuffer(raw, dtype=dtype)
            if array.size != int(np.prod(shape, dtype=np.int64)):
                raise PacketError("array byte count does not match shape")
            return array.reshape(shape).copy()
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise PacketError(f"invalid array: {error}") from error
    if marker == "result":
        try:
            return SmplerXResult(
                frame_id=_decode(value["frame_id"]),
                completed_time_s=value["completed_time_s"],
                model_id=value["model_id"],
                model_hash=value["model_hash"],
                arrays={name: _decode(item) for name, item in value["arrays"].items()},
                valid=value["valid"],
                reason=value.get("reason", ""),
                metadata=value.get("metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PacketError(f"invalid result: {error}") from error
    return {key: _decode(item) for key, item in value.items()}


def encode_packet(message_type: str, payload: dict[str, Any]) -> bytes:
    if not message_type or not isinstance(payload, dict):
        raise PacketError("message type and mapping payload are required")
    body = json.dumps({"version": PROTOCOL_VERSION, "type": message_type,
                       "payload": _encode(payload)}, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    if len(body) > MAX_PACKET_BYTES:
        raise PacketError("packet is too large")
    return struct.pack(">I", len(body)) + body


def decode_packet(packet: bytes) -> dict[str, Any]:
    if len(packet) < 4:
        raise PacketError("packet has no length prefix")
    length = struct.unpack(">I", packet[:4])[0]
    if length > MAX_PACKET_BYTES or len(packet) != length + 4:
        raise PacketError("packet length is invalid")
    try:
        message = json.loads(packet[4:].decode("utf-8"))
        if message.get("version") != PROTOCOL_VERSION:
            raise PacketError("unsupported protocol version")
        if not isinstance(message.get("type"), str) or not isinstance(message.get("payload"), dict):
            raise PacketError("packet type or payload is invalid")
        return {"version": message["version"], "type": message["type"],
                "payload": _decode(message["payload"])}
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError) as error:
        raise PacketError(f"invalid JSON packet: {error}") from error


class PacketReader:
    def __init__(self):
        self._buffer = bytearray()
        self._expected: int | None = None

    def feed(self, data: bytes) -> list[bytes]:
        self._buffer.extend(data)
        packets = []
        while True:
            if self._expected is None:
                if len(self._buffer) < 4:
                    break
                self._expected = struct.unpack(">I", self._buffer[:4])[0]
                if self._expected > MAX_PACKET_BYTES:
                    self._buffer.clear()
                    self._expected = None
                    raise PacketError("packet is too large")
            if len(self._buffer) < self._expected + 4:
                break
            packet = bytes(self._buffer[:self._expected + 4])
            decode_packet(packet)
            packets.append(packet)
            del self._buffer[:self._expected + 4]
            self._expected = None
        return packets


class SmplerXClient:
    """Bounded scheduler around a transport with ``send`` and ``poll`` methods."""

    def __init__(self, transport, *, session_id: str):
        if not session_id:
            raise ValueError("session_id must not be empty")
        self.transport = transport
        self.session_id = session_id
        self._active = False
        self._pending: tuple[BodyFrameId, np.ndarray] | None = None
        self._last_sequence = -1
        self._active_frame = None

    def submit(self, frame_id: BodyFrameId, rgb: np.ndarray) -> None:
        if frame_id.session_id != self.session_id:
            raise PacketError("frame belongs to another session")
        if frame_id.sequence <= self._last_sequence:
            raise PacketError("frame sequence is not increasing")
        sample = (frame_id, np.asarray(rgb))
        if sample[1].ndim != 3 or sample[1].shape[2] != 3 or sample[1].dtype != np.uint8:
            raise ValueError("rgb must have shape (height,width,3) and dtype uint8")
        self._last_sequence = frame_id.sequence
        if self._active:
            self._pending = sample
            return
        self._send(sample)

    def _send(self, sample):
        frame_id, rgb = sample
        self.transport.send(encode_packet("frame", {"frame_id": frame_id, "rgb": rgb}))
        self._active = True
        self._active_frame = frame_id

    def poll(self) -> SmplerXResult | None:
        value = self.transport.poll()
        if value is None:
            return None
        if isinstance(value, bytes):
            message = decode_packet(value)
            if message["type"] != "result":
                raise PacketError("worker message is not a result")
            value = message["payload"].get("result")
        if not isinstance(value, SmplerXResult):
            raise PacketError("worker result has an invalid type")
        if value.frame_id.session_id != self.session_id:
            raise PacketError("worker result belongs to an old session")
        if value.frame_id != self._active_frame:
            raise PacketError('worker result does not match the active frame')
        self._active = False
        self._active_frame = None
        if self._pending is not None:
            pending, self._pending = self._pending, None
            self._send(pending)
        return value

    def close(self):
        self._pending = None
        self._active = False
        self._active_frame = None
        close = getattr(self.transport, 'close', None)
        if close is not None:
            close()
