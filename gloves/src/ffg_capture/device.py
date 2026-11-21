"""Linker FFG serial connection and device-neutral FFG source."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .api import FFGSample
from .config import DeviceConfig, project_path
from .console import CallbackLogger
from .frames import EXPECTED_FFG_CHANNELS


class FFGConnectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SerialPortInfo:
    device: str
    description: str = ""
    hwid: str = ""
    manufacturer: str = ""
    product: str = ""
    serial_number: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def discover_serial_ports() -> list[SerialPortInfo]:
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise FFGConnectionError(
            "缺少 pyserial。请在 PyCharm 终端运行: pip install -r requirements.txt"
        ) from exc

    ports = []
    for item in list_ports.comports():
        ports.append(
            SerialPortInfo(
                device=item.device,
                description=item.description or "",
                hwid=item.hwid or "",
                manufacturer=item.manufacturer or "",
                product=item.product or "",
                serial_number=item.serial_number or "",
            )
        )

    def priority(item: SerialPortInfo) -> tuple[int, str]:
        text = " ".join(
            (item.device, item.description, item.manufacturer, item.product, item.hwid)
        ).lower()
        usb_words = ("usb", "serial", "ch340", "cp210", "ftdi", "silicon", "uart")
        return (0 if any(word in text for word in usb_words) else 1, item.device)

    return sorted(ports, key=priority)


def _vendor_module():
    try:
        from .transport.vendor import linkerforce
    except ImportError as exc:
        raise FFGConnectionError(
            "无法加载 FFG 串口驱动。请先安装 pyserial: pip install -r requirements.txt"
        ) from exc
    linkerforce.LINKERFORCE_ABNORMAL_LOG_FILE_PATH = project_path(
        "outputs/logs/linkerforce_abnormal.log"
    )
    linkerforce.LINKERFORCE_PACKET_LOG_DIR = project_path(
        "outputs/logs/linkerforce_packets"
    )
    return linkerforce


class FFGDevice:
    """Connect to one Linker FFG without starting robot-hand retargeting."""

    def __init__(
        self,
        config: DeviceConfig,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.logger = logger
        self._callback_logger = CallbackLogger(logger)
        self._reader = None
        self._port: str | None = None
        self._baudrate: int | None = None
        self._actual_side = "unknown"
        self._version: str | None = None
        self._product_version: str | None = None
        self._sequence = 0
        self._last_frame_count = 0
        self._latest: FFGSample | None = None
        self._lock = threading.Lock()
        self._started = False

    @property
    def is_running(self) -> bool:
        return bool(
            self._started
            and self._reader is not None
            and getattr(self._reader, "connflag", False)
        )

    @property
    def stream_id(self) -> str:
        return f"ffg_{self._actual_side if self._actual_side != 'unknown' else self.config.side}"

    @property
    def info(self) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "side": self._actual_side,
            "port": self._port,
            "baudrate": self._baudrate,
            "version": self._version,
            "product_version": self._product_version,
            "channels": EXPECTED_FFG_CHANNELS,
            "unit": "rad",
        }

    def _expected_hand_enum(self, side: str | None = None):
        linkerforce = _vendor_module()
        requested = (side or self.config.side).lower()
        if requested == "left":
            return linkerforce.HandType.left
        if requested == "right":
            return linkerforce.HandType.right
        return linkerforce.HandType.left

    def _attempt_baudrate(
        self,
        port: str,
        baudrate: int,
        *,
        expected_side: str | None = None,
    ) -> bool:
        linkerforce = _vendor_module()
        hand_enum = self._expected_hand_enum(expected_side)
        query_loops = max(
            0,
            round(
                (self.config.serial_query_interval_ms / 1000.0)
                / float(linkerforce.READ_INTERVAL)
            ),
        )
        linkerforce.QUERY_INTERVAL = query_loops
        reader = linkerforce.ForceSerialReader(
            hand_enum,
            excludelist=[],
            baudrates=[baudrate],
            isdebug=self.config.serial_debug,
            logger=self._callback_logger,
            log_position_packets=False,
        )
        try:
            if not reader.openserial(port=port, baudrate=baudrate):
                return False
            if not reader.query_version_sync(
                retry_count=4,
                response_wait=0.25,
                log_version_hint=False,
            ):
                reader.stop()
                return False

            detected = str(reader.handtype or "").lower()
            if detected not in {"left", "right"}:
                reader.stop()
                return False
            if self.config.side != "auto" and detected != self.config.side:
                reader.stop()
                self.logger.warning(
                    "串口 %s 检测到 %s 手，但配置要求 %s 手",
                    port,
                    detected,
                    self.config.side,
                )
                return False

            reader.start()
            reader.write_packet(reader.pack_01_data())
            self._reader = reader
            self._port = port
            self._baudrate = baudrate
            self._actual_side = detected
            self._version = reader.version
            self._product_version = getattr(reader, "product_version", None)
            self._last_frame_count = int(reader.position_frame_count)
            self._started = True
            self.logger.info(
                "FFG 已连接: %s @ %d, 手侧=%s, 版本=%s, 产品=%s, 查询周期≈%.0fms",
                port,
                baudrate,
                detected,
                self._version,
                self._product_version,
                self.config.serial_query_interval_ms,
            )
            return True
        except Exception:
            try:
                reader.stop()
            except Exception:
                pass
            raise

    def connect(self, port: str | None = None) -> None:
        if self._started:
            return

        requested_port = (port or self.config.port).strip()
        if requested_port.lower() == "auto":
            discovered = discover_serial_ports()
            candidates = (
                [
                    item
                    for item in discovered
                    if not any(
                        word
                        in f"{item.device} {item.description} {item.manufacturer} {item.product}".lower()
                        for word in ("bluetooth", "蓝牙", "bthenum")
                    )
                ]
                if self.config.skip_bluetooth_ports
                else discovered
            )
            if not candidates and discovered:
                raise FFGConnectionError(
                    "只发现蓝牙串口，已自动跳过。请连接 FFG USB 数据线，"
                    "或在配置中明确填写实际 COM 端口。"
                )
        else:
            candidates = [SerialPortInfo(device=requested_port)]
        if not candidates:
            raise FFGConnectionError(
                "没有发现可用串口。请确认 FFG 已通过数据线连接并开机。"
            )

        failures: list[str] = []
        for candidate in candidates:
            self.logger.info("尝试串口: %s %s", candidate.device, candidate.description)
            for baudrate in self.config.baudrates:
                self.logger.info("  尝试波特率 %d", baudrate)
                try:
                    if self._attempt_baudrate(candidate.device, baudrate):
                        return
                except Exception as exc:
                    failures.append(f"{candidate.device}@{baudrate}: {exc}")
                    self.logger.debug(
                        "串口尝试失败 %s@%s: %s",
                        candidate.device,
                        baudrate,
                        exc,
                    )
        detail = "\n".join(failures[-8:]) if failures else "设备没有返回有效版本帧"
        raise FFGConnectionError(
            "未识别到 Linker FFG。\n"
            f"检查端口: {', '.join(item.device for item in candidates)}\n"
            "确认数据线支持传输、设备已开机、波特率配置正确。\n"
            f"详细失败信息:\n{detail}"
        )

    def read_new(self, timeout_s: float = 0.2) -> FFGSample | None:
        if not self._started or self._reader is None:
            return None

        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() <= deadline:
            reader = self._reader
            frame_count = int(reader.position_frame_count)
            if frame_count > self._last_frame_count:
                positions = np.asarray(reader.poslist, dtype=np.float64).copy()
                if positions.shape != (EXPECTED_FFG_CHANNELS,):
                    time.sleep(0.001)
                    continue
                force = np.asarray(reader.realforcelist, dtype=np.float64).reshape(-1)
                if force.size != 5:
                    force = None
                with self._lock:
                    self._sequence += 1
                    sample = FFGSample(
                        host_time_ns=time.monotonic_ns(),
                        sequence=self._sequence,
                        side=self._actual_side,
                        positions_rad=positions,
                        force_response=force,
                        valid=bool(np.isfinite(positions).all()),
                        status="tracking"
                        if getattr(reader, "connflag", False)
                        else "disconnected",
                        metadata={
                            "device_kind": "linker_ffg",
                            "port": self._port,
                            "baudrate": self._baudrate,
                            "version": self._version,
                            "product_version": self._product_version,
                            "driver_frame_count": frame_count,
                        },
                    )
                    self._latest = sample
                    self._last_frame_count = frame_count
                return sample
            time.sleep(0.001)
        return None

    def latest(self) -> FFGSample | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        reader, self._reader = self._reader, None
        self._started = False
        if reader is not None:
            try:
                reader.stop()
            except Exception as exc:
                self.logger.debug("关闭 FFG 串口时出现异常: %s", exc)


class FFGSource:
    """SampleSource implementation for a real Linker FFG."""

    def __init__(self, config: DeviceConfig, logger: logging.Logger) -> None:
        self.device = FFGDevice(config, logger)
        self.stream_id = f"ffg_{config.side}"

    @property
    def is_running(self) -> bool:
        return self.device.is_running

    def start(self) -> None:
        self.device.connect()
        self.stream_id = self.device.stream_id

    def read(self, timeout_s: float = 0.2) -> FFGSample | None:
        return self.device.read_new(timeout_s)

    def latest(self) -> FFGSample | None:
        return self.device.latest()

    def stop(self) -> None:
        self.device.stop()


class SyntheticSource:
    """Deterministic FFG-shaped samples for offline tests and demonstrations."""

    def __init__(
        self,
        *,
        fps: float = 50.0,
        side: str = "right",
        duration_s: float | None = None,
        still_seconds: float = 0.0,
        seed: int = 7,
    ) -> None:
        self.stream_id = f"ffg_{side}"
        self.side = side
        self.fps = float(fps)
        self.duration_s = duration_s
        self.still_seconds = float(still_seconds)
        self._rng = np.random.default_rng(seed)
        self._sequence = 0
        self._started = False
        self._start_monotonic = 0.0
        self._latest: FFGSample | None = None

    @property
    def is_running(self) -> bool:
        return self._started

    def start(self) -> None:
        self._started = True
        self._start_monotonic = time.monotonic()

    def read(self, timeout_s: float = 0.2) -> FFGSample | None:
        if not self._started:
            return None
        elapsed = time.monotonic() - self._start_monotonic
        if self.duration_s is not None and elapsed >= self.duration_s:
            self._started = False
            return None

        movement_elapsed = max(0.0, elapsed - self.still_seconds)
        phase = 2.0 * np.pi * movement_elapsed * 0.5
        base = (1.0 - np.cos(phase)) * 0.5
        positions = np.empty(EXPECTED_FFG_CHANNELS, dtype=np.float64)
        for index in range(EXPECTED_FFG_CHANNELS):
            finger_gain = 0.55 + 0.08 * (index // 4)
            ripple = 0.04 * np.sin(phase + index * 0.13)
            positions[index] = 0.12 + finger_gain * base + ripple
        positions += self._rng.normal(0.0, 0.0015, size=positions.shape)
        positions = np.clip(positions, 0.0, 2.0 * np.pi)

        self._sequence += 1
        sample = FFGSample(
            host_time_ns=time.monotonic_ns(),
            sequence=self._sequence,
            side=self.side,
            positions_rad=positions,
            force_response=np.zeros(5, dtype=np.float64),
            device_time_ns=None,
            valid=True,
            status="tracking",
            metadata={
                "device_kind": "synthetic_ffg",
                "fps": self.fps,
                "synthetic": True,
            },
        )
        self._latest = sample
        time.sleep(1.0 / self.fps)
        return sample

    def latest(self) -> FFGSample | None:
        return self._latest

    def stop(self) -> None:
        self._started = False
