# 类 InputSource[next_observation/close]（封装数据/流程）；链路：相机/回放→原始观测→应用/重定向
"""Sensor-only interface: no-new-sample is None, EOF is StopIteration."""
from typing import Protocol
from teleoperation.contracts.observations import RawObservation


class InputSource(Protocol):
    def next_observation(self) -> RawObservation | None: ...
    def close(self) -> None: ...
