"""Sensor-only interface: no-new-sample is None, EOF is StopIteration."""
from typing import Protocol
from teleoperation.contracts.observations import RawObservation


class InputSource(Protocol):
    def next_observation(self) -> RawObservation | None: ...
    def close(self) -> None: ...
