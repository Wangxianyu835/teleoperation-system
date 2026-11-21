"""Standalone Linker FFG acquisition package."""

from .api import FFGSample, FFGSampleSource, MotionSample, SampleSource
from .frames import EXPECTED_FFG_CHANNELS, FFGChannelLayout

__all__ = [
    "EXPECTED_FFG_CHANNELS",
    "FFGChannelLayout",
    "FFGSample",
    "FFGSampleSource",
    "MotionSample",
    "SampleSource",
]

__version__ = "0.1.0"
