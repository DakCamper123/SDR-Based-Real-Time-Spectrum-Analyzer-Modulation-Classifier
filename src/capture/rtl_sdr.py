"""RTL-SDR Hardware receiver interface with automatic fallback to synthetic IQ generator."""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class RTLSDRReceiver:
    """Wrapper around pyrtlsdr for capturing live RF signals."""

    def __init__(
        self,
        center_freq: float = 100e6,  # 100 MHz default (FM band)
        sample_rate: float = 2.4e6,  # 2.4 MSPS
        gain: str | float = "auto",
        fallback_synthetic: bool = True,
    ):
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.fallback_synthetic = fallback_synthetic
        self.sdr = None
        self.is_hardware_connected = False
        self._init_hardware()

    def _init_hardware(self) -> None:
        """Attempt to initialize the RTL-SDR dongle."""
        try:
            from rtlsdr import RtlSdr

            self.sdr = RtlSdr()
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.center_freq
            self.sdr.gain = self.gain
            self.is_hardware_connected = True
            logger.info("RTL-SDR hardware successfully initialized.")
        except Exception as e:
            self.is_hardware_connected = False
            self.sdr = None
            if self.fallback_synthetic:
                logger.warning(
                    "RTL-SDR device not detected or librtlsdr missing (%s). Falling back to synthetic generator mode.",
                    e,
                )
            else:
                raise RuntimeError("Failed to connect to RTL-SDR device.") from e

    def read_samples(self, num_samples: int = 2048) -> np.ndarray:
        """Read complex IQ samples from hardware or fallback generator."""
        if self.is_hardware_connected and self.sdr is not None:
            samples = self.sdr.read_samples(num_samples)
            return np.array(samples, dtype=np.complex64)

        # Fallback mode
        from .generator import SyntheticIQGenerator

        gen = SyntheticIQGenerator(sample_rate=self.sample_rate)
        return gen.generate("FM", num_samples=num_samples, snr_db=18.0)

    def set_center_freq(self, freq_hz: float) -> None:
        """Retune center frequency."""
        self.center_freq = freq_hz
        if self.is_hardware_connected and self.sdr:
            self.sdr.center_freq = freq_hz

    def close(self) -> None:
        """Release RTL-SDR hardware resources."""
        if self.sdr:
            try:
                self.sdr.close()
            except Exception:
                pass
            self.sdr = None
            self.is_hardware_connected = False
