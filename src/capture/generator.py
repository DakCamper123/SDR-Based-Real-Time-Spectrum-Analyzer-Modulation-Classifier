"""Synthetic IQ signal generator for offline testing and ML dataset generation.

Supports AM, FM, BPSK, QPSK, and 16-QAM modulations with AWGN channel simulation.
"""

from typing import Tuple
import numpy as np


class SyntheticIQGenerator:
    """Generates synthetic modulated baseband IQ samples for testing and training."""

    SUPPORTED_MODULATIONS = ["AM", "FM", "BPSK", "QPSK", "16QAM"]

    def __init__(self, sample_rate: float = 2.4e6):
        """Initialize generator with sample rate in Hz (default: 2.4 MHz standard RTL-SDR rate)."""
        self.sample_rate = sample_rate

    def _add_awgn(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add Additive White Gaussian Noise (AWGN) to complex signal for a target SNR."""
        sig_power = np.mean(np.abs(signal) ** 2)
        if sig_power == 0:
            return signal
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise_std = np.sqrt(noise_power / 2.0)
        noise = np.random.normal(0, noise_std, size=signal.shape) + 1j * np.random.normal(
            0, noise_std, size=signal.shape
        )
        return signal + noise

    def generate_am(self, num_samples: int = 2048, mod_freq: float = 5e3, mod_index: float = 0.8) -> np.ndarray:
        """Generate DSB-AM modulated complex baseband signal."""
        t = np.arange(num_samples) / self.sample_rate
        audio = np.cos(2 * np.pi * mod_freq * t)
        baseband = (1.0 + mod_index * audio) + 0j
        return baseband.astype(np.complex64)

    def generate_fm(self, num_samples: int = 2048, mod_freq: float = 5e3, freq_dev: float = 25e3) -> np.ndarray:
        """Generate Wideband/Narrowband FM modulated complex baseband signal."""
        t = np.arange(num_samples) / self.sample_rate
        audio = np.cos(2 * np.pi * mod_freq * t)
        phase = 2 * np.pi * freq_dev * np.cumsum(audio) / self.sample_rate
        return np.exp(1j * phase).astype(np.complex64)

    def generate_bpsk(self, num_samples: int = 2048, symbol_rate: float = 100e3) -> np.ndarray:
        """Generate BPSK modulated complex baseband signal."""
        samples_per_symbol = max(1, int(self.sample_rate / symbol_rate))
        num_symbols = int(np.ceil(num_samples / samples_per_symbol))
        bits = np.random.randint(0, 2, num_symbols)
        symbols = 2 * bits - 1  # Map 0 -> -1, 1 -> +1
        iq = np.repeat(symbols, samples_per_symbol)[:num_samples] + 0j
        return iq.astype(np.complex64)

    def generate_qpsk(self, num_samples: int = 2048, symbol_rate: float = 100e3) -> np.ndarray:
        """Generate QPSK modulated complex baseband signal."""
        samples_per_symbol = max(1, int(self.sample_rate / symbol_rate))
        num_symbols = int(np.ceil(num_samples / samples_per_symbol))
        bits_i = 2 * np.random.randint(0, 2, num_symbols) - 1
        bits_q = 2 * np.random.randint(0, 2, num_symbols) - 1
        symbols = (bits_i + 1j * bits_q) / np.sqrt(2)
        iq = np.repeat(symbols, samples_per_symbol)[:num_samples]
        return iq.astype(np.complex64)

    def generate_16qam(self, num_samples: int = 2048, symbol_rate: float = 100e3) -> np.ndarray:
        """Generate 16-QAM modulated complex baseband signal."""
        samples_per_symbol = max(1, int(self.sample_rate / symbol_rate))
        num_symbols = int(np.ceil(num_samples / samples_per_symbol))
        constellation = np.array([-3, -1, 1, 3])
        sym_i = np.random.choice(constellation, num_symbols)
        sym_q = np.random.choice(constellation, num_symbols)
        symbols = (sym_i + 1j * sym_q) / np.sqrt(10)  # Normalized power
        iq = np.repeat(symbols, samples_per_symbol)[:num_samples]
        return iq.astype(np.complex64)

    def generate(
        self,
        modulation: str,
        num_samples: int = 2048,
        snr_db: float = 20.0,
        freq_offset: float = 0.0,
    ) -> np.ndarray:
        """Generate specified modulation with optional frequency offset and AWGN."""
        mod = modulation.upper()
        if mod == "AM":
            iq = self.generate_am(num_samples)
        elif mod == "FM":
            iq = self.generate_fm(num_samples)
        elif mod == "BPSK":
            iq = self.generate_bpsk(num_samples)
        elif mod in ("QPSK", "4QAM"):
            iq = self.generate_qpsk(num_samples)
        elif mod in ("16QAM", "QAM16", "QAM"):
            iq = self.generate_16qam(num_samples)
        else:
            raise ValueError(f"Unsupported modulation '{modulation}'. Choose from {self.SUPPORTED_MODULATIONS}")

        if freq_offset != 0.0:
            t = np.arange(num_samples) / self.sample_rate
            iq = iq * np.exp(1j * 2 * np.pi * freq_offset * t)

        return self._add_awgn(iq, snr_db)
