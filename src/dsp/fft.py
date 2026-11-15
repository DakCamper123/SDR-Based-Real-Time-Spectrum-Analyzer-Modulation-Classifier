"""Digital Signal Processing (DSP) utilities: Windowed FFT, PSD, and Waterfall processing."""

from typing import Tuple
import numpy as np


def compute_fft_bins(
    iq_samples: np.ndarray,
    n_fft: int = 1024,
    window_type: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute centered FFT magnitude in dBFS from complex IQ samples.

    Returns:
        (fft_mag_db, phase_rad)
    """
    if len(iq_samples) < n_fft:
        # Zero-pad if needed
        iq_samples = np.pad(iq_samples, (0, n_fft - len(iq_samples)))
    else:
        iq_samples = iq_samples[:n_fft]

    # Select window
    if window_type == "hann":
        win = np.hanning(n_fft)
    elif window_type == "hamming":
        win = np.hamming(n_fft)
    elif window_type == "blackman":
        win = np.blackman(n_fft)
    else:
        win = np.ones(n_fft)

    windowed = iq_samples * win
    fft_val = np.fft.fftshift(np.fft.fft(windowed, n=n_fft))
    mag = np.abs(fft_val) + 1e-12  # avoid log(0)
    mag_db = 20 * np.log10(mag / n_fft)
    phase = np.angle(fft_val)
    return mag_db.astype(np.float32), phase.astype(np.float32)


def compute_psd(
    iq_samples: np.ndarray,
    sample_rate: float,
    center_freq: float,
    n_fft: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density (PSD) and corresponding absolute frequency axis.

    Returns:
        (frequencies_hz, psd_dbm)
    """
    mag_db, _ = compute_fft_bins(iq_samples, n_fft=n_fft)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sample_rate)) + center_freq
    return freqs, mag_db


def create_waterfall_row(
    iq_samples: np.ndarray,
    n_fft: int = 512,
    min_db: float = -100.0,
    max_db: float = -10.0,
) -> np.ndarray:
    """Normalize FFT bins to [0, 255] uint8 row suitable for WebSocket/Canvas waterfall streaming."""
    mag_db, _ = compute_fft_bins(iq_samples, n_fft=n_fft)
    # Clip and normalize between min_db and max_db
    clipped = np.clip(mag_db, min_db, max_db)
    normalized = (clipped - min_db) / (max_db - min_db) * 255.0
    return normalized.astype(np.uint8)
