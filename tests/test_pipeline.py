"""Unit tests for SDR capture, DSP, and ML pipeline."""

import pytest
import numpy as np
import torch

from src.capture.generator import SyntheticIQGenerator
from src.capture.rtl_sdr import RTLSDRReceiver
from src.dsp.fft import compute_fft_bins, compute_psd, create_waterfall_row
from src.classifier.model import ModulationCNN


def test_synthetic_generator():
    gen = SyntheticIQGenerator(sample_rate=2.4e6)
    for mod in ["AM", "FM", "BPSK", "QPSK", "16QAM"]:
        samples = gen.generate(mod, num_samples=1024, snr_db=15.0)
        assert len(samples) == 1024
        assert samples.dtype == np.complex64


def test_dsp_fft():
    gen = SyntheticIQGenerator(sample_rate=2.4e6)
    samples = gen.generate("FM", num_samples=2048)

    mag_db, phase = compute_fft_bins(samples, n_fft=512)
    assert len(mag_db) == 512
    assert len(phase) == 512
    assert mag_db.dtype == np.float32

    freqs, psd = compute_psd(samples, sample_rate=2.4e6, center_freq=100e6, n_fft=512)
    assert len(freqs) == 512
    assert len(psd) == 512

    waterfall_row = create_waterfall_row(samples, n_fft=512)
    assert len(waterfall_row) == 512
    assert waterfall_row.dtype == np.uint8


def test_receiver_fallback():
    receiver = RTLSDRReceiver(fallback_synthetic=True)
    samples = receiver.read_samples(num_samples=1024)
    assert len(samples) == 1024
    assert samples.dtype == np.complex64


def test_cnn_model():
    model = ModulationCNN(num_classes=5, input_samples=1024)
    model.eval()

    # Synthetic IQ to tensor
    samples = np.random.randn(1024) + 1j * np.random.randn(1024)
    tensor = ModulationCNN.iq_to_tensor(samples, target_len=1024)
    assert tensor.shape == (1, 2, 1024)

    with torch.no_grad():
        logits = model(tensor)
    assert logits.shape == (1, 5)
