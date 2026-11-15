# SDR-Based Real-Time Spectrum Analyzer & Modulation Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end software-defined radio (SDR) platform combining real-time radio frequency (RF) signal processing, deep learning automatic modulation classification (AMC), and an interactive browser-based waterfall spectrum analyzer.

---

## 🎯 Core Concept
- **Signal Acquisition**: Captures live RF baseband IQ samples (FM broadcast, ADS-B aviation telemetry, LoRa, ISM bands) via **RTL-SDR** (with automatic fallback to a high-fidelity synthetic IQ generator for offline testing).
- **DSP Filtering Pipeline**: Computes windowed Fast Fourier Transforms (FFT), Power Spectral Density (PSD), Welch estimation, and waterfall bin generation in real time.
- **Deep Learning AMC**: Employs a 1D Convolutional Neural Network (CNN) operating directly on raw $(I, Q)$ baseband tensors to classify analog and digital modulations (**AM, FM, BPSK, QPSK, 16-QAM**).
- **Live Visualization**: Streams high-framerate spectrum and waterfall spectrograms to an interactive web dashboard over WebSockets at ~30 FPS.

---

## 🏗️ Architecture

```
                                  ┌───────────────────────────┐
                                  │   RTL-SDR USB Hardware    │
                                  └─────────────┬─────────────┘
                                                │ (or Synthetic Fallback)
                                                ▼
                                  ┌───────────────────────────┐
                                  │   Baseband IQ Receiver    │
                                  └─────────────┬─────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │        DSP Pipeline         │                   │   CNN Modulation Engine     │
        │ • Windowing (Hann/Hamming)  │                   │ • Raw IQ (2, N) Tensors     │
        │ • Centered FFT & PSD        │                   │ • Conv1D Feature Extraction │
        │ • Waterfall Row Scaling     │                   │ • Softmax Classification    │
        └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │     FastAPI WebSocket     │
                                  │          Streamer         │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │    Interactive Web UI     │
                                  │  PSD + Waterfall + Labels │
                                  └───────────────────────────┘
```

---

## 📁 Repository Structure

```
├── src/
│   ├── capture/             # Hardware interface & synthetic test signals
│   │   ├── rtl_sdr.py       # pyrtlsdr receiver with auto-fallback
│   │   └── generator.py     # Synthetic AM, FM, BPSK, QPSK, 16-QAM generator
│   ├── dsp/                 # Signal processing core
│   │   └── fft.py           # Windowed FFT, PSD, and waterfall row scaling
│   ├── classifier/          # Deep learning modulation recognition
│   │   └── model.py         # 1D-CNN modulation classifier on raw IQ
│   └── server/              # Backend server
│       └── app.py           # FastAPI WebSocket streaming server
├── web/
│   └── index.html           # Real-time spectrum & waterfall browser viewer
├── tests/
│   └── test_pipeline.py     # Unit tests for capture, DSP, and ML
├── data/                    # Datasets and sample IQ files (.dat / .npy)
├── models/                  # Saved weights and checkpoints (.pt / .onnx)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## ⚡ Quickstart

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/DakCamper123/SDR-Based-Real-Time-Spectrum-Analyzer-Modulation-Classifier.git
cd "SDR-Based-Real-Time-Spectrum-Analyzer-Modulation-Classifier"

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Streaming Server

```bash
uvicorn src.server.app:app --reload --port 8000
```

> **Note**: If an RTL-SDR dongle is detected, the server automatically captures live over-the-air signals. If not plugged in, it automatically runs in **Synthetic Fallback Mode**, generating simulated RF modulations for seamless development.

### 3. Open the Spectrum Viewer

Open `web/index.html` in your web browser, or navigate to:
```
http://localhost:8000/docs
```
for the interactive FastAPI Swagger API documentation.

---

## 🧪 Running Tests

Verify the DSP and ML pipeline components:

```bash
pytest tests/
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
