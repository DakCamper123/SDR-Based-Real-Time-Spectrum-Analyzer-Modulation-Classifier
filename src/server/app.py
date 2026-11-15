"""FastAPI backend server with WebSocket streaming for real-time spectrum & modulation inference."""

import asyncio
import json
import logging
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.capture.rtl_sdr import RTLSDRReceiver
from src.dsp.fft import create_waterfall_row, compute_psd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SDR Spectrum Analyzer & Modulation Classifier API",
    version="0.1.0",
    description="Real-time RF waterfall streaming and CNN-based modulation classification",
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global SDR receiver instance (defaults to synthetic mode if no hardware present)
receiver = RTLSDRReceiver(center_freq=100.0e6, sample_rate=2.4e6)


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "mode": "hardware" if receiver.is_hardware_connected else "synthetic_fallback"}


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    return {
        "center_freq_mhz": receiver.center_freq / 1e6,
        "sample_rate_msps": receiver.sample_rate / 1e6,
        "is_hardware_connected": receiver.is_hardware_connected,
    }


@app.post("/api/config")
def set_config(center_freq_mhz: float) -> Dict[str, Any]:
    receiver.set_center_freq(center_freq_mhz * 1e6)
    return {"status": "updated", "center_freq_mhz": center_freq_mhz}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Streams live waterfall FFT bins and mock/model predictions over WebSocket at ~30 FPS."""
    await websocket.accept()
    logger.info("Client connected to waterfall stream.")
    try:
        while True:
            # Read IQ batch
            samples = receiver.read_samples(num_samples=2048)

            # Compute waterfall row (512 uint8 FFT bins)
            waterfall_bins = create_waterfall_row(samples, n_fft=512)

            payload = {
                "timestamp": asyncio.get_event_loop().time(),
                "center_freq": receiver.center_freq,
                "sample_rate": receiver.sample_rate,
                "waterfall": waterfall_bins.tolist(),
                # Placeholder inference payload (linked when model is loaded)
                "prediction": {
                    "modulation": "FM" if not receiver.is_hardware_connected else "ANALYZING",
                    "confidence": 0.95,
                },
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.033)  # ~30 Hz refresh rate
    except WebSocketDisconnect:
        logger.info("Client disconnected from stream.")
    except Exception as e:
        logger.error("Streaming error: %s", e)
        await websocket.close()
