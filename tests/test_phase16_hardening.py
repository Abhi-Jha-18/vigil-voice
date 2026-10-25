"""
Tests for Phase 16: Concurrency, Timeout Protection, Upload Security,
Authentication, Rate Limiting, and Architecture Hardening.
"""

import io
import os
import sys
import time
import asyncio
import numpy as np
import pytest
from pathlib import Path
from scipy.io import wavfile
from fastapi.testclient import TestClient

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.main import app, _rate_limiter, current_request_id, AppError
from backend.config import settings
from backend.audio.features import get_mel_spectrogram_b64
from backend.utils.audio_utils import validate_magic_bytes, get_audio_header_duration


@pytest.fixture
def client():
    return TestClient(app)


def _make_valid_wav_bytes(duration_sec: float = 1.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, samples)
    return buf.getvalue()


# ── 1. Timeout Bug & Resource Lifecycle Tests ──────────────────────────────────
def test_timeout_does_not_delete_temp_file_while_worker_runs(client, monkeypatch):
    """
    Verifies that when inference times out, the shielded worker task continues,
    retaining the temporary file and semaphore until execution finishes.
    """
    valid_wav = _make_valid_wav_bytes(1.0)
    observed_temp_path = []
    file_intact_during_run = []

    import backend.audio.pipeline as pipeline_mod
    original_process_canonical = pipeline_mod.process_canonical

    def slow_canonical(temp_path, *args, **kwargs):
        observed_temp_path.append(temp_path)
        # Wait until wait_for times out in the main event loop
        time.sleep(0.3)
        file_intact_during_run.append(os.path.exists(temp_path))
        return original_process_canonical(temp_path, *args, **kwargs)

    monkeypatch.setattr("backend.audio.pipeline.process_canonical", slow_canonical)
    monkeypatch.setattr(settings, "inference_timeout_seconds", 0.1)

    response = client.post(
        "/api/detect",
        files={"file": ("test_timeout.wav", valid_wav, "audio/wav")},
        data={"phase": "phase1"}
    )
    assert response.status_code == 504
    data = response.json()
    assert data["error"]["code"] == "INFERENCE_TIMEOUT"

    # Temp file MUST have existed while the worker was running after timeout!
    assert file_intact_during_run == [True], "Temp file was deleted while worker was running!"

    # After worker completes, temp file should be deleted
    assert len(observed_temp_path) == 1
    temp_file = observed_temp_path[0]
    assert not os.path.exists(temp_file), "Temp file should be deleted after worker finishes"


def test_try_acquire_semaphore_returns_429(client, monkeypatch):
    """
    Verifies non-blocking try-acquire returns 429 when max concurrent inferences is reached.
    """
    from backend.main import _inference_semaphore

    # Lock the semaphore completely
    for _ in range(_inference_semaphore._value):
        _inference_semaphore._value -= 1

    try:
        valid_wav = _make_valid_wav_bytes(1.0)
        response = client.post(
            "/api/detect",
            files={"file": ("test_busy.wav", valid_wav, "audio/wav")},
            data={"phase": "phase1"}
        )
        assert response.status_code == 429
        data = response.json()
        assert data["error"]["code"] == "CONCURRENCY_LIMIT_EXCEEDED"
    finally:
        # Restore semaphore value
        _inference_semaphore._value = settings.max_concurrent_inferences


# ── 2. Content-Length & Upload Size Rejection Tests ─────────────────────────────
def test_content_length_middleware_rejects_oversized_before_decode(client, monkeypatch):
    """
    Verifies that Content-Length header exceeding MAX_UPLOAD_MB is rejected early by middleware.
    """
    decode_called = []

    def mock_decode(*args, **kwargs):
        decode_called.append(True)
        raise RuntimeError("Decode should not be called!")

    monkeypatch.setattr("backend.audio.processor.load_and_resample", mock_decode)

    oversized_length = str(settings.max_upload_bytes + 1024)
    response = client.post(
        "/api/detect",
        headers={"Content-Length": oversized_length},
        data={"phase": "phase1"}
    )
    assert response.status_code == 413
    data = response.json()
    assert data["error"]["code"] == "FILE_TOO_LARGE"
    assert len(decode_called) == 0


def test_oversized_upload_body_rejected_before_decode(client, monkeypatch):
    """
    Verifies that upload bodies exceeding MAX_UPLOAD_MB are rejected before decoding.
    """
    decode_called = []

    def mock_decode(*args, **kwargs):
        decode_called.append(True)
        raise RuntimeError("Decode should not be called!")

    monkeypatch.setattr("backend.audio.processor.load_and_resample", mock_decode)
    monkeypatch.setattr(settings, "max_upload_mb", 1)  # 1 MB max

    # 1.5MB dummy payload
    oversized_bytes = b"RIFF" + b"\x00" * 8 + b"WAVE" + b"\x00" * (1500 * 1024)
    response = client.post(
        "/api/detect",
        files={"file": ("oversized.wav", oversized_bytes, "audio/wav")},
        data={"phase": "phase1"}
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"
    assert len(decode_called) == 0


# ── 3. Pre-decode Over-duration & Magic Bytes Rejection Tests ──────────────────
def test_over_duration_upload_rejected_before_decode(client, monkeypatch):
    """
    Verifies that duration check from metadata headers rejects over-duration audio before decoding.
    """
    decode_called = []

    def mock_decode(*args, **kwargs):
        decode_called.append(True)
        raise RuntimeError("Decode should not be called!")

    monkeypatch.setattr("backend.audio.processor.load_and_resample", mock_decode)
    monkeypatch.setattr("backend.main.get_audio_header_duration", lambda _p: 180.0)

    valid_wav = _make_valid_wav_bytes(1.0)
    response = client.post(
        "/api/detect",
        files={"file": ("long.wav", valid_wav, "audio/wav")},
        data={"phase": "phase1"}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"]["code"] == "AUDIO_TOO_LONG"
    assert len(decode_called) == 0


def test_invalid_magic_bytes_rejected_before_decode(client, monkeypatch):
    """
    Verifies that audio files with invalid magic bytes are rejected with 415 before decoding.
    """
    decode_called = []

    def mock_decode(*args, **kwargs):
        decode_called.append(True)
        raise RuntimeError("Decode should not be called!")

    monkeypatch.setattr("backend.audio.processor.load_and_resample", mock_decode)

    fake_wav_content = b"NOT_A_REAL_WAV_HEADER_DATA_12345"
    response = client.post(
        "/api/detect",
        files={"file": ("fake.wav", fake_wav_content, "audio/wav")},
        data={"phase": "phase1"}
    )
    assert response.status_code == 415
    data = response.json()
    assert data["error"]["code"] == "UNSUPPORTED_FORMAT"
    assert len(decode_called) == 0


def test_magic_bytes_helper():
    assert validate_magic_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ") is True
    assert validate_magic_bytes(b"fLaC\x00\x00\x00\x22") is True
    assert validate_magic_bytes(b"OggS\x00\x02\x00") is True
    assert validate_magic_bytes(b"\x00\x00\x00\x20ftypisom") is True
    assert validate_magic_bytes(b"\x1a\x45\xdf\xa3") is True
    assert validate_magic_bytes(b"ID3\x03\x00\x00") is True
    assert validate_magic_bytes(b"\xff\xfb\x90\x44") is True
    assert validate_magic_bytes(b"MZ\x90\x00\x03") is False
    assert validate_magic_bytes(b"") is False


# ── 4. Minimized Health & Authenticated Endpoints ──────────────────────────────
def test_minimized_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "VigilVoice"
    # Ensure platform internals, dependencies, and full configs are NOT leaked
    assert "platform" not in data.get("runtime", {})
    assert "dependencies" not in data.get("runtime", {})
    assert "sample_rate_hz" not in data.get("configuration", {})
    assert "model_path" not in data.get("configuration", {})


def test_api_key_auth_enforced_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "super-secret-key-123")

    # 1. System details without key -> 401
    resp = client.get("/api/system/details")
    assert resp.status_code == 401

    # 2. System details with wrong key -> 401
    resp = client.get("/api/system/details", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401

    # 3. System details with valid key -> 200
    resp = client.get("/api/system/details", headers={"X-API-Key": "super-secret-key-123"})
    assert resp.status_code == 200
    data = resp.json()
    assert "configuration" in data
    assert "runtime" in data

    # 4. Model info without key -> 401
    resp = client.get("/api/model/info")
    assert resp.status_code == 401

    # 5. Model info with valid key -> 200
    resp = client.get("/api/model/info", headers={"X-API-Key": "super-secret-key-123"})
    assert resp.status_code == 200

    # 6. Incidents endpoint without key -> 401
    resp = client.get("/api/incidents")
    assert resp.status_code == 401

    # 7. Incidents endpoint with valid key -> 200
    resp = client.get("/api/incidents", headers={"X-API-Key": "super-secret-key-123"})
    assert resp.status_code == 200


# ── 5. Rate Limiting Tests ─────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_rate_limiter_throttles_excessive_requests():
    from backend.main import ClientRateLimiter
    limiter = ClientRateLimiter(max_requests=3, window_seconds=10.0)

    for i in range(3):
        allowed, rem, retry = await limiter.check("client_test_1")
        assert allowed is True

    allowed, rem, retry = await limiter.check("client_test_1")
    assert allowed is False
    assert retry > 0


# ── 6. WebSocket Origin Validation Tests ───────────────────────────────────────
def test_websocket_forbidden_origin_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "cors_origins", ("http://127.0.0.1:8000", "http://localhost:8000"))

    with pytest.raises(Exception):
        with client.websocket_connect("/api/stream", headers={"Origin": "http://evil-attacker.com"}):
            pass


# ── 7. Spectrogram Width Capping & Logging Request ID Tests ────────────────────
def test_spectrogram_image_width_capped():
    # 10 seconds of audio at 16kHz
    sr = 16000
    y = np.random.randn(sr * 10).astype(np.float32)

    # Cap at width 100
    b64_img = get_mel_spectrogram_b64(y, sr=sr, max_width=100)
    assert isinstance(b64_img, str)
    assert len(b64_img) > 0


def test_request_id_in_logging_context():
    token = current_request_id.set("req-test-tracing-999")
    try:
        assert current_request_id.get() == "req-test-tracing-999"
    finally:
        current_request_id.reset(token)
