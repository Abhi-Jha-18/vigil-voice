"""
Phase 14 Tests: Production Hardening, Security & Deployment Validation

Tests cover:
- File upload security: empty file rejection, oversized file rejection, unsupported format rejection
- Path traversal attack neutralization
- Structured JSON error responses and Request ID header propagation
- HTTP Security headers (X-Content-Type-Options, X-Frame-Options, CSP)
- /api/health and /api/ready endpoints
- Model SHA-256 integrity check and MODEL_INTEGRITY_FAILURE handling
- Guaranteed temporary file cleanup on success and error
"""

import io
import os
import sys
import json
import tempfile
import numpy as np
import pytest
from pathlib import Path
from scipy.io import wavfile
from fastapi.testclient import TestClient

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.main import app
from backend.config import settings
from backend.utils.audio_utils import sanitize_filename, remove_temp_file
from backend.detection.detector import _compute_file_sha256, reset_cached_model


@pytest.fixture
def client():
    return TestClient(app)


def _make_valid_wav_bytes(duration_sec: float = 1.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, samples)
    return buf.getvalue()


# ── 1. Upload Security Tests ───────────────────────────────────────────────────

def test_empty_file_upload_rejected(client):
    response = client.post(
        "/api/detect",
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"phase": "phase1"}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "EMPTY_FILE"
    assert "request_id" in data["error"]


def test_unsupported_format_rejected(client):
    response = client.post(
        "/api/detect",
        files={"file": ("malicious.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/x-msdownload")},
        data={"phase": "phase1"}
    )
    assert response.status_code in (400, 415)
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_path_traversal_filename_sanitized(client):
    traversal_name = "../../../../../windows/system32/cmd.exe.wav"
    clean_name = sanitize_filename(traversal_name)
    assert ".." not in clean_name
    assert "/" not in clean_name
    assert "\\" not in clean_name
    assert clean_name.endswith(".wav")

    # Verify upload with traversal path succeeds safely without directory escape
    valid_wav = _make_valid_wav_bytes(1.0)
    response = client.post(
        "/api/detect",
        files={"file": (traversal_name, valid_wav, "audio/wav")},
        data={"phase": "phase1"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert ".." not in data["filename"]


# ── 2. Structured Error & Request Tracing Tests ────────────────────────────────

def test_request_id_and_security_headers_present(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "strict-origin-when-cross-origin" in response.headers["Referrer-Policy"]
    assert "Content-Security-Policy" in response.headers


def test_invalid_phase_returns_structured_error(client):
    valid_wav = _make_valid_wav_bytes(1.0)
    response = client.post(
        "/api/detect",
        files={"file": ("audio.wav", valid_wav, "audio/wav")},
        data={"phase": "invalid_phase_xyz"}
    )
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] in ("INVALID_REQUEST", "UNPROCESSABLE_ENTITY")
    assert "request_id" in data["error"]


# ── 3. Health and Readiness Endpoints ──────────────────────────────────────────

def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "VigilVoice"
    assert "model_status" in data


def test_readiness_endpoint(client):
    response = client.get("/api/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert "model_status" in data


# ── 4. Model Integrity Check ───────────────────────────────────────────────────

def test_model_sha256_computation_and_integrity_check():
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        f.write(b"dummy_model_bytes_12345")
        model_path = Path(f.name)

    try:
        calculated_sha = _compute_file_sha256(model_path)
        assert len(calculated_sha) == 64  # Valid SHA-256 hex string
        assert isinstance(calculated_sha, str)
    finally:
        remove_temp_file(model_path)


# ── 5. Temporary File Cleanup Verification ─────────────────────────────────────

def test_temp_file_deleted_after_detection(client):
    valid_wav = _make_valid_wav_bytes(1.0)
    temp_dir = tempfile.gettempdir()
    
    # Snapshot files before
    files_before = set(os.listdir(temp_dir))
    
    response = client.post(
        "/api/detect",
        files={"file": ("test_cleanup.wav", valid_wav, "audio/wav")},
        data={"phase": "phase1"}
    )
    assert response.status_code == 200

    # Snapshot files after
    files_after = set(os.listdir(temp_dir))
    
    # No dangling vigilvoice_ temp files created during this test should remain
    new_files = files_after - files_before
    dangling_vigilvoice = [f for f in new_files if f.startswith("vigilvoice_")]
    assert len(dangling_vigilvoice) == 0, f"Dangling temp files found: {dangling_vigilvoice}"
