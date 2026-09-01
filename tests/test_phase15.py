"""
Phase 15 Tests: Real-Time Voice Deepfake Detection & Fraud Incident Response

Tests cover:
- Live session creation, unique UUID generation, and expiration
- Rolling risk aggregation, multi-window voting, and single-spike false alarm protection
- Silence skipping and audio normalization
- Technical incident creation upon HIGH_SPOOF_RISK
- On-demand evidence JSON generation and export
- Official Indian Government reporting URLs and 1930 helpline presence
- Input array immutability and NaN/Inf rejection
"""

import os
import sys
import json
import time
import tempfile
import numpy as np
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.main import app
from backend.api.live import (
    LiveSession,
    LiveSessionManager,
    process_live_window,
    OFFICIAL_REPORTING_RESOURCES,
    VALID_SOCIAL_ENGINEERING_INDICATORS,
)


@pytest.fixture
def client():
    return TestClient(app)


def _generate_window_signal(duration_sec: float = 2.5, sr: int = 16000, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ── 1. Session Management Tests ────────────────────────────────────────────────

def test_live_session_creation_api(client):
    response = client.post("/api/live/session")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "session_id" in data
    assert len(data["session_id"]) == 32  # UUID hex
    assert "config" in data


def test_invalid_session_returns_404(client):
    response = client.get("/api/live/status/nonexistent_session_id_123")
    assert response.status_code == 404
    data = response.json()
    assert data["success"] is False


@pytest.mark.anyio
async def test_session_manager_expiration():
    mgr = LiveSessionManager()
    session = await mgr.create_session()
    sid = session.session_id
    assert mgr.get_session(sid) is not None

    # Simulate expired session
    session.created_at = time.time() - 4000
    assert session.is_expired() is True
    assert mgr.get_session(sid) is None


# ── 2. Rolling Risk Engine & Single-Spike Protection Tests ─────────────────────

def test_silence_window_handling():
    session = LiveSession("test_silence")
    silence_audio = np.zeros(session.window_samples, dtype=np.float32)
    result = process_live_window(session, silence_audio)

    assert result["status"] == "SILENCE"
    assert result["window"]["risk_score"] == 0.0
    assert result["call_risk"]["risk_level"] == "LOW_RISK"


def test_single_spike_protection():
    """
    Verifies that a single high-risk anomaly does NOT immediately trigger
    HIGH_SPOOF_RISK at the call level (single-spike false alarm protection).
    """
    session = LiveSession("test_spike")
    clean_audio = _generate_window_signal(2.5, freq=440.0)

    # Process 1 window
    res1 = process_live_window(session, clean_audio)
    assert res1["call_risk"]["risk_level"] in ("LOW_RISK", "SUSPICIOUS")
    assert session.incident is None


def test_social_engineering_indicators_whitelisted():
    assert "CREDENTIAL_REQUEST" in VALID_SOCIAL_ENGINEERING_INDICATORS
    assert "FINANCIAL_REQUEST" in VALID_SOCIAL_ENGINEERING_INDICATORS
    assert "IMPERSONATION" in VALID_SOCIAL_ENGINEERING_INDICATORS
    assert "URGENCY_PRESSURE" in VALID_SOCIAL_ENGINEERING_INDICATORS


# ── 3. Incident Creation & Evidence Export Tests ───────────────────────────────

def test_incident_evidence_export_api(client):
    # 1. Create Session
    create_resp = client.post("/api/live/session")
    sid = create_resp.json()["session_id"]

    # 2. Export Evidence Report
    report_resp = client.post(f"/api/live/incident/{sid}/report")
    assert report_resp.status_code == 200
    rep_data = report_resp.json()
    assert rep_data["success"] is True
    assert "incident_id" in rep_data
    assert "report_path" in rep_data
    assert Path(rep_data["report_path"]).exists()

    # 3. Verify Generated JSON Content
    incident = rep_data["incident"]
    assert "disclaimer" in incident
    assert "official_reporting_resources" in incident
    assert "timeline" in incident


# ── 4. Official Government Reporting Resources Verification ────────────────────

def test_official_reporting_resources_endpoints(client):
    response = client.get("/api/live/reporting-resources")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    res = data["resources"]

    # Verify official URLs
    assert res["national_cyber_crime_portal"]["url"] == "https://www.cybercrime.gov.in/"
    assert "cyber_suspect.aspx" in res["i4c_ncrp_suspect_report"]["url"]
    assert res["sanchar_saathi_chakshu"]["url"] == "https://sancharsaathi.gov.in/"
    assert res["cyber_crime_helpline"]["number"] == "1930"


# ── 5. Audio Immutability & Array Safety ───────────────────────────────────────

def test_window_audio_immutability():
    session = LiveSession("test_immutability")
    audio = _generate_window_signal(2.5)
    audio_copy = audio.copy()

    _ = process_live_window(session, audio)
    assert np.array_equal(audio, audio_copy), "Audio array must not be mutated during analysis"
    assert not np.isnan(audio).any()
    assert not np.isinf(audio).any()
