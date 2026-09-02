"""
VigilVoice — Real-Time Voice Deepfake Detection & Incident Response (Phase 15)
==============================================================================
Provides real-time streaming audio analysis, rolling spoof-risk aggregation with
single-spike protection, technical incident creation, and official cybercrime reporting links.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from backend.config import PROJECT_ROOT, settings
from backend.audio.processor import normalize_audio
from backend.audio.features import extract_mfcc_for_model
from backend.detection.detector import (
    SimpleCNNDetector,
    get_model_status,
    is_cnn_model_available,
    _load_cnn_model,
<<<<<<< HEAD
    heuristic_score_from_mfcc,
=======
    run_ai_detection,
>>>>>>> 20153ec (fix(detection): refine Phase 1 acoustic heuristic and Phase 2 CNN live inference, add model metadata and evaluation reports)
)

logger = logging.getLogger("vigilvoice.live")
router = APIRouter(prefix="/api/live", tags=["Live Detection & Incident Response"])

# ── Official Indian Government Cybercrime Reporting Resources ──────────────────
OFFICIAL_REPORTING_RESOURCES = {
    "national_cyber_crime_portal": {
        "name": "National Cyber Crime Reporting Portal",
        "url": "https://www.cybercrime.gov.in/",
        "purpose": "Official Government of India portal to report cybercrime incidents and financial fraud."
    },
    "i4c_ncrp_suspect_report": {
        "name": "I4C / NCRP Report Suspect Identifier",
        "url": "https://www.cybercrime.gov.in/Webform/cyber_suspect.aspx",
        "purpose": "Report suspicious identifiers such as phone numbers, emails, bank accounts, or websites."
    },
    "sanchar_saathi_chakshu": {
        "name": "Sanchar Saathi — Chakshu Portal",
        "url": "https://sancharsaathi.gov.in/",
        "purpose": "Report suspected fraudulent communication received via call, SMS, or WhatsApp."
    },
    "cyber_crime_helpline": {
        "number": "1930",
        "description": "For immediate cyber/financial fraud assistance, contact the national cybercrime helpline at 1930."
    }
}

VALID_SOCIAL_ENGINEERING_INDICATORS = {
    "CREDENTIAL_REQUEST",
    "FINANCIAL_REQUEST",
    "PERSONAL_INFORMATION_REQUEST",
    "IMPERSONATION",
    "URGENCY_PRESSURE"
}


# ── Data Models ────────────────────────────────────────────────────────────────

@dataclass
class WindowDetection:
    window_id: int
    start_time: float
    end_time: float
    fake_probability: float
    real_probability: float
    risk_score: float  # 0 to 100
    risk_level: str    # LOW_RISK, SUSPICIOUS, HIGH_SPOOF_RISK
    status: str        # ANALYZED or SILENCE
    confidence: float
    is_suspicious: bool


@dataclass
class LiveIncident:
    incident_id: str
    session_id: str
    created_at: float
    duration_seconds: float
    analyzed_windows: int
    suspicious_windows: int
    average_spoof_score: float
    peak_spoof_score: float
    risk_level: str
    model_status: str
    model_sha256: Optional[str]
    timeline: List[dict]
    social_engineering_indicators: List[str]
    disclaimer: str
    official_reporting_resources: dict

    def to_dict(self) -> dict:
        return asdict(self)


class LiveSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_activity = time.time()
        self.sample_rate = settings.live_sample_rate
        self.window_samples = int(self.sample_rate * settings.live_window_seconds)
        self.hop_samples = int(self.sample_rate * settings.live_hop_seconds)
        
        self.buffer = np.array([], dtype=np.float32)
        self.total_processed_samples = 0
        self.window_counter = 0
        self.recent_windows: List[WindowDetection] = []
        self.all_windows: List[WindowDetection] = []
        
        self.current_risk_level = "LOW_RISK"
        self.current_risk_score = 0.0
        self.incident: Optional[LiveIncident] = None
        self.social_engineering_flags: List[str] = []

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > settings.live_max_session_seconds

    def add_audio_chunk(self, chunk: np.ndarray) -> None:
        self.last_activity = time.time()
        max_samples = int(settings.live_max_buffer_mb * 1024 * 1024 / 4)
        if len(self.buffer) + len(chunk) > max_samples:
            # Drop oldest samples to protect against memory exhaustion
            self.buffer = self.buffer[len(chunk):]
        self.buffer = np.concatenate((self.buffer, chunk))

    def get_ready_window(self) -> Optional[np.ndarray]:
        if len(self.buffer) >= self.window_samples:
            return self.buffer[:self.window_samples]
        return None

    def advance_buffer(self) -> None:
        self.buffer = self.buffer[self.hop_samples:]
        self.total_processed_samples += self.hop_samples


class LiveSessionManager:
    """Manages active real-time calling sessions and memory lifecycle."""
    def __init__(self):
        self._sessions: Dict[str, LiveSession] = {}
        self._lock = asyncio.Lock()

    async def create_session(self) -> LiveSession:
        async with self._lock:
            # Clean expired sessions
            now = time.time()
            expired = [sid for sid, s in self._sessions.items() if s.is_expired() or (now - s.last_activity > 600)]
            for sid in expired:
                del self._sessions[sid]

            session_id = uuid.uuid4().hex
            session = LiveSession(session_id)
            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> Optional[LiveSession]:
        session = self._sessions.get(session_id)
        if session and not session.is_expired():
            return session
        return None

    async def remove_session(self, session_id: str) -> None:
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]


session_manager = LiveSessionManager()


# ── Rolling Risk Aggregator & Window Inference ─────────────────────────────────

def process_live_window(session: LiveSession, window_audio: np.ndarray) -> dict:
    """
    Analyzes a 2.5s audio window and updates rolling call-level risk state.
    Implements single-spike false alarm protection.
    """
    sr = session.sample_rate
    start_sec = session.total_processed_samples / sr
    end_sec = start_sec + (session.window_samples / sr)
    model_status = get_model_status()

    # 1. Voice Activity Energy Check (skip silence)
    rms_energy = float(np.sqrt(np.mean(window_audio ** 2)))
    if rms_energy < 0.005:
        detection = WindowDetection(
            window_id=session.window_counter,
            start_time=start_sec,
            end_time=end_sec,
            fake_probability=0.0,
            real_probability=1.0,
            risk_score=0.0,
            risk_level="LOW_RISK",
            status="SILENCE",
            confidence=0.0,
            is_suspicious=False
        )
        session.window_counter += 1
        return {
            "status": "SILENCE",
            "model_status": model_status,
            "session_id": session.session_id,
            "window": asdict(detection),
            "call_risk": {
                "risk_level": session.current_risk_level,
                "risk_score": session.current_risk_score,
                "potential_spoof_detected": session.current_risk_level == "HIGH_SPOOF_RISK"
            },
            "incident_created": session.incident is not None
        }

    # 2. Preprocess and Extract Features (MFCC-only — live path must stay < hop interval)
    y_norm = normalize_audio(window_audio)
    mfcc = extract_mfcc_for_model(y_norm, sr=sr)

    # 3. Model Inference (Guarded). The CNN emits logits; convert with sigmoid.
    model, status = _load_cnn_model()
    if status in ("REAL_MODEL", "DEMO_MODEL") and model is not None:
        inp_tensor = torch.from_numpy(np.ascontiguousarray(mfcc, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            logit = model(inp_tensor).reshape(-1)[0]
            cnn_prob = float(torch.sigmoid(logit).cpu())
        heur_prob = heuristic_score_from_mfcc(mfcc)
        real_prob = float(np.clip(0.75 * cnn_prob + 0.25 * heur_prob, 0.01, 0.99))
    else:
        real_prob = heuristic_score_from_mfcc(mfcc)
    fake_prob = float(1.0 - real_prob)
    risk_score = float(np.clip(fake_prob * 100.0, 0.0, 100.0))
    is_suspicious = fake_prob >= settings.live_suspicious_threshold

    window_risk_level = (
        "HIGH_SPOOF_RISK" if fake_prob >= settings.live_high_risk_threshold
        else "SUSPICIOUS" if is_suspicious
        else "LOW_RISK"
    )

    detection = WindowDetection(
        window_id=session.window_counter,
        start_time=start_sec,
        end_time=end_sec,
        fake_probability=fake_prob,
        real_probability=real_prob,
        risk_score=risk_score,
        risk_level=window_risk_level,
        status="ANALYZED",
        confidence=float(max(real_prob, fake_prob) * 100.0),
        is_suspicious=is_suspicious
    )
    session.window_counter += 1

    # 4. Rolling History Maintenance (Last 10 windows)
    session.recent_windows.append(detection)
    session.all_windows.append(detection)
    if len(session.recent_windows) > 10:
        session.recent_windows.pop(0)

    # 5. Call-Level Risk Aggregation with Single-Spike Protection
    suspicious_count = sum(1 for w in session.recent_windows if w.is_suspicious)
    high_spike_count = sum(1 for w in session.recent_windows if w.fake_probability >= settings.live_high_risk_threshold)
    avg_fake_prob = float(np.mean([w.fake_probability for w in session.recent_windows if w.status == "ANALYZED"])) if session.recent_windows else 0.0
    
    # Aggregated Score calculation
    session.current_risk_score = float(np.clip(avg_fake_prob * 100.0, 0.0, 100.0))

    if high_spike_count >= settings.live_min_suspicious_windows and avg_fake_prob >= 0.70:
        session.current_risk_level = "HIGH_SPOOF_RISK"
    elif suspicious_count >= 1 or high_spike_count == 1:
        session.current_risk_level = "SUSPICIOUS"
    else:
        session.current_risk_level = "LOW_RISK"

    # 6. Incident Creation upon High Spoof Risk
    if session.current_risk_level == "HIGH_SPOOF_RISK" and session.incident is None:
        from backend.detection.detector import _compute_file_sha256
        model_sha = None
        if settings.model_path.is_file():
            try:
                model_sha = _compute_file_sha256(settings.model_path)
            except Exception:
                pass

        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        session.incident = LiveIncident(
            incident_id=incident_id,
            session_id=session.session_id,
            created_at=time.time(),
            duration_seconds=end_sec,
            analyzed_windows=len(session.all_windows),
            suspicious_windows=sum(1 for w in session.all_windows if w.is_suspicious),
            average_spoof_score=float(np.mean([w.fake_probability for w in session.all_windows])),
            peak_spoof_score=float(np.max([w.fake_probability for w in session.all_windows])),
            risk_level="HIGH_SPOOF_RISK",
            model_status=model_status,
            model_sha256=model_sha,
            timeline=[asdict(w) for w in session.all_windows[-20:]],
            social_engineering_indicators=session.social_engineering_flags,
            disclaimer="Detection is probabilistic and indicates acoustic anomalies associated with synthetic speech. VigilVoice does not make legal accusations or automatically file reports.",
            official_reporting_resources=OFFICIAL_REPORTING_RESOURCES
        )
        logger.warning(f"Live Call Incident {incident_id} created for session {session.session_id}")

    return {
        "status": "ANALYZED",
        "model_status": model_status,
        "session_id": session.session_id,
        "window": asdict(detection),
        "call_risk": {
            "risk_level": session.current_risk_level,
            "risk_score": session.current_risk_score,
            "potential_spoof_detected": session.current_risk_level == "HIGH_SPOOF_RISK"
        },
        "incident": session.incident.to_dict() if session.incident else None
    }


# ── REST Endpoints ─────────────────────────────────────────────────────────────

@router.post("/session")
async def create_live_session():
    """Initializes a new real-time call analysis session."""
    session = await session_manager.create_session()
    return {
        "success": True,
        "session_id": session.session_id,
        "created_at": session.created_at,
        "model_status": get_model_status(),
        "config": {
            "sample_rate": session.sample_rate,
            "window_seconds": settings.live_window_seconds,
            "hop_seconds": settings.live_hop_seconds,
            "high_risk_threshold": settings.live_high_risk_threshold,
            "suspicious_threshold": settings.live_suspicious_threshold,
        }
    }


@router.get("/status/{session_id}")
async def get_session_status(session_id: str):
    """Retrieves current risk state, window history, and active incident for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Live session not found or expired.")
        
    return {
        "success": True,
        "session_id": session.session_id,
        "model_status": get_model_status(),
        "duration_seconds": (session.total_processed_samples / session.sample_rate),
        "current_risk_level": session.current_risk_level,
        "current_risk_score": session.current_risk_score,
        "recent_windows": [asdict(w) for w in session.recent_windows],
        "incident": session.incident.to_dict() if session.incident else None,
        "reporting_resources": OFFICIAL_REPORTING_RESOURCES
    }


@router.get("/reporting-resources")
async def get_reporting_resources():
    """Provides navigation links to official Indian Government cybercrime reporting portals."""
    return {
        "success": True,
        "resources": OFFICIAL_REPORTING_RESOURCES,
        "guidance": "If you suspect financial fraud or voice impersonation, preserve incident details and use these official government portals or call 1930."
    }


@router.post("/incident/{session_id}/report")
async def generate_incident_evidence_report(session_id: str):
    """Saves structured incident evidence JSON to reports/incidents/{incident_id}.json."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Live session not found.")
        
    if not session.incident:
        # Create an on-demand incident summary even if high risk wasn't fully triggered
        from backend.detection.detector import _compute_file_sha256
        model_sha = None
        if settings.model_path.is_file():
            try:
                model_sha = _compute_file_sha256(settings.model_path)
            except Exception:
                pass
                
        incident_id = f"CALL-{uuid.uuid4().hex[:8].upper()}"
        all_scores = [w.fake_probability for w in session.all_windows] if session.all_windows else [0.0]
        session.incident = LiveIncident(
            incident_id=incident_id,
            session_id=session.session_id,
            created_at=time.time(),
            duration_seconds=(session.total_processed_samples / session.sample_rate),
            analyzed_windows=len(session.all_windows),
            suspicious_windows=sum(1 for w in session.all_windows if w.is_suspicious),
            average_spoof_score=float(np.mean(all_scores)),
            peak_spoof_score=float(np.max(all_scores)),
            risk_level=session.current_risk_level,
            model_status=get_model_status(),
            model_sha256=model_sha,
            timeline=[asdict(w) for w in session.all_windows[-25:]],
            social_engineering_indicators=session.social_engineering_flags,
            disclaimer="Detection is probabilistic and indicates acoustic anomalies associated with synthetic speech. VigilVoice does not make legal accusations or automatically file reports.",
            official_reporting_resources=OFFICIAL_REPORTING_RESOURCES
        )

    incident_dir = PROJECT_ROOT / "reports" / "incidents"
    incident_dir.mkdir(parents=True, exist_ok=True)
    report_file = incident_dir / f"{session.incident.incident_id}.json"
    
    with open(report_file, "w") as f:
        json.dump(session.incident.to_dict(), f, indent=2)

    return {
        "success": True,
        "incident_id": session.incident.incident_id,
        "report_path": str(report_file),
        "incident": session.incident.to_dict()
    }


# ── WebSocket Audio Streaming Endpoint ─────────────────────────────────────────

@router.websocket("/audio/{session_id}")
async def websocket_live_audio(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint receiving live float32 PCM audio stream.
    Processes 2.5s sliding windows (1.0s hop) asynchronously and returns detection results.
    """
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found or expired")
        return

    await websocket.accept()
    logger.info(f"WebSocket live streaming connected for session {session_id}")

    try:
        while True:
            # Receive float32 PCM byte chunks from browser AudioContext
            message = await websocket.receive()

            # Starlette sends a raw disconnect dict before raising WebSocketDisconnect;
            # stop the loop immediately so we don't call receive() on a closed socket.
            if message.get("type") == "websocket.disconnect":
                logger.info(f"WebSocket disconnect message received for session {session_id}")
                break

            if "bytes" in message:
                raw_bytes = message["bytes"]
                if len(raw_bytes) == 0:
                    continue
                audio_chunk = np.frombuffer(raw_bytes, dtype=np.float32)
                session.add_audio_chunk(audio_chunk)

                # Process all ready overlapping windows
                while True:
                    window = session.get_ready_window()
                    if window is None:
                        break

                    # Execute CPU ML inference in worker thread to prevent blocking event loop
                    result = await run_in_threadpool(process_live_window, session, window)
                    await websocket.send_json(result)
                    session.advance_buffer()

            elif "text" in message:
                # Handle control messages (e.g. social engineering flag toggles)
                try:
                    payload = json.loads(message["text"])
                    action = payload.get("action")
                    if action == "add_indicator":
                        indicator = payload.get("indicator")
                        if indicator in VALID_SOCIAL_ENGINEERING_INDICATORS and indicator not in session.social_engineering_flags:
                            session.social_engineering_flags.append(indicator)
                            await websocket.send_json({
                                "status": "INDICATOR_UPDATED",
                                "indicators": session.social_engineering_flags
                            })
                except Exception:
                    pass

    except WebSocketDisconnect:
        logger.info(f"WebSocket live streaming disconnected for session {session_id}")
    except Exception as e:
        logger.exception(f"WebSocket live streaming error for session {session_id}: {e}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
