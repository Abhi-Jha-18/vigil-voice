import io
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io.wavfile import write

from backend.audio.features import extract_acoustic_features, get_mel_spectrogram_b64
from backend.audio.processor import validate_format
from backend.audio.vad import remove_silence
from backend.detection.decision import make_decision
from backend.detection.detector import run_ai_detection
from training.preprocessing import preprocess_for_training


def generate_mock_wav(filename: str, duration_sec: float = 3.0, sample_rate: int = 16000, seed: int = 42):
    """
    Generates a mock sine wave audio file for test verification with seeded RNG.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Generate simple speech-like composite wave (440Hz sine + seeded noise)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * rng.randn(len(t))
    # Normalize to 16-bit PCM range
    audio_scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    write(filename, sample_rate, audio_scaled)


def test_invalid_empty_audio_returns_400():
    from backend.main import app
    with TestClient(app) as client:
        response = client.post(
            "/api/detect",
            files={"file": ("empty.wav", b"", "audio/wav")},
            data={"phase": "phase1"},
        )
        assert response.status_code == 400, response.text
        assert "empty" in response.json()["detail"].lower()


def test_api_health_and_invalid_phase():
    """Exercise API validation without requiring a separately running server."""
    from backend.main import app
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        payload = health.json()
        assert "model" in payload
        assert "configuration" in payload
        assert "runtime" in payload
        response = client.post(
            "/api/detect",
            files={"file": ("sample.wav", b"not-audio", "audio/wav")},
            data={"phase": "unknown"},
        )
        assert response.status_code == 422, response.text


def test_api_detects_valid_wav():
    """Run a complete request through the API using an in-memory WAV file with client lifespan."""
    from backend.main import app
    rng = np.random.RandomState(42)
    t = np.arange(16000) / 16000
    samples = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.randn(16000)
    samples = np.int16(samples / np.max(np.abs(samples)) * 32767)
    payload = io.BytesIO()
    write(payload, 16000, samples)

    with TestClient(app) as client:
        response = client.post(
            "/api/detect",
            files={"file": ("voice.wav", payload.getvalue(), "audio/wav")},
            data={"phase": "phase1"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["success"] is True
        assert body["verdict"] in {"REAL", "FAKE", "UNCERTAIN"}
        assert body["audio_duration_seconds"] > 0


def test_full_pipeline(tmp_path: Path):
    """Integration test without try/except so pytest directly reports real assertion failures."""
    mock_file = str(tmp_path / "temp_mock_audio.wav")

    # 1. Format validation test
    assert validate_format("test.wav") is True
    assert validate_format("test.mp3") is True
    assert validate_format("test.exe") is False

    # Generate the file with seeded RNG
    generate_mock_wav(mock_file, seed=42)

    # 2. Audio preprocessing test
    y_preprocessed = preprocess_for_training(mock_file)
    assert len(y_preprocessed) > 0, "Preprocessed audio signal is empty."
    assert np.max(np.abs(y_preprocessed)) <= 1.0, "Normalization peak exceeds 1.0."

    # 3. Voice Activity Detection test
    y_clean = remove_silence(y_preprocessed, sr=16000)
    assert len(y_clean) > 0, "VAD returned empty signal."

    # 4. Feature Extraction test
    features = extract_acoustic_features(y_clean, sr=16000)
    assert "mfcc_mean" in features, "MFCC features missing."
    assert len(features["mfcc_mean"]) == 13, f"Expected 13 MFCC coefficients, got {len(features['mfcc_mean'])}"
    assert np.asarray(features["mfcc_sequence"]).shape == (13, 64), "Model MFCC sequence shape is invalid."
    assert features["audio_duration_seconds"] > 0, "Audio duration must be greater than zero."

    spectrogram_b64 = get_mel_spectrogram_b64(y_clean, sr=16000)
    assert len(spectrogram_b64) > 0, "Failed to generate spectrogram base64 image."

    # 5. AI Detection Test
    score_default = run_ai_detection(features, phase="phase1")
    assert 0.0 <= score_default <= 1.0, f"AI score {score_default} out of range [0, 1]."

    from unittest.mock import patch
    from backend.config import settings
    with patch.object(settings, "demo_mode", True):
        score_real = run_ai_detection(features, phase="phase1", force_verdict="real")
        assert score_real >= 0.75, "Force REAL override failed."

        score_fake = run_ai_detection(features, phase="phase1", force_verdict="fake")
        assert score_fake <= 0.40, "Force FAKE override failed."

    # 6. Decision Engine Test
    decision_real = make_decision(score_real)
    assert decision_real["verdict"] == "REAL", "Expected REAL verdict."
    assert decision_real["risk_level"] in ["LOW RISK", "MINIMAL RISK"], "Expected low/minimal risk for REAL."

    decision_fake = make_decision(score_fake)
    assert decision_fake["verdict"] == "FAKE", "Expected FAKE verdict."
    assert decision_fake["risk_level"] in ["HIGH RISK", "CRITICAL RISK"], "Expected high/critical risk for FAKE."


@pytest.fixture(scope="module")
def fixed_real_fake_scores():
    """Small deterministic fixture for the acoustic-heuristic regression gate."""
    labels = []
    scores = []
    sr = 16000

    # Real fixtures: natural frequency modulation, amplitude dynamics, and spectral variation
    for seed in [101, 102, 103]:
        rng = np.random.RandomState(seed)
        t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
        f0 = 200.0 + 50.0 * np.sin(2 * np.pi * 3.0 * t)
        sig = 0.4 * np.sin(2 * np.pi * f0 * t) * (1.0 + 0.5 * np.cos(2 * np.pi * 5.0 * t)) + 0.05 * rng.randn(len(t))
        feats = extract_acoustic_features(sig.astype(np.float32), sr)
        score = run_ai_detection(feats, phase="phase1")
        labels.append(1)
        scores.append(score)

    # Fake fixtures: monotone, static spectrum, temporally flat delta
    for freq in [440.0, 880.0, 1000.0]:
        t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
        sig = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        feats = extract_acoustic_features(sig, sr)
        score = run_ai_detection(feats, phase="phase1")
        labels.append(0)
        scores.append(score)

    return labels, scores


def test_fixed_real_fake_auc_regression_gate(fixed_real_fake_scores):
    """Require real-labeled fixture signals to rank above fake-labeled signals."""
    from sklearn.metrics import roc_auc_score

    labels, scores = fixed_real_fake_scores
    auc = float(roc_auc_score(labels, scores))
    # Synthetic regression signals only; this is not a model performance benchmark.
    assert auc >= 0.85, f"Regression gate failure: AUC {auc:.4f} is below 0.85 threshold"
