import os
import sys
import io
import numpy as np
from scipy.io.wavfile import write
from fastapi.testclient import TestClient

# Add the project root to the python path to import our modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from backend.audio.processor import validate_format
from training.preprocessing import preprocess_for_training
from backend.audio.vad import remove_silence
from backend.audio.features import extract_acoustic_features, get_mel_spectrogram_b64
from backend.detection.detector import run_ai_detection
from backend.detection.decision import make_decision

def generate_mock_wav(filename: str, duration_sec: float = 3.0, sample_rate: int = 16000):
    """
    Generates a mock sine wave audio file for test verification.
    """
    print(f"Generating mock audio file: {filename}...")
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Generate simple speech-like composite wave (440Hz sine + noise)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    # Normalize to 16-bit PCM range
    audio_scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    write(filename, sample_rate, audio_scaled)
    print("Mock audio file created successfully.")

def test_invalid_empty_audio_returns_400():
    from backend.main import app
    response = TestClient(app).post(
        "/api/detect",
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"phase": "phase1"},
    )
    assert response.status_code == 400, response.text
    assert "empty" in response.json()["detail"].lower()
    print("[OK] Invalid empty audio file returned a 400 error instead of crashing the server.")


def test_api_health_and_invalid_phase():
    """Exercise API validation without requiring a separately running server."""
    from backend.main import app
    client = TestClient(app)
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
    print("[OK] API health and phase validation passed.")


def test_api_detects_valid_wav():
    """Run a complete request through the API using an in-memory WAV file."""
    from backend.main import app
    samples = (0.4 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 32767).astype(np.int16)
    payload = io.BytesIO()
    write(payload, 16000, samples)
    response = TestClient(app).post(
        "/api/detect",
        files={"file": ("voice.wav", payload.getvalue(), "audio/wav")},
        data={"phase": "phase1"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    assert body["verdict"] in {"REAL", "FAKE", "UNCERTAIN"}
    assert body["audio_duration_seconds"] > 0
    print("[OK] Full API request passed.")


def test_full_pipeline():
    print("=" * 60)
    print("          STARTING AUDIO PIPELINE INTEGRATION TEST")
    print("=" * 60)
    
    mock_file = os.path.join(project_root, "tests", "temp_mock_audio.wav")
    
    try:
        # 1. Format validation test
        print("\n[Step 1] Testing Format Validation...")
        assert validate_format("test.wav") is True
        assert validate_format("test.mp3") is True
        assert validate_format("test.exe") is False
        print("[OK] Format validation passed.")

        # Generate the file
        generate_mock_wav(mock_file)

        # 2. Audio preprocessing test
        print("\n[Step 2] Testing Preprocessing (Loading, Mono, Resample, Normalize)...")
        y_preprocessed = preprocess_for_training(mock_file)
        assert len(y_preprocessed) > 0, "Preprocessed audio signal is empty."
        assert np.max(np.abs(y_preprocessed)) <= 1.0, "Normalization peak exceeds 1.0."
        print(f"[OK] Preprocessing passed. Signal length: {len(y_preprocessed)} samples.")

        # 3. Voice Activity Detection test
        print("\n[Step 3] Testing Voice Activity Detection (Silence removal)...")
        y_clean = remove_silence(y_preprocessed, sr=16000)
        assert len(y_clean) > 0, "VAD returned empty signal."
        print(f"[OK] VAD passed. Active signal length: {len(y_clean)} samples.")

        # 4. Feature Extraction test
        print("\n[Step 4] Testing Feature Extraction & Spectrogram Rendering...")
        features = extract_acoustic_features(y_clean, sr=16000)
        assert "mfcc_mean" in features, "MFCC features missing."
        assert len(features["mfcc_mean"]) == 13, f"Expected 13 MFCC coefficients, got {len(features['mfcc_mean'])}"
        assert np.asarray(features["mfcc_sequence"]).shape == (13, 64), "Model MFCC sequence shape is invalid."
        assert features["audio_duration_seconds"] > 0, "Audio duration must be greater than zero."
        
        spectrogram_b64 = get_mel_spectrogram_b64(y_clean, sr=16000)
        assert len(spectrogram_b64) > 0, "Failed to generate spectrogram base64 image."
        print("[OK] Feature extraction passed.")
        print(f"  Duration: {features['audio_duration_seconds']:.2f}s")
        print(f"  ZCR Mean: {features['zero_crossing_rate_mean']:.4f}")
        print(f"  Spectral Centroid Mean: {features['spectral_centroid_mean']:.2f} Hz")
        print(f"  Mel Spectrogram B64 Length: {len(spectrogram_b64)} chars")

        # 5. AI Detection Test
        print("\n[Step 5] Testing AI Detection Engine...")
        # Test default
        score_default = run_ai_detection(features, phase="phase1")
        assert 0.0 <= score_default <= 1.0, f"AI score {score_default} out of range [0, 1]."
        
        # Test overrides
        score_real = run_ai_detection(features, phase="phase1", force_verdict="real")
        assert score_real >= 0.75, "Force REAL override failed."
        
        score_fake = run_ai_detection(features, phase="phase1", force_verdict="fake")
        assert score_fake <= 0.40, "Force FAKE override failed."
        print("[OK] AI Detection Engine passed.")

        # 6. Decision Engine Test
        print("\n[Step 6] Testing Decision Engine...")
        decision_real = make_decision(score_real)
        assert decision_real["verdict"] == "REAL", "Expected REAL verdict."
        assert decision_real["risk_level"] in ["LOW RISK", "MINIMAL RISK"], "Expected low/minimal risk for REAL."
        
        decision_fake = make_decision(score_fake)
        assert decision_fake["verdict"] == "FAKE", "Expected FAKE verdict."
        assert decision_fake["risk_level"] in ["HIGH RISK", "CRITICAL RISK"], "Expected high/critical risk for FAKE."
        print("[OK] Decision Engine passed.")
        print(f"  REAL Score {score_real:.4f} -> {decision_real['verdict']} ({decision_real['confidence']}%) | {decision_real['risk_level']}")
        print(f"  FAKE Score {score_fake:.4f} -> {decision_fake['verdict']} ({decision_fake['confidence']}%) | {decision_fake['risk_level']}")

        print("\n" + "=" * 60)
        print("          ALL INTEGRATION PIPELINE TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] PIPELINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Clean up temp file
        if os.path.exists(mock_file):
            os.remove(mock_file)
            print("\nCleaned up test temporary files.")

if __name__ == "__main__":
    test_invalid_empty_audio_returns_400()
    test_api_health_and_invalid_phase()
    test_api_detects_valid_wav()
    test_full_pipeline()
