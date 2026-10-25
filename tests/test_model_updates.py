import numpy as np
import pytest
import torch
import torch.nn as nn

from backend.audio.features import (
    extract_mfcc_for_model,
    extract_acoustic_features,
    mfcc_crops,
    _cmn,
    _pad_or_crop,
    N_MFCC,
    N_TIME_STEPS,
    HOP_LENGTH,
)
from backend.detection.detector import (
    CNNLSTMDetector,
    SimpleCNNDetector,
    heuristic_score_from_mfcc,
    _heuristic_score_from_arrays,
    run_phase2_cnn_detection,
    run_segment_detection,
)
from backend.audio.pipeline import AudioSegment


def test_padded_region_is_exactly_zero():
    """
    Test 1:
    Verify that padding after CMN ensures the padded region is exactly zero.
    Before this fix, _cmn(_pad_or_crop(mfcc)) resulted in non-zero padded values (-mean).
    """
    # 0.5s audio at 16kHz produces ~16 frames (less than N_TIME_STEPS = 64)
    sr = 16000
    duration_sec = 0.5
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    import librosa
    raw_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    actual_frames = raw_mfcc.shape[1]
    assert actual_frames < N_TIME_STEPS, f"Test requires < {N_TIME_STEPS} frames, got {actual_frames}"

    # Check extract_mfcc_for_model
    mfcc_model = extract_mfcc_for_model(y, sr=sr)
    assert mfcc_model.shape == (N_MFCC, N_TIME_STEPS)
    padded_region_model = mfcc_model[:, actual_frames:]
    assert np.all(padded_region_model == 0.0), "Padded region in extract_mfcc_for_model must be exactly 0.0"

    # Check extract_acoustic_features
    features = extract_acoustic_features(y, sr=sr)
    seq = np.asarray(features["mfcc_sequence"])
    assert seq.shape == (N_MFCC, N_TIME_STEPS)
    padded_region_features = seq[:, actual_frames:]
    assert np.all(padded_region_features == 0.0), "Padded region in extract_acoustic_features must be exactly 0.0"


def test_mfcc_crops_shapes():
    """
    Test 2:
    Verify mfcc_crops with short, exact, and long MFCC inputs.
    """
    # Short: T = 30 < 64 -> padded to 64, exactly 1 crop
    mfcc_short = np.random.randn(13, 30).astype(np.float32)
    crops_short = mfcc_crops(mfcc_short, n=64, hop=32)
    assert crops_short.shape == (1, 13, 64)
    assert np.all(crops_short[0, :, 30:] == 0.0)

    # Exact: T = 64 -> 1 crop
    mfcc_exact = np.random.randn(13, 64).astype(np.float32)
    crops_exact = mfcc_crops(mfcc_exact, n=64, hop=32)
    assert crops_exact.shape == (1, 13, 64)

    # Long: T = 128 -> starts at 0, 32, 64 -> 3 crops
    mfcc_long = np.random.randn(13, 128).astype(np.float32)
    crops_long = mfcc_crops(mfcc_long, n=64, hop=32)
    assert crops_long.shape == (3, 13, 64)

    # Non-multiple: T = 100 -> starts: 0, 32, and tail at 100-64=36
    mfcc_tail = np.random.randn(13, 100).astype(np.float32)
    crops_tail = mfcc_crops(mfcc_tail, n=64, hop=32)
    assert crops_tail.shape[0] >= 2
    assert crops_tail.shape[1:] == (13, 64)


def test_heuristic_no_hardcoded_centroid_bias_and_excludes_c0():
    """
    Test 3 & 4:
    Verify heuristic_score_from_mfcc:
    - computes std on mfcc[1:] (excluding c0 energy)
    - defaults sc_mean to None and skips centroid term (no hardcoded 1500.0 bias)
    """
    mfcc = np.zeros((13, 100), dtype=np.float32)
    # Put massive variance ONLY on c0 (row 0)
    mfcc[0, :] = np.linspace(-100, 100, 100)
    # Rows 1..12 have very low variance (frozen vocoder-like)
    mfcc[1:, :] = 0.01 * np.random.randn(12, 100)

    # If c0 were included, std_variance would be huge (> 18), biasing score toward REAL.
    # Since c0 is excluded, std_variance is very small (< 6), so score drops toward FAKE.
    score = heuristic_score_from_mfcc(mfcc)
    assert score < 0.50, f"Expected spoof detection due to frozen c1..c12, got {score}"

    # Check that sc_mean=None skips the centroid term compared to hardcoded 1500
    score_no_sc = _heuristic_score_from_arrays(mfcc_stds=[10.0], sc_mean=None)
    score_hardcoded_1500 = _heuristic_score_from_arrays(mfcc_stds=[10.0], sc_mean=1500.0)
    # 1500 falls in [1100, 2200] which previously added +0.08
    assert score_hardcoded_1500 == pytest.approx(score_no_sc + 0.08)


def test_cnnlstm_detector_returns_logits():
    """
    Test 5:
    Verify CNNLSTMDetector returns raw logits and does not end in Sigmoid.
    """
    model = CNNLSTMDetector()
    model.eval()

    # Verify self.fc is a single Linear layer or does not have Sigmoid
    for module in model.fc.modules():
        assert not isinstance(module, nn.Sigmoid), "CNNLSTMDetector should not contain nn.Sigmoid"

    # Forward pass on dummy batch
    x = torch.randn(2, 1, 13, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)

    # SimpleCNNDetector also returns logits
    cnn = SimpleCNNDetector()
    cnn.eval()
    with torch.no_grad():
        cnn_out = cnn(x)
    assert cnn_out.shape == (2, 1)


def test_inference_multi_crop_average_logits():
    """
    Test multi-crop inference in run_segment_detection.
    """
    sr = 16000
    # 5.0 seconds of audio: produces multiple 64-frame crops
    t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    seg = AudioSegment(
        segment_id=0,
        start_time=0.0,
        end_time=5.0,
        duration=5.0,
        status="GOOD",
        audio_data=y,
    )

    preds = run_segment_detection([seg], phase="phase1")
    assert len(preds) == 1
    assert 0.0 <= preds[0].real_probability <= 1.0


import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from scipy.io import wavfile

from backend.main import app
from backend.config import settings
from backend.detection.detector import (
    run_phase1_stub_detection,
    run_phase2_cnn_detection,
    run_ai_detection,
    _get_checkpoint_status,
    reset_cached_model,
)
from backend.detection.decision import THRESHOLD_REAL, THRESHOLD_FAKE, classify_score
from backend.api.live import LiveSession, LiveIncident, process_live_window


def _make_dummy_wav_bytes(duration_sec: float = 1.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, samples)
    return buf.getvalue()


def test_demo_mode_false_ignores_force_verdict_detector_and_api():
    """
    Requirement 1:
    Gate force_verdict: ignore force_verdict unless settings.demo_mode is True.
    Test detector functions and API endpoint /api/detect with demo_mode False vs True.
    """
    sr = 16000
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    features = extract_acoustic_features(y, sr=sr)
    seg = AudioSegment(
        segment_id=0,
        start_time=0.0,
        end_time=1.0,
        duration=1.0,
        status="GOOD",
        audio_data=y,
    )

    client = TestClient(app)
    wav_bytes = _make_dummy_wav_bytes()

    # 1. With demo_mode = False: force_verdict must be ignored
    with patch.object(settings, "demo_mode", False):
        score_p1 = run_phase1_stub_detection(features, force_verdict="real")
        assert score_p1 < 0.50, f"force_verdict should be ignored when demo_mode=False, got {score_p1}"

        preds = run_segment_detection([seg], phase="phase1", force_verdict="real")
        assert len(preds) == 1
        assert preds[0].real_probability < 0.50

        # API level: sending force_verdict with demo_mode=False succeeds with 200 and ignores override
        resp = client.post(
            "/api/detect",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={"phase": "phase1", "force_verdict": "real"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["demo_override"] is False
        assert body["verdict"] in {"FAKE", "UNCERTAIN"}

    # 2. With demo_mode = True: force_verdict is honored
    with patch.object(settings, "demo_mode", True):
        score_forced_real = run_phase1_stub_detection(features, force_verdict="real")
        assert score_forced_real >= 0.75
        score_forced_fake = run_phase1_stub_detection(features, force_verdict="fake")
        assert score_forced_fake <= 0.40

        preds_forced = run_segment_detection([seg], phase="phase1", force_verdict="real")
        assert preds_forced[0].real_probability >= 0.75

        resp_demo = client.post(
            "/api/detect",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={"phase": "phase1", "force_verdict": "real"},
        )
        assert resp_demo.status_code == 200
        body_demo = resp_demo.json()
        assert body_demo["demo_override"] is True
        assert body_demo["verdict"] == "REAL"


def test_is_synthetic_metadata_determines_model_status():
    """
    Requirement 4:
    Model status: decide DEMO_MODEL vs REAL_MODEL from an explicit boolean `is_synthetic`
    in the metadata JSON (not the substring 'DEMO'). Missing metadata -> DEMO_MODEL.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_file = tmp_path / "model.pth"
        meta_file = tmp_path / "model_metadata.json"

        # Create dummy weights
        torch.save({"dummy": torch.tensor([1.0])}, model_file)

        # 1. is_synthetic: True -> DEMO_MODEL
        meta_file.write_text(json.dumps({"is_synthetic": True}))
        assert _get_checkpoint_status(model_file) == "DEMO_MODEL"

        # 2. is_synthetic: False -> REAL_MODEL (even if name has DEMO)
        meta_file.write_text(json.dumps({"is_synthetic": False, "dataset_identifier": "DEMO_DATASET"}))
        assert _get_checkpoint_status(model_file) == "REAL_MODEL"

        # 3. Missing metadata -> DEMO_MODEL
        meta_file.unlink()
        assert _get_checkpoint_status(model_file) == "DEMO_MODEL"


def test_checksum_mismatch_triggers_model_integrity_failure_and_503():
    """
    Requirement:
    Checksum mismatch -> MODEL_INTEGRITY_FAILURE and /api/ready returns 503.
    """
    client = TestClient(app)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_file = tmp_path / "corrupted_model.pth"
        meta_file = tmp_path / "corrupted_model_metadata.json"

        # Write dummy model file and wrong sha256 in metadata
        model_file.write_bytes(b"some model weights")
        meta_file.write_text(json.dumps({
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "is_synthetic": False
        }))

        with patch.object(settings, "model_path", model_file):
            reset_cached_model()
            try:
                status = _get_checkpoint_status(model_file)
                assert status == "MODEL_INTEGRITY_FAILURE"

                ready_resp = client.get("/api/ready")
                assert ready_resp.status_code == 503
                payload = ready_resp.json()
                assert payload["ready"] is False
                assert payload["model_status"] == "MODEL_INTEGRITY_FAILURE"
            finally:
                reset_cached_model()


def test_effective_phase_reporting_and_demo_override():
    """
    Requirements 3 & 5:
    Report effective phase in responses (phases 3 and 4 fall back to phase 2).
    Add 'demo_override' to every detection response and to incident evidence JSON.
    """
    client = TestClient(app)
    wav_bytes = _make_dummy_wav_bytes()

    for requested_phase in ["phase3", "phase4"]:
        resp = client.post(
            "/api/detect",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={"phase": requested_phase},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["phase_used"] == "phase2"
        assert "demo_override" in body
        assert isinstance(body["demo_override"], bool)


def test_centralized_thresholds_and_blend_mode():
    """
    Requirements 6 & 7:
    - Keep REAL/FAKE/UNCERTAIN thresholds in one place (decision.py).
    - Add config blend_mode (cnn_only | blend), default cnn_only when status is REAL_MODEL.
    - Heuristic must not override a CNN 'fake' verdict by default.
    """
    assert THRESHOLD_REAL == 0.75
    assert THRESHOLD_FAKE == 0.40

    v_real, c_real = classify_score(0.80)
    assert v_real == "REAL"
    assert c_real > 0

    v_fake, c_fake = classify_score(0.30)
    assert v_fake == "FAKE"
    assert c_fake > 0

    v_unc, _ = classify_score(0.55)
    assert v_unc == "UNCERTAIN"

    # Test blend_mode default resolution
    with patch.object(settings, "blend_mode", None):
        assert settings.resolve_blend_mode("REAL_MODEL") == "cnn_only"
        assert settings.resolve_blend_mode("DEMO_MODEL") == "blend"

    # Explicit override in config
    with patch.object(settings, "blend_mode", "blend"):
        assert settings.resolve_blend_mode("REAL_MODEL") == "blend"

