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
