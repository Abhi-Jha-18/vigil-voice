import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from backend.audio.features import (
    LOGMEL_DELTA_FEATURE_VERSION,
    extract_model_features,
    feature_crops,
)
from backend.audio.pipeline import AudioSegment
from backend.detection.detector import (
    SimpleCNNDetector,
    validate_checkpoint_feature_version,
)
from training.audio_augmentations import apply_packet_loss


def test_logmel_delta_features_are_normalized_and_have_three_channels():
    rng = np.random.default_rng(12)
    waveform = rng.normal(0.0, 0.1, 16_000).astype(np.float32)

    features = extract_model_features(
        waveform, feature_version=LOGMEL_DELTA_FEATURE_VERSION
    )
    crops = feature_crops(features)

    assert features.shape[0:2] == (3, 64)
    assert np.allclose(features.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(features.std(axis=-1), 1.0, atol=1e-4)
    assert crops.shape[1:] == (3, 64, 64)
    model = SimpleCNNDetector(feature_version=LOGMEL_DELTA_FEATURE_VERSION).eval()
    with torch.no_grad():
        assert model(torch.from_numpy(crops[:1])).shape == (1, 1)


def test_checkpoint_feature_version_mismatch_is_rejected(tmp_path: Path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"placeholder")
    metadata = tmp_path / "model_metadata.json"
    metadata.write_text(json.dumps({"feature_version": "mfcc_v1"}), encoding="utf-8")

    with pytest.raises(ValueError, match="feature_version mismatch"):
        validate_checkpoint_feature_version(checkpoint, LOGMEL_DELTA_FEATURE_VERSION)


def test_packet_loss_is_repeatable_and_preserves_waveform_length():
    waveform = np.ones(3_200, dtype=np.float32)
    first = apply_packet_loss(waveform, loss_rate=0.25, seed=42)
    second = apply_packet_loss(waveform, loss_rate=0.25, seed=42)

    assert len(first) == len(waveform)
    assert np.array_equal(first, second)
    assert np.any(first == 0)


def test_segment_inference_returns_gradcam_spectrogram():
    from backend.detection import detector

    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = (0.4 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    segment = AudioSegment(0, 0.0, 1.0, 1.0, "GOOD", audio)
    model = SimpleCNNDetector().eval()

    with (
        patch.object(detector, "_load_cnn_model", return_value=(model, "REAL_MODEL")),
        patch.object(detector, "get_model_thresholds", return_value=(None, None)),
        patch.object(detector.settings, "resolve_blend_mode", return_value="cnn_only"),
    ):
        predictions = detector.run_segment_detection([segment], phase="phase2")

    assert len(predictions) == 1
    assert predictions[0].gradcam
    assert predictions[0].to_dict()["gradcam"] == predictions[0].gradcam
