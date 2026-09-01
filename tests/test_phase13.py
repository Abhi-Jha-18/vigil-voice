"""
Phase 13 Tests: Robustness, Generalization & Adversarial Audio Testing

Tests cover:
- Noise transformation mathematics & bounds
- Resampling preservation & restoration
- Volume gain adjustments & clipping prevention
- Audio truncation & zero-padding
- Simulated reverberation convolution
- Combined condition application
- Seed determinism & reproducibility
- Robustness evaluation blocked status handling when missing checkpoint
- Scorecard JSON generation structure
"""

import os
import sys
import json
import tempfile
import numpy as np
import pytest
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from training.robustness import (
    add_noise,
    resample_audio,
    change_volume,
    truncate_audio,
    simulate_reverberation,
    apply_transformation,
    run_robustness_evaluation,
)


# ── Fixture Helper ─────────────────────────────────────────────────────────────

def _generate_test_signal(duration_sec: float = 1.0, sr: int = 16000, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ── 1. Transformation Tests ────────────────────────────────────────────────────

def test_add_noise_bounds_and_shape():
    y = _generate_test_signal(1.0)
    noisy = add_noise(y, sr=16000, snr_db=10.0, seed=42)

    assert noisy.shape == y.shape
    assert not np.isnan(noisy).any()
    assert not np.isinf(noisy).any()
    assert np.max(np.abs(noisy)) <= 1.0
    # Noisy signal should differ from clean signal
    assert not np.allclose(y, noisy)


def test_add_noise_reproducibility():
    y = _generate_test_signal(1.0)
    noisy1 = add_noise(y, sr=16000, snr_db=10.0, seed=42)
    noisy2 = add_noise(y, sr=16000, snr_db=10.0, seed=42)
    noisy_diff_seed = add_noise(y, sr=16000, snr_db=10.0, seed=99)

    assert np.array_equal(noisy1, noisy2), "Same seed must produce identical noisy audio"
    assert not np.array_equal(noisy1, noisy_diff_seed), "Different seeds should produce different noise"


def test_resample_audio_shape():
    y = _generate_test_signal(1.0, sr=16000)
    # Downsample to 8kHz and resample back to 16kHz
    resampled = resample_audio(y, orig_sr=16000, target_sr=8000)

    assert not np.isnan(resampled).any()
    assert not np.isinf(resampled).any()
    # Output length should match original audio array length approximately
    assert abs(len(resampled) - len(y)) <= 16  # allow tiny rounding boundary


def test_change_volume_clipping_prevention():
    y = _generate_test_signal(1.0)
    # Boost by +20 dB
    boosted = change_volume(y, gain_db=20.0)

    assert not np.isnan(boosted).any()
    assert not np.isinf(boosted).any()
    assert np.max(np.abs(boosted)) <= 1.0  # Clipped safely


def test_truncate_audio_lengths():
    y = _generate_test_signal(3.0, sr=16000)  # 48000 samples
    truncated_1s = truncate_audio(y, sr=16000, duration_seconds=1.0)
    assert len(truncated_1s) == 16000

    short_y = _generate_test_signal(0.5, sr=16000)  # 8000 samples
    padded_3s = truncate_audio(short_y, sr=16000, duration_seconds=3.0)
    assert len(padded_3s) == 48000


def test_simulate_reverberation():
    y = _generate_test_signal(1.0)
    reverb = simulate_reverberation(y, sr=16000, decay=5.0)

    assert len(reverb) == len(y)
    assert not np.isnan(reverb).any()
    assert not np.isinf(reverb).any()
    assert np.max(np.abs(reverb)) <= 1.0


def test_apply_transformation_suite():
    y = _generate_test_signal(2.0)
    conditions = [
        "clean", "noise_20db", "noise_10db", "resample_8khz",
        "volume_-6db", "short_audio_1s", "reverberation_mild",
        "combined_noise_volume", "combined_resample_noise"
    ]
    for cond in conditions:
        out = apply_transformation(y, sr=16000, condition_name=cond, seed=42)
        assert len(out) > 0, f"Transformation {cond} returned empty array"
        assert not np.isnan(out).any(), f"Transformation {cond} produced NaNs"
        assert not np.isinf(out).any(), f"Transformation {cond} produced Infs"


def test_transformation_does_not_mutate_original():
    y_orig = _generate_test_signal(2.0)
    y_copy = y_orig.copy()
    
    _ = apply_transformation(y_orig, sr=16000, condition_name="noise_10db", seed=42)
    _ = apply_transformation(y_orig, sr=16000, condition_name="volume_+6db", seed=42)
    _ = apply_transformation(y_orig, sr=16000, condition_name="resample_8khz", seed=42)
    
    assert np.array_equal(y_orig, y_copy), "Transformation functions must not mutate the original audio array"


# ── 2. Guard Check: Missing Dataset / Checkpoint Handling ──────────────────────

def test_robustness_evaluation_blocked_when_missing_checkpoint():
    with tempfile.TemporaryDirectory() as d:
        temp_dir = Path(d)
        fake_args = type("Args", (), {
            "manifest": temp_dir / "nonexistent_manifest.csv",
            "checkpoint": temp_dir / "nonexistent_model.pth",
            "reports_dir": temp_dir / "reports",
            "seed": 42,
            "device": "cpu"
        })()

        scorecard = run_robustness_evaluation(fake_args)
        assert scorecard["status"] == "BLOCKED"
        assert "ROBUSTNESS EVALUATION BLOCKED" in scorecard.get("reason", "") or "missing" in scorecard.get("reason", "").lower()
        assert (temp_dir / "reports" / "robustness_report.json").exists()
