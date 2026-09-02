"""
Phase 11 Tests: Real Anti-Spoofing Dataset + Production Training Pipeline

Tests cover:
- Real metadata loading
- Invalid metadata rejection
- Missing audio rejection  
- Duplicate detection
- Speaker-disjoint splitting
- Split integrity
- Feature tensor shape
- No synthetic fallback enforcement
- Model forward pass
- Checkpoint save/load
- Deterministic seed behavior
- Model status detection
"""
import csv
import io
import json
import os
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from training.dataset import (
    AudioRecord, AntiSpoofDataset, dataset_statistics, load_metadata,
    normalize_label, validate_dataset, REQUIRED_COLUMNS
)
from training.splits import make_speaker_disjoint_splits, split_integrity_issues
from training.train import set_seed


# ── Fixtures ────────────────────────────────────────────────────────────────────

def _write_wav(path: Path, *, seconds: float = 1.0, amplitude: int = 5000, sr: int = 16000) -> None:
    """Write a minimal valid WAV fixture (not suitable for training, only unit tests)."""
    frames = max(1, int(sr * seconds))
    t = np.linspace(0, seconds, frames, dtype=np.float32)
    signal = (np.sin(2 * np.pi * 220 * t) * amplitude).astype(np.int16)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sr)
        out.writeframes(signal.tobytes())


def _metadata_row(filepath: str, speaker: str, label: int = 1, split: str = "train",
                  dataset_source: str = "real_corpus") -> dict:
    return {
        "filepath": filepath,
        "label": label,
        "speaker_id": speaker,
        "dataset_source": dataset_source,
        "attack_type": "bona_fide" if label else "tts",
        "generator_id": "none" if label else "fixture_tts",
        "language": "en",
        "channel": "microphone",
        "split": split,
    }


def _write_manifest(path: Path, rows: list[dict]) -> None:
    fields = list(REQUIRED_COLUMNS)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ── 1. Real metadata loading ────────────────────────────────────────────────────

def test_real_metadata_loading():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        manifest = root / "meta.csv"
        _write_manifest(manifest, [
            _metadata_row("data/raw/spk01_01.wav", "spk01", label=1),
            _metadata_row("data/raw/spk02_01.wav", "spk02", label=0),
        ])
        records = load_metadata(manifest)
        assert len(records) == 2
        assert records[0].label == 1
        assert records[1].label == 0
        assert records[0].speaker_id == "spk01"


# ── 2. Invalid metadata rejection ──────────────────────────────────────────────

def test_invalid_metadata_missing_columns():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        manifest = root / "bad_meta.csv"
        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "label"])
            writer.writeheader()
            writer.writerow({"filepath": "a.wav", "label": "1"})
        with pytest.raises(ValueError, match="missing required columns"):
            load_metadata(manifest)


def test_invalid_label_rejected():
    with pytest.raises(ValueError, match="label must identify"):
        normalize_label("unknown_label")


# ── 3. Missing audio rejection ─────────────────────────────────────────────────

def test_missing_audio_flagged_in_validation():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        manifest = root / "meta.csv"
        _write_manifest(manifest, [
            _metadata_row("nonexistent/file.wav", "spk01"),
        ])
        report = validate_dataset(manifest, root)
        codes = {issue.code for issue in report.issues}
        assert "missing_file" in codes
        assert not report.valid


# ── 4. Duplicate detection ─────────────────────────────────────────────────────

def test_duplicate_filepath_flagged():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_wav(root / "audio.wav")
        manifest = root / "meta.csv"
        _write_manifest(manifest, [
            _metadata_row("audio.wav", "spk01"),
            _metadata_row("audio.wav", "spk02"),  # Same file, different speaker
        ])
        report = validate_dataset(manifest, root)
        codes = {issue.code for issue in report.issues}
        assert "duplicate_filepath" in codes


def test_duplicate_audio_content_flagged():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_wav(root / "audio1.wav")
        import shutil
        shutil.copy(root / "audio1.wav", root / "audio2.wav")  # Same content, different name
        manifest = root / "meta.csv"
        _write_manifest(manifest, [
            _metadata_row("audio1.wav", "spk01"),
            _metadata_row("audio2.wav", "spk02"),
        ])
        report = validate_dataset(manifest, root)
        codes = {issue.code for issue in report.issues}
        assert "duplicate_audio" in codes


# ── 5 & 6. Speaker-disjoint splitting and split integrity ──────────────────────

def test_speaker_disjoint_splits():
    records = [
        AudioRecord(f"audio-{i}.wav", i % 2, f"speaker-{i}", "corpus", "bona_fide", "none", "en", "mic")
        for i in range(9)
    ]
    assigned = make_speaker_disjoint_splits(records, seed=42)
    train_speakers = {r.speaker_id for r in assigned if r.split == "train"}
    val_speakers = {r.speaker_id for r in assigned if r.split == "validation"}
    test_speakers = {r.speaker_id for r in assigned if r.split == "test"}

    assert train_speakers.isdisjoint(val_speakers), "Train/val speaker leak"
    assert train_speakers.isdisjoint(test_speakers), "Train/test speaker leak"
    assert val_speakers.isdisjoint(test_speakers), "Val/test speaker leak"
    assert not split_integrity_issues(assigned), "Split integrity issues found"


def test_augmentation_leakage_detected():
    leaked = [
        AudioRecord("a.wav", 1, "spk-a", "corpus", "bona_fide", "none", "en", "mic", "train", augmentation_of="orig-1"),
        AudioRecord("b.wav", 0, "spk-b", "corpus", "tts", "sys-a", "en", "mic", "test", augmentation_of="orig-1"),
    ]
    issues = split_integrity_issues(leaked)
    assert any("source/augmentation overlap" in issue for issue in issues)


# ── 7. Dataset item loading (unit fixture) ─────────────────────────────────────

def test_dataset_item_loads_with_real_audio():
    import torch
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        raw_dir = root / "data" / "raw"
        raw_dir.mkdir(parents=True)
        _write_wav(raw_dir / "spk01_real.wav", seconds=3.0)
        _write_wav(raw_dir / "spk02_fake.wav", seconds=3.0)

        manifest = root / "manifest.csv"
        _write_manifest(manifest, [
            _metadata_row(str(raw_dir / "spk01_real.wav"), "spk01", label=1, split="train"),
            _metadata_row(str(raw_dir / "spk02_fake.wav"), "spk02", label=0, split="train"),
        ])

        dataset = AntiSpoofDataset(manifest, split="train", root_dir=root)
        assert len(dataset) == 2
        tensor, label, meta = dataset[0]
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert isinstance(meta, dict)
        assert "speaker_id" in meta


# ── 8. Feature tensor shape ────────────────────────────────────────────────────

def test_feature_tensor_shape():
    import torch
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        raw_dir = root / "data" / "raw"
        raw_dir.mkdir(parents=True)
        _write_wav(raw_dir / "spk01_real.wav", seconds=3.0)
        manifest = root / "manifest.csv"
        _write_manifest(manifest, [
            _metadata_row(str(raw_dir / "spk01_real.wav"), "spk01", label=1, split="train"),
        ])
        dataset = AntiSpoofDataset(manifest, split="train", root_dir=root)
        tensor, _, _ = dataset[0]
        # Expects (1, N_MFCC, N_TIME_STEPS) = (1, 13, 64)
        assert tensor.shape == (1, 13, 64), f"Unexpected shape: {tensor.shape}"


# ── 9. No synthetic fallback ───────────────────────────────────────────────────

def test_demo_dataset_rejected():
    """Training must fail if the manifest only contains DEMO_DATA fixtures."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        raw_dir = root / "data" / "raw"
        raw_dir.mkdir(parents=True)
        _write_wav(raw_dir / "demo_file.wav", seconds=3.0)
        manifest = root / "manifest.csv"
        _write_manifest(manifest, [
            {
                "filepath": str(raw_dir / "demo_file.wav"),
                "label": "1", "speaker_id": "spk01",
                "dataset_source": "DEMO_DATA_NOT_FOR_EVALUATION",
                "attack_type": "bona_fide", "generator_id": "demo_tone_generator",
                "language": "en", "channel": "microphone", "split": "train",
            }
        ])
        with pytest.raises(ValueError, match="REAL DATASET REQUIRED"):
            AntiSpoofDataset(manifest, split="train", root_dir=root)


# ── 10. Model forward pass ─────────────────────────────────────────────────────

def test_model_forward_pass():
    import torch
    from backend.detection.detector import SimpleCNNDetector
    model = SimpleCNNDetector()
    model.eval()
    # Batch of 2, (1, 13, 64)
    x = torch.randn(2, 1, 13, 64)
    with torch.no_grad():
        out = torch.sigmoid(model(x))
    assert out.shape == (2, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Sigmoid outputs should be in [0, 1]"


# ── 11. Checkpoint save/load ───────────────────────────────────────────────────

def test_checkpoint_save_and_load():
    import torch
    from backend.detection.detector import SimpleCNNDetector
    with tempfile.TemporaryDirectory() as d:
        out_path = Path(d) / "test_model.pth"

        # Train a tiny model, save it
        model = SimpleCNNDetector()
        torch.save(model.state_dict(), out_path)
        assert out_path.exists()

        # Load it back and verify structure
        loaded = SimpleCNNDetector()
        loaded.load_state_dict(torch.load(out_path, map_location="cpu", weights_only=True))
        
        model.eval()
        loaded.eval()

        x = torch.randn(1, 1, 13, 64)
        with torch.no_grad():
            original_out = model(x)
            loaded_out = loaded(x)
        assert torch.allclose(original_out, loaded_out, atol=1e-6), "Loaded model should produce same outputs"


# ── 12. Deterministic seed behavior ──────────────────────────────────────────

def test_deterministic_seed():
    """Same seed should produce identical model weight initialization."""
    import torch
    from backend.detection.detector import SimpleCNNDetector
    set_seed(42)
    model_a = SimpleCNNDetector()

    set_seed(42)
    model_b = SimpleCNNDetector()

    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert torch.allclose(param_a, param_b), f"Parameter {name_a} differs between same-seed models"


def test_different_seed_produces_different_weights():
    import torch
    from backend.detection.detector import SimpleCNNDetector
    set_seed(1)
    model_a = SimpleCNNDetector()

    set_seed(999)
    model_b = SimpleCNNDetector()

    # With overwhelmingly high probability, random init with different seeds differs
    first_a = next(model_a.parameters())
    first_b = next(model_b.parameters())
    assert not torch.allclose(first_a, first_b), "Different seeds should (almost certainly) differ"
