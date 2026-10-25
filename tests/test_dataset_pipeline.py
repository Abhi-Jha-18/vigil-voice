import csv
import wave
from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf
import torch

from backend.audio.features import HOP_LENGTH, N_FFT, N_MELS, N_MFCC, _cmn, mfcc_crops
from backend.audio.pipeline import process_canonical
from training.dataset import (
    AntiSpoofDataset,
    AudioRecord,
    dataset_statistics,
    hash_decoded_audio,
    validate_dataset,
)
from training.splits import (
    make_generator_disjoint_splits,
    make_speaker_disjoint_splits,
    split_integrity_issues,
    write_metadata,
)


def _write_wav(path: Path, *, seconds: float = 0.5, amplitude: int = 1000, seed: int = 42) -> None:
    frames = max(1, int(16_000 * seconds))
    if amplitude == 0:
        samples = np.zeros(frames, dtype=np.int16).tobytes()
    else:
        rng = np.random.RandomState(seed)
        t = np.linspace(0, seconds, frames, endpoint=False, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t)
        noise = rng.normal(0, 0.05, frames)
        samples = ((sine + noise) * (amplitude / 1.1)).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(samples)


def _metadata_row(filepath: str, speaker: str, label: int = 1) -> dict[str, str | int]:
    return {
        "filepath": filepath,
        "label": label,
        "speaker_id": speaker,
        "dataset_source": "fixture",
        "attack_type": "bona_fide" if label else "tts",
        "generator_id": "none" if label else "fixture_tts",
        "language": "en",
        "channel": "microphone",
        "split": "",
    }


def test_dataset_validation_detects_audio_and_metadata_problems(tmp_path: Path):
    root = tmp_path
    _write_wav(root / "good.wav")
    _write_wav(root / "silent.wav", amplitude=0)
    _write_wav(root / "short.wav", seconds=0.05)
    (root / "corrupt.wav").write_bytes(b"not audio")
    fields = ["filepath", "label", "speaker_id", "dataset_source", "attack_type", "generator_id", "language", "channel", "split"]
    rows = [
        _metadata_row("good.wav", "speaker-a"),
        _metadata_row("silent.wav", "speaker-b"),
        _metadata_row("short.wav", "speaker-c"),
        _metadata_row("corrupt.wav", "speaker-d"),
        _metadata_row("missing.wav", "speaker-e"),
        _metadata_row("good.txt", "speaker-f"),
        _metadata_row("good.wav", "speaker-g"),
    ]
    metadata = root / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = validate_dataset(metadata, root)
    codes = {issue.code for issue in report.issues}
    assert {"silence_only", "sample_too_short", "corrupt_audio", "missing_file", "unsupported_format", "duplicate_filepath"} <= codes
    assert not report.valid

    issues_by_code = {issue.code: issue.filepath for issue in report.issues}
    assert issues_by_code.get("silence_only") == "silent.wav"
    assert issues_by_code.get("sample_too_short") == "short.wav"
    assert issues_by_code.get("corrupt_audio") == "corrupt.wav"
    assert issues_by_code.get("missing_file") == "missing.wav"
    assert issues_by_code.get("unsupported_format") == "good.txt"
    assert issues_by_code.get("duplicate_filepath") == "good.wav"


def test_speaker_and_source_disjoint_splits_and_statistics():
    records = [
        AudioRecord(
            filepath=f"audio-{index}.wav",
            label=index % 2,
            speaker_id=f"speaker-{index}",
            dataset_source="fixture",
            attack_type="tts" if index % 2 == 0 else "bona_fide",
            generator_id="generator-a",
            language="en",
            channel="mic",
            source_id=f"source-{index}",
        )
        for index in range(12)
    ]
    assigned = make_speaker_disjoint_splits(records, seed=10)
    assert not split_integrity_issues(assigned)
    assert {record.split for record in assigned} == {"train", "validation", "test"}
    stats = dataset_statistics(assigned)
    assert stats["samples"] == 12
    assert stats["attack_types"]["tts"] == 6

    leaked = [
        AudioRecord(
            filepath="a.wav",
            label=1,
            speaker_id="speaker-a",
            dataset_source="fixture",
            attack_type="bona_fide",
            generator_id="none",
            language="en",
            channel="mic",
            split="train",
            augmentation_of="original-1",
        ),
        AudioRecord(
            filepath="b.wav",
            label=0,
            speaker_id="speaker-b",
            dataset_source="fixture",
            attack_type="replay",
            generator_id="replay-a",
            language="en",
            channel="mic",
            split="test",
            augmentation_of="original-1",
        ),
    ]
    assert any("source/augmentation overlap" in issue for issue in split_integrity_issues(leaked))


def test_split_integrity_fails_when_split_lacks_class():
    """split_integrity_issues must fail if any of train/validation/test lacks either class."""
    records = [
        AudioRecord(filepath="tr1.wav", label=1, speaker_id="s1", dataset_source="f", attack_type="bona_fide", generator_id="none", language="en", channel="mic", split="train"),
        AudioRecord(filepath="tr0.wav", label=0, speaker_id="s2", dataset_source="f", attack_type="tts", generator_id="g1", language="en", channel="mic", split="train"),
        # validation lacks label 0
        AudioRecord(filepath="v1.wav", label=1, speaker_id="s3", dataset_source="f", attack_type="bona_fide", generator_id="none", language="en", channel="mic", split="validation"),
        AudioRecord(filepath="te1.wav", label=1, speaker_id="s4", dataset_source="f", attack_type="bona_fide", generator_id="none", language="en", channel="mic", split="test"),
        AudioRecord(filepath="te0.wav", label=0, speaker_id="s5", dataset_source="f", attack_type="tts", generator_id="g1", language="en", channel="mic", split="test"),
    ]
    issues = split_integrity_issues(records)
    assert any("validation" in issue and "class 0" in issue for issue in issues)


def test_generator_disjoint_unseen_attack_split():
    """Generator-disjoint unseen attack test split using generator_id."""
    records = [
        AudioRecord(
            filepath=f"real-{i}.wav",
            label=1,
            speaker_id=f"spk-real-{i}",
            dataset_source="fixture",
            attack_type="bona_fide",
            generator_id="none",
            language="en",
            channel="mic",
        )
        for i in range(6)
    ] + [
        AudioRecord(
            filepath=f"seen-{i}.wav",
            label=0,
            speaker_id=f"spk-seen-{i}",
            dataset_source="fixture",
            attack_type="tts",
            generator_id="gen-seen",
            language="en",
            channel="mic",
        )
        for i in range(6)
    ] + [
        AudioRecord(
            filepath=f"unseen-{i}.wav",
            label=0,
            speaker_id=f"spk-unseen-{i}",
            dataset_source="fixture",
            attack_type="tts",
            generator_id="gen-unseen",
            language="en",
            channel="mic",
        )
        for i in range(4)
    ]
    assigned = make_generator_disjoint_splits(records, seed=42, unseen_generators=["gen-unseen"])
    unseen_records = [r for r in assigned if r.split == "test_unseen"]
    assert len(unseen_records) == 4
    assert all(r.generator_id == "gen-unseen" for r in unseen_records)

    train_gens = {r.generator_id for r in assigned if r.split == "train" and r.label == 0}
    assert "gen-unseen" not in train_gens
    assert "gen-seen" in train_gens


def test_content_hash_duplicate_across_splits(tmp_path: Path):
    """Content-hash duplicate check across splits by hashing decoded audio."""
    root = tmp_path
    f1 = root / "train_audio.wav"
    f2 = root / "test_audio.wav"
    _write_wav(f1, seed=123)
    # Write identical audio signal to f2 (different path, same decoded samples)
    _write_wav(f2, seed=123)

    h1 = hash_decoded_audio(f1)
    h2 = hash_decoded_audio(f2)
    assert h1 == h2

    records = [
        AudioRecord(filepath=str(f1), label=1, speaker_id="s1", dataset_source="f", attack_type="bona_fide", generator_id="none", language="en", channel="mic", split="train"),
        AudioRecord(filepath="tr0.wav", label=0, speaker_id="s2", dataset_source="f", attack_type="tts", generator_id="g1", language="en", channel="mic", split="train"),
        AudioRecord(filepath="v1.wav", label=1, speaker_id="s3", dataset_source="f", attack_type="bona_fide", generator_id="none", language="en", channel="mic", split="validation"),
        AudioRecord(filepath="v0.wav", label=0, speaker_id="s4", dataset_source="f", attack_type="tts", generator_id="g1", language="en", channel="mic", split="validation"),
        AudioRecord(filepath=str(f2), label=1, speaker_id="s5", dataset_source="f", attack_type="bona_fide", generator_id="none", language="en", channel="mic", split="test"),
        AudioRecord(filepath="te0.wav", label=0, speaker_id="s6", dataset_source="f", attack_type="tts", generator_id="g1", language="en", channel="mic", split="test"),
    ]
    content_hashes = {str(f1): h1, str(f2): h2}
    issues = split_integrity_issues(records, content_hashes=content_hashes)
    assert any("duplicate content hash overlap across splits" in issue and h1 in issue for issue in issues)


def test_feature_parity_training_vs_inference(tmp_path: Path):
    """Same audio file through training dataset path and inference path gives identical tensors within tolerance."""
    root = tmp_path
    audio_path = root / "test_parity.wav"
    _write_wav(audio_path, seconds=2.0, seed=99)

    record = AudioRecord(
        filepath=str(audio_path),
        label=1,
        speaker_id="parity_speaker",
        dataset_source="fixture",
        attack_type="bona_fide",
        generator_id="none",
        language="en",
        channel="mic",
        split="train",
    )
    manifest = root / "manifest.csv"
    write_metadata([record], manifest)

    # 1. Training path
    dataset = AntiSpoofDataset(manifest, split="train", root_dir=root, augment=False)
    training_tensor, _, _ = dataset[0]

    # 2. Inference path
    prep_res = process_canonical(str(audio_path))
    assert prep_res.quality.status in ("GOOD", "DEGRADED")
    y_inf = prep_res.segments[0].audio_data.astype(np.float32, copy=False)
    raw_mfcc = librosa.feature.mfcc(
        y=y_inf,
        sr=16000,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    cmn_mfcc = _cmn(raw_mfcc)
    inf_crops = mfcc_crops(cmn_mfcc, n=64, hop=32)
    inference_tensor = torch.from_numpy(np.ascontiguousarray(inf_crops[0], dtype=np.float32)).unsqueeze(0)

    assert training_tensor.shape == inference_tensor.shape
    assert torch.allclose(training_tensor, inference_tensor, atol=1e-5, rtol=1e-5)


def test_evaluate_seen_vs_unseen_attacks(tmp_path: Path):
    """evaluate.py must report metrics for seen vs unseen attacks separately."""
    import json
    from types import SimpleNamespace
    from backend.detection.detector import SimpleCNNDetector
    from training.evaluate import evaluate_model

    root = tmp_path
    chk_path = root / "model.pth"
    model = SimpleCNNDetector()
    torch.save(model.state_dict(), chk_path)

    # Create audio files
    real_wav = root / "real.wav"
    seen_wav = root / "seen.wav"
    unseen_wav = root / "unseen.wav"
    _write_wav(real_wav, seconds=1.0, seed=1)
    _write_wav(seen_wav, seconds=1.0, seed=2)
    _write_wav(unseen_wav, seconds=1.0, seed=3)

    records = [
        AudioRecord(filepath=str(real_wav), label=1, speaker_id="s1", dataset_source="f", attack_type="bona_fide", generator_id="none", language="en", channel="mic", split="train"),
        AudioRecord(filepath=str(seen_wav), label=0, speaker_id="s2", dataset_source="f", attack_type="tts", generator_id="gen_seen", language="en", channel="mic", split="train"),
        # Test split with bonafide, seen attack, and unseen attack
        AudioRecord(filepath=str(real_wav), label=1, speaker_id="s3", dataset_source="f", attack_type="bona_fide", generator_id="none", language="en", channel="mic", split="test"),
        AudioRecord(filepath=str(seen_wav), label=0, speaker_id="s4", dataset_source="f", attack_type="tts", generator_id="gen_seen", language="en", channel="mic", split="test"),
        AudioRecord(filepath=str(unseen_wav), label=0, speaker_id="s5", dataset_source="f", attack_type="tts", generator_id="gen_unseen", language="en", channel="mic", split="test"),
    ]
    manifest = root / "manifest.csv"
    write_metadata(records, manifest)

    out_report = root / "eval_report.json"
    args = SimpleNamespace(
        manifest=manifest,
        checkpoint=chk_path,
        output=out_report,
        batch_size=4,
        device="cpu",
        split="test",
        allow_synthetic=True,
        deploy=False,
        max_eer=0.05,
    )
    evaluate_model(args)

    assert out_report.is_file()
    with open(out_report) as f:
        data = json.load(f)

    assert "seen_attacks" in data
    assert "unseen_attacks" in data
    assert data["seen_attacks"]["samples"] == 1
    assert data["unseen_attacks"]["samples"] == 1
    assert "accuracy" in data["seen_attacks"]
    assert "eer" in data["seen_attacks"]
    assert "accuracy" in data["unseen_attacks"]
    assert "eer" in data["unseen_attacks"]
