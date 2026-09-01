import csv
import tempfile
import wave
from pathlib import Path

import numpy as np

from training.dataset import AudioRecord, dataset_statistics, validate_dataset
from training.splits import make_speaker_disjoint_splits, split_integrity_issues


def _write_wav(path: Path, *, seconds: float = 0.5, amplitude: int = 1000) -> None:
    frames = max(1, int(16_000 * seconds))
    samples = (np.ones(frames, dtype=np.int16) * amplitude).tobytes()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(samples)


def _metadata_row(filepath: str, speaker: str, label: int = 1) -> dict[str, str | int]:
    return {
        "filepath": filepath, "label": label, "speaker_id": speaker,
        "dataset_source": "fixture", "attack_type": "bona_fide" if label else "tts",
        "generator_id": "none" if label else "fixture_tts", "language": "en",
        "channel": "microphone", "split": "",
    }


def test_dataset_validation_detects_audio_and_metadata_problems():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
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


def test_speaker_and_source_disjoint_splits_and_statistics():
    records = [
        AudioRecord(f"audio-{index}.wav", index % 2, f"speaker-{index}", "fixture", "tts" if index % 2 == 0 else "bona_fide", "generator-a", "en", "mic", source_id=f"source-{index}")
        for index in range(6)
    ]
    assigned = make_speaker_disjoint_splits(records, seed=10)
    assert not split_integrity_issues(assigned)
    assert {record.split for record in assigned} == {"train", "validation", "test"}
    stats = dataset_statistics(assigned)
    assert stats["samples"] == 6
    assert stats["attack_types"]["tts"] == 3

    leaked = [
        AudioRecord("a.wav", 1, "speaker-a", "fixture", "bona_fide", "none", "en", "mic", "train", augmentation_of="original-1"),
        AudioRecord("b.wav", 0, "speaker-b", "fixture", "replay", "replay-a", "en", "mic", "test", augmentation_of="original-1"),
    ]
    assert any("source/augmentation overlap" in issue for issue in split_integrity_issues(leaked))
