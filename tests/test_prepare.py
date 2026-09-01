import csv
import json
import subprocess
import tempfile
import wave
from pathlib import Path
import sys

import numpy as np


def _write_wav(path: Path, *, seconds: float = 0.5, amplitude: int = 1000, freq: float = 440.0) -> None:
    frames = max(1, int(16_000 * seconds))
    t = np.linspace(0, seconds, frames, endpoint=False)
    signal = (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.int16)
    samples = signal.tobytes()
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


def test_prepare_script_success():
    """Test that the prepare script runs successfully with valid data."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_wav(root / "1.wav", freq=220.0)
        _write_wav(root / "2.wav", freq=440.0)
        _write_wav(root / "3.wav", freq=880.0)
        
        fields = ["filepath", "label", "speaker_id", "dataset_source", "attack_type", "generator_id", "language", "channel", "split"]
        rows = [
            _metadata_row("1.wav", "speaker-a", 1),
            _metadata_row("2.wav", "speaker-b", 0),
            _metadata_row("3.wav", "speaker-c", 1),
        ]
        
        metadata = root / "metadata.csv"
        with metadata.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            
        output_manifest = root / "splits/manifest.csv"
        
        result = subprocess.run([
            sys.executable, "data/prepare.py",
            "--metadata", str(metadata),
            "--root", str(root),
            "--output", str(output_manifest)
        ], capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
        
        assert result.returncode == 0
        assert "Dataset validation successful." in result.stdout
        assert "Generating speaker-disjoint splits..." in result.stdout
        assert "Dataset Preparation Complete." in result.stdout
        assert "Dataset Statistics:" in result.stdout
        assert output_manifest.exists()
        

def test_prepare_script_validation_failure():
    """Test that the prepare script exits with 1 on invalid data."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        # missing files
        
        fields = ["filepath", "label", "speaker_id", "dataset_source", "attack_type", "generator_id", "language", "channel", "split"]
        rows = [
            _metadata_row("missing.wav", "speaker-a", 1),
        ]
        
        metadata = root / "metadata.csv"
        with metadata.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            
        output_manifest = root / "splits/manifest.csv"
        
        result = subprocess.run([
            sys.executable, "data/prepare.py",
            "--metadata", str(metadata),
            "--root", str(root),
            "--output", str(output_manifest)
        ], capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
        
        assert result.returncode == 1
        assert "Dataset validation failed." in result.stderr
        assert "missing_file" in result.stderr
        assert not output_manifest.exists()
