"""
VigilVoice - Synthetic Audio Data Generator
============================================
[DEMO ONLY] — NOT FOR MODEL TRAINING OR EVALUATION CLAIMS.

Generates waveform fixtures for dashboard and integration-test demonstrations.

REAL samples: Multi-harmonic speech-like signals with natural pitch variation,
              amplitude modulation, and gaussian noise — mimicking human voice.

FAKE samples: Monotone, narrow-bandwidth robotic tones with little variance,
              simulating TTS/vocoder artifacts typical of cloned speech.

Usage:
    python data/generate_synthetic_data.py
"""

import os
import sys
import csv
import numpy as np
from scipy.io.wavfile import write

# Ensure we run from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

RAW_DIR = os.path.join(project_root, "data", "raw")
META_CSV = os.path.join(project_root, "data", "metadata.csv")
SAMPLE_RATE = 16000


def generate_real_audio(duration_sec: float, base_freq: float = None) -> np.ndarray:
    """
    Simulate authentic human voice characteristics:
    - Multiple harmonics (fundamental + overtones)
    - Natural pitch jitter (~1-3% variation)
    - Amplitude envelope (rising/falling like a syllable)
    - Background gaussian noise
    """
    if base_freq is None:
        # Random fundamental frequency: male ~85-180 Hz, female ~165-255 Hz
        base_freq = np.random.uniform(100, 240)

    num_samples = int(SAMPLE_RATE * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Jitter: small pitch micro-variation per frame
    jitter_rate = np.random.uniform(0.01, 0.03)
    freq_variation = base_freq + jitter_rate * base_freq * np.sin(2 * np.pi * 3 * t)

    # Build signal from fundamental + harmonics
    audio = np.zeros(num_samples)
    num_harmonics = np.random.randint(4, 8)
    for harmonic in range(1, num_harmonics + 1):
        amplitude = 1.0 / harmonic  # Natural harmonic roll-off
        phase = np.random.uniform(0, 2 * np.pi)
        audio += amplitude * np.sin(2 * np.pi * (freq_variation * harmonic) * t + phase)

    # Apply syllable-like amplitude envelope
    envelope = np.hanning(num_samples)
    audio *= envelope

    # Add gaussian noise (natural microphone + room noise)
    noise_level = np.random.uniform(0.02, 0.08)
    audio += noise_level * np.random.randn(num_samples)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.85

    return audio.astype(np.float32)


def generate_fake_audio(duration_sec: float) -> np.ndarray:
    """
    Simulate synthetic/TTS voice characteristics:
    - Near-monotone pitch (minimal variation)
    - Very few harmonics (narrow bandwidth)
    - Periodic and mechanical waveform patterns
    - Low spectral variance (flatter MFCC distribution)
    """
    num_samples = int(SAMPLE_RATE * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # TTS typically has a fixed, unvarying pitch
    base_freq = np.random.uniform(140, 180)  # Narrow mid-range

    # Very small pitch variation — robotic stability
    jitter_rate = np.random.uniform(0.001, 0.005)
    freq_variation = base_freq + jitter_rate * base_freq * np.sin(2 * np.pi * 1.5 * t)

    # Only 2-3 harmonics — narrow spectrum typical of vocoders
    audio = np.zeros(num_samples)
    num_harmonics = np.random.randint(2, 4)
    for harmonic in range(1, num_harmonics + 1):
        amplitude = 1.0 / harmonic
        audio += amplitude * np.sin(2 * np.pi * (freq_variation * harmonic) * t)

    # Flat box envelope (no natural rise/fall)
    envelope = np.ones(num_samples)
    # Apply very slight fade-in/out only at edges
    fade_len = int(0.02 * SAMPLE_RATE)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    audio *= envelope

    # Minimal background noise (TTS is much cleaner than real recordings)
    noise_level = np.random.uniform(0.002, 0.01)
    audio += noise_level * np.random.randn(num_samples)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.85

    return audio.astype(np.float32)


def save_wav(filepath: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
    """Save float32 audio array as 16-bit PCM WAV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    audio_int16 = np.int16(audio * 32767)
    write(filepath, sr, audio_int16)


def generate_all_samples():
    """Read metadata.csv and generate all listed WAV files."""
    print("=" * 55)
    print("  VigilVoice — Synthetic Audio Data Generator")
    print("  [WARNING] DEMO DATA ONLY - NOT FOR MODEL TRAINING")
    print("=" * 55)

    if not os.path.exists(META_CSV):
        print(f"[ERROR] metadata.csv not found at: {META_CSV}")
        sys.exit(1)

    generated_real = 0
    generated_fake = 0
    skipped = 0

    with open(META_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        filepath = os.path.join(project_root, row["filepath"])
        label = int(row["label"])
        duration = float(row.get("duration_sec", 3.0))

        if os.path.exists(filepath):
            print(f"  [SKIP] Already exists: {os.path.basename(filepath)}")
            skipped += 1
            continue

        if label == 1:
            audio = generate_real_audio(duration)
            tag = "REAL"
            generated_real += 1
        else:
            audio = generate_fake_audio(duration)
            tag = "FAKE"
            generated_fake += 1

        save_wav(filepath, audio)
        print(f"  [GEN]  {tag} -> {os.path.basename(filepath)} ({duration:.1f}s)")

    print("=" * 55)
    print(f"  Generated: {generated_real} REAL, {generated_fake} FAKE samples")
    if skipped:
        print(f"  Skipped (already exist): {skipped}")
    print(f"  Output directory: {RAW_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    generate_all_samples()
