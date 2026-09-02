"""
VigilVoice - Synthetic Audio Data Generator
============================================
Generates waveform fixtures for dashboard demos AND a larger local training
corpus used to fit the CNN when a licensed anti-spoof dataset is not present.

REAL samples: formant-synthesized speech-like signals with pitch jitter,
              moving vowels, unvoiced bursts, and microphone noise.

FAKE samples: vocoder / neural-TTS / replay / voice-conversion artifacts —
              stable F0, few harmonics, frame-periodic buzz, band-limiting.

Usage:
    python data/generate_synthetic_data.py
    python data/generate_synthetic_data.py --speakers 24 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from scipy.io.wavfile import write
from scipy.signal import lfilter, butter, resample_poly

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

RAW_DIR = os.path.join(project_root, "data", "raw")
META_CSV = os.path.join(project_root, "data", "metadata.csv")
MANIFEST_CSV = os.path.join(project_root, "data", "splits", "manifest.csv")
SAMPLE_RATE = 16000

VOWELS = {
    "a": (730.0, 1090.0, 2440.0),
    "e": (530.0, 1840.0, 2480.0),
    "i": (270.0, 2290.0, 3010.0),
    "o": (570.0, 840.0, 2410.0),
    "u": (300.0, 870.0, 2240.0),
}


def _normalize(audio: np.ndarray, peak: float = 0.85) -> np.ndarray:
    m = float(np.max(np.abs(audio)))
    if m > 0:
        audio = audio / m * peak
    return audio.astype(np.float32)


def _formant_filter(signal: np.ndarray, f1: float, f2: float, f3: float, sr: int) -> np.ndarray:
    """Cascade three two-pole resonators (Klatt-style formants)."""
    y = signal
    for freq, bw in ((f1, 90.0), (f2, 110.0), (f3, 170.0)):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * freq / sr
        b = [1.0]
        a = [1.0, -2.0 * r * np.cos(theta), r * r]
        y = lfilter(b, a, y)
    return y


def _glottal_source(f0: np.ndarray, sr: int, n_harmonics: int) -> np.ndarray:
    phase = np.cumsum(2.0 * np.pi * f0 / sr)
    audio = np.zeros_like(f0)
    for h in range(1, n_harmonics + 1):
        audio += (1.0 / h) * np.sin(h * phase)
    return audio


def generate_real_audio(duration_sec: float, rng: np.random.Generator) -> np.ndarray:
    """Natural-ish speech: moving formants, jitter, unvoiced bursts, noise."""
    n = int(SAMPLE_RATE * duration_sec)
    t = np.arange(n) / SAMPLE_RATE

    f0_base = float(rng.uniform(95, 230))
    intonation = 0.10 * np.sin(2 * np.pi * rng.uniform(0.4, 1.6) * t)
    vibrato = 0.025 * np.sin(2 * np.pi * rng.uniform(3.5, 6.5) * t)
    jitter = rng.normal(0.0, 0.018, size=n)
    f0 = np.clip(f0_base * (1.0 + intonation + vibrato + jitter), 70.0, 380.0)

    n_harmonics = int(rng.integers(8, 16))
    source = _glottal_source(f0, SAMPLE_RATE, n_harmonics)

    vowel_names = list(VOWELS.keys())
    chunk = max(int(0.16 * SAMPLE_RATE), 1)
    filtered = np.zeros(n, dtype=np.float64)
    pos = 0
    while pos < n:
        end = min(pos + chunk + int(rng.integers(-400, 800)), n)
        f1, f2, f3 = VOWELS[vowel_names[int(rng.integers(0, len(vowel_names)))]]
        f1 *= float(rng.uniform(0.92, 1.08))
        f2 *= float(rng.uniform(0.92, 1.08))
        piece = _formant_filter(source[pos:end], f1, f2, f3, SAMPLE_RATE)
        # short overlap-add to avoid clicks
        if pos > 0:
            fade = min(160, len(piece), pos)
            ramp = np.linspace(0, 1, fade)
            filtered[pos:pos + fade] = filtered[pos:pos + fade] * (1 - ramp) + piece[:fade] * ramp
            filtered[pos + fade:end] = piece[fade:]
        else:
            filtered[pos:end] = piece
        pos = end

    # Syllable-like amplitude modulation
    syll_hz = float(rng.uniform(3.0, 6.0))
    envelope = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * syll_hz * t))
    fade = int(0.03 * SAMPLE_RATE)
    envelope[:fade] *= np.linspace(0, 1, fade)
    envelope[-fade:] *= np.linspace(1, 0, fade)
    audio = filtered * envelope

    # Unvoiced consonant bursts (fricative-like noise)
    n_bursts = int(rng.integers(2, 5))
    for _ in range(n_bursts):
        b0 = int(rng.integers(0, max(1, n - 800)))
        b1 = min(n, b0 + int(rng.integers(400, 1200)))
        noise = rng.normal(0, 0.35, size=b1 - b0)
        b, a = butter(2, 1800 / (SAMPLE_RATE / 2), btype="high")
        audio[b0:b1] += lfilter(b, a, noise) * np.hanning(b1 - b0)

    audio += rng.normal(0, float(rng.uniform(0.025, 0.07)), size=n)
    return _normalize(audio)


def generate_fake_vocoder(duration_sec: float, rng: np.random.Generator) -> np.ndarray:
    """Pulse-vocoder: almost-fixed F0, few harmonics, 10 ms frame buzz."""
    n = int(SAMPLE_RATE * duration_sec)
    t = np.arange(n) / SAMPLE_RATE
    f0_base = float(rng.uniform(140, 185))
    f0 = f0_base * (1.0 + 0.002 * np.sin(2 * np.pi * 1.2 * t))
    n_harmonics = int(rng.integers(2, 5))
    audio = _glottal_source(f0, SAMPLE_RATE, n_harmonics)

    f1, f2, f3 = VOWELS["e"]
    audio = _formant_filter(audio, f1, f2, f3, SAMPLE_RATE)

    # Frame-periodic energy (typical hop-size artifact ~10 ms)
    hop = int(0.010 * SAMPLE_RATE)
    gate = np.ones(n)
    for i in range(0, n, hop):
        gate[i:min(i + 3, n)] *= 1.35
    audio *= gate

    # Metallic residual buzz
    audio += 0.08 * np.sin(2 * np.pi * 4000 * t)
    fade = int(0.015 * SAMPLE_RATE)
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    audio *= env
    audio += rng.normal(0, 0.004, size=n)
    return _normalize(audio)


def generate_fake_tts(duration_sec: float, rng: np.random.Generator) -> np.ndarray:
    """Neural-TTS-like: overly smooth F0, regular syllables, missing fricatives."""
    n = int(SAMPLE_RATE * duration_sec)
    t = np.arange(n) / SAMPLE_RATE
    f0_base = float(rng.uniform(155, 200))
    # Perfectly smooth intonation, no jitter
    f0 = f0_base * (1.0 + 0.04 * np.sin(2 * np.pi * 0.6 * t))
    audio = _glottal_source(f0, SAMPLE_RATE, n_harmonics=6)
    f1, f2, f3 = VOWELS[["a", "o", "e"][int(rng.integers(0, 3))]]
    audio = _formant_filter(audio, f1, f2 * 0.95, f3 * 0.9, SAMPLE_RATE)

    # Overly regular syllable clock
    clock = 0.5 + 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)) ** 2
    audio *= clock
    # Spectral hole / high-frequency roll-off typical of neural vocoders
    b, a = butter(4, 3800 / (SAMPLE_RATE / 2), btype="low")
    audio = lfilter(b, a, audio)
    audio += 0.05 * np.sin(2 * np.pi * 3500 * t)
    audio += rng.normal(0, 0.003, size=n)
    return _normalize(audio)


def generate_fake_replay(duration_sec: float, rng: np.random.Generator) -> np.ndarray:
    """Replay / telephony: band-limited, mild distortion, room echo."""
    base = generate_real_audio(duration_sec, rng)
    # Band-limit like a phone channel
    b, a = butter(3, [300 / (SAMPLE_RATE / 2), 3400 / (SAMPLE_RATE / 2)], btype="band")
    audio = lfilter(b, a, base)
    # Cheap room echo
    delay = int(0.045 * SAMPLE_RATE)
    echoed = np.zeros_like(audio)
    echoed[delay:] += 0.28 * audio[:-delay]
    audio = audio + echoed
    # Soft clip (compression)
    audio = np.tanh(audio * 1.8)
    return _normalize(audio)


def generate_fake_conversion(duration_sec: float, rng: np.random.Generator) -> np.ndarray:
    """Voice-conversion: formants shifted, F0 locked, phase-vocoder roughness."""
    n = int(SAMPLE_RATE * duration_sec)
    t = np.arange(n) / SAMPLE_RATE
    f0 = np.full(n, float(rng.uniform(110, 160)))
    audio = _glottal_source(f0, SAMPLE_RATE, n_harmonics=7)
    # Shifted formants (converted speaker)
    f1, f2, f3 = 450.0, 1600.0, 2700.0
    audio = _formant_filter(audio, f1, f2, f3, SAMPLE_RATE)
    # Phase-vocoder roughness via 8 kHz downsample/upsample
    down = resample_poly(audio, 1, 2)
    audio = resample_poly(down, 2, 1)[:n]
    audio += 0.06 * np.sin(2 * np.pi * 2200 * t)
    audio += rng.normal(0, 0.006, size=n)
    return _normalize(audio)


FAKE_GENERATORS = {
    "vocoder": generate_fake_vocoder,
    "tts": generate_fake_tts,
    "replay": generate_fake_replay,
    "voice_conversion": generate_fake_conversion,
}


def save_wav(filepath: str, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    write(filepath, sr, audio_int16)


def _split_for_speaker(speaker_idx: int, n_speakers: int) -> str:
    # Speaker-disjoint: ~70% train / 15% val / 15% test
    train_end = max(1, int(round(n_speakers * 0.70)))
    val_end = max(train_end + 1, int(round(n_speakers * 0.85)))
    if speaker_idx < train_end:
        return "train"
    if speaker_idx < val_end:
        return "validation"
    return "test"


def generate_training_corpus(n_speakers: int = 24, seed: int = 42, per_class: int = 2) -> list[dict]:
    """
    Build a speaker-disjoint synthetic corpus.

    Each speaker contributes `per_class` bona-fide clips and `per_class` spoof clips.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST_CSV), exist_ok=True)

    attack_cycle = list(FAKE_GENERATORS.keys())
    rows: list[dict] = []

    print("=" * 60)
    print("  VigilVoice — Local Training Corpus Generator")
    print("  Distinctive formant-synth REAL vs vocoder/TTS/replay FAKE")
    print("=" * 60)

    for spk in range(1, n_speakers + 1):
        speaker_id = f"spk{spk:02d}"
        split = _split_for_speaker(spk - 1, n_speakers)

        for k in range(1, per_class + 1):
            duration = float(rng.uniform(2.6, 3.6))
            audio = generate_real_audio(duration, rng)
            rel = f"data/raw/real_{speaker_id}_{k:02d}.wav"
            save_wav(os.path.join(project_root, rel), audio)
            rows.append({
                "filepath": rel,
                "label": 1,
                "speaker_id": speaker_id,
                "dataset_source": "DEMO_DATA_NOT_FOR_EVALUATION",
                "attack_type": "bona_fide",
                "generator_id": "formant_synth",
                "language": "en",
                "channel": "synthetic",
                "split": split,
                "source_id": f"real_{speaker_id}_{k:02d}",
                "augmentation_of": "",
                "duration_sec": f"{duration:.2f}",
            })
            print(f"  [GEN] REAL  {os.path.basename(rel):28s} {duration:.1f}s  split={split}")

        for k in range(1, per_class + 1):
            duration = float(rng.uniform(2.6, 3.6))
            attack = attack_cycle[(spk + k) % len(attack_cycle)]
            audio = FAKE_GENERATORS[attack](duration, rng)
            rel = f"data/raw/fake_{attack}_{speaker_id}_{k:02d}.wav"
            save_wav(os.path.join(project_root, rel), audio)
            rows.append({
                "filepath": rel,
                "label": 0,
                "speaker_id": speaker_id,
                "dataset_source": "DEMO_DATA_NOT_FOR_EVALUATION",
                "attack_type": attack,
                "generator_id": f"synth_{attack}",
                "language": "en",
                "channel": "synthetic",
                "split": split,
                "source_id": f"fake_{attack}_{speaker_id}_{k:02d}",
                "augmentation_of": "",
                "duration_sec": f"{duration:.2f}",
            })
            print(f"  [GEN] FAKE  {os.path.basename(rel):28s} {duration:.1f}s  {attack:16s} split={split}")

    fieldnames = [
        "filepath", "label", "speaker_id", "dataset_source", "attack_type",
        "generator_id", "language", "channel", "split", "source_id",
        "augmentation_of", "duration_sec",
    ]
    with open(META_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest_fields = [f for f in fieldnames if f != "duration_sec"]
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in manifest_fields})

    n_real = sum(1 for r in rows if r["label"] == 1)
    n_fake = sum(1 for r in rows if r["label"] == 0)
    print("=" * 60)
    print(f"  Wrote {n_real} REAL + {n_fake} FAKE  ({len(rows)} total)")
    print(f"  metadata : {META_CSV}")
    print(f"  manifest : {MANIFEST_CSV}")
    print("=" * 60)
    return rows


def generate_all_samples():
    """Backward-compatible entry: build the training corpus."""
    generate_training_corpus()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate VigilVoice synthetic corpus")
    parser.add_argument("--speakers", type=int, default=24)
    parser.add_argument("--per-class", type=int, default=2, help="Clips per class per speaker")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_training_corpus(n_speakers=args.speakers, seed=args.seed, per_class=args.per_class)
