"""Local telephony-channel simulations shared by training and robustness tests."""

from __future__ import annotations

from functools import lru_cache
import io
import shutil
import subprocess
import warnings

import librosa
import numpy as np
import soundfile as sf

_warned_missing_codec = False

class CodecUnavailableError(RuntimeError):
    """Raised when the requested FFmpeg codec cannot be used locally."""


@lru_cache(maxsize=1)
def available_ffmpeg_codecs() -> dict[str, bool]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"g711": False, "amr": False}
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unable to inspect FFmpeg encoders: {result.stderr.strip()}")
    return {
        "g711": "pcm_mulaw" in result.stdout,
        "amr": "libopencore_amrnb" in result.stdout,
    }


def bandlimit_telephony(y: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Apply an 8 kHz telephony bandwidth by down/up-sampling."""
    if sample_rate != 16000:
        raise ValueError("Telephony band-limiting currently requires 16 kHz input.")
    if len(y) == 0:
        return y.astype(np.float32, copy=False)
    downsampled = librosa.resample(y.astype(np.float32), orig_sr=sample_rate, target_sr=8000)
    return librosa.resample(downsampled, orig_sr=8000, target_sr=sample_rate).astype(np.float32)


def simulate_ffmpeg_codec(
    y: np.ndarray,
    sample_rate: int = 16000,
    codec: str = "g711",
) -> np.ndarray:
    """Encode and decode a 16 kHz waveform using FFmpeg G.711 mu-law or AMR-NB."""
    if codec not in {"g711", "amr"}:
        raise ValueError(f"Unsupported telephony codec: {codec}")
    if not available_ffmpeg_codecs().get(codec, False):
        raise CodecUnavailableError(
            f"FFmpeg codec '{codec}' is unavailable. Install FFmpeg with the required encoder."
        )
    if len(y) == 0:
        return y.astype(np.float32, copy=False)

    ffmpeg = shutil.which("ffmpeg")
    wav_input = io.BytesIO()
    sf.write(wav_input, y.astype(np.float32, copy=False), sample_rate, format="WAV", subtype="PCM_16")
    encoder = (
        ["-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", "-f", "wav"]
        if codec == "g711"
        else ["-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", "12.2k", "-f", "amr"]
    )
    encoded = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *encoder, "pipe:1"],
        input=wav_input.getvalue(),
        capture_output=True,
        timeout=30,
        check=False,
    )
    if encoded.returncode != 0 or not encoded.stdout:
        raise CodecUnavailableError(
            f"FFmpeg failed to encode {codec}: {encoded.stderr.decode(errors='replace').strip()}"
        )

    decoded = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
            "-ar", str(sample_rate), "-ac", "1", "-f", "f32le", "-acodec",
            "pcm_f32le", "pipe:1",
        ],
        input=encoded.stdout,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if decoded.returncode != 0 or not decoded.stdout:
        raise CodecUnavailableError(
            f"FFmpeg failed to decode {codec}: {decoded.stderr.decode(errors='replace').strip()}"
        )
    result = np.frombuffer(decoded.stdout, dtype="<f4").copy()
    if len(result) > len(y):
        return result[: len(y)]
    if len(result) < len(y):
        return np.pad(result, (0, len(y) - len(result))).astype(np.float32)
    return result.astype(np.float32, copy=False)


def apply_packet_loss(
    y: np.ndarray,
    sample_rate: int = 16000,
    loss_rate: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """Zero random 20 ms packets to approximate dropped telephony frames."""
    if not 0 <= loss_rate <= 1:
        raise ValueError("loss_rate must be between 0 and 1.")
    result = y.astype(np.float32, copy=True)
    frame_size = max(1, round(sample_rate * 0.02))
    rng = np.random.default_rng(seed)
    for start in range(0, len(result), frame_size):
        if rng.random() < loss_rate:
            result[start : start + frame_size] = 0.0
    return result


def add_background_noise(
    y: np.ndarray,
    snr_db: float = 15.0,
    seed: int | None = None,
) -> np.ndarray:
    """Mix in white background noise at a requested SNR."""
    if len(y) == 0:
        return y.astype(np.float32, copy=False)
    signal = y.astype(np.float32, copy=False)
    power = float(np.mean(signal**2))
    if power < 1e-10:
        return signal.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(power / (10 ** (snr_db / 10))), size=len(signal))
    return np.clip(signal + noise, -1.0, 1.0).astype(np.float32)


def add_reverberation(y: np.ndarray, sample_rate: int = 16000, seed: int | None = None) -> np.ndarray:
    """Add a short, decaying synthetic room impulse response."""
    if len(y) == 0:
        return y.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    ir_length = max(1, round(sample_rate * 0.12))
    time = np.arange(ir_length, dtype=np.float32) / sample_rate
    impulse = np.exp(-5.0 * time) * rng.normal(0.0, 0.08, size=ir_length)
    impulse[0] = 1.0
    result = np.convolve(y.astype(np.float32, copy=False), impulse, mode="full")[: len(y)]
    peak = float(np.max(np.abs(result)))
    return (result / max(peak, 1.0)).astype(np.float32)


def augment_telephony(y: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Apply a randomized, reproducible-per-process telephony augmentation chain."""
    global _warned_missing_codec
    rng = np.random.default_rng(int(np.random.randint(0, 2**32 - 1)))
    result = y.astype(np.float32, copy=True)
    if rng.random() < 0.65:
        result = bandlimit_telephony(result, sample_rate)

    codecs = available_ffmpeg_codecs()
    usable_codecs = [codec for codec, enabled in codecs.items() if enabled]
    if not usable_codecs and not _warned_missing_codec:
        warnings.warn(
            "FFmpeg G.711/AMR encoders are unavailable; applying the other telephony augmentations only.",
            RuntimeWarning,
            stacklevel=2,
        )
        _warned_missing_codec = True
    if usable_codecs and rng.random() < 0.5:
        codec = usable_codecs[int(rng.integers(len(usable_codecs)))]
        result = simulate_ffmpeg_codec(result, sample_rate, codec)

    if rng.random() < 0.35:
        result = apply_packet_loss(result, sample_rate, loss_rate=0.02, seed=int(rng.integers(2**32)))
    if rng.random() < 0.5:
        result = add_background_noise(
            result, snr_db=float(rng.uniform(10, 25)), seed=int(rng.integers(2**32))
        )
    if rng.random() < 0.35:
        result = add_reverberation(result, sample_rate, seed=int(rng.integers(2**32)))
    return result
