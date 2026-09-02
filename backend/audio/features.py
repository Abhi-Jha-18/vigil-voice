import io
import base64
import threading

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_MFCC = 13
N_TIME_STEPS = 64
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 40

_PLOT_LOCK = threading.Lock()


def _pad_or_crop(mfcc: np.ndarray, n_steps: int = N_TIME_STEPS) -> np.ndarray:
    if mfcc.shape[1] < n_steps:
        return np.pad(mfcc, ((0, 0), (0, n_steps - mfcc.shape[1])), mode="constant")
    return mfcc[:, :n_steps]


def _cmn(mfcc: np.ndarray) -> np.ndarray:
    """Cepstral mean normalization — keeps variance cues, removes loudness bias."""
    return mfcc - mfcc.mean(axis=1, keepdims=True)


def extract_mfcc_for_model(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Fast MFCC path used by the CNN (training and inference).
    Returns float32 array of shape (N_MFCC, N_TIME_STEPS).
    """
    if y is None or len(y) == 0:
        return np.zeros((N_MFCC, N_TIME_STEPS), dtype=np.float32)

    mfcc = librosa.feature.mfcc(
        y=y.astype(np.float32, copy=False),
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    return _cmn(_pad_or_crop(mfcc)).astype(np.float32, copy=False)


def get_mel_spectrogram_b64(y: np.ndarray, sr: int = 16000) -> str:
    """
    Computes the Mel Spectrogram and returns a base64 encoded PNG image string.
    Uses plt.imsave (no full figure/colorbar) so dashboard rendering stays fast.
    """
    if y is None or len(y) == 0:
        return ""

    try:
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, fmax=8000, n_fft=1024, hop_length=256,
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        buf = io.BytesIO()
        with _PLOT_LOCK:
            plt.imsave(buf, S_dB, format="png", cmap="magma", origin="lower")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return ""


def extract_acoustic_features(y: np.ndarray, sr: int = 16000) -> dict:
    """
    Extracts numerical features from the audio signal.
    STFT is computed once and reused for centroid / rolloff / flatness.
    """
    if y is None or len(y) == 0:
        return {
            "mfcc_mean": [],
            "mfcc_std": [],
            "mfcc_sequence": np.zeros((N_MFCC, N_TIME_STEPS), dtype=np.float32),
            "spectral_centroid_mean": 0.0,
            "spectral_rolloff_mean": 0.0,
            "zero_crossing_rate_mean": 0.0,
            "spectral_flatness_mean": 0.0,
            "mfcc_delta_std": 0.0,
            "audio_duration_seconds": 0.0,
        }

    y = y.astype(np.float32, copy=False)
    duration = float(len(y)) / sr

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_pow = np.square(S)
    mel = librosa.feature.melspectrogram(
        S=S_pow, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)

    mfcc_mean = np.mean(mfcc, axis=1).tolist()
    mfcc_std = np.std(mfcc, axis=1).tolist()
    mfcc_for_model = _cmn(_pad_or_crop(mfcc)).astype(np.float32, copy=False)

    if mfcc.shape[1] > 1:
        mfcc_delta_std = float(np.mean(np.std(np.diff(mfcc, axis=1), axis=1)))
    else:
        mfcc_delta_std = 0.0

    sc_mean = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    sr_mean = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)))

    return {
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
        "mfcc_sequence": mfcc_for_model,
        "spectral_centroid_mean": sc_mean,
        "spectral_rolloff_mean": sr_mean,
        "zero_crossing_rate_mean": zcr_mean,
        "spectral_flatness_mean": flatness,
        "mfcc_delta_std": mfcc_delta_std,
        "audio_duration_seconds": duration,
    }
