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
MFCC_FEATURE_VERSION = "mfcc_v1"
LOGMEL_DELTA_FEATURE_VERSION = "logmel_delta_v1"
SUPPORTED_FEATURE_VERSIONS = {
    MFCC_FEATURE_VERSION,
    LOGMEL_DELTA_FEATURE_VERSION,
}

_PLOT_LOCK = threading.Lock()


def _pad_or_crop(mfcc: np.ndarray, n_steps: int = N_TIME_STEPS) -> np.ndarray:
    if mfcc.shape[1] < n_steps:
        return np.pad(mfcc, ((0, 0), (0, n_steps - mfcc.shape[1])), mode="constant")
    return mfcc[:, :n_steps]


def _cmn(mfcc: np.ndarray) -> np.ndarray:
    """Cepstral mean normalization — keeps variance cues, removes loudness bias."""
    if mfcc.shape[1] == 0:
        return mfcc
    return mfcc - mfcc.mean(axis=1, keepdims=True)


def mfcc_crops(mfcc: np.ndarray, n: int = N_TIME_STEPS, hop: int = 32) -> np.ndarray:
    """
    Extract overlapping n-frame crops from an MFCC matrix of shape (N_MFCC, T).
    Returns an ndarray of shape (num_crops, N_MFCC, n).
    If T < n, pads with zeros along the time axis so at least one crop is returned.
    """
    if mfcc is None or mfcc.size == 0 or mfcc.ndim != 2:
        return np.zeros((1, N_MFCC, n), dtype=np.float32)

    n_mfcc, t = mfcc.shape
    if t < n:
        padded = np.pad(mfcc, ((0, 0), (0, n - t)), mode="constant")
        return padded[np.newaxis, ...].astype(np.float32, copy=False)

    starts = list(range(0, t - n + 1, hop))
    if not starts:
        starts = [0]
    elif starts[-1] != t - n:
        starts.append(t - n)

    crops = [mfcc[:, s : s + n] for s in starts]
    return np.stack(crops, axis=0).astype(np.float32, copy=False)


def feature_crops(features: np.ndarray, n: int = N_TIME_STEPS, hop: int = 32) -> np.ndarray:
    """Crop model features shaped (channels, bins, time) into fixed-size windows."""
    if features is None or features.size == 0 or features.ndim != 3:
        raise ValueError("Model features must have shape (channels, bins, time)")

    channels, bins, frames = features.shape
    if frames < n:
        padded = np.pad(features, ((0, 0), (0, 0), (0, n - frames)), mode="constant")
        return padded[np.newaxis, ...].astype(np.float32, copy=False)

    starts = list(range(0, frames - n + 1, hop))
    if starts[-1] != frames - n:
        starts.append(frames - n)
    return np.stack([features[:, :, start : start + n] for start in starts]).astype(
        np.float32, copy=False
    )


def extract_model_features(
    y: np.ndarray,
    sr: int = 16000,
    feature_version: str = MFCC_FEATURE_VERSION,
) -> np.ndarray:
    """Extract model input features as (channels, bins, time)."""
    if feature_version == MFCC_FEATURE_VERSION:
        if y is None or len(y) == 0:
            return np.zeros((1, N_MFCC, 0), dtype=np.float32)
        mfcc = librosa.feature.mfcc(
            y=y.astype(np.float32, copy=False),
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        return _cmn(mfcc)[np.newaxis, ...].astype(np.float32, copy=False)
    if feature_version != LOGMEL_DELTA_FEATURE_VERSION:
        raise ValueError(f"Unsupported feature version: {feature_version}")
    if y is None or len(y) == 0:
        return np.zeros((3, 64, 0), dtype=np.float32)

    mel_power = librosa.feature.melspectrogram(
        y=y.astype(np.float32, copy=False),
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=64,
        fmax=min(8000, sr // 2),
    )
    log_mel = librosa.power_to_db(mel_power, ref=np.max)
    if log_mel.shape[1] >= 3:
        width = min(9, log_mel.shape[1] if log_mel.shape[1] % 2 else log_mel.shape[1] - 1)
        delta = librosa.feature.delta(log_mel, width=width)
        delta_delta = librosa.feature.delta(log_mel, order=2, width=width)
    else:
        delta = np.zeros_like(log_mel)
        delta_delta = np.zeros_like(log_mel)

    channels = []
    for values in (log_mel, delta, delta_delta):
        mean = values.mean(axis=1, keepdims=True)
        variance = values.var(axis=1, keepdims=True)
        channels.append((values - mean) / np.sqrt(np.maximum(variance, 1e-8)))
    return np.stack(channels).astype(np.float32, copy=False)


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
    return _pad_or_crop(_cmn(mfcc)).astype(np.float32, copy=False)


def get_mel_spectrogram_b64(y: np.ndarray, sr: int = 16000, max_width: int = 800) -> str:
    """
    Computes the Mel Spectrogram and returns a base64 encoded PNG image string.
    Uses plt.imsave (no full figure/colorbar) so dashboard rendering stays fast.
    Caps the spectrogram image width to max_width to prevent oversized payloads.
    """
    if y is None or len(y) == 0:
        return ""

    try:
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, fmax=8000, n_fft=1024, hop_length=256,
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        if S_dB.shape[1] > max_width:
            step = int(np.ceil(S_dB.shape[1] / max_width))
            S_dB = S_dB[:, ::step]
            if S_dB.shape[1] > max_width:
                S_dB = S_dB[:, :max_width]

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


def render_gradcam_b64(
    activation_map: np.ndarray,
    audio: np.ndarray | None = None,
    sr: int = 16000,
) -> str:
    """Render Grad-CAM alone or over the corresponding log-mel spectrogram."""
    if activation_map is None or activation_map.size == 0:
        return ""
    buf = io.BytesIO()
    with _PLOT_LOCK:
        if audio is None or len(audio) == 0:
            plt.imsave(buf, activation_map, format="png", cmap="magma", origin="lower")
        else:
            mel = librosa.feature.melspectrogram(
                y=audio.astype(np.float32, copy=False),
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH // 2,
                n_mels=64,
                fmax=min(8000, sr // 2),
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            frames = np.linspace(0, activation_map.shape[1] - 1, log_mel.shape[1])
            time_map = np.stack([
                np.interp(frames, np.arange(activation_map.shape[1]), row)
                for row in activation_map
            ])
            figure, axis = plt.subplots(figsize=(5, 2.5), dpi=100)
            axis.imshow(log_mel, origin="lower", aspect="auto", cmap="gray")
            axis.imshow(time_map, origin="lower", aspect="auto", cmap="magma", alpha=0.55, vmin=0, vmax=1)
            axis.axis("off")
            figure.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(figure)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def extract_acoustic_features(
    y: np.ndarray,
    sr: int = 16000,
    feature_version: str | None = None,
) -> dict:
    """
    Extracts numerical features from the audio signal.
    STFT is computed once and reused for centroid / rolloff / flatness.
    """
    from backend.audio.config import default_config

    feature_version = feature_version or default_config.feature_version
    if y is None or len(y) == 0:
        return {
            "mfcc_mean": [],
            "mfcc_std": [],
            "mfcc_sequence": np.zeros((N_MFCC, N_TIME_STEPS), dtype=np.float32),
            "mfcc_unpadded": np.zeros((N_MFCC, 0), dtype=np.float32),
            "spectral_centroid_mean": 0.0,
            "spectral_rolloff_mean": 0.0,
            "zero_crossing_rate_mean": 0.0,
            "spectral_flatness_mean": 0.0,
            "mfcc_delta_std": 0.0,
            "audio_duration_seconds": 0.0,
            "model_features": np.zeros(
                (3, 64, N_TIME_STEPS)
                if feature_version == LOGMEL_DELTA_FEATURE_VERSION
                else (1, N_MFCC, N_TIME_STEPS),
                dtype=np.float32,
            ),
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
    mfcc_std = np.std(mfcc[1:], axis=1).tolist() if mfcc.shape[0] > 1 else np.std(mfcc, axis=1).tolist()
    cmn_mfcc = _cmn(mfcc)
    mfcc_for_model = _pad_or_crop(cmn_mfcc).astype(np.float32, copy=False)

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
        "mfcc_unpadded": cmn_mfcc.astype(np.float32, copy=False),
        "spectral_centroid_mean": sc_mean,
        "spectral_rolloff_mean": sr_mean,
        "zero_crossing_rate_mean": zcr_mean,
        "spectral_flatness_mean": flatness,
        "mfcc_delta_std": mfcc_delta_std,
        "audio_duration_seconds": duration,
        "model_features": (
            cmn_mfcc[np.newaxis, ...].astype(np.float32, copy=False)
            if feature_version == MFCC_FEATURE_VERSION
            else extract_model_features(y, sr=sr, feature_version=feature_version)
        ),
    }
