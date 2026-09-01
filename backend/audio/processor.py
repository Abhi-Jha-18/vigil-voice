import os
import librosa
import numpy as np
import soundfile as sf

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".flac", ".webm", ".aac"}

def validate_format(filename: str) -> bool:
    """
    Validates if the file format is supported based on its extension.
    """
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def _load_with_av(filepath: str, target_sr: int = 16000) -> tuple[np.ndarray, int, int]:
    """
    Decodes audio from containers (e.g. MP4, M4A, WEBM) using PyAV.
    """
    import av
    try:
        container = av.open(filepath)
    except Exception as exc:
        raise ValueError("Could not open media file.") from exc

    audio_streams = [s for s in container.streams if s.type == "audio"]
    if not audio_streams:
        container.close()
        raise ValueError("No audio stream found in the uploaded file.")

    stream = audio_streams[0]
    orig_channels = getattr(stream.codec_context, "channels", None) or 1
    if hasattr(stream.codec_context, "layout") and stream.codec_context.layout:
        orig_channels = len(stream.codec_context.layout.channels)

    resampler = av.AudioResampler(
        format="fltp",
        layout="mono",
        rate=target_sr,
    )

    chunks = []
    try:
        for frame in container.decode(stream):
            for resampled_frame in resampler.resample(frame):
                arr = resampled_frame.to_ndarray()
                if arr.ndim > 1:
                    arr = arr.reshape(-1)
                chunks.append(arr)

        for resampled_frame in resampler.resample(None):
            arr = resampled_frame.to_ndarray()
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            chunks.append(arr)
    except Exception as exc:
        container.close()
        raise ValueError("Error decoding audio stream from container.") from exc

    container.close()

    if not chunks:
        raise ValueError("Uploaded audio file contains no readable samples.")

    y = np.concatenate(chunks).astype(np.float32)
    return y, target_sr, orig_channels

def load_and_resample(filepath: str, target_sr: int = 16000) -> tuple[np.ndarray, int, int]:
    """
    Loads audio, resamples to target_sr, and converts it to mono safely.
    Supports MP4, M4A, WAV, MP3, OGG, FLAC, WEBM, AAC.
    Returns:
        y (np.ndarray): Audio time series
        sr (int): Target sampling rate
        original_channels (int): Number of channels in the original file
    """
    if not filepath or not os.path.exists(filepath):
        raise ValueError("Audio file is missing or unreadable.")

    if os.path.getsize(filepath) == 0:
        raise ValueError("Uploaded audio file is empty.")

    original_channels = 1
    ext = os.path.splitext(filepath)[1].lower()

    # For MP4/M4A video/audio containers, prefer PyAV
    if ext in {".mp4", ".m4a"}:
        try:
            return _load_with_av(filepath, target_sr=target_sr)
        except Exception:
            pass  # Fallback to standard flow if av fails

    try:
        # First try librosa
        y, sr = librosa.load(filepath, sr=target_sr, mono=True)
    except Exception:
        try:
            # Fallback to soundfile
            y, sr = sf.read(filepath, always_2d=False)
            if y.ndim > 1:
                original_channels = y.shape[1]
                y = np.mean(y, axis=1)
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        except Exception:
            try:
                # Fallback to PyAV for any other container formats
                return _load_with_av(filepath, target_sr=target_sr)
            except Exception as exc:
                raise ValueError("Could not decode the uploaded audio file. Please upload a valid audio file.") from exc

    if y.size == 0:
        raise ValueError("Uploaded audio file contains no readable samples.")

    # Try to find original channels if librosa loaded it
    if original_channels == 1:
        try:
            info = sf.info(filepath)
            original_channels = info.channels
        except Exception:
            pass

    return y.astype(np.float32, copy=False), target_sr, original_channels

def normalize_audio(y: np.ndarray, method: str = "peak") -> np.ndarray:
    """
    Applies normalization to the audio signal.
    """
    if method == "peak":
        max_val = np.max(np.abs(y))
        if max_val > 0:
            return y / max_val
    return y

