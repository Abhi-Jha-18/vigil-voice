import os
import gc
import time
import tempfile
import numpy as np
from scipy.io import wavfile

from backend.audio.config import PreprocessingConfig, default_config
from backend.audio.pipeline import process_canonical
from training.preprocessing import preprocess_for_training

def generate_sine_wave(duration_sec: float, sample_rate: int = 16000, frequency: float = 440.0, amplitude: float = 0.5):
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio = amplitude * np.sin(2 * np.pi * frequency * t)
    audio_scaled = np.int16(audio * 32767)
    return audio_scaled

def _safe_remove(filepath, retries=5, delay=0.3):
    """Remove a file with retries for Windows file-locking issues."""
    for attempt in range(retries):
        try:
            gc.collect()
            os.remove(filepath)
            return
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                pass  # Silently skip if still locked after retries

def test_preprocessing_good_audio():
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_path = temp_wav.name
            audio = generate_sine_wave(3.0)
            wavfile.write(temp_path, 16000, audio)
        
        result = process_canonical(temp_path)
        assert result.quality.status in ["GOOD", "DEGRADED"]
        assert result.quality.metrics.total_duration >= 2.9
        assert result.quality.metrics.speech_duration > 0
        assert len(result.audio_data) > 0
        assert len(result.segments) > 0
    finally:
        if temp_path:
            _safe_remove(temp_path)

def test_preprocessing_silent_audio():
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_path = temp_wav.name
            audio = generate_sine_wave(3.0, amplitude=0.0)
            wavfile.write(temp_path, 16000, audio)
        
        result = process_canonical(temp_path)
        assert result.quality.status == "INSUFFICIENT"
        assert "silence-only" in result.quality.reason.lower()
    finally:
        if temp_path:
            _safe_remove(temp_path)

def test_preprocessing_too_short():
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_path = temp_wav.name
            audio = generate_sine_wave(0.1)
            wavfile.write(temp_path, 16000, audio)
        
        result = process_canonical(temp_path)
        assert result.quality.status == "INSUFFICIENT"
        assert "too short" in result.quality.reason.lower()
    finally:
        if temp_path:
            _safe_remove(temp_path)

def test_preprocessing_clipping():
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_path = temp_wav.name
            # Generate an audio file that is clipped
            t = np.linspace(0, 3.0, int(16000 * 3.0), endpoint=False)
            audio = 5.0 * np.sin(2 * np.pi * 440.0 * t)  # Will exceed 1.0 and clip if we just truncate
            audio = np.clip(audio, -1.0, 1.0)
            audio_scaled = np.int16(audio * 32767)
            wavfile.write(temp_path, 16000, audio_scaled)
        
        # Our config peak-normalizes by default, but wait, the clipping metric is 
        # calculated before VAD, after normalization. Since we clip at 1.0, peak normalization
        # will keep many values at 1.0.
        result = process_canonical(temp_path)
        assert result.quality.status in ["DEGRADED", "GOOD"] # Depending on thresholds, but shouldn't fail
        assert result.quality.metrics.clipping_ratio > 0
    finally:
        if temp_path:
            _safe_remove(temp_path)

def test_training_inference_consistency():
    """
    Test proving that training and inference reference the same preprocessing configuration.
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_path = temp_wav.name
            audio = generate_sine_wave(3.0)
            wavfile.write(temp_path, 16000, audio)
        
        backend_result = process_canonical(temp_path)
        training_result = preprocess_for_training(temp_path)
        
        # The arrays should be exactly identical
        np.testing.assert_array_equal(backend_result.audio_data, training_result)
    finally:
        if temp_path:
            _safe_remove(temp_path)

def test_preprocessing_mp4_audio():
    """
    Test that MP4 files with audio streams decode correctly and work in the canonical pipeline.
    """
    import av
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_mp4:
            temp_path = temp_mp4.name
        
        # Write an MP4 file with an AAC audio stream
        out_container = av.open(temp_path, mode="w", format="mp4")
        stream = out_container.add_stream("aac", rate=44100)
        stream.layout = "mono"

        t = np.linspace(0, 3.0, int(44100 * 3.0), endpoint=False)
        samples = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="flt", layout="mono")
        frame.rate = 44100
        for packet in stream.encode(frame):
            out_container.mux(packet)
        for packet in stream.encode(None):
            out_container.mux(packet)
        out_container.close()

        result = process_canonical(temp_path)
        assert result.quality.status in ["GOOD", "DEGRADED"]
        assert result.quality.metrics.total_duration >= 2.9
        assert len(result.audio_data) > 0
        assert len(result.segments) > 0
    finally:
        if temp_path:
            _safe_remove(temp_path)

if __name__ == "__main__":
    test_preprocessing_good_audio()
    test_preprocessing_silent_audio()
    test_preprocessing_too_short()
    test_preprocessing_clipping()
    test_training_inference_consistency()
    test_preprocessing_mp4_audio()
    print("All preprocessing tests passed.")

