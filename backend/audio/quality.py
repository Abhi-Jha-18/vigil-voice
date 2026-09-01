import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

from backend.audio.config import PreprocessingConfig

@dataclass
class QualityMetrics:
    total_duration: float
    speech_duration: float
    silence_ratio: float
    rms_energy: float
    clipping_ratio: float
    sample_rate: int
    channel_count: int
    snr_estimate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_duration": self.total_duration,
            "speech_duration": self.speech_duration,
            "silence_ratio": self.silence_ratio,
            "rms_energy": self.rms_energy,
            "clipping_ratio": self.clipping_ratio,
            "sample_rate": self.sample_rate,
            "channel_count": self.channel_count,
            "snr_estimate": self.snr_estimate
        }

@dataclass
class QualityResult:
    status: str  # GOOD, DEGRADED, INSUFFICIENT
    metrics: QualityMetrics
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "metrics": self.metrics.to_dict(),
            "reason": self.reason
        }

def analyze_quality(
    y: np.ndarray, 
    sr: int, 
    y_speech: Optional[np.ndarray], 
    config: PreprocessingConfig,
    original_channels: int = 1
) -> QualityResult:
    """
    Analyzes audio array for quality.
    
    Args:
        y: Original audio array (mono, resampled)
        sr: Sample rate
        y_speech: Audio array after VAD (silence removed). If None, calculated from y.
        config: Preprocessing configuration
        original_channels: Number of channels before mono mixdown
    """
    total_duration = len(y) / sr if len(y) > 0 else 0.0
    
    # Calculate speech duration
    if y_speech is not None:
        speech_duration = len(y_speech) / sr if len(y_speech) > 0 else 0.0
    else:
        # We don't have VAD output yet, so we just use total duration
        speech_duration = total_duration
        
    silence_ratio = 1.0 - (speech_duration / total_duration) if total_duration > 0 else 1.0
    
    # RMS Energy
    rms_energy = float(np.sqrt(np.mean(y**2))) if len(y) > 0 else 0.0
    
    # Clipping ratio (samples at or very close to max amplitude of 1.0)
    if len(y) > 0:
        clipping_threshold = 0.99
        clipping_samples = np.sum(np.abs(y) >= clipping_threshold)
        clipping_ratio = float(clipping_samples / len(y))
    else:
        clipping_ratio = 0.0
        
    # SNR Estimate (simple: signal variance / noise floor variance)
    snr_estimate = None
    if y_speech is not None and len(y_speech) > 0 and len(y) > len(y_speech):
        # We can estimate noise from the non-speech parts (rough estimate)
        # But for simplicity without exact masks, we'll skip complex SNR for now 
        # unless requested.
        pass

    metrics = QualityMetrics(
        total_duration=total_duration,
        speech_duration=speech_duration,
        silence_ratio=silence_ratio,
        rms_energy=rms_energy,
        clipping_ratio=clipping_ratio,
        sample_rate=sr,
        channel_count=original_channels,
        snr_estimate=snr_estimate
    )
    
    # Check against thresholds
    if total_duration < config.min_audio_duration_seconds:
        return QualityResult("INSUFFICIENT", metrics, f"Audio too short (total duration: {total_duration:.2f}s < {config.min_audio_duration_seconds}s)")
        
    if total_duration > config.max_audio_duration_seconds:
        return QualityResult("INSUFFICIENT", metrics, f"Audio too long (total duration: {total_duration:.2f}s > {config.max_audio_duration_seconds}s)")
        
    if rms_energy < 1e-4:
        return QualityResult("INSUFFICIENT", metrics, "Audio is silence-only or extremely quiet.")
        
    if speech_duration < config.min_speech_duration_seconds:
        return QualityResult("INSUFFICIENT", metrics, f"Insufficient speech duration ({speech_duration:.2f}s < {config.min_speech_duration_seconds}s)")
        
    if clipping_ratio > 0.05:
        return QualityResult("DEGRADED", metrics, "Severe clipping detected (excessive distortion).")
        
    if silence_ratio > 0.9:
        return QualityResult("DEGRADED", metrics, "Very high silence ratio.")

    return QualityResult("GOOD", metrics)
