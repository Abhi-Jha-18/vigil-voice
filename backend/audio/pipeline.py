import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from backend.audio.config import PreprocessingConfig, default_config
from backend.audio.processor import load_and_resample, normalize_audio
from backend.audio.vad import remove_silence
from backend.audio.quality import analyze_quality, QualityResult

@dataclass
class AudioSegment:
    segment_id: int
    start_time: float
    end_time: float
    duration: float
    status: str
    audio_data: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status
        }

@dataclass
class PreprocessingResult:
    audio_data: np.ndarray
    quality: QualityResult
    segments: List[AudioSegment]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality": self.quality.to_dict(),
            "segments": [s.to_dict() for s in self.segments]
        }

def segment_audio(y: np.ndarray, sr: int, config: PreprocessingConfig, status: str) -> List[AudioSegment]:
    """Splits audio into overlapping segments."""
    segments = []
    
    # Avoid segmenting if we don't have enough data for even one segment
    if len(y) == 0:
        return segments
        
    segment_length = int(config.segment_duration_seconds * sr)
    hop_length = int((config.segment_duration_seconds - config.segment_overlap_seconds) * sr)
    
    if hop_length <= 0:
        hop_length = segment_length # fallback if overlap is misconfigured
        
    for i, start in enumerate(range(0, len(y), hop_length)):
        if i >= config.max_segments:
            break
            
        end = min(start + segment_length, len(y))
        segment_data = y[start:end]
        
        # Only add segment if it's long enough, or if it's the only data we have
        if len(segment_data) >= segment_length * 0.5 or (i == 0 and len(segments) == 0):
            segments.append(AudioSegment(
                segment_id=i,
                start_time=start / sr,
                end_time=end / sr,
                duration=len(segment_data) / sr,
                status=status,
                audio_data=segment_data
            ))
            
    return segments

def process_canonical(filepath: str, config: PreprocessingConfig = default_config) -> PreprocessingResult:
    """
    The canonical preprocessing pipeline shared across all parts of the application.
    """
    # 1. Decode & Resample & Mono
    y, sr, original_channels = load_and_resample(filepath, target_sr=config.sample_rate)
    
    # 2. Early Quality Check (pre-VAD)
    initial_quality = analyze_quality(y, sr, None, config, original_channels)
    if initial_quality.status == "INSUFFICIENT":
        return PreprocessingResult(
            audio_data=y, 
            quality=initial_quality, 
            segments=[]
        )
        
    # 3. Normalize
    if config.normalization_method == "peak":
        y = normalize_audio(y, method="peak")
        
    # 4. VAD
    y_speech = remove_silence(y, sr=sr, top_db=config.vad_top_db)
    
    # 5. Final Quality Check (post-VAD)
    final_quality = analyze_quality(y, sr, y_speech, config, original_channels)
    
    if final_quality.status == "INSUFFICIENT":
        return PreprocessingResult(
            audio_data=y_speech,
            quality=final_quality,
            segments=[]
        )
        
    # 6. Segmentation
    segments = segment_audio(y_speech, sr, config, final_quality.status)
    
    return PreprocessingResult(
        audio_data=y_speech,
        quality=final_quality,
        segments=segments
    )
