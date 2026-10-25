"""Central configuration for audio preprocessing."""

from dataclasses import dataclass
import os
from typing import Set

@dataclass(frozen=True)
class PreprocessingConfig:
    """Canonical configuration for all audio preprocessing in the project."""
    sample_rate: int = 16000
    channels: int = 1
    normalization_method: str = "peak"
    min_audio_duration_seconds: float = 0.25
    max_audio_duration_seconds: float = 120.0
    min_speech_duration_seconds: float = 0.25
    vad_top_db: int = 30
    segment_duration_seconds: float = 3.0
    segment_overlap_seconds: float = 1.0
    max_segments: int = 10
    supported_extensions: tuple = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac")
    feature_version: str = "mfcc_v1"

    def to_dict(self) -> dict:
        feature_version = os.getenv("FEATURE_VERSION", self.feature_version).strip()
        if feature_version not in {"mfcc_v1", "logmel_delta_v1"}:
            raise ValueError(f"Unsupported FEATURE_VERSION: {feature_version}")
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "normalization_method": self.normalization_method,
            "min_audio_duration_seconds": self.min_audio_duration_seconds,
            "max_audio_duration_seconds": self.max_audio_duration_seconds,
            "min_speech_duration_seconds": self.min_speech_duration_seconds,
            "vad_top_db": self.vad_top_db,
            "segment_duration_seconds": self.segment_duration_seconds,
            "segment_overlap_seconds": self.segment_overlap_seconds,
            "max_segments": self.max_segments,
            "supported_extensions": list(self.supported_extensions),
            "feature_version": feature_version,
        }

# Shared singleton config for the canonical pipeline
default_config = PreprocessingConfig(
    feature_version=os.getenv("FEATURE_VERSION", "mfcc_v1").strip()
)
if default_config.feature_version not in {"mfcc_v1", "logmel_delta_v1"}:
    raise ValueError(f"Unsupported FEATURE_VERSION: {default_config.feature_version}")
