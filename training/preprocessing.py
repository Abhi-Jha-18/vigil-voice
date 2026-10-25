"""
Shared preprocessing interface for training.
Ensures that training uses the exact same preprocessing configuration and pipeline
as the backend inference (single canonical pipeline).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from backend.audio.config import PreprocessingConfig, default_config
from backend.audio.pipeline import (
    AudioSegment,
    PreprocessingResult,
    process_canonical,
    segment_audio,
)

__all__ = [
    "preprocess_for_training",
    "process_canonical",
    "PreprocessingResult",
    "AudioSegment",
    "segment_audio",
    "default_config",
    "PreprocessingConfig",
]


def preprocess_for_training(
    filepath: str | Path,
    config: PreprocessingConfig = default_config,
) -> np.ndarray:
    """
    Applies the canonical preprocessing pipeline for training data.
    If the audio is INSUFFICIENT, raises ValueError so training can skip it.
    Otherwise, returns the processed VAD-cleaned audio array.
    """
    result = process_canonical(str(filepath), config=config)

    if result.quality.status == "INSUFFICIENT":
        reason = result.quality.reason or "Unknown reason"
        raise ValueError(f"Audio quality insufficient for training: {reason}")

    return result.audio_data
