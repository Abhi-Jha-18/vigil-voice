"""
Shared preprocessing interface for training.
Ensures that training uses the exact same preprocessing configuration and pipeline
as the backend inference.
"""

import numpy as np
from backend.audio.pipeline import process_canonical, PreprocessingResult
from backend.audio.config import default_config

def preprocess_for_training(filepath: str) -> np.ndarray:
    """
    Applies the canonical preprocessing pipeline for training data.
    If the audio is INSUFFICIENT, raises ValueError so training can skip it.
    Otherwise, returns the processed VAD-cleaned audio array.
    """
    result = process_canonical(filepath, config=default_config)
    
    if result.quality.status == "INSUFFICIENT":
        reason = result.quality.reason or "Unknown reason"
        raise ValueError(f"Audio quality insufficient for training: {reason}")
        
    return result.audio_data
