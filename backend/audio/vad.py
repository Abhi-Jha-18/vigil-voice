import numpy as np
import librosa

def remove_silence(y: np.ndarray, sr: int = 16000, top_db: int = 30) -> np.ndarray:
    """
    Applies Voice Activity Detection (VAD) by splitting the audio into non-silent intervals
    and concatenating them. This filters out silent and low-energy segments.
    
    Args:
        y (np.ndarray): Preprocessed audio signal
        sr (int): Sampling rate
        top_db (int): The threshold (in decibels) below reference to consider as silence.
                     A lower value means more sensitive silence removal.
                     
    Returns:
        np.ndarray: Concatenated active audio segments.
    """
    # If the signal is too short or quiet, return as is
    if len(y) == 0 or np.max(np.abs(y)) < 0.01:
        return y
        
    try:
        # librosa.effects.split splits audio into non-silent intervals
        intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
        
        if len(intervals) == 0:
            return y  # Return original if everything was considered silence
            
        # Extract and concatenate active segments
        active_segments = [y[start:end] for start, end in intervals]
        y_active = np.concatenate(active_segments)
        
        # If silence removal stripped too much or failed, return original signal
        if len(y_active) < 100:
            return y
            
        return y_active
        
    except Exception as e:
        print(f"VAD warning (falling back to original signal): {e}")
        return y
