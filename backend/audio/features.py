import io
import base64
import numpy as np
import librosa
import matplotlib
# Use the non-interactive Agg backend to avoid threading/GUI errors in web servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_MFCC = 13
N_TIME_STEPS = 64

def get_mel_spectrogram_b64(y: np.ndarray, sr: int = 16000) -> str:
    """
    Computes the Mel Spectrogram and returns a base64 encoded PNG image string.
    """
    if len(y) == 0:
        return ""
        
    try:
        # Compute Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Plot Spectrogram
        plt.figure(figsize=(6, 3), dpi=100)
        # Deep space / neon color themes: 'magma', 'viridis', 'plasma'
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram', color='white', fontsize=10)
        plt.tight_layout()
        
        # Style the plot for dark mode
        fig = plt.gcf()
        fig.patch.set_facecolor('#111026')  # Dark violet background to match dashboard
        ax = plt.gca()
        ax.set_facecolor('#111026')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#111026', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # Cleanup plot to free memory
        plt.close()
        
        return image_base64
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        # Always make sure to close the plot if open to avoid memory leak
        plt.close()
        return ""

def extract_acoustic_features(y: np.ndarray, sr: int = 16000) -> dict:
    """
    Extracts numerical features from the audio signal.
    Returns a dict with serializable values.
    """
    if len(y) == 0:
        return {
            "mfcc_mean": [],
            "mfcc_std": [],
            "mfcc_sequence": [],
            "spectral_centroid_mean": 0.0,
            "spectral_rolloff_mean": 0.0,
            "zero_crossing_rate_mean": 0.0,
            "audio_duration_seconds": 0.0
        }
        
    # Calculate duration
    duration = float(len(y)) / sr
    
    # MFCC extraction (typically 13 coefficients is standard for speech processing)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1).tolist()
    mfcc_std = np.std(mfcc, axis=1).tolist()

    # Keep the same fixed-size time-frequency representation used in training.
    # This is intentionally returned alongside the summary statistics: the CNN
    # needs temporal information, while the dashboard displays the summaries.
    if mfcc.shape[1] < N_TIME_STEPS:
        mfcc_for_model = np.pad(
            mfcc, ((0, 0), (0, N_TIME_STEPS - mfcc.shape[1])), mode="constant"
        )
    else:
        mfcc_for_model = mfcc[:, :N_TIME_STEPS]
    
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    sc_mean = float(np.mean(spectral_centroid))
    
    # Spectral Roll-off
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    sr_mean = float(np.mean(spectral_rolloff))
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = float(np.mean(zcr))
    
    # Return features dictionary
    return {
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
        "mfcc_sequence": mfcc_for_model.astype(np.float32).tolist(),
        "spectral_centroid_mean": sc_mean,
        "spectral_rolloff_mean": sr_mean,
        "zero_crossing_rate_mean": zcr_mean,
        "audio_duration_seconds": duration
    }
