"""
VigilVoice — AI Detection Engine
=================================
Houses model architectures and the multi-phase detection interface.

Phase 1: Heuristic scoring using acoustic features (always available)
Phase 2: CNN model loaded from models/cnn_weight.pth
Phase 3: CNN-LSTM (architecture stub — requires training data)
Phase 4: Wav2Vec (architecture stub — requires pre-trained weights)
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from backend.config import settings

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH   = settings.model_path
N_MFCC       = 13
N_TIME_STEPS = 64   # Must match the value used during training

# Module-level cached model (loaded once on first Phase 2 call)
_cnn_model     = None
_cnn_status    = None   # REAL_MODEL, DEMO_MODEL, HEURISTIC_ONLY, MODEL_UNAVAILABLE


# ── Model Architectures ────────────────────────────────────────────────────────

class SimpleCNNDetector(nn.Module):
    """
    Phase 2: CNN Model Architecture for MFCC spectrogram analysis.
    Input shape: (batch, 1, N_MFCC, N_TIME_STEPS)  →  e.g. (B, 1, 13, 64)
    """
    def __init__(self):
        super(SimpleCNNDetector, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)


class CNNLSTMDetector(nn.Module):
    """
    Phase 3: CNN-LSTM Architecture to capture temporal structures.
    (Skeleton — requires full training data to activate)
    """
    def __init__(self):
        super(CNNLSTMDetector, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(
            input_size=16 * 6,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, f, t = x.shape
        cnn_out = self.cnn(x)
        # Reshape: treat time axis as LSTM sequence
        _, ch, fh, tw = cnn_out.shape
        cnn_out = cnn_out.permute(0, 3, 1, 2).reshape(b, tw, ch * fh)
        lstm_out, _ = self.lstm(cnn_out)
        return self.fc(lstm_out[:, -1, :])


# ── Model Loading ──────────────────────────────────────────────────────────────

def _compute_file_sha256(path: Path) -> str:
    """Calculates the SHA-256 checksum of a file."""
    import hashlib
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _get_checkpoint_status(model_file: Path) -> str:
    """Reads metadata and verifies SHA-256 integrity to determine model status."""
    import json
    metadata_file = model_file.with_name(model_file.stem + "_metadata.json")
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                
                # Check for recorded checksum to verify integrity
                recorded_hash = metadata.get("model_sha256") or metadata.get("expected_sha256")
                if recorded_hash:
                    actual_hash = _compute_file_sha256(model_file)
                    if recorded_hash.lower() != actual_hash.lower():
                        print(f"[Security Warning] Checksum mismatch for '{model_file}': expected {recorded_hash}, got {actual_hash}")
                        return "MODEL_INTEGRITY_FAILURE"

                if metadata.get("dataset_identifier") == "REAL_DATASET_REQUIRED":
                    return "MODEL_UNAVAILABLE"
                if "DEMO" in str(metadata.get("dataset_identifier", "")).upper():
                    return "DEMO_MODEL"
                return "REAL_MODEL"
        except Exception as e:
            print(f"[Detector] Metadata parsing error: {e}")
            pass

    # If no metadata file, check if it's the old demo checkpoint by path name
    if "demo" in model_file.name.lower():
        return "DEMO_MODEL"
    # Legacy unversioned checkpoints are considered demo
    return "DEMO_MODEL"

def _load_cnn_model() -> tuple:
    """
    Load the SimpleCNNDetector from disk.
    Returns (model, status_string).
    """
    global _cnn_model, _cnn_status

    if _cnn_status is not None:
        return _cnn_model, _cnn_status

    model_file = MODEL_PATH
    if not model_file.is_file():
        print(f"[Detector] Phase 2 model not found at '{model_file}' — "
              f"falling back to Phase 1 heuristic.")
        _cnn_status = "MODEL_UNAVAILABLE"
        return None, _cnn_status

    try:
        status = _get_checkpoint_status(model_file)
        if status == "MODEL_INTEGRITY_FAILURE":
            _cnn_status = "MODEL_INTEGRITY_FAILURE"
            return None, _cnn_status

        model = SimpleCNNDetector()
        state = torch.load(model_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _cnn_model = model
        _cnn_status = status
        print(f"[Detector] Phase 2 CNN loaded from '{model_file}'. Status: {_cnn_status}")
        return model, _cnn_status
    except Exception as e:
        print(f"[Detector] Failed to load Phase 2 model: {e} — falling back.")
        _cnn_status = "MODEL_UNAVAILABLE"
        return None, _cnn_status


def reset_cached_model():
    """Testing helper to clear cached model instance and status."""
    global _cnn_model, _cnn_status
    _cnn_model = None
    _cnn_status = None


def is_cnn_model_available() -> bool:
    """Public helper — returns whether ANY trusted trained CNN checkpoint exists."""
    _, status = _load_cnn_model()
    return status in ("REAL_MODEL", "DEMO_MODEL")

def get_model_status() -> str:
    """Returns the precise model capability status."""
    _, status = _load_cnn_model()
    if status == "MODEL_INTEGRITY_FAILURE":
        return "MODEL_INTEGRITY_FAILURE"
    if not is_cnn_model_available():
        return "HEURISTIC_ONLY"
    return status


# ── Phase 1: Heuristic Detector ────────────────────────────────────────────────

def run_phase1_stub_detection(features: dict, force_verdict: str = None) -> float:
    """
    Calculates a score (0.0 – 1.0) representing the probability of audio being REAL.
    Higher score = more likely REAL. Lower score = more likely FAKE/synthetic.

    If force_verdict is set, returns an engineered score for that outcome.
    """
    if force_verdict:
        force_verdict = force_verdict.lower()
        if force_verdict == "real":
            return float(np.random.uniform(0.82, 0.98))
        elif force_verdict == "fake":
            return float(np.random.uniform(0.02, 0.18))
        elif force_verdict == "uncertain":
            return float(np.random.uniform(0.45, 0.65))

    # Heuristic based on MFCC variance + spectral centroid
    try:
        mfcc_stds  = features.get("mfcc_std", [])
        sc_mean    = features.get("spectral_centroid_mean", 1500)
        score_base = 0.5

        # Higher spectral centroid → robotic synthetic voice → lower score
        if sc_mean > 2500:
            score_base -= 0.15
        elif sc_mean < 1200:
            score_base += 0.10

        # Higher MFCC std → natural expressive voice → higher score
        if mfcc_stds:
            std_variance = np.mean(mfcc_stds)
            if std_variance > 18:
                score_base += 0.25
            elif std_variance < 8:
                score_base -= 0.25

        return float(np.clip(score_base, 0.05, 0.95))
    except Exception as e:
        print(f"[Detector] Heuristic error: {e}")
        return 0.5


# ── Phase 2: CNN Detector ──────────────────────────────────────────────────────

def run_phase2_cnn_detection(features: dict, force_verdict: str = None) -> float:
    """
    Runs the trained SimpleCNNDetector using the fixed-length MFCC sequence.
    Falls back to Phase 1 if the model is not available.
    """
    if force_verdict:
        return run_phase1_stub_detection(features, force_verdict)

    model, status = _load_cnn_model()
    if status == "MODEL_UNAVAILABLE":
        return run_phase1_stub_detection(features)

    try:
        mfcc_sequence = np.asarray(features.get("mfcc_sequence", []), dtype=np.float32)
        if mfcc_sequence.shape != (N_MFCC, N_TIME_STEPS):
            return run_phase1_stub_detection(features)

        # The values intentionally remain on the same scale as training/train.py.
        # Unlike the previous mean-vector tiling, this retains speech dynamics.
        tensor = torch.from_numpy(mfcc_sequence[np.newaxis, np.newaxis, :, :])

        with torch.no_grad():
            score = model(tensor).item()

        return float(np.clip(score, 0.01, 0.99))
    except Exception as e:
        print(f"[Detector] Phase 2 inference error: {e} — falling back.")
        return run_phase1_stub_detection(features)


# ── Main Interface ─────────────────────────────────────────────────────────────

def run_ai_detection(features: dict, phase: str = "phase1",
                     force_verdict: str = None) -> float:
    """
    Multi-phase detection interface.
    Returns the REAL probability score (0.0 – 1.0).
    """
    if phase == "phase2":
        return run_phase2_cnn_detection(features, force_verdict)
    elif phase == "phase3":
        print("[Detector] Phase 3 CNN-LSTM not yet trained — using Phase 2 fallback.")
        return run_phase2_cnn_detection(features, force_verdict)
    elif phase == "phase4":
        print("[Detector] Phase 4 Wav2Vec not yet available — using Phase 2 fallback.")
        return run_phase2_cnn_detection(features, force_verdict)
    else:
        # phase1 or default
        return run_phase1_stub_detection(features, force_verdict)


def run_segment_detection(segments: list, phase: str = "phase1", force_verdict: str = None) -> list:
    """
    Runs ML prediction on an array of AudioSegment objects.
    Returns a list of SegmentPrediction objects.
    """
    from backend.detection.aggregator import SegmentPrediction
    from backend.audio.features import extract_acoustic_features
    from backend.config import settings

    predictions = []
    
    for seg in segments:
        # Extract features just for this segment
        # In a real heavy model, we'd batch these for performance, but this fits the architecture
        features = extract_acoustic_features(seg.audio_data, sr=settings.sample_rate)
        
        real_prob = run_ai_detection(features, phase=phase, force_verdict=force_verdict)
        fake_prob = 1.0 - real_prob
        
        if real_prob >= 0.75:
            pred_label = "REAL"
            conf = real_prob
        elif real_prob <= 0.40:
            pred_label = "FAKE"
            conf = fake_prob
        else:
            pred_label = "UNCERTAIN"
            conf = 1.0 - 2 * abs(real_prob - 0.5)
            
        predictions.append(SegmentPrediction(
            segment_id=seg.segment_id,
            start_time=seg.start_time,
            end_time=seg.end_time,
            fake_probability=fake_prob,
            real_probability=real_prob,
            predicted_label=pred_label,
            confidence=conf
        ))
        
    return predictions
