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

    Frequency pooling uses kernel (1, 2) so the 13 MFCC bins are not crushed
    before AdaptiveAvgPool — higher cepstra carry spoof-discriminative cues.
    """
    def __init__(self):
        super(SimpleCNNDetector, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(64, 1)
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


def warmup_cnn_model() -> str:
    """Load weights and run a dummy forward pass so the first user request is fast."""
    model, status = _load_cnn_model()
    if model is not None:
        dummy = torch.zeros(1, 1, N_MFCC, N_TIME_STEPS)
        with torch.inference_mode():
            model(dummy)
    return status


# ── Feature helpers ────────────────────────────────────────────────────────────

def _as_mfcc_array(features: dict) -> np.ndarray:
    seq = features.get("mfcc_sequence", [])
    mfcc = np.asarray(seq, dtype=np.float32)
    if mfcc.ndim != 2:
        return np.zeros((N_MFCC, N_TIME_STEPS), dtype=np.float32)
    if mfcc.shape[0] != N_MFCC:
        return np.zeros((N_MFCC, N_TIME_STEPS), dtype=np.float32)
    if mfcc.shape[1] < N_TIME_STEPS:
        mfcc = np.pad(mfcc, ((0, 0), (0, N_TIME_STEPS - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :N_TIME_STEPS]
    return mfcc


def _mfcc_to_tensor(mfcc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(mfcc, dtype=np.float32)[np.newaxis, np.newaxis, :, :])


def _cnn_real_probability(model: nn.Module, mfcc_batch: np.ndarray) -> np.ndarray:
    """
    mfcc_batch: (B, N_MFCC, N_TIME_STEPS)
    Returns P(REAL) in [0.01, 0.99] as a 1-D numpy array.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(mfcc_batch, dtype=np.float32)).unsqueeze(1)
    with torch.inference_mode():
        logits = model(tensor).reshape(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return np.clip(probs.astype(np.float64), 0.01, 0.99)


# ── Phase 1: Heuristic Detector ────────────────────────────────────────────────

def _heuristic_score_from_arrays(
    mfcc_stds,
    sc_mean: float,
    zcr_mean: float = 0.05,
    mfcc_sequence=None,
    flatness: float = 0.1,
    delta_std: float = None,
) -> float:
    """
    Acoustic anti-spoof heuristic using cues that transfer to real recordings:
    natural speech has higher cepstral / temporal variance; vocoders are too
    stable, too clean, or show unnatural spectral centroids.
    """
    score = 0.50

    if mfcc_stds is not None and len(mfcc_stds) > 0:
        std_variance = float(np.mean(mfcc_stds))
        if std_variance > 18:
            score += 0.22
        elif std_variance > 12:
            score += 0.10
        elif std_variance < 6:
            score -= 0.28
        elif std_variance < 9:
            score -= 0.14

    if sc_mean > 2800:
        score -= 0.16
    elif sc_mean > 2400:
        score -= 0.08
    elif sc_mean < 900:
        score -= 0.10
    elif 1100 <= sc_mean <= 2200:
        score += 0.08

    if zcr_mean < 0.02:
        score -= 0.08
    elif zcr_mean > 0.18:
        score -= 0.06

    if flatness is not None:
        if flatness < 0.02:
            score -= 0.08  # overly peaky / buzzy vocoder
        elif flatness > 0.35:
            score -= 0.05

    seq = None
    if mfcc_sequence is not None:
        seq = np.asarray(mfcc_sequence, dtype=np.float32)
        if seq.ndim == 2 and seq.shape[1] > 2:
            if delta_std is None:
                delta_std = float(np.mean(np.std(np.diff(seq, axis=1), axis=1)))

    if delta_std is not None:
        if delta_std < 1.5:
            score -= 0.20  # temporally frozen → synthetic
        elif delta_std < 3.0:
            score -= 0.08
        elif delta_std > 8.0:
            score += 0.12

    return float(np.clip(score, 0.05, 0.95))


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

    try:
        return _heuristic_score_from_arrays(
            mfcc_stds=features.get("mfcc_std", []),
            sc_mean=float(features.get("spectral_centroid_mean", 1500) or 1500),
            zcr_mean=float(features.get("zero_crossing_rate_mean", 0.05) or 0.05),
            mfcc_sequence=features.get("mfcc_sequence"),
            flatness=float(features.get("spectral_flatness_mean", 0.1) or 0.1),
            delta_std=features.get("mfcc_delta_std"),
        )
    except Exception as e:
        print(f"[Detector] Heuristic error: {e}")
        return 0.5


def heuristic_score_from_mfcc(mfcc: np.ndarray) -> float:
    """Fast heuristic using only a (13, T) MFCC matrix — used for live/segment paths."""
    if mfcc is None or mfcc.size == 0:
        return 0.5
    stds = np.std(mfcc, axis=1)
    delta_std = float(np.mean(np.std(np.diff(mfcc, axis=1), axis=1))) if mfcc.shape[1] > 2 else 4.0
    return _heuristic_score_from_arrays(
        mfcc_stds=stds,
        sc_mean=1500.0,
        mfcc_sequence=mfcc,
        delta_std=delta_std,
    )


# ── Phase 2: CNN Detector ──────────────────────────────────────────────────────

def run_phase2_cnn_detection(features: dict, force_verdict: str = None) -> float:
    """
    Runs the trained SimpleCNNDetector using the fixed-length MFCC sequence.
    Falls back to Phase 1 if the model is not available.
    Blends a light acoustic prior so out-of-domain clips are less brittle.
    """
    if force_verdict:
        return run_phase1_stub_detection(features, force_verdict)

    model, status = _load_cnn_model()
    if status == "MODEL_UNAVAILABLE" or model is None:
        return run_phase1_stub_detection(features)

    try:
        mfcc_sequence = _as_mfcc_array(features)
        if mfcc_sequence.shape != (N_MFCC, N_TIME_STEPS):
            return run_phase1_stub_detection(features)

        cnn_score = float(_cnn_real_probability(model, mfcc_sequence[np.newaxis, ...])[0])
        heuristic_score = run_phase1_stub_detection(features)
        # CNN dominates; heuristic regularizes out-of-domain recordings.
        score = 0.75 * cnn_score + 0.25 * heuristic_score
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
    Batches CNN inference so a 10-segment clip is one forward pass, not ten.
    """
    from backend.detection.aggregator import SegmentPrediction
    from backend.audio.features import extract_mfcc_for_model
    from backend.config import settings as _settings

    if not segments:
        return []

    predictions = []
    use_cnn = phase in ("phase2", "phase3", "phase4") and not force_verdict
    model, status = (_load_cnn_model() if use_cnn else (None, None))
    cnn_ready = use_cnn and status in ("REAL_MODEL", "DEMO_MODEL") and model is not None

    mfccs = [extract_mfcc_for_model(seg.audio_data, sr=_settings.sample_rate) for seg in segments]

    if force_verdict:
        real_probs = [run_phase1_stub_detection({}, force_verdict) for _ in segments]
    elif cnn_ready:
        batch = np.stack(mfccs, axis=0)
        cnn_probs = _cnn_real_probability(model, batch)
        heur_probs = np.array([heuristic_score_from_mfcc(m) for m in mfccs], dtype=np.float64)
        real_probs = np.clip(0.75 * cnn_probs + 0.25 * heur_probs, 0.01, 0.99).tolist()
    else:
        real_probs = [heuristic_score_from_mfcc(m) for m in mfccs]

    for seg, real_prob in zip(segments, real_probs):
        real_prob = float(real_prob)
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
