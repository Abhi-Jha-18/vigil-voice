import numpy as np
from typing import Dict, Any, List

def calculate_risk_score(
    max_fake_prob: float, 
    avg_fake_prob: float, 
    temporal_variance: float,
    suspicious_segments_count: int,
    total_segments: int,
    quality_status: str
) -> int:
    """
    Converts detection evidence into a security-oriented risk score (0-100).
    Note: This is a prototype decision-support score, not a scientifically validated risk score.
    """
    if total_segments == 0:
        return 0
        
    # Base risk is heavily driven by the maximum localized fake probability (max_risk strategy)
    base_score = max_fake_prob * 100
    
    # Penalize (increase risk) if multiple segments are suspicious
    suspicious_ratio = suspicious_segments_count / total_segments
    if suspicious_ratio > 0.5:
        base_score += 10
    
    # Penalize heavily if temporal variance is extremely low and fake prob is high 
    # (consistent spoofing throughout the call)
    if temporal_variance < 0.05 and max_fake_prob > 0.7:
        base_score += 15
        
    # Slightly reduce risk confidence if audio quality is DEGRADED
    if quality_status == "DEGRADED":
        # We don't drop risk completely, but we temper extreme scores slightly 
        # because the model might be hallucinating on noise.
        if base_score > 50:
            base_score *= 0.9
            
    # Clip to 0-100
    return int(np.clip(base_score, 0, 100))

def get_risk_level(score: int) -> str:
    """Maps a 0-100 score to a categorical risk level."""
    if score <= 30:
        return "LOW"
    elif score <= 60:
        return "MEDIUM"
    elif score <= 80:
        return "HIGH"
    else:
        return "CRITICAL"

def generate_risk_report(
    max_fake_prob: float, 
    avg_fake_prob: float, 
    temporal_variance: float,
    suspicious_segments: List[Any],
    total_segments: int,
    quality_status: str
) -> Dict[str, Any]:
    """Generates the full dynamic risk engine payload."""
    score = calculate_risk_score(
        max_fake_prob, 
        avg_fake_prob, 
        temporal_variance, 
        len(suspicious_segments), 
        total_segments, 
        quality_status
    )
    level = get_risk_level(score)
    
    # Determine overall confidence based on variance and quality
    confidence = "HIGH"
    if quality_status == "DEGRADED" or temporal_variance > 0.2:
        confidence = "MEDIUM"
    if quality_status == "INSUFFICIENT":
        confidence = "LOW"
        
    return {
        "risk_score": score,
        "risk_level": level,
        "fake_probability": float(round(max_fake_prob, 4)),
        "confidence": confidence,
        "suspicious_segments": [s.to_dict() for s in suspicious_segments],
        # The 'signals' array will be populated by the explainability layer
        "signals": [],
        "_disclaimer": "This risk score is for prototype decision-support only. It does not make automatic financial or critical security decisions."
    }
