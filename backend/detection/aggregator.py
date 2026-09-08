import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class SegmentPrediction:
    segment_id: int
    start_time: float
    end_time: float
    fake_probability: float
    real_probability: float
    predicted_label: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start_time": float(round(self.start_time, 3)),
            "end_time": float(round(self.end_time, 3)),
            "fake_probability": float(round(self.fake_probability, 4)),
            "real_probability": float(round(self.real_probability, 4)),
            "predicted_label": self.predicted_label,
            "confidence": float(round(self.confidence, 4))
        }

@dataclass
class AggregationResult:
    overall_probability: float  # P(REAL)
    maximum_fake_probability: float
    average_fake_probability: float
    suspicious_segments: List[SegmentPrediction]
    temporal_consistency: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_probability": float(round(self.overall_probability, 4)),
            "maximum_fake_probability": float(round(self.maximum_fake_probability, 4)),
            "average_fake_probability": float(round(self.average_fake_probability, 4)),
            "suspicious_segments": [s.to_dict() for s in self.suspicious_segments],
            "temporal_consistency": self.temporal_consistency
        }

def calculate_temporal_consistency(segments: List[SegmentPrediction], fake_threshold: float = 0.6) -> Dict[str, Any]:
    """Calculates metrics around contiguous sequences of fake segments."""
    if not segments:
        return {"contiguous_fake_segments": 0, "variance": 0.0, "is_consistent": True}
        
    fake_probs = [s.fake_probability for s in segments]
    variance = float(np.var(fake_probs))
    
    max_contiguous = 0
    current_contiguous = 0
    
    for prob in fake_probs:
        if prob >= fake_threshold:
            current_contiguous += 1
            max_contiguous = max(max_contiguous, current_contiguous)
        else:
            current_contiguous = 0
            
    # If variance is high, it means the model oscillates between real/fake, which 
    # could indicate a localized injection or an unreliable prediction sequence.
    is_consistent = variance < 0.1
            
    return {
        "contiguous_fake_segments": max_contiguous,
        "variance": float(round(variance, 4)),
        "is_consistent": is_consistent
    }

def aggregate_predictions(
    segments: List[SegmentPrediction], 
    strategy: str = "max_risk",
    suspicious_threshold: float = 0.6
) -> AggregationResult:
    """
    Aggregates a list of segment predictions into a single call-level result.
    
    Strategies:
    - 'max_risk': The call's overall FAKE probability is the maximum of any single segment.
      This is highly sensitive to isolated deepfake injections.
    - 'average': Simple mean of all segment probabilities.
    """
    if not segments:
        return AggregationResult(0.5, 0.0, 0.0, [], {})
        
    fake_probs = [s.fake_probability for s in segments]
    
    max_fake_prob = float(np.max(fake_probs))
    avg_fake_prob = float(np.mean(fake_probs))
    
    suspicious_segments = [s for s in segments if s.fake_probability >= suspicious_threshold]
    temporal_consistency = calculate_temporal_consistency(segments, suspicious_threshold)
    
    if strategy == "max_risk":
        overall_fake_prob = max_fake_prob
    elif strategy == "average":
        overall_fake_prob = avg_fake_prob
    elif strategy in ("robust_risk", "hybrid"):
        # Single-spike false alarm protection:
        # If there is only 1 isolated spike in a clip with >= 3 segments and the average fake probability
        # across all segments is low (<0.35), do not allow an isolated anomaly to dictate 90%+ fake confidence.
        if len(segments) >= 3 and len(suspicious_segments) <= 1 and avg_fake_prob < 0.35:
            # Weighted blend of average (60%) and peak (40%) to temper the false alarm
            overall_fake_prob = float(np.clip(0.60 * avg_fake_prob + 0.40 * max_fake_prob, 0.0, 1.0))
        else:
            overall_fake_prob = max_fake_prob
    else:
        # Default to max_risk for security
        overall_fake_prob = max_fake_prob
        
    overall_real_prob = 1.0 - overall_fake_prob
    
    return AggregationResult(
        overall_probability=overall_real_prob,
        maximum_fake_probability=max_fake_prob,
        average_fake_probability=avg_fake_prob,
        suspicious_segments=suspicious_segments,
        temporal_consistency=temporal_consistency
    )
