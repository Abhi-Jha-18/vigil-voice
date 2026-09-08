import pytest
from backend.detection.aggregator import (
    SegmentPrediction, 
    aggregate_predictions,
    calculate_temporal_consistency
)

def create_mock_segment(segment_id, fake_prob):
    return SegmentPrediction(
        segment_id=segment_id,
        start_time=segment_id * 3.0,
        end_time=segment_id * 3.0 + 3.0,
        fake_probability=fake_prob,
        real_probability=1.0 - fake_prob,
        predicted_label="FAKE" if fake_prob > 0.6 else "REAL",
        confidence=fake_prob if fake_prob > 0.6 else (1.0 - fake_prob)
    )

def test_max_risk_strategy():
    # 3 REAL segments, 1 FAKE segment
    segments = [
        create_mock_segment(0, 0.1),
        create_mock_segment(1, 0.2),
        create_mock_segment(2, 0.95), # High risk spike
        create_mock_segment(3, 0.15)
    ]
    
    result = aggregate_predictions(segments, strategy="max_risk")
    
    assert result.maximum_fake_probability == 0.95
    assert result.overall_probability == pytest.approx(0.05) # 1.0 - 0.95 (FAKE)
    assert len(result.suspicious_segments) == 1
    assert result.suspicious_segments[0].segment_id == 2
    
def test_average_strategy():
    segments = [
        create_mock_segment(0, 0.0),
        create_mock_segment(1, 0.0),
        create_mock_segment(2, 0.6),
        create_mock_segment(3, 0.2)
    ]
    
    result = aggregate_predictions(segments, strategy="average")
    
    assert result.average_fake_probability == 0.2
    assert result.overall_probability == 0.8
    assert len(result.suspicious_segments) == 1
    
def test_temporal_consistency_contiguous():
    segments = [
        create_mock_segment(0, 0.9),
        create_mock_segment(1, 0.95),
        create_mock_segment(2, 0.85),
        create_mock_segment(3, 0.2)
    ]
    
    metrics = calculate_temporal_consistency(segments, fake_threshold=0.6)
    
    assert metrics["contiguous_fake_segments"] == 3
    # Variance of [0.9, 0.95, 0.85, 0.2] is relatively high because of the 0.2
    # but let's just check the contiguous logic
    
def test_temporal_consistency_isolated():
    segments = [
        create_mock_segment(0, 0.1),
        create_mock_segment(1, 0.95),
        create_mock_segment(2, 0.1),
        create_mock_segment(3, 0.1)
    ]
    
    metrics = calculate_temporal_consistency(segments, fake_threshold=0.6)
    
    assert metrics["contiguous_fake_segments"] == 1
    # Variance of [0.1, 0.95, 0.1, 0.1] is high, so is_consistent should be False
    assert not metrics["is_consistent"]
    
def test_empty_segments():
    result = aggregate_predictions([])
    
    assert result.overall_probability == 0.5
    assert result.maximum_fake_probability == 0.0
    assert result.average_fake_probability == 0.0
    assert len(result.suspicious_segments) == 0

def test_robust_risk_strategy_isolated_spike():
    # 3 authentic segments, 1 isolated anomalous spike
    segments = [
        create_mock_segment(0, 0.10),
        create_mock_segment(1, 0.90), # isolated false alarm spike
        create_mock_segment(2, 0.10),
        create_mock_segment(3, 0.10),
    ]
    result = aggregate_predictions(segments, strategy="robust_risk")
    # Peak is 0.90, but single-spike protection dampens overall fake prob
    assert result.maximum_fake_probability == 0.90
    assert result.overall_probability > 0.40  # Not crushed down to 0.10

def test_robust_risk_strategy_repeated_spoofing():
    # Repeated spoofing segments -> must enforce maximum risk
    segments = [
        create_mock_segment(0, 0.10),
        create_mock_segment(1, 0.92),
        create_mock_segment(2, 0.88),
        create_mock_segment(3, 0.10),
    ]
    result = aggregate_predictions(segments, strategy="robust_risk")
    assert result.maximum_fake_probability == 0.92
    assert result.overall_probability == pytest.approx(0.08)  # Enforces full max risk

def test_blend_cnn_and_heuristic_divergence():
    from backend.detection.detector import blend_cnn_and_heuristic
    
    # Divergence: CNN claims fake (0.02) but Heuristic confirms authentic voice dynamics (0.85)
    blended = blend_cnn_and_heuristic(0.02, 0.85)
    assert blended >= 0.45, f"Expected uncertain buffer >= 0.45, got {blended}"

    # Concordant fake: both detect spoof
    blended_fake = blend_cnn_and_heuristic(0.05, 0.10)
    assert blended_fake < 0.20, f"Expected strong fake < 0.20, got {blended_fake}"

    # Concordant authentic: both confirm real
    blended_real = blend_cnn_and_heuristic(0.90, 0.85)
    assert blended_real > 0.80, f"Expected strong real > 0.80, got {blended_real}"

if __name__ == "__main__":
    test_max_risk_strategy()
    test_average_strategy()
    test_temporal_consistency_contiguous()
    test_temporal_consistency_isolated()
    test_empty_segments()
    test_robust_risk_strategy_isolated_spike()
    test_robust_risk_strategy_repeated_spoofing()
    test_blend_cnn_and_heuristic_divergence()
    print("All segment aggregation tests passed.")
