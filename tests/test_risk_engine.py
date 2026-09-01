import pytest
from backend.detection.risk import calculate_risk_score, get_risk_level

def test_risk_score_boundaries():
    # If max fake prob is 0.3, base score is 30.
    # No other penalties.
    score1 = calculate_risk_score(
        max_fake_prob=0.3,
        avg_fake_prob=0.2,
        temporal_variance=0.1,
        suspicious_segments_count=0,
        total_segments=10,
        quality_status="GOOD"
    )
    assert score1 == 30
    assert get_risk_level(score1) == "LOW"

    # Base 60, no penalties -> 60
    score2 = calculate_risk_score(
        max_fake_prob=0.6,
        avg_fake_prob=0.5,
        temporal_variance=0.1,
        suspicious_segments_count=4,
        total_segments=10,
        quality_status="GOOD"
    )
    assert score2 == 60
    assert get_risk_level(score2) == "MEDIUM"
    
    # Base 60, suspicious ratio > 0.5 (+10) -> 70
    score3 = calculate_risk_score(
        max_fake_prob=0.6,
        avg_fake_prob=0.5,
        temporal_variance=0.1,
        suspicious_segments_count=6,
        total_segments=10,
        quality_status="GOOD"
    )
    assert score3 == 70
    assert get_risk_level(score3) == "HIGH"
    
    # Base 75, consistent spoofing (+15) -> 90
    score4 = calculate_risk_score(
        max_fake_prob=0.75,
        avg_fake_prob=0.7,
        temporal_variance=0.01,
        suspicious_segments_count=8,
        total_segments=10,
        quality_status="GOOD"
    )
    assert score4 == 100 # 75 + 10 (ratio) + 15 (consistent) = 100
    assert get_risk_level(score4) == "CRITICAL"

def test_quality_degraded_penalty():
    # Base 80, but degraded quality tempers it (80 * 0.9 = 72)
    score = calculate_risk_score(
        max_fake_prob=0.8,
        avg_fake_prob=0.7,
        temporal_variance=0.1,
        suspicious_segments_count=1,
        total_segments=10,
        quality_status="DEGRADED"
    )
    assert score == 72
    assert get_risk_level(score) == "HIGH"

if __name__ == "__main__":
    test_risk_score_boundaries()
    test_quality_degraded_penalty()
    print("Risk engine tests passed.")
