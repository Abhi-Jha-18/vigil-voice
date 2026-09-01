import pytest
from backend.detection.recommendations import get_recommendation

def test_recommendations():
    # LOW Risk
    rec_low = get_recommendation("LOW")
    assert rec_low["risk_level"] == "LOW"
    assert "Continue normal interaction" in rec_low["recommendation"]
    assert rec_low["recommended_verification_method"] == "Standard authentication"
    
    # MEDIUM Risk
    rec_medium = get_recommendation("MEDIUM")
    assert rec_medium["risk_level"] == "MEDIUM"
    assert "secondary identity verification" in rec_medium["recommendation"]
    assert "SMS OTP" in rec_medium["recommended_verification_method"]
    
    # HIGH Risk
    rec_high = get_recommendation("HIGH", suspicious_segments_count=3)
    assert rec_high["risk_level"] == "HIGH"
    assert "not relying on voice alone" in rec_high["recommendation"]
    assert "Push Notification" in rec_high["recommended_verification_method"]
    assert "across 3 segments" in rec_high["reason"]
    
    # CRITICAL Risk
    rec_critical = get_recommendation("CRITICAL")
    assert rec_critical["risk_level"] == "CRITICAL"
    assert "pausing the sensitive action immediately" in rec_critical["recommendation"]
    assert "manual callback" in rec_critical["recommended_verification_method"]
    
    # High strength evidence
    evidence = [{"description": "Unusually high spectral centroid.", "strength": "HIGH"}]
    rec_evidence = get_recommendation("HIGH", suspicious_segments_count=1, evidence_signals=evidence)
    assert "Unusually high spectral centroid" in rec_evidence["reason"]

if __name__ == "__main__":
    test_recommendations()
    print("Recommendation tests passed.")
