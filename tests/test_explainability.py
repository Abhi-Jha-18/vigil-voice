import pytest
from backend.detection.explainability import generate_evidence

def test_spectral_anomalies():
    # High spectral centroid
    evidence1 = generate_evidence(
        features={"spectral_centroid_mean": 3200},
        temporal_variance=0.1,
        suspicious_segments_count=0,
        total_segments=10
    )
    assert len(evidence1) == 1
    assert evidence1[0]["type"] == "spectral"
    assert evidence1[0]["strength"] == "HIGH"
    
    # Low spectral centroid
    evidence2 = generate_evidence(
        features={"spectral_centroid_mean": 900},
        temporal_variance=0.1,
        suspicious_segments_count=0,
        total_segments=10
    )
    assert len(evidence2) == 1
    assert evidence2[0]["type"] == "spectral"
    assert evidence2[0]["strength"] == "MEDIUM"
    
    # Normal spectral centroid
    evidence3 = generate_evidence(
        features={"spectral_centroid_mean": 1500},
        temporal_variance=0.1,
        suspicious_segments_count=0,
        total_segments=10
    )
    assert len(evidence3) == 0

def test_prosodic_anomalies():
    evidence = generate_evidence(
        features={"spectral_centroid_mean": 1500},
        temporal_variance=0.01,
        suspicious_segments_count=9,
        total_segments=10
    )
    
    assert len(evidence) == 2
    types = [e["type"] for e in evidence]
    assert "prosodic" in types
    
    # Check that high suspicious ratio generates HIGH strength evidence
    ratio_evidence = [e for e in evidence if "out of" in e["description"]][0]
    assert ratio_evidence["strength"] == "HIGH"
    
    # Check that low variance consistent spoofing generates evidence
    variance_evidence = [e for e in evidence if "variance" in e["description"].lower()][0]
    assert variance_evidence["strength"] == "HIGH"

if __name__ == "__main__":
    test_spectral_anomalies()
    test_prosodic_anomalies()
    print("Explainability tests passed.")
