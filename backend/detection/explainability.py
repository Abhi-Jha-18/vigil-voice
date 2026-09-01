from typing import Dict, Any, List

def generate_evidence(
    features: Dict[str, Any], 
    temporal_variance: float, 
    suspicious_segments_count: int,
    total_segments: int
) -> List[Dict[str, str]]:
    """
    Generates explainability evidence signals based on acoustic features and temporal properties.
    Returns a list of evidence dicts.
    """
    evidence = []
    
    # 1. Spectral anomalies
    sc_mean = features.get("spectral_centroid_mean", 1500)
    if sc_mean > 2500:
        evidence.append({
            "type": "spectral",
            "description": "Unusually high spectral centroid observed, a possible indicator of synthetic or vocoder-based generation anomaly.",
            "strength": "HIGH" if sc_mean > 3000 else "MEDIUM"
        })
    elif sc_mean < 1200:
        evidence.append({
            "type": "spectral",
            "description": "Unusually low spectral centroid observed, which can be a contributing signal for muffled replay attacks.",
            "strength": "MEDIUM"
        })
        
    # 2. Acoustic Feature Deviations
    mfcc_mean_list = features.get("mfcc_mean", [])
    if mfcc_mean_list and len(mfcc_mean_list) > 0:
        # Check first MFCC (energy/loudness proxy)
        mfcc_0 = mfcc_mean_list[0]
        if mfcc_0 < -300:
            evidence.append({
                "type": "acoustic",
                "description": "Overall MFCC energy profile deviates from natural expressive speech distributions.",
                "strength": "LOW"
            })
            
    # 3. Temporal / Prosodic consistency
    if total_segments > 0:
        suspicious_ratio = suspicious_segments_count / total_segments
        if suspicious_ratio > 0.5:
            evidence.append({
                "type": "prosodic",
                "description": f"Model evidence suggests {suspicious_segments_count} out of {total_segments} segments exhibit spoofing characteristics.",
                "strength": "HIGH" if suspicious_ratio > 0.8 else "MEDIUM"
            })
            
        if temporal_variance < 0.05 and suspicious_ratio > 0.3:
            evidence.append({
                "type": "prosodic",
                "description": "Extremely low variance in predictions suggests consistent, homogeneous generation across the entire audio file.",
                "strength": "HIGH"
            })

    return evidence
