from typing import Dict, Any, List

def get_recommendation(
    risk_level: str,
    suspicious_segments_count: int = 0,
    evidence_signals: List[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Generates actionable security recommendations based on the risk level.
    Never automatically approves or rejects transactions.
    """
    evidence_signals = evidence_signals or []
    
    # 1. Synthesize a contextual reason
    reason = "No significant synthetic speech indicators detected."
    if risk_level == "CRITICAL":
        reason = "Severe synthetic speech indicators detected, suggesting high probability of voice cloning or spoofing."
    elif risk_level == "HIGH":
        if suspicious_segments_count > 1:
            reason = f"Strong synthetic speech indicators detected across {suspicious_segments_count} segments."
        else:
            reason = "Strong synthetic speech indicators detected."
    elif risk_level == "MEDIUM":
        reason = "Moderate acoustic anomalies detected that warrant caution."
        
    # Append any specific high-strength evidence to the reason
    high_strength_evidence = [e["description"] for e in evidence_signals if e.get("strength") == "HIGH"]
    if high_strength_evidence:
        reason += f" Details: {high_strength_evidence[0]}"

    # 2. Map Recommendation and Verification Method
    if risk_level == "LOW":
        recommendation = "Continue normal interaction. Model indicates low risk of voice spoofing."
        method = "Standard authentication"
    elif risk_level == "MEDIUM":
        recommendation = "Recommend secondary identity verification before proceeding with sensitive actions."
        method = "SMS OTP or Security Question"
    elif risk_level == "HIGH":
        recommendation = "Recommend not relying on voice alone for sensitive actions. Perform independent identity verification before accepting instructions."
        method = "Push Notification Approval or Video Verification"
    elif risk_level == "CRITICAL":
        recommendation = "Recommend pausing the sensitive action immediately and escalating the call for manual security verification."
        method = "Out-of-band manual callback or IT Security Escalation"
    else:
        recommendation = "Unknown risk level. Proceed with standard caution."
        method = "Standard authentication"
        
    return {
        "risk_level": risk_level,
        "reason": reason,
        "recommendation": recommendation,
        "recommended_verification_method": method
    }
