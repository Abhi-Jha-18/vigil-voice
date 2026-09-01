def make_decision(real_score: float) -> dict:
    """
    Evaluates the realness score (0.0 to 1.0) and assigns a verdict,
    confidence percentage, and risk level.
    
    Scores represent the probability that the audio is REAL.
    - REAL: >= 0.75
    - FAKE: <= 0.40
    - UNCERTAIN: 0.40 < score < 0.75
    """
    # 1. Determine Verdict and Confidence
    if real_score >= 0.75:
        verdict = "REAL"
        # Confidence is relative to the REAL class
        confidence = real_score
    elif real_score <= 0.40:
        verdict = "FAKE"
        # Confidence is relative to the FAKE class
        confidence = 1.0 - real_score
    else:
        verdict = "UNCERTAIN"
        # For uncertain, confidence is the degree of uncertainty
        # (closer to 0.5 means higher uncertainty, so we scale it)
        confidence = 1.0 - 2 * abs(real_score - 0.5) # Higher value = more uncertain

    # Convert confidence to percentage
    confidence_pct = float(round(confidence * 100, 2))

    # 2. Map Risk Level
    if verdict == "REAL":
        if real_score >= 0.90:
            risk_level = "LOW RISK"
        else:
            risk_level = "MINIMAL RISK"
    elif verdict == "FAKE":
        if real_score <= 0.15:
            risk_level = "CRITICAL RISK"
        else:
            risk_level = "HIGH RISK"
    else:
        risk_level = "MODERATE RISK"
        
    return {
        "verdict": verdict,
        "confidence": confidence_pct,
        "raw_score": float(round(real_score, 4)),
        "risk_level": risk_level
    }
