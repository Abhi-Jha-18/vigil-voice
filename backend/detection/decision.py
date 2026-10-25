THRESHOLD_REAL = 0.75
THRESHOLD_FAKE = 0.40


def classify_score(real_score: float) -> tuple[str, float]:
    """
    Evaluates realness score (0.0 to 1.0) and assigns verdict and raw confidence.
    Single source of truth for REAL/FAKE/UNCERTAIN classification thresholds.

    - REAL: >= 0.75
    - FAKE: <= 0.40
    - UNCERTAIN: 0.40 < real_score < 0.75
    """
    if real_score >= THRESHOLD_REAL:
        return "REAL", float(real_score)
    elif real_score <= THRESHOLD_FAKE:
        return "FAKE", float(1.0 - real_score)
    else:
        return "UNCERTAIN", float(1.0 - 2 * abs(real_score - 0.5))


def make_decision(real_score: float) -> dict:
    """
    Evaluates the realness score (0.0 to 1.0) and assigns a verdict,
    confidence percentage, and risk level.
    """
    # 1. Determine Verdict and Confidence from single source of truth
    verdict, confidence = classify_score(real_score)

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
