THRESHOLD_REAL = 0.75
THRESHOLD_FAKE = 0.40


def classify_score(
    real_score: float,
    threshold_real: float | None = None,
    threshold_fake: float | None = None,
) -> tuple[str, float]:
    """
    Evaluates realness score (0.0 to 1.0) and assigns verdict and raw confidence.
    Single source of truth for REAL/FAKE/UNCERTAIN classification thresholds.
    Accepts calibrated threshold_real and threshold_fake from model calibration metadata,
    defaulting to THRESHOLD_REAL (0.75) and THRESHOLD_FAKE (0.40).
    """
    t_real = threshold_real if threshold_real is not None else THRESHOLD_REAL
    t_fake = threshold_fake if threshold_fake is not None else THRESHOLD_FAKE

    if real_score >= t_real:
        return "REAL", float(real_score)
    elif real_score <= t_fake:
        return "FAKE", float(1.0 - real_score)
    else:
        return "UNCERTAIN", float(1.0 - 2 * abs(real_score - 0.5))


def make_decision(
    real_score: float,
    threshold_real: float | None = None,
    threshold_fake: float | None = None,
) -> dict:
    """
    Evaluates the realness score (0.0 to 1.0) and assigns a verdict,
    confidence percentage, and risk level.
    """
    # 1. Determine Verdict and Confidence from single source of truth
    verdict, confidence = classify_score(
        real_score,
        threshold_real=threshold_real,
        threshold_fake=threshold_fake,
    )

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
