import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes standard accuracy.
    """
    return float(np.mean(y_true == y_pred))

def calculate_metrics(y_true: np.ndarray, y_pred_scores: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Computes precision, recall, f1, and accuracy at a given threshold.
    Note: y_true should be 1 for REAL, 0 for FAKE.
    """
    y_pred = (y_pred_scores >= threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = float((tp + tn) / len(y_true))
    far = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    frr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_scores))
    except ValueError:
        roc_auc = 0.5  # Only one class present in evaluation
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": far,
        "frr": frr,
        "roc_auc": roc_auc
    }

def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """
    Calculates the Equal Error Rate (EER) which is standard for ASVspoof evaluations.
    EER is the point where the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR).
    
    Args:
        y_true: Ground truth labels (1 for REAL, 0 for FAKE).
        y_scores: Decision scores (probability of being REAL).
        
    Returns:
        eer (float): The Equal Error Rate.
        threshold (float): The threshold corresponding to EER.
    """
    # FAR = FP / (TN + FP) -> False Alarm Rate / False Acceptance
    # FRR = FN / (TP + FN) -> Miss Rate / False Rejection
    
    # We negate scores since roc_curve expects higher scores for the positive class (REAL)
    # FAR is FP rate, FRR is 1 - TP rate (since TP rate is recall = TP/(TP+FN) = 1 - FRR)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    
    fnr = 1 - tpr
    
    # Find the point where FPR (FAR) and FNR (FRR) are closest
    idx = np.nanargmin(np.absolute(fpr - fnr))
    
    eer = float((fpr[idx] + fnr[idx]) / 2)
    threshold = float(thresholds[idx])
    
    return eer, threshold
