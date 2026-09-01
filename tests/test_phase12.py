import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Fix python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.metrics import calculate_eer, calculate_metrics

def test_eer_perfect_separation():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    
    eer, threshold = calculate_eer(y_true, y_scores)
    
    # Perfect separation means EER should be 0.0
    assert eer == 0.0
    assert 0.3 <= threshold <= 0.7

def test_eer_random_guess():
    # If scores are inverse or completely overlapping
    y_true = np.array([1, 0, 1, 0])
    y_scores = np.array([0.5, 0.5, 0.5, 0.5])
    
    eer, threshold = calculate_eer(y_true, y_scores)
    
    # Should be close to 0.5 for random guessing
    assert eer == pytest.approx(0.5, abs=0.1)

def test_calculate_metrics_far_frr_roc():
    y_true = np.array([1, 1, 0, 0])
    y_scores = np.array([0.8, 0.4, 0.6, 0.2])
    # Predictions at 0.5 threshold:
    # y_true = [1, 1, 0, 0]
    # y_pred = [1, 0, 1, 0]
    # TP = 1 (idx 0), FN = 1 (idx 1), FP = 1 (idx 2), TN = 1 (idx 3)
    
    metrics = calculate_metrics(y_true, y_scores, threshold=0.5)
    
    assert metrics['accuracy'] == 0.5
    assert metrics['far'] == 0.5  # FP / (TN + FP) = 1 / 2 = 0.5
    assert metrics['frr'] == 0.5  # FN / (TP + FN) = 1 / 2 = 0.5
    
    # For ROC AUC: 
    # Positive class (1) scores: 0.8, 0.4
    # Negative class (0) scores: 0.6, 0.2
    # All pairs (pos, neg):
    # (0.8, 0.6) -> 1
    # (0.8, 0.2) -> 1
    # (0.4, 0.6) -> 0
    # (0.4, 0.2) -> 1
    # Concordant pairs = 3. Total pairs = 4. AUC = 0.75
    assert metrics['roc_auc'] == 0.75

def test_calculate_metrics_single_class():
    y_true = np.array([1, 1, 1])
    y_scores = np.array([0.9, 0.8, 0.7])
    
    metrics = calculate_metrics(y_true, y_scores, threshold=0.5)
    # Should safely handle ROC-AUC without crashing by returning 0.5
    assert metrics['roc_auc'] == 0.5
    assert metrics['accuracy'] == 1.0
