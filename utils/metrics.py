"""
Evaluation metrics for anomaly detection
"""

import numpy as np
from typing import Tuple, List, Dict
from sklearn.metrics import precision_recall_curve, auc, roc_curve

def calculate_detection_delay(scores: np.ndarray,
                            labels: np.ndarray,
                            threshold: float) -> float:
    """
    Calculate average detection delay for attacks
    
    Args:
        scores: Anomaly scores
        labels: True attack labels (0/1)
        threshold: Detection threshold
        
    Returns:
        float: Average detection delay in samples
    """
    # Find attack start points
    attack_starts = np.where(np.diff(labels) == 1)[0] + 1
    
    if len(attack_starts) == 0:
        return 0
    
    delays = []
    for start in attack_starts:
        # Find first detection after attack start
        detections = np.where(scores[start:] > threshold)[0]
        if len(detections) > 0:
            delays.append(detections[0])
            
    return np.mean(delays) if delays else np.inf

def calculate_etapr(scores: np.ndarray,
                   labels: np.ndarray,
                   window: int = 10) -> Tuple[float, float, float]:
    """
    Calculate enhanced Time-series aware Precision and Recall (eTaPR)
    
    Args:
        scores: Anomaly scores
        labels: True attack labels (0/1)
        window: Range overlap window size
        
    Returns:
        tuple: (precision, recall, f1)
    """
    # Convert scores to binary predictions using optimal threshold
    fpr, tpr, thresholds = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    predictions = (scores > threshold).astype(int)
    
    # Find ranges
    pred_ranges = _get_ranges(predictions)
    true_ranges = _get_ranges(labels)
    
    # Calculate overlaps
    tp = 0
    fp = 0
    fn = 0
    
    for pr in pred_ranges:
        matched = False
        for tr in true_ranges:
            if _ranges_overlap(pr, tr, window):
                tp += 1
                matched = True
                break
        if not matched:
            fp += 1
            
    for tr in true_ranges:
        matched = False
        for pr in pred_ranges:
            if _ranges_overlap(tr, pr, window):
                matched = True
                break
        if not matched:
            fn += 1
            
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def _get_ranges(binary_sequence: np.ndarray) -> List[Tuple[int, int]]:
    """Get start and end indices of continuous ranges of 1s"""
    ranges = []
    start = None
    
    for i, val in enumerate(binary_sequence):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            ranges.append((start, i-1))
            start = None
            
    if start is not None:
        ranges.append((start, len(binary_sequence)-1))
        
    return ranges

def _ranges_overlap(range1: Tuple[int, int],
                   range2: Tuple[int, int],
                   window: int) -> bool:
    """Check if two ranges overlap within window size"""
    return not (range1[1] + window < range2[0] or range2[1] + window < range1[0])

def evaluate_model(scores: np.ndarray,
                  labels: np.ndarray,
                  thresholds: np.ndarray = None) -> Dict[str, float]:
    """
    Comprehensive model evaluation
    
    Args:
        scores: Anomaly scores
        labels: True attack labels (0/1)
        thresholds: Array of thresholds to try
        
    Returns:
        dict: Dictionary containing various metrics
    """
    if thresholds is None:
        thresholds = np.linspace(min(scores), max(scores), 100)
        
    # Calculate PR curve
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate eTaPR
    etapr_p, etapr_r, etapr_f1 = calculate_etapr(scores, labels)
    
    # Calculate detection delay
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    delay = calculate_detection_delay(scores, labels, optimal_threshold)
    
    return {
        'PR_AUC': pr_auc,
        'ROC_AUC': roc_auc,
        'eTaPR_Precision': etapr_p,
        'eTaPR_Recall': etapr_r,
        'eTaPR_F1': etapr_f1,
        'Detection_Delay': delay,
        'Optimal_Threshold': optimal_threshold
    }

def evaluate_by_attack_type(scores: np.ndarray,
                          labels: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance for different attack types
    
    Args:
        scores: Anomaly scores
        labels: Dictionary mapping attack types to label arrays
        
    Returns:
        dict: Dictionary containing metrics for each attack type
    """
    results = {}
    
    for attack_type, attack_labels in labels.items():
        results[attack_type] = evaluate_model(scores, attack_labels)
        
    return results
