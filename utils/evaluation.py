import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

class Evaluator:
    """Evaluation metrics for HAI Security Dataset"""
    
    @staticmethod
    def calculate_basic_metrics(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate basic classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary containing metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_prob is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob, average='weighted')
            
        return metrics
    
    @staticmethod
    def calculate_confusion_matrix(y_true: np.ndarray,
                                 y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def calculate_etapr(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       theta_p: float = 0.5,
                       theta_r: float = 0.5) -> Dict[str, float]:
        """
        Calculate enhanced Time-series Aware Precision and Recall (eTaPR)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            theta_p: Precision threshold
            theta_r: Recall threshold
            
        Returns:
            Dictionary containing eTaPR metrics
        """
        def find_ranges(y: np.ndarray) -> List[Tuple[int, int]]:
            """Find continuous ranges of 1s in binary array"""
            ranges = []
            start = None
            
            for i, val in enumerate(y):
                if val == 1 and start is None:
                    start = i
                elif val == 0 and start is not None:
                    ranges.append((start, i-1))
                    start = None
                    
            if start is not None:
                ranges.append((start, len(y)-1))
                
            return ranges
        
        # Find ranges for true and predicted anomalies
        true_ranges = find_ranges(y_true)
        pred_ranges = find_ranges(y_pred)
        
        # Calculate overlaps
        overlaps = []
        for tr_start, tr_end in true_ranges:
            for pr_start, pr_end in pred_ranges:
                # Check if ranges overlap
                if not (pr_end < tr_start or pr_start > tr_end):
                    overlap_start = max(tr_start, pr_start)
                    overlap_end = min(tr_end, pr_end)
                    overlap_length = overlap_end - overlap_start + 1
                    true_length = tr_end - tr_start + 1
                    pred_length = pr_end - pr_start + 1
                    
                    overlaps.append({
                        'overlap': overlap_length,
                        'true_length': true_length,
                        'pred_length': pred_length
                    })
        
        # Calculate eTaPR metrics
        if not overlaps:
            return {'etapr_precision': 0.0, 'etapr_recall': 0.0, 'etapr_f1': 0.0}
        
        # Calculate precision
        precision_scores = [o['overlap'] / o['pred_length'] for o in overlaps]
        precision = sum(s >= theta_p for s in precision_scores) / len(pred_ranges) if pred_ranges else 0
        
        # Calculate recall
        recall_scores = [o['overlap'] / o['true_length'] for o in overlaps]
        recall = sum(s >= theta_r for s in recall_scores) / len(true_ranges) if true_ranges else 0
        
        # Calculate F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'etapr_precision': precision,
            'etapr_recall': recall,
            'etapr_f1': f1
        }
    
    @staticmethod
    def calculate_detection_delay(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                timestamps: np.ndarray) -> Dict[str, float]:
        """
        Calculate attack detection delay
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            timestamps: Array of timestamps
            
        Returns:
            Dictionary containing delay metrics
        """
        def find_attack_starts(y: np.ndarray) -> List[int]:
            """Find indices where attacks start"""
            return np.where(np.diff(np.concatenate(([0], y))) == 1)[0]
        
        true_starts = find_attack_starts(y_true)
        pred_starts = find_attack_starts(y_pred)
        
        delays = []
        for ts in true_starts:
            # Find first detection after attack start
            detections = pred_starts[pred_starts >= ts]
            if len(detections) > 0:
                delay = timestamps[detections[0]] - timestamps[ts]
                delays.append(delay)
        
        if not delays:
            return {'mean_delay': np.inf, 'median_delay': np.inf}
            
        return {
            'mean_delay': np.mean(delays),
            'median_delay': np.median(delays)
        }
    
    @staticmethod
    def calculate_false_alarm_rate(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 window_size: int = 100) -> float:
        """
        Calculate false alarm rate using sliding window
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            window_size: Size of sliding window
            
        Returns:
            False alarm rate
        """
        false_alarms = 0
        n_windows = len(y_true) - window_size + 1
        
        for i in range(n_windows):
            window_true = y_true[i:i+window_size]
            window_pred = y_pred[i:i+window_size]
            
            # Count as false alarm if prediction is 1 but true is 0
            if np.any(window_pred > window_true):
                false_alarms += 1
                
        return false_alarms / n_windows if n_windows > 0 else 0.0
