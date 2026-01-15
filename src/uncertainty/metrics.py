"""
Uncertainty Metrics for Calibration Assessment
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import roc_auc_score


def expected_calibration_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15
) -> Dict[str, float]:
    """Compute Expected Calibration Error (ECE)."""
    if predictions.dim() == 2:
        predictions = predictions.flatten()
        targets = targets.flatten()
    
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    mce = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracy = targets[mask].mean()
            bin_confidence = predictions[mask].mean()
            bin_count = mask.sum()
            
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
            
            ece += (bin_count / len(predictions)) * abs(bin_accuracy - bin_confidence)
            mce = max(mce, abs(bin_accuracy - bin_confidence))
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts
    }


def brier_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Brier Score."""
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    return float(np.mean((predictions - targets) ** 2))


def negative_log_likelihood(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-8
) -> float:
    """Compute Negative Log-Likelihood (NLL)."""
    predictions = predictions.clamp(eps, 1 - eps)
    nll = -(targets * torch.log(predictions) + 
            (1 - targets) * torch.log(1 - predictions))
    return float(nll.mean().cpu().numpy())


def uncertainty_coverage(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor,
    coverage_levels: List[float] = [0.5, 0.7, 0.9, 0.95]
) -> Dict[str, float]:
    """Compute uncertainty coverage metrics."""
    if predictions.dim() == 2:
        if uncertainties.dim() == 2:
            uncertainties = uncertainties.mean(dim=1)
        errors = (predictions - targets).abs().mean(dim=1)
    else:
        errors = (predictions - targets).abs()
    
    predictions = predictions.cpu().numpy()
    uncertainties = uncertainties.cpu().numpy()
    errors = errors.cpu().numpy()
    
    sorted_indices = np.argsort(uncertainties)
    sorted_errors = errors[sorted_indices]
    
    results = {}
    n = len(errors)
    
    for level in coverage_levels:
        k = int(n * level)
        confident_error = sorted_errors[:k].mean() if k > 0 else 0
        uncertain_error = sorted_errors[k:].mean() if k < n else 0
        
        results[f'error_at_{int(level*100)}pct'] = float(confident_error)
        results[f'error_above_{int(level*100)}pct'] = float(uncertain_error)
    
    if len(uncertainties) > 1:
        correlation = np.corrcoef(uncertainties, errors)[0, 1]
        results['uncertainty_error_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
    else:
        results['uncertainty_error_correlation'] = 0.0
    
    median_error = np.median(errors)
    high_error = (errors > median_error).astype(float)
    
    if len(np.unique(high_error)) > 1:
        results['uncertainty_auroc'] = float(roc_auc_score(high_error, uncertainties))
    else:
        results['uncertainty_auroc'] = 0.5
    
    return results


def compute_calibration_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    uncertainties: Optional[torch.Tensor] = None,
    n_bins: int = 15
) -> Dict[str, float]:
    """Compute comprehensive calibration metrics."""
    metrics = {}
    
    ece_results = expected_calibration_error(predictions, targets, n_bins)
    metrics['ece'] = ece_results['ece']
    metrics['mce'] = ece_results['mce']
    
    metrics['brier_score'] = brier_score(predictions, targets)
    metrics['nll'] = negative_log_likelihood(predictions, targets)
    
    if uncertainties is not None:
        coverage = uncertainty_coverage(predictions, uncertainties, targets)
        metrics.update(coverage)
    
    return metrics
