"""
Uncertainty Quantification Module for SatBird
==============================================

Supported Methods:
    1. MC Dropout: Monte Carlo Dropout for epistemic uncertainty
    2. Deep Ensemble: Multi-model ensemble
    3. OOD Detection: Out-of-distribution detection
"""

from .mc_dropout import MCDropoutEstimator, enable_mc_dropout
from .deep_ensemble import DeepEnsembleEstimator
from .ood_detection import OODDetector
from .uncertainty_estimator import UncertaintyEstimator
from .metrics import (
    compute_calibration_metrics,
    expected_calibration_error,
    brier_score,
    negative_log_likelihood,
    uncertainty_coverage
)

__all__ = [
    'UncertaintyEstimator',
    'MCDropoutEstimator',
    'DeepEnsembleEstimator',
    'OODDetector',
    'enable_mc_dropout',
    'compute_calibration_metrics',
    'expected_calibration_error',
    'brier_score',
    'negative_log_likelihood',
    'uncertainty_coverage'
]
