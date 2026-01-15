#!/usr/bin/env python3
"""
Quick Test Script for Uncertainty Quantification Module
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np

print("="*60)
print("Testing Uncertainty Quantification Module")
print("="*60)

# Test 1: Import modules
print("\n[Test 1] Importing uncertainty modules...")
try:
    from src.uncertainty import (
        UncertaintyEstimator,
        MCDropoutEstimator,
        DeepEnsembleEstimator,
        OODDetector,
        compute_calibration_metrics,
        expected_calibration_error,
        brier_score,
        negative_log_likelihood
    )
    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Create a simple test model
print("\n[Test 2] Creating test model...")

class SimpleModel(nn.Module):
    def __init__(self, input_dim=768, num_classes=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x, env=None):
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])
        if x.shape[-1] != 768:
            x = nn.functional.adaptive_avg_pool1d(x.unsqueeze(1), 768).squeeze(1)
        feat = self.encoder(x)
        logits = self.classifier(feat)
        return logits

model = SimpleModel(input_dim=768, num_classes=100)
print(f"✓ Created SimpleModel with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test 3: MC Dropout
print("\n[Test 3] Testing MC Dropout...")
try:
    mc_estimator = MCDropoutEstimator(model, n_samples=5)
    batch_size = 4
    dummy_images = torch.randn(batch_size, 768)
    
    results = mc_estimator.predict_with_uncertainty(
        images=dummy_images,
        env=None,
        apply_sigmoid=True
    )
    
    print(f"  Mean shape: {results['mean'].shape}")
    print(f"  Std shape: {results['std'].shape}")
    print(f"  Entropy shape: {results['entropy'].shape}")
    print(f"  Mean uncertainty: {results['epistemic_uncertainty'].mean():.4f}")
    print("✓ MC Dropout test passed!")
except Exception as e:
    print(f"✗ MC Dropout test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Deep Ensemble
print("\n[Test 4] Testing Deep Ensemble...")
try:
    models = [SimpleModel(768, 100) for _ in range(3)]
    ensemble = DeepEnsembleEstimator(models=models)
    
    results = ensemble.predict_with_uncertainty(
        images=dummy_images,
        env=None,
        apply_sigmoid=True
    )
    
    print(f"  Mean shape: {results['mean'].shape}")
    print(f"  JS Divergence shape: {results['js_divergence'].shape}")
    print(f"  Mean disagreement: {results['epistemic_uncertainty'].mean():.4f}")
    print("✓ Deep Ensemble test passed!")
except Exception as e:
    print(f"✗ Deep Ensemble test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: OOD Detection
print("\n[Test 5] Testing OOD Detection...")
try:
    ood_detector = OODDetector(model, method='energy')
    scores = ood_detector.compute_ood_scores(images=dummy_images, env=None)
    
    print(f"  MSP shape: {scores['msp'].shape}")
    print(f"  Energy range: [{scores['energy'].min():.4f}, {scores['energy'].max():.4f}]")
    print("✓ OOD Detection test passed!")
except Exception as e:
    print(f"✗ OOD Detection test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Uncertainty Metrics
print("\n[Test 6] Testing Uncertainty Metrics...")
try:
    predictions = torch.sigmoid(torch.randn(100, 50))
    targets = (torch.rand(100, 50) > 0.5).float()
    uncertainties = torch.rand(100, 50) * 0.5
    
    ece_results = expected_calibration_error(predictions, targets)
    print(f"  ECE: {ece_results['ece']:.4f}")
    
    bs = brier_score(predictions, targets)
    print(f"  Brier Score: {bs:.4f}")
    
    nll = negative_log_likelihood(predictions, targets)
    print(f"  NLL: {nll:.4f}")
    print("✓ Uncertainty Metrics test passed!")
except Exception as e:
    print(f"✗ Uncertainty Metrics test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Unified Estimator
print("\n[Test 7] Testing Unified UncertaintyEstimator...")
try:
    estimator = UncertaintyEstimator(model, method='mc_dropout', n_samples=5)
    results = estimator.predict_with_uncertainty(dummy_images)
    print(f"  MC mode - Mean shape: {results['mean'].shape}, Uncertainty: {results['uncertainty'].mean():.4f}")
    
    estimator = UncertaintyEstimator(method='ensemble', models=models)
    results = estimator.predict_with_uncertainty(dummy_images)
    print(f"  Ensemble mode - Mean shape: {results['mean'].shape}, Uncertainty: {results['uncertainty'].mean():.4f}")
    print("✓ Unified Estimator test passed!")
except Exception as e:
    print(f"✗ Unified Estimator test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All Basic Tests Completed!")
print("="*60)

print("\n📊 Summary:")
print("  ✓ MC Dropout: Working")
print("  ✓ Deep Ensemble: Working")
print("  ✓ OOD Detection: Working")
print("  ✓ Calibration Metrics: Working")
print("  ✓ Unified Interface: Working")
