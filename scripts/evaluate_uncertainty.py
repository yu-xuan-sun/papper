#!/usr/bin/env python3
"""
Uncertainty Evaluation Script for SatBird Models
Supports MC Dropout, Deep Ensemble, and OOD Detection
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

# Import SatBird components
from src.trainer.trainer import EbirdTask, EbirdDataModule
from src.utils.config_utils import load_opts

# Import uncertainty modules
from src.uncertainty import (
    UncertaintyEstimator,
    MCDropoutEstimator,
    DeepEnsembleEstimator,
    OODDetector,
    compute_calibration_metrics,
    expected_calibration_error,
    brier_score,
    negative_log_likelihood,
    uncertainty_coverage
)


def load_config(config_path, base_dir):
    """Load YAML config file."""
    default_config = os.path.join(base_dir, "configs/defaults.yaml")
    opts = load_opts(config_path, default=default_config, commandline_opts={})
    opts.base_dir = base_dir
    return opts


def evaluate_uncertainty(model, dataloader, method='mc_dropout', n_samples=20, device='cuda'):
    """Run uncertainty evaluation on the dataset."""
    
    print(f"\n{'='*60}")
    print(f"Evaluating with {method} method")
    print(f"{'='*60}")
    
    # Get the underlying model (not the PL wrapper)
    inner_model = model.model
    
    # Initialize estimator
    if method == 'mc_dropout':
        estimator = UncertaintyEstimator(
            inner_model, 
            method='mc_dropout',
            n_samples=n_samples
        )
    elif method == 'ood':
        estimator = UncertaintyEstimator(
            inner_model,
            method='ood'
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Collect predictions and uncertainties
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    all_epistemic = []
    all_aleatoric = []
    
    inner_model.to(device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating ({method})"):
            images = batch['sat'].to(device)
            targets = batch['target'].to(device)
            env = batch.get('env')
            if env is not None:
                env = env.to(device)
            
            # Get predictions with uncertainty
            results = estimator.predict_with_uncertainty(
                images=images,
                env=env,
                apply_sigmoid=True
            )
            
            all_predictions.append(results['mean'].cpu())
            all_targets.append(targets.cpu())
            all_uncertainties.append(results['uncertainty'].cpu())
            
            if 'epistemic_uncertainty' in results:
                all_epistemic.append(results['epistemic_uncertainty'].cpu())
            if 'aleatoric_uncertainty' in results:
                all_aleatoric.append(results['aleatoric_uncertainty'].cpu())
    
    # Concatenate all results
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    uncertainties = torch.cat(all_uncertainties, dim=0)
    
    if all_epistemic:
        epistemic = torch.cat(all_epistemic, dim=0)
    else:
        epistemic = None
    
    if all_aleatoric:
        aleatoric = torch.cat(all_aleatoric, dim=0)
    else:
        aleatoric = None
    
    return {
        'predictions': predictions,
        'targets': targets,
        'uncertainties': uncertainties,
        'epistemic': epistemic,
        'aleatoric': aleatoric
    }


def compute_metrics(results):
    """Compute all uncertainty and calibration metrics."""
    predictions = results['predictions']
    targets = results['targets']
    uncertainties = results['uncertainties']
    
    metrics = {}
    
    # Calibration metrics
    print("\nComputing calibration metrics...")
    ece_results = expected_calibration_error(predictions, targets)
    metrics['ece'] = ece_results['ece']
    metrics['mce'] = ece_results['mce']
    
    metrics['brier_score'] = brier_score(predictions, targets)
    metrics['nll'] = negative_log_likelihood(predictions, targets)
    
    # Coverage metrics
    print("Computing uncertainty coverage...")
    coverage_results = uncertainty_coverage(predictions, targets, uncertainties)
    metrics.update(coverage_results)
    
    # Additional metrics
    metrics['mean_uncertainty'] = uncertainties.mean().item()
    metrics['std_uncertainty'] = uncertainties.std().item()
    
    if results['epistemic'] is not None:
        metrics['mean_epistemic'] = results['epistemic'].mean().item()
    if results['aleatoric'] is not None:
        metrics['mean_aleatoric'] = results['aleatoric'].mean().item()
    
    # Performance metrics
    print("Computing performance metrics...")
    # Top-k accuracy
    for k in [10, 20, 30]:
        _, pred_indices = predictions.topk(k, dim=1)
        correct = targets.gather(1, pred_indices).sum(dim=1)
        accuracy = (correct > 0).float().mean().item()
        metrics[f'top{k}_accuracy'] = accuracy
    
    # mAP
    from sklearn.metrics import average_precision_score
    try:
        mAP = average_precision_score(targets.numpy(), predictions.numpy(), average='macro')
        metrics['mAP'] = mAP
    except:
        metrics['mAP'] = 0.0
    
    return metrics


def analyze_uncertainty_quality(results, metrics, output_dir):
    """Analyze the quality of uncertainty estimates."""
    predictions = results['predictions']
    targets = results['targets']
    uncertainties = results['uncertainties']
    
    analysis = {}
    
    # 1. Uncertainty-Error Correlation
    errors = torch.abs(predictions - targets).mean(dim=1)
    from scipy import stats
    correlation, p_value = stats.spearmanr(uncertainties.numpy(), errors.numpy())
    analysis['uncertainty_error_correlation'] = correlation
    analysis['correlation_p_value'] = p_value
    
    # 2. Selective Prediction Analysis
    print("\nSelective Prediction Analysis:")
    print("-" * 40)
    
    # Sort by uncertainty
    sorted_indices = torch.argsort(uncertainties)
    
    for keep_ratio in [0.5, 0.7, 0.9]:
        n_keep = int(len(sorted_indices) * keep_ratio)
        selected_indices = sorted_indices[:n_keep]
        
        selected_preds = predictions[selected_indices]
        selected_targets = targets[selected_indices]
        
        # Top-30 accuracy on selected samples
        _, pred_indices = selected_preds.topk(30, dim=1)
        correct = selected_targets.gather(1, pred_indices).sum(dim=1)
        accuracy = (correct > 0).float().mean().item()
        
        analysis[f'selective_top30_acc_{int(keep_ratio*100)}pct'] = accuracy
        print(f"  Keep {int(keep_ratio*100)}% (lowest uncertainty): Top-30 = {accuracy:.4f}")
    
    # 3. High vs Low Uncertainty Analysis
    median_uncertainty = uncertainties.median()
    low_uncertainty_mask = uncertainties < median_uncertainty
    high_uncertainty_mask = uncertainties >= median_uncertainty
    
    low_preds = predictions[low_uncertainty_mask]
    low_targets = targets[low_uncertainty_mask]
    high_preds = predictions[high_uncertainty_mask]
    high_targets = targets[high_uncertainty_mask]
    
    # Accuracy for low vs high uncertainty
    _, low_indices = low_preds.topk(30, dim=1)
    low_correct = low_targets.gather(1, low_indices).sum(dim=1)
    low_acc = (low_correct > 0).float().mean().item()
    
    _, high_indices = high_preds.topk(30, dim=1)
    high_correct = high_targets.gather(1, high_indices).sum(dim=1)
    high_acc = (high_correct > 0).float().mean().item()
    
    analysis['low_uncertainty_top30_acc'] = low_acc
    analysis['high_uncertainty_top30_acc'] = high_acc
    analysis['uncertainty_separation'] = low_acc - high_acc
    
    print(f"\n  Low uncertainty samples: Top-30 = {low_acc:.4f}")
    print(f"  High uncertainty samples: Top-30 = {high_acc:.4f}")
    print(f"  Separation: {low_acc - high_acc:.4f}")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description='Evaluate Uncertainty Quantification')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--method', type=str, default='mc_dropout',
                       choices=['mc_dropout', 'ood'],
                       help='Uncertainty method')
    parser.add_argument('--n_samples', type=int, default=20,
                       help='Number of MC samples (for MC Dropout)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--output_dir', type=str, default='outputs/uncertainty_analysis',
                       help='Output directory for results')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to evaluate')
    
    args = parser.parse_args()
    
    # Base directory
    base_dir = str(project_root)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Uncertainty Quantification Evaluation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Method: {args.method}")
    print(f"Device: {args.device}")
    
    # Load config
    config_path = os.path.join(base_dir, args.config)
    opts = load_config(config_path, base_dir)
    
    # Override batch size
    opts.data.loaders.batch_size = args.batch_size
    
    # Load model
    print("\nLoading model from checkpoint...")
    checkpoint_path = os.path.join(base_dir, args.checkpoint)
    model = EbirdTask.load_from_checkpoint(
        checkpoint_path,
        opts=opts,
        strict=False
    )
    model.to(args.device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model type: {type(model.model).__name__}")
    
    # Setup data module
    print("\nSetting up data loader...")
    dm = EbirdDataModule(opts)
    dm.setup(stage='test')
    
    if args.split == 'test':
        dataloader = dm.test_dataloader()
    elif args.split == 'val':
        dataloader = dm.val_dataloader()
    else:
        dataloader = dm.train_dataloader()
    
    print(f"Data loader ready: {len(dataloader)} batches")
    
    # Run evaluation
    results = evaluate_uncertainty(
        model=model,
        dataloader=dataloader,
        method=args.method,
        n_samples=args.n_samples,
        device=args.device
    )
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Analyze uncertainty quality
    analysis = analyze_uncertainty_quality(results, metrics, output_dir)
    
    # Combine all results
    all_results = {
        'config': {
            'checkpoint': args.checkpoint,
            'config_file': args.config,
            'method': args.method,
            'n_samples': args.n_samples,
            'split': args.split
        },
        'metrics': metrics,
        'analysis': analysis
    }
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    print("\n📊 Performance Metrics:")
    for k in [10, 20, 30]:
        print(f"  Top-{k} Accuracy: {metrics[f'top{k}_accuracy']:.4f}")
    print(f"  mAP: {metrics['mAP']:.4f}")
    
    print("\n📈 Calibration Metrics:")
    print(f"  ECE: {metrics['ece']:.4f}")
    print(f"  MCE: {metrics['mce']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  NLL: {metrics['nll']:.4f}")
    
    print("\n🎯 Uncertainty Quality:")
    print(f"  Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")
    print(f"  Uncertainty-Error Correlation: {analysis['uncertainty_error_correlation']:.4f}")
    print(f"  Uncertainty Separation: {analysis['uncertainty_separation']:.4f}")
    
    print("\n🔍 Selective Prediction:")
    print(f"  Keep 50% (lowest uncertainty): {analysis['selective_top30_acc_50pct']:.4f}")
    print(f"  Keep 70% (lowest uncertainty): {analysis['selective_top30_acc_70pct']:.4f}")
    print(f"  Keep 90% (lowest uncertainty): {analysis['selective_top30_acc_90pct']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f'uncertainty_results_{args.method}_{timestamp}.json'
    
    # Convert numpy types for JSON
    def convert_to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    return all_results


if __name__ == '__main__':
    main()
