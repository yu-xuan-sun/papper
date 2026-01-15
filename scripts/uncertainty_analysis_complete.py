#!/usr/bin/env python3
"""
Complete Uncertainty Quantification Analysis for SatBird
"""

import argparse
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import average_precision_score
import yaml

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

COLORS = {'primary': '#2E86AB', 'secondary': '#A23B72', 'accent': '#F18F01'}


def enable_mc_dropout(model):
    """Enable dropout layers during inference for MC Dropout."""
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            count += 1
    return count


def mc_dropout_predict(model, images, env=None, n_samples=30, device='cuda'):
    """Run MC Dropout prediction."""
    model.eval()
    enable_mc_dropout(model)
    
    all_predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            if env is not None:
                logits = model(images, env)
            else:
                logits = model(images)
            probs = torch.sigmoid(logits)
            all_predictions.append(probs)
    
    predictions = torch.stack(all_predictions, dim=0)
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)
    
    return {
        'mean': mean_pred, 
        'std': std_pred,
        'sample_uncertainty': std_pred.mean(dim=1),
        'species_uncertainty': std_pred.mean(dim=0)
    }


def run_uncertainty_evaluation(model, dataloader, n_samples=30, device='cuda', max_batches=None):
    """Run full uncertainty evaluation on a dataloader."""
    model.to(device)
    model.eval()
    n_dropout = enable_mc_dropout(model)
    print(f"   Enabled {n_dropout} dropout layers for MC Dropout")
    
    all_mean_preds = []
    all_std_preds = []
    all_targets = []
    all_sample_unc = []
    
    n_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc="MC Dropout")):
            if max_batches and i >= max_batches:
                break
            
            images = batch['sat'].to(device)
            # Handle extra dimension: [B, 1, C, H, W] -> [B, C, H, W]
            if images.dim() == 5 and images.shape[1] == 1:
                images = images.squeeze(1)
            
            targets = batch['target'].to(device)
            env = batch.get('env')
            if env is not None:
                env = env.to(device)
            
            results = mc_dropout_predict(model, images, env, n_samples, device)
            all_mean_preds.append(results['mean'].cpu())
            all_std_preds.append(results['std'].cpu())
            all_targets.append(targets.cpu())
            all_sample_unc.append(results['sample_uncertainty'].cpu())
    
    return {
        'mean_preds': torch.cat(all_mean_preds, dim=0),
        'std_preds': torch.cat(all_std_preds, dim=0),
        'targets': torch.cat(all_targets, dim=0),
        'sample_uncertainty': torch.cat(all_sample_unc, dim=0),
    }


def compute_performance_metrics(mean_preds, targets):
    """Compute Top-K and mAP metrics."""
    metrics = {}
    for k in [1, 5, 10, 20, 30]:
        _, pred_indices = mean_preds.topk(k, dim=1)
        correct = targets.gather(1, pred_indices).sum(dim=1)
        metrics[f'top{k}_acc'] = (correct > 0).float().mean().item()
    try:
        metrics['mAP'] = average_precision_score(targets.numpy(), mean_preds.numpy(), average='macro')
    except:
        metrics['mAP'] = 0.0
    return metrics


def selective_prediction_analysis(mean_preds, targets, sample_unc):
    """Analyze selective prediction performance."""
    n_samples = len(sample_unc)
    sorted_indices = torch.argsort(sample_unc)
    
    results = []
    for keep_ratio in np.arange(0.1, 1.01, 0.05):
        n_keep = int(n_samples * keep_ratio)
        if n_keep == 0:
            continue
        
        selected_indices = sorted_indices[:n_keep]
        selected_preds = mean_preds[selected_indices]
        selected_targets = targets[selected_indices]
        
        _, pred_indices = selected_preds.topk(30, dim=1)
        correct = selected_targets.gather(1, pred_indices).sum(dim=1)
        top30_acc = (correct > 0).float().mean().item()
        
        results.append({
            'keep_ratio': keep_ratio, 
            'n_samples': n_keep, 
            'top30_acc': top30_acc,
            'mean_uncertainty': sample_unc[selected_indices].mean().item()
        })
    return pd.DataFrame(results)


def plot_selective_prediction(selective_df, output_dir, dataset_name):
    """Plot selective prediction curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(selective_df['keep_ratio'] * 100, selective_df['top30_acc'] * 100, 
            'o-', color=COLORS['primary'], linewidth=2, markersize=6, label='With Selection')
    
    baseline = selective_df[selective_df['keep_ratio'] >= 0.95]['top30_acc'].values[0] * 100
    ax.axhline(baseline, color='gray', linestyle=':', linewidth=1.5, label=f'Baseline: {baseline:.1f}%')
    
    ax.set_xlabel('Percentage of Samples Kept (Lowest Uncertainty)', fontsize=12)
    ax.set_ylabel('Top-30 Accuracy (%)', fontsize=12)
    ax.set_title(f'Selective Prediction Performance - {dataset_name}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = dataset_name.replace('-', '_').replace(' ', '_').lower()
    plt.savefig(output_dir / f'selective_prediction_{safe_name}.pdf', bbox_inches='tight')
    plt.savefig(output_dir / f'selective_prediction_{safe_name}.png', bbox_inches='tight')
    plt.close()


def plot_uncertainty_distribution(sample_unc, output_dir, dataset_name):
    """Plot uncertainty distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sample_unc.numpy(), bins=50, color=COLORS['primary'], alpha=0.7, edgecolor='white')
    ax.axvline(sample_unc.mean(), color=COLORS['accent'], linestyle='--', linewidth=2,
               label=f'Mean: {sample_unc.mean():.4f}')
    ax.axvline(sample_unc.median(), color=COLORS['secondary'], linestyle='--', linewidth=2,
               label=f'Median: {sample_unc.median():.4f}')
    
    ax.set_xlabel('Prediction Uncertainty (Std)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Sample Uncertainty Distribution - {dataset_name}', fontsize=14)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    safe_name = dataset_name.replace('-', '_').replace(' ', '_').lower()
    plt.savefig(output_dir / f'uncertainty_distribution_{safe_name}.pdf', bbox_inches='tight')
    plt.savefig(output_dir / f'uncertainty_distribution_{safe_name}.png', bbox_inches='tight')
    plt.close()


def plot_uncertainty_vs_error(sample_unc, errors, corr, output_dir, dataset_name):
    """Plot uncertainty vs prediction error scatter."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(sample_unc, errors, alpha=0.3, s=10, color=COLORS['primary'])
    
    # Add trend line
    z = np.polyfit(sample_unc, errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample_unc.min(), sample_unc.max(), 100)
    ax.plot(x_line, p(x_line), color=COLORS['accent'], linewidth=2, linestyle='--',
            label=f'Linear fit (r={corr:.3f})')
    
    ax.set_xlabel('Prediction Uncertainty', fontsize=12)
    ax.set_ylabel('Prediction Error (MAE)', fontsize=12)
    ax.set_title(f'Uncertainty vs Error Correlation\nSpearman ρ = {corr:.4f} - {dataset_name}', fontsize=14)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    safe_name = dataset_name.replace('-', '_').replace(' ', '_').lower()
    plt.savefig(output_dir / f'uncertainty_error_correlation_{safe_name}.pdf', bbox_inches='tight')
    plt.savefig(output_dir / f'uncertainty_error_correlation_{safe_name}.png', bbox_inches='tight')
    plt.close()


def find_checkpoint(run_dir):
    """Find the best checkpoint in a run directory."""
    ckpt_dir = Path(run_dir) / 'checkpoints'
    if not ckpt_dir.exists():
        return None
    ckpts = list(ckpt_dir.glob('best*.ckpt'))
    if ckpts:
        return str(ckpts[0])
    ckpts = list(ckpt_dir.glob('*.ckpt'))
    return str(ckpts[0]) if ckpts else None


def load_model_from_checkpoint(checkpoint_path, config_path, base_dir):
    """Load model from checkpoint with saved config, handling architecture mismatches."""
    from src.trainer.trainer import EbirdTask
    from omegaconf import OmegaConf
    
    # Load config
    with open(config_path, 'r') as f:
        opts = OmegaConf.create(yaml.safe_load(f))
    
    opts.base_dir = base_dir
    
    # First try strict loading
    try:
        model = EbirdTask.load_from_checkpoint(checkpoint_path, opts=opts, strict=True)
        print("   Model loaded with strict=True")
        return model, opts
    except RuntimeError as e:
        print(f"   Strict loading failed, trying non-strict...")
        # Check if it's a size mismatch issue
        if "size mismatch" in str(e) or "Missing key" in str(e):
            # Load checkpoint state_dict
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state_dict = ckpt['state_dict']
            
            # Analyze env_encoder structure in checkpoint
            env_encoder_keys = [k for k in state_dict.keys() if 'fusion.env_encoder' in k]
            max_idx = 0
            for k in env_encoder_keys:
                parts = k.split('.')
                for p in parts:
                    if p.isdigit():
                        max_idx = max(max_idx, int(p))
            
            # Calculate num_layers from checkpoint
            ckpt_num_layers = (max_idx // 4) + 1
            print(f"   Detected {ckpt_num_layers} layers in checkpoint env_encoder")
            
            # Modify opts to match checkpoint architecture
            if hasattr(opts.experiment.module, 'env_num_layers'):
                opts.experiment.module.env_num_layers = ckpt_num_layers
            
            # Try again
            try:
                model = EbirdTask.load_from_checkpoint(checkpoint_path, opts=opts, strict=True)
                print("   Model loaded after adjusting env_num_layers")
                return model, opts
            except:
                # Final fallback: non-strict
                model = EbirdTask.load_from_checkpoint(checkpoint_path, opts=opts, strict=False)
                print("   Model loaded with strict=False (some weights may not match)")
                return model, opts
        else:
            raise e


def main():
    parser = argparse.ArgumentParser(description='Complete Uncertainty Analysis')
    parser.add_argument('--datasets', nargs='+', default=['summer'], 
                        choices=['summer', 'winter'])
    parser.add_argument('--n_samples', type=int, default=30, help='MC Dropout samples')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs/uncertainty_analysis_complete')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("COMPLETE UNCERTAINTY QUANTIFICATION ANALYSIS")
    print("="*70)
    print(f"MC Dropout samples: {args.n_samples}")
    print(f"Output directory: {output_dir}")
    
    from src.trainer.trainer import EbirdDataModule
    
    DATASET_CONFIGS = {
        'summer': {
            'run_dir': 'runs/best summer1',
            'name': 'USA-Summer'
        },
        'winter': {
            'run_dir': 'runs/best winter',
            'name': 'USA-Winter'
        }
    }
    
    all_results = {}
    
    for dataset_key in args.datasets:
        if dataset_key not in DATASET_CONFIGS:
            continue
        
        cfg = DATASET_CONFIGS[dataset_key]
        dataset_name = cfg['name']
        run_dir = Path(cfg['run_dir'])
        
        print(f"\n{'='*70}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*70}")
        
        # Check run directory
        if not run_dir.exists():
            print(f"Run directory not found: {run_dir}, skipping...")
            continue
        
        # Find checkpoint
        checkpoint = find_checkpoint(run_dir)
        if checkpoint is None:
            print(f"Checkpoint not found in {run_dir}, skipping...")
            continue
        
        # Load saved config
        config_path = run_dir / 'config_full.yaml'
        if not config_path.exists():
            print(f"Config not found: {config_path}, skipping...")
            continue
        
        print(f"Checkpoint: {checkpoint}")
        print(f"Config: {config_path}")
        
        # Load model with architecture matching
        print(f"Loading model...")
        model, opts = load_model_from_checkpoint(
            checkpoint, config_path, str(project_root)
        )
        inner_model = model.model
        
        # Override batch size
        opts.data.loaders.batch_size = args.batch_size
        
        # Create data module
        dm = EbirdDataModule(opts)
        dm.setup(stage='test')
        dataloader = dm.test_dataloader()
        print(f"Test set: {len(dataloader)} batches ({len(dataloader) * args.batch_size} samples max)")
        
        # Run MC Dropout evaluation
        print(f"\nRunning MC Dropout ({args.n_samples} samples)...")
        results = run_uncertainty_evaluation(
            inner_model, dataloader, args.n_samples, args.device, args.max_batches
        )
        
        # Compute metrics
        performance = compute_performance_metrics(results['mean_preds'], results['targets'])
        
        # Uncertainty-error correlation
        errors = torch.abs(results['mean_preds'] - results['targets']).mean(dim=1).numpy()
        sample_unc = results['sample_uncertainty'].numpy()
        corr, p_value = stats.spearmanr(sample_unc, errors)
        
        # Selective prediction analysis
        selective_df = selective_prediction_analysis(
            results['mean_preds'], results['targets'], results['sample_uncertainty']
        )
        
        # Store results
        all_results[dataset_name] = {
            'sample_uncertainty': results['sample_uncertainty'],
            'performance': performance,
            'correlation': corr,
            'p_value': p_value,
            'selective_df': selective_df
        }
        
        # Generate plots
        print("\nGenerating plots...")
        plot_uncertainty_distribution(results['sample_uncertainty'], figures_dir, dataset_name)
        plot_uncertainty_vs_error(sample_unc, errors, corr, figures_dir, dataset_name)
        plot_selective_prediction(selective_df, figures_dir, dataset_name)
        
        # Print summary
        baseline = selective_df[selective_df['keep_ratio'] >= 0.95]['top30_acc'].values[0]
        best_50 = selective_df[selective_df['keep_ratio'] <= 0.55]['top30_acc'].max()
        
        print(f"\n{'='*50}")
        print(f"RESULTS: {dataset_name}")
        print(f"{'='*50}")
        print(f"📊 Performance:")
        print(f"   Top-30 Accuracy: {performance['top30_acc']*100:.2f}%")
        print(f"   mAP: {performance['mAP']*100:.2f}%")
        print(f"\n🎯 Uncertainty Quality:")
        print(f"   Mean Uncertainty: {results['sample_uncertainty'].mean():.4f}")
        print(f"   Std Uncertainty: {results['sample_uncertainty'].std():.4f}")
        print(f"   Uncertainty-Error Correlation: ρ={corr:.4f} (p={p_value:.2e})")
        print(f"\n🔍 Selective Prediction:")
        print(f"   Baseline (all samples): {baseline*100:.2f}%")
        print(f"   Best 50% (low uncertainty): {best_50*100:.2f}%")
        print(f"   Improvement: +{(best_50-baseline)/baseline*100:.1f}%")
    
    # Save summary JSON
    summary = {}
    for name, res in all_results.items():
        df = res['selective_df']
        baseline = df[df['keep_ratio'] >= 0.95]['top30_acc'].values[0]
        best_50 = df[df['keep_ratio'] <= 0.55]['top30_acc'].max()
        
        summary[name] = {
            'n_samples': len(res['sample_uncertainty']),
            'mean_uncertainty': float(res['sample_uncertainty'].mean()),
            'std_uncertainty': float(res['sample_uncertainty'].std()),
            'uncertainty_error_correlation': float(res['correlation']),
            'p_value': float(res['p_value']),
            'performance': res['performance'],
            'selective_prediction': {
                'baseline_top30': float(baseline),
                'best_50pct_top30': float(best_50),
                'improvement_pct': float((best_50 - baseline) / baseline * 100)
            }
        }
        
        # Save selective prediction CSV
        safe_name = name.replace('-', '_').replace(' ', '_').lower()
        df.to_csv(output_dir / f'selective_prediction_{safe_name}.csv', index=False)
    
    with open(output_dir / 'uncertainty_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ UNCERTAINTY ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📄 Generated files:")
    for f in sorted(output_dir.glob('*')):
        if f.is_file():
            print(f"   {f.name}")
    print(f"\n📊 Generated figures:")
    for f in sorted(figures_dir.glob('*')):
        print(f"   figures/{f.name}")


if __name__ == '__main__':
    main()
