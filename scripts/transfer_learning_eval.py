#!/usr/bin/env python3
"""
Transfer Learning Evaluation and Comparison Script

This script evaluates and compares transfer learning experiments across:
1. Season Transfer: USA-Summer -> USA-Winter
2. Geographic Transfer: USA -> Kenya
3. Species Transfer: Bird -> Butterfly
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def parse_args():
    parser = argparse.ArgumentParser(description='Transfer Learning Evaluation')
    parser.add_argument('--experiment_type', type=str, default='all',
                       choices=['season', 'geo', 'species', 'all'],
                       help='Type of transfer experiment to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used')
    parser.add_argument('--output_dir', type=str, default='results/transfer_learning',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--runs_dir', type=str, default='runs', help='Directory containing run outputs')
    return parser.parse_args()


def load_metrics(run_dir: str) -> Optional[Dict]:
    """Load metrics from a run directory."""
    metrics_file = os.path.join(run_dir, 'metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    # Try to load from tensorboard events
    event_files = glob.glob(os.path.join(run_dir, 'events.out.tfevents.*'))
    if event_files:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(run_dir)
            ea.Reload()
            metrics = {}
            for tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                if events:
                    metrics[tag] = events[-1].value  # Get last value
            return metrics
        except ImportError:
            print("Warning: tensorboard not installed, cannot read event files")
    
    # Try to load from CSV
    csv_file = os.path.join(run_dir, 'metrics.csv')
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if len(df) > 0:
            return df.iloc[-1].to_dict()
    
    return None


def find_best_checkpoint(run_dir: str) -> Optional[str]:
    """Find the best checkpoint in a run directory."""
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    if os.path.exists(ckpt_dir):
        # Look for best checkpoint
        best_ckpts = glob.glob(os.path.join(ckpt_dir, '*best*.ckpt'))
        if best_ckpts:
            return best_ckpts[0]
        
        # Look for last checkpoint
        last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
        if os.path.exists(last_ckpt):
            return last_ckpt
    
    return None


def collect_experiment_results(runs_dir: str, seed: int, experiment_type: str = 'all') -> Dict:
    """Collect results from all transfer experiments."""
    results = {
        'season': {},
        'geo': {},
        'species': {}
    }
    
    strategies = ['linear', 'adapter', 'finetune']
    
    experiment_types = ['season', 'geo', 'species'] if experiment_type == 'all' else [experiment_type]
    
    for exp_type in experiment_types:
        for strategy in strategies:
            run_name = f"{exp_type}_transfer_{strategy}_seed{seed}"
            run_dir = os.path.join(runs_dir, run_name)
            
            if os.path.exists(run_dir):
                metrics = load_metrics(run_dir)
                checkpoint = find_best_checkpoint(run_dir)
                
                results[exp_type][strategy] = {
                    'metrics': metrics,
                    'checkpoint': checkpoint,
                    'run_dir': run_dir
                }
                print(f"✓ Found results for {run_name}")
            else:
                print(f"✗ No results found for {run_name}")
    
    return results


def create_comparison_table(results: Dict) -> pd.DataFrame:
    """Create a comparison table of all experiments."""
    rows = []
    
    metric_names = ['test_map', 'test_top10', 'test_top30', 'val_map', 'val_loss']
    
    for exp_type, strategies in results.items():
        for strategy, data in strategies.items():
            if data.get('metrics'):
                row = {
                    'Experiment': exp_type.capitalize(),
                    'Strategy': strategy.capitalize(),
                }
                
                for metric in metric_names:
                    value = data['metrics'].get(metric, data['metrics'].get(f'test/{metric}', 'N/A'))
                    if isinstance(value, float):
                        row[metric] = f"{value:.4f}"
                    else:
                        row[metric] = str(value)
                
                rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        return df
    else:
        return pd.DataFrame()


def plot_comparison_bar(results: Dict, output_dir: str, metric: str = 'test_map'):
    """Create bar chart comparing all experiments."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    experiments = []
    strategies = []
    values = []
    
    strategy_order = ['linear', 'adapter', 'finetune']
    exp_order = ['season', 'geo', 'species']
    
    for exp_type in exp_order:
        if exp_type in results:
            for strategy in strategy_order:
                if strategy in results[exp_type]:
                    data = results[exp_type][strategy]
                    if data.get('metrics'):
                        value = data['metrics'].get(metric, data['metrics'].get(f'test/{metric}'))
                        if value is not None and isinstance(value, (int, float)):
                            experiments.append(exp_type.capitalize())
                            strategies.append(strategy.capitalize())
                            values.append(float(value))
    
    if not values:
        print("No data available for plotting")
        return
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Experiment': experiments,
        'Strategy': strategies,
        'Value': values
    })
    
    # Plot
    x = np.arange(len(set(experiments)))
    width = 0.25
    
    strategy_colors = {'Linear': '#1f77b4', 'Adapter': '#ff7f0e', 'Finetune': '#2ca02c'}
    
    for i, strategy in enumerate(['Linear', 'Adapter', 'Finetune']):
        strategy_data = df[df['Strategy'] == strategy]
        if len(strategy_data) > 0:
            bars = ax.bar(x + i * width, strategy_data['Value'], width, 
                         label=strategy, color=strategy_colors.get(strategy, 'gray'))
            
            # Add value labels
            for bar, val in zip(bars, strategy_data['Value']):
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Transfer Experiment Type')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Transfer Learning Comparison: {metric.replace("_", " ").title()}')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Season\n(Summer→Winter)', 'Geographic\n(USA→Kenya)', 'Species\n(Bird→Butterfly)'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'comparison_{metric}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, f'comparison_{metric}.png')}")


def plot_strategy_effectiveness(results: Dict, output_dir: str):
    """Plot strategy effectiveness across different transfer types."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    exp_names = {
        'season': 'Season Transfer\n(Summer → Winter)',
        'geo': 'Geographic Transfer\n(USA → Kenya)',
        'species': 'Species Transfer\n(Bird → Butterfly)'
    }
    
    metrics_to_plot = ['test_map', 'test_top10', 'test_top30']
    
    for idx, (exp_type, exp_data) in enumerate(results.items()):
        if not exp_data:
            continue
            
        ax = axes[idx]
        
        strategies = []
        metric_values = {m: [] for m in metrics_to_plot}
        
        for strategy in ['linear', 'adapter', 'finetune']:
            if strategy in exp_data and exp_data[strategy].get('metrics'):
                strategies.append(strategy.capitalize())
                for metric in metrics_to_plot:
                    val = exp_data[strategy]['metrics'].get(metric, 0)
                    if isinstance(val, (int, float)):
                        metric_values[metric].append(float(val))
                    else:
                        metric_values[metric].append(0)
        
        if strategies:
            x = np.arange(len(strategies))
            width = 0.25
            
            for i, (metric, values) in enumerate(metric_values.items()):
                if values:
                    ax.bar(x + i * width, values, width, 
                          label=metric.replace('test_', '').replace('_', ' ').title())
            
            ax.set_xticks(x + width)
            ax.set_xticklabels(strategies)
            ax.set_ylabel('Score')
            ax.set_title(exp_names.get(exp_type, exp_type))
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Transfer Strategy Comparison Across Experiments', fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'strategy_effectiveness.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'strategy_effectiveness.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'strategy_effectiveness.png')}")


def plot_transfer_difficulty(results: Dict, output_dir: str):
    """Plot showing relative difficulty of different transfer scenarios."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate average performance for each transfer type
    exp_scores = {}
    for exp_type, strategies in results.items():
        scores = []
        for strategy, data in strategies.items():
            if data.get('metrics'):
                val = data['metrics'].get('test_map', data['metrics'].get('val_map', 0))
                if isinstance(val, (int, float)):
                    scores.append(float(val))
        if scores:
            exp_scores[exp_type] = np.mean(scores)
    
    if not exp_scores:
        print("No data available for difficulty plot")
        return
    
    # Sort by score (higher = easier transfer)
    sorted_exps = sorted(exp_scores.items(), key=lambda x: x[1], reverse=True)
    
    names = [exp[0].capitalize() for exp in sorted_exps]
    scores = [exp[1] for exp in sorted_exps]
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green to red
    bars = ax.barh(names, scores, color=colors[:len(names)])
    
    # Add annotations
    for bar, score in zip(bars, scores):
        ax.annotate(f'{score:.4f}',
                   xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                   xytext=(5, 0),
                   textcoords="offset points",
                   ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Average Test mAP')
    ax.set_title('Transfer Difficulty Comparison\n(Higher score = Easier transfer)')
    ax.grid(axis='x', alpha=0.3)
    
    # Add difficulty annotations
    ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'transfer_difficulty.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'transfer_difficulty.png')}")


def generate_report(results: Dict, output_dir: str):
    """Generate a comprehensive markdown report."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_lines = [
        "# Transfer Learning Experiment Report",
        "",
        "## Overview",
        "",
        "This report summarizes the transfer learning experiments for the HGCP+FDA model.",
        "",
        "### Transfer Scenarios:",
        "1. **Season Transfer**: USA-Summer → USA-Winter (same species, different season)",
        "2. **Geographic Transfer**: USA → Kenya (different species, different region)", 
        "3. **Species Transfer**: Bird → Butterfly (different taxa, same region)",
        "",
        "### Transfer Strategies:",
        "- **Linear Probe**: Freeze all pretrained weights, only train new classifier",
        "- **Adapter Tune**: Freeze backbone, train adapters + HGCP + FDA + classifier",
        "- **Full Fine-tune**: Train all components with lower learning rate for backbone",
        "",
        "## Results Summary",
        "",
    ]
    
    # Create comparison table
    df = create_comparison_table(results)
    if len(df) > 0:
        report_lines.append("### Performance Comparison")
        report_lines.append("")
        report_lines.append(df.to_markdown(index=False))
        report_lines.append("")
    
    # Detailed analysis for each experiment type
    for exp_type, exp_name in [('season', 'Season Transfer'), 
                               ('geo', 'Geographic Transfer'),
                               ('species', 'Species Transfer')]:
        if exp_type in results and results[exp_type]:
            report_lines.append(f"## {exp_name}")
            report_lines.append("")
            
            best_strategy = None
            best_score = -1
            
            for strategy, data in results[exp_type].items():
                if data.get('metrics'):
                    score = data['metrics'].get('test_map', 0)
                    if isinstance(score, (int, float)) and score > best_score:
                        best_score = score
                        best_strategy = strategy
            
            if best_strategy:
                report_lines.append(f"**Best Strategy**: {best_strategy.capitalize()} (mAP: {best_score:.4f})")
            report_lines.append("")
            
            for strategy, data in results[exp_type].items():
                report_lines.append(f"### {strategy.capitalize()}")
                if data.get('metrics'):
                    report_lines.append("```")
                    for k, v in data['metrics'].items():
                        if isinstance(v, float):
                            report_lines.append(f"  {k}: {v:.4f}")
                        else:
                            report_lines.append(f"  {k}: {v}")
                    report_lines.append("```")
                if data.get('checkpoint'):
                    report_lines.append(f"Checkpoint: `{data['checkpoint']}`")
                report_lines.append("")
    
    # Key findings
    report_lines.extend([
        "## Key Findings",
        "",
        "1. **Season Transfer** typically shows the best results due to similar data distribution",
        "2. **Geographic Transfer** faces challenges due to different species and environmental variables",
        "3. **Species Transfer** tests cross-taxa generalization capability",
        "4. **Adapter Tuning** often provides a good balance between performance and efficiency",
        "",
        "## Recommendations",
        "",
        "- For similar domains (season transfer): Linear Probe may be sufficient",
        "- For different domains (geo/species): Adapter Tuning recommended",
        "- Full Fine-tune best for maximum performance when compute budget allows",
        "",
        "---",
        "*Report generated automatically by transfer_learning_eval.py*"
    ])
    
    # Write report
    report_path = os.path.join(output_dir, 'transfer_learning_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report saved to: {report_path}")
    
    # Also save results as JSON
    results_json = {}
    for exp_type, strategies in results.items():
        results_json[exp_type] = {}
        for strategy, data in strategies.items():
            results_json[exp_type][strategy] = {
                'metrics': data.get('metrics'),
                'checkpoint': data.get('checkpoint')
            }
    
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results JSON saved to: {json_path}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Transfer Learning Evaluation")
    print("=" * 60)
    
    # Collect results
    results = collect_experiment_results(args.runs_dir, args.seed, args.experiment_type)
    
    # Generate report
    generate_report(results, args.output_dir)
    
    # Create visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        plot_comparison_bar(results, args.output_dir, 'test_map')
        plot_comparison_bar(results, args.output_dir, 'test_top10')
        plot_strategy_effectiveness(results, args.output_dir)
        plot_transfer_difficulty(results, args.output_dir)
    
    # Print summary table
    df = create_comparison_table(results)
    if len(df) > 0:
        print("\n" + "=" * 60)
        print("Summary Table")
        print("=" * 60)
        print(df.to_string(index=False))
    
    print("\n✅ Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
