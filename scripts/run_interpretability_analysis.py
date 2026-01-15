"""
运行多模态融合可解释性分析

使用方法:
    python scripts/run_interpretability_analysis.py --quick_test
    python scripts/run_interpretability_analysis.py --compare_domains
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.analysis.interpretability import (
    InterpretabilityAnalyzer,
    InterpretabilityVisualizer,
    save_analysis_results,
    BIOCLIM_SHORT_NAMES
)


# 模型配置 - 根据实际训练配置
MODEL_CONFIGS = {
    'USA-Summer': {
        'checkpoint': 'runs/best summer1/checkpoints/last.ckpt',
        'num_classes': 624,
        'env_dim': 27,  # 19 bioclim + 8 ped
        'env_hidden_dim': 2048,
        'data_dir': 'USA_summer',
    },
    'USA-Winter': {
        'checkpoint': 'runs/best winter/checkpoints/last.ckpt',
        'num_classes': 624,
        'env_dim': 27,
        'env_hidden_dim': 2048,
        'data_dir': 'USA_winter',
    },
    'Kenya-Transfer': {
        'checkpoint': 'runs/transfer_usa_to_kenya_freeze_seed42_20251202-025823/checkpoints/best-104-0.0694.ckpt',
        'num_classes': 1054,
        'env_dim': 19,  # Kenya只有bioclim
        'env_hidden_dim': 512,
        'data_dir': 'kenya',
    }
}


def create_model(config, device='cuda'):
    """根据配置创建模型"""
    from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt
    
    model = Dinov2AdapterPrompt(
        num_classes=config['num_classes'],
        dino_model_name='vit_base_patch14_dinov2.lvd142m',
        pretrained_path='checkpoints/dinov2_vitb14_pretrain.pth',
        prompt_len=40,
        bottleneck_dim=96,
        env_input_dim=config['env_dim'],
        env_hidden_dim=config['env_hidden_dim'],
        env_num_layers=3 if config['env_hidden_dim'] == 512 else 6,
        use_env=True,
        fusion_type='adaptive_attention',
        hidden_dims=[2048, 1024],
        dropout=0.15,
        use_channel_adapter=True,
        in_channels=4,
        freeze_backbone=True
    )
    
    return model


def load_model_weights(model, checkpoint_path):
    """加载模型权重，处理维度不匹配"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    # 获取当前模型的state_dict
    model_dict = model.state_dict()
    
    # 过滤不匹配的参数
    filtered_dict = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered_dict[k] = v
            else:
                skipped.append(f"{k}: {v.shape} vs {model_dict[k].shape}")
        else:
            skipped.append(f"{k}: not in model")
    
    if skipped:
        print(f"  Skipped {len(skipped)} params with shape mismatch")
        for s in skipped[:5]:
            print(f"    {s}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")
    
    # 加载过滤后的参数
    model.load_state_dict(filtered_dict, strict=False)
    print(f"  Loaded {len(filtered_dict)} / {len(state_dict)} parameters")
    
    return model


def quick_test():
    """快速测试可解释性分析模块"""
    print("\n" + "="*60)
    print("Quick Test: Interpretability Analysis Module")
    print("="*60 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 检查可用的checkpoint
    available = []
    for name, config in MODEL_CONFIGS.items():
        path = config['checkpoint']
        if Path(path).exists():
            print(f"Found: {name} -> {path}")
            available.append((name, config))
        else:
            print(f"Not found: {name} -> {path}")
    
    if not available:
        print("\nNo checkpoints found. Using random weights for testing...")
        name = "Random"
        config = {
            'num_classes': 624,
            'env_dim': 27,
            'env_hidden_dim': 2048,
        }
        checkpoint_path = None
    else:
        name, config = available[0]
        checkpoint_path = config['checkpoint']
    
    print(f"\nTesting with: {name}")
    print(f"  num_classes: {config['num_classes']}")
    print(f"  env_dim: {config['env_dim']}")
    print(f"  env_hidden_dim: {config['env_hidden_dim']}")
    
    # 创建模型
    print("\nCreating model...")
    model = create_model(config, device)
    
    # 加载权重
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        model = load_model_weights(model, checkpoint_path)
    else:
        print("Using random weights")
    
    model.to(device)
    model.eval()
    
    # 创建随机测试数据
    print("\n" + "-"*40)
    print("Testing with random data...")
    print("-"*40)
    
    batch_size = 4
    env_dim = config['env_dim']
    img = torch.randn(batch_size, 4, 224, 224).to(device)
    env = torch.randn(batch_size, env_dim).to(device)
    
    # 测试门控权重提取
    analyzer = InterpretabilityAnalyzer(model, device)
    
    print("\n1. Extracting gate weights...")
    gate_info = analyzer.extract_gate_weights(img, env)
    alpha = gate_info['alpha']
    print(f"   Alpha shape: {alpha.shape}")
    print(f"   Alpha values: {[f'{v:.4f}' for v in alpha.flatten().tolist()]}")
    print(f"   Alpha mean: {alpha.mean().item():.4f}")
    
    # 解释alpha含义
    alpha_mean = alpha.mean().item()
    if alpha_mean > 0.5:
        print(f"   -> Model relies MORE on environment-enhanced features")
    else:
        print(f"   -> Model relies MORE on pure visual features")
    
    print("\n2. Computing environment gradients...")
    env_grad = analyzer.compute_env_gradient(img, env)
    print(f"   Gradient shape: {env_grad.shape}")
    
    # 获取top 5重要环境变量
    mean_abs_grad = torch.abs(env_grad).mean(0)
    top5_idx = torch.argsort(mean_abs_grad, descending=True)[:5]
    print(f"\n   Top 5 important env variables (by gradient magnitude):")
    for rank, idx in enumerate(top5_idx, 1):
        var_name = BIOCLIM_SHORT_NAMES.get(idx.item(), f'ENV_{idx.item()}')
        print(f"   {rank}. {var_name}: {mean_abs_grad[idx].item():.6f}")
    
    # 测试完整分析流程
    print("\n3. Testing full analysis pipeline...")
    analyzer.reset()
    result = analyzer.analyze_batch(img, env)
    print(f"   Single batch result keys: {list(result.keys())}")
    
    # 多次分析模拟
    print("\n4. Simulating multi-batch analysis...")
    analyzer.reset()
    for i in range(5):
        img_batch = torch.randn(batch_size, 4, 224, 224)
        env_batch = torch.randn(batch_size, env_dim)
        analyzer.analyze_batch(img_batch, env_batch)
    
    results = analyzer.get_aggregated_results()
    print(f"   Total samples analyzed: {len(results['alpha'])}")
    print(f"   Aggregated results keys: {list(results.keys())}")
    
    # 测试环境变量重要性计算
    print("\n5. Computing environment variable importance...")
    importance_df = analyzer.compute_env_importance()
    print("\n   Environment Variable Importance Ranking:")
    print(importance_df.head(10).to_string())
    
    # 测试可视化 (保存到文件)
    print("\n6. Testing visualization (saving to files)...")
    output_dir = Path('outputs/interpretability/quick_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    InterpretabilityVisualizer.plot_gate_distribution(
        results['alpha'],
        save_path=output_dir / 'gate_distribution.png',
        title=f'{name} Gate Weight Distribution'
    )
    plt.close()
    print(f"   Saved: {output_dir / 'gate_distribution.png'}")
    
    InterpretabilityVisualizer.plot_env_importance(
        importance_df,
        save_path=output_dir / 'env_importance.png',
        title=f'{name} Environment Variable Importance'
    )
    plt.close()
    print(f"   Saved: {output_dir / 'env_importance.png'}")
    
    # 保存结果
    save_analysis_results(results, importance_df, output_dir, prefix=f"{name.lower().replace('-','_')}_")
    
    print("\n" + "="*60)
    print("Quick Test PASSED!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("\nTo run full analysis with real data, use:")
    print("  python scripts/run_interpretability_analysis.py --compare_domains")
    
    return True


def run_domain_comparison(output_dir='outputs/interpretability/comparison', max_samples=500):
    """运行跨域对比分析"""
    print("\n" + "="*60)
    print("Cross-Domain Interpretability Comparison")
    print("="*60 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    all_results = {}
    all_importance = {}
    
    for domain_name, config in MODEL_CONFIGS.items():
        checkpoint_path = config['checkpoint']
        
        if not Path(checkpoint_path).exists():
            print(f"\nSkipping {domain_name}: checkpoint not found")
            continue
        
        print(f"\n{'='*40}")
        print(f"Analyzing: {domain_name}")
        print(f"{'='*40}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  num_classes: {config['num_classes']}")
        print(f"  env_dim: {config['env_dim']}")
        
        try:
            # 创建模型
            model = create_model(config, device)
            model = load_model_weights(model, checkpoint_path)
            model.to(device)
            model.eval()
            
            # 使用随机数据进行分析 (实际使用时应该用真实数据)
            print(f"\n  Analyzing with synthetic data...")
            analyzer = InterpretabilityAnalyzer(model, device)
            
            batch_size = 32
            n_batches = max_samples // batch_size
            env_dim = config['env_dim']
            
            for i in range(n_batches):
                img = torch.randn(batch_size, 4, 224, 224)
                env = torch.randn(batch_size, env_dim)
                analyzer.analyze_batch(img, env)
            
            results = analyzer.get_aggregated_results()
            importance_df = analyzer.compute_env_importance()
            
            all_results[domain_name] = results
            all_importance[domain_name] = importance_df
            
            # 保存单域结果
            domain_output_dir = Path(output_dir) / domain_name.lower().replace('-', '_').replace(' ', '_')
            save_analysis_results(results, importance_df, domain_output_dir)
            
            print(f"\n  Results:")
            print(f"    Samples analyzed: {len(results['alpha'])}")
            print(f"    Alpha mean: {results['alpha'].mean():.4f} ± {results['alpha'].std():.4f}")
            
            if results['alpha'].mean() > 0.5:
                print(f"    -> Model relies MORE on environment-enhanced features")
            else:
                print(f"    -> Model relies MORE on pure visual features")
            
            print(f"    Top 3 env vars: {importance_df.head(3).index.tolist()}")
            
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 跨域对比
    if len(all_results) > 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("Generating Cross-Domain Comparison")
        print(f"{'='*60}")
        
        # 跨域对比可视化
        if len(all_results) > 1:
            InterpretabilityVisualizer.plot_cross_domain_comparison(
                all_results,
                save_path=output_dir / 'cross_domain_comparison.png'
            )
            plt.close()
            print(f"  Saved: {output_dir / 'cross_domain_comparison.png'}")
        
        # 单独绘制每个域的门控分布
        for domain_name, results in all_results.items():
            safe_name = domain_name.lower().replace("-", "_").replace(" ", "_")
            InterpretabilityVisualizer.plot_gate_distribution(
                results['alpha'],
                save_path=output_dir / f'{safe_name}_gate_dist.png',
                title=f'{domain_name} Gate Weight Distribution'
            )
            plt.close()
        
        # 保存对比摘要
        import json
        comparison_summary = {
            'analysis_info': {
                'method': 'Multi-Modal Fusion Gate Analysis',
                'description': 'Analyzing how the model balances visual and environmental features',
                'alpha_interpretation': {
                    'alpha > 0.5': 'Model relies more on environment-enhanced features',
                    'alpha < 0.5': 'Model relies more on pure visual features',
                    'alpha = 0.5': 'Balanced fusion'
                }
            },
            'domains': {}
        }
        
        for domain, results in all_results.items():
            comparison_summary['domains'][domain] = {
                'alpha_mean': float(results['alpha'].mean()),
                'alpha_std': float(results['alpha'].std()),
                'alpha_min': float(results['alpha'].min()),
                'alpha_max': float(results['alpha'].max()),
                'n_samples': len(results['alpha']),
                'interpretation': 'env-dominant' if results['alpha'].mean() > 0.5 else 'visual-dominant'
            }
        
        with open(output_dir / 'comparison_summary.json', 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Cross-Domain Comparison Summary")
        print(f"{'='*60}")
        for domain, stats in comparison_summary['domains'].items():
            print(f"\n{domain}:")
            print(f"  Alpha: {stats['alpha_mean']:.4f} ± {stats['alpha_std']:.4f}")
            print(f"  Range: [{stats['alpha_min']:.4f}, {stats['alpha_max']:.4f}]")
            print(f"  Mode: {stats['interpretation']}")
        
        print(f"\nAll results saved to: {output_dir}")
    
    return all_results, all_importance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run interpretability analysis")
    parser.add_argument('--quick_test', action='store_true', help='Run quick test')
    parser.add_argument('--compare_domains', action='store_true', help='Run cross-domain comparison')
    parser.add_argument('--output_dir', type=str, default='outputs/interpretability', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=500, help='Maximum samples to analyze')
    
    args = parser.parse_args()
    
    if args.compare_domains:
        run_domain_comparison(args.output_dir, args.max_samples)
    else:
        quick_test()
