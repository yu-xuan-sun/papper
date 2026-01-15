#!/usr/bin/env python3
"""
全样本可解释性分析脚本 - V3
使用 get_gate_value() 获取真正有意义的门控值（0-1范围）
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader

# 设置绘图风格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# 环境变量名称映射
BIOCLIM_SHORT_NAMES = {
    0: 'AnnualTemp',
    1: 'DiurnalRange',
    2: 'Isotherm',
    3: 'TempSeason',
    4: 'MaxTempWarm',
    5: 'MinTempCold',
    6: 'TempRange',
    7: 'TempWetQ',
    8: 'TempDryQ',
    9: 'TempWarmQ',
    10: 'TempColdQ',
    11: 'AnnualPrecip',
    12: 'PrecipWetM',
    13: 'PrecipDryM',
    14: 'PrecipSeason',
    15: 'PrecipWetQ',
    16: 'PrecipDryQ',
    17: 'PrecipWarmQ',
    18: 'PrecipColdQ',
    19: 'Elevation',
    20: 'PopDensity',
    21: 'Forest',
    22: 'Urban',
    23: 'Water',
    24: 'Crop',
    25: 'Grass',
    26: 'NDVI',
}


# 域配置
DOMAIN_CONFIGS = {
    'USA-Summer': {
        'checkpoint': 'runs/best summer2/checkpoints/best-48-0.0516.ckpt',
        'num_classes': 624,
        'env_dim': 27,
        'data_dir': 'USA_summer',
        'prompt_len': 40,
        'bottleneck_dim': 96,
        'env_hidden_dim': 2048,
        'env_num_layers': 3,
        'unfreeze_last_n': 4,
    },
    'USA-Winter': {
        'checkpoint': 'runs/best winter/checkpoints/best-80-0.0479.ckpt',
        'num_classes': 670,
        'env_dim': 27,
        'data_dir': 'USA_winter',
        'prompt_len': 25,
        'bottleneck_dim': 96,
        'env_hidden_dim': 2048,
        'env_num_layers': 3,
        'unfreeze_last_n': 0,
    },
    'Kenya-Transfer': {
        'checkpoint': 'runs/transfer_usa_to_kenya_freeze_seed42_20251202-025823/checkpoints/best-104-0.0694.ckpt',
        'num_classes': 673,
        'env_dim': 27,
        'data_dir': 'kenya',
        'prompt_len': 40,
        'bottleneck_dim': 96,
        'env_hidden_dim': 2048,
        'env_num_layers': 3,
        'unfreeze_last_n': 4,
    }
}


def create_model(config, device='cuda'):
    """创建模型"""
    from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt
    
    model = Dinov2AdapterPrompt(
        num_classes=config['num_classes'],
        dino_model_name='vit_base_patch14_dinov2.lvd142m',
        pretrained_path='checkpoints/dinov2_vitb14_pretrain.pth',
        prompt_len=config['prompt_len'],
        bottleneck_dim=config['bottleneck_dim'],
        env_input_dim=config['env_dim'],
        env_hidden_dim=config['env_hidden_dim'],
        env_num_layers=config['env_num_layers'],
        use_env=True,
        fusion_type='adaptive_attention',
        hidden_dims=[2048, 1024],
        dropout=0.15,
        use_channel_adapter=True,
        in_channels=4,
        freeze_backbone=True,
        unfreeze_last_n_blocks=config.get('unfreeze_last_n', 0),
    )
    
    return model


def load_model_weights(model, checkpoint_path):
    """加载模型权重"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 移除 'model.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    return model


def create_dataloader(data_dir: str, split: str = 'test', batch_size: int = 64):
    """创建数据加载器"""
    from src.dataset.dataloader import EbirdVisionDataset
    
    data_path = Path(data_dir)
    stats_path = data_path / 'stats'
    
    # 加载统计数据
    env_means = np.load(stats_path / 'env_means.npy')
    env_stds = np.load(stats_path / 'env_stds.npy')
    img_means = np.load(stats_path / 'means_rgbnir.npy')
    img_stds = np.load(stats_path / 'stds_rgbnir.npy')
    
    # 加载split文件
    split_df = pd.read_csv(data_path / f'{split}_split.csv')
    
    # 创建transforms
    from torchvision import transforms as trsfs
    transform = trsfs.Compose([
        trsfs.Normalize(mean=img_means, std=img_stds),
    ])
    
    # 创建数据集
    dataset = EbirdVisionDataset(
        df_paths=split_df,
        data_base_dir=str(data_path),
        bands=['r', 'g', 'b', 'nir'],
        env=['ped', 'bioclim'],
        env_var_sizes={'ped': 8, 'bioclim': 19},
        transforms=None,
        mode=split,
        datatype='refl',
        target='probs',
        concat_env_to_sat=False,  # 环境特征作为单独向量
    )
    
    # 设置归一化
    dataset.env_means = env_means
    dataset.env_stds = env_stds
    dataset.img_means = img_means
    dataset.img_stds = img_stds
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader, len(dataset)


class FullInterpretabilityAnalyzer:
    """全样本可解释性分析器"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.gate_values = []
        self.final_alphas = []
        self.env_gradients = []
        self.env_features = []
        self.predictions = []
        
    def reset(self):
        self.gate_values = []
        self.final_alphas = []
        self.env_gradients = []
        self.env_features = []
        self.predictions = []
    
    @torch.no_grad()
    def extract_gate_info(self, img: torch.Tensor, env: torch.Tensor):
        """提取门控信息"""
        img = img.to(self.device)
        env = env.to(self.device)
        
        visual_feat = self.model.forward_visual_features(img)
        fusion = self.model.fusion
        
        if fusion is None:
            batch_size = img.size(0)
            return (
                torch.ones(batch_size, 1, device=self.device) * 0.5,
                torch.ones(batch_size, 1, device=self.device) * 0.5
            )
        
        # 使用 get_gate_value 获取第一次sigmoid后的值
        if hasattr(fusion, 'get_gate_value'):
            gate_value = fusion.get_gate_value(visual_feat, env)
        else:
            env_norm = fusion.env_input_norm(env)
            img_norm = fusion.img_input_norm(visual_feat)
            env_encoded = fusion.env_encoder(env_norm)
            concat_feat = torch.cat([img_norm, env_encoded], dim=1)
            gate_value = fusion.gate(concat_feat)
        
        temp = fusion.temperature.abs().clamp(min=0.01)
        final_alpha = torch.sigmoid(gate_value / temp)
        
        return gate_value, final_alpha
    
    def compute_env_gradient(self, img: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """计算环境变量梯度"""
        img = img.to(self.device)
        env = env.to(self.device).requires_grad_(True)
        
        logits = self.model(img, env)
        target = logits.argmax(dim=1)
        
        batch_size = logits.size(0)
        target_scores = logits[torch.arange(batch_size, device=self.device), target]
        
        grad = torch.autograd.grad(
            outputs=target_scores.sum(),
            inputs=env,
            create_graph=False,
            retain_graph=False
        )[0]
        
        return grad.detach()
    
    def analyze_dataloader(self, dataloader, desc: str = "Analyzing", compute_gradients: bool = True):
        """分析整个数据集"""
        self.reset()
        
        for batch in tqdm(dataloader, desc=desc):
            # 解包batch
            if isinstance(batch, dict):
                img = batch.get('sat')
                env = batch.get('env')
            else:
                continue
            
            if env is None or img is None:
                continue
            
            # 提取门控信息
            gate_value, final_alpha = self.extract_gate_info(img, env)
            self.gate_values.append(gate_value.cpu())
            self.final_alphas.append(final_alpha.cpu())
            self.env_features.append(env.cpu() if isinstance(env, torch.Tensor) else torch.tensor(env))
            
            # 计算梯度
            if compute_gradients:
                try:
                    grad = self.compute_env_gradient(img, env)
                    self.env_gradients.append(grad.cpu())
                except Exception as e:
                    pass
            
            # 获取预测
            with torch.no_grad():
                logits = self.model(img.to(self.device), env.to(self.device))
                pred = torch.sigmoid(logits)
                self.predictions.append(pred.cpu())
    
    def get_results(self) -> dict:
        """获取聚合结果"""
        results = {}
        
        if self.gate_values:
            results['gate_value'] = torch.cat(self.gate_values, dim=0).numpy()
        if self.final_alphas:
            results['final_alpha'] = torch.cat(self.final_alphas, dim=0).numpy()
        if self.env_gradients:
            results['env_gradient'] = torch.cat(self.env_gradients, dim=0).numpy()
        if self.env_features:
            results['env_features'] = torch.cat(self.env_features, dim=0).numpy()
        if self.predictions:
            results['predictions'] = torch.cat(self.predictions, dim=0).numpy()
        
        return results
    
    def compute_env_importance(self) -> pd.DataFrame:
        """计算环境变量重要性"""
        results = self.get_results()
        
        if 'env_gradient' not in results:
            return pd.DataFrame()
        
        gradients = results['env_gradient']
        env_dim = gradients.shape[1]
        
        importance = {
            'mean_abs_grad': np.abs(gradients).mean(axis=0),
            'std_grad': gradients.std(axis=0),
            'median_abs_grad': np.median(np.abs(gradients), axis=0),
            'positive_ratio': (gradients > 0).mean(axis=0),
        }
        
        var_names = [BIOCLIM_SHORT_NAMES.get(i, f'ENV_{i}') for i in range(env_dim)]
        df = pd.DataFrame(importance, index=var_names)
        df['rank'] = df['mean_abs_grad'].rank(ascending=False)
        df = df.sort_values('mean_abs_grad', ascending=False)
        
        return df


def plot_gate_comparison(results_dict: dict, save_dir: Path):
    """对比门控值 vs 最终alpha"""
    n = len(results_dict)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    
    if n == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, (domain, results) in enumerate(results_dict.items()):
        gate_values = results['gate_value'].flatten()
        final_alphas = results['final_alpha'].flatten()
        
        axes[idx, 0].hist(gate_values, bins=50, color=colors[idx % 3], alpha=0.7, edgecolor='black')
        axes[idx, 0].axvline(gate_values.mean(), color='red', linestyle='--', linewidth=2, 
                            label=f'Mean: {gate_values.mean():.4f}')
        axes[idx, 0].axvline(0.5, color='green', linestyle=':', linewidth=2, label='Balanced (0.5)')
        axes[idx, 0].set_xlabel('Gate Value (First Sigmoid)')
        axes[idx, 0].set_ylabel('Count')
        axes[idx, 0].set_title(f'{domain}: Gate Network Output\n(Mean={gate_values.mean():.4f}, Std={gate_values.std():.4f})')
        axes[idx, 0].legend()
        axes[idx, 0].set_xlim(0, 1)
        
        axes[idx, 1].hist(final_alphas, bins=50, color=colors[idx % 3], alpha=0.7, edgecolor='black')
        axes[idx, 1].axvline(final_alphas.mean(), color='red', linestyle='--', linewidth=2, 
                            label=f'Mean: {final_alphas.mean():.4f}')
        axes[idx, 1].set_xlabel('Final Alpha (After Temp Scaling)')
        axes[idx, 1].set_ylabel('Count')
        axes[idx, 1].set_title(f'{domain}: Final Alpha\n(Mean={final_alphas.mean():.4f}, Std={final_alphas.std():.4f})')
        axes[idx, 1].legend()
        axes[idx, 1].set_xlim(0, 1)
    
    plt.tight_layout()
    save_path = save_dir / 'gate_vs_alpha_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_gate_distribution_detailed(results_dict: dict, save_dir: Path):
    """详细的门控值分布分析"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (domain, results) in enumerate(results_dict.items()):
        gate_values = results['gate_value'].flatten()
        axes[0, 0].hist(gate_values, bins=50, alpha=0.5, label=domain, color=colors[i % 3])
    axes[0, 0].set_xlabel('Gate Value')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Gate Value Distribution (All Domains)')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 1)
    
    for i, (domain, results) in enumerate(results_dict.items()):
        gate_values = results['gate_value'].flatten()
        sns.kdeplot(gate_values, ax=axes[0, 1], label=domain, fill=True, alpha=0.3, color=colors[i % 3])
    axes[0, 1].set_xlabel('Gate Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Gate Value KDE (All Domains)')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 1)
    
    gate_data = [results['gate_value'].flatten() for results in results_dict.values()]
    labels = list(results_dict.keys())
    
    bp = axes[1, 0].boxplot(gate_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1, 0].set_ylabel('Gate Value')
    axes[1, 0].set_title('Gate Value Distribution by Domain')
    axes[1, 0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    stats_data = []
    for domain, results in results_dict.items():
        g = results['gate_value'].flatten()
        stats_data.append({
            'Domain': domain,
            'N': len(g),
            'Mean': f"{g.mean():.4f}",
            'Std': f"{g.std():.4f}",
            'Min': f"{g.min():.4f}",
            'Max': f"{g.max():.4f}",
            'Median': f"{np.median(g):.4f}",
        })
    
    stats_df = pd.DataFrame(stats_data)
    axes[1, 1].axis('off')
    table = axes[1, 1].table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    axes[1, 1].set_title('Gate Value Statistics', pad=20)
    
    plt.tight_layout()
    save_path = save_dir / 'gate_distribution_detailed.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return stats_df


def plot_env_importance_comparison(importance_dict: dict, save_dir: Path, top_k: int = 15):
    """对比各域的环境变量重要性"""
    n = len(importance_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 8))
    
    if n == 1:
        axes = [axes]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, (domain, df) in enumerate(importance_dict.items()):
        if df.empty:
            continue
        df_top = df.head(top_k)
        y_pos = np.arange(len(df_top))
        
        axes[idx].barh(y_pos, df_top['mean_abs_grad'], color=colors[idx % 3], alpha=0.8, edgecolor='black')
        axes[idx].set_yticks(y_pos)
        axes[idx].set_yticklabels(df_top.index)
        axes[idx].set_xlabel('Mean |Gradient|')
        axes[idx].set_title(f'{domain}\nTop-{top_k} Environment Variables')
        axes[idx].invert_yaxis()
        
        for i, (name, row) in enumerate(df_top.iterrows()):
            axes[idx].text(row['mean_abs_grad'] + 0.0005, i, f"{row['mean_abs_grad']:.4f}", va='center', fontsize=9)
    
    plt.tight_layout()
    save_path = save_dir / 'env_importance_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_gradient_heatmap(results_dict: dict, save_dir: Path, n_samples: int = 200):
    """环境变量梯度热力图"""
    n = len(results_dict)
    fig, axes = plt.subplots(n, 1, figsize=(16, 5 * n))
    
    if n == 1:
        axes = [axes]
    
    for idx, (domain, results) in enumerate(results_dict.items()):
        if 'env_gradient' not in results:
            continue
        
        grad = results['env_gradient']
        env_dim = grad.shape[1]
        
        if len(grad) > n_samples:
            indices = np.random.choice(len(grad), n_samples, replace=False)
            grad_subset = grad[indices]
        else:
            grad_subset = grad
        
        mean_abs_grad = np.abs(grad).mean(axis=0)
        sorted_idx = np.argsort(mean_abs_grad)[::-1]
        var_names = [BIOCLIM_SHORT_NAMES.get(i, f'V{i}') for i in sorted_idx]
        grad_sorted = grad_subset[:, sorted_idx]
        
        vmax = np.percentile(np.abs(grad_sorted), 95)
        im = axes[idx].imshow(grad_sorted.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        axes[idx].set_yticks(range(env_dim))
        axes[idx].set_yticklabels(var_names, fontsize=8)
        axes[idx].set_xlabel('Samples')
        axes[idx].set_ylabel('Environment Variables')
        axes[idx].set_title(f'{domain}: Environment Variable Gradients')
        plt.colorbar(im, ax=axes[idx], label='Gradient')
    
    plt.tight_layout()
    save_path = save_dir / 'gradient_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def analyze_single_domain(domain_name: str, config: dict, device: str = 'cuda', batch_size: int = 64):
    """分析单个域"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {domain_name}")
    print(f"{'='*60}")
    
    # 创建数据加载器
    dataloader, n_samples = create_dataloader(config['data_dir'], split='test', batch_size=batch_size)
    print(f"Total test samples: {n_samples}")
    
    # 创建并加载模型
    print(f"Loading checkpoint: {config['checkpoint']}")
    model = create_model(config, device)
    model = load_model_weights(model, config['checkpoint'])
    model.to(device)
    model.eval()
    
    # 分析
    analyzer = FullInterpretabilityAnalyzer(model, device)
    analyzer.analyze_dataloader(dataloader, desc=f"Analyzing {domain_name}")
    
    results = analyzer.get_results()
    importance_df = analyzer.compute_env_importance()
    
    # 打印统计
    print(f"\n--- {domain_name} Statistics ---")
    gate_values = results['gate_value'].flatten()
    final_alphas = results['final_alpha'].flatten()
    print(f"Gate Value: mean={gate_values.mean():.4f}, std={gate_values.std():.4f}, "
          f"min={gate_values.min():.4f}, max={gate_values.max():.4f}")
    print(f"Final Alpha: mean={final_alphas.mean():.4f}, std={final_alphas.std():.4f}")
    
    if not importance_df.empty:
        print(f"\nTop 5 Important Variables:")
        for i, (name, row) in enumerate(importance_df.head(5).iterrows()):
            print(f"  {i+1}. {name}: {row['mean_abs_grad']:.6f}")
    
    return results, importance_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='outputs/interpretability_v3')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    data_dir = output_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    all_results = {}
    all_importance = {}
    
    for domain_name, config in DOMAIN_CONFIGS.items():
        if not Path(config['checkpoint']).exists():
            print(f"Warning: Checkpoint not found: {config['checkpoint']}, skipping {domain_name}")
            continue
        
        results, importance = analyze_single_domain(
            domain_name=domain_name,
            config=config,
            device=args.device,
            batch_size=args.batch_size
        )
        
        all_results[domain_name] = results
        all_importance[domain_name] = importance
        
        # 保存数据
        key = domain_name.lower().replace("-", "_")
        np.save(data_dir / f'{key}_gate_values.npy', results['gate_value'])
        np.save(data_dir / f'{key}_final_alphas.npy', results['final_alpha'])
        if 'env_gradient' in results:
            np.save(data_dir / f'{key}_env_gradients.npy', results['env_gradient'])
        importance.to_csv(data_dir / f'{key}_importance.csv')
    
    if not all_results:
        print("No valid domains found!")
        return
    
    # 生成可视化
    print(f"\n{'='*60}")
    print("Generating Visualizations...")
    print(f"{'='*60}")
    
    plot_gate_comparison(all_results, figures_dir)
    stats_df = plot_gate_distribution_detailed(all_results, figures_dir)
    stats_df.to_csv(data_dir / 'gate_statistics.csv', index=False)
    plot_env_importance_comparison(all_importance, figures_dir)
    plot_gradient_heatmap(all_results, figures_dir)
    
    # 保存摘要
    summary = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'domains': list(all_results.keys()),
        'statistics': {}
    }
    
    for domain, results in all_results.items():
        g = results['gate_value'].flatten()
        summary['statistics'][domain] = {
            'n_samples': len(g),
            'gate_mean': float(g.mean()),
            'gate_std': float(g.std()),
            'gate_min': float(g.min()),
            'gate_max': float(g.max()),
            'top_5_env_vars': all_importance[domain].head(5).index.tolist() if domain in all_importance and not all_importance[domain].empty else []
        }
    
    with open(output_dir / 'analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
