#!/usr/bin/env python3
"""
不确定性量化分析 - Monte Carlo Dropout
为Ecological Informatics论文增加创新点
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, '/sunyuxuan/satbird')

from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt

# 配置 - 使用adaptive_attention融合类型
DOMAINS = {
    'USA-Summer': {
        'checkpoint': 'runs/best summer2/checkpoints/best-48-0.0516.ckpt',
        'data_dir': 'USA_summer',
        'num_classes': 624,
        'env_dim': 27,
    },
    'USA-Winter': {
        'checkpoint': 'runs/best winter/checkpoints/best-80-0.0479.ckpt',
        'data_dir': 'USA_winter',
        'num_classes': 670,
        'env_dim': 27,
    },
}

MC_SAMPLES = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_sample(data_dir, hotspot_id):
    env_path = os.path.join(data_dir, 'environmental', f'{hotspot_id}.npy')
    img_path = os.path.join(data_dir, 'images', f'{hotspot_id}.tif')
    
    if not os.path.exists(env_path) or not os.path.exists(img_path):
        return None
    
    env_data = np.load(env_path).astype(np.float32)
    
    with rasterio.open(img_path) as src:
        img = src.read().astype(np.float32)
    
    img = np.clip(img / 10000.0, 0, 1)
    
    if img.shape[1] != 224 or img.shape[2] != 224:
        from torchvision.transforms.functional import resize
        img = resize(torch.from_numpy(img), [224, 224]).numpy()
    
    if len(env_data.shape) > 1:
        env_data = env_data.mean(axis=(1, 2)) if len(env_data.shape) == 3 else env_data.flatten()
    
    return {
        'image': torch.from_numpy(img).unsqueeze(0),
        'env': torch.from_numpy(env_data[:27]).unsqueeze(0)
    }


def load_model(ckpt_path, num_classes, env_dim=27):
    print(f"Loading: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 使用adaptive_attention融合类型 - 匹配checkpoint
    model = Dinov2AdapterPrompt(
        num_classes=num_classes,
        dino_model_name='vit_base_patch14_dinov2.lvd142m',
        pretrained_path='checkpoints/dinov2_vitb14_pretrain.pth',
        prompt_len=40,
        bottleneck_dim=96,
        env_input_dim=env_dim,
        env_hidden_dim=2048,  # 匹配checkpoint
        env_num_layers=3,
        use_env=True,
        fusion_type='adaptive_attention',  # 关键：使用自适应融合
        hidden_dims=[2048, 1024],
        dropout=0.15,
        use_channel_adapter=True,
        in_channels=4,
        freeze_backbone=True,
    )
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('model.', '') if k.startswith('model.') else k
        new_state_dict[new_k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(DEVICE)
    
    return model


def enable_dropout(model):
    dropout_count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
            dropout_count += 1
    return dropout_count


def mc_dropout_inference(model, images, env, n_samples=30):
    model.eval()
    enable_dropout(model)
    
    all_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(images, env)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
    
    all_preds = np.array(all_preds)
    mean_pred = all_preds.mean(axis=0)
    pred_std = all_preds.std(axis=0)
    
    return mean_pred, pred_std, all_preds


def compute_entropy(probs):
    epsilon = 1e-10
    probs = np.clip(probs, epsilon, 1 - epsilon)
    entropy = -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
    return entropy


def analyze_domain(domain_name, config, output_dir):
    print(f"\n{'='*70}")
    print(f"分析域: {domain_name}")
    print(f"{'='*70}")
    
    model = load_model(config['checkpoint'], config['num_classes'], config['env_dim'])
    
    # 显示启用的dropout数量
    dropout_count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            dropout_count += 1
    print(f"模型中有 {dropout_count} 个Dropout层")
    
    test_df = pd.read_csv(os.path.join(config['data_dir'], 'test_split.csv'))
    print(f"测试样本数: {len(test_df)}")
    
    all_mean_preds = []
    all_pred_stds = []
    
    print(f"开始MC Dropout采样 (n={MC_SAMPLES})...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
        hotspot_id = row['hotspot_id']
        sample = load_sample(config['data_dir'], hotspot_id)
        
        if sample is None:
            continue
        
        images = sample['image'].to(DEVICE)
        env = sample['env'].to(DEVICE)
        env = torch.nan_to_num(env, nan=0.0)
        
        mean_pred, pred_std, _ = mc_dropout_inference(model, images, env, MC_SAMPLES)
        
        all_mean_preds.append(mean_pred[0])
        all_pred_stds.append(pred_std[0])
    
    all_mean_preds = np.array(all_mean_preds)
    all_pred_stds = np.array(all_pred_stds)
    
    print(f"成功分析样本数: {len(all_mean_preds)}")
    
    sample_uncertainty = all_pred_stds.mean(axis=1)
    species_uncertainty = all_pred_stds.mean(axis=0)
    
    entropy = compute_entropy(all_mean_preds)
    sample_entropy = entropy.mean(axis=1)
    species_entropy = entropy.mean(axis=0)
    
    print(f"\n【样本级不确定性】")
    print(f"  平均std: {sample_uncertainty.mean():.4f} +/- {sample_uncertainty.std():.4f}")
    print(f"  平均熵: {sample_entropy.mean():.4f}")
    print(f"  高不确定性样本(>0.05): {(sample_uncertainty > 0.05).sum()}/{len(sample_uncertainty)}")
    
    print(f"\n【物种级不确定性】")
    print(f"  平均std: {species_uncertainty.mean():.4f} +/- {species_uncertainty.std():.4f}")
    print(f"  高不确定性物种(>0.05): {(species_uncertainty > 0.05).sum()}/{len(species_uncertainty)}")
    
    # Top-10 高不确定性物种
    sorted_idx = np.argsort(species_uncertainty)[::-1]
    print(f"\n【Top-10 高不确定性物种】")
    for i, idx in enumerate(sorted_idx[:10], 1):
        print(f"  {i:2d}. Species {idx}: std={species_uncertainty[idx]:.4f}")
    
    domain_dir = output_dir / domain_name.lower().replace('-', '_')
    domain_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(domain_dir / 'mean_predictions.npy', all_mean_preds)
    np.save(domain_dir / 'prediction_std.npy', all_pred_stds)
    np.save(domain_dir / 'sample_uncertainty.npy', sample_uncertainty)
    np.save(domain_dir / 'species_uncertainty.npy', species_uncertainty)
    np.save(domain_dir / 'entropy.npy', entropy)
    
    summary = {
        'domain': domain_name,
        'mc_samples': MC_SAMPLES,
        'num_samples': len(all_mean_preds),
        'num_species': all_mean_preds.shape[1],
        'sample_uncertainty': {
            'mean': float(sample_uncertainty.mean()),
            'std': float(sample_uncertainty.std()),
            'min': float(sample_uncertainty.min()),
            'max': float(sample_uncertainty.max()),
        },
        'species_uncertainty': {
            'mean': float(species_uncertainty.mean()),
            'std': float(species_uncertainty.std()),
            'min': float(species_uncertainty.min()),
            'max': float(species_uncertainty.max()),
        },
    }
    
    with open(domain_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return {
        'sample_uncertainty': sample_uncertainty,
        'species_uncertainty': species_uncertainty,
        'sample_entropy': sample_entropy,
        'species_entropy': species_entropy,
        'summary': summary,
    }


def generate_visualizations(results, output_dir):
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    plt.rcParams['figure.dpi'] = 150
    
    # 图1: 样本不确定性分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#FF6B6B', '#4ECDC4']
    
    for i, (domain, data) in enumerate(results.items()):
        axes[i].hist(data['sample_uncertainty'], bins=50, alpha=0.7, color=colors[i])
        axes[i].set_xlabel('Sample Uncertainty (std)', fontsize=11)
        axes[i].set_ylabel('Frequency', fontsize=11)
        axes[i].set_title(f'{domain}\nMean: {data["sample_uncertainty"].mean():.4f}', fontsize=12, fontweight='bold')
        axes[i].axvline(data['sample_uncertainty'].mean(), color='red', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'sample_uncertainty_distribution.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(fig_dir / 'sample_uncertainty_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("sample_uncertainty_distribution.pdf")
    
    # 图2: 物种不确定性排名
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (domain, data) in enumerate(results.items()):
        sorted_unc = np.sort(data['species_uncertainty'])[::-1]
        axes[i].bar(range(len(sorted_unc)), sorted_unc, alpha=0.7, color=colors[i])
        axes[i].set_xlabel('Species (sorted by uncertainty)', fontsize=11)
        axes[i].set_ylabel('Species Uncertainty (std)', fontsize=11)
        axes[i].set_title(f'{domain}\nMean: {data["species_uncertainty"].mean():.4f}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'species_uncertainty_ranking.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(fig_dir / 'species_uncertainty_ranking.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("species_uncertainty_ranking.pdf")
    
    # 图3: 不确定性 vs 熵
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (domain, data) in enumerate(results.items()):
        axes[i].scatter(data['sample_uncertainty'], data['sample_entropy'], alpha=0.3, s=5, c=colors[i])
        axes[i].set_xlabel('Sample Uncertainty (std)', fontsize=11)
        axes[i].set_ylabel('Sample Entropy', fontsize=11)
        axes[i].set_title(f'{domain}', fontsize=12, fontweight='bold')
        corr = np.corrcoef(data['sample_uncertainty'], data['sample_entropy'])[0, 1]
        axes[i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[i].transAxes, fontsize=11, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'uncertainty_vs_entropy.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(fig_dir / 'uncertainty_vs_entropy.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("uncertainty_vs_entropy.pdf")
    
    # 图4: 跨域对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    domains = list(results.keys())
    x = np.arange(len(domains))
    width = 0.35
    
    sample_means = [results[d]['summary']['sample_uncertainty']['mean'] for d in domains]
    species_means = [results[d]['summary']['species_uncertainty']['mean'] for d in domains]
    
    bars1 = ax.bar(x - width/2, sample_means, width, label='Sample Uncertainty', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, species_means, width, label='Species Uncertainty', color='#4ECDC4', alpha=0.8)
    
    ax.set_ylabel('Mean Uncertainty (std)', fontsize=12)
    ax.set_title('Cross-Domain Uncertainty Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'cross_domain_uncertainty.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(fig_dir / 'cross_domain_uncertainty.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("cross_domain_uncertainty.pdf")


def main():
    output_dir = Path('outputs/uncertainty_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Monte Carlo Dropout 不确定性量化分析")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"MC Samples: {MC_SAMPLES}")
    
    results = {}
    
    for domain_name, config in DOMAINS.items():
        results[domain_name] = analyze_domain(domain_name, config, output_dir)
    
    print("\n生成可视化图表...")
    generate_visualizations(results, output_dir)
    
    overall_summary = {domain: results[domain]['summary'] for domain in results}
    with open(output_dir / 'overall_summary.json', 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
