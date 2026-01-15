#!/usr/bin/env python3
"""
视觉特征重要性分析
分析遥感影像各波段对预测的贡献
"""

import sys
sys.path.insert(0, '/sunyuxuan/satbird')

import os
import json
import torch
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt

# 配置
DOMAINS = {
    'USA-Summer': {
        'checkpoint': 'runs/best summer2/checkpoints/best-48-0.0516.ckpt',
        'data_dir': 'USA_summer',
        'num_classes': 624,
        'env_dim': 27,
        'env_hidden_dim': 2048,
    },
    'USA-Winter': {
        'checkpoint': 'runs/best winter/checkpoints/best-80-0.0479.ckpt',
        'data_dir': 'USA_winter',
        'num_classes': 670,
        'env_dim': 27,
        'env_hidden_dim': 2048,
    },
}

BAND_NAMES = ['Red', 'Green', 'Blue', 'NIR']


def load_sample(data_dir, hotspot_id, env_dim=27):
    """加载样本"""
    env_path = os.path.join(data_dir, 'environmental', f'{hotspot_id}.npy')
    img_path = os.path.join(data_dir, 'images', f'{hotspot_id}.tif')
    
    if not os.path.exists(env_path) or not os.path.exists(img_path):
        return None
    
    env_data = np.load(env_path).astype(np.float32)
    if len(env_data.shape) > 1:
        env_data = env_data.mean(axis=(1, 2)) if len(env_data.shape) == 3 else env_data.flatten()
    env_data = env_data[:env_dim]
    
    with rasterio.open(img_path) as src:
        img = src.read().astype(np.float32)
    
    img = np.clip(img / 10000.0, 0, 1)
    
    if img.shape[1] != 224 or img.shape[2] != 224:
        from torchvision.transforms.functional import resize
        img = resize(torch.from_numpy(img), [224, 224]).numpy()
    
    return {
        'image': torch.from_numpy(img).unsqueeze(0),
        'env': torch.from_numpy(env_data).unsqueeze(0)
    }


def load_model(ckpt_path, num_classes, env_dim, env_hidden_dim, device='cuda'):
    """加载模型"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    model = Dinov2AdapterPrompt(
        num_classes=num_classes,
        dino_model_name='vit_base_patch14_dinov2.lvd142m',
        pretrained_path='checkpoints/dinov2_vitb14_pretrain.pth',
        prompt_len=40,
        bottleneck_dim=96,
        env_input_dim=env_dim,
        env_hidden_dim=env_hidden_dim,
        env_num_layers=3,
        use_env=True,
        fusion_type='adaptive_attention',
        hidden_dims=[2048, 1024],
        dropout=0.15,
        use_channel_adapter=True,
        in_channels=4,
        freeze_backbone=True,
    )
    
    new_state_dict = {k[6:] if k.startswith('model.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model


def compute_visual_importance(model, image, env, device='cuda'):
    """计算视觉特征各波段的重要性（梯度）"""
    image = image.to(device).requires_grad_(True)
    env = env.to(device)
    
    model.zero_grad()
    logits = model(image, env)
    target_score = logits[:, logits.argmax(dim=1)].sum()
    target_score.backward()
    
    # 获取图像梯度 [1, 4, 224, 224]
    img_grad = image.grad.detach()
    
    # 计算每个波段的平均绝对梯度
    band_importance = img_grad.abs().mean(dim=(2, 3)).cpu().numpy().flatten()  # [4]
    
    return band_importance


def analyze_feature_contribution(model, image, env, device='cuda'):
    """分析视觉 vs 环境特征贡献"""
    image = image.to(device)
    env = env.to(device)
    
    with torch.no_grad():
        # 1. 完整预测
        full_output = model(image, env)
        full_pred = full_output.argmax(dim=1)
        full_score = full_output[:, full_pred].item()
        
        # 2. 仅视觉（环境置零）
        zero_env = torch.zeros_like(env)
        visual_only_output = model(image, zero_env)
        visual_only_score = visual_only_output[:, full_pred].item()
        
        # 3. 仅环境（图像置为均值）
        mean_image = torch.full_like(image, 0.5)
        env_only_output = model(mean_image, env)
        env_only_score = env_only_output[:, full_pred].item()
    
    return {
        'full_score': full_score,
        'visual_only_score': visual_only_score,
        'env_only_score': env_only_score,
        'visual_contribution': visual_only_score / (full_score + 1e-8),
        'env_contribution': env_only_score / (full_score + 1e-8),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    output_dir = Path('outputs/interpretability_v3')
    figures_dir = output_dir / 'figures'
    
    all_band_importance = {}
    all_contributions = {}
    
    for name, config in DOMAINS.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {name}")
        print(f"{'='*60}")
        
        model = load_model(
            config['checkpoint'], config['num_classes'],
            config['env_dim'], config['env_hidden_dim'], device
        )
        
        split_file = os.path.join(config['data_dir'], 'test_split.csv')
        df = pd.read_csv(split_file)
        hotspot_ids = df['hotspot_id'].tolist()[:500]  # 取500个样本
        
        band_importance_list = []
        contributions_list = []
        
        for hid in tqdm(hotspot_ids, desc=f"Analyzing {name}"):
            sample = load_sample(config['data_dir'], hid, config['env_dim'])
            if sample is None:
                continue
            
            try:
                # 波段重要性
                band_imp = compute_visual_importance(model, sample['image'], sample['env'], device)
                band_importance_list.append(band_imp)
                
                # 特征贡献
                contrib = analyze_feature_contribution(model, sample['image'], sample['env'], device)
                contributions_list.append(contrib)
            except Exception as e:
                continue
        
        all_band_importance[name] = np.array(band_importance_list)
        all_contributions[name] = contributions_list
        
        # 打印统计
        band_imp_mean = all_band_importance[name].mean(axis=0)
        band_imp_norm = band_imp_mean / band_imp_mean.sum()
        
        print(f"\n📸 Band Importance (Gradient-based):")
        for i, (band, imp) in enumerate(zip(BAND_NAMES, band_imp_norm)):
            print(f"   {band}: {imp*100:.1f}%")
        
        contrib_mean = {
            'visual': np.mean([c['visual_contribution'] for c in contributions_list]),
            'env': np.mean([c['env_contribution'] for c in contributions_list]),
        }
        print(f"\n🔄 Feature Contribution (Ablation):")
        print(f"   Visual-only: {contrib_mean['visual']*100:.1f}%")
        print(f"   Env-only: {contrib_mean['env']*100:.1f}%")
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 波段重要性对比
    x = np.arange(4)
    width = 0.35
    
    summer_imp = all_band_importance['USA-Summer'].mean(axis=0)
    summer_imp = summer_imp / summer_imp.sum()
    winter_imp = all_band_importance['USA-Winter'].mean(axis=0)
    winter_imp = winter_imp / winter_imp.sum()
    
    axes[0].bar(x - width/2, summer_imp * 100, width, label='USA-Summer', color='#2ecc71')
    axes[0].bar(x + width/2, winter_imp * 100, width, label='USA-Winter', color='#3498db')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(BAND_NAMES)
    axes[0].set_ylabel('Importance (%)')
    axes[0].set_title('Satellite Image Band Importance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 视觉 vs 环境贡献
    domains = list(all_contributions.keys())
    visual_contribs = [np.mean([c['visual_contribution'] for c in all_contributions[d]]) * 100 for d in domains]
    env_contribs = [np.mean([c['env_contribution'] for c in all_contributions[d]]) * 100 for d in domains]
    
    x = np.arange(len(domains))
    axes[1].bar(x - width/2, visual_contribs, width, label='Visual Features', color='#e74c3c')
    axes[1].bar(x + width/2, env_contribs, width, label='Environmental Features', color='#3498db')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(domains)
    axes[1].set_ylabel('Contribution to Prediction (%)')
    axes[1].set_title('Visual vs Environmental Feature Contribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'visual_importance.png', dpi=150)
    plt.savefig(figures_dir / 'visual_importance.pdf', dpi=300)
    print(f"\n✅ Saved: visual_importance.png/pdf")
    plt.close()
    
    # 保存结果
    results = {
        'band_importance': {
            name: {band: float(imp) for band, imp in zip(BAND_NAMES, 
                   all_band_importance[name].mean(axis=0) / all_band_importance[name].mean(axis=0).sum())}
            for name in all_band_importance
        },
        'feature_contribution': {
            name: {
                'visual': float(np.mean([c['visual_contribution'] for c in all_contributions[name]])),
                'env': float(np.mean([c['env_contribution'] for c in all_contributions[name]])),
            }
            for name in all_contributions
        }
    }
    
    with open(output_dir / 'visual_importance.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("VISUAL FEATURE ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
