#!/usr/bin/env python3
"""
Kenya Transfer专用分析脚本 (v3 - 修复NaN问题)
Kenya模型使用不同配置: hidden_dim=512, env_dim=19 (仅bioclim)
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

# Kenya配置
CHECKPOINT = 'runs/transfer_usa_to_kenya_freeze_seed42_20251202-025823/checkpoints/best-104-0.0694.ckpt'
DATA_DIR = 'kenya'
NUM_CLASSES = 1054
ENV_HIDDEN_DIM = 512
ENV_DIM = 19

# 环境变量名称
ENV_NAMES = [
    'BIO1 (Annual Mean Temp)',
    'BIO2 (Mean Diurnal Range)',
    'BIO3 (Isothermality)',
    'BIO4 (Temp Seasonality)',
    'BIO5 (Max Temp Warmest)',
    'BIO6 (Min Temp Coldest)',
    'BIO7 (Temp Annual Range)',
    'BIO8 (Mean Temp Wettest)',
    'BIO9 (Mean Temp Driest)',
    'BIO10 (Mean Temp Warmest)',
    'BIO11 (Mean Temp Coldest)',
    'BIO12 (Annual Precip)',
    'BIO13 (Precip Wettest Month)',
    'BIO14 (Precip Driest Month)',
    'BIO15 (Precip Seasonality)',
    'BIO16 (Precip Wettest Quarter)',
    'BIO17 (Precip Driest Quarter)',
    'BIO18 (Precip Warmest Quarter)',
    'BIO19 (Precip Coldest Quarter)',
]


def load_sample(data_dir, hotspot_id, env_dim=19):
    """加载单个样本"""
    env_path = os.path.join(data_dir, 'environmental', f'{hotspot_id}.npy')
    img_path = os.path.join(data_dir, 'images', f'{hotspot_id}.tif')
    
    if not os.path.exists(env_path) or not os.path.exists(img_path):
        return None
    
    env_data = np.load(env_path).astype(np.float32)
    
    # 处理3D环境数据 (19, H, W) -> (19,)
    if len(env_data.shape) == 3:
        env_data = np.nanmean(env_data, axis=(1, 2))
    elif len(env_data.shape) == 2:
        env_data = env_data.flatten()
    
    # 检查NaN
    if np.isnan(env_data).any():
        return None
    
    # 只取前env_dim维
    env_data = env_data[:env_dim]
    
    with rasterio.open(img_path) as src:
        img = src.read().astype(np.float32)
    
    # 归一化图像
    img = np.clip(img / 10000.0, 0, 1)
    
    if img.shape[1] != 224 or img.shape[2] != 224:
        from torchvision.transforms.functional import resize
        img = resize(torch.from_numpy(img), [224, 224]).numpy()
    
    return {
        'image': torch.from_numpy(img).unsqueeze(0),
        'env': torch.from_numpy(env_data).unsqueeze(0)
    }


def load_kenya_model(ckpt_path, device='cuda'):
    """加载Kenya模型"""
    print(f"Loading Kenya model: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 创建模型
    model = Dinov2AdapterPrompt(
        num_classes=NUM_CLASSES,
        dino_model_name='vit_base_patch14_dinov2.lvd142m',
        pretrained_path='checkpoints/dinov2_vitb14_pretrain.pth',
        prompt_len=40,
        bottleneck_dim=96,
        adapter_dropout=0.1,
        use_layer_specific_prompts=True,
        env_input_dim=ENV_DIM,
        env_hidden_dim=ENV_HIDDEN_DIM,
        env_num_layers=3,
        use_env=True,
        fusion_type='adaptive_attention',
        hidden_dims=[2048, 1024],
        dropout=0.15,
        use_channel_adapter=True,
        in_channels=4,
        freeze_backbone=True,
    )
    
    # 加载权重
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model


def extract_gate_and_gradient(model, image, env, device='cuda'):
    """提取门控值和环境梯度"""
    image = image.to(device)
    env = env.to(device)
    
    # 提取门控值
    with torch.no_grad():
        visual_feat = model.forward_visual_features(image)
        fusion = model.fusion
        
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
    
    # 计算梯度
    model.zero_grad()
    env_grad = env.clone().requires_grad_(True)
    logits = model(image, env_grad)
    target_score = logits[:, logits.argmax(dim=1)].sum()
    target_score.backward()
    gradient = env_grad.grad.detach()
    
    return {
        'gate_value': gate_value.cpu().numpy().flatten()[0],
        'final_alpha': final_alpha.cpu().numpy().flatten()[0],
        'env_gradient': gradient.cpu().numpy().flatten()
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    output_dir = Path('outputs/interpretability_v3')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}")
        return
    
    model = load_kenya_model(CHECKPOINT, device)
    
    split_file = os.path.join(DATA_DIR, 'test_split.csv')
    df = pd.read_csv(split_file)
    hotspot_ids = df['hotspot_id'].tolist()
    
    print(f"\nTotal Kenya test samples: {len(hotspot_ids)}")
    
    gate_values = []
    final_alphas = []
    gradients = []
    skipped = 0
    
    for hid in tqdm(hotspot_ids, desc="Analyzing Kenya-Transfer"):
        sample = load_sample(DATA_DIR, hid, env_dim=ENV_DIM)
        if sample is None:
            skipped += 1
            continue
        
        try:
            result = extract_gate_and_gradient(model, sample['image'], sample['env'], device)
            gate_values.append(result['gate_value'])
            final_alphas.append(result['final_alpha'])
            gradients.append(result['env_gradient'])
        except Exception as e:
            skipped += 1
            continue
    
    gate_values = np.array(gate_values)
    final_alphas = np.array(final_alphas)
    gradients = np.array(gradients)
    
    print(f"\n{'='*60}")
    print(f"Kenya-Transfer Analysis Results")
    print(f"{'='*60}")
    print(f"Samples analyzed: {len(gate_values)} (skipped {skipped})")
    print(f"Gate Value: mean={gate_values.mean():.4f}, std={gate_values.std():.4f}, "
          f"min={gate_values.min():.4f}, max={gate_values.max():.4f}")
    print(f"Final Alpha: mean={final_alphas.mean():.4f}, std={final_alphas.std():.4f}")
    
    # 保存数据
    np.save(output_dir / 'kenya_transfer_gate_values.npy', gate_values)
    np.save(output_dir / 'kenya_transfer_final_alphas.npy', final_alphas)
    np.save(output_dir / 'kenya_transfer_env_gradients.npy', gradients)
    print(f"\n✅ Saved: kenya_transfer_*.npy")
    
    # 环境变量重要性
    mean_abs_grad = np.abs(gradients).mean(axis=0)
    importance = mean_abs_grad / mean_abs_grad.sum()
    
    print(f"\n--- Top-5 Environmental Variables for Kenya ---")
    sorted_idx = np.argsort(importance)[::-1]
    for rank, idx in enumerate(sorted_idx[:5], 1):
        print(f"  {rank}. {ENV_NAMES[idx]}: {importance[idx]*100:.1f}%")
    
    # 绘制图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(gate_values, bins=50, density=True, alpha=0.7, color='orange', edgecolor='darkorange')
    axes[0].axvline(gate_values.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {gate_values.mean():.4f}')
    axes[0].set_xlabel('Gate Value (First Sigmoid Output)')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Kenya-Transfer: Gate Value Distribution (n={len(gate_values)})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    y_pos = np.arange(len(importance))
    colors = ['#e74c3c' if i in sorted_idx[:5] else '#3498db' for i in range(len(importance))]
    axes[1].barh(y_pos, importance, color=colors, alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f'BIO{i+1}' for i in range(len(importance))])
    axes[1].set_xlabel('Relative Importance')
    axes[1].set_title('Kenya-Transfer: Environmental Variable Importance')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / 'kenya_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(figures_dir / 'kenya_analysis.pdf', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: kenya_analysis.png/pdf")
    plt.close()
    
    # 更新summary
    kenya_summary = {
        'Kenya-Transfer': {
            'n_samples': int(len(gate_values)),
            'gate_mean': float(gate_values.mean()),
            'gate_std': float(gate_values.std()),
            'gate_min': float(gate_values.min()),
            'gate_max': float(gate_values.max()),
            'alpha_mean': float(final_alphas.mean()),
            'alpha_std': float(final_alphas.std()),
            'top5_env': {f'BIO{sorted_idx[i]+1}': float(importance[sorted_idx[i]]) for i in range(5)}
        }
    }
    
    existing_summary_path = output_dir / 'analysis_summary.json'
    if existing_summary_path.exists():
        with open(existing_summary_path, 'r') as f:
            existing = json.load(f)
        existing.update(kenya_summary)
        final_summary = existing
    else:
        final_summary = kenya_summary
    
    with open(existing_summary_path, 'w') as f:
        json.dump(final_summary, f, indent=2)
    print(f"✅ Updated: analysis_summary.json")
    
    print(f"\n{'='*60}")
    print(f"Kenya Analysis Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
