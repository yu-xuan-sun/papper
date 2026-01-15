#!/usr/bin/env python3
"""
快速门控值分析脚本
直接从checkpoint加载并分析gate_value
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
import seaborn as sns

sys.path.insert(0, '/sunyuxuan/satbird')

from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt

# 配置
DOMAINS = {
    'USA-Summer': {
        'checkpoint': 'runs/test_seed42_20251219-153117/checkpoints/best-77-0.0517.ckpt',
        'data_dir': 'USA_summer',
        'num_classes': 624,
    },
    'USA-Winter': {
        'checkpoint': 'runs/best winter/checkpoints/best-80-0.0479.ckpt',
        'data_dir': 'USA_winter',
        'num_classes': 670,
    },
    'Kenya-Transfer': {
        'checkpoint': 'runs/transfer_usa_to_kenya_freeze_seed42_20251202-025823/checkpoints/best-104-0.0694.ckpt',
        'data_dir': 'kenya',
        'num_classes': 673,
    }
}

ENV_NAMES = [
    'BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9',
    'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19',
    'Aspect', 'OrganicC', 'Slope', 'TPI', 'Clay', 'BulkDens', 'CEC', 'Elev'
]


def load_sample(data_dir, hotspot_id):
    """加载单个样本"""
    env_path = os.path.join(data_dir, 'environmental', f'{hotspot_id}.npy')
    img_path = os.path.join(data_dir, 'images', f'{hotspot_id}.tif')
    
    if not os.path.exists(env_path) or not os.path.exists(img_path):
        return None
    
    env_data = np.load(env_path).astype(np.float32)
    
    with rasterio.open(img_path) as src:
        img = src.read().astype(np.float32)
    
    # 归一化图像
    img = np.clip(img / 10000.0, 0, 1)
    
    if img.shape[1] != 224 or img.shape[2] != 224:
        from torchvision.transforms.functional import resize
        img = resize(torch.from_numpy(img), [224, 224]).numpy()
    
    # 环境变量 flatten
    if len(env_data.shape) > 1:
        env_data = env_data.mean(axis=(1, 2)) if len(env_data.shape) == 3 else env_data.flatten()
    
    return {
        'image': torch.from_numpy(img).unsqueeze(0),
        'env': torch.from_numpy(env_data[:27]).unsqueeze(0)  # 取前27维
    }


def load_model(ckpt_path, num_classes, device='cuda'):
    """加载模型"""
    print(f"Loading: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 分析state_dict确定参数
    env_dim = 27
    for k in state_dict:
        if 'fusion.env_encoder.0.weight' in k:
            env_dim = state_dict[k].shape[1]
            break
    
    # 创建模型
    model = Dinov2AdapterPrompt(
        num_classes=num_classes,
        dino_model_name='vit_base_patch14_dinov2.lvd142m',
        pretrained_path='checkpoints/dinov2_vitb14_pretrain.pth',
        prompt_len=40,
        bottleneck_dim=96,
        env_input_dim=env_dim,
        env_hidden_dim=2048,
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


def analyze_domain(name, config, max_samples=None, device='cuda'):
    """分析单个域"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")
    
    if not os.path.exists(config['checkpoint']):
        print(f"Checkpoint not found: {config['checkpoint']}")
        return None
    
    # 加载模型
    model = load_model(config['checkpoint'], config['num_classes'], device)
    
    # 加载测试集
    split_file = os.path.join(config['data_dir'], 'test_split.csv')
    df = pd.read_csv(split_file)
    hotspot_ids = df['hotspot_id'].tolist()
    
    if max_samples:
        hotspot_ids = hotspot_ids[:max_samples]
    
    print(f"Total samples: {len(hotspot_ids)}")
    
    # 分析
    gate_values = []
    final_alphas = []
    gradients = []
    
    for hid in tqdm(hotspot_ids, desc=f"Analyzing {name}"):
        sample = load_sample(config['data_dir'], hid)
        if sample is None:
            continue
        
        try:
            result = extract_gate_and_gradient(model, sample['image'], sample['env'], device)
            gate_values.append(result['gate_value'])
            final_alphas.append(result['final_alpha'])
            gradients.append(result['env_gradient'])
        except Exception as e:
            pass
    
    gate_values = np.array(gate_values)
    final_alphas = np.array(final_alphas)
    gradients = np.array(gradients) if gradients else np.array([])
    
    print(f"\n--- {name} Statistics ---")
    print(f"Gate Value: mean={gate_values.mean():.4f}, std={gate_values.std():.4f}, "
          f"min={gate_values.min():.4f}, max={gate_values.max():.4f}")
    print(f"Final Alpha: mean={final_alphas.mean():.4f}, std={final_alphas.std():.4f}")
    
    return {
        'gate_value': gate_values,
        'final_alpha': final_alphas,
        'env_gradient': gradients
    }


def plot_results(all_results, output_dir):
    """绘制结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 门控值对比
    fig, axes = plt.subplots(len(all_results), 2, figsize=(14, 5 * len(all_results)))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, (name, results) in enumerate(all_results.items()):
        gate = results['gate_value']
        alpha = results['final_alpha']
        
        axes[idx, 0].hist(gate, bins=50, color=colors[idx % 3], alpha=0.7, edgecolor='black')
        axes[idx, 0].axvline(gate.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {gate.mean():.4f}')
        axes[idx, 0].axvline(0.5, color='green', linestyle=':', linewidth=2, label='Balanced (0.5)')
        axes[idx, 0].set_xlabel('Gate Value (First Sigmoid)')
        axes[idx, 0].set_title(f'{name}: Gate Network Output\n(Mean={gate.mean():.4f}, Std={gate.std():.4f})')
        axes[idx, 0].legend()
        axes[idx, 0].set_xlim(0, 1)
        
        axes[idx, 1].hist(alpha, bins=50, color=colors[idx % 3], alpha=0.7, edgecolor='black')
        axes[idx, 1].axvline(alpha.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {alpha.mean():.4f}')
        axes[idx, 1].set_xlabel('Final Alpha (After Temp Scaling)')
        axes[idx, 1].set_title(f'{name}: Final Alpha\n(Mean={alpha.mean():.4f}, Std={alpha.std():.4f})')
        axes[idx, 1].legend()
        axes[idx, 1].set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gate_vs_alpha.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'gate_vs_alpha.png'}")
    plt.close()
    
    # 环境变量重要性对比
    fig, axes = plt.subplots(1, len(all_results), figsize=(7 * len(all_results), 10))
    if len(all_results) == 1:
        axes = [axes]
    
    for idx, (name, results) in enumerate(all_results.items()):
        if len(results['env_gradient']) == 0:
            continue
        
        grad = results['env_gradient']
        mean_abs = np.abs(grad).mean(axis=0)
        
        # 排序
        sorted_idx = np.argsort(mean_abs)[::-1]
        top_k = min(15, len(mean_abs))
        
        var_names = [ENV_NAMES[i] if i < len(ENV_NAMES) else f'V{i}' for i in sorted_idx[:top_k]]
        values = mean_abs[sorted_idx[:top_k]]
        
        y_pos = np.arange(top_k)
        axes[idx].barh(y_pos, values, color=colors[idx % 3], alpha=0.8, edgecolor='black')
        axes[idx].set_yticks(y_pos)
        axes[idx].set_yticklabels(var_names)
        axes[idx].set_xlabel('Mean |Gradient|')
        axes[idx].set_title(f'{name}\nTop-{top_k} Environment Variables')
        axes[idx].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'env_importance.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'env_importance.png'}")
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    output_dir = Path('outputs/interpretability_v3')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for name, config in DOMAINS.items():
        results = analyze_domain(name, config, max_samples=None, device=device)  # 全样本
        if results:
            all_results[name] = results
            
            # 保存数据
            key = name.lower().replace('-', '_')
            np.save(output_dir / f'{key}_gate_values.npy', results['gate_value'])
            np.save(output_dir / f'{key}_final_alphas.npy', results['final_alpha'])
            if len(results['env_gradient']) > 0:
                np.save(output_dir / f'{key}_env_gradients.npy', results['env_gradient'])
    
    if all_results:
        plot_results(all_results, output_dir / 'figures')
        
        # 保存摘要
        summary = {}
        for name, results in all_results.items():
            gate = results['gate_value']
            summary[name] = {
                'n_samples': len(gate),
                'gate_mean': float(gate.mean()),
                'gate_std': float(gate.std()),
                'gate_min': float(gate.min()),
                'gate_max': float(gate.max()),
            }
        
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Analysis Complete!")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
