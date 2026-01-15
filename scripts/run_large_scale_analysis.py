#!/usr/bin/env python3
"""
大规模可解释性分析脚本
- 使用更多样本（每域1000+）
- 使用修复后的模型（无双重sigmoid）
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

sys.path.insert(0, '/sunyuxuan/satbird')
from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt


# 环境变量名称
BIOCLIM_NAMES = [
    'BIO1: Annual Mean Temp', 'BIO2: Mean Diurnal Range', 'BIO3: Isothermality',
    'BIO4: Temp Seasonality', 'BIO5: Max Temp Warmest', 'BIO6: Min Temp Coldest',
    'BIO7: Temp Annual Range', 'BIO8: Mean Temp Wettest', 'BIO9: Mean Temp Driest',
    'BIO10: Mean Temp Warmest Q', 'BIO11: Mean Temp Coldest Q', 'BIO12: Annual Precip',
    'BIO13: Precip Wettest Month', 'BIO14: Precip Driest Month', 'BIO15: Precip Seasonality',
    'BIO16: Precip Wettest Q', 'BIO17: Precip Driest Q', 'BIO18: Precip Warmest Q',
    'BIO19: Precip Coldest Q'
]

PED_NAMES = [
    'PED1: Aspect', 'PED2: Organic Carbon', 'PED3: Slope', 'PED4: TPI',
    'PED5: Clay', 'PED6: Bulk Density', 'PED7: Cation Exchange', 'PED8: Elevation'
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
    
    img = np.clip(img / 10000.0, 0, 1)
    
    if img.shape[1] != 224 or img.shape[2] != 224:
        from torchvision.transforms.functional import resize
        img = resize(torch.from_numpy(img), [224, 224]).numpy()
    
    return {'image': torch.from_numpy(img), 'env_data': torch.from_numpy(env_data)}


def extract_gate_and_gradients(model, image, env_data, device):
    """提取门控值和环境变量梯度"""
    model.eval()
    
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    if len(env_data.shape) == 3:
        env_data = env_data.unsqueeze(0)
    
    image = image.to(device)
    env_data = env_data.to(device)
    env_data.requires_grad_(True)
    
    # 前向传播
    output = model(image, env_data)
    
    # 计算梯度
    target = output.sum()
    target.backward()
    
    env_grad = env_data.grad.detach()
    
    # 提取alpha (通过hook或直接计算)
    with torch.no_grad():
        if hasattr(model, 'channel_adapter') and model.channel_adapter is not None:
            x = model.channel_adapter(image)
        else:
            x = image
        
        visual_features = model.encoder(x)
        env_processed = model.env_encoder(env_data.detach())
        
        if hasattr(model, 'cross_modal_fusion') and model.cross_modal_fusion is not None:
            _, alpha = model.cross_modal_fusion(visual_features, env_processed, return_alpha=True)
            alpha_val = alpha.cpu().item()
        else:
            alpha_val = None
    
    # 环境变量重要性 = |gradient| * |value|
    env_importance = (env_grad.abs() * env_data.detach().abs()).mean(dim=(2, 3)).cpu().numpy()[0]
    
    return {
        'alpha': alpha_val,
        'env_importance': env_importance,
        'env_grad_norm': env_grad.abs().mean(dim=(2, 3)).cpu().numpy()[0]
    }


def analyze_domain(name, data_dir, ckpt_path, max_samples=1000, device='cuda'):
    """分析单个域"""
    print(f"\n{'='*70}")
    print(f"分析域: {name}")
    print(f"{'='*70}")
    
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint不存在: {ckpt_path}")
        return None
    
    split_file = os.path.join(data_dir, 'test_split.csv')
    if not os.path.exists(split_file):
        print(f"  Split文件不存在")
        return None
    
    # 加载checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 推断参数
    num_classes = state_dict['model.classifier.2.weight'].shape[0]
    env_hidden_dim = state_dict['model.fusion.env_encoder.0.weight'].shape[0]
    
    print(f"  num_classes={num_classes}, env_hidden_dim={env_hidden_dim}")
    
    # 检测环境维度
    env_dim = 19 if 'kenya' in data_dir.lower() else 27
    
    # 创建模型
    model = Dinov2AdapterPrompt(
        num_classes=num_classes,
        pretrained=False,
        model_size='base',
        use_adapter=True,
        use_prompt=True,
        adapter_dim=64,
        prompt_len=4,
        env_input_dim=env_dim,
        env_hidden_dim=env_hidden_dim,
        use_env=True
    )
    
    # 转换state_dict key
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"  加载: {len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected")
    
    model = model.to(device)
    model.eval()
    
    # 加载数据
    df = pd.read_csv(split_file)
    total_samples = len(df)
    hotspot_ids = df['hotspot_id'].tolist()[:max_samples]
    
    print(f"  分析样本: {len(hotspot_ids)}/{total_samples}")
    
    # 分析
    alphas = []
    all_importance = []
    
    for hid in tqdm(hotspot_ids, desc=f"分析 {name}"):
        sample = load_sample(data_dir, hid)
        if sample is None:
            continue
        
        try:
            info = extract_gate_and_gradients(model, sample['image'], sample['env_data'], device)
            if info['alpha'] is not None:
                alphas.append(info['alpha'])
            all_importance.append(info['env_importance'])
        except Exception as e:
            continue
    
    print(f"\n  成功分析: {len(all_importance)} 个样本")
    
    if len(all_importance) == 0:
        return None
    
    # 计算统计
    all_importance = np.array(all_importance)
    mean_importance = all_importance.mean(axis=0)
    
    # 归一化
    mean_importance = mean_importance / (mean_importance.sum() + 1e-8)
    
    # Top 5
    top5_idx = np.argsort(mean_importance)[::-1][:5].tolist()
    
    env_names = BIOCLIM_NAMES[:env_dim] if env_dim <= 19 else BIOCLIM_NAMES + PED_NAMES[:env_dim-19]
    
    # Alpha统计
    alpha_stats = {}
    if len(alphas) > 0:
        alphas = np.array(alphas)
        alpha_stats = {
            'mean': float(alphas.mean()),
            'std': float(alphas.std()),
            'min': float(alphas.min()),
            'max': float(alphas.max())
        }
        print(f"\n  Alpha统计: mean={alphas.mean():.4f}, std={alphas.std():.4f}, range=[{alphas.min():.4f}, {alphas.max():.4f}]")
    
    print(f"\n  Top 5 环境变量:")
    for i, idx in enumerate(top5_idx):
        print(f"    {i+1}. [{idx:2d}] {env_names[idx]}: {mean_importance[idx]:.4f}")
    
    return {
        'domain': name,
        'num_samples': len(all_importance),
        'total_samples': total_samples,
        'env_dim': env_dim,
        'env_names': env_names,
        'alpha_stats': alpha_stats,
        'env_importance': {
            'mean_importance': mean_importance.tolist(),
            'top_5_indices': top5_idx,
            'top_5_importance': [mean_importance[i] for i in top5_idx],
            'top_5_names': [env_names[i] for i in top5_idx]
        }
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 域配置
    configs = {
        'USA_Summer': {
            'data_dir': '/sunyuxuan/satbird/USA_summer',
            'checkpoint': '/sunyuxuan/satbird/runs/dinov2_v10_nir_enhanced_summer_seed42_20251201-153435/checkpoints/best-56-0.0522.ckpt'
        },
        'USA_Winter': {
            'data_dir': '/sunyuxuan/satbird/USA_winter',
            'checkpoint': '/sunyuxuan/satbird/runs/best winter/best_model.pt'
        },
        'Kenya_Transfer': {
            'data_dir': '/sunyuxuan/satbird/kenya',
            'checkpoint': '/sunyuxuan/satbird/runs/transfer_usa_to_kenya_linear_seed42_20251202-023844/best_model.pt'
        }
    }
    
    results = {}
    
    for name, cfg in configs.items():
        try:
            result = analyze_domain(
                name=name,
                data_dir=cfg['data_dir'],
                ckpt_path=cfg['checkpoint'],
                max_samples=2000,  # 增加到2000样本
                device=device
            )
            if result:
                results[name] = result
        except Exception as e:
            print(f"域 {name} 分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    out_dir = '/sunyuxuan/satbird/outputs/interpretability_v2'
    os.makedirs(out_dir, exist_ok=True)
    
    for name, result in results.items():
        with open(os.path.join(out_dir, f'{name}_analysis.json'), 'w') as f:
            json.dump(result, f, indent=2)
    
    # 汇总
    print("\n" + "="*70)
    print("分析汇总")
    print("="*70)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  样本: {result['num_samples']}/{result['total_samples']}")
        if result['alpha_stats']:
            print(f"  Alpha: {result['alpha_stats']['mean']:.4f} ± {result['alpha_stats']['std']:.4f}")
        print(f"  Top 3 变量: {', '.join(result['env_importance']['top_5_names'][:3])}")
    
    print(f"\n结果已保存到: {out_dir}")


if __name__ == '__main__':
    main()
