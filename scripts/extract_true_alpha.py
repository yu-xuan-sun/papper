#!/usr/bin/env python3
"""
提取真正的门控值（在双重sigmoid问题之前的值）
对于已训练的模型，我们需要：
1. 获取gate网络的输出（已经经过第一次sigmoid，范围0-1）
2. 这个值才是真正反映模态权重的数值
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, '/sunyuxuan/satbird')

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


def infer_params(state_dict):
    """从state_dict推断模型参数"""
    params = {}
    
    for k in state_dict.keys():
        if 'attn_adapter.down_project.weight' in k:
            params['bottleneck_dim'] = state_dict[k].shape[0]
            break
    
    for k in state_dict.keys():
        if 'prompts.0.prompts' in k:
            params['prompt_len'] = state_dict[k].shape[1]
            break
    
    for k in state_dict.keys():
        if 'fusion.env_encoder.0.weight' in k:
            params['env_hidden_dim'] = state_dict[k].shape[0]
            env_input = state_dict[k].shape[1]
            break
    
    # 计算env_num_layers
    encoder_layers = [k for k in state_dict.keys() if 'fusion.env_encoder' in k and 'weight' in k and 'norm' not in k.lower()]
    params['env_num_layers'] = len([k for k in encoder_layers if '.weight' in k]) // 2
    
    # 检查channel_adapter
    params['use_channel_adapter'] = any('channel_adapter' in k for k in state_dict.keys())
    
    return params


def extract_true_gate(model, image, env_data, device):
    """
    提取真正的门控值（gate网络的原始输出）
    在旧代码中，gate网络最后有sigmoid，输出在0-1之间
    这个值才是真正有意义的模态权重指示器
    """
    model.eval()
    
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    if len(env_data.shape) == 3:
        env_data = env_data.unsqueeze(0)
    
    image = image.to(device)
    env_data = env_data.to(device)
    env_data.requires_grad_(True)
    
    with torch.no_grad():
        # 通道适配
        if hasattr(model, 'channel_adapter') and model.channel_adapter is not None:
            x = model.channel_adapter(image)
        else:
            x = image
        
        # 获取视觉特征
        visual_features = model.encoder(x)
        
        # 获取环境特征
        env_processed = model.env_encoder(env_data)
        
        # 获取融合模块
        fusion = model.cross_modal_fusion
        
        # 手动执行融合，获取gate的原始输出
        env_features = fusion.env_input_norm(env_processed)
        img_features = fusion.img_input_norm(visual_features)
        
        env_encoded = fusion.env_encoder(env_features)
        
        # 获取gate输出（这是第一次sigmoid后的值，范围0-1）
        concat_feat = torch.cat([img_features, env_encoded], dim=1)
        gate_raw = fusion.gate(concat_feat)  # 这包含了第一个sigmoid
        
        # 当前forward中的alpha计算（包含第二次sigmoid）
        temp = fusion.temperature.abs().clamp(min=0.01)
        alpha_final = torch.sigmoid(gate_raw / temp)
        
        return {
            'gate_raw': gate_raw.cpu().item(),  # 第一次sigmoid后的值（有意义）
            'alpha_final': alpha_final.cpu().item(),  # 最终alpha（接近1）
            'temperature': temp.cpu().item()
        }


def extract_with_gradients(model, image, env_data, device):
    """提取环境变量梯度重要性"""
    model.eval()
    
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    if len(env_data.shape) == 3:
        env_data = env_data.unsqueeze(0)
    
    image = image.to(device)
    env_data = env_data.to(device).requires_grad_(True)
    
    # 前向传播
    output = model(image, env_data)
    
    # 计算梯度
    target = output.abs().sum()
    target.backward()
    
    env_grad = env_data.grad.detach()
    
    # 环境变量重要性
    env_importance = (env_grad.abs() * env_data.detach().abs()).mean(dim=(2, 3)).cpu().numpy()[0]
    
    # 获取gate值
    gate_info = extract_true_gate(model, image, env_data.detach(), device)
    
    return {
        'gate_raw': gate_info['gate_raw'],
        'alpha_final': gate_info['alpha_final'],
        'temperature': gate_info['temperature'],
        'env_importance': env_importance
    }


def analyze_domain(name, data_dir, ckpt_path, max_samples=2000, device='cuda'):
    """分析单个域"""
    print(f"\n{'='*70}")
    print(f"分析域: {name}")
    print(f"{'='*70}")
    
    from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt
    
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint不存在: {ckpt_path}")
        return None
    
    # 加载checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 处理key
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k[6:] if k.startswith('model.') else k
        new_state_dict[new_k] = v
    
    # 推断参数
    params = infer_params(new_state_dict)
    
    # 获取num_classes
    for k, v in new_state_dict.items():
        if 'classifier' in k and 'weight' in k and v.dim() == 2:
            if v.shape[0] > 100:
                num_classes = v.shape[0]
    
    # 获取env_dim
    for k, v in new_state_dict.items():
        if 'fusion.env_encoder.0.weight' in k:
            env_dim = v.shape[1]
            break
    
    print(f"  num_classes={num_classes}, env_dim={env_dim}")
    print(f"  params: {params}")
    
    # 创建模型
    model = Dinov2AdapterPrompt(
        num_classes=num_classes,
        prompt_len=params.get('prompt_len', 40),
        bottleneck_dim=params.get('bottleneck_dim', 96),
        env_input_dim=env_dim,
        env_hidden_dim=params.get('env_hidden_dim', 2048),
        env_num_layers=params.get('env_num_layers', 3),
        use_env=True,
        use_channel_adapter=params.get('use_channel_adapter', True),
        in_channels=4
    )
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"  加载: {len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected")
    
    model = model.to(device)
    model.eval()
    
    # 加载数据
    split_file = os.path.join(data_dir, 'test_split.csv')
    df = pd.read_csv(split_file)
    total_samples = len(df)
    hotspot_ids = df['hotspot_id'].tolist()[:max_samples]
    
    print(f"  分析样本: {len(hotspot_ids)}/{total_samples}")
    
    # 分析
    gate_raws = []
    alpha_finals = []
    all_importance = []
    
    for hid in tqdm(hotspot_ids, desc=f"分析 {name}"):
        sample = load_sample(data_dir, hid)
        if sample is None:
            continue
        
        try:
            info = extract_with_gradients(model, sample['image'], sample['env_data'], device)
            gate_raws.append(info['gate_raw'])
            alpha_finals.append(info['alpha_final'])
            all_importance.append(info['env_importance'])
        except Exception as e:
            continue
        
        model.zero_grad()
    
    print(f"\n  成功分析: {len(all_importance)} 个样本")
    
    if len(all_importance) == 0:
        return None
    
    # 计算统计
    gate_raws = np.array(gate_raws)
    alpha_finals = np.array(alpha_finals)
    all_importance = np.array(all_importance)
    
    mean_importance = all_importance.mean(axis=0)
    mean_importance_norm = mean_importance / (mean_importance.sum() + 1e-8)
    
    top5_idx = np.argsort(mean_importance_norm)[::-1][:5].tolist()
    
    env_names = BIOCLIM_NAMES[:env_dim] if env_dim <= 19 else BIOCLIM_NAMES + PED_NAMES[:env_dim-19]
    
    print(f"\n  Gate统计 (真正的模态权重指示):")
    print(f"    Gate Raw: {gate_raws.mean():.4f} ± {gate_raws.std():.4f}")
    print(f"    Range: [{gate_raws.min():.4f}, {gate_raws.max():.4f}]")
    print(f"    (值>0.5表示偏向环境特征，<0.5表示偏向视觉特征)")
    
    print(f"\n  Alpha Final (受双重sigmoid影响，接近1):")
    print(f"    Mean: {alpha_finals.mean():.6f}")
    
    print(f"\n  Top 5 环境变量:")
    for i, idx in enumerate(top5_idx):
        print(f"    {i+1}. [{idx:2d}] {env_names[idx]}: {mean_importance_norm[idx]:.4f}")
    
    return {
        'domain': name,
        'num_samples': len(all_importance),
        'total_samples': total_samples,
        'env_dim': env_dim,
        'env_names': env_names,
        'gate_stats': {
            'mean': float(gate_raws.mean()),
            'std': float(gate_raws.std()),
            'min': float(gate_raws.min()),
            'max': float(gate_raws.max()),
            'interpretation': '>0.5 = env-dominant, <0.5 = visual-dominant'
        },
        'alpha_stats': {
            'mean': float(alpha_finals.mean()),
            'note': 'Affected by double-sigmoid bug, always near 1'
        },
        'env_importance': {
            'mean_importance': mean_importance_norm.tolist(),
            'top_5_indices': top5_idx,
            'top_5_importance': [float(mean_importance_norm[i]) for i in top5_idx],
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
            'checkpoint': '/sunyuxuan/satbird/runs/best winter/checkpoints/best-80-0.0479.ckpt'
        },
        'Kenya_Transfer': {
            'data_dir': '/sunyuxuan/satbird/kenya',
            'checkpoint': '/sunyuxuan/satbird/runs/transfer_usa_to_kenya_linear_seed42_20251202-023844/checkpoints/best-73-0.0694.ckpt'
        }
    }
    
    results = {}
    
    for name, cfg in configs.items():
        try:
            result = analyze_domain(
                name=name,
                data_dir=cfg['data_dir'],
                ckpt_path=cfg['checkpoint'],
                max_samples=2000,
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
    print("📊 分析汇总")
    print("="*70)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  样本: {result['num_samples']}/{result['total_samples']} ({result['num_samples']/result['total_samples']*100:.1f}%)")
        print(f"  Gate Raw: {result['gate_stats']['mean']:.4f} ± {result['gate_stats']['std']:.4f}")
        print(f"  Top 3: {', '.join(result['env_importance']['top_5_names'][:3])}")
    
    print(f"\n✅ 结果已保存到: {out_dir}")


if __name__ == '__main__':
    main()
