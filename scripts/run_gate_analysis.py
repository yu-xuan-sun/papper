#!/usr/bin/env python3
"""修复版可解释性分析 - 提取真实门控值"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

sys.path.insert(0, '/sunyuxuan/satbird')
from src.models.dinov2_adapter import DINOv2AdapterModel


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
    
    return {'image': torch.from_numpy(img), 'env_data': torch.from_numpy(env_data)}


def extract_gate_info(model, image, env_data, device):
    model.eval()
    
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    if len(env_data.shape) == 3:
        env_data = env_data.unsqueeze(0)
    
    image = image.to(device)
    env_data = env_data.to(device)
    
    with torch.no_grad():
        if hasattr(model, 'channel_adapter'):
            x = model.channel_adapter(image)
        else:
            x = image
        
        if model.use_adapter:
            features = model._forward_with_adapters(x)
        else:
            features = model.backbone.forward_features(x)
        
        if isinstance(features, dict):
            visual_features = features['x_norm_clstok']
        else:
            visual_features = features[:, 0]
        
        env_processed = model.env_fusion.env_encoder(env_data)
        gate_input = torch.cat([visual_features, env_processed], dim=-1)
        gate_raw = model.env_fusion.gate(gate_input)
        
        temp = model.env_fusion.temperature.item()
        alpha = torch.sigmoid(gate_raw / temp)
        
        return {'gate_raw': gate_raw.cpu().item(), 'alpha': alpha.cpu().item(), 'temperature': temp}


def analyze_domain(name, data_dir, ckpt_path, max_samples=100, device='cuda'):
    print(f"\n{'='*60}\nDomain: {name}\n{'='*60}")
    
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}")
        return None
    
    split_file = os.path.join(data_dir, 'test_split.csv')
    if not os.path.exists(split_file):
        print(f"  Split file not found")
        return None
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    
    num_classes = state_dict['classifier.weight'].shape[0]
    env_hidden_dim = state_dict['env_fusion.env_encoder.1.weight'].shape[0]
    print(f"  num_classes={num_classes}, env_hidden_dim={env_hidden_dim}")
    
    model = DINOv2AdapterModel(
        num_classes=num_classes, pretrained=False, use_adapter=True, use_prompt=True,
        use_env_data=True, model_size='base', freeze_backbone=True,
        bottleneck_dim=64, prompt_len=4, env_input_channels=27,
        env_hidden_dim=env_hidden_dim, env_num_layers=3
    )
    
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"  Load: {len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected")
    
    model = model.to(device)
    model.eval()
    
    df = pd.read_csv(split_file)
    hotspot_ids = df['hotspot_id'].tolist()[:max_samples]
    
    gate_raws, alphas = [], []
    
    for hid in tqdm(hotspot_ids, desc="Analyzing"):
        sample = load_sample(data_dir, hid)
        if sample is None:
            continue
        try:
            info = extract_gate_info(model, sample['image'], sample['env_data'], device)
            gate_raws.append(info['gate_raw'])
            alphas.append(info['alpha'])
        except Exception as e:
            print(f"\n  Error {hid}: {e}")
    
    print(f"\nAnalyzed: {len(gate_raws)} samples")
    
    if len(gate_raws) > 0:
        gate_raws = np.array(gate_raws)
        alphas = np.array(alphas)
        
        print(f"\nGate Raw (after 1st sigmoid, range 0-1):")
        print(f"  Mean: {gate_raws.mean():.4f}, Std: {gate_raws.std():.4f}")
        print(f"  Range: [{gate_raws.min():.4f}, {gate_raws.max():.4f}]")
        print(f"Alpha (final): {alphas.mean():.6f}, temp={info['temperature']:.4f}")
        
        return {
            'domain': name, 'samples': len(gate_raws),
            'gate_raw_mean': float(gate_raws.mean()), 'gate_raw_std': float(gate_raws.std()),
            'gate_raw_min': float(gate_raws.min()), 'gate_raw_max': float(gate_raws.max()),
            'alpha_mean': float(alphas.mean()), 'temperature': info['temperature']
        }
    return None


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    configs = {
        'USA_Summer': {'data_dir': '/sunyuxuan/satbird/USA_summer',
                       'checkpoint': '/sunyuxuan/satbird/runs/dinov2_v10_nir_enhanced_summer_seed42_20251201-153435/best_model.pt'},
        'USA_Winter': {'data_dir': '/sunyuxuan/satbird/USA_winter',
                       'checkpoint': '/sunyuxuan/satbird/runs/best winter/best_model.pt'},
        'Kenya_Transfer': {'data_dir': '/sunyuxuan/satbird/kenya',
                          'checkpoint': '/sunyuxuan/satbird/runs/transfer_usa_to_kenya_linear_seed42_20251202-023844/best_model.pt'}
    }
    
    results = {}
    for name, cfg in configs.items():
        result = analyze_domain(name, cfg['data_dir'], cfg['checkpoint'], max_samples=100, device=device)
        if result:
            results[name] = result
    
    out_dir = '/sunyuxuan/satbird/outputs/interpretability_fixed'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'gate_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  Gate Raw: {r['gate_raw_mean']:.4f} ± {r['gate_raw_std']:.4f} [{r['gate_raw_min']:.4f}, {r['gate_raw_max']:.4f}]")
        print(f"  Alpha: {r['alpha_mean']:.6f} (temp={r['temperature']:.4f})")


if __name__ == '__main__':
    main()
