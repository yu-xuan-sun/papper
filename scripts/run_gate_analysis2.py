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
from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt


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
        # 通道适配
        if hasattr(model, 'channel_adapter') and model.channel_adapter is not None:
            x = model.channel_adapter(image)
        else:
            x = image
        
        # 视觉特征
        visual_features = model.encoder(x)
        
        # 环境特征
        env_processed = model.env_encoder(env_data)
        
        # 融合模块
        if hasattr(model, 'cross_modal_fusion') and model.cross_modal_fusion is not None:
            # 获取融合后的特征和alpha
            fused, alpha = model.cross_modal_fusion(visual_features, env_processed, return_alpha=True)
            return {'alpha': alpha.cpu().item()}
        else:
            return {'alpha': None}


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
    
    num_classes = state_dict['classifier.2.weight'].shape[0]
    env_hidden_dim = state_dict['env_encoder.encoder.0.weight'].shape[0]
    print(f"  num_classes={num_classes}, env_hidden_dim={env_hidden_dim}")
    
    model = Dinov2AdapterPrompt(
        num_classes=num_classes,
        pretrained=False,
        model_size='base',
        use_adapter=True,
        use_prompt=True,
        adapter_dim=64,
        prompt_len=4,
        env_input_dim=27,
        env_hidden_dim=env_hidden_dim,
        use_env=True
    )
    
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"  Load: {len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected")
    
    model = model.to(device)
    model.eval()
    
    df = pd.read_csv(split_file)
    hotspot_ids = df['hotspot_id'].tolist()[:max_samples]
    
    alphas = []
    
    for hid in tqdm(hotspot_ids, desc="Analyzing"):
        sample = load_sample(data_dir, hid)
        if sample is None:
            continue
        try:
            info = extract_gate_info(model, sample['image'], sample['env_data'], device)
            if info['alpha'] is not None:
                alphas.append(info['alpha'])
        except Exception as e:
            print(f"\n  Error {hid}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print(f"\nAnalyzed: {len(alphas)} samples")
    
    if len(alphas) > 0:
        alphas = np.array(alphas)
        print(f"\nAlpha Statistics:")
        print(f"  Mean: {alphas.mean():.4f}, Std: {alphas.std():.4f}")
        print(f"  Range: [{alphas.min():.4f}, {alphas.max():.4f}]")
        
        return {
            'domain': name, 'samples': len(alphas),
            'alpha_mean': float(alphas.mean()), 'alpha_std': float(alphas.std()),
            'alpha_min': float(alphas.min()), 'alpha_max': float(alphas.max())
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
        print(f"  Alpha: {r['alpha_mean']:.4f} +/- {r['alpha_std']:.4f} [{r['alpha_min']:.4f}, {r['alpha_max']:.4f}]")


if __name__ == '__main__':
    main()
