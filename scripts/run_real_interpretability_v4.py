#!/usr/bin/env python3
"""
真实数据可解释性分析脚本 - 完整版本，自动推断所有模型参数
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import traceback
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.interpretability import InterpretabilityAnalyzer

# BIOCLIM变量名称
BIOCLIM_NAMES = [
    'BIO1: Annual Mean Temperature',
    'BIO2: Mean Diurnal Range',
    'BIO3: Isothermality',
    'BIO4: Temperature Seasonality',
    'BIO5: Max Temp Warmest Month',
    'BIO6: Min Temp Coldest Month',
    'BIO7: Temperature Annual Range',
    'BIO8: Mean Temp Wettest Quarter',
    'BIO9: Mean Temp Driest Quarter',
    'BIO10: Mean Temp Warmest Quarter',
    'BIO11: Mean Temp Coldest Quarter',
    'BIO12: Annual Precipitation',
    'BIO13: Precip Wettest Month',
    'BIO14: Precip Driest Month',
    'BIO15: Precipitation Seasonality',
    'BIO16: Precip Wettest Quarter',
    'BIO17: Precip Driest Quarter',
    'BIO18: Precip Warmest Quarter',
    'BIO19: Precip Coldest Quarter',
]

PED_NAMES = [
    'PED1: Soil pH',
    'PED2: Organic Carbon',
    'PED3: Sand Content',
    'PED4: Clay Content',
    'PED5: Silt Content',
    'PED6: Bulk Density',
    'PED7: Cation Exchange',
    'PED8: Elevation',
]


def get_domain_config(domain: str):
    configs = {
        'USA-Summer': {
            'checkpoint': 'runs/best summer2/checkpoints/best-48-0.0516.ckpt',
            'data_dir': 'USA_summer',
            'num_species': 624,
            'env_dim': 27,
            'test_csv': 'test_split.csv'
        },
        'USA-Winter': {
            'checkpoint': 'runs/best winter/checkpoints/best-80-0.0479.ckpt',
            'data_dir': 'USA_winter',
            'num_species': 670,
            'env_dim': 27,
            'test_csv': 'test_split.csv'
        },
        'Kenya-Transfer': {
            'checkpoint': 'runs/transfer_usa_to_kenya_freeze_seed42_20251202-025823/checkpoints/best-104-0.0694.ckpt',
            'data_dir': 'kenya',
            'num_species': 1054,
            'env_dim': 19,
            'test_csv': 'test_split.csv'
        }
    }
    return configs.get(domain)


def infer_model_params_from_checkpoint(checkpoint_path: str):
    """从checkpoint推断所有模型参数"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    params = {}
    
    # 推断bottleneck_dim
    for k in state_dict.keys():
        if 'attn_adapter.down_project.weight' in k:
            params['bottleneck_dim'] = state_dict[k].shape[0]
            break
    
    # 推断prompt_len
    for k in state_dict.keys():
        if 'prompts.0.prompts' in k:
            params['prompt_len'] = state_dict[k].shape[1]
            break
    
    # 推断env_num_layers
    env_encoder_keys = [k for k in state_dict.keys() if 'fusion.env_encoder' in k and 'weight' in k]
    max_idx = 0
    for k in env_encoder_keys:
        parts = k.split('.')
        for i, p in enumerate(parts):
            if p == 'env_encoder':
                max_idx = max(max_idx, int(parts[i+1]))
    params['env_num_layers'] = 3 if max_idx <= 9 else 6
    
    # 推断env_hidden_dim (从gate.0.weight的输出维度)
    for k in state_dict.keys():
        if 'fusion.gate.0.weight' in k:
            params['env_hidden_dim'] = state_dict[k].shape[0]
            break
    
    # 检查是否使用channel_adapter
    params['use_channel_adapter'] = any('channel_adapter' in k for k in state_dict.keys())
    
    print(f"  Inferred: bottleneck_dim={params.get('bottleneck_dim')}, prompt_len={params.get('prompt_len')}")
    print(f"  env_num_layers={params.get('env_num_layers')}, env_hidden_dim={params.get('env_hidden_dim')}")
    print(f"  use_channel_adapter={params.get('use_channel_adapter')}")
    
    return params, checkpoint


def load_model_direct(checkpoint_path: str, num_species: int, env_dim: int, device: str):
    from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt
    
    params, checkpoint = infer_model_params_from_checkpoint(checkpoint_path)
    
    model = Dinov2AdapterPrompt(
        num_classes=num_species,
        pretrained_path='checkpoints/dinov2_vitb14_pretrain.pth',
        dino_model_name='vit_base_patch14_dinov2.lvd142m',
        use_env=True,
        env_input_dim=env_dim,
        env_hidden_dim=params.get('env_hidden_dim', 2048),
        env_num_layers=params.get('env_num_layers', 3),
        fusion_type='adaptive_attention',
        prompt_len=params.get('prompt_len', 40),
        bottleneck_dim=params.get('bottleneck_dim', 96),
        dropout=0.3,
        use_dropkey=True,
        dropkey_rate=0.15,
        use_channel_adapter=params.get('use_channel_adapter', True),
        in_channels=4 if params.get('use_channel_adapter') else 3,
        channel_adapter_type='learned',
        freeze_backbone=True
    )
    
    state_dict = checkpoint.get('state_dict', checkpoint)
    cleaned_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
    if not cleaned_state_dict:
        cleaned_state_dict = state_dict
    
    result = model.load_state_dict(cleaned_state_dict, strict=False)
    loaded_count = len(cleaned_state_dict) - len(result.missing_keys)
    print(f"  Loaded: {loaded_count}/{len(cleaned_state_dict)} params")
    if result.missing_keys:
        print(f"  Missing: {len(result.missing_keys)} keys")
    
    model.to(device)
    model.eval()
    return model, params


def load_image(img_path: str):
    """加载图像，支持tif和npy格式"""
    if img_path.endswith('.tif'):
        import rasterio
        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)
    else:
        img = np.load(img_path).astype(np.float32)
        if img.ndim == 3 and img.shape[0] > img.shape[2]:
            img = np.transpose(img, (2, 0, 1))
    return img


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return {
        'sat': torch.stack([b['sat'] for b in batch]),
        'env': torch.stack([b['env'] for b in batch]),
        'target': torch.stack([b['target'] for b in batch]),
        'hotspot_id': [b['hotspot_id'] for b in batch]
    }


def load_dataset_simple(config: dict, base_dir: str, use_channel_adapter: bool, max_samples: int = None):
    from torch.utils.data import Dataset, DataLoader
    from skimage.transform import resize
    
    data_dir = os.path.join(base_dir, config['data_dir'])
    test_csv = os.path.join(data_dir, config['test_csv'])
    
    df_test = pd.read_csv(test_csv)
    print(f"Test set: {len(df_test)} samples")
    
    in_channels = 4 if use_channel_adapter else 3
    
    class SimpleDataset(Dataset):
        def __init__(self, df, data_dir, env_dim, num_species, in_channels):
            self.df = df.reset_index(drop=True)
            self.data_dir = data_dir
            self.env_dim = env_dim
            self.num_species = num_species
            self.in_channels = in_channels
            self.images_dir = os.path.join(data_dir, 'images')
            self.env_dir = os.path.join(data_dir, 'environmental')
            self.targets_dir = os.path.join(data_dir, 'targets')
            
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            try:
                row = self.df.iloc[idx]
                hotspot_id = row['hotspot_id']
                
                img_path = None
                for ext in ['.tif', '.npy', '.png', '.jpg']:
                    candidate = os.path.join(self.images_dir, f'{hotspot_id}{ext}')
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                
                if img_path is None:
                    return None
                
                img = load_image(img_path)
                
                if self.in_channels == 4:
                    if img.shape[0] >= 4:
                        img_out = img[:4]
                    elif img.shape[0] == 3:
                        img_out = np.concatenate([img, img[1:2]], axis=0)
                    else:
                        img_out = np.repeat(img[:1], 4, axis=0)
                else:
                    if img.shape[0] >= 3:
                        img_out = img[:3]
                    else:
                        img_out = np.repeat(img[:1], 3, axis=0)
                
                img_hwc = img_out.transpose(1, 2, 0)
                img_resized = resize(img_hwc, (224, 224), preserve_range=True, anti_aliasing=True)
                img_resized = img_resized.transpose(2, 0, 1).astype(np.float32)
                
                if img_resized.max() > 1:
                    img_resized = img_resized / img_resized.max()
                
                env_data = []
                bioclim_path = os.path.join(self.env_dir, 'bioclim', f'{hotspot_id}.npy')
                if os.path.exists(bioclim_path):
                    bioclim = np.load(bioclim_path).astype(np.float32)
                    if bioclim.ndim > 1:
                        bioclim = bioclim.reshape(bioclim.shape[0], -1).mean(axis=1)
                    env_data.append(bioclim)
                
                if self.env_dim > 19:
                    ped_path = os.path.join(self.env_dir, 'ped', f'{hotspot_id}.npy')
                    if os.path.exists(ped_path):
                        ped = np.load(ped_path).astype(np.float32)
                        if ped.ndim > 1:
                            ped = ped.reshape(ped.shape[0], -1).mean(axis=1)
                        env_data.append(ped)
                
                if env_data:
                    env = np.concatenate(env_data)
                else:
                    env = np.zeros(self.env_dim, dtype=np.float32)
                
                if len(env) < self.env_dim:
                    env = np.pad(env, (0, self.env_dim - len(env)))
                elif len(env) > self.env_dim:
                    env = env[:self.env_dim]
                
                target_path = os.path.join(self.targets_dir, f'{hotspot_id}.npy')
                if os.path.exists(target_path):
                    target = np.load(target_path).astype(np.float32)
                else:
                    target = np.zeros(self.num_species, dtype=np.float32)
                
                return {
                    'sat': torch.from_numpy(img_resized),
                    'env': torch.from_numpy(env),
                    'target': torch.from_numpy(target),
                    'hotspot_id': hotspot_id
                }
            except Exception as e:
                return None
    
    if max_samples and max_samples < len(df_test):
        df_test = df_test.sample(n=max_samples, random_state=42)
        print(f"Sampled {max_samples} samples")
    
    dataset = SimpleDataset(df_test, data_dir, config['env_dim'], config['num_species'], in_channels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    return dataloader


def run_analysis(domain: str, max_samples: int = 500, output_dir: str = 'outputs/interpretability'):
    config = get_domain_config(domain)
    if config is None:
        raise ValueError(f"Unknown domain: {domain}")
    
    base_dir = str(project_root)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Running interpretability analysis for {domain}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    checkpoint_path = os.path.join(base_dir, config['checkpoint'])
    print(f"Loading model from: {checkpoint_path}")
    model, params = load_model_direct(checkpoint_path, config['num_species'], config['env_dim'], device)
    
    use_channel_adapter = params.get('use_channel_adapter', True)
    print(f"\nLoading data from: {config['data_dir']}")
    print(f"Input channels: {4 if use_channel_adapter else 3}")
    dataloader = load_dataset_simple(config, base_dir, use_channel_adapter, max_samples)
    
    analyzer = InterpretabilityAnalyzer(model, device=device)
    
    all_alphas = []
    all_env_grads = []
    sample_count = 0
    error_count = 0
    
    print(f"\nAnalyzing samples...")
    import warnings
    warnings.filterwarnings('ignore')
    
    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue
        try:
            images = batch['sat'].to(device)
            env = batch['env'].to(device)
            
            gate_info = analyzer.extract_gate_weights(images, env)
            if gate_info and 'alpha' in gate_info:
                alphas = gate_info['alpha'].cpu().numpy()
                for a in alphas.flatten():
                    all_alphas.append(float(a))
            
            env_grad = analyzer.compute_env_gradient(images, env)
            if env_grad is not None:
                grad_np = env_grad.abs().cpu().numpy()
                for g in grad_np:
                    all_env_grads.append(g)
            
            sample_count += images.shape[0]
            
            if (batch_idx + 1) % 5 == 0:
                alpha_mean = np.mean(all_alphas) if all_alphas else 0
                print(f"  Batch {batch_idx + 1}: {sample_count} samples, avg_alpha={alpha_mean:.4f}")
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"  Batch error: {str(e)[:200]}")
    
    print(f"\nTotal samples analyzed: {sample_count}")
    print(f"Total errors: {error_count}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备环境变量名称
    env_names = BIOCLIM_NAMES[:config['env_dim']]
    if config['env_dim'] > 19:
        env_names = BIOCLIM_NAMES + PED_NAMES[:config['env_dim']-19]
    
    results = {
        'domain': domain,
        'num_samples': sample_count,
        'env_dim': config['env_dim'],
        'env_names': env_names,
        'alpha_stats': {},
        'env_importance': {}
    }
    
    if all_alphas:
        all_alphas = np.array(all_alphas)
        results['alpha_stats'] = {
            'mean': float(np.mean(all_alphas)),
            'std': float(np.std(all_alphas)),
            'min': float(np.min(all_alphas)),
            'max': float(np.max(all_alphas)),
            'median': float(np.median(all_alphas)),
            'q25': float(np.percentile(all_alphas, 25)),
            'q75': float(np.percentile(all_alphas, 75))
        }
        
        print(f"\n{'='*60}")
        print(f"Results for {domain}:")
        print(f"  Alpha (env weight): {results['alpha_stats']['mean']:.4f} ± {results['alpha_stats']['std']:.4f}")
        print(f"  Alpha range: [{results['alpha_stats']['min']:.4f}, {results['alpha_stats']['max']:.4f}]")
        
        mean_alpha = results['alpha_stats']['mean']
        if mean_alpha > 0.7:
            print(f"  >> Environment-dominant: model relies more on env features")
        elif mean_alpha < 0.3:
            print(f"  >> Visual-dominant: model relies more on satellite images")
        else:
            print(f"  >> Balanced: model uses both modalities")
    
    if all_env_grads:
        env_grads = np.array(all_env_grads)
        mean_importance = np.mean(np.abs(env_grads), axis=0)
        mean_importance = mean_importance / (mean_importance.sum() + 1e-8)
        top_indices = np.argsort(mean_importance)[::-1][:5]
        
        results['env_importance'] = {
            'mean_importance': mean_importance.tolist(),
            'top_5_indices': top_indices.tolist(),
            'top_5_importance': mean_importance[top_indices].tolist(),
            'top_5_names': [env_names[i] if i < len(env_names) else f'Var_{i}' for i in top_indices]
        }
        print(f"\n  Top 5 most important environmental variables:")
        for i, idx in enumerate(top_indices):
            name = env_names[idx] if idx < len(env_names) else f'Var_{idx}'
            print(f"    {i+1}. [{idx:2d}] {name}: {mean_importance[idx]:.4f}")
    
    output_path = os.path.join(output_dir, f'{domain.replace("-", "_")}_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='USA-Summer', 
                        choices=['USA-Summer', 'USA-Winter', 'Kenya-Transfer', 'all'])
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--output_dir', type=str, default='outputs/interpretability')
    args = parser.parse_args()
    
    all_results = {}
    
    if args.domain == 'all':
        for domain in ['USA-Summer', 'USA-Winter', 'Kenya-Transfer']:
            try:
                result = run_analysis(domain, args.max_samples, args.output_dir)
                all_results[domain] = result
            except Exception as e:
                print(f"Error analyzing {domain}: {e}")
                traceback.print_exc()
    else:
        all_results[args.domain] = run_analysis(args.domain, args.max_samples, args.output_dir)
    
    # 保存汇总结果
    if len(all_results) > 1:
        summary_path = os.path.join(args.output_dir, 'cross_domain_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCross-domain summary saved to: {summary_path}")
