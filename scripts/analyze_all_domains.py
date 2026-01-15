"""
全样本可解释性分析 - 完整版
"""
import sys
sys.path.insert(0, '/sunyuxuan/satbird')

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

def infer_config_from_checkpoint(checkpoint_path):
    """从checkpoint推断模型配置"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            clean_state[k[6:]] = v
        else:
            clean_state[k] = v
    
    env_dim = clean_state.get('fusion.env_input_norm.weight', torch.zeros(27)).shape[0]
    hidden_dim = clean_state.get('fusion.env_encoder.0.weight', torch.zeros(512, 27)).shape[0]
    num_classes = clean_state.get('classifier.8.weight', torch.zeros(624, 1024)).shape[0]
    
    return {
        'env_dim': env_dim,
        'hidden_dim': hidden_dim,
        'num_classes': num_classes,
        'state_dict': clean_state
    }

def load_model(checkpoint_path, device):
    """加载模型"""
    from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt
    
    print(f"Loading: {checkpoint_path}")
    inferred = infer_config_from_checkpoint(checkpoint_path)
    print(f"  Inferred: env_dim={inferred['env_dim']}, hidden_dim={inferred['hidden_dim']}, num_classes={inferred['num_classes']}")
    
    model = Dinov2AdapterPrompt(
        num_classes=inferred['num_classes'],
        dino_model_name='vit_base_patch14_dinov2.lvd142m',
        pretrained_path='checkpoints/dinov2_vitb14_pretrain.pth',
        prompt_len=40,
        bottleneck_dim=96,
        adapter_layers=list(range(12)),
        env_input_dim=inferred['env_dim'],
        env_hidden_dim=inferred['hidden_dim'],
        use_env=True,
        fusion_type='adaptive_attention',
        hidden_dims=[2048, 1024],
        dropout=0.1,
        use_channel_adapter=True,
        in_channels=4,
        freeze_backbone=True,
    )
    
    model.load_state_dict(inferred['state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    return model

def load_sample(idx, data_dir, split_file, channels=4, env_vars=['bioclim']):
    """加载单个样本"""
    import pandas as pd
    import rasterio
    
    split_df = pd.read_csv(Path(data_dir) / split_file)
    row = split_df.iloc[idx]
    hotspot_id = row['hotspot_id']
    
    img_path = Path(data_dir) / 'images' / f'{hotspot_id}.tif'
    with rasterio.open(img_path) as src:
        image = src.read()
    
    image = image[:channels].astype(np.float32)
    
    means_path = Path(data_dir) / 'stats' / 'means_rgbnir.npy'
    stds_path = Path(data_dir) / 'stats' / 'stds_rgbnir.npy'
    
    if means_path.exists() and stds_path.exists():
        means = np.load(means_path)[:channels]
        stds = np.load(stds_path)[:channels]
        image = (image - means.reshape(-1, 1, 1)) / (stds.reshape(-1, 1, 1) + 1e-6)
    
    from torchvision import transforms
    image = torch.from_numpy(image)
    image = transforms.Resize((224, 224))(image)
    
    env_list = []
    for env_var in env_vars:
        env_path = Path(data_dir) / 'environmental' / f'{hotspot_id}_{env_var}.npy'
        if env_path.exists():
            env_data = np.load(env_path)
            env_list.append(env_data)
    
    env = np.concatenate(env_list) if env_list else np.zeros(27)
    
    env_means_path = Path(data_dir) / 'stats' / 'env_means.npy'
    env_stds_path = Path(data_dir) / 'stats' / 'env_stds.npy'
    
    if env_means_path.exists() and env_stds_path.exists():
        env_means = np.load(env_means_path)
        env_stds = np.load(env_stds_path)
        if len(env_means) == len(env):
            env = (env - env_means) / (env_stds + 1e-6)
    
    return image, torch.from_numpy(env.astype(np.float32))

def analyze_domain(name, config, device):
    """分析单个域"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")
    
    model = load_model(config['checkpoint'], device)
    
    import pandas as pd
    split_df = pd.read_csv(Path(config['data_dir']) / 'test_split.csv')
    num_samples = len(split_df)
    print(f"Total samples: {num_samples}")
    
    gate_values = []
    final_alphas = []
    env_gradients = []
    
    for idx in tqdm(range(num_samples), desc=f"Analyzing {name}"):
        try:
            image, env = load_sample(
                idx, config['data_dir'], 'test_split.csv',
                channels=4, env_vars=config.get('env_vars', ['bioclim'])
            )
            
            image = image.unsqueeze(0).to(device)
            env = env.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image, env)
                
                # 使用正确的方法提取visual features
                vis_features = model.forward_visual_features(image)
                
                gate_val = model.fusion.get_gate_value(vis_features, env).cpu().numpy().item()
                gate_values.append(gate_val)
                
                temp = model.fusion.temperature.abs().clamp(min=0.01).item()
                alpha = 1.0 / (1.0 + np.exp(-gate_val / temp))
                final_alphas.append(alpha)
            
            # 梯度分析
            model.zero_grad()
            env_grad = env.clone().detach().requires_grad_(True)
            
            vis_features = model.forward_visual_features(image)
            fused = model.fusion(vis_features, env_grad)
            logits = model.classifier(fused)
            loss = logits.sum()
            loss.backward()
            
            if env_grad.grad is not None:
                grad = env_grad.grad.abs().cpu().numpy()
                env_gradients.append(grad)
                
        except Exception as e:
            if idx < 5:
                print(f"Error at sample {idx}: {e}")
            continue
    
    gate_values = np.array(gate_values)
    final_alphas = np.array(final_alphas)
    env_gradients = np.array(env_gradients).squeeze() if env_gradients else np.array([])
    
    print(f"\n--- {name} Statistics ---")
    print(f"Gate Value: mean={gate_values.mean():.4f}, std={gate_values.std():.4f}, min={gate_values.min():.4f}, max={gate_values.max():.4f}")
    print(f"Final Alpha: mean={final_alphas.mean():.4f}, std={final_alphas.std():.4f}")
    
    return {
        'gate_values': gate_values,
        'final_alphas': final_alphas,
        'env_gradients': env_gradients
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    domains = {
        'Kenya-Transfer': {
            'checkpoint': 'runs/transfer_usa_to_kenya_freeze_seed42_20251202-025823/checkpoints/best-104-0.0694.ckpt',
            'data_dir': 'kenya',
            'env_vars': ['bioclim'],
        }
    }
    
    output_dir = Path('outputs/interpretability_v3')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, config in domains.items():
        results = analyze_domain(name, config, device)
        
        prefix = name.lower().replace('-', '_')
        np.save(output_dir / f'{prefix}_gate_values.npy', results['gate_values'])
        np.save(output_dir / f'{prefix}_final_alphas.npy', results['final_alphas'])
        np.save(output_dir / f'{prefix}_env_gradients.npy', results['env_gradients'])
    
    print(f"\n✅ Analysis complete!")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
