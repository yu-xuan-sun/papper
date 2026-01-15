#!/usr/bin/env python3
"""
EVP实验运行脚本 - 独立版本
直接加载数据并训练EVP模型
"""

import os
import sys
sys.path.insert(0, '/sunyuxuan/satbird')

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json

# 导入项目模块
from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt
from src.models.env_aware_prompt import EnvironmentPromptGenerator


class EVPDinov2Model(nn.Module):
    """带EVP的DINOv2模型"""
    
    def __init__(
        self,
        base_model: Dinov2AdapterPrompt,
        env_dim: int = 27,
        use_evp: bool = True,
        evp_hidden_dim: int = 512,
        evp_dropout: float = 0.1,
        freeze_base: bool = False
    ):
        super().__init__()
        
        self.base_model = base_model
        self.use_evp = use_evp
        self.env_dim = env_dim
        
        self.embed_dim = base_model.embed_dim
        self.prompt_len = base_model.prompt_len
        self.num_layers = len(base_model.dino.blocks)
        
        if use_evp:
            self.evp_generator = EnvironmentPromptGenerator(
                env_dim=env_dim,
                prompt_len=self.prompt_len,
                embed_dim=self.embed_dim,
                hidden_dim=evp_hidden_dim,
                num_layers=self.num_layers,
                use_layer_specific=True,
                use_residual=True,
                use_gating=True,
                dropout=evp_dropout
            )
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
    
    def forward(self, img: torch.Tensor, env: torch.Tensor = None) -> torch.Tensor:
        batch_size = img.size(0)
        
        # Channel adapter
        if self.base_model.channel_adapter is not None:
            x = self.base_model.channel_adapter(img)
        elif self.base_model.in_channels > 3:
            x = img[:, :3, :, :]
        else:
            x = img
        
        # Patch embedding
        x = self.base_model.dino.patch_embed(x)
        
        # Add CLS token
        if self.base_model.dino.cls_token is not None:
            cls_token = self.base_model.dino.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        x = self.base_model.dino.pos_drop(x)
        
        # EVP forward
        if self.use_evp and env is not None:
            for i, block in enumerate(self.base_model.dino.blocks):
                prompts, _ = self.evp_generator(env, layer_idx=i)
                x_with_prompts = torch.cat([x, prompts], dim=1)
                x_with_prompts = block(x_with_prompts)
                x = x_with_prompts[:, :-self.prompt_len, :]
        else:
            x = self.base_model.adapted_encoder(x)
        
        x = self.base_model.dino.norm(x)
        visual_feat = x[:, 0]
        
        # Fusion
        if self.base_model.use_env and env is not None:
            if self.base_model.fusion_type == "adaptive_attention":
                fused_feat = self.base_model.fusion(visual_feat, env)
            else:
                fused_feat = visual_feat
        else:
            fused_feat = visual_feat
        
        logits = self.base_model.classifier(fused_feat)
        return logits
    
    def get_evp_gates(self, env):
        if not self.use_evp:
            return []
        _, gates = self.evp_generator.get_all_prompts(env)
        return gates


def main():
    parser = argparse.ArgumentParser(description='EVP Experiment')
    parser.add_argument('--dataset', type=str, default='summer', 
                       choices=['summer', 'winter', 'kenya'])
    parser.add_argument('--use_evp', action='store_true', default=True)
    parser.add_argument('--freeze_base', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据集配置
    dataset_configs = {
        'summer': {'data_dir': 'USA_summer', 'num_classes': 624},
        'winter': {'data_dir': 'USA_winter', 'num_classes': 624},
        'kenya': {'data_dir': 'kenya', 'num_classes': 756}
    }
    
    config = dataset_configs[args.dataset]
    
    print(f"\n{'='*60}")
    print(f"EVP Experiment - {args.dataset.upper()}")
    print(f"{'='*60}")
    print(f"Use EVP: {args.use_evp}")
    print(f"Freeze base: {args.freeze_base}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # 创建基础模型
    print("\nCreating base model...")
    base_model = Dinov2AdapterPrompt(
        num_classes=config['num_classes'],
        dino_model_name='vit_base_patch14_dinov2.lvd142m',
        pretrained_path='checkpoints/dinov2_vitb14_pretrain.pth',
        prompt_len=40,
        bottleneck_dim=96,
        adapter_layers=None,
        adapter_dropout=0.1,
        env_input_dim=27,
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
    
    # 创建EVP模型
    print("Creating EVP model...")
    model = EVPDinov2Model(
        base_model=base_model,
        env_dim=27,
        use_evp=args.use_evp,
        evp_hidden_dim=512,
        evp_dropout=0.1,
        freeze_base=args.freeze_base
    ).to(device)
    
    # 加载checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)
    
    # 统计参数
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
    
    # 测试前向传播
    print("\nTesting forward pass...")
    model.eval()
    dummy_img = torch.randn(2, 4, 224, 224).to(device)
    dummy_env = torch.randn(2, 27).to(device)
    with torch.no_grad():
        output = model(dummy_img, dummy_env)
    print(f"✅ Forward pass successful! Output shape: {output.shape}")
    
    # 测试EVP gates
    gates = model.get_evp_gates(dummy_env)
    if gates:
        print(f"\nEVP gate values (all 12 layers):")
        for i, g in enumerate(gates):
            print(f"  Layer {i:2d}: {g.mean().item():.4f}")
    
    if args.test_only:
        print("\n✅ Test-only mode completed!")
        return
    
    # 使用合成数据进行快速训练测试
    print("\n" + "="*60)
    print("Running synthetic training test...")
    print("="*60)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # 分组参数：EVP使用更高学习率
    evp_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'evp_generator' in name:
                evp_params.append(param)
            else:
                other_params.append(param)
    
    print(f"\nParameter groups:")
    print(f"  EVP params: {sum(p.numel() for p in evp_params):,}")
    print(f"  Other params: {sum(p.numel() for p in other_params):,}")
    
    optimizer = torch.optim.AdamW([
        {'params': evp_params, 'lr': args.lr * 2, 'weight_decay': 0.01},  # EVP用更高学习率
        {'params': other_params, 'lr': args.lr, 'weight_decay': 0.05}
    ])
    
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    model.train()
    print("\nTraining iterations:")
    for i in range(10):
        # 合成数据
        img = torch.randn(args.batch_size, 4, 224, 224).to(device)
        env = torch.randn(args.batch_size, 27).to(device)
        target = torch.zeros(args.batch_size, config['num_classes']).to(device)
        # 随机生成一些正样本
        for j in range(args.batch_size):
            n_pos = torch.randint(5, 20, (1,)).item()
            pos_idx = torch.randint(0, config['num_classes'], (n_pos,))
            target[j, pos_idx] = 1.0
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(img, env)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(img, env)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # 计算TopK
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            k = int(target.sum(dim=1).mean().item())
            k = max(1, min(k, logits.size(1)))
            _, topk_indices = probs.topk(k, dim=1)
            topk_acc = target.gather(1, topk_indices).sum(dim=1).mean().item()
        
        print(f"  Iteration {i+1:2d}/10: loss = {loss.item():.4f}, topk_acc = {topk_acc:.4f}")
    
    # 查看训练后的EVP gates
    model.eval()
    with torch.no_grad():
        gates = model.get_evp_gates(dummy_env)
    
    print(f"\nEVP gate values after training:")
    for i, g in enumerate(gates):
        print(f"  Layer {i:2d}: {g.mean().item():.4f}")
    
    print("\n" + "="*60)
    print("✅ Synthetic training test passed!")
    print("="*60)
    
    # 保存模型
    save_dir = f"runs/evp_test_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存EVP模块参数
    evp_state = {k: v for k, v in model.state_dict().items() if 'evp_generator' in k}
    torch.save(evp_state, os.path.join(save_dir, 'evp_weights.pth'))
    print(f"\nEVP weights saved to {save_dir}/evp_weights.pth")
    
    # 保存完整模型
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    print(f"Full model saved to {save_dir}/model.pth")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. To train with real data, use:")
    print(f"   python train.py --config configs/SatBird-USA-{args.dataset}/evp_{args.dataset}.yaml")
    print("\n2. Or integrate EVP into existing trainer:")
    print("   - Modify src/trainer/trainer.py to use EVPDinov2Model")
    print("   - Add EVP configuration to your YAML config")


if __name__ == '__main__':
    main()
