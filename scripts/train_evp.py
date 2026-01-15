#!/usr/bin/env python3
"""
EVP (Environment-aware Visual Prompt Tuning) 训练脚本
支持在原有DINOv2 Adapter+Prompt模型基础上添加EVP功能
"""

import os
import sys
sys.path.insert(0, '/sunyuxuan/satbird')

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import copy

# 导入基础模型组件
from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt
from src.models.env_aware_prompt import EnvironmentPromptGenerator


class EVPDinov2Model(nn.Module):
    """
    带EVP的DINOv2模型
    在原有Dinov2AdapterPrompt基础上，将静态Prompt替换为环境感知的动态Prompt
    """
    
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
        
        # 获取基础模型参数
        self.embed_dim = base_model.embed_dim
        self.prompt_len = base_model.prompt_len
        self.num_layers = len(base_model.dino.blocks)
        
        if use_evp:
            # 创建EVP生成器
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
            print(f"✨ EVP enabled: env_dim={env_dim}, prompt_len={self.prompt_len}")
        
        if freeze_base:
            # 冻结基础模型，只训练EVP
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("🔒 Base model frozen, only training EVP")
        
        self._print_params()
    
    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
    
    def forward(
        self, 
        img: torch.Tensor, 
        env: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            img: [B, C, H, W] 输入图像
            env: [B, env_dim] 环境特征
            
        Returns:
            logits: [B, num_classes] 分类输出
        """
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
        
        # 使用EVP或静态Prompt
        if self.use_evp and env is not None:
            # EVP: 根据环境特征动态生成Prompt
            for i, block in enumerate(self.base_model.dino.blocks):
                # 生成环境感知的prompts
                prompts, gate = self.evp_generator(env, layer_idx=i)
                
                # 添加prompts到序列
                x_with_prompts = torch.cat([x, prompts], dim=1)
                
                # 通过Transformer block
                x_with_prompts = block(x_with_prompts)
                
                # 移除prompts
                x = x_with_prompts[:, :-self.prompt_len, :]
        else:
            # 使用原始的静态Prompt (fallback)
            x = self.base_model.adapted_encoder(x)
        
        # Final normalization
        x = self.base_model.dino.norm(x)
        
        # 取CLS token
        visual_feat = x[:, 0]
        
        # 融合环境特征
        if self.base_model.use_env and env is not None:
            if self.base_model.fusion_type == "adaptive_attention":
                fused_feat = self.base_model.fusion(visual_feat, env)
            elif self.base_model.fusion_type == "cross_attention":
                env_feat = self.base_model.env_encoder(env)
                fused_feat = self.base_model.fusion(visual_feat, env_feat)
            elif self.base_model.fusion_type == "concat":
                env_feat = self.base_model.env_encoder(env)
                fused_feat = torch.cat([visual_feat, env_feat], dim=1)
            else:
                fused_feat = visual_feat
        else:
            fused_feat = visual_feat
        
        # 分类头
        logits = self.base_model.classifier(fused_feat)
        
        return logits
    
    def get_evp_gate_values(self, env: torch.Tensor) -> List[torch.Tensor]:
        """获取所有层的EVP门控值，用于可解释性分析"""
        if not self.use_evp:
            return []
        
        _, all_gates = self.evp_generator.get_all_prompts(env)
        return all_gates


def create_evp_model(
    checkpoint_path: str = None,
    num_classes: int = 624,
    env_dim: int = 27,
    in_channels: int = 4,
    use_evp: bool = True,
    freeze_base: bool = False,
    device: str = 'cuda'
) -> EVPDinov2Model:
    """
    创建EVP模型的便捷函数
    
    Args:
        checkpoint_path: 预训练模型路径 (可选)
        num_classes: 类别数
        env_dim: 环境特征维度
        in_channels: 输入通道数
        use_evp: 是否使用EVP
        freeze_base: 是否冻结基础模型
        device: 设备
        
    Returns:
        EVPDinov2Model
    """
    # 创建基础模型
    base_model = Dinov2AdapterPrompt(
        num_classes=num_classes,
        dino_model_name='vit_base_patch14_dinov2.lvd142m',
        pretrained_path='checkpoints/dinov2_vitb14_pretrain.pth',
        prompt_len=40,
        bottleneck_dim=96,
        adapter_layers=None,  # 所有层
        adapter_dropout=0.1,
        env_input_dim=env_dim,
        env_hidden_dim=2048,
        env_num_layers=3,
        use_env=True,
        fusion_type='adaptive_attention',
        hidden_dims=[2048, 1024],
        dropout=0.15,
        use_channel_adapter=True,
        in_channels=in_channels,
        freeze_backbone=True,
    )
    
    # 加载预训练权重 (如果有)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📥 Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in state_dict:
            # PyTorch Lightning格式
            state_dict = {k.replace('model.', ''): v for k, v in state_dict['state_dict'].items()}
        base_model.load_state_dict(state_dict, strict=False)
    
    # 创建EVP模型
    evp_model = EVPDinov2Model(
        base_model=base_model,
        env_dim=env_dim,
        use_evp=use_evp,
        evp_hidden_dim=512,
        evp_dropout=0.1,
        freeze_base=freeze_base
    )
    
    return evp_model.to(device)


def test_evp_model():
    """测试EVP模型"""
    print("="*60)
    print("Testing EVP Model")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 创建模型
    model = create_evp_model(
        checkpoint_path=None,  # 不加载预训练
        num_classes=624,
        env_dim=27,
        in_channels=4,
        use_evp=True,
        freeze_base=False,
        device=device
    )
    
    model.eval()
    
    # 测试前向传播
    batch_size = 2
    img = torch.randn(batch_size, 4, 224, 224).to(device)
    env = torch.randn(batch_size, 27).to(device)
    
    with torch.no_grad():
        output = model(img, env)
    
    print(f"\n✅ Forward pass successful!")
    print(f"   Input image shape: {img.shape}")
    print(f"   Input env shape: {env.shape}")
    print(f"   Output shape: {output.shape}")
    
    # 测试EVP门控值
    gate_values = model.get_evp_gate_values(env)
    print(f"\n📊 EVP Gate values:")
    for i, gate in enumerate(gate_values[:3]):
        print(f"   Layer {i}: {gate.mean().item():.4f}")
    print(f"   ... ({len(gate_values)} layers total)")
    
    # 测试梯度流
    model.train()
    output = model(img, env)
    loss = output.sum()
    loss.backward()
    
    evp_grad_norm = 0
    for name, param in model.named_parameters():
        if 'evp_generator' in name and param.grad is not None:
            evp_grad_norm += param.grad.norm().item()
    
    print(f"\n🔄 Gradient flow:")
    print(f"   EVP gradient norm: {evp_grad_norm:.4f}")
    
    print("\n" + "="*60)
    print("✅ All EVP Model tests passed!")
    print("="*60)
    
    return model


if __name__ == '__main__':
    test_evp_model()
