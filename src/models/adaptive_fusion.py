"""
Adaptive Environmental Fusion (AEF) - V2
支持可配置的环境编码器层数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveEnvironmentalFusion(nn.Module):
    """自适应环境特征融合模块 - 修复版 (Fix Double Sigmoid)"""
    
    def __init__(self, img_dim=768, env_dim=27, hidden_dim=2048, num_heads=8, 
                 dropout=0.2, num_layers=3):
        super().__init__()
        
        self.img_dim = img_dim
        self.env_dim = env_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input normalization
        self.env_input_norm = nn.LayerNorm(env_dim)
        self.img_input_norm = nn.LayerNorm(img_dim)
        
        # 环境编码器
        self.env_encoder = self._build_env_encoder(env_dim, hidden_dim, img_dim, num_layers, dropout)
        
        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 🔴 核心修改：门控网络移除最后的 Sigmoid
        # 现在的输出是 Logits (-inf 到 +inf)，而不是 (0, 1)
        self.gate = nn.Sequential(
            nn.Linear(img_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, img_dim),
            nn.LayerNorm(img_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(img_dim, 1)
            # [Deleted] nn.Sigmoid()  <-- 移除了这行
        )
        
        # Temperature (可学习的温度参数)
        self.temperature = nn.Parameter(torch.tensor(0.1))
        
        self.norm1 = nn.LayerNorm(img_dim)
        self.norm2 = nn.LayerNorm(img_dim)
        
        self._init_weights()
        
        print(f"  AdaptiveEnvironmentalFusion (Fixed): env_dim={env_dim}, hidden_dim={hidden_dim}, "
              f"num_layers={num_layers}, num_heads={num_heads}")
    
    def _build_env_encoder(self, env_dim, hidden_dim, output_dim, num_layers, dropout):
        """构建可配置层数的环境编码器 (保持不变)"""
        layers = []
        if num_layers == 1:
            layers.extend([
                nn.Linear(env_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            ])
        elif num_layers == 2:
            layers.extend([
                nn.Linear(env_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
            ])
        else:
            layers.extend([
                nn.Linear(env_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            for _ in range(num_layers - 2):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
            layers.extend([
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
            ])
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, img_features, env_features, return_alpha=False):
        """前向传播"""
        # Normalize inputs
        env_features = self.env_input_norm(env_features)
        img_features = self.img_input_norm(img_features)
        
        # 编码环境特征
        env_encoded = self.env_encoder(env_features)
        
        # Cross-attention
        img_feat_expanded = img_features.unsqueeze(1)
        env_feat_expanded = env_encoded.unsqueeze(1)
        
        attn_out, _ = self.cross_attn(
            query=img_feat_expanded,
            key=env_feat_expanded,
            value=env_feat_expanded
        )
        
        attn_out = attn_out.squeeze(1)
        attn_out = self.norm1(attn_out + img_features)
        
        # 门控机制
        concat_feat = torch.cat([img_features, env_encoded], dim=1)
        gate_logits = self.gate(concat_feat)  # 现在是 Logits
        
        # 温度缩放 + sigmoid
        # 逻辑修复：现在 gate_logits 可以是负数
        # 如果 gate_logits = -5, temp = 0.1 -> -50 -> sigmoid ≈ 0.0 (完全关闭)
        # 如果 gate_logits = +5, temp = 0.1 -> +50 -> sigmoid ≈ 1.0 (完全开启)
        alpha = gate_logits / torch.clamp(self.temperature.abs(), min=0.01)
        alpha = torch.sigmoid(alpha)
        
        # 自适应融合
        fused = alpha * attn_out + (1 - alpha) * img_features
        fused = self.norm2(fused)
        
        if return_alpha:
            return fused, gate_logits # 返回 logits 以便观察原始值
        return fused
    
    def get_gate_value(self, img_features, env_features):
        """获取门控网络的原始输出 Logits"""
        env_features = self.env_input_norm(env_features)
        img_features = self.img_input_norm(img_features)
        env_encoded = self.env_encoder(env_features)
        concat_feat = torch.cat([img_features, env_encoded], dim=1)
        return self.gate(concat_feat)  # 返回 Logits


class SimpleConcatFusion(nn.Module):
    """简单拼接融合 - 用于对比实验"""
    
    def __init__(self, img_dim=768, env_dim=27, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        
        # 环境编码器
        layers = [nn.Linear(env_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, img_dim))
        
        self.env_encoder = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(img_dim * 2)
        
    def forward(self, img_features, env_features, return_alpha=False):
        env_encoded = self.env_encoder(env_features)
        fused = torch.cat([img_features, env_encoded], dim=-1)
        fused = self.norm(fused)
        if return_alpha:
            return fused, None
        return fused


class NoEnvFusion(nn.Module):
    """不使用环境特征 - 用于消融实验"""
    
    def __init__(self, img_dim=768, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(img_dim)
        
    def forward(self, img_features, env_features=None, return_alpha=False):
        out = self.norm(img_features)
        if return_alpha:
            return out, torch.zeros(img_features.size(0), 1, device=img_features.device)
        return out
