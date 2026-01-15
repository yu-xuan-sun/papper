"""
HGCP + FDA 集成模型
在 HGCP (Hierarchical Geo-Contextual Prompts) 基础上增加 FDA (Frequency-Decoupled Domain Adaptation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class FrequencyDecomposition(nn.Module):
    """傅里叶频率分解模块"""
    
    def __init__(self, low_freq_ratio: float = 0.25, learnable: bool = True):
        super().__init__()
        self.low_freq_ratio = low_freq_ratio
        self.learnable = learnable
        
        if learnable:
            self.freq_threshold = nn.Parameter(torch.tensor(low_freq_ratio))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
        ratio = torch.sigmoid(self.freq_threshold) if self.learnable else self.low_freq_ratio
        center_h, center_w = H // 2, W // 2
        mask_h = int(H * ratio)
        mask_w = int(W * ratio)
        
        low_mask = torch.zeros(H, W, device=x.device)
        h_start = max(0, center_h - mask_h // 2)
        h_end = min(H, center_h + mask_h // 2)
        w_start = max(0, center_w - mask_w // 2)
        w_end = min(W, center_w + mask_w // 2)
        low_mask[h_start:h_end, w_start:w_end] = 1.0
        
        low_mask = low_mask.unsqueeze(0).unsqueeze(0)
        high_mask = 1.0 - low_mask
        
        low_fft = x_fft_shifted * low_mask
        high_fft = x_fft_shifted * high_mask
        
        low_fft_unshifted = torch.fft.ifftshift(low_fft, dim=(-2, -1))
        high_fft_unshifted = torch.fft.ifftshift(high_fft, dim=(-2, -1))
        
        low_freq = torch.fft.ifft2(low_fft_unshifted, norm='ortho').real
        high_freq = torch.fft.ifft2(high_fft_unshifted, norm='ortho').real
        
        return low_freq, high_freq


class FrequencyBranchEncoder(nn.Module):
    """频率分支编码器"""
    
    def __init__(self, in_channels: int = 3, hidden_dim: int = 128, out_dim: int = 768):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AdaptiveFrequencyGate(nn.Module):
    """环境引导的自适应频率门控"""
    
    def __init__(self, feat_dim: int = 768, env_dim: int = 27, init_low_bias: float = 0.6):
        super().__init__()
        
        self.gate_net = nn.Sequential(
            nn.Linear(feat_dim * 2 + env_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 2),
        )
        
        with torch.no_grad():
            self.gate_net[-1].bias[0] = math.log(init_low_bias / (1 - init_low_bias))
            self.gate_net[-1].bias[1] = math.log((1 - init_low_bias) / init_low_bias)
    
    def forward(self, low_feat, high_feat, env=None):
        if env is not None:
            gate_input = torch.cat([low_feat, high_feat, env], dim=-1)
        else:
            B = low_feat.size(0)
            dummy_env = torch.zeros(B, 27, device=low_feat.device)
            gate_input = torch.cat([low_feat, high_feat, dummy_env], dim=-1)
        
        gate_logits = self.gate_net(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        return gate_weights[:, 0:1], gate_weights[:, 1:2]


class FDAModule(nn.Module):
    """FDA 模块"""
    
    def __init__(
        self,
        in_channels: int = 4,
        feat_dim: int = 768,
        env_dim: int = 27,
        hidden_dim: int = 128,
        low_freq_ratio: float = 0.25,
        learnable_freq: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.freq_decomp = FrequencyDecomposition(low_freq_ratio, learnable_freq)
        self.low_encoder = FrequencyBranchEncoder(in_channels, hidden_dim, feat_dim)
        self.high_encoder = FrequencyBranchEncoder(in_channels, hidden_dim, feat_dim)
        self.gate = AdaptiveFrequencyGate(feat_dim, env_dim)
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
        )
        
        total = sum(p.numel() for p in self.parameters())
        print(f"🌊 FDA Module: feat_dim={feat_dim}, params={total/1e6:.2f}M")
    
    def forward(self, img, env=None):
        low_freq, high_freq = self.freq_decomp(img)
        low_feat = self.low_encoder(low_freq)
        high_feat = self.high_encoder(high_freq)
        low_weight, high_weight = self.gate(low_feat, high_feat, env)
        fda_feat = low_weight * low_feat + high_weight * high_feat
        fda_feat = self.fusion(fda_feat)
        
        info = {'low_weight': low_weight.mean().item(), 'high_weight': high_weight.mean().item()}
        return fda_feat, info


class Dinov2HGCP_FDA(nn.Module):
    """HGCP + FDA 集成模型"""
    
    def __init__(
        self,
        hgcp_model,
        env_dim: int = 27,
        use_fda: bool = True,
        fda_hidden_dim: int = 128,
        fda_low_freq_ratio: float = 0.25,
        fda_learnable_freq: bool = True,
        fda_fusion_weight: float = 0.3,
        fda_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hgcp_model = hgcp_model
        self.use_fda = use_fda
        self.fda_fusion_weight = fda_fusion_weight
        
        self.embed_dim = hgcp_model.embed_dim
        self.in_channels = hgcp_model.base_model.in_channels
        
        if use_fda:
            self.fda_module = FDAModule(
                in_channels=self.in_channels,
                feat_dim=self.embed_dim,
                env_dim=env_dim,
                hidden_dim=fda_hidden_dim,
                low_freq_ratio=fda_low_freq_ratio,
                learnable_freq=fda_learnable_freq,
                dropout=fda_dropout,
            )
            
            self.fda_fusion = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU(),
                nn.Dropout(fda_dropout),
            )
        
        self._print_params()
    
    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        hgcp_params = sum(p.numel() for p in self.hgcp_model.parameters())
        fda_params = sum(p.numel() for p in self.fda_module.parameters()) if self.use_fda else 0
        
        print(f"📊 HGCP+FDA Model Summary:")
        print(f"   HGCP: {hgcp_params/1e6:.2f}M")
        print(f"   FDA:  {fda_params/1e6:.2f}M")
        print(f"   Total: {total/1e6:.2f}M (Trainable: {trainable/1e6:.2f}M)")
    
    def forward(self, img, env=None):
        batch_size = img.size(0)
        
        # 通道适配
        if self.hgcp_model.base_model.channel_adapter is not None:
            x = self.hgcp_model.base_model.channel_adapter(img)
        elif self.hgcp_model.base_model.in_channels > 3:
            x = img[:, :3, :, :]
        else:
            x = img
        
        # Patch embedding
        x = self.hgcp_model.base_model.dino.patch_embed(x)
        
        if self.hgcp_model.base_model.dino.cls_token is not None:
            cls_token = self.hgcp_model.base_model.dino.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        x = x + self.hgcp_model.base_model.dino.pos_embed
        x = self.hgcp_model.base_model.dino.pos_drop(x)
        
        if env is not None:
            env = torch.nan_to_num(env, nan=0.0)
        
        prompt_len = self.hgcp_model.prompt_len
        for layer_idx, block in enumerate(self.hgcp_model.base_model.dino.blocks):
            if self.hgcp_model.use_hgcp and env is not None:
                dynamic_prompts, _ = self.hgcp_model.hgcp(env, layer_idx)
                prompts_to_use = dynamic_prompts
            else:
                if self.hgcp_model.base_model.adapted_encoder.use_layer_specific_prompts:
                    prompts_to_use = self.hgcp_model.base_model.adapted_encoder.prompts[layer_idx](batch_size)
                else:
                    prompts_to_use = self.hgcp_model.base_model.adapted_encoder.shared_prompts(batch_size)
            
            x_with_prompts = torch.cat([x, prompts_to_use], dim=1)
            x_with_prompts = block(x_with_prompts)
            x = x_with_prompts[:, :-prompt_len, :]
        
        x = self.hgcp_model.base_model.dino.norm(x)
        cls_features = x[:, 0]
        
        # FDA 特征
        if self.use_fda:
            fda_feat, _ = self.fda_module(img, env)
            combined_feat = torch.cat([cls_features, fda_feat], dim=-1)
            fused_feat = self.fda_fusion(combined_feat)
            cls_features = (1 - self.fda_fusion_weight) * cls_features + self.fda_fusion_weight * fused_feat
        
        # 环境融合
        if self.hgcp_model.base_model.use_env and env is not None:
            try:
                from src.models.adaptive_fusion import AdaptiveEnvironmentalFusion
                ADAPTIVE_FUSION_AVAILABLE = True
            except ImportError:
                ADAPTIVE_FUSION_AVAILABLE = False
            
            if self.hgcp_model.base_model.fusion_type == "adaptive_attention" and ADAPTIVE_FUSION_AVAILABLE:
                fused_feat = self.hgcp_model.base_model.fusion(cls_features, env)
            elif self.hgcp_model.base_model.fusion_type in ["cross_attention", "adaptive_attention"]:
                env_feat = self.hgcp_model.base_model.env_encoder(env)
                fused_feat = self.hgcp_model.base_model.fusion(cls_features, env_feat)
            elif self.hgcp_model.base_model.fusion_type == "concat":
                env_feat = self.hgcp_model.base_model.env_encoder(env)
                fused_feat = torch.cat([cls_features, env_feat], dim=1)
            else:
                fused_feat = cls_features
        else:
            fused_feat = cls_features
        
        logits = self.hgcp_model.base_model.classifier(fused_feat)
        return logits
    
    def is_backbone_frozen(self):
        if hasattr(self.hgcp_model.base_model, 'is_backbone_frozen'):
            return self.hgcp_model.base_model.is_backbone_frozen()
        return False


def create_dinov2_hgcp_fda_model(config):
    """创建 HGCP + FDA 集成模型"""
    from src.models.hierarchical_geo_prompt import create_dinov2_hgcp_model
    
    hgcp_model = create_dinov2_hgcp_model(config)
    
    module_cfg = config.experiment.module
    data_cfg = config.data
    
    env_dim = sum(data_cfg.env_var_sizes) if hasattr(data_cfg, 'env_var_sizes') else 27
    
    model = Dinov2HGCP_FDA(
        hgcp_model=hgcp_model,
        env_dim=env_dim,
        use_fda=getattr(module_cfg, 'use_fda', True),
        fda_hidden_dim=getattr(module_cfg, 'fda_hidden_dim', 128),
        fda_low_freq_ratio=getattr(module_cfg, 'fda_low_freq_ratio', 0.25),
        fda_learnable_freq=getattr(module_cfg, 'fda_learnable_freq', True),
        fda_fusion_weight=getattr(module_cfg, 'fda_fusion_weight', 0.3),
        fda_dropout=getattr(module_cfg, 'fda_dropout', 0.1),
    )
    
    print("✅ Created HGCP + FDA integrated model")
    return model
