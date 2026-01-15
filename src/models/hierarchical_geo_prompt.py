"""
Hierarchical Geo-Contextual Prompts (HGCP)
改进版动态提示生成器

相比EVP的创新点:
1. 全层使用但差异化策略 (浅层纹理抑制/中层边缘增强/深层语义先验)
2. HyperNetwork替代线性投影 (更强非线性建模能力)
3. 层间信息传递 (hierarchical reasoning)
4. 通道级注意力门控 (而非标量门控)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import math


class HyperNetworkPromptGenerator(nn.Module):
    """
    HyperNetwork 动态生成 Prompt
    """
    
    def __init__(
        self,
        env_dim: int = 27,
        hidden_dim: int = 256,
        prompt_len: int = 10,
        embed_dim: int = 768,
        rank: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        self.rank = rank
        
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.low_rank_proj = nn.Linear(hidden_dim, prompt_len * rank)
        
        self.hyper_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rank * embed_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, env: torch.Tensor) -> torch.Tensor:
        B = env.size(0)
        h = self.env_encoder(env)
        low_rank = self.low_rank_proj(h)
        low_rank = low_rank.view(B, self.prompt_len, self.rank)
        weight = self.hyper_net(h)
        weight = weight.view(B, self.rank, self.embed_dim)
        prompts = torch.bmm(low_rank, weight)
        return prompts


class LayerSpecificStrategy(nn.Module):
    """层级特定策略"""
    
    def __init__(
        self,
        env_dim: int = 27,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        prompt_len: int = 10,
        num_layers: int = 12,
        rank: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        
        self.shallow_layers = list(range(0, 4))
        self.middle_layers = list(range(4, 8))
        self.deep_layers = list(range(8, 12))
        
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.shallow_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, prompt_len * embed_dim),
        )
        self.shallow_scale = nn.Parameter(torch.tensor(0.1))
        
        self.middle_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prompt_len * embed_dim),
        )
        self.middle_scale = nn.Parameter(torch.tensor(0.2))
        
        self.deep_generator = HyperNetworkPromptGenerator(
            env_dim=hidden_dim,
            hidden_dim=hidden_dim,
            prompt_len=prompt_len,
            embed_dim=embed_dim,
            rank=rank,
            dropout=dropout
        )
        self.deep_scale = nn.Parameter(torch.tensor(0.3))
        
        self.base_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, prompt_len, embed_dim) * 0.02)
            for _ in range(num_layers)
        ])
        
        self.layer_embed = nn.Embedding(num_layers, hidden_dim // 4)
        
        self.layer_modulation = nn.ModuleList([
            nn.Linear(hidden_dim // 4, embed_dim) for _ in range(num_layers)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, env: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B = env.size(0)
        env_encoded = self.env_encoder(env)
        
        layer_emb = self.layer_embed(
            torch.tensor([layer_idx], device=env.device)
        ).expand(B, -1)
        
        layer_mod = self.layer_modulation[layer_idx](layer_emb)
        
        if layer_idx in self.shallow_layers:
            dynamic = self.shallow_generator(env_encoded)
            dynamic = dynamic.view(B, self.prompt_len, self.embed_dim)
            scale = torch.sigmoid(self.shallow_scale)
        elif layer_idx in self.middle_layers:
            dynamic = self.middle_generator(env_encoded)
            dynamic = dynamic.view(B, self.prompt_len, self.embed_dim)
            scale = torch.sigmoid(self.middle_scale)
        else:
            dynamic = self.deep_generator(env_encoded)
            scale = torch.sigmoid(self.deep_scale)
        
        dynamic = dynamic * (1 + layer_mod.unsqueeze(1) * 0.1)
        
        base = self.base_prompts[layer_idx].expand(B, -1, -1)
        prompts = base + scale * dynamic
        
        return prompts, scale


class ChannelWiseGating(nn.Module):
    """通道级注意力门控"""
    
    def __init__(
        self,
        env_dim: int = 27,
        embed_dim: int = 768,
        hidden_dim: int = 128,
        num_layers: int = 12,
        init_value: float = -2.0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        self.env_proj = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.GELU(),
        )
        
        self.gate_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, embed_dim))
            for _ in range(num_layers)
        ])
        
        for gate_net in self.gate_nets:
            nn.init.zeros_(gate_net[0].weight)
            nn.init.constant_(gate_net[0].bias, init_value)
    
    def forward(self, env: torch.Tensor, layer_idx: int) -> torch.Tensor:
        h = self.env_proj(env)
        gate_logits = self.gate_nets[layer_idx](h)
        gate = torch.sigmoid(gate_logits)
        return gate


class HierarchicalGeoContextualPrompts(nn.Module):
    """HGCP: Hierarchical Geo-Contextual Prompts"""
    
    def __init__(
        self,
        env_dim: int = 27,
        prompt_len: int = 10,
        embed_dim: int = 768,
        num_layers: int = 12,
        hidden_dim: int = 256,
        rank: int = 16,
        dropout: float = 0.1,
        use_channel_gate: bool = True,
        gate_init: float = -2.0
    ):
        super().__init__()
        
        self.env_dim = env_dim
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_channel_gate = use_channel_gate
        
        self.strategy_generator = LayerSpecificStrategy(
            env_dim=env_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            prompt_len=prompt_len,
            num_layers=num_layers,
            rank=rank,
            dropout=dropout
        )
        
        if use_channel_gate:
            self.channel_gate = ChannelWiseGating(
                env_dim=env_dim,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim // 2,
                num_layers=num_layers,
                init_value=gate_init
            )
        
        self._print_info()
    
    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✨ HGCP initialized: env_dim={self.env_dim}, prompt_len={self.prompt_len}")
        print(f"   Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    
    def forward(self, env: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = env.size(0)
        env = torch.nan_to_num(env, nan=0.0)
        
        prompts, scale = self.strategy_generator(env, layer_idx)
        info = {'scale': scale}
        
        if self.use_channel_gate:
            gate = self.channel_gate(env, layer_idx)
            gate_expanded = gate.unsqueeze(1)
            base = self.strategy_generator.base_prompts[layer_idx].expand(B, -1, -1)
            prompts = base + gate_expanded * (prompts - base)
            info['gate_mean'] = gate.mean()
            info['gate'] = gate
        
        return prompts, info
    
    def get_all_prompts(self, env: torch.Tensor) -> Tuple[Dict[int, torch.Tensor], Dict[int, Dict]]:
        all_prompts = {}
        all_info = {}
        for layer_idx in range(self.num_layers):
            prompts, info = self.forward(env, layer_idx)
            all_prompts[layer_idx] = prompts
            all_info[layer_idx] = info
        return all_prompts, all_info


class Dinov2HGCP(nn.Module):
    """带 HGCP 的 DINOv2 模型"""
    
    def __init__(
        self,
        base_model,
        env_dim: int = 27,
        use_hgcp: bool = True,
        hgcp_hidden_dim: int = 256,
        hgcp_dropout: float = 0.1,
        hgcp_rank: int = 16,
        use_channel_gate: bool = True,
        gate_init: float = -2.0,
        freeze_base: bool = False
    ):
        super().__init__()
        
        self.base_model = base_model
        self.use_hgcp = use_hgcp
        self.env_dim = env_dim
        
        self.embed_dim = base_model.embed_dim
        self.prompt_len = base_model.prompt_len
        self.num_layers = len(base_model.dino.blocks)
        
        if use_hgcp:
            self.hgcp = HierarchicalGeoContextualPrompts(
                env_dim=env_dim,
                prompt_len=self.prompt_len,
                embed_dim=self.embed_dim,
                num_layers=self.num_layers,
                hidden_dim=hgcp_hidden_dim,
                rank=hgcp_rank,
                dropout=hgcp_dropout,
                use_channel_gate=use_channel_gate,
                gate_init=gate_init
            )
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("Base model frozen")
        
        self._print_params()
    
    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        base_params = sum(p.numel() for p in self.base_model.parameters())
        hgcp_params = sum(p.numel() for p in self.hgcp.parameters()) if self.use_hgcp else 0
        print(f"📊 HGCP Model: Base={base_params/1e6:.2f}M, HGCP={hgcp_params/1e6:.2f}M, Total={total/1e6:.2f}M")
    
    def forward(self, img: torch.Tensor, env: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = img.size(0)
        
        if self.base_model.channel_adapter is not None:
            x = self.base_model.channel_adapter(img)
        elif self.base_model.in_channels > 3:
            x = img[:, :3, :, :]
        else:
            x = img
        
        x = self.base_model.dino.patch_embed(x)
        
        if self.base_model.dino.cls_token is not None:
            cls_token = self.base_model.dino.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        x = x + self.base_model.dino.pos_embed
        x = self.base_model.dino.pos_drop(x)
        
        if env is not None:
            env = torch.nan_to_num(env, nan=0.0)
        
        for layer_idx, block in enumerate(self.base_model.dino.blocks):
            if self.use_hgcp and env is not None:
                dynamic_prompts, _ = self.hgcp(env, layer_idx)
                prompts_to_use = dynamic_prompts
            else:
                if self.base_model.adapted_encoder.use_layer_specific_prompts:
                    prompts_to_use = self.base_model.adapted_encoder.prompts[layer_idx](batch_size)
                else:
                    prompts_to_use = self.base_model.adapted_encoder.shared_prompts(batch_size)
            
            x_with_prompts = torch.cat([x, prompts_to_use], dim=1)
            x_with_prompts = block(x_with_prompts)
            x = x_with_prompts[:, :-self.prompt_len, :]
        
        x = self.base_model.dino.norm(x)
        cls_features = x[:, 0]
        
        if self.base_model.use_env and env is not None:
            try:
                from src.models.adaptive_fusion import AdaptiveEnvironmentalFusion
                ADAPTIVE_FUSION_AVAILABLE = True
            except ImportError:
                ADAPTIVE_FUSION_AVAILABLE = False
            
            if self.base_model.fusion_type == "adaptive_attention" and ADAPTIVE_FUSION_AVAILABLE:
                fused_feat = self.base_model.fusion(cls_features, env)
            elif self.base_model.fusion_type in ["cross_attention", "adaptive_attention"]:
                env_feat = self.base_model.env_encoder(env)
                fused_feat = self.base_model.fusion(cls_features, env_feat)
            elif self.base_model.fusion_type == "concat":
                env_feat = self.base_model.env_encoder(env)
                fused_feat = torch.cat([cls_features, env_feat], dim=1)
            else:
                fused_feat = cls_features
        else:
            fused_feat = cls_features
        
        logits = self.base_model.classifier(fused_feat)
        return logits


def create_dinov2_hgcp_model(config):
    """创建 DINOv2 HGCP 模型"""
    from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt
    
    module_cfg = config.experiment.module
    data_cfg = config.data
    
    env_dim = sum(data_cfg.env_var_sizes) if hasattr(data_cfg, 'env_var_sizes') else 27
    num_classes = data_cfg.total_species
    
    base_model = Dinov2AdapterPrompt(
        num_classes=num_classes,
        dino_model_name=getattr(module_cfg, 'dino_model', 'vit_base_patch14_dinov2.lvd142m'),
        pretrained_path=getattr(module_cfg, 'pretrained_path', 'checkpoints/dinov2_vitb14_pretrain.pth'),
        prompt_len=getattr(module_cfg, 'prompt_len', 10),
        bottleneck_dim=getattr(module_cfg, 'bottleneck_dim', 64),
        env_input_dim=env_dim,
        env_hidden_dim=getattr(module_cfg, 'env_hidden_dim', 512),
        env_num_layers=getattr(module_cfg, 'env_num_layers', 3),
        use_env=True,
        fusion_type=getattr(module_cfg, 'fusion_type', 'adaptive_attention'),
        use_channel_adapter=getattr(module_cfg, 'use_channel_adapter', True),
        in_channels=getattr(module_cfg, 'in_channels', 4),
        channel_adapter_type=getattr(module_cfg, 'channel_adapter_type', 'learned'),
        freeze_backbone=getattr(module_cfg, 'freeze_backbone', True),
        unfreeze_last_n_blocks=getattr(module_cfg, 'unfreeze_last_n_blocks', 0),
        use_dropkey=getattr(module_cfg, 'use_dropkey', False),
        dropkey_rate=getattr(module_cfg, 'dropkey_rate', 0.1),
        hidden_dims=getattr(module_cfg, 'hidden_dims', [1024, 512]),
        dropout=getattr(module_cfg, 'dropout', 0.2),
    )
    
    hgcp_model = Dinov2HGCP(
        base_model=base_model,
        env_dim=env_dim,
        use_hgcp=getattr(module_cfg, 'use_hgcp', True),
        hgcp_hidden_dim=getattr(module_cfg, 'hgcp_hidden_dim', 256),
        hgcp_dropout=getattr(module_cfg, 'hgcp_dropout', 0.1),
        hgcp_rank=getattr(module_cfg, 'hgcp_rank', 16),
        use_channel_gate=getattr(module_cfg, 'use_channel_gate', True),
        gate_init=getattr(module_cfg, 'hgcp_gate_init', -2.0),
        freeze_base=getattr(module_cfg, 'freeze_base_for_hgcp', False),
    )
    
    print("✅ Created DINOv2 HGCP model")
    return hgcp_model
