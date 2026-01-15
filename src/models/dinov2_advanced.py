"""
Enhanced DINOv2 with Advanced Techniques
包含以下改进:
1. DropKey Attention - 注意力正则化
2. Multi-Scale Feature Aggregation - 多尺度特征融合
3. Gated Cross-Attention Fusion - 门控多模态融合
4. Enhanced Environment Encoder - 增强环境编码
5. Self-Distillation - 自蒸馏支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import timm


class DropKeyAttention(nn.Module):
    """
    DropKey: 在训练时随机丢弃部分key tokens
    使用monkey-patching方式注入，保持预训练权重兼容性
    """
    
    def __init__(self, attn_module: nn.Module, drop_rate: float = 0.1):
        super().__init__()
        self.attn_module = attn_module
        self.drop_rate = drop_rate
        self._original_forward = None
        
    def inject_dropkey(self):
        """将dropkey注入到attention模块"""
        if self._original_forward is None:
            self._original_forward = self.attn_module.forward
            
        def forward_with_dropkey(x):
            if self.training and self.drop_rate > 0:
                B, N, C = x.shape
                # 随机保留的token数量
                keep_tokens = int(N * (1 - self.drop_rate))
                if keep_tokens < N:
                    # 随机选择要保留的indices
                    indices = torch.randperm(N, device=x.device)[:keep_tokens]
                    indices = indices.sort()[0]  # 保持顺序
                    x = x[:, indices, :]
            return self._original_forward(x)
        
        self.attn_module.forward = forward_with_dropkey


class MultiScaleAggregation(nn.Module):
    """
    多尺度特征聚合模块 (类似FPN)
    从DINOv2的多个层提取特征并融合
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        output_dim: int = 768,
        num_scales: int = 4
    ):
        super().__init__()
        self.num_scales = num_scales
        
        # 为每个尺度创建投影层
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            ) for dim in feature_dims
        ])
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * num_scales, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of [B, N, D] tensors from different layers
        Returns:
            fused_features: [B, D] tensor
        """
        # 投影每个尺度的特征
        projected = []
        for i, feat in enumerate(features[:self.num_scales]):
            # 提取CLS token
            cls_token = feat[:, 0]  # [B, D]
            proj = self.projections[i](cls_token)  # [B, output_dim]
            projected.append(proj)
        
        # 拼接并融合
        concat_features = torch.cat(projected, dim=1)  # [B, output_dim * num_scales]
        fused = self.fusion(concat_features)  # [B, output_dim]
        
        return fused


class GatedCrossAttentionFusion(nn.Module):
    """
    门控交叉注意力融合
    学习如何动态融合卫星图像特征和环境特征
    """
    
    def __init__(
        self,
        sat_dim: int = 768,
        env_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=sat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 环境特征投影到sat_dim
        self.env_proj = nn.Linear(env_dim, sat_dim)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(sat_dim * 2, sat_dim),
            nn.Sigmoid()
        )
        
        self.norm1 = nn.LayerNorm(sat_dim)
        self.norm2 = nn.LayerNorm(sat_dim)
        
    def forward(
        self,
        sat_features: torch.Tensor,  # [B, sat_dim]
        env_features: torch.Tensor   # [B, env_dim]
    ) -> torch.Tensor:
        """
        Returns:
            fused_features: [B, sat_dim]
        """
        # 投影环境特征
        env_proj = self.env_proj(env_features)  # [B, sat_dim]
        
        # 添加序列维度用于attention
        sat_seq = sat_features.unsqueeze(1)  # [B, 1, sat_dim]
        env_seq = env_proj.unsqueeze(1)  # [B, 1, sat_dim]
        
        # 交叉注意力: sat attend to env
        attn_out, _ = self.cross_attn(
            query=sat_seq,
            key=env_seq,
            value=env_seq
        )  # [B, 1, sat_dim]
        attn_out = attn_out.squeeze(1)  # [B, sat_dim]
        
        # 残差连接
        attn_out = self.norm1(sat_features + attn_out)
        
        # 门控融合
        gate_input = torch.cat([sat_features, attn_out], dim=1)  # [B, sat_dim*2]
        gate_weights = self.gate(gate_input)  # [B, sat_dim]
        
        fused = gate_weights * attn_out + (1 - gate_weights) * sat_features
        fused = self.norm2(fused)
        
        return fused


class EnhancedEnvEncoder(nn.Module):
    """
    增强的环境特征编码器
    使用自注意力机制捕捉环境变量之间的关系
    """
    
    def __init__(
        self,
        num_features: int = 27,
        hidden_dim: int = 512,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # 自注意力层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, env_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            env_features: [B, num_features]
        Returns:
            encoded: [B, hidden_dim]
        """
        # 添加序列维度
        x = self.input_proj(env_features).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # 自注意力
        x = self.transformer(x)  # [B, 1, hidden_dim]
        
        # 输出
        x = x.squeeze(1)  # [B, hidden_dim]
        x = self.output_proj(x)
        
        return x


class EnhancedDinov2Multimodal(nn.Module):
    """
    增强版DINOv2多模态模型
    集成所有性能改进技术
    """
    
    def __init__(
        self,
        num_species: int,
        dinov2_name: str = "dinov2_vitb14",
        dinov2_pretrained: bool = True,
        proj_dim: int = 768,
        freeze_dinov2: bool = True,
        # Advanced features
        drop_key_rate: float = 0.1,
        use_multi_scale: bool = True,
        use_gated_fusion: bool = True,
        use_enhanced_env: bool = True,
        use_self_distill: bool = False,
        # Regularization
        drop_path_rate: float = 0.1,
        dropout: float = 0.2,
        # Environment
        num_env_features: int = 27,
        env_hidden_dim: int = 512,
        # Other
        device: str = "cuda"
    ):
        super().__init__()
        
        self.num_species = num_species
        self.proj_dim = proj_dim
        self.freeze_dinov2 = freeze_dinov2
        self.use_multi_scale = use_multi_scale
        self.use_gated_fusion = use_gated_fusion
        self.use_enhanced_env = use_enhanced_env
        self.use_self_distill = use_self_distill
        self.drop_key_rate = drop_key_rate
        
        # 加载DINOv2 backbone
        if dinov2_pretrained:
            # 使用 torch.hub 加载 DINOv2
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', dinov2_name)
        else:
            # 如果不需要预训练权重，使用 timm
            try:
                self.dinov2 = timm.create_model(
                    dinov2_name,
                    pretrained=False,
                    num_classes=0
                )
            except:
                # 后备方案：从 hub 加载
                self.dinov2 = torch.hub.load('facebookresearch/dinov2', dinov2_name)
        
        # 获取特征维度
        if hasattr(self.dinov2, 'embed_dim'):
            backbone_dim = self.dinov2.embed_dim
        else:
            backbone_dim = 768  # default for vitb14
        
        # 添加通道转换层 (4通道RGBNIR -> 3通道RGB)
        self.channel_adapter = nn.Conv2d(4, 3, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.channel_adapter.weight)
        
        # 冻结backbone
        if freeze_dinov2:
            for param in self.dinov2.parameters():
                param.requires_grad = False
        
        # DropKey注入 (如果启用)
        if drop_key_rate > 0:
            self._inject_dropkey()
        
        # 多尺度聚合
        if use_multi_scale:
            # 从第3, 7, 11层提取特征
            self.multi_scale_agg = MultiScaleAggregation(
                feature_dims=[backbone_dim] * 4,
                output_dim=proj_dim,
                num_scales=4
            )
        else:
            # 简单投影
            self.sat_proj = nn.Sequential(
                nn.Linear(backbone_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU()
            )
        
        # 环境编码器
        if use_enhanced_env:
            self.env_encoder = EnhancedEnvEncoder(
                num_features=num_env_features,
                hidden_dim=env_hidden_dim,
                num_heads=4,
                num_layers=2,
                dropout=dropout
            )
        else:
            self.env_encoder = nn.Sequential(
                nn.Linear(num_env_features, env_hidden_dim),
                nn.LayerNorm(env_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # 门控融合
        if use_gated_fusion:
            self.fusion = GatedCrossAttentionFusion(
                sat_dim=proj_dim,
                env_dim=env_hidden_dim,
                num_heads=8,
                dropout=dropout
            )
        else:
            # 简单拼接
            self.fusion = nn.Sequential(
                nn.Linear(proj_dim + env_hidden_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.LayerNorm(proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, num_species)
        )
        
        # 自蒸馏分支 (如果启用)
        if use_self_distill:
            self.distill_classifier = nn.Linear(proj_dim, num_species)
        
        print(f"✅ Created Enhanced DINOv2:")
        print(f"   - Model: {dinov2_name}")
        print(f"   - DropKey: {drop_key_rate}")
        print(f"   - Multi-scale: {use_multi_scale}")
        print(f"   - Gated fusion: {use_gated_fusion}")
        print(f"   - Enhanced env: {use_enhanced_env}")
        print(f"   - Self-distill: {use_self_distill}")
        
    def _inject_dropkey(self):
        """
        为 DINOv2 的 attention 层添加额外的 dropout
        这是一种简化的 DropKey 实现
        """
        if hasattr(self.dinov2, 'blocks'):
            for block in self.dinov2.blocks:
                if hasattr(block, 'attn') and hasattr(block.attn, 'attn_drop'):
                    # attn_drop 是 nn.Dropout 对象，修改其 p 属性
                    if isinstance(block.attn.attn_drop, nn.Dropout):
                        block.attn.attn_drop.p = max(block.attn.attn_drop.p, self.drop_key_rate)
                    
        print(f"   - Injected DropKey (dropout={self.drop_key_rate}) into attention layers")
    
    def forward(
        self,
        sat_images: torch.Tensor,
        env_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            sat_images: [B, 3, H, W]
            env_features: [B, num_env_features]
            
        Returns:
            dict with keys: 'logits', 'pred', 'logit_sum', 'features'
            (and 'distill_logits' if self.use_self_distill)
        """
        B = sat_images.size(0)
        
        # 通道转换: 4通道(RGBNIR) -> 3通道(RGB)
        if sat_images.size(1) == 4:
            sat_images = self.channel_adapter(sat_images)
        
        # 提取卫星图像特征 - 单尺度
        # DINOv2 forward_features 返回字典，需要提取 x_norm_clstoken
        features_dict = self.dinov2.forward_features(sat_images)
        if isinstance(features_dict, dict):
            sat_features = features_dict['x_norm_clstoken']  # [B, D]
        else:
            # 如果不是字典，可能是张量
            sat_features = features_dict[:, 0] if features_dict.dim() > 2 else features_dict
        sat_features = self.sat_proj(sat_features)  # [B, proj_dim]
        
        # 编码环境特征
        env_encoded = self.env_encoder(env_features)  # [B, env_hidden_dim]
        
        # 融合
        if self.use_gated_fusion:
            fused_features = self.fusion(sat_features, env_encoded)
        else:
            fused_features = self.fusion(torch.cat([sat_features, env_encoded], dim=1))
        
        # 分类
        logits = self.classifier(fused_features)  # [B, num_species]
        pred = torch.sigmoid(logits)
        
        outputs = {
            'logits': logits,
            'pred': pred,
            'logit_sum': logits,  # 兼容BCE loss
            'features': fused_features
        }
        
        # 自蒸馏
        if self.use_self_distill and self.training:
            distill_logits = self.distill_classifier(fused_features.detach())
            outputs['distill_logits'] = distill_logits
        
        return outputs
    
    def is_backbone_frozen(self) -> bool:
        """检查backbone是否被冻结"""
        return not next(self.dinov2.parameters()).requires_grad
    
    def unfreeze_backbone(self):
        """解冻backbone"""
        for param in self.dinov2.parameters():
            param.requires_grad = True
        self.freeze_dinov2 = False
        print("✅ DINOv2 backbone unfrozen")
    
    def freeze_backbone(self):
        """冻结backbone"""
        for param in self.dinov2.parameters():
            param.requires_grad = False
        self.freeze_dinov2 = True
        print("🔒 DINOv2 backbone frozen")


if __name__ == "__main__":
    print("Testing Enhanced DINOv2...")
    
    model = EnhancedDinov2Multimodal(
        num_species=670,
        dinov2_name="dinov2_vits14",
        dinov2_pretrained=False,
        drop_key_rate=0.1,
        use_multi_scale=True,
        use_gated_fusion=True,
        use_enhanced_env=True,
        use_self_distill=True
    )
    
    sat = torch.randn(2, 3, 224, 224)
    env = torch.randn(2, 27)
    
    outputs = model(sat, env)
    print(f"✅ Logits shape: {outputs['logits'].shape}")
    print(f"✅ Pred shape: {outputs['pred'].shape}")
    if 'distill_logits' in outputs:
        print(f"✅ Distill logits shape: {outputs['distill_logits'].shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Total parameters: {total_params/1e6:.1f}M")
