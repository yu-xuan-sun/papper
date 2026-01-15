"""
FDA-Net: Frequency-Decoupled Domain Adaptation Network
频率解耦域自适应网络

核心创新点：
1. 傅里叶变换分解：将图像分解为低频（全局结构）和高频（局部纹理）
2. 双分支特征提取：分别编码域不变（低频）和域特定（高频）特征
3. 自适应频率门控：根据环境变量动态调整低高频融合权重
4. 可选域判别器：通过对抗训练增强跨域泛化能力
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from torch.autograd import Function


class FrequencyDecomposition(nn.Module):
    """
    频率域分解模块
    使用FFT将图像分解为低频和高频成分
    """
    
    def __init__(
        self,
        low_freq_ratio: float = 0.25,
        learnable_threshold: bool = True,
        smooth_transition: bool = True
    ):
        super().__init__()
        self.smooth_transition = smooth_transition
        
        # 可学习的频率阈值
        if learnable_threshold:
            self.freq_threshold = nn.Parameter(torch.tensor(low_freq_ratio))
        else:
            self.register_buffer('freq_threshold', torch.tensor(low_freq_ratio))
        
        self.learnable = learnable_threshold
        print(f"🌊 FrequencyDecomposition: ratio={low_freq_ratio}, learnable={learnable_threshold}")
    
    def _create_frequency_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """创建频率掩码"""
        # 计算频率坐标
        freq_y = torch.fft.fftfreq(H, device=device).view(-1, 1)
        freq_x = torch.fft.fftfreq(W, device=device).view(1, -1)
        
        # 归一化频率距离 [0, 1]
        freq_dist = torch.sqrt(freq_y ** 2 + freq_x ** 2)
        freq_dist = freq_dist / freq_dist.max()
        
        # 获取阈值 (确保在合理范围内)
        threshold = torch.sigmoid(self.freq_threshold) if self.learnable else self.freq_threshold
        threshold = torch.clamp(threshold, 0.05, 0.95)
        
        if self.smooth_transition:
            # 平滑过渡 (sigmoid-based)
            sharpness = 20.0
            low_mask = torch.sigmoid(-sharpness * (freq_dist - threshold))
            high_mask = 1 - low_mask
        else:
            # 硬切分
            low_mask = (freq_dist <= threshold).float()
            high_mask = 1 - low_mask
        
        return low_mask, high_mask, threshold
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        输入: x [B, C, H, W]
        输出: low_freq_img, high_freq_img, info_dict
        """
        B, C, H, W = x.shape
        
        # FFT (2D on spatial dimensions)
        x_fft = torch.fft.fft2(x)
        x_fft_shifted = torch.fft.fftshift(x_fft)  # 将零频移到中心
        
        # 创建频率掩码
        low_mask, high_mask, threshold = self._create_frequency_mask(H, W, x.device)
        
        # 应用掩码
        low_fft = x_fft_shifted * low_mask.unsqueeze(0).unsqueeze(0)
        high_fft = x_fft_shifted * high_mask.unsqueeze(0).unsqueeze(0)
        
        # 逆FFT
        low_fft_unshifted = torch.fft.ifftshift(low_fft)
        high_fft_unshifted = torch.fft.ifftshift(high_fft)
        
        low_img = torch.fft.ifft2(low_fft_unshifted).real
        high_img = torch.fft.ifft2(high_fft_unshifted).real
        
        # 计算能量比例用于监控
        total_energy = torch.abs(x_fft_shifted).pow(2).sum()
        low_energy = torch.abs(low_fft).pow(2).sum()
        energy_ratio = (low_energy / (total_energy + 1e-8)).item()
        
        info = {
            'freq_threshold': threshold.item() if isinstance(threshold, torch.Tensor) else threshold,
            'low_energy_ratio': energy_ratio
        }
        
        return low_img, high_img, info


class FrequencyBranchEncoder(nn.Module):
    """
    频率分支编码器
    为低频/高频分支设计的轻量级CNN特征提取器
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        out_dim: int = 768,
        num_layers: int = 3,
        dropout: float = 0.1,
        branch_type: str = "low"  # "low" or "high"
    ):
        super().__init__()
        
        # 根据分支类型调整卷积核大小
        # 低频分支：较大感受野
        # 高频分支：较小感受野保留细节
        if branch_type == "low":
            kernel_sizes = [7, 5, 3]
        else:
            kernel_sizes = [3, 3, 3]
        
        layers = []
        curr_channels = in_channels
        
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else hidden_dim * 2
            k = kernel_sizes[min(i, len(kernel_sizes)-1)]
            padding = k // 2
            
            layers.extend([
                nn.Conv2d(curr_channels, out_ch, k, stride=2, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.Dropout2d(dropout)
            ])
            curr_channels = out_ch
        
        self.encoder = nn.Sequential(*layers)
        
        # 全局池化 + 投影
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(curr_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
        print(f"  📊 {branch_type.upper()} Branch: in={in_channels}, hidden={hidden_dim}, out={out_dim}")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x [B, C, H, W]
        输出: features [B, out_dim]
        """
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return x


class AdaptiveFrequencyGate(nn.Module):
    """
    自适应频率门控
    根据环境变量动态调整低频/高频特征的融合权重
    """
    
    def __init__(
        self,
        feat_dim: int = 768,
        env_dim: int = 27,
        hidden_dim: int = 128,
        init_low_bias: float = 0.6
    ):
        super().__init__()
        
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim + feat_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        
        # 初始化偏向低频
        nn.init.zeros_(self.gate_net[-1].weight)
        self.gate_net[-1].bias.data = torch.tensor([init_low_bias, 1 - init_low_bias])
        
        print(f"  🚪 AdaptiveFrequencyGate: feat={feat_dim}, env={env_dim}, init_low={init_low_bias}")
    
    def forward(
        self, 
        low_feat: torch.Tensor, 
        high_feat: torch.Tensor,
        env: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        输入: low_feat [B, D], high_feat [B, D], env [B, env_dim]
        输出: fused_feat [B, D], info_dict
        """
        if env is not None:
            env_emb = self.env_encoder(env)
            gate_input = torch.cat([env_emb, low_feat, high_feat], dim=-1)
        else:
            env_emb = torch.zeros(low_feat.size(0), self.env_encoder[0].in_features, device=low_feat.device)
            env_emb = self.env_encoder(env_emb)
            gate_input = torch.cat([env_emb, low_feat, high_feat], dim=-1)
        
        # 计算门控权重 [B, 2]
        gate_weights = F.softmax(self.gate_net(gate_input), dim=-1)
        low_weight = gate_weights[:, 0:1]
        high_weight = gate_weights[:, 1:2]
        
        # 加权融合
        fused = low_weight * low_feat + high_weight * high_feat
        
        info = {
            'low_weight': low_weight.mean().item(),
            'high_weight': high_weight.mean().item()
        }
        
        return fused, info


class GradientReversal(Function):
    """梯度反转层，用于对抗训练"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainDiscriminator(nn.Module):
    """域判别器（可选）"""
    
    def __init__(
        self,
        feat_dim: int = 768,
        hidden_dim: int = 256,
        num_domains: int = 2,
        gradient_reversal: bool = True
    ):
        super().__init__()
        self.gradient_reversal = gradient_reversal
        
        self.discriminator = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_domains),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if self.gradient_reversal:
            x = GradientReversal.apply(x, alpha)
        return self.discriminator(x)


class FDAModule(nn.Module):
    """
    完整的FDA模块
    组合频率分解、双分支编码、自适应门控
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 768,
        env_dim: int = 27,
        hidden_dim: int = 256,
        low_freq_ratio: float = 0.25,
        learnable_freq: bool = True,
        use_domain_disc: bool = False,
        num_domains: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.use_domain_disc = use_domain_disc
        
        # 频率分解
        self.freq_decomp = FrequencyDecomposition(
            low_freq_ratio=low_freq_ratio,
            learnable_threshold=learnable_freq,
            smooth_transition=True
        )
        
        # 双分支编码器
        self.low_branch = FrequencyBranchEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_dim=feat_dim,
            num_layers=3,
            dropout=dropout,
            branch_type="low"
        )
        
        self.high_branch = FrequencyBranchEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_dim=feat_dim,
            num_layers=3,
            dropout=dropout,
            branch_type="high"
        )
        
        # 自适应门控
        self.freq_gate = AdaptiveFrequencyGate(
            feat_dim=feat_dim,
            env_dim=env_dim,
            hidden_dim=hidden_dim // 2
        )
        
        # 域判别器（可选）
        if use_domain_disc:
            self.domain_disc = DomainDiscriminator(
                feat_dim=feat_dim,
                hidden_dim=hidden_dim,
                num_domains=num_domains
            )
        
        self._print_info()
    
    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✨ FDAModule initialized: feat_dim={self.feat_dim}")
        print(f"   Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    
    def forward(
        self, 
        x: torch.Tensor, 
        env: Optional[torch.Tensor] = None,
        return_domain_pred: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        输入: x [B, C, H, W], env [B, env_dim]
        输出: fused_features [B, feat_dim], info_dict
        """
        # 频率分解
        low_img, high_img, decomp_info = self.freq_decomp(x)
        
        # 双分支编码
        low_feat = self.low_branch(low_img)
        high_feat = self.high_branch(high_img)
        
        # 自适应融合
        fused_feat, gate_info = self.freq_gate(low_feat, high_feat, env)
        
        # 整合信息
        info = {**decomp_info, **gate_info}
        
        # 域判别（如果启用）
        if return_domain_pred and self.use_domain_disc:
            domain_pred = self.domain_disc(fused_feat)
            info['domain_pred'] = domain_pred
        
        return fused_feat, info


class Dinov2FDA(nn.Module):
    """
    DINOv2 + FDA 联合模型
    """
    
    def __init__(
        self,
        base_model,
        num_classes: int = 670,
        use_fda: bool = True,
        fda_hidden_dim: int = 256,
        fda_low_freq_ratio: float = 0.25,
        fda_learnable_freq: bool = True,
        fda_fusion_weight: float = 0.3,
        use_domain_disc: bool = False,
        fda_dropout: float = 0.1
    ):
        super().__init__()
        
        self.base_model = base_model
        self.use_fda = use_fda
        self.num_classes = num_classes
        
        # 获取base model的特征维度
        feat_dim = base_model.embed_dim
        env_dim = getattr(base_model, 'env_input_dim', 27)
        
        if use_fda:
            self.fda = FDAModule(
                in_channels=3,
                feat_dim=feat_dim,
                env_dim=env_dim,
                hidden_dim=fda_hidden_dim,
                low_freq_ratio=fda_low_freq_ratio,
                learnable_freq=fda_learnable_freq,
                use_domain_disc=use_domain_disc,
                dropout=fda_dropout
            )
            
            # 特征融合层
            self.fusion_layer = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.GELU(),
                nn.Dropout(fda_dropout)
            )
            
            # 可学习的融合权重
            self.fusion_alpha = nn.Parameter(torch.tensor(fda_fusion_weight))
        
        # 打印模型信息
        self._print_summary()
    
    def _print_summary(self):
        base_params = sum(p.numel() for p in self.base_model.parameters())
        if self.use_fda:
            fda_params = sum(p.numel() for p in self.fda.parameters())
            fusion_params = sum(p.numel() for p in self.fusion_layer.parameters())
            total = base_params + fda_params + fusion_params
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"📊 FDA Model Summary:")
            print(f"   Base: {base_params/1e6:.2f}M")
            print(f"   FDA:  {fda_params/1e6:.2f}M")
            print(f"   Total: {total/1e6:.2f}M (Trainable: {trainable/1e6:.2f}M)")
        print("✅ Created DINOv2 + FDA model")
    
    def forward(self, img: torch.Tensor, env: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        输入: img [B, C, H, W], env [B, env_dim]
        输出: logits [B, num_classes]
        """
        batch_size = img.size(0)
        
        # ===== 图像尺寸处理 =====
        _, C, H, W = img.shape
        if H != 224 or W != 224:
            img_resized = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            img_resized = img
        
        # ===== DINOv2 分支 =====
        if self.base_model.channel_adapter is not None:
            x_dino = self.base_model.channel_adapter(img_resized)
        elif self.base_model.in_channels > 3:
            x_dino = img_resized[:, :3, :, :]
        else:
            x_dino = img_resized
        
        # Patch embedding
        x = self.base_model.dino.patch_embed(x_dino)
        
        # CLS token
        if self.base_model.dino.cls_token is not None:
            cls_token = self.base_model.dino.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        x = x + self.base_model.dino.pos_embed
        x = self.base_model.dino.pos_drop(x)
        
        # Transformer blocks with prompts
        prompt_len = self.base_model.prompt_len
        for layer_idx, block in enumerate(self.base_model.dino.blocks):
            if self.base_model.adapted_encoder.use_layer_specific_prompts:
                prompts = self.base_model.adapted_encoder.prompts[layer_idx](batch_size)
            else:
                prompts = self.base_model.adapted_encoder.shared_prompts(batch_size)
            
            x_with_prompts = torch.cat([x, prompts], dim=1)
            x_with_prompts = block(x_with_prompts)
            x = x_with_prompts[:, :-prompt_len, :]
        
        x = self.base_model.dino.norm(x)
        dino_features = x[:, 0]  # CLS token
        
        # ===== FDA 分支 =====
        if self.use_fda:
            # FDA使用原始分辨率图像以保留更多频率信息
            fda_features, fda_info = self.fda(img, env)
            
            # 特征融合
            alpha = torch.sigmoid(self.fusion_alpha)
            combined = torch.cat([dino_features, fda_features], dim=-1)
            combined_features = self.fusion_layer(combined)
            combined_features = (1 - alpha) * dino_features + alpha * combined_features
        else:
            combined_features = dino_features
        
        # ===== 环境特征融合 =====
        if self.base_model.use_env and env is not None:
            if hasattr(self.base_model, 'fusion') and self.base_model.fusion is not None:
                if self.base_model.fusion_type == "adaptive_attention":
                    combined_features = self.base_model.fusion(combined_features, env)
                elif hasattr(self.base_model, 'env_encoder') and self.base_model.env_encoder is not None:
                    env_feat = self.base_model.env_encoder(env)
                    combined_features = self.base_model.fusion(combined_features, env_feat)
                else:
                    combined_features = self.base_model.fusion(combined_features, env)
        
        # ===== 分类头 =====
        logits = self.base_model.classifier(combined_features)
        
        return logits


def create_dinov2_fda_model(opts) -> Dinov2FDA:
    """
    从配置创建 DINOv2 + FDA 模型
    """
    from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt
    
    module_cfg = opts.experiment.module
    data_cfg = opts.data
    
    # 直接使用total_species
    num_species = data_cfg.total_species
    
    # 获取classifier配置
    classifier_cfg = getattr(module_cfg, 'classifier', None)
    hidden_dims = list(getattr(classifier_cfg, 'hidden_dims', [1024, 512])) if classifier_cfg else [1024, 512]
    classifier_dropout = getattr(classifier_cfg, 'dropout', 0.3) if classifier_cfg else 0.3
    
    # 获取adapter_layers配置
    adapter_layers = getattr(module_cfg, 'adapter_layers', [9, 10, 11])
    if hasattr(adapter_layers, '__iter__') and not isinstance(adapter_layers, str):
        adapter_layers = list(adapter_layers)
    else:
        adapter_layers = [9, 10, 11]
    
    # 创建base model
    base_model = Dinov2AdapterPrompt(
        num_classes=num_species,
        dino_model_name="vit_base_patch14_dinov2.lvd142m",
        pretrained_path="checkpoints/dinov2_vitb14_pretrain.pth",
        prompt_len=getattr(module_cfg, 'num_tokens', 10),
        bottleneck_dim=getattr(module_cfg, 'adapter_dim', 64),
        adapter_layers=adapter_layers,
        adapter_dropout=getattr(module_cfg, 'prompt_dropout', 0.1),
        use_layer_specific_prompts=True,
        env_input_dim=getattr(module_cfg, 'num_env_features', 27),
        env_hidden_dim=getattr(module_cfg, 'env_hidden_dim', 256),
        env_num_layers=getattr(module_cfg, 'env_num_layers', 2),
        use_env=True,
        fusion_type="cross_attention",
        hidden_dims=hidden_dims,
        dropout=classifier_dropout,
        use_channel_adapter=getattr(module_cfg, 'use_channel_adapter', False),
        in_channels=getattr(module_cfg, 'in_channels', 3),
        freeze_backbone=getattr(module_cfg, 'freeze', True)
    )
    
    # 创建FDA模型
    model = Dinov2FDA(
        base_model=base_model,
        num_classes=num_species,
        use_fda=getattr(module_cfg, 'use_fda', True),
        fda_hidden_dim=getattr(module_cfg, 'fda_hidden_dim', 256),
        fda_low_freq_ratio=getattr(module_cfg, 'fda_low_freq_ratio', 0.25),
        fda_learnable_freq=getattr(module_cfg, 'fda_learnable_freq', True),
        fda_fusion_weight=getattr(module_cfg, 'fda_fusion_weight', 0.3),
        use_domain_disc=getattr(module_cfg, 'use_domain_disc', False),
        fda_dropout=getattr(module_cfg, 'fda_dropout', 0.1)
    )
    
    return model


__all__ = [
    'FrequencyDecomposition',
    'FrequencyBranchEncoder', 
    'AdaptiveFrequencyGate',
    'FDAModule',
    'DomainDiscriminator',
    'Dinov2FDA',
    'create_dinov2_fda_model'
]
