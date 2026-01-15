"""
多通道输入适配器

将4通道(RGBNIR)或其他通道数的输入适配到DINOv2的3通道输入
两种策略:
1. 通道映射: 学习一个1x1卷积将N通道映射到3通道
2. 特征融合: 分别处理RGB和额外通道，然后融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChannelAdapter(nn.Module):
    """
    通道适配器 - 将任意通道数映射到3通道
    
    使用1x1卷积，可以保留预训练权重的空间特征提取能力
    """
    
    def __init__(
        self, 
        in_channels: int = 4,
        out_channels: int = 3,
        pretrained_rgb_weights: Optional[torch.Tensor] = None,
        trainable: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1x1卷积进行通道映射
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # 初始化策略: RGB通道使用单位映射，NIR通道使用小随机值
        with torch.no_grad():
            self.conv.weight.zero_()
            # 前3个通道直接传递 (单位映射)
            min_channels = min(3, in_channels, out_channels)
            self.conv.weight[:min_channels, :min_channels, 0, 0] = torch.eye(min_channels)
            # 额外通道使用小随机值
            if in_channels > 3:
                nn.init.normal_(self.conv.weight[:, 3:, :, :], std=0.01)
        
        # 是否可训练
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False
        
        print(f"ChannelAdapter: {in_channels} -> {out_channels} channels")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W]
        Returns:
            [B, out_channels, H, W]
        """
        return self.conv(x)


class NIREnhancedAdapter(nn.Module):
    """
    NIR增强适配器
    
    策略: RGB通道直接使用，NIR通道用于增强特定特征
    适合卫星图像中植被检测任务
    """
    
    def __init__(
        self,
        nir_weight: float = 0.3,
        learnable: bool = True
    ):
        super().__init__()
        
        self.nir_weight = nn.Parameter(
            torch.tensor(nir_weight),
            requires_grad=learnable
        )
        
        # NIR增强卷积
        self.nir_enhance = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=1)
        )
        
        # 初始化
        for m in self.nir_enhance.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        print(f"NIREnhancedAdapter: initial nir_weight={nir_weight}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4, H, W] - RGBNIR
        Returns:
            [B, 3, H, W] - 增强后的RGB
        """
        rgb = x[:, :3, :, :]  # [B, 3, H, W]
        nir = x[:, 3:4, :, :]  # [B, 1, H, W]
        
        # NIR增强
        nir_feat = self.nir_enhance(nir)
        
        # 融合
        weight = torch.sigmoid(self.nir_weight)
        enhanced = rgb + weight * nir_feat
        
        return enhanced


class MultiChannelPatchEmbed(nn.Module):
    """
    多通道Patch Embedding
    
    直接修改ViT的patch embedding层以支持更多通道
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 4,
        embed_dim: int = 768,
        pretrained_patch_embed: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # 新的patch embedding
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # 如果有预训练权重，初始化前3个通道
        if pretrained_patch_embed is not None:
            with torch.no_grad():
                pretrained_weight = pretrained_patch_embed.proj.weight  # [embed_dim, 3, P, P]
                # 复制RGB通道权重
                self.proj.weight[:, :3, :, :] = pretrained_weight
                # 额外通道使用小随机值
                if in_channels > 3:
                    nn.init.normal_(self.proj.weight[:, 3:, :, :], std=0.01)
                # 偏置
                if pretrained_patch_embed.proj.bias is not None:
                    self.proj.bias = nn.Parameter(pretrained_patch_embed.proj.bias.clone())
            print(f"MultiChannelPatchEmbed: Initialized from pretrained weights")
        
        print(f"MultiChannelPatchEmbed: {in_channels}ch x {img_size}x{img_size} -> {self.num_patches} patches x {embed_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W]
        Returns:
            [B, num_patches, embed_dim]
        """
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


__all__ = ['ChannelAdapter', 'NIREnhancedAdapter', 'MultiChannelPatchEmbed', 'SpectralAttentionAdapter', 'SpectralIndexAdapter', 'DualStreamAdapter', 'AdaptiveChannelMixer']


class SpectralAttentionAdapter(nn.Module):
    """
    光谱注意力适配器 - 创新方法1
    
    使用可学习的注意力机制为每个通道分配重要性权重
    相比简单的1x1卷积，这种方法可以：
    1. 动态调整每个波段的贡献
    2. 学习通道间的依赖关系
    3. 参数量仅略微增加
    """
    
    def __init__(
        self, 
        in_channels: int = 4,
        out_channels: int = 3,
        reduction: int = 2,
        use_spatial_attention: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 通道注意力: 学习每个波段的重要性
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 [B, C, 1, 1]
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 通道映射: 1x1卷积
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        
        # 可选的空间注意力
        self.use_spatial_attention = use_spatial_attention
        if use_spatial_attention:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            )
        
        # 初始化
        self._init_weights()
        
        print(f"SpectralAttentionAdapter: {in_channels}->{out_channels}, spatial_att={use_spatial_attention}")
    
    def _init_weights(self):
        # 1x1卷积初始化：RGB通道单位映射
        with torch.no_grad():
            nn.init.zeros_(self.conv.weight)
            min_ch = min(3, self.in_channels, self.out_channels)
            self.conv.weight[:min_ch, :min_ch, 0, 0] = torch.eye(min_ch)
            if self.in_channels > 3:
                nn.init.normal_(self.conv.weight[:, 3:, :, :], std=0.01)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4, H, W] - RGBNIR
        Returns:
            [B, 3, H, W] - 增强的RGB
        """
        # 通道注意力加权
        att = self.channel_attention(x)  # [B, 4, 1, 1]
        x_weighted = x * att  # 每个通道加权
        
        # 1x1卷积映射
        out = self.conv(x_weighted)
        
        # 可选的空间注意力
        if self.use_spatial_attention:
            spatial_att = self.spatial_attention(out)
            out = out * spatial_att
        
        return out


class SpectralIndexAdapter(nn.Module):
    """
    光谱指数适配器 - 创新方法2
    
    受遥感领域启发，利用波段比值/差值构建类似NDVI的特征
    例如: NDVI = (NIR - Red) / (NIR + Red)
    这种方法可以显式建模植被、水体等地物特征
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        num_indices: int = 3,
        learnable_indices: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_indices = num_indices
        
        if learnable_indices:
            # 可学习的波段组合系数
            self.index_weights = nn.Parameter(torch.randn(num_indices, in_channels, 2))
            nn.init.normal_(self.index_weights, std=0.1)
        else:
            # 固定的遥感指数 (NDVI, NDWI, SAVI等)
            self.register_buffer('index_weights', self._get_predefined_indices())
        
        # 融合网络：原始通道 + 光谱指数 -> RGB
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + num_indices, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        
        # 初始化融合网络
        for m in self.fusion.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        print(f"SpectralIndexAdapter: {in_channels}->{out_channels}, indices={num_indices}, learnable={learnable_indices}")
    
    def _get_predefined_indices(self):
        """预定义的遥感光谱指数 (RGBN顺序)"""
        indices = torch.zeros(3, 4, 2)
        # NDVI = (NIR - Red) / (NIR + Red)
        indices[0, 3, 0] = 1.0   # NIR分子
        indices[0, 0, 0] = -1.0  # Red分子
        indices[0, 3, 1] = 1.0   # NIR分母
        indices[0, 0, 1] = 1.0   # Red分母
        
        # NDWI = (Green - NIR) / (Green + NIR)
        indices[1, 1, 0] = 1.0   # Green分子
        indices[1, 3, 0] = -1.0  # NIR分子
        indices[1, 1, 1] = 1.0   # Green分母
        indices[1, 3, 1] = 1.0   # NIR分母
        
        # 简化EVI
        indices[2, 3, 0] = 2.5   # NIR分子
        indices[2, 0, 0] = -2.5  # Red分子
        indices[2, 3, 1] = 1.0   # NIR分母
        indices[2, 0, 1] = 6.0   # Red分母
        
        return indices
    
    def compute_indices(self, x: torch.Tensor) -> torch.Tensor:
        """计算光谱指数"""
        indices = []
        
        for i in range(self.num_indices):
            # 计算分子和分母
            numerator = (x * self.index_weights[i, :, 0].view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
            denominator = (x * self.index_weights[i, :, 1].view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
            
            # 归一化差异指数
            idx = numerator / (denominator.abs() + 1e-6)
            indices.append(idx)
        
        return torch.cat(indices, dim=1)  # [B, num_indices, H, W]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算光谱指数
        spectral_indices = self.compute_indices(x)
        
        # 拼接原始通道和指数特征
        combined = torch.cat([x, spectral_indices], dim=1)
        
        # 融合并映射到3通道
        out = self.fusion(combined)
        
        return out


class DualStreamAdapter(nn.Module):
    """
    双流适配器 - 创新方法3
    
    RGB和NIR分别处理，然后通过注意力机制融合
    优势：
    1. 保留RGB的预训练知识
    2. NIR独立提取特征后融合
    3. 可以学习最优融合策略
    """
    
    def __init__(
        self,
        nir_hidden_dim: int = 16,
        fusion_type: str = 'cross_attention'  # 'concat' | 'add' | 'cross_attention'
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # RGB流：直接透传（保持预训练）
        self.rgb_stream = nn.Identity()
        
        # NIR流：单独处理
        self.nir_stream = nn.Sequential(
            nn.Conv2d(1, nir_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(nir_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(nir_hidden_dim, nir_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(nir_hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 融合模块
        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Conv2d(3 + nir_hidden_dim, 32, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, kernel_size=1)
            )
        elif fusion_type == 'cross_attention':
            self.fusion = nn.Sequential(
                nn.Conv2d(nir_hidden_dim, 3, kernel_size=1),  # NIR->RGB维度
                nn.Sigmoid()  # 注意力权重
            )
        else:  # add
            self.fusion = nn.Conv2d(nir_hidden_dim, 3, kernel_size=1)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        print(f"DualStreamAdapter: fusion={fusion_type}, nir_dim={nir_hidden_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4, H, W] - RGBNIR
        Returns:
            [B, 3, H, W]
        """
        rgb = x[:, :3, :, :]   # [B, 3, H, W]
        nir = x[:, 3:4, :, :]  # [B, 1, H, W]
        
        # RGB流
        rgb_feat = self.rgb_stream(rgb)
        
        # NIR流
        nir_feat = self.nir_stream(nir)  # [B, hidden, H, W]
        
        # 融合
        if self.fusion_type == 'concat':
            combined = torch.cat([rgb_feat, nir_feat], dim=1)
            out = self.fusion(combined)
        elif self.fusion_type == 'cross_attention':
            attention = self.fusion(nir_feat)  # [B, 3, H, W]
            out = rgb_feat + rgb_feat * attention  # 注意力加权
        else:  # add
            nir_contribution = self.fusion(nir_feat)
            out = rgb_feat + nir_contribution
        
        return out


class AdaptiveChannelMixer(nn.Module):
    """
    自适应通道混合器 - 创新方法4
    
    使用小型MLP学习通道间的非线性组合
    相比1x1卷积（线性变换），MLP可以学习更复杂的通道交互
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        hidden_ratio: int = 2,
        use_instance_norm: bool = True
    ):
        super().__init__()
        
        hidden_dim = in_channels * hidden_ratio
        
        # 通道混合MLP (逐像素应用)
        self.mixer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.InstanceNorm2d(hidden_dim) if use_instance_norm else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.InstanceNorm2d(hidden_dim) if use_instance_norm else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        )
        
        # 残差连接（如果维度匹配）
        self.use_residual = (in_channels >= out_channels)
        if self.use_residual:
            self.residual_proj = nn.Conv2d(3, out_channels, kernel_size=1)
            # 初始化残差投影为RGB通道
            with torch.no_grad():
                nn.init.zeros_(self.residual_proj.weight)
                self.residual_proj.weight[:3, :3, 0, 0] = torch.eye(3)
        
        # 初始化
        for m in self.mixer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        print(f"AdaptiveChannelMixer: {in_channels}->{out_channels}, hidden={hidden_dim}, residual={self.use_residual}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mixer(x)
        
        if self.use_residual:
            residual = self.residual_proj(x[:, :3, :, :])  # 只用前3通道做残差
            out = out + residual
        
        return out
