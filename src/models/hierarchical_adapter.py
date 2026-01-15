"""
Hierarchical Multi-Scale Adapter (HMA)
分层多尺度适配器模块
"""

import torch
import torch.nn as nn


class Adapter(nn.Module):
    """
    基础Adapter模块 (Bottleneck架构)
    
    架构: input -> down_project -> GELU -> dropout -> up_project -> residual
    """
    def __init__(self, input_dim, bottleneck_dim, dropout=0.1):
        super().__init__()
        
        self.bottleneck_dim = bottleneck_dim
        
        # Down projection
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        
        # Activation
        self.nonlinearity = nn.GELU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Up projection
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """小方差初始化，确保训练初期稳定"""
        nn.init.normal_(self.down_project.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_project.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, D] 输入特征
        Returns:
            [B, N, D] 输出特征
        """
        residual = x
        
        # Bottleneck transformation
        x = self.down_project(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.up_project(x)
        
        # Residual connection
        x = x + residual
        
        return x


class HierarchicalMultiScaleAdapter(nn.Module):
    """
    分层多尺度Adapter
    
    设计思想:
    - Early layers (0-3): 小维度 (64) - 捕获低级视觉特征 (边缘、纹理)
    - Middle layers (4-7): 中等维度 (96) - 捕获中级语义特征 (部位形状)
    - Late layers (8-11): 大维度 (128) - 捕获高级判别特征 (物种特异性)
    
    优势:
    1. 参数效率: 早期层用更少参数
    2. 多尺度: 不同层捕获不同粒度的特征
    3. 灵活性: 可以针对不同层定制adapter
    """
    
    def __init__(self, 
                 embed_dim=768,
                 layer_configs=None,
                 dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 默认配置: 3个stage
        if layer_configs is None:
            layer_configs = {
                'early': {
                    'bottleneck_dim': 64,
                    'layers': [0, 1, 2, 3],
                    'description': 'Low-level visual features'
                },
                'middle': {
                    'bottleneck_dim': 96,
                    'layers': [4, 5, 6, 7],
                    'description': 'Mid-level semantic features'
                },
                'late': {
                    'bottleneck_dim': 128,
                    'layers': [8, 9, 10, 11],
                    'description': 'High-level discriminative features'
                }
            }
        
        self.layer_configs = layer_configs
        
        # 为每个层创建对应的adapter
        self.adapters = nn.ModuleDict()
        
        print("\n[HierarchicalMultiScaleAdapter] Creating adapters:")
        for stage_name, config in layer_configs.items():
            bottleneck_dim = config['bottleneck_dim']
            layer_ids = config['layers']
            description = config.get('description', '')
            
            print(f"  {stage_name}: dim={bottleneck_dim}, layers={layer_ids} - {description}")
            
            for layer_id in layer_ids:
                self.adapters[str(layer_id)] = Adapter(
                    input_dim=embed_dim,
                    bottleneck_dim=bottleneck_dim,
                    dropout=dropout
                )
        
        # 多尺度特征融合权重 (可选)
        # 早期:中期:后期 = 1:2:7
        self.use_fusion = False  # 简化版本先不用
        if self.use_fusion:
            self.fusion_weights = nn.Parameter(
                torch.tensor([0.1, 0.2, 0.7])
            )
        
        print(f"[HierarchicalMultiScaleAdapter] Total adapters: {len(self.adapters)}")
        self._print_params()
    
    def _print_params(self):
        """打印参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def forward(self, x, layer_id):
        """
        前向传播
        
        Args:
            x: [B, N, D] 输入特征
            layer_id: int, 当前层的ID
        
        Returns:
            [B, N, D] 输出特征
        """
        layer_key = str(layer_id)
        
        if layer_key in self.adapters:
            return self.adapters[layer_key](x)
        else:
            # 如果该层没有adapter，直接返回原始特征
            return x
    
    def get_adapter_dim(self, layer_id):
        """获取指定层的adapter维度"""
        layer_key = str(layer_id)
        if layer_key in self.adapters:
            return self.adapters[layer_key].bottleneck_dim
        return None


class DropKeyAttention(nn.Module):
    """
    DropKey: 在Attention中随机drop掉部分keys
    
    作用:
    1. 正则化: 防止过拟合
    2. 鲁棒性: 提高对部分特征缺失的鲁棒性
    3. 泛化性: 强制模型学习更diverse的特征表示
    
    实现:
    - 训练时: 随机mask掉部分keys (保留CLS token)
    - 推理时: 不使用dropout
    """
    
    def __init__(self, dropkey_rate=0.1):
        super().__init__()
        self.dropkey_rate = dropkey_rate
    
    def forward(self, x):
        """
        Args:
            x: [B, N, D] 输入tokens
        Returns:
            x: [B, N, D] 处理后的tokens
        """
        if not self.training or self.dropkey_rate <= 0:
            return x
        
        B, N, D = x.shape
        
        # 生成随机mask (保留概率 = 1 - drop_rate)
        keep_prob = 1.0 - self.dropkey_rate
        mask = torch.rand(B, N, device=x.device) < keep_prob
        
        # 永远保留CLS token (第0个位置)
        mask[:, 0] = True
        
        # 应用mask
        mask = mask.unsqueeze(-1).float()  # [B, N, 1]
        x = x * mask
        
        # 缩放补偿 (类似dropout)
        x = x / keep_prob
        
        return x


def create_hierarchical_adapter(config):
    """
    工厂函数: 从配置创建HierarchicalMultiScaleAdapter
    
    Args:
        config: 配置对象
    
    Returns:
        HierarchicalMultiScaleAdapter 实例
    """
    model_cfg = config.experiment.module
    
    # 读取配置
    embed_dim = 768  # DINOv2-Base的embed_dim
    
    # 读取层级配置
    if hasattr(model_cfg, 'adapter_configs') and model_cfg.adapter_configs:
        layer_configs = {}
        for stage_name, stage_cfg in model_cfg.adapter_configs.items():
            layer_configs[stage_name] = {
                'bottleneck_dim': stage_cfg.get('dim', 64),
                'layers': stage_cfg.get('layers', []),
                'description': stage_cfg.get('description', '')
            }
    else:
        # 使用默认配置
        layer_configs = None
    
    dropout = getattr(model_cfg, 'adapter_dropout', 0.1)
    
    return HierarchicalMultiScaleAdapter(
        embed_dim=embed_dim,
        layer_configs=layer_configs,
        dropout=dropout
    )
