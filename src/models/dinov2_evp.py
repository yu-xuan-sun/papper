"""
DINOv2 with EVP (Environment-aware Visual Prompt Tuning)
核心创新: 根据环境数据动态生成视觉提示

V3: 
- 低秩分解大幅减少参数量
- 只在最后N层使用EVP（减少对早期层干扰）
- 更强的门控机制（初始gate接近0）
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict

from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt
from src.models.env_aware_prompt import EnvironmentPromptGenerator


class Dinov2EVP(nn.Module):
    """
    带EVP的DINOv2模型
    在原有Dinov2AdapterPrompt基础上，将静态Prompt替换为环境感知的动态Prompt
    
    V3: 
    - 低秩分解，EVP模块仅增加~0.9M参数（只有4层）
    - 只在最后4层使用EVP，减少对早期特征的干扰
    - 初始gate≈0.05，让模型从baseline开始逐渐学习EVP贡献
    """
    
    def __init__(
        self,
        base_model: Dinov2AdapterPrompt,
        env_dim: int = 27,
        use_evp: bool = True,
        evp_hidden_dim: int = 256,
        evp_dropout: float = 0.1,
        evp_rank: int = 16,
        freeze_base: bool = False,
        evp_layers: Optional[List[int]] = None,  # 新增：指定EVP层
        evp_gate_init: float = -3.0  # 新增：gate初始值 (sigmoid(-3)≈0.047)
    ):
        super().__init__()
        
        self.base_model = base_model
        self.use_evp = use_evp
        self.env_dim = env_dim
        
        # 获取基础模型参数
        self.embed_dim = base_model.embed_dim
        self.prompt_len = base_model.prompt_len
        self.num_layers = len(base_model.dino.blocks)
        
        # 默认只在最后4层使用EVP
        if evp_layers is None:
            self.evp_layers = list(range(self.num_layers - 4, self.num_layers))
        else:
            self.evp_layers = evp_layers
        
        self.evp_layers_set = set(self.evp_layers)
        
        if use_evp:
            # 创建轻量级EVP生成器 (V3)
            self.evp_generator = EnvironmentPromptGenerator(
                env_dim=env_dim,
                prompt_len=self.prompt_len,
                embed_dim=self.embed_dim,
                hidden_dim=evp_hidden_dim,
                num_layers=self.num_layers,
                use_layer_specific=True,
                use_residual=True,
                use_gating=True,
                dropout=evp_dropout,
                rank=evp_rank,
                evp_layers=self.evp_layers,  # 只在指定层使用EVP
                gate_init_value=evp_gate_init  # 初始gate接近0
            )
            print(f"✨ EVP V3 enabled:")
            print(f"   env_dim={env_dim}, prompt_len={self.prompt_len}, rank={evp_rank}")
            print(f"   evp_layers={self.evp_layers} (only last {len(self.evp_layers)} layers)")
            print(f"   gate_init={evp_gate_init} (sigmoid={torch.sigmoid(torch.tensor(evp_gate_init)).item():.4f})")
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("🔒 Base model frozen, only training EVP")
        
        self._print_params()
    
    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 分别统计
        base_params = sum(p.numel() for p in self.base_model.parameters())
        evp_params = sum(p.numel() for p in self.evp_generator.parameters()) if self.use_evp else 0
        
        print(f"📊 EVP V3 Model Statistics:")
        print(f"   Base model: {base_params:,} ({base_params/1e6:.2f}M)")
        print(f"   EVP module: {evp_params:,} ({evp_params/1e6:.2f}M)")
        print(f"   Total: {total:,} ({total/1e6:.2f}M)")
        print(f"   Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    
    def _is_evp_layer(self, layer_idx: int) -> bool:
        """检查该层是否使用EVP"""
        return layer_idx in self.evp_layers_set
    
    def forward(
        self, 
        img: torch.Tensor, 
        env: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
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
        
        # Position embedding + dropout
        x = x + self.base_model.dino.pos_embed
        x = self.base_model.dino.pos_drop(x)
        
        # 处理环境特征
        if env is not None:
            env = torch.nan_to_num(env, nan=0.0)
        
        # 通过所有blocks
        for layer_idx, block in enumerate(self.base_model.dino.blocks):
            # 判断是否使用EVP
            use_evp_this_layer = (
                self.use_evp and 
                env is not None and 
                self._is_evp_layer(layer_idx)
            )
            
            if use_evp_this_layer:
                # EVP层：生成环境感知的动态prompts
                dynamic_prompts, _ = self.evp_generator(env, layer_idx=layer_idx)
                prompts_to_use = dynamic_prompts
            else:
                # 非EVP层：使用原有adapted_encoder中的静态prompts
                if self.base_model.adapted_encoder.use_layer_specific_prompts:
                    prompts_to_use = self.base_model.adapted_encoder.prompts[layer_idx](batch_size)
                else:
                    prompts_to_use = self.base_model.adapted_encoder.shared_prompts(batch_size)
            
            # 添加prompts到序列
            x_with_prompts = torch.cat([x, prompts_to_use], dim=1)
            
            # 通过block
            x_with_prompts = block(x_with_prompts)
            
            # 移除prompts
            x = x_with_prompts[:, :-self.prompt_len, :]
        
        # Final norm
        x = self.base_model.dino.norm(x)
        
        # 获取CLS token特征
        cls_features = x[:, 0]
        
        # 环境特征融合 (复用base_model的fusion)
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
        
        # 分类
        logits = self.base_model.classifier(fused_feat)
        
        return logits
    
    def get_evp_gate_values(self, env: torch.Tensor) -> Dict[int, torch.Tensor]:
        """获取所有EVP层的门控值"""
        if not self.use_evp:
            return {}
        
        _, all_gates = self.evp_generator.get_all_prompts(env)
        return all_gates


def create_dinov2_evp_model(config) -> Dinov2EVP:
    """
    创建DINOv2 EVP模型的工厂函数
    """
    module_cfg = config.experiment.module
    data_cfg = config.data
    
    # 获取环境特征维度
    env_dim = sum(data_cfg.env_var_sizes) if hasattr(data_cfg, 'env_var_sizes') else 27
    num_classes = data_cfg.total_species
    
    # 创建基础模型
    base_model = Dinov2AdapterPrompt(
        num_classes=num_classes,
        dino_model_name=getattr(module_cfg, 'dino_model', 'vit_base_patch14_dinov2.lvd142m'),
        pretrained_path=getattr(module_cfg, 'pretrained_path', 'checkpoints/dinov2_vitb14_pretrain.pth'),
        prompt_len=getattr(module_cfg, 'prompt_len', 40),
        bottleneck_dim=getattr(module_cfg, 'bottleneck_dim', 96),
        env_input_dim=env_dim,
        env_hidden_dim=getattr(module_cfg, 'env_hidden_dim', 2048),
        env_num_layers=getattr(module_cfg, 'env_num_layers', 6),
        use_env=True,
        fusion_type=getattr(module_cfg, 'fusion_type', 'adaptive_attention'),
        use_channel_adapter=getattr(module_cfg, 'use_channel_adapter', True),
        in_channels=getattr(module_cfg, 'in_channels', 4),
        channel_adapter_type=getattr(module_cfg, 'channel_adapter_type', 'learned'),
        freeze_backbone=getattr(module_cfg, 'freeze_backbone', True),
        unfreeze_last_n_blocks=getattr(module_cfg, 'unfreeze_last_n_blocks', 4),
        use_dropkey=getattr(module_cfg, 'use_dropkey', True),
        dropkey_rate=getattr(module_cfg, 'dropkey_rate', 0.15),
        hidden_dims=getattr(module_cfg, 'hidden_dims', [2048, 1024]),
        dropout=getattr(module_cfg, 'dropout', 0.15),
    )
    
    # 解析EVP层配置
    evp_layers_config = getattr(module_cfg, 'evp_layers', None)
    if evp_layers_config is not None:
        evp_layers = list(evp_layers_config)
    else:
        evp_layers = None  # 使用默认（最后4层）
    
    # 创建EVP包装器 (V3)
    evp_model = Dinov2EVP(
        base_model=base_model,
        env_dim=env_dim,
        use_evp=getattr(module_cfg, 'use_evp', True),
        evp_hidden_dim=getattr(module_cfg, 'evp_hidden_dim', 256),
        evp_dropout=getattr(module_cfg, 'evp_dropout', 0.1),
        evp_rank=getattr(module_cfg, 'evp_rank', 16),
        freeze_base=getattr(module_cfg, 'freeze_base_for_evp', False),
        evp_layers=evp_layers,
        evp_gate_init=getattr(module_cfg, 'evp_gate_init', -3.0),  # 新增配置
    )
    
    print("✅ Created DINOv2 EVP model (V3 - Last 4 Layers + Strong Gate)")
    
    return evp_model
