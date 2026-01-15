"""
Environment-aware Visual Prompt Tuning (EVP)
核心创新: 根据环境数据动态生成视觉提示

V3: 使用低秩分解 + 只在指定层使用EVP + 更强门控机制
- 只在最后N层使用EVP（减少对早期层干扰）
- 使用更强的门控机制（初始gate接近0）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import math


class EnvironmentPromptGenerator(nn.Module):
    """
    环境感知提示生成器 (Environment-aware Prompt Generator, EPG)
    根据环境特征动态生成视觉prompts
    
    V3: 
    - 低秩分解大幅减少参数量
    - 支持只在指定层使用EVP
    - 更强的门控机制（初始gate接近0）
    """
    
    def __init__(
        self,
        env_dim: int = 27,
        prompt_len: int = 40,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 12,
        use_layer_specific: bool = True,
        use_residual: bool = True,
        use_gating: bool = True,
        dropout: float = 0.1,
        rank: int = 16,
        evp_layers: Optional[List[int]] = None,  # 指定使用EVP的层索引
        gate_init_value: float = -3.0  # 初始gate值 (sigmoid(-3)≈0.047)
    ):
        super().__init__()
        
        self.env_dim = env_dim
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_layer_specific = use_layer_specific
        self.use_residual = use_residual
        self.use_gating = use_gating
        self.rank = rank
        self.gate_init_value = gate_init_value
        
        # 默认只在最后4层使用EVP
        if evp_layers is None:
            self.evp_layers = list(range(num_layers - 4, num_layers))  # [8, 9, 10, 11] for 12 layers
        else:
            self.evp_layers = evp_layers
        
        self.evp_layers_set = set(self.evp_layers)
        print(f"   EVP active layers: {self.evp_layers} (only last {len(self.evp_layers)} layers)")
        
        # 环境特征编码器 (共享)
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # 只为EVP层创建生成器
        if use_layer_specific:
            self.prompt_projectors = nn.ModuleDict()
            self.feature_expanders = nn.ParameterDict()
            
            for layer_idx in self.evp_layers:
                self.prompt_projectors[str(layer_idx)] = nn.Linear(hidden_dim, rank * prompt_len)
                self.feature_expanders[str(layer_idx)] = nn.Parameter(
                    torch.randn(rank, embed_dim) * 0.02
                )
        else:
            self.prompt_projector = nn.Linear(hidden_dim, rank * prompt_len)
            self.feature_expander = nn.Parameter(torch.randn(rank, embed_dim) * 0.02)
        
        # 可学习的基础Prompts（只为EVP层创建）
        if use_residual:
            if use_layer_specific:
                self.base_prompts = nn.ParameterDict()
                for layer_idx in self.evp_layers:
                    self.base_prompts[str(layer_idx)] = nn.Parameter(
                        torch.randn(1, prompt_len, embed_dim) * 0.02
                    )
            else:
                self.base_prompts = nn.Parameter(
                    torch.randn(1, prompt_len, embed_dim) * 0.02
                )
        
        # 更强的门控网络（初始值接近0）
        if use_gating:
            num_evp_layers = len(self.evp_layers)
            self.gate_hidden = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
            )
            # 独立的gate偏置，可以精确控制初始值
            self.gate_linear = nn.Linear(hidden_dim // 4, num_evp_layers)
            
            # 初始化gate使其接近0
            # sigmoid(gate_init_value) ≈ 0 when gate_init_value << 0
            nn.init.zeros_(self.gate_linear.weight)
            nn.init.constant_(self.gate_linear.bias, gate_init_value)
            
            # 创建层索引到gate索引的映射
            self.layer_to_gate_idx = {layer: i for i, layer in enumerate(self.evp_layers)}
        
        self._init_weights()
        self._print_params()
    
    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and 'gate' not in name:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"   EnvironmentPromptGenerator V3: {total:,} params ({total/1e6:.2f}M)")
        print(f"   Initial gate value: sigmoid({self.gate_init_value:.2f}) = {torch.sigmoid(torch.tensor(self.gate_init_value)).item():.4f}")
    
    def is_evp_layer(self, layer_idx: int) -> bool:
        """检查该层是否使用EVP"""
        return layer_idx in self.evp_layers_set
    
    def forward(
        self, 
        env_features: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        生成环境感知的prompts
        
        Args:
            env_features: [B, env_dim] 环境特征
            layer_idx: 层索引
            
        Returns:
            prompts: [B, prompt_len, embed_dim] 或 None（如果该层不使用EVP）
            gate_value: [B, 1] 门控值 或 None
        """
        # 如果该层不使用EVP，返回None
        if layer_idx is not None and not self.is_evp_layer(layer_idx):
            return None, None
        
        batch_size = env_features.size(0)
        env_features = torch.nan_to_num(env_features, nan=0.0)
        
        # 编码环境特征
        env_encoded = self.env_encoder(env_features)  # [B, hidden_dim]
        
        # 低秩生成动态prompts
        if self.use_layer_specific:
            if layer_idx is None:
                raise ValueError("layer_idx is required when use_layer_specific=True")
            
            layer_key = str(layer_idx)
            low_rank = self.prompt_projectors[layer_key](env_encoded)
            low_rank = low_rank.view(batch_size, self.prompt_len, self.rank)
            dynamic_prompts = torch.matmul(low_rank, self.feature_expanders[layer_key])
        else:
            low_rank = self.prompt_projector(env_encoded)
            low_rank = low_rank.view(batch_size, self.prompt_len, self.rank)
            dynamic_prompts = torch.matmul(low_rank, self.feature_expander)
        
        # 计算门控值
        gate_value = None
        if self.use_gating:
            gate_hidden = self.gate_hidden(env_encoded)
            all_gates = torch.sigmoid(self.gate_linear(gate_hidden))  # [B, num_evp_layers]
            
            gate_idx = self.layer_to_gate_idx[layer_idx]
            gate_value = all_gates[:, gate_idx:gate_idx+1]  # [B, 1]
        
        # 残差连接
        if self.use_residual:
            if self.use_layer_specific:
                base = self.base_prompts[str(layer_idx)].expand(batch_size, -1, -1)
            else:
                base = self.base_prompts.expand(batch_size, -1, -1)
            
            if self.use_gating:
                gate_expanded = gate_value.unsqueeze(-1)  # [B, 1, 1]
                prompts = base + gate_expanded * dynamic_prompts
            else:
                prompts = base + dynamic_prompts
        else:
            if self.use_gating:
                prompts = gate_value.unsqueeze(-1) * dynamic_prompts
            else:
                prompts = dynamic_prompts
        
        return prompts, gate_value
    
    def get_all_prompts(
        self, 
        env_features: torch.Tensor
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """获取所有EVP层的prompts"""
        all_prompts = {}
        all_gates = {}
        
        for layer_idx in self.evp_layers:
            prompts, gate = self.forward(env_features, layer_idx)
            if prompts is not None:
                all_prompts[layer_idx] = prompts
            if gate is not None:
                all_gates[layer_idx] = gate
        
        return all_prompts, all_gates


class EnvironmentAwarePromptEncoder(nn.Module):
    """
    环境感知的Prompt编码器
    将EVP与Transformer blocks整合
    """
    
    def __init__(
        self,
        vit_blocks: nn.ModuleList,
        prompt_len: int = 40,
        embed_dim: int = 768,
        bottleneck_dim: int = 96,
        env_dim: int = 27,
        adapter_layers: Optional[List[int]] = None,
        adapter_dropout: float = 0.1,
        prompt_dropout: float = 0.1,
        use_residual_prompts: bool = True,
        use_gating: bool = True,
        rank: int = 16,
        evp_layers: Optional[List[int]] = None,
        gate_init_value: float = -3.0
    ):
        super().__init__()
        
        self.num_layers = len(vit_blocks)
        self.vit_blocks = vit_blocks
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        self.adapter_layers = adapter_layers if adapter_layers else list(range(self.num_layers))
        
        # 创建EVP生成器
        self.prompt_generator = EnvironmentPromptGenerator(
            env_dim=env_dim,
            prompt_len=prompt_len,
            embed_dim=embed_dim,
            hidden_dim=256,
            num_layers=self.num_layers,
            use_layer_specific=True,
            use_residual=use_residual_prompts,
            use_gating=use_gating,
            dropout=prompt_dropout,
            rank=rank,
            evp_layers=evp_layers,
            gate_init_value=gate_init_value
        )
        
        # Adapter模块 (optional)
        self.adapters = nn.ModuleList()
        for i in range(self.num_layers):
            if i in self.adapter_layers:
                self.adapters.append(
                    Adapter(
                        embed_dim=embed_dim,
                        bottleneck_dim=bottleneck_dim,
                        dropout=adapter_dropout
                    )
                )
            else:
                self.adapters.append(nn.Identity())
        
        self.prompt_dropout = nn.Dropout(prompt_dropout)
        
        print(f"✨ EnvironmentAwarePromptEncoder V3 initialized:")
        print(f"   - num_layers: {self.num_layers}")
        print(f"   - prompt_len: {prompt_len}")
        print(f"   - env_dim: {env_dim}")
        print(f"   - rank: {rank}")
        print(f"   - evp_layers: {self.prompt_generator.evp_layers}")
        print(f"   - gate_init_value: {gate_init_value}")
        print(f"   - adapter_layers: {self.adapter_layers}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        env_features: torch.Tensor,
        return_gate_values: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[int, torch.Tensor]]]:
        """前向传播"""
        gate_values = {} if return_gate_values else None
        
        for i, block in enumerate(self.vit_blocks):
            prompts, gate = self.prompt_generator(env_features, layer_idx=i)
            
            if prompts is not None:
                # EVP层：使用动态prompts
                prompts = self.prompt_dropout(prompts)
                
                if return_gate_values and gate is not None:
                    gate_values[i] = gate
                
                x_with_prompts = torch.cat([x, prompts], dim=1)
                x_with_prompts = block(x_with_prompts)
                x = x_with_prompts[:, :-self.prompt_len, :]
            else:
                # 非EVP层：不添加prompts，直接通过block
                x = block(x)
            
            if i in self.adapter_layers:
                x = x + self.adapters[i](x)
        
        return x, gate_values


class Adapter(nn.Module):
    """轻量级Adapter模块"""
    
    def __init__(
        self,
        embed_dim: int = 768,
        bottleneck_dim: int = 96,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.up_proj(self.act(self.down_proj(x))))


class EVPInterpretability:
    """EVP可解释性分析工具"""
    
    @staticmethod
    @torch.enable_grad()
    def compute_env_sensitivity(
        prompt_generator: EnvironmentPromptGenerator,
        env_features: torch.Tensor,
        layer_idx: int
    ) -> Optional[torch.Tensor]:
        if not prompt_generator.is_evp_layer(layer_idx):
            return None
        
        env_features = env_features.clone().detach().requires_grad_(True)
        prompts, _ = prompt_generator(env_features, layer_idx)
        if prompts is None:
            return None
        
        prompt_norm = prompts.norm(dim=(1, 2))
        prompt_norm.sum().backward()
        sensitivity = env_features.grad.abs()
        return sensitivity
    
    @staticmethod
    def analyze_gate_patterns(
        prompt_generator: EnvironmentPromptGenerator,
        env_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        _, all_gates = prompt_generator.get_all_prompts(env_features)
        
        if not all_gates:
            return {}
        
        # 转换为tensor
        gate_list = [all_gates[layer_idx] for layer_idx in sorted(all_gates.keys())]
        gates_tensor = torch.stack(gate_list, dim=1)  # [B, num_evp_layers, 1]
        
        return {
            'mean_gate': gates_tensor.mean(dim=0),
            'std_gate': gates_tensor.std(dim=0),
            'layer_gates': gates_tensor,
            'evp_layers': prompt_generator.evp_layers
        }


if __name__ == '__main__':
    print("="*60)
    print("Testing EVP Modules (V3 - Low Rank + Last 4 Layers + Strong Gate)")
    print("="*60)
    
    # 测试EnvironmentPromptGenerator
    print("\n1. Testing EnvironmentPromptGenerator...")
    epg = EnvironmentPromptGenerator(
        env_dim=27, prompt_len=40, embed_dim=768,
        hidden_dim=256, num_layers=12, 
        use_layer_specific=True, use_residual=True, use_gating=True,
        rank=16,
        evp_layers=None,  # 默认最后4层
        gate_init_value=-3.0  # 初始gate≈0.047
    )
    
    # 参数统计
    total_params = sum(p.numel() for p in epg.parameters())
    print(f"\n   Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    
    env = torch.randn(4, 27)
    
    # 测试非EVP层
    print("\n2. Testing non-EVP layer (layer 0)...")
    prompts, gate = epg(env, layer_idx=0)
    print(f"   Layer 0 (non-EVP): prompts={prompts}, gate={gate}")
    
    # 测试EVP层
    print("\n3. Testing EVP layer (layer 8)...")
    prompts, gate = epg(env, layer_idx=8)
    print(f"   Layer 8 (EVP): prompts shape={prompts.shape}")
    print(f"   Initial gate value: {gate.mean().item():.4f} (expected ≈0.047)")
    
    # 获取所有EVP prompts
    print("\n4. Testing get_all_prompts...")
    all_prompts, all_gates = epg.get_all_prompts(env)
    print(f"   Active layers: {list(all_prompts.keys())}")
    print(f"   Gate values: {[f'{all_gates[k].mean().item():.4f}' for k in sorted(all_gates.keys())]}")
    
    print("\n" + "="*60)
    print("✅ All EVP V3 tests passed!")
    print("="*60)
