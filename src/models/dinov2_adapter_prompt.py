"""
DINOv2 with Adapter and Prompt Tuning for Bird Species Classification
Based on Parameter-Efficient Fine-Tuning (PEFT) strategies
Inspired by WhisPAr architecture

Key Features:
- Adapter modules in each transformer block
- Learnable prompt tokens at each layer
- Frozen backbone with efficient tuning
- Multi-modal fusion (satellite + environmental)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple, Dict, Any, List
import math

# Import V6 enhanced modules
try:
    from src.models.adaptive_fusion import AdaptiveEnvironmentalFusion
    ADAPTIVE_FUSION_AVAILABLE = True
except ImportError:
    ADAPTIVE_FUSION_AVAILABLE = False
    print("[Warning] AdaptiveEnvironmentalFusion not available, using default fusion")

try:
    from src.models.dropkey import DropKey
    DROPKEY_AVAILABLE = True
except ImportError:
    DROPKEY_AVAILABLE = False
    print("[Warning] DropKey not available, using standard Dropout")

# Import Channel Adapter for multi-channel input
try:
    from src.models.channel_adapter import (ChannelAdapter, NIREnhancedAdapter,
                                              SpectralAttentionAdapter, SpectralIndexAdapter,
                                              DualStreamAdapter, AdaptiveChannelMixer)
    CHANNEL_ADAPTER_AVAILABLE = True
except ImportError:
    CHANNEL_ADAPTER_AVAILABLE = False
    print("[Warning] ChannelAdapter not available")





def interpolate_pos_embed(pos_embed_checkpoint: torch.Tensor, pos_embed_model: torch.Tensor) -> torch.Tensor:
    """
    Interpolate position embeddings to match model size
    
    Args:
        pos_embed_checkpoint: Position embeddings from checkpoint [1, N_old, D]
        pos_embed_model: Position embeddings from model [1, N_new, D]
    Returns:
        Interpolated position embeddings [1, N_new, D]
    """
    # Remove CLS token
    pos_embed_checkpoint_no_cls = pos_embed_checkpoint[:, 1:, :]
    pos_embed_model_no_cls = pos_embed_model[:, 1:, :]
    
    # Get dimensions
    N_old = pos_embed_checkpoint_no_cls.shape[1]
    N_new = pos_embed_model_no_cls.shape[1]
    D = pos_embed_checkpoint.shape[2]
    
    # Calculate grid sizes
    old_size = int(N_old ** 0.5)
    new_size = int(N_new ** 0.5)
    
    print(f"    Interpolating: {old_size}x{old_size} ({N_old} patches) -> {new_size}x{new_size} ({N_new} patches)")
    
    # Reshape to 2D grid [1, old_size, old_size, D]
    pos_embed_checkpoint_grid = pos_embed_checkpoint_no_cls.reshape(1, old_size, old_size, D)
    # Permute to [1, D, old_size, old_size] for F.interpolate
    pos_embed_checkpoint_grid = pos_embed_checkpoint_grid.permute(0, 3, 1, 2)
    
    # Interpolate using bilinear interpolation
    pos_embed_new = F.interpolate(
        pos_embed_checkpoint_grid,
        size=(new_size, new_size),
        mode='bilinear',
        align_corners=False
    )
    
    # Permute back to [1, new_size, new_size, D] and reshape to [1, N_new, D]
    pos_embed_new = pos_embed_new.permute(0, 2, 3, 1).reshape(1, N_new, D)
    
    # Add back CLS token (use original model's CLS token position embedding)
    cls_token_pos = pos_embed_model[:, :1, :]
    pos_embed_new = torch.cat([cls_token_pos, pos_embed_new], dim=1)
    
    return pos_embed_new


class Adapter(nn.Module):
    """
    Bottleneck adapter module for parameter-efficient fine-tuning
    """
    def __init__(self, input_dim: int, bottleneck_dim: int, dropout: float = 0.1):
        super(Adapter, self).__init__()
        self.bottleneck_dim = bottleneck_dim
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.nonlinearity = nn.GELU()  # Use GELU for better performance
        self.dropout = nn.Dropout(dropout)
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        
        # Careful initialization for stability
        torch.nn.init.normal_(self.down_project.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.up_project.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.down_project.bias)
        torch.nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_project(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.up_project(x)
        x = x + residual  # Residual connection
        return x


class PromptTokens(nn.Module):
    """
    Learnable prompt tokens prepended to input sequence
    """
    def __init__(self, prompt_len: int, embed_dim: int):
        super(PromptTokens, self).__init__()
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        # Initialize prompts with small random values
        self.prompts = nn.Parameter(torch.randn(1, prompt_len, embed_dim) * 0.02)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        # Expand prompts for the batch
        return self.prompts.expand(batch_size, -1, -1)


class AdaptedTransformerBlock(nn.Module):
    """
    Transformer block with adapter modules and prompt support
    """
    def __init__(
        self,
        original_block: nn.Module,
        input_dim: int,
        bottleneck_dim: int,
        adapter_dropout: float = 0.1,
        use_adapter_attn: bool = True,
        use_adapter_mlp: bool = True
    ):
        super(AdaptedTransformerBlock, self).__init__()
        
        # Keep original components frozen
        self.original_block = original_block
        for param in self.original_block.parameters():
            param.requires_grad = False
        
        # Add adapters
        self.use_adapter_attn = use_adapter_attn
        self.use_adapter_mlp = use_adapter_mlp
        
        if use_adapter_attn:
            self.attn_adapter = Adapter(input_dim, bottleneck_dim, adapter_dropout)
        if use_adapter_mlp:
            self.mlp_adapter = Adapter(input_dim, bottleneck_dim, adapter_dropout)
        
        # Unfreeze layer norms for better adaptation
        if hasattr(original_block, 'norm1'):
            for param in original_block.norm1.parameters():
                param.requires_grad = True
        if hasattr(original_block, 'norm2'):
            for param in original_block.norm2.parameters():
                param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DINOv2 uses pre-norm architecture
        if hasattr(self.original_block, 'norm1') and hasattr(self.original_block, 'attn'):
            # Attention path
            x_norm = self.original_block.norm1(x)
            attn_out = self.original_block.attn(x_norm)
            if self.use_adapter_attn:
                attn_out = self.attn_adapter(attn_out)
            x = x + attn_out
            
            # MLP path
            x_norm = self.original_block.norm2(x)
            mlp_out = self.original_block.mlp(x_norm)
            if self.use_adapter_mlp:
                mlp_out = self.mlp_adapter(mlp_out)
            x = x + mlp_out
        else:
            # Fallback: use original block as-is
            x = self.original_block(x)
        
        return x


class PromptAdaptedEncoder(nn.Module):
    """
    Vision Transformer encoder with prompt tuning and adapters
    """
    def __init__(
        self,
        vit_blocks: nn.ModuleList,
        prompt_len: int,
        embed_dim: int,
        bottleneck_dim: int,
        adapter_layers: Optional[List[int]] = None,
        adapter_dropout: float = 0.1,
        use_layer_specific_prompts: bool = True
    ):
        super(PromptAdaptedEncoder, self).__init__()
        
        self.num_blocks = len(vit_blocks)
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        self.use_layer_specific_prompts = use_layer_specific_prompts
        
        # Determine which layers get adapters (default: all)
        if adapter_layers is None:
            adapter_layers = list(range(self.num_blocks))
        self.adapter_layers = adapter_layers
        
        # Wrap each block with adapters
        self.blocks = nn.ModuleList()
        for i, block in enumerate(vit_blocks):
            if i in adapter_layers:
                adapted_block = AdaptedTransformerBlock(
                    block, embed_dim, bottleneck_dim, adapter_dropout
                )
            else:
                # Keep original block frozen
                for param in block.parameters():
                    param.requires_grad = False
                adapted_block = block
            self.blocks.append(adapted_block)
        
        # Layer-specific prompts or shared prompts
        if use_layer_specific_prompts:
            # Each layer has its own prompt tokens
            self.prompts = nn.ModuleList([
                PromptTokens(prompt_len, embed_dim) for _ in range(self.num_blocks)
            ])
        else:
            # Shared prompts across layers
            self.shared_prompts = PromptTokens(prompt_len, embed_dim)


class EnhancedEnvEncoder(nn.Module):
    """
    Enhanced environmental feature encoder with residual connections
    """
    def __init__(
        self,
        input_dim: int = 27,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super(EnhancedEnvEncoder, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer outputs the target dimension
                layers.extend([
                    nn.Linear(current_dim, output_dim),
                    nn.LayerNorm(output_dim)
                ])
            else:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
                current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CrossModalFusion(nn.Module):
    """
    Cross-modal fusion with gated attention mechanism
    """
    def __init__(self, dim: int, num_heads: int = 12, dropout: float = 0.1):
        super(CrossModalFusion, self).__init__()
        
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(
        self,
        visual_feat: torch.Tensor,
        env_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual_feat: [B, D]
            env_feat: [B, D]
        Returns:
            fused_feat: [B, D]
        """
        # Reshape for multi-head attention
        B = visual_feat.size(0)
        
        # Query from visual, Key/Value from env
        q = self.q_proj(visual_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(env_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(env_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).contiguous().view(B, self.dim)
        out = self.out_proj(out)
        out = self.norm1(out + visual_feat)
        
        # Gated fusion
        concat_feat = torch.cat([visual_feat, env_feat], dim=1)
        gate = self.gate(concat_feat)
        fused = gate * out + (1 - gate) * visual_feat
        fused = self.norm2(fused)
        
        return fused


class Dinov2AdapterPrompt(nn.Module):
    """
    DINOv2 with Adapter and Prompt Tuning for multi-modal bird classification
    """
    def __init__(
        self,
        num_classes: int,
        dino_model_name: str = "vit_base_patch14_dinov2.lvd142m",
        pretrained_path: Optional[str] = None,
        # Prompt & Adapter settings
        prompt_len: int = 10,
        bottleneck_dim: int = 64,
        adapter_layers: Optional[List[int]] = None,
        adapter_dropout: float = 0.1,
        use_layer_specific_prompts: bool = True,
        # Environmental encoder settings
        env_input_dim: int = 27,
        env_hidden_dim: int = 512,
        env_num_layers: int = 3,
        use_env: bool = True,
        # Fusion settings
        fusion_type: str = "cross_attention",
        # Classifier settings
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        # DropKey regularization (V6 Enhanced)
        use_dropkey: bool = False,
        dropkey_rate: float = 0.1,
        # Inference-time calibration
        logit_temperature: float = 1.0,
        # Channel Adapter settings (for 4-channel input)
        use_channel_adapter: bool = False,
        in_channels: int = 3,
        channel_adapter_type: str = "learned",  # "learned" | "nir_enhanced"
        # Training settings
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0
    ):
        super(Dinov2AdapterPrompt, self).__init__()
        
        self.num_classes = num_classes
        self.use_env = use_env
        self.fusion_type = fusion_type
        self.prompt_len = prompt_len
        self.logit_temperature = float(logit_temperature)
        self.use_channel_adapter = use_channel_adapter
        self.in_channels = in_channels
        
        # Initialize channel adapter for multi-channel input
        if use_channel_adapter and in_channels != 3:
            if CHANNEL_ADAPTER_AVAILABLE:
                # 支持多种通道适配器类型
                if channel_adapter_type == "nir_enhanced":
                    print(f"Using NIREnhancedAdapter for {in_channels}-channel input")
                    self.channel_adapter = NIREnhancedAdapter(nir_weight=0.3, learnable=True)
                elif channel_adapter_type == "spectral_attention":
                    print(f"Using SpectralAttentionAdapter for {in_channels}-channel input")
                    self.channel_adapter = SpectralAttentionAdapter(
                        in_channels=in_channels,
                        out_channels=3,
                        reduction=2,
                        use_spatial_attention=False
                    )
                elif channel_adapter_type == "spectral_index":
                    print(f"Using SpectralIndexAdapter for {in_channels}-channel input")
                    self.channel_adapter = SpectralIndexAdapter(
                        in_channels=in_channels,
                        out_channels=3,
                        num_indices=3,
                        learnable_indices=True
                    )
                elif channel_adapter_type == "dual_stream":
                    print(f"Using DualStreamAdapter for {in_channels}-channel input")
                    self.channel_adapter = DualStreamAdapter(
                        nir_hidden_dim=16,
                        fusion_type='cross_attention'
                    )
                elif channel_adapter_type == "adaptive_mixer":
                    print(f"Using AdaptiveChannelMixer for {in_channels}-channel input")
                    self.channel_adapter = AdaptiveChannelMixer(
                        in_channels=in_channels,
                        out_channels=3,
                        hidden_ratio=2,
                        use_instance_norm=True
                    )
                else:  # "learned" or default
                    print(f"Using ChannelAdapter: {in_channels} -> 3 channels")
                    self.channel_adapter = ChannelAdapter(
                        in_channels=in_channels,
                        out_channels=3,
                        pretrained_rgb_weights=None,
                        trainable=True
                    )
            else:
                print(f"ChannelAdapter requested but not available, using simple slice")
                self.channel_adapter = None
        else:
            self.channel_adapter = None
        
        # Load DINOv2 backbone
        print(f"Loading DINOv2 model: {dino_model_name}")
        try:
            if pretrained_path and pretrained_path.lower() != "none":
                self.dino = timm.create_model(
                    dino_model_name,
                    pretrained=False,
                    num_classes=0,
                    img_size=224
                )
                state_dict = torch.load(pretrained_path, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']

                # Interpolate position embeddings if size mismatch
                if "pos_embed" in state_dict:
                    pos_embed_checkpoint = state_dict["pos_embed"]
                    pos_embed_model = self.dino.pos_embed
                    if pos_embed_checkpoint.shape != pos_embed_model.shape:
                        print(f"  Position embedding size mismatch:")
                        print(f"    Checkpoint: {pos_embed_checkpoint.shape}")
                        print(f"    Model:      {pos_embed_model.shape}")
                        state_dict["pos_embed"] = interpolate_pos_embed(pos_embed_checkpoint, pos_embed_model)
                        print(f"  ✅ Successfully interpolated position embeddings")
                self.dino.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights from {pretrained_path}")
            else:
                self.dino = timm.create_model(
                    dino_model_name,
                    pretrained=True,
                    num_classes=0,
                    img_size=224
                )
                print("Loaded pretrained DINOv2 from timm")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating model without pretrained weights")
            self.dino = timm.create_model(dino_model_name, pretrained=False, num_classes=0, img_size=224)
        
        # Get model config
        self.embed_dim = self.dino.embed_dim
        self.num_patches = self.dino.patch_embed.num_patches
        
        # Freeze backbone initially
        if freeze_backbone:
            for param in self.dino.parameters():
                param.requires_grad = False
            print(f"Froze DINOv2 backbone")
        
        # Optionally unfreeze last N blocks
        if unfreeze_last_n_blocks > 0:
            num_blocks = len(self.dino.blocks)
            for i in range(num_blocks - unfreeze_last_n_blocks, num_blocks):
                for param in self.dino.blocks[i].parameters():
                    param.requires_grad = True
            print(f"Unfroze last {unfreeze_last_n_blocks} blocks")
        
        # Replace transformer blocks with adapted versions
        print(f"Adding adapters and prompts (bottleneck_dim={bottleneck_dim}, prompt_len={prompt_len})")
        self.adapted_encoder = PromptAdaptedEncoder(
            vit_blocks=self.dino.blocks,
            prompt_len=prompt_len,
            embed_dim=self.embed_dim,
            bottleneck_dim=bottleneck_dim,
            adapter_layers=adapter_layers,
            adapter_dropout=adapter_dropout,
            use_layer_specific_prompts=use_layer_specific_prompts
        )
        
        # Replace the blocks in dino
        self.dino.blocks = self.adapted_encoder.blocks

        # Apply DropKey to attention layers (V6 Enhanced)
        if use_dropkey and DROPKEY_AVAILABLE and dropkey_rate > 0:
            print(f"🔑 Applying DropKey(p={dropkey_rate}) to attention layers...")
            dropkey_count = 0
            for block in self.dino.blocks:
                # AdaptedTransformerBlock wraps the original block
                target_block = block.original_block if hasattr(block, 'original_block') else block
                
                if hasattr(target_block, 'attn') and hasattr(target_block.attn, 'attn_drop'):
                    # Replace attention dropout with DropKey
                    target_block.attn.attn_drop = DropKey(p=dropkey_rate)
                    dropkey_count += 1
            print(f"   ✓ Replaced attn_drop with DropKey in {dropkey_count} layers")
            print(f"   ✓ proj_drop and MLP dropout remain as standard Dropout")
        elif use_dropkey and not DROPKEY_AVAILABLE:
            print(f"⚠️  DropKey requested but not available, using standard Dropout")
        
        # Environmental encoder
        if self.use_env:
            print(f"Using environmental features (input_dim={env_input_dim})")
            
            # Fusion module - V6 Enhanced: Adaptive Fusion
            if fusion_type == "adaptive_attention" and ADAPTIVE_FUSION_AVAILABLE:
                print("✨ Using V6 Enhanced: AdaptiveEnvironmentalFusion")
                self.fusion = AdaptiveEnvironmentalFusion(
                    num_layers=env_num_layers,
                    img_dim=self.embed_dim,
                    env_dim=env_input_dim,
                    hidden_dim=env_hidden_dim,
                    num_heads=12,
                    dropout=dropout
                )
                self.env_encoder = None  # Fusion包含了环境编码
                classifier_input_dim = self.embed_dim
            elif fusion_type == "cross_attention" or fusion_type == "adaptive_attention":
                # 回退到标准fusion
                if fusion_type == "adaptive_attention":
                    print("[Warning] AdaptiveFusion not available, using CrossModalFusion")
                self.env_encoder = EnhancedEnvEncoder(
                    input_dim=env_input_dim,
                    hidden_dim=env_hidden_dim,
                    output_dim=self.embed_dim,
                    num_layers=env_num_layers,
                    dropout=dropout
                )
                self.fusion = CrossModalFusion(
                    dim=self.embed_dim,
                    num_heads=12,
                    dropout=dropout
                )
                classifier_input_dim = self.embed_dim
            elif fusion_type == "concat":
                self.env_encoder = EnhancedEnvEncoder(
                    input_dim=env_input_dim,
                    hidden_dim=env_hidden_dim,
                    output_dim=self.embed_dim,
                    num_layers=env_num_layers,
                    dropout=dropout
                )
                self.fusion = None
                classifier_input_dim = self.embed_dim * 2
            else:
                raise ValueError(f"Unknown fusion_type: {fusion_type}")
        else:
            self.env_encoder = None
            self.fusion = None
            classifier_input_dim = self.embed_dim

        if hidden_dims is None:
            hidden_dims = [2048, 1024]
        
        classifier_layers = []
        current_dim = classifier_input_dim
        
        for hidden_dim in hidden_dims:
            classifier_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        # Final output layer
        classifier_layers.append(nn.Linear(current_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Initialize classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        print(f"Classifier: {classifier_input_dim} -> {hidden_dims} -> {num_classes}")
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        """Print the number of trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def forward_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract visual features from DINOv2 with prompts"""
        batch_size = x.size(0)
        
        # Apply channel adapter for multi-channel input
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        elif self.in_channels > 3:
            # Simple fallback: use first 3 channels
            x = x[:, :3, :, :]
        
        # Patch embedding
        x = self.dino.patch_embed(x)
        
        # Add position embeddings and class token
        if self.dino.cls_token is not None:
            cls_token = self.dino.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        x = self.dino.pos_drop(x)
        
        # Pass through adapted transformer blocks with prompts
        for i, block in enumerate(self.dino.blocks):
            # Get layer-specific or shared prompts
            if self.adapted_encoder.use_layer_specific_prompts:
                prompts = self.adapted_encoder.prompts[i](batch_size)
            else:
                prompts = self.adapted_encoder.shared_prompts(batch_size)
            
            # Prepend prompts
            x_with_prompts = torch.cat([x, prompts], dim=1)
            
            # Pass through block
            x_with_prompts = block(x_with_prompts)
            
            # Remove prompts
            x = x_with_prompts[:, :-self.prompt_len, :]
        
        # Apply final norm
        x = self.dino.norm(x)
        
        # Extract CLS token
        cls_token = x[:, 0]
        
        return cls_token
    
    def forward(
        self,
        img: torch.Tensor,
        env: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            img: Input images [B, 3, H, W]
            env: Environmental features [B, env_dim] (optional)
        Returns:
            logits: [B, num_classes]
        """
        # Extract visual features
        visual_feat = self.forward_visual_features(img)
        
        # Process environmental features and fuse
        if self.use_env and env is not None:
            # V6 Enhanced: adaptive_attention uses raw env features
            if self.fusion_type == "adaptive_attention" and ADAPTIVE_FUSION_AVAILABLE:
                fused_feat = self.fusion(visual_feat, env)  # raw env
            elif self.fusion_type == "cross_attention" or self.fusion_type == "adaptive_attention":
                env_feat = self.env_encoder(env)
                fused_feat = self.fusion(visual_feat, env_feat)
            elif self.fusion_type == "concat":
                env_feat = self.env_encoder(env)
                fused_feat = torch.cat([visual_feat, env_feat], dim=1)
            else:
                raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        else:
            fused_feat = visual_feat
        
        # Classification
        logits = self.classifier(fused_feat)
        
        return logits


def create_dinov2_adapter_prompt_model(config) -> Dinov2AdapterPrompt:
    """Factory function to create model from config"""
    model_cfg = config.experiment.module
    
    # Extract parameters with defaults
    num_classes = config.data.total_species
    
    # Adapter & Prompt settings
    prompt_len = getattr(model_cfg, 'prompt_len', 10)
    bottleneck_dim = getattr(model_cfg, 'bottleneck_dim', 64)
    adapter_layers = getattr(model_cfg, 'adapter_layers', None)
    adapter_dropout = getattr(model_cfg, 'adapter_dropout', 0.1)
    use_layer_specific_prompts = getattr(model_cfg, 'use_layer_specific_prompts', True)
    
    # Environmental settings
    env_input_dim = sum(config.data.env_var_sizes) if hasattr(config.data, 'env_var_sizes') else len(config.data.env) if hasattr(config.data, 'env') else 0
    use_env = env_input_dim > 0
    env_hidden_dim = getattr(model_cfg, 'env_hidden_dim', 512)
    env_num_layers = getattr(model_cfg, 'env_num_layers', 3)
    
    # Fusion settings
    fusion_type = getattr(model_cfg, 'fusion_type', 'cross_attention')
    
    # Classifier settings
    hidden_dims = getattr(model_cfg, 'hidden_dims', [2048, 1024])
    dropout = getattr(model_cfg, 'dropout', 0.3)
    
    # Training settings
    
    # DropKey settings (V6 Enhanced)
    use_dropkey = getattr(model_cfg, 'use_dropkey', False)
    dropkey_rate = getattr(model_cfg, 'dropkey_rate', 0.1)
    
    freeze_backbone = getattr(model_cfg, 'freeze_backbone', True)
    unfreeze_last_n_blocks = getattr(model_cfg, 'unfreeze_last_n_blocks', 0)
    
    # Model name and pretrained path
    dino_model_name = getattr(model_cfg, 'dino_model', 'vit_base_patch14_dinov2.lvd142m')
    pretrained_path = getattr(model_cfg, 'pretrained_path', None)
    
    # Channel adapter settings (for RGBNIR 4-channel input)
    use_channel_adapter = getattr(model_cfg, 'use_channel_adapter', False)
    in_channels = getattr(model_cfg, 'in_channels', 3)
    channel_adapter_type = getattr(model_cfg, 'channel_adapter_type', 'learned')
    
    model = Dinov2AdapterPrompt(
        num_classes=num_classes,
        dino_model_name=dino_model_name,
        pretrained_path=pretrained_path,
        prompt_len=prompt_len,
        bottleneck_dim=bottleneck_dim,
        adapter_layers=adapter_layers,
        adapter_dropout=adapter_dropout,
        use_layer_specific_prompts=use_layer_specific_prompts,
        env_input_dim=env_input_dim,
        env_hidden_dim=env_hidden_dim,
        env_num_layers=env_num_layers,
        use_env=use_env,
        fusion_type=fusion_type,
        hidden_dims=hidden_dims,
        use_dropkey=use_dropkey,
        dropkey_rate=dropkey_rate,
        dropout=dropout,
        use_channel_adapter=use_channel_adapter,
        in_channels=in_channels,
        channel_adapter_type=channel_adapter_type,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks
    )
    
    return model


# Explicitly export symbols
__all__ = [
    'Dinov2AdapterPrompt',
    'create_dinov2_adapter_prompt_model',
    'ADAPTIVE_FUSION_AVAILABLE',
    'AdaptedTransformerBlock',
    'Adapter',
    'PromptTokens',
    'PromptAdaptedEncoder',
    'CrossModalFusion',
    'EnhancedEnvEncoder',
    'interpolate_pos_embed'
]
