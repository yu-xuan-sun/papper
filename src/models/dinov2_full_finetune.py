"""
DINOv2 Full Fine-tuning Model
Simple DINOv2 with full parameter training, no adapters, no prompts
Just DINOv2 backbone + classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Dict, Any
import os


def interpolate_pos_embed(pos_embed_checkpoint: torch.Tensor, pos_embed_model: torch.Tensor) -> torch.Tensor:
    """Interpolate position embeddings to match model size"""
    pos_embed_checkpoint_no_cls = pos_embed_checkpoint[:, 1:, :]
    pos_embed_model_no_cls = pos_embed_model[:, 1:, :]
    
    N_old = pos_embed_checkpoint_no_cls.shape[1]
    N_new = pos_embed_model_no_cls.shape[1]
    D = pos_embed_checkpoint.shape[2]
    
    old_size = int(N_old ** 0.5)
    new_size = int(N_new ** 0.5)
    
    print(f"    Interpolating: {old_size}x{old_size} ({N_old} patches) -> {new_size}x{new_size} ({N_new} patches)")
    
    pos_embed_checkpoint_grid = pos_embed_checkpoint_no_cls.reshape(1, old_size, old_size, D)
    pos_embed_checkpoint_grid = pos_embed_checkpoint_grid.permute(0, 3, 1, 2)
    
    pos_embed_new = F.interpolate(
        pos_embed_checkpoint_grid,
        size=(new_size, new_size),
        mode='bilinear',
        align_corners=False
    )
    
    pos_embed_new = pos_embed_new.permute(0, 2, 3, 1).reshape(1, N_new, D)
    cls_token_pos = pos_embed_model[:, :1, :]
    pos_embed_new = torch.cat([cls_token_pos, pos_embed_new], dim=1)
    
    return pos_embed_new


class Dinov2FullFinetune(nn.Module):
    """
    Simple DINOv2 model with full parameter fine-tuning
    No adapters, no prompts, just backbone + classifier
    """
    
    def __init__(
        self,
        num_species: int,
        dino_model: str = "vit_base_patch14_dinov2.lvd142m",
        pretrained_path: str = "checkpoints/dinov2_vitb14_pretrain.pth",
        hidden_dims: list = [1024],
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        img_size: int = 224,
    ):
        super().__init__()
        
        self.num_species = num_species
        self.dino_model_name = dino_model
        self.img_size = img_size
        
        # Create DINOv2 backbone via timm with correct image size
        print(f"Loading DINOv2 model: {dino_model} with img_size={img_size}")
        self.backbone = timm.create_model(
            dino_model,
            pretrained=False,
            num_classes=0,  # Remove classification head
            drop_path_rate=drop_path_rate,
            img_size=img_size,  # Specify image size
        )
        
        # Load pretrained weights
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)
        else:
            print(f"[Warning] Pretrained weights not found at {pretrained_path}, using random init")
        
        # Get embedding dimension
        self.embed_dim = self.backbone.embed_dim
        print(f"DINOv2 embedding dimension: {self.embed_dim}")
        
        # Build classification head
        self.classifier = self._build_classifier(self.embed_dim, num_species, hidden_dims, dropout)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def _load_pretrained_weights(self, pretrained_path: str):
        """Load pretrained weights with position embedding interpolation"""
        print(f"Loading pretrained weights from {pretrained_path}")
        
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Get model's current position embedding
        model_pos_embed = self.backbone.pos_embed
        
        # Check if position embedding sizes match
        if 'pos_embed' in state_dict:
            checkpoint_pos_embed = state_dict['pos_embed']
            if checkpoint_pos_embed.shape != model_pos_embed.shape:
                print(f"  Position embedding size mismatch:")
                print(f"    Checkpoint: {checkpoint_pos_embed.shape}")
                print(f"    Model:      {model_pos_embed.shape}")
                
                # Interpolate position embeddings
                state_dict['pos_embed'] = interpolate_pos_embed(checkpoint_pos_embed, model_pos_embed)
                print(f"  ✅ Successfully interpolated position embeddings")
        
        # Load weights (allow missing keys for classification head)
        missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")
        
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    def _build_classifier(self, input_dim: int, num_classes: int, hidden_dims: list, dropout: float):
        """Build MLP classification head"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, env: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Image tensor [B, C, H, W]
            env: Environmental features (ignored in this model)
        
        Returns:
            Dictionary with 'logits' key (NOT 'pred') so trainer applies sigmoid
        """
        # Handle 4-channel input by taking first 3 channels (RGB)
        if x.shape[1] == 4:
            x = x[:, :3, :, :]
        
        # Forward through backbone
        features = self.backbone(x)  # [B, embed_dim]
        
        # Classification
        logits = self.classifier(features)  # [B, num_species]
        
        # Return 'logits' key (not 'pred') so trainer will apply sigmoid
        return {
            'logits': logits,
            'features': features,
        }


def create_dinov2_full_finetune_model(opts) -> Dinov2FullFinetune:
    """Factory function to create model from config"""
    module_cfg = opts.experiment.module
    data_cfg = opts.data
    
    # Get number of species
    num_species = data_cfg.get('total_species', 670)
    
    # Get image size
    img_size = data_cfg.get('image_size', 224)
    
    # Model config
    dino_model = module_cfg.get('dino_model', 'vit_base_patch14_dinov2.lvd142m')
    pretrained_path = module_cfg.get('pretrained_path', 'checkpoints/dinov2_vitb14_pretrain.pth')
    hidden_dims = module_cfg.get('hidden_dims', [1024])
    dropout = module_cfg.get('dropout', 0.1)
    drop_path_rate = module_cfg.get('drop_path_rate', 0.1)
    
    model = Dinov2FullFinetune(
        num_species=num_species,
        dino_model=dino_model,
        pretrained_path=pretrained_path,
        hidden_dims=hidden_dims,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        img_size=img_size,
    )
    
    return model
