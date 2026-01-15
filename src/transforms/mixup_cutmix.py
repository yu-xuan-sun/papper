"""
Mixup and CutMix implementations for multi-label classification
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class MixupCutmix(nn.Module):
    """Combined Mixup and CutMix data augmentation"""
    
    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        switch_prob: float = 0.5,
        label_smoothing: float = 0.0,
        num_classes: int = 670
    ):
        super().__init__()
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        
    def forward(
        self, 
        images: torch.Tensor, 
        targets: torch.Tensor,
        env_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if np.random.rand() > self.prob:
            return images, targets, env_features
            
        if np.random.rand() < self.switch_prob:
            return self._mixup(images, targets, env_features)
        else:
            return self._cutmix(images, targets, env_features)
    
    def _mixup(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        env_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size = images.size(0)
        
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0
            
        index = torch.randperm(batch_size, device=images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        
        mixed_env = None
        if env_features is not None:
            mixed_env = lam * env_features + (1 - lam) * env_features[index]
        
        return mixed_images, mixed_targets, mixed_env
    
    def _cutmix(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        env_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size = images.size(0)
        _, _, H, W = images.shape
        
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1.0
        
        index = torch.randperm(batch_size, device=images.device)
        
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        
        mixed_env = None
        if env_features is not None:
            mixed_env = lam * env_features + (1 - lam) * env_features[index]
        
        return mixed_images, mixed_targets, mixed_env


class MixupOnly(nn.Module):
    """Mixup-only augmentation"""
    
    def __init__(
        self,
        alpha: float = 0.2,
        prob: float = 0.5,
        num_classes: int = 670
    ):
        super().__init__()
        self.alpha = alpha
        self.prob = prob
        self.num_classes = num_classes
        
    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        env_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if np.random.rand() > self.prob:
            return images, targets, env_features
            
        batch_size = images.size(0)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
            
        index = torch.randperm(batch_size, device=images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        
        mixed_env = None
        if env_features is not None:
            mixed_env = lam * env_features + (1 - lam) * env_features[index]
        
        return mixed_images, mixed_targets, mixed_env


class CutmixOnly(nn.Module):
    """CutMix-only augmentation"""
    
    def __init__(
        self,
        alpha: float = 1.0,
        prob: float = 0.5,
        num_classes: int = 670
    ):
        super().__init__()
        self.alpha = alpha
        self.prob = prob
        self.num_classes = num_classes
        
    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        env_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if np.random.rand() > self.prob:
            return images, targets, env_features
            
        batch_size = images.size(0)
        _, _, H, W = images.shape
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        index = torch.randperm(batch_size, device=images.device)
        
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        
        mixed_env = None
        if env_features is not None:
            mixed_env = lam * env_features + (1 - lam) * env_features[index]
        
        return mixed_images, mixed_targets, mixed_env


def build_mixup_cutmix(
    mode: str = "mixup_cutmix",
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    prob: float = 0.5,
    switch_prob: float = 0.5,
    num_classes: int = 670
) -> nn.Module:
    """Factory function to build Mixup/CutMix augmentation"""
    if mode == "mixup_cutmix":
        return MixupCutmix(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=prob,
            switch_prob=switch_prob,
            num_classes=num_classes
        )
    elif mode == "mixup":
        return MixupOnly(
            alpha=mixup_alpha,
            prob=prob,
            num_classes=num_classes
        )
    elif mode == "cutmix":
        return CutmixOnly(
            alpha=cutmix_alpha,
            prob=prob,
            num_classes=num_classes
        )
    elif mode == "none" or mode is None:
        return None
    else:
        raise ValueError(f"Unknown mixup_cutmix mode: {mode}")
