"""
Domain Adaptation for Cross-Region Species Distribution Modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy (MMD) Loss"""
    
    def __init__(self, kernel_type: str = 'rbf', kernel_mul: float = 2.0, kernel_num: int = 5):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
    
    def gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_source = source.size(0)
        n_target = target.size(0)
        n_total = n_source + n_target
        
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0)
        total1 = total.unsqueeze(1)
        L2_distance = ((total0 - total1) ** 2).sum(dim=2)
        
        bandwidth = torch.sum(L2_distance) / (n_total ** 2 - n_total)
        bandwidth = bandwidth.clamp(min=1e-8)
        
        bandwidth_list = [
            bandwidth * (self.kernel_mul ** (i - self.kernel_num // 2))
            for i in range(self.kernel_num)
        ]
        
        kernel_val = sum([
            torch.exp(-L2_distance / bw.clamp(min=1e-8)) 
            for bw in bandwidth_list
        ]) / self.kernel_num
        
        return kernel_val
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_source = source.size(0)
        n_target = target.size(0)
        
        if n_source == 0 or n_target == 0:
            return torch.tensor(0.0, device=source.device)
        
        kernels = self.gaussian_kernel(source, target)
        
        XX = kernels[:n_source, :n_source]
        YY = kernels[n_source:, n_source:]
        XY = kernels[:n_source, n_source:]
        
        mmd = (XX.sum() / (n_source * n_source) + 
               YY.sum() / (n_target * n_target) - 
               2 * XY.sum() / (n_source * n_target))
        
        return mmd.clamp(min=0)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class DomainDiscriminator(nn.Module):
    """域判别器"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


class DomainAdaptiveModule(nn.Module):
    """域适应模块"""
    
    def __init__(
        self,
        feature_dim: int = 768,
        use_mmd: bool = True,
        use_adversarial: bool = True,
        mmd_weight: float = 0.1,
        adv_weight: float = 0.1
    ):
        super().__init__()
        
        self.use_mmd = use_mmd
        self.use_adversarial = use_adversarial
        self.mmd_weight = mmd_weight
        self.adv_weight = adv_weight
        self.lambda_ = 1.0
        
        if use_mmd:
            self.mmd_loss = MMDLoss()
            print(f"✓ MMD Loss enabled (weight={mmd_weight})")
        
        if use_adversarial:
            self.domain_discriminator = DomainDiscriminator(feature_dim)
            print(f"✓ Adversarial DA enabled (weight={adv_weight})")
    
    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_
    
    def compute_da_loss(
        self,
        source_feat: torch.Tensor,
        target_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=source_feat.device)
        
        if self.use_mmd:
            mmd = self.mmd_loss(source_feat, target_feat)
            loss_dict['mmd'] = mmd.item()
            total_loss = total_loss + self.mmd_weight * mmd
        
        if self.use_adversarial:
            source_feat_rev = GradientReversalFunction.apply(source_feat, self.lambda_)
            target_feat_rev = GradientReversalFunction.apply(target_feat, self.lambda_)
            
            source_domain = self.domain_discriminator(source_feat_rev)
            target_domain = self.domain_discriminator(target_feat_rev)
            
            source_labels = torch.zeros_like(source_domain)
            target_labels = torch.ones_like(target_domain)
            
            adv_loss = F.binary_cross_entropy_with_logits(
                torch.cat([source_domain, target_domain], dim=0),
                torch.cat([source_labels, target_labels], dim=0)
            )
            
            loss_dict['adv'] = adv_loss.item()
            total_loss = total_loss + self.adv_weight * adv_loss
        
        loss_dict['total_da'] = total_loss.item()
        return total_loss, loss_dict


class LambdaScheduler:
    """GRL lambda调度器"""
    
    def __init__(self, gamma: float = 10.0, max_epochs: int = 100):
        self.gamma = gamma
        self.max_epochs = max_epochs
    
    def get_lambda(self, epoch: int) -> float:
        p = epoch / self.max_epochs
        return 2.0 / (1.0 + np.exp(-self.gamma * p)) - 1.0


if __name__ == '__main__':
    print("Testing Domain Adaptation modules...")
    
    mmd = MMDLoss()
    x1 = torch.randn(32, 768)
    x2 = torch.randn(32, 768) + 1.0
    loss = mmd(x1, x2)
    print(f"✓ MMD Loss: {loss.item():.6f}")
    
    disc = DomainDiscriminator(input_dim=768)
    pred = disc(x1)
    print(f"✓ Domain prediction shape: {pred.shape}")
    
    da = DomainAdaptiveModule(feature_dim=768)
    da_loss, loss_dict = da.compute_da_loss(x1, x2)
    print(f"✓ DA Loss: {da_loss.item():.6f}")
    
    print("✅ All tests passed!")
