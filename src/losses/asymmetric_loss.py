"""
Asymmetric Loss for Multi-Label Classification
论文: Asymmetric Loss For Multi-Label Classification (ICCV 2021)
适用于SatBird的多标签不平衡分类任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification
    
    对正负样本使用不对称的焦点损失权重:
    - 正样本(稀有物种): 权重高，不惩罚
    - 负样本(常见情况): 简单样本权重低，难样本权重高
    
    Args:
        gamma_neg: 负样本focusing参数 (推荐4.0)
        gamma_pos: 正样本focusing参数 (推荐1.0)  
        clip: 概率裁剪阈值 (推荐0.05)
        eps: 数值稳定性参数
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Args:
            logits: 模型输出logits [batch_size, num_classes]
            targets: 目标标签 [batch_size, num_classes]
            reduction: 'mean', 'sum', or 'none'
        """
        # 计算sigmoid概率
        probs = torch.sigmoid(logits)
        
        # 概率裁剪(仅对负样本)
        probs_clipped = torch.where(
            targets == 1,
            probs,
            probs.add(self.clip).clamp(max=1)
        )
        
        # 计算交叉熵
        xs_pos = probs_clipped
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        
        xs_neg = 1 - probs_clipped  
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        
        # 应用不对称焦点权重
        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                asymmetric_w_pos = torch.pow(
                    1 - probs, self.gamma_pos
                ) if self.gamma_pos > 0 else 1.0
                
                asymmetric_w_neg = torch.pow(
                    probs, self.gamma_neg
                ) if self.gamma_neg > 0 else 1.0
            
            los_pos *= asymmetric_w_pos
            los_neg *= asymmetric_w_neg
        else:
            asymmetric_w_pos = torch.pow(1 - probs, self.gamma_pos)
            asymmetric_w_neg = torch.pow(probs, self.gamma_neg)
            los_pos *= asymmetric_w_pos
            los_neg *= asymmetric_w_neg
        
        # 合并损失
        loss = -(los_pos + los_neg)
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def extra_repr(self) -> str:
        return (
            f"gamma_neg={self.gamma_neg}, "
            f"gamma_pos={self.gamma_pos}, "
            f"clip={self.clip}"
        )


class AsymmetricLossOptimized(nn.Module):
    """优化版ASL，减少内存占用"""
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        pos_mask = (targets == 1)
        neg_mask = ~pos_mask
        
        loss = torch.zeros_like(probs)
        
        # 处理正样本
        if pos_mask.any():
            probs_pos = probs[pos_mask]
            weight_pos = torch.pow(1 - probs_pos, self.gamma_pos) if self.gamma_pos > 0 else 1.0
            loss[pos_mask] = -weight_pos * torch.log(probs_pos.clamp(min=self.eps))
        
        # 处理负样本
        if neg_mask.any():
            probs_neg = probs[neg_mask]
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)
            weight_neg = torch.pow(probs_neg, self.gamma_neg) if self.gamma_neg > 0 else 1.0
            loss[neg_mask] = -weight_neg * torch.log((1 - probs_neg).clamp(min=self.eps))
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    print("Testing AsymmetricLoss...")
    
    batch_size = 32
    num_classes = 670
    
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    loss = criterion(logits, targets)
    print(f"✅ ASL Loss: {loss.item():.4f}")
    
    loss.backward()
    print(f"✅ Gradients computed: {logits.grad.shape}")
    
    bce = nn.BCEWithLogitsLoss()
    bce_loss = bce(logits.detach(), targets)
    print(f"�� BCE Loss: {bce_loss.item():.4f} vs ASL: {loss.item():.4f}")
