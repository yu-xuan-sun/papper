# Src code of loss functions
import torch
from torchmetrics import Metric
import torch.nn as nn

eps = 1e-7


class CustomCrossEntropyLoss:
    def __init__(self, lambd_pres=1, lambd_abs=1, label_smoothing=0.0):
        super().__init__()
        # print('in my custom')
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres
        self.label_smoothing = label_smoothing

    def __call__(self, pred, target, reduction='mean'):
        """
        target: ground truth
        pred: prediction
        reduction: mean, sum, none
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # For multi-label case, smooth the targets
            # Convert hard 0/1 targets to soft targets
            # target = 1 -> 1 - label_smoothing
            # target = 0 -> label_smoothing
            target_smooth = target * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        else:
            target_smooth = target
        
        loss = (-self.lambd_pres * target_smooth * torch.log(pred + eps) - self.lambd_abs * (1 - target_smooth) * torch.log(1 - pred + eps))
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:  # reduction = None
            loss = loss

        return loss


class RMSLELoss(nn.Module):
    """
    root mean squared log error
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(target + 1)))


class CustomFocalLoss:
    def __init__(self, alpha=1, gamma=2):
        """
        build on top of binary cross entropy as implemented before
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, pred, target):
        ce_loss = (- target * torch.log(pred + eps) - (1 - target) * torch.log(1 - pred + eps)).mean()
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


class CustomCrossEntropy(Metric):
    def __init__(self, lambd_pres=1, lambd_abs=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres
        self.add_state("correct", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.FloatTensor([0]), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        target: target distribution
        pred: predicted distribution
        """
        self.correct += (-self.lambd_pres * target * torch.log(pred) - self.lambd_abs * (1 - target) * torch.log(1 - pred)).sum()
        self.total += target.numel()

    def compute(self):
        return (self.correct / self.total)


class WeightedCustomCrossEntropyLoss:
    def __init__(self, lambd_pres=1, lambd_abs=1, label_smoothing=0.0):
        super().__init__()
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres
        self.label_smoothing = label_smoothing

    def __call__(self, pred, target, weights=1):
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            target_smooth = target * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        else:
            target_smooth = target
        
        loss = (weights * (
                -self.lambd_pres * target_smooth * torch.log(pred + eps) - self.lambd_abs * (1 - target_smooth) * torch.log(1 - pred + eps))).mean()

        return loss


class MultiTaskLoss(nn.Module):
    """
    多任务损失: 结合交叉熵(分类)和MSE(回归)
    用于同时优化物种存在性判断和丰度预测
    """
    def __init__(self, ce_weight=0.6, mse_weight=0.4, 
                 lambd_pres=1.0, lambd_abs=1.0, label_smoothing=0.0):
        super().__init__()
        self.ce_loss = CustomCrossEntropyLoss(
            lambd_pres=lambd_pres, 
            lambd_abs=lambd_abs,
            label_smoothing=label_smoothing
        )
        self.mse_loss = nn.MSELoss()
        self.ce_weight = ce_weight
        self.mse_weight = mse_weight
    
    def __call__(self, pred, target):
        """
        pred: 模型预测 [B, num_species]
        target: 真实标签 [B, num_species]
        """
        ce = self.ce_loss(pred, target)
        mse = self.mse_loss(pred, target)
        total_loss = self.ce_weight * ce + self.mse_weight * mse
        return total_loss
