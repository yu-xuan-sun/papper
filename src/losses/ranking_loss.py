"""
Ranking Loss for Multi-label Classification
优化topk指标的排序损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-7


class ListwiseLoss(nn.Module):
    """
    ListNet Loss - 优化整体排序质量
    适合提升topk性能
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes] 模型预测logits
            targets: [batch_size, num_classes] 二值标签
        """
        # 转换为概率分布
        pred_probs = F.softmax(logits / self.temperature, dim=-1)
        
        # 目标分布: 正类概率=1/k, 负类=0
        k = targets.sum(dim=-1, keepdim=True).clamp(min=1)
        target_probs = targets / k
        target_probs = F.normalize(target_probs, p=1, dim=-1)
        
        # KL散度
        loss = F.kl_div(
            pred_probs.log(),
            target_probs,
            reduction='batchmean'
        )
        
        return loss


class ApproxNDCGLoss(nn.Module):
    """
    近似NDCG Loss - 直接优化排序质量
    更适合topk场景
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size, num_classes]
        """
        batch_size = logits.size(0)
        
        # 计算排序位置的增益
        pred_scores = torch.sigmoid(logits / self.temperature)
        
        # 对每个样本计算NDCG
        loss = 0
        for i in range(batch_size):
            pred = pred_scores[i]
            target = targets[i]
            
            # 理想排序
            ideal_order = torch.argsort(target, descending=True)
            ideal_dcg = self._dcg(target[ideal_order])
            
            # 预测排序
            pred_order = torch.argsort(pred, descending=True)
            pred_dcg = self._dcg(target[pred_order])
            
            # NDCG
            ndcg = pred_dcg / (ideal_dcg + 1e-8)
            loss += (1 - ndcg)
        
        return loss / batch_size
    
    def _dcg(self, relevances):
        """计算DCG"""
        positions = torch.arange(1, len(relevances) + 1, device=relevances.device)
        discounts = 1.0 / torch.log2(positions + 1)
        return (relevances * discounts).sum()


class CombinedLossWithRanking(nn.Module):
    """
    组合损失: CE + Ranking
    平衡分类精度和排序质量
    """
    def __init__(
        self,
        num_classes,
        label_smoothing=0.05,
        presence_weight=1.2,
        absence_weight=1.0,
        ranking_weight=0.2,
        ranking_type='listwise'
    ):
        super().__init__()
        
        # 分类损失
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.label_smoothing = label_smoothing
        self.presence_weight = presence_weight
        self.absence_weight = absence_weight
        
        # 排序损失
        self.ranking_weight = ranking_weight
        if ranking_type == 'listwise':
            self.ranking_loss = ListwiseLoss(temperature=1.0)
        else:
            self.ranking_loss = ApproxNDCGLoss(temperature=1.0)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size, num_classes]
        """
        # Label smoothing
        if self.label_smoothing > 0:
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets
        
        # 分类损失
        ce_loss = self.ce_loss(logits, targets_smooth)
        
        # 类别权重
        weights = torch.where(
            targets == 1,
            torch.tensor(self.presence_weight, device=targets.device),
            torch.tensor(self.absence_weight, device=targets.device)
        )
        ce_loss = (ce_loss * weights).mean()
        
        # 排序损失
        rank_loss = self.ranking_loss(logits, targets)
        
        # 组合
        total_loss = ce_loss + self.ranking_weight * rank_loss
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'rank_loss': rank_loss.item(),
            'total_loss': total_loss.item()
        }
