"""
改进的Ranking Loss实现
针对Top-k指标优化

包含:
1. ListMLE Loss - 直接优化排序似然
2. ApproxNDCG Loss - 近似NDCG优化  
3. 类别权重 - 处理不平衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

eps = 1e-7


class ListMLELoss(nn.Module):
    """
    ListMLE Loss - 直接优化排序的似然
    比ListNet更直接地优化排序质量
    """
    def __init__(self, temperature=1.0, top_k=30):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes] 模型预测
            targets: [batch_size, num_classes] 真实标签
        """
        batch_size, num_classes = logits.shape
        device = logits.device
        
        # 获取真实的top-k物种索引
        k = min(self.top_k, num_classes)
        _, true_topk_idx = torch.topk(targets, k, dim=-1)
        
        # 获取这些位置的预测分数
        pred_scores = torch.gather(logits, 1, true_topk_idx) / self.temperature
        
        # ListMLE: 计算排列概率的负对数似然
        loss = torch.zeros(batch_size, device=device)
        for i in range(k):
            remaining_scores = pred_scores[:, i:]
            log_sum_exp = torch.logsumexp(remaining_scores, dim=-1)
            loss = loss + log_sum_exp - pred_scores[:, i]
        
        return loss.mean() / k


class ImprovedListwiseLoss(nn.Module):
    """
    改进的Listwise Loss
    使用KL散度但只关注top-k
    """
    def __init__(self, temperature=1.0, top_k=30):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
    
    def forward(self, logits, targets):
        batch_size, num_classes = logits.shape
        k = min(self.top_k, num_classes)
        
        # 只对top-k位置计算
        _, top_idx = torch.topk(targets, k, dim=-1)
        
        # 获取top-k的logits和targets
        top_logits = torch.gather(logits, 1, top_idx)
        top_targets = torch.gather(targets, 1, top_idx)
        
        # 在top-k范围内计算softmax
        pred_probs = F.softmax(top_logits / self.temperature, dim=-1)
        target_probs = F.softmax(top_targets / self.temperature, dim=-1)
        
        # KL散度
        loss = F.kl_div(
            pred_probs.log(),
            target_probs,
            reduction='batchmean'
        )
        
        return loss


class ImprovedCombinedLoss(nn.Module):
    """
    改进的组合损失
    BCE + ListMLE + 类别权重
    """
    def __init__(
        self,
        num_classes,
        class_frequencies=None,
        label_smoothing=0.05,
        presence_weight=1.5,
        absence_weight=1.0,
        ranking_weight=0.3,
        ranking_type='listmle',
        ranking_top_k=30,
        use_class_weights=True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.presence_weight = presence_weight
        self.absence_weight = absence_weight
        self.ranking_weight = ranking_weight
        self.use_class_weights = use_class_weights
        
        # 类别权重
        if class_frequencies is not None and use_class_weights:
            freqs = np.array(class_frequencies)
            weights = 1.0 / (freqs + 1)
            weights = weights / weights.sum() * len(weights)
            weights = np.clip(weights, 0.1, 10.0)
            self.register_buffer('class_weights', torch.tensor(weights, dtype=torch.float32))
        else:
            self.class_weights = None
        
        # 排序损失
        if ranking_type == 'listmle':
            self.ranking_loss = ListMLELoss(top_k=ranking_top_k)
        else:
            self.ranking_loss = ImprovedListwiseLoss(top_k=ranking_top_k)
    
    def forward(self, logits, targets):
        # Label smoothing
        if self.label_smoothing > 0:
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets
        
        # BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        
        # 正负样本权重
        pos_weight = torch.where(
            targets > 0.5,
            torch.full_like(targets, self.presence_weight),
            torch.full_like(targets, self.absence_weight)
        )
        bce_loss = bce_loss * pos_weight
        
        # 类别权重
        if self.class_weights is not None:
            bce_loss = bce_loss * self.class_weights.unsqueeze(0).to(bce_loss.device)
        
        bce_loss = bce_loss.mean()
        
        # 排序损失
        rank_loss = self.ranking_loss(logits, targets)
        
        # 组合
        total_loss = (1 - self.ranking_weight) * bce_loss + self.ranking_weight * rank_loss
        
        return total_loss


def load_species_frequencies(freq_path):
    """加载物种频率"""
    return np.load(freq_path)
