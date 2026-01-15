"""
修复版Ranking Loss v2
针对SatBird物种分布预测任务优化

关键设计:
1. 保持BCE为主损失保证回归精度 (MAE/MSE)
2. 辅助损失只做轻微的排序调整
3. 动态loss归一化确保尺度匹配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

eps = 1e-7


class SoftMarginRankLoss(nn.Module):
    """
    软Margin排名损失
    只轻微调整预测，不破坏BCE学到的回归能力
    """
    def __init__(self, top_k=30, margin=0.05, temperature=1.0):
        super().__init__()
        self.top_k = top_k
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, logits, targets):
        batch_size, num_classes = logits.shape
        device = logits.device
        k = min(self.top_k, num_classes)
        
        # 获取真实的top-k索引和分数
        _, true_topk_idx = torch.topk(targets, k, dim=-1)
        topk_logits = torch.gather(logits, 1, true_topk_idx)
        
        # 获取预测的top-k索引
        _, pred_topk_idx = torch.topk(logits, k, dim=-1)
        
        # 计算预测top-k与真实top-k的重叠 (作为reward)
        # 越多重叠说明模型已经预测得很好了
        overlap = torch.zeros(batch_size, device=device)
        for b in range(batch_size):
            set_true = set(true_topk_idx[b].tolist())
            set_pred = set(pred_topk_idx[b].tolist())
            overlap[b] = len(set_true & set_pred) / k
        
        # 基于overlap的自适应权重
        # 重叠越少，loss权重越大
        adaptive_weight = 1.0 - overlap  # [B]
        
        # 对于真实top-k位置，希望logits高
        # 使用sigmoid让梯度平滑
        probs = torch.sigmoid(topk_logits / self.temperature)
        loss = -torch.log(probs + eps).mean(dim=1)  # [B]
        
        # 加权
        loss = (loss * adaptive_weight).mean()
        
        return loss


class ListwiseKLLoss(nn.Module):
    """
    Listwise KL散度损失
    只在top-k范围内做概率分布匹配
    """
    def __init__(self, top_k=30, temperature=2.0):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
    
    def forward(self, logits, targets):
        batch_size, num_classes = logits.shape
        k = min(self.top_k, num_classes)
        
        # 获取top-k
        _, top_idx = torch.topk(targets, k, dim=-1)
        top_logits = torch.gather(logits, 1, top_idx)
        top_targets = torch.gather(targets, 1, top_idx)
        
        # 在top-k内计算softmax分布
        pred_dist = F.softmax(top_logits / self.temperature, dim=-1)
        targ_dist = F.softmax(top_targets / self.temperature, dim=-1)
        
        # KL散度 (target || pred)
        kl = F.kl_div(pred_dist.log(), targ_dist, reduction='batchmean')
        
        return kl


class FixedCombinedLoss(nn.Module):
    """
    修复版组合损失 v2
    
    核心思路:
    - BCE占主导地位 (≥85%) - 保证回归精度
    - 排序损失只做轻微调整 (≤15%)
    - 动态归一化确保loss scale匹配
    """
    def __init__(
        self,
        num_classes,
        class_frequencies=None,
        presence_weight=1.5,
        absence_weight=1.0,
        ranking_weight=0.1,  # 更小的权重!
        ranking_top_k=30,
        use_class_weights=False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.presence_weight = presence_weight
        self.absence_weight = absence_weight
        self.ranking_weight = ranking_weight
        
        # 类别权重
        if class_frequencies is not None and use_class_weights:
            freqs = np.array(class_frequencies)
            weights = 1.0 / (freqs + 1)
            weights = weights / weights.sum() * len(weights)
            weights = np.clip(weights, 0.1, 10.0)
            self.register_buffer('class_weights', torch.tensor(weights, dtype=torch.float32))
        else:
            self.class_weights = None
        
        # 使用KL散度做轻微排序调整
        self.ranking_loss = ListwiseKLLoss(top_k=ranking_top_k, temperature=2.0)
        
        # EMA用于动态归一化
        self.register_buffer('bce_ema', torch.tensor(0.7))
        self.register_buffer('rank_ema', torch.tensor(0.1))
        self.ema_decay = 0.99
    
    def forward(self, logits, targets):
        # === BCE损失 (主损失) ===
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
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
        
        # === Ranking损失 (辅助损失) ===
        rank_loss = self.ranking_loss(logits, targets)
        
        # 动态归一化
        if self.training:
            with torch.no_grad():
                self.bce_ema = self.ema_decay * self.bce_ema + (1 - self.ema_decay) * bce_loss
                self.rank_ema = self.ema_decay * self.rank_ema + (1 - self.ema_decay) * rank_loss
        
        # 归一化rank_loss
        scale = (self.bce_ema / (self.rank_ema + eps)).clamp(0.1, 10.0)
        rank_loss_scaled = rank_loss * scale
        
        # === 组合 ===
        total_loss = (1 - self.ranking_weight) * bce_loss + self.ranking_weight * rank_loss_scaled
        
        return total_loss


class SimpleCombinedLoss(nn.Module):
    """
    简化版组合损失 - 最稳定的选择
    
    只使用加权BCE + 轻微的top-k惩罚
    不添加复杂的ranking loss
    """
    def __init__(
        self,
        num_classes,
        presence_weight=2.0,
        absence_weight=1.0,
        topk_bonus_weight=0.05,  # 非常小的权重
        top_k=30
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.presence_weight = presence_weight
        self.absence_weight = absence_weight
        self.topk_bonus_weight = topk_bonus_weight
        self.top_k = top_k
    
    def forward(self, logits, targets):
        batch_size, num_classes = logits.shape
        k = min(self.top_k, num_classes)
        
        # === 主损失: 加权BCE ===
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 对高target值位置给更高权重
        weight = torch.where(
            targets > 0.5,
            torch.full_like(targets, self.presence_weight),
            torch.full_like(targets, self.absence_weight)
        )
        
        # 额外对真实top-k位置加权
        _, true_topk_idx = torch.topk(targets, k, dim=-1)
        topk_mask = torch.zeros_like(targets).scatter_(1, true_topk_idx, 1.0)
        weight = weight + topk_mask * 0.5  # 对top-k额外+0.5权重
        
        bce_loss = (bce_loss * weight).mean()
        
        # === 辅助损失: Top-K bonus ===
        # 如果预测的top-k与真实top-k重叠，给予奖励(减少loss)
        _, pred_topk_idx = torch.topk(logits, k, dim=-1)
        
        # 计算重叠比例
        pred_onehot = torch.zeros_like(logits).scatter_(1, pred_topk_idx, 1)
        true_onehot = torch.zeros_like(targets).scatter_(1, true_topk_idx, 1)
        overlap = (pred_onehot * true_onehot).sum(dim=1) / k  # [B]
        
        # 重叠越多，bonus loss越小
        topk_loss = (1 - overlap).mean()
        
        # === 组合 ===
        total_loss = bce_loss + self.topk_bonus_weight * topk_loss
        
        return total_loss


# 测试
if __name__ == "__main__":
    batch_size = 4
    num_classes = 624
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.rand(batch_size, num_classes)
    
    print("Loss测试 v2:")
    print("-" * 40)
    
    soft_rank = SoftMarginRankLoss(top_k=30)
    print(f"SoftMarginRankLoss: {soft_rank(logits, targets):.4f}")
    
    listwise_kl = ListwiseKLLoss(top_k=30)
    print(f"ListwiseKLLoss: {listwise_kl(logits, targets):.4f}")
    
    fixed_combined = FixedCombinedLoss(num_classes=num_classes)
    print(f"FixedCombinedLoss: {fixed_combined(logits, targets):.4f}")
    
    simple_combined = SimpleCombinedLoss(num_classes=num_classes)
    print(f"SimpleCombinedLoss: {simple_combined(logits, targets):.4f}")
    
    # 对比标准BCE
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    print(f"Standard BCE: {bce:.4f}")
