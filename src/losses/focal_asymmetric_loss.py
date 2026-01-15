"""
Focal-Asymmetric Hybrid Loss (FAHL)
结合Focal Loss和Asymmetric Loss处理long-tail多标签分类问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalAsymmetricHybridLoss(nn.Module):
    """
    混合损失函数，结合Focal Loss和Asymmetric Loss的优势
    - Focal Loss: 处理类别不平衡，关注难样本
    - Asymmetric Loss: 对正负样本使用不对称的处理策略
    - Class-Balanced: 基于样本数量的类别权重
    """
    def __init__(self, 
                 focal_gamma=2.0,
                 focal_alpha=0.25,
                 asl_gamma_neg=4.0,
                 asl_gamma_pos=1.0,
                 asl_clip=0.05,
                 focal_weight=0.6,
                 asl_weight=0.4,
                 use_class_balance=True,
                 cb_beta=0.9999):
        super().__init__()
        
        # Focal Loss参数
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
        # Asymmetric Loss参数
        self.asl_gamma_neg = asl_gamma_neg
        self.asl_gamma_pos = asl_gamma_pos
        self.asl_clip = asl_clip
        
        # 损失权重
        self.focal_weight = focal_weight
        self.asl_weight = asl_weight
        
        # Class-Balanced Loss
        self.use_class_balance = use_class_balance
        self.cb_beta = cb_beta
        
        # 类别权重 (需要通过set_class_weights设置)
        self.register_buffer('cb_weights', None)
    
    def set_class_weights(self, samples_per_class):
        """
        计算Class-Balanced Loss权重
        
        Args:
            samples_per_class: Tensor [num_classes], 每个类别的样本数
        
        Formula:
            effective_num = 1 - β^n
            weight = (1 - β) / effective_num
        """
        if not self.use_class_balance:
            return
        
        samples_per_class = samples_per_class.float()
        effective_num = 1.0 - torch.pow(self.cb_beta, samples_per_class)
        weights = (1.0 - self.cb_beta) / (effective_num + 1e-8)
        
        # 归一化权重
        weights = weights / weights.sum() * len(weights)
        self.cb_weights = weights
        
        print(f"[FAHL] Class-Balanced weights initialized:")
        print(f"  Range: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"  Mean: {weights.mean():.4f}, Std: {weights.std():.4f}")
    
    def focal_loss(self, inputs, targets):
        """
        Focal Loss: FL = -α_t * (1 - p_t)^γ * log(p_t)
        
        处理类别不平衡，让模型关注难分类样本
        
        Args:
            inputs: [B, C] logits
            targets: [B, C] binary labels
        Returns:
            loss: scalar
        """
        # BCE loss
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # 计算概率
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # 计算alpha_t
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        
        # Focal term
        focal_term = (1 - p_t) ** self.focal_gamma
        
        # Final loss
        loss = alpha_t * focal_term * BCE_loss
        
        # 应用类别权重
        if self.cb_weights is not None:
            loss = loss * self.cb_weights.unsqueeze(0).to(loss.device)
        
        return loss.mean()
    
    def asymmetric_loss(self, inputs, targets):
        """
        Asymmetric Loss: 对正负样本使用不对称的focusing策略
        
        - 正样本: 使用较小的gamma (更关注所有正样本)
        - 负样本: 使用较大的gamma + margin clipping (下采样easy negatives)
        
        Args:
            inputs: [B, C] logits
            targets: [B, C] binary labels
        Returns:
            loss: scalar
        """
        # 计算概率
        xs_pos = torch.sigmoid(inputs)
        xs_neg = 1.0 - xs_pos
        
        # 负样本使用margin clipping
        xs_neg = (xs_neg + self.asl_clip).clamp(max=1)
        
        # 计算log loss
        los_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        
        # 不对称focusing
        los_pos = los_pos * ((1 - xs_pos) ** self.asl_gamma_pos)
        los_neg = los_neg * (xs_pos ** self.asl_gamma_neg)
        
        # 总loss
        loss = -(los_pos + los_neg)
        
        return loss.mean()
    
    def forward(self, inputs, targets):
        """
        混合损失函数
        
        Args:
            inputs: [B, num_classes] logits
            targets: [B, num_classes] binary labels (0 or 1)
        
        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss components
        """
        # 计算两种loss
        focal = self.focal_loss(inputs, targets)
        asl = self.asymmetric_loss(inputs, targets)
        
        # 加权组合
        total_loss = self.focal_weight * focal + self.asl_weight * asl
        
        loss_dict = {
            'focal': focal.item(),
            'asl': asl.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


def compute_class_weights(dataset_path, num_classes):
    """
    从数据集计算每个类别的样本数，用于Class-Balanced Loss
    
    Args:
        dataset_path: 数据集路径
        num_classes: 类别数量
    
    Returns:
        samples_per_class: Tensor [num_classes]
    """
    import pandas as pd
    import os
    
    # 读取训练集CSV
    train_csv = os.path.join(dataset_path, "train_split.csv")
    if not os.path.exists(train_csv):
        print(f"[Warning] {train_csv} not found, using uniform weights")
        return torch.ones(num_classes)
    
    train_df = pd.read_csv(train_csv)
    
    # 读取物种列表
    species_list_path = os.path.join(dataset_path, "species_list.txt")
    if not os.path.exists(species_list_path):
        print(f"[Warning] {species_list_path} not found, using uniform weights")
        return torch.ones(num_classes)
    
    with open(species_list_path, 'r') as f:
        species_list = [line.strip() for line in f.readlines()]
    
    # 统计每个物种的样本数
    samples_per_class = []
    for species in species_list:
        if species in train_df.columns:
            count = (train_df[species] > 0).sum()
            samples_per_class.append(max(count, 1))  # 避免0
        else:
            samples_per_class.append(1)
    
    samples_per_class = torch.tensor(samples_per_class, dtype=torch.float32)
    
    print(f"[Class Weights] Computed for {len(samples_per_class)} classes")
    print(f"  Min samples: {samples_per_class.min().item():.0f}")
    print(f"  Max samples: {samples_per_class.max().item():.0f}")
    print(f"  Mean samples: {samples_per_class.mean().item():.1f}")
    
    return samples_per_class
