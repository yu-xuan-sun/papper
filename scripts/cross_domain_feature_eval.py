#!/usr/bin/env python
"""
跨域特征评估 - 使用源域backbone提取特征，目标域训练线性分类器
这是标准的迁移学习评估方法
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_features(model, dataloader, device='cuda'):
    """使用模型提取特征（不使用分类头）"""
    model.eval()
    model.to(device)
    
    all_features = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            x = batch['sat'].squeeze(1).to(device)
            y = batch['target']
            
            # 只提取backbone特征
            if hasattr(model, 'model'):
                # 获取backbone输出
                backbone = model.model
                if hasattr(backbone, 'get_features'):
                    features = backbone.get_features(x)
                else:
                    # 手动提取
                    if hasattr(backbone, 'channel_adapter') and backbone.channel_adapter is not None:
                        x = backbone.channel_adapter(x)
                    features = backbone.backbone(x)
            else:
                features = model(x)
            
            if isinstance(features, dict):
                features = features.get('features', features.get('pred'))
            
            all_features.append(features.cpu().numpy())
            all_targets.append(y.numpy())
    
    return np.concatenate(all_features), np.concatenate(all_targets)


def linear_probe_eval(train_features, train_targets, test_features, test_targets):
    """线性探测评估"""
    print(f"Training linear classifier on {train_features.shape[0]} samples...")
    print(f"Feature dim: {train_features.shape[1]}, Classes: {train_targets.shape[1]}")
    
    # 对每个类别训练一个二分类器
    n_classes = train_targets.shape[1]
    predictions = np.zeros((test_features.shape[0], n_classes))
    
    for i in tqdm(range(n_classes), desc="Training classifiers"):
        if train_targets[:, i].sum() < 5:  # 样本太少跳过
            continue
            
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
        clf.fit(train_features, train_targets[:, i])
        predictions[:, i] = clf.predict_proba(test_features)[:, 1]
    
    # 计算mAP
    valid_classes = train_targets.sum(axis=0) >= 5
    mAP = average_precision_score(
        test_targets[:, valid_classes], 
        predictions[:, valid_classes],
        average='macro'
    )
    
    return mAP, predictions


if __name__ == '__main__':
    print("Cross-domain feature evaluation")
    print("Usage: python cross_domain_feature_eval.py --source_ckpt <path> --source_data <name> --target_data <name>")
