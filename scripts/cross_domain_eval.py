#!/usr/bin/env python
"""
跨域泛化评估脚本
用于在不同数据集上测试已训练模型的迁移能力
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
import pytorch_lightning as pl
from sklearn.metrics import average_precision_score, f1_score


def load_checkpoint(ckpt_path, config_path):
    """加载训练好的模型"""
    from src.trainer.trainer import EbirdTask
    
    # 加载配置
    config = OmegaConf.load(config_path)
    
    # 创建模型
    task = EbirdTask(config)
    
    # 加载权重
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    task.load_state_dict(checkpoint['state_dict'])
    
    return task, config


def evaluate_cross_domain(
    model,
    source_config,
    target_dataset: str,
    target_data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4
):
    """
    跨域评估
    
    Args:
        model: 训练好的模型
        source_config: 源域配置
        target_dataset: 目标数据集名称
        target_data_dir: 目标数据目录
        batch_size: batch大小
        num_workers: 数据加载workers
    """
    from src.dataset.satbird import SatBirdDataModule
    
    # 修改配置指向目标数据集
    target_config = source_config.copy()
    target_config.data.data_dir = target_data_dir
    target_config.data.dataset_name = target_dataset
    target_config.data.files.base = target_data_dir
    target_config.data.loaders.batch_size = batch_size
    target_config.data.loaders.num_workers = num_workers
    
    # 加载目标数据集
    datamodule = SatBirdDataModule(target_config)
    datamodule.setup('test')
    
    # 评估
    model.eval()
    model.cuda()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            x = batch['sat'].squeeze(1).cuda()
            env = batch.get('env')
            if env is not None:
                env = env.cuda()
            y = batch['target'].cuda()
            
            outputs = model(x, env)
            if isinstance(outputs, dict):
                pred = outputs.get('pred', outputs.get('logits'))
            else:
                pred = outputs
            
            pred = torch.sigmoid(pred)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # 计算指标
    results = {}
    
    # mAP
    try:
        results['mAP'] = average_precision_score(targets, preds, average='macro')
    except:
        results['mAP'] = 0.0
    
    # Top-K accuracy
    for k in [10, 30, 50]:
        topk_acc = compute_topk_accuracy(preds, targets, k)
        results[f'top{k}_acc'] = topk_acc
    
    # Per-class AP
    per_class_ap = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            ap = average_precision_score(targets[:, i], preds[:, i])
            per_class_ap.append(ap)
    results['mean_per_class_ap'] = np.mean(per_class_ap) if per_class_ap else 0.0
    
    return results


def compute_topk_accuracy(preds, targets, k):
    """计算Top-K准确率"""
    n_samples = preds.shape[0]
    correct = 0
    
    for i in range(n_samples):
        pred_topk = np.argsort(preds[i])[-k:]
        target_species = np.where(targets[i] > 0)[0]
        
        if len(target_species) > 0:
            hits = len(set(pred_topk) & set(target_species))
            correct += hits / min(k, len(target_species))
    
    return correct / n_samples


def main():
    parser = argparse.ArgumentParser(description='Cross-domain evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to source config')
    parser.add_argument('--target_dataset', type=str, required=True, 
                        choices=['USA_summer', 'USA_winter', 'kenya'],
                        help='Target dataset name')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # 目标数据目录映射
    target_dirs = {
        'USA_summer': 'USA_summer',
        'USA_winter': 'USA_winter',
        'kenya': 'kenya'
    }
    
    print(f"Loading model from {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, args.config)
    
    print(f"Evaluating on {args.target_dataset}")
    results = evaluate_cross_domain(
        model=model,
        source_config=config,
        target_dataset=args.target_dataset,
        target_data_dir=target_dirs[args.target_dataset],
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print("\n" + "="*50)
    print(f"Cross-Domain Results: {args.target_dataset}")
    print("="*50)
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()
