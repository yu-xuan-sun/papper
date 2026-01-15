#!/usr/bin/env python
"""
跨域迁移脚本 - 只迁移Backbone权重
用于物种数/环境变量不同的数据集之间迁移
"""

import torch
import argparse
from pathlib import Path


def transfer_backbone_weights(source_ckpt, target_config_species, target_env_dim):
    """
    从源checkpoint提取可迁移的权重
    
    Args:
        source_ckpt: 源模型checkpoint路径
        target_config_species: 目标数据集物种数
        target_env_dim: 目标数据集环境变量维度
    
    Returns:
        可迁移的state_dict
    """
    checkpoint = torch.load(source_ckpt, map_location='cpu')
    source_state = checkpoint['state_dict']
    
    # 需要排除的层 (与数据集相关)
    exclude_patterns = [
        'model.classifier',      # 分类头 (物种数相关)
        'model.fusion.env',      # 环境编码器 (环境变量维度相关)
        'model.env_encoder',     # 环境编码器
        'model.fusion.gate',     # 门控网络 (可能与环境相关)
    ]
    
    # 可迁移的层 (与数据集无关)
    transferable_patterns = [
        'model.backbone',        # DINOv2 backbone
        'model.channel_adapter', # 通道适配器
        'model.adapted_blocks',  # Adapter模块
        'model.prompt',          # Prompt模块
        'model.fusion.cross_attn', # Cross-attention (如果维度匹配)
        'model.fusion.norm',     # LayerNorm
    ]
    
    transferred_state = {}
    excluded_keys = []
    
    for key, value in source_state.items():
        should_exclude = any(pattern in key for pattern in exclude_patterns)
        
        if should_exclude:
            excluded_keys.append(key)
        else:
            transferred_state[key] = value
    
    print(f"=== 权重迁移统计 ===")
    print(f"源checkpoint总参数: {len(source_state)}")
    print(f"可迁移参数: {len(transferred_state)}")
    print(f"排除参数: {len(excluded_keys)}")
    
    print(f"\n排除的层 (需要重新初始化):")
    for key in excluded_keys[:10]:  # 只显示前10个
        print(f"  - {key}")
    if len(excluded_keys) > 10:
        print(f"  ... 共{len(excluded_keys)}个")
    
    return transferred_state, excluded_keys


def create_transfer_checkpoint(source_ckpt, output_path):
    """创建可迁移的checkpoint文件"""
    transferred_state, excluded = transfer_backbone_weights(source_ckpt, None, None)
    
    # 保存
    torch.save({
        'state_dict': transferred_state,
        'excluded_keys': excluded,
        'source_checkpoint': str(source_ckpt),
    }, output_path)
    
    print(f"\n✓ 已保存可迁移权重到: {output_path}")
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='源checkpoint路径')
    parser.add_argument('--output', type=str, default=None, help='输出路径')
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.source.replace('.ckpt', '_backbone_only.ckpt')
    
    create_transfer_checkpoint(args.source, args.output)
