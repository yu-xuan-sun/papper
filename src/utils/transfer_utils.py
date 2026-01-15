"""
跨域权重迁移工具
"""

import torch
from typing import Dict, List, Tuple


def load_transferable_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    exclude_patterns: List[str] = None,
    strict: bool = False
) -> Tuple[List[str], List[str]]:
    """
    加载可迁移的权重（跳过维度不匹配的层）
    
    Args:
        model: 目标模型
        checkpoint_path: 源checkpoint路径
        exclude_patterns: 要排除的层名模式
        strict: 是否严格匹配
    
    Returns:
        (loaded_keys, skipped_keys)
    """
    if exclude_patterns is None:
        exclude_patterns = ['classifier', 'fc', 'head']
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        source_state = checkpoint['state_dict']
    else:
        source_state = checkpoint
    
    model_state = model.state_dict()
    
    loaded_keys = []
    skipped_keys = []
    
    for key, value in source_state.items():
        # 检查是否在排除列表中
        should_exclude = any(pattern in key for pattern in exclude_patterns)
        
        if should_exclude:
            skipped_keys.append(f"{key} (excluded by pattern)")
            continue
        
        # 检查key是否存在
        if key not in model_state:
            skipped_keys.append(f"{key} (not in model)")
            continue
        
        # 检查维度是否匹配
        if value.shape != model_state[key].shape:
            skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state[key].shape})")
            continue
        
        # 加载权重
        model_state[key] = value
        loaded_keys.append(key)
    
    # 加载到模型
    model.load_state_dict(model_state, strict=False)
    
    print(f"\n=== 跨域权重迁移 ===")
    print(f"成功加载: {len(loaded_keys)} 个参数")
    print(f"跳过: {len(skipped_keys)} 个参数")
    
    if skipped_keys:
        print(f"\n跳过的参数 (前10个):")
        for key in skipped_keys[:10]:
            print(f"  - {key}")
        if len(skipped_keys) > 10:
            print(f"  ... 共 {len(skipped_keys)} 个")
    
    return loaded_keys, skipped_keys


def get_backbone_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    从checkpoint中提取backbone权重
    排除分类头和环境编码器
    """
    exclude_patterns = [
        'classifier',
        'fc', 
        'head',
        'env_encoder',
        'fusion.env',
        'fusion.gate',
    ]
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    source_state = checkpoint.get('state_dict', checkpoint)
    
    backbone_state = {}
    for key, value in source_state.items():
        if not any(pattern in key for pattern in exclude_patterns):
            backbone_state[key] = value
    
    return backbone_state
