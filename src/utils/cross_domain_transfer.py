"""
跨域迁移学习工具
支持从一个数据集迁移到另一个数据集
处理物种数不同、环境变量维度不同的情况
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict


class CrossDomainTransfer:
    """跨域迁移学习管理器"""
    
    # 可迁移的层 (与数据集无关)
    TRANSFERABLE_PATTERNS = [
        'dino',               # DINOv2 backbone
        'backbone',           # 通用backbone
        'channel_adapter',    # 通道适配器
        'adapted_encoder',    # 适配后的编码器
        'attn_adapter',       # 注意力adapter
        'mlp_adapter',        # MLP adapter
        'prompt',             # Prompt模块
        'patch_embed',        # Patch embedding
        'cls_token',          # CLS token
        'pos_embed',          # Position embedding
    ]
    
    # 不可迁移的层 (与数据集相关)
    NON_TRANSFERABLE_PATTERNS = [
        'classifier',         # 分类头 (物种数相关)
        'env_encoder',        # 环境编码器 (环境变量维度相关)
        'env_input_norm',     # 环境输入归一化
        'fusion.env',         # 融合模块的环境部分
    ]
    
    @staticmethod
    def load_transferable_weights(
        model: nn.Module,
        source_checkpoint: str,
        transfer_mode: str = 'freeze_backbone',
        strict: bool = False,
        verbose: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        加载可迁移的权重
        
        Args:
            model: 目标模型
            source_checkpoint: 源checkpoint路径
            transfer_mode: 迁移模式
                - 'freeze_backbone': 冻结backbone，只训练分类头
                - 'finetune_all': 加载权重后全部可训练
                - 'linear_probe': 只训练最后的分类层
            strict: 是否严格匹配
            verbose: 是否打印详细信息
            
        Returns:
            (loaded_keys, skipped_keys)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"跨域权重迁移")
            print(f"{'='*60}")
            print(f"源checkpoint: {source_checkpoint}")
            print(f"迁移模式: {transfer_mode}")
        
        # 加载源checkpoint
        checkpoint = torch.load(source_checkpoint, map_location='cpu')
        if 'state_dict' in checkpoint:
            source_state = checkpoint['state_dict']
        else:
            source_state = checkpoint
        
        # 处理model.前缀 - 如果源checkpoint中有model.前缀，去掉它
        processed_source_state = {}
        for key, value in source_state.items():
            # 去掉'model.'前缀
            if key.startswith('model.'):
                new_key = key[6:]  # 去掉'model.'
            else:
                new_key = key
            processed_source_state[new_key] = value
        
        source_state = processed_source_state
        
        # 获取模型当前state
        model_state = model.state_dict()
        
        loaded_keys = []
        skipped_keys = []
        shape_mismatch_keys = []
        
        for key, source_value in source_state.items():
            # 检查是否应该跳过 (不可迁移的层)
            should_skip = any(
                pattern in key 
                for pattern in CrossDomainTransfer.NON_TRANSFERABLE_PATTERNS
            )
            
            if should_skip:
                skipped_keys.append((key, "excluded (domain-specific)"))
                continue
            
            # 检查key是否存在于目标模型
            if key not in model_state:
                skipped_keys.append((key, "not in target model"))
                continue
            
            # 检查维度是否匹配
            target_value = model_state[key]
            if source_value.shape != target_value.shape:
                shape_mismatch_keys.append((key, source_value.shape, target_value.shape))
                skipped_keys.append((key, f"shape mismatch {source_value.shape} vs {target_value.shape}"))
                continue
            
            # 加载权重
            model_state[key] = source_value
            loaded_keys.append(key)
        
        # 应用权重
        model.load_state_dict(model_state, strict=False)
        
        if verbose:
            print(f"\n✅ 成功加载: {len(loaded_keys)} 个参数")
            print(f"⏭️  跳过: {len(skipped_keys)} 个参数")
            
            if shape_mismatch_keys:
                print(f"\n⚠️  维度不匹配的参数:")
                for key, src_shape, tgt_shape in shape_mismatch_keys[:5]:
                    print(f"   {key}: {src_shape} → {tgt_shape}")
                if len(shape_mismatch_keys) > 5:
                    print(f"   ... 共 {len(shape_mismatch_keys)} 个")
            
            # 显示跳过的domain-specific参数
            domain_specific = [(k, r) for k, r in skipped_keys if 'domain-specific' in r]
            if domain_specific:
                print(f"\n📋 Domain-specific参数 (重新初始化):")
                for k, _ in domain_specific[:5]:
                    print(f"   {k}")
                if len(domain_specific) > 5:
                    print(f"   ... 共 {len(domain_specific)} 个")
        
        # 根据迁移模式设置冻结策略
        CrossDomainTransfer._apply_freeze_strategy(model, transfer_mode, verbose)
        
        return loaded_keys, [k[0] for k in skipped_keys]
    
    @staticmethod
    def _apply_freeze_strategy(
        model: nn.Module, 
        transfer_mode: str,
        verbose: bool = True
    ):
        """应用冻结策略"""
        
        if transfer_mode == 'freeze_backbone':
            # 冻结所有已迁移的层，只训练新初始化的层
            frozen_count = 0
            trainable_count = 0
            
            for name, param in model.named_parameters():
                # 检查是否是可迁移的(需要冻结的)层
                should_freeze = any(
                    pattern in name 
                    for pattern in CrossDomainTransfer.TRANSFERABLE_PATTERNS
                )
                
                if should_freeze:
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    param.requires_grad = True
                    trainable_count += 1
            
            if verbose:
                print(f"\n🔒 冻结策略: freeze_backbone")
                print(f"   冻结参数组: {frozen_count}")
                print(f"   可训练参数组: {trainable_count}")
                
        elif transfer_mode == 'linear_probe':
            # 只训练分类器
            frozen = 0
            trainable = 0
            for name, param in model.named_parameters():
                if 'classifier' in name:
                    param.requires_grad = True
                    trainable += 1
                else:
                    param.requires_grad = False
                    frozen += 1
            
            if verbose:
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"\n🔒 冻结策略: linear_probe")
                print(f"   只训练分类头，可训练参数: {trainable_params:,}")
                print(f"   冻结参数组: {frozen}, 可训练参数组: {trainable}")
                
        elif transfer_mode == 'finetune_all':
            # 全部可训练
            for param in model.parameters():
                param.requires_grad = True
            
            if verbose:
                total = sum(p.numel() for p in model.parameters())
                print(f"\n�� 冻结策略: finetune_all")
                print(f"   全部可训练，参数量: {total:,}")
        
        else:
            raise ValueError(f"Unknown transfer_mode: {transfer_mode}")
    
    @staticmethod
    def get_parameter_groups(
        model: nn.Module,
        base_lr: float = 1e-4,
        backbone_lr_scale: float = 0.1,
        new_layer_lr_scale: float = 10.0
    ) -> List[Dict]:
        """
        获取分层学习率的参数组
        
        Args:
            model: 模型
            base_lr: 基础学习率
            backbone_lr_scale: backbone学习率缩放因子
            new_layer_lr_scale: 新层学习率缩放因子
            
        Returns:
            参数组列表，用于optimizer
        """
        backbone_params = []
        new_layer_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            is_new_layer = any(
                pattern in name 
                for pattern in CrossDomainTransfer.NON_TRANSFERABLE_PATTERNS
            )
            
            is_backbone = any(
                pattern in name
                for pattern in ['dino', 'backbone', 'patch_embed', 'cls_token', 'pos_embed']
            )
            
            if is_new_layer:
                new_layer_params.append(param)
            elif is_backbone:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = []
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': base_lr * backbone_lr_scale,
                'name': 'backbone'
            })
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr,
                'name': 'adapter_prompt'
            })
        
        if new_layer_params:
            param_groups.append({
                'params': new_layer_params,
                'lr': base_lr * new_layer_lr_scale,
                'name': 'new_layers'
            })
        
        return param_groups


def transfer_and_freeze(
    model: nn.Module,
    source_checkpoint: str,
    transfer_mode: str = 'freeze_backbone'
) -> nn.Module:
    """
    便捷函数：加载权重并应用冻结策略
    
    Example:
        model = create_model(kenya_config)
        model = transfer_and_freeze(model, 'usa_model.ckpt', 'freeze_backbone')
    """
    CrossDomainTransfer.load_transferable_weights(
        model, source_checkpoint, transfer_mode
    )
    return model


# =====================================================
# 测试函数
# =====================================================
def test_transfer():
    """测试跨域迁移功能"""
    print("=" * 60)
    print("测试跨域迁移功能")
    print("=" * 60)
    
    # 这里可以添加测试代码
    pass


if __name__ == "__main__":
    test_transfer()


# Additional transfer modes for HGCP+FDA model
class HGCPFDATransfer(CrossDomainTransfer):
    """Extended transfer support for HGCP+FDA model"""
    
    # HGCP+FDA specific patterns
    HGCP_PATTERNS = ['hgcp', 'hierarchical', 'geo_prompt', 'gate']
    FDA_PATTERNS = ['fda', 'freq', 'low_freq', 'high_freq', 'frequency']
    ADAPTER_PATTERNS = ['adapter', 'prompt', 'channel_adapter']
    BACKBONE_PATTERNS = ['dino', 'backbone', 'patch_embed', 'cls_token', 'pos_embed', 'blocks']
    
    @staticmethod
    def apply_transfer_strategy(
        model: nn.Module,
        transfer_mode: str,
        unfreeze_last_n_blocks: int = 0,
        verbose: bool = True
    ) -> Dict[str, int]:
        """
        Apply transfer learning strategy for HGCP+FDA model
        
        Args:
            model: The model to configure
            transfer_mode: One of ['linear_probe', 'adapter_tune', 'finetune']
            unfreeze_last_n_blocks: Number of last backbone blocks to unfreeze
            verbose: Print detailed info
            
        Returns:
            Dict with counts of frozen/trainable parameters
        """
        stats = {'frozen': 0, 'trainable': 0, 'total_params': 0}
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Transfer Strategy: {transfer_mode}")
            print(f"{'='*50}")
        
        if transfer_mode == 'linear_probe':
            # Only train classifier, freeze everything else
            for name, param in model.named_parameters():
                if 'classifier' in name or 'fc' in name:
                    param.requires_grad = True
                    stats['trainable'] += param.numel()
                else:
                    param.requires_grad = False
                    stats['frozen'] += param.numel()
                stats['total_params'] += param.numel()
                    
        elif transfer_mode == 'adapter_tune':
            # Freeze backbone, train adapters + HGCP + FDA + classifier
            for name, param in model.named_parameters():
                # Check if it's a backbone parameter (to freeze)
                is_backbone = any(
                    pattern in name.lower() 
                    for pattern in HGCPFDATransfer.BACKBONE_PATTERNS
                )
                # Check if it's an adapter/HGCP/FDA parameter (to train)
                is_adapter = any(
                    pattern in name.lower()
                    for pattern in HGCPFDATransfer.ADAPTER_PATTERNS + 
                                   HGCPFDATransfer.HGCP_PATTERNS +
                                   HGCPFDATransfer.FDA_PATTERNS
                )
                is_classifier = 'classifier' in name.lower() or 'fc' in name.lower()
                is_env_encoder = 'env' in name.lower()
                
                # Handle backbone blocks with unfreeze_last_n_blocks
                if is_backbone and 'blocks' in name.lower() and unfreeze_last_n_blocks > 0:
                    # Extract block number
                    import re
                    block_match = re.search(r'blocks\.(\d+)', name)
                    if block_match:
                        block_num = int(block_match.group(1))
                        total_blocks = 12  # DINOv2 ViT-B has 12 blocks
                        if block_num >= total_blocks - unfreeze_last_n_blocks:
                            is_backbone = False  # Unfreeze this block
                
                if is_backbone and not is_adapter:
                    param.requires_grad = False
                    stats['frozen'] += param.numel()
                else:
                    param.requires_grad = True
                    stats['trainable'] += param.numel()
                stats['total_params'] += param.numel()
                    
        elif transfer_mode == 'finetune':
            # Train all with differential learning rates
            # Partially unfreeze backbone (last n blocks)
            for name, param in model.named_parameters():
                is_backbone = any(
                    pattern in name.lower() 
                    for pattern in HGCPFDATransfer.BACKBONE_PATTERNS
                )
                
                # Handle backbone blocks with unfreeze_last_n_blocks
                should_freeze_backbone = is_backbone and 'blocks' in name.lower()
                if should_freeze_backbone and unfreeze_last_n_blocks > 0:
                    import re
                    block_match = re.search(r'blocks\.(\d+)', name)
                    if block_match:
                        block_num = int(block_match.group(1))
                        total_blocks = 12
                        if block_num >= total_blocks - unfreeze_last_n_blocks:
                            should_freeze_backbone = False
                
                if should_freeze_backbone:
                    param.requires_grad = False
                    stats['frozen'] += param.numel()
                else:
                    param.requires_grad = True
                    stats['trainable'] += param.numel()
                stats['total_params'] += param.numel()
        else:
            raise ValueError(f"Unknown transfer_mode: {transfer_mode}")
        
        if verbose:
            print(f"  Frozen parameters: {stats['frozen']:,}")
            print(f"  Trainable parameters: {stats['trainable']:,}")
            print(f"  Total parameters: {stats['total_params']:,}")
            print(f"  Trainable ratio: {100*stats['trainable']/stats['total_params']:.2f}%")
        
        return stats
    
    @staticmethod
    def get_optimizer_param_groups(
        model: nn.Module,
        base_lr: float = 1e-4,
        transfer_mode: str = 'adapter_tune'
    ) -> list:
        """
        Get parameter groups with differentiated learning rates
        
        Returns:
            List of dicts for optimizer
        """
        param_groups = []
        
        # Separate parameters by type
        backbone_params = []
        adapter_params = []
        hgcp_params = []
        fda_params = []
        classifier_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            name_lower = name.lower()
            
            if any(p in name_lower for p in ['classifier', 'fc']):
                classifier_params.append(param)
            elif any(p in name_lower for p in HGCPFDATransfer.HGCP_PATTERNS):
                hgcp_params.append(param)
            elif any(p in name_lower for p in HGCPFDATransfer.FDA_PATTERNS):
                fda_params.append(param)
            elif any(p in name_lower for p in HGCPFDATransfer.ADAPTER_PATTERNS):
                adapter_params.append(param)
            elif any(p in name_lower for p in HGCPFDATransfer.BACKBONE_PATTERNS):
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        # Create param groups with different learning rates
        lr_scales = {
            'linear_probe': {'backbone': 0.0, 'adapter': 0.0, 'hgcp': 0.0, 'fda': 0.0, 'classifier': 1.0, 'other': 0.0},
            'adapter_tune': {'backbone': 0.1, 'adapter': 1.0, 'hgcp': 1.0, 'fda': 1.0, 'classifier': 2.0, 'other': 1.0},
            'finetune': {'backbone': 0.1, 'adapter': 1.0, 'hgcp': 1.0, 'fda': 1.0, 'classifier': 2.0, 'other': 1.0}
        }
        
        scales = lr_scales.get(transfer_mode, lr_scales['adapter_tune'])
        
        if backbone_params and scales['backbone'] > 0:
            param_groups.append({'params': backbone_params, 'lr': base_lr * scales['backbone'], 'name': 'backbone'})
        if adapter_params and scales['adapter'] > 0:
            param_groups.append({'params': adapter_params, 'lr': base_lr * scales['adapter'], 'name': 'adapter'})
        if hgcp_params and scales['hgcp'] > 0:
            param_groups.append({'params': hgcp_params, 'lr': base_lr * scales['hgcp'], 'name': 'hgcp'})
        if fda_params and scales['fda'] > 0:
            param_groups.append({'params': fda_params, 'lr': base_lr * scales['fda'], 'name': 'fda'})
        if classifier_params and scales['classifier'] > 0:
            param_groups.append({'params': classifier_params, 'lr': base_lr * scales['classifier'], 'name': 'classifier'})
        if other_params and scales['other'] > 0:
            param_groups.append({'params': other_params, 'lr': base_lr * scales['other'], 'name': 'other'})
        
        return param_groups
