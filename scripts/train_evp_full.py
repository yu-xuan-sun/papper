#!/usr/bin/env python3
"""
EVP (Environment-aware Visual Prompt Tuning) 完整训练脚本
支持USA-Summer, USA-Winter, Kenya数据集
"""

import os
import sys
sys.path.insert(0, '/sunyuxuan/satbird')

import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt
from src.models.env_aware_prompt import EnvironmentPromptGenerator
from src.datamodule.datamodule import EBirdDataModule


class EVPDinov2Model(nn.Module):
    """带EVP的DINOv2模型"""
    
    def __init__(
        self,
        base_model: Dinov2AdapterPrompt,
        env_dim: int = 27,
        use_evp: bool = True,
        evp_hidden_dim: int = 512,
        evp_dropout: float = 0.1,
        freeze_base: bool = False
    ):
        super().__init__()
        
        self.base_model = base_model
        self.use_evp = use_evp
        self.env_dim = env_dim
        
        self.embed_dim = base_model.embed_dim
        self.prompt_len = base_model.prompt_len
        self.num_layers = len(base_model.dino.blocks)
        
        if use_evp:
            self.evp_generator = EnvironmentPromptGenerator(
                env_dim=env_dim,
                prompt_len=self.prompt_len,
                embed_dim=self.embed_dim,
                hidden_dim=evp_hidden_dim,
                num_layers=self.num_layers,
                use_layer_specific=True,
                use_residual=True,
                use_gating=True,
                dropout=evp_dropout
            )
            print(f"✨ EVP enabled: env_dim={env_dim}, prompt_len={self.prompt_len}")
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("🔒 Base model frozen, only training EVP")
    
    def forward(self, img: torch.Tensor, env: torch.Tensor = None) -> torch.Tensor:
        batch_size = img.size(0)
        
        # Channel adapter
        if self.base_model.channel_adapter is not None:
            x = self.base_model.channel_adapter(img)
        elif self.base_model.in_channels > 3:
            x = img[:, :3, :, :]
        else:
            x = img
        
        # Patch embedding
        x = self.base_model.dino.patch_embed(x)
        
        # Add CLS token
        if self.base_model.dino.cls_token is not None:
            cls_token = self.base_model.dino.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        x = self.base_model.dino.pos_drop(x)
        
        # EVP forward
        if self.use_evp and env is not None:
            for i, block in enumerate(self.base_model.dino.blocks):
                prompts, _ = self.evp_generator(env, layer_idx=i)
                x_with_prompts = torch.cat([x, prompts], dim=1)
                x_with_prompts = block(x_with_prompts)
                x = x_with_prompts[:, :-self.prompt_len, :]
        else:
            x = self.base_model.adapted_encoder(x)
        
        x = self.base_model.dino.norm(x)
        visual_feat = x[:, 0]
        
        # Fusion
        if self.base_model.use_env and env is not None:
            if self.base_model.fusion_type == "adaptive_attention":
                fused_feat = self.base_model.fusion(visual_feat, env)
            else:
                fused_feat = visual_feat
        else:
            fused_feat = visual_feat
        
        logits = self.base_model.classifier(fused_feat)
        return logits


class EVPLightningModule(pl.LightningModule):
    """EVP训练的Lightning模块"""
    
    def __init__(self, config, use_evp=True, freeze_base=False):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        model_cfg = config.experiment.module
        
        # 创建基础模型
        base_model = Dinov2AdapterPrompt(
            num_classes=config.data.total_species,
            dino_model_name=getattr(model_cfg, 'dino_model', 'vit_base_patch14_dinov2.lvd142m'),
            pretrained_path=getattr(model_cfg, 'pretrained_path', 'checkpoints/dinov2_vitb14_pretrain.pth'),
            prompt_len=getattr(model_cfg, 'prompt_len', 40),
            bottleneck_dim=getattr(model_cfg, 'bottleneck_dim', 96),
            adapter_layers=getattr(model_cfg, 'adapter_layers', None),
            adapter_dropout=getattr(model_cfg, 'adapter_dropout', 0.1),
            env_input_dim=sum(config.data.env_var_sizes) if hasattr(config.data, 'env_var_sizes') else 27,
            env_hidden_dim=getattr(model_cfg, 'env_hidden_dim', 2048),
            env_num_layers=getattr(model_cfg, 'env_num_layers', 3),
            use_env=True,
            fusion_type=getattr(model_cfg, 'fusion_type', 'adaptive_attention'),
            hidden_dims=getattr(model_cfg, 'hidden_dims', [2048, 1024]),
            dropout=getattr(model_cfg, 'dropout', 0.15),
            use_channel_adapter=getattr(model_cfg, 'use_channel_adapter', True),
            in_channels=len(config.data.bands) if hasattr(config.data, 'bands') else 4,
            freeze_backbone=getattr(model_cfg, 'freeze_backbone', True),
        )
        
        env_dim = sum(config.data.env_var_sizes) if hasattr(config.data, 'env_var_sizes') else 27
        
        # 创建EVP模型
        self.model = EVPDinov2Model(
            base_model=base_model,
            env_dim=env_dim,
            use_evp=use_evp,
            evp_hidden_dim=512,
            evp_dropout=0.1,
            freeze_base=freeze_base
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 统计参数
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Total: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    
    def forward(self, img, env):
        return self.model(img, env)
    
    def training_step(self, batch, batch_idx):
        img, env, target = batch['image'], batch['env'], batch['target']
        logits = self(img, env)
        loss = self.criterion(logits, target)
        
        # 计算TopK准确率
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            k = int(target.sum(dim=1).mean().item())
            k = max(1, min(k, logits.size(1)))
            _, topk_indices = probs.topk(k, dim=1)
            topk_acc = target.gather(1, topk_indices).sum(dim=1).mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_topk', topk_acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, env, target = batch['image'], batch['env'], batch['target']
        logits = self(img, env)
        loss = self.criterion(logits, target)
        
        probs = torch.sigmoid(logits)
        k = int(target.sum(dim=1).mean().item())
        k = max(1, min(k, logits.size(1)))
        _, topk_indices = probs.topk(k, dim=1)
        topk_acc = target.gather(1, topk_indices).sum(dim=1).mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_topk', topk_acc, prog_bar=True)
        return {'val_loss': loss, 'val_topk': topk_acc}
    
    def configure_optimizers(self):
        # 分组参数
        evp_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'evp_generator' in name:
                    evp_params.append(param)
                else:
                    other_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': evp_params, 'lr': self.config.experiment.module.lr * 2},  # EVP用更高学习率
            {'params': other_params, 'lr': self.config.experiment.module.lr}
        ], weight_decay=self.config.experiment.module.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config.experiment.trainer.max_epochs,
            eta_min=1e-7
        )
        
        return [optimizer], [scheduler]


@hydra.main(config_path="../configs", config_name="defaults", version_base=None)
def main(config: DictConfig):
    """主训练函数"""
    print("="*60)
    print("EVP Training Script")
    print("="*60)
    
    # 解析额外参数
    use_evp = config.get('use_evp', True)
    freeze_base = config.get('freeze_base', False)
    
    print(f"Configuration:")
    print(f"  - Dataset: {config.data.dataset}")
    print(f"  - Use EVP: {use_evp}")
    print(f"  - Freeze base: {freeze_base}")
    print(f"  - Max epochs: {config.experiment.trainer.max_epochs}")
    
    # 创建数据模块
    dm = EBirdDataModule(config)
    
    # 创建模型
    model = EVPLightningModule(config, use_evp=use_evp, freeze_base=freeze_base)
    
    # 设置保存路径
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"evp_{config.data.dataset}_seed{config.experiment.seed}_{timestamp}"
    
    # 回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath=f"runs/{run_name}",
            filename="best-{epoch:02d}-{val_topk:.4f}",
            monitor="val_topk",
            mode="max",
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor="val_topk",
            patience=15,
            mode="max"
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Logger
    loggers = [
        TensorBoardLogger("logs/evp", name=run_name),
        CSVLogger("logs/evp", name=run_name)
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.experiment.trainer.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=loggers,
        precision=16,  # 混合精度
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.experiment.trainer.get('accumulate_grad_batches', 1),
        val_check_interval=1.0,
        log_every_n_steps=50,
    )
    
    # 训练
    print("\n🚀 Starting training...")
    trainer.fit(model, dm)
    
    # 测试最佳模型
    print("\n📊 Testing best model...")
    trainer.test(model, dm, ckpt_path='best')
    
    print(f"\n✅ Training completed! Results saved to runs/{run_name}")


if __name__ == '__main__':
    main()
