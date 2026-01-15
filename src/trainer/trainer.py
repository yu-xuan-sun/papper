"""
general trainer: supports training Resnet18, Satlas, SATMAE, and DINOv2 multimodal
"""
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import models

from src.dataset.dataloader import EbirdVisionDataset, get_subset
from src.losses.losses import CustomCrossEntropyLoss, WeightedCustomCrossEntropyLoss, RMSLELoss, CustomFocalLoss, MultiTaskLoss

from src.losses.asymmetric_loss import AsymmetricLoss
from src.losses.improved_ranking_loss import ListMLELoss, ImprovedListwiseLoss, ImprovedCombinedLoss
from src.losses.fixed_ranking_loss import FixedCombinedLoss, SimpleCombinedLoss
from src.trainer.utils import get_target_size, get_nb_bands, get_scheduler, init_first_layer_weights, \
    load_from_checkpoint
from src.transforms.transforms import get_transforms
from src.transforms.mixup_cutmix import build_mixup_cutmix  # ✅ 加入 Mixup/CutMix
from src.models.vit import ViTFinetune
from src.models.dinov2_multimodal import Dinov2Multimodal  # ✅ 加入 dinov2
# from src.models.enhanced_convnext import EnhancedConvNeXtMultimodal  # ✅ 加入 enhanced_convnext
# from src.models.dinov2_advanced import EnhancedDinov2Multimodal  # ✅ 加入 dinov2_advanced

# Placeholder classes for type checking (modules not implemented yet)
class EnhancedConvNeXtMultimodal:
    pass
class EnhancedDinov2Multimodal:
    pass

from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt  # ✅ 加入 dinov2_adapter_prompt
from src.models.dinov2_full_finetune import Dinov2FullFinetune  # ✅ 加入 dinov2_full_finetune
from src.models.dinov2_evp import Dinov2EVP  # ✅ 加入 EVP 模型

from src.models.hierarchical_geo_prompt import Dinov2HGCP  # ✅ 加入 HGCP 模型
from src.models.fda_net import Dinov2FDA  # ✅ 加入 FDA 模型
from src.models.hgcp_fda import Dinov2HGCP_FDA
class EbirdTask(pl.LightningModule):
    def __init__(self, opts, **kwargs: Any) -> None:
        """initializes a new Lightning Module to train"""

        super().__init__()
        self.save_hyperparameters(opts)
        self.opts = opts
        self.means = None

        self.freeze_backbone = self.opts.experiment.module.freeze
        # get target vector size (number of species we consider)
        self.subset = get_subset(self.opts.data.target.subset, self.opts.data.total_species)
        self.target_size = get_target_size(opts, self.subset)
        print("Predicting ", self.target_size, "species")

        self.target_type = self.opts.data.target.type

        # define self.learning_rate to enable learning rate finder
        self.learning_rate = self.opts.experiment.module.lr
        self.sigmoid_activation = nn.Sigmoid()

        self.config_task(opts, **kwargs)
        # 用于各阶段的指标累积（必须在 __init__ 内初始化，不能在类体）
        self.test_preds = []
        self.test_targets = []
        self.val_preds = []
        self.val_targets = []

        self._finetune_schedule = getattr(self.opts.experiment.module, "finetune_schedule", None)
        if self._finetune_schedule is not None:
            self._freeze_epochs = int(getattr(self._finetune_schedule, "freeze_epochs", -1))
            self._lr_after_unfreeze = float(getattr(self._finetune_schedule, "lr_after_unfreeze", self.learning_rate))
            self._lr_warmup_epochs = int(getattr(self._finetune_schedule, "lr_warmup_epochs", 0))
        else:
            self._freeze_epochs = None
            self._lr_after_unfreeze = self.learning_rate
            self._lr_warmup_epochs = 0
        self._has_unfrozen_backbone = False
        self._stage2_epoch_start = 0
        self._warmup_start_lr = None

        # Initialize Mixup/CutMix augmentation
        mixup_config = getattr(self.opts.experiment.module, "mixup_cutmix", None)
        if mixup_config is not None and getattr(mixup_config, "enabled", False):
            self.mixup_cutmix = build_mixup_cutmix(
                mode=getattr(mixup_config, "mode", "mixup_cutmix"),
                mixup_alpha=getattr(mixup_config, "mixup_alpha", 0.2),
                cutmix_alpha=getattr(mixup_config, "cutmix_alpha", 1.0),
                prob=getattr(mixup_config, "prob", 0.5),
                switch_prob=getattr(mixup_config, "switch_prob", 0.5),
                num_classes=self.target_size
            )
            print(f"✅ Mixup/CutMix enabled: mode={mixup_config.mode}, prob={mixup_config.prob}")
        else:
            self.mixup_cutmix = None

    def _compute_regression_metrics(self, preds: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        """返回基于全类的 MAE / MSE（对多类逐元素计算后再全局平均）。"""
        mae = torch.mean(torch.abs(preds - targets))
        mse = torch.mean((preds - targets) ** 2)
        return {"mae": mae, "mse": mse}

    def _compute_topk_metrics(self, preds: Tensor, targets: Tensor, ks=(1, 5, 10, 30)) -> Dict[str, Tensor]:
        """按照 `metrics.py` 中的定义计算 Top-K 指标。"""
        if preds.ndim != 2:
            preds = preds.view(preds.size(0), -1)
        if targets.ndim != 2:
            targets = targets.view(targets.size(0), -1)

        preds = preds.float()
        targets = targets.float()

        n, c = preds.shape
        if c == 0:
            return {}

        max_k = min(max(ks), c)
        pred_topk = torch.topk(preds, k=max_k, dim=1, largest=True).indices
        target_topk = torch.topk(targets, k=max_k, dim=1, largest=True).indices
        non_zero_counts = torch.count_nonzero(targets, dim=1)

        accum: Dict[int, Tensor] = {k: preds.new_tensor(0.0) for k in ks}
        totals: Dict[int, int] = {k: 0 for k in ks}

        for i in range(n):
            ki = int(non_zero_counts[i].item())
            if ki == 0:
                continue
            for k in ks:
                kk = min(k, c)
                pred_idx = pred_topk[i, :kk]
                if ki >= kk:
                    target_idx = target_topk[i, :kk]
                    denom = kk
                else:
                    target_idx = target_topk[i, :ki]
                    denom = ki
                matches = (pred_idx.unsqueeze(1) == target_idx.unsqueeze(0)).any(dim=1).float().sum()
                accum[k] += matches / denom
                totals[k] += 1

        metrics: Dict[str, Tensor] = {}
        for k in ks:
            total = totals[k]
            if total > 0:
                metrics[f"top{k}_acc"] = accum[k] / total
        return metrics

    def _compute_dynamic_topk(self, preds: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        """对应 `CustomTopK` 的动态 Top-K 指标，按样本真实物种数自适应。"""
        if preds.ndim != 2:
            preds = preds.view(preds.size(0), -1)
        if targets.ndim != 2:
            targets = targets.view(targets.size(0), -1)

        preds = preds.float()
        targets = targets.float()

        n, c = preds.shape
        if c == 0:
            return {}

        non_zero_counts = torch.count_nonzero(targets, dim=1)
        total = 0
        accum = preds.new_tensor(0.0)

        for i in range(n):
            ki = int(non_zero_counts[i].item())
            if ki <= 0:
                continue
            ki = min(ki, c)
            pred_idx = torch.topk(preds[i], k=ki, largest=True).indices
            target_idx = torch.topk(targets[i], k=ki, largest=True).indices
            hits = (pred_idx.unsqueeze(1) == target_idx.unsqueeze(0)).any(dim=1).float().sum()
            accum += hits / ki
            total += 1

        if total == 0:
            return {}
        return {"topk": accum / total}

    def config_task(self, opts, **kwargs: Any) -> None:

        pres_weight = getattr(self.opts.losses, "presence_weight", 1.0)
        abs_weight = getattr(self.opts.losses, "absence_weight", 1.0)
        module_cfg = self.opts.experiment.module
        use_weighted_loss = getattr(module_cfg, "use_weighted_loss", False)

        if self.opts.losses.criterion == "MSE":
            self.criterion = nn.MSELoss()
        elif self.opts.losses.criterion == "MAE":
            self.criterion = nn.L1Loss()
        elif self.opts.losses.criterion == "RMSLE":
            self.criterion = RMSLELoss()
        elif self.opts.losses.criterion == "Focal":
            alpha = getattr(self.opts.losses, "focal_alpha", 1.0)
            gamma = getattr(self.opts.losses, "focal_gamma", 2.0)
            self.criterion = CustomFocalLoss(alpha=alpha, gamma=gamma)
        elif self.opts.losses.criterion == "ASL":
            gamma_neg = getattr(self.opts.losses, "asl_gamma_neg", 4.0)
            gamma_pos = getattr(self.opts.losses, "asl_gamma_pos", 1.0)
            clip = getattr(self.opts.losses, "asl_clip", 0.05)
            self.criterion = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)
        elif self.opts.losses.criterion == "multi_task":
            ce_weight = getattr(self.opts.losses, "ce_weight", 0.6)
            mse_weight = getattr(self.opts.losses, "mse_weight", 0.4)
            label_smoothing = getattr(self.opts.losses, "label_smoothing", 0.0)
            self.criterion = MultiTaskLoss(
                ce_weight=ce_weight,
                mse_weight=mse_weight,
                lambd_pres=pres_weight,
                lambd_abs=abs_weight,
                label_smoothing=label_smoothing
            )
            print(f"Training with MultiTask Loss (CE={ce_weight}, MSE={mse_weight}, smoothing={label_smoothing})")
        elif self.opts.losses.criterion == "listmle":
            # ListMLE Loss for ranking optimization
            ranking_top_k = getattr(self.opts.experiment.module, "listwise_top_k", 30)
            ranking_weight = getattr(self.opts.experiment.module, "ranking_loss_weight", 0.3)
            label_smoothing = getattr(self.opts.experiment.module, "label_smoothing", 0.05)
            self.criterion = ImprovedCombinedLoss(
                num_classes=self.target_size,
                class_frequencies=None,
                label_smoothing=label_smoothing,
                presence_weight=pres_weight,
                absence_weight=abs_weight,
                ranking_weight=ranking_weight,
                ranking_type='listmle',
                ranking_top_k=ranking_top_k,
                use_class_weights=False
            )
            print(f"Training with ListMLE Loss (ranking_weight={ranking_weight}, top_k={ranking_top_k})")
        elif self.opts.losses.criterion == "fixed_combined":
            # 修复版组合损失
            ranking_top_k = getattr(self.opts.experiment.module, "listwise_top_k", 30)
            ranking_weight = getattr(self.opts.experiment.module, "ranking_loss_weight", 0.1)
            self.criterion = FixedCombinedLoss(
                num_classes=self.target_size,
                class_frequencies=None,
                presence_weight=pres_weight,
                absence_weight=abs_weight,
                ranking_weight=ranking_weight,
                ranking_top_k=ranking_top_k,
                use_class_weights=False
            )
            print(f"Training with Fixed Combined Loss (ranking_weight={ranking_weight}, top_k={ranking_top_k})")
        elif self.opts.losses.criterion == "simple_combined":
            # 简化版组合损失 - 最稳定
            ranking_top_k = getattr(self.opts.experiment.module, "listwise_top_k", 30)
            topk_bonus = getattr(self.opts.experiment.module, "topk_bonus_weight", 0.05)
            self.criterion = SimpleCombinedLoss(
                num_classes=self.target_size,
                presence_weight=pres_weight,
                absence_weight=abs_weight,
                topk_bonus_weight=topk_bonus,
                top_k=ranking_top_k
            )
            print(f"Training with Simple Combined Loss (presence_weight={pres_weight}, topk_bonus={topk_bonus})")
        else:
            # target is num checklists reporting species i / total number of checklists at a hotspot
            if use_weighted_loss:
                self.criterion = WeightedCustomCrossEntropyLoss(lambd_pres=pres_weight, lambd_abs=abs_weight)
                print("Training with Weighted CE Loss")
            else:
                self.criterion = CustomCrossEntropyLoss(lambd_pres=pres_weight, lambd_abs=abs_weight)
                print("Training with Custom CE Loss")

        self.model = self.get_sat_model()

    def get_sat_model(self):
        model_type = self.opts.experiment.module.model

        if model_type == "satlas":
            print('using Satlas model')
            self.feature_extractor = torchvision.models.swin_transformer.swin_v2_b()
            full_state_dict = torch.load(self.opts.experiment.module.resume, map_location=torch.device('cpu'))
            swin_prefix = 'backbone.backbone.'
            swin_state_dict = {k[len(swin_prefix):]: v for k, v in full_state_dict.items() if k.startswith(swin_prefix)}

            self.feature_extractor.load_state_dict(swin_state_dict)
            self.feature_extractor.to('cuda:0')
            print("initialized network, freezing weights")

            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.model = nn.Linear(1000, self.target_size)

        elif model_type == "satmae":
            print('using SatMAE model')
            satmae = ViTFinetune(img_size=224, patch_size=16, in_chans=3, num_classes=self.target_size, embed_dim=1024,
                                 depth=24, num_heads=16, mlp_ratio=4, drop_rate=0.1, )
            satmae = load_from_checkpoint(self.opts.experiment.module.resume, satmae)
            satmae.to('cuda')
            in_feat = satmae.fc.in_features
            satmae.fc = nn.Sequential()
            print("initialized network, freezing weights")
            self.feature_extractor = satmae
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.model = nn.Linear(in_feat, self.target_size)

        elif model_type == "resnet18":
            self.model = models.resnet18(pretrained=self.opts.experiment.module.pretrained)
            if self.opts.experiment.module.transfer_weights == "SECO":
                print("Initializing with SeCo weights")
                with open(self.opts.experiment.module.resume, "rb") as file:
                    enc = pickle.load(file)
                pretrained = list(enc.items())
                model_dict = dict(self.model.state_dict())
                count = 0
                for key, value in model_dict.items():
                    if not key.startswith("fc"):
                        if not key.startswith("conv1") and not key.startswith("bn1"):
                            layer_name, weights = pretrained[count]
                            model_dict[key] = weights
                        count += 1
            if len(self.opts.data.bands) != 3 or len(self.opts.data.env) > 0:
                self.bands = self.opts.data.bands + self.opts.data.env
                orig_channels = self.model.conv1.in_channels
                weights = self.model.conv1.weight.data.clone()
                self.model.conv1 = nn.Conv2d(get_nb_bands(self.bands), 64, kernel_size=(7, 7), stride=(2, 2),
                                             padding=(3, 3), bias=False)
                if self.opts.experiment.module.pretrained:
                    self.model.conv1.weight.data = init_first_layer_weights(get_nb_bands(self.bands), weights)
            if self.opts.experiment.module.transfer_weights == "USA":
                print("Transferring USA weights")
                ckpt = torch.load(self.opts.experiment.module.resume)
                self.model.fc = nn.Sequential()
                loaded_dict = ckpt['state_dict']
                model_dict = self.model.state_dict()
                for key_model, key_pretrained in zip(model_dict.keys(), loaded_dict.keys()):
                    if key_model == 'conv1.weight':
                        continue
                    model_dict[key_model] = loaded_dict[key_pretrained]
                self.model.load_state_dict(model_dict)
                if self.freeze_backbone:
                    print("initialized network, freezing weights")
                    for param in self.model.parameters():
                        param.requires_grad = False
            self.model.fc = nn.Linear(512, self.target_size)

        elif model_type == "dinov2":   # ✅ 新增分支
            print("using Dinov2 multimodal model")
            module_cfg = self.opts.experiment.module
            adapter_reduction_val = module_cfg.get("adapter_reduction", None)
            adapter_reduction = int(adapter_reduction_val) if adapter_reduction_val not in (None, "", "null") else None
            self.model = Dinov2Multimodal(
                num_species=self.target_size,
                dinov2_name=module_cfg.get("dinov2_name", "dinov2_vits14"),
                dinov2_pretrained=module_cfg.get("dinov2_pretrained", True),
                n_prompts=module_cfg.get("n_prompts", 8),
                env_backbone=module_cfg.get("env_backbone", "resnet18"),
                proj_dim=module_cfg.get("proj_dim", 768),
                freeze_dinov2=module_cfg.get("freeze_dinov2", True),
                device="cuda" if torch.cuda.is_available() else "cpu",
                use_grad_ckpt=module_cfg.get("use_grad_ckpt", False),
                use_channels_last=module_cfg.get("use_channels_last", False),
                use_block_prompts=module_cfg.get("use_block_prompts", False),
                block_prompt_length=module_cfg.get("block_prompt_length", 2),
                block_prompt_dropout=float(module_cfg.get("block_prompt_dropout", 0.0)),
                adapter_reduction=adapter_reduction,
                adapter_dropout=float(module_cfg.get("adapter_dropout", 0.0)),
                condition_prompts_on_env=module_cfg.get("condition_prompts_on_env", False),
                train_block_layer_norm=module_cfg.get("train_block_layer_norm", True),
                use_global_prompt=module_cfg.get("use_global_prompt", True),
                train_prompts_when_frozen=module_cfg.get("train_prompts_when_frozen", True),
            )

        elif model_type == "enhanced_convnext":   # ✅ 新增 EnhancedConvNeXt 分支
            print("using Enhanced ConvNeXt multimodal model")
            module_cfg = self.opts.experiment.module
            self.model = EnhancedConvNeXtMultimodal(
                num_species=self.target_size,
                num_env_features=module_cfg.get("num_env_features", 27),
                variant=module_cfg.get("convnext_variant", "small"),
                in_channels=module_cfg.get("in_channels", 4),
                use_bifpn=module_cfg.get("use_bifpn", True),
                env_hidden_dim=module_cfg.get("env_hidden_dim", 512),
                fusion_heads=module_cfg.get("fusion_heads", 12),
                dropout=module_cfg.get("dropout", 0.2),
                pretrained=module_cfg.get("pretrained", True),
            )
            print(f"✅ Created Enhanced ConvNeXt model:")
            print(f"   - Variant: {module_cfg.get('convnext_variant', 'small')}")
            print(f"   - BiFPN: {module_cfg.get('use_bifpn', True)}")
            print(f"   - Env features: {module_cfg.get('num_env_features', 27)}")
            print(f"   - Species: {self.target_size}")

        elif model_type == "dinov2_adapter_prompt":   # u2705 DINOv2 with Adapter and Prompt Tuning
            print("Using DINOv2 with Adapter and Prompt Tuning (PEFT)")
            from src.models.dinov2_adapter_prompt import create_dinov2_adapter_prompt_model
            self.model = create_dinov2_adapter_prompt_model(self.opts)
            print(f"✅ Created DINOv2 Adapter+Prompt model")
            # ========== Cross-domain transfer support ==========
            transfer_from = getattr(self.opts.experiment.module, "transfer_from", None)
            if transfer_from is not None:
                from src.utils.cross_domain_transfer import CrossDomainTransfer
                transfer_mode = getattr(self.opts.experiment.module, "transfer_mode", "freeze_backbone")
                print(f"🔄 Cross-domain transfer enabled:")
                print(f"   - Source checkpoint: {transfer_from}")
                print(f"   - Transfer mode: {transfer_mode}")
                # 使用静态方法
                loaded_keys, skipped_keys = CrossDomainTransfer.load_transferable_weights(self.model, transfer_from, transfer_mode)
                print(f"   - Loaded {len(loaded_keys)} parameter groups")
                print(f"   - Skipped {len(skipped_keys)} domain-specific parameters")
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                print(f"   - Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        elif model_type == "dinov2_full_finetune":   # ✅ DINOv2 Full Fine-tuning
            print("Using DINOv2 Full Fine-tuning (all parameters trainable)")
            from src.models.dinov2_full_finetune import create_dinov2_full_finetune_model
            self.model = create_dinov2_full_finetune_model(self.opts)
            print(f"✅ Created DINOv2 Full Finetune model")

        elif model_type == "dinov2_evp":   # ✅ DINOv2 with EVP
            print("Using DINOv2 with EVP (Environment-aware Visual Prompt Tuning)")
            from src.models.dinov2_evp import create_dinov2_evp_model
            self.model = create_dinov2_evp_model(self.opts)
            print(f"✅ Created DINOv2 EVP model")


        elif model_type == "dinov2_hgcp":   # ✅ DINOv2 with HGCP
            print("Using DINOv2 with HGCP (Hierarchical Geo-Contextual Prompts)")
            from src.models.hierarchical_geo_prompt import create_dinov2_hgcp_model
            self.model = create_dinov2_hgcp_model(self.opts)
            print(f"✅ Created DINOv2 HGCP model")

        elif model_type == "dinov2_hgcp_fda":   # ✅ DINOv2 with HGCP + FDA
            print("Using DINOv2 with HGCP + FDA")
            from src.models.hgcp_fda import create_dinov2_hgcp_fda_model
            self.model = create_dinov2_hgcp_fda_model(self.opts)
            print(f"✅ Created DINOv2 HGCP+FDA model")

        elif model_type == "dinov2_fda":   # ✅ DINOv2 with FDA (Frequency-Decoupled Adaptation)
            print("Using DINOv2 with FDA (Frequency-Decoupled Domain Adaptation)")
            from src.models.fda_net import create_dinov2_fda_model
            self.model = create_dinov2_fda_model(self.opts)
            print(f"✅ Created DINOv2 FDA model")

        elif model_type == "dinov2_advanced":   # ✅ 新增 DINOv2 Advanced 分支
            print("using DINOv2 Advanced multimodal model with 8 techniques")
            module_cfg = self.opts.experiment.module
            self.model = EnhancedDinov2Multimodal(
                num_species=self.target_size,
                dinov2_name=module_cfg.get("dinov2_name", "dinov2_vitb14"),
                dinov2_pretrained=module_cfg.get("dinov2_pretrained", True),
                proj_dim=module_cfg.get("proj_dim", 768),
                num_env_features=module_cfg.get("num_env_features", 27),
                env_hidden_dim=module_cfg.get("env_hidden_dim", 512),
                dropout=module_cfg.get("dropout", 0.2),
                drop_path_rate=module_cfg.get("drop_path_rate", 0.1),
                drop_key_rate=module_cfg.get("drop_key_rate", 0.1),
                use_multi_scale=module_cfg.get("use_multi_scale", True),
                use_gated_fusion=module_cfg.get("use_gated_fusion", True),
                use_enhanced_env=module_cfg.get("use_enhanced_env", True),
                use_self_distill=module_cfg.get("use_self_distill", True),
                freeze_dinov2=module_cfg.get("freeze_dinov2", True),
            )
            print(f"✅ Created DINOv2 Advanced model:")
            print(f"   - Backbone: {module_cfg.get('dinov2_name', 'dinov2_vitb14')}")
            print(f"   - DropKey rate: {module_cfg.get('drop_key_rate', 0.1)}")
            print(f"   - Multi-scale: {module_cfg.get('use_multi_scale', True)}")
            print(f"   - Gated fusion: {module_cfg.get('use_gated_fusion', True)}")
            print(f"   - Enhanced env: {module_cfg.get('use_enhanced_env', True)}")
            print(f"   - Self-distill: {module_cfg.get('use_self_distill', True)}")
            print(f"   - Species: {self.target_size}")

        else:
            raise ValueError(f"Model type '{model_type}' is not valid")

        return self.model

    def forward(self, x: Tensor, env: Optional[Tensor] = None) -> Any:
        if isinstance(self.model, (Dinov2Multimodal, EnhancedConvNeXtMultimodal, EnhancedDinov2Multimodal, Dinov2AdapterPrompt, Dinov2FullFinetune, Dinov2EVP, Dinov2HGCP, Dinov2HGCP_FDA)):
            return self.model(x, env)
        else:
            return self.model(x)

    def _set_optimizer_lr(self, lr: float) -> None:
        if not hasattr(self, "trainer") or self.trainer is None:
            return
        optimizers = getattr(self.trainer, "optimizers", None)
        if not optimizers:
            return
        for optimizer in optimizers:
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        self.log("stage_lr", lr, prog_bar=True, on_epoch=True, logger=True)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if not isinstance(self.model, Dinov2Multimodal):
            return
        self.log("backbone_frozen", float(self.model.is_backbone_frozen()), prog_bar=True, on_epoch=True, logger=True)
        if self._finetune_schedule is None or self._freeze_epochs is None or self._freeze_epochs < 0:
            return

        if (not self._has_unfrozen_backbone) and self.current_epoch >= self._freeze_epochs:
            self.model.set_backbone_trainable(True)
            self._has_unfrozen_backbone = True
            self._stage2_epoch_start = self.current_epoch
            if self._lr_warmup_epochs > 0:
                self._warmup_start_lr = self._lr_after_unfreeze * 0.2
                self._set_optimizer_lr(self._warmup_start_lr)
            else:
                self._set_optimizer_lr(self._lr_after_unfreeze)
            return

        if self._has_unfrozen_backbone and self._lr_warmup_epochs > 0:
            elapsed = self.current_epoch - self._stage2_epoch_start
            if elapsed < self._lr_warmup_epochs:
                start_lr = self._warmup_start_lr if self._warmup_start_lr is not None else self._lr_after_unfreeze * 0.2
                alpha = float(elapsed + 1) / float(max(self._lr_warmup_epochs, 1))
                new_lr = start_lr + (self._lr_after_unfreeze - start_lr) * min(max(alpha, 0.0), 1.0)
                self._set_optimizer_lr(new_lr)
            elif elapsed == self._lr_warmup_epochs:
                self._set_optimizer_lr(self._lr_after_unfreeze)

    # ⚠️ training_step, validation_step, test_step 可以保持原有逻辑，
    # 但如果用 dinov2，需要 dataloader 输出 {'sat': image, 'env': ..., 'target': ...}

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        if isinstance(self.model, (Dinov2Multimodal, EnhancedConvNeXtMultimodal, EnhancedDinov2Multimodal, Dinov2AdapterPrompt, Dinov2FullFinetune, Dinov2EVP, Dinov2HGCP, Dinov2HGCP_FDA)):
            x = batch.get("sat").squeeze(1)
            env = batch.get("env")
            y = batch["target"]
            
            # ✅ Apply Mixup/CutMix augmentation during training
            if self.training and self.mixup_cutmix is not None:
                x, y, env = self.mixup_cutmix(x, y, env)
            
            outputs = self.model(x, env)
            
            # 支持字典格式的输出
            if isinstance(outputs, dict):
                pred = outputs.get("pred", outputs.get("logits"))
                if pred is None:
                    raise ValueError("Model output dict must contain 'pred' or 'logits' key")
                # 如果 pred 是 logits，需要通过 sigmoid
                if "pred" not in outputs:
                    pred = self.sigmoid_activation(pred).type_as(y)
                loss = self.criterion(pred, y)
            else:
                pred = self.sigmoid_activation(outputs).type_as(y)
                loss = self.criterion(pred, y)
        else:
            x = batch['sat'].squeeze(1)
            y = batch['target']
            
            # ✅ Apply Mixup/CutMix augmentation during training
            if self.training and self.mixup_cutmix is not None:
                x, y, _ = self.mixup_cutmix(x, y, None)
            
            y_hat = self.forward(x)
            pred = self.sigmoid_activation(y_hat).type_as(y)
            loss = self.criterion(pred, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        if isinstance(self.model, (Dinov2Multimodal, EnhancedConvNeXtMultimodal, EnhancedDinov2Multimodal, Dinov2AdapterPrompt, Dinov2FullFinetune, Dinov2EVP, Dinov2HGCP, Dinov2HGCP_FDA)):
            x = batch.get("sat").squeeze(1)
            env = batch.get("env")
            y = batch["target"]
            outputs = self.model(x, env)
            
            # 支持字典格式的输出
            if isinstance(outputs, dict):
                pred = outputs.get("pred", outputs.get("logits"))
                if pred is None:
                    raise ValueError("Model output dict must contain 'pred' or 'logits' key")
                # 如果 pred 是 logits，需要通过 sigmoid
                if "pred" not in outputs:
                    pred = self.sigmoid_activation(pred).type_as(y)
                loss = self.criterion(pred, y)
            else:
                pred = self.sigmoid_activation(outputs).type_as(y)
                loss = self.criterion(pred, y)
        else:
            x = batch['sat'].squeeze(1)
            y = batch['target']
            y_hat = self.forward(x)
            pred = self.sigmoid_activation(y_hat).type_as(y)
            loss = self.criterion(pred, y)

        self.log("val_loss", loss, on_epoch=True)
        try:
            self.val_preds.append(pred.detach().float().cpu())
            self.val_targets.append(y.detach().float().cpu())
        except Exception:
            pass

    def on_validation_epoch_end(self) -> None:
        if self.val_preds and self.val_targets:
            preds = torch.cat(self.val_preds, dim=0)
            targets = torch.cat(self.val_targets, dim=0)
            reg = self._compute_regression_metrics(preds, targets)
            self.log("val_mae", reg["mae"], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log("val_mse", reg["mse"], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            topk = self._compute_topk_metrics(preds, targets)
            for k, v in topk.items():
                self.log(f"val_{k}", v, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            dyn_topk = self._compute_dynamic_topk(preds, targets)
            for name, value in dyn_topk.items():
                self.log(f"val_{name}", value, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.val_preds.clear()
            self.val_targets.clear()

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        if isinstance(self.model, (Dinov2Multimodal, EnhancedConvNeXtMultimodal, EnhancedDinov2Multimodal, Dinov2AdapterPrompt, Dinov2FullFinetune, Dinov2EVP, Dinov2HGCP, Dinov2HGCP_FDA)):
            x = batch.get("sat").squeeze(1)
            env = batch.get("env")
            y = batch["target"]
            outputs = self.model(x, env)
            # 支持字典格式的输出
            if isinstance(outputs, dict):
                pred = outputs.get("pred", outputs.get("logits"))
                if pred is None:
                    raise ValueError("Model output dict must contain 'pred' or 'logits' key")
                # 如果 pred 是 logits，需要通过 sigmoid
                if "pred" not in outputs:
                    pred = self.sigmoid_activation(pred).type_as(y)
            else:
                pred = self.sigmoid_activation(outputs).type_as(y)
            loss = self.criterion(pred, y)
        else:
            x = batch['sat'].squeeze(1)
            y = batch['target']
            y_hat = self.forward(x)
            pred = self.sigmoid_activation(y_hat).type_as(y)
            loss = self.criterion(pred, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        try:
            self.test_preds.append(pred.detach().float().cpu())
            self.test_targets.append(y.detach().float().cpu())
        except Exception:
            pass

    def on_test_epoch_end(self) -> None:
        if self.test_preds and self.test_targets:
            preds = torch.cat(self.test_preds, dim=0)
            targets = torch.cat(self.test_targets, dim=0)
            reg = self._compute_regression_metrics(preds, targets)
            self.log("test_mae", reg["mae"], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log("test_mse", reg["mse"], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            topk = self._compute_topk_metrics(preds, targets)
            for k, v in topk.items():
                self.log(f"test_{k}", v, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            dyn_topk = self._compute_dynamic_topk(preds, targets)
            for name, value in dyn_topk.items():
                self.log(f"test_{name}", value, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.test_preds.clear()
            self.test_targets.clear()

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_scheduler(optimizer, self.opts)
        if scheduler is None:
            return optimizer
        else:
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "frequency": 1}}


class EbirdDataModule(pl.LightningDataModule):
    def __init__(self, opts) -> None:
        super().__init__()
        self.opts = opts
        self.seed = self.opts.program.seed
        self.batch_size = self.opts.data.loaders.batch_size
        self.num_workers = self.opts.data.loaders.num_workers
        self.data_base_dir = self.opts.data.files.base
        self.targets_folder = self.opts.data.files.targets_folder
        self.env_data_folder = self.opts.data.files.env_data_folder
        self.images_folder = self.opts.data.files.images_folder
        self.df_train = pd.read_csv(os.path.join(self.data_base_dir, self.opts.data.files.train))
        self.df_val = pd.read_csv(os.path.join(self.data_base_dir, self.opts.data.files.val))
        self.df_test = pd.read_csv(os.path.join(self.data_base_dir, self.opts.data.files.test))
        self.bands = self.opts.data.bands
        self.env = self.opts.data.env
        self.env_var_sizes = self.opts.data.env_var_sizes
        self.datatype = self.opts.data.datatype
        self.target = self.opts.data.target.type
        self.subset = self.opts.data.target.subset
        self.res = self.opts.data.multiscale
        self.use_loc = self.opts.loc.use
        self.num_species = self.opts.data.total_species
        # 若使用 dinov2 或 enhanced_convnext，多模态模型内部会单独处理 env，避免拼接到图像通道
        _model_name = self.opts.experiment.module.model
        self.concat_env_to_sat = False if _model_name in ("dinov2", "enhanced_convnext", "dinov2_advanced", "dinov2_adapter_prompt", "dinov2_evp", "dinov2_hgcp", "dinov2_fda", "dinov2_hgcp_fda") else True
        # Check if using channel adapter (for multi-channel input like RGBNIR)
        module_cfg = self.opts.experiment.module
        use_channel_adapter = getattr(module_cfg, 'use_channel_adapter', False)
        in_channels = getattr(module_cfg, 'in_channels', 3)
        
        # Don't drop to RGB if using channel adapter with more channels
        if use_channel_adapter and in_channels > 3:
            self.drop_to_rgb = False
            print(f"Using channel adapter: {in_channels} channels -> keeping all channels")
        else:
            self.drop_to_rgb = True if _model_name in ["dinov2", "dinov2_adapter_prompt", "dinov2_evp", "dinov2_hgcp", "dinov2_fda", "dinov2_hgcp_fda", "dinov2_advanced"] else False

    def setup(self, stage: Optional[str] = None) -> None:
        self.all_train_dataset = EbirdVisionDataset(
            df_paths=self.df_train, data_base_dir=self.data_base_dir,
            bands=self.bands, env=self.env, env_var_sizes=self.env_var_sizes,
            transforms=get_transforms(self.opts, "train"), mode="train",
            datatype=self.datatype, target=self.target,
            targets_folder=self.targets_folder,
            env_data_folder=self.env_data_folder,
            images_folder=self.images_folder, subset=self.subset, res=self.res,
            use_loc=self.use_loc, num_species=self.num_species,
            concat_env_to_sat=self.concat_env_to_sat,
            drop_to_rgb=self.drop_to_rgb)

        self.all_test_dataset = EbirdVisionDataset(
            df_paths=self.df_test, data_base_dir=self.data_base_dir,
            bands=self.bands, env=self.env, env_var_sizes=self.env_var_sizes,
            transforms=get_transforms(self.opts, "val"), mode="test",
            datatype=self.datatype, target=self.target,
            targets_folder=self.targets_folder,
            env_data_folder=self.env_data_folder,
            images_folder=self.images_folder, subset=self.subset, res=self.res,
            use_loc=self.use_loc, num_species=self.num_species,
            concat_env_to_sat=self.concat_env_to_sat,
            drop_to_rgb=self.drop_to_rgb)

        self.all_val_dataset = EbirdVisionDataset(
            df_paths=self.df_val, data_base_dir=self.data_base_dir,
            bands=self.bands, env=self.env, env_var_sizes=self.env_var_sizes,
            transforms=get_transforms(self.opts, "val"), mode="val",
            datatype=self.datatype, target=self.target,
            targets_folder=self.targets_folder,
            env_data_folder=self.env_data_folder,
            images_folder=self.images_folder, subset=self.subset, res=self.res,
            use_loc=self.use_loc, num_species=self.num_species,
            concat_env_to_sat=self.concat_env_to_sat,
            drop_to_rgb=self.drop_to_rgb)

        self.train_dataset = self.all_train_dataset
        self.test_dataset = self.all_test_dataset
        self.val_dataset = self.all_val_dataset

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
            persistent_workers=True if self.num_workers and self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers and self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            persistent_workers=True if self.num_workers and self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers and self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            persistent_workers=True if self.num_workers and self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers and self.num_workers > 0 else None,
        )
