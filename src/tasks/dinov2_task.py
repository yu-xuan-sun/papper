# src/tasks/dinov2_task.py
import json
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import MeanAbsoluteError

from src.models.dinov2_multimodal import Dinov2Multimodal


class Dinov2Task(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        # 处理 optimizer 超参数，避免 config.optimizer 是 str 时出错
        if isinstance(config.optimizer, dict):
            lr = float(config.optimizer.get("lr", 1e-4))
            weight_decay = float(config.optimizer.get("weight_decay", 0.01))
        else:
            lr = 1e-4
            weight_decay = 0.01

        # 处理 experiment.module
        module_cfg = getattr(config.experiment, "module", {})
        proj_dim = int(module_cfg.get("proj_dim", 768)) if isinstance(module_cfg, dict) else 768
        freeze_dinov2 = bool(module_cfg.get("freeze_dinov2", True)) if isinstance(module_cfg, dict) else True

        # 保存关键超参
        self.save_hyperparameters({
            "lr": lr,
            "weight_decay": weight_decay,
            "proj_dim": proj_dim,
            "freeze_dinov2": freeze_dinov2,
        })

        # 保存完整配置
        self.config = config


        num_species = int(config.data.total_species)
        module_cfg = config.experiment.module
        dinov2_name = module_cfg.get("dinov2_name", "vit_small_patch14_224.dinov2")
        n_prompts = int(module_cfg.get("n_prompts", 8))
        env_backbone = module_cfg.get("env_backbone", "resnet18")
        freeze_dinov2 = bool(module_cfg.get("freeze_dinov2", True))
        proj_dim = int(module_cfg.get("proj_dim", 768))
        dinov2_pretrained = bool(module_cfg.get("dinov2_pretrained", True))
        use_grad_ckpt = bool(module_cfg.get("use_grad_ckpt", False))
        use_channels_last = bool(module_cfg.get("use_channels_last", False))
        use_global_prompt = bool(module_cfg.get("use_global_prompt", True))
        use_block_prompts = bool(module_cfg.get("use_block_prompts", False))
        block_prompt_length = int(module_cfg.get("block_prompt_length", 2))
        block_prompt_dropout = float(module_cfg.get("block_prompt_dropout", 0.0))
        adapter_reduction_val = module_cfg.get("adapter_reduction", None)
        adapter_reduction = int(adapter_reduction_val) if adapter_reduction_val not in (None, "", "null") else None
        adapter_dropout = float(module_cfg.get("adapter_dropout", 0.0))
        condition_prompts_on_env = bool(module_cfg.get("condition_prompts_on_env", False))
        train_block_layer_norm = bool(module_cfg.get("train_block_layer_norm", True))

        self.model = Dinov2Multimodal(
            num_species=num_species,
            dinov2_name=dinov2_name,
            dinov2_pretrained=dinov2_pretrained,
            n_prompts=n_prompts,
            env_backbone=env_backbone,
            proj_dim=proj_dim,
            freeze_dinov2=freeze_dinov2,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_grad_ckpt=use_grad_ckpt,
            use_channels_last=use_channels_last,
            use_block_prompts=use_block_prompts,
            block_prompt_length=block_prompt_length,
            block_prompt_dropout=block_prompt_dropout,
            adapter_reduction=adapter_reduction,
            adapter_dropout=adapter_dropout,
            condition_prompts_on_env=condition_prompts_on_env,
            train_block_layer_norm=train_block_layer_norm,
            use_global_prompt=use_global_prompt,
        )

        # loss 权重
        # loss 权重
        self.pres_w = float(config.losses.get("hurdle_presence_w", 1.0))
        self.cond_w = float(config.losses.get("hurdle_cond_w", 1.0))

        # Ranking Loss配置
        self.use_ranking_loss = bool(module_cfg.get("use_ranking_loss", False))
        if self.use_ranking_loss:
            from src.losses.ranking_loss import ListwiseLoss, ApproxNDCGLoss
            
            self.ranking_loss_weight = float(module_cfg.get("ranking_loss_weight", 0.2))
            self.ranking_loss_type = module_cfg.get("ranking_loss_type", "listwise")
            
            if self.ranking_loss_type == "listwise":
                top_k = int(module_cfg.get("listwise_top_k", 30))
                temperature = float(module_cfg.get("ranking_temperature", 1.0))
                self.ranking_loss_fn = ListwiseLoss(top_k=top_k, temperature=temperature)
                print(f"✓ Using Listwise Ranking Loss (top_k={top_k}, temp={temperature})")
            
            elif self.ranking_loss_type == "approx_ndcg":
                top_k = int(module_cfg.get("ndcg_top_k", 30))
                temperature = float(module_cfg.get("ndcg_temperature", 0.1))
                self.ranking_loss_fn = ApproxNDCGLoss(top_k=top_k, temperature=temperature)
                print(f"✓ Using ApproxNDCG Ranking Loss (top_k={top_k}, temp={temperature})")
            
            else:
                raise ValueError(f"Unknown ranking_loss_type: {self.ranking_loss_type}")
            
            print(f"✓ Ranking loss weight: {self.ranking_loss_weight:.2f}")

        # metrics
        self.train_mae = MeanAbsoluteError()

    self.val_preds = []
    self.val_targets = []
    self.test_preds = []
    self.test_targets = []
    self.proto_records: list[Dict[str, Any]] = []
        logging_cfg = getattr(config, "logging", {})
        try:
            self.proto_topk = int(logging_cfg.get("proto_topk", 5))  # type: ignore[attr-defined]
        except Exception:
            self.proto_topk = 5

    def forward(self, x_img, x_env):
        return self.model(x_img, x_env)

    def _unpack_batch(self, batch: Dict[str, Any]):
        """
        兼容不同 dataloader 输出格式
        """
        meta: Dict[str, Any] = {}
        if "hotspot_id" in batch:
            meta["hotspot_id"] = batch["hotspot_id"]
        if "image" in batch and "target" in batch:
            images = batch["image"]
            target = batch["target"]
            env = batch.get("env", None)

        elif "sat" in batch and "target" in batch:
            images = batch["sat"]
            target = batch["target"]
            env = {
                "bioclim": batch.get("bioclim", None),
                "ped": batch.get("ped", None),
                "num_complete_checklists": batch.get("num_complete_checklists", None),
                "hotspot_id": batch.get("hotspot_id", None),
            }

        elif "images" in batch and "targets" in batch:
            images = batch["images"]
            target = batch["targets"]
            env = batch.get("env", None)

        else:
            raise ValueError(f"_unpack_batch 收到未知格式: keys={batch.keys()}")

        return images, env, target, meta

    def _compute_losses(self, outputs: Dict[str, Any], targets: torch.Tensor):
        """
        outputs: dict from model.forward
        targets: (B,S) float tensor in [0,1]
        """
        p_probs = outputs["presence_prob"]
        a_vals = outputs["conditional_abundance"]
        preds = outputs["pred"]

        presence_labels = (targets > 0).float()
        p_probs = torch.clamp(p_probs, 1e-6, 1.0 - 1e-6)

        # presence loss
        loss_presence = F.binary_cross_entropy(p_probs, presence_labels)

        # conditional loss
        mask = presence_labels.bool()
        if mask.any():
            loss_cond = F.mse_loss(a_vals[mask], targets[mask])
        else:
            loss_cond = torch.tensor(0.0, device=targets.device)

        total_loss = self.pres_w * loss_presence + self.cond_w * loss_cond
        
        # 新增: Ranking Loss (仅在训练时计算)
        ranking_loss = None
        if self.use_ranking_loss and self.training:
            ranking_loss = self.ranking_loss_fn(preds, targets)
            # 组合loss: (1-λ)*原始loss + λ*Ranking
            total_loss = (1 - self.ranking_loss_weight) * total_loss + \
                         self.ranking_loss_weight * ranking_loss
        
        return total_loss, loss_presence, loss_cond, preds, ranking_loss

    def training_step(self, batch, batch_idx):
        images, env, target, _ = self._unpack_batch(batch)
        outputs = self.forward(images, env)
        total_loss, loss_presence, loss_cond, preds, ranking_loss = self._compute_losses(outputs, target)

        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_presence", loss_presence, prog_bar=False, on_epoch=True)
        self.log("train_loss_cond", loss_cond, prog_bar=False, on_epoch=True)
        if ranking_loss is not None:
            self.log("train_ranking_loss", ranking_loss, prog_bar=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, env, target, meta = self._unpack_batch(batch)
        outputs = self.forward(images, env)
        total_loss, loss_presence, loss_cond, preds, _ = self._compute_losses(outputs, target)

        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss_presence", loss_presence, prog_bar=False, on_epoch=True)
        self.log("val_loss_cond", loss_cond, prog_bar=False, on_epoch=True)
        self.val_preds.append(preds.detach().float().cpu())
        self.val_targets.append(target.detach().float().cpu())
        return total_loss

    def test_step(self, batch, batch_idx):
        images, env, target, meta = self._unpack_batch(batch)
        outputs = self.forward(images, env)
        total_loss, _, _, preds, _ = self._compute_losses(outputs, target)

        mae = F.l1_loss(preds, target)
        mse = F.mse_loss(preds, target)
        self.log("test_loss", total_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_mse", mse, prog_bar=False, on_step=False, on_epoch=True)
        self.test_preds.append(preds.detach().float().cpu())
        self.test_targets.append(target.detach().float().cpu())
        proto_sim = outputs.get("proto_sim")
        if proto_sim is not None:
            self._record_proto_interpretation(proto_sim, meta)
        return {"test_loss": total_loss.detach()}

    def configure_optimizers(self):
        base_lr = float(self.config.optimizer.get("lr", 1e-4))
        wd = float(self.config.optimizer.get("weight_decay", 0.01))

        optimizer = torch.optim.AdamW(self.parameters(), lr=base_lr, weight_decay=wd)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

    # -------------------------
    # Evaluation helpers
    # -------------------------
    def _compute_topk_metrics(self, preds: torch.Tensor, targets: torch.Tensor, ks=(1, 5, 10, 30)) -> Dict[str, torch.Tensor]:
        if preds.ndim != 2:
            preds = preds.view(preds.size(0), -1)
        if targets.ndim != 2:
            targets = targets.view(targets.size(0), -1)

        n, c = preds.shape
        maxk = min(max(ks), c)
        pred_topk = torch.topk(preds, k=maxk, dim=1, largest=True).indices
        target_topk = torch.topk(targets, k=maxk, dim=1, largest=True).indices

        metrics: Dict[str, torch.Tensor] = {}
        for k in ks:
            kk = min(k, c)
            if kk <= 0:
                continue
            pred_top = pred_topk[:, :kk]
            target_top = target_topk[:, :kk]
            matches = (pred_top.unsqueeze(2) == target_top.unsqueeze(1)).any(dim=2).float().sum(dim=1)
            metrics[f"top{k}_acc"] = (matches / kk).mean()
        return metrics

    def _compute_dynamic_topk(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
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

    def on_validation_epoch_end(self) -> None:
        if not self.val_preds:
            return
        preds = torch.cat(self.val_preds, dim=0)
        targets = torch.cat(self.val_targets, dim=0)
        mae = torch.mean(torch.abs(preds - targets))
        mse = torch.mean((preds - targets) ** 2)
        self.log("val_mae", mae, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
        self.log("val_mse", mse, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
        topk = self._compute_topk_metrics(preds, targets)
        for k, v in topk.items():
            self.log(f"val_{k}", v, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
        dyn_topk = self._compute_dynamic_topk(preds, targets)
        for name, value in dyn_topk.items():
            self.log(f"val_{name}", value, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
        self.val_preds.clear()
        self.val_targets.clear()

    def on_test_epoch_end(self) -> None:
        if self.test_preds:
            preds = torch.cat(self.test_preds, dim=0)
            targets = torch.cat(self.test_targets, dim=0)
            mae = torch.mean(torch.abs(preds - targets))
            mse = torch.mean((preds - targets) ** 2)
            self.log("test_mae_epoch", mae, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
            self.log("test_mse_epoch", mse, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
            topk = self._compute_topk_metrics(preds, targets)
            for k, v in topk.items():
                self.log(f"test_{k}", v, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
            dyn_topk = self._compute_dynamic_topk(preds, targets)
            for name, value in dyn_topk.items():
                self.log(f"test_{name}", value, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
            self.test_preds.clear()
            self.test_targets.clear()

        save_path = getattr(self.config, "save_preds_path", "")
        if self.proto_records and save_path:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as fp:
                json.dump(self.proto_records, fp, ensure_ascii=False, indent=2)
            self.proto_records.clear()

    def _record_proto_interpretation(self, proto_sim: torch.Tensor, meta: Dict[str, Any]) -> None:
        save_path = getattr(self.config, "save_preds_path", "")
        if not save_path:
            return
        hotspot_ids = meta.get("hotspot_id")
        if hotspot_ids is None:
            return
        if isinstance(hotspot_ids, torch.Tensor):
            hotspot_ids = hotspot_ids.tolist()

        top_k = min(self.proto_topk, proto_sim.shape[1])
        values, indices = torch.topk(proto_sim.detach().cpu(), k=top_k, dim=1)

        for i in range(values.size(0)):
            record = {
                "hotspot_id": hotspot_ids[i] if i < len(hotspot_ids) else None,
                "top_prototype_indices": indices[i].tolist(),
                "top_prototype_scores": values[i].tolist(),
            }
            self.proto_records.append(record)
