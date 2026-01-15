import os
import time
import json
from typing import Any, Dict, cast

import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, DictConfig, ListConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from src.utils.config_utils import load_opts
import src.trainer.trainer as general_trainer
from src.utils.compute_normalization_stats import (
    compute_means_stds_env_vars,
    compute_means_stds_images,
    compute_means_stds_images_visual
)
import torch
import torch.multiprocessing as mp


def _to_primitive(o):
    if isinstance(o, (DictConfig, ListConfig)):
        return OmegaConf.to_container(o, resolve=True)
    return o


def _extract_hparams(config) -> Dict[str, Any]:
    hp = {}
    try:
        hp["model"] = _to_primitive(config.experiment.module.model)
        hp["fc_type"] = _to_primitive(config.experiment.module.fc)
        hp["pretrained"] = _to_primitive(config.experiment.module.pretrained)
        hp["bands"] = _to_primitive(config.data.bands)
        hp["datatype"] = _to_primitive(config.data.datatype)
        hp["batch_size"] = _to_primitive(config.data.loaders.batch_size)
        hp["num_workers"] = _to_primitive(config.data.loaders.num_workers)
        hp["seed"] = _to_primitive(config.experiment.seed)
        hp["optimizer"] = _to_primitive(config.optimizer)
        hp["loss"] = _to_primitive(config.losses.criterion)
        hp["max_epochs"] = _to_primitive(config.max_epochs)
    except Exception as e:
        print(f"[WARN] extracting hparams failed: {e}")
    return hp


def _build_run_dir(config, base_dir: str, seed: int) -> str:
    run_name = getattr(config.logging, "run_name", "default_run")
    root_save = config.save_path if config.save_path else "runs"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, root_save, f"{run_name}_seed{seed}_{timestamp}")
    return run_dir


def _setup_loggers(config, run_dir: str):
    loggers = []
    if getattr(config.logging, "enable_tensorboard", True):
        tb_logger = TensorBoardLogger(
            save_dir=run_dir,
            name="tb_logs",
            default_hp_metric=False
        )
        loggers.append(tb_logger)
    if getattr(config.logging, "enable_csv", True):
        csv_logger = CSVLogger(
            save_dir=run_dir,
            name="csv_logs"
        )
        loggers.append(csv_logger)
    return loggers


def _find_latest_last_ckpt(save_root: str, run_name_prefix: str, exclude_dir: str = "") -> str:
    """
    在 save_root 下查找与 run_name_prefix 开头的历史 run 目录，按时间戳逆序，
    返回第一个存在 checkpoints/last.ckpt 的完整路径；若找不到则返回空串。
    exclude_dir 用于排除当前新建的 run 目录。
    """
    if not os.path.isdir(save_root):
        return ""
    try:
        candidates = []
        for name in os.listdir(save_root):
            full = os.path.join(save_root, name)
            if not os.path.isdir(full):
                continue
            if not name.startswith(run_name_prefix):
                continue
            if exclude_dir and os.path.abspath(full) == os.path.abspath(exclude_dir):
                continue
            candidates.append(full)
        # 按目录名（包含时间戳）逆序排序，越新的越靠前
        candidates.sort(reverse=True)
        for d in candidates:
            last_ckpt = os.path.join(d, "checkpoints", "last.ckpt")
            if os.path.exists(last_ckpt):
                return last_ckpt
    except Exception:
        pass
    return ""


@hydra.main(version_base=None, config_path="configs", config_name="hydra")
def main(opts):
    # 更稳健的多进程 & 共享设置，避免资源共享异常
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError:
        pass
    hydra_opts = dict(OmegaConf.to_container(opts))
    args = hydra_opts.pop("args", None)
    if args is None:
        raise ValueError("Missing 'args' section in hydra config.")

    base_dir = args.get("base_dir", "")
    run_id = args.get("run_id", 1)
    # Coerce run_id to int safely (handles empty string or non-numeric)
    if isinstance(run_id, str):
        run_id = int(run_id) if run_id.strip().isdigit() else 1
    elif not isinstance(run_id, int):
        try:
            run_id = int(run_id)
        except Exception:
            run_id = 1
    if not base_dir:
        base_dir = get_original_cwd()

    config_path = os.path.join(base_dir, args["config"])
    default_config = os.path.join(base_dir, "configs/defaults.yaml")
    print(f"using config  {config_path}")
    print(default_config)
    config = load_opts(config_path, default=default_config, commandline_opts=hydra_opts)

    base_seed = config.experiment.seed
    global_seed = base_seed + (run_id - 1)
    pl.seed_everything(global_seed, workers=True)

    run_dir = _build_run_dir(config, base_dir, global_seed)
    os.makedirs(run_dir, exist_ok=True)
    config.base_dir = base_dir
    config.run_dir = run_dir

    # 启用 TF32 与 cuDNN benchmark，提高训练吞吐
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("[Optimization] Enabled TF32 and cuDNN benchmark")
        except Exception:
            pass

    # ========================
    # 均值方差统计
    # ========================
    if hasattr(config.data, "env") and len(config.data.env) > 0:
        (config.variables.bioclim_means,
         config.variables.bioclim_std,
         config.variables.ped_means,
         config.variables.ped_std) = compute_means_stds_env_vars(
            root_dir=config.data.files.base,
            train_csv=config.data.files.train,
            env=config.data.env,
            env_data_folder=config.data.files.env_data_folder,
            output_file_means=config.data.files.env_means,
            output_file_std=config.data.files.env_stds
        )
    if config.data.datatype == "refl":
        (config.variables.rgbnir_means,
         config.variables.rgbnir_std) = compute_means_stds_images(
            root_dir=config.data.files.base,
            train_csv=config.data.files.train,
            output_file_means=config.data.files.rgbnir_means,
            output_file_std=config.data.files.rgbnir_stds
        )
    elif config.data.datatype == "img" and not config.data.transforms[4].normalize_by_255:
        (config.variables.visual_means,
         config.variables.visual_stds) = compute_means_stds_images_visual(
            root_dir=config.data.files.base,
            train_csv=config.data.files.train,
            output_file_means=config.data.files.rgb_means,
            output_file_std=config.data.files.rgb_stds
        )

    # 保存配置与 hparams
    with open(os.path.join(run_dir, "config_full.yaml"), "w") as f:
        OmegaConf.save(config=config, f=f)
    hparams = _extract_hparams(config)
    with open(os.path.join(run_dir, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2, ensure_ascii=False)

    print("Using trainer entrypoint..")
    print(f"Run directory: {run_dir}")
    print(f"Predicting {config.data.total_species} species")
    print(f"Datatype={config.data.datatype} Bands={config.data.bands}")
    model_name = config.experiment.module.model
    print(f"Model = {model_name}")

    # ========================
    # 模型 / 数据模块分支
    # ========================
    if model_name == "dinov2_hurdle":
        try:
            from src.models.dinov2_hurdle_module import DinoHurdleModule
        except ImportError as e:
            raise ImportError(f"无法导入 DinoHurdleModule，确认 src/models/dinov2_hurdle_module.py 存在: {e}")
        if not hasattr(config.data, "total_species"):
            raise ValueError("缺少 data.total_species，请在 YAML 中定义。")
        task = DinoHurdleModule(config)
        datamodule = general_trainer.EbirdDataModule(config)
    else:
        task = general_trainer.EbirdTask(config)
        datamodule = general_trainer.EbirdDataModule(config)

    # ========================
    # Trainer 参数整理
    # ========================
    raw_trainer_cfg = getattr(config, "trainer", None)
    if raw_trainer_cfg is None:
        trainer_args: Dict[str, Any] = {}
    else:
        trainer_args = OmegaConf.to_object(raw_trainer_cfg) or {}
    if not isinstance(trainer_args, dict):
        trainer_args = {}

    loggers = _setup_loggers(config, run_dir)
    trainer_args["logger"] = loggers if len(loggers) > 1 else loggers[0]

    if "callbacks" not in trainer_args or trainer_args["callbacks"] is None:
        trainer_args["callbacks"] = []
    elif not isinstance(trainer_args["callbacks"], list):
        trainer_args["callbacks"] = list(trainer_args["callbacks"])

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_topk",
        dirpath=ckpt_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="max",
        save_last=True,
        save_weights_only=False,
        auto_insert_metric_name=False
    )
    trainer_args["callbacks"].append(checkpoint_callback)
    trainer_args["overfit_batches"] = config.overfit_batches
    trainer_args["max_epochs"] = config.max_epochs
    trainer_args["max_steps"] = getattr(config, "max_steps", -1)

    # 默认补充的稳定性参数（若 YAML 已设则不会覆盖）
    trainer_args.setdefault("accelerator", "gpu" if torch.cuda.is_available() else "cpu")
    trainer_args.setdefault("devices", 1)
    trainer_args.setdefault("precision", "16-mixed")
    trainer_args.setdefault("deterministic", True)
    trainer_args.setdefault("gradient_clip_val", 5.0)
    trainer_args.setdefault("log_every_n_steps", 50)

    # Auto LR (可选)
    if getattr(config, "auto_lr_find", False):
        tmp_trainer = pl.Trainer(**trainer_args)
        lr_finder = tmp_trainer.tuner.lr_find(task, datamodule=datamodule)
        new_lr = lr_finder.suggestion()
        print(f"[AutoLR] suggested lr={new_lr}")
        # 尽量兼容旧写法
        if hasattr(task, "hparams"):
            task.hparams.lr = new_lr
            task.hparams.learning_rate = new_lr
        if hasattr(task, "lr"):
            task.lr = new_lr

    # 确保 accelerator 被正确设置（修复潜在的 None 值）
    if trainer_args.get("accelerator") is None:
        trainer_args["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"

    # 移除 PyTorch Lightning 2.0+ 中废弃的参数
    deprecated_params = [
        'checkpoint_callback', 'weights_summary', 'weights_save_path',
        'progress_bar_refresh_rate', 'num_processes', 'gpus',
        'log_gpu_memory', 'auto_select_gpus', 'tpu_cores',
        'terminate_on_nan', 'amp_backend', 'replace_sampler_ddp',
        'process_position', 'flush_logs_every_n_steps',
        'prepare_data_per_node', 'reload_dataloaders_every_epoch',
        'track_grad_norm', 'auto_lr_find', 'auto_scale_batch_size',
        'benchmark', 'profiler', 'resume_from_checkpoint',
        'move_metrics_to_cpu'
    ]
    for param in deprecated_params:
        if param in trainer_args:
            print(f"[Warning] Removing deprecated parameter: {param}={trainer_args[param]}")
            del trainer_args[param]
    
    # 移除 PyTorch Lightning 2.0+ 中废弃的参数
    deprecated_params = [
        'checkpoint_callback', 'weights_summary', 'weights_save_path',
        'progress_bar_refresh_rate', 'num_processes', 'gpus',
        'log_gpu_memory', 'auto_select_gpus', 'tpu_cores',
        'terminate_on_nan', 'amp_backend', 'replace_sampler_ddp',
        'process_position', 'flush_logs_every_n_steps',
        'prepare_data_per_node', 'reload_dataloaders_every_epoch',
        'track_grad_norm', 'auto_lr_find', 'auto_scale_batch_size',
        'benchmark', 'profiler', 'resume_from_checkpoint',
        'move_metrics_to_cpu'
    ]
    for param in deprecated_params:
        if param in trainer_args:
            print(f"[Warning] Removing deprecated parameter: {param}={trainer_args[param]}")
            del trainer_args[param]
    
    # 处理 strategy=None 的情况
    if trainer_args.get("strategy") is None:
        trainer_args.pop("strategy", None)
    
    # 处理 strategy=None 的情况
    if trainer_args.get("strategy") is None:
        trainer_args.pop("strategy", None)
    
    trainer = pl.Trainer(**trainer_args)

    # 推理 / 评估模式
    if getattr(config, "load_ckpt_path", ""):
        ckpt_path = config.load_ckpt_path
        print(f"[Eval] 使用已训练权重: {ckpt_path}")
        trainer.validate(model=task, datamodule=datamodule, ckpt_path=ckpt_path)
        trainer.test(model=task, datamodule=datamodule, ckpt_path=ckpt_path)
        print("Evaluation finished.")
        return

    # 训练（支持断点续训）
    # 优先：当前 run 目录下的 last.ckpt；否则：到历史 runs 中按时间戳搜索最近的 last.ckpt
    last_ckpt_curr = os.path.join(ckpt_dir, 'last.ckpt')
    ckpt_resume = last_ckpt_curr if os.path.exists(last_ckpt_curr) else None
    if ckpt_resume is None:
        root_save = config.save_path if config.save_path else "runs"
        # 前缀与 _build_run_dir 一致：{run_name}_seed{seed}_
        run_name = getattr(config.logging, "run_name", "default_run")
        run_prefix = f"{run_name}_seed{global_seed}_"
        history_last = _find_latest_last_ckpt(os.path.join(base_dir, root_save), run_prefix, exclude_dir=run_dir)
        ckpt_resume = history_last or None
    if ckpt_resume:
        print(f"[Resume] 从断点恢复训练: {ckpt_resume}")
    else:
        print("[Resume] 未找到 last.ckpt，从头开始训练")

    # 尝试用完整 checkpoint 恢复；若是历史上仅保存权重的 ckpt，则退化为只加载模型权重
    try:
        trainer.fit(model=task, datamodule=datamodule, ckpt_path=ckpt_resume)
    except (KeyError, RuntimeError) as e:
        msg = str(e)
        fallback_needed = False
        if ckpt_resume:
            if isinstance(e, KeyError) and "contains only the model" in msg:
                fallback_needed = True
            elif isinstance(e, RuntimeError) and ("Missing key(s)" in msg or "Unexpected key(s)" in msg):
                print(f"[Resume-Fallback] state_dict 键不匹配: {msg.splitlines()[0] if msg else msg}")
                fallback_needed = True
        if fallback_needed:
            print("[Resume-Fallback] 尝试仅加载可匹配的模型权重并重新开始优化器状态。")
            try:
                ckpt = torch.load(ckpt_resume, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt)
                missing, unexpected = task.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"[Resume-Fallback] Missing keys: {len(missing)}")
                if unexpected:
                    print(f"[Resume-Fallback] Unexpected keys: {len(unexpected)}")
            except Exception as e2:
                print(f"[Resume-Fallback] 加载模型权重失败: {e2}")
            clean_args = dict(trainer_args)
            cbs = list(clean_args.get("callbacks", []))
            cbs = [cb for cb in cbs if getattr(cb, "__class__", type("x", (), {})).__name__ != "GradientAccumulationScheduler"]
            clean_args["callbacks"] = cbs
            trainer_fresh = pl.Trainer(**clean_args)
            trainer_fresh.fit(model=task, datamodule=datamodule, ckpt_path=None)
        else:
            raise
    # 测试（best）
    trainer.test(model=task, datamodule=datamodule, ckpt_path="best")

    print("Training finished.")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Logs & artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()
