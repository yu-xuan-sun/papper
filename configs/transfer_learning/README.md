# Transfer Learning Experiments for HGCP+FDA Model

## 概述

本目录包含三种迁移学习场景的配置文件：

1. **季节迁移 (Season Transfer)**: USA-Summer → USA-Winter
2. **地域迁移 (Geographic Transfer)**: USA → Kenya
3. **物种迁移 (Species Transfer)**: Bird → Butterfly

## 实验设计

### 源模型
- 模型：HGCP+FDA (Hierarchical Gated Cross-modal Prompt + Frequency Domain Adaptation)
- 训练数据：USA-Summer (670种鸟类)
- Checkpoint: `runs/hgcp_fda_summer_seed42/checkpoints/last.ckpt`

### 迁移策略

| 策略 | Backbone | Adapter | HGCP | FDA | Classifier | 学习率 |
|------|----------|---------|------|-----|------------|--------|
| Linear Probe | ❄️ 冻结 | ❄️ 冻结 | ❄️ 冻结 | ❄️ 冻结 | 🔥 训练 | 1e-3 |
| Adapter Tune | ❄️ 冻结 | 🔥 训练 | 🔥 训练 | 🔥 训练 | 🔥 训练 | 3e-4 |
| Full Fine-tune | 🔥 部分 | 🔥 训练 | 🔥 训练 | 🔥 训练 | 🔥 训练 | 1e-4 |

## 配置文件说明

### 季节迁移 (USA-Summer → USA-Winter)
- `season_transfer_linear.yaml` - Linear Probe策略
- `season_transfer_adapter.yaml` - Adapter Tune策略
- `season_transfer_finetune.yaml` - Full Fine-tune策略

**特点**: 
- 相同物种 (670种)
- 相同图像通道 (RGBNIR)
- 相同环境变量 (27个)
- `reinit_classifier: false` (可复用分类器)

### 地域迁移 (USA → Kenya)
- `geo_transfer_linear.yaml` - Linear Probe策略
- `geo_transfer_adapter.yaml` - Adapter Tune策略
- `geo_transfer_finetune.yaml` - Full Fine-tune策略

**特点**:
- 不同物种 (670 → 1054种)
- 不同图像通道 (RGBNIR → RGB)
- 不同环境变量 (27 → 19个)
- `reinit_classifier: true` (必须重新初始化)
- `reinit_env_encoder: true` (环境编码器需重新初始化)

### 物种迁移 (Bird → Butterfly)
- `species_transfer_linear.yaml` - Linear Probe策略
- `species_transfer_adapter.yaml` - Adapter Tune策略
- `species_transfer_finetune.yaml` - Full Fine-tune策略

**特点**:
- 不同物种类群 (670种鸟 → 172种蝴蝶)
- 相同图像通道 (RGBNIR)
- 相同环境变量 (27个)
- `reinit_classifier: true` (必须重新初始化)

## 运行实验

### 运行所有实验
```bash
bash scripts/run_transfer_experiments.sh
```

### 运行特定实验
```bash
# 只运行季节迁移
bash scripts/run_transfer_experiments.sh --experiment season

# 只运行地域迁移
bash scripts/run_transfer_experiments.sh --experiment geo

# 只运行物种迁移
bash scripts/run_transfer_experiments.sh --experiment species
```

### 单独运行配置
```bash
# 季节迁移 - Linear Probe
python train.py --config configs/transfer_learning/season_transfer_linear.yaml

# 地域迁移 - Adapter Tune
python train.py --config configs/transfer_learning/geo_transfer_adapter.yaml

# 物种迁移 - Full Fine-tune
python train.py --config configs/transfer_learning/species_transfer_finetune.yaml
```

## 评估结果

```bash
# 生成评估报告
python scripts/transfer_learning_eval.py --visualize

# 只评估特定实验
python scripts/transfer_learning_eval.py --experiment_type season --visualize
```

## 预期结果

| 迁移类型 | 难度 | 预期mAP范围 |
|----------|------|-------------|
| 季节迁移 | 低 | 0.35-0.45 |
| 地域迁移 | 高 | 0.15-0.25 |
| 物种迁移 | 中-高 | 0.20-0.35 |

## 关键发现

1. **季节迁移**通常表现最好，因为数据分布相似
2. **地域迁移**面临更大挑战，需要适应不同的物种和环境
3. **物种迁移**测试模型的跨类群泛化能力
4. **Adapter Tuning**通常在性能和效率之间提供良好平衡

## 文件结构

```
configs/transfer_learning/
├── README.md                        # 本文档
├── season_transfer_linear.yaml      # 季节-Linear
├── season_transfer_adapter.yaml     # 季节-Adapter
├── season_transfer_finetune.yaml    # 季节-Finetune
├── geo_transfer_linear.yaml         # 地域-Linear
├── geo_transfer_adapter.yaml        # 地域-Adapter
├── geo_transfer_finetune.yaml       # 地域-Finetune
├── species_transfer_linear.yaml     # 物种-Linear
├── species_transfer_adapter.yaml    # 物种-Adapter
└── species_transfer_finetune.yaml   # 物种-Finetune
```
