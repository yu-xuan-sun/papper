#!/bin/bash
# ================================================================
# Ensemble Training Script
# 训练3个不同seed的模型用于集成学习
# ================================================================

echo "============================================================"
echo "Ensemble Training - 3 models with different seeds"
echo "============================================================"

cd /sunyuxuan/satbird

# 激活conda环境
source /sunyuxuan/miniconda3/etc/profile.d/conda.sh
conda activate satbird1

# 训练模型 1 (seed=1001)
echo ""
echo "[1/3] Training model with seed 1001..."
echo "============================================================"
python train.py args.config=configs/ensemble/ensemble_summer_seed1.yaml

# 训练模型 2 (seed=1002)
echo ""
echo "[2/3] Training model with seed 1002..."
echo "============================================================"
python train.py args.config=configs/ensemble/ensemble_summer_seed2.yaml

# 训练模型 3 (seed=1003)
echo ""
echo "[3/3] Training model with seed 1003..."
echo "============================================================"
python train.py args.config=configs/ensemble/ensemble_summer_seed3.yaml

echo ""
echo "============================================================"
echo "All training completed!"
echo "============================================================"
echo ""
echo "To evaluate the ensemble, run:"
echo "python scripts/ensemble_evaluate.py \\"
echo "    --run_dirs runs/ensemble_summer_seed1* runs/ensemble_summer_seed2* runs/ensemble_summer_seed3* \\"
echo "    --config configs/ensemble/ensemble_summer_seed1.yaml"
