#!/bin/bash
# ================================================================
# SatBird 改进实验运行脚本
# ================================================================

# 激活conda环境
source /sunyuxuan/miniconda3/etc/profile.d/conda.sh
conda activate satbird1

cd /sunyuxuan/satbird

echo "============================================"
echo "SatBird 改进实验"
echo "============================================"

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# ================================================================
# 消融实验
# ================================================================

echo ""
echo "============================================"
echo "消融实验1: ListMLE Loss改进"
echo "============================================"
python train.py args.config=configs/SatBird-USA-summer/ablation_listmle.yaml

echo ""
echo "============================================"
echo "消融实验2: 类别权重"
echo "============================================"
python train.py args.config=configs/SatBird-USA-summer/ablation_class_weights.yaml

echo ""
echo "============================================"
echo "消融实验3: 无Adapter (验证Adapter贡献)"
echo "============================================"
python train.py args.config=configs/SatBird-USA-summer/ablation_no_adapter.yaml

echo ""
echo "============================================"
echo "所有实验完成!"
echo "============================================"
