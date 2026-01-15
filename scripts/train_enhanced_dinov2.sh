#!/bin/bash
# Quick Start Script for Enhanced DINOv2 Training

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="/data1/sunyuxuan/SatBird"
cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Enhanced DINOv2 Training${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查环境
echo -e "\n${YELLOW}[1/4] 检查环境...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python未安装${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python: $(python --version)${NC}"

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ CUDA可用${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1
else
    echo -e "${YELLOW}⚠️ CUDA不可用${NC}"
fi

# 检查数据集
echo -e "\n${YELLOW}[2/4] 检查数据集...${NC}"
DATASET="USA_winter"
DATA_DIR="$PROJECT_ROOT/$DATASET"

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ 数据集不存在: $DATA_DIR${NC}"
    exit 1
fi

required_files=("train_split.csv" "valid_split.csv" "test_split.csv" "species_list.txt")
for file in "${required_files[@]}"; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo -e "${RED}❌ 缺少文件: $file${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ 数据集完整${NC}"

# 检查配置文件
echo -e "\n${YELLOW}[3/4] 检查配置...${NC}"
CONFIG="configs/SatBird-USA-winter/dinov2_advanced.yaml"
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}❌ 配置文件不存在: $CONFIG${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 配置文件: $CONFIG${NC}"

# 启动训练
echo -e "\n${YELLOW}[4/4] 启动训练...${NC}"
echo -e "${BLUE}========================================${NC}\n"

read -p "确认开始训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}训练已取消${NC}"
    exit 0
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 创建日志目录
LOG_DIR="$PROJECT_ROOT/runs/dinov2_advanced_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

# 训练命令
echo -e "${GREEN}🚀 开始训练...${NC}\n"
python train.py \
    args.config="$CONFIG" \
    experiment.seed=42 \
    trainer.max_epochs=100 \
    trainer.precision="16-mixed" \
    data.batch_size=48 \
    2>&1 | tee "$LOG_DIR/training.log"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}✅ 训练完成！${NC}"
    echo -e "日志: $LOG_DIR/training.log"
else
    echo -e "\n${RED}❌ 训练失败 (exit code: $EXIT_CODE)${NC}"
    exit $EXIT_CODE
fi
