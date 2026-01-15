#!/bin/bash
# =============================================================================
# Transfer Learning Experiments for HGCP+FDA Model
# =============================================================================
# Three transfer scenarios:
# 1. Season Transfer: USA-Summer -> USA-Winter (same species)
# 2. Geographic Transfer: USA -> Kenya (different species/region)
# 3. Species Transfer: Bird -> Butterfly (different taxa)
# =============================================================================

set -e

# Configuration
PROJECT_ROOT="/data1/sunyuxuan/SatBird"
SOURCE_CHECKPOINT="runs/hgcp_fda_summer_seed42/checkpoints/last.ckpt"
SEED=42
GPU_ID=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored headers
print_header() {
    echo ""
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
    echo ""
}

# Function to run experiment
run_experiment() {
    local config=$1
    local exp_name=$2
    local description=$3
    
    echo -e "${YELLOW}Starting: ${description}${NC}"
    echo "Config: ${config}"
    echo "Experiment: ${exp_name}"
    
    if [ -f "${PROJECT_ROOT}/${config}" ]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
            --config ${config} \
            experiment.seed=${SEED} \
            experiment.exp_name=${exp_name}_seed${SEED}
        
        echo -e "${GREEN}✓ Completed: ${exp_name}${NC}"
    else
        echo -e "${RED}✗ Config not found: ${config}${NC}"
        return 1
    fi
}

# Function to evaluate model
evaluate_model() {
    local config=$1
    local checkpoint=$2
    local exp_name=$3
    
    echo -e "${YELLOW}Evaluating: ${exp_name}${NC}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python test.py \
        --config ${config} \
        load_ckpt=${checkpoint}
    
    echo -e "${GREEN}✓ Evaluation completed: ${exp_name}${NC}"
}

# Check for source checkpoint
check_source_checkpoint() {
    if [ ! -f "${PROJECT_ROOT}/${SOURCE_CHECKPOINT}" ]; then
        echo -e "${RED}Error: Source checkpoint not found at ${SOURCE_CHECKPOINT}${NC}"
        echo "Please train the HGCP+FDA model on USA-Summer first."
        echo "Run: python train.py --config configs/SatBird-USA-summer/hgcp_fda_summer.yaml"
        exit 1
    fi
    echo -e "${GREEN}✓ Source checkpoint found: ${SOURCE_CHECKPOINT}${NC}"
}

# Parse command line arguments
EXPERIMENT_TYPE="all"
MODE="train"

while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment|-e)
            EXPERIMENT_TYPE="$2"
            shift 2
            ;;
        --mode|-m)
            MODE="$2"
            shift 2
            ;;
        --gpu|-g)
            GPU_ID="$2"
            shift 2
            ;;
        --seed|-s)
            SEED="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -e, --experiment TYPE   Experiment type: season, geo, species, all (default: all)"
            echo "  -m, --mode MODE         Mode: train, eval, both (default: train)"
            echo "  -g, --gpu GPU_ID        GPU ID (default: 0)"
            echo "  -s, --seed SEED         Random seed (default: 42)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd ${PROJECT_ROOT}

# =============================================================================
# Main Execution
# =============================================================================

print_header "Transfer Learning Experiments for HGCP+FDA Model"

echo "Configuration:"
echo "  - Experiment Type: ${EXPERIMENT_TYPE}"
echo "  - Mode: ${MODE}"
echo "  - GPU: ${GPU_ID}"
echo "  - Seed: ${SEED}"
echo "  - Source Checkpoint: ${SOURCE_CHECKPOINT}"
echo ""

# Check source checkpoint exists
check_source_checkpoint

# =============================================================================
# Season Transfer: USA-Summer -> USA-Winter
# =============================================================================
if [ "$EXPERIMENT_TYPE" = "season" ] || [ "$EXPERIMENT_TYPE" = "all" ]; then
    print_header "Season Transfer: USA-Summer -> USA-Winter"
    
    if [ "$MODE" = "train" ] || [ "$MODE" = "both" ]; then
        # Linear Probe
        run_experiment \
            "configs/transfer_learning/season_transfer_linear.yaml" \
            "season_transfer_linear" \
            "Season Transfer - Linear Probe"
        
        # Adapter Tune
        run_experiment \
            "configs/transfer_learning/season_transfer_adapter.yaml" \
            "season_transfer_adapter" \
            "Season Transfer - Adapter Tune"
        
        # Full Fine-tune
        run_experiment \
            "configs/transfer_learning/season_transfer_finetune.yaml" \
            "season_transfer_finetune" \
            "Season Transfer - Full Fine-tune"
    fi
    
    if [ "$MODE" = "eval" ] || [ "$MODE" = "both" ]; then
        echo "Evaluating season transfer models..."
        for strategy in linear adapter finetune; do
            checkpoint="runs/season_transfer_${strategy}_seed${SEED}/checkpoints/last.ckpt"
            if [ -f "$checkpoint" ]; then
                evaluate_model \
                    "configs/transfer_learning/season_transfer_${strategy}.yaml" \
                    "$checkpoint" \
                    "season_transfer_${strategy}"
            fi
        done
    fi
fi

# =============================================================================
# Geographic Transfer: USA -> Kenya
# =============================================================================
if [ "$EXPERIMENT_TYPE" = "geo" ] || [ "$EXPERIMENT_TYPE" = "all" ]; then
    print_header "Geographic Transfer: USA -> Kenya"
    
    if [ "$MODE" = "train" ] || [ "$MODE" = "both" ]; then
        # Linear Probe
        run_experiment \
            "configs/transfer_learning/geo_transfer_linear.yaml" \
            "geo_transfer_linear" \
            "Geographic Transfer - Linear Probe"
        
        # Adapter Tune
        run_experiment \
            "configs/transfer_learning/geo_transfer_adapter.yaml" \
            "geo_transfer_adapter" \
            "Geographic Transfer - Adapter Tune"
        
        # Full Fine-tune
        run_experiment \
            "configs/transfer_learning/geo_transfer_finetune.yaml" \
            "geo_transfer_finetune" \
            "Geographic Transfer - Full Fine-tune"
    fi
    
    if [ "$MODE" = "eval" ] || [ "$MODE" = "both" ]; then
        echo "Evaluating geographic transfer models..."
        for strategy in linear adapter finetune; do
            checkpoint="runs/geo_transfer_${strategy}_seed${SEED}/checkpoints/last.ckpt"
            if [ -f "$checkpoint" ]; then
                evaluate_model \
                    "configs/transfer_learning/geo_transfer_${strategy}.yaml" \
                    "$checkpoint" \
                    "geo_transfer_${strategy}"
            fi
        done
    fi
fi

# =============================================================================
# Species Transfer: Bird -> Butterfly
# =============================================================================
if [ "$EXPERIMENT_TYPE" = "species" ] || [ "$EXPERIMENT_TYPE" = "all" ]; then
    print_header "Species Transfer: Bird -> Butterfly"
    
    if [ "$MODE" = "train" ] || [ "$MODE" = "both" ]; then
        # Linear Probe
        run_experiment \
            "configs/transfer_learning/species_transfer_linear.yaml" \
            "species_transfer_linear" \
            "Species Transfer - Linear Probe"
        
        # Adapter Tune
        run_experiment \
            "configs/transfer_learning/species_transfer_adapter.yaml" \
            "species_transfer_adapter" \
            "Species Transfer - Adapter Tune"
        
        # Full Fine-tune
        run_experiment \
            "configs/transfer_learning/species_transfer_finetune.yaml" \
            "species_transfer_finetune" \
            "Species Transfer - Full Fine-tune"
    fi
    
    if [ "$MODE" = "eval" ] || [ "$MODE" = "both" ]; then
        echo "Evaluating species transfer models..."
        for strategy in linear adapter finetune; do
            checkpoint="runs/species_transfer_${strategy}_seed${SEED}/checkpoints/last.ckpt"
            if [ -f "$checkpoint" ]; then
                evaluate_model \
                    "configs/transfer_learning/species_transfer_${strategy}.yaml" \
                    "$checkpoint" \
                    "species_transfer_${strategy}"
            fi
        done
    fi
fi

# =============================================================================
# Generate Comparison Report
# =============================================================================
print_header "Generating Comparison Report"

python scripts/transfer_learning_eval.py \
    --experiment_type ${EXPERIMENT_TYPE} \
    --seed ${SEED} \
    --output_dir results/transfer_learning

print_header "All experiments completed!"

echo "Results saved to: results/transfer_learning/"
echo ""
echo "Summary of experiments:"
echo "  1. Season Transfer (USA-Summer -> USA-Winter): 3 strategies"
echo "  2. Geographic Transfer (USA -> Kenya): 3 strategies"  
echo "  3. Species Transfer (Bird -> Butterfly): 3 strategies"
echo ""
echo "To view detailed results, run:"
echo "  python scripts/transfer_learning_eval.py --visualize"
