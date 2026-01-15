#!/bin/bash
# Training script for DINOv2 with Adapter and Prompt Tuning
# Parameter-Efficient Fine-Tuning (PEFT) approach

echo "=================================================="
echo "DINOv2 Adapter+Prompt Training Script"
echo "=================================================="

# Set working directory
cd /data1/sunyuxuan/SatBird

# Configuration file
CONFIG="configs/SatBird-USA-winter/dinov2_adapter_prompt_v3.yaml"

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Configuration file not found: $CONFIG"
    exit 1
fi

echo "Configuration: $CONFIG"
echo "Dataset: SatBird-USA-winter (624 species)"
echo "Model: DINOv2 with Adapter (bottleneck_dim=64) + Prompt Tuning (length=10)"
echo ""

# Run training with different seeds for robustness
for SEED in 42 123 456; do
    echo "=================================================="
    echo "Training with seed: $SEED"
    echo "=================================================="
    
    python train.py \
        --config-name dinov2_adapter_prompt_v3 \
        experiment.seed=$SEED \
        experiment.exp_name="dinov2_adapter_prompt_v3_seed${SEED}" \
        trainer.devices=1 \
        experiment.training.batch_size=32 \
        experiment.module.learning_rate=0.0001 \
        experiment.module.bottleneck_dim=64 \
        experiment.module.prompt_len=10 \
        experiment.module.freeze_backbone=true \
        experiment.module.unfreeze_last_n_blocks=0 \
        experiment.module.fusion_type="cross_attention"
    
    if [ $? -eq 0 ]; then
        echo "✅ Training completed successfully for seed $SEED"
    else
        echo "❌ Training failed for seed $SEED"
        exit 1
    fi
    
    echo ""
done

echo "=================================================="
echo "All training runs completed!"
echo "=================================================="

# Optionally run testing on best checkpoint
echo "To test the model, run:"
echo "python test.py --config-name dinov2_adapter_prompt_v3 --ckpt_path <path_to_best_checkpoint>"
