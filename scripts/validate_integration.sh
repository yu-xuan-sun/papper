#!/bin/bash
# Validation script for DINOv2 Adapter+Prompt integration

echo "=================================================="
echo "DINOv2 Adapter+Prompt Integration Validation"
echo "=================================================="
echo ""

SUCCESS_COUNT=0
TOTAL_CHECKS=5

# Check 1: Model file exists
echo "[1/5] Checking model file..."
if [ -f "src/models/dinov2_adapter_prompt.py" ]; then
    SIZE=$(wc -l < src/models/dinov2_adapter_prompt.py)
    echo "✅ Model file exists ($SIZE lines)"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ Model file NOT found"
fi
echo ""

# Check 2: Config file exists
echo "[2/5] Checking configuration file..."
if [ -f "configs/SatBird-USA-winter/dinov2_adapter_prompt_v3.yaml" ]; then
    echo "✅ Configuration file exists"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ Configuration file NOT found"
fi
echo ""

# Check 3: Trainer integration
echo "[3/5] Checking trainer.py integration..."
if grep -q "dinov2_adapter_prompt" src/trainer/trainer.py; then
    echo "✅ Model registered in trainer.py"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ Model NOT registered in trainer.py"
fi
echo ""

# Check 4: drop_to_rgb modification
echo "[4/5] Checking drop_to_rgb setting..."
if grep -q 'drop_to_rgb.*dinov2_adapter_prompt' src/trainer/trainer.py; then
    echo "✅ drop_to_rgb includes dinov2_adapter_prompt"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ drop_to_rgb NOT properly configured"
fi
echo ""

# Check 5: Python import test
echo "[5/5] Testing Python import..."
cd /data1/sunyuxuan/SatBird
python -c "
import sys
sys.path.insert(0, '.')
try:
    from src.models.dinov2_adapter_prompt import Dinov2AdapterPrompt, create_dinov2_adapter_prompt_model
    print('✅ Python import successful')
    exit(0)
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
" 2>&1
if [ $? -eq 0 ]; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
fi
echo ""

# Summary
echo "=================================================="
echo "Validation Summary: $SUCCESS_COUNT/$TOTAL_CHECKS checks passed"
echo "=================================================="

if [ $SUCCESS_COUNT -eq $TOTAL_CHECKS ]; then
    echo "✅ All checks passed! Ready to train."
    echo ""
    echo "To start training, run:"
    echo "  bash scripts/train_dinov2_adapter_prompt.sh"
    exit 0
else
    echo "❌ Some checks failed. Please review the errors above."
    exit 1
fi
