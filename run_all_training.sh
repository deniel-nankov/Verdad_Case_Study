#!/bin/bash
# RUN ALL DRL & HYBRID TRAINING
# This script runs both training scripts sequentially

echo "=========================================="
echo "🚀 COMPREHENSIVE DRL & HYBRID ML+DRL TRAINING"
echo "=========================================="
echo ""
echo "This will train:"
echo "  1. Robust DRL (200 episodes) - ~10-15 minutes"
echo "  2. Hybrid ML+DRL (100 episodes) - ~15-20 minutes"
echo ""
echo "Total time: ~25-35 minutes"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to start..."
sleep 5

echo ""
echo "=========================================="
echo "STEP 1/2: Training Robust DRL..."
echo "=========================================="
.venv/bin/python3 scripts/training/train_drl_robust.py

if [ $? -eq 0 ]; then
    echo "✅ Robust DRL training complete!"
else
    echo "❌ Robust DRL training failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "STEP 2/2: Training Hybrid ML+DRL..."
echo "=========================================="
.venv/bin/python3 scripts/training/train_hybrid_ml_drl.py

if [ $? -eq 0 ]; then
    echo "✅ Hybrid ML+DRL training complete!"
else
    echo "❌ Hybrid ML+DRL training failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ ALL TRAINING COMPLETE!"
echo "=========================================="
echo ""
echo "Models saved:"
echo "  - drl_best_model.pth"
echo "  - hybrid_ml_drl_model.pth"
echo "  - ml_prediction_engine.pkl"
echo ""
echo "Charts saved:"
echo "  - drl_training_results.png"
echo "  - hybrid_ml_drl_results.png"
echo ""
echo "🎉 Ready for deployment!"
