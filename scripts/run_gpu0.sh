#!/bin/bash
# ============================================================
# GPU 0 (96GB Blackwell) — Dir1 LoRA
# 執行: CUDA_VISIBLE_DEVICES=0 bash scripts/run_gpu0.sh
# 或直接: bash scripts/run_gpu0.sh  (預設使用 GPU 0)
# ============================================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
set -e
cd "$(dirname "$0")/.."

VENV_PY=".venv/bin/python"
LOG_DIR="benchmark_results"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "=============================="
echo " GPU 0 Benchmark Start: $TIMESTAMP"
echo " CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo " Directions: dir1_lora"
echo " Max Steps: 300"
echo "=============================="

# --- Dir1: Standard LoRA ---
echo ""
echo "[GPU 0] Running Dir1: Standard LoRA (q+v, r=4, 669696 params)"
$VENV_PY scripts/run_benchmark.py \
    --direction dir1_lora \
    --games ls20,vc33,s5i5,dc22,wa30 \
    --max-steps 100 \
    --cuda cuda:0 \
    --output-dir "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/dir1_lora_${TIMESTAMP}.log"

echo ""
echo "[GPU 0] ✅ Dir1 complete"
echo ""
echo "=============================="
echo " GPU 0 ALL DONE: $(date +%Y%m%d_%H%M%S)"
echo "=============================="
