#!/bin/bash
# run_task06_pipeline.slurm — Task06 DNA Helix 16B 蒸餾流水線
# ================================================================
# 總計 50B Tokens，3 Phase 訓練：
#
#   Step 1: 數據打包 (Type A + Type B Parquet)
#   Step 2: CoT 思考鏈數據生成 (Qwen3-Coder-30B enable_thinking)
#   Step 3: Teacher 推論 → Top-128 Logit Shards (100份)
#   Step 4: Phase 1 —  1B 純蒸餾暖身    (只餵 Type A, α=1.0, β=0.5)
#   Step 5: Phase 2 — 49B 雙螺旋混合共生 (A:B=4:1, α=1.0, β動態)
#
# 使用方式：
#   sbatch run_task06_pipeline.slurm        # 全部跑
#   bash   run_task06_pipeline.slurm step4  # 只跑 Phase 1
#   bash   run_task06_pipeline.slurm step5  # 只跑 Phase 2

#SBATCH --job-name=dna16b_distill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=999:00:00
#SBATCH --output=task06_pipeline_%j.log

set -e  # 任何步驟失敗立即停止

# ── 目錄配置 ──
TASK_DIR="/home/tim9510019/wisdomCoreV2/task06"
DATA_DIR="/home/tim9510019/task06_data"
LOGIT_DIR="${DATA_DIR}/logit_shards"
TYPE_A_DIR="${DATA_DIR}/type_A"
TYPE_B_DIR="${DATA_DIR}/type_B"
COT_DIR="${DATA_DIR}/cot_generated"
CKPT_DIR="${HOME}/task06_checkpoints"

mkdir -p "${DATA_DIR}" "${LOGIT_DIR}" "${TYPE_A_DIR}" "${TYPE_B_DIR}" "${COT_DIR}" "${CKPT_DIR}"

STEP="${1:-all}"

# ================================================================
# Step 1: 數據集打包 (Data-Type A + B)
# ================================================================
if [ "${STEP}" = "step1" ] || [ "${STEP}" = "all" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Step 1: 打包 Data-Type A + B 數據集         ║"
    echo "╚══════════════════════════════════════════════╝"

    bash "${TASK_DIR}/prepare_data_task06.slurm" stage1

    echo "✅ Step 1 完成！"
fi

# ================================================================
# Step 2: CoT 思考鏈自生成（波動頭暖身前的知識播種）
# ================================================================
if [ "${STEP}" = "step2" ] || [ "${STEP}" = "all" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Step 2: Qwen3-Coder CoT 思考鏈生成 (6B tok) ║"
    echo "╚══════════════════════════════════════════════╝"

    bash "${TASK_DIR}/prepare_data_task06.slurm" stage2

    echo "✅ Step 2 完成！CoT 數據已合入 Type A。"
fi

# ================================================================
# Step 3: Teacher 推論 → Top-128 Logit Shards (100份)
# ================================================================
if [ "${STEP}" = "step3" ] || [ "${STEP}" = "all" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Step 3: Teacher 推論 → 100份 Logit Shards   ║"
    echo "╚══════════════════════════════════════════════╝"

    # 執行 Teacher 推論生成 shards
    bash "${TASK_DIR}/prepare_data_task06.slurm" stage3

    # 建立軟連結供 trainDNA16B_distill.py (硬編碼 ~/task06_data/logit_shards/) 讀取
    ln -sfn logit_shards_qwen3_coder_30b "${DATA_DIR}/logit_shards"
    echo "🔗 建立軟連結：${DATA_DIR}/logit_shards -> logit_shards_qwen3_coder_30b"

    echo "✅ Step 3 完成！100 份 Logit Shards 已就緒並連結。"
fi


# ================================================================
# Step 4: Phase 1 —— 1B 純蒸餾暖身
#   - 只餵 Data-Type A (≤2048 tok)
#   - 波動頭與粒子頭在糾纏架構下協同蒸餾，共同學習 Qwen3-Coder Soft Labels
#   - α=1.0，β=0.5
# ================================================================
if [ "${STEP}" = "step4" ] || [ "${STEP}" = "all" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Step 4: Phase 1 — 1B 純蒸餾暖身             ║"
    echo "║  只餵 Type A | α=1.0 | β=0.5 | A_only mode  ║"
    echo "╚══════════════════════════════════════════════╝"

    python "${TASK_DIR}/trainDNA16B_distill.py" --phase 1

    echo "✅ Step 4 (Phase 1 純蒸餾暖身) 完成！"
fi

# ================================================================
# Step 5: Phase 2 —— 49B 雙螺旋混合共生
#   - A:B = 4:1 混合 Batch（每批 80% Type A + 20% Type B）
#   - WPCE 因果糾纏正式啟動：粒子頭 AP_Base 引導 Δθ
#   - 波動頭 AW 作為引力透鏡反向修正粒子頭注意力
#   - A型 α=1.0 KL | B型 β=5.0 負熵 | 動態切換
# ================================================================
if [ "${STEP}" = "step5" ] || [ "${STEP}" = "all" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  Step 5: Phase 2 — 49B 雙螺旋混合共生                ║"
    echo "║  A:B=4:1 | A型 α=1.0 | B型 β=5.0 | mixed mode      ║"
    echo "╚══════════════════════════════════════════════════════╝"

    python "${TASK_DIR}/trainDNA16B_distill.py" --phase 2

    echo "✅ Step 5 (Phase 2 雙螺旋混合共生) 完成！"
fi

# ================================================================
# 完成總結
# ================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  🧬 Task06 DNA Helix 16B Coder 蒸餾流水線全部完成！        ║"
echo "║                                                          ║"
echo "║  Phase 1:  1B 純蒸餾暖身  ✅                              ║"
echo "║  Phase 2: 49B 雙螺旋共生  ✅                              ║"
echo "║  Total  : 50B Tokens      ✅                              ║"
echo "║                                                          ║"
echo "║  模型保存位置：${CKPT_DIR}                                  ║"
echo "║  HuggingFace：tim9510019/DNA-Helix-16B-Coder             ║"
echo "╚══════════════════════════════════════════════════════════╝"
