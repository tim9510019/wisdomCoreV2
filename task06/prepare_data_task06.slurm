#!/bin/bash
# prepare_data_task06.slurm — Task06 蒸餾訓練資料完整準備腳本（修正版）
# ================================================================
# 完成後產出物：
#   ~/task06_data/type_A/                    ── Type A Parquet (≤1024 tok)
#   ~/task06_data/type_B/                    ── Type B Parquet (32K~128K tok)
#   ~/task06_data/logit_shards_qwen3_coder_30b/ ── 100份 Top-128 Logit Shards
#   ~/task06_data/train_type_A_combined.parquet  ── 永久保留，供換 Teacher 重用
#   ~/task06_data/dataset_stats.json         ── Token 統計報告
#
# 使用方式：
#   sbatch prepare_data_task06.slurm          # 全部準備
#   bash   prepare_data_task06.slurm stage1   # 只做 HF 下載 + Parquet 打包
#   bash   prepare_data_task06.slurm stage2   # 只做 CoT 自生成
#   bash   prepare_data_task06.slurm stage3   # 只做 Teacher Logit 推論
#   bash   prepare_data_task06.slurm verify   # 只做最終驗證

#SBATCH --job-name=task06_data_prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --output=task06_data_prep_%j.log
#SBATCH --error=task06_data_prep_%j.err

set -euo pipefail

# ── 目錄設定 ──
TASK_DIR="/home/tim9510019/wisdomCoreV2/task06"
DATA_ROOT="${HOME}/task06_data"
TYPE_A_DIR="${DATA_ROOT}/type_A"
TYPE_B_DIR="${DATA_ROOT}/type_B"
COT_DIR="${DATA_ROOT}/cot_generated"
LOGIT_DIR="${DATA_ROOT}/logit_shards_qwen3_coder_30b"
LOG_DIR="${DATA_ROOT}/logs"
COMBINED_A="${DATA_ROOT}/train_type_A_combined.parquet"
STATS_FILE="${DATA_ROOT}/dataset_stats.json"
TEACHER_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"

mkdir -p "${TYPE_A_DIR}" "${TYPE_B_DIR}" "${COT_DIR}" "${LOGIT_DIR}" "${LOG_DIR}"

STAGE="${1:-all}"

# ── 顏色輸出 ──
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
section() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    printf  "║  %-56s  ║\n" "$*"
    echo "╚══════════════════════════════════════════════════════════╝"
}

# ================================================================
# STAGE 1: HuggingFace 數據下載 + 打包 Parquet
# ================================================================
if [ "${STAGE}" = "stage1" ] || [ "${STAGE}" = "all" ]; then
    section "Stage 1: HuggingFace 數據下載 + 打包 Parquet"
    info "開始時間：$(date)"

    python3 "${TASK_DIR}/stage1_download.py" \
        --type_a_dir "${TYPE_A_DIR}" \
        --type_b_dir "${TYPE_B_DIR}" \
        --tokenizer  "${TEACHER_MODEL}" \
        --steps      "1a,1b,1c,1d,1e" \
        2>&1 | tee "${LOG_DIR}/stage1.log"

    info "Stage 1 完成：$(date)"
fi

# ================================================================
# STAGE 2: CoT 思考鏈自生成（Qwen3-Coder-30B, enable_thinking）
# ================================================================
if [ "${STAGE}" = "stage2" ] || [ "${STAGE}" = "all" ]; then
    section "Stage 2: Qwen3-Coder-30B CoT 思考鏈生成（目標 6B tokens）"
    info "開始時間：$(date)"

    python3 "${TASK_DIR}/prepare_cot_distill.py" \
        --teacher_model  "${TEACHER_MODEL}" \
        --output_dir     "${COT_DIR}"       \
        --target_tokens  60000000           \
        --max_new_tokens 2048               \
        --temperature    0.7                \
        --seed           2026               \
        2>&1 | tee "${LOG_DIR}/stage2_cot.log"

    info "複製 CoT Parquet 到 Type A 目錄..."
    find "${COT_DIR}" -name "*.parquet" -exec cp {} "${TYPE_A_DIR}/" \;

    info "Stage 2 完成：$(date)"
fi

# ================================================================
# STAGE 3: 合併 Type A + Teacher 推論 → Top-128 Logit Shards
# ================================================================
if [ "${STAGE}" = "stage3" ] || [ "${STAGE}" = "all" ]; then
    section "Stage 3: 合併 Type A + Teacher 推論 → 100份 Logit Shards"
    info "開始時間：$(date)"

    # ── 合併所有 Type A Parquet ──
    info "合併所有 Type A Parquet 成單一文件（永久保留供換 Teacher 重用）..."
    python3 "${TASK_DIR}/stage3_merge.py" \
        --type_a_dir "${TYPE_A_DIR}" \
        --output     "${COMBINED_A}" \
        2>&1 | tee "${LOG_DIR}/stage3_merge.log"

    TOTAL_ROWS=$(python3 -c "
import pyarrow.parquet as pq
m = pq.read_metadata('${COMBINED_A}')
print(m.num_rows)
" 2>/dev/null || echo "unknown")
    info "Type A 合併完成：${TOTAL_ROWS} 樣本 → ${COMBINED_A}"

    # ── Teacher 推論 ──
    info "啟動雙卡平行 Qwen3-Coder-30B Teacher 推論（Top-128 Logits × 100 Shards）..."

    RESUME_SHARD=0
    EXISTING=$(find "${LOGIT_DIR}" -maxdepth 1 -name "*.safetensors" | wc -l)
    if [ "${EXISTING}" -gt 0 ]; then
        RESUME_SHARD=${EXISTING}
        warn "發現 ${EXISTING} 份已存在的 Shard，將從 Shard ${RESUME_SHARD} 繼續..."
    fi

    SHARD_MID=$(( (100 + RESUME_SHARD) / 2 ))

    if [ "${RESUME_SHARD}" -lt 100 ]; then
        # GPU 0
        CUDA_VISIBLE_DEVICES=0 python3 -u "${TASK_DIR}/run_teacher_inference.py" \
            --input_parquet  "${COMBINED_A}"      \
            --output_dir     "${LOGIT_DIR}"       \
            --teacher_model  "${TEACHER_MODEL}"   \
            --shard_count    100                  \
            --top_k          128                  \
            --batch_size     32                   \
            --max_length     1024                 \
            --shard_start    "${RESUME_SHARD}"    \
            --shard_end      "${SHARD_MID}"       \
            --skip_update_parquet \
            > "${LOG_DIR}/stage3_teacher_gpu0.log" 2>&1 &
        PID0=$!
        
        # GPU 1
        CUDA_VISIBLE_DEVICES=1 python3 -u "${TASK_DIR}/run_teacher_inference.py" \
            --input_parquet  "${COMBINED_A}"      \
            --output_dir     "${LOGIT_DIR}"       \
            --teacher_model  "${TEACHER_MODEL}"   \
            --shard_count    100                  \
            --top_k          128                  \
            --batch_size     32                   \
            --max_length     1024                 \
            --shard_start    "${SHARD_MID}"       \
            --shard_end      100                  \
            --skip_update_parquet \
            > "${LOG_DIR}/stage3_teacher_gpu1.log" 2>&1 &
        PID1=$!
        
        info "已背景啟動雙卡推論任務。GPU 0 PID: ${PID0}, GPU 1 PID: ${PID1}"
        info "正在等待推論任務完成 (日誌輸出至 ${LOG_DIR}/stage3_teacher_gpu*.log)..."
        
        wait "${PID0}"
        wait "${PID1}"
        
        info "雙卡推論全部完成！正在更新 combined parquet 索引欄位..."
        python3 -c "
import pyarrow.parquet as pq
import pyarrow as pa
import math
table = pq.read_table('${COMBINED_A}')
total_samples = len(table)
samples_per_shard = math.ceil(total_samples / 100)
logit_shard_ids = [i // samples_per_shard for i in range(total_samples)]
logit_offsets = [i % samples_per_shard for i in range(total_samples)]
shard_id_idx = table.schema.get_field_index('logit_shard_id')
offset_idx = table.schema.get_field_index('logit_offset')
if shard_id_idx != -1:
    table = table.set_column(shard_id_idx, 'logit_shard_id', pa.array(logit_shard_ids, type=pa.int32()))
else:
    table = table.append_column('logit_shard_id', pa.array(logit_shard_ids, type=pa.int32()))
if offset_idx != -1:
    table = table.set_column(offset_idx, 'logit_offset', pa.array(logit_offsets, type=pa.int64()))
else:
    table = table.append_column('logit_offset', pa.array(logit_offsets, type=pa.int64()))
pq.write_table(table, '${COMBINED_A}', compression='snappy')
"
        info "Parquet 索引欄位更新成功！"
    fi

    SHARD_COUNT=$(find "${LOGIT_DIR}" -maxdepth 1 -name "*.safetensors" | wc -l)
    info "完成 Logit Shard：${SHARD_COUNT} / 100"
    info "Stage 3 完成：$(date)"
fi

# ================================================================
# VERIFY: 最終統計驗證
# ================================================================
if [ "${STAGE}" = "verify" ] || [ "${STAGE}" = "all" ]; then
    section "最終驗證：Token 統計 + 文件完整性"

    python3 "${TASK_DIR}/stage_verify.py" \
        --type_a_dir "${TYPE_A_DIR}" \
        --type_b_dir "${TYPE_B_DIR}" \
        --cot_dir    "${COT_DIR}"    \
        --logit_dir  "${LOGIT_DIR}"  \
        --stats_out  "${STATS_FILE}" \
        2>&1 | tee "${LOG_DIR}/verify.log"
fi

# ================================================================
# 完成
# ================================================================
section "Task06 資料準備完成"
echo ""
echo "  ├── ${TYPE_A_DIR}/"
echo "  │       Type A Parquet（蒸餾型，≤1024 tok）"
echo "  ├── ${TYPE_B_DIR}/"
echo "  │       Type B Parquet（長文型，32K~128K tok）"
echo "  ├── ${LOGIT_DIR}/"
echo "  │       logit_shard_000~099.safetensors（Top-128 Logits）"
echo "  ├── ${COMBINED_A}"
echo "  │       永久保留 → 換 Teacher 只需重跑 Stage 3"
echo "  └── ${STATS_FILE}"
echo ""
echo "  下一步訓練："
echo "    python trainDNA16B_distill.py --phase 1"
echo "    python trainDNA16B_distill.py --phase 2"
echo "    python trainDNA16B_distill.py --phase 3"
echo ""
