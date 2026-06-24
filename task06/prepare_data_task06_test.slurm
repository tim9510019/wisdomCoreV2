#!/bin/bash
# prepare_data_task06_test.slurm — Task06 蒸餾訓練資料測試準備腳本（支援任意 GB 指定）
# ================================================================
# 完成後產出物：
#   ~/task06_data/type_A/                         ── Type A Parquet (≤2048 tok)
#   ~/task06_data/type_B/                         ── Type B Parquet (長文)
#   ~/task06_data/type_B_cot/                     ── Type B-CoT Parquet
#   ~/task06_data/logit_shards_qwen3_coder_30b/   ── Top-128 Logit Shards
#   ~/task06_data/train_type_A_combined.parquet   ── 永久保留，供換 Teacher 重用
#   ~/task06_data/dataset_stats.json              ── Token 統計報告
#
# 使用方式（第 1 參數：stage，第 2 參數：目標 GB 數）：
#
#   sbatch prepare_data_task06_test.slurm                    # 全部準備，預設 100 GB
#   sbatch prepare_data_task06_test.slurm all 50             # 全部準備，限制 50 GB
#   sbatch prepare_data_task06_test.slurm all 200            # 全部準備，限制 200 GB
#
#   bash prepare_data_task06_test.slurm stage1               # 只做 Stage1，預設 100 GB
#   bash prepare_data_task06_test.slurm stage1 30            # 只做 Stage1，限制 30 GB
#   bash prepare_data_task06_test.slurm stage2 50            # 只做 Stage2，限制 50 GB
#   bash prepare_data_task06_test.slurm stage3               # 只做 Stage3（Teacher 推論）
#   bash prepare_data_task06_test.slurm verify               # 只做最終驗證
#
# GB 預算說明：
#   - 控制 pack_distill_dataset.py 的 Parquet 總量（--max_gb）
#   - 同時按比例換算 CoT tokens 上限（stage2）
#   - Logit Shard 容量另計（約等於 Type-A tokens × 128 × 2 bytes）
#   - 範例：100 GB 預算 → ~24B tokens Parquet + ~100GB Logit Shards

#SBATCH --job-name=task06_data_prep_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=task06_data_prep_test_%j.log
#SBATCH --error=task06_data_prep_test_%j.err

set -euo pipefail

# ================================================================
# 參數解析
#   $1 = stage (all / stage1 / stage2 / stage3 / verify)，預設 all
#   $2 = 目標 GB 數（整數或小數），預設 100
# ================================================================
STAGE="${1:-all}"
MAX_GB="${2:-100}"

# 驗證 MAX_GB 是合法正數
if ! echo "${MAX_GB}" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || \
   [ "$(echo "${MAX_GB} <= 0" | bc -l)" = "1" ]; then
    echo "[ERROR] 第 2 參數 MAX_GB 必須是正數（例如 50、100、200.5）"
    echo "        用法：bash $0 [stage] [GB數]"
    exit 1
fi

# ── 目錄設定 ──
TASK_DIR="/home/tim9510019/wisdomCoreV2/task06"
DATA_ROOT="${HOME}/task06_data"
TYPE_A_DIR="${DATA_ROOT}/type_A"
TYPE_B_DIR="${DATA_ROOT}/type_B"
TYPE_B_COT_DIR="${DATA_ROOT}/type_B_cot"
COT_DIR="${DATA_ROOT}/cot_generated"
LOGIT_DIR="${DATA_ROOT}/logit_shards_qwen3_coder_30b"
LOG_DIR="${DATA_ROOT}/logs"
COMBINED_A="${DATA_ROOT}/train_type_A_combined.parquet"
STATS_FILE="${DATA_ROOT}/dataset_stats.json"
TEACHER_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"

mkdir -p "${TYPE_A_DIR}" "${TYPE_B_DIR}" "${TYPE_B_COT_DIR}" \
         "${COT_DIR}" "${LOGIT_DIR}" "${LOG_DIR}"

# ── 顏色輸出 ──
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
detail()  { echo -e "${CYAN}[CONF]${NC}  $*"; }
section() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    printf  "║  %-56s  ║\n" "$*"
    echo "╚══════════════════════════════════════════════════════════╝"
}

# ================================================================
# 從 MAX_GB 換算各 Stage 的 token/token 上限
#
# 比例關係（基於 100 GB 基準）：
#   整體 Parquet        → MAX_GB GB（直接傳給 --max_gb）
#   CoT Stage2 tokens   → 按比例：100GB基準=30M tokens → N GB = N/100×30M
# ================================================================

# CoT tokens 上限：100 GB → 30,000,000 tokens（測試版比例）
COT_TARGET_TOKENS=$(echo "${MAX_GB} * 300000" | bc | awk '{printf "%d", $1}')
# 最小值保護：至少 1,000,000 tokens
if [ "${COT_TARGET_TOKENS}" -lt 1000000 ]; then
    COT_TARGET_TOKENS=1000000
fi

# 打印配置摘要
section "Task06 測試資料準備（GB 預算：${MAX_GB} GB）"
detail "STAGE        = ${STAGE}"
detail "MAX_GB       = ${MAX_GB} GB"
detail "COT tokens   = $(echo "scale=1; ${COT_TARGET_TOKENS}/1000000" | bc)M tokens"
detail "資料根目錄   = ${DATA_ROOT}"
detail "Teacher 模型 = ${TEACHER_MODEL}"
echo ""

# ================================================================
# STAGE 1: HuggingFace 數據下載 + 打包 Parquet
#   --max_gb 傳入 pack_distill_dataset.py，自動等比縮放所有 target_tokens
# ================================================================
if [ "${STAGE}" = "stage1" ] || [ "${STAGE}" = "all" ]; then
    section "Stage 1: HF 數據下載 + 打包 Parquet（預算 ${MAX_GB} GB）"
    info "開始時間：$(date)"

    python3 "${TASK_DIR}/stage1_download.py" \
        --type_a_dir "${TYPE_A_DIR}"     \
        --type_b_dir "${TYPE_B_DIR}"     \
        --tokenizer  "${TEACHER_MODEL}"  \
        --steps      "1a,1b,1c,1d,1e"   \
        --max_gb     "${MAX_GB}"         \
        2>&1 | tee "${LOG_DIR}/stage1_test.log"

    info "Stage 1 完成：$(date)"
fi

# ================================================================
# STAGE 2: CoT 思考鏈自生成（tokens 上限按 GB 比例換算）
# ================================================================
if [ "${STAGE}" = "stage2" ] || [ "${STAGE}" = "all" ]; then
    section "Stage 2: Qwen3 CoT 生成（目標 ${COT_TARGET_TOKENS} tokens）"
    info "開始時間：$(date)"

    python3 "${TASK_DIR}/prepare_cot_distill.py" \
        --teacher_model  "${TEACHER_MODEL}"        \
        --output_dir     "${COT_DIR}"              \
        --target_tokens  "${COT_TARGET_TOKENS}"    \
        --max_new_tokens 2048                      \
        --temperature    0.7                       \
        --seed           2026                      \
        2>&1 | tee "${LOG_DIR}/stage2_cot_test.log"

    info "複製 CoT Parquet 到 Type A 目錄..."
    find "${COT_DIR}" -name "*.parquet" -exec cp {} "${TYPE_A_DIR}/" \;

    info "Stage 2 完成：$(date)"
fi

# ================================================================
# STAGE 3: 合併 Type A + Teacher 推論 → Top-128 Logit Shards
#   Shard 數量根據 MAX_GB 動態計算（每個 Shard 約 1 GB）
#   公式：SHARD_COUNT = ceil(MAX_GB / 1)，最少 10，最多 500
# ================================================================
if [ "${STAGE}" = "stage3" ] || [ "${STAGE}" = "all" ]; then

    # 動態計算 Shard 數（取整，確保每個 Shard ≈ 1 GB）
    SHARD_COUNT=$(echo "${MAX_GB}" | awk '{n=int($1+0.5); if(n<10) n=10; if(n>500) n=500; print n}')

    section "Stage 3: 合併 Type A + Teacher 推論（${SHARD_COUNT} 份 Shards）"
    info "開始時間：$(date)"

    # ── 合併所有 Type A Parquet ──
    info "合併所有 Type A Parquet 成單一文件..."
    python3 "${TASK_DIR}/stage3_merge.py" \
        --type_a_dir "${TYPE_A_DIR}" \
        --output     "${COMBINED_A}" \
        2>&1 | tee "${LOG_DIR}/stage3_merge_test.log"

    TOTAL_ROWS=$(python3 -c "
import pyarrow.parquet as pq
m = pq.read_metadata('${COMBINED_A}')
print(m.num_rows)
" 2>/dev/null || echo "unknown")
    info "Type A 合併完成：${TOTAL_ROWS} 樣本 → ${COMBINED_A}"

    # ── Teacher 推論（雙卡平行）──
    info "啟動雙卡平行 Teacher 推論（Top-128 Logits × ${SHARD_COUNT} Shards）..."

    RESUME_SHARD=0
    EXISTING=$(find "${LOGIT_DIR}" -maxdepth 1 -name "*.safetensors" | wc -l)
    if [ "${EXISTING}" -gt 0 ]; then
        RESUME_SHARD=${EXISTING}
        warn "發現 ${EXISTING} 份已存在的 Shard，將從 Shard ${RESUME_SHARD} 繼續..."
    fi

    SHARD_MID=$(( (SHARD_COUNT + RESUME_SHARD) / 2 ))

    if [ "${RESUME_SHARD}" -lt "${SHARD_COUNT}" ]; then
        # GPU 0：負責前半段 Shard
        CUDA_VISIBLE_DEVICES=0 python3 -u "${TASK_DIR}/run_teacher_inference.py" \
            --input_parquet  "${COMBINED_A}"       \
            --output_dir     "${LOGIT_DIR}"        \
            --teacher_model  "${TEACHER_MODEL}"    \
            --shard_count    "${SHARD_COUNT}"      \
            --top_k          128                   \
            --batch_size     32                    \
            --max_length     2048                  \
            --shard_start    "${RESUME_SHARD}"     \
            --shard_end      "${SHARD_MID}"        \
            --skip_update_parquet \
            > "${LOG_DIR}/stage3_teacher_gpu0_test.log" 2>&1 &
        PID0=$!

        # GPU 1：負責後半段 Shard
        CUDA_VISIBLE_DEVICES=1 python3 -u "${TASK_DIR}/run_teacher_inference.py" \
            --input_parquet  "${COMBINED_A}"       \
            --output_dir     "${LOGIT_DIR}"        \
            --teacher_model  "${TEACHER_MODEL}"    \
            --shard_count    "${SHARD_COUNT}"      \
            --top_k          128                   \
            --batch_size     32                    \
            --max_length     2048                  \
            --shard_start    "${SHARD_MID}"        \
            --shard_end      "${SHARD_COUNT}"      \
            --skip_update_parquet \
            > "${LOG_DIR}/stage3_teacher_gpu1_test.log" 2>&1 &
        PID1=$!

        info "已背景啟動雙卡推論任務。GPU 0 PID: ${PID0}, GPU 1 PID: ${PID1}"
        info "等待推論完成（日誌輸出至 ${LOG_DIR}/stage3_teacher_gpu*_test.log）..."

        wait "${PID0}"
        wait "${PID1}"

        info "雙卡推論全部完成！更新 combined parquet 索引欄位..."
        python3 -c "
import pyarrow.parquet as pq
import pyarrow as pa, math
table = pq.read_table('${COMBINED_A}')
total_samples = len(table)
shard_count   = ${SHARD_COUNT}
samples_per_shard = math.ceil(total_samples / shard_count)
logit_shard_ids = [i // samples_per_shard for i in range(total_samples)]
logit_offsets   = [i %  samples_per_shard for i in range(total_samples)]
for col, arr, typ in [
    ('logit_shard_id', logit_shard_ids, pa.int32()),
    ('logit_offset',   logit_offsets,   pa.int64()),
]:
    idx = table.schema.get_field_index(col)
    a   = pa.array(arr, type=typ)
    table = table.set_column(idx, col, a) if idx != -1 else table.append_column(col, a)
pq.write_table(table, '${COMBINED_A}', compression='snappy')
print('Parquet 索引欄位更新成功！')
"
    fi

    DONE_COUNT=$(find "${LOGIT_DIR}" -maxdepth 1 -name "*.safetensors" | wc -l)
    info "完成 Logit Shard：${DONE_COUNT} / ${SHARD_COUNT}"
    info "Stage 3 完成：$(date)"
fi

# ================================================================
# VERIFY: 最終統計驗證
# ================================================================
if [ "${STAGE}" = "verify" ] || [ "${STAGE}" = "all" ]; then
    section "最終驗證：Token 統計 + 文件完整性"

    python3 "${TASK_DIR}/stage_verify.py" \
        --type_a_dir "${TYPE_A_DIR}"  \
        --type_b_dir "${TYPE_B_DIR}"  \
        --cot_dir    "${COT_DIR}"     \
        --logit_dir  "${LOGIT_DIR}"   \
        --stats_out  "${STATS_FILE}"  \
        2>&1 | tee "${LOG_DIR}/verify_test.log"
fi

# ================================================================
# 完成摘要
# ================================================================
section "Task06 測試資料準備完成（預算 ${MAX_GB} GB）"
echo ""
echo "  ├── ${TYPE_A_DIR}/"
echo "  │       Type A Parquet（蒸餾型，≤2048 tok）"
echo "  ├── ${TYPE_B_DIR}/"
echo "  │       Type B Parquet（長文純自回歸）"
echo "  ├── ${TYPE_B_COT_DIR}/"
echo "  │       Type B-CoT Parquet（短文+THINK前綴）"
echo "  ├── ${LOGIT_DIR}/"
echo "  │       logit_shard_*.safetensors（共 ${SHARD_COUNT:-N/A} 份）"
echo "  ├── ${COMBINED_A}"
echo "  └── ${STATS_FILE}"
echo ""
echo "  下一步訓練："
echo "    python trainDNA16B_distill.py --phase 1"
echo ""
echo "  重跑（不同 GB 上限）範例："
echo "    bash $0 all 50     # 50 GB"
echo "    bash $0 all 200    # 200 GB"
echo "    bash $0 stage1 30  # 只做 Stage1，30 GB"
echo ""
