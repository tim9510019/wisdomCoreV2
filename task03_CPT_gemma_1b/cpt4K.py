"""
cpt4k.py — AGIV2 持續預訓練：第二階 (Intermediate) 4K 數據引擎 (絕對無重複極限版)
===========================================================
第一性原理實踐：
1. 視界擴展：維持 N+B 物理長度 4000，強迫啟動全局共振與潛在記憶。
2. 原生隔絕符徵用：使用 <unused0> 確保零拓撲碰撞與硬體對齊保護。
3. 極端語義淨化：徹底剔除 QA 與 NIAH，配置 80% Text + 20% Code。
4. 絕對無重複：完全捨棄 Epoch 輪迴，接入 100BT 文本與巨獸矩陣。
5. 異質流觀測與純淨過濾：共用 utils 模組打破同質性阻塞。
"""

import os
import sys
import random
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from utils import (
    AsyncS3Prefetcher,
    AsyncRedPajamaFilter,
    verify_existing_matrix,
    get_boundary_ids,
    find_topological_boundary,
)

# =====================================================================
# [ 全局配置區 ] 可調參數 (Tunable Parameters)
# =====================================================================
# 模型與環境設定
MODEL_ID = "google/gemma-3-1b-it"
RANDOM_SEED = 2026
DOC_SEP_TOKEN = "<unused0>"
OUTPUT_DIR = "./agiv2_stage2_4K"
OUTPUT_FILENAME = "agiv2_stage2_N_B_4K.parquet"

# 物理長度與序列總數
TOTAL_SEQ_LEN = 4000
TARGET_SEQUENCES = 2750000

# 資料來源與配比
RATIOS = {"long_text": 0.80, "code_smollm": 0.10, "code_redpajama": 0.10}
DATA_SOURCES = {
    "long_text": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-100BT",
        "split": "train",
    },
    "code_smollm": {
        "path": "HuggingFaceTB/smollm-corpus",
        "name": "python-edu",
        "split": "train",
    },
    "code_redpajama": {
        "path": "MBZUAI-LLM/SlimPajama-627B-DC",
        "name": "default",
        "split": "train",
    },
}

# 寫入、緩衝與異步效能參數
WRITE_BATCH_SIZE = 5000
BUFFER_WATERMARK = TOTAL_SEQ_LEN * 100
ASYNC_MAX_WORKERS = 32
ASYNC_QUEUE_SIZE = 500
SHUFFLE_BUFFER_SIZE = 10000

# 拓撲邊界與目標分割參數 (N)
TARGET_N_MIN = 1500
TARGET_N_MAX = 3200
BOUNDARY_OFFSET_LARGE = 800
BOUNDARY_MARGIN_LARGE = 100
BOUNDARY_OFFSET_SMALL = 400
BOUNDARY_MARGIN_SMALL = 100

# 實體檢驗設定
EXACT_MATCH_CHECK = False


# =====================================================================
# [ 執行邏輯區 ] PyArrow 超光速直寫引擎與動態合成器
# =====================================================================
class HyperDriveTopologicalBuilder4K:
    def __init__(self):
        print(
            "🚀 啟動量子疊加態資料引擎：AGIV2 (Stage 2 - 4K 視界擴展，純淨代碼過濾版)"
        )
        random.seed(RANDOM_SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        self.doc_sep_id = self.tokenizer.convert_tokens_to_ids(DOC_SEP_TOKEN)
        print(
            f"🔗 實體定錨：已將 {DOC_SEP_TOKEN} (ID: {self.doc_sep_id}) 徵用為跨文檔物理隔絕牆。"
        )

        self.total_seq_len = TOTAL_SEQ_LEN
        self.boundary_ids = get_boundary_ids(self.tokenizer)

        print("掛載開源巨獸資料流 (Streaming Parquet)...")
        self.streams = {}
        for key, conf in DATA_SOURCES.items():
            kwargs = {"path": conf["path"], "split": conf["split"], "streaming": True}
            if conf.get("name"):
                kwargs["name"] = conf["name"]

            raw_dataset = load_dataset(**kwargs)

            if key == "code_redpajama":
                raw_dataset = raw_dataset.shuffle(
                    seed=RANDOM_SEED, buffer_size=SHUFFLE_BUFFER_SIZE
                )

            raw_stream = iter(raw_dataset)

            if key == "code_smollm":
                self.streams[key] = AsyncS3Prefetcher(
                    raw_stream,
                    max_workers=ASYNC_MAX_WORKERS,
                    queue_size=ASYNC_QUEUE_SIZE,
                )
            elif key == "code_redpajama":
                self.streams[key] = AsyncRedPajamaFilter(
                    raw_stream, queue_size=ASYNC_QUEUE_SIZE
                )
            else:
                self.streams[key] = raw_stream

        self.schema = pa.schema(
            [
                ("input_ids", pa.list_(pa.int32())),
                ("n_split_index", pa.int32()),
                ("b_length", pa.int32()),
            ]
        )

    def _get_raw_tokens(self, source_type: str) -> List[int]:
        try:
            if source_type in ["code_smollm", "code_redpajama"]:
                text = self.streams[source_type].get_next_text()
            else:
                sample = next(self.streams[source_type])
                text = sample.get("text", sample.get("content", sample.get("code", "")))

            if not text or not text.strip():
                return []
            return self.tokenizer.encode(text, add_special_tokens=False)

        except StopIteration:
            raise RuntimeError(f"[物理質量枯竭] 資料流 '{source_type}' 已徹底耗盡！")
        except Exception:
            return []

    def build_and_save(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        print(f"準備將 {TARGET_SEQUENCES} 筆 4K 長脈絡教材寫入實體磁區: {output_file}")

        seq_count = 0
        batch_data = []
        global_token_buffer = []

        regular_sources = list(RATIOS.keys())
        regular_weights = list(RATIOS.values())

        with pq.ParquetWriter(output_file, self.schema) as writer:
            pbar = tqdm(
                total=TARGET_SEQUENCES, desc="HyperDrive 4K Writing", smoothing=0.1
            )

            while seq_count < TARGET_SEQUENCES:
                while len(global_token_buffer) < self.total_seq_len:
                    source_type = random.choices(
                        regular_sources, weights=regular_weights, k=1
                    )[0]
                    doc_tokens = self._get_raw_tokens(source_type)
                    if doc_tokens:
                        global_token_buffer.extend(doc_tokens)
                        global_token_buffer.append(self.doc_sep_id)

                current_seq = global_token_buffer[: self.total_seq_len]
                global_token_buffer = global_token_buffer[self.total_seq_len :]

                raw_target_n = random.randint(TARGET_N_MIN, TARGET_N_MAX)
                calibrated_n = find_topological_boundary(
                    current_seq,
                    raw_target_n,
                    self.doc_sep_id,
                    self.boundary_ids,
                    offset_large=BOUNDARY_OFFSET_LARGE,
                    margin_large=BOUNDARY_MARGIN_LARGE,
                    offset_small=BOUNDARY_OFFSET_SMALL,
                    margin_small=BOUNDARY_MARGIN_SMALL,
                )

                batch_data.append(
                    {
                        "input_ids": current_seq,
                        "n_split_index": calibrated_n,
                        "b_length": self.total_seq_len - calibrated_n,
                    }
                )

                seq_count += 1
                pbar.update(1)

                if len(batch_data) >= WRITE_BATCH_SIZE:
                    table = pa.Table.from_pylist(batch_data, schema=self.schema)
                    writer.write_table(table)
                    batch_data = []

            pbar.close()
            if batch_data:
                table = pa.Table.from_pylist(batch_data, schema=self.schema)
                writer.write_table(table)

        print("✅ 算力完全釋放：Stage 2 (4K) 絕對無重複拓撲封裝完成。")


if __name__ == "__main__":
    output_f = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    if verify_existing_matrix(
        output_f, TARGET_SEQUENCES, tag="4K 實體", exact_match=EXACT_MATCH_CHECK
    ):
        sys.exit(0)

    builder = HyperDriveTopologicalBuilder4K()
    builder.build_and_save()

    os._exit(0)
