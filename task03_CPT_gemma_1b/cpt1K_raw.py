"""
cpt1k.py — AGIV2 持續預訓練：第一階 (Beginner) 1K 數據引擎 (多線程 I/O 突破版)
===========================================================
第一性原理實踐：
1. 視界鎖定：維持 N+B 物理長度 1000，專注於局部 SDPA 高精度映射。
2. 絕對物理隔絕：徵用 Gemma 3 原生未使用的 <unused0> 作為文檔分隔牆。
3. 極端語義淨化：徹底剔除 QA 與 NIAH，配置 80% Text + 20% Code。
4. 拓撲觀測：超光速緩衝池與 O(1) 邊界尋路。
5. 異步實體映射：共用 utils 模組，打破 I/O 阻塞。
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

from utils import AsyncS3Prefetcher, verify_existing_matrix, get_boundary_ids, find_topological_boundary

# =====================================================================
# [ 全局配置區 ] 可調參數 (Tunable Parameters)
# =====================================================================
# 模型與環境設定
MODEL_ID = "google/gemma-3-1b-it"
RANDOM_SEED = 2026
DOC_SEP_TOKEN = "<unused0>" 
OUTPUT_DIR = "./agiv2_stage1_1K"
OUTPUT_FILENAME = "agiv2_stage1_N_B.parquet"

# 物理長度與序列總數
TOTAL_SEQ_LEN = 1000 
TARGET_SEQUENCES = 5370000 

# 資料來源與配比
RATIOS = {"short_text": 0.80, "short_code": 0.20}
DATA_SOURCES = {
    "short_text": {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train"},
    "short_code": {"path": "HuggingFaceTB/smollm-corpus", "name": "python-edu", "split": "train"}
}

# 寫入、緩衝與異步效能參數
WRITE_BATCH_SIZE = 10000 
BUFFER_WATERMARK = TOTAL_SEQ_LEN * 100 
ASYNC_MAX_WORKERS = 32
ASYNC_QUEUE_SIZE = 500

# 拓撲邊界與目標分割參數 (N)
TARGET_N_MIN = 300
TARGET_N_MAX = 800
BOUNDARY_OFFSET_LARGE = 300
BOUNDARY_MARGIN_LARGE = 50
BOUNDARY_OFFSET_SMALL = 150
BOUNDARY_MARGIN_SMALL = 50

# 實體檢驗設定
EXACT_MATCH_CHECK = True

# =====================================================================
# [ 執行邏輯區 ] PyArrow 超光速直寫引擎
# =====================================================================
class HyperDriveTopologicalBuilder:
    def __init__(self):
        print("🚀 啟動量子疊加態資料引擎：AGIV2 (並行 S3 預取 | O(1) 邊界尋路 | 極端語義淨化)")
        random.seed(RANDOM_SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        self.doc_sep_id = self.tokenizer.convert_tokens_to_ids(DOC_SEP_TOKEN)
        self.total_seq_len = TOTAL_SEQ_LEN
        self.ratios = RATIOS
        
        self.boundary_ids = get_boundary_ids(self.tokenizer)

        print("掛載開源異質資料流 (Streaming Parquet)...")
        self.streams = {}
        for key, conf in DATA_SOURCES.items():
            kwargs = {"path": conf["path"], "split": conf["split"], "streaming": True}
            if conf.get("name"): kwargs["name"] = conf["name"]
            
            raw_stream = iter(load_dataset(**kwargs))
            if key == "short_code":
                self.streams[key] = AsyncS3Prefetcher(
                    raw_stream, 
                    max_workers=ASYNC_MAX_WORKERS, 
                    queue_size=ASYNC_QUEUE_SIZE
                )
            else:
                self.streams[key] = raw_stream

        self.schema = pa.schema([
            ('input_ids', pa.list_(pa.int32())),
            ('n_split_index', pa.int32()),
            ('b_length', pa.int32())
        ])

    def _get_next_document_tokens(self) -> List[int]:
        source_type = random.choices(list(self.ratios.keys()), weights=list(self.ratios.values()), k=1)[0]
        try:
            if source_type == "short_code":
                text = self.streams["short_code"].get_next_text()
            else:
                sample = next(self.streams["short_text"])
                text = sample.get("text", sample.get("prompt", ""))
            
            if not text or not text.strip(): return []
            return self.tokenizer.encode(text, add_special_tokens=False)
            
        except StopIteration:
            raise RuntimeError(f"[物理質量枯竭] 資料流 '{source_type}' 已完全耗盡。")
        except Exception:
            return []

    def build_and_save(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        print(f"準備將 {TARGET_SEQUENCES} 筆語意完整教材寫入實體磁區: {output_file}")
        
        seq_count = 0
        batch_data = []
        global_token_buffer = [] 
        
        with pq.ParquetWriter(output_file, self.schema) as writer:
            pbar = tqdm(total=TARGET_SEQUENCES, desc="HyperDrive Writing", smoothing=0.1)
            
            while seq_count < TARGET_SEQUENCES:
                while len(global_token_buffer) < BUFFER_WATERMARK:
                    doc_tokens = self._get_next_document_tokens()
                    if doc_tokens:
                        global_token_buffer.extend(doc_tokens)
                        global_token_buffer.append(self.doc_sep_id)
                
                current_seq = global_token_buffer[:self.total_seq_len]
                global_token_buffer = global_token_buffer[self.total_seq_len:]
                
                raw_target_n = random.randint(TARGET_N_MIN, TARGET_N_MAX)
                calibrated_n = find_topological_boundary(
                    current_seq, raw_target_n, self.doc_sep_id, self.boundary_ids,
                    offset_large=BOUNDARY_OFFSET_LARGE, margin_large=BOUNDARY_MARGIN_LARGE, 
                    offset_small=BOUNDARY_OFFSET_SMALL, margin_small=BOUNDARY_MARGIN_SMALL
                )
                
                batch_data.append({
                    "input_ids": current_seq,
                    "n_split_index": calibrated_n, 
                    "b_length": self.total_seq_len - calibrated_n 
                })
                
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

        print("✅ 算力完全釋放：超光速封裝完成，資料庫已定錨。")

if __name__ == "__main__":
    output_f = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    if verify_existing_matrix(output_f, TARGET_SEQUENCES, tag="1K 實體", exact_match=EXACT_MATCH_CHECK):
        sys.exit(0)
        
    builder = HyperDriveTopologicalBuilder()
    builder.build_and_save()
    
    os._exit(0)