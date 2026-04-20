"""
cpt1k.py — AGIV2 持續預訓練：第一階 (Beginner) 1K 數據引擎 (多線程 I/O 突破版)
===========================================================
第一性原理實踐：
1. 視界鎖定：維持 N+B 物理長度 1000，專注於局部 SDPA 高精度映射。
2. 絕對物理隔絕：徵用 Gemma 3 原生未使用的 <unused0> 作為文檔分隔牆。
3. 極端語義淨化：徹底剔除 QA 與 NIAH，配置 80% Text + 20% Code。
4. 拓撲觀測：超光速緩衝池與 O(1) 邊界尋路。
5. 異步實體映射 (NEW)：啟動 32 執行緒的背景 S3 預取引擎，打破 I/O 阻塞。
"""
import os
import sys
import random
import threading
import queue
import concurrent.futures
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

import boto3
import botocore
from smart_open import open as smart_open

# =====================================================================
# [ 雲端觀測區 ] S3 實體連線初始化
# =====================================================================
# 使用無簽名配置以進行高頻率匿名存取
s3_client = boto3.client(
    's3',
    region_name='us-west-2',
    config=botocore.config.Config(signature_version=botocore.UNSIGNED)
)

# =====================================================================
# [ 全局配置區 ] 物理隔絕矩陣參數
# =====================================================================
MODEL_ID = "google/gemma-3-1b-it"
RANDOM_SEED = 2026
DOC_SEP_TOKEN = "<unused0>" 

TOTAL_SEQ_LEN = 1000 
TARGET_SEQUENCES = 5370000 
OUTPUT_DIR = "./agiv2_stage1_1K"

RATIOS = {"short_text": 0.80, "short_code": 0.20}
DATA_SOURCES = {
    "short_text": {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train"},
    "short_code": {"path": "HuggingFaceTB/smollm-corpus", "name": "python-edu", "split": "train"}
}

WRITE_BATCH_SIZE = 10000 
BUFFER_WATERMARK = TOTAL_SEQ_LEN * 100 

# =====================================================================
# [ 異步引擎區 ] 解決 S3 網路 I/O 阻塞的並行預取器
# =====================================================================
class AsyncS3Prefetcher:
    """背景非同步抓取 S3 程式碼的緩衝池"""
    def __init__(self, stream, max_workers=32, queue_size=500):
        self.stream = stream
        self.queue = queue.Queue(maxsize=queue_size)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.stop_event = threading.Event()
        self.producer_thread = threading.Thread(target=self._produce, daemon=True)
        self.producer_thread.start()
        print(f"⚡ 異步預取引擎啟動：{max_workers} 執行緒全速映射 S3 實體檔案...")

    def _fetch(self, blob_id: str) -> str:
        try:
            s3_url = f"s3://softwareheritage/content/{blob_id}"
            with smart_open(s3_url, "rb", compression=".gz", transport_params={"client": s3_client}) as f:
                return f.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def _produce(self):
        try:
            for sample in self.stream:
                if self.stop_event.is_set():
                    break
                blob_id = sample.get("blob_id")
                if blob_id:
                    # 將抓取任務丟給線程池，並將 Future 物件放入 Queue
                    future = self.executor.submit(self._fetch, blob_id)
                    self.queue.put(future) # 若 Queue 滿了會自動阻塞，不吃光記憶體
        except Exception as e:
            print(f"\n⚠️ 預取引擎發生異常: {e}")
        finally:
            self.queue.put(None) # 發送終止信號

    def get_next_text(self) -> str:
        """主執行緒呼叫此方法，O(1) 取得已下載的程式碼"""
        future = self.queue.get()
        if future is None:
            raise StopIteration("S3 程式碼資料流已枯竭")
        return future.result()

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
        
        self.boundary_ids = set()
        boundary_chars = ['\n', '.', '?', '!', ';', '。', '！', '？', '；']
        test_strings = boundary_chars + [f" {c}" for c in boundary_chars] + [f"a{c}" for c in boundary_chars]
        for s in test_strings:
            tokens = self.tokenizer.encode(s, add_special_tokens=False)
            if tokens:
                self.boundary_ids.add(tokens[-1]) 

        print("掛載開源異質資料流 (Streaming Parquet)...")
        self.streams = {}
        for key, conf in DATA_SOURCES.items():
            kwargs = {"path": conf["path"], "split": conf["split"], "streaming": True}
            if conf["name"]: kwargs["name"] = conf["name"]
            
            raw_stream = iter(load_dataset(**kwargs))
            if key == "short_code":
                # 將原始 code stream 包裝進我們寫好的異步預取器
                self.streams[key] = AsyncS3Prefetcher(raw_stream, max_workers=32, queue_size=500)
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
                # 從預取器直接拿取已經抓好/正在抓的 text
                text = self.streams["short_code"].get_next_text()
            else:
                sample = next(self.streams["short_text"])
                text = sample.get("text", sample.get("prompt", ""))
            
            if not text or not text.strip():
                return []
                
            return self.tokenizer.encode(text, add_special_tokens=False)
            
        except StopIteration:
            raise RuntimeError(f"[物理質量枯竭] 資料流 '{source_type}' 已完全耗盡。")
        except Exception:
            return []

    def _find_topological_boundary(self, seq: List[int], target_n: int) -> int:
        for offset in range(300):
            left_idx = target_n - offset
            right_idx = target_n + offset
            if left_idx > 50 and seq[left_idx] == self.doc_sep_id: return left_idx + 1  
            if right_idx < len(seq) - 50 and seq[right_idx] == self.doc_sep_id: return right_idx + 1

        for offset in range(150):
            left_idx = target_n - offset
            if left_idx > 50 and seq[left_idx] in self.boundary_ids: return left_idx + 1  
            right_idx = target_n + offset
            if right_idx < len(seq) - 50 and seq[right_idx] in self.boundary_ids: return right_idx + 1
                
        return target_n

    def build_and_save(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, "agiv2_stage1_N_B.parquet")
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
                
                raw_target_n = random.randint(300, 800)
                calibrated_n = self._find_topological_boundary(current_seq, raw_target_n)
                
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
    output_f = os.path.join(OUTPUT_DIR, "agiv2_stage1_N_B.parquet")
    if os.path.exists(output_f):
        print("實體矩陣已存在，請確認是否需要刪除重建。")
        sys.exit(0)
        
    builder = HyperDriveTopologicalBuilder()
    builder.build_and_save()
    
    os._exit(0)