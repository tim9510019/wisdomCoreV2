"""
cpt4k.py — AGIV2 持續預訓練：第二階 (Intermediate) 4K 數據引擎 (絕對無重複極限版)
===========================================================
第一性原理實踐：
1. 視界擴展：維持 N+B 物理長度 4000，強迫啟動全局共振與潛在記憶。
2. 原生隔絕符徵用：使用 <unused0> 確保零拓撲碰撞與硬體對齊保護。
3. 極端語義淨化：徹底剔除 QA 與 NIAH，配置 80% Text + 20% Code。
4. 絕對無重複：完全捨棄 Epoch 輪迴，接入 100BT 文本與巨獸矩陣。
5. 異質流觀測：針對無實體矩陣 (smollm) 啟動 S3 預取。
6. 絕對純度過濾 (FIXED)：撤除無效的目錄映射，改用內部自旋鎖定，100% 防禦配比稀釋與語義污染。
"""
import os
import sys
import json
import random
import threading
import queue
import concurrent.futures
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Iterator, Dict, List, Any
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

import boto3
import botocore
from smart_open import open as smart_open

# =====================================================================
# [ 雲端觀測區 ] S3 實體連線初始化
# =====================================================================
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

TOTAL_SEQ_LEN = 4000 
TARGET_SEQUENCES = 2750000 
OUTPUT_DIR = "./agiv2_stage2_4K"

RATIOS = {"long_text": 0.80, "code_smollm": 0.10, "code_redpajama": 0.10}

# 🌟 數據源巨獸化與精準鎖定
DATA_SOURCES = {
    "long_text":      {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-100BT", "split": "train"},
    "code_smollm":    {"path": "HuggingFaceTB/smollm-corpus", "name": "python-edu", "split": "train"},
    # 🛡️ 撤除 data_dir，避免 HF 虛擬目錄解析崩潰，改由程式內部邏輯全盤接管
    "code_redpajama": {"path": "MBZUAI-LLM/SlimPajama-627B-DC", "name": "default", "split": "train"} 
}

WRITE_BATCH_SIZE = 5000 
BUFFER_WATERMARK = TOTAL_SEQ_LEN * 100 

# =====================================================================
# [ 異步引擎區 ] 解決 S3 網路 I/O 阻塞的並行預取器
# =====================================================================
class AsyncS3Prefetcher:
    def __init__(self, stream, max_workers=32, queue_size=500):
        self.stream = stream
        self.queue = queue.Queue(maxsize=queue_size)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.stop_event = threading.Event()
        self.producer_thread = threading.Thread(target=self._produce, daemon=True)
        self.producer_thread.start()
        print(f"⚡ 異步預取引擎啟動：{max_workers} 執行緒全速映射 S3 實體檔案 (針對 smollm)...")

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
                    future = self.executor.submit(self._fetch, blob_id)
                    self.queue.put(future) 
        except Exception as e:
            print(f"\n⚠️ 預取引擎發生異常: {e}")
        finally:
            self.queue.put(None) 

    def get_next_text(self) -> str:
        future = self.queue.get()
        if future is None:
            raise StopIteration("S3 程式碼資料流已枯竭")
        return future.result()

# =====================================================================
# [ 觀測邏輯區 ] 實體拓撲檢驗
# =====================================================================
def verify_existing_matrix() -> bool:
    output_file = os.path.join(OUTPUT_DIR, "agiv2_stage2_N_B_4K.parquet")
    if not os.path.exists(output_file):
        print("🌌 觀測結果：4K 實體磁區不存在，準備無中生有。")
        return False
        
    try:
        pf = pq.ParquetFile(output_file)
        num_rows = pf.metadata.num_rows
        print(f"📊 當前序列數 (Rows): {num_rows} / 預期序列數: {TARGET_SEQUENCES}")
        if num_rows >= TARGET_SEQUENCES * 0.99: 
            print("✅ 拓撲完整性達標！4K 實體矩陣已定錨。")
            return True
        return False
    except Exception as e:
        print(f"⚠️ 磁區讀取失敗 ({e})。準備啟動引擎覆寫...")
        return False

# =====================================================================
# [ 執行邏輯區 ] PyArrow 超光速直寫引擎與動態合成器
# =====================================================================
class HyperDriveTopologicalBuilder4K:
    def __init__(self):
        print("🚀 啟動量子疊加態資料引擎：AGIV2 (Stage 2 - 4K 視界擴展，純淨代碼過濾版)")
        random.seed(RANDOM_SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        self.doc_sep_id = self.tokenizer.convert_tokens_to_ids(DOC_SEP_TOKEN)
        print(f"🔗 實體定錨：已將 {DOC_SEP_TOKEN} (ID: {self.doc_sep_id}) 徵用為跨文檔物理隔絕牆。")

        self.total_seq_len = TOTAL_SEQ_LEN
        
        print("預計算高維度拓撲邊界 Token IDs...")
        self.boundary_ids = set()
        boundary_chars = ['\n', '.', '?', '!', ';', '。', '！', '？', '；']
        for s in boundary_chars + [f" {c}" for c in boundary_chars] + [f"a{c}" for c in boundary_chars]:
            tokens = self.tokenizer.encode(s, add_special_tokens=False)
            if tokens: self.boundary_ids.add(tokens[-1]) 

        print("掛載開源巨獸資料流 (Streaming Parquet)...")
        self.streams = {}
        for key, conf in DATA_SOURCES.items():
            kwargs = {
                "path": conf["path"], 
                "split": conf["split"], 
                "streaming": True
            }
            if conf.get("name"): kwargs["name"] = conf["name"]
                
            raw_stream = iter(load_dataset(**kwargs))
            
            if key == "code_smollm":
                self.streams[key] = AsyncS3Prefetcher(raw_stream, max_workers=32, queue_size=500)
            else:
                self.streams[key] = raw_stream

        self.schema = pa.schema([
            ('input_ids', pa.list_(pa.int32())),
            ('n_split_index', pa.int32()),
            ('b_length', pa.int32())
        ])

    def _get_raw_tokens(self, source_type: str) -> List[int]:
        """🌟 絕對不輪迴：只提取真實且無重複的語義，加入內部自旋鎖定機制"""
        try:
            if source_type == "code_smollm":
                text = self.streams[source_type].get_next_text()
            elif source_type == "code_redpajama":
                # 🛡️ 內部自旋鎖定：持續消耗資料流，直到抽中 GitHub 磁區為止，確保配比不被稀釋
                while True:
                    sample = next(self.streams[source_type])
                    
                    # 容錯解析：標籤可能在頂層，也可能在 meta 欄位中
                    set_name = sample.get("redpajama_set_name")
                    if not set_name:
                        meta = sample.get("meta", {})
                        if isinstance(meta, str):
                            try: meta = json.loads(meta)
                            except: meta = {}
                        set_name = meta.get("redpajama_set_name", "")
                    
                    # 嚴格觀測：命中 GitHub 磁區才跳出迴圈提取文本
                    if set_name == "RedPajamaGithub":
                        text = sample.get("text", sample.get("content", sample.get("code", "")))
                        break
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

    def _find_topological_boundary(self, seq: List[int], target_n: int) -> int:
        for offset in range(800):
            left_idx, right_idx = target_n - offset, target_n + offset
            if left_idx > 100 and seq[left_idx] == self.doc_sep_id: return left_idx + 1  
            if right_idx < len(seq) - 100 and seq[right_idx] == self.doc_sep_id: return right_idx + 1

        for offset in range(400):
            left_idx, right_idx = target_n - offset, target_n + offset
            if left_idx > 100 and seq[left_idx] in self.boundary_ids: return left_idx + 1  
            if right_idx < len(seq) - 100 and seq[right_idx] in self.boundary_ids: return right_idx + 1
                
        return target_n

    def build_and_save(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, "agiv2_stage2_N_B_4K.parquet")
        print(f"準備將 {TARGET_SEQUENCES} 筆 4K 長脈絡教材寫入實體磁區: {output_file}")
        
        seq_count = 0
        batch_data = []
        global_token_buffer = [] 
        
        regular_sources = list(RATIOS.keys())
        regular_weights = list(RATIOS.values())
        
        with pq.ParquetWriter(output_file, self.schema) as writer:
            pbar = tqdm(total=TARGET_SEQUENCES, desc="HyperDrive 4K Writing", smoothing=0.1)
            
            while seq_count < TARGET_SEQUENCES:
                while len(global_token_buffer) < self.total_seq_len:
                    source_type = random.choices(regular_sources, weights=regular_weights, k=1)[0]
                    doc_tokens = self._get_raw_tokens(source_type)
                    if doc_tokens:
                        global_token_buffer.extend(doc_tokens)
                        global_token_buffer.append(self.doc_sep_id)
                
                current_seq = global_token_buffer[:self.total_seq_len]
                global_token_buffer = global_token_buffer[self.total_seq_len:]
                
                raw_target_n = random.randint(1500, 3200)
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

        print("✅ 算力完全釋放：Stage 2 (4K) 絕對無重複拓撲封裝完成。")

if __name__ == "__main__":
    if verify_existing_matrix(): sys.exit(0)
    builder = HyperDriveTopologicalBuilder4K()
    builder.build_and_save()
    
    os._exit(0)