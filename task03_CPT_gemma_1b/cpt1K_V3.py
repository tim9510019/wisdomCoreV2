"""
cpt1K_V3.py — AGIV3 持續預訓練：純 Byte 態 1K 數據引擎
===========================================================
第一性原理實踐：
1. 視界鎖定：不再依賴 Token，將物理長度嚴格鎖定在 4000 Bytes，保障 VRAM 絕對穩定。
2. 徹底消滅字典：捨棄 Tokenizer，改用 Bytearray 緩衝池，達成純淨的 UTF-8 採樣。
3. 安全量子坍縮：使用 errors='ignore' 解碼，物理阻斷中文字節被截斷引發的亂碼。
4. 自然語義邊界：動態尋找標點符號，切分 text_past 與 text_future。
"""
import os
import sys
import random
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from datasets import load_dataset

# 載入你不依賴 Tokenizer 的通用工具
# (注意：這裡移除了 get_boundary_ids 與 find_topological_boundary)
from utils import AsyncS3Prefetcher, verify_existing_matrix

# =====================================================================
# [ 全局配置區 ] 
# =====================================================================
RANDOM_SEED = 2026
OUTPUT_DIR = "./agiv2_stage1_1K"
# 🌟 檔名直接對齊 AGIV3 的 DataLoader 要求
OUTPUT_FILENAME = "agiv3_raw_text.parquet"

# 物理長度設定：1000 Tokens 大約對應 3500~4500 Bytes。我們鎖死在 4000。
TOTAL_BYTE_LEN = 4000 
TARGET_SEQUENCES = 5370000 

RATIOS = {"short_text": 0.80, "short_code": 0.20}
DATA_SOURCES = {
    "short_text": {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train"},
    "short_code": {"path": "HuggingFaceTB/smollm-corpus", "name": "python-edu", "split": "train"}
}

WRITE_BATCH_SIZE = 10000 
BUFFER_WATERMARK_BYTES = TOTAL_BYTE_LEN * 100 
ASYNC_MAX_WORKERS = 32
ASYNC_QUEUE_SIZE = 500
EXACT_MATCH_CHECK = True

# =====================================================================
# [ 文本拓撲邊界尋路器 ] 取代原本的 Token 邊界計算
# =====================================================================
def find_text_boundary(text: str, target_idx: int, margin: int = 200) -> int:
    """在純文字中尋找最接近目標點的自然標點或空白，保障切割語義完整"""
    left_bound = max(0, target_idx - margin)
    left_window = text[left_bound:target_idx]

    # 權重順序：雙換行 > 單換行 > 句號/問號 > 逗號 > 空白
    delimiters = ['\n\n', '\n', '。', '.', '！', '!', '？', '?', '，', ',', ' ']
    for delim in delimiters:
        pos = left_window.rfind(delim)
        if pos != -1:
            return left_bound + pos + len(delim)
    return target_idx

# =====================================================================
# [ 執行邏輯區 ] PyArrow 超光速直寫引擎 (純文字 Byte 版)
# =====================================================================
class HyperDriveByteBuilder:
    def __init__(self):
        print("🚀 啟動量子疊加態資料引擎：AGIV3 (純 Byte 態 | 絕對顯存鎖定 | 動態語義尋路)")
        random.seed(RANDOM_SEED)
        self.ratios = RATIOS
        self.total_byte_len = TOTAL_BYTE_LEN
        
        # 定義專屬的文檔分隔符號 (轉為 Bytes)
        self.doc_sep_bytes = b"\n\n[DOC_SEP]\n\n"

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

        # 🌟 寫入綱要改為純文字，由 DataLoader 在最後一刻才轉換為 0~255 Tensor
        self.schema = pa.schema([
            ('text_past', pa.string()),
            ('text_future', pa.string())
        ])

    def _get_next_document_bytes(self) -> bytes:
        source_type = random.choices(list(self.ratios.keys()), weights=list(self.ratios.values()), k=1)[0]
        try:
            if source_type == "short_code":
                text = self.streams["short_code"].get_next_text()
            else:
                sample = next(self.streams["short_text"])
                text = sample.get("text", sample.get("prompt", ""))
            
            if not text or not text.strip(): return b""
            return text.encode('utf-8')
            
        except StopIteration:
            raise RuntimeError(f"[物理質量枯竭] 資料流 '{source_type}' 已完全耗盡。")
        except Exception:
            return b""

    def build_and_save(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        print(f"準備將 {TARGET_SEQUENCES} 筆語意完整教材寫入實體磁區: {output_file}")
        
        seq_count = 0
        batch_data = []
        global_byte_buffer = bytearray() 
        
        with pq.ParquetWriter(output_file, self.schema) as writer:
            pbar = tqdm(total=TARGET_SEQUENCES, desc="HyperDrive Writing", smoothing=0.1)
            
            while seq_count < TARGET_SEQUENCES:
                # 填充 Buffer 直到滿足水位線
                while len(global_byte_buffer) < BUFFER_WATERMARK_BYTES:
                    doc_bytes = self._get_next_document_bytes()
                    if doc_bytes:
                        global_byte_buffer.extend(doc_bytes)
                        global_byte_buffer.extend(self.doc_sep_bytes)
                
                # 🔪 物理切割：精準切下 4000 Bytes
                current_chunk_bytes = global_byte_buffer[:self.total_byte_len]
                del global_byte_buffer[:self.total_byte_len]
                
                # 🛡️ 安全坍縮：忽略被腰斬的 UTF-8 字節，還原為乾淨字串
                current_text = current_chunk_bytes.decode('utf-8', errors='ignore')
                
                # 尋找動態切割點 (約在 30% ~ 70% 的位置)
                raw_target_n = int(len(current_text) * random.uniform(0.3, 0.7))
                calibrated_n = find_text_boundary(current_text, raw_target_n, margin=150)
                
                batch_data.append({
                    "text_past": current_text[:calibrated_n],
                    "text_future": current_text[calibrated_n:]
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
    if verify_existing_matrix(output_f, TARGET_SEQUENCES, tag="AGIV3 純文字實體", exact_match=EXACT_MATCH_CHECK):
        sys.exit(0)
        
    builder = HyperDriveByteBuilder()
    builder.build_and_save()
    
    os._exit(0)