"""
cpt1k.py — AGIV2 持續預訓練：第一階 (Beginner) 1K 數據引擎
===========================================================
第一性原理實踐：
1. 視界鎖定：維持 N+B 物理長度 1000，專注於局部 SDPA 高精度映射。
2. 絕對物理隔絕：徵用 Gemma 3 原生未使用的 <unused0> 作為文檔分隔牆，
   完全避免詞表擴增導致的維度碰撞與硬體對齊(262144)失效。
3. 極端語義淨化：徹底剔除 QA 與 NIAH，配置 80% Text + 20% Code。
   強制洗除 -it 模型的指令慣性，逼迫 Local SDPA 專注於高熵文本的連續自迴歸接龍。
4. 拓撲觀測：超光速緩衝池與 O(1) 邊界尋路。
"""
import os
import sys
import random
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Iterator, Dict, List, Any
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# =====================================================================
# [ 全局配置區 ] 物理隔絕矩陣參數 (Physical Isolation Matrix)
# =====================================================================
MODEL_ID = "google/gemma-3-1b-it"
RANDOM_SEED = 2026

# 【核心突破】直接徵用原生空白符號，拋棄詞表擴充與整數寄生
DOC_SEP_TOKEN = "<unused0>" 

# N+B 總物理長度鎖定為 1000 (資料總數維持不變)
TOTAL_SEQ_LEN = 1000 
TARGET_SEQUENCES = 5370000 
OUTPUT_DIR = "./agiv2_stage1_1K"

# 🌟 語義淨化配比 (徹底移除 QA 與 NIAH，強迫洗掉 -it 對話慣性)
RATIOS = {"short_text": 0.80, "short_code": 0.20}

# 全線接入高質量 Parquet 原生矩陣
DATA_SOURCES = {
    "short_text": {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train"},
    "short_code": {"path": "HuggingFaceTB/smollm-corpus", "name": "python-edu", "split": "train"}
}

WRITE_BATCH_SIZE = 10000 
BUFFER_WATERMARK = TOTAL_SEQ_LEN * 100 

# =====================================================================
# [ 觀測邏輯區 ] 實體拓撲檢驗 (Topology Verification)
# =====================================================================

def verify_existing_matrix() -> bool:
    output_file = os.path.join(OUTPUT_DIR, "agiv2_stage1_N_B.parquet")
    if not os.path.exists(output_file):
        print("🌌 觀測結果：實體磁區不存在，準備無中生有。")
        return False
        
    print(f"🔍 偵測到既有實體磁區：{output_file}，啟動物理完整性驗證...")
    try:
        pf = pq.ParquetFile(output_file)
        num_rows = pf.metadata.num_rows
        print(f"📊 當前序列數 (Rows): {num_rows} / 預期序列數: {TARGET_SEQUENCES}")
        
        if num_rows == TARGET_SEQUENCES:
            print("✅ 拓撲完整性 100%！實體矩陣已完美定錨，無須重複運算。")
            return True
        else:
            print("⚠️ 序列數量不符，宇宙發生了坍縮。準備啟動超光速引擎進行覆寫...")
            return False
    except Exception as e:
        print(f"⚠️ 磁區讀取失敗 ({e})，檔案內部結構已損毀。準備啟動引擎覆寫...")
        return False

# =====================================================================
# [ 執行邏輯區 ] PyArrow 超光速直寫引擎
# =====================================================================

class HyperDriveTopologicalBuilder:
    def __init__(self):
        print("🚀 啟動量子疊加態資料引擎：AGIV2 (超光速緩衝版 | O(1) 邊界尋路 | 極端語義淨化)")
        random.seed(RANDOM_SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # 取得原生預留 ID，絕對不會發生越界或改變矩陣大小
        self.doc_sep_id = self.tokenizer.convert_tokens_to_ids(DOC_SEP_TOKEN)
        print(f"🔗 實體定錨：已將 {DOC_SEP_TOKEN} (ID: {self.doc_sep_id}) 徵用為跨文檔物理隔絕牆。")

        self.total_seq_len = TOTAL_SEQ_LEN
        self.ratios = RATIOS
        
        print("預計算高維度拓撲邊界 Token IDs...")
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
            self.streams[key] = iter(load_dataset(**kwargs))

        self.schema = pa.schema([
            ('input_ids', pa.list_(pa.int32())),
            ('n_split_index', pa.int32()),
            ('b_length', pa.int32())
        ])

    def _get_next_document_tokens(self) -> List[int]:
        source_type = random.choices(list(self.ratios.keys()), weights=list(self.ratios.values()), k=1)[0]
        try:
            sample = next(self.streams[source_type])
            
            # 🌟 移除 QA 特殊邏輯，純粹提取連續文本
            text = sample.get("text", sample.get("prompt", ""))
            
            return self.tokenizer.encode(text, add_special_tokens=False)
            
        except StopIteration:
            raise RuntimeError(f"[物理質量枯竭] 資料流 '{source_type}' 已完全耗盡。")
        except Exception:
            return []

    def _find_topological_boundary(self, seq: List[int], target_n: int) -> int:
        for offset in range(300):
            left_idx = target_n - offset
            right_idx = target_n + offset
            if left_idx > 50 and seq[left_idx] == self.doc_sep_id:
                return left_idx + 1  
            if right_idx < len(seq) - 50 and seq[right_idx] == self.doc_sep_id:
                return right_idx + 1

        for offset in range(150):
            left_idx = target_n - offset
            if left_idx > 50 and seq[left_idx] in self.boundary_ids:
                return left_idx + 1  
            
            right_idx = target_n + offset
            if right_idx < len(seq) - 50 and seq[right_idx] in self.boundary_ids:
                return right_idx + 1
                
        return target_n

    def build_and_save(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, "agiv2_stage1_N_B.parquet")
        print(f"準備將 {TARGET_SEQUENCES} 筆語意完整 N->N+B 教材，光速寫入實體磁區: {output_file}")
        
        seq_count = 0
        batch_data = []
        global_token_buffer = [] 
        
        with pq.ParquetWriter(output_file, self.schema) as writer:
            pbar = tqdm(total=TARGET_SEQUENCES, desc="HyperDrive Writing")
            
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
    if verify_existing_matrix():
        sys.exit(0)
        
    builder = HyperDriveTopologicalBuilder()
    builder.build_and_save()
    
    os._exit(0)