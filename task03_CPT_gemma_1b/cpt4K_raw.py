"""
cpt4k.py — AGIV2 持續預訓練：第二階 (Intermediate) 4K 數據引擊 (絕對無重複極限版)
===========================================================
第一性原理實踐：
1. 視界擴展：維持 N+B 物理長度 4000，強迫啟動全局共振與潛在記憶。
2. 原生隔絕符徵用：使用 <unused0> 確保零拓撲碰撞與硬體對齊保護。
3. 極端語義淨化：徹底剔除 QA 與 NIAH，配置 80% Text + 20% Code。
4. 絕對無重複 (Zero-Repetition)：完全捨棄 Epoch 輪迴，接入 100BT 文本與 59B 程式碼巨獸矩陣，
   確保模型在 110 億 Tokens 的訓練流中，看見的每一個字都是全新的物理實體。
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

# 【核心突破】直接徵用原生空白符號，避免任何實體矩陣干擾
DOC_SEP_TOKEN = "<unused0>" 

# 定錨 4K 物理長度與總序列數 (資料總數維持不變，總計需消耗 ~11B Tokens)
TOTAL_SEQ_LEN = 4000 
TARGET_SEQUENCES = 2750000 
OUTPUT_DIR = "./agiv2_stage2_4K"

# 🌟 語義淨化配比 (80% 純文本，20% 多樣化純 CODE)
RATIOS = {"long_text": 0.80, "code_smollm": 0.10, "code_redpajama": 0.10}

# 🌟 數據源巨獸化 (10BT -> 100BT, 並新增 RedPajama 59B Github 矩陣)
DATA_SOURCES = {
    "long_text":      {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-100BT", "split": "train"},
    "code_smollm":    {"path": "HuggingFaceTB/smollm-corpus", "name": "python-edu", "split": "train"},
    "code_redpajama": {"path": "togethercomputer/RedPajama-Data-1T", "name": "github", "split": "train"}
}

WRITE_BATCH_SIZE = 5000 
BUFFER_WATERMARK = TOTAL_SEQ_LEN * 100 

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
        print("🚀 啟動量子疊加態資料引擎：AGIV2 (Stage 2 - 4K 視界擴展，無重複巨量版)")
        random.seed(RANDOM_SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # 取得原生預留 ID，保持 100% 矩陣維度穩定
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
            kwargs = {"path": conf["path"], "split": conf["split"], "streaming": True}
            if conf["name"]: kwargs["name"] = conf["name"]
            self.streams[key] = iter(load_dataset(**kwargs))

        self.schema = pa.schema([
            ('input_ids', pa.list_(pa.int32())),
            ('n_split_index', pa.int32()),
            ('b_length', pa.int32())
        ])

    def _get_raw_tokens(self, source_type: str) -> List[int]:
        """🌟 絕對不輪迴：只提取真實且無重複的語義，遇缺即爆錯"""
        try:
            sample = next(self.streams[source_type])
            text = sample.get("text", sample.get("content", sample.get("code", "")))
            return self.tokenizer.encode(text, add_special_tokens=False)
        except StopIteration:
            raise RuntimeError(f"[物理質量枯竭] 資料流 '{source_type}' 已徹底耗盡，無法滿足您的無重複要求！")
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
        
        # 動態計算來源權重
        regular_sources = list(RATIOS.keys())
        regular_weights = list(RATIOS.values())
        
        with pq.ParquetWriter(output_file, self.schema) as writer:
            pbar = tqdm(total=TARGET_SEQUENCES, desc="HyperDrive 4K Writing")
            
            while seq_count < TARGET_SEQUENCES:
                # 專注於全局緩衝池拼接 (純粹 Text 與 Code 的無盡接龍)
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