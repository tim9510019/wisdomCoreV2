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
DOC_SEP_TOKEN = "<|doc_sep|>"

# N+B 總物理長度鎖定為 1000
TOTAL_SEQ_LEN = 1000 
TARGET_SEQUENCES = 5370000 
OUTPUT_DIR = "./agiv2_stage1_1K"

# 三位一體的黃金配比
RATIOS = {"short_text": 0.60, "short_code": 0.20, "short_qa": 0.10, "short_niah": 0.10}

# 全線接入高質量 Parquet 原生矩陣
DATA_SOURCES = {
    "short_text": {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train"},
    "short_code": {"path": "HuggingFaceTB/smollm-corpus", "name": "python-edu", "split": "train"}, 
    "short_qa":   {"path": "Open-Orca/OpenOrca", "name": "default", "split": "train"},      
    "short_niah": {"path": "HuggingFaceTB/smollm-corpus", "name": "cosmopedia-v2", "split": "train"} 
}

# 記憶體防護：每收集滿 10,000 筆序列寫入一次硬碟
WRITE_BATCH_SIZE = 10000 
# 全局緩衝池水位線 (維持約 10 萬 Tokens 的儲備，極小化 I/O 中斷)
BUFFER_WATERMARK = TOTAL_SEQ_LEN * 100 

# =====================================================================
# [ 觀測邏輯區 ] 實體拓撲檢驗 (Topology Verification)
# =====================================================================

def verify_existing_matrix() -> bool:
    """
    第一性原理：在啟動龐大算力前，先進行 O(1) 的宏觀狀態觀測。
    若矩陣已達完美狀態，則直接凍結時間，節省邊際效用。
    """
    output_file = os.path.join(OUTPUT_DIR, "agiv2_stage1_N_B.parquet")
    if not os.path.exists(output_file):
        print("🌌 觀測結果：實體磁區不存在，準備無中生有。")
        return False
        
    print(f"🔍 偵測到既有實體磁區：{output_file}，啟動物理完整性驗證...")
    try:
        # 使用 ParquetFile 讀取 metadata，速度極快且不佔記憶體
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
        print("🚀 啟動量子疊加態資料引擎：AGIV2 (超光速緩衝版 | O(1) 邊界尋路)")
        random.seed(RANDOM_SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        if DOC_SEP_TOKEN not in self.tokenizer.get_vocab():
            print(f"[警告] 詞表中未發現 {DOC_SEP_TOKEN}，將使用 <EOS> 替代。")
            self.doc_sep_id = self.tokenizer.eos_token_id
        else:
            self.doc_sep_id = self.tokenizer.convert_tokens_to_ids(DOC_SEP_TOKEN)

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
            
            if source_type == "short_qa":
                sys_prompt = sample.get("system_prompt", "")
                question = sample.get("question", "")
                response = sample.get("response", "")
                text_parts = [p for p in (sys_prompt, question, response) if p and p.strip()]
                text = "\n".join(text_parts)
            else:
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
    # 1. 執行觀測：檢查既有宇宙是否已經完美
    if verify_existing_matrix():
        sys.exit(0)  # 完美則直接退出
        
    # 2. 啟動坍縮：執行超光速引擎重新生成
    builder = HyperDriveTopologicalBuilder()
    builder.build_and_save()
    
    # 3. 絕對靜止協議：繞過 Python 的垃圾回收 (GC)，強制瞬間凍結進程。
    # 這能完美避開 Hugging Face streaming datasets 在關閉時觸發的假性 Core Dumped 報錯。
    os._exit(0)