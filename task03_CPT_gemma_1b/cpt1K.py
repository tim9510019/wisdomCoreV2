import os
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

RATIOS = {"short_text": 0.60, "short_code": 0.20, "short_qa": 0.10, "short_niah": 0.10}

DATA_SOURCES = {
    "short_text": {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train"},
    "short_code": {"path": "HuggingFaceTB/smollm-corpus", "name": "python-edu", "split": "train"}, 
    "short_qa":   {"path": "Open-Orca/OpenOrca", "name": "default", "split": "train"},      
    "short_niah": {"path": "HuggingFaceTB/smollm-corpus", "name": "cosmopedia-v2", "split": "train"} 
}

# 記憶體防護：每收集滿多少筆序列寫入一次硬碟
WRITE_BATCH_SIZE = 10000 

# =====================================================================
# [ 執行邏輯區 ] PyArrow 實體直寫引擎 (Absolute Deadlock Immunity)
# =====================================================================

class PyArrowTopologicalBuilder:
    def __init__(self):
        print("啟動量子疊加態資料引擎：AGIV2 (PyArrow 底層直寫版 | 絕對防死結)")
        random.seed(RANDOM_SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        if DOC_SEP_TOKEN not in self.tokenizer.get_vocab():
            print(f"[警告] 詞表中未發現 {DOC_SEP_TOKEN}，將使用 <EOS> 替代。")
            self.doc_sep_id = self.tokenizer.eos_token_id
        else:
            self.doc_sep_id = self.tokenizer.convert_tokens_to_ids(DOC_SEP_TOKEN)

        self.total_seq_len = TOTAL_SEQ_LEN
        self.ratios = RATIOS
        
        print("掛載開源異質資料流 (Streaming Parquet)...")
        self.streams = {}
        for key, conf in DATA_SOURCES.items():
            kwargs = {"path": conf["path"], "split": conf["split"], "streaming": True}
            if conf["name"]: kwargs["name"] = conf["name"]
            # 建立單純的 Python Iterator
            self.streams[key] = iter(load_dataset(**kwargs))

        # 定義絕對的實體 Schema (寫入 Parquet 的資料表結構)
        self.schema = pa.schema([
            ('input_ids', pa.list_(pa.int32())),
            ('n_split_index', pa.int32()),
            ('b_length', pa.int32())
        ])

    def _get_next_document(self) -> tuple[List[int], str]:
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
            
            return self.tokenizer.encode(text, add_special_tokens=False), source_type
            
        except StopIteration:
            raise RuntimeError(f"[物理質量枯竭] 資料流 '{source_type}' 已完全耗盡。")
        except Exception as e:
            return [], source_type

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
            if left_idx > 50:
                token_str = self.tokenizer.decode([seq[left_idx]])
                if '\n' in token_str or token_str.strip() in ['.', '?', '!', ';']:
                    return left_idx + 1  
            right_idx = target_n + offset
            if right_idx < len(seq) - 50:
                token_str = self.tokenizer.decode([seq[right_idx]])
                if '\n' in token_str or token_str.strip() in ['.', '?', '!', ';']:
                    return right_idx + 1
        return target_n

    def build_and_save(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, "agiv2_stage1_N_B.parquet")
        print(f"準備將 {TARGET_SEQUENCES} 筆語意完整 N->N+B 教材，分批寫入實體磁區: {output_file}")
        
        current_seq = []
        seq_count = 0
        batch_data = []
        
        # 啟動 PyArrow 的 Parquet 寫入引擎
        with pq.ParquetWriter(output_file, self.schema) as writer:
            pbar = tqdm(total=TARGET_SEQUENCES, desc="Direct Writing Parquet")
            
            while seq_count < TARGET_SEQUENCES:
                doc_tokens, _ = self._get_next_document()
                if not doc_tokens: continue
                
                doc_tokens.append(self.doc_sep_id)
                
                while doc_tokens:
                    remaining_space = self.total_seq_len - len(current_seq)
                    if len(doc_tokens) <= remaining_space:
                        current_seq.extend(doc_tokens)
                        doc_tokens = []
                    else:
                        current_seq.extend(doc_tokens[:remaining_space])
                        doc_tokens = doc_tokens[remaining_space:]
                    
                    if len(current_seq) == self.total_seq_len:
                        raw_target_n = random.randint(300, 800)
                        calibrated_n = self._find_topological_boundary(current_seq, raw_target_n)
                        
                        # 收集完成的序列
                        batch_data.append({
                            "input_ids": current_seq,
                            "n_split_index": calibrated_n, 
                            "b_length": self.total_seq_len - calibrated_n 
                        })
                        
                        current_seq = []
                        seq_count += 1
                        pbar.update(1)
                        
                        # 記憶體防護：批次寫入硬碟
                        if len(batch_data) >= WRITE_BATCH_SIZE:
                            table = pa.Table.from_pylist(batch_data, schema=self.schema)
                            writer.write_table(table)
                            batch_data = [] # 清空記憶體
                        
                        if seq_count >= TARGET_SEQUENCES:
                            break
            pbar.close()
            
            # 清空最後剩餘的緩衝區
            if batch_data:
                table = pa.Table.from_pylist(batch_data, schema=self.schema)
                writer.write_table(table)

        print("✅ 第一性原理實踐：資料已無損寫入硬碟。再也沒有多執行緒死結。")

if __name__ == "__main__":
    builder = PyArrowTopologicalBuilder()
    builder.build_and_save()