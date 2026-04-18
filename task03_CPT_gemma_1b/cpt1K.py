import os
import random
from typing import Iterator, Dict, List
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# =====================================================================
# [ 全局配置區 ] 核心參數控制台 (Global Configuration Console)
# =====================================================================

# 1. 系統與模型基礎設定
MODEL_ID = "google/gemma-3-1b-it"
RANDOM_SEED = 2026
DOC_SEP_TOKEN = "<|doc_sep|>"

# 2. 課程量化指標 (Curriculum Spec)
MAX_SEQ_LEN = 1024
TARGET_SEQUENCES = 10000  # 測試用；實際 SOTA 藍圖滿載應設為 5,500,000
OUTPUT_DIR = "./agiv2_stage1_1k_dataset"

# 3. 資料配比矩陣 (Ratios)
RATIOS = {
    "short_text": 0.60,
    "short_code": 0.20,
    "short_qa": 0.10,
    "short_niah": 0.10
}

# 4. 異質資料來源定義 (Dataset Streams)
DATA_SOURCES = {
    "short_text": {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train"},
    "short_code": {"path": "bigcode/the-stack-smol", "name": None, "split": "train"},
    "short_qa":   {"path": "squad_v2", "name": None, "split": "train"},
    # 依藍圖指示，資源有限時 NIAH 可由 QA/輕量腳本替代
    "short_niah": {"path": "squad_v2", "name": None, "split": "validation"} 
}
STREAM_BATCH_SIZE = 1

# =====================================================================
# [ 執行邏輯區 ] 引擎與打包器實作
# =====================================================================

class Stage1CoursewareBuilder:
    def __init__(self):
        print("啟動量子疊加態資料引擎：Stage 1 (1K Beginner)")
        random.seed(RANDOM_SEED)
        
        # 初始化詞表
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # 詞表外科手術檢查
        if DOC_SEP_TOKEN not in self.tokenizer.get_vocab():
            print(f"[警告] 詞表中未發現 {DOC_SEP_TOKEN}，將暫時使用 <EOS> 替代以維持防護。")
            self.doc_sep_id = self.tokenizer.eos_token_id
        else:
            self.doc_sep_id = self.tokenizer.convert_tokens_to_ids(DOC_SEP_TOKEN)

        self.max_seq_len = MAX_SEQ_LEN
        self.ratios = RATIOS
        
        # 動態掛載資料流
        print("掛載異質資料流 (Streaming)...")
        self.streams = {}
        for key, conf in DATA_SOURCES.items():
            kwargs = {"path": conf["path"], "split": conf["split"], "streaming": True}
            if conf["name"]:
                kwargs["name"] = conf["name"]
            
            self.streams[key] = load_dataset(**kwargs).iter(batch_size=STREAM_BATCH_SIZE)

    def _get_next_document_tokens(self) -> List[int]:
        """基於機率分佈抽取下一個文檔，並轉換為 Tokens"""
        source_type = random.choices(list(self.ratios.keys()), weights=list(self.ratios.values()), k=1)[0]
        try:
            sample = next(self.streams[source_type])[0]
            
            # 根據資料集結構提取文字
            if source_type == "short_text":
                text = sample["text"]
            elif source_type == "short_code":
                text = sample["content"]
            else: # QA / NIAH
                ans = sample['answers']['text'][0] if sample['answers']['text'] else 'None'
                text = f"Question: {sample['question']} Context: {sample['context']} Answer: {ans}"
            
            return self.tokenizer.encode(text, add_special_tokens=False)
        except StopIteration:
            return []
        except Exception:
            return []

    def generate_packed_sequences(self) -> Iterator[Dict[str, List[int]]]:
        """無損拼接與絕對屏障隔離邏輯"""
        current_seq = []
        seq_count = 0
        
        pbar = tqdm(total=TARGET_SEQUENCES, desc="Packing 1K Sequences")
        
        while seq_count < TARGET_SEQUENCES:
            doc_tokens = self._get_next_document_tokens()
            if not doc_tokens: continue
            
            # 插入跨文檔防堵標籤
            doc_tokens.append(self.doc_sep_id)
            
            while doc_tokens:
                remaining_space = self.max_seq_len - len(current_seq)
                if len(doc_tokens) <= remaining_space:
                    current_seq.extend(doc_tokens)
                    doc_tokens = []
                else:
                    current_seq.extend(doc_tokens[:remaining_space])
                    doc_tokens = doc_tokens[remaining_space:]
                
                if len(current_seq) == self.max_seq_len:
                    yield {"input_ids": current_seq}
                    current_seq = []
                    seq_count += 1
                    pbar.update(1)
                    
                    if seq_count >= TARGET_SEQUENCES:
                        break
        pbar.close()

    def build_and_save(self):
        """將序列化的流數據固化為磁碟格式"""
        print(f"開始構建 {TARGET_SEQUENCES} 筆教材，準備寫入 {OUTPUT_DIR}...")
        
        dataset = Dataset.from_generator(self.generate_packed_sequences)
        dataset.save_to_disk(OUTPUT_DIR)
        print("✅ 第一階：入門教材構建與封裝完成。")

if __name__ == "__main__":
    builder = Stage1CoursewareBuilder()
    builder.build_and_save()