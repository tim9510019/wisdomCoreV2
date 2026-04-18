import os
import random
from typing import Iterator, Dict, List, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# =====================================================================
# [ 全局配置區 ] 物理隔絕矩陣參數
# =====================================================================
MODEL_ID = "google/gemma-3-1b-it"
RANDOM_SEED = 2026
DOC_SEP_TOKEN = "<|doc_sep|>"

# N+B 總物理長度鎖定為 1000
TOTAL_SEQ_LEN = 1000 
TARGET_SEQUENCES = 5370000 
OUTPUT_DIR = "./agiv2_stage1_N_plus_B_formless"

RATIOS = {"short_text": 0.60, "short_code": 0.20, "short_qa": 0.10, "short_niah": 0.10}

DATA_SOURCES = {
    "short_text": {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train"},
    "short_code": {"path": "bigcode/starcoderdata", "name": "python", "split": "train"}, 
    "short_qa":   {"path": "Open-Orca/OpenOrca", "name": None, "split": "train"},      
    "short_niah": {"path": "m-a-p/Cosmopedia", "name": "v2", "split": "train"} 
}

# =====================================================================
# [ 執行邏輯區 ] 物理隔絕資料生成器 (去格式化版)
# =====================================================================

class FormlessAsymmetricBuilder:
    def __init__(self):
        print("啟動量子疊加態資料引擎：AGIV2 (N -> N+B 去格式化純淨版)")
        random.seed(RANDOM_SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        if DOC_SEP_TOKEN not in self.tokenizer.get_vocab():
            self.doc_sep_id = self.tokenizer.eos_token_id
        else:
            self.doc_sep_id = self.tokenizer.convert_tokens_to_ids(DOC_SEP_TOKEN)

        self.total_seq_len = TOTAL_SEQ_LEN
        self.ratios = RATIOS
        self.streams = {}
        
        for key, conf in DATA_SOURCES.items():
            kwargs = {"path": conf["path"], "split": conf["split"], "streaming": True}
            if conf["name"]: kwargs["name"] = conf["name"]
            self.streams[key] = load_dataset(**kwargs).iter(batch_size=1)

    def _get_next_document(self) -> tuple[List[int], str]:
        """返回 Token 列表以及資料來源類型。徹底剝除 SFT 格式。"""
        source_type = random.choices(list(self.ratios.keys()), weights=list(self.ratios.values()), k=1)[0]
        try:
            sample = next(self.streams[source_type])[0]
            
            if source_type == "short_qa":
                # 【破除模板桎梏】捨棄 System/Question/Answer 標籤
                # 讓模型學習「敘述前提 -> 提出疑問 -> 給出解答」的自然語義過渡
                sys_prompt = sample.get("system_prompt", "")
                question = sample.get("question", "")
                response = sample.get("response", "")
                # 僅用自然換行或空白銜接，不加入任何人工提示詞
                text_parts = [p for p in (sys_prompt, question, response) if p.strip()]
                text = "\n".join(text_parts)
                
            elif source_type == "short_text":
                text = sample.get("text", "")
            elif source_type == "short_code":
                text = sample.get("content", "") 
            else: # short_niah
                text = sample.get("text", sample.get("prompt", ""))
            
            return self.tokenizer.encode(text, add_special_tokens=False), source_type
            
        except StopIteration:
            raise RuntimeError(f"[物理質量枯竭] 資料流 '{source_type}' 已完全耗盡。")
        except Exception:
            return [], source_type

    def _determine_dynamic_n(self, dominant_source: str) -> int:
        """
        動態邊界 $N$：不再依賴任務的固定結構，而是讓模型在任意長度的
        過去 (X_past) 中，學習推導未來 (X_future)。
        """
        # 放寬 N 的震盪範圍，給予模型更多極端長度差異的訓練
        # 讓模型學會：無論看了 50 個字還是 950 個字，都能鎖定全局拓撲並預測 B
        return random.randint(50, 950)

    def generate_packed_sequences(self) -> Iterator[Dict[str, Any]]:
        """打包序列並注入物理斷層線"""
        current_seq = []
        dominant_sources = [] 
        seq_count = 0
        
        pbar = tqdm(total=TARGET_SEQUENCES, desc="Packing Formless N->N+B Sequences")
        
        try:
            while seq_count < TARGET_SEQUENCES:
                doc_tokens, source_type = self._get_next_document()
                if not doc_tokens: continue
                
                # 保留唯一必要的人工標籤：絕對防護的文檔分隔符
                doc_tokens.append(self.doc_sep_id)
                
                while doc_tokens:
                    remaining_space = self.total_seq_len - len(current_seq)
                    if len(doc_tokens) <= remaining_space:
                        current_seq.extend(doc_tokens)
                        dominant_sources.append((source_type, len(doc_tokens)))
                        doc_tokens = []
                    else:
                        current_seq.extend(doc_tokens[:remaining_space])
                        dominant_sources.append((source_type, remaining_space))
                        doc_tokens = doc_tokens[remaining_space:]
                    
                    if len(current_seq) == self.total_seq_len:
                        main_source = max(dominant_sources, key=lambda x: x[1])[0]
                        n_split = self._determine_dynamic_n(main_source)
                        
                        yield {
                            "input_ids": current_seq,
                            "n_split_index": n_split, 
                            "b_length": self.total_seq_len - n_split 
                        }
                        
                        current_seq = []
                        dominant_sources = []
                        seq_count += 1
                        pbar.update(1)
                        
                        if seq_count >= TARGET_SEQUENCES:
                            break
        finally:
            pbar.close()

    def build_and_save(self):
        print(f"寫入 {TARGET_SEQUENCES} 筆無相 N->N+B 教材至 {OUTPUT_DIR}...")
        dataset = Dataset.from_generator(self.generate_packed_sequences)
        dataset.save_to_disk(OUTPUT_DIR)
        print("✅ 物理隔絕純淨資料庫構建完成。")

if __name__ == "__main__":
    builder = FormlessAsymmetricBuilder()
    builder.build_and_save()