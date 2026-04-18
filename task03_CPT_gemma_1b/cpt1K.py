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

TOTAL_SEQ_LEN = 1000 
TARGET_SEQUENCES = 5370000 
OUTPUT_DIR = "./agiv2_stage1_1K"

RATIOS = {"short_text": 0.60, "short_code": 0.20, "short_qa": 0.10, "short_niah": 0.10}

DATA_SOURCES = {
    "short_text": {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train"},
    "short_code": {"path": "bigcode/starcoderdata", "name": "python", "split": "train"}, 
    "short_qa":   {"path": "Open-Orca/OpenOrca", "name": None, "split": "train"},      
    "short_niah": {"path": "m-a-p/Cosmopedia", "name": "v2", "split": "train"} 
}

# =====================================================================
# [ 執行邏輯區 ] 拓撲邊界感知生成器
# =====================================================================

class TopologicalAsymmetricBuilder:
    def __init__(self):
        print("啟動量子疊加態資料引擎：AGIV2 (N -> N+B 語意拓撲完整版)")
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
        """剝除 SFT 格式，維持純淨知識流動"""
        source_type = random.choices(list(self.ratios.keys()), weights=list(self.ratios.values()), k=1)[0]
        try:
            sample = next(self.streams[source_type])[0]
            
            if source_type == "short_qa":
                sys_prompt = sample.get("system_prompt", "")
                question = sample.get("question", "")
                response = sample.get("response", "")
                text_parts = [p for p in (sys_prompt, question, response) if p.strip()]
                text = "\n".join(text_parts)
            elif source_type == "short_text":
                text = sample.get("text", "")
            elif source_type == "short_code":
                text = sample.get("content", "") 
            else:
                text = sample.get("text", sample.get("prompt", ""))
            
            return self.tokenizer.encode(text, add_special_tokens=False), source_type
            
        except StopIteration:
            raise RuntimeError(f"[物理質量枯竭] 資料流 '{source_type}' 已完全耗盡。")
        except Exception:
            return [], source_type

    def _find_topological_boundary(self, seq: List[int], target_n: int) -> int:
        """
        核心物理防線：確保 N 切割在完整的語意拓撲節點上。
        搜尋優先級：1. 文檔分隔符 -> 2. 換行符號 -> 3. 句點。
        """
        # 1. 絕對物理斷點掃描：向外尋找最近的 doc_sep_id
        for offset in range(300):
            left_idx = target_n - offset
            right_idx = target_n + offset
            
            if left_idx > 50 and seq[left_idx] == self.doc_sep_id:
                return left_idx + 1  # 切在分隔符之後，N 包含完整文檔
            if right_idx < len(seq) - 50 and seq[right_idx] == self.doc_sep_id:
                return right_idx + 1

        # 2. 語意拓撲斷點掃描：解碼單一 Token 尋找換行或句點
        # 為了效能，我們只在 target_n 附近的 150 個 Token 內尋找
        for offset in range(150):
            left_idx = target_n - offset
            if left_idx > 50:
                token_str = self.tokenizer.decode([seq[left_idx]])
                if '\n' in token_str or token_str.strip() in ['.', '?', '!', ';']:
                    return left_idx + 1  # N 完整包含該句子/段落
            
            right_idx = target_n + offset
            if right_idx < len(seq) - 50:
                token_str = self.tokenizer.decode([seq[right_idx]])
                if '\n' in token_str or token_str.strip() in ['.', '?', '!', ';']:
                    return right_idx + 1
        
        # 3. 邊際妥協：若該區塊是超長無標點字串 (如 Base64)，退化為原始切割
        return target_n

    def generate_packed_sequences(self) -> Iterator[Dict[str, Any]]:
        """打包序列並注入經過校準的物理斷層線"""
        current_seq = []
        seq_count = 0
        
        pbar = tqdm(total=TARGET_SEQUENCES, desc="Packing Topological N->N+B Sequences")
        
        try:
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
                        # 1. 先決定一個原始的目標 N (保持不同長度的震盪訓練)
                        raw_target_n = random.randint(100, 900)
                        
                        # 2. 啟動外科手術：尋找最近的完美拓撲邊界
                        calibrated_n = self._find_topological_boundary(current_seq, raw_target_n)
                        
                        yield {
                            "input_ids": current_seq,
                            "n_split_index": calibrated_n, 
                            "b_length": self.total_seq_len - calibrated_n 
                        }
                        
                        current_seq = []
                        seq_count += 1
                        pbar.update(1)
                        
                        if seq_count >= TARGET_SEQUENCES:
                            break
        finally:
            pbar.close()

    def build_and_save(self):
        print(f"寫入 {TARGET_SEQUENCES} 筆語意完整 N->N+B 教材至 {OUTPUT_DIR}...")
        dataset = Dataset.from_generator(self.generate_packed_sequences)
        dataset.save_to_disk(OUTPUT_DIR)
        print("✅ 物理隔絕與語意拓撲校準資料庫構建完成。")

if __name__ == "__main__":
    builder = TopologicalAsymmetricBuilder()
    builder.build_and_save()