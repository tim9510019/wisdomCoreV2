"""
cpt4k.py — AGIV2 持續預訓練：第二階 (Intermediate) 4K 數據引擊
===========================================================
第一性原理實踐：
1. 視界擴展：維持 N+B 物理長度 4000，強迫啟動全局共振與潛在記憶。
2. 原生隔絕符徵用：使用 <unused0> 確保零拓撲碰撞與硬體對齊保護。
3. 絕對雙針合成：動態生成無語義關聯的 UUID 雙針，強制在長脈絡中進行跨度檢索，
   鍛鍊 Tier-0 Structural Semantic Router 的極端特徵極化能力。
4. 拓撲分流：混合資料流 (Buffer) 與嚴格區塊 (Block) 雙軌並行。
"""
import os
import sys
import random
import uuid
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

# 定錨 4K 物理長度與總序列數
TOTAL_SEQ_LEN = 4000 
TARGET_SEQUENCES = 2750000 
OUTPUT_DIR = "./agiv2_stage2_4K"

# 第二階段配比 (Stage 2: Intermediate)
RATIOS = {"long_text": 0.50, "single_code": 0.30, "long_qa": 0.10, "double_niah": 0.10}

# 數據源升級
DATA_SOURCES = {
    "long_text":   {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train"},
    "single_code": {"path": "HuggingFaceTB/smollm-corpus", "name": "python-edu", "split": "train"}, 
    "long_qa":     {"path": "Open-Orca/OpenOrca", "name": "default", "split": "train"}
    # double_niah 不需掛載外部資料庫，將以 long_text 為草堆進行實時動態合成
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
        print("🚀 啟動量子疊加態資料引擎：AGIV2 (Stage 2 - 4K 視界擴展與動態雙針)")
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

    def _get_raw_tokens(self, source_type: str) -> List[int]:
        """從指定的資料流抽取一段原始 Token 序列"""
        try:
            sample = next(self.streams[source_type])
            if source_type == "long_qa":
                text = "\n".join([p for p in (sample.get("system_prompt", ""), sample.get("question", ""), sample.get("response", "")) if p and p.strip()])
            else:
                text = sample.get("text", sample.get("prompt", ""))
            return self.tokenizer.encode(text, add_special_tokens=False)
        except StopIteration:
            raise RuntimeError(f"[物理質量枯竭] 資料流 '{source_type}' 已耗盡。")
        except Exception:
            return []

    def _generate_dynamic_niah_block(self) -> Dict[str, Any]:
        """
        核心突破：實時動態合成雙針矩陣
        強制構造長度精準為 4000 的獨立序列，確保 N (草堆+針+問題) 與 B (答案) 物理隔絕。
        """
        # 1. 鍛造雙針特徵 (採用無規律 UUID 避免預訓練記憶干擾)
        key_alpha = uuid.uuid4().hex[:8]
        key_beta = uuid.uuid4().hex[:8]
        
        needle_1 = self.tokenizer.encode(f"\n[系統底層日誌：覆寫核心矩陣，存取金鑰 Alpha 為 {key_alpha}]\n", add_special_tokens=False)
        needle_2 = self.tokenizer.encode(f"\n[系統底層日誌：路由通道解鎖，存取金鑰 Beta 為 {key_beta}]\n", add_special_tokens=False)
        
        query = self.tokenizer.encode(f"\n{DOC_SEP_TOKEN}\n[系統查核] 請根據上述文本的底層日誌，精確提取存取金鑰 Alpha 與 Beta。\n答案：", add_special_tokens=False)
        answer = self.tokenizer.encode(f"金鑰 Alpha 為 {key_alpha}，金鑰 Beta 為 {key_beta}。{DOC_SEP_TOKEN}", add_special_tokens=False)
        
        # 2. 計算剩餘草堆容積
        reserved_len = len(needle_1) + len(needle_2) + len(query) + len(answer)
        target_haystack_len = self.total_seq_len - reserved_len
        
        # 3. 採集足量的背景草堆 (使用 long_text)
        haystack = []
        while len(haystack) < target_haystack_len:
            haystack.extend(self._get_raw_tokens("long_text"))
            haystack.append(self.doc_sep_id)
            
        haystack = haystack[:target_haystack_len] # 精準切斷
        
        # 4. 隨機決定雙針的物理深度 (分佈於 10%~40% 與 60%~80% 區間)
        idx_1 = random.randint(int(target_haystack_len * 0.1), int(target_haystack_len * 0.4))
        idx_2 = random.randint(int(target_haystack_len * 0.6), int(target_haystack_len * 0.8))
        
        # 5. 拓撲重組
        final_seq = (
            haystack[:idx_1] + 
            needle_1 + 
            haystack[idx_1:idx_2] + 
            needle_2 + 
            haystack[idx_2:] + 
            query + 
            answer
        )
        
        # 6. 計算絕對隔絕點：剛好在 query 結束、answer 開始的位置
        n_split = len(final_seq) - len(answer)
        
        return {
            "input_ids": final_seq,
            "n_split_index": n_split,
            "b_length": len(answer)
        }

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
        
        # 剝離雙針比例，重新計算常規數據源的權重
        regular_sources = [k for k in RATIOS.keys() if k != "double_niah"]
        regular_weights = [RATIOS[k] for k in regular_sources]
        
        with pq.ParquetWriter(output_file, self.schema) as writer:
            pbar = tqdm(total=TARGET_SEQUENCES, desc="HyperDrive 4K Writing")
            
            while seq_count < TARGET_SEQUENCES:
                # 第一層分流：是否觸發動態雙針？
                if random.random() < RATIOS["double_niah"]:
                    batch_data.append(self._generate_dynamic_niah_block())
                else:
                    # 第二層分流：常規全局緩衝池拼接
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

        print("✅ 算力完全釋放：Stage 2 (4K) 動態雙針拓撲封裝完成。")

if __name__ == "__main__":
    if verify_existing_matrix(): sys.exit(0)
    builder = HyperDriveTopologicalBuilder4K()
    builder.build_and_save()
    os._exit(0)