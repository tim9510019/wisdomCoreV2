import os
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

# --- 2026 SOTA Roadmap 對齊參數 ---
MODEL_ID = "google/gemma-3-1b-it"
SAVE_PATH = "./agiv2_stage1_tridata_v5"
NUM_SAMPLES_TOTAL = 18000  # Roadmap 階段一：TRIDATA [cite: 3, 4]
TARGET_LENGTHS = [4096, 8192, 16384, 32768] # 階段一主攻 32K 區間 [cite: 4, 9]
SEED = 2026

random.seed(SEED)
np.random.seed(SEED)

class QuantumRouterEngineV5:
    def __init__(self):
        print(f"🚀 [啟動] 量子疊加態數據引擎 V5 - TRIDATA 修正版")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # 多維度數據源：模擬 Roadmap 階段二的四通道預演 [cite: 8, 9]
        self.squad = load_dataset("squad_v2", split="train")
        self.fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train[:100000]")
        self.haystack_texts = [ex["text"] for ex in self.fineweb]
        
        # 修正後的代碼通道：確保所有模板在傳導路徑上具備一致性
        self.code_templates = [
            ("void update_router(int gate_id) {{ if (gate_id == {0}) active = true; }}", True),
            ("export const ROUTE_CONFIG = {{ mode: 'FFT', depth: {0} }};", True),
            ("def check_gradient_hook(grad): return grad * 1.5 # Heart Booster ID: {0}", True),
            ("/* Quantum Tunneling Protocol Static Segment */", False)
        ]

    def _generate_trinity_logic(self):
        """建構三位一體邏輯：主體 A + 值 B + 邏輯狀態 C [cite: 4, 15]"""
        q1 = self.squad[random.randint(0, len(self.squad)-1)]
        q2 = self.squad[random.randint(0, len(self.squad)-1)]
        
        subject = q1['context'][:60].strip().replace('\n', ' ')
        # 處理可能的空答案 
        ans_list = q2['answers']['text']
        secret_val = ans_list[0] if ans_list else f"TOKEN_{random.randint(1000,9999)}"
        
        fmt = random.choice(["json", "xml", "natural", "protocol"])
        if fmt == "json":
            fa = f'{{"node": "ALPHA", "subject": "{subject}"}}'
            fb = f'{{"node": "BETA", "value": "{secret_val}", "status": "verified"}}'
        elif fmt == "protocol":
            fa = f"SET_SUBJECT >>> {subject} [SIG_A]"
            fb = f"SET_VALUE >>> {secret_val} [SIG_B]"
        else:
            fa = f"實體標識為「{subject}」。"
            fb = f"該實體的解碼核心值為：{secret_val}。"

        # 強負樣本邏輯：撤銷門 (Revocation Gate) [cite: 5, 15]
        fc = random.choice([
            f"!! 警告：與 {subject} 相關的所有路徑已在執行前被撤銷 !!",
            f"SYSTEM_OVERRIDE: 解碼值 {secret_val} 已失效，請標記為 REDACTED。",
            f"if (entity == '{subject}') {{ return NULL; // ACCESS_DENIED }}"
        ])
        
        query = f"結合主體 '{subject}' 與其對應的解碼值，請給出最終執行碼。若邏輯門顯示無效，回應 'REDACTED'。"
        return fa, fb, fc, query, secret_val

    def _get_heterogeneous_filler(self, length):
        """模擬異質熵背景 (四通道比例預演) [cite: 9]"""
        tokens = []
        while len(tokens) < length:
            r = random.random()
            if r < 0.7: # 通用文本通道
                text = random.choice(self.haystack_texts)
            else: # 代碼/結構化通道：修正後的格式化邏輯
                tpl, has_placeholder = random.choice(self.code_templates)
                text = tpl.format(random.randint(100, 999)) if has_placeholder else tpl
            
            new_tokens = self.tokenizer(text, add_special_tokens=False).input_ids
            tokens.extend(new_tokens)
        return tokens[:length]

    def create_triplet(self, target_base):
        """生成對齊梯度強心針所需的三元組 [cite: 5, 6, 22]"""
        target_len = int(target_base * random.uniform(0.98, 1.02))
        fa, fb, fc, query, _ = self._generate_trinity_logic()
        
        q_ids = self.tokenizer(query, max_length=128, truncation=True, padding="max_length").input_ids
        t_a = self.tokenizer(fa, add_special_tokens=False).input_ids
        t_b = self.tokenizer(fb, add_special_tokens=False).input_ids
        t_c = self.tokenizer(fc, add_special_tokens=False).input_ids
        
        # NIAH 插針位置優化 [cite: 7, 12, 17]
        pos_a = random.uniform(0.05, 0.45)
        pos_b = random.uniform(pos_a + 0.15, 0.90) 
        pos_c = random.uniform(0.10, 0.95)
        
        # 預留緩衝空間計算總長度
        content_len = len(t_a) + max(len(t_b), len(t_c))
        filler_len = max(0, target_len - content_len - 20) 
        base_filler = self._get_heterogeneous_filler(filler_len)

        def inject(filler, content_list):
            res = list(filler)
            for p, content in sorted(content_list, key=lambda x: x[0], reverse=True):
                idx = int(len(filler) * p)
                res[idx:idx] = content
            return res

        # 正樣本：路徑 A -> B [cite: 1, 18]
        pos_final = inject(base_filler, [(pos_a, t_a), (pos_b, t_b)])
        # 負樣本：路徑 A -> C (邏輯截斷) [cite: 1, 18]
        neg_final = inject(base_filler, [(pos_a, t_a), (pos_c, t_c)])

        return {
            "query_ids": q_ids,
            "pos_ids": self._finalize(pos_final, target_len),
            "neg_ids": self._finalize(neg_final, target_len)
        }

    def _finalize(self, ids, length):
        if len(ids) > length: return ids[:length]
        return ids + [self.tokenizer.pad_token_id] * (length - len(ids))

def main():
    engine = QuantumRouterEngineV5()
    data = {"query_ids": [], "pos_ids": [], "neg_ids": []}
    
    samples_per_len = NUM_SAMPLES_TOTAL // len(TARGET_LENGTHS)
    
    for base_len in TARGET_LENGTHS:
        print(f"🌌 正在編織 {base_len} 寬度的 TRIDATA 對抗場...")
        for _ in tqdm(range(samples_per_len)):
            sample = engine.create_triplet(base_len)
            data["query_ids"].append(sample["query_ids"])
            data["pos_ids"].append(sample["pos_ids"])
            data["neg_ids"].append(sample["neg_ids"])

    ds = Dataset.from_dict(data)
    ds.save_to_disk(SAVE_PATH)
    print(f"\n✅ V5 修正版數據集已就緒於 {SAVE_PATH}！ [cite: 3]")
    print(f"📊 階段一指標啟動：Loss < 3.0 ")

if __name__ == "__main__":
    main()