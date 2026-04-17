"""
prepareRouterAC.py — 物理隔離版數據引擎 (多樣性完全恢復)
===========================================================
- N 區塊: 異質背景 + 隨機插針 + 提問 (Query 鎖定於絕對尾端)
- B 區塊: 純答案 + PAD 填充至 512
"""
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

MODEL_ID = "google/gemma-3-1b-it"
SAVE_PATH = "./agiv2_stage1_tridata_v5_ac_1k"
NUM_SAMPLES_TOTAL = 200000  
TARGET_LENGTHS = [1024] 
B_SIZE = 512
SEED = 2026

random.seed(SEED)
np.random.seed(SEED)

class QuantumRouterEngineAC:
    def __init__(self):
        print(f"🚀 [啟動] 量子疊加態數據引擎 AC 版 (多樣性保留 + N->N+B)")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.squad = load_dataset("squad_v2", split="train")
        self.fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train[:100000]")
        self.haystack_texts = [ex["text"] for ex in self.fineweb]
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.code_templates = [
            ("void update_router(int gate_id) {{ if (gate_id == {0}) active = true; }}", True),
            ("export const ROUTE_CONFIG = {{ mode: 'FFT', depth: {0} }};", True),
            ("def check_gradient_hook(grad): return grad * 1.5 # Heart Booster ID: {0}", True),
            ("/* Quantum Tunneling Protocol Static Segment */", False)
        ]

    def _generate_trinity_logic(self):
        q1 = self.squad[random.randint(0, len(self.squad)-1)]
        q2 = self.squad[random.randint(0, len(self.squad)-1)]
        subject = q1['context'][:60].strip().replace('\n', ' ')
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

        fc = random.choice([
            f"!! 警告：與 {subject} 相關的所有路徑已在執行前被撤銷 !!",
            f"SYSTEM_OVERRIDE: 解碼值 {secret_val} 已失效，請標記為 REDACTED。",
            f"if (entity == '{subject}') {{ return NULL; // ACCESS_DENIED }}"
        ])
        
        query = f"\n[系統指令] 結合 '{subject}' 的解碼值給出結果，若失效回應 'REDACTED'。\n輸出："
        return fa, fb, fc, query, secret_val

    def _get_heterogeneous_filler(self, length):
        tokens = []
        while len(tokens) < length:
            if random.random() < 0.7:
                text = random.choice(self.haystack_texts)
            else:
                tpl, has_ph = random.choice(self.code_templates)
                text = tpl.format(random.randint(100, 999)) if has_ph else tpl
            tokens.extend(self.tokenizer(text, add_special_tokens=False).input_ids)
        return tokens[:length]

    def create_triplet_ac(self, target_base):
        target_len = int(target_base * random.uniform(0.98, 1.02))
        fa, fb, fc, query, secret_val = self._generate_trinity_logic()
        
        q_ids = self.tokenizer.encode(query, add_special_tokens=False)
        t_a = self.tokenizer.encode(fa, add_special_tokens=False)
        t_b = self.tokenizer.encode(fb, add_special_tokens=False)
        t_c = self.tokenizer.encode(fc, add_special_tokens=False)
        
        n_len = target_len - B_SIZE
        available_filler_len = n_len - len(q_ids)
        base_filler = self._get_heterogeneous_filler(available_filler_len)
        
        def inject_into_ocean(filler, content_list):
            res = list(filler)
            for p, content in sorted(content_list, key=lambda x: x[0], reverse=True):
                idx = int(len(filler) * p)
                res = res[:idx] + content + res[idx:] # 安全插入，不覆蓋
            return res[:available_filler_len] + q_ids # 安全截斷背景，保證 Query 在絕對尾部

        pos_a = random.uniform(0.05, 0.45)
        pos_b, pos_c = random.uniform(pos_a + 0.15, 0.90), random.uniform(0.10, 0.95)

        n_pos_ids = inject_into_ocean(base_filler, [(pos_a, t_a), (pos_b, t_b)])
        n_neg_ids = inject_into_ocean(base_filler, [(pos_a, t_a), (pos_c, t_c)])

        def build_b_block(ans_text):
            ans_ids = self.tokenizer.encode(ans_text, add_special_tokens=False)
            if len(ans_ids) > B_SIZE: return ans_ids[:B_SIZE]
            return ans_ids + [self.tokenizer.pad_token_id] * (B_SIZE - len(ans_ids))

        pos_final = n_pos_ids + build_b_block(f" {secret_val}")
        neg_final = n_neg_ids + build_b_block(" REDACTED")

        return {"pos_ids": pos_final, "neg_ids": neg_final}

def main():
    engine = QuantumRouterEngineAC()
    data = {"pos_ids": [], "neg_ids": []}
    samples_per_len = NUM_SAMPLES_TOTAL // len(TARGET_LENGTHS)
    
    for base_len in TARGET_LENGTHS:
        print(f"🌌 正在編織 {base_len} 寬度的 AC 異質對抗場...")
        for _ in tqdm(range(samples_per_len)):
            sample = engine.create_triplet_ac(base_len)
            data["pos_ids"].append(sample["pos_ids"])
            data["neg_ids"].append(sample["neg_ids"])

    ds = Dataset.from_dict(data)
    ds.save_to_disk(SAVE_PATH)
    print(f"\n✅ 數據庫構建完成！已儲存至 {SAVE_PATH}")

if __name__ == "__main__":
    main()