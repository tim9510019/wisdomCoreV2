"""
prepareRouter_V3.py — 路由激發終極壓力版 (針對 AGIV2G 65M 參數優化)
=============================================================================
核心邏輯：雙針邏輯推理。強制模型聯動跨 Chunk 資訊 (Fact A + Fact B)，
若不激活 gate_mem 則無法區分正負樣本。
"""
import os
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

# --- 設定中心 ---
MODEL_ID = "google/gemma-3-1b-it"
SAVE_PATH = "./agiv2_router_v3_max"
NUM_SAMPLES_PER_LENGTH = 4500  # 4種長度共 18,000 筆，確保單 EPOCH 飽和 
TARGET_LENGTHS = [4096, 8192, 16384, 32768] # 直攻長文本路由區間
SEED = 2026

random.seed(SEED)

def main():
    print(f"🔥 啟動【路由激發-雙針推理】生成器... 目標：打破 mem_max 鎖定狀態。")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 載入 SQuAD 與 FineWeb-Edu
    squad = load_dataset("squad_v2", split="train")
    fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train[:100000]")
    haystack_texts = [ex["text"] for ex in fineweb]
    
    def get_filler_tokens(target_num):
        tokens = []
        while len(tokens) < target_num:
            text = random.choice(haystack_texts)
            new_tokens = tokenizer(text, add_special_tokens=False).input_ids
            tokens.extend(new_tokens)
        return tokens[:target_num]

    data = {"query_ids": [], "pos_ids": [], "neg_ids": []}

    for target_length in TARGET_LENGTHS:
        print(f"\n⚡ 正在構建 {target_length // 1024}K 推理高壓數據...")
        
        for _ in tqdm(range(NUM_SAMPLES_PER_LENGTH)):
            # 1. 構建雙針邏輯：問題需要聯動 A 與 B
            # 隨機挑選兩個 SQuAD 樣本來組合
            qa1 = squad[random.randint(0, len(squad)-1)]
            qa2 = squad[random.randint(0, len(squad)-1)]
            
            # 正樣本針：Fact A (開頭) + Fact B (結尾)
            fact_a = f"根據機密協議 Alpha，參與者包括 {qa1['context'][:50]}。"
            fact_b = f"根據機密協議 Beta，最終的執行碼為 {qa2['answers']['text'][0] if qa2['answers']['text'] else 'XYZ-789'}。"
            query = f"結合協議 Alpha 與 Beta，請問最終執行碼是什麼？"
            
            pos_a_str = f"\n[檔案 A]: {fact_a}\n"
            pos_b_str = f"\n[檔案 B]: {fact_b}\n"
            
            # 負樣本針：Fact A (開頭) + 干擾 Fact C (結尾)
            neg_c_str = f"\n[檔案 B]: 根據過期協議 Gamma，目前的執行碼已被撤銷，暫無有效代碼。\n"

            # 2. Token 化
            q_ids = tokenizer(query, max_length=128, truncation=True, padding="max_length").input_ids
            n_a_ids = tokenizer(pos_a_str, add_special_tokens=False).input_ids
            n_b_ids = tokenizer(pos_b_str, add_special_tokens=False).input_ids
            n_c_ids = tokenizer(neg_c_str, add_special_tokens=False).input_ids

            # 3. 執行精確插針 (長度守恆) [cite: 1]
            # 計算背景空間
            pos_filler_len = target_length - len(n_a_ids) - len(n_b_ids)
            neg_filler_len = target_length - len(n_a_ids) - len(n_c_ids)
            
            base_filler = get_filler_tokens(max(pos_filler_len, neg_filler_len))
            
            # 分別插入開頭 5%-15% 與 結尾 85%-95% 位置，確保跨過多個 Global Block
            idx_a = int(pos_filler_len * random.uniform(0.05, 0.15))
            idx_b = int(pos_filler_len * random.uniform(0.85, 0.95))
            
            pos_final = (base_filler[:idx_a] + n_a_ids + 
                         base_filler[idx_a:idx_b] + n_b_ids + 
                         base_filler[idx_b:pos_filler_len])
            
            # 負樣本在同樣的 idx_b 插入錯誤資訊
            neg_final = (base_filler[:idx_a] + n_a_ids + 
                         base_filler[idx_a:idx_b] + n_c_ids + 
                         base_filler[idx_b:neg_filler_len])

            # 雙重檢查
            assert len(pos_final) == target_length
            assert len(neg_final) == target_length

            data["query_ids"].append(q_ids)
            data["pos_ids"].append(pos_final)
            data["neg_ids"].append(neg_final)

    # 4. 塌縮與保存
    ds = Dataset.from_dict(data)
    ds.save_to_disk(SAVE_PATH)
    print(f"✅ 成功！已生成 {len(ds)} 筆高壓對比數據，準備強推路由。")

if __name__ == "__main__":
    main()