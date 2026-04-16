"""
prepareRouter_V2.py — 專業級長文本 (1K~32K) 對比學習資料生成器
=============================================================
融合 SQuAD 邏輯問答與 FineWeb-Edu 真實學術背景，打造最高難度草堆。
"""
import os
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

MODEL_ID = "google/gemma-3-1b-it"
SAVE_PATH = "./agiv2_router_32k_triplets"
NUM_SAMPLES_PER_LENGTH = 1000 # 每個長度級別生成的數量
TARGET_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]

def main():
    print("🚀 啟動專業級長文本路由資料生成器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 1. 載入高品質的「針」(真實問答與證據) 與「草堆」(真實學術長文)
    print("📚 載入 SQuAD v2 (問答針) 與 FineWeb-Edu (背景草堆)...")
    squad = load_dataset("squad_v2", split="train[:10000]")
    fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train[:50000]")
    
    # 將草堆轉為純文本列表，方便快速抽取
    haystack_texts = [ex["text"] for ex in fineweb]
    
    def get_random_haystack(target_tokens):
        """從 FineWeb 隨機拼接文本，直到滿足目標 Token 長度"""
        text = ""
        while len(tokenizer.tokenize(text)) < target_tokens:
            text += random.choice(haystack_texts) + "\n"
        # 粗略截斷，保留一點餘裕
        return text

    data = {"query_ids": [], "pos_ids": [], "neg_ids": []}
    
    # 2. 針對不同長度級別 (1K ~ 32K) 進行動態插針
    for target_length in TARGET_LENGTHS:
        print(f"\n🧬 正在生成 {target_length // 1024}K 長度級別的對比三元組...")
        
        for i in tqdm(range(NUM_SAMPLES_PER_LENGTH)):
            # 隨機抽取一個 SQuAD 問答作為 Target
            qa = squad[random.randint(0, len(squad)-1)]
            query = qa["question"]
            
            # 正確的針 (Context + Answer)
            correct_context = qa["context"]
            answer = qa["answers"]["text"][0] if len(qa["answers"]["text"]) > 0 else "Unknown"
            positive_needle = f"\n[重要證據]: 關於問題「{query}」，資料顯示：{correct_context} 結論是 {answer}。\n"
            
            # 困難反例的針 (Hard Negative) - 借用其他題目的答案或混淆視聽
            wrong_qa = squad[random.randint(0, len(squad)-1)]
            wrong_answer = wrong_qa["answers"]["text"][0] if len(wrong_qa["answers"]["text"]) > 0 else "None"
            negative_needle = f"\n[過期情報]: 關於問題「{query}」，先前的錯誤紀錄曾指出：{correct_context} 但事實上是 {wrong_answer} (此為干擾)。\n"
            
            # 獲取足夠長度的真實背景雜訊
            bg_text = get_random_haystack(target_length - 200) # 預留針的長度
            
            # 隨機切點，將針動態插入草堆的任意位置 (平移不變性訓練)
            insert_pos = int(len(bg_text) * random.random())
            
            full_pos_text = bg_text[:insert_pos] + positive_needle + bg_text[insert_pos:]
            full_neg_text = bg_text[:insert_pos] + negative_needle + bg_text[insert_pos:]
            
            # Tokenize 並嚴格對齊該級別的長度 (截斷或填充)
            q_ids = tokenizer(query, max_length=128, truncation=True, padding="max_length")["input_ids"]
            p_ids = tokenizer(full_pos_text, max_length=target_length, truncation=True, padding="max_length")["input_ids"]
            n_ids = tokenizer(full_neg_text, max_length=target_length, truncation=True, padding="max_length")["input_ids"]
            
            data["query_ids"].append(q_ids)
            data["pos_ids"].append(p_ids)
            data["neg_ids"].append(n_ids)

    print("\n💾 正在保存混合長度之專業訓練集...")
    ds = Dataset.from_dict(data)
    splits = ds.train_test_split(test_size=0.05)
    splits.save_to_disk(SAVE_PATH)
    print(f"✅ SOTA 路由資料準備完成！涵蓋 1K 到 32K，已保存至 {SAVE_PATH}")

if __name__ == "__main__":
    main()