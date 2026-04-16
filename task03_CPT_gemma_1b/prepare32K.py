import os
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets, interleave_datasets

# ==========================================
# 🌟 全域參數控制中心 (Configuration) - 極速精準版
# ==========================================
# --- 模型與輸出 ---
MODEL_ID = "google/gemma-3-1b-it"
SAVE_PATH = "./agiv2_pretokenized_32k_mixed"
MAX_LENGTH = 32768

# --- 目標數量 ---
TOTAL_TARGET = 40050  # 總目標 32K 筆數
TEST_SIZE = 50        # 驗證集切分大小

# --- 系統效能 ---
NUM_PROC = 16         # 多核心處理數量 
SEED = 42             # 亂數種子，確保結果可重現

# --- 各通道目標比例 (順應資料集物理限制，重新調配) ---
RATIO_SHORT = 0.50    # 目標 20,025 筆 (吸收了 Code 的缺口)
RATIO_BOOK  = 0.30    # 目標 12,015 筆
RATIO_CODE  = 0.10    # 目標 4,005 筆 (避免觸碰 7.68M 的資料集天花板)
# NIAH 將自動使用剩餘的 0.10 (4,005 筆)

# --- 原始資料撈取量上限 (精準邊界：只 Tokenize 剛好夠用的量) ---
# 這些數值是根據歷史產能精算出的「微幅溢出」安全邊界，拒絕浪費算力。
RAW_EDU_LIMIT = 420000    # (短文) 40萬 + 35萬 = 75萬，足夠產出 2萬筆 32K
RAW_COS_LIMIT = 380000    
RAW_BOOK_LIMIT = 4500     # (書籍) 4500 本足夠切出 1.3萬筆 32K
RAW_CODE_LIMIT = 4200000  # (程式碼) 約需 1000 筆原始碼才湊滿 1 筆 32K，故設定 420 萬

def main():
    # --- 動態計算各通道精確需求 ---
    N_SHORT = int(TOTAL_TARGET * RATIO_SHORT)
    N_BOOK  = int(TOTAL_TARGET * RATIO_BOOK)
    N_CODE  = int(TOTAL_TARGET * RATIO_CODE)
    N_NIAH  = TOTAL_TARGET - N_SHORT - N_BOOK - N_CODE 

    print(f"🚀 啟動大規模處理任務，總目標: {TOTAL_TARGET} 筆 32K 樣本")
    print(f"📊 通道分配: 短文 {N_SHORT} | 書籍 {N_BOOK} | 程式碼 {N_CODE} | NIAH {N_NIAH}")
    print("啟動 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # ==========================================
    # 1. 極速 Packing 管線 (動態偵測、無差別對待)
    # ==========================================
    def tokenize_func(examples):
        # 抱持未知的態度，無差別捕捉字串欄位
        valid_cols = [k for k, v in examples.items() if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str)]
        
        target_col = None
        if "text" in valid_cols: target_col = "text"
        elif "content" in valid_cols: target_col = "content"
        elif "code" in valid_cols: target_col = "code"
        elif valid_cols: target_col = valid_cols[0] 
            
        if target_col is None:
            return {"input_ids": []} 
            
        texts = examples[target_col]
        texts = [t if t is not None else "" for t in texts]
        
        if len(texts) == 0:
            return {"input_ids": []}
            
        return tokenizer(texts, add_special_tokens=False)

    def group_texts(examples):
        concatenated_ids = []
        for ids in examples["input_ids"]:
            concatenated_ids.extend(ids)
            concatenated_ids.append(tokenizer.eos_token_id)
            
        total_length = len(concatenated_ids)
        total_length = (total_length // MAX_LENGTH) * MAX_LENGTH
        
        result = {
            "input_ids": [
                concatenated_ids[i : i + MAX_LENGTH] for i in range(0, total_length, MAX_LENGTH)
            ]
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def process_channel(dataset, desc_name, target_n):
        print(f"\n[{desc_name}] 開始 Tokenize (原始資料量: {len(dataset)})...")
        tokenized = dataset.map(
            tokenize_func, 
            batched=True, 
            num_proc=NUM_PROC, 
            remove_columns=dataset.column_names,
            desc=f"{desc_name} Tokenizing"
        )
        print(f"[{desc_name}] 開始進行 32K Packing...")
        packed = tokenized.map(
            group_texts, 
            batched=True, 
            batch_size=1000, 
            num_proc=NUM_PROC, 
            remove_columns=tokenized.column_names,
            desc=f"{desc_name} Packing"
        )
        
        current_len = len(packed)
        if current_len < target_n:
            print(f"⚠️ 狀態警告：[{desc_name}] 原始資料轉換後不足！僅生成 {current_len} 筆，未達目標 {target_n} 筆。請調高上方的 RAW_LIMIT。")
            return packed 
        else:
            print(f"✅ [{desc_name}] 成功生成超出預期的 {current_len} 筆，精準截斷至目標的 {target_n} 筆。")
            return packed.select(range(target_n))

    # ==========================================
    # 2. 載入三大文本通道 (由全域參數精準控制)
    # ==========================================
    print(f"\n📦 載入並混合通道 A：短文百科 (最大擷取量: {RAW_EDU_LIMIT + RAW_COS_LIMIT} 筆)...")
    ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split=f"train[:{RAW_EDU_LIMIT}]")
    ds_cos = load_dataset("HuggingFaceTB/cosmopedia", name="stanford", split=f"train[:{RAW_COS_LIMIT}]")
    ds_short_raw = interleave_datasets([ds_edu, ds_cos], probabilities=[0.6, 0.4])
    packed_short = process_channel(ds_short_raw, "短文通道", N_SHORT)

    print(f"\n📦 載入通道 B：長文書籍 (最大擷取量: {RAW_BOOK_LIMIT} 本)...")
    ds_book_raw = load_dataset("emozilla/pg19", split=f"train[:{RAW_BOOK_LIMIT}]")
    packed_book = process_channel(ds_book_raw, "書籍通道", N_BOOK)

    print(f"\n📦 載入通道 C：程式碼 (最大擷取量: {RAW_CODE_LIMIT} 筆)...")
    ds_code_raw = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split=f"train[:{RAW_CODE_LIMIT}]")
    packed_code = process_channel(ds_code_raw, "程式碼通道", N_CODE)

    # ==========================================
    # 3. 生成通道 D：NIAH 撈針 (無損精確生成)
    # ==========================================
    print(f"\n💉 瞬間生成通道 D：合成撈針 (精確生成 {N_NIAH} 筆)...")
    eos_id = tokenizer.eos_token_id
    filler_tokens = tokenizer("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. Programming is the art of expressing solutions to problems. The quick brown fox jumps over the lazy dog. ", add_special_tokens=False)['input_ids']

    niah_data = {"input_ids": [], "labels": []}
    for _ in tqdm(range(N_NIAH), desc="NIAH 生成中"):
        ntype = random.randint(0, 5)
        val = str(random.randint(10000, 99999))
        instruction = "There is important information hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n"
        needle = f"The secret passkey is {val}. Remember it. {val} is the secret passkey."
        qa = f"\nWhat is the secret passkey? The secret passkey is {val}"
        
        if ntype == 1:
            code_val = f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))}-{random.randint(10, 99)}"
            needle = f"The activation code is {code_val}. Remember it. {code_val} is the activation code."
            qa = f"\nWhat is the activation code? The activation code is {code_val}"
        elif ntype == 2:
            name = random.choice(['FALCON', 'PHOENIX', 'GLACIER', 'NEBULA', 'OBSIDIAN'])
            needle = f"The project codename is {name}. Remember it. {name} is the project codename."
            qa = f"\nWhat is the project codename? The project codename is {name}"
        elif ntype == 3:
            instruction = "以下文字中隱藏了一條重要資訊，請仔細閱讀並記住它。我稍後會問你相關問題。\n\n"
            needle = f"機密通關密碼是 {val}。請務必記住這個密碼。{val} 就是機密通關密碼。"
            qa = f"\n機密通關密碼是什麼？機密通關密碼是 {val}"
        elif ntype == 4:
            code_val = f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))}-{random.randint(10, 99)}"
            instruction = "請仔細閱讀以下文字，其中隱藏著一條關鍵資訊。讀完後請回答問題。\n\n"
            needle = f"啟動代碼是 {code_val}。請務必記住這個代碼。{code_val} 就是啟動代碼。"
            qa = f"\n啟動代碼是什麼？啟動代碼是 {code_val}"
        elif ntype == 5:
            name = random.choice(['雷霆行動', '銀河計畫', '鳳凰協議', '極光任務'])
            instruction = "請仔細閱讀以下文字，其中隱藏著一條關鍵資訊。讀完後請回答問題。\n\n"
            needle = f"專案代號是{name}。請務必記住這個代號。{name}就是專案代號。"
            qa = f"\n專案代號是什麼？專案代號是{name}"

        inst_ids = tokenizer(instruction, add_special_tokens=False)['input_ids']
        needle_ids = tokenizer(" " + needle + " ", add_special_tokens=False)['input_ids']
        qa_ids = tokenizer(qa, add_special_tokens=False)['input_ids'] + [eos_id]

        reserved = len(inst_ids) + len(needle_ids) + len(qa_ids)
        filler_budget = max(0, MAX_LENGTH - reserved)
        repeats = (filler_budget // len(filler_tokens)) + 2
        all_filler = (filler_tokens * repeats)[:filler_budget]

        depth = random.uniform(0.0, 1.0)
        split_pos = int(len(all_filler) * depth)

        chunk = inst_ids + all_filler[:split_pos] + needle_ids + all_filler[split_pos:] + qa_ids
        chunk = chunk[:MAX_LENGTH]
        
        niah_data["input_ids"].append(chunk)
        niah_data["labels"].append(chunk)

    packed_niah = Dataset.from_dict(niah_data)

    # ==========================================
    # 4. 終極融合與保存
    # ==========================================
    print(f"\n🔀 正在完美比例融合四通道並隨機打亂 (Seed={SEED})...")
    final_mixed = concatenate_datasets([packed_short, packed_book, packed_code, packed_niah])
    final_mixed = final_mixed.shuffle(seed=SEED)

    print(f"✂️ 分割訓練集與驗證集 (驗證集設定: {TEST_SIZE} 筆)...")
    final_splits = final_mixed.train_test_split(test_size=TEST_SIZE)
    
    print(f"💾 寫入實體檔案至 {SAVE_PATH} ...")
    final_splits.save_to_disk(SAVE_PATH)
    print("✅ 全規模預處理順利完成！資料的無序量子態已塌縮為確定的高階訓練數據。")

if __name__ == "__main__":
    main()