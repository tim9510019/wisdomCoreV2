"""
genCPTAC.py — AGIV2 CPT 階段解碼 (雙擎異步融合：NAR 宏觀鎖定 + AR 微觀接龍版)
=========================================================================
核心機制：
1. 認知對齊：訓練期已採用非對稱解耦，推論時必須採用 AR (自迴歸) 迴圈來釋放 Local SDPA 的文法能力。
2. 宏觀鎖定：每一次前向傳播皆強制傳入 n_split_index = N，確保 M_global (NAR引擎) 絕對不會被新生成的字元污染。
3. 權重對齊：使用 strict=True 確保 CPT 階段解凍的 Routing 與 LayerNorm 完美載入。
4. 駐波破除：引入 Temperature 與 Repetition Penalty 打破 Greedy Decoding 造成的語義引力井。
5. 負載控制：引入 MAX_PROMPT_LENGTH 限制 1B 模型的最大注意力負載，避免表徵崩潰。
6. 實驗介入：直接透過參數強制覆寫 gate_fft 與 gate_mem (無 Sigmoid，0=全關，1=全開)。
"""
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from AGIV2GAC import AGIV2G
from utils import AGIV2GForCausalLM

# ==========================================
# [ 全局配置區 ] 統一調控台 (所有可調參數)
# ==========================================
# --- 1. 路徑與基礎設定 ---
MODEL_ID = "google/gemma-3-1b-it"  # 若已重新使用 pt 版訓練，可直接改為 "google/gemma-3-1b-pt"
CHECKPOINT_PATH = "./agiv2_cpt_checkpoints_1KS/best_cpt_model.pth"
DATASET_PATH = "./agiv2_stage1_1K/agiv2_stage1_N_B.parquet"
RANDOM_SEED = 2026
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SIZE = 50                     # 驗證集切割大小

# --- 2. 推論解碼參數 (打破貪婪對稱性與限制認知負載) ---
TEST_SAMPLE_INDEX = 0              # 🎯 指定要測試的驗證集樣本 (範圍: 0 ~ TEST_SIZE-1)
MAX_DISPLAY_TOKENS = 128           # 最大生成 Token 數量
TEMPERATURE = 0.1                  # 溫度 (越低越保守，越高越隨機，設為 0.0 則退回 Greedy)
REPETITION_PENALTY = 1.2           # 重複懲罰係數 (> 1.0 表示開啟懲罰，避免無限迴圈)
MAX_PROMPT_LENGTH = 512            # 限制 1B 模型的最大輸入長度 (保留最靠近生成的 Token)

# --- 3. AGIV2G 模型物理架構參數 ---
VOCAB_SIZE = 262144
D_DIM = 1152
C_DIM = 256
HIDDEN_DIM = 6912
NUM_BLOCKS = 26
HEAD_DIM = 256
USE_GC = False                     # CausalLM 封裝是否使用 Gradient Checkpointing (推論時通常為 False)

# --- 4. 🎛️ Routing Gates 實驗控制台 (直觀開關) ---
# 設定為 1.0 為全開，0.0 為全關。設定為 None 則不介入，維持模型原權重。
OVERRIDE_GATE_FFT = 0.0            # 🎯 獨立控制 FFT 閘門
OVERRIDE_GATE_MEM = 0.0            # 🎯 獨立控制 MEM 閘門

# ==========================================
# [ 核心函式區 ] 
# ==========================================

def load_cpt_model():
    print(f"🧬 [載入] 正在初始化 AGIV2G 物理隔離架構 (CPT 模式)...")
    
    # 透過全局參數初始化基礎骨架
    base = AGIV2G(
        vocab_size=VOCAB_SIZE, 
        D=D_DIM, 
        C=C_DIM,
        hidden_dim=HIDDEN_DIM, 
        num_blocks=NUM_BLOCKS,
        head_dim=HEAD_DIM
    )
    # 封裝為 CausalLM 以匹配 trainCPTAC_1K.py 的儲存結構
    model = AGIV2GForCausalLM(base, use_gc=USE_GC)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"📬 [權重] 正在載入 CPT 結構穩定權重: {CHECKPOINT_PATH}")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        # 嚴格檢查模型骨架與 LayerNorm/Routing 權重是否完美對齊
        model.load_state_dict(state_dict, strict=True)
        print("✅ CPT 權重對齊完成 (結構穩固：Routing + LayerNorm Unfrozen)。")
    else:
        print(f"⚠️ 找不到權重檔：{CHECKPOINT_PATH}，請確認防作弊版本的訓練是否已產出檔案。")
        sys.exit(1)
        
    return model.to(DEVICE).to(torch.bfloat16).eval()

def apply_gate_override(model, fft_val, mem_val):
    """介入干預：直接透過數值強制覆寫載入後的 Routing Gates 權重"""
    if fft_val is None and mem_val is None:
        return
        
    print(f"\n⚠️ [實驗介入] 正在強制覆寫 Routing Gates 狀態：")
    core = model.base_model
    override_count = 0
    
    # 關閉梯度計算下直接修改底層參數的資料
    with torch.no_grad():
        for block in core.blocks:
            if fft_val is not None and hasattr(block, 'gate_fft'):
                block.gate_fft.data.fill_(fft_val)
                override_count += 1
            if mem_val is not None and hasattr(block, 'gate_mem'):
                block.gate_mem.data.fill_(mem_val)
                override_count += 1
                
    print(f"✅ 已成功覆寫 {override_count} 個獨立的閘門參數。")

def check_routing_gates(model):
    print("\n🔍 [分析] 檢查當前路由閘門 (Routing Gates) 數值狀態：")
    core = model.base_model
    for i, block in enumerate(core.blocks):
        if hasattr(block, 'gate_fft'):
            fft_val = block.gate_fft.item()
            mem_val = block.gate_mem.item()
            print(f"  ▶ Block {i:02d} | gate_fft: {fft_val:+.4f} | gate_mem: {mem_val:+.4f}")
    print("-" * 50)

@torch.no_grad()
def generate_cpt_autoregressive(model, tokenizer, prompt_ids, expected_b_size, max_display, temperature, repetition_penalty):
    """
    雙擎異步解碼：M_global 鎖死於 N，Local SDPA 逐字 AR 生成 (包含駐波破除)。
    """
    N_len = len(prompt_ids)
    
    # 初始輸入：只有 N 區塊 (沒有任何 PAD 污染)
    current_ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    
    # 🌟 核心防護：鎖定物理斷層點。無論迴圈跑多少次，n_split 永遠固定在 N
    n_split_tensor = torch.tensor([N_len], dtype=torch.long, device=DEVICE)
    
    print(f"\n🚀 [雙擎解碼] 啟動 AR + NAR 異步融合生成 (初始長度 N={N_len})")
    print(f"✨ [狀態] 宏觀引擎 M_global (NAR) 已鎖定於前 {N_len} 個 Token。")
    print(f"✨ [參數] Temperature={temperature}, Repetition_Penalty={repetition_penalty}")
    print(f"✨ [狀態] 微觀引擎 Local SDPA (AR) 啟動字元接龍...\n")
    
    generated_ids = []
    avg_conf = 0.0
    
    for step in range(max_display):
        # 🌟 每次前向傳播都必須傳入 n_split_index，防止未來字元流回 Phase I/II
        outputs = model(input_ids=current_ids, n_split_index=n_split_tensor)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
            
        # 取出最後一個位置的 Logit 進行預測 (AR 本質)
        next_token_logits = logits[0, -1, :].clone()
        
        # 🌟 引入重複懲罰 (Repetition Penalty) 破除駐波
        if repetition_penalty > 1.0 and len(generated_ids) > 0:
            for token_id in set(generated_ids):
                if next_token_logits[token_id] < 0:
                    next_token_logits[token_id] *= repetition_penalty
                else:
                    next_token_logits[token_id] /= repetition_penalty

        # 🌟 引入溫度 (Temperature) 打破絕對對稱性
        if temperature > 0.0:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_id = next_token.item()
            avg_conf += probs[next_token_id].item()
        else:
            # 降維至 Greedy Decoding (Temperature = 0)
            probs = F.softmax(next_token_logits, dim=-1)
            max_prob, next_token = torch.max(probs, dim=-1)
            next_token_id = next_token.item()
            avg_conf += max_prob.item()
        
        # 遇到 EOS 終止
        if next_token_id == tokenizer.eos_token_id:
            break
            
        generated_ids.append(next_token_id)
        
        # 🌟 將新預測的 Token 拼接回去，推進 AR 迴圈
        current_ids = torch.cat([current_ids, torch.tensor([[next_token_id]], device=DEVICE)], dim=1)
            
    avg_conf = avg_conf / max(1, len(generated_ids))
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"📊 預測平均信心度: {avg_conf:.4f}")
    print(f"✅ 模型 AR+NAR 融合輸出：\n{output_text}")
    print("-" * 50)
    
    return output_text

def get_eval_sample(sample_index):
    """確保提取與訓練期完全相同的獨立驗證集樣本。"""
    if not os.path.exists(DATASET_PATH):
        print(f"⚠️ 找不到實體 Parquet 資料集: {DATASET_PATH}")
        sys.exit(1)
        
    dataset_dict = load_dataset("parquet", data_files=DATASET_PATH)
    # 使用全局參數 TEST_SIZE 控制切割
    dataset = dataset_dict['train'].train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
    eval_ds = dataset['test']
    
    # 邊界防護
    if sample_index < 0 or sample_index >= len(eval_ds):
        print(f"⚠️ 警告: 指定的樣本索引 ({sample_index}) 超出範圍，強制重置為 0。")
        sample_index = 0
        
    print(f"📦 [資料集] 成功提取獨立驗證集 第 {sample_index} 筆樣本 (總數: {len(eval_ds)})")
    return eval_ds[sample_index]

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 1. 取得指定測試資料與斷層點
    sample = get_eval_sample(TEST_SAMPLE_INDEX)
    full_seq = sample["input_ids"]
    n_split = sample["n_split_index"]
    
    prompt_ids = full_seq[:n_split]
    
    # --- 🌟 核心修改：限縮上下文長度，減輕 1B 模型認知負載 ---
    if len(prompt_ids) > MAX_PROMPT_LENGTH:
        print(f"✂️ [注意] 原始上下文長度 ({len(prompt_ids)}) 超出負載限制，自動截斷保留最靠近生成的後 {MAX_PROMPT_LENGTH} 個 Token。")
        prompt_ids = prompt_ids[-MAX_PROMPT_LENGTH:]
    # --------------------------------------------------------
    
    true_b_ids = full_seq[n_split:]
    b_size = len(true_b_ids)
    
    # 2. 載入無洩漏之 CPT 模型
    model = load_cpt_model()
    
    # 🔧 介入：獨立套用 Gate FFT 與 Gate MEM 覆寫實驗
    apply_gate_override(model, OVERRIDE_GATE_FFT, OVERRIDE_GATE_MEM)
    
    # 觀測：檢查最終進入推論的權重狀態
    check_routing_gates(model)
    
    # 印出完整輸入 (A 區塊)，判斷上下文邏輯
    input_text = tokenizer.decode([i for i in prompt_ids if i != 0], skip_special_tokens=True)
    print(f"\n📥 [輸入上下文 Input Context] A 區塊內容 (長度: {len(prompt_ids)} Token)：\n{input_text}")
    print("-" * 50)
    
    # 3. 對照 Ground Truth
    true_ans_text = tokenizer.decode([i for i in true_b_ids[:MAX_DISPLAY_TOKENS] if i != 0], skip_special_tokens=True)
    
    print(f"💡 [標準答案 Ground Truth] B 區塊內容 (前 {MAX_DISPLAY_TOKENS} Token)：\n{true_ans_text}")
    print("-" * 50)
    
    print("🎯 測試啟動：AGIV2 CPT 雙擎異步解碼驗證")
    
    # 4. 啟動生成 (帶入全局配置參數)
    generate_cpt_autoregressive(
        model=model, 
        tokenizer=tokenizer, 
        prompt_ids=prompt_ids, 
        expected_b_size=b_size, 
        max_display=MAX_DISPLAY_TOKENS,
        temperature=TEMPERATURE,
        repetition_penalty=REPETITION_PENALTY
    )