"""
genCPTAC.py — AGIV2 CPT 階段解碼 (雙擎異步融合：NAR 宏觀鎖定 + AR 微觀接龍版)
=========================================================================
核心機制：
1. 認知對齊：訓練期已採用非對稱解耦，推論時必須採用 AR (自迴歸) 迴圈來釋放 Local SDPA 的文法能力。
2. 宏觀鎖定：每一次前向傳播皆強制傳入 n_split_index = N，確保 M_global (NAR引擎) 絕對不會被新生成的字元污染。
3. 權重對齊：使用 strict=True 確保 CPT 階段解凍的 Routing 與 LayerNorm 完美載入。
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
# [ 全局配置區 ]
# ==========================================
MODEL_ID = "google/gemma-3-1b-it"
CHECKPOINT_PATH = "./agiv2_cpt_checkpoints_1K/best_cpt_model.pth"
DATASET_PATH = "./agiv2_stage1_1K/agiv2_stage1_N_B.parquet"
RANDOM_SEED = 2026
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_cpt_model():
    print(f"🧬 [載入] 正在初始化 AGIV2G 物理隔離架構 (CPT 模式)...")
    
    # 初始化基礎骨架
    base = AGIV2G(
        vocab_size=262144, 
        D=1152, 
        C=256,
        hidden_dim=6912, 
        num_blocks=26,
        head_dim=256
    )
    # 封裝為 CausalLM 以匹配 trainCPTAC_1K.py 的儲存結構
    model = AGIV2GForCausalLM(base, use_gc=False)
    
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

def check_routing_gates(model):
    print("\n🔍 [分析] 檢查路由閘門 (Routing Gates) CPT 收斂狀況：")
    core = model.base_model
    for i, block in enumerate(core.blocks):
        if hasattr(block, 'gate_fft'):
            fft_val = block.gate_fft.item()
            mem_val = block.gate_mem.item()
            print(f"  ▶ Block {i:02d} | gate_fft: {fft_val:+.4f} | gate_mem: {mem_val:+.4f}")
    print("-" * 50)

@torch.no_grad()
def generate_cpt_autoregressive(model, tokenizer, prompt_ids, expected_b_size, max_display=64):
    """
    雙擎異步解碼：M_global 鎖死於 N，Local SDPA 逐字 AR 生成。
    """
    N_len = len(prompt_ids)
    
    # 初始輸入：只有 N 區塊 (沒有任何 PAD 污染)
    current_ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    
    # 🌟 核心防護：鎖定物理斷層點。無論迴圈跑多少次，n_split 永遠固定在 N
    n_split_tensor = torch.tensor([N_len], dtype=torch.long, device=DEVICE)
    
    print(f"\n🚀 [雙擎解碼] 啟動 AR + NAR 異步融合生成 (初始長度 N={N_len})")
    print(f"✨ [狀態] 宏觀引擎 M_global (NAR) 已鎖定於前 {N_len} 個 Token。")
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
        next_token_logits = logits[0, -1, :]
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
        
        # 可選：即時印出打字機效果
        # print(tokenizer.decode([next_token_id]), end="", flush=True)
            
    avg_conf = avg_conf / max(1, len(generated_ids))
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"📊 預測平均信心度: {avg_conf:.4f}")
    print(f"✅ 模型 AR+NAR 融合輸出：\n{output_text}")
    print("-" * 50)
    
    return output_text

def get_eval_sample():
    """確保提取與訓練期完全相同的獨立驗證集樣本。"""
    if not os.path.exists(DATASET_PATH):
        print(f"⚠️ 找不到實體 Parquet 資料集: {DATASET_PATH}")
        sys.exit(1)
        
    dataset_dict = load_dataset("parquet", data_files=DATASET_PATH)
    dataset = dataset_dict['train'].train_test_split(test_size=50, seed=RANDOM_SEED)
    eval_ds = dataset['test']
    
    print(f"📦 [資料集] 成功提取獨立驗證集樣本 (總數: {len(eval_ds)})")
    return eval_ds[0]

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 1. 取得測試資料與斷層點
    sample = get_eval_sample()
    full_seq = sample["input_ids"]
    n_split = sample["n_split_index"]
    
    prompt_ids = full_seq[:n_split]
    true_b_ids = full_seq[n_split:]
    b_size = len(true_b_ids)
    
    # 2. 載入無洩漏之 CPT 模型
    model = load_cpt_model()
    check_routing_gates(model)
    
    # 3. 對照 Ground Truth
    max_tokens = 64
    true_ans_text = tokenizer.decode([i for i in true_b_ids[:max_tokens] if i != 0], skip_special_tokens=True)
    
    print(f"\n💡 [標準答案 Ground Truth] B 區塊內容 (前 {max_tokens} Token)：\n{true_ans_text}")
    print("\n🎯 測試啟動：AGIV2 CPT 雙擎異步解碼驗證")
    
    # 4. 啟動生成 (改呼叫 AR 版本)
    generate_cpt_autoregressive(
        model=model, 
        tokenizer=tokenizer, 
        prompt_ids=prompt_ids, 
        expected_b_size=b_size, 
        max_display=max_tokens
    )