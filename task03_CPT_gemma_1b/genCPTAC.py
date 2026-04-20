"""
genCPTAC.py — AGIV2 CPT 階段解碼 (平行映射防洩漏對齊版)
=========================================================================
核心機制：
1. 認知對齊：訓練期已嚴格執行 B 區塊 PAD 抹除，推論時恢復單次 Forward 平行拔取 (Single-Shot Extraction)。
2. 權重對齊：使用 strict=True 確保 CPT 階段解凍的 Routing 與 LayerNorm 完美載入。
3. 輸出防護：精準提取 CausalLMOutput 中的 logits，避免 dict 存取錯誤。
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
            # 重新訓練後，這裡應呈現健康的極化分佈，而非全作弊的 +1.0000
            print(f"  ▶ Block {i:02d} | gate_fft: {fft_val:+.4f} | gate_mem: {mem_val:+.4f}")
    print("-" * 50)

@torch.no_grad()
def generate_cpt_parallel(model, tokenizer, prompt_ids, expected_b_size, max_display=64):
    """
    並行映射解碼：單次 Forward 提取整個 B 區塊。
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    N_len = len(prompt_ids)
    
    # [物理隔離重現] 構造輸入：N (已知) + B (全部填 PAD，對齊訓練期的防作弊機制)
    input_ids = prompt_ids + [pad_id] * expected_b_size
    current_ids = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    print(f"\n🚀 [平行映射] 總長度 L={N_len + expected_b_size} (N={N_len}, B={expected_b_size})")
    print(f"✨ [執行模式] 非自迴歸 Single-Shot 平行提取...")
    
    # 執行單次 Forward
    outputs = model(input_ids=current_ids)
    
    # 🌟 核心防護：從 CausalLMOutput 中明確提取 logits Tensor
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    elif isinstance(outputs, dict):
        logits = outputs['logits']
    else:
        logits = outputs
    
    # 根據平移量對齊：提取整個 B 區塊的預測 Logits
    # (N_len-1 位置預測第 N_len 個 Token)
    pred_logits = logits[0, N_len-1 : N_len-1 + expected_b_size, :]
    
    # 計算預測信心度
    probs = F.softmax(pred_logits, dim=-1)
    max_probs, pred_ids_full = torch.max(probs, dim=-1)
    avg_conf = max_probs[:max_display].mean().item()
    
    # 截斷邏輯：遇到 EOS 或是 PAD 停止輸出
    final_ids = []
    for pid in pred_ids_full.tolist():
        if pid == tokenizer.eos_token_id or pid == pad_id:
            break
        final_ids.append(pid)
        if len(final_ids) >= max_display:
            break
            
    output_text = tokenizer.decode(final_ids, skip_special_tokens=True)
    
    print(f"📊 預測平均信心度: {avg_conf:.4f}")
    print(f"✅ 模型平行輸出 (前 {max_display} Token)：\n{output_text}")
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
    print("\n🎯 測試啟動：AGIV2 CPT 物理隔離平行映射驗證")
    
    # 4. 啟動生成
    generate_cpt_parallel(
        model=model, 
        tokenizer=tokenizer, 
        prompt_ids=prompt_ids, 
        expected_b_size=b_size, 
        max_display=max_tokens
    )